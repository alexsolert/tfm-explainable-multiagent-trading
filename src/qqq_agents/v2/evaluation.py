"""Walk-forward V2 con modelos calibrados, pesos regularizados y ablaciones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import BacktestResult, run_backtest
from qqq_agents.config import AppConfig
from qqq_agents.v2.allocation import TacticalAllocator
from qqq_agents.v2.config import V2Config
from qqq_agents.v2.diagnostics import probability_diagnostics
from qqq_agents.v2.models import fit_temporal_probability_model, regularized_quality_weights


@dataclass(frozen=True)
class V2WalkForwardResult:
    decisions: pd.DataFrame
    strategy: BacktestResult
    baselines: dict[str, BacktestResult]
    ablations: dict[str, BacktestResult]
    training_audit: pd.DataFrame
    model_leaderboard: pd.DataFrame
    probability_report: dict[str, object]


def _available_group(frame: pd.DataFrame, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if name in frame.columns)


def _action(previous: float, desired: float) -> str:
    if desired > previous:
        return "BUY"
    if desired < previous:
        return "SELL"
    return "HOLD"


def _moving_average_position(frame: pd.DataFrame) -> pd.Series:
    close = frame["close"]
    sma_50 = close / (1 + frame["distance_sma_50"])
    sma_200 = close / (1 + frame["distance_sma_200"])
    return (sma_50 > sma_200).fillna(False).astype(float)


def _compact_logistic_position(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    random_seed: int,
) -> pd.Series:
    positions: list[pd.Series] = []
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        cutoff = period_start - pd.Timedelta(days=1)
        training = frame.loc[
            (frame.index <= cutoff) & (frame["target_end_date"] <= cutoff),
            [*feature_names, "target_up"],
        ].dropna()
        evaluation = frame.loc[
            (frame.index >= period_start) & (frame.index <= period_end), feature_names
        ].dropna()
        if training.empty or evaluation.empty or training["target_up"].nunique() < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, random_state=random_seed),
        )
        model.fit(training.loc[:, feature_names], training["target_up"].astype(int))
        positive_index = list(model.classes_).index(1)
        probability = model.predict_proba(evaluation)[:, positive_index]
        positions.append(pd.Series((probability >= 0.5).astype(float), index=evaluation.index))
    if not positions:
        raise ValueError("Compact logistic V2 baseline produced no positions")
    return pd.concat(positions).sort_index().reindex(frame.loc[start:end].index).ffill().fillna(1.0)


def _ablation_positions(decisions: pd.DataFrame, config: V2Config) -> dict[str, pd.Series]:
    risk = decisions["risk_probability"]
    raw_risk = decisions["risk_raw_probability"]
    direction = decisions["direction_probability"]
    moderate = config.allocation.moderate_risk_probability
    severe = config.allocation.severe_risk_probability
    bearish = config.allocation.bearish_probability_threshold
    risk_only = pd.Series(1.0, index=decisions.index)
    risk_only.loc[risk >= moderate] = 0.5
    risk_only.loc[risk >= severe] = 0.0
    directional = pd.Series(np.where(direction < bearish, 0.5, 1.0), index=decisions.index)
    binary = pd.Series(np.where(risk >= severe, 0.0, 1.0), index=decisions.index)
    uncalibrated = pd.Series(1.0, index=decisions.index)
    uncalibrated.loc[raw_risk >= moderate] = 0.5
    uncalibrated.loc[raw_risk >= severe] = 0.0
    return {
        "structural_long": pd.Series(1.0, index=decisions.index),
        "risk_only": risk_only,
        "directional_only": directional,
        "binary_calibrated_risk": binary,
        "uncalibrated_risk": uncalibrated,
    }


def run_v2_walk_forward(
    frame: pd.DataFrame,
    *,
    app_config: AppConfig,
    v2_config: V2Config,
    start: str | None = None,
    end: str | None = None,
    allow_protected: bool = False,
) -> V2WalkForwardResult:
    prepared = frame.copy().sort_index()
    prepared["target_end_date"] = pd.to_datetime(prepared["target_end_date"])
    start_at = pd.Timestamp(start or v2_config.development.start)
    end_at = pd.Timestamp(end or v2_config.development.end)
    if end_at >= pd.Timestamp(v2_config.development.protected_test_start) and not allow_protected:
        raise ValueError("Development evaluation cannot open the protected V2 test period")
    evaluation = prepared.loc[(prepared.index >= start_at) & (prepared.index <= end_at)]
    if evaluation.empty:
        raise ValueError("V2 evaluation period contains no observations")

    feature_groups = {
        "technical": _available_group(prepared, v2_config.features.technical),
        "momentum": _available_group(prepared, v2_config.features.momentum),
        "regime": _available_group(prepared, v2_config.features.regime),
        "risk": _available_group(prepared, v2_config.features.risk),
    }
    required_groups = ("technical", "momentum", "risk")
    if any(not feature_groups[name] for name in required_groups):
        raise ValueError("V2 requires technical, momentum and risk features")

    allocator = TacticalAllocator(v2_config.allocation)
    records: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    leaderboard: list[dict[str, object]] = []
    current_exposure = 1.0
    for year in range(start_at.year, end_at.year + 1):
        period_start = max(start_at, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end_at, pd.Timestamp(year=year, month=12, day=31))
        cutoff = period_start - pd.Timedelta(days=1)
        eligible = prepared.loc[
            (prepared.index <= cutoff) & (prepared["target_end_date"] <= cutoff)
        ]
        if len(eligible) < v2_config.development.minimum_training_rows:
            raise ValueError(f"Only {len(eligible)} V2 training rows available at {cutoff.date()}")

        models = {}
        for agent_id, names in feature_groups.items():
            if not names:
                continue
            target = "target_risk" if agent_id == "risk" else "target_up"
            model = fit_temporal_probability_model(
                eligible,
                feature_names=names,
                target_column=target,
                config=v2_config.models,
                random_seed=v2_config.evaluation.random_seed,
            )
            models[agent_id] = model
            audit.append(
                {
                    "agent_id": agent_id,
                    "cutoff": cutoff,
                    "training_rows": int(len(eligible.loc[:, [*names, target]].dropna())),
                    "training_start": eligible.index.min(),
                    "training_end": eligible.index.max(),
                    "maximum_target_end_date": eligible["target_end_date"].max(),
                    "selected_model": model.selected_model,
                    "selected_raw_brier": model.selected_raw_brier,
                    "selected_raw_auc": model.selected_raw_auc,
                    "features": ",".join(names),
                }
            )
            for diagnostic in model.diagnostics:
                leaderboard.append(
                    {
                        "agent_id": agent_id,
                        "cutoff": cutoff,
                        "candidate": diagnostic.name,
                        "observations": diagnostic.observations,
                        "raw_brier": diagnostic.raw_brier,
                        "raw_auc": diagnostic.raw_auc,
                        "selected": diagnostic.name == model.selected_model,
                    }
                )

        directional_models = {
            name: model
            for name, model in models.items()
            if name != "risk"
            and model.selected_raw_auc >= v2_config.models.minimum_direction_auc
        }
        weights = (
            regularized_quality_weights(
                directional_models,
                shrinkage=v2_config.models.adaptive_weight_shrinkage,
            )
            if directional_models
            else {}
        )
        risk_agent_active = models["risk"].selected_raw_auc >= v2_config.models.minimum_risk_auc
        rows = evaluation.loc[(evaluation.index >= period_start) & (evaluation.index <= period_end)]
        for timestamp, row in rows.iterrows():
            observation = row.to_frame().T
            probabilities: dict[str, float] = {}
            raw_probabilities: dict[str, float] = {}
            for agent_id, model in models.items():
                raw_probabilities[agent_id] = float(model.predict_raw_probability(observation)[0])
                probabilities[agent_id] = float(model.predict_probability(observation)[0])
            direction = (
                sum(weights[name] * probabilities[name] for name in directional_models)
                if directional_models
                else 0.5
            )
            risk = probabilities["risk"] if risk_agent_active else 0.0
            allocation = allocator.allocate(
                direction_probability=direction,
                risk_probability=risk,
            )
            desired = allocation.exposure
            action = _action(current_exposure, desired)
            record: dict[str, object] = {
                "date": timestamp,
                "action": action,
                "operation_executed": action != "HOLD",
                "previous_exposure": current_exposure,
                "desired_position": desired,
                "exposure_change": desired - current_exposure,
                "direction_probability": direction,
                "risk_probability": risk,
                "risk_raw_probability": raw_probabilities["risk"],
                "risk_agent_active": risk_agent_active,
                "risk_tier": allocation.risk_tier,
                "rationale": allocation.rationale,
            }
            for agent_id, probability in probabilities.items():
                record[f"{agent_id}_probability"] = probability
                record[f"{agent_id}_raw_probability"] = raw_probabilities[agent_id]
                if agent_id != "risk":
                    record[f"{agent_id}_weight"] = weights.get(agent_id, 0.0)
                    record[f"{agent_id}_active"] = agent_id in directional_models
                record[f"{agent_id}_model"] = models[agent_id].selected_model
            records.append(record)
            current_exposure = desired

    decisions = pd.DataFrame.from_records(records).set_index("date").sort_index()
    close = evaluation.loc[decisions.index, "close"]
    costs = app_config.experiment.transaction_cost_bps
    strategy = run_backtest(close, decisions["desired_position"], transaction_cost_bps=costs)
    baselines = {
        "buy_and_hold": run_backtest(close, buy_and_hold(close), transaction_cost_bps=costs),
        "sma_50_200": run_backtest(
            close,
            _moving_average_position(evaluation.loc[decisions.index]),
            transaction_cost_bps=costs,
        ),
    }
    compact_names = tuple(
        dict.fromkeys(
            [
                *feature_groups["technical"],
                *feature_groups["momentum"],
                *feature_groups["regime"],
            ]
        )
    )
    logistic_position = _compact_logistic_position(
        prepared,
        feature_names=compact_names,
        start=start_at,
        end=end_at,
        random_seed=v2_config.evaluation.random_seed,
    ).reindex(decisions.index)
    baselines["single_logistic_agent"] = run_backtest(
        close,
        logistic_position,
        transaction_cost_bps=costs,
    )
    ablations = {
        name: run_backtest(close, position, transaction_cost_bps=costs)
        for name, position in _ablation_positions(decisions, v2_config).items()
    }

    report: dict[str, object] = {
        "direction": probability_diagnostics(
            evaluation.loc[decisions.index, "target_up"],
            decisions["direction_probability"],
            bins=v2_config.evaluation.calibration_bins,
        ),
        "risk": probability_diagnostics(
            evaluation.loc[decisions.index, "target_risk"],
            decisions["risk_probability"],
            bins=v2_config.evaluation.calibration_bins,
        ),
        "by_year": {},
    }
    for year, yearly_decisions in decisions.groupby(decisions.index.year):
        yearly_observations = evaluation.loc[yearly_decisions.index]
        report["by_year"][str(year)] = {
            "direction": probability_diagnostics(
                yearly_observations["target_up"], yearly_decisions["direction_probability"]
            ),
            "risk": probability_diagnostics(
                yearly_observations["target_risk"], yearly_decisions["risk_probability"]
            ),
        }
    return V2WalkForwardResult(
        decisions=decisions,
        strategy=strategy,
        baselines=baselines,
        ablations=ablations,
        training_audit=pd.DataFrame.from_records(audit),
        model_leaderboard=pd.DataFrame.from_records(leaderboard),
        probability_report=report,
    )
