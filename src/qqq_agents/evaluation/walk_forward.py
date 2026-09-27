"""Walk-forward expansivo para el MVP cuantitativo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qqq_agents.agents.quantitative import (
    MOMENTUM_FEATURES,
    RISK_FEATURES,
    TECHNICAL_FEATURES,
    QuantitativeAgent,
)
from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import BacktestResult, run_backtest
from qqq_agents.config import AppConfig
from qqq_agents.coordinator import DeterministicCoordinator
from qqq_agents.personality import load_personality
from qqq_agents.schemas import Action, Personality

LOGISTIC_FEATURES = tuple(dict.fromkeys([*TECHNICAL_FEATURES, *MOMENTUM_FEATURES, *RISK_FEATURES]))


@dataclass(frozen=True)
class WalkForwardResult:
    decisions: pd.DataFrame
    strategy: BacktestResult
    baselines: dict[str, BacktestResult]
    training_audit: pd.DataFrame


def fit_agents_at_cutoff(
    frame: pd.DataFrame,
    *,
    config: AppConfig,
    cutoff: pd.Timestamp,
) -> tuple[dict[str, QuantitativeAgent], list[dict[str, object]]]:
    eligible = frame.loc[(frame.index <= cutoff) & (frame["target_end_date"] <= cutoff)].copy()
    if eligible.empty:
        raise ValueError(f"No leakage-safe training data available through {cutoff.date()}")

    specifications = {
        "technical": (TECHNICAL_FEATURES, "target_up", "direction"),
        "momentum": (MOMENTUM_FEATURES, "target_up", "direction"),
        "risk": (RISK_FEATURES, "target_risk", "risk"),
    }
    agents: dict[str, QuantitativeAgent] = {}
    audit: list[dict[str, object]] = []
    for agent_id, (feature_names, target_column, target_kind) in specifications.items():
        model_config = config.models[agent_id]
        complete = eligible.loc[:, [*feature_names, target_column]].dropna()
        agent = QuantitativeAgent.fit(
            agent_id=agent_id,
            frame=eligible,
            feature_names=feature_names,
            target_column=target_column,
            target_kind=target_kind,  # type: ignore[arg-type]
            random_seed=config.project.random_seed,
            n_estimators=model_config.n_estimators or 300,
            max_depth=model_config.max_depth or 6,
            min_samples_leaf=model_config.min_samples_leaf or 10,
            model_version=f"{agent_id}-rf-through-{cutoff.date()}",
        )
        agents[agent_id] = agent
        audit.append(
            {
                "agent_id": agent_id,
                "cutoff": cutoff,
                "training_rows": len(complete),
                "training_start": complete.index.min(),
                "training_end": complete.index.max(),
                "maximum_target_end_date": eligible.loc[complete.index, "target_end_date"].max(),
                "positive_rate": float(complete[target_column].astype(float).mean()),
            }
        )
    return agents, audit


def _moving_average_position(frame: pd.DataFrame) -> pd.Series:
    close = frame["close"]
    sma_50 = close / (1 + frame["distance_sma_50"])
    sma_200 = close / (1 + frame["distance_sma_200"])
    return (sma_50 > sma_200).fillna(False).astype(float)


def _logistic_walk_forward_position(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    random_seed: int,
) -> pd.Series:
    """Single-model baseline refitted at the same yearly boundaries as the framework."""

    positions: list[pd.Series] = []
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        cutoff = period_start - pd.Timedelta(days=1)
        training = frame.loc[
            (frame.index <= cutoff) & (frame["target_end_date"] <= cutoff),
            [*LOGISTIC_FEATURES, "target_up"],
        ].dropna()
        evaluation = frame.loc[
            (frame.index >= period_start) & (frame.index <= period_end), LOGISTIC_FEATURES
        ].dropna()
        if training.empty or evaluation.empty:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=random_seed,
            ),
        )
        model.fit(training.loc[:, LOGISTIC_FEATURES], training["target_up"].astype(int))
        positive_class_index = list(model.classes_).index(1)
        probability = model.predict_proba(evaluation)[:, positive_class_index]
        positions.append(pd.Series((probability >= 0.5).astype(float), index=evaluation.index))
    if not positions:
        raise ValueError("The logistic baseline produced no validation positions")
    return pd.concat(positions).sort_index().rename("single_logistic_agent")


def run_quantitative_walk_forward(
    frame: pd.DataFrame,
    *,
    config: AppConfig,
    validation_start: str = "2020-01-01",
    validation_end: str = "2022-12-31",
    include_shap: bool = False,
    personality_name: str = "conservative",
) -> WalkForwardResult:
    """Retrain at each calendar-year boundary and decide weekly thereafter."""

    prepared = frame.copy().sort_index()
    prepared["target_end_date"] = pd.to_datetime(prepared["target_end_date"])
    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    validation = prepared.loc[(prepared.index >= start) & (prepared.index <= end)].copy()
    if validation.empty:
        raise ValueError("The requested validation period contains no observations")

    profile = load_personality(f"configs/personalities/{personality_name}.yaml")
    coordinator = DeterministicCoordinator(
        weights=profile.transform_weights(config.coordinator.weights),
        buy_threshold=max(
            0.01, min(1.0, config.coordinator.buy_threshold + profile.buy_threshold_delta)
        ),
        sell_threshold=config.coordinator.sell_threshold,
        risk_veto_probability=max(
            0.0,
            min(
                1.0,
                config.coordinator.risk_veto_probability + profile.risk_veto_probability_delta,
            ),
        ),
    )
    current_position = 0.0
    records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []

    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        cutoff = period_start - pd.Timedelta(days=1)
        agents, audit = fit_agents_at_cutoff(prepared, config=config, cutoff=cutoff)
        audit_records.extend(audit)
        shap_explainers: dict[str, object] = {}
        if include_shap:
            from qqq_agents.explainability import TreeShapAgentExplainer

            shap_explainers = {
                agent_id: TreeShapAgentExplainer(agent) for agent_id, agent in agents.items()
            }

        year_rows = validation.loc[
            (validation.index >= period_start) & (validation.index <= period_end)
        ]
        for timestamp, row in year_rows.iterrows():
            observation = row.to_dict()
            signals = [
                profile.transform_signal(
                    agent.evaluate(
                        as_of=timestamp.date(),
                        observation=observation,
                        personality=Personality.CONSERVATIVE,
                    )
                )
                for agent in agents.values()
            ]
            trace = coordinator.decide(
                as_of=timestamp.date(),
                signals=signals,
                current_position=current_position,
                created_at=datetime.combine(timestamp.date(), time.min, tzinfo=UTC),
            )
            if trace.action is Action.BUY:
                current_position = 1.0
            elif trace.action is Action.SELL:
                current_position = 0.0

            record: dict[str, object] = {
                "date": timestamp,
                "action": trace.action.value,
                "desired_position": current_position,
                "score_before_veto": trace.score_before_veto,
                "risk_veto_triggered": trace.risk_veto_triggered,
                "risk_veto_reason": trace.risk_veto_reason,
                "coordinator_version": trace.coordinator_version,
                "personality": profile.name.value,
            }
            for signal in signals:
                record[f"{signal.agent_id}_signal"] = signal.signal
                record[f"{signal.agent_id}_confidence"] = signal.confidence
                record[f"{signal.agent_id}_model_version"] = signal.model_version
                if signal.veto_probability is not None:
                    record[f"{signal.agent_id}_veto_probability"] = signal.veto_probability
                if include_shap:
                    explainer = shap_explainers[signal.agent_id]
                    attributions = explainer.explain(observation, top_n=5)  # type: ignore[attr-defined]
                    record[f"{signal.agent_id}_shap"] = json.dumps(
                        [item.to_dict() for item in attributions], sort_keys=True
                    )
            records.append(record)

    decisions = pd.DataFrame.from_records(records).set_index("date").sort_index()
    close = validation.loc[decisions.index, "close"]
    transaction_cost = config.experiment.transaction_cost_bps
    strategy = run_backtest(
        close,
        decisions["desired_position"],
        transaction_cost_bps=transaction_cost,
    )
    baselines = {
        "buy_and_hold": run_backtest(
            close,
            buy_and_hold(close),
            transaction_cost_bps=transaction_cost,
        ),
        "sma_50_200": run_backtest(
            close,
            _moving_average_position(validation.loc[decisions.index]),
            transaction_cost_bps=transaction_cost,
        ),
    }
    logistic_position = _logistic_walk_forward_position(
        prepared,
        start=start,
        end=end,
        random_seed=config.project.random_seed,
    ).reindex(decisions.index)
    baselines["single_logistic_agent"] = run_backtest(
        close,
        logistic_position,
        transaction_cost_bps=transaction_cost,
    )
    training_audit = pd.DataFrame.from_records(audit_records)
    return WalkForwardResult(
        decisions=decisions,
        strategy=strategy,
        baselines=baselines,
        training_audit=training_audit,
    )
