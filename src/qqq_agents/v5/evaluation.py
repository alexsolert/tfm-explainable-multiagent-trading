"""Leakage-safe multi-frequency evaluation and policy selection for V5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import BacktestResult, run_backtest
from qqq_agents.backtesting.metrics import calculate_metrics
from qqq_agents.config import AppConfig
from qqq_agents.v4.models import fit_daily_classifier
from qqq_agents.v5.config import V5Config
from qqq_agents.v5.features import RISK_FEATURES, VOLATILITY_FEATURES
from qqq_agents.v5.models import fit_volatility_laboratory
from qqq_agents.v5.policies import PolicySelection, build_policy_positions, pareto_select_policy


@dataclass(frozen=True)
class V5WalkForwardResult:
    decisions: pd.DataFrame
    policies: dict[str, BacktestResult]
    policy_details: dict[str, pd.DataFrame]
    selected_policy: str
    strategy: BacktestResult
    baselines: dict[str, BacktestResult]
    policy_selection: PolicySelection
    training_audit: pd.DataFrame
    model_leaderboard: pd.DataFrame


def _trend_score(frame: pd.DataFrame) -> pd.Series:
    signals = pd.concat(
        [
            frame["distance_sma_50"].gt(0),
            frame["distance_sma_200"].gt(0),
            frame["momentum_20"].gt(0),
            frame["momentum_60"].gt(0),
        ],
        axis=1,
    )
    return signals.mean(axis=1)


def _period_metrics(backtest: BacktestResult, start: str, end: str) -> dict[str, float]:
    history = backtest.history.loc[start:end]
    return calculate_metrics(
        returns=history["strategy_return"],
        turnover=history["turnover"],
        asset_returns=history["asset_return"],
        positions=history["applied_position"],
        periods_per_year=252,
    )


def run_v5_walk_forward(
    frame: pd.DataFrame,
    *,
    app_config: AppConfig,
    v5_config: V5Config,
    start: str | None = None,
    end: str | None = None,
    allow_prospective: bool = False,
) -> V5WalkForwardResult:
    prepared = frame.copy().sort_index()
    start_at = pd.Timestamp(start or v5_config.period.evaluation_start)
    end_at = pd.Timestamp(end or v5_config.period.development_end)
    if end_at >= pd.Timestamp(v5_config.period.prospective_start) and not allow_prospective:
        raise ValueError("V5 development cannot open its prospective period")
    evaluation = prepared.loc[start_at:end_at]
    if evaluation.empty:
        raise ValueError("V5 evaluation period is empty")

    required = [*RISK_FEATURES, *VOLATILITY_FEATURES, "distance_sma_50", "distance_sma_200"]
    rows = evaluation.dropna(subset=required).copy()
    daily_trend = _trend_score(rows)
    weekly_observations = daily_trend.resample("W-FRI").last()
    weekly_trend = weekly_observations.reindex(rows.index, method="ffill").fillna(daily_trend)

    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    leaderboard: list[dict[str, object]] = []
    for year in range(start_at.year, end_at.year + 1):
        period_start = max(start_at, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end_at, pd.Timestamp(year=year, month=12, day=31))
        cutoff = period_start - pd.Timedelta(days=1)
        period_rows = rows.loc[period_start:period_end]
        if period_rows.empty:
            continue

        risk_models = {}
        risk_active = {}
        risk_positive_rate = {}
        maximum_end_dates: list[pd.Timestamp] = []
        for horizon in v5_config.risk.horizons:
            target = f"target_tail_{horizon}"
            end_column = f"target_end_date_{horizon}"
            training = prepared.loc[
                (prepared.index <= cutoff)
                & (pd.to_datetime(prepared[end_column]) <= cutoff)
            ]
            available_training_rows = len(training.dropna(subset=list(RISK_FEATURES)))
            if available_training_rows < v5_config.period.minimum_training_rows:
                raise ValueError(f"Insufficient V5 risk training rows at {cutoff.date()}")
            model = fit_daily_classifier(
                training,
                features=RISK_FEATURES,
                target=target,
                config=v5_config.models,  # type: ignore[arg-type]
            )
            risk_models[horizon] = model
            risk_active[horizon] = model.selected_auc >= v5_config.risk.activation_auc
            risk_positive_rate[horizon] = float(training[target].dropna().mean())
            maximum_end_dates.append(pd.Timestamp(training[end_column].max()))
            for diagnostic in model.diagnostics:
                leaderboard.append(
                    {
                        "cutoff": cutoff,
                        "agent": f"risk_{horizon}",
                        "candidate": diagnostic.name,
                        "observations": diagnostic.observations,
                        "loss": diagnostic.brier,
                        "auc": diagnostic.auc,
                        "selected": diagnostic.name == model.selected_model,
                    }
                )

        volatility_models = {}
        for horizon in (5, 20):
            target = f"future_realized_vol_{horizon}"
            end_column = f"vol_target_end_date_{horizon}"
            training = prepared.loc[
                (prepared.index <= cutoff)
                & (pd.to_datetime(prepared[end_column]) <= cutoff)
            ]
            model = fit_volatility_laboratory(
                training,
                features=VOLATILITY_FEATURES,
                target=target,
                config=v5_config.models,
            )
            volatility_models[horizon] = model
            maximum_end_dates.append(pd.Timestamp(training[end_column].max()))
            for diagnostic in model.diagnostics:
                leaderboard.append(
                    {
                        "cutoff": cutoff,
                        "agent": f"volatility_{horizon}",
                        "candidate": diagnostic.name,
                        "observations": diagnostic.observations,
                        "loss": diagnostic.mae,
                        "auc": np.nan,
                        "selected": diagnostic.name == model.selected_model,
                    }
                )

        risk_predictions = {
            horizon: (
                risk_models[horizon].predict(period_rows)
                if risk_active[horizon]
                else np.zeros(len(period_rows))
            )
            for horizon in v5_config.risk.horizons
        }
        volatility_predictions = {
            horizon: volatility_models[horizon].predict(period_rows)
            for horizon in (5, 20)
        }
        audits.append(
            {
                "cutoff": cutoff,
                "maximum_target_end_date": max(maximum_end_dates),
                **{
                    f"risk_{horizon}_auc": risk_models[horizon].selected_auc
                    for horizon in v5_config.risk.horizons
                },
                **{
                    f"risk_{horizon}_brier": risk_models[horizon].selected_brier
                    for horizon in v5_config.risk.horizons
                },
                **{
                    f"risk_{horizon}_positive_rate": risk_positive_rate[horizon]
                    for horizon in v5_config.risk.horizons
                },
                **{
                    f"risk_{horizon}_model": risk_models[horizon].selected_model
                    for horizon in v5_config.risk.horizons
                },
                **{
                    f"risk_{horizon}_active": risk_active[horizon]
                    for horizon in v5_config.risk.horizons
                },
                **{
                    f"volatility_{horizon}_model": volatility_models[horizon].selected_model
                    for horizon in (5, 20)
                },
                **{
                    f"volatility_{horizon}_mae": volatility_models[horizon].temporal_mae
                    for horizon in (5, 20)
                },
                **{
                    f"volatility_{horizon}_upper_coverage": volatility_models[
                        horizon
                    ].upper_coverage
                    for horizon in (5, 20)
                },
            }
        )

        for offset, (timestamp, row) in enumerate(period_rows.iterrows()):
            probabilities = {
                horizon: float(risk_predictions[horizon][offset])
                for horizon in v5_config.risk.horizons
            }
            relative_intensities = {
                horizon: probabilities[horizon] / max(risk_positive_rate[horizon], 1e-6)
                for horizon in v5_config.risk.horizons
            }
            active_weight = sum(
                weight
                for horizon, weight in zip(
                    v5_config.risk.horizons,
                    v5_config.risk.horizon_weights,
                    strict=True,
                )
                if risk_active[horizon]
            )
            weighted_intensity = sum(
                weight * relative_intensities[horizon]
                for horizon, weight in zip(
                    v5_config.risk.horizons,
                    v5_config.risk.horizon_weights,
                    strict=True,
                )
            )
            score = weighted_intensity / active_weight if active_weight else 0.0
            vol_5, upper_5 = volatility_predictions[5]
            vol_20, upper_20 = volatility_predictions[20]
            point_forecast = 0.4 * float(vol_5[offset]) + 0.6 * float(vol_20[offset])
            upper_forecast = 0.4 * float(upper_5[offset]) + 0.6 * float(upper_20[offset])
            records.append(
                {
                    "date": timestamp,
                    **{
                        f"risk_probability_{horizon}": value
                        for horizon, value in probabilities.items()
                    },
                    **{
                        f"risk_active_{horizon}": risk_active[horizon]
                        for horizon in v5_config.risk.horizons
                    },
                    **{
                        f"risk_base_rate_{horizon}": risk_positive_rate[horizon]
                        for horizon in v5_config.risk.horizons
                    },
                    **{
                        f"risk_intensity_{horizon}": relative_intensities[horizon]
                        for horizon in v5_config.risk.horizons
                    },
                    **{
                        f"risk_model_{horizon}": risk_models[horizon].selected_model
                        for horizon in v5_config.risk.horizons
                    },
                    "risk_score": score,
                    "forecast_volatility": point_forecast,
                    "upper_volatility": upper_forecast,
                    "forecast_volatility_5": float(vol_5[offset]),
                    "forecast_volatility_20": float(vol_20[offset]),
                    "volatility_model_5": volatility_models[5].selected_model,
                    "volatility_model_20": volatility_models[20].selected_model,
                    "daily_trend_score": float(daily_trend.loc[timestamp]),
                    "weekly_trend_score": float(weekly_trend.loc[timestamp]),
                    "cash_return": float(row["cash_return"]),
                }
            )

    decisions = pd.DataFrame.from_records(records).set_index("date").sort_index()
    aligned = evaluation.loc[decisions.index]
    close = aligned["close"]
    cash = aligned["cash_return"]
    costs = app_config.experiment.transaction_cost_bps
    policy_details = {
        policy: build_policy_positions(
            decisions,
            policy=policy,
            config=v5_config.allocation,
        )
        for policy in v5_config.selection.policy_candidates
    }
    policies = {
        policy: run_backtest(
            close,
            detail["desired_position"],
            transaction_cost_bps=costs,
            periods_per_year=252,
            cash_return=cash,
        )
        for policy, detail in policy_details.items()
    }
    sma_50 = close / (1 + aligned["distance_sma_50"])
    sma_200 = close / (1 + aligned["distance_sma_200"])
    trailing_vol_position = (
        v5_config.allocation.target_volatility / aligned["rv_20"].replace(0, np.nan)
    ).clip(v5_config.allocation.minimum_exposure, 1.0).fillna(1.0)
    baselines = {
        "buy_and_hold": run_backtest(
            close,
            buy_and_hold(close),
            transaction_cost_bps=costs,
            periods_per_year=252,
            cash_return=cash,
        ),
        "sma_50_200": run_backtest(
            close,
            (sma_50 > sma_200).astype(float),
            transaction_cost_bps=costs,
            periods_per_year=252,
            cash_return=cash,
        ),
        "trailing_volatility_target": run_backtest(
            close,
            trailing_vol_position,
            transaction_cost_bps=costs,
            periods_per_year=252,
            cash_return=cash,
        ),
    }
    selection_metrics = {
        name: _period_metrics(
            value,
            v5_config.period.evaluation_start,
            v5_config.period.selection_end,
        )
        for name, value in policies.items()
    }
    benchmark_metrics = _period_metrics(
        baselines["buy_and_hold"],
        v5_config.period.evaluation_start,
        v5_config.period.selection_end,
    )
    policy_selection = pareto_select_policy(
        selection_metrics,
        benchmark=benchmark_metrics,
        config=v5_config.selection,
    )
    selected = policy_selection.selected_policy
    decisions = decisions.join(
        policy_details[selected].add_prefix("selected_")
    )
    return V5WalkForwardResult(
        decisions=decisions,
        policies=policies,
        policy_details=policy_details,
        selected_policy=selected,
        strategy=policies[selected],
        baselines=baselines,
        policy_selection=policy_selection,
        training_audit=pd.DataFrame.from_records(audits),
        model_leaderboard=pd.DataFrame.from_records(leaderboard),
    )
