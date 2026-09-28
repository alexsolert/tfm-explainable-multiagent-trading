"""Expanding, purged walk-forward evaluation for V6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_agents.backtesting.engine import BacktestResult
from qqq_agents.backtesting.metrics import calculate_metrics
from qqq_agents.config import AppConfig
from qqq_agents.v4.models import fit_daily_classifier
from qqq_agents.v5.models import fit_volatility_laboratory
from qqq_agents.v6.backtest import run_exposure_backtest
from qqq_agents.v6.config import V6Config
from qqq_agents.v6.features import CORE_FEATURES, RISK_FEATURES, VOLATILITY_FEATURES
from qqq_agents.v6.models import fit_return_model
from qqq_agents.v6.policies import (
    ExposureSelection,
    build_exposure_policy,
    build_guarded_trend_policy,
    build_trend_exposure_policy,
    policy_name,
    select_exposure_policy,
)


@dataclass(frozen=True)
class V6WalkForwardResult:
    decisions: pd.DataFrame
    policies: dict[str, BacktestResult]
    policy_details: dict[str, pd.DataFrame]
    selected_policy: str
    strategy: BacktestResult
    baselines: dict[str, BacktestResult]
    policy_selection: ExposureSelection
    training_audit: pd.DataFrame
    model_leaderboard: pd.DataFrame


def _trend_score(frame: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            frame["distance_sma_50"].gt(0),
            frame["distance_sma_200"].gt(0),
            frame["momentum_20"].gt(0),
            frame["momentum_60"].gt(0),
        ],
        axis=1,
    ).mean(axis=1)


def _period_metrics(backtest: BacktestResult, start: str, end: str) -> dict[str, float]:
    history = backtest.history.loc[start:end]
    return calculate_metrics(
        returns=history["strategy_return"],
        turnover=history["turnover"],
        asset_returns=history["asset_return"],
        positions=history["applied_position"],
        periods_per_year=252,
    )


def run_v6_walk_forward(
    frame: pd.DataFrame,
    *,
    app_config: AppConfig,
    v6_config: V6Config,
    start: str | None = None,
    end: str | None = None,
) -> V6WalkForwardResult:
    prepared = frame.copy().sort_index()
    start_at = pd.Timestamp(start or v6_config.period.evaluation_start)
    end_at = pd.Timestamp(end or v6_config.period.development_end)
    evaluation = prepared.loc[start_at:end_at]
    if evaluation.empty:
        raise ValueError("V6 evaluation period is empty")
    required = [*CORE_FEATURES, *VOLATILITY_FEATURES]
    rows = evaluation.dropna(subset=required).copy()
    daily_trend = _trend_score(rows)
    weekly = daily_trend.resample("W-FRI").last().reindex(rows.index, method="ffill")
    weekly_trend = weekly.fillna(daily_trend)

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
        maximum_end_dates: list[pd.Timestamp] = []

        risk_models = {}
        risk_active = {}
        risk_base_rate = {}
        for horizon in v6_config.risk.horizons:
            target = f"target_tail_{horizon}"
            end_column = f"target_end_date_{horizon}"
            training = prepared.loc[
                (prepared.index <= cutoff)
                & (pd.to_datetime(prepared[end_column]) <= cutoff)
            ]
            available = len(training.dropna(subset=[*RISK_FEATURES, target]))
            if available < v6_config.period.minimum_training_rows:
                raise ValueError(f"Insufficient V6 risk training rows at {cutoff.date()}")
            model = fit_daily_classifier(
                training,
                features=RISK_FEATURES,
                target=target,
                config=v6_config.models,  # type: ignore[arg-type]
            )
            risk_models[horizon] = model
            risk_active[horizon] = model.selected_auc >= v6_config.risk.activation_auc
            risk_base_rate[horizon] = float(training[target].dropna().mean())
            maximum_end_dates.append(pd.Timestamp(training[end_column].max()))
            for diagnostic in model.diagnostics:
                leaderboard.append(
                    {
                        "cutoff": cutoff,
                        "agent": f"risk_{horizon}",
                        "candidate": diagnostic.name,
                        "observations": diagnostic.observations,
                        "loss": diagnostic.brier,
                        "correlation": diagnostic.auc,
                        "selected": diagnostic.name == model.selected_model,
                    }
                )

        return_models = {}
        for horizon in v6_config.return_model.horizons:
            target = f"target_normalized_return_{horizon}"
            end_column = f"return_target_end_date_{horizon}"
            training = prepared.loc[
                (prepared.index <= cutoff)
                & (pd.to_datetime(prepared[end_column]) <= cutoff)
            ]
            model = fit_return_model(
                training,
                features=CORE_FEATURES,
                target=target,
                config=v6_config.return_model,
            )
            return_models[horizon] = model
            maximum_end_dates.append(pd.Timestamp(training[end_column].max()))
            for diagnostic in model.diagnostics:
                leaderboard.append(
                    {
                        "cutoff": cutoff,
                        "agent": f"return_{horizon}",
                        "candidate": diagnostic.name,
                        "observations": diagnostic.observations,
                        "loss": diagnostic.mae,
                        "correlation": diagnostic.correlation,
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
                config=v6_config.models,  # type: ignore[arg-type]
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
                        "correlation": np.nan,
                        "selected": diagnostic.name == model.selected_model,
                    }
                )

        risk_predictions = {
            horizon: (
                risk_models[horizon].predict(period_rows)
                if risk_active[horizon]
                else np.zeros(len(period_rows))
            )
            for horizon in v6_config.risk.horizons
        }
        return_predictions = {
            horizon: return_models[horizon].predict(period_rows)
            for horizon in v6_config.return_model.horizons
        }
        volatility_predictions = {
            horizon: volatility_models[horizon].predict(period_rows) for horizon in (5, 20)
        }
        audits.append(
            {
                "cutoff": cutoff,
                "maximum_target_end_date": max(maximum_end_dates),
                **{
                    f"risk_{horizon}_auc": risk_models[horizon].selected_auc
                    for horizon in v6_config.risk.horizons
                },
                **{
                    f"return_{horizon}_correlation": return_models[
                        horizon
                    ].temporal_correlation
                    for horizon in v6_config.return_model.horizons
                },
                **{
                    f"return_{horizon}_model": return_models[horizon].selected_model
                    for horizon in v6_config.return_model.horizons
                },
            }
        )

        for offset, (timestamp, row) in enumerate(period_rows.iterrows()):
            active_weight = sum(
                weight
                for horizon, weight in zip(
                    v6_config.risk.horizons,
                    v6_config.risk.horizon_weights,
                    strict=True,
                )
                if risk_active[horizon]
            )
            weighted_risk = sum(
                weight
                * float(risk_predictions[horizon][offset])
                / max(risk_base_rate[horizon], 1e-6)
                for horizon, weight in zip(
                    v6_config.risk.horizons,
                    v6_config.risk.horizon_weights,
                    strict=True,
                )
                if risk_active[horizon]
            )
            risk_score = weighted_risk / active_weight if active_weight else 0.0
            return_score = sum(
                weight * float(return_predictions[horizon][offset])
                for horizon, weight in zip(
                    v6_config.return_model.horizons,
                    v6_config.return_model.horizon_weights,
                    strict=True,
                )
            )
            vol_5, _ = volatility_predictions[5]
            vol_20, _ = volatility_predictions[20]
            forecast_volatility = 0.4 * float(vol_5[offset]) + 0.6 * float(vol_20[offset])
            records.append(
                {
                    "date": timestamp,
                    "risk_score": risk_score,
                    "return_score": return_score,
                    "forecast_volatility": forecast_volatility,
                    "daily_trend_score": float(daily_trend.loc[timestamp]),
                    "weekly_trend_score": float(weekly_trend.loc[timestamp]),
                    "cash_return": float(row["cash_return"]),
                    **{
                        f"risk_probability_{horizon}": float(
                            risk_predictions[horizon][offset]
                        )
                        for horizon in v6_config.risk.horizons
                    },
                    **{
                        f"return_prediction_{horizon}": float(
                            return_predictions[horizon][offset]
                        )
                        for horizon in v6_config.return_model.horizons
                    },
                }
            )

    decisions = pd.DataFrame.from_records(records).set_index("date").sort_index()
    aligned = evaluation.loc[decisions.index]
    close = aligned["close"]
    cash = aligned["cash_return"]
    costs = app_config.experiment.transaction_cost_bps
    joint_policies = {
        policy_name(maximum, "joint"): build_exposure_policy(
            decisions,
            maximum_exposure=maximum,
            config=v6_config.allocation,
        )
        for maximum in v6_config.allocation.maximum_exposures
    }
    trend_policies = {
        policy_name(maximum, "trend"): build_trend_exposure_policy(
            decisions,
            maximum_exposure=maximum,
            config=v6_config.allocation,
        )
        for maximum in v6_config.allocation.maximum_exposures
    }
    guarded_policies = {
        policy_name(maximum, "guarded_trend"): build_guarded_trend_policy(
            decisions,
            maximum_exposure=maximum,
            config=v6_config.allocation,
        )
        for maximum in v6_config.allocation.maximum_exposures
    }
    policy_details = {**joint_policies, **trend_policies, **guarded_policies}
    policies = {
        name: run_exposure_backtest(
            close,
            detail["desired_position"],
            maximum_exposure=float(detail["maximum_exposure"].iloc[0]),
            transaction_cost_bps=costs,
            borrowing_spread_bps=v6_config.allocation.borrowing_spread_bps,
            cash_return=cash,
        )
        for name, detail in policy_details.items()
    }
    baselines = {
        "buy_and_hold": run_exposure_backtest(
            close,
            pd.Series(1.0, index=close.index),
            maximum_exposure=1.0,
            transaction_cost_bps=costs,
            borrowing_spread_bps=v6_config.allocation.borrowing_spread_bps,
            cash_return=cash,
        ),
        "constant_125": run_exposure_backtest(
            close,
            pd.Series(1.25, index=close.index),
            maximum_exposure=1.25,
            transaction_cost_bps=costs,
            borrowing_spread_bps=v6_config.allocation.borrowing_spread_bps,
            cash_return=cash,
        ),
    }
    selection_metrics = {
        name: _period_metrics(
            value, v6_config.period.evaluation_start, v6_config.period.selection_end
        )
        for name, value in policies.items()
    }
    benchmark_metrics = _period_metrics(
        baselines["buy_and_hold"],
        v6_config.period.evaluation_start,
        v6_config.period.selection_end,
    )
    selection = select_exposure_policy(
        selection_metrics,
        benchmark=benchmark_metrics,
        config=v6_config.selection,
    )
    selected = selection.selected_policy
    decisions = decisions.join(policy_details[selected].add_prefix("selected_"))
    return V6WalkForwardResult(
        decisions=decisions,
        policies=policies,
        policy_details=policy_details,
        selected_policy=selected,
        strategy=policies[selected],
        baselines=baselines,
        policy_selection=selection,
        training_audit=pd.DataFrame.from_records(audits),
        model_leaderboard=pd.DataFrame.from_records(leaderboard),
    )
