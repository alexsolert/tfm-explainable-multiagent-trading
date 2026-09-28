"""Daily hierarchical walk-forward evaluation for V4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import BacktestResult, run_backtest
from qqq_agents.config import AppConfig
from qqq_agents.v4.config import V4Config
from qqq_agents.v4.features import DIRECTION_FEATURES, RISK_FEATURES, VOLATILITY_FEATURES
from qqq_agents.v4.models import fit_daily_classifier, fit_har_volatility_model


@dataclass(frozen=True)
class V4WalkForwardResult:
    decisions: pd.DataFrame
    strategy: BacktestResult
    baselines: dict[str, BacktestResult]
    ablations: dict[str, BacktestResult]
    training_audit: pd.DataFrame
    model_leaderboard: pd.DataFrame


def _apply_policy(
    *,
    forecast_volatility: float,
    risk_probability: float,
    trend_score: float,
    direction_probability: float,
    direction_active: bool,
    previous_exposure: float,
    config: V4Config,
) -> tuple[float, dict[str, float]]:
    allocation = config.allocation
    volatility_exposure = float(
        np.clip(
            allocation.target_volatility / max(forecast_volatility, 1e-4),
            allocation.minimum_exposure,
            1.0,
        )
    )
    risk_cap = float(
        np.clip(
            1 - allocation.risk_sensitivity * max(risk_probability - allocation.risk_tolerance, 0),
            allocation.minimum_exposure,
            1.0,
        )
    )
    trend_cap = 1.0 if trend_score >= 0.5 else allocation.bearish_trend_cap
    direction_cap = (
        float(
            np.clip(
                allocation.direction_floor
                + 2 * direction_probability * (1 - allocation.direction_floor),
                allocation.direction_floor,
                1,
            )
        )
        if direction_active and direction_probability < 0.5
        else 1.0
    )
    raw = min(volatility_exposure, risk_cap, trend_cap, direction_cap)
    smoothed = allocation.smoothing * raw + (1 - allocation.smoothing) * previous_exposure
    desired = (
        previous_exposure
        if abs(smoothed - previous_exposure) < allocation.minimum_rebalance
        else float(np.clip(smoothed, allocation.minimum_exposure, 1.0))
    )
    return desired, {
        "volatility_exposure": volatility_exposure,
        "risk_cap": risk_cap,
        "trend_cap": trend_cap,
        "direction_cap": direction_cap,
        "raw_exposure": raw,
    }


def _trend_score(row: pd.Series) -> float:
    signals = (
        row["distance_sma_50"] > 0,
        row["distance_sma_200"] > 0,
        row["momentum_20"] > 0,
        row["momentum_60"] > 0,
    )
    return float(np.mean(signals))


def _ablation_positions(decisions: pd.DataFrame, config: V4Config) -> dict[str, pd.Series]:
    minimum = config.allocation.minimum_exposure
    risk_only = decisions["risk_cap"].clip(lower=minimum)
    vol_only = decisions["volatility_exposure"].clip(lower=minimum)
    trend_only = decisions["trend_cap"].clip(lower=minimum)
    direction_only = decisions["direction_cap"].clip(lower=minimum)
    return {
        "risk_only": risk_only,
        "volatility_only": vol_only,
        "trend_only": trend_only,
        "direction_only": direction_only,
    }


def run_v4_walk_forward(
    frame: pd.DataFrame,
    *,
    app_config: AppConfig,
    v4_config: V4Config,
    start: str | None = None,
    end: str | None = None,
    allow_prospective: bool = False,
) -> V4WalkForwardResult:
    prepared = frame.copy().sort_index()
    start_at = pd.Timestamp(start or v4_config.period.evaluation_start)
    end_at = pd.Timestamp(end or v4_config.period.development_end)
    if end_at >= pd.Timestamp(v4_config.period.prospective_start) and not allow_prospective:
        raise ValueError("V4 development cannot open its prospective period")
    evaluation = prepared.loc[start_at:end_at]
    if evaluation.empty:
        raise ValueError("V4 evaluation period is empty")

    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    leaderboard: list[dict[str, object]] = []
    previous_exposure = 1.0
    for year in range(start_at.year, end_at.year + 1):
        period_start = max(start_at, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end_at, pd.Timestamp(year=year, month=12, day=31))
        cutoff = period_start - pd.Timedelta(days=1)
        classification_training = prepared.loc[
            (prepared.index <= cutoff)
            & (pd.to_datetime(prepared["target_end_date"]) <= cutoff)
        ]
        volatility_training = prepared.loc[
            (prepared.index <= cutoff)
            & (pd.to_datetime(prepared["vol_target_end_date"]) <= cutoff)
        ]
        available_risk_rows = len(
            classification_training.dropna(subset=list(RISK_FEATURES))
        )
        if available_risk_rows < v4_config.period.minimum_training_rows:
            raise ValueError(f"Insufficient V4 training rows at {cutoff.date()}")
        risk_model = fit_daily_classifier(
            classification_training,
            features=RISK_FEATURES,
            target="target_risk",
            config=v4_config.models,
        )
        direction_model = fit_daily_classifier(
            classification_training,
            features=DIRECTION_FEATURES,
            target="target_up",
            config=v4_config.models,
        )
        volatility_model = fit_har_volatility_model(
            volatility_training,
            features=VOLATILITY_FEATURES,
            config=v4_config.models,
        )
        risk_active = risk_model.selected_auc >= v4_config.models.minimum_risk_auc
        direction_active = direction_model.selected_auc >= v4_config.models.minimum_direction_auc
        audits.append(
            {
                "cutoff": cutoff,
                "training_rows": len(classification_training),
                "maximum_target_end_date": classification_training["target_end_date"].max(),
                "risk_model": risk_model.selected_model,
                "risk_auc": risk_model.selected_auc,
                "risk_brier": risk_model.selected_brier,
                "risk_active": risk_active,
                "direction_model": direction_model.selected_model,
                "direction_auc": direction_model.selected_auc,
                "direction_brier": direction_model.selected_brier,
                "direction_active": direction_active,
                "har_volatility_mae": volatility_model.temporal_mae,
            }
        )
        for agent, model in (("risk", risk_model), ("direction", direction_model)):
            for diagnostic in model.diagnostics:
                leaderboard.append(
                    {
                        "cutoff": cutoff,
                        "agent": agent,
                        "candidate": diagnostic.name,
                        "observations": diagnostic.observations,
                        "brier": diagnostic.brier,
                        "auc": diagnostic.auc,
                        "selected": diagnostic.name == model.selected_model,
                    }
                )

        rows = evaluation.loc[period_start:period_end].dropna(
            subset=[*RISK_FEATURES, *DIRECTION_FEATURES, *VOLATILITY_FEATURES]
        )
        risk_probability = risk_model.predict(rows) if risk_active else np.zeros(len(rows))
        direction_probability = direction_model.predict(rows)
        volatility_forecast = volatility_model.predict(rows)
        for timestamp, (_, row), risk, direction, volatility in zip(
            rows.index,
            rows.iterrows(),
            risk_probability,
            direction_probability,
            volatility_forecast,
            strict=True,
        ):
            trend = _trend_score(row)
            desired, components = _apply_policy(
                forecast_volatility=float(volatility),
                risk_probability=float(risk),
                trend_score=trend,
                direction_probability=float(direction),
                direction_active=direction_active,
                previous_exposure=previous_exposure,
                config=v4_config,
            )
            records.append(
                {
                    "date": timestamp,
                    "desired_position": desired,
                    "previous_exposure": previous_exposure,
                    "risk_probability": risk,
                    "risk_active": risk_active,
                    "direction_probability": direction,
                    "direction_active": direction_active,
                    "forecast_volatility": volatility,
                    "trend_score": trend,
                    "cash_return": row["cash_return"],
                    "risk_model": risk_model.selected_model,
                    "direction_model": direction_model.selected_model,
                    **components,
                }
            )
            previous_exposure = desired

    decisions = pd.DataFrame.from_records(records).set_index("date").sort_index()
    close = evaluation.loc[decisions.index, "close"]
    cash = evaluation.loc[decisions.index, "cash_return"]
    costs = app_config.experiment.transaction_cost_bps
    strategy = run_backtest(
        close,
        decisions["desired_position"],
        transaction_cost_bps=costs,
        periods_per_year=252,
        cash_return=cash,
    )
    aligned = evaluation.loc[decisions.index]
    sma_50 = close / (1 + aligned["distance_sma_50"])
    sma_200 = close / (1 + aligned["distance_sma_200"])
    sma_position = (sma_50 > sma_200).astype(float)
    trailing_vol_position = (
        v4_config.allocation.target_volatility
        / evaluation.loc[decisions.index, "rv_20"].replace(0, np.nan)
    ).clip(v4_config.allocation.minimum_exposure, 1.0).fillna(1.0)
    baselines = {
        "buy_and_hold": run_backtest(
            close,
            buy_and_hold(close),
            transaction_cost_bps=costs,
            periods_per_year=252,
            cash_return=cash,
        ),
        "sma_50_200": run_backtest(
            close, sma_position, transaction_cost_bps=costs, periods_per_year=252, cash_return=cash
        ),
        "trailing_volatility_target": run_backtest(
            close,
            trailing_vol_position,
            transaction_cost_bps=costs,
            periods_per_year=252,
            cash_return=cash,
        ),
    }
    ablations = {
        name: run_backtest(
            close,
            position,
            transaction_cost_bps=costs,
            periods_per_year=252,
            cash_return=cash,
        )
        for name, position in _ablation_positions(decisions, v4_config).items()
    }
    return V4WalkForwardResult(
        decisions=decisions,
        strategy=strategy,
        baselines=baselines,
        ablations=ablations,
        training_audit=pd.DataFrame.from_records(audits),
        model_leaderboard=pd.DataFrame.from_records(leaderboard),
    )
