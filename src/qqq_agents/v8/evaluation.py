"""Evaluation and benchmark construction for V8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_agents.backtesting.engine import BacktestResult
from qqq_agents.backtesting.metrics import calculate_metrics
from qqq_agents.v7.backtest import run_v7_backtest
from qqq_agents.v8.agents import build_agent_signals
from qqq_agents.v8.config import V8Config
from qqq_agents.v8.coordinator import coordinate_profile, decision_action


@dataclass(frozen=True)
class V8Evaluation:
    decisions: pd.DataFrame
    strategies: dict[str, BacktestResult]
    baselines: dict[str, BacktestResult]
    matched_baselines: dict[str, BacktestResult]
    subperiods: dict[str, dict[str, dict[str, float]]]


def _period_metrics(
    result: BacktestResult, start: str, end: str
) -> dict[str, float]:
    history = result.history.loc[start:end]
    return calculate_metrics(
        returns=history["strategy_return"],
        turnover=history["turnover"],
        asset_returns=history["asset_return"],
        positions=history["applied_position"],
        periods_per_year=252,
    )


def _run(
    close: pd.Series,
    cash: pd.Series,
    exposure: pd.Series,
    maximum: float,
    config: V8Config,
) -> BacktestResult:
    return run_v7_backtest(
        close,
        exposure,
        maximum_exposure=maximum,
        transaction_cost_bps=config.costs.transaction_cost_bps,
        borrowing_spread_bps=config.costs.borrowing_spread_bps,
        cash_return=cash,
    )


def evaluate_v8(frame: pd.DataFrame, config: V8Config) -> V8Evaluation:
    close = frame["close"].sort_index()
    cash = frame["cash_return"].reindex(close.index).ffill().fillna(0.0)
    signals = build_agent_signals(frame, config.signals)
    decisions = signals.copy()
    strategies: dict[str, BacktestResult] = {}
    details: dict[str, pd.DataFrame] = {}
    for profile in config.profiles:
        detail = coordinate_profile(signals, profile)
        details[profile.name] = detail
        strategies[profile.name] = _run(
            close,
            cash,
            detail["desired_position"],
            profile.maximum_exposure,
            config,
        )
        decisions = decisions.join(
            detail.add_prefix(f"{profile.name}_"), how="left"
        )
        previous = detail["desired_position"].shift(1).fillna(0.0)
        decisions[f"{profile.name}_action"] = [
            decision_action(current, prior)
            for current, prior in zip(
                detail["desired_position"], previous, strict=True
            )
        ]

    baseline_exposures = sorted(
        {1.0, *(profile.maximum_exposure for profile in config.profiles)}
    )
    baselines = {
        f"constant_{int(exposure * 100)}": _run(
            close,
            cash,
            pd.Series(exposure, index=close.index),
            exposure,
            config,
        )
        for exposure in baseline_exposures
    }
    sma = close.rolling(250).mean()
    weekly_sma = sma.resample("W-FRI").last().reindex(close.index, method="ffill")
    weekly_close = close.resample("W-FRI").last().reindex(close.index, method="ffill")
    baselines["sma_250_cash"] = _run(
        close,
        cash,
        pd.Series(np.where(weekly_close > weekly_sma, 1.0, 0.0), index=close.index),
        1.0,
        config,
    )
    weekly_volatility = (
        frame["volatility_20"].resample("W-FRI").last().reindex(close.index, method="ffill")
    )
    volatility_exposure = (0.35 / weekly_volatility).clip(lower=0.25, upper=1.50)
    baselines["volatility_target_35"] = _run(
        close, cash, volatility_exposure, 1.50, config
    )

    selection = (config.period.selection_start, config.period.selection_end)
    matched_baselines = {}
    for profile in config.profiles:
        average = float(
            strategies[profile.name]
            .history.loc[selection[0] : selection[1], "applied_position"]
            .mean()
        )
        matched_baselines[f"matched_{profile.name}"] = _run(
            close,
            cash,
            pd.Series(average, index=close.index),
            max(1.0, average),
            config,
        )

    periods = {
        "selection": selection,
        "retrospective": (
            config.period.retrospective_start,
            config.period.retrospective_end,
        ),
    }
    all_results = {**strategies, **baselines, **matched_baselines}
    subperiods = {
        period: {
            name: _period_metrics(result, start, end)
            for name, result in all_results.items()
        }
        for period, (start, end) in periods.items()
    }
    return V8Evaluation(
        decisions=decisions,
        strategies=strategies,
        baselines=baselines,
        matched_baselines=matched_baselines,
        subperiods=subperiods,
    )
