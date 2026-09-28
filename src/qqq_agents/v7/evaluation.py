"""Evaluation of predeclared V7 policies against fair leverage controls."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qqq_agents.backtesting.engine import BacktestResult
from qqq_agents.backtesting.metrics import calculate_metrics
from qqq_agents.v7.backtest import run_v7_backtest
from qqq_agents.v7.config import V7Config
from qqq_agents.v7.policies import build_v7_policy


@dataclass(frozen=True)
class V7Evaluation:
    policies: dict[str, BacktestResult]
    policy_details: dict[str, pd.DataFrame]
    baselines: dict[str, BacktestResult]
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


def evaluate_v7(frame: pd.DataFrame, config: V7Config) -> V7Evaluation:
    close = frame["close"].sort_index()
    cash = frame["cash_return"].reindex(close.index).fillna(0.0)
    details = {policy.name: build_v7_policy(frame, policy) for policy in config.policies}
    policies = {
        policy.name: run_v7_backtest(
            close,
            details[policy.name]["desired_position"],
            maximum_exposure=policy.maximum_exposure,
            transaction_cost_bps=config.costs.transaction_cost_bps,
            borrowing_spread_bps=config.costs.borrowing_spread_bps,
            cash_return=cash,
        )
        for policy in config.policies
    }
    baseline_exposures = (1.0, 1.5, 1.75)
    baselines = {
        f"constant_{int(exposure * 100)}": run_v7_backtest(
            close,
            pd.Series(exposure, index=close.index),
            maximum_exposure=exposure,
            transaction_cost_bps=config.costs.transaction_cost_bps,
            borrowing_spread_bps=config.costs.borrowing_spread_bps,
            cash_return=cash,
        )
        for exposure in baseline_exposures
    }
    periods = {
        "selection": (config.period.selection_start, config.period.selection_end),
        "retrospective": (
            config.period.retrospective_start,
            config.period.retrospective_end,
        ),
    }
    combined = {**policies, **baselines}
    subperiods = {
        period: {
            name: _period_metrics(result, start, end)
            for name, result in combined.items()
        }
        for period, (start, end) in periods.items()
    }
    return V7Evaluation(
        policies=policies,
        policy_details=details,
        baselines=baselines,
        subperiods=subperiods,
    )
