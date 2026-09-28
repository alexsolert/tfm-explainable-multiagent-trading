"""Backtest for V7 exposure up to two times QQQ."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.backtesting.engine import BacktestResult
from qqq_agents.backtesting.metrics import calculate_metrics


def run_v7_backtest(
    close: pd.Series,
    desired_position: pd.Series,
    *,
    maximum_exposure: float,
    transaction_cost_bps: float,
    borrowing_spread_bps: float,
    cash_return: pd.Series,
) -> BacktestResult:
    if maximum_exposure < 1 or maximum_exposure > 2:
        raise ValueError("V7 maximum exposure must remain in [1, 2]")
    aligned_close, signal = close.align(desired_position, join="inner")
    if aligned_close.empty or not signal.dropna().between(0, maximum_exposure).all():
        raise ValueError("Invalid V7 exposure series")
    cash = cash_return.reindex(aligned_close.index).ffill().fillna(0.0)
    if not np.isfinite(cash).all():
        raise ValueError("V7 cash returns must be finite")

    asset_return = aligned_close.pct_change().fillna(0.0)
    applied = signal.shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(applied.abs())
    cost = turnover * transaction_cost_bps / 10_000
    borrowed = (applied - 1).clip(lower=0)
    spread = (1 + borrowing_spread_bps / 10_000) ** (1 / 252) - 1
    financing = borrowed * (cash + spread)
    cash_contribution = (1 - applied.clip(upper=1)) * cash
    strategy_return = applied * asset_return + cash_contribution - financing - cost
    history = pd.DataFrame(
        {
            "close": aligned_close,
            "desired_position": signal,
            "applied_position": applied,
            "asset_return": asset_return,
            "cash_return": cash,
            "financing_cost": financing,
            "turnover": turnover,
            "cost": cost,
            "strategy_return": strategy_return,
            "equity": (1 + strategy_return).cumprod(),
        }
    )
    return BacktestResult(
        history=history,
        metrics=calculate_metrics(
            returns=strategy_return,
            turnover=turnover,
            asset_returns=asset_return,
            positions=applied,
            periods_per_year=252,
        ),
    )
