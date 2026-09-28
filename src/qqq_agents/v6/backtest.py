"""Backtest engine for bounded leveraged QQQ exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.backtesting.engine import BacktestResult
from qqq_agents.backtesting.metrics import calculate_metrics


def run_exposure_backtest(
    close: pd.Series,
    desired_position: pd.Series,
    *,
    maximum_exposure: float,
    transaction_cost_bps: float,
    borrowing_spread_bps: float,
    cash_return: pd.Series,
    periods_per_year: int = 252,
) -> BacktestResult:
    if maximum_exposure < 1 or maximum_exposure > 1.25:
        raise ValueError("V6 maximum exposure must remain in [1, 1.25]")
    aligned_close, signal = close.align(desired_position, join="inner")
    if aligned_close.empty:
        raise ValueError("No overlapping V6 observations")
    if not signal.dropna().between(0, maximum_exposure).all():
        raise ValueError("V6 desired exposure exceeds its declared bounds")
    aligned_cash = cash_return.reindex(aligned_close.index).ffill().fillna(0.0)
    if not np.isfinite(aligned_cash).all():
        raise ValueError("V6 cash returns must be finite")

    raw_asset_return = aligned_close.pct_change()
    asset_return = raw_asset_return.fillna(0.0)
    applied = signal.shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(applied.abs())
    transaction_cost = turnover * transaction_cost_bps / 10_000
    borrowed = (applied - 1).clip(lower=0)
    spread_daily = (1 + borrowing_spread_bps / 10_000) ** (1 / periods_per_year) - 1
    financing_cost = borrowed * (aligned_cash + spread_daily)
    cash_contribution = (1 - applied.clip(upper=1)) * aligned_cash
    strategy_return = (
        applied * asset_return + cash_contribution - financing_cost - transaction_cost
    )
    history = pd.DataFrame(
        {
            "close": aligned_close,
            "desired_position": signal,
            "applied_position": applied,
            "asset_return": asset_return,
            "cash_return": aligned_cash,
            "cash_contribution": cash_contribution,
            "borrowed_fraction": borrowed,
            "financing_cost": financing_cost,
            "turnover": turnover,
            "cost": transaction_cost,
            "strategy_return": strategy_return,
            "equity": (1 + strategy_return).cumprod(),
        }
    )
    return BacktestResult(
        history=history,
        metrics=calculate_metrics(
            returns=strategy_return,
            turnover=turnover,
            asset_returns=raw_asset_return,
            positions=applied,
            periods_per_year=periods_per_year,
        ),
    )
