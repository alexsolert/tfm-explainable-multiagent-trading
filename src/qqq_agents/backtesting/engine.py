"""Motor long-only sencillo con ejecucion desplazada y costes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_agents.backtesting.metrics import calculate_metrics


@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    close: pd.Series,
    desired_position: pd.Series,
    *,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 52,
    cash_return: pd.Series | None = None,
) -> BacktestResult:
    """Apply each decision to the following return, never to its own period.

    ``cash_return`` represents the return earned during each observation by the
    uninvested fraction.  Keeping it optional preserves the frozen V1/V2
    experiments, whose cash assumption was zero.
    """

    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")
    aligned_close, aligned_signal = close.align(desired_position, join="inner")
    if aligned_close.empty:
        raise ValueError("No overlapping observations between prices and positions")
    if not aligned_close.index.is_monotonic_increasing:
        raise ValueError("Backtest observations must be chronologically ordered")
    if not aligned_signal.dropna().between(0, 1).all():
        raise ValueError("Long-only desired positions must remain in [0, 1]")

    raw_asset_return = aligned_close.pct_change()
    asset_return = raw_asset_return.fillna(0.0)
    applied_position = aligned_signal.shift(1).fillna(0.0)
    turnover = applied_position.diff().abs().fillna(applied_position.abs())
    costs = turnover * transaction_cost_bps / 10_000
    if cash_return is None:
        aligned_cash_return = pd.Series(0.0, index=aligned_close.index)
    else:
        aligned_cash_return = cash_return.reindex(aligned_close.index).ffill().fillna(0.0)
        if not np.isfinite(aligned_cash_return).all():
            raise ValueError("cash_return must contain only finite values")
        if (aligned_cash_return <= -1).any():
            raise ValueError("cash_return cannot be less than or equal to -100%")
    strategy_return = (
        applied_position * asset_return
        + (1 - applied_position) * aligned_cash_return
        - costs
    )

    history = pd.DataFrame(
        {
            "close": aligned_close,
            "desired_position": aligned_signal,
            "applied_position": applied_position,
            "asset_return": asset_return,
            "cash_return": aligned_cash_return,
            "cash_contribution": (1 - applied_position) * aligned_cash_return,
            "turnover": turnover,
            "cost": costs,
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
            positions=applied_position,
            periods_per_year=periods_per_year,
        ),
    )
