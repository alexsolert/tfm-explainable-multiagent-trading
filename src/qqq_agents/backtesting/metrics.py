"""Metricas financieras del protocolo experimental."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_metrics(
    *,
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = 52,
) -> dict[str, float]:
    clean_returns = returns.fillna(0.0).astype(float)
    if clean_returns.empty:
        raise ValueError("At least one return is required")

    equity = (1 + clean_returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1)
    years = len(clean_returns) / periods_per_year
    annualized_return = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    annualized_volatility = float(clean_returns.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = (
        float(clean_returns.mean() / clean_returns.std(ddof=1) * math.sqrt(periods_per_year))
        if clean_returns.std(ddof=1) > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1
    maximum_drawdown = float(drawdown.min())
    total_turnover = float(turnover.fillna(0).sum()) if turnover is not None else 0.0

    values = {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": maximum_drawdown,
        "total_turnover": total_turnover,
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Calculated metrics contain non-finite values")
    return values
