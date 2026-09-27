"""Posiciones de referencia para validar el motor y contextualizar el MVP."""

from __future__ import annotations

import pandas as pd


def buy_and_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index, name="buy_and_hold")


def moving_average_crossover(
    close: pd.Series,
    *,
    fast_window: int = 50,
    slow_window: int = 200,
) -> pd.Series:
    if fast_window <= 0 or slow_window <= fast_window:
        raise ValueError("Expected 0 < fast_window < slow_window")
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    position = (fast > slow).astype(float)
    position.loc[slow.isna()] = 0.0
    return position.rename("moving_average_crossover")
