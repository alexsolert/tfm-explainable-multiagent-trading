"""Indicadores, objetivos y muestreo semanal del experimento."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "return_1d",
    "log_return_1d",
    "distance_sma_20",
    "distance_sma_50",
    "distance_sma_200",
    "ema_spread_12_26",
    "macd",
    "macd_signal",
    "macd_histogram",
    "rsi_14",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "atr_14_pct",
    "drawdown_252",
    "volume_zscore_20",
)

TARGET_COLUMNS = ("forward_return", "future_min_return", "target_up", "target_risk")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.where(average_loss.ne(0), 100.0)


def build_features(
    market: pd.DataFrame,
    *,
    prediction_horizon: int = 5,
    risk_event_threshold: float = -0.03,
) -> pd.DataFrame:
    """Build backward-looking features and explicitly separated future labels."""

    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    if not market.index.is_monotonic_increasing or market.index.has_duplicates:
        raise ValueError("Market observations must be unique and chronologically ordered")

    result = market.copy().astype(float)
    close = result["close"]
    result["return_1d"] = close.pct_change()
    result["log_return_1d"] = np.log(close).diff()

    for window in (20, 50, 200):
        sma = close.rolling(window=window, min_periods=window).mean()
        result[f"distance_sma_{window}"] = close / sma - 1

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["ema_spread_12_26"] = ema_12 / ema_26 - 1
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    result["rsi_14"] = _rsi(close)

    for window in (5, 20, 60):
        result[f"momentum_{window}"] = close.pct_change(window)

    result["volatility_20"] = result["return_1d"].rolling(20).std() * np.sqrt(252)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14_pct"] = true_range.rolling(14).mean() / close
    result["drawdown_252"] = close / close.rolling(252, min_periods=20).max() - 1
    volume_mean = result["volume"].rolling(20).mean()
    volume_std = result["volume"].rolling(20).std()
    result["volume_zscore_20"] = (result["volume"] - volume_mean) / volume_std

    result["forward_return"] = close.shift(-prediction_horizon) / close - 1
    future_paths = pd.concat(
        [close.shift(-step) / close - 1 for step in range(1, prediction_horizon + 1)],
        axis=1,
    )
    result["future_min_return"] = future_paths.min(axis=1, skipna=False)

    valid_target = result["forward_return"].notna()
    result["target_up"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[valid_target, "target_up"] = (result.loc[valid_target, "forward_return"] > 0).astype(
        "int8"
    )
    valid_risk = result["future_min_return"].notna()
    result["target_risk"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[valid_risk, "target_risk"] = (
        result.loc[valid_risk, "future_min_return"] <= risk_event_threshold
    ).astype("int8")

    return result


def sample_decisions(frame: pd.DataFrame, frequency: str = "W-FRI") -> pd.DataFrame:
    """Select the last real market observation in every decision period."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Decision sampling requires a DatetimeIndex")
    return frame.groupby(pd.Grouper(freq=frequency)).tail(1).copy()
