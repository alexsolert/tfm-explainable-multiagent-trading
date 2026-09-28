"""Long-history, point-in-time features and multi-horizon targets for V6."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.features import build_features
from qqq_agents.v3.features import cash_returns_from_yield
from qqq_agents.v6.config import ReturnModelConfig, RiskConfig

CORE_FEATURES = (
    "return_1d",
    "distance_sma_20",
    "distance_sma_50",
    "distance_sma_200",
    "rsi_14",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "rv_5",
    "rv_20",
    "rv_60",
    "downside_vol_20",
    "return_skew_20",
    "atr_14_pct",
    "drawdown_252",
    "volume_zscore_20",
    "vix_level",
    "vix_change_5",
    "qqq_relative_spy_20",
)

RISK_FEATURES = (
    "rv_5",
    "rv_20",
    "rv_60",
    "downside_vol_20",
    "return_skew_20",
    "atr_14_pct",
    "drawdown_252",
    "volume_zscore_20",
    "vix_level",
    "vix_change_5",
    "qqq_relative_spy_20",
)

VOLATILITY_FEATURES = (
    "log_rv_1",
    "log_rv_5",
    "log_rv_20",
    "downside_vol_20",
    "vix_log",
)


def _future_paths(close: pd.Series, horizon: int) -> pd.DataFrame:
    return pd.concat(
        [(close.shift(-step) / close - 1).rename(step) for step in range(1, horizon + 1)],
        axis=1,
    )


def _future_realized_volatility(returns: pd.Series, horizon: int) -> pd.Series:
    squares = pd.concat(
        [returns.shift(-step).pow(2).rename(step) for step in range(1, horizon + 1)],
        axis=1,
    )
    complete = squares.notna().sum(axis=1) == horizon
    return np.sqrt(squares.mean(axis=1) * 252).where(complete)


def build_v6_research_frame(
    markets: dict[str, pd.DataFrame],
    annual_yield_percent: pd.Series,
    *,
    risk_config: RiskConfig,
    return_config: ReturnModelConfig,
) -> pd.DataFrame:
    required = {"qqq", "spy", "vix"}
    missing = required - set(markets)
    if missing:
        raise ValueError(f"V6 is missing core market snapshots: {sorted(missing)}")
    maximum_horizon = max(*risk_config.horizons, *return_config.horizons)
    frame = build_features(markets["qqq"], prediction_horizon=maximum_horizon)
    index = frame.index
    close = frame["close"]
    returns = frame["return_1d"]
    spy = markets["spy"]["close"].reindex(index).ffill()
    vix = markets["vix"]["close"].reindex(index).ffill()

    frame["rv_1"] = returns.abs() * np.sqrt(252)
    for window in (5, 20, 60):
        frame[f"rv_{window}"] = returns.rolling(window).std() * np.sqrt(252)
    negative = returns.clip(upper=0)
    frame["downside_vol_20"] = np.sqrt(negative.pow(2).rolling(20).mean() * 252)
    frame["return_skew_20"] = returns.rolling(20).skew()
    frame["log_rv_1"] = np.log(frame["rv_1"].clip(lower=1e-5))
    frame["log_rv_5"] = np.log(frame["rv_5"].clip(lower=1e-5))
    frame["log_rv_20"] = np.log(frame["rv_20"].clip(lower=1e-5))
    frame["vix_level"] = vix
    frame["vix_log"] = np.log(vix.clip(lower=1e-5))
    frame["vix_change_5"] = vix.pct_change(5)
    frame["qqq_relative_spy_20"] = close.pct_change(20) - spy.pct_change(20)
    frame["cash_return"] = cash_returns_from_yield(annual_yield_percent, index)

    dates = pd.Series(index, index=index)
    for horizon, multiplier in zip(
        risk_config.horizons,
        risk_config.volatility_multipliers,
        strict=True,
    ):
        paths = _future_paths(close, horizon)
        complete = paths.notna().sum(axis=1) == horizon
        drawdown = paths.min(axis=1).where(complete)
        threshold = -multiplier * frame["rv_20"] * np.sqrt(horizon / 252)
        frame[f"future_drawdown_{horizon}"] = drawdown
        frame[f"dynamic_threshold_{horizon}"] = threshold
        frame[f"target_tail_{horizon}"] = (drawdown <= threshold).astype(float).where(complete)
        frame[f"target_end_date_{horizon}"] = dates.shift(-horizon)
        frame[f"future_realized_vol_{horizon}"] = _future_realized_volatility(
            returns, horizon
        )
        frame[f"vol_target_end_date_{horizon}"] = dates.shift(-horizon)

    for horizon in return_config.horizons:
        forward = close.shift(-horizon) / close - 1
        scale = frame["rv_20"] * np.sqrt(horizon / 252)
        frame[f"forward_return_{horizon}"] = forward
        frame[f"target_normalized_return_{horizon}"] = forward / scale.clip(lower=1e-4)
        frame[f"return_target_end_date_{horizon}"] = dates.shift(-horizon)
    return frame
