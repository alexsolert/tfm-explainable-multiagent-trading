"""Daily point-in-time features for volatility, risk, trend and direction agents."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.features import build_features
from qqq_agents.v3.features import cash_returns_from_yield

RISK_FEATURES = (
    "rv_5",
    "rv_20",
    "rv_60",
    "downside_vol_20",
    "atr_14_pct",
    "drawdown_252",
    "vix_level",
    "vix_change_5",
    "vix_term_slope",
    "credit_momentum_20",
    "breadth_above_sma_200",
)

DIRECTION_FEATURES = (
    "distance_sma_20",
    "distance_sma_50",
    "distance_sma_200",
    "rsi_14",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "qqq_relative_spy_20",
    "semiconductor_momentum_20",
    "small_cap_momentum_20",
    "credit_momentum_20",
    "breadth_positive_20",
    "vix_term_slope",
)

VOLATILITY_FEATURES = (
    "log_rv_1",
    "log_rv_5",
    "log_rv_20",
    "vix_log",
    "vix_term_slope",
)


def _future_realized_volatility(returns: pd.Series, horizon: int = 20) -> pd.Series:
    future_squares = pd.concat(
        [returns.shift(-step).pow(2) for step in range(1, horizon + 1)], axis=1
    )
    return np.sqrt(future_squares.mean(axis=1) * 252)


def build_daily_research_frame(
    markets: dict[str, pd.DataFrame],
    annual_yield_percent: pd.Series,
    *,
    prediction_horizon: int = 5,
    risk_event_threshold: float = -0.03,
) -> pd.DataFrame:
    required = {"qqq", "spy", "iwm", "smh", "tlt", "vix", "vix3m", "hyg", "lqd", "qqqe"}
    missing = required - set(markets)
    if missing:
        raise ValueError(f"V4 is missing market snapshots: {sorted(missing)}")
    qqq = markets["qqq"].copy()
    result = build_features(
        qqq,
        prediction_horizon=prediction_horizon,
        risk_event_threshold=risk_event_threshold,
    )
    index = result.index
    closes = pd.DataFrame(index=index)
    for alias, market in markets.items():
        closes[alias] = market["close"].reindex(index).ffill()

    daily_return = result["return_1d"]
    result["rv_1"] = daily_return.abs() * np.sqrt(252)
    for window in (5, 20, 60):
        result[f"rv_{window}"] = daily_return.rolling(window).std() * np.sqrt(252)
    negative = daily_return.clip(upper=0)
    result["downside_vol_20"] = np.sqrt(negative.pow(2).rolling(20).mean() * 252)
    result["return_skew_20"] = daily_return.rolling(20).skew()
    result["log_rv_1"] = np.log(result["rv_1"].clip(lower=1e-5))
    result["log_rv_5"] = np.log(result["rv_5"].clip(lower=1e-5))
    result["log_rv_20"] = np.log(result["rv_20"].clip(lower=1e-5))

    result["vix_level"] = closes["vix"]
    result["vix_log"] = np.log(closes["vix"].clip(lower=1e-5))
    result["vix_change_5"] = closes["vix"].pct_change(5)
    result["vix_term_slope"] = closes["vix"] / closes["vix3m"] - 1
    result["qqq_relative_spy_20"] = closes["qqq"].pct_change(20) - closes["spy"].pct_change(20)
    result["semiconductor_momentum_20"] = closes["smh"].pct_change(20)
    result["small_cap_momentum_20"] = closes["iwm"].pct_change(20)
    credit_ratio = closes["hyg"] / closes["lqd"]
    result["credit_momentum_20"] = credit_ratio.pct_change(20)

    breadth_assets = ("qqq", "spy", "iwm", "smh", "qqqe")
    positive = pd.concat(
        [(closes[name].pct_change(20) > 0).rename(name) for name in breadth_assets], axis=1
    )
    above_long_average = pd.concat(
        [
            (closes[name] > closes[name].rolling(200).mean()).rename(name)
            for name in breadth_assets
        ],
        axis=1,
    )
    result["breadth_positive_20"] = positive.mean(axis=1)
    result["breadth_above_sma_200"] = above_long_average.mean(axis=1)
    result["future_realized_vol_20"] = _future_realized_volatility(daily_return, 20)
    result["vol_target_end_date"] = pd.Series(index, index=index).shift(-20)
    result["cash_return"] = cash_returns_from_yield(annual_yield_percent, index)
    return result
