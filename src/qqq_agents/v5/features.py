"""Volatility-normalised multi-horizon targets for V5."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.v4.features import RISK_FEATURES as V4_RISK_FEATURES
from qqq_agents.v4.features import build_daily_research_frame
from qqq_agents.v5.config import RiskConfig

RISK_FEATURES = (
    *V4_RISK_FEATURES,
    "return_skew_20",
    "breadth_positive_20",
    "qqq_relative_spy_20",
    "semiconductor_momentum_20",
)

VOLATILITY_FEATURES = (
    "log_rv_1",
    "log_rv_5",
    "log_rv_20",
    "downside_vol_20",
    "vix_log",
    "vix_term_slope",
    "credit_momentum_20",
)


def _future_path_return(close: pd.Series, horizon: int) -> pd.DataFrame:
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
    result = np.sqrt(squares.mean(axis=1) * 252)
    return result.where(complete)


def build_v5_research_frame(
    markets: dict[str, pd.DataFrame],
    annual_yield_percent: pd.Series,
    *,
    risk_config: RiskConfig,
) -> pd.DataFrame:
    frame = build_daily_research_frame(
        markets,
        annual_yield_percent,
        prediction_horizon=max(risk_config.horizons),
        risk_event_threshold=-0.04,
    )
    close = frame["close"]
    returns = frame["return_1d"]
    dates = pd.Series(frame.index, index=frame.index)
    for horizon, multiplier in zip(
        risk_config.horizons,
        risk_config.volatility_multipliers,
        strict=True,
    ):
        path = _future_path_return(close, horizon)
        complete = path.notna().sum(axis=1) == horizon
        future_drawdown = path.min(axis=1).where(complete)
        dynamic_threshold = -multiplier * frame["rv_20"] * np.sqrt(horizon / 252)
        target = (future_drawdown <= dynamic_threshold).astype(float).where(complete)
        frame[f"future_drawdown_{horizon}"] = future_drawdown
        frame[f"dynamic_threshold_{horizon}"] = dynamic_threshold
        frame[f"target_tail_{horizon}"] = target
        frame[f"target_end_date_{horizon}"] = dates.shift(-horizon)
        frame[f"future_realized_vol_{horizon}"] = _future_realized_volatility(
            returns, horizon
        )
        frame[f"vol_target_end_date_{horizon}"] = dates.shift(-horizon)
    return frame
