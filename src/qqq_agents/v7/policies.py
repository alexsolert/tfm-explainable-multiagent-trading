"""Point-in-time, weekly V7 exposure policies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.v7.config import PolicyConfig


def build_v7_policy(frame: pd.DataFrame, config: PolicyConfig) -> pd.DataFrame:
    """Build a daily exposure series from signals fixed at each weekly close."""

    close = frame["close"].sort_index()
    volatility = frame["volatility_20"].reindex(close.index)
    average = close.rolling(config.moving_average_sessions).mean()
    weekly_close = close.resample("W-FRI").last()
    weekly_average = average.resample("W-FRI").last()
    weekly_volatility = volatility.resample("W-FRI").last()
    favourable = weekly_close > weekly_average

    if config.family == "trend_guard":
        guarded = weekly_volatility > float(config.volatility_guard)
        weekly_exposure = pd.Series(
            np.where(
                favourable & ~guarded,
                config.maximum_exposure,
                config.defensive_exposure,
            ),
            index=weekly_close.index,
        )
        state = pd.Series(
            np.select(
                [guarded, favourable],
                ["VOLATILITY_GUARD", "TREND_FAVOURABLE"],
                default="TREND_DEFENSIVE",
            ),
            index=weekly_close.index,
        )
    else:
        scaled = (float(config.volatility_target) / weekly_volatility).clip(
            lower=float(config.minimum_exposure), upper=config.maximum_exposure
        )
        weekly_exposure = scaled.where(favourable, config.defensive_exposure)
        state = pd.Series(
            np.where(favourable, "VOLATILITY_SCALED", "TREND_DEFENSIVE"),
            index=weekly_close.index,
        )

    return pd.DataFrame(
        {
            "desired_position": weekly_exposure.reindex(
                close.index, method="ffill"
            ).fillna(config.defensive_exposure),
            "policy_state": state.reindex(close.index, method="ffill").fillna(
                "WARMUP"
            ),
        },
        index=close.index,
    )
