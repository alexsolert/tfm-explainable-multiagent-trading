"""Transparent V8 agents evaluated at weekly decision times."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.v8.config import SignalConfig


def build_agent_signals(frame: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    """Return daily-aligned outputs whose values update only at a weekly close."""

    close = frame["close"].sort_index()
    components: dict[str, pd.Series] = {}
    for sessions in config.moving_average_sessions:
        components[f"above_sma_{sessions}"] = close.gt(close.rolling(sessions).mean())
    for sessions in config.momentum_sessions:
        components[f"positive_momentum_{sessions}"] = close.pct_change(sessions).gt(0)
    trend_components = pd.DataFrame(components, index=close.index)
    daily_trend = trend_components.mean(axis=1)
    volatility = frame["volatility_20"].reindex(close.index)
    drawdown = frame["drawdown_252"].reindex(close.index)
    relative_strength = frame.get(
        "qqq_relative_spy_20", pd.Series(0.0, index=close.index)
    ).reindex(close.index)

    weekly = pd.DataFrame(
        {
            "trend_score": daily_trend.resample("W-FRI").last(),
            "forecast_volatility": volatility.resample("W-FRI").last(),
            "drawdown_252": drawdown.resample("W-FRI").last(),
            "relative_strength_20": relative_strength.resample("W-FRI").last(),
        }
    )
    weekly["signal_as_of"] = weekly.index
    weekly["trend_vote"] = 2 * weekly["trend_score"] - 1
    weekly["volatility_risk"] = np.select(
        [
            weekly["forecast_volatility"] >= config.severe_volatility,
            weekly["forecast_volatility"] >= config.high_volatility,
        ],
        [1.0, 0.5],
        default=0.0,
    )
    weekly["drawdown_risk"] = np.select(
        [
            weekly["drawdown_252"] <= config.severe_drawdown,
            weekly["drawdown_252"] <= config.moderate_drawdown,
        ],
        [1.0, 0.5],
        default=0.0,
    )
    weekly["relative_strength_vote"] = np.sign(
        weekly["relative_strength_20"].fillna(0.0)
    )
    weekly["risk_score"] = weekly[
        ["volatility_risk", "drawdown_risk"]
    ].max(axis=1)
    weekly["committee_score"] = (
        0.70 * weekly["trend_vote"]
        + 0.10 * weekly["relative_strength_vote"]
        - 0.20 * weekly["risk_score"]
    )
    return weekly.reindex(close.index, method="ffill")
