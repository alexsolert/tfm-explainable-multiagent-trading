"""Deterministic V8 coordinator and human-readable decision traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.v8.config import ProfileConfig


def _explanation(row: pd.Series, state: str, target: float) -> str:
    return (
        f"Estado {state}; tendencia {row['trend_score']:.0%}; "
        f"volatilidad anualizada {row['forecast_volatility']:.1%}; "
        f"drawdown a 252 sesiones {row['drawdown_252']:.1%}; "
        f"exposición propuesta {target:.0%}."
    )


def coordinate_profile(
    signals: pd.DataFrame, profile: ProfileConfig
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for timestamp, row in signals.iterrows():
        trend = float(row["trend_score"])
        volatility = max(float(row["forecast_volatility"]), 1e-6)
        severe_volatility = float(row["volatility_risk"]) >= 1.0
        high_volatility = float(row["volatility_risk"]) >= 0.5
        if severe_volatility:
            target = profile.severe_risk_exposure
            state = "SEVERE_RISK"
        elif trend < 0.5:
            target = profile.defensive_exposure
            state = "DEFENSIVE"
        elif profile.volatility_target is not None:
            target = float(
                np.clip(
                    profile.volatility_target / volatility,
                    profile.minimum_exposure,
                    profile.maximum_exposure,
                )
            )
            state = "VOLATILITY_MANAGED"
        elif high_volatility:
            target = profile.defensive_exposure
            state = "VOLATILITY_GUARD"
        else:
            target = profile.maximum_exposure
            state = "GROWTH"
        target = float(np.clip(target, 0, profile.maximum_exposure))
        confidence = float(
            np.clip(
                0.5
                + 0.5 * abs(trend - 0.5) * 2
                + 0.15 * float(severe_volatility),
                0,
                1,
            )
        )
        records.append(
            {
                "date": timestamp,
                "profile": profile.name,
                "profile_label": profile.label,
                "desired_position": target,
                "policy_state": state,
                "confidence": confidence,
                "explanation": _explanation(row, state, target),
            }
        )
    return pd.DataFrame.from_records(records).set_index("date")


def decision_action(current: float, previous: float, tolerance: float = 0.05) -> str:
    change = current - previous
    if change > tolerance:
        return "INCREASE"
    if change < -tolerance:
        return "REDUCE"
    return "HOLD"
