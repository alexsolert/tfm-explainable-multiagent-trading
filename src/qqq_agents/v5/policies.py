"""Deterministic policy candidates and Pareto selection for V5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_agents.v5.config import AllocationConfig, SelectionConfig


@dataclass(frozen=True)
class PolicySelection:
    selected_policy: str
    leaderboard: pd.DataFrame
    pareto_policies: tuple[str, ...]
    target_constraints_met: bool


def _volatility_cap(row: pd.Series, config: AllocationConfig) -> float:
    conservative_forecast = max(float(row["forecast_volatility"]), 1e-4)
    return float(
        np.clip(
            config.target_volatility / conservative_forecast,
            config.minimum_exposure,
            1.0,
        )
    )


def build_policy_positions(
    decisions: pd.DataFrame,
    *,
    policy: str,
    config: AllocationConfig,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    previous = 1.0
    state = "NORMAL"
    calm_days = 0
    for timestamp, row in decisions.iterrows():
        risk = float(row["risk_score"])
        trend = float(row["weekly_trend_score"])
        volatility_cap = _volatility_cap(row, config)
        risk_cap = float(
            np.clip(
                1 - config.risk_sensitivity * max(risk - config.risk_low, 0),
                config.minimum_exposure,
                1.0,
            )
        )
        trend_cap = 1.0 if trend >= 0.5 else config.bearish_trend_cap

        if policy == "risk_only":
            target = risk_cap
            state = "RISK"
        elif policy == "trend_only":
            target = trend_cap
            state = "TREND"
        elif policy == "minimum_caps":
            target = min(volatility_cap, risk_cap, trend_cap)
            state = "MINIMUM"
        elif policy == "confirmation":
            if risk >= config.risk_high and trend < 0.5:
                target, state = config.defensive_cap, "CONFIRMED_DEFENSIVE"
            elif risk >= config.risk_high or trend < 0.25:
                target, state = config.alert_cap, "UNCONFIRMED_ALERT"
            else:
                target, state = volatility_cap, "CONFIRMED_NORMAL"
        elif policy == "asymmetric_state_machine":
            defensive = risk >= config.risk_high or (risk >= config.risk_low and trend < 0.5)
            calm = risk < config.risk_low and trend >= 0.5
            if defensive:
                state = "DEFENSIVE"
                calm_days = 0
            elif state == "DEFENSIVE":
                state = "RECOVERY"
                calm_days = 1 if calm else 0
            elif state == "RECOVERY":
                calm_days = calm_days + 1 if calm else 0
                if calm_days >= config.recovery_confirmations:
                    state = "NORMAL"
            elif risk >= config.risk_low or trend < 0.5:
                state = "ALERT"
            else:
                state = "NORMAL"
            state_cap = {
                "NORMAL": 1.0,
                "ALERT": config.alert_cap,
                "DEFENSIVE": config.defensive_cap,
                "RECOVERY": config.recovery_cap,
            }[state]
            target = min(volatility_cap, state_cap)
        else:
            raise ValueError(f"Unknown V5 policy: {policy}")

        target = float(np.clip(target, config.minimum_exposure, 1.0))
        if target > previous:
            target = min(target, previous + config.maximum_daily_reentry)
        if abs(target - previous) < config.minimum_rebalance:
            target = previous
        records.append(
            {
                "date": timestamp,
                "desired_position": target,
                "policy_state": state,
                "risk_cap": risk_cap,
                "trend_cap": trend_cap,
                "volatility_cap": volatility_cap,
            }
        )
        previous = target
    return pd.DataFrame.from_records(records).set_index("date")


def pareto_select_policy(
    metrics: dict[str, dict[str, float]],
    *,
    benchmark: dict[str, float],
    config: SelectionConfig,
) -> PolicySelection:
    rows = []
    benchmark_return = benchmark["annualized_return"]
    benchmark_drawdown = abs(benchmark["maximum_drawdown"])
    for name, values in metrics.items():
        return_retention = (
            values["annualized_return"] / benchmark_return if benchmark_return else 0.0
        )
        drawdown_reduction = (
            1 - abs(values["maximum_drawdown"]) / benchmark_drawdown
            if benchmark_drawdown
            else 0.0
        )
        rows.append(
            {
                "policy": name,
                **values,
                "return_retention": return_retention,
                "drawdown_reduction": drawdown_reduction,
            }
        )
    frame = pd.DataFrame(rows).set_index("policy")
    objectives = pd.DataFrame(
        {
            "annualized_return": frame["annualized_return"],
            "sharpe_ratio": frame["sharpe_ratio"],
            "maximum_drawdown": frame["maximum_drawdown"],
            "negative_turnover": -frame["total_turnover"],
        }
    )
    pareto: list[str] = []
    for candidate in objectives.index:
        other = objectives.drop(index=candidate)
        dominated = ((other >= objectives.loc[candidate]).all(axis=1) & (
            other > objectives.loc[candidate]
        ).any(axis=1)).any()
        if not dominated:
            pareto.append(str(candidate))
    eligible = frame.loc[pareto]
    constrained = eligible.loc[
        (eligible["return_retention"] >= config.minimum_return_retention)
        & (eligible["drawdown_reduction"] >= config.target_drawdown_reduction)
        & (eligible["sharpe_ratio"] >= benchmark["sharpe_ratio"])
    ]
    constraints_met = not constrained.empty
    pool = constrained if constraints_met else eligible
    score = (
        pool["sharpe_ratio"]
        + 1.5 * pool["annualized_return"]
        + 0.5 * pool["maximum_drawdown"]
        - 0.002 * pool["total_turnover"]
    )
    selected = str(score.idxmax())
    frame["pareto"] = frame.index.isin(pareto)
    frame["selected"] = frame.index == selected
    return PolicySelection(
        selected_policy=selected,
        leaderboard=frame.reset_index(),
        pareto_policies=tuple(pareto),
        target_constraints_met=constraints_met,
    )
