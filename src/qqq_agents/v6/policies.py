"""Risk-managed exposure policies and development-only selection for V6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qqq_agents.v6.config import AllocationConfig, SelectionConfig


@dataclass(frozen=True)
class ExposureSelection:
    selected_policy: str
    leaderboard: pd.DataFrame
    constraints_met: bool


def policy_name(maximum_exposure: float, family: str = "joint") -> str:
    return f"{family}_{int(round(maximum_exposure * 100)):03d}"


def build_exposure_policy(
    decisions: pd.DataFrame,
    *,
    maximum_exposure: float,
    config: AllocationConfig,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    previous = 1.0
    for timestamp, row in decisions.iterrows():
        risk = float(row["risk_score"])
        trend = float(row["weekly_trend_score"])
        return_score = float(row["return_score"])
        forecast_volatility = max(float(row["forecast_volatility"]), 1e-4)
        volatility_target = float(
            np.clip(
                config.target_volatility / forecast_volatility,
                config.minimum_exposure,
                maximum_exposure,
            )
        )
        risk_cap = float(
            np.clip(
                1 - config.risk_sensitivity * max(risk - config.risk_low, 0),
                config.minimum_exposure,
                1.0,
            )
        )
        favourable = (
            trend >= config.favourable_trend
            and return_score >= config.minimum_return_score
            and risk < config.risk_low
        )
        if risk >= config.risk_high:
            target = min(config.defensive_cap, volatility_target)
            state = "DEFENSIVE"
        elif trend < 0.5:
            target = min(config.bearish_trend_cap, risk_cap, volatility_target)
            state = "BEARISH"
        elif favourable:
            target = volatility_target
            state = "FAVOURABLE"
        else:
            target = min(1.0, risk_cap, volatility_target)
            state = "NORMAL"

        target = float(np.clip(target, config.minimum_exposure, maximum_exposure))
        if target > previous:
            target = min(target, previous + config.maximum_daily_reentry)
        if abs(target - previous) < config.minimum_rebalance:
            target = previous
        records.append(
            {
                "date": timestamp,
                "desired_position": target,
                "policy_state": state,
                "favourable": favourable,
                "risk_cap": risk_cap,
                "volatility_target": volatility_target,
                "maximum_exposure": maximum_exposure,
            }
        )
        previous = target
    return pd.DataFrame.from_records(records).set_index("date")


def build_trend_exposure_policy(
    decisions: pd.DataFrame,
    *,
    maximum_exposure: float,
    config: AllocationConfig,
) -> pd.DataFrame:
    """Parsimonious challenger: leverage the broad trend, de-risk below 50%."""

    records: list[dict[str, object]] = []
    for timestamp, row in decisions.iterrows():
        trend = float(row["weekly_trend_score"])
        if trend >= 0.5:
            target, state = maximum_exposure, "TREND_FAVOURABLE"
        else:
            target, state = config.bearish_trend_cap, "TREND_BEARISH"
        records.append(
            {
                "date": timestamp,
                "desired_position": float(target),
                "policy_state": state,
                "favourable": trend >= 0.5,
                "risk_cap": np.nan,
                "volatility_target": np.nan,
                "maximum_exposure": maximum_exposure,
            }
        )
    return pd.DataFrame.from_records(records).set_index("date")


def build_guarded_trend_policy(
    decisions: pd.DataFrame,
    *,
    maximum_exposure: float,
    config: AllocationConfig,
) -> pd.DataFrame:
    """Trend exposure with a sparse veto under exceptional forecast volatility."""

    records: list[dict[str, object]] = []
    for timestamp, row in decisions.iterrows():
        trend = float(row["weekly_trend_score"])
        forecast_volatility = float(row["forecast_volatility"])
        guarded = forecast_volatility > config.volatility_guard
        if guarded:
            target, state = config.guard_exposure, "VOLATILITY_GUARD"
        elif trend >= 0.5:
            target, state = maximum_exposure, "TREND_FAVOURABLE"
        else:
            target, state = config.bearish_trend_cap, "TREND_BEARISH"
        records.append(
            {
                "date": timestamp,
                "desired_position": float(target),
                "policy_state": state,
                "favourable": trend >= 0.5 and not guarded,
                "risk_cap": np.nan,
                "volatility_target": config.guard_exposure if guarded else np.nan,
                "maximum_exposure": maximum_exposure,
            }
        )
    return pd.DataFrame.from_records(records).set_index("date")


def select_exposure_policy(
    metrics: dict[str, dict[str, float]],
    *,
    benchmark: dict[str, float],
    config: SelectionConfig,
    block_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    benchmark_blocks: dict[str, dict[str, float]] | None = None,
) -> ExposureSelection:
    rows = []
    benchmark_drawdown = abs(benchmark["maximum_drawdown"])
    for name, values in metrics.items():
        rows.append(
            {
                "policy": name,
                **values,
                "return_improvement": (
                    values["annualized_return"] - benchmark["annualized_return"]
                ),
                "sharpe_improvement": values["sharpe_ratio"] - benchmark["sharpe_ratio"],
                "drawdown_ratio": (
                    abs(values["maximum_drawdown"]) / benchmark_drawdown
                    if benchmark_drawdown
                    else 0.0
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index("policy")
    if block_metrics is not None and benchmark_blocks is not None:
        for policy in frame.index:
            comparisons = [
                {
                    "return_gap": values["annualized_return"]
                    - benchmark_blocks[block]["annualized_return"],
                    "sharpe_gap": values["sharpe_ratio"]
                    - benchmark_blocks[block]["sharpe_ratio"],
                    "drawdown_gap": values["maximum_drawdown"]
                    - benchmark_blocks[block]["maximum_drawdown"],
                }
                for block, values in block_metrics[str(policy)].items()
            ]
            frame.loc[policy, "worst_block_return_gap"] = min(
                value["return_gap"] for value in comparisons
            )
            frame.loc[policy, "worst_block_sharpe_gap"] = min(
                value["sharpe_gap"] for value in comparisons
            )
            frame.loc[policy, "worst_block_drawdown_gap"] = min(
                value["drawdown_gap"] for value in comparisons
            )
            frame.loc[policy, "block_return_win_rate"] = np.mean(
                [value["return_gap"] >= 0 for value in comparisons]
            )
            frame.loc[policy, "block_sharpe_win_rate"] = np.mean(
                [value["sharpe_gap"] >= 0 for value in comparisons]
            )
            frame.loc[policy, "block_drawdown_win_rate"] = np.mean(
                [value["drawdown_gap"] >= 0 for value in comparisons]
            )
    else:
        for column in (
            "worst_block_return_gap",
            "worst_block_sharpe_gap",
            "worst_block_drawdown_gap",
            "block_return_win_rate",
            "block_sharpe_win_rate",
            "block_drawdown_win_rate",
        ):
            frame[column] = 0.0
    constrained = frame.loc[
        (frame["return_improvement"] >= config.minimum_return_improvement)
        & (frame["sharpe_improvement"] >= config.minimum_sharpe_improvement)
        & (frame["drawdown_ratio"] <= config.maximum_drawdown_ratio)
    ]
    constraints_met = not constrained.empty
    pool = constrained if constraints_met else frame
    score = (
        2.0 * pool["annualized_return"]
        + pool["sharpe_ratio"]
        + 0.5 * pool["maximum_drawdown"]
        - 0.001 * pool["total_turnover"]
        + 0.50 * pool["worst_block_return_gap"]
        + 0.25 * pool["worst_block_sharpe_gap"]
        + 0.25 * pool["worst_block_drawdown_gap"]
    )
    selected = str(score.idxmax())
    frame["constraints_met"] = frame.index.isin(constrained.index)
    frame["selected"] = frame.index == selected
    return ExposureSelection(
        selected_policy=selected,
        leaderboard=frame.reset_index(),
        constraints_met=constraints_met,
    )
