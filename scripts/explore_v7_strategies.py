"""Explore parsimonious leveraged QQQ trend policies without touching the V6 freeze."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_agents.backtesting.metrics import calculate_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "v6_daily.csv"
SELECTION = ("2004-01-01", "2022-12-31")
RETROSPECTIVE = ("2023-01-01", "2026-08-31")
BLOCKS = (
    ("early", "2004-01-01", "2009-12-31"),
    ("middle", "2010-01-01", "2016-12-31"),
    ("late", "2017-01-01", "2022-12-31"),
)


def backtest(
    close: pd.Series,
    cash: pd.Series,
    desired: pd.Series,
    *,
    cost_bps: float = 10.0,
    borrowing_spread_bps: float = 150.0,
) -> pd.DataFrame:
    asset_return = close.pct_change().fillna(0.0)
    applied = desired.shift(1).fillna(0.0)
    turnover = applied.diff().abs().fillna(applied.abs())
    trading_cost = turnover * cost_bps / 10_000
    borrowed = (applied - 1).clip(lower=0)
    spread_daily = (1 + borrowing_spread_bps / 10_000) ** (1 / 252) - 1
    financing = borrowed * (cash + spread_daily)
    cash_contribution = (1 - applied.clip(upper=1)) * cash
    strategy_return = (
        applied * asset_return + cash_contribution - financing - trading_cost
    )
    return pd.DataFrame(
        {
            "strategy_return": strategy_return,
            "asset_return": asset_return,
            "applied_position": applied,
            "turnover": turnover,
        }
    )


def metrics(history: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sample = history.loc[start:end]
    return calculate_metrics(
        returns=sample["strategy_return"],
        turnover=sample["turnover"],
        asset_returns=sample["asset_return"],
        positions=sample["applied_position"],
        periods_per_year=252,
    )


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["date"], index_col="date").sort_index()
    close = data["close"]
    cash = data["cash_return"].fillna(0.0)
    benchmark = backtest(close, cash, pd.Series(1.0, index=data.index))
    benchmark_selection = metrics(benchmark, *SELECTION)
    benchmark_retrospective = metrics(benchmark, *RETROSPECTIVE)

    rows: list[dict[str, object]] = []
    for window, maximum, defensive, buffer, guard in product(
        (100, 150, 200, 250),
        (1.25, 1.5, 1.75, 2.0),
        (0.0, 0.25, 0.5, 0.7, 1.0),
        (0.0, 0.01, 0.02),
        (None, 0.40, 0.35),
    ):
        average = close.rolling(window).mean()
        weekly_close = close.resample("W-FRI").last().reindex(close.index, method="ffill")
        weekly_average = average.resample("W-FRI").last().reindex(close.index, method="ffill")
        favourable = weekly_close > weekly_average * (1 + buffer)
        desired = pd.Series(np.where(favourable, maximum, defensive), index=data.index)
        if guard is not None:
            desired = desired.mask(data["volatility_20"] > guard, defensive)
        history = backtest(close, cash, desired)
        selection = metrics(history, *SELECTION)
        retrospective = metrics(history, *RETROSPECTIVE)
        block_values = [metrics(history, start, end) for _, start, end in BLOCKS]
        block_benchmarks = [metrics(benchmark, start, end) for _, start, end in BLOCKS]
        rows.append(
            {
                "window": window,
                "maximum": maximum,
                "defensive": defensive,
                "buffer": buffer,
                "guard": guard,
                "selection_return": selection["annualized_return"],
                "selection_sharpe": selection["sharpe_ratio"],
                "selection_drawdown": selection["maximum_drawdown"],
                "selection_return_gap": selection["annualized_return"]
                - benchmark_selection["annualized_return"],
                "selection_sharpe_gap": selection["sharpe_ratio"]
                - benchmark_selection["sharpe_ratio"],
                "worst_block_return_gap": min(
                    value["annualized_return"] - base["annualized_return"]
                    for value, base in zip(block_values, block_benchmarks, strict=True)
                ),
                "block_return_win_rate": np.mean(
                    [
                        value["annualized_return"] >= base["annualized_return"]
                        for value, base in zip(block_values, block_benchmarks, strict=True)
                    ]
                ),
                "retrospective_return": retrospective["annualized_return"],
                "retrospective_sharpe": retrospective["sharpe_ratio"],
                "retrospective_drawdown": retrospective["maximum_drawdown"],
                "retrospective_return_gap": retrospective["annualized_return"]
                - benchmark_retrospective["annualized_return"],
            }
        )
    results = pd.DataFrame(rows)
    eligible = results.loc[
        (results["selection_return_gap"] > 0)
        & (results["selection_sharpe_gap"] > 0)
        & (results["selection_drawdown"] >= benchmark_selection["maximum_drawdown"])
        & (results["block_return_win_rate"] >= 2 / 3)
    ].copy()
    eligible["robust_score"] = (
        2 * eligible["selection_return_gap"]
        + eligible["selection_sharpe_gap"]
        + 0.5 * eligible["worst_block_return_gap"]
    )
    columns = [
        "window",
        "maximum",
        "defensive",
        "buffer",
        "guard",
        "selection_return",
        "selection_sharpe",
        "selection_drawdown",
        "selection_return_gap",
        "worst_block_return_gap",
        "block_return_win_rate",
        "retrospective_return",
        "retrospective_sharpe",
        "retrospective_drawdown",
        "retrospective_return_gap",
        "robust_score",
    ]
    print("Benchmark selection:", benchmark_selection)
    print("Benchmark retrospective:", benchmark_retrospective)
    print("Eligible:", len(eligible), "of", len(results))
    print(eligible.nlargest(30, "robust_score")[columns].to_string(index=False))


if __name__ == "__main__":
    main()
