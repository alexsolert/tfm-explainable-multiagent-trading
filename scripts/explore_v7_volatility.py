"""Explore weekly volatility-managed trend exposure as a V7 challenger."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from explore_v7_strategies import (
    BLOCKS,
    DATA,
    RETROSPECTIVE,
    SELECTION,
    backtest,
    metrics,
)


def main() -> None:
    data = pd.read_csv(DATA, parse_dates=["date"], index_col="date").sort_index()
    close = data["close"]
    cash = data["cash_return"].fillna(0.0)
    benchmark = backtest(close, cash, pd.Series(1.0, index=data.index))
    benchmark_selection = metrics(benchmark, *SELECTION)
    benchmark_retrospective = metrics(benchmark, *RETROSPECTIVE)
    block_benchmarks = [metrics(benchmark, start, end) for _, start, end in BLOCKS]
    rows: list[dict[str, object]] = []

    for window, target, maximum, floor, defensive in product(
        (150, 200, 250),
        (0.20, 0.25, 0.30, 0.35),
        (1.50, 1.75, 2.00),
        (0.25, 0.50, 0.70),
        (0.25, 0.50, 0.70, 1.00),
    ):
        weekly_close = close.resample("W-FRI").last()
        weekly_average = close.rolling(window).mean().resample("W-FRI").last()
        weekly_volatility = data["volatility_20"].resample("W-FRI").last()
        scaled = (target / weekly_volatility).clip(lower=floor, upper=maximum)
        weekly_position = scaled.where(weekly_close > weekly_average, defensive)
        desired = weekly_position.reindex(close.index, method="ffill").fillna(defensive)
        history = backtest(close, cash, desired)
        selection = metrics(history, *SELECTION)
        retrospective = metrics(history, *RETROSPECTIVE)
        block_values = [metrics(history, start, end) for _, start, end in BLOCKS]
        rows.append(
            {
                "window": window,
                "target": target,
                "maximum": maximum,
                "floor": floor,
                "defensive": defensive,
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
    print("Eligible:", len(eligible), "of", len(results))
    print(eligible.nlargest(30, "robust_score").to_string(index=False))


if __name__ == "__main__":
    main()
