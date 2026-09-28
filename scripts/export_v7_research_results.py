"""Evaluate the predeclared V7 profiles and export a compact audit bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qqq_agents.backtesting.metrics import calculate_metrics
from qqq_agents.v7 import evaluate_v7, load_v7_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "v6_daily.csv"
DESTINATION = ROOT / "research_results" / "v7"
BLOCKS = {
    "early": ("2004-01-01", "2009-12-31"),
    "middle": ("2010-01-01", "2016-12-31"),
    "late": ("2017-01-01", "2022-12-31"),
}


def _metrics(history: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    sample = history.loc[start:end]
    return calculate_metrics(
        returns=sample["strategy_return"],
        turnover=sample["turnover"],
        asset_returns=sample["asset_return"],
        positions=sample["applied_position"],
        periods_per_year=252,
    )


def main() -> None:
    frame = pd.read_csv(SOURCE, parse_dates=["date"], index_col="date").sort_index()
    config = load_v7_config(ROOT / "configs/v7.yaml")
    evaluation = evaluate_v7(frame, config)
    combined = {**evaluation.policies, **evaluation.baselines}
    blocks = {
        block: {
            name: _metrics(result.history, start, end)
            for name, result in combined.items()
        }
        for block, (start, end) in BLOCKS.items()
    }
    financing_stress = {}
    for spread in (150, 300, 500, 800):
        stressed = config.model_copy(
            update={
                "costs": config.costs.model_copy(
                    update={"borrowing_spread_bps": spread}
                )
            }
        )
        result = evaluate_v7(frame, stressed)
        financing_stress[str(spread)] = {
            name: result.subperiods["retrospective"][name]
            for name in result.policies
        }

    payload = {
        "version": config.version,
        "selection_uses_only_data_through": config.period.selection_end,
        "retrospective_assessment_is_not_holdout": True,
        "policies": [policy.model_dump(mode="json") for policy in config.policies],
        "subperiods": evaluation.subperiods,
        "selection_robustness_blocks": blocks,
        "robustness": {"borrowing_spread_bps": financing_stress},
        "interpretation": (
            "V7 maps a return-risk frontier; leverage is not evidence of forecasting alpha. "
            "Compare each policy with constant-exposure controls as well as QQQ Buy & Hold."
        ),
    }
    rows = []
    for period, values in evaluation.subperiods.items():
        for name, metrics in values.items():
            rows.append({"period": period, "strategy": name, **metrics})
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(DESTINATION / "leaderboard.csv", index=False)
    print(f"Exported V7 results to {DESTINATION}")


if __name__ == "__main__":
    main()
