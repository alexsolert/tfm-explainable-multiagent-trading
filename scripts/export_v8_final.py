"""Generate the final V8 audit, dashboard bundle and current paper decision."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from qqq_agents.backtesting.metrics import calculate_metrics
from qqq_agents.v2.diagnostics import (
    circular_block_bootstrap_difference,
    probability_of_backtest_overfitting,
)
from qqq_agents.v4.diagnostics import deflated_sharpe_probability
from qqq_agents.v8 import evaluate_v8, load_v8_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "v6_daily.csv"
ARTIFACTS = ROOT / "artifacts" / "v8_final"
RESEARCH = ROOT / "research_results" / "v8"
DEMO = ROOT / "demo_data" / "v8_final"
BLOCKS = {
    "early": ("2004-01-01", "2009-12-31"),
    "middle": ("2010-01-01", "2016-12-31"),
    "late": ("2017-01-01", "2022-12-31"),
}
STRESS_PERIODS = {
    "global_financial_crisis": ("2007-10-01", "2009-06-30"),
    "covid": ("2020-02-01", "2020-12-31"),
    "rate_shock_2022": ("2022-01-01", "2022-12-31"),
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


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def main() -> None:
    frame = pd.read_csv(SOURCE, parse_dates=["date"], index_col="date").sort_index()
    config = load_v8_config(ROOT / "configs/v8.yaml")
    evaluation = evaluate_v8(frame, config)
    all_results = {
        **evaluation.strategies,
        **evaluation.baselines,
        **evaluation.matched_baselines,
    }
    selection_slice = slice(
        config.period.selection_start, config.period.selection_end
    )
    retrospective_slice = slice(
        config.period.retrospective_start, config.period.retrospective_end
    )
    blocks = {
        block: {
            name: _metrics(result.history, start, end)
            for name, result in all_results.items()
        }
        for block, (start, end) in BLOCKS.items()
    }
    stress_periods = {
        period: {
            name: _metrics(result.history, start, end)
            for name, result in all_results.items()
        }
        for period, (start, end) in STRESS_PERIODS.items()
    }

    bootstrap: dict[str, dict[str, object]] = {}
    dsr: dict[str, object] = {}
    for name, strategy in evaluation.strategies.items():
        benchmark = evaluation.baselines["constant_100"]
        matched = evaluation.matched_baselines[f"matched_{name}"]
        bootstrap[name] = {}
        for period, period_slice in (
            ("selection", selection_slice),
            ("retrospective", retrospective_slice),
        ):
            bootstrap[name][period] = {
                "vs_buy_and_hold": circular_block_bootstrap_difference(
                    strategy.history.loc[period_slice, "strategy_return"],
                    benchmark.history.loc[period_slice, "strategy_return"],
                    samples=config.robustness.bootstrap_samples,
                    block_length=config.robustness.bootstrap_block_sessions,
                    random_seed=config.robustness.random_seed,
                    periods_per_year=252,
                ),
                "vs_equal_average_exposure": circular_block_bootstrap_difference(
                    strategy.history.loc[period_slice, "strategy_return"],
                    matched.history.loc[period_slice, "strategy_return"],
                    samples=config.robustness.bootstrap_samples,
                    block_length=config.robustness.bootstrap_block_sessions,
                    random_seed=config.robustness.random_seed,
                    periods_per_year=252,
                ),
            }
        dsr[name] = deflated_sharpe_probability(
            strategy.history.loc[selection_slice, "strategy_return"],
            trials=config.robustness.candidate_trials,
            periods_per_year=252,
        )

    family = pd.DataFrame(
        {
            name: result.history.loc[selection_slice, "strategy_return"]
            for name, result in all_results.items()
        }
    )
    pbo = probability_of_backtest_overfitting(
        family, partitions=8, periods_per_year=252
    )
    latest = evaluation.decisions.dropna().iloc[-1]
    paper_decision = {
        "as_of": latest.name,
        "framework_version": config.version,
        "agents": {
            "trend": {
                "score": latest["trend_score"],
                "vote": latest["trend_vote"],
            },
            "volatility": {
                "annualized_forecast": latest["forecast_volatility"],
                "risk_level": latest["volatility_risk"],
            },
            "drawdown": {
                "current_252": latest["drawdown_252"],
                "risk_level": latest["drawdown_risk"],
                "authority": "advisory",
            },
            "relative_strength": {
                "qqq_vs_spy_20": latest["relative_strength_20"],
                "vote": latest["relative_strength_vote"],
            },
        },
        "committee_score": latest["committee_score"],
        "profiles": {
            profile.name: {
                "label": profile.label,
                "exposure": latest[f"{profile.name}_desired_position"],
                "action": latest[f"{profile.name}_action"],
                "state": latest[f"{profile.name}_policy_state"],
                "signal_strength": latest[f"{profile.name}_signal_strength"],
                "explanation": latest[f"{profile.name}_explanation"],
            }
            for profile in config.profiles
        },
        "execution": "paper_only",
        "disclaimer": "Experimental academic output; not investment advice.",
    }
    payload = {
        "version": config.version,
        "selection_uses_only_data_through": config.period.selection_end,
        "retrospective_assessment_is_not_holdout": True,
        "profile_definitions": [
            profile.model_dump(mode="json") for profile in config.profiles
        ],
        "subperiods": evaluation.subperiods,
        "selection_robustness_blocks": blocks,
        "stress_periods": stress_periods,
        "robustness": {
            "block_bootstrap": bootstrap,
            "deflated_sharpe": dsr,
            "candidate_family_cscv": pbo,
        },
        "latest_paper_decision": paper_decision,
        "claims": {
            "supported": (
                "V8 provides explainable risk profiles that improve different points of the "
                "historical QQQ return-risk frontier under explicit costs and financing."
            ),
            "not_supported": (
                "The experiment does not prove persistent alpha or guarantee future "
                "outperformance."
            ),
        },
    }
    rows = [
        {"period": period, "strategy": name, **metrics}
        for period, values in evaluation.subperiods.items()
        for name, metrics in values.items()
    ]
    equity = pd.DataFrame(
        {
            name: result.history["equity"]
            for name, result in {
                **evaluation.strategies,
                "buy_and_hold": evaluation.baselines["constant_100"],
                "volatility_target_35": evaluation.baselines[
                    "volatility_target_35"
                ],
            }.items()
        }
    ).loc[config.period.selection_start : config.period.retrospective_end]
    decisions = (
        evaluation.decisions.loc[
            config.period.selection_start : config.period.retrospective_end
        ]
        .resample("W-FRI")
        .last()
        .dropna(subset=["trend_score"])
    )

    for destination in (ARTIFACTS, DEMO):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        )
        decisions.to_csv(destination / "decisions.csv", index_label="date")
        equity.to_csv(destination / "equity.csv", index_label="date")
        (destination / "current_decision.json").write_text(
            json.dumps(paper_decision, indent=2, default=_json_default), encoding="utf-8"
        )

    RESEARCH.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARTIFACTS / "metrics.json", RESEARCH / "metrics.json")
    pd.DataFrame(rows).to_csv(RESEARCH / "leaderboard.csv", index=False)
    shutil.copy2(
        ARTIFACTS / "current_decision.json", RESEARCH / "current_decision.json"
    )
    print(f"Exported V8 final bundle through {paper_decision['as_of'].date()}")


if __name__ == "__main__":
    main()
