import json
from pathlib import Path

import pandas as pd
import pytest

from qqq_agents.dashboard.data import load_dashboard_artifacts, parse_attribution_cell

ROOT = Path(__file__).resolve().parents[1]


def test_parse_attribution_cell_requires_a_list() -> None:
    payload = json.dumps(
        [{"feature": "rsi_14", "value": 52.0, "contribution": 0.12, "method": "shap"}]
    )

    parsed = parse_attribution_cell(payload)

    assert parsed[0]["feature"] == "rsi_14"
    with pytest.raises(ValueError):
        parse_attribution_cell('{"feature": "rsi_14"}')


def test_dashboard_loader_reads_expected_artifacts(tmp_path) -> None:
    walk_forward = tmp_path / "walk_forward"
    walk_forward.mkdir()
    (walk_forward / "validation_metrics.json").write_text(
        json.dumps({"quantitative_multiagent": {"sharpe_ratio": 1.0}}), encoding="utf-8"
    )
    index = pd.DatetimeIndex(["2022-01-07"], name="date")
    pd.DataFrame({"action": ["HOLD"]}, index=index).to_csv(
        walk_forward / "validation_decisions.csv"
    )
    pd.DataFrame({"multiagent": [1.0]}, index=index).to_csv(walk_forward / "validation_equity.csv")
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    (llm_dir / "dry_run_2022-01-07.json").write_text(
        json.dumps({"packet": {"as_of": "2022-01-07"}, "mode": "deterministic_dry_run"}),
        encoding="utf-8",
    )
    hybrid_dir = tmp_path / "hybrid_validation"
    hybrid_dir.mkdir()
    (hybrid_dir / "metrics.json").write_text(
        json.dumps({"hybrid_multiagent": {"sharpe_ratio": 1.1}}), encoding="utf-8"
    )
    pd.DataFrame({"action": ["HOLD"]}, index=index).to_csv(hybrid_dir / "decisions.csv")
    pd.DataFrame({"equity": [1.0]}, index=index).to_csv(hybrid_dir / "strategy.csv")
    final_quantitative_dir = tmp_path / "final_quantitative"
    final_quantitative_dir.mkdir()
    (final_quantitative_dir / "final_test_metrics.json").write_text(
        json.dumps({"quantitative_multiagent": {"sharpe_ratio": 0.9}}), encoding="utf-8"
    )
    pd.DataFrame({"action": ["HOLD"]}, index=index).to_csv(
        final_quantitative_dir / "final_test_decisions.csv"
    )
    pd.DataFrame({"multiagent": [1.0]}, index=index).to_csv(
        final_quantitative_dir / "final_test_equity.csv"
    )
    final_hybrid_dir = tmp_path / "hybrid_final_test"
    final_hybrid_dir.mkdir()
    (final_hybrid_dir / "metrics.json").write_text(
        json.dumps({"hybrid_multiagent": {"sharpe_ratio": 0.9}}), encoding="utf-8"
    )
    pd.DataFrame({"action": ["HOLD"]}, index=index).to_csv(final_hybrid_dir / "decisions.csv")
    pd.DataFrame({"equity": [1.0]}, index=index).to_csv(final_hybrid_dir / "strategy.csv")

    artifacts = load_dashboard_artifacts(tmp_path)

    assert artifacts.metrics["quantitative_multiagent"]["sharpe_ratio"] == 1.0
    assert artifacts.decisions.index[0] == pd.Timestamp("2022-01-07")
    assert artifacts.lime_cases is None
    assert artifacts.llm_traces[pd.Timestamp("2022-01-07")]["mode"] == "deterministic_dry_run"
    assert artifacts.hybrid_metrics["hybrid_multiagent"]["sharpe_ratio"] == 1.1
    assert artifacts.hybrid_decisions.loc[pd.Timestamp("2022-01-07"), "action"] == "HOLD"
    assert artifacts.final_quantitative_metrics["quantitative_multiagent"]["sharpe_ratio"] == 0.9
    assert artifacts.final_hybrid_metrics["hybrid_multiagent"]["sharpe_ratio"] == 0.9
    assert artifacts.final_hybrid_strategy.loc[pd.Timestamp("2022-01-07"), "equity"] == 1.0


def test_versioned_demo_bundle_is_self_contained() -> None:
    artifacts = load_dashboard_artifacts(ROOT / "demo_data")

    assert len(artifacts.final_hybrid_decisions) == 105
    assert artifacts.final_hybrid_metrics["test_period_consulted"] is True
    assert artifacts.final_quantitative_metrics["quantitative_multiagent"]["sharpe_ratio"] > 0
    assert artifacts.lime_cases["test_period_consulted"] is False
    assert artifacts.v2_metrics["protected_test_consulted"] is False
    assert artifacts.v2_metrics["deployment_recommendation"]["champion"] == "sma_50_200"
    assert artifacts.v2_model_leaderboard["selected"].any()
    assert artifacts.v2_protected_metrics["protected_test_consulted"] is True
    assert artifacts.v2_protected_metrics["deployment_recommendation"]["champion"] == "sma_50_200"
    assert artifacts.v4_metrics["prospective_period_consulted"] is False
    assert artifacts.v4_metrics["configuration_evaluations"] == 1_660
    assert len(artifacts.v4_decisions) > 2_000
    assert "v4_multiagent" in artifacts.v4_equity
