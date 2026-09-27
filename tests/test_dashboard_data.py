import json

import pandas as pd
import pytest

from qqq_agents.dashboard.data import load_dashboard_artifacts, parse_attribution_cell


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

    artifacts = load_dashboard_artifacts(tmp_path)

    assert artifacts.metrics["quantitative_multiagent"]["sharpe_ratio"] == 1.0
    assert artifacts.decisions.index[0] == pd.Timestamp("2022-01-07")
    assert artifacts.lime_cases is None
    assert artifacts.llm_traces[pd.Timestamp("2022-01-07")]["mode"] == "deterministic_dry_run"
