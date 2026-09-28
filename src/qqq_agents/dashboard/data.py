"""Lectura validada de artefactos generados fuera de Streamlit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DashboardArtifacts:
    metrics: dict[str, Any]
    decisions: pd.DataFrame
    equity: pd.DataFrame
    lime_cases: dict[str, Any] | None
    llm_traces: dict[pd.Timestamp, dict[str, Any]]
    hybrid_metrics: dict[str, Any] | None
    hybrid_decisions: pd.DataFrame | None
    hybrid_strategy: pd.DataFrame | None
    final_quantitative_metrics: dict[str, Any] | None
    final_quantitative_decisions: pd.DataFrame | None
    final_quantitative_equity: pd.DataFrame | None
    final_hybrid_metrics: dict[str, Any] | None
    final_hybrid_decisions: pd.DataFrame | None
    final_hybrid_strategy: pd.DataFrame | None
    v2_metrics: dict[str, Any] | None
    v2_decisions: pd.DataFrame | None
    v2_equity: pd.DataFrame | None
    v2_model_leaderboard: pd.DataFrame | None
    v2_protected_metrics: dict[str, Any] | None
    v2_protected_decisions: pd.DataFrame | None
    v2_protected_equity: pd.DataFrame | None
    v3_metrics: dict[str, Any] | None
    v3_decisions: pd.DataFrame | None
    v3_equity: pd.DataFrame | None
    v4_metrics: dict[str, Any] | None
    v4_decisions: pd.DataFrame | None
    v4_equity: pd.DataFrame | None


def _optional_bundle(
    paths: dict[str, Path],
) -> tuple[dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame | None]:
    if not all(path.exists() for path in paths.values()):
        return None, None, None
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    decisions = pd.read_csv(paths["decisions"], index_col="date", parse_dates=["date"])
    strategy = pd.read_csv(paths["strategy"], index_col="date", parse_dates=["date"])
    return metrics, decisions.sort_index(), strategy.sort_index()


def parse_attribution_cell(value: object) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Attribution payload must be a JSON list of objects")
    return parsed


def load_dashboard_artifacts(root: str | Path = "artifacts") -> DashboardArtifacts:
    artifact_root = Path(root)
    walk_forward = artifact_root / "walk_forward"
    required = {
        "metrics": walk_forward / "validation_metrics.json",
        "decisions": walk_forward / "validation_decisions.csv",
        "equity": walk_forward / "validation_equity.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dashboard artifacts. Run 'qqq-agents walk-forward --with-shap'. "
            f"Missing: {missing}"
        )

    metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
    decisions = pd.read_csv(required["decisions"], index_col="date", parse_dates=["date"])
    equity = pd.read_csv(required["equity"], index_col="date", parse_dates=["date"])
    lime_path = artifact_root / "explainability" / "lime_cases.json"
    lime_cases = json.loads(lime_path.read_text(encoding="utf-8")) if lime_path.exists() else None
    llm_traces: dict[pd.Timestamp, dict[str, Any]] = {}
    llm_root = artifact_root / "llm"
    for pattern in ("dry_run_*.json", "pilot_*.json"):
        for path in sorted(llm_root.glob(pattern)):
            payload = json.loads(path.read_text(encoding="utf-8"))
            as_of = pd.Timestamp(payload["packet"]["as_of"])
            llm_traces[as_of] = payload
    hybrid_root = artifact_root / "hybrid_validation"
    hybrid_paths = {
        "metrics": hybrid_root / "metrics.json",
        "decisions": hybrid_root / "decisions.csv",
        "strategy": hybrid_root / "strategy.csv",
    }
    hybrid_metrics, hybrid_decisions, hybrid_strategy = _optional_bundle(hybrid_paths)
    final_quantitative_root = artifact_root / "final_quantitative"
    final_quantitative_paths = {
        "metrics": final_quantitative_root / "final_test_metrics.json",
        "decisions": final_quantitative_root / "final_test_decisions.csv",
        "strategy": final_quantitative_root / "final_test_equity.csv",
    }
    (
        final_quantitative_metrics,
        final_quantitative_decisions,
        final_quantitative_equity,
    ) = _optional_bundle(final_quantitative_paths)
    final_hybrid_root = artifact_root / "hybrid_final_test"
    final_hybrid_paths = {
        "metrics": final_hybrid_root / "metrics.json",
        "decisions": final_hybrid_root / "decisions.csv",
        "strategy": final_hybrid_root / "strategy.csv",
    }
    final_hybrid_metrics, final_hybrid_decisions, final_hybrid_strategy = _optional_bundle(
        final_hybrid_paths
    )
    v2_root = artifact_root / "v2_development"
    v2_paths = {
        "metrics": v2_root / "metrics.json",
        "decisions": v2_root / "decisions.csv",
        "strategy": v2_root / "equity.csv",
    }
    v2_metrics, v2_decisions, v2_equity = _optional_bundle(v2_paths)
    v2_leaderboard_path = v2_root / "model_leaderboard.csv"
    v2_model_leaderboard = (
        pd.read_csv(v2_leaderboard_path, parse_dates=["cutoff"])
        if v2_leaderboard_path.exists()
        else None
    )
    v2_protected_root = artifact_root / "v2_protected_test"
    v2_protected_paths = {
        "metrics": v2_protected_root / "metrics.json",
        "decisions": v2_protected_root / "decisions.csv",
        "strategy": v2_protected_root / "equity.csv",
    }
    v2_protected_metrics, v2_protected_decisions, v2_protected_equity = _optional_bundle(
        v2_protected_paths
    )
    v3_root = artifact_root / "v3_development"
    v3_paths = {
        "metrics": v3_root / "metrics.json",
        "decisions": v3_root / "decisions.csv",
        "strategy": v3_root / "equity.csv",
    }
    v3_metrics, v3_decisions, v3_equity = _optional_bundle(v3_paths)
    v4_root = artifact_root / "v4_development"
    v4_paths = {
        "metrics": v4_root / "metrics.json",
        "decisions": v4_root / "decisions.csv",
        "strategy": v4_root / "equity.csv",
    }
    v4_metrics, v4_decisions, v4_equity = _optional_bundle(v4_paths)
    return DashboardArtifacts(
        metrics=metrics,
        decisions=decisions.sort_index(),
        equity=equity.sort_index(),
        lime_cases=lime_cases,
        llm_traces=llm_traces,
        hybrid_metrics=hybrid_metrics,
        hybrid_decisions=hybrid_decisions,
        hybrid_strategy=hybrid_strategy,
        final_quantitative_metrics=final_quantitative_metrics,
        final_quantitative_decisions=final_quantitative_decisions,
        final_quantitative_equity=final_quantitative_equity,
        final_hybrid_metrics=final_hybrid_metrics,
        final_hybrid_decisions=final_hybrid_decisions,
        final_hybrid_strategy=final_hybrid_strategy,
        v2_metrics=v2_metrics,
        v2_decisions=v2_decisions,
        v2_equity=v2_equity,
        v2_model_leaderboard=v2_model_leaderboard,
        v2_protected_metrics=v2_protected_metrics,
        v2_protected_decisions=v2_protected_decisions,
        v2_protected_equity=v2_protected_equity,
        v3_metrics=v3_metrics,
        v3_decisions=v3_decisions,
        v3_equity=v3_equity,
        v4_metrics=v4_metrics,
        v4_decisions=v4_decisions,
        v4_equity=v4_equity,
    )
