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
    return DashboardArtifacts(
        metrics=metrics,
        decisions=decisions.sort_index(),
        equity=equity.sort_index(),
        lime_cases=lime_cases,
        llm_traces=llm_traces,
    )
