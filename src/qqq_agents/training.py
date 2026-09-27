"""Entrenamiento de los especialistas cuantitativos sin consultar el test final."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_agents.agents.quantitative import (
    MOMENTUM_FEATURES,
    RISK_FEATURES,
    TECHNICAL_FEATURES,
    QuantitativeAgent,
)
from qqq_agents.config import AppConfig


def train_quantitative_agents(
    frame: pd.DataFrame,
    *,
    config: AppConfig,
    cutoff: str,
    output_dir: str | Path = "artifacts/models",
) -> dict[str, Any]:
    cutoff_timestamp = pd.Timestamp(cutoff)
    prepared = frame.copy()
    prepared["target_end_date"] = pd.to_datetime(prepared["target_end_date"])
    development = prepared.loc[
        (prepared.index <= cutoff_timestamp) & (prepared["target_end_date"] <= cutoff_timestamp)
    ].copy()
    if development.empty:
        raise ValueError("No leakage-safe development observations are available")

    specifications = {
        "technical": (TECHNICAL_FEATURES, "target_up", "direction"),
        "momentum": (MOMENTUM_FEATURES, "target_up", "direction"),
        "risk": (RISK_FEATURES, "target_risk", "risk"),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "cutoff": str(cutoff_timestamp.date()),
        "test_period_consulted": False,
        "training_start": str(development.index.min().date()),
        "training_end": str(development.index.max().date()),
        "maximum_target_end_date": str(development["target_end_date"].max().date()),
        "agents": {},
    }
    for agent_id, (feature_names, target_column, target_kind) in specifications.items():
        model_config = config.models[agent_id]
        complete = development.loc[:, [*feature_names, target_column]].dropna()
        agent = QuantitativeAgent.fit(
            agent_id=agent_id,
            frame=development,
            feature_names=feature_names,
            target_column=target_column,
            target_kind=target_kind,  # type: ignore[arg-type]
            random_seed=config.project.random_seed,
            n_estimators=model_config.n_estimators or 300,
            max_depth=model_config.max_depth or 6,
            min_samples_leaf=model_config.min_samples_leaf or 10,
            model_version=f"{agent_id}-rf-development-v1",
        )
        model_path = destination / f"{agent_id}.joblib"
        agent.save(model_path)
        manifest["agents"][agent_id] = {
            "model_path": str(model_path),
            "features": list(feature_names),
            "target": target_column,
            "rows": len(complete),
            "positive_rate": float(complete[target_column].astype(float).mean()),
        }

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
