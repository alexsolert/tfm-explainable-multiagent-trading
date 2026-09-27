"""Seleccion y explicacion LIME de decisiones representativas."""

from __future__ import annotations

from typing import Any

import pandas as pd

from qqq_agents.config import AppConfig
from qqq_agents.evaluation.walk_forward import fit_agents_at_cutoff
from qqq_agents.explainability.local import LimeAgentExplainer


def select_representative_cases(decisions: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if decisions.empty:
        raise ValueError("Representative cases require at least one decision")

    selected: dict[str, pd.Timestamp] = {}
    buys = decisions.loc[decisions["action"] == "BUY"]
    sells = decisions.loc[decisions["action"] == "SELL"]
    vetoes = decisions.loc[decisions["risk_veto_triggered"].astype(bool)]
    selected["strongest_buy"] = (
        buys["score_before_veto"].idxmax()
        if not buys.empty
        else decisions["score_before_veto"].idxmax()
    )
    selected["strongest_sell"] = (
        sells["score_before_veto"].idxmin()
        if not sells.empty
        else decisions["score_before_veto"].idxmin()
    )
    selected["highest_risk_veto"] = (
        vetoes["risk_veto_probability"].idxmax()
        if not vetoes.empty
        else decisions["risk_veto_probability"].idxmax()
    )
    signal_columns = ["technical_signal", "momentum_signal", "risk_signal"]
    disagreement = decisions.loc[:, signal_columns].std(axis=1)
    selected["largest_disagreement"] = disagreement.idxmax()
    return selected


def generate_lime_cases(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    config: AppConfig,
    num_samples: int = 1_000,
) -> dict[str, Any]:
    prepared = frame.copy().sort_index()
    prepared["target_end_date"] = pd.to_datetime(prepared["target_end_date"])
    selected = select_representative_cases(decisions)
    cases: list[dict[str, Any]] = []
    models_by_year: dict[int, dict[str, object]] = {}

    for case_type, timestamp in selected.items():
        year = timestamp.year
        cutoff = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=1)
        if year not in models_by_year:
            agents, _ = fit_agents_at_cutoff(prepared, config=config, cutoff=cutoff)
            models_by_year[year] = agents
        agents = models_by_year[year]
        training = prepared.loc[
            (prepared.index <= cutoff) & (prepared["target_end_date"] <= cutoff)
        ]
        observation = prepared.loc[timestamp].to_dict()
        decision = decisions.loc[timestamp]

        for agent_id, agent in agents.items():
            explainer = LimeAgentExplainer(
                agent,  # type: ignore[arg-type]
                training,
                random_seed=config.project.random_seed,
            )
            attributions = explainer.explain(
                observation,
                top_n=5,
                num_samples=num_samples,
            )
            cases.append(
                {
                    "case_type": case_type,
                    "date": str(timestamp.date()),
                    "action": decision["action"],
                    "score_before_veto": float(decision["score_before_veto"]),
                    "risk_veto_triggered": bool(decision["risk_veto_triggered"]),
                    "agent_id": agent_id,
                    "model_version": agent.model_version,  # type: ignore[attr-defined]
                    "attributions": [item.to_dict() for item in attributions],
                }
            )
    return {
        "test_period_consulted": False,
        "selection": {name: str(value.date()) for name, value in selected.items()},
        "cases": cases,
    }
