"""Evaluacion temporal del comite hibrido sobre un periodo cerrado."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qqq_agents.backtesting import BacktestResult, run_backtest
from qqq_agents.config import AppConfig
from qqq_agents.llm.committee import LLMCommittee
from qqq_agents.llm.contracts import LLMCallResult
from qqq_agents.llm.pilot import PilotCase, coordinate_pilot
from qqq_agents.llm.providers import StructuredLLMClient
from qqq_agents.schemas import Action

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class HybridPeriodResult:
    decisions: pd.DataFrame
    strategy: BacktestResult
    llm_traces: tuple[tuple[LLMCallResult, ...], ...]
    estimated_cost_usd: float
    incremental_estimated_cost_usd: float


async def _evaluate_cases(
    cases: tuple[PilotCase, ...],
    *,
    client: StructuredLLMClient,
    concurrency: int,
    progress: ProgressCallback | None,
) -> tuple[tuple[LLMCallResult, ...], ...]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    semaphore = asyncio.Semaphore(concurrency)
    committee = LLMCommittee(client)

    async def evaluate_one(index: int, case: PilotCase) -> tuple[int, tuple[LLMCallResult, ...]]:
        async with semaphore:
            return index, await committee.evaluate(case.packet)

    tasks = [asyncio.create_task(evaluate_one(index, case)) for index, case in enumerate(cases)]
    ordered: list[tuple[LLMCallResult, ...] | None] = [None] * len(cases)
    completed = 0
    for task in asyncio.as_completed(tasks):
        index, results = await task
        ordered[index] = results
        completed += 1
        if progress is not None:
            progress(completed, len(cases))
    if any(result is None for result in ordered):
        raise RuntimeError("At least one LLM evaluation did not complete")
    return tuple(result for result in ordered if result is not None)


async def run_hybrid_period(
    *,
    cases: tuple[PilotCase, ...],
    close: pd.Series,
    config: AppConfig,
    client: StructuredLLMClient,
    concurrency: int = 5,
    progress: ProgressCallback | None = None,
) -> HybridPeriodResult:
    """Evaluate LLM roles concurrently, then coordinate positions chronologically."""

    if not cases:
        raise ValueError("At least one pilot case is required")
    llm_traces = await _evaluate_cases(
        cases,
        client=client,
        concurrency=concurrency,
        progress=progress,
    )
    records: list[dict[str, object]] = []
    current_position = 0.0
    for case, results in zip(cases, llm_traces, strict=True):
        trace = coordinate_pilot(
            case=case,
            config=config,
            results=results,
            current_position=current_position,
        )
        if trace.action is Action.BUY:
            current_position = 1.0
        elif trace.action is Action.SELL:
            current_position = 0.0
        record: dict[str, object] = {
            "date": pd.Timestamp(case.packet.as_of),
            "action": trace.action.value,
            "desired_position": current_position,
            "score_before_veto": trace.score_before_veto,
            "risk_veto_triggered": trace.risk_veto_triggered,
            "risk_veto_reason": trace.risk_veto_reason,
            "coordinator_version": trace.coordinator_version,
            "personality": case.packet.personality.value,
            "quantitative_proposed_action": (
                case.packet.proposed_action.value if case.packet.proposed_action else None
            ),
        }
        for signal in case.quantitative_signals:
            record[f"{signal.agent_id}_signal"] = signal.signal
            record[f"{signal.agent_id}_confidence"] = signal.confidence
        for result in results:
            prefix = result.role.value
            record[f"{prefix}_signal"] = result.assessment.signal
            record[f"{prefix}_confidence"] = result.assessment.confidence
            record[f"{prefix}_model"] = result.model
            record[f"{prefix}_input_tokens"] = result.input_tokens
            record[f"{prefix}_output_tokens"] = result.output_tokens
            record[f"{prefix}_cached"] = result.cached
            record[f"{prefix}_justification"] = result.assessment.justification
            record[f"{prefix}_evidence_ids"] = json.dumps(result.assessment.evidence_ids)
            record[f"{prefix}_limitations"] = json.dumps(result.assessment.limitations)
        for contribution in trace.contributions:
            record[f"{contribution.agent_id}_contribution"] = contribution.contribution
        records.append(record)

    decisions = pd.DataFrame.from_records(records).set_index("date").sort_index()
    strategy = run_backtest(
        close.reindex(decisions.index),
        decisions["desired_position"],
        transaction_cost_bps=config.experiment.transaction_cost_bps,
    )
    all_results = [result for trace in llm_traces for result in trace]
    estimated_cost = sum(
        result.estimated_cost(
            input_price=config.llm.input_price_per_million,
            output_price=config.llm.output_price_per_million,
        )
        for result in all_results
    )
    incremental_cost = sum(
        result.estimated_cost(
            input_price=config.llm.input_price_per_million,
            output_price=config.llm.output_price_per_million,
        )
        for result in all_results
        if not result.cached
    )
    return HybridPeriodResult(
        decisions=decisions,
        strategy=strategy,
        llm_traces=llm_traces,
        estimated_cost_usd=estimated_cost,
        incremental_estimated_cost_usd=incremental_cost,
    )


def save_hybrid_period(
    result: HybridPeriodResult,
    *,
    output_dir: str | Path,
    period_name: str,
    start: str,
    end: str,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(destination / "decisions.csv", index=True)
    result.strategy.history.to_csv(destination / "strategy.csv", index=True)
    metrics: dict[str, object] = {
        "period_name": period_name,
        "period": {"start": start, "end": end},
        "test_period_consulted": period_name == "final_test",
        "hybrid_multiagent": result.strategy.metrics,
        "decision_counts": result.decisions["action"].value_counts().to_dict(),
        "risk_veto_count": int(result.decisions["risk_veto_triggered"].sum()),
        "llm_estimated_cost_usd": result.estimated_cost_usd,
        "llm_incremental_estimated_cost_usd": result.incremental_estimated_cost_usd,
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics
