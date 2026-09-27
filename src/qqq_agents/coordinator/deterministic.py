"""Agregacion ponderada con confianza y veto de riesgo."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime

from qqq_agents.schemas import (
    Action,
    AgentSignal,
    DecisionTrace,
    WeightedContribution,
)


class DeterministicCoordinator:
    """Combine normalized signals without delegating the final action to an LLM."""

    version = "deterministic-weighted-v1"

    def __init__(
        self,
        *,
        weights: dict[str, float],
        buy_threshold: float,
        sell_threshold: float,
        risk_veto_probability: float,
    ) -> None:
        if not weights or any(weight < 0 for weight in weights.values()):
            raise ValueError("Weights must be non-negative and non-empty")
        if not -1 <= sell_threshold < 0 < buy_threshold <= 1:
            raise ValueError("Expected sell_threshold < 0 < buy_threshold within [-1, 1]")
        if not 0 <= risk_veto_probability <= 1:
            raise ValueError("risk_veto_probability must remain in [0, 1]")
        self.weights = weights.copy()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.risk_veto_probability = risk_veto_probability

    def decide(
        self,
        *,
        as_of: date,
        signals: Iterable[AgentSignal],
        current_position: float,
        created_at: datetime | None = None,
    ) -> DecisionTrace:
        if current_position not in (0.0, 1.0):
            raise ValueError("The MVP only supports binary long-only positions")

        signal_list = list(signals)
        if not signal_list:
            raise ValueError("At least one agent signal is required")
        if len({signal.agent_id for signal in signal_list}) != len(signal_list):
            raise ValueError("Agent identifiers must be unique within one decision")
        if any(signal.as_of != as_of for signal in signal_list):
            raise ValueError("All signals must share the coordinator decision date")

        raw_terms: list[tuple[AgentSignal, float, float]] = []
        denominator = 0.0
        for signal in signal_list:
            weight = self.weights.get(signal.agent_id, 0.0)
            effective_weight = weight * signal.confidence
            denominator += effective_weight
            raw_terms.append((signal, weight, effective_weight))
        if denominator <= 0:
            raise ValueError("Signals have no positive effective weight")

        contributions = tuple(
            WeightedContribution(
                agent_id=signal.agent_id,
                signal=signal.signal,
                confidence=signal.confidence,
                weight=weight,
                contribution=effective_weight * signal.signal / denominator,
            )
            for signal, weight, effective_weight in raw_terms
        )
        score = sum(item.contribution for item in contributions)
        score = max(-1.0, min(1.0, score))

        if score > self.buy_threshold:
            provisional_action = Action.BUY
        elif score < self.sell_threshold:
            provisional_action = Action.SELL
        else:
            provisional_action = Action.HOLD

        vetoes = [
            signal
            for signal in signal_list
            if signal.veto_probability is not None
            and signal.veto_probability >= self.risk_veto_probability
        ]
        veto_triggered = bool(vetoes)
        veto_reason = None
        action = provisional_action
        if veto_triggered:
            highest_risk = max(vetoes, key=lambda signal: signal.veto_probability or 0.0)
            action = Action.SELL if current_position == 1.0 else Action.HOLD
            veto_reason = highest_risk.veto_reason or (
                f"{highest_risk.agent_id} veto probability "
                f"{highest_risk.veto_probability:.3f} reached the configured threshold"
            )

        return DecisionTrace(
            as_of=as_of,
            created_at=created_at or datetime.now(UTC),
            action=action,
            score_before_veto=score,
            risk_veto_triggered=veto_triggered,
            risk_veto_reason=veto_reason,
            contributions=contributions,
            coordinator_version=self.version,
        )
