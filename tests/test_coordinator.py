from datetime import UTC, date, datetime

import pytest

from qqq_agents.coordinator import DeterministicCoordinator
from qqq_agents.schemas import Action, AgentSignal

DECISION_DATE = date(2024, 1, 5)


def make_signal(
    agent_id: str,
    signal: float,
    confidence: float = 1.0,
    veto_probability: float | None = None,
) -> AgentSignal:
    return AgentSignal(
        agent_id=agent_id,
        agent_type="quantitative",
        as_of=DECISION_DATE,
        signal=signal,
        confidence=confidence,
        explanation="Test signal.",
        model_name="test",
        model_version="v1",
        veto_probability=veto_probability,
        veto_reason="Risk threshold exceeded." if veto_probability is not None else None,
    )


def coordinator() -> DeterministicCoordinator:
    return DeterministicCoordinator(
        weights={"technical": 0.4, "momentum": 0.3, "risk": 0.3},
        buy_threshold=0.2,
        sell_threshold=-0.2,
        risk_veto_probability=0.65,
    )


def test_weighted_positive_signals_produce_buy() -> None:
    trace = coordinator().decide(
        as_of=DECISION_DATE,
        signals=[make_signal("technical", 0.8), make_signal("momentum", 0.5)],
        current_position=0.0,
        created_at=datetime(2024, 1, 5, tzinfo=UTC),
    )

    assert trace.action is Action.BUY
    assert trace.risk_veto_triggered is False
    assert sum(item.contribution for item in trace.contributions) == pytest.approx(
        trace.score_before_veto
    )


def test_risk_veto_blocks_entry() -> None:
    trace = coordinator().decide(
        as_of=DECISION_DATE,
        signals=[
            make_signal("technical", 0.9),
            make_signal("momentum", 0.8),
            make_signal("risk", -0.2, veto_probability=0.8),
        ],
        current_position=0.0,
    )

    assert trace.score_before_veto > 0.2
    assert trace.action is Action.HOLD
    assert trace.risk_veto_triggered is True


def test_risk_veto_exits_an_existing_position() -> None:
    trace = coordinator().decide(
        as_of=DECISION_DATE,
        signals=[make_signal("technical", 0.9), make_signal("risk", -0.8, veto_probability=0.9)],
        current_position=1.0,
    )

    assert trace.action is Action.SELL
    assert trace.risk_veto_reason == "Risk threshold exceeded."


def test_mismatched_signal_date_is_rejected() -> None:
    wrong_date = make_signal("technical", 0.5).model_copy(update={"as_of": date(2024, 1, 6)})

    with pytest.raises(ValueError, match="share the coordinator decision date"):
        coordinator().decide(
            as_of=DECISION_DATE,
            signals=[wrong_date],
            current_position=0.0,
        )
