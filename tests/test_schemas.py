from datetime import date

import pytest
from pydantic import ValidationError

from qqq_agents.schemas import AgentSignal, Personality


def test_agent_signal_accepts_normalized_values() -> None:
    signal = AgentSignal(
        agent_id="technical",
        agent_type="quantitative",
        as_of=date(2024, 1, 5),
        signal=0.4,
        confidence=0.7,
        explanation="Trend remains positive.",
        model_name="random_forest",
        model_version="v1",
        personality=Personality.CONSERVATIVE,
    )

    assert signal.signal == 0.4
    assert signal.personality is Personality.CONSERVATIVE


def test_agent_signal_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        AgentSignal(
            agent_id="technical",
            agent_type="quantitative",
            as_of=date(2024, 1, 5),
            signal=1.2,
            confidence=0.7,
            explanation="Invalid signal.",
            model_name="random_forest",
            model_version="v1",
        )
