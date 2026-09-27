from datetime import date

import pytest

from qqq_agents.personality import load_personality
from qqq_agents.schemas import AgentSignal, Personality


def test_conservative_profile_increases_normalized_risk_weight() -> None:
    profile = load_personality("configs/personalities/conservative.yaml")
    weights = profile.transform_weights({"technical": 0.4, "momentum": 0.3, "risk": 0.3})

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["risk"] > 0.3


def test_confidence_floor_can_neutralize_a_signal() -> None:
    profile = load_personality("configs/personalities/conservative.yaml")
    raw = AgentSignal(
        agent_id="technical",
        agent_type="quantitative",
        as_of=date(2024, 1, 5),
        signal=0.8,
        confidence=0.51,
        explanation="Test.",
        model_name="test",
        model_version="v1",
    )

    adjusted = profile.transform_signal(raw)

    assert adjusted.signal == 0.0
    assert adjusted.personality is Personality.CONSERVATIVE
