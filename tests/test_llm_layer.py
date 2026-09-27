import asyncio
from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from qqq_agents.config import load_config
from qqq_agents.llm import CachedLLMClient, LLMCommittee, MarketContextPacket, MockLLMClient
from qqq_agents.llm.contracts import AgentRole, DatedHeadline
from qqq_agents.llm.pilot import MARKET_FEATURES, execute_pilot, load_pilot_case
from qqq_agents.schemas import Personality


def packet() -> MarketContextPacket:
    return MarketContextPacket(
        as_of=date(2022, 1, 7),
        personality=Personality.CONSERVATIVE,
        market_features={"volatility_20": 0.22, "rsi_14": 54.0},
        quantitative_signals={"technical": 0.4, "momentum": 0.2, "risk": -0.1},
    )


def test_context_packet_rejects_future_headlines() -> None:
    with pytest.raises(ValidationError, match="Future evidence"):
        MarketContextPacket(
            as_of=date(2022, 1, 7),
            personality=Personality.CONSERVATIVE,
            market_features={},
            quantitative_signals={},
            headlines=(
                DatedHeadline(
                    evidence_id="news-1",
                    published_on=date(2022, 1, 8),
                    text="Future headline",
                    source="test",
                ),
            ),
        )


def test_mock_committee_produces_three_structured_agent_signals() -> None:
    results = asyncio.run(LLMCommittee(MockLLMClient()).evaluate(packet()))
    signals = [
        result.to_agent_signal(as_of=packet().as_of, personality=Personality.CONSERVATIVE)
        for result in results
    ]

    assert [result.role for result in results] == [
        AgentRole.MARKET_CONTEXT,
        AgentRole.SENTIMENT,
        AgentRole.STRATEGIC_VALIDATOR,
    ]
    assert len(signals) == 3
    assert signals[1].signal == 0.0
    assert signals[1].confidence == 0.0


def test_cache_avoids_repeating_identical_calls(tmp_path) -> None:
    mock = MockLLMClient()
    cached = CachedLLMClient(mock, tmp_path)

    first = asyncio.run(cached.evaluate(AgentRole.MARKET_CONTEXT, packet()))
    second = asyncio.run(cached.evaluate(AgentRole.MARKET_CONTEXT, packet()))

    assert mock.call_count == 1
    assert first.cached is False
    assert second.cached is True
    assert first.assessment == second.assessment


def test_pilot_uses_only_selected_development_decision(tmp_path) -> None:
    dates = pd.to_datetime(["2022-12-23", "2022-12-30"])
    features = pd.DataFrame(
        {name: [0.1, 0.2] for name in MARKET_FEATURES},
        index=pd.Index(dates, name="date"),
    )
    decisions = pd.DataFrame(
        {
            "action": ["BUY", "HOLD"],
            "desired_position": [1.0, 1.0],
            "personality": ["conservative", "conservative"],
            "technical_signal": [0.4, 0.2],
            "technical_confidence": [0.7, 0.6],
            "technical_model_version": ["technical-v1", "technical-v1"],
            "momentum_signal": [0.3, 0.1],
            "momentum_confidence": [0.7, 0.6],
            "momentum_model_version": ["momentum-v1", "momentum-v1"],
            "risk_signal": [0.5, 0.4],
            "risk_confidence": [0.8, 0.7],
            "risk_model_version": ["risk-v1", "risk-v1"],
            "risk_veto_probability": [0.25, 0.3],
        },
        index=pd.Index(dates, name="date"),
    )
    feature_path = tmp_path / "features.csv"
    decision_path = tmp_path / "decisions.csv"
    features.to_csv(feature_path)
    decisions.to_csv(decision_path)

    case = load_pilot_case(features_path=feature_path, decisions_path=decision_path)
    payload = asyncio.run(
        execute_pilot(
            case=case,
            config=load_config("configs/base.yaml"),
            client=MockLLMClient(),
            mode="test",
        )
    )

    assert case.packet.as_of == date(2022, 12, 30)
    assert case.current_position == 1.0
    assert payload["test_period_consulted"] is False
    assert len(payload["llm_results"]) == 3
    assert len(payload["combined_decision"]["contributions"]) == 6


def test_pilot_rejects_final_test_date(tmp_path) -> None:
    dates = pd.to_datetime(["2023-01-06"])
    features = pd.DataFrame(
        {name: [0.1] for name in MARKET_FEATURES},
        index=pd.Index(dates, name="date"),
    )
    decisions = pd.DataFrame(
        {"action": ["HOLD"], "desired_position": [0.0]},
        index=pd.Index(dates, name="date"),
    )
    feature_path = tmp_path / "features.csv"
    decision_path = tmp_path / "decisions.csv"
    features.to_csv(feature_path)
    decisions.to_csv(decision_path)

    with pytest.raises(ValueError, match="No development-period decisions"):
        load_pilot_case(
            features_path=feature_path,
            decisions_path=decision_path,
            requested_date="2023-01-06",
        )
