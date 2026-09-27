"""Secuencia jerarquica de roles LLM con validacion estrategica final."""

from __future__ import annotations

import asyncio

from qqq_agents.llm.contracts import (
    AgentRole,
    LLMCallResult,
    MarketContextPacket,
    PriorAssessment,
)
from qqq_agents.llm.providers import StructuredLLMClient


class LLMCommittee:
    def __init__(self, client: StructuredLLMClient) -> None:
        self.client = client

    async def evaluate(self, packet: MarketContextPacket) -> tuple[LLMCallResult, ...]:
        context, sentiment = await asyncio.gather(
            self.client.evaluate(AgentRole.MARKET_CONTEXT, packet),
            self.client.evaluate(AgentRole.SENTIMENT, packet),
        )
        prior = tuple(
            PriorAssessment(
                role=result.role,
                signal=result.assessment.signal,
                confidence=result.assessment.confidence,
                justification=result.assessment.justification,
            )
            for result in (context, sentiment)
        )
        validator_packet = packet.model_copy(update={"prior_assessments": prior})
        validator = await self.client.evaluate(AgentRole.STRATEGIC_VALIDATOR, validator_packet)
        return context, sentiment, validator
