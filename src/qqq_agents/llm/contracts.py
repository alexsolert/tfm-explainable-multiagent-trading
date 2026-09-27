"""Esquemas fechados de entrada, salida y consumo de los agentes LLM."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qqq_agents.schemas import Action, AgentSignal, Evidence, Personality


class AgentRole(StrEnum):
    MARKET_CONTEXT = "market_context"
    SENTIMENT = "sentiment"
    STRATEGIC_VALIDATOR = "strategic_validator"


class DatedHeadline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    published_on: date
    text: str = Field(min_length=1, max_length=1_000)
    source: str = Field(min_length=1)


class PriorAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: AgentRole
    signal: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    justification: str


class MarketContextPacket(BaseModel):
    """All and only the information visible to an LLM on one decision date."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: date
    ticker: str = "QQQ"
    personality: Personality
    market_features: dict[str, float]
    quantitative_signals: dict[str, float]
    proposed_action: Action | None = None
    headlines: tuple[DatedHeadline, ...] = Field(default=(), max_length=20)
    prior_assessments: tuple[PriorAssessment, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def reject_future_evidence(self) -> MarketContextPacket:
        future = [item.evidence_id for item in self.headlines if item.published_on > self.as_of]
        if future:
            raise ValueError(f"Future evidence is not allowed: {future}")
        return self


class LLMAssessment(BaseModel):
    """Structured output: concise rationale, not hidden chain-of-thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    justification: str = Field(min_length=10, max_length=1_500)
    evidence_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class LLMCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: AgentRole
    model: str
    prompt_version: str
    assessment: LLMAssessment
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached: bool = False

    def estimated_cost(self, *, input_price: float, output_price: float) -> float:
        return (self.input_tokens * input_price + self.output_tokens * output_price) / 1_000_000

    def to_agent_signal(self, *, as_of: date, personality: Personality) -> AgentSignal:
        evidence = tuple(
            Evidence(name=evidence_id, value=True, source="llm_context_packet")
            for evidence_id in self.assessment.evidence_ids
        )
        return AgentSignal(
            agent_id=self.role.value,
            agent_type="llm",
            as_of=as_of,
            signal=self.assessment.signal,
            confidence=self.assessment.confidence,
            explanation=self.assessment.justification,
            evidence=evidence,
            model_name=self.model,
            model_version=f"{self.model}:{self.prompt_version}",
            personality=personality,
        )
