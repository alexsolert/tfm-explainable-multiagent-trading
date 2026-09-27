"""Contratos compartidos por agentes y coordinador."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Action(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Personality(StrEnum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    OPPORTUNISTIC = "opportunistic"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: float | str | bool
    source: str


class AgentSignal(BaseModel):
    """Normalized output produced by every quantitative or LLM agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    agent_type: str
    as_of: date
    signal: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()
    model_name: str
    model_version: str
    personality: Personality = Personality.CONSERVATIVE
    veto_probability: float | None = Field(default=None, ge=0, le=1)
    veto_reason: str | None = None


class WeightedContribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    signal: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    contribution: float


class DecisionTrace(BaseModel):
    """Auditable record of the coordinator's final decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: date
    created_at: datetime
    action: Action
    score_before_veto: float = Field(ge=-1, le=1)
    risk_veto_triggered: bool
    risk_veto_reason: str | None = None
    contributions: tuple[WeightedContribution, ...]
    coordinator_version: str
