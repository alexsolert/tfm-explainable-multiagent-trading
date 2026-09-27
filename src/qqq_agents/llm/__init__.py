"""Agentes LLM estructurados y desacoplados del proveedor."""

from qqq_agents.llm.committee import LLMCommittee
from qqq_agents.llm.contracts import AgentRole, LLMCallResult, MarketContextPacket
from qqq_agents.llm.evaluation import run_hybrid_period, save_hybrid_period
from qqq_agents.llm.pilot import execute_pilot, load_pilot_case, save_pilot_result
from qqq_agents.llm.providers import (
    AutoGenOpenAIClient,
    CachedLLMClient,
    EvidenceGatedLLMClient,
    EvidenceValidatedLLMClient,
    MockLLMClient,
)

__all__ = [
    "AgentRole",
    "AutoGenOpenAIClient",
    "CachedLLMClient",
    "EvidenceGatedLLMClient",
    "EvidenceValidatedLLMClient",
    "LLMCallResult",
    "LLMCommittee",
    "MarketContextPacket",
    "MockLLMClient",
    "execute_pilot",
    "load_pilot_case",
    "run_hybrid_period",
    "save_pilot_result",
    "save_hybrid_period",
]
