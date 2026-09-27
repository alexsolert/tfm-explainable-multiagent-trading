"""Agentes LLM estructurados y desacoplados del proveedor."""

from qqq_agents.llm.committee import LLMCommittee
from qqq_agents.llm.contracts import AgentRole, LLMCallResult, MarketContextPacket
from qqq_agents.llm.pilot import execute_pilot, load_pilot_case, save_pilot_result
from qqq_agents.llm.providers import AutoGenOpenAIClient, CachedLLMClient, MockLLMClient

__all__ = [
    "AgentRole",
    "AutoGenOpenAIClient",
    "CachedLLMClient",
    "LLMCallResult",
    "LLMCommittee",
    "MarketContextPacket",
    "MockLLMClient",
    "execute_pilot",
    "load_pilot_case",
    "save_pilot_result",
]
