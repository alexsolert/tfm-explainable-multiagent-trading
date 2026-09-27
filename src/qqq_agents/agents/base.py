"""Interfaz estable que permite sustituir implementaciones de agentes."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, runtime_checkable

from qqq_agents.schemas import AgentSignal, Personality


@runtime_checkable
class Agent(Protocol):
    agent_id: str

    def evaluate(
        self,
        *,
        as_of: date,
        observation: dict[str, Any],
        personality: Personality,
    ) -> AgentSignal:
        """Evaluate one dated observation without accessing future information."""
        ...
