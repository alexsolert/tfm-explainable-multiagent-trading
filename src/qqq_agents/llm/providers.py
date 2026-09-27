"""Clientes intercambiables, cache y adaptador AutoGen/OpenAI."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Protocol

from qqq_agents.llm.contracts import (
    AgentRole,
    LLMAssessment,
    LLMCallResult,
    MarketContextPacket,
)
from qqq_agents.llm.prompts import PROMPT_VERSION, ROLE_PROMPTS


class StructuredLLMClient(Protocol):
    model: str

    async def evaluate(self, role: AgentRole, packet: MarketContextPacket) -> LLMCallResult: ...


class MockLLMClient:
    """Deterministic local substitute used to test orchestration without paid calls."""

    model = "deterministic-mock-v1"

    def __init__(self) -> None:
        self.call_count = 0

    async def evaluate(self, role: AgentRole, packet: MarketContextPacket) -> LLMCallResult:
        self.call_count += 1
        quantitative_mean = (
            sum(packet.quantitative_signals.values()) / len(packet.quantitative_signals)
            if packet.quantitative_signals
            else 0.0
        )
        if role is AgentRole.SENTIMENT and not packet.headlines:
            assessment = LLMAssessment(
                signal=0.0,
                confidence=0.0,
                justification=(
                    "No se aportaron titulares fechados; el sentimiento permanece neutral."
                ),
                limitations=("No existe evidencia textual disponible.",),
            )
        elif role is AgentRole.STRATEGIC_VALIDATOR:
            assessment = LLMAssessment(
                signal=max(-1.0, min(1.0, quantitative_mean)),
                confidence=0.6,
                justification=(
                    "La acción propuesta se contrastó con las señales de agentes disponibles."
                ),
                limitations=("Proveedor determinista de prueba, sin razonamiento semántico.",),
            )
        else:
            assessment = LLMAssessment(
                signal=max(-1.0, min(1.0, quantitative_mean)),
                confidence=0.55,
                justification=(
                    "El contexto cuantitativo fechado refleja el balance direccional indicado."
                ),
                limitations=("Proveedor determinista de prueba, sin contexto externo.",),
            )
        return LLMCallResult(
            role=role,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            assessment=assessment,
        )


class EvidenceGatedLLMClient:
    """Avoid paid inference when a role has no admissible evidence."""

    def __init__(self, client: StructuredLLMClient) -> None:
        self.client = client
        self.model = client.model

    async def evaluate(self, role: AgentRole, packet: MarketContextPacket) -> LLMCallResult:
        if role is AgentRole.SENTIMENT and not packet.headlines:
            return LLMCallResult(
                role=role,
                model="evidence-gate-v1",
                prompt_version=PROMPT_VERSION,
                assessment=LLMAssessment(
                    signal=0.0,
                    confidence=0.0,
                    justification=(
                        "No existen titulares fechados admisibles; el sentimiento se fija como "
                        "neutral sin invocar un modelo de lenguaje."
                    ),
                    limitations=(
                        "Las noticias históricas quedan pospuestas como mejora posterior.",
                    ),
                ),
            )
        return await self.client.evaluate(role, packet)


class CachedLLMClient:
    def __init__(self, client: StructuredLLMClient, cache_dir: str | Path) -> None:
        self.client = client
        self.model = client.model
        self.cache_dir = Path(cache_dir)

    def _path(self, role: AgentRole, packet: MarketContextPacket) -> Path:
        identity = json.dumps(
            {
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "role": role.value,
                "packet": packet.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return self.cache_dir / role.value / f"{digest}.json"

    async def evaluate(self, role: AgentRole, packet: MarketContextPacket) -> LLMCallResult:
        path = self._path(role, packet)
        if path.exists():
            return LLMCallResult.model_validate_json(path.read_text(encoding="utf-8")).model_copy(
                update={"cached": True}
            )
        result = await self.client.evaluate(role, packet)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


class AutoGenOpenAIClient:
    """One-shot structured AutoGen agents with no cross-date conversation memory."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        reasoning_effort: str = "low",
        max_output_tokens: int = 300,
    ) -> None:
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        self.model = model
        self._model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_output_tokens,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "gpt-5",
                "structured_output": True,
            },
        )

    async def evaluate(self, role: AgentRole, packet: MarketContextPacket) -> LLMCallResult:
        from autogen_agentchat.agents import AssistantAgent

        agent = AssistantAgent(
            name=role.value,
            model_client=self._model_client,
            system_message=ROLE_PROMPTS[role],
            output_content_type=LLMAssessment,
            reflect_on_tool_use=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="^Pydantic serializer warnings:",
                category=UserWarning,
            )
            result = await agent.run(task=packet.model_dump_json())
        message = result.messages[-1]
        assessment = (
            message.content
            if isinstance(message.content, LLMAssessment)
            else LLMAssessment.model_validate_json(str(message.content))
        )
        usage = message.models_usage
        return LLMCallResult(
            role=role,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            assessment=assessment,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def close(self) -> None:
        await self._model_client.close()
