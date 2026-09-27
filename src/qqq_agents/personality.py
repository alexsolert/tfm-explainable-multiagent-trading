"""Configuracion transversal de personalidades cuantitativas."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from qqq_agents.schemas import AgentSignal, Personality


class PersonalityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Personality
    signal_multiplier: float = Field(gt=0)
    confidence_floor: float = Field(ge=0, le=1)
    buy_threshold_delta: float = 0.0
    risk_weight_multiplier: float = Field(default=1.0, gt=0)
    momentum_weight_multiplier: float = Field(default=1.0, gt=0)
    risk_veto_probability_delta: float = 0.0

    def transform_weights(self, base_weights: dict[str, float]) -> dict[str, float]:
        weights = base_weights.copy()
        if "risk" in weights:
            weights["risk"] *= self.risk_weight_multiplier
        if "momentum" in weights:
            weights["momentum"] *= self.momentum_weight_multiplier
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Personality produced weights with no positive mass")
        return {name: value / total for name, value in weights.items()}

    def transform_signal(self, signal: AgentSignal) -> AgentSignal:
        adjusted_signal = signal.signal * self.signal_multiplier
        if signal.confidence < self.confidence_floor:
            adjusted_signal = 0.0
        adjusted_signal = max(-1.0, min(1.0, adjusted_signal))
        return signal.model_copy(update={"signal": adjusted_signal, "personality": self.name})


def load_personality(path: str | Path) -> PersonalityProfile:
    with Path(path).open(encoding="utf-8") as stream:
        return PersonalityProfile.model_validate(yaml.safe_load(stream))
