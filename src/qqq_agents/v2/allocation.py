"""Politica long-biased con tres niveles de exposicion y riesgo gradual."""

from __future__ import annotations

from dataclasses import dataclass

from qqq_agents.v2.config import AllocationConfig


@dataclass(frozen=True)
class AllocationDecision:
    exposure: float
    risk_tier: str
    rationale: str


class TacticalAllocator:
    def __init__(self, config: AllocationConfig) -> None:
        self.config = config

    def allocate(
        self, *, direction_probability: float, risk_probability: float
    ) -> AllocationDecision:
        cash, reduced, full = self.config.exposure_levels
        if risk_probability >= self.config.severe_risk_probability:
            return AllocationDecision(
                cash,
                "severe",
                "Calibrated downside risk reached the preregistered severe threshold.",
            )
        if risk_probability >= self.config.moderate_risk_probability:
            return AllocationDecision(
                reduced,
                "moderate",
                "Calibrated downside risk calls for partial rather than binary de-risking.",
            )
        if direction_probability < self.config.bearish_probability_threshold:
            return AllocationDecision(
                reduced,
                "directional_caution",
                "Directional evidence is bearish but downside risk is not severe.",
            )
        return AllocationDecision(
            full,
            "normal",
            "No sufficiently strong evidence justifies abandoning the structural QQQ exposure.",
        )
