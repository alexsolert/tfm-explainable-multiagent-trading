"""Long-biased continuous allocation for V3 forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qqq_agents.v3.config import AllocationConfig


@dataclass(frozen=True)
class ContinuousAllocation:
    exposure: float
    raw_exposure: float
    expected_excess_return: float
    downside_penalty: float
    risk_penalty: float
    rationale: str


class ContinuousAllocator:
    def __init__(self, config: AllocationConfig) -> None:
        self.config = config

    def allocate(
        self,
        *,
        expected_return: float,
        downside_quantile: float,
        risk_probability: float,
        cash_return: float,
        previous_exposure: float,
    ) -> ContinuousAllocation:
        excess = expected_return - cash_return
        downside_penalty = max(-downside_quantile - self.config.downside_tolerance, 0.0)
        risk_penalty = max(risk_probability - self.config.risk_tolerance, 0.0)
        raw = float(
            np.clip(
                self.config.structural_exposure
                + self.config.return_sensitivity * excess
                - self.config.downside_sensitivity * downside_penalty
                - self.config.risk_sensitivity * risk_penalty,
                0.0,
                1.0,
            )
        )
        change = raw - previous_exposure
        if abs(change) < self.config.minimum_rebalance:
            exposure = previous_exposure
            rationale = "Cambio inferior a la banda mínima; se evita rotación innecesaria."
        else:
            bounded_change = float(
                np.clip(change, -self.config.maximum_step, self.config.maximum_step)
            )
            exposure = float(np.clip(previous_exposure + bounded_change, 0.0, 1.0))
            rationale = (
                "Exposición continua ajustada por retorno esperado frente al efectivo "
                "y por el cuantil bajista previsto."
            )
        return ContinuousAllocation(
            exposure=exposure,
            raw_exposure=raw,
            expected_excess_return=excess,
            downside_penalty=downside_penalty,
            risk_penalty=risk_penalty,
            rationale=rationale,
        )
