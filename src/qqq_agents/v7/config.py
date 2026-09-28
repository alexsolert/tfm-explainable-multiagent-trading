"""Strict configuration for the V7 growth experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PeriodConfig(StrictModel):
    selection_start: str
    selection_end: str
    retrospective_start: str
    retrospective_end: str

    @model_validator(mode="after")
    def chronological(self) -> PeriodConfig:
        if not (
            self.selection_start <= self.selection_end
            < self.retrospective_start
            <= self.retrospective_end
        ):
            raise ValueError("V7 periods must be chronological and non-overlapping")
        return self


class PolicyConfig(StrictModel):
    name: str
    family: Literal["trend_guard", "volatility_managed_trend"]
    moving_average_sessions: int = Field(ge=100, le=300)
    maximum_exposure: float = Field(ge=1, le=2)
    defensive_exposure: float = Field(ge=0, le=1)
    volatility_guard: float | None = Field(default=None, ge=0.2, le=1.5)
    volatility_target: float | None = Field(default=None, ge=0.1, le=0.5)
    minimum_exposure: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def family_parameters(self) -> PolicyConfig:
        if self.family == "trend_guard" and self.volatility_guard is None:
            raise ValueError("trend_guard requires volatility_guard")
        if self.family == "volatility_managed_trend" and (
            self.volatility_target is None or self.minimum_exposure is None
        ):
            raise ValueError(
                "volatility_managed_trend requires volatility_target and minimum_exposure"
            )
        return self


class CostConfig(StrictModel):
    transaction_cost_bps: float = Field(ge=0, le=100)
    borrowing_spread_bps: float = Field(ge=0, le=1000)


class V7Config(StrictModel):
    version: str
    period: PeriodConfig
    policies: tuple[PolicyConfig, ...]
    costs: CostConfig


def load_v7_config(path: str | Path = "configs/v7.yaml") -> V7Config:
    with Path(path).open(encoding="utf-8") as stream:
        return V7Config.model_validate(yaml.safe_load(stream))
