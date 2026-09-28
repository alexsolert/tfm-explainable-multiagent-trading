"""Strict configuration for V8."""

from __future__ import annotations

from pathlib import Path

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
            raise ValueError("V8 periods must be chronological and non-overlapping")
        return self


class SignalConfig(StrictModel):
    moving_average_sessions: tuple[int, ...]
    momentum_sessions: tuple[int, ...]
    high_volatility: float = Field(gt=0.2, lt=1)
    severe_volatility: float = Field(gt=0.2, lt=1.5)
    moderate_drawdown: float = Field(gt=-1, lt=0)
    severe_drawdown: float = Field(gt=-1, lt=0)

    @model_validator(mode="after")
    def ordered_thresholds(self) -> SignalConfig:
        if self.high_volatility >= self.severe_volatility:
            raise ValueError("V8 volatility thresholds must be ordered")
        if self.severe_drawdown >= self.moderate_drawdown:
            raise ValueError("V8 drawdown thresholds must be ordered")
        return self


class ProfileConfig(StrictModel):
    name: str
    label: str
    maximum_exposure: float = Field(ge=1, le=2)
    defensive_exposure: float = Field(ge=0, le=1)
    severe_risk_exposure: float = Field(ge=0, le=1)
    volatility_target: float | None = Field(default=None, ge=0.1, le=0.5)
    minimum_exposure: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def valid_exposure_order(self) -> ProfileConfig:
        if self.severe_risk_exposure > self.defensive_exposure:
            raise ValueError("V8 severe-risk exposure cannot exceed defensive exposure")
        if self.minimum_exposure > self.maximum_exposure:
            raise ValueError("V8 minimum exposure cannot exceed maximum exposure")
        return self


class CostConfig(StrictModel):
    transaction_cost_bps: float = Field(ge=0, le=100)
    borrowing_spread_bps: float = Field(ge=0, le=1000)


class RobustnessConfig(StrictModel):
    bootstrap_samples: int = Field(ge=100)
    bootstrap_block_sessions: int = Field(ge=5)
    candidate_trials: int = Field(ge=1)
    random_seed: int


class V8Config(StrictModel):
    version: str
    period: PeriodConfig
    signals: SignalConfig
    profiles: tuple[ProfileConfig, ...]
    costs: CostConfig
    robustness: RobustnessConfig

    @model_validator(mode="after")
    def unique_profiles(self) -> V8Config:
        names = [profile.name for profile in self.profiles]
        if len(names) != len(set(names)):
            raise ValueError("V8 profile names must be unique")
        return self


def load_v8_config(path: str | Path = "configs/v8.yaml") -> V8Config:
    with Path(path).open(encoding="utf-8") as stream:
        return V8Config.model_validate(yaml.safe_load(stream))
