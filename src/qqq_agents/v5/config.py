"""Configuration for the multi-frequency V5 research track."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PeriodConfig(StrictModel):
    evaluation_start: str
    selection_end: str
    retrospective_start: str
    development_end: str
    prospective_start: str
    minimum_training_rows: int = Field(ge=750)

    @model_validator(mode="after")
    def chronological_periods(self) -> PeriodConfig:
        if not (
            self.evaluation_start <= self.selection_end
            < self.retrospective_start
            <= self.development_end
            < self.prospective_start
        ):
            raise ValueError("V5 periods must be chronological and non-overlapping")
        return self


class RiskConfig(StrictModel):
    horizons: tuple[int, ...]
    volatility_multipliers: tuple[float, ...]
    horizon_weights: tuple[float, ...]
    activation_auc: float = Field(ge=0.5, le=0.8)

    @model_validator(mode="after")
    def aligned_horizons(self) -> RiskConfig:
        if not (
            len(self.horizons)
            == len(self.volatility_multipliers)
            == len(self.horizon_weights)
        ):
            raise ValueError("V5 risk horizons, multipliers and weights must align")
        if sorted(self.horizons) != list(self.horizons) or min(self.horizons) < 2:
            raise ValueError("V5 risk horizons must be increasing and at least two sessions")
        if abs(sum(self.horizon_weights) - 1) > 1e-9:
            raise ValueError("V5 risk horizon weights must sum to one")
        return self


class ModelConfig(StrictModel):
    classifier_candidates: tuple[Literal["logistic", "histogram_gradient_boosting"], ...]
    volatility_candidates: tuple[
        Literal["har_ridge", "histogram_gradient_boosting", "trailing_realized", "vix"],
        ...,
    ]
    validation_splits: int = Field(ge=2, le=8)
    purge_sessions: int = Field(ge=5, le=40)
    random_seed: int


class AllocationConfig(StrictModel):
    target_volatility: float = Field(gt=0.1, lt=0.5)
    minimum_exposure: float = Field(ge=0, le=0.75)
    alert_cap: float = Field(gt=0, le=1)
    defensive_cap: float = Field(gt=0, le=1)
    recovery_cap: float = Field(gt=0, le=1)
    bearish_trend_cap: float = Field(gt=0, le=1)
    risk_low: float = Field(gt=0, le=5)
    risk_high: float = Field(gt=0, le=5)
    risk_sensitivity: float = Field(gt=0)
    recovery_confirmations: int = Field(ge=1, le=20)
    maximum_daily_reentry: float = Field(gt=0, le=0.5)
    minimum_rebalance: float = Field(ge=0, lt=0.25)

    @model_validator(mode="after")
    def ordered_limits(self) -> AllocationConfig:
        if self.risk_low >= self.risk_high:
            raise ValueError("risk_low must be below risk_high")
        if not self.defensive_cap <= self.recovery_cap <= self.alert_cap:
            raise ValueError("V5 state caps must be ordered")
        return self


class SelectionConfig(StrictModel):
    policy_candidates: tuple[
        Literal[
            "risk_only",
            "trend_only",
            "minimum_caps",
            "confirmation",
            "asymmetric_state_machine",
        ],
        ...,
    ]
    minimum_return_retention: float = Field(gt=0, le=1.5)
    target_drawdown_reduction: float = Field(ge=0, lt=1)


class DataConfig(StrictModel):
    start: str
    raw_directory: Path
    processed_path: Path
    assets: dict[str, str]
    cash_yield_ticker: str


class V5Config(StrictModel):
    version: str
    period: PeriodConfig
    risk: RiskConfig
    models: ModelConfig
    allocation: AllocationConfig
    selection: SelectionConfig
    data: DataConfig


def load_v5_config(path: str | Path = "configs/v5.yaml") -> V5Config:
    with Path(path).open(encoding="utf-8") as stream:
        return V5Config.model_validate(yaml.safe_load(stream))
