"""Strict configuration for the V6 risk-managed exposure experiment."""

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
    minimum_training_rows: int = Field(ge=500)

    @model_validator(mode="after")
    def chronological_periods(self) -> PeriodConfig:
        if not (
            self.evaluation_start <= self.selection_end
            < self.retrospective_start
            <= self.development_end
        ):
            raise ValueError("V6 periods must be chronological and non-overlapping")
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
            raise ValueError("V6 risk settings must align")
        if abs(sum(self.horizon_weights) - 1) > 1e-9:
            raise ValueError("V6 risk weights must sum to one")
        return self


class ReturnModelConfig(StrictModel):
    horizons: tuple[int, ...]
    horizon_weights: tuple[float, ...]
    candidates: tuple[Literal["ridge", "histogram_gradient_boosting"], ...]
    validation_splits: int = Field(ge=2, le=8)
    purge_sessions: int = Field(ge=5, le=40)
    random_seed: int

    @model_validator(mode="after")
    def aligned_horizons(self) -> ReturnModelConfig:
        if len(self.horizons) != len(self.horizon_weights):
            raise ValueError("V6 return horizons and weights must align")
        if abs(sum(self.horizon_weights) - 1) > 1e-9:
            raise ValueError("V6 return weights must sum to one")
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
    maximum_exposures: tuple[float, ...]
    target_volatility: float = Field(gt=0.1, lt=0.5)
    minimum_exposure: float = Field(ge=0, le=0.75)
    bearish_trend_cap: float = Field(gt=0, le=1)
    defensive_cap: float = Field(gt=0, le=1)
    risk_low: float = Field(gt=0)
    risk_high: float = Field(gt=0)
    risk_sensitivity: float = Field(gt=0)
    favourable_trend: float = Field(ge=0.5, le=1)
    minimum_return_score: float
    maximum_daily_reentry: float = Field(gt=0, le=0.5)
    minimum_rebalance: float = Field(ge=0, lt=0.25)
    borrowing_spread_bps: float = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def valid_limits(self) -> AllocationConfig:
        if self.risk_low >= self.risk_high:
            raise ValueError("V6 risk limits must be ordered")
        if tuple(sorted(self.maximum_exposures)) != self.maximum_exposures:
            raise ValueError("V6 maximum exposures must be ordered")
        if not self.maximum_exposures or self.maximum_exposures[0] < 1:
            raise ValueError("V6 must retain an unlevered control")
        if self.maximum_exposures[-1] > 1.25:
            raise ValueError("V6 exposure is capped at 125%")
        return self


class SelectionConfig(StrictModel):
    minimum_return_improvement: float
    maximum_drawdown_ratio: float = Field(gt=0, le=1.5)
    minimum_sharpe_improvement: float


class DataConfig(StrictModel):
    start: str
    end: str
    raw_directory: Path
    processed_path: Path
    assets: dict[str, str]
    cash_yield_ticker: str


class V6Config(StrictModel):
    version: str
    period: PeriodConfig
    risk: RiskConfig
    return_model: ReturnModelConfig
    models: ModelConfig
    allocation: AllocationConfig
    selection: SelectionConfig
    data: DataConfig


def load_v6_config(path: str | Path = "configs/v6.yaml") -> V6Config:
    with Path(path).open(encoding="utf-8") as stream:
        return V6Config.model_validate(yaml.safe_load(stream))
