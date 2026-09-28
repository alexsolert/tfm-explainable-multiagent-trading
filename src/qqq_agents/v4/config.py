"""Strict experimental configuration for daily V4."""

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
    validation_start: str
    development_end: str
    prospective_start: str
    minimum_training_rows: int = Field(ge=750)

    @model_validator(mode="after")
    def chronological_periods(self) -> PeriodConfig:
        if not (
            self.evaluation_start <= self.selection_end
            < self.validation_start
            <= self.development_end
            < self.prospective_start
        ):
            raise ValueError("V4 periods must be chronological and non-overlapping")
        return self


class ModelConfig(StrictModel):
    classifier_candidates: tuple[Literal["logistic", "histogram_gradient_boosting"], ...]
    validation_splits: int = Field(ge=2, le=8)
    purge_sessions: int = Field(ge=5, le=30)
    minimum_direction_auc: float = Field(ge=0.5, le=0.7)
    minimum_risk_auc: float = Field(ge=0.5, le=0.7)
    random_seed: int


class AllocationConfig(StrictModel):
    target_volatility: float = Field(gt=0.1, lt=0.5)
    minimum_exposure: float = Field(ge=0, le=0.75)
    bearish_trend_cap: float = Field(gt=0, le=1)
    risk_tolerance: float = Field(gt=0, lt=1)
    risk_sensitivity: float = Field(gt=0)
    direction_floor: float = Field(ge=0, le=1)
    smoothing: float = Field(gt=0, le=1)
    minimum_rebalance: float = Field(ge=0, lt=0.25)


class DataConfig(StrictModel):
    start: str
    end: str
    raw_directory: Path
    processed_path: Path
    assets: dict[str, str]
    cash_yield_ticker: str
    risk_event_threshold: float = Field(gt=-0.2, lt=0)


class V4Config(StrictModel):
    version: str
    period: PeriodConfig
    models: ModelConfig
    allocation: AllocationConfig
    data: DataConfig


def load_v4_config(path: str | Path = "configs/v4.yaml") -> V4Config:
    with Path(path).open(encoding="utf-8") as stream:
        return V4Config.model_validate(yaml.safe_load(stream))
