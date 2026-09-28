"""Strict configuration for the prospective V3 research track."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchPeriodConfig(StrictModel):
    development_start: str
    development_end: str
    prospective_start: str
    minimum_training_dates: int = Field(ge=104)

    @model_validator(mode="after")
    def periods_do_not_overlap(self) -> ResearchPeriodConfig:
        if self.development_end >= self.prospective_start:
            raise ValueError("V3 development must end before the prospective period")
        return self


class ModelConfig(StrictModel):
    expected_return_candidates: tuple[Literal["ridge", "histogram_gradient_boosting"], ...]
    risk_candidates: tuple[Literal["logistic", "histogram_gradient_boosting"], ...]
    validation_splits: int = Field(ge=2, le=8)
    embargo_decisions: int = Field(ge=0, le=4)
    downside_quantile: float = Field(gt=0.05, lt=0.5)
    random_seed: int


class AllocationConfig(StrictModel):
    structural_exposure: float = Field(ge=0, le=1)
    return_sensitivity: float = Field(gt=0)
    downside_sensitivity: float = Field(gt=0)
    downside_tolerance: float = Field(gt=0, lt=0.2)
    risk_sensitivity: float = Field(gt=0)
    risk_tolerance: float = Field(gt=0, lt=1)
    minimum_rebalance: float = Field(ge=0, lt=0.5)
    maximum_step: float = Field(gt=0, le=1)


class DataConfig(StrictModel):
    start: str
    end: str
    raw_directory: Path
    panel_path: Path
    cash_path: Path
    assets: dict[str, str]
    cash_yield_ticker: str


class EvaluationConfig(StrictModel):
    target_volatility: float = Field(gt=0, lt=1)
    volatility_window: int = Field(ge=4, le=52)


class V3Config(StrictModel):
    version: str
    research: ResearchPeriodConfig
    models: ModelConfig
    allocation: AllocationConfig
    data: DataConfig
    evaluation: EvaluationConfig


def load_v3_config(path: str | Path = "configs/v3.yaml") -> V3Config:
    with Path(path).open(encoding="utf-8") as stream:
        return V3Config.model_validate(yaml.safe_load(stream))
