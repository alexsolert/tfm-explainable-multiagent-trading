"""Configuracion independiente y estricta para no modificar retrospectivamente V1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DevelopmentConfig(StrictModel):
    start: str
    end: str
    protected_test_start: str
    protected_test_end: str
    minimum_training_rows: int = Field(ge=104)


class ModelSelectionConfig(StrictModel):
    candidates: tuple[Literal["logistic", "random_forest", "histogram_gradient_boosting"], ...]
    calibration_splits: int = Field(ge=2, le=8)
    embargo_decisions: int = Field(ge=0, le=4)
    probability_clip: float = Field(gt=0, lt=0.1)
    adaptive_weight_shrinkage: float = Field(ge=0, le=1)
    minimum_direction_auc: float = Field(ge=0.5, le=1)
    minimum_risk_auc: float = Field(ge=0.5, le=1)

    @model_validator(mode="after")
    def candidates_are_unique(self) -> ModelSelectionConfig:
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("V2 model candidates must be unique")
        return self


class AllocationConfig(StrictModel):
    exposure_levels: tuple[float, float, float]
    bearish_probability_threshold: float = Field(gt=0, lt=0.5)
    moderate_risk_probability: float = Field(gt=0, lt=1)
    severe_risk_probability: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_levels(self) -> AllocationConfig:
        if tuple(sorted(self.exposure_levels)) != self.exposure_levels:
            raise ValueError("Exposure levels must be ordered")
        if self.exposure_levels != (0.0, 0.5, 1.0):
            raise ValueError("V2 is preregistered for exposure levels 0, 0.5 and 1")
        if self.moderate_risk_probability >= self.severe_risk_probability:
            raise ValueError("Moderate risk threshold must be below severe risk threshold")
        return self


class FeatureConfig(StrictModel):
    technical: tuple[str, ...]
    momentum: tuple[str, ...]
    risk: tuple[str, ...]
    regime: tuple[str, ...]

    @model_validator(mode="after")
    def remove_accidental_overlap(self) -> FeatureConfig:
        groups = (self.technical, self.momentum, self.risk, self.regime)
        flattened = [name for group in groups for name in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("V2 specialist feature groups must not overlap")
        return self


class ContextConfig(StrictModel):
    start: str
    end: str
    raw_directory: Path
    processed_path: Path
    tickers: dict[str, str]


class EvaluationConfig(StrictModel):
    transaction_cost_scenarios_bps: tuple[float, ...]
    calibration_bins: int = Field(ge=3, le=20)
    bootstrap_samples: int = Field(ge=100)
    bootstrap_block_weeks: int = Field(ge=2)
    random_seed: int


class V2Config(StrictModel):
    version: str
    development: DevelopmentConfig
    models: ModelSelectionConfig
    allocation: AllocationConfig
    features: FeatureConfig
    context: ContextConfig
    evaluation: EvaluationConfig


def load_v2_config(path: str | Path = "configs/v2.yaml") -> V2Config:
    with Path(path).open(encoding="utf-8") as stream:
        return V2Config.model_validate(yaml.safe_load(stream))
