"""Carga y validacion de la configuracion experimental."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    random_seed: int = 42
    timezone: str


class DataConfig(StrictModel):
    ticker: str
    start: str
    end: str
    raw_path: Path
    processed_path: Path
    auto_adjust: bool = True


class ExperimentConfig(StrictModel):
    decision_frequency: str
    prediction_horizon_sessions: int = Field(gt=0)
    train_start: str
    validation_start: str
    test_start: str
    test_end: str
    transaction_cost_bps: float = Field(ge=0)
    initial_capital: float = Field(gt=0)
    risk_event_threshold: float = Field(lt=0)


class CoordinatorConfig(StrictModel):
    buy_threshold: float = Field(gt=0, le=1)
    sell_threshold: float = Field(ge=-1, lt=0)
    risk_veto_probability: float = Field(ge=0, le=1)
    weights: dict[str, float]

    @model_validator(mode="after")
    def validate_weights(self) -> CoordinatorConfig:
        if not self.weights or any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Coordinator weights must be non-negative and non-empty")
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("Coordinator weights must sum to 1.0")
        return self


class ModelConfig(StrictModel):
    kind: str
    n_estimators: int | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, gt=0)
    min_samples_leaf: int | None = Field(default=None, gt=0)


class LLMConfig(StrictModel):
    enabled: bool
    provider: str
    model: str
    reasoning_effort: Literal["minimal", "low", "medium", "high"]
    max_output_tokens: int = Field(gt=0)
    cache: bool
    budget_usd: float = Field(ge=0)
    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)


class AppConfig(StrictModel):
    project: ProjectConfig
    data: DataConfig
    experiment: ExperimentConfig
    coordinator: CoordinatorConfig
    models: dict[str, ModelConfig]
    llm: LLMConfig


def load_config(path: str | Path = "configs/base.yaml") -> AppConfig:
    """Load a YAML configuration and reject unknown or invalid fields."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return AppConfig.model_validate(payload)
