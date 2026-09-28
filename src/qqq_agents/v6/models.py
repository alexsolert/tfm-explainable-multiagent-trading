"""Purged temporal return models for V6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qqq_agents.v6.config import ReturnModelConfig


def _folds(length: int, splits: int, purge: int):
    fold_size = max(length // (splits + 1), 1)
    for fold in range(splits):
        validation_start = length - (splits - fold) * fold_size
        validation_end = min(validation_start + fold_size, length)
        training_end = validation_start - purge
        if training_end > 0:
            yield np.arange(training_end), np.arange(validation_start, validation_end)


def _estimator(name: str, seed: int) -> BaseEstimator:
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=25.0))
    if name == "histogram_gradient_boosting":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=160,
            max_depth=2,
            min_samples_leaf=50,
            learning_rate=0.03,
            l2_regularization=5.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown V6 return model: {name}")


@dataclass(frozen=True)
class ReturnDiagnostic:
    name: str
    observations: int
    mae: float
    correlation: float


@dataclass
class ReturnModel:
    features: tuple[str, ...]
    selected_model: str
    estimator: BaseEstimator
    diagnostics: tuple[ReturnDiagnostic, ...]
    temporal_mae: float
    temporal_correlation: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(frame.loc[:, self.features]), dtype=float)


def fit_return_model(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    config: ReturnModelConfig,
) -> ReturnModel:
    training = frame.loc[:, [*features, target]].dropna().sort_index()
    diagnostics: list[ReturnDiagnostic] = []
    for name in config.candidates:
        predictions = np.full(len(training), np.nan)
        template = _estimator(name, config.random_seed)
        for train_indices, validation_indices in _folds(
            len(training), config.validation_splits, config.purge_sessions
        ):
            fitted = clone(template).fit(
                training.loc[:, features].iloc[train_indices],
                training[target].iloc[train_indices].clip(-4, 4),
            )
            predictions[validation_indices] = fitted.predict(
                training.loc[:, features].iloc[validation_indices]
            )
        valid = np.isfinite(predictions)
        observed = training[target].to_numpy(dtype=float)[valid]
        predicted = predictions[valid]
        if not len(predicted):
            continue
        correlation = np.corrcoef(observed, predicted)[0, 1]
        diagnostics.append(
            ReturnDiagnostic(
                name=name,
                observations=int(valid.sum()),
                mae=float(mean_absolute_error(observed, predicted)),
                correlation=float(correlation) if np.isfinite(correlation) else 0.0,
            )
        )
    if not diagnostics:
        raise ValueError(f"No V6 return candidate produced predictions for {target}")
    selected = min(diagnostics, key=lambda item: (item.mae, -item.correlation, item.name))
    estimator = _estimator(selected.name, config.random_seed).fit(
        training.loc[:, features], training[target].clip(-4, 4)
    )
    return ReturnModel(
        features=features,
        selected_model=selected.name,
        estimator=estimator,
        diagnostics=tuple(diagnostics),
        temporal_mae=selected.mae,
        temporal_correlation=selected.correlation,
    )
