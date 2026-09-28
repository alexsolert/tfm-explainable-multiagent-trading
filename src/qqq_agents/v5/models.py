"""Temporal volatility laboratory used by V5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qqq_agents.v5.config import ModelConfig


def _folds(length: int, splits: int, purge: int):
    fold_size = max(length // (splits + 1), 1)
    for fold in range(splits):
        validation_start = length - (splits - fold) * fold_size
        validation_end = min(validation_start + fold_size, length)
        training_end = validation_start - purge
        if training_end > 0:
            yield np.arange(training_end), np.arange(validation_start, validation_end)


def _point_estimator(name: str, seed: int) -> BaseEstimator:
    if name == "har_ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "histogram_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=180,
            max_depth=3,
            min_samples_leaf=35,
            learning_rate=0.035,
            l2_regularization=3.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown fitted volatility model: {name}")


@dataclass(frozen=True)
class VolatilityDiagnostic:
    name: str
    observations: int
    mae: float


@dataclass
class VolatilityLaboratoryModel:
    features: tuple[str, ...]
    selected_model: str
    estimator: BaseEstimator | None
    upper_estimator: BaseEstimator
    diagnostics: tuple[VolatilityDiagnostic, ...]
    temporal_mae: float
    upper_pinball_loss: float
    upper_coverage: float

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.selected_model in {"har_ridge", "histogram_gradient_boosting"}:
            if self.estimator is None:
                raise RuntimeError("Fitted volatility estimator is missing")
            point = np.exp(self.estimator.predict(frame.loc[:, self.features]))
        elif self.selected_model == "trailing_realized":
            point = frame["rv_20"].to_numpy(dtype=float)
        elif self.selected_model == "vix":
            point = frame["vix_level"].to_numpy(dtype=float) / 100
        else:
            raise ValueError(f"Unknown selected volatility model: {self.selected_model}")
        upper = np.exp(self.upper_estimator.predict(frame.loc[:, self.features]))
        point = np.clip(point, 0.03, 1.50)
        upper = np.maximum(np.clip(upper, 0.03, 1.50), point)
        return point, upper


def fit_volatility_laboratory(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    config: ModelConfig,
) -> VolatilityLaboratoryModel:
    required = [*features, target, "rv_20", "vix_level"]
    training = frame.loc[:, required].dropna().sort_index()
    diagnostics: list[VolatilityDiagnostic] = []
    for name in config.volatility_candidates:
        predicted = np.full(len(training), np.nan)
        for train_indices, validation_indices in _folds(
            len(training), config.validation_splits, config.purge_sessions
        ):
            validation = training.iloc[validation_indices]
            if name in {"har_ridge", "histogram_gradient_boosting"}:
                estimator = clone(_point_estimator(name, config.random_seed)).fit(
                    training.loc[:, features].iloc[train_indices],
                    np.log(training[target].iloc[train_indices].clip(lower=1e-5)),
                )
                forecast = np.exp(estimator.predict(validation.loc[:, features]))
            elif name == "trailing_realized":
                forecast = validation["rv_20"].to_numpy(dtype=float)
            elif name == "vix":
                forecast = validation["vix_level"].to_numpy(dtype=float) / 100
            else:
                raise ValueError(f"Unknown volatility candidate: {name}")
            predicted[validation_indices] = forecast
        valid = np.isfinite(predicted)
        diagnostics.append(
            VolatilityDiagnostic(
                name=name,
                observations=int(valid.sum()),
                mae=float(
                    mean_absolute_error(training[target].to_numpy()[valid], predicted[valid])
                ),
            )
        )
    selected = min(diagnostics, key=lambda item: (item.mae, item.name))
    estimator: BaseEstimator | None = None
    if selected.name in {"har_ridge", "histogram_gradient_boosting"}:
        estimator = clone(_point_estimator(selected.name, config.random_seed)).fit(
            training.loc[:, features], np.log(training[target].clip(lower=1e-5))
        )

    upper_template = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.75,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=35,
        learning_rate=0.035,
        l2_regularization=3.0,
        random_state=config.random_seed,
    )
    upper_oof = np.full(len(training), np.nan)
    for train_indices, validation_indices in _folds(
        len(training), config.validation_splits, config.purge_sessions
    ):
        fitted = clone(upper_template).fit(
            training.loc[:, features].iloc[train_indices],
            np.log(training[target].iloc[train_indices].clip(lower=1e-5)),
        )
        upper_oof[validation_indices] = np.exp(
            fitted.predict(training.loc[:, features].iloc[validation_indices])
        )
    upper_valid = np.isfinite(upper_oof)
    upper_observed = training[target].to_numpy()[upper_valid]
    upper_prediction = upper_oof[upper_valid]
    upper_estimator = clone(upper_template).fit(
        training.loc[:, features], np.log(training[target].clip(lower=1e-5))
    )
    return VolatilityLaboratoryModel(
        features=features,
        selected_model=selected.name,
        estimator=estimator,
        upper_estimator=upper_estimator,
        diagnostics=tuple(diagnostics),
        temporal_mae=selected.mae,
        upper_pinball_loss=float(mean_pinball_loss(upper_observed, upper_prediction, alpha=0.75)),
        upper_coverage=float((upper_observed <= upper_prediction).mean()),
    )
