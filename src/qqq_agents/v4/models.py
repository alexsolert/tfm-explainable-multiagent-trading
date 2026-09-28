"""Purged temporal models for daily V4 agents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qqq_agents.v4.config import ModelConfig


def _classifier(name: str, random_seed: int) -> BaseEstimator:
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2_000, random_state=random_seed),
        )
    if name == "histogram_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=160,
            max_depth=3,
            min_samples_leaf=35,
            learning_rate=0.035,
            l2_regularization=3.0,
            random_state=random_seed,
        )
    raise ValueError(f"Unknown V4 classifier: {name}")


def _positive_probability(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)  # type: ignore[attr-defined]
    return np.asarray(model.predict_proba(features)[:, classes.index(1)], dtype=float)  # type: ignore[attr-defined]


def _folds(length: int, splits: int, purge: int):
    fold_size = max(length // (splits + 1), 1)
    for fold in range(splits):
        validation_start = length - (splits - fold) * fold_size
        validation_end = min(validation_start + fold_size, length)
        training_end = validation_start - purge
        if training_end > 0:
            yield np.arange(training_end), np.arange(validation_start, validation_end)


@dataclass(frozen=True)
class ClassifierDiagnostic:
    name: str
    observations: int
    brier: float
    auc: float


@dataclass
class CalibratedDailyClassifier:
    features: tuple[str, ...]
    estimator: BaseEstimator
    calibrator: LogisticRegression
    selected_model: str
    diagnostics: tuple[ClassifierDiagnostic, ...]
    selected_brier: float
    selected_auc: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = _positive_probability(self.estimator, frame.loc[:, self.features])
        bounded = np.clip(raw, 0.001, 0.999)
        logit = np.log(bounded / (1 - bounded)).reshape(-1, 1)
        return self.calibrator.predict_proba(logit)[:, 1]


def fit_daily_classifier(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    config: ModelConfig,
) -> CalibratedDailyClassifier:
    training = frame.loc[:, [*features, target]].dropna().sort_index()
    if training[target].nunique() < 2:
        raise ValueError(f"V4 target {target} requires two classes")
    diagnostics: list[ClassifierDiagnostic] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in config.classifier_candidates:
        estimator = _classifier(name, config.random_seed)
        probability = np.full(len(training), np.nan)
        for train_indices, validation_indices in _folds(
            len(training), config.validation_splits, config.purge_sessions
        ):
            observed_train = training[target].iloc[train_indices].astype(int)
            if observed_train.nunique() < 2:
                continue
            fitted = clone(estimator).fit(
                training.loc[:, features].iloc[train_indices], observed_train
            )
            probability[validation_indices] = _positive_probability(
                fitted, training.loc[:, features].iloc[validation_indices]
            )
        valid = np.isfinite(probability)
        observed = training[target].to_numpy(dtype=int)[valid]
        predicted = probability[valid]
        if not len(predicted) or len(np.unique(observed)) < 2:
            continue
        diagnostic = ClassifierDiagnostic(
            name=name,
            observations=len(predicted),
            brier=float(brier_score_loss(observed, predicted)),
            auc=float(roc_auc_score(observed, predicted)),
        )
        diagnostics.append(diagnostic)
        predictions[name] = predicted, observed
    if not diagnostics:
        raise ValueError(f"No V4 candidate produced temporal predictions for {target}")
    selected = min(diagnostics, key=lambda item: (item.brier, -item.auc, item.name))
    oof_probability, oof_observed = predictions[selected.name]
    bounded = np.clip(oof_probability, 0.001, 0.999)
    logit = np.log(bounded / (1 - bounded)).reshape(-1, 1)
    calibrator = LogisticRegression(C=10.0, max_iter=2_000, random_state=config.random_seed)
    calibrator.fit(logit, oof_observed)
    estimator = _classifier(selected.name, config.random_seed)
    estimator.fit(training.loc[:, features], training[target].astype(int))
    return CalibratedDailyClassifier(
        features=features,
        estimator=estimator,
        calibrator=calibrator,
        selected_model=selected.name,
        diagnostics=tuple(diagnostics),
        selected_brier=selected.brier,
        selected_auc=selected.auc,
    )


@dataclass
class HarVolatilityModel:
    features: tuple[str, ...]
    estimator: BaseEstimator
    temporal_mae: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        log_forecast = self.estimator.predict(frame.loc[:, self.features])
        return np.exp(log_forecast)


def fit_har_volatility_model(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    config: ModelConfig,
) -> HarVolatilityModel:
    target = "future_realized_vol_20"
    training = frame.loc[:, [*features, target]].dropna().sort_index()
    observed: list[float] = []
    predicted: list[float] = []
    template = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    for train_indices, validation_indices in _folds(
        len(training), config.validation_splits, config.purge_sessions
    ):
        fitted = clone(template).fit(
            training.loc[:, features].iloc[train_indices],
            np.log(training[target].iloc[train_indices].clip(lower=1e-5)),
        )
        forecast = np.exp(fitted.predict(training.loc[:, features].iloc[validation_indices]))
        predicted.extend(forecast)
        observed.extend(training[target].iloc[validation_indices])
    if not predicted:
        raise ValueError("HAR temporal validation produced no volatility forecasts")
    estimator = clone(template).fit(
        training.loc[:, features], np.log(training[target].clip(lower=1e-5))
    )
    return HarVolatilityModel(
        features=features,
        estimator=estimator,
        temporal_mae=float(mean_absolute_error(observed, predicted)),
    )
