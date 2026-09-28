"""Seleccion temporal de modelos y calibracion Platt sin mezclar futuro y pasado."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qqq_agents.v2.config import ModelSelectionConfig


def _candidate(name: str, *, random_seed: int) -> BaseEstimator:
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=random_seed),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            class_weight=None,
            random_state=random_seed,
            n_jobs=-1,
        )
    if name == "histogram_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=3,
            min_samples_leaf=15,
            learning_rate=0.04,
            l2_regularization=1.0,
            random_state=random_seed,
        )
    raise ValueError(f"Unknown V2 candidate: {name}")


def _positive_probability(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)  # type: ignore[attr-defined]
    positive_index = classes.index(1)
    return np.asarray(model.predict_proba(features)[:, positive_index], dtype=float)  # type: ignore[attr-defined]


def _logit(probability: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(probability, clip, 1 - clip)
    return np.log(bounded / (1 - bounded)).reshape(-1, 1)


@dataclass(frozen=True)
class CandidateDiagnostic:
    name: str
    observations: int
    raw_brier: float
    raw_auc: float


@dataclass
class TemporalProbabilityModel:
    feature_names: tuple[str, ...]
    estimator: BaseEstimator
    calibrator: LogisticRegression | None
    selected_model: str
    diagnostics: tuple[CandidateDiagnostic, ...]
    selected_raw_brier: float
    selected_raw_auc: float
    probability_clip: float

    def predict_raw_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return _positive_probability(self.estimator, frame.loc[:, self.feature_names])

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw_probability(frame)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(_logit(raw, self.probability_clip))[:, 1]


def _oof_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    target_end_dates: pd.Series,
    *,
    splits: int,
    embargo_decisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.full(len(features), np.nan, dtype=float)
    splitter = TimeSeriesSplit(n_splits=splits)
    for train_indices, validation_indices in splitter.split(features):
        validation_start = features.index[validation_indices[0]]
        train_indices = train_indices[
            target_end_dates.iloc[train_indices].to_numpy() < validation_start
        ]
        if embargo_decisions:
            train_indices = train_indices[:-embargo_decisions]
        if not len(train_indices):
            continue
        train_target = target.iloc[train_indices]
        if train_target.nunique() < 2:
            continue
        fitted = clone(estimator)
        fitted.fit(features.iloc[train_indices], train_target)
        probabilities[validation_indices] = _positive_probability(
            fitted, features.iloc[validation_indices]
        )
    valid = np.isfinite(probabilities)
    return probabilities[valid], target.to_numpy(dtype=int)[valid]


def fit_temporal_probability_model(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    target_column: str,
    config: ModelSelectionConfig,
    random_seed: int,
) -> TemporalProbabilityModel:
    training = (
        frame.loc[:, [*feature_names, target_column, "target_end_date"]]
        .dropna()
        .sort_index()
    )
    if training.empty or training[target_column].nunique() < 2:
        raise ValueError(f"V2 requires two target classes for {target_column}")
    features = training.loc[:, feature_names]
    target = training[target_column].astype(int)
    diagnostics: list[CandidateDiagnostic] = []
    candidate_oof: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in config.candidates:
        estimator = _candidate(name, random_seed=random_seed)
        probability, observed = _oof_probabilities(
            estimator,
            features,
            target,
            pd.to_datetime(training["target_end_date"]),
            splits=config.calibration_splits,
            embargo_decisions=config.embargo_decisions,
        )
        if not len(probability) or len(np.unique(observed)) < 2:
            continue
        diagnostic = CandidateDiagnostic(
            name=name,
            observations=len(probability),
            raw_brier=float(brier_score_loss(observed, probability)),
            raw_auc=float(roc_auc_score(observed, probability)),
        )
        diagnostics.append(diagnostic)
        candidate_oof[name] = probability, observed
    if not diagnostics:
        raise ValueError("No V2 candidate produced valid temporal predictions")

    selected = min(diagnostics, key=lambda item: (item.raw_brier, -item.raw_auc, item.name))
    raw_probability, observed = candidate_oof[selected.name]
    calibrator: LogisticRegression | None = None
    if len(np.unique(observed)) == 2:
        calibrator = LogisticRegression(C=10.0, max_iter=2_000, random_state=random_seed)
        calibrator.fit(_logit(raw_probability, config.probability_clip), observed)

    final_estimator = _candidate(selected.name, random_seed=random_seed)
    final_estimator.fit(features, target)
    return TemporalProbabilityModel(
        feature_names=feature_names,
        estimator=final_estimator,
        calibrator=calibrator,
        selected_model=selected.name,
        diagnostics=tuple(diagnostics),
        selected_raw_brier=selected.raw_brier,
        selected_raw_auc=selected.raw_auc,
        probability_clip=config.probability_clip,
    )


def regularized_quality_weights(
    models: dict[str, TemporalProbabilityModel], *, shrinkage: float
) -> dict[str, float]:
    """Shrink inverse-Brier weights towards equal weights to limit regime chasing."""

    if not models:
        raise ValueError("At least one directional model is required")
    raw = {name: 1 / max(model.selected_raw_brier, 1e-6) for name, model in models.items()}
    raw_total = sum(raw.values())
    equal = 1 / len(models)
    weights = {
        name: shrinkage * equal + (1 - shrinkage) * value / raw_total
        for name, value in raw.items()
    }
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}
