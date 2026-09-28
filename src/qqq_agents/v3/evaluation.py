"""Leakage-safe cross-asset walk-forward evaluation for V3 development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import BacktestResult, run_backtest
from qqq_agents.config import AppConfig
from qqq_agents.v3.allocation import ContinuousAllocator
from qqq_agents.v3.config import V3Config
from qqq_agents.v3.features import PANEL_FEATURES, RISK_FEATURES


@dataclass(frozen=True)
class V3WalkForwardResult:
    decisions: pd.DataFrame
    strategy: BacktestResult
    baselines: dict[str, BacktestResult]
    training_audit: pd.DataFrame
    model_leaderboard: pd.DataFrame


def _candidate(name: str, random_seed: int) -> BaseEstimator:
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "histogram_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=3,
            min_samples_leaf=25,
            learning_rate=0.04,
            l2_regularization=2.0,
            random_state=random_seed,
        )
    raise ValueError(f"Unknown V3 model candidate: {name}")


def _risk_candidate(name: str, random_seed: int) -> BaseEstimator:
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2_000, random_state=random_seed),
        )
    if name == "histogram_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=3,
            min_samples_leaf=25,
            learning_rate=0.04,
            l2_regularization=2.0,
            random_state=random_seed,
        )
    raise ValueError(f"Unknown V3 risk candidate: {name}")


def _positive_probability(model: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)  # type: ignore[attr-defined]
    return np.asarray(model.predict_proba(features)[:, classes.index(1)], dtype=float)  # type: ignore[attr-defined]


def _temporal_mae(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    splits: int,
    embargo: int,
) -> tuple[float, int]:
    dates = pd.DatetimeIndex(frame.index.get_level_values("date").unique()).sort_values()
    fold_size = max(len(dates) // (splits + 1), 1)
    predicted: list[float] = []
    observed: list[float] = []
    for fold in range(splits):
        validation_start_index = len(dates) - (splits - fold) * fold_size
        validation_end_index = min(validation_start_index + fold_size, len(dates))
        if validation_start_index <= embargo:
            continue
        training_dates = dates[: validation_start_index - embargo]
        validation_dates = dates[validation_start_index:validation_end_index]
        training = frame.loc[frame.index.get_level_values("date").isin(training_dates)]
        validation = frame.loc[frame.index.get_level_values("date").isin(validation_dates)]
        if training.empty or validation.empty:
            continue
        fitted = clone(estimator).fit(training.loc[:, features], training["forward_return"])
        predicted.extend(fitted.predict(validation.loc[:, features]))
        observed.extend(validation["forward_return"])
    if not predicted:
        raise ValueError("V3 temporal validation produced no forecasts")
    return float(mean_absolute_error(observed, predicted)), len(predicted)


def _temporal_risk_predictions(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    splits: int,
    embargo: int,
) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(frame.index.get_level_values("date").unique()).sort_values()
    fold_size = max(len(dates) // (splits + 1), 1)
    predicted: list[float] = []
    observed: list[int] = []
    for fold in range(splits):
        validation_start_index = len(dates) - (splits - fold) * fold_size
        validation_end_index = min(validation_start_index + fold_size, len(dates))
        if validation_start_index <= embargo:
            continue
        training_dates = dates[: validation_start_index - embargo]
        validation_dates = dates[validation_start_index:validation_end_index]
        training = frame.loc[frame.index.get_level_values("date").isin(training_dates)]
        validation = frame.loc[frame.index.get_level_values("date").isin(validation_dates)]
        if training["target_risk"].nunique() < 2 or validation.empty:
            continue
        fitted = clone(estimator).fit(training.loc[:, features], training["target_risk"])
        predicted.extend(_positive_probability(fitted, validation.loc[:, features]))
        observed.extend(validation["target_risk"].astype(int))
    if not predicted or len(set(observed)) < 2:
        raise ValueError("V3 temporal risk validation produced no valid forecasts")
    return np.asarray(predicted), np.asarray(observed)


def _volatility_target_position(
    close: pd.Series, *, target: float, window: int, periods_per_year: int = 52
) -> pd.Series:
    realised = close.pct_change().rolling(window).std() * np.sqrt(periods_per_year)
    return (target / realised.replace(0, np.nan)).clip(0, 1).fillna(1.0)


def run_v3_walk_forward(
    panel: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    app_config: AppConfig,
    v3_config: V3Config,
    start: str | None = None,
    end: str | None = None,
    allow_prospective: bool = False,
) -> V3WalkForwardResult:
    prepared = panel.copy().sort_index()
    if prepared.index.names != ["date", "asset"]:
        raise ValueError("V3 panel index must be ['date', 'asset']")
    features = tuple(name for name in PANEL_FEATURES if name in prepared.columns)
    if len(features) != len(PANEL_FEATURES):
        missing = set(PANEL_FEATURES) - set(features)
        raise ValueError(f"V3 panel is missing features: {sorted(missing)}")
    start_at = pd.Timestamp(start or v3_config.research.development_start)
    end_at = pd.Timestamp(end or v3_config.research.development_end)
    if end_at >= pd.Timestamp(v3_config.research.prospective_start) and not allow_prospective:
        raise ValueError("V3 development cannot open the prospective period")

    qqq = prepared.xs("qqq", level="asset").loc[start_at:end_at]
    if qqq.empty:
        raise ValueError("V3 QQQ evaluation period contains no observations")
    cash = cash_returns.reindex(qqq.index).ffill().fillna(0.0)
    allocator = ContinuousAllocator(v3_config.allocation)
    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    leaderboard: list[dict[str, object]] = []
    previous_exposure = v3_config.allocation.structural_exposure

    for year in range(start_at.year, end_at.year + 1):
        period_start = max(start_at, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end_at, pd.Timestamp(year=year, month=12, day=31))
        cutoff = period_start - pd.Timedelta(days=1)
        target_dates = pd.to_datetime(prepared["target_end_date"])
        eligible = prepared.loc[
            (prepared.index.get_level_values("date") <= cutoff) & (target_dates <= cutoff),
            [*features, "forward_return", "target_risk", "target_end_date"],
        ].dropna()
        date_count = eligible.index.get_level_values("date").nunique()
        if date_count < v3_config.research.minimum_training_dates:
            raise ValueError(f"Only {date_count} V3 training dates available at {cutoff.date()}")

        diagnostics: list[tuple[str, float, int]] = []
        for name in v3_config.models.expected_return_candidates:
            mae, observations = _temporal_mae(
                _candidate(name, v3_config.models.random_seed),
                eligible,
                features,
                splits=v3_config.models.validation_splits,
                embargo=v3_config.models.embargo_decisions,
            )
            diagnostics.append((name, mae, observations))
            leaderboard.append(
                {
                    "cutoff": cutoff,
                    "target": "expected_return",
                    "candidate": name,
                    "mae": mae,
                    "observations": observations,
                }
            )
        selected_name, selected_mae, _ = min(diagnostics, key=lambda item: (item[1], item[0]))
        expected_model = _candidate(selected_name, v3_config.models.random_seed)
        expected_model.fit(eligible.loc[:, features], eligible["forward_return"])
        downside_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=v3_config.models.downside_quantile,
            max_iter=150,
            max_depth=3,
            min_samples_leaf=25,
            learning_rate=0.04,
            l2_regularization=2.0,
            random_state=v3_config.models.random_seed,
        ).fit(eligible.loc[:, features], eligible["forward_return"])
        risk_training = eligible.loc[
            eligible.index.get_level_values("asset") == "qqq"
        ].copy()
        risk_diagnostics: list[tuple[str, float, float, np.ndarray, np.ndarray]] = []
        for name in v3_config.models.risk_candidates:
            probability, observed = _temporal_risk_predictions(
                _risk_candidate(name, v3_config.models.random_seed),
                risk_training,
                RISK_FEATURES,
                splits=v3_config.models.validation_splits,
                embargo=v3_config.models.embargo_decisions,
            )
            brier = float(brier_score_loss(observed, probability))
            auc = float(roc_auc_score(observed, probability))
            risk_diagnostics.append((name, brier, auc, probability, observed))
            leaderboard.append(
                {
                    "cutoff": cutoff,
                    "target": "risk_event",
                    "candidate": name,
                    "mae": np.nan,
                    "brier": brier,
                    "auc": auc,
                    "observations": len(observed),
                }
            )
        risk_name, risk_brier, risk_auc, risk_oof, risk_observed = min(
            risk_diagnostics, key=lambda item: (item[1], -item[2], item[0])
        )
        bounded_risk_oof = np.clip(risk_oof, 0.001, 0.999)
        risk_logit = np.log(bounded_risk_oof / (1 - bounded_risk_oof)).reshape(-1, 1)
        risk_calibrator = LogisticRegression(
            C=10.0, max_iter=2_000, random_state=v3_config.models.random_seed
        ).fit(risk_logit, risk_observed)
        risk_model = _risk_candidate(risk_name, v3_config.models.random_seed)
        risk_model.fit(risk_training.loc[:, RISK_FEATURES], risk_training["target_risk"])
        audits.append(
            {
                "cutoff": cutoff,
                "training_rows": len(eligible),
                "training_dates": date_count,
                "training_assets": eligible.index.get_level_values("asset").nunique(),
                "maximum_target_end_date": eligible["target_end_date"].max(),
                "selected_model": selected_name,
                "selected_mae": selected_mae,
                "selected_risk_model": risk_name,
                "selected_risk_brier": risk_brier,
                "selected_risk_auc": risk_auc,
                "risk_training_rows": len(risk_training),
            }
        )
        rows = qqq.loc[period_start:period_end].dropna(subset=list(features))
        expected = expected_model.predict(rows.loc[:, features])
        downside = downside_model.predict(rows.loc[:, features])
        raw_risk = _positive_probability(risk_model, rows.loc[:, RISK_FEATURES])
        bounded_raw_risk = np.clip(raw_risk, 0.001, 0.999)
        risk = risk_calibrator.predict_proba(
            np.log(bounded_raw_risk / (1 - bounded_raw_risk)).reshape(-1, 1)
        )[:, 1]
        for timestamp, expected_return, downside_quantile, risk_probability in zip(
            rows.index, expected, downside, risk, strict=True
        ):
            allocation = allocator.allocate(
                expected_return=float(expected_return),
                downside_quantile=float(downside_quantile),
                risk_probability=float(risk_probability),
                cash_return=float(cash.loc[timestamp]),
                previous_exposure=previous_exposure,
            )
            records.append(
                {
                    "date": timestamp,
                    "expected_return": expected_return,
                    "downside_quantile": downside_quantile,
                    "risk_probability": risk_probability,
                    "cash_return": cash.loc[timestamp],
                    "previous_exposure": previous_exposure,
                    "raw_exposure": allocation.raw_exposure,
                    "desired_position": allocation.exposure,
                    "expected_excess_return": allocation.expected_excess_return,
                    "downside_penalty": allocation.downside_penalty,
                    "risk_penalty": allocation.risk_penalty,
                    "selected_model": selected_name,
                    "selected_risk_model": risk_name,
                    "rationale": allocation.rationale,
                }
            )
            previous_exposure = allocation.exposure

    decisions = pd.DataFrame.from_records(records).set_index("date").sort_index()
    close = qqq.loc[decisions.index, "close"]
    cash = cash.reindex(decisions.index)
    costs = app_config.experiment.transaction_cost_bps
    strategy = run_backtest(
        close,
        decisions["desired_position"],
        transaction_cost_bps=costs,
        cash_return=cash,
    )
    sma_50 = close / (1 + qqq.loc[decisions.index, "distance_sma_50"])
    sma_200 = close / (1 + qqq.loc[decisions.index, "distance_sma_200"])
    baselines = {
        "buy_and_hold": run_backtest(
            close, buy_and_hold(close), transaction_cost_bps=costs, cash_return=cash
        ),
        "sma_50_200": run_backtest(
            close,
            (sma_50 > sma_200).astype(float),
            transaction_cost_bps=costs,
            cash_return=cash,
        ),
        "volatility_target": run_backtest(
            close,
            _volatility_target_position(
                close,
                target=v3_config.evaluation.target_volatility,
                window=v3_config.evaluation.volatility_window,
            ),
            transaction_cost_bps=costs,
            cash_return=cash,
        ),
    }
    return V3WalkForwardResult(
        decisions=decisions,
        strategy=strategy,
        baselines=baselines,
        training_audit=pd.DataFrame.from_records(audits),
        model_leaderboard=pd.DataFrame.from_records(leaderboard),
    )
