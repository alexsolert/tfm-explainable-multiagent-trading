"""Agentes cuantitativos basados en clasificadores probabilisticos."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from qqq_agents.schemas import AgentSignal, Evidence, Personality

TECHNICAL_FEATURES = (
    "distance_sma_20",
    "distance_sma_50",
    "distance_sma_200",
    "ema_spread_12_26",
    "macd",
    "macd_signal",
    "macd_histogram",
    "rsi_14",
    "volume_zscore_20",
)

MOMENTUM_FEATURES = (
    "return_1d",
    "log_return_1d",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volume_zscore_20",
)

RISK_FEATURES = (
    "return_1d",
    "momentum_5",
    "momentum_20",
    "volatility_20",
    "atr_14_pct",
    "drawdown_252",
    "volume_zscore_20",
)


class QuantitativeAgent:
    """A fitted specialist that conforms to the common agent contract."""

    def __init__(
        self,
        *,
        agent_id: str,
        feature_names: tuple[str, ...],
        target_kind: Literal["direction", "risk"],
        model: RandomForestClassifier,
        model_version: str,
    ) -> None:
        self.agent_id = agent_id
        self.feature_names = feature_names
        self.target_kind = target_kind
        self.model = model
        self.model_version = model_version

    @classmethod
    def fit(
        cls,
        *,
        agent_id: str,
        frame: pd.DataFrame,
        feature_names: tuple[str, ...],
        target_column: str,
        target_kind: Literal["direction", "risk"],
        random_seed: int,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        model_version: str,
    ) -> QuantitativeAgent:
        required = [*feature_names, target_column]
        training = frame.loc[:, required].dropna()
        if training.empty:
            raise ValueError(f"No complete training rows available for {agent_id}")
        labels = training[target_column].astype(int)
        if labels.nunique() < 2:
            raise ValueError(f"{agent_id} requires both target classes")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        )
        model.fit(training.loc[:, feature_names], labels)
        return cls(
            agent_id=agent_id,
            feature_names=feature_names,
            target_kind=target_kind,
            model=model,
            model_version=model_version,
        )

    def evaluate(
        self,
        *,
        as_of: date,
        observation: dict[str, Any],
        personality: Personality,
    ) -> AgentSignal:
        missing = [name for name in self.feature_names if name not in observation]
        if missing:
            raise ValueError(f"Missing features for {self.agent_id}: {missing}")
        row = pd.DataFrame(
            [[observation[name] for name in self.feature_names]], columns=self.feature_names
        )
        if row.isna().any().any():
            raise ValueError(f"{self.agent_id} cannot evaluate missing feature values")

        positive_class_index = list(self.model.classes_).index(1)
        probability = float(self.model.predict_proba(row)[0, positive_class_index])
        confidence = max(probability, 1 - probability)
        if self.target_kind == "risk":
            signal = 1 - 2 * probability
            veto_probability = probability
            explanation = (
                f"Estimated adverse-event probability is {probability:.1%}; "
                "higher values reduce long exposure."
            )
        else:
            signal = 2 * probability - 1
            veto_probability = None
            explanation = (
                f"Estimated probability of a positive forward return is {probability:.1%}."
            )

        importance_order = self.model.feature_importances_.argsort()[::-1][:3]
        evidence = tuple(
            Evidence(
                name=self.feature_names[index],
                value=float(observation[self.feature_names[index]]),
                source="market_features",
            )
            for index in importance_order
        )
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type="quantitative",
            as_of=as_of,
            signal=signal,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            model_name="random_forest_classifier",
            model_version=self.model_version,
            personality=personality,
            veto_probability=veto_probability,
            veto_reason=(
                "The estimated probability of an adverse five-session event exceeded the limit."
                if veto_probability is not None
                else None
            ),
        )

    def save(self, destination: str | Path) -> None:
        output_path = Path(destination)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, source: str | Path) -> QuantitativeAgent:
        agent = joblib.load(source)
        if not isinstance(agent, cls):
            raise TypeError(f"Serialized object at {source} is not a QuantitativeAgent")
        return agent
