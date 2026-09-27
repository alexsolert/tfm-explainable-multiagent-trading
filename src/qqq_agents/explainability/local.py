"""Adaptadores SHAP y LIME con una salida comun y serializable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from qqq_agents.agents.quantitative import QuantitativeAgent


@dataclass(frozen=True)
class FeatureAttribution:
    feature: str
    value: float
    contribution: float
    method: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class TreeShapAgentExplainer:
    """Local TreeSHAP explanations for the positive model class."""

    def __init__(self, agent: QuantitativeAgent) -> None:
        import shap

        self.agent = agent
        self.explainer = shap.TreeExplainer(agent.model)
        self.positive_class_index = list(agent.model.classes_).index(1)

    def explain(
        self,
        observation: dict[str, Any],
        *,
        top_n: int | None = None,
    ) -> list[FeatureAttribution]:
        row = pd.DataFrame(
            [[observation[name] for name in self.agent.feature_names]],
            columns=self.agent.feature_names,
        )
        raw_values = self.explainer.shap_values(row)
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[self.positive_class_index])[0]
        else:
            array = np.asarray(raw_values)
            if array.ndim == 3:
                values = array[0, :, self.positive_class_index]
            elif array.ndim == 2:
                values = array[0]
            else:
                raise ValueError(f"Unsupported SHAP value shape: {array.shape}")

        attributions = [
            FeatureAttribution(
                feature=name,
                value=float(observation[name]),
                contribution=float(contribution),
                method="shap",
            )
            for name, contribution in zip(self.agent.feature_names, values, strict=True)
        ]
        attributions.sort(key=lambda item: abs(item.contribution), reverse=True)
        return attributions[:top_n] if top_n is not None else attributions


class LimeAgentExplainer:
    """LIME explanation for representative decisions, not the full backtest."""

    def __init__(
        self,
        agent: QuantitativeAgent,
        training_features: pd.DataFrame,
        *,
        random_seed: int = 42,
    ) -> None:
        from lime.lime_tabular import LimeTabularExplainer

        self.agent = agent
        self.positive_class_index = list(agent.model.classes_).index(1)
        complete = training_features.loc[:, agent.feature_names].dropna()
        if complete.empty:
            raise ValueError("LIME requires at least one complete training observation")
        self.explainer = LimeTabularExplainer(
            complete.to_numpy(dtype=float),
            feature_names=list(agent.feature_names),
            class_names=[str(value) for value in agent.model.classes_],
            mode="classification",
            discretize_continuous=True,
            random_state=random_seed,
        )

    def explain(
        self,
        observation: dict[str, Any],
        *,
        top_n: int = 5,
        num_samples: int = 1_000,
    ) -> list[FeatureAttribution]:
        row = np.asarray([observation[name] for name in self.agent.feature_names], dtype=float)

        def predict_proba(values: np.ndarray) -> np.ndarray:
            named_values = pd.DataFrame(values, columns=self.agent.feature_names)
            return self.agent.model.predict_proba(named_values)

        explanation = self.explainer.explain_instance(
            row,
            predict_proba,
            labels=(self.positive_class_index,),
            num_features=min(top_n, len(self.agent.feature_names)),
            num_samples=num_samples,
        )
        weights = explanation.as_map()[self.positive_class_index]
        return [
            FeatureAttribution(
                feature=self.agent.feature_names[index],
                value=float(observation[self.agent.feature_names[index]]),
                contribution=float(weight),
                method="lime",
            )
            for index, weight in weights
        ]
