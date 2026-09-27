from datetime import date

import numpy as np
import pandas as pd
import pytest

from qqq_agents.agents.quantitative import QuantitativeAgent
from qqq_agents.schemas import Personality


def training_frame() -> pd.DataFrame:
    feature = np.linspace(-1, 1, 80)
    return pd.DataFrame({"feature": feature, "target": (feature > 0).astype(int)})


def fitted_agent(target_kind: str = "direction") -> QuantitativeAgent:
    return QuantitativeAgent.fit(
        agent_id="test_agent",
        frame=training_frame(),
        feature_names=("feature",),
        target_column="target",
        target_kind=target_kind,  # type: ignore[arg-type]
        random_seed=42,
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=2,
        model_version="test-v1",
    )


def test_direction_agent_returns_normalized_contract() -> None:
    signal = fitted_agent().evaluate(
        as_of=date(2024, 1, 5),
        observation={"feature": 0.8},
        personality=Personality.CONSERVATIVE,
    )

    assert -1 <= signal.signal <= 1
    assert 0.5 <= signal.confidence <= 1
    assert signal.veto_probability is None
    assert signal.evidence[0].name == "feature"


def test_risk_agent_exposes_veto_probability() -> None:
    signal = fitted_agent(target_kind="risk").evaluate(
        as_of=date(2024, 1, 5),
        observation={"feature": 0.8},
        personality=Personality.CONSERVATIVE,
    )

    assert signal.veto_probability is not None
    assert signal.signal == pytest.approx(1 - 2 * signal.veto_probability)


def test_agent_rejects_missing_features() -> None:
    with pytest.raises(ValueError, match="Missing features"):
        fitted_agent().evaluate(
            as_of=date(2024, 1, 5),
            observation={},
            personality=Personality.CONSERVATIVE,
        )
