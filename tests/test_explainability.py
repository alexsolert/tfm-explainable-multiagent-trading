import numpy as np
import pandas as pd

from qqq_agents.agents.quantitative import QuantitativeAgent
from qqq_agents.explainability import LimeAgentExplainer, TreeShapAgentExplainer
from qqq_agents.explainability.cases import select_representative_cases


def fitted_agent_and_data() -> tuple[QuantitativeAgent, pd.DataFrame]:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "feature_a": rng.normal(size=100),
            "feature_b": rng.normal(size=100),
        }
    )
    frame["target"] = (frame["feature_a"] + 0.2 * frame["feature_b"] > 0).astype(int)
    agent = QuantitativeAgent.fit(
        agent_id="test_agent",
        frame=frame,
        feature_names=("feature_a", "feature_b"),
        target_column="target",
        target_kind="direction",
        random_seed=42,
        n_estimators=20,
        max_depth=3,
        min_samples_leaf=2,
        model_version="test-v1",
    )
    return agent, frame


def test_tree_shap_returns_ranked_serializable_attributions() -> None:
    agent, frame = fitted_agent_and_data()
    observation = frame.iloc[-1].to_dict()

    attributions = TreeShapAgentExplainer(agent).explain(observation, top_n=2)

    assert len(attributions) == 2
    assert {item.feature for item in attributions} == {"feature_a", "feature_b"}
    assert all(item.method == "shap" for item in attributions)
    assert all(isinstance(item.to_dict(), dict) for item in attributions)


def test_lime_returns_selected_local_attributions() -> None:
    agent, frame = fitted_agent_and_data()
    observation = frame.iloc[-1].to_dict()

    attributions = LimeAgentExplainer(agent, frame, random_seed=42).explain(
        observation, top_n=2, num_samples=200
    )

    assert len(attributions) == 2
    assert all(item.method == "lime" for item in attributions)


def test_representative_case_selection_covers_conflict_and_veto() -> None:
    index = pd.date_range("2022-01-07", periods=4, freq="W-FRI")
    decisions = pd.DataFrame(
        {
            "action": ["BUY", "HOLD", "SELL", "HOLD"],
            "score_before_veto": [0.6, 0.1, -0.7, 0.0],
            "risk_veto_triggered": [False, True, False, False],
            "risk_veto_probability": [0.2, 0.9, 0.3, 0.4],
            "technical_signal": [0.8, 0.2, -0.7, 0.9],
            "momentum_signal": [0.7, 0.1, -0.6, -0.9],
            "risk_signal": [0.4, -0.8, -0.5, 0.0],
        },
        index=index,
    )

    selected = select_representative_cases(decisions)

    assert selected["strongest_buy"] == index[0]
    assert selected["strongest_sell"] == index[2]
    assert selected["highest_risk_veto"] == index[1]
    assert selected["largest_disagreement"] == index[3]
