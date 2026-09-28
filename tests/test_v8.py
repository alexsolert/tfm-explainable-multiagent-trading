from pathlib import Path

import numpy as np
import pandas as pd

from qqq_agents.v8 import evaluate_v8, load_v8_config
from qqq_agents.v8.agents import build_agent_signals
from qqq_agents.v8.coordinator import coordinate_profile, decision_action

ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    index = pd.date_range("2019-01-01", "2022-12-30", freq="B")
    close = pd.Series(100 * np.exp(np.linspace(0, 0.8, len(index))), index=index)
    return pd.DataFrame(
        {
            "close": close,
            "volatility_20": 0.20,
            "drawdown_252": 0.0,
            "qqq_relative_spy_20": 0.01,
            "cash_return": 0.0,
        },
        index=index,
    )


def test_v8_has_three_ordered_profiles() -> None:
    config = load_v8_config(ROOT / "configs/v8.yaml")
    assert [profile.name for profile in config.profiles] == [
        "conservative",
        "balanced",
        "aggressive",
    ]
    assert [profile.maximum_exposure for profile in config.profiles] == [
        1.15,
        1.50,
        1.75,
    ]


def test_agents_update_only_after_weekly_close() -> None:
    signals = build_agent_signals(
        _frame(), load_v8_config(ROOT / "configs/v8.yaml").signals
    )
    assert signals.loc["2021-01-04":"2021-01-07", "trend_score"].nunique() == 1
    assert signals["trend_score"].dropna().between(0, 1).all()
    assert signals["risk_score"].dropna().between(0, 1).all()


def test_severe_risk_vetoes_every_profile() -> None:
    config = load_v8_config(ROOT / "configs/v8.yaml")
    signals = build_agent_signals(_frame(), config.signals).dropna().tail(2).copy()
    signals.loc[:, "volatility_risk"] = 1.0
    for profile in config.profiles:
        decision = coordinate_profile(signals, profile)
        assert (decision["desired_position"] == profile.severe_risk_exposure).all()
        assert (decision["policy_state"] == "SEVERE_RISK").all()


def test_decision_action_describes_exposure_change() -> None:
    assert decision_action(1.5, 0.7) == "INCREASE"
    assert decision_action(0.5, 1.0) == "REDUCE"
    assert decision_action(1.0, 1.02) == "HOLD"


def test_v8_evaluation_includes_conventional_and_matched_baselines() -> None:
    config = load_v8_config(ROOT / "configs/v8.yaml")
    config = config.model_copy(
        update={
            "period": config.period.model_copy(
                update={
                    "selection_start": "2019-01-01",
                    "selection_end": "2020-12-31",
                    "retrospective_start": "2021-01-01",
                    "retrospective_end": "2022-12-30",
                }
            )
        }
    )
    result = evaluate_v8(_frame(), config)
    assert set(result.strategies) == {"conservative", "balanced", "aggressive"}
    assert "sma_250_cash" in result.baselines
    assert "volatility_target_35" in result.baselines
    assert set(result.matched_baselines) == {
        "matched_conservative",
        "matched_balanced",
        "matched_aggressive",
    }
    assert "balanced_explanation" in result.decisions
