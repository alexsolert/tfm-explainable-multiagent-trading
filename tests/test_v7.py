from pathlib import Path

import pandas as pd
import pytest

from qqq_agents.v7 import evaluate_v7, load_v7_config
from qqq_agents.v7.backtest import run_v7_backtest
from qqq_agents.v7.policies import build_v7_policy

ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    close = pd.Series(range(100, 500), index=index, dtype=float)
    return pd.DataFrame(
        {"close": close, "volatility_20": 0.20, "cash_return": 0.0}, index=index
    )


def test_v7_config_predeclares_two_growth_profiles() -> None:
    config = load_v7_config(ROOT / "configs/v7.yaml")
    assert config.version == "7.0-growth-frontier"
    assert {policy.name for policy in config.policies} == {
        "robust_growth",
        "high_growth",
    }


def test_v7_weekly_signal_is_bounded_and_point_in_time() -> None:
    frame = _frame()
    policy = load_v7_config(ROOT / "configs/v7.yaml").policies[0]
    detail = build_v7_policy(frame, policy)
    assert detail["desired_position"].between(0, policy.maximum_exposure).all()
    assert detail.loc["2021-01-04":"2021-01-07", "desired_position"].nunique() == 1


def test_v7_backtest_delays_signal_and_charges_financing() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    result = run_v7_backtest(
        pd.Series([100, 101, 102, 103], index=index),
        pd.Series(1.75, index=index),
        maximum_exposure=1.75,
        transaction_cost_bps=10,
        borrowing_spread_bps=150,
        cash_return=pd.Series(0.0001, index=index),
    )
    assert result.history.iloc[0]["applied_position"] == 0
    assert result.history.iloc[1]["applied_position"] == pytest.approx(1.75)
    assert result.history.iloc[1]["financing_cost"] > 0


def test_v7_evaluation_includes_equal_leverage_controls() -> None:
    config = load_v7_config(ROOT / "configs/v7.yaml")
    config = config.model_copy(
        update={
            "period": config.period.model_copy(
                update={
                    "selection_start": "2020-01-01",
                    "selection_end": "2020-12-31",
                    "retrospective_start": "2021-01-01",
                    "retrospective_end": "2021-07-13",
                }
            )
        }
    )
    result = evaluate_v7(_frame(), config)
    assert set(result.baselines) == {"constant_100", "constant_150", "constant_175"}
    assert set(result.subperiods) == {"selection", "retrospective"}
