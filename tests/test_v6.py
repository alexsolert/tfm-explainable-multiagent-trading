import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_agents.config import load_config
from qqq_agents.v6 import build_v6_research_frame, load_v6_config, run_v6_walk_forward
from qqq_agents.v6.backtest import run_exposure_backtest
from qqq_agents.v6.policies import (
    build_exposure_policy,
    build_guarded_trend_policy,
    build_trend_exposure_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def _market(seed: int, index: pd.DatetimeIndex, volatility: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, volatility, len(index))
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.007,
            "low": close * 0.993,
            "close": close,
            "volume": rng.integers(1_000, 3_000, len(index)),
        },
        index=index,
    )


@pytest.fixture(scope="module")
def v6_frame() -> pd.DataFrame:
    index = pd.date_range("2010-01-04", "2019-12-31", freq="B")
    markets = {
        "qqq": _market(1, index, 0.012),
        "spy": _market(2, index),
        "vix": _market(3, index),
    }
    qqq_move = markets["qqq"]["close"].pct_change().abs().fillna(0)
    markets["vix"]["close"] = 18 + 120 * qqq_move
    config = load_v6_config(ROOT / "configs/v6.yaml")
    return build_v6_research_frame(
        markets,
        pd.Series(2.0, index=index),
        risk_config=config.risk,
        return_config=config.return_model,
    )


def test_v6_targets_are_multi_horizon_and_point_in_time(v6_frame: pd.DataFrame) -> None:
    for horizon in (5, 10, 20):
        valid = v6_frame[f"return_target_end_date_{horizon}"].notna()
        assert (
            pd.to_datetime(v6_frame.loc[valid, f"return_target_end_date_{horizon}"]).to_numpy()
            > v6_frame.index[valid].to_numpy()
        ).all()
        assert v6_frame[f"target_normalized_return_{horizon}"].notna().sum() > 1_000


def test_leveraged_backtest_charges_financing_and_delays_signal() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    close = pd.Series([100, 101, 102, 103, 104], index=index)
    result = run_exposure_backtest(
        close,
        pd.Series(1.25, index=index),
        maximum_exposure=1.25,
        transaction_cost_bps=0,
        borrowing_spread_bps=150,
        cash_return=pd.Series(0.0001, index=index),
    )
    assert result.history.iloc[0]["applied_position"] == 0
    assert result.history.iloc[1]["applied_position"] == pytest.approx(1.25)
    assert result.history.iloc[1]["financing_cost"] > 0


def test_v6_leverage_requires_joint_confirmation() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="B")
    decisions = pd.DataFrame(
        {
            "risk_score": [0.5, 0.5, 2.0, 0.5],
            "weekly_trend_score": [1.0, 0.25, 1.0, 1.0],
            "return_score": [0.2, 0.2, 0.2, -0.2],
            "forecast_volatility": [0.18] * 4,
        },
        index=index,
    )
    detail = build_exposure_policy(
        decisions,
        maximum_exposure=1.25,
        config=load_v6_config(ROOT / "configs/v6.yaml").allocation,
    )
    assert detail.iloc[0]["policy_state"] == "FAVOURABLE"
    assert detail.iloc[0]["desired_position"] > 1
    assert detail.iloc[1]["desired_position"] <= 1
    assert detail.iloc[2]["policy_state"] == "DEFENSIVE"
    assert detail.iloc[3]["desired_position"] <= 1


def test_v6_trend_policy_is_parsimonious() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="B")
    detail = build_trend_exposure_policy(
        pd.DataFrame({"weekly_trend_score": [0.75, 0.25]}, index=index),
        maximum_exposure=1.20,
        config=load_v6_config(ROOT / "configs/v6.yaml").allocation,
    )
    assert detail["desired_position"].tolist() == [1.20, 0.70]


def test_v6_volatility_guard_overrides_favourable_trend() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="B")
    detail = build_guarded_trend_policy(
        pd.DataFrame(
            {
                "weekly_trend_score": [1.0, 1.0],
                "forecast_volatility": [0.30, 0.50],
            },
            index=index,
        ),
        maximum_exposure=1.15,
        config=load_v6_config(ROOT / "configs/v6.yaml").allocation,
    )
    assert detail["desired_position"].tolist() == [1.15, 0.70]
    assert detail.iloc[1]["policy_state"] == "VOLATILITY_GUARD"


def test_v6_walk_forward_is_purged_and_bounded(v6_frame: pd.DataFrame) -> None:
    config = load_v6_config(ROOT / "configs/v6.yaml")
    config = config.model_copy(
        update={
            "period": config.period.model_copy(update={"minimum_training_rows": 500}),
            "models": config.models.model_copy(
                update={
                    "classifier_candidates": ("logistic",),
                    "volatility_candidates": ("har_ridge", "trailing_realized"),
                    "validation_splits": 2,
                }
            ),
            "return_model": config.return_model.model_copy(
                update={"candidates": ("ridge",), "validation_splits": 2}
            ),
        }
    )
    result = run_v6_walk_forward(
        v6_frame,
        app_config=load_config(ROOT / "configs/base.yaml"),
        v6_config=config,
        start="2018-01-01",
        end="2018-12-31",
    )
    assert (
        result.training_audit["maximum_target_end_date"] <= result.training_audit["cutoff"]
    ).all()
    assert result.decisions["selected_desired_position"].between(0, 1.25).all()
    assert result.selected_policy in result.policies


def test_versioned_v6_result_beats_buy_and_hold_but_preserves_uncertainty() -> None:
    metrics = json.loads(
        (ROOT / "research_results/v6/metrics.json").read_text(encoding="utf-8")
    )
    retrospective = metrics["subperiods"]["retrospective_assessment"]
    strategy = retrospective["v6_selected"]
    benchmark = retrospective["baselines"]["buy_and_hold"]

    assert metrics["selected_policy"] == "guarded_trend_115"
    assert metrics["retrospective_assessment_is_not_holdout"] is True
    assert strategy["annualized_return"] > benchmark["annualized_return"]
    assert strategy["sharpe_ratio"] > benchmark["sharpe_ratio"]
    assert strategy["maximum_drawdown"] > benchmark["maximum_drawdown"]
    bootstrap = metrics["robustness"]["retrospective_block_bootstrap"]["buy_and_hold"]
    assert bootstrap["ci_95_lower"] < 0 < bootstrap["ci_95_upper"]
