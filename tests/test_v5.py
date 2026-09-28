import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_agents.config import load_config
from qqq_agents.v5 import build_v5_research_frame, load_v5_config, run_v5_walk_forward
from qqq_agents.v5.policies import build_policy_positions, pareto_select_policy

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
def v5_frame() -> pd.DataFrame:
    index = pd.date_range("2012-01-03", "2019-12-31", freq="B")
    aliases = ("qqq", "spy", "iwm", "smh", "tlt", "vix", "vix3m", "hyg", "lqd", "qqqe")
    markets = {
        alias: _market(seed, index, volatility=0.012 if alias == "qqq" else 0.01)
        for seed, alias in enumerate(aliases, start=1)
    }
    qqq_move = markets["qqq"]["close"].pct_change().abs().fillna(0)
    markets["vix"]["close"] = 20 + 120 * qqq_move
    markets["vix3m"]["close"] = 22 + 80 * qqq_move
    return build_v5_research_frame(
        markets,
        pd.Series(2.0, index=index),
        risk_config=load_v5_config().risk,
    )


def test_v5_targets_are_normalised_and_point_in_time(v5_frame: pd.DataFrame) -> None:
    for horizon in (5, 10, 20):
        valid = (
            v5_frame[f"target_end_date_{horizon}"].notna()
            & v5_frame[f"dynamic_threshold_{horizon}"].notna()
        )
        assert (v5_frame.loc[valid, f"dynamic_threshold_{horizon}"] < 0).all()
        assert (
            pd.to_datetime(v5_frame.loc[valid, f"target_end_date_{horizon}"]).to_numpy()
            > v5_frame.index[valid].to_numpy()
        ).all()
        assert v5_frame[f"target_tail_{horizon}"].dropna().isin([0.0, 1.0]).all()


def test_asymmetric_policy_cuts_fast_and_reenters_gradually() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    decisions = pd.DataFrame(
        {
            "risk_score": [0.8, 2.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            "weekly_trend_score": [1.0] * 8,
            "forecast_volatility": [0.2] * 8,
        },
        index=index,
    )
    policy = build_policy_positions(
        decisions,
        policy="asymmetric_state_machine",
        config=load_v5_config().allocation,
    )

    assert policy.iloc[1]["policy_state"] == "DEFENSIVE"
    assert policy.iloc[1]["desired_position"] == pytest.approx(0.45)
    assert (policy["desired_position"].diff().dropna().loc[index[2]:] <= 0.10 + 1e-12).all()


def test_pareto_selection_uses_declared_policy_family() -> None:
    metrics = {
        "high_return": {
            "annualized_return": 0.20,
            "sharpe_ratio": 1.0,
            "maximum_drawdown": -0.20,
            "total_turnover": 3.0,
        },
        "low_risk": {
            "annualized_return": 0.18,
            "sharpe_ratio": 1.1,
            "maximum_drawdown": -0.15,
            "total_turnover": 4.0,
        },
    }
    selected = pareto_select_policy(
        metrics,
        benchmark={
            "annualized_return": 0.20,
            "sharpe_ratio": 0.9,
            "maximum_drawdown": -0.20,
        },
        config=load_v5_config().selection,
    )

    assert set(selected.pareto_policies) == {"high_return", "low_risk"}
    assert selected.selected_policy in metrics


def test_v5_walk_forward_is_purged_and_selects_on_declared_period(
    v5_frame: pd.DataFrame,
) -> None:
    config = load_v5_config()
    fast_models = config.models.model_copy(
        update={
            "classifier_candidates": ("logistic",),
            "volatility_candidates": ("har_ridge", "trailing_realized"),
            "validation_splits": 2,
        }
    )
    config = config.model_copy(update={"models": fast_models})
    result = run_v5_walk_forward(
        v5_frame,
        app_config=load_config(),
        v5_config=config,
        start="2018-01-01",
        end="2018-12-31",
    )

    assert result.selected_policy in config.selection.policy_candidates
    assert set(result.policies) == set(config.selection.policy_candidates)
    assert (
        result.training_audit["maximum_target_end_date"] <= result.training_audit["cutoff"]
    ).all()
    assert result.decisions["selected_desired_position"].between(0, 1).all()


def test_v5_cannot_open_prospective_period(v5_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="prospective period"):
        run_v5_walk_forward(
            v5_frame,
            app_config=load_config(),
            v5_config=load_v5_config(),
            start="2026-09-29",
            end="2026-10-02",
        )


def test_v5_frozen_files_match_lock() -> None:
    lock = json.loads((ROOT / "configs/v5.lock.json").read_text(encoding="utf-8"))

    for relative, expected in lock["frozen_files"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected
    assert lock["selected_policy"] == "trend_only"
    assert lock["prospective_period_consulted"] is False


def test_versioned_v5_results_remain_retrospective() -> None:
    metrics = json.loads(
        (ROOT / "research_results/v5/metrics.json").read_text(encoding="utf-8")
    )
    retrospective = metrics["subperiods"]["retrospective_assessment"]

    assert metrics["prospective_period_consulted"] is False
    assert metrics["retrospective_assessment_is_not_holdout"] is True
    assert metrics["selected_policy"] == "trend_only"
    assert retrospective["v5_selected"]["sharpe_ratio"] > retrospective["baselines"][
        "buy_and_hold"
    ]["sharpe_ratio"]
