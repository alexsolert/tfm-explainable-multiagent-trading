import numpy as np
import pandas as pd
import pytest

from qqq_agents.config import load_config
from qqq_agents.v4 import (
    build_daily_research_frame,
    deflated_sharpe_probability,
    load_v4_config,
    run_v4_walk_forward,
)
from qqq_agents.v4.features import DIRECTION_FEATURES, RISK_FEATURES, VOLATILITY_FEATURES


def _market(seed: int, index: pd.DatetimeIndex, volatility: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, volatility, len(index))
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, len(index))),
            "high": close * 1.007,
            "low": close * 0.993,
            "close": close,
            "volume": rng.integers(1_000, 3_000, len(index)),
        },
        index=index,
    )


def synthetic_v4_frame() -> pd.DataFrame:
    index = pd.date_range("2012-01-03", "2021-12-31", freq="B")
    aliases = ("qqq", "spy", "iwm", "smh", "tlt", "vix", "vix3m", "hyg", "lqd", "qqqe")
    markets = {
        alias: _market(seed, index, volatility=0.012 if alias == "qqq" else 0.01)
        for seed, alias in enumerate(aliases, start=1)
    }
    markets["vix"]["close"] = 20 + 100 * markets["qqq"]["close"].pct_change().abs().fillna(0)
    markets["vix3m"]["close"] = 22 + 80 * markets["qqq"]["close"].pct_change().abs().fillna(0)
    annual_yield = pd.Series(2.0, index=index)
    return build_daily_research_frame(markets, annual_yield)


def test_daily_frame_contains_agent_specific_features_and_targets() -> None:
    frame = synthetic_v4_frame()

    assert set(RISK_FEATURES) <= set(frame.columns)
    assert set(DIRECTION_FEATURES) <= set(frame.columns)
    assert set(VOLATILITY_FEATURES) <= set(frame.columns)
    assert {"future_realized_vol_20", "cash_return", "vol_target_end_date"} <= set(
        frame.columns
    )
    valid = frame["vol_target_end_date"].notna()
    assert (
        pd.to_datetime(frame.loc[valid, "vol_target_end_date"]).to_numpy()
        > frame.index[valid].to_numpy()
    ).all()


def test_v4_daily_walk_forward_is_leakage_safe() -> None:
    config = load_v4_config()
    fast_models = config.models.model_copy(
        update={"classifier_candidates": ("logistic",), "validation_splits": 2}
    )
    config = config.model_copy(update={"models": fast_models})
    result = run_v4_walk_forward(
        synthetic_v4_frame(),
        app_config=load_config(),
        v4_config=config,
        start="2018-01-01",
        end="2019-12-31",
    )

    assert result.decisions["desired_position"].between(0, 1).all()
    assert (
        result.training_audit["maximum_target_end_date"] <= result.training_audit["cutoff"]
    ).all()
    assert {"risk_only", "volatility_only", "trend_only", "direction_only"} <= set(
        result.ablations
    )


def test_v4_development_cannot_open_prospective_period() -> None:
    with pytest.raises(ValueError, match="prospective period"):
        run_v4_walk_forward(
            synthetic_v4_frame(),
            app_config=load_config(),
            v4_config=load_v4_config(),
            start="2026-09-01",
            end="2026-09-30",
        )


def test_deflated_sharpe_penalizes_multiple_testing() -> None:
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0005, 0.01, 750))

    few_trials = deflated_sharpe_probability(returns, trials=2)
    many_trials = deflated_sharpe_probability(returns, trials=1_660)

    assert many_trials["expected_maximum_null_sharpe"] > few_trials[
        "expected_maximum_null_sharpe"
    ]
    assert many_trials["deflated_sharpe_probability"] < few_trials[
        "deflated_sharpe_probability"
    ]
