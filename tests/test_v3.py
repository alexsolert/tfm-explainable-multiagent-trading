import numpy as np
import pandas as pd
import pytest

from qqq_agents.backtesting.engine import run_backtest
from qqq_agents.config import load_config
from qqq_agents.v3 import (
    ContinuousAllocator,
    build_cross_asset_panel,
    cash_returns_from_yield,
    load_v3_config,
    run_v3_walk_forward,
)
from qqq_agents.v3.features import PANEL_FEATURES


def _market(seed: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(index))))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000, 2_000, len(index)),
        },
        index=index,
    )


def test_backtest_credits_cash_to_uninvested_fraction() -> None:
    index = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    close = pd.Series([100.0, 100.0, 100.0], index=index)
    cash = pd.Series([0.01, 0.01, 0.01], index=index)
    result = run_backtest(
        close,
        pd.Series(0.0, index=index),
        transaction_cost_bps=0,
        cash_return=cash,
    )

    assert result.history["strategy_return"].tolist() == [0.01, 0.01, 0.01]
    assert result.history["cash_contribution"].tolist() == [0.01, 0.01, 0.01]


def test_continuous_allocator_limits_turnover_and_downside() -> None:
    allocator = ContinuousAllocator(load_v3_config().allocation)
    cautious = allocator.allocate(
        expected_return=-0.01,
        downside_quantile=-0.08,
        risk_probability=0.4,
        cash_return=0.001,
        previous_exposure=0.9,
    )
    stable = allocator.allocate(
        expected_return=0.001,
        downside_quantile=-0.02,
        risk_probability=0.1,
        cash_return=0.001,
        previous_exposure=0.9,
    )

    assert cautious.exposure < 0.9
    assert cautious.exposure >= 0.65
    assert stable.exposure == 0.9


def test_cross_asset_panel_contains_only_contemporaneous_features() -> None:
    index = pd.date_range("2017-01-02", "2021-12-31", freq="B")
    markets = {
        name: _market(seed, index)
        for seed, name in enumerate(("qqq", "spy", "iwm", "smh", "tlt", "vix"), start=1)
    }
    panel = build_cross_asset_panel(markets)

    assert panel.index.names == ["date", "asset"]
    assert set(panel.index.get_level_values("asset")) == {"qqq", "spy", "iwm", "smh", "tlt"}
    assert set(PANEL_FEATURES) <= set(panel.columns)
    valid_target = panel["target_end_date"].notna()
    target_dates = pd.to_datetime(panel.loc[valid_target, "target_end_date"])
    feature_dates = panel.index.get_level_values("date")[valid_target]
    assert (target_dates.to_numpy() > feature_dates.to_numpy()).all()


def test_cash_conversion_is_non_negative_and_weekly() -> None:
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    annual_yield = pd.Series(5.0, index=dates)
    returns = cash_returns_from_yield(annual_yield, dates)

    assert returns.iloc[0] == 0.0
    assert (returns.iloc[1:] > 0).all()
    assert returns.iloc[1] == pytest.approx((1.05 ** (7 / 365)) - 1)
    assert returns.index.equals(dates)


def _synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2015-01-02", "2021-12-31", freq="W-FRI")
    records = []
    rng = np.random.default_rng(9)
    for asset_number, asset in enumerate(("qqq", "spy", "iwm", "smh", "tlt")):
        returns = rng.normal(0.002, 0.025, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        for position, timestamp in enumerate(dates):
            record = {
                "date": timestamp,
                "asset": asset,
                "close": close[position],
                "forward_return": returns[position + 1] if position + 1 < len(dates) else np.nan,
                "target_risk": (
                    int(returns[position + 1] <= -0.025)
                    if position + 1 < len(dates)
                    else np.nan
                ),
                "target_end_date": (
                    dates[position + 1] if position + 1 < len(dates) else pd.NaT
                ),
            }
            for feature_number, name in enumerate(PANEL_FEATURES, start=1):
                record[name] = (
                    float(name == f"asset_{asset}")
                    if name.startswith("asset_")
                    else rng.normal(asset_number / 100, 0.1) + feature_number / 1000
                )
            records.append(record)
    return pd.DataFrame(records).set_index(["date", "asset"]).sort_index()


def test_v3_walk_forward_is_panel_based_and_leakage_safe() -> None:
    config = load_v3_config()
    fast_models = config.models.model_copy(
        update={"expected_return_candidates": ("ridge",), "validation_splits": 2}
    )
    config = config.model_copy(update={"models": fast_models})
    panel = _synthetic_panel()
    qqq_dates = panel.xs("qqq", level="asset").index
    cash = pd.Series(0.0005, index=qqq_dates)
    result = run_v3_walk_forward(
        panel,
        cash,
        app_config=load_config(),
        v3_config=config,
        start="2020-01-01",
        end="2021-12-31",
    )

    assert result.decisions["desired_position"].between(0, 1).all()
    assert (result.training_audit["training_assets"] == 5).all()
    assert (
        result.training_audit["maximum_target_end_date"] <= result.training_audit["cutoff"]
    ).all()
    assert "volatility_target" in result.baselines


def test_v3_development_cannot_open_prospective_period() -> None:
    with pytest.raises(ValueError, match="prospective period"):
        run_v3_walk_forward(
            _synthetic_panel(),
            pd.Series(dtype=float),
            app_config=load_config(),
            v3_config=load_v3_config(),
            start="2026-09-01",
            end="2026-09-30",
        )
