from datetime import date

import numpy as np
import pandas as pd
import pytest

from qqq_agents.config import load_config
from qqq_agents.data.context import build_context_features
from qqq_agents.news import HistoricalNewsStore
from qqq_agents.v2 import load_v2_config, run_v2_walk_forward
from qqq_agents.v2.allocation import TacticalAllocator


def test_v2_configuration_separates_specialist_features() -> None:
    config = load_v2_config()
    groups = (
        config.features.technical,
        config.features.momentum,
        config.features.risk,
        config.features.regime,
    )
    flattened = [name for group in groups for name in group]

    assert len(flattened) == len(set(flattened))
    assert config.development.protected_test_start == "2025-01-01"


def test_tactical_allocator_uses_three_exposure_levels() -> None:
    allocator = TacticalAllocator(load_v2_config().allocation)

    assert allocator.allocate(direction_probability=0.6, risk_probability=0.1).exposure == 1.0
    assert allocator.allocate(direction_probability=0.3, risk_probability=0.1).exposure == 0.5
    assert allocator.allocate(direction_probability=0.6, risk_probability=0.25).exposure == 0.5
    assert allocator.allocate(direction_probability=0.6, risk_probability=0.4).exposure == 0.0


def synthetic_v2_frame() -> pd.DataFrame:
    index = pd.date_range("2015-01-02", "2021-12-31", freq="W-FRI")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.002, 0.025, len(index))
    frame = pd.DataFrame(index=index)
    frame["close"] = 100 * np.exp(np.cumsum(returns))
    frame["target_end_date"] = frame.index + pd.Timedelta(days=7)
    frame["target_up"] = (np.roll(returns, -1) > 0).astype(int)
    frame["target_risk"] = (np.roll(returns, -1) < -0.025).astype(int)
    frame.loc[frame.index[-1], ["target_up", "target_risk"]] = [0, 0]
    names = {
        "distance_sma_50",
        "distance_sma_200",
        "ema_spread_12_26",
        "rsi_14",
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "volatility_20",
        "atr_14_pct",
        "drawdown_252",
    }
    for offset, name in enumerate(sorted(names), start=1):
        frame[name] = rng.normal(offset / 100, 0.1, len(index))
    return frame


def test_v2_walk_forward_is_leakage_safe_and_gradual() -> None:
    config = load_v2_config()
    fast_models = config.models.model_copy(
        update={"candidates": ("logistic",), "calibration_splits": 2}
    )
    config = config.model_copy(update={"models": fast_models})
    result = run_v2_walk_forward(
        synthetic_v2_frame(),
        app_config=load_config(),
        v2_config=config,
        start="2020-01-01",
        end="2021-12-31",
    )

    assert set(result.decisions["desired_position"]).issubset({0.0, 0.5, 1.0})
    assert result.decisions["operation_executed"].sum() <= len(result.decisions)
    assert (
        result.training_audit["maximum_target_end_date"] <= result.training_audit["cutoff"]
    ).all()
    assert {"structural_long", "risk_only", "directional_only"} <= set(result.ablations)


def test_v2_development_cannot_open_protected_period() -> None:
    with pytest.raises(ValueError, match="protected V2 test"):
        run_v2_walk_forward(
            synthetic_v2_frame(),
            app_config=load_config(),
            v2_config=load_v2_config(),
            start="2020-01-01",
            end="2025-01-03",
        )


def test_context_features_are_backward_looking(monkeypatch, tmp_path) -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="B")
    market = pd.DataFrame(
        {
            "open": np.arange(100, 180),
            "high": np.arange(101, 181),
            "low": np.arange(99, 179),
            "close": np.arange(100, 180),
            "volume": 1_000,
        },
        index=index,
    )
    external = market.copy()

    def fake_load(_path):
        return external

    monkeypatch.setattr("qqq_agents.data.context.load_market_data", fake_load)
    (tmp_path / "spy.csv").touch()
    (tmp_path / "vix.csv").touch()
    features = build_context_features(
        market,
        tickers={"spy": "SPY", "vix": "^VIX"},
        raw_directory=tmp_path,
    )

    # Missing files are deliberately ignored; once snapshots exist, no future shifts are used.
    assert features.index.equals(market.index)
    assert {"spy_momentum_20", "vix_level", "vix_change_20"} <= set(features.columns)
    assert not any("future" in column for column in features.columns)


def test_historical_news_store_never_returns_future_evidence() -> None:
    frame = pd.DataFrame(
        {
            "evidence_id": ["past", "future"],
            "published_at": ["2024-01-04T12:00:00Z", "2024-01-06T12:00:00Z"],
            "text": ["Past headline", "Future headline"],
            "source": ["test", "test"],
            "ticker": ["QQQ", "QQQ"],
        }
    )
    headlines = HistoricalNewsStore(frame).as_of(date(2024, 1, 5))

    assert [headline.evidence_id for headline in headlines] == ["past"]
