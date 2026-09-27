import numpy as np
import pandas as pd

from qqq_agents.config import load_config
from qqq_agents.evaluation import run_quantitative_walk_forward


def processed_weekly_frame() -> pd.DataFrame:
    index = pd.date_range("2017-01-06", "2021-12-31", freq="W-FRI")
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(index=index)
    frame["close"] = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.025, len(index))))
    frame["target_end_date"] = frame.index + pd.Timedelta(days=7)
    frame["target_up"] = (np.arange(len(index)) % 2).astype(int)
    frame["target_risk"] = (np.arange(len(index)) % 7 == 0).astype(int)
    feature_names = {
        "return_1d",
        "log_return_1d",
        "distance_sma_20",
        "distance_sma_50",
        "distance_sma_200",
        "ema_spread_12_26",
        "macd",
        "macd_signal",
        "macd_histogram",
        "rsi_14",
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "volatility_20",
        "atr_14_pct",
        "drawdown_252",
        "volume_zscore_20",
    }
    for offset, name in enumerate(sorted(feature_names), start=1):
        frame[name] = rng.normal(offset / 100, 0.1, len(index))
    return frame


def test_walk_forward_never_trains_beyond_each_cutoff() -> None:
    result = run_quantitative_walk_forward(
        processed_weekly_frame(),
        config=load_config(),
        validation_start="2020-01-01",
        validation_end="2021-12-31",
    )

    assert not result.decisions.empty
    assert (
        result.training_audit["maximum_target_end_date"] <= result.training_audit["cutoff"]
    ).all()
    assert result.decisions.index.min() >= pd.Timestamp("2020-01-01")
    assert result.decisions.index.max() <= pd.Timestamp("2021-12-31")
    assert result.decisions["desired_position"].between(0, 1).all()
    assert "single_logistic_agent" in result.baselines
