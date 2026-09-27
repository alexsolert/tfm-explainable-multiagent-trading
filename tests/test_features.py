import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from qqq_agents.features.technical import FEATURE_COLUMNS, build_features, sample_decisions


def synthetic_market(rows: int = 320) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.3, rows)), index=index)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 1_300_000, rows),
        },
        index=index,
    )


def test_feature_values_do_not_change_when_only_future_prices_change() -> None:
    original = synthetic_market()
    changed = original.copy()
    cutoff = original.index[250]
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 1.5

    original_features = build_features(original).loc[:cutoff, FEATURE_COLUMNS]
    changed_features = build_features(changed).loc[:cutoff, FEATURE_COLUMNS]

    assert_frame_equal(original_features, changed_features)


def test_incomplete_future_targets_remain_missing() -> None:
    features = build_features(synthetic_market(), prediction_horizon=5)

    assert features["target_up"].tail(5).isna().all()
    assert features["target_risk"].tail(5).isna().all()
    assert features["target_end_date"].tail(5).isna().all()


def test_weekly_sampling_keeps_real_market_dates() -> None:
    features = build_features(synthetic_market())
    sampled = sample_decisions(features)

    assert sampled.index.isin(features.index).all()
    assert sampled.index.is_monotonic_increasing
