"""Robustness diagnostics for the daily V4 research track."""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd


def deflated_sharpe_probability(
    returns: pd.Series,
    *,
    trials: int,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Conservative DSR-style probability after selection and non-normality.

    The expected maximum Sharpe assumes ``trials`` independent null strategies. This is
    deliberately conservative because many V4 configurations are correlated variants.
    """

    clean = returns.dropna().astype(float)
    if len(clean) < 30:
        raise ValueError("Deflated Sharpe requires at least 30 return observations")
    if trials < 2:
        raise ValueError("Deflated Sharpe requires at least two trials")
    standard_deviation = float(clean.std(ddof=1))
    if standard_deviation <= 0:
        raise ValueError("Deflated Sharpe requires non-zero return variance")

    observations = len(clean)
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    expected_max_standard_normal = (
        (1 - euler_gamma) * normal.inv_cdf(1 - 1 / trials)
        + euler_gamma * normal.inv_cdf(1 - 1 / (trials * math.e))
    )
    observed_period_sharpe = float(clean.mean() / standard_deviation)
    expected_max_period_sharpe = expected_max_standard_normal / math.sqrt(observations)
    skewness = float(clean.skew())
    kurtosis = float(clean.kurt() + 3)
    variance_adjustment = (
        1
        - skewness * observed_period_sharpe
        + ((kurtosis - 1) / 4) * observed_period_sharpe**2
    )
    z_score = (
        (observed_period_sharpe - expected_max_period_sharpe)
        * math.sqrt(observations - 1)
        / math.sqrt(max(variance_adjustment, 1e-12))
    )
    return {
        "observations": observations,
        "trials": trials,
        "observed_sharpe": observed_period_sharpe * math.sqrt(periods_per_year),
        "expected_maximum_null_sharpe": expected_max_period_sharpe
        * math.sqrt(periods_per_year),
        "deflated_sharpe_probability": float(normal.cdf(z_score)),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def return_family(backtests: dict[str, object]) -> pd.DataFrame:
    """Align strategy-return series from a named family of BacktestResult objects."""

    series = {
        name: result.history["strategy_return"].rename(name)
        for name, result in backtests.items()
    }
    return pd.concat(series.values(), axis=1).dropna()
