"""Diagnosticos probabilisticos y financieros del protocolo V2."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def probability_diagnostics(
    observed: pd.Series,
    probability: pd.Series,
    *,
    bins: int = 5,
) -> dict[str, object]:
    aligned_observed, aligned_probability = observed.align(probability, join="inner")
    valid = aligned_observed.notna() & aligned_probability.notna()
    y = aligned_observed.loc[valid].astype(int)
    p = aligned_probability.loc[valid].astype(float).clip(0, 1)
    if y.empty or y.nunique() < 2:
        raise ValueError("Probability diagnostics require both observed classes")
    prediction = p >= 0.5
    edges = np.linspace(0, 1, bins + 1)
    assignments = pd.cut(p, bins=edges, include_lowest=True, duplicates="drop")
    calibration: list[dict[str, float | int]] = []
    ece = 0.0
    for _, group in pd.DataFrame({"observed": y, "probability": p, "bin": assignments}).groupby(
        "bin", observed=False
    ):
        if group.empty:
            continue
        predicted_mean = float(group["probability"].mean())
        observed_rate = float(group["observed"].mean())
        count = int(len(group))
        ece += count / len(y) * abs(predicted_mean - observed_rate)
        calibration.append(
            {
                "predicted_probability": predicted_mean,
                "observed_frequency": observed_rate,
                "count": count,
            }
        )
    return {
        "observations": int(len(y)),
        "positive_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": float(ece),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "calibration": calibration,
    }


def circular_block_bootstrap_difference(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    samples: int,
    block_length: int,
    random_seed: int,
    periods_per_year: int = 52,
) -> dict[str, float]:
    strategy, benchmark = strategy_returns.align(benchmark_returns, join="inner")
    values = np.column_stack([strategy.fillna(0).to_numpy(), benchmark.fillna(0).to_numpy()])
    if len(values) < block_length:
        raise ValueError("Bootstrap block exceeds the available return history")
    rng = np.random.default_rng(random_seed)
    blocks_needed = math.ceil(len(values) / block_length)
    differences = np.empty(samples, dtype=float)
    for sample in range(samples):
        starts = rng.integers(0, len(values), size=blocks_needed)
        indices = np.concatenate(
            [(np.arange(start, start + block_length) % len(values)) for start in starts]
        )[: len(values)]
        selected = values[indices]
        annualized = (1 + selected).prod(axis=0) ** (periods_per_year / len(values)) - 1
        differences[sample] = annualized[0] - annualized[1]
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "probability_strategy_outperforms": float((differences > 0).mean()),
    }


def probability_of_backtest_overfitting(
    returns: pd.DataFrame,
    *,
    partitions: int = 8,
    periods_per_year: int = 52,
) -> dict[str, object]:
    """CSCV estimate of how often the in-sample winner falls below median out of sample."""

    clean = returns.dropna().astype(float)
    if partitions % 2 or partitions < 4:
        raise ValueError("CSCV requires an even number of at least four partitions")
    if len(clean) < partitions * 2 or clean.shape[1] < 2:
        raise ValueError("CSCV requires multiple strategies and enough observations")
    blocks = [
        np.asarray(block, dtype=int)
        for block in np.array_split(np.arange(len(clean)), partitions)
    ]
    below_median: list[bool] = []
    selected_counts = {name: 0 for name in clean.columns}

    def sharpe(values: pd.DataFrame) -> pd.Series:
        standard_deviation = values.std(ddof=1).replace(0, np.nan)
        return values.mean() / standard_deviation * math.sqrt(periods_per_year)

    for selected_blocks in combinations(range(partitions), partitions // 2):
        train_index = np.concatenate([blocks[index] for index in selected_blocks])
        test_blocks = [index for index in range(partitions) if index not in selected_blocks]
        test_index = np.concatenate([blocks[index] for index in test_blocks])
        train_sharpe = sharpe(clean.iloc[train_index])
        test_sharpe = sharpe(clean.iloc[test_index])
        if train_sharpe.isna().all() or test_sharpe.isna().all():
            continue
        winner = str(train_sharpe.idxmax())
        selected_counts[winner] += 1
        percentile = float(test_sharpe.rank(pct=True, method="average").loc[winner])
        below_median.append(percentile <= 0.5)
    if not below_median:
        raise ValueError("CSCV produced no valid train/test combinations")
    return {
        "partitions": partitions,
        "combinations": len(below_median),
        "probability_of_backtest_overfitting": float(np.mean(below_median)),
        "in_sample_winner_counts": selected_counts,
    }
