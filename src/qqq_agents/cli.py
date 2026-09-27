"""Comandos reproducibles para preparar los primeros artefactos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import run_backtest
from qqq_agents.config import load_config
from qqq_agents.data import download_market_data, load_market_data
from qqq_agents.features import build_features, sample_decisions
from qqq_agents.training import train_quantitative_agents


def _download(config_path: Path) -> None:
    config = load_config(config_path)
    frame = download_market_data(
        ticker=config.data.ticker,
        start=config.data.start,
        end=config.data.end,
        destination=config.data.raw_path,
        auto_adjust=config.data.auto_adjust,
    )
    print(f"Saved {len(frame)} daily observations to {config.data.raw_path}")


def _prepare(config_path: Path) -> None:
    config = load_config(config_path)
    market = load_market_data(config.data.raw_path)
    features = build_features(
        market,
        prediction_horizon=config.experiment.prediction_horizon_sessions,
        risk_event_threshold=config.experiment.risk_event_threshold,
    )
    decisions = sample_decisions(features, config.experiment.decision_frequency)
    config.data.processed_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(config.data.processed_path, index=True)
    print(f"Saved {len(decisions)} weekly observations to {config.data.processed_path}")


def _baselines(config_path: Path, through: str) -> None:
    """Evaluate deterministic baselines only on the development period."""

    config = load_config(config_path)
    frame = pd.read_csv(config.data.processed_path, index_col="date", parse_dates=["date"])
    frame = frame.loc[:through].copy()
    if frame.empty:
        raise ValueError(f"No processed observations available through {through}")

    close = frame["close"]
    sma_50 = close / (1 + frame["distance_sma_50"])
    sma_200 = close / (1 + frame["distance_sma_200"])
    positions = {
        "buy_and_hold": buy_and_hold(close),
        "sma_50_200": (sma_50 > sma_200).fillna(False).astype(float),
    }
    reports: dict[str, dict[str, float]] = {}
    history = pd.DataFrame(index=frame.index)
    for name, position in positions.items():
        result = run_backtest(
            close,
            position,
            transaction_cost_bps=config.experiment.transaction_cost_bps,
        )
        reports[name] = result.metrics
        history[f"{name}_equity"] = result.history["equity"]

    artifact_dir = Path("artifacts/baselines")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "development_metrics.json"
    history_path = artifact_dir / "development_equity.csv"
    payload = {
        "period": {"start": str(frame.index.min().date()), "end": str(frame.index.max().date())},
        "test_period_consulted": False,
        "metrics": reports,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    history.to_csv(history_path, index=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved development baseline artifacts to {artifact_dir}")


def _train_quant(config_path: Path, through: str) -> None:
    config = load_config(config_path)
    frame = pd.read_csv(config.data.processed_path, index_col="date", parse_dates=["date"])
    manifest = train_quantitative_agents(frame, config=config, cutoff=through)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to the experiment YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="Download and normalize QQQ market data.")
    subparsers.add_parser("prepare", help="Build weekly features and future labels.")
    baseline_parser = subparsers.add_parser(
        "baselines", help="Evaluate baselines without opening the final test period."
    )
    baseline_parser.add_argument(
        "--through",
        default="2022-12-31",
        help="Inclusive final date; defaults to the end of the development period.",
    )
    training_parser = subparsers.add_parser(
        "train-quant", help="Fit quantitative agents on leakage-safe development observations."
    )
    training_parser.add_argument(
        "--through",
        default="2022-12-31",
        help="Latest allowed feature and target date.",
    )
    args = parser.parse_args()

    if args.command == "download":
        _download(args.config)
    elif args.command == "prepare":
        _prepare(args.config)
    elif args.command == "baselines":
        _baselines(args.config, args.through)
    elif args.command == "train-quant":
        _train_quant(args.config, args.through)


if __name__ == "__main__":
    main()
