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
from qqq_agents.evaluation import run_quantitative_walk_forward
from qqq_agents.explainability import generate_lime_cases
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


def _walk_forward(config_path: Path, *, include_shap: bool, personality: str) -> None:
    config = load_config(config_path)
    frame = pd.read_csv(config.data.processed_path, index_col="date", parse_dates=["date"])
    result = run_quantitative_walk_forward(
        frame,
        config=config,
        validation_start=config.experiment.validation_start,
        validation_end="2022-12-31",
        include_shap=include_shap,
        personality_name=personality,
    )
    output_dir = Path("artifacts/walk_forward")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(output_dir / "validation_decisions.csv", index=True)
    result.strategy.history.to_csv(output_dir / "validation_strategy.csv", index=True)
    result.training_audit.to_csv(output_dir / "training_audit.csv", index=False)
    metrics = {
        "period": {"start": config.experiment.validation_start, "end": "2022-12-31"},
        "personality": personality,
        "test_period_consulted": False,
        "quantitative_multiagent": result.strategy.metrics,
        "baselines": {name: value.metrics for name, value in result.baselines.items()},
        "decision_counts": result.decisions["action"].value_counts().to_dict(),
        "risk_veto_count": int(result.decisions["risk_veto_triggered"].sum()),
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved walk-forward validation artifacts to {output_dir}")


def _lime_cases(config_path: Path, *, num_samples: int) -> None:
    config = load_config(config_path)
    frame = pd.read_csv(config.data.processed_path, index_col="date", parse_dates=["date"])
    decisions_path = Path("artifacts/walk_forward/validation_decisions.csv")
    if not decisions_path.exists():
        raise FileNotFoundError("Run 'qqq-agents walk-forward --with-shap' before LIME cases")
    decisions = pd.read_csv(decisions_path, index_col="date", parse_dates=["date"])
    payload = generate_lime_cases(
        frame,
        decisions,
        config=config,
        num_samples=num_samples,
    )
    output_path = Path("artifacts/explainability/lime_cases.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["selection"], indent=2, sort_keys=True))
    print(f"Saved {len(payload['cases'])} LIME explanations to {output_path}")


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
    walk_forward_parser = subparsers.add_parser(
        "walk-forward", help="Run expanding yearly validation without opening the final test."
    )
    walk_forward_parser.add_argument(
        "--with-shap",
        action="store_true",
        help="Persist the five strongest TreeSHAP factors for every agent decision.",
    )
    walk_forward_parser.add_argument(
        "--personality",
        choices=("conservative", "aggressive", "opportunistic"),
        default="conservative",
        help="Personality profile applied across agents and coordinator.",
    )
    lime_parser = subparsers.add_parser(
        "lime-cases", help="Explain representative validation decisions with LIME."
    )
    lime_parser.add_argument(
        "--num-samples",
        type=int,
        default=1_000,
        help="Perturbations generated by LIME for each local explanation.",
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
    elif args.command == "walk-forward":
        _walk_forward(
            args.config,
            include_shap=args.with_shap,
            personality=args.personality,
        )
    elif args.command == "lime-cases":
        _lime_cases(args.config, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
