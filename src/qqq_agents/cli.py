"""Comandos reproducibles para preparar los primeros artefactos."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd

from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import run_backtest
from qqq_agents.config import load_config
from qqq_agents.data import download_market_data, load_market_data
from qqq_agents.evaluation import run_quantitative_walk_forward
from qqq_agents.explainability import generate_lime_cases
from qqq_agents.features import build_features, sample_decisions
from qqq_agents.llm import (
    AutoGenOpenAIClient,
    CachedLLMClient,
    MockLLMClient,
    execute_pilot,
    load_pilot_case,
    save_pilot_result,
)
from qqq_agents.llm.pilot import cached_spend, conservative_call_cost_bound
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
    equity = pd.DataFrame({"multiagent": result.strategy.history["equity"]})
    for name, baseline in result.baselines.items():
        equity[name] = baseline.history["equity"]
    equity.to_csv(output_dir / "validation_equity.csv", index=True)
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


async def _llm_case(config_path: Path, *, requested_date: str | None, real: bool) -> None:
    config = load_config(config_path)
    final_development_date = str(
        (pd.Timestamp(config.experiment.test_start) - pd.Timedelta(days=1)).date()
    )
    case = load_pilot_case(
        features_path=config.data.processed_path,
        decisions_path="artifacts/walk_forward/validation_decisions.csv",
        requested_date=requested_date,
        latest_allowed_date=final_development_date,
    )
    raw_client: AutoGenOpenAIClient | MockLLMClient
    cache_name: str
    if real:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add the key locally."
            )
        cache_name = "openai"
        cache_dir = Path("artifacts/llm/cache") / cache_name
        prior_spend = cached_spend(
            cache_dir,
            input_price=config.llm.input_price_per_million,
            output_price=config.llm.output_price_per_million,
        )
        call_bound = conservative_call_cost_bound(config, case.packet)
        if prior_spend + call_bound > config.llm.budget_usd:
            raise SystemExit(
                f"LLM budget guard blocked the pilot: ${prior_spend:.4f} recorded and "
                f"up to ${call_bound:.4f} reserved against a ${config.llm.budget_usd:.2f} cap."
            )
        raw_client = AutoGenOpenAIClient(
            model=config.llm.model,
            api_key=api_key,
            reasoning_effort=config.llm.reasoning_effort,
            max_output_tokens=config.llm.max_output_tokens,
        )
        mode = "openai_pilot"
    else:
        cache_name = "mock"
        cache_dir = Path("artifacts/llm/cache") / cache_name
        raw_client = MockLLMClient()
        mode = "deterministic_dry_run"

    client = CachedLLMClient(raw_client, cache_dir) if config.llm.cache else raw_client
    try:
        payload = await execute_pilot(case=case, config=config, client=client, mode=mode)
    finally:
        if isinstance(raw_client, AutoGenOpenAIClient):
            await raw_client.close()

    filename = "pilot" if real else "dry_run"
    output_path = Path("artifacts/llm") / f"{filename}_{case.packet.as_of}.json"
    save_pilot_result(payload, output_path)
    decision = payload["combined_decision"]
    assert isinstance(decision, dict)
    print(
        json.dumps(
            {
                "as_of": str(case.packet.as_of),
                "mode": mode,
                "quantitative_action": case.packet.proposed_action,
                "combined_action": decision["action"],
                "incremental_estimated_cost_usd": payload["incremental_estimated_cost_usd"],
                "test_period_consulted": False,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Saved LLM pilot trace to {output_path}")


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
    llm_dry_parser = subparsers.add_parser(
        "llm-dry-run", help="Validate the complete LLM route with a free deterministic provider."
    )
    llm_dry_parser.add_argument(
        "--date",
        help="Validation decision date; defaults to the latest allowed development date.",
    )
    llm_pilot_parser = subparsers.add_parser(
        "llm-pilot", help="Run three bounded OpenAI calls for one validation decision."
    )
    llm_pilot_parser.add_argument(
        "--date",
        help="Validation decision date; defaults to the latest allowed development date.",
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
    elif args.command == "llm-dry-run":
        asyncio.run(_llm_case(args.config, requested_date=args.date, real=False))
    elif args.command == "llm-pilot":
        asyncio.run(_llm_case(args.config, requested_date=args.date, real=True))


if __name__ == "__main__":
    main()
