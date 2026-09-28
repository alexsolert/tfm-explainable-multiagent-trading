"""Comandos reproducibles para preparar los primeros artefactos."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from qqq_agents.backtesting.baselines import buy_and_hold
from qqq_agents.backtesting.engine import run_backtest
from qqq_agents.backtesting.metrics import calculate_metrics
from qqq_agents.config import load_config
from qqq_agents.data import (
    build_context_features,
    download_context_bundle,
    download_market_data,
    download_yield_data,
    load_market_data,
    load_yield_data,
)
from qqq_agents.evaluation import run_quantitative_walk_forward
from qqq_agents.explainability import generate_lime_cases
from qqq_agents.features import build_features, rebuild_decision_interval_labels, sample_decisions
from qqq_agents.llm import (
    AutoGenOpenAIClient,
    CachedLLMClient,
    EvidenceGatedLLMClient,
    EvidenceValidatedLLMClient,
    MockLLMClient,
    execute_pilot,
    load_pilot_case,
    run_hybrid_period,
    save_hybrid_period,
    save_pilot_result,
)
from qqq_agents.llm.pilot import (
    cached_spend,
    conservative_call_cost_bound,
    load_pilot_cases,
)
from qqq_agents.training import train_quantitative_agents
from qqq_agents.v2 import load_v2_config, run_v2_walk_forward
from qqq_agents.v2.diagnostics import (
    circular_block_bootstrap_difference,
    probability_of_backtest_overfitting,
)
from qqq_agents.v3 import (
    build_cross_asset_panel,
    cash_returns_from_yield,
    load_v3_config,
    run_v3_walk_forward,
)
from qqq_agents.v4 import (
    build_daily_research_frame,
    deflated_sharpe_probability,
    load_v4_config,
    return_family,
    run_v4_walk_forward,
)
from qqq_agents.v5 import build_v5_research_frame, load_v5_config, run_v5_walk_forward


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


def _v2_download_context(v2_config_path: Path) -> None:
    config = load_v2_config(v2_config_path)
    if pd.Timestamp(config.context.end) > pd.Timestamp(config.development.protected_test_start):
        raise ValueError("Context download would open the protected V2 test period")
    observations = download_context_bundle(
        tickers=config.context.tickers,
        start=config.context.start,
        end=config.context.end,
        destination=config.context.raw_directory,
    )
    print(json.dumps({"protected_test_consulted": False, "observations": observations}, indent=2))


def _v2_prepare(config_path: Path, v2_config_path: Path) -> None:
    app_config = load_config(config_path)
    v2_config = load_v2_config(v2_config_path)
    market = load_market_data(app_config.data.raw_path)
    if market.index.max() >= pd.Timestamp(v2_config.development.protected_test_start):
        raise ValueError("Primary market snapshot includes the protected V2 test period")
    features = build_features(
        market,
        prediction_horizon=app_config.experiment.prediction_horizon_sessions,
        risk_event_threshold=app_config.experiment.risk_event_threshold,
    )
    context = build_context_features(
        market,
        tickers=v2_config.context.tickers,
        raw_directory=v2_config.context.raw_directory,
    )
    weekly = sample_decisions(
        features.join(context),
        app_config.experiment.decision_frequency,
    )
    weekly = rebuild_decision_interval_labels(
        weekly,
        market["close"],
        risk_event_threshold=app_config.experiment.risk_event_threshold,
    )
    v2_config.context.processed_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(v2_config.context.processed_path, index=True)
    available = [name for name in v2_config.features.regime if name in weekly.columns]
    print(
        json.dumps(
            {
                "rows": len(weekly),
                "destination": str(v2_config.context.processed_path),
                "regime_features": available,
                "protected_test_consulted": False,
            },
            indent=2,
        )
    )


def _config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


V2_FROZEN_PATHS = (
    Path("configs/v2.yaml"),
    Path("src/qqq_agents/v2/config.py"),
    Path("src/qqq_agents/v2/models.py"),
    Path("src/qqq_agents/v2/allocation.py"),
    Path("src/qqq_agents/v2/diagnostics.py"),
    Path("src/qqq_agents/v2/evaluation.py"),
    Path("src/qqq_agents/data/context.py"),
    Path("src/qqq_agents/features/technical.py"),
    Path("docs/experiments/v2-protocol.md"),
)

V5_FROZEN_PATHS = (
    Path("configs/v5.yaml"),
    Path("src/qqq_agents/v5/config.py"),
    Path("src/qqq_agents/v5/features.py"),
    Path("src/qqq_agents/v5/models.py"),
    Path("src/qqq_agents/v5/policies.py"),
    Path("src/qqq_agents/v5/evaluation.py"),
    Path("docs/experiments/v5-multifrequency-protocol.md"),
)


def _v2_file_digests() -> dict[str, str]:
    return {str(path): _config_digest(path) for path in V2_FROZEN_PATHS}


def _v5_file_digests() -> dict[str, str]:
    return {str(path): _config_digest(path) for path in V5_FROZEN_PATHS}


def _save_v2_result(
    result,
    *,
    app_config,
    v2_config,
    output_dir: Path,
    start: str,
    end: str,
    protected_test_consulted: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(output_dir / "decisions.csv", index=True)
    result.strategy.history.to_csv(output_dir / "strategy.csv", index=True)
    result.training_audit.to_csv(output_dir / "training_audit.csv", index=False)
    result.model_leaderboard.to_csv(output_dir / "model_leaderboard.csv", index=False)
    equity = pd.DataFrame({"v2_multiagent": result.strategy.history["equity"]})
    for name, value in result.baselines.items():
        equity[name] = value.history["equity"]
    for name, value in result.ablations.items():
        equity[f"ablation_{name}"] = value.history["equity"]
    equity.to_csv(output_dir / "equity.csv", index=True)

    close = result.strategy.history["close"]
    cost_scenarios = {}
    for cost in v2_config.evaluation.transaction_cost_scenarios_bps:
        scenario = run_backtest(
            close,
            result.decisions["desired_position"],
            transaction_cost_bps=cost,
        )
        cost_scenarios[str(cost)] = scenario.metrics
    bootstrap = {
        name: circular_block_bootstrap_difference(
            result.strategy.history["strategy_return"],
            value.history["strategy_return"],
            samples=v2_config.evaluation.bootstrap_samples,
            block_length=v2_config.evaluation.bootstrap_block_weeks,
            random_seed=v2_config.evaluation.random_seed,
        )
        for name, value in result.baselines.items()
    }
    return_candidates = {"v2_multiagent": result.strategy.history["strategy_return"]}
    return_candidates.update(
        {name: value.history["strategy_return"] for name, value in result.baselines.items()}
    )
    return_candidates.update(
        {
            f"ablation_{name}": value.history["strategy_return"]
            for name, value in result.ablations.items()
        }
    )
    overfitting = probability_of_backtest_overfitting(pd.DataFrame(return_candidates))
    sma_metrics = result.baselines["sma_50_200"].metrics
    vs_sma = bootstrap["sma_50_200"]
    multiagent_eligible = (
        result.strategy.metrics["sharpe_ratio"] > sma_metrics["sharpe_ratio"]
        and vs_sma["probability_strategy_outperforms"] >= 0.90
    )
    evaluated_recommendation = {
        "champion": "v2_multiagent" if multiagent_eligible else "sma_50_200",
        "challenger": "sma_50_200" if multiagent_eligible else "v2_multiagent",
        "multiagent_promotion_eligible": multiagent_eligible,
        "rule": "Promote V2 only if its Sharpe exceeds SMA and bootstrap P(outperformance) >= 0.90",
    }
    deployment_recommendation = evaluated_recommendation
    if protected_test_consulted:
        development_path = Path("artifacts/v2_development/metrics.json")
        if not development_path.exists():
            raise FileNotFoundError("Protected reporting requires frozen development metrics")
        development_metrics = json.loads(development_path.read_text(encoding="utf-8"))
        deployment_recommendation = development_metrics["deployment_recommendation"]
    payload = {
        "version": v2_config.version,
        "period": {"start": start, "end": end},
        "protected_test_consulted": protected_test_consulted,
        "v2_multiagent": result.strategy.metrics,
        "baselines": {name: value.metrics for name, value in result.baselines.items()},
        "ablations": {name: value.metrics for name, value in result.ablations.items()},
        "cost_scenarios": cost_scenarios,
        "bootstrap_vs_baselines": bootstrap,
        "backtest_overfitting": overfitting,
        "deployment_recommendation": deployment_recommendation,
        "period_evaluated_recommendation": evaluated_recommendation,
        "probability_report": result.probability_report,
        "decision_counts": result.decisions["action"].value_counts().to_dict(),
        "operation_count": int(result.decisions["operation_executed"].sum()),
        "exposure_counts": {
            str(key): int(value)
            for key, value in result.decisions["desired_position"].value_counts().items()
        },
        "transaction_cost_bps": app_config.experiment.transaction_cost_bps,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved V2 artifacts to {output_dir}")


def _v2_evaluate(
    config_path: Path,
    v2_config_path: Path,
    *,
    protected: bool,
) -> None:
    app_config = load_config(config_path)
    v2_config = load_v2_config(v2_config_path)
    source = (
        Path("data/processed/v2_protected_features.csv")
        if protected
        else (
            v2_config.context.processed_path
            if v2_config.context.processed_path.exists()
            else app_config.data.processed_path
        )
    )
    if not source.exists():
        raise FileNotFoundError(f"Missing V2 data snapshot: {source}")
    frame = pd.read_csv(source, index_col="date", parse_dates=["date"])
    if protected:
        start = v2_config.development.protected_test_start
        end = v2_config.development.protected_test_end
        output = Path("artifacts/v2_protected_test")
    else:
        start = v2_config.development.start
        end = v2_config.development.end
        output = Path("artifacts/v2_development")
    result = run_v2_walk_forward(
        frame,
        app_config=app_config,
        v2_config=v2_config,
        start=start,
        end=end,
        allow_protected=protected,
    )
    _save_v2_result(
        result,
        app_config=app_config,
        v2_config=v2_config,
        output_dir=output,
        start=start,
        end=end,
        protected_test_consulted=protected,
    )


def _v3_download(v3_config_path: Path) -> None:
    config = load_v3_config(v3_config_path)
    config.data.raw_directory.mkdir(parents=True, exist_ok=True)
    observations: dict[str, int] = {}
    for alias, ticker in config.data.assets.items():
        frame = download_market_data(
            ticker=ticker,
            start=config.data.start,
            end=config.data.end,
            destination=config.data.raw_directory / f"{alias}.csv",
            auto_adjust=True,
        )
        observations[alias] = len(frame)
    cash = download_yield_data(
        ticker=config.data.cash_yield_ticker,
        start=config.data.start,
        end=config.data.end,
        destination=config.data.raw_directory / "cash_yield.csv",
    )
    observations["cash_yield"] = len(cash)
    print(json.dumps({"observations": observations}, indent=2, sort_keys=True))


def _v3_prepare(config_path: Path, v3_config_path: Path) -> None:
    app_config = load_config(config_path)
    config = load_v3_config(v3_config_path)
    markets = {
        alias: load_market_data(config.data.raw_directory / f"{alias}.csv")
        for alias in config.data.assets
    }
    panel = build_cross_asset_panel(
        markets,
        decision_frequency=app_config.experiment.decision_frequency,
        risk_event_threshold=app_config.experiment.risk_event_threshold,
    )
    config.data.panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.data.panel_path, index=True)
    qqq_dates = panel.xs("qqq", level="asset").index
    cash_yield = load_yield_data(config.data.raw_directory / "cash_yield.csv")
    cash = cash_returns_from_yield(cash_yield, qqq_dates)
    cash.to_csv(config.data.cash_path, index_label="date")
    print(
        json.dumps(
            {
                "panel_rows": len(panel),
                "panel_assets": panel.index.get_level_values("asset").nunique(),
                "panel_destination": str(config.data.panel_path),
                "cash_observations": len(cash),
                "cash_destination": str(config.data.cash_path),
            },
            indent=2,
        )
    )


def _v3_development(config_path: Path, v3_config_path: Path) -> None:
    app_config = load_config(config_path)
    config = load_v3_config(v3_config_path)
    panel = pd.read_csv(
        config.data.panel_path,
        index_col=["date", "asset"],
        parse_dates=["date", "target_end_date"],
    )
    cash = pd.read_csv(config.data.cash_path, index_col="date", parse_dates=["date"])[
        "cash_return"
    ]
    result = run_v3_walk_forward(
        panel,
        cash,
        app_config=app_config,
        v3_config=config,
    )
    output = Path("artifacts/v3_development")
    output.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(output / "decisions.csv", index=True)
    result.strategy.history.to_csv(output / "strategy.csv", index=True)
    result.training_audit.to_csv(output / "training_audit.csv", index=False)
    result.model_leaderboard.to_csv(output / "model_leaderboard.csv", index=False)
    equity = pd.DataFrame({"v3_multiagent": result.strategy.history["equity"]})
    for name, baseline in result.baselines.items():
        equity[name] = baseline.history["equity"]
    equity.to_csv(output / "equity.csv", index=True)

    def period_metrics(backtest, start: str, end: str) -> dict[str, float]:
        history = backtest.history.loc[start:end]
        return calculate_metrics(
            returns=history["strategy_return"],
            turnover=history["turnover"],
            asset_returns=history["asset_return"],
            positions=history["applied_position"],
        )

    periods = {
        "selection_2020_2024": ("2020-01-01", "2024-12-31"),
        "internal_validation_2025_2026": ("2025-01-01", config.research.development_end),
    }
    subperiods = {}
    for period_name, (period_start, period_end) in periods.items():
        subperiods[period_name] = {
            "period": {"start": period_start, "end": period_end},
            "v3_multiagent": period_metrics(result.strategy, period_start, period_end),
            "baselines": {
                name: period_metrics(value, period_start, period_end)
                for name, value in result.baselines.items()
            },
        }
    payload = {
        "version": config.version,
        "period": {
            "start": config.research.development_start,
            "end": config.research.development_end,
        },
        "prospective_period_consulted": False,
        "v3_multiagent": result.strategy.metrics,
        "baselines": {name: value.metrics for name, value in result.baselines.items()},
        "subperiods": subperiods,
        "cash_return_included": True,
        "training_assets": sorted(panel.index.get_level_values("asset").unique()),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved V3 development artifacts to {output}")


def _v4_download(v4_config_path: Path) -> None:
    config = load_v4_config(v4_config_path)
    config.data.raw_directory.mkdir(parents=True, exist_ok=True)
    observations: dict[str, int] = {}
    for alias, ticker in config.data.assets.items():
        frame = download_market_data(
            ticker=ticker,
            start=config.data.start,
            end=config.data.end,
            destination=config.data.raw_directory / f"{alias}.csv",
            auto_adjust=True,
        )
        observations[alias] = len(frame)
    cash = download_yield_data(
        ticker=config.data.cash_yield_ticker,
        start=config.data.start,
        end=config.data.end,
        destination=config.data.raw_directory / "cash_yield.csv",
    )
    observations["cash_yield"] = len(cash)
    print(json.dumps({"observations": observations}, indent=2, sort_keys=True))


def _v4_prepare(config_path: Path, v4_config_path: Path) -> None:
    app_config = load_config(config_path)
    config = load_v4_config(v4_config_path)
    markets = {
        alias: load_market_data(config.data.raw_directory / f"{alias}.csv")
        for alias in config.data.assets
    }
    cash_yield = load_yield_data(config.data.raw_directory / "cash_yield.csv")
    frame = build_daily_research_frame(
        markets,
        cash_yield,
        prediction_horizon=app_config.experiment.prediction_horizon_sessions,
        risk_event_threshold=config.data.risk_event_threshold,
    )
    config.data.processed_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.data.processed_path, index=True)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "start": str(frame.index.min().date()),
                "end": str(frame.index.max().date()),
                "destination": str(config.data.processed_path),
            },
            indent=2,
        )
    )


def _v4_development(config_path: Path, v4_config_path: Path) -> None:
    app_config = load_config(config_path)
    config = load_v4_config(v4_config_path)
    frame = pd.read_csv(
        config.data.processed_path,
        index_col="date",
        parse_dates=["date", "target_end_date", "vol_target_end_date"],
    )
    result = run_v4_walk_forward(frame, app_config=app_config, v4_config=config)
    output = Path("artifacts/v4_development")
    output.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(output / "decisions.csv", index=True)
    result.strategy.history.to_csv(output / "strategy.csv", index=True)
    result.training_audit.to_csv(output / "training_audit.csv", index=False)
    result.model_leaderboard.to_csv(output / "model_leaderboard.csv", index=False)
    equity = pd.DataFrame({"v4_multiagent": result.strategy.history["equity"]})
    for name, value in result.baselines.items():
        equity[name] = value.history["equity"]
    for name, value in result.ablations.items():
        equity[f"ablation_{name}"] = value.history["equity"]
    equity.to_csv(output / "equity.csv", index=True)

    def period_metrics(backtest, start: str, end: str) -> dict[str, float]:
        history = backtest.history.loc[start:end]
        return calculate_metrics(
            returns=history["strategy_return"],
            turnover=history["turnover"],
            asset_returns=history["asset_return"],
            positions=history["applied_position"],
            periods_per_year=252,
        )

    periods = {
        "selection": (config.period.evaluation_start, config.period.selection_end),
        "internal_validation": (
            config.period.validation_start,
            config.period.development_end,
        ),
    }
    subperiods = {}
    for name, (start, end) in periods.items():
        subperiods[name] = {
            "period": {"start": start, "end": end},
            "v4_multiagent": period_metrics(result.strategy, start, end),
            "baselines": {
                key: period_metrics(value, start, end)
                for key, value in result.baselines.items()
            },
            "ablations": {
                key: period_metrics(value, start, end)
                for key, value in result.ablations.items()
            },
        }
    validation_start = config.period.validation_start
    validation_end = config.period.development_end
    validation_strategy_returns = result.strategy.history.loc[
        validation_start:validation_end, "strategy_return"
    ]
    bootstrap = {
        name: circular_block_bootstrap_difference(
            validation_strategy_returns,
            value.history.loc[validation_start:validation_end, "strategy_return"],
            samples=5_000,
            block_length=20,
            random_seed=config.models.random_seed,
            periods_per_year=252,
        )
        for name, value in result.baselines.items()
    }
    candidate_family = {
        "v4_multiagent": result.strategy,
        **result.baselines,
        **{f"ablation_{name}": value for name, value in result.ablations.items()},
    }
    candidate_returns = return_family(candidate_family).loc[
        validation_start:validation_end
    ]
    robustness = {
        "validation_block_bootstrap": bootstrap,
        "deflated_sharpe": deflated_sharpe_probability(
            validation_strategy_returns,
            trials=1_660,
            periods_per_year=252,
        ),
        "candidate_family_cscv": probability_of_backtest_overfitting(
            candidate_returns,
            partitions=8,
            periods_per_year=252,
        ),
        "cscv_scope_note": (
            "CSCV covers the eight exported strategy/baseline/ablation return series, not all "
            "1,660 policy evaluations. The DSR trial count applies the broader selection penalty."
        ),
    }
    payload = {
        "version": config.version,
        "period": {
            "start": config.period.evaluation_start,
            "end": config.period.development_end,
        },
        "prospective_period_consulted": False,
        "decision_frequency": "daily",
        "v4_multiagent": result.strategy.metrics,
        "baselines": {name: value.metrics for name, value in result.baselines.items()},
        "ablations": {name: value.metrics for name, value in result.ablations.items()},
        "subperiods": subperiods,
        "configuration_evaluations": 1_660,
        "robustness": robustness,
        "research_scope": {
            "risk_targets": [-0.02, -0.03, -0.04, -0.05],
            "volatility_forecasts": ["HAR", "realized_20", "blend", "maximum"],
            "target_volatility_range": [0.18, 0.32],
            "trend_cap_range": [0.50, 1.00],
            "selection_period_only": True,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved V4 development artifacts to {output}")


def _v5_prepare(v5_config_path: Path) -> None:
    config = load_v5_config(v5_config_path)
    markets = {
        alias: load_market_data(config.data.raw_directory / f"{alias}.csv")
        for alias in config.data.assets
    }
    cash_yield = load_yield_data(config.data.raw_directory / "cash_yield.csv")
    frame = build_v5_research_frame(markets, cash_yield, risk_config=config.risk)
    config.data.processed_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.data.processed_path, index=True)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "start": str(frame.index.min().date()),
                "end": str(frame.index.max().date()),
                "destination": str(config.data.processed_path),
                "risk_horizons": list(config.risk.horizons),
            },
            indent=2,
        )
    )


def _v5_development(config_path: Path, v5_config_path: Path) -> None:
    app_config = load_config(config_path)
    config = load_v5_config(v5_config_path)
    date_columns = [
        *(f"target_end_date_{horizon}" for horizon in config.risk.horizons),
        *(f"vol_target_end_date_{horizon}" for horizon in config.risk.horizons),
    ]
    frame = pd.read_csv(
        config.data.processed_path,
        index_col="date",
        parse_dates=["date", *date_columns],
    )
    result = run_v5_walk_forward(frame, app_config=app_config, v5_config=config)
    output = Path("artifacts/v5_development")
    output.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(output / "decisions.csv", index=True)
    result.training_audit.to_csv(output / "training_audit.csv", index=False)
    result.model_leaderboard.to_csv(output / "model_leaderboard.csv", index=False)
    result.policy_selection.leaderboard.to_csv(output / "policy_leaderboard.csv", index=False)
    equity = pd.DataFrame(
        {
            f"policy_{name}": value.history["equity"]
            for name, value in result.policies.items()
        }
    )
    for name, value in result.baselines.items():
        equity[name] = value.history["equity"]
    equity.to_csv(output / "equity.csv", index=True)

    def enriched_metrics(backtest, start: str, end: str) -> dict[str, float]:
        history = backtest.history.loc[start:end]
        metrics = calculate_metrics(
            returns=history["strategy_return"],
            turnover=history["turnover"],
            asset_returns=history["asset_return"],
            positions=history["applied_position"],
            periods_per_year=252,
        )
        returns = history["strategy_return"].dropna()
        threshold = returns.quantile(0.05)
        metrics["expected_shortfall_5"] = float(returns.loc[returns <= threshold].mean())
        metrics["calmar_ratio"] = (
            metrics["annualized_return"] / abs(metrics["maximum_drawdown"])
            if metrics["maximum_drawdown"]
            else 0.0
        )
        return metrics

    periods = {
        "selection": (config.period.evaluation_start, config.period.selection_end),
        "retrospective_assessment": (
            config.period.retrospective_start,
            config.period.development_end,
        ),
    }
    subperiods = {}
    for period_name, (start, end) in periods.items():
        subperiods[period_name] = {
            "period": {"start": start, "end": end},
            "selected_policy": result.selected_policy,
            "v5_selected": enriched_metrics(result.strategy, start, end),
            "policies": {
                name: enriched_metrics(value, start, end)
                for name, value in result.policies.items()
            },
            "baselines": {
                name: enriched_metrics(value, start, end)
                for name, value in result.baselines.items()
            },
        }

    retrospective_start = config.period.retrospective_start
    retrospective_end = config.period.development_end
    strategy_returns = result.strategy.history.loc[
        retrospective_start:retrospective_end, "strategy_return"
    ]
    bootstrap = {
        name: circular_block_bootstrap_difference(
            strategy_returns,
            value.history.loc[retrospective_start:retrospective_end, "strategy_return"],
            samples=5_000,
            block_length=20,
            random_seed=config.models.random_seed,
            periods_per_year=252,
        )
        for name, value in result.baselines.items()
    }
    candidate_family = {
        **{f"policy_{name}": value for name, value in result.policies.items()},
        **result.baselines,
    }
    candidate_returns = return_family(candidate_family).loc[
        retrospective_start:retrospective_end
    ]
    robustness = {
        "retrospective_block_bootstrap": bootstrap,
        "deflated_sharpe": deflated_sharpe_probability(
            strategy_returns,
            trials=len(config.selection.policy_candidates),
            periods_per_year=252,
        ),
        "candidate_family_cscv": probability_of_backtest_overfitting(
            candidate_returns,
            partitions=8,
            periods_per_year=252,
        ),
        "cost_sensitivity": {},
        "execution_delay": {},
    }
    selected_position = result.policy_details[result.selected_policy]["desired_position"]
    aligned_close = frame.loc[selected_position.index, "close"]
    aligned_cash = frame.loc[selected_position.index, "cash_return"]
    for cost in (0, 5, 10, 20, 30):
        sensitivity = run_backtest(
            aligned_close,
            selected_position,
            transaction_cost_bps=cost,
            periods_per_year=252,
            cash_return=aligned_cash,
        )
        robustness["cost_sensitivity"][str(cost)] = enriched_metrics(
            sensitivity, retrospective_start, retrospective_end
        )
    for delay in (1, 2):
        delayed_position = selected_position.shift(delay - 1).fillna(0.0)
        sensitivity = run_backtest(
            aligned_close,
            delayed_position,
            transaction_cost_bps=app_config.experiment.transaction_cost_bps,
            periods_per_year=252,
            cash_return=aligned_cash,
        )
        robustness["execution_delay"][str(delay)] = enriched_metrics(
            sensitivity, retrospective_start, retrospective_end
        )
    stress_periods = {
        "2018_q4": ("2018-09-01", "2018-12-31"),
        "covid_2020": ("2020-02-01", "2020-06-30"),
        "bear_2022": ("2022-01-01", "2022-12-31"),
    }
    stress = {
        name: {
            "period": {"start": start, "end": end},
            "v5_selected": enriched_metrics(result.strategy, start, end),
            "buy_and_hold": enriched_metrics(result.baselines["buy_and_hold"], start, end),
        }
        for name, (start, end) in stress_periods.items()
    }
    payload = {
        "version": config.version,
        "period": {
            "start": config.period.evaluation_start,
            "end": config.period.development_end,
        },
        "prospective_start": config.period.prospective_start,
        "prospective_period_consulted": False,
        "retrospective_assessment_is_not_holdout": True,
        "decision_frequency": {"risk": "daily", "trend": "weekly"},
        "selected_policy": result.selected_policy,
        "selection": {
            "pareto_policies": list(result.policy_selection.pareto_policies),
            "target_constraints_met": result.policy_selection.target_constraints_met,
            "policy_candidates": list(config.selection.policy_candidates),
        },
        "v5_selected": result.strategy.metrics,
        "policies": {name: value.metrics for name, value in result.policies.items()},
        "baselines": {name: value.metrics for name, value in result.baselines.items()},
        "subperiods": subperiods,
        "stress_periods": stress,
        "robustness": robustness,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved V5 development artifacts to {output}")


def _freeze_v5(v5_config_path: Path) -> None:
    metrics_path = Path("artifacts/v5_development/metrics.json")
    if not metrics_path.exists():
        raise SystemExit("Run 'qqq-agents v5-development' before freezing V5")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = load_v5_config(v5_config_path)
    payload = {
        "config": str(v5_config_path),
        "sha256": _config_digest(v5_config_path),
        "frozen_files": _v5_file_digests(),
        "selected_policy": metrics["selected_policy"],
        "prospective_start": config.period.prospective_start,
        "development_metrics_sha256": _config_digest(metrics_path),
        "prospective_period_consulted": False,
    }
    destination = Path("configs/v5.lock.json")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _verify_v5_lock(v5_config_path: Path) -> dict[str, object]:
    lock_path = Path("configs/v5.lock.json")
    if not lock_path.exists():
        raise SystemExit("Freeze V5 with 'qqq-agents v5-freeze' before shadow evaluation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("sha256") != _config_digest(v5_config_path):
        raise SystemExit("configs/v5.yaml changed after V5 freeze")
    current_files = _v5_file_digests()
    changed = [
        path
        for path, digest in lock.get("frozen_files", {}).items()
        if current_files.get(path) != digest
    ]
    if changed:
        raise SystemExit(f"V5 implementation changed after freeze: {changed}")
    return lock


def _v5_shadow(config_path: Path, v5_config_path: Path, through: str) -> None:
    lock = _verify_v5_lock(v5_config_path)
    config = load_v5_config(v5_config_path)
    app_config = load_config(config_path)
    requested_end = pd.Timestamp(through)
    if requested_end < pd.Timestamp(config.period.prospective_start):
        raise SystemExit(
            f"V5 shadow mode begins on {config.period.prospective_start}; requested {through}"
        )
    download_end = (requested_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw_root = Path("data/raw/v5_shadow")
    raw_root.mkdir(parents=True, exist_ok=True)
    markets = {}
    for alias, ticker in config.data.assets.items():
        markets[alias] = download_market_data(
            ticker=ticker,
            start=config.data.start,
            end=download_end,
            destination=raw_root / f"{alias}.csv",
            auto_adjust=True,
        )
    cash_yield = download_yield_data(
        ticker=config.data.cash_yield_ticker,
        start=config.data.start,
        end=download_end,
        destination=raw_root / "cash_yield.csv",
    )
    frame = build_v5_research_frame(markets, cash_yield, risk_config=config.risk)
    result = run_v5_walk_forward(
        frame,
        app_config=app_config,
        v5_config=config,
        end=through,
        allow_prospective=True,
    )
    if result.selected_policy != lock["selected_policy"]:
        raise RuntimeError("Frozen V5 policy selection changed during shadow evaluation")
    as_of = result.decisions.index[-1]
    row = result.decisions.iloc[-1]
    signal = {
        "as_of": str(as_of.date()),
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": "shadow_only_no_execution",
        "selected_policy": result.selected_policy,
        "desired_qqq_exposure": float(row["selected_desired_position"]),
        "policy_state": str(row["selected_policy_state"]),
        "risk_score": float(row["risk_score"]),
        "forecast_volatility": float(row["forecast_volatility"]),
        "upper_volatility": float(row["upper_volatility"]),
        "daily_trend_score": float(row["daily_trend_score"]),
        "weekly_trend_score": float(row["weekly_trend_score"]),
        "risk_probabilities": {
            str(horizon): float(row[f"risk_probability_{horizon}"])
            for horizon in config.risk.horizons
        },
        "model_versions": {
            **{
                f"risk_{horizon}": str(row[f"risk_model_{horizon}"])
                for horizon in config.risk.horizons
            },
            "volatility_5": str(row["volatility_model_5"]),
            "volatility_20": str(row["volatility_model_20"]),
        },
        "policy_candidates": {
            policy: {
                "role": "champion" if policy == result.selected_policy else "challenger",
                "desired_qqq_exposure": float(detail.iloc[-1]["desired_position"]),
                "policy_state": str(detail.iloc[-1]["policy_state"]),
            }
            for policy, detail in result.policy_details.items()
        },
        "prospective_outcomes_used_for_tuning": False,
    }
    output = Path("artifacts/v5_shadow/signals")
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(signal, indent=2, sort_keys=True)
    (output / f"{as_of.date()}.json").write_text(encoded, encoding="utf-8")
    (output / "latest.json").write_text(encoded, encoding="utf-8")
    print(encoded)


def _freeze_v2(v2_config_path: Path) -> None:
    digest = _config_digest(v2_config_path)
    destination = Path("configs/v2.lock.json")
    payload = {
        "config": str(v2_config_path),
        "sha256": digest,
        "frozen_files": _v2_file_digests(),
        "protected_test_opened": False,
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _verify_v2_lock(v2_config_path: Path) -> None:
    lock_path = Path("configs/v2.lock.json")
    if not lock_path.exists():
        raise SystemExit("Freeze V2 with 'qqq-agents v2-freeze' before opening its protected test")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("sha256") != _config_digest(v2_config_path):
        raise SystemExit("configs/v2.yaml changed after its protocol was frozen")
    expected_files = lock.get("frozen_files", {})
    current_files = _v2_file_digests()
    changed = [path for path, digest in expected_files.items() if current_files.get(path) != digest]
    if changed:
        raise SystemExit(f"V2 implementation changed after freeze: {changed}")


def _open_v2_protected_data(config_path: Path, v2_config_path: Path) -> None:
    """Download a separate snapshot only after the V2 protocol is frozen."""

    _verify_v2_lock(v2_config_path)
    app_config = load_config(config_path)
    v2_config = load_v2_config(v2_config_path)
    end_exclusive = str(
        (pd.Timestamp(v2_config.development.protected_test_end) + pd.Timedelta(days=1)).date()
    )
    protected_root = Path("data/raw/v2_protected")
    market = download_market_data(
        ticker=app_config.data.ticker,
        start=v2_config.context.start,
        end=end_exclusive,
        destination=protected_root / "qqq.csv",
        auto_adjust=app_config.data.auto_adjust,
    )
    context_root = protected_root / "context"
    observations = download_context_bundle(
        tickers=v2_config.context.tickers,
        start=v2_config.context.start,
        end=end_exclusive,
        destination=context_root,
    )
    features = build_features(
        market,
        prediction_horizon=app_config.experiment.prediction_horizon_sessions,
        risk_event_threshold=app_config.experiment.risk_event_threshold,
    )
    context = build_context_features(
        market,
        tickers=v2_config.context.tickers,
        raw_directory=context_root,
    )
    weekly = sample_decisions(features.join(context), app_config.experiment.decision_frequency)
    weekly = rebuild_decision_interval_labels(
        weekly,
        market["close"],
        risk_event_threshold=app_config.experiment.risk_event_threshold,
    )
    destination = Path("data/processed/v2_protected_features.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(destination, index=True)
    lock_path = Path("configs/v2.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protected_test_opened"] = True
    lock["protected_snapshot_end"] = v2_config.development.protected_test_end
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "qqq_observations": len(market),
                "context_observations": observations,
                "weekly_rows": len(weekly),
                "destination": str(destination),
                "protected_test_opened": True,
            },
            indent=2,
        )
    )


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


def _quantitative_period(
    config_path: Path,
    *,
    start: str,
    end: str,
    include_shap: bool,
    personality: str,
    output_dir: Path,
    file_prefix: str,
    test_period_consulted: bool,
) -> None:
    config = load_config(config_path)
    frame = pd.read_csv(config.data.processed_path, index_col="date", parse_dates=["date"])
    result = run_quantitative_walk_forward(
        frame,
        config=config,
        validation_start=start,
        validation_end=end,
        include_shap=include_shap,
        personality_name=personality,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(output_dir / f"{file_prefix}_decisions.csv", index=True)
    result.strategy.history.to_csv(output_dir / f"{file_prefix}_strategy.csv", index=True)
    result.training_audit.to_csv(output_dir / f"{file_prefix}_training_audit.csv", index=False)
    equity = pd.DataFrame({"multiagent": result.strategy.history["equity"]})
    for name, baseline in result.baselines.items():
        equity[name] = baseline.history["equity"]
    equity.to_csv(output_dir / f"{file_prefix}_equity.csv", index=True)
    metrics = {
        "period": {"start": start, "end": end},
        "personality": personality,
        "test_period_consulted": test_period_consulted,
        "quantitative_multiagent": result.strategy.metrics,
        "baselines": {name: value.metrics for name, value in result.baselines.items()},
        "decision_counts": result.decisions["action"].value_counts().to_dict(),
        "risk_veto_count": int(result.decisions["risk_veto_triggered"].sum()),
    }
    (output_dir / f"{file_prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved quantitative period artifacts to {output_dir}")


def _walk_forward(config_path: Path, *, include_shap: bool, personality: str) -> None:
    config = load_config(config_path)
    validation_end = str((pd.Timestamp(config.experiment.test_start) - pd.Timedelta(days=1)).date())
    _quantitative_period(
        config_path,
        start=config.experiment.validation_start,
        end=validation_end,
        include_shap=include_shap,
        personality=personality,
        output_dir=Path("artifacts/walk_forward"),
        file_prefix="validation",
        test_period_consulted=False,
    )


def _final_quantitative(config_path: Path, *, include_shap: bool, personality: str) -> None:
    config = load_config(config_path)
    _quantitative_period(
        config_path,
        start=config.experiment.test_start,
        end=config.experiment.test_end,
        include_shap=include_shap,
        personality=personality,
        output_dir=Path("artifacts/final_quantitative"),
        file_prefix="final_test",
        test_period_consulted=True,
    )


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
        cache_name = "openai_gated_v1"
        cache_dir = Path("artifacts/llm/cache") / cache_name
        prior_spend = cached_spend(
            Path("artifacts/llm/cache"),
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
        cache_name = "mock_gated_v1"
        cache_dir = Path("artifacts/llm/cache") / cache_name
        raw_client = MockLLMClient()
        mode = "deterministic_dry_run"

    gated_client = EvidenceGatedLLMClient(raw_client)
    cached_client = CachedLLMClient(gated_client, cache_dir) if config.llm.cache else gated_client
    client = EvidenceValidatedLLMClient(cached_client)
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


async def _hybrid_period(
    config_path: Path,
    *,
    concurrency: int,
    provider: str,
    final_test: bool,
) -> None:
    config = load_config(config_path)
    if final_test:
        start = config.experiment.test_start
        end = config.experiment.test_end
        decisions_path = "artifacts/final_quantitative/final_test_decisions.csv"
        period_stem = "final_test"
    else:
        start = config.experiment.validation_start
        end = str((pd.Timestamp(config.experiment.test_start) - pd.Timedelta(days=1)).date())
        decisions_path = "artifacts/walk_forward/validation_decisions.csv"
        period_stem = "validation"
    cases = load_pilot_cases(
        features_path=config.data.processed_path,
        decisions_path=decisions_path,
        start=start,
        end=end,
        latest_allowed_date=end,
    )
    cache_root = Path("artifacts/llm/cache")
    raw_client: AutoGenOpenAIClient | MockLLMClient
    if provider == "openai":
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is missing from the local .env file")
        prior_spend = cached_spend(
            cache_root,
            input_price=config.llm.input_price_per_million,
            output_price=config.llm.output_price_per_million,
        )
        reserved_cost = sum(conservative_call_cost_bound(config, case.packet) for case in cases)
        if prior_spend + reserved_cost > config.llm.budget_usd:
            raise SystemExit(
                f"LLM budget guard blocked {period_stem}: ${prior_spend:.4f} recorded and "
                f"${reserved_cost:.4f} reserved against a ${config.llm.budget_usd:.2f} cap."
            )
        raw_client = AutoGenOpenAIClient(
            model=config.llm.model,
            api_key=api_key,
            reasoning_effort=config.llm.reasoning_effort,
            max_output_tokens=config.llm.max_output_tokens,
        )
        cache_name = "openai_gated_v1"
        output_dir = f"artifacts/hybrid_{period_stem}"
        period_name = period_stem
    else:
        raw_client = MockLLMClient()
        cache_name = f"mock_{period_stem}_gated_v1"
        output_dir = f"artifacts/hybrid_{period_stem}_mock"
        period_name = f"{period_stem}_mock"
    gated_client = EvidenceGatedLLMClient(raw_client)
    cached_client = CachedLLMClient(gated_client, cache_root / cache_name)
    client = EvidenceValidatedLLMClient(cached_client)
    last_reported = 0

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_reported
        if completed == total or completed - last_reported >= 10:
            print(f"LLM {period_stem} progress: {completed}/{total}", flush=True)
            last_reported = completed

    features = pd.read_csv(config.data.processed_path, index_col="date", parse_dates=["date"])
    close = features["close"]
    try:
        result = await run_hybrid_period(
            cases=cases,
            close=close,
            config=config,
            client=client,
            concurrency=concurrency,
            progress=report_progress,
        )
    finally:
        if isinstance(raw_client, AutoGenOpenAIClient):
            await raw_client.close()
    metrics = save_hybrid_period(
        result,
        output_dir=output_dir,
        period_name=period_name,
        start=start,
        end=end,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved hybrid validation artifacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--v2-config",
        type=Path,
        default=Path("configs/v2.yaml"),
        help="Path to the independent V2 experimental configuration.",
    )
    parser.add_argument(
        "--v3-config",
        type=Path,
        default=Path("configs/v3.yaml"),
        help="Path to the independent V3 research configuration.",
    )
    parser.add_argument(
        "--v4-config",
        type=Path,
        default=Path("configs/v4.yaml"),
        help="Path to the independent daily V4 research configuration.",
    )
    parser.add_argument(
        "--v5-config",
        type=Path,
        default=Path("configs/v5.yaml"),
        help="Path to the independent multi-frequency V5 research configuration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="Download and normalize QQQ market data.")
    subparsers.add_parser("prepare", help="Build weekly features and future labels.")
    subparsers.add_parser(
        "v2-download-context",
        help="Download only pre-2025 regime data without opening the protected V2 test.",
    )
    subparsers.add_parser(
        "v2-prepare",
        help="Build compact QQQ and external-regime features for V2.",
    )
    subparsers.add_parser(
        "v2-development",
        help="Run calibrated V2 over the diagnostic 2020-2024 development period.",
    )
    subparsers.add_parser(
        "v2-freeze",
        help="Hash and freeze the V2 specification before downloading protected data.",
    )
    v2_open_parser = subparsers.add_parser(
        "v2-open-protected-data",
        help="Download 2025-2026 snapshots only after V2 has been frozen.",
    )
    v2_open_parser.add_argument("--confirm-frozen-spec", action="store_true")
    v2_test_parser = subparsers.add_parser(
        "v2-protected-test",
        help="Open the frozen 2025-2026 V2 test after explicit confirmation.",
    )
    v2_test_parser.add_argument("--confirm-frozen-spec", action="store_true")
    subparsers.add_parser(
        "v3-download",
        help="Download the cross-asset and T-bill snapshots used by V3.",
    )
    subparsers.add_parser(
        "v3-prepare",
        help="Build the point-in-time cross-asset panel and cash returns.",
    )
    subparsers.add_parser(
        "v3-development",
        help="Run V3 only over its declared development period.",
    )
    subparsers.add_parser("v4-download", help="Download daily V4 context snapshots.")
    subparsers.add_parser("v4-prepare", help="Build the daily V4 research dataset.")
    subparsers.add_parser(
        "v4-development",
        help="Run daily V4 over selection and internal-validation periods.",
    )
    subparsers.add_parser(
        "v5-prepare",
        help="Build volatility-normalised multi-horizon targets for V5.",
    )
    subparsers.add_parser(
        "v5-development",
        help="Run V5 policy selection and retrospective robustness diagnostics.",
    )
    subparsers.add_parser(
        "v5-freeze",
        help="Freeze V5 code, policy and prospective start before shadow evaluation.",
    )
    v5_shadow_parser = subparsers.add_parser(
        "v5-shadow",
        help="Generate a non-executable prospective V5 signal from the frozen policy.",
    )
    v5_shadow_parser.add_argument(
        "--through",
        required=True,
        help="Inclusive market-data date in YYYY-MM-DD format.",
    )
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
    final_quant_parser = subparsers.add_parser(
        "final-quantitative",
        help="Open the frozen 2023-2024 quantitative test exactly once.",
    )
    final_quant_parser.add_argument(
        "--confirm-frozen-spec",
        action="store_true",
        help="Required acknowledgement that no further tuning will follow.",
    )
    final_quant_parser.add_argument("--with-shap", action="store_true")
    final_quant_parser.add_argument(
        "--personality",
        choices=("conservative", "aggressive", "opportunistic"),
        default="conservative",
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
    hybrid_validation_parser = subparsers.add_parser(
        "hybrid-validation",
        help="Evaluate the complete hybrid committee over 2020-2022 with cached LLM calls.",
    )
    hybrid_validation_parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum simultaneous validation dates.",
    )
    hybrid_validation_parser.add_argument(
        "--provider",
        choices=("mock", "openai"),
        default="mock",
        help="Use free local validation by default; OpenAI requires explicit selection.",
    )
    hybrid_final_parser = subparsers.add_parser(
        "hybrid-final-test",
        help="Evaluate the frozen hybrid committee on 2023-2024.",
    )
    hybrid_final_parser.add_argument(
        "--confirm-frozen-spec",
        action="store_true",
        help="Required acknowledgement that the final period cannot be used for tuning.",
    )
    hybrid_final_parser.add_argument("--concurrency", type=int, default=5)
    hybrid_final_parser.add_argument(
        "--provider",
        choices=("mock", "openai"),
        default="mock",
    )
    args = parser.parse_args()

    if args.command == "download":
        _download(args.config)
    elif args.command == "prepare":
        _prepare(args.config)
    elif args.command == "v2-download-context":
        _v2_download_context(args.v2_config)
    elif args.command == "v2-prepare":
        _v2_prepare(args.config, args.v2_config)
    elif args.command == "v2-development":
        _v2_evaluate(args.config, args.v2_config, protected=False)
    elif args.command == "v2-freeze":
        _freeze_v2(args.v2_config)
    elif args.command == "v2-open-protected-data":
        if not args.confirm_frozen_spec:
            raise SystemExit("Pass --confirm-frozen-spec to download the protected V2 snapshot")
        _open_v2_protected_data(args.config, args.v2_config)
    elif args.command == "v2-protected-test":
        if not args.confirm_frozen_spec:
            raise SystemExit("Pass --confirm-frozen-spec to open the protected V2 test")
        _verify_v2_lock(args.v2_config)
        _v2_evaluate(args.config, args.v2_config, protected=True)
    elif args.command == "v3-download":
        _v3_download(args.v3_config)
    elif args.command == "v3-prepare":
        _v3_prepare(args.config, args.v3_config)
    elif args.command == "v3-development":
        _v3_development(args.config, args.v3_config)
    elif args.command == "v4-download":
        _v4_download(args.v4_config)
    elif args.command == "v4-prepare":
        _v4_prepare(args.config, args.v4_config)
    elif args.command == "v4-development":
        _v4_development(args.config, args.v4_config)
    elif args.command == "v5-prepare":
        _v5_prepare(args.v5_config)
    elif args.command == "v5-development":
        _v5_development(args.config, args.v5_config)
    elif args.command == "v5-freeze":
        _freeze_v5(args.v5_config)
    elif args.command == "v5-shadow":
        _v5_shadow(args.config, args.v5_config, args.through)
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
    elif args.command == "final-quantitative":
        if not args.confirm_frozen_spec:
            raise SystemExit("Pass --confirm-frozen-spec to open the final test period")
        _final_quantitative(
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
    elif args.command == "hybrid-validation":
        asyncio.run(
            _hybrid_period(
                args.config,
                concurrency=args.concurrency,
                provider=args.provider,
                final_test=False,
            )
        )
    elif args.command == "hybrid-final-test":
        if not args.confirm_frozen_spec:
            raise SystemExit("Pass --confirm-frozen-spec to open the final test period")
        asyncio.run(
            _hybrid_period(
                args.config,
                concurrency=args.concurrency,
                provider=args.provider,
                final_test=True,
            )
        )


if __name__ == "__main__":
    main()
