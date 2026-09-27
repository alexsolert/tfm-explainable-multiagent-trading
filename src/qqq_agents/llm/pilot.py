"""Construccion y ejecucion controlada de un caso LLM de validacion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path

import pandas as pd

from qqq_agents.config import AppConfig
from qqq_agents.coordinator import DeterministicCoordinator
from qqq_agents.llm.committee import LLMCommittee
from qqq_agents.llm.contracts import LLMCallResult, MarketContextPacket
from qqq_agents.llm.providers import StructuredLLMClient
from qqq_agents.personality import load_personality
from qqq_agents.schemas import Action, AgentSignal, Personality

MARKET_FEATURES = (
    "close",
    "return_1d",
    "distance_sma_20",
    "distance_sma_50",
    "distance_sma_200",
    "ema_spread_12_26",
    "macd_histogram",
    "rsi_14",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "atr_14_pct",
    "drawdown_252",
    "volume_zscore_20",
)
QUANTITATIVE_AGENT_IDS = ("technical", "momentum", "risk")


@dataclass(frozen=True)
class PilotCase:
    packet: MarketContextPacket
    quantitative_signals: tuple[AgentSignal, ...]
    current_position: float


def _read_indexed_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame.sort_index()


def load_pilot_case(
    *,
    features_path: str | Path,
    decisions_path: str | Path,
    requested_date: str | None = None,
    latest_allowed_date: str = "2022-12-31",
) -> PilotCase:
    """Load one development-period decision without consulting final-test rows."""

    features = _read_indexed_csv(features_path)
    decisions = _read_indexed_csv(decisions_path)
    allowed_end = pd.Timestamp(latest_allowed_date)
    eligible = decisions.loc[decisions.index <= allowed_end]
    if eligible.empty:
        raise ValueError("No development-period decisions are available")
    selected = pd.Timestamp(requested_date).normalize() if requested_date else eligible.index[-1]
    if selected > allowed_end:
        raise ValueError(
            f"The LLM pilot cannot consult {selected.date()}; latest allowed date is "
            f"{allowed_end.date()}"
        )
    if selected not in eligible.index:
        raise ValueError(f"No validation decision is available for {selected.date()}")
    if selected not in features.index:
        raise ValueError(f"No market features are available for {selected.date()}")

    decision = eligible.loc[selected]
    observation = features.loc[selected]
    personality = Personality(str(decision["personality"]))
    packet = MarketContextPacket(
        as_of=selected.date(),
        personality=personality,
        market_features={name: float(observation[name]) for name in MARKET_FEATURES},
        quantitative_signals={
            agent_id: float(decision[f"{agent_id}_signal"]) for agent_id in QUANTITATIVE_AGENT_IDS
        },
        proposed_action=Action(str(decision["action"])),
    )

    signals: list[AgentSignal] = []
    for agent_id in QUANTITATIVE_AGENT_IDS:
        veto_probability = float(decision["risk_veto_probability"]) if agent_id == "risk" else None
        signals.append(
            AgentSignal(
                agent_id=agent_id,
                agent_type="quantitative",
                as_of=selected.date(),
                signal=float(decision[f"{agent_id}_signal"]),
                confidence=float(decision[f"{agent_id}_confidence"]),
                explanation="Stored output from the leakage-safe walk-forward validation.",
                model_name="random_forest_classifier",
                model_version=str(decision[f"{agent_id}_model_version"]),
                personality=personality,
                veto_probability=veto_probability,
                veto_reason=(
                    "The estimated adverse-event probability exceeded the configured limit."
                    if veto_probability is not None
                    else None
                ),
            )
        )

    preceding = eligible.loc[eligible.index < selected, "desired_position"]
    current_position = float(preceding.iloc[-1]) if not preceding.empty else 0.0
    return PilotCase(packet, tuple(signals), current_position)


def cached_spend(
    cache_dir: str | Path,
    *,
    input_price: float,
    output_price: float,
) -> float:
    """Estimate already incurred spend from unique persisted API responses."""

    total = 0.0
    for path in Path(cache_dir).glob("**/*.json"):
        result = LLMCallResult.model_validate_json(path.read_text(encoding="utf-8"))
        total += result.estimated_cost(input_price=input_price, output_price=output_price)
    return total


def conservative_call_cost_bound(config: AppConfig, packet: MarketContextPacket) -> float:
    """Pre-call guardrail; token accounting after each response remains authoritative."""

    # Three input packets plus prompt overhead. One UTF-8 token is conservatively
    # approximated as no more than three characters for this bounded payload.
    packet_tokens = math.ceil(len(packet.model_dump_json()) / 3)
    input_tokens = 3 * (packet_tokens + 1_000)
    output_tokens = 3 * config.llm.max_output_tokens
    return (
        input_tokens * config.llm.input_price_per_million
        + output_tokens * config.llm.output_price_per_million
    ) / 1_000_000


async def execute_pilot(
    *,
    case: PilotCase,
    config: AppConfig,
    client: StructuredLLMClient,
    mode: str,
) -> dict[str, object]:
    results = await LLMCommittee(client).evaluate(case.packet)
    profile = load_personality(f"configs/personalities/{case.packet.personality.value}.yaml")
    llm_signals = tuple(
        profile.transform_signal(
            result.to_agent_signal(
                as_of=case.packet.as_of,
                personality=case.packet.personality,
            )
        )
        for result in results
    )
    coordinator = DeterministicCoordinator(
        weights=profile.transform_weights(config.coordinator.weights),
        buy_threshold=max(
            0.01,
            min(1.0, config.coordinator.buy_threshold + profile.buy_threshold_delta),
        ),
        sell_threshold=config.coordinator.sell_threshold,
        risk_veto_probability=max(
            0.0,
            min(
                1.0,
                config.coordinator.risk_veto_probability + profile.risk_veto_probability_delta,
            ),
        ),
    )
    trace = coordinator.decide(
        as_of=case.packet.as_of,
        signals=(*case.quantitative_signals, *llm_signals),
        current_position=case.current_position,
        created_at=datetime.combine(case.packet.as_of, time.min, tzinfo=UTC),
    )
    incremental_cost = sum(
        result.estimated_cost(
            input_price=config.llm.input_price_per_million,
            output_price=config.llm.output_price_per_million,
        )
        for result in results
        if not result.cached
    )
    return {
        "mode": mode,
        "test_period_consulted": False,
        "packet": case.packet.model_dump(mode="json"),
        "llm_results": [result.model_dump(mode="json") for result in results],
        "incremental_estimated_cost_usd": incremental_cost,
        "combined_decision": trace.model_dump(mode="json"),
    }


def save_pilot_result(payload: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
