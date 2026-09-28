"""Idempotent, non-executing V8 paper-portfolio engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qqq_agents.v8.agents import build_agent_signals
from qqq_agents.v8.config import ProfileConfig, V8Config
from qqq_agents.v8.coordinator import coordinate_profile, decision_action


@dataclass(frozen=True)
class PaperAccountState:
    account: str
    profile: str
    currency: str
    equity: float
    exposure: float
    equivalent_qqq_shares: float
    last_market_date: str | None
    last_price: float | None
    runs: int


def validate_paper_frame(
    frame: pd.DataFrame, *, requested_through: str | pd.Timestamp
) -> dict[str, object]:
    required = {
        "close",
        "volatility_20",
        "drawdown_252",
        "qqq_relative_spy_20",
        "cash_return",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Paper frame is missing columns: {sorted(missing)}")
    if len(frame) < 300:
        raise ValueError("Paper frame requires at least 300 observations")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Paper frame dates must be unique and increasing")
    latest = pd.Timestamp(frame.index.max()).normalize()
    requested = pd.Timestamp(requested_through).normalize()
    if latest > requested:
        raise ValueError("Paper snapshot contains observations after the requested date")
    lag_days = int((requested - latest).days)
    if lag_days > 7:
        raise ValueError(
            f"Paper snapshot is stale: latest market date {latest.date()} is "
            f"{lag_days} calendar days behind {requested.date()}"
        )
    returns = frame["close"].pct_change().dropna()
    if (returns.abs() > 0.50).any():
        raise ValueError("Paper snapshot contains an implausible one-session QQQ move")
    latest_row = frame.loc[latest]
    if not np.isfinite(latest_row[list(required)].astype(float)).all():
        raise ValueError("Paper snapshot has non-finite values on its latest date")
    return {
        "requested_through": str(requested.date()),
        "latest_market_date": str(latest.date()),
        "calendar_lag_days": lag_days,
        "observations": len(frame),
        "fresh": True,
    }


def default_account(account: str, profile: str, initial_capital: float) -> PaperAccountState:
    if initial_capital <= 0:
        raise ValueError("Initial paper capital must be positive")
    return PaperAccountState(
        account=account,
        profile=profile,
        currency="USD",
        equity=float(initial_capital),
        exposure=0.0,
        equivalent_qqq_shares=0.0,
        last_market_date=None,
        last_price=None,
        runs=0,
    )


def load_account(
    path: Path, *, account: str, profile: str, initial_capital: float
) -> PaperAccountState:
    if not path.exists():
        return default_account(account, profile, initial_capital)
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = PaperAccountState(**payload)
    if state.account != account or state.profile != profile:
        raise ValueError("Stored paper account identity does not match the request")
    return state


def _mark_to_market(
    state: PaperAccountState,
    frame: pd.DataFrame,
    *,
    borrowing_spread_bps: float,
) -> tuple[float, dict[str, float]]:
    if state.last_market_date is None:
        return state.equity, {
            "asset_return_since_previous": 0.0,
            "paper_return_since_previous": 0.0,
            "financing_cost": 0.0,
            "cash_contribution": 0.0,
        }
    previous = pd.Timestamp(state.last_market_date)
    latest = pd.Timestamp(frame.index.max())
    if latest < previous:
        raise ValueError("Paper data predates the stored account state")
    if latest == previous:
        return state.equity, {
            "asset_return_since_previous": 0.0,
            "paper_return_since_previous": 0.0,
            "financing_cost": 0.0,
            "cash_contribution": 0.0,
        }
    period = frame.loc[(frame.index > previous) & (frame.index <= latest)]
    if period.empty or state.last_price is None:
        raise ValueError("Paper snapshot does not bridge the previous account date")
    prior_close = float(state.last_price)
    equity = float(state.equity)
    financing_total = 0.0
    cash_total = 0.0
    spread_daily = (1 + borrowing_spread_bps / 10_000) ** (1 / 252) - 1
    for _, row in period.iterrows():
        asset_return = float(row["close"] / prior_close - 1)
        cash_return = float(row["cash_return"])
        financing = max(state.exposure - 1, 0) * (cash_return + spread_daily)
        cash_contribution = (1 - min(state.exposure, 1)) * cash_return
        paper_return = state.exposure * asset_return + cash_contribution - financing
        equity *= 1 + paper_return
        financing_total += financing
        cash_total += cash_contribution
        prior_close = float(row["close"])
    asset_return_total = float(frame.loc[latest, "close"] / state.last_price - 1)
    return equity, {
        "asset_return_since_previous": asset_return_total,
        "paper_return_since_previous": float(equity / state.equity - 1),
        "financing_cost": financing_total,
        "cash_contribution": cash_total,
    }


def generate_paper_decision(
    frame: pd.DataFrame,
    *,
    config: V8Config,
    profile: ProfileConfig,
    state: PaperAccountState,
    snapshot_sha256: str,
    config_sha256: str,
    implementation_sha256: str,
    generated_at_utc: str,
) -> tuple[dict[str, Any], PaperAccountState]:
    signals = build_agent_signals(frame, config.signals).dropna()
    if signals.empty:
        raise ValueError("No complete V8 paper signal is available")
    signal = signals.iloc[-1]
    decisions = coordinate_profile(signals.tail(2), profile)
    current = decisions.iloc[-1]
    target = float(current["desired_position"])
    market_date = pd.Timestamp(signals.index[-1])
    signal_date = pd.Timestamp(signal["signal_as_of"])
    price = float(frame.loc[market_date, "close"])
    equity_before_trade, performance = _mark_to_market(
        state,
        frame,
        borrowing_spread_bps=config.costs.borrowing_spread_bps,
    )
    current_equivalent_shares = state.exposure * equity_before_trade / price
    action = decision_action(target, state.exposure)
    turnover = abs(target - state.exposure)
    transaction_cost = (
        turnover * equity_before_trade * config.costs.transaction_cost_bps / 10_000
    )
    equity_after_trade = equity_before_trade - transaction_cost
    target_shares = target * equity_after_trade / price
    equivalent_trade_shares = target_shares - current_equivalent_shares
    next_state = PaperAccountState(
        account=state.account,
        profile=state.profile,
        currency=state.currency,
        equity=equity_after_trade,
        exposure=target,
        equivalent_qqq_shares=target_shares,
        last_market_date=str(market_date.date()),
        last_price=price,
        runs=state.runs + 1,
    )
    record: dict[str, Any] = {
        "record_schema": "v8-paper-1",
        "framework_version": config.version,
        "mode": "paper_only_no_execution",
        "generated_at_utc": generated_at_utc,
        "market_as_of": str(market_date.date()),
        "signal_as_of": str(signal_date.date()),
        "snapshot_sha256": snapshot_sha256,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "account": state.account,
        "currency": state.currency,
        "profile": {"name": profile.name, "label": profile.label},
        "agents": {
            "trend": {
                "score": float(signal["trend_score"]),
                "vote": float(signal["trend_vote"]),
            },
            "volatility": {
                "annualized": float(signal["forecast_volatility"]),
                "risk_level": float(signal["volatility_risk"]),
                "authority": "hard_veto_only_when_extreme",
            },
            "drawdown": {
                "value_252": float(signal["drawdown_252"]),
                "risk_level": float(signal["drawdown_risk"]),
                "authority": "advisory",
            },
            "relative_strength": {
                "qqq_vs_spy_20": float(signal["relative_strength_20"]),
                "vote": float(signal["relative_strength_vote"]),
                "authority": "advisory",
            },
        },
        "committee_score": float(signal["committee_score"]),
        "decision": {
            "action": action,
            "previous_exposure": state.exposure,
            "target_exposure": target,
            "policy_state": str(current["policy_state"]),
            "signal_strength": float(current["signal_strength"]),
            "explanation": str(current["explanation"]),
        },
        "paper_portfolio": {
            "price": price,
            "equity_before_trade": equity_before_trade,
            "transaction_cost": transaction_cost,
            "equity_after_trade": equity_after_trade,
            "current_equivalent_qqq_shares": current_equivalent_shares,
            "target_equivalent_qqq_shares": target_shares,
            "equivalent_trade_shares": equivalent_trade_shares,
            **performance,
        },
        "safety": {
            "broker_connected": False,
            "orders_submitted": False,
            "leverage_is_synthetic": target > 1,
            "disclaimer": "Academic paper simulation; not investment advice.",
        },
    }
    digest_source = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["record_sha256"] = hashlib.sha256(digest_source.encode()).hexdigest()
    return record, next_state


def persist_paper_run(
    record: dict[str, Any],
    next_state: PaperAccountState,
    *,
    root: Path,
) -> tuple[Path, bool]:
    profile = str(record["profile"]["name"])
    account = str(record["account"])
    market_date = str(record["market_as_of"])
    runs = root / "runs"
    accounts = root / "accounts"
    runs.mkdir(parents=True, exist_ok=True)
    accounts.mkdir(parents=True, exist_ok=True)
    record_path = runs / f"{market_date}-{account}-{profile}.json"
    account_path = accounts / f"{account}-{profile}.json"
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        same_inputs = all(
            existing.get(field) == record.get(field)
            for field in (
                "snapshot_sha256",
                "config_sha256",
                "implementation_sha256",
                "account",
            )
        ) and existing.get("profile") == record.get("profile")
        if not same_inputs:
            raise FileExistsError(
                f"Immutable paper record already exists with different inputs: {record_path}"
            )
        return record_path, False
    encoded = json.dumps(record, indent=2, sort_keys=True)
    record_path.write_text(encoded, encoding="utf-8")
    (root / "latest.json").write_text(encoded, encoding="utf-8")
    account_path.write_text(
        json.dumps(asdict(next_state), indent=2, sort_keys=True), encoding="utf-8"
    )
    ledger_path = root / "ledger.csv"
    ledger_row = pd.DataFrame(
        [
            {
                "market_as_of": record["market_as_of"],
                "signal_as_of": record["signal_as_of"],
                "account": account,
                "profile": profile,
                "action": record["decision"]["action"],
                "previous_exposure": record["decision"]["previous_exposure"],
                "target_exposure": record["decision"]["target_exposure"],
                "price": record["paper_portfolio"]["price"],
                "equity": record["paper_portfolio"]["equity_after_trade"],
                "transaction_cost": record["paper_portfolio"]["transaction_cost"],
                "record_sha256": record["record_sha256"],
            }
        ]
    )
    ledger_row.to_csv(
        ledger_path,
        mode="a" if ledger_path.exists() else "w",
        header=not ledger_path.exists(),
        index=False,
    )
    return record_path, True
