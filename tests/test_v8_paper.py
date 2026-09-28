import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qqq_agents.v8 import load_v8_config
from qqq_agents.v8.paper import (
    default_account,
    generate_paper_decision,
    load_account,
    persist_paper_run,
    validate_paper_frame,
)

ROOT = Path(__file__).resolve().parents[1]


def _frame(end: str = "2024-12-31") -> pd.DataFrame:
    index = pd.date_range("2022-01-03", end, freq="B")
    close = pd.Series(100 * np.exp(np.arange(len(index)) * 0.0005), index=index)
    return pd.DataFrame(
        {
            "close": close,
            "volatility_20": 0.18,
            "drawdown_252": -0.03,
            "qqq_relative_spy_20": 0.01,
            "cash_return": 0.0001,
        },
        index=index,
    )


def test_paper_frame_rejects_stale_snapshot() -> None:
    with pytest.raises(ValueError, match="stale"):
        validate_paper_frame(_frame(), requested_through="2025-01-31")


def test_paper_decision_is_non_executing_and_costed() -> None:
    frame = _frame()
    config = load_v8_config(ROOT / "configs/v8.yaml")
    profile = next(value for value in config.profiles if value.name == "balanced")
    state = default_account("demo", profile.name, 10_000)
    record, next_state = generate_paper_decision(
        frame,
        config=config,
        profile=profile,
        state=state,
        snapshot_sha256="snapshot",
        config_sha256="config",
        implementation_sha256="implementation",
        generated_at_utc="2025-01-01T00:00:00+00:00",
    )
    assert record["mode"] == "paper_only_no_execution"
    assert record["safety"]["broker_connected"] is False
    assert record["safety"]["orders_submitted"] is False
    assert record["decision"]["target_exposure"] == 1.5
    assert record["paper_portfolio"]["transaction_cost"] > 0
    assert next_state.equity < 10_000
    assert next_state.exposure == 1.5


def test_paper_records_are_immutable_and_idempotent(tmp_path: Path) -> None:
    frame = _frame()
    config = load_v8_config(ROOT / "configs/v8.yaml")
    profile = config.profiles[0]
    state = default_account("demo", profile.name, 10_000)
    record, next_state = generate_paper_decision(
        frame,
        config=config,
        profile=profile,
        state=state,
        snapshot_sha256="snapshot",
        config_sha256="config",
        implementation_sha256="implementation",
        generated_at_utc="2025-01-01T00:00:00+00:00",
    )
    path, created = persist_paper_run(record, next_state, root=tmp_path)
    assert created is True
    assert path.exists()
    _, created_again = persist_paper_run(record, next_state, root=tmp_path)
    assert created_again is False
    assert len(pd.read_csv(tmp_path / "ledger.csv")) == 1
    changed = {**record, "snapshot_sha256": "different"}
    with pytest.raises(FileExistsError, match="Immutable"):
        persist_paper_run(changed, next_state, root=tmp_path)
    stored = load_account(
        tmp_path / "accounts" / f"demo-{profile.name}.json",
        account="demo",
        profile=profile.name,
        initial_capital=10_000,
    )
    assert stored.runs == 1
    assert json.loads(path.read_text(encoding="utf-8"))["record_sha256"]


def test_existing_account_is_marked_to_market() -> None:
    first = _frame("2024-12-20")
    second = _frame("2024-12-31")
    config = load_v8_config(ROOT / "configs/v8.yaml")
    profile = config.profiles[1]
    record, state = generate_paper_decision(
        first,
        config=config,
        profile=profile,
        state=default_account("demo", profile.name, 10_000),
        snapshot_sha256="one",
        config_sha256="config",
        implementation_sha256="implementation",
        generated_at_utc="2024-12-21T00:00:00+00:00",
    )
    assert record["paper_portfolio"]["paper_return_since_previous"] == 0
    later, later_state = generate_paper_decision(
        second,
        config=config,
        profile=profile,
        state=state,
        snapshot_sha256="two",
        config_sha256="config",
        implementation_sha256="implementation",
        generated_at_utc="2025-01-01T00:00:00+00:00",
    )
    assert later["paper_portfolio"]["asset_return_since_previous"] > 0
    assert later["paper_portfolio"]["paper_return_since_previous"] > 0
    assert later_state.runs == 2
