import pandas as pd
import pytest

from qqq_agents.backtesting.baselines import buy_and_hold, moving_average_crossover
from qqq_agents.backtesting.engine import run_backtest


def test_decision_is_applied_to_next_period_return() -> None:
    index = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    close = pd.Series([100.0, 110.0, 121.0, 133.1], index=index)
    desired_position = pd.Series([1.0, 1.0, 1.0, 1.0], index=index)

    result = run_backtest(close, desired_position, transaction_cost_bps=0)

    assert result.history["strategy_return"].iloc[0] == 0.0
    assert result.history["strategy_return"].iloc[1] == pytest.approx(0.1)


def test_transaction_cost_is_charged_when_position_changes() -> None:
    index = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    desired_position = pd.Series([1.0, 0.0, 1.0, 1.0], index=index)

    result = run_backtest(close, desired_position, transaction_cost_bps=10)

    assert result.history["cost"].sum() == pytest.approx(0.003)


def test_baselines_return_long_only_positions() -> None:
    close = pd.Series(range(1, 251), dtype=float)

    hold = buy_and_hold(close)
    crossover = moving_average_crossover(close)

    assert hold.eq(1.0).all()
    assert crossover.between(0, 1).all()
