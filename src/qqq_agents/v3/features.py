"""Point-in-time cross-asset panel and cash-return construction for V3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_agents.features import build_features, rebuild_decision_interval_labels, sample_decisions

PANEL_FEATURES = (
    "distance_sma_50",
    "distance_sma_200",
    "rsi_14",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "atr_14_pct",
    "drawdown_252",
    "market_momentum_20",
    "small_cap_momentum_20",
    "semiconductor_momentum_20",
    "bond_momentum_20",
    "cross_asset_breadth_20",
    "vix_level",
    "vix_change_20",
    "asset_iwm",
    "asset_qqq",
    "asset_smh",
    "asset_spy",
)

RISK_FEATURES = ("volatility_20", "atr_14_pct", "drawdown_252")


def build_cross_asset_panel(
    markets: dict[str, pd.DataFrame],
    *,
    decision_frequency: str = "W-FRI",
    risk_event_threshold: float = -0.03,
) -> pd.DataFrame:
    """Pool asset histories while preserving a common point-in-time feature definition."""

    required = {"qqq", "spy", "iwm", "smh", "tlt"}
    missing = required - set(markets)
    if missing:
        raise ValueError(f"V3 panel is missing required assets: {sorted(missing)}")
    common_index = markets["qqq"].index
    closes = pd.DataFrame(index=common_index)
    for alias, market in markets.items():
        closes[alias] = market["close"].reindex(common_index).ffill()

    context = pd.DataFrame(index=common_index)
    context["market_momentum_20"] = closes["spy"].pct_change(20)
    context["small_cap_momentum_20"] = closes["iwm"].pct_change(20)
    context["semiconductor_momentum_20"] = closes["smh"].pct_change(20)
    context["bond_momentum_20"] = closes["tlt"].pct_change(20)
    breadth = pd.concat(
        [(closes[name].pct_change(20) > 0).rename(name) for name in required], axis=1
    )
    context["cross_asset_breadth_20"] = breadth.mean(axis=1)
    if "vix" in closes:
        context["vix_level"] = closes["vix"]
        context["vix_change_20"] = closes["vix"].pct_change(20)

    panels: list[pd.DataFrame] = []
    for alias in sorted(required):
        features = build_features(markets[alias]).join(context)
        features.index.name = "date"
        weekly = sample_decisions(features, decision_frequency)
        weekly = rebuild_decision_interval_labels(
            weekly,
            markets[alias]["close"],
            risk_event_threshold=risk_event_threshold,
        )
        for identity in ("iwm", "qqq", "smh", "spy"):
            weekly[f"asset_{identity}"] = float(alias == identity)
        weekly["asset"] = alias
        panels.append(weekly.reset_index())
    panel = pd.concat(panels, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.set_index(["date", "asset"]).sort_index()


def cash_returns_from_yield(
    annual_yield_percent: pd.Series,
    decision_index: pd.DatetimeIndex,
    *,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Convert a quoted annualised T-bill yield into realised decision-period returns."""

    annual = annual_yield_percent.sort_index().astype(float).reindex(decision_index).ffill()
    if (annual.dropna() <= -100).any():
        raise ValueError("Annual yield cannot be less than or equal to -100%")
    daily = (1 + annual / 100) ** (1 / trading_days_per_year) - 1
    dates = pd.Series(decision_index, index=decision_index)
    previous = dates.shift(1)
    elapsed = (dates - previous).dt.days.fillna(0).clip(lower=0)
    result = (1 + daily) ** elapsed - 1
    result.name = "cash_return"
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
