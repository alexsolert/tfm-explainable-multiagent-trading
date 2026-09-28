"""Datos de regimen externos, almacenados en snapshots separados y fechados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qqq_agents.data.market import download_market_data, load_market_data


def download_context_bundle(
    *,
    tickers: dict[str, str],
    start: str,
    end: str,
    destination: str | Path,
) -> dict[str, int]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    observations: dict[str, int] = {}
    for alias, ticker in tickers.items():
        frame = download_market_data(
            ticker=ticker,
            start=start,
            end=end,
            destination=root / f"{alias}.csv",
            auto_adjust=True,
        )
        observations[alias] = len(frame)
    return observations


def build_context_features(
    qqq_market: pd.DataFrame,
    *,
    tickers: dict[str, str],
    raw_directory: str | Path,
) -> pd.DataFrame:
    """Create only backward-looking context features on the QQQ trading calendar."""

    root = Path(raw_directory)
    close = qqq_market["close"].rename("qqq")
    closes = pd.DataFrame(index=qqq_market.index)
    closes["qqq"] = close
    for alias in tickers:
        path = root / f"{alias}.csv"
        if not path.exists():
            continue
        external = load_market_data(path)
        closes[alias] = external["close"].reindex(closes.index).ffill()

    result = pd.DataFrame(index=closes.index)
    for alias in ("spy", "iwm", "smh", "tlt"):
        if alias in closes:
            result[f"{alias}_momentum_20"] = closes[alias].pct_change(20)
            result[f"{alias}_momentum_60"] = closes[alias].pct_change(60)
    if "spy" in closes:
        result["qqq_relative_spy_20"] = close.pct_change(20) - closes["spy"].pct_change(20)
        result["qqq_relative_spy_60"] = close.pct_change(60) - closes["spy"].pct_change(60)
    if "vix" in closes:
        result["vix_level"] = closes["vix"]
        result["vix_change_20"] = closes["vix"].pct_change(20)
    breadth_members = [
        result[f"{alias}_momentum_20"] > 0
        for alias in ("spy", "iwm", "smh")
        if f"{alias}_momentum_20" in result
    ]
    if breadth_members:
        result["breadth_proxy_20"] = pd.concat(breadth_members, axis=1).mean(axis=1)
    return result
