"""Descarga reproducible y validacion basica de datos OHLCV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _normalize_yfinance_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No market data returned for {ticker}")

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        ticker_level = normalized.columns.get_level_values(-1)
        if ticker in ticker_level:
            normalized = normalized.xs(ticker, axis=1, level=-1)
        else:
            normalized.columns = normalized.columns.get_level_values(0)

    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized]
    normalized.index = pd.DatetimeIndex(normalized.index).tz_localize(None)
    normalized.index.name = "date"
    normalized = normalized.sort_index()

    missing = set(REQUIRED_COLUMNS) - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required market columns: {sorted(missing)}")
    if normalized.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates")
    if not normalized.index.is_monotonic_increasing:
        raise ValueError("Market data must be ordered chronologically")
    if (normalized[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Market prices must be strictly positive")

    return normalized.loc[:, list(REQUIRED_COLUMNS)]


def download_market_data(
    *,
    ticker: str,
    start: str,
    end: str,
    destination: str | Path,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV data and persist the normalized snapshot as CSV."""

    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        actions=False,
        multi_level_index=False,
    )
    normalized = _normalize_yfinance_frame(frame, ticker)
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=True)
    return normalized


def load_market_data(path: str | Path) -> pd.DataFrame:
    """Load a previously downloaded snapshot without contacting the provider."""

    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return _normalize_yfinance_frame(frame, ticker="stored_snapshot")
