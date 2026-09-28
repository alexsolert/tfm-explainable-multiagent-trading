"""Adaptador neutral a proveedor para noticias verificables y conocidas ex ante."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from qqq_agents.llm.contracts import DatedHeadline

REQUIRED_COLUMNS = ("evidence_id", "published_at", "text", "source")


class HistoricalNewsStore:
    def __init__(self, frame: pd.DataFrame) -> None:
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Historical news is missing columns: {sorted(missing)}")
        normalized = frame.copy()
        normalized["published_at"] = pd.to_datetime(normalized["published_at"], utc=True)
        if normalized["evidence_id"].duplicated().any():
            raise ValueError("News evidence identifiers must be unique")
        self.frame = normalized.sort_values("published_at")

    @classmethod
    def from_csv(cls, path: str | Path) -> HistoricalNewsStore:
        return cls(pd.read_csv(path))

    def as_of(
        self,
        decision_date: date,
        *,
        lookback_days: int = 7,
        limit: int = 20,
        ticker: str | None = "QQQ",
    ) -> tuple[DatedHeadline, ...]:
        cutoff = pd.Timestamp(decision_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(
            microseconds=1
        )
        start = cutoff - pd.Timedelta(days=lookback_days)
        eligible = self.frame.loc[
            (self.frame["published_at"] >= start) & (self.frame["published_at"] <= cutoff)
        ]
        if ticker is not None and "ticker" in eligible.columns:
            eligible = eligible.loc[eligible["ticker"].fillna(ticker).isin((ticker, "MARKET"))]
        selected = eligible.tail(limit)
        return tuple(
            DatedHeadline(
                evidence_id=str(row.evidence_id),
                published_on=row.published_at.date(),
                text=str(row.text),
                source=str(row.source),
            )
            for row in selected.itertuples(index=False)
        )
