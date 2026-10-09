"""What a market-data provider can be asked for. A provider implements only what it supports."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from app.core.errors import NotSupported


class Provider:
    """Base class. Methods a provider does not support raise ``NotSupported``."""

    name = "provider"
    #: Datasets this provider can serve: history, quote, profile, actions, news, economic, earnings.
    supports: frozenset[str] = frozenset()

    def available(self) -> tuple[bool, str]:
        """Whether the provider can be used right now, and why not if it cannot."""
        return True, ""

    # Each method returns raw provider data. The unified service adds the envelope.
    def history(self, symbol: str, start: date | None, end: date | None, interval: str) -> pd.DataFrame:
        raise NotSupported(f"{self.name} does not provide price history.")

    def quote(self, symbol: str) -> dict[str, Any]:
        """Keys: price, market_timestamp (aware datetime or None), realtime (bool), and optional
        previous_close, open, high, low, volume, change, change_pct."""
        raise NotSupported(f"{self.name} does not provide quotes.")

    def profile(self, symbol: str) -> dict[str, Any]:
        raise NotSupported(f"{self.name} does not provide company profiles.")

    def corporate_actions(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        """Columns: Dividends, Stock Splits, indexed by ex-date."""
        raise NotSupported(f"{self.name} does not provide corporate actions.")

    def news(self, symbol: str | None, query: str | None, limit: int) -> list[dict[str, Any]]:
        """Items with: title, summary, publisher, link, published_at (ISO string or None)."""
        raise NotSupported(f"{self.name} does not provide news.")

    def economic_series(self, series_id: str, start: date | None, end: date | None) -> pd.Series:
        raise NotSupported(f"{self.name} does not provide economic series.")

    def earnings_calendar(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        raise NotSupported(f"{self.name} does not provide an earnings calendar.")


def to_utc(value: Any) -> datetime | None:
    """Best-effort conversion of a provider timestamp to an aware UTC datetime."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT:
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()
