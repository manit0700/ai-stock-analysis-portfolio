"""Yahoo Finance through the ``yfinance`` library.

Free and unofficial. Quotes can be delayed by 15 minutes or more depending on the exchange, so
nothing from this provider is ever labelled real-time. ``yfinance`` is imported only when the
provider is used, so the rest of the application runs without it.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.core.errors import ProviderError
from app.marketdata.providers.base import Provider, to_utc

_INTRADAY_LOOKBACK_DAYS = {"1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "90m": 59, "1h": 729}


def _local_date(stamp: Any) -> date:
    """The exchange-local calendar date of a bar (Yahoo indexes bars in the exchange's time zone)."""
    return pd.Timestamp(stamp).date()


class YahooProvider(Provider):
    name = "yahoo"
    supports = frozenset({"history", "quote", "profile", "actions", "news"})

    def __init__(self, yf_module: Any | None = None) -> None:
        self._yf = yf_module

    def available(self) -> tuple[bool, str]:
        try:
            self._module()
        except ProviderError as exc:
            return False, exc.message
        return True, ""

    def _module(self) -> Any:
        if self._yf is None:
            try:
                import yfinance as yf
            except ImportError:
                raise ProviderError("The yfinance package is not installed.", details={"provider": self.name}) from None
            self._yf = yf
        return self._yf

    def _ticker(self, symbol: str) -> Any:
        return self._module().Ticker(symbol.upper())

    def history(self, symbol: str, start: date | None, end: date | None, interval: str) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"interval": interval, "auto_adjust": False, "actions": True}
        if interval in _INTRADAY_LOOKBACK_DAYS:
            earliest = datetime.now(timezone.utc).date() - timedelta(days=_INTRADAY_LOOKBACK_DAYS[interval])
            start = max(start, earliest) if start else earliest
        if start:
            kwargs["start"] = start.isoformat()
        if end:
            kwargs["end"] = (end + timedelta(days=1)).isoformat()  # yfinance treats ``end`` as exclusive
        if not start and not end:
            kwargs["period"] = "max"
        try:
            frame = self._ticker(symbol).history(**kwargs)
        except Exception as exc:  # noqa: BLE001 - yfinance raises many unrelated types
            raise ProviderError(f"yahoo history for {symbol.upper()} failed ({type(exc).__name__}).", details={"provider": self.name}) from None
        if frame is None or len(frame) == 0:
            raise ProviderError(f"yahoo returned no price history for {symbol.upper()}.", details={"provider": self.name}, code="no_data")
        return frame

    def quote(self, symbol: str) -> dict[str, Any]:
        """Latest one-minute bar of the current session, else the last daily close."""
        ticker = self._ticker(symbol)
        try:
            bars = ticker.history(period="1d", interval="1m", auto_adjust=False, actions=False)
        except Exception:  # noqa: BLE001
            bars = None
        previous_close = None
        try:
            daily = ticker.history(period="5d", interval="1d", auto_adjust=False, actions=False)
        except Exception as exc:  # noqa: BLE001
            if bars is None or len(bars) == 0:
                raise ProviderError(f"yahoo quote for {symbol.upper()} failed ({type(exc).__name__}).", details={"provider": self.name}) from None
            daily = None
        if bars is not None and len(bars):
            last = bars.dropna(subset=["Close"]).iloc[-1:]
            if len(last):
                stamp = to_utc(last.index[-1])
                session_day = _local_date(last.index[-1])
                if daily is not None and len(daily):
                    closes = daily.dropna(subset=["Close"])
                    before = closes[[_local_date(ix) < session_day for ix in closes.index]]
                    if len(before):
                        previous_close = float(before["Close"].iloc[-1])
                day = bars.dropna(subset=["Close"])
                return {
                    "price": float(last["Close"].iloc[-1]),
                    "market_timestamp": stamp,
                    "realtime": False,
                    "previous_close": previous_close,
                    "open": float(day["Open"].iloc[0]),
                    "high": float(day["High"].max()),
                    "low": float(day["Low"].min()),
                    "volume": float(day["Volume"].sum()),
                    "basis": "last one-minute bar",
                }
        if daily is None or len(daily) == 0:
            raise ProviderError(f"yahoo returned no quote for {symbol.upper()}.", details={"provider": self.name}, code="no_data")
        daily = daily.dropna(subset=["Close"])
        row = daily.iloc[-1]
        return {
            "price": float(row["Close"]),
            "market_timestamp": None,  # a daily bar has a date, not a time; the service derives the close time
            "session_date": pd.Timestamp(daily.index[-1]).date().isoformat(),
            "realtime": False,
            "previous_close": float(daily["Close"].iloc[-2]) if len(daily) > 1 else None,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": float(row["Volume"]),
            "basis": "last daily close",
        }

    def profile(self, symbol: str) -> dict[str, Any]:
        try:
            info = dict(self._ticker(symbol).info or {})
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"yahoo profile for {symbol.upper()} failed ({type(exc).__name__}).", details={"provider": self.name}) from None
        if not info:
            raise ProviderError(f"yahoo returned no profile for {symbol.upper()}.", details={"provider": self.name}, code="no_data")
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "exchange": info.get("exchange"),
            "currency": info.get("currency"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "market_cap": info.get("marketCap"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "summary": info.get("longBusinessSummary"),
        }

    def corporate_actions(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        try:
            actions = self._ticker(symbol).actions
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"yahoo corporate actions for {symbol.upper()} failed ({type(exc).__name__}).", details={"provider": self.name}) from None
        if actions is None or len(actions) == 0:
            return pd.DataFrame(columns=["Dividends", "Stock Splits"])
        actions = actions.copy()
        actions.index = pd.to_datetime(actions.index).tz_localize(None) if actions.index.tz is not None else pd.to_datetime(actions.index)
        if start:
            actions = actions[actions.index >= pd.Timestamp(start)]
        if end:
            actions = actions[actions.index <= pd.Timestamp(end)]
        return actions

    def news(self, symbol: str | None, query: str | None, limit: int) -> list[dict[str, Any]]:
        if not symbol:
            raise ProviderError("yahoo news needs a ticker symbol.", details={"provider": self.name}, code="not_supported")
        try:
            raw = self._ticker(symbol).news or []
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"yahoo news for {symbol.upper()} failed ({type(exc).__name__}).", details={"provider": self.name}) from None
        items: list[dict[str, Any]] = []
        for entry in raw[:limit]:
            content = entry.get("content") if isinstance(entry.get("content"), dict) else entry
            published = content.get("pubDate") or content.get("displayTime")
            if published is None and entry.get("providerPublishTime"):
                published = datetime.fromtimestamp(entry["providerPublishTime"], tz=timezone.utc).isoformat()
            provider = content.get("provider")
            link = content.get("canonicalUrl") or content.get("clickThroughUrl")
            items.append(
                {
                    "title": content.get("title") or "",
                    "summary": content.get("summary") or content.get("description"),
                    "publisher": (provider or {}).get("displayName") if isinstance(provider, dict) else entry.get("publisher"),
                    "link": (link or {}).get("url") if isinstance(link, dict) else entry.get("link"),
                    "published_at": str(published) if published else None,
                }
            )
        return [item for item in items if item["title"]]
