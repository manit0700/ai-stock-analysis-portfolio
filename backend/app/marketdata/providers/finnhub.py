"""Finnhub (free tier): quotes, company profile, company news, earnings dates.

The key is sent in the ``X-Finnhub-Token`` header, never in the URL, so it cannot end up in a
request log. It stays on the server: the browser asks this backend for quotes.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.core.errors import ProviderError
from app.core.secrets import Secret
from app.marketdata.http import HttpClient
from app.marketdata.providers.base import Provider

BASE_URL = "https://finnhub.io/api/v1"


class FinnhubProvider(Provider):
    name = "finnhub"
    supports = frozenset({"quote", "profile", "news", "earnings"})

    def __init__(self, api_key: Secret, http: HttpClient) -> None:
        self._key = api_key
        self._http = http

    def available(self) -> tuple[bool, str]:
        return (True, "") if self._key else (False, "FINNHUB_API_KEY is not set.")

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        if not self._key:
            raise ProviderError("FINNHUB_API_KEY is not set.", details={"provider": self.name}, code="not_configured")
        return self._http.get(f"{BASE_URL}{path}", params=params, headers={"X-Finnhub-Token": self._key.get()})

    def quote(self, symbol: str) -> dict[str, Any]:
        data = self._get("/quote", {"symbol": symbol.upper()})
        price = data.get("c") if isinstance(data, dict) else None
        if not price:
            raise ProviderError(f"finnhub returned no quote for {symbol.upper()}.", details={"provider": self.name}, code="no_data")
        stamp = datetime.fromtimestamp(data["t"], tz=timezone.utc) if data.get("t") else None
        return {
            "price": float(price),
            "market_timestamp": stamp,
            "realtime": True,  # Finnhub states US quotes are real-time; the service still checks the age
            "previous_close": data.get("pc"),
            "open": data.get("o"),
            "high": data.get("h"),
            "low": data.get("l"),
            "change": data.get("d"),
            "change_pct": data.get("dp"),
            "basis": "last trade",
        }

    def profile(self, symbol: str) -> dict[str, Any]:
        data = self._get("/stock/profile2", {"symbol": symbol.upper()})
        if not data:
            raise ProviderError(f"finnhub returned no profile for {symbol.upper()}.", details={"provider": self.name}, code="no_data")
        cap = data.get("marketCapitalization")
        return {
            "symbol": symbol.upper(),
            "name": data.get("name"),
            "exchange": data.get("exchange"),
            "currency": data.get("currency"),
            "sector": None,
            "industry": data.get("finnhubIndustry"),
            "country": data.get("country"),
            "website": data.get("weburl"),
            "market_cap": float(cap) * 1e6 if cap else None,  # Finnhub reports millions
            "shares_outstanding": float(data["shareOutstanding"]) * 1e6 if data.get("shareOutstanding") else None,
            "ipo_date": data.get("ipo"),
            "logo": data.get("logo"),
        }

    def news(self, symbol: str | None, query: str | None, limit: int) -> list[dict[str, Any]]:
        if symbol:
            today = datetime.now(timezone.utc).date()
            raw = self._get("/company-news", {"symbol": symbol.upper(), "from": (today - timedelta(days=7)).isoformat(), "to": today.isoformat()})
        else:
            raw = self._get("/news", {"category": query or "general"})
        items = []
        for entry in (raw or [])[:limit]:
            stamp = datetime.fromtimestamp(entry["datetime"], tz=timezone.utc).isoformat() if entry.get("datetime") else None
            items.append(
                {
                    "title": entry.get("headline") or "",
                    "summary": entry.get("summary") or None,
                    "publisher": entry.get("source"),
                    "link": entry.get("url"),
                    "published_at": stamp,
                    "image": entry.get("image") or None,
                    "category": entry.get("category"),
                }
            )
        return [item for item in items if item["title"]]

    def earnings_calendar(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        data = self._get("/calendar/earnings", {"symbol": symbol.upper(), "from": start.isoformat(), "to": end.isoformat()})
        rows = data.get("earningsCalendar", []) if isinstance(data, dict) else []
        return [
            {
                "date": row.get("date"),
                "hour": row.get("hour"),
                "eps_estimate": row.get("epsEstimate"),
                "eps_actual": row.get("epsActual"),
                "revenue_estimate": row.get("revenueEstimate"),
                "revenue_actual": row.get("revenueActual"),
                "quarter": row.get("quarter"),
                "year": row.get("year"),
            }
            for row in rows
        ]
