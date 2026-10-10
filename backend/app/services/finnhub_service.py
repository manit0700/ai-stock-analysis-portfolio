"""Finnhub for the older dashboard routes (company profile, financials, news, analyst trends).

It no longer talks to Finnhub on its own. It borrows the process-wide Finnhub client from
``app.marketdata.clients``, so the key travels in the ``X-Finnhub-Token`` header (never in the
URL), errors never contain the key, and this module and the v2 data layer share one free-plan
budget. Answers are cached so a dashboard refresh does not spend the quota again.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.core.config import get_settings
from app.core.errors import MarketVisionError
from app.marketdata.cache import TTLCache
from app.marketdata.clients import shared_client

log = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"
_analyzer = SentimentIntensityAnalyzer()
_CACHE = TTLCache(maxsize=256)
#: Seconds each endpoint's answer is reused.
_TTL = {
    "/quote": 10.0,
    "/stock/profile2": 86400.0,
    "/stock/metric": 21600.0,
    "/stock/recommendation": 21600.0,
    "/calendar/earnings": 21600.0,
    "/company-news": 600.0,
    "/news": 600.0,
}


def _configured() -> bool:
    settings = get_settings()
    return bool(settings.finnhub_api_key) and not settings.offline


def _get(path: str, params: dict) -> dict | list | None:
    """GET a Finnhub endpoint. ``None`` when unconfigured or when the call fails (logged, redacted)."""
    settings = get_settings()
    if not _configured():
        return None
    cache_key = f"{path}|{json.dumps(params, sort_keys=True)}"
    hit = _CACHE.get(cache_key, _TTL.get(path, 300.0))
    if hit is not None:
        return hit[0]
    try:
        data = shared_client("finnhub").get(
            f"{_BASE}{path}", params=params, headers={"X-Finnhub-Token": settings.finnhub_api_key.get()}
        )
    except MarketVisionError as exc:
        log.warning("Finnhub %s failed: %s", path, exc.message)
        return None
    _CACHE.set(cache_key, data)
    return data


def clear_cache() -> None:
    _CACHE.clear()


def _sentiment(text: str) -> float:
    """VADER compound score: -1 (very negative) to +1 (very positive)."""
    return round(_analyzer.polarity_scores(text)["compound"], 3)


def _sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


class FinnhubService:
    def is_available(self) -> bool:
        """Configured (key set, not offline). Whether Finnhub answers is checked by the provider probe."""
        return _configured()

    def get_quote(self, ticker: str) -> dict[str, Any] | None:
        data = _get("/quote", {"symbol": ticker.upper()})
        if not data or data.get("c") is None:
            return None
        return {
            "current_price": data.get("c"),
            "change": data.get("d"),
            "change_pct": data.get("dp"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "prev_close": data.get("pc"),
        }

    def get_company_profile(self, ticker: str) -> dict[str, Any] | None:
        data = _get("/stock/profile2", {"symbol": ticker.upper()})
        if not data or not data.get("name"):
            return None
        return {
            "name": data.get("name"),
            "ticker": data.get("ticker"),
            "exchange": data.get("exchange"),
            "industry": data.get("finnhubIndustry"),
            "website": data.get("weburl"),
            "logo": data.get("logo"),
            "market_cap": data.get("marketCapitalization"),
            "shares_outstanding": data.get("shareOutstanding"),
            "ipo_date": data.get("ipo"),
            "country": data.get("country"),
        }

    def get_basic_financials(self, ticker: str) -> dict[str, Any] | None:
        data = _get("/stock/metric", {"symbol": ticker.upper(), "metric": "all"})
        if not data or "metric" not in data:
            return None
        m = data["metric"]
        return {
            "pe_ttm": m.get("peTTM"),
            "pb": m.get("pbAnnual"),
            "eps_ttm": m.get("epsTTM"),
            "revenue_growth_ttm": m.get("revenueGrowthTTMYoy"),
            "roe_ttm": m.get("roeTTM"),
            "debt_equity": m.get("totalDebt/totalEquityAnnual"),
            "current_ratio": m.get("currentRatioAnnual"),
            "52w_high": m.get("52WeekHigh"),
            "52w_low": m.get("52WeekLow"),
            "52w_return": m.get("52WeekPriceReturnDaily"),
            "beta": m.get("beta"),
            "dividend_yield": m.get("dividendYieldIndicatedAnnual"),
        }

    def get_news(self, ticker: str, days_back: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        """Company news with VADER sentiment scores."""
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=days_back)
        data = _get("/company-news", {
            "symbol": ticker.upper(),
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
        })
        if not data or not isinstance(data, list):
            return []

        results = []
        for item in data[:limit]:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            text_for_sentiment = f"{headline}. {summary}" if summary else headline
            score = _sentiment(text_for_sentiment)
            results.append({
                "title": headline,
                "summary": summary or None,
                "publisher": item.get("source"),
                "link": item.get("url"),
                "published_at": datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc).isoformat() if item.get("datetime") else None,
                "sentiment_score": score,
                "sentiment_label": _sentiment_label(score),
                "category": item.get("category"),
                "image": item.get("image"),
            })
        return results

    def get_market_news(self, category: str = "general", limit: int = 8) -> list[dict[str, Any]]:
        """Broad market news (not ticker-specific)."""
        data = _get("/news", {"category": category})
        if not data or not isinstance(data, list):
            return []
        results = []
        for item in data[:limit]:
            headline = item.get("headline", "")
            score = _sentiment(headline)
            results.append({
                "title": headline,
                "summary": item.get("summary"),
                "publisher": item.get("source"),
                "link": item.get("url"),
                "published_at": datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc).isoformat() if item.get("datetime") else None,
                "sentiment_score": score,
                "sentiment_label": _sentiment_label(score),
            })
        return results

    def get_recommendation_trends(self, ticker: str) -> dict[str, Any] | None:
        data = _get("/stock/recommendation", {"symbol": ticker.upper()})
        if not data or not isinstance(data, list) or not data:
            return None
        latest = data[0]
        total = sum([
            latest.get("strongBuy", 0),
            latest.get("buy", 0),
            latest.get("hold", 0),
            latest.get("sell", 0),
            latest.get("strongSell", 0),
        ])
        return {
            "period": latest.get("period"),
            "strong_buy": latest.get("strongBuy", 0),
            "buy": latest.get("buy", 0),
            "hold": latest.get("hold", 0),
            "sell": latest.get("sell", 0),
            "strong_sell": latest.get("strongSell", 0),
            "total_analysts": total,
            "consensus": (
                "buy" if (latest.get("strongBuy", 0) + latest.get("buy", 0)) > (latest.get("sell", 0) + latest.get("strongSell", 0))
                else "sell" if (latest.get("sell", 0) + latest.get("strongSell", 0)) > (latest.get("strongBuy", 0) + latest.get("buy", 0))
                else "hold"
            ),
        }

    def get_earnings_calendar(self, ticker: str) -> dict[str, Any] | None:
        to_date = datetime.now(timezone.utc) + timedelta(days=90)
        from_date = datetime.now(timezone.utc) - timedelta(days=30)
        data = _get("/calendar/earnings", {
            "symbol": ticker.upper(),
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
        })
        if not data or not data.get("earningsCalendar"):
            return None
        upcoming = [e for e in data["earningsCalendar"] if e.get("date", "") >= datetime.now(timezone.utc).strftime("%Y-%m-%d")]
        if not upcoming:
            return None
        next_earnings = upcoming[0]
        return {
            "date": next_earnings.get("date"),
            "eps_estimate": next_earnings.get("epsEstimate"),
            "revenue_estimate": next_earnings.get("revenueEstimate"),
            "quarter": next_earnings.get("quarter"),
            "year": next_earnings.get("year"),
        }
