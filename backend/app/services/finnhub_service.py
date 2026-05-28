from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load .env from backend/ directory regardless of working directory
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

_API_KEY = os.getenv("FINNHUB_API_KEY", "")
_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 10
_analyzer = SentimentIntensityAnalyzer()


def _get(path: str, params: dict) -> dict | list | None:
    if not _API_KEY:
        return None
    try:
        params["token"] = _API_KEY
        r = requests.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("Finnhub %s failed: %s", path, exc)
        return None


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
        return bool(_API_KEY)

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
