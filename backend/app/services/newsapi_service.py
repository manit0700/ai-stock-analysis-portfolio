"""NewsAPI for the older dashboard routes, with VADER sentiment on each headline.

Shares the process-wide NewsAPI client (key in the ``X-Api-Key`` header, one 100-a-day budget
with the v2 data layer) and caches answers for ten minutes. In staging or production it stays
off unless the owner declares a paid plan, because the free Developer plan forbids that use.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.core.config import get_settings
from app.core.errors import MarketVisionError
from app.marketdata.cache import TTLCache
from app.marketdata.clients import shared_client

log = logging.getLogger(__name__)

_BASE = "https://newsapi.org/v2"
_TTL_SECONDS = 600.0
_analyzer = SentimentIntensityAnalyzer()
_CACHE = TTLCache(maxsize=128)


def _configured() -> bool:
    settings = get_settings()
    return bool(settings.newsapi_key) and not settings.offline and not settings.newsapi_blocked_reason


def _get(path: str, params: dict) -> dict | None:
    settings = get_settings()
    if not _configured():
        return None
    cache_key = f"{path}|{json.dumps(params, sort_keys=True)}"
    hit = _CACHE.get(cache_key, _TTL_SECONDS)
    if hit is not None:
        return hit[0]
    try:
        data = shared_client("newsapi").get(f"{_BASE}{path}", params=params, headers={"X-Api-Key": settings.newsapi_key.get()})
    except MarketVisionError as exc:
        log.warning("NewsAPI %s failed: %s", path, exc.message)
        return None
    _CACHE.set(cache_key, data)
    return data


def clear_cache() -> None:
    _CACHE.clear()


def _sentiment(text: str) -> float:
    return round(_analyzer.polarity_scores(text)["compound"], 3)


def _sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def _enrich(article: dict) -> dict:
    title = article.get("title") or ""
    description = article.get("description") or ""
    text = f"{title}. {description}" if description else title
    score = _sentiment(text)
    return {
        "title": title,
        "summary": description or None,
        "publisher": (article.get("source") or {}).get("name"),
        "link": article.get("url"),
        "published_at": article.get("publishedAt"),
        "image": article.get("urlToImage"),
        "sentiment_score": score,
        "sentiment_label": _sentiment_label(score),
    }


class NewsApiService:
    def is_available(self) -> bool:
        return _configured()

    def get_stock_news(self, ticker: str, company_name: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """Search for news about a specific stock ticker/company."""
        query = f"{ticker} stock" if not company_name else f"{ticker} OR {company_name}"
        data = _get("/everything", {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 20),
        })
        if not data or data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        return [_enrich(a) for a in articles[:limit] if a.get("title") and a["title"] != "[Removed]"]

    def get_market_headlines(self, limit: int = 10) -> list[dict[str, Any]]:
        """Top business/finance headlines."""
        data = _get("/top-headlines", {
            "category": "business",
            "language": "en",
            "pageSize": min(limit, 20),
        })
        if not data or data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        return [_enrich(a) for a in articles[:limit] if a.get("title") and a["title"] != "[Removed]"]

    def search_news(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Free-form news search."""
        data = _get("/everything", {
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(limit, 20),
        })
        if not data or data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        return [_enrich(a) for a in articles[:limit] if a.get("title") and a["title"] != "[Removed]"]
