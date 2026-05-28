from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

_API_KEY = os.getenv("NEWSAPI_KEY", "")
_BASE = "https://newsapi.org/v2"
_TIMEOUT = 10
_analyzer = SentimentIntensityAnalyzer()


def _sentiment(text: str) -> float:
    return round(_analyzer.polarity_scores(text)["compound"], 3)


def _sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def _get(path: str, params: dict) -> dict | None:
    if not _API_KEY:
        return None
    try:
        params["apiKey"] = _API_KEY
        r = requests.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("NewsAPI %s failed: %s", path, exc)
        return None


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
        return bool(_API_KEY)

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
