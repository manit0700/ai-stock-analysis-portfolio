"""NewsAPI.org (free developer plan).

The developer plan delays articles by 24 hours and allows about 100 requests a day, so results
are always labelled delayed and are cached. The key travels in the ``X-Api-Key`` header.
"""
from __future__ import annotations

from typing import Any

from app.core.errors import ProviderError
from app.core.secrets import Secret
from app.marketdata.http import HttpClient
from app.marketdata.providers.base import Provider

BASE_URL = "https://newsapi.org/v2"


class NewsApiProvider(Provider):
    name = "newsapi"
    supports = frozenset({"news"})

    def __init__(self, api_key: Secret, http: HttpClient) -> None:
        self._key = api_key
        self._http = http

    def available(self) -> tuple[bool, str]:
        return (True, "") if self._key else (False, "NEWSAPI_KEY is not set.")

    def news(self, symbol: str | None, query: str | None, limit: int) -> list[dict[str, Any]]:
        if not self._key:
            raise ProviderError("NEWSAPI_KEY is not set.", details={"provider": self.name}, code="not_configured")
        headers = {"X-Api-Key": self._key.get()}
        size = max(1, min(int(limit), 20))
        if symbol or query:
            text = query or f"{symbol.upper()} stock"
            params = {"q": text, "language": "en", "sortBy": "publishedAt" if symbol else "relevancy", "pageSize": size}
            payload = self._http.get(f"{BASE_URL}/everything", params=params, headers=headers)
        else:
            payload = self._http.get(f"{BASE_URL}/top-headlines", params={"category": "business", "language": "en", "pageSize": size}, headers=headers)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise ProviderError("newsapi returned an error status.", details={"provider": self.name})
        items = []
        for article in payload.get("articles", [])[:limit]:
            title = article.get("title") or ""
            if not title or title == "[Removed]":
                continue
            items.append(
                {
                    "title": title,
                    "summary": article.get("description") or None,
                    "publisher": (article.get("source") or {}).get("name"),
                    "link": article.get("url"),
                    "published_at": article.get("publishedAt"),
                    "image": article.get("urlToImage") or None,
                }
            )
        return items
