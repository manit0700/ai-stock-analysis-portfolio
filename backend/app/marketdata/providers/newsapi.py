"""NewsAPI.org.

The free Developer plan delays articles by 24 hours, searches at most a month back, allows 100
requests a day, and "cannot be used in a staging or production environment" (newsapi.org/pricing).
So results are always labelled delayed and cached, and the provider refuses to run in staging or
production unless the owner declares a paid plan. The key travels in the ``X-Api-Key`` header.
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

    def __init__(self, api_key: Secret, http: HttpClient, *, blocked_reason: str = "") -> None:
        self._key = api_key
        self._http = http
        self._blocked = blocked_reason

    def available(self) -> tuple[bool, str]:
        if not self._key:
            return False, "NEWSAPI_KEY is not set."
        if self._blocked:
            return False, self._blocked
        return True, ""

    def news(self, symbol: str | None, query: str | None, limit: int) -> list[dict[str, Any]]:
        ok, reason = self.available()
        if not ok:
            raise ProviderError(reason, details={"provider": self.name}, code="not_configured")
        headers = {"X-Api-Key": self._key.get()}
        size = max(1, min(int(limit), 20))
        if symbol or query:
            text = query or f"{symbol.upper()} stock"
            params = {"q": text, "language": "en", "sortBy": "publishedAt" if symbol else "relevancy", "pageSize": size}
            payload = self._http.get(f"{BASE_URL}/everything", params=params, headers=headers)
        else:
            payload = self._http.get(f"{BASE_URL}/top-headlines", params={"category": "business", "language": "en", "pageSize": size}, headers=headers)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            # NewsAPI error bodies carry a short code such as "rateLimited" or "apiKeyInvalid".
            code = str(payload.get("code", "unknown"))[:40] if isinstance(payload, dict) else "unknown"
            raise ProviderError(f"newsapi returned an error status ({code}).", details={"provider": self.name, "newsapi_code": code})
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
