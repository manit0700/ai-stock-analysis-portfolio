"""One HTTP client per keyed provider, shared by every caller in the process.

The v2 data layer and the older dashboard services used to keep separate clients, so each spent
the free quota on its own and together they could exceed it. Everything now borrows the client
below, whose token bucket is sized to stay inside the provider's published free-plan limit even
in the first minute after start-up (burst + rate x 60 <= limit).

Budgets are per process. Two processes (for example two uvicorn workers) each get their own.
"""
from __future__ import annotations

import threading

from app.core.config import Settings, get_settings
from app.marketdata.http import HttpClient, TokenBucket

#: provider -> (tokens per second, burst). Comments give the published free-plan limit.
BUDGETS: dict[str, tuple[float, float]] = {
    "finnhub": (0.9, 5.0),  # free plan: 60 calls/minute -> at most 59 in any first minute
    "fred": (1.8, 10.0),  # FRED API: about 120 requests/minute -> at most 118
    "newsapi": (90.0 / 86400.0, 10.0),  # Developer plan: 100 requests/day -> at most 100
}

_clients: dict[str, HttpClient] = {}
_lock = threading.Lock()


def shared_client(provider: str, settings: Settings | None = None) -> HttpClient:
    """The process-wide client for ``provider`` (finnhub, fred or newsapi)."""
    if provider not in BUDGETS:
        raise KeyError(f"No shared client is defined for {provider!r}.")
    with _lock:
        client = _clients.get(provider)
        if client is None:
            settings = settings or get_settings()
            rate, burst = BUDGETS[provider]
            client = HttpClient(
                provider,
                timeout=settings.http_timeout_seconds,
                max_retries=settings.http_max_retries,
                rate_limiter=TokenBucket(rate, burst),
            )
            _clients[provider] = client
        return client


def reset_shared_clients() -> None:
    """Forget the shared clients (tests that change settings or swap the HTTP session)."""
    with _lock:
        _clients.clear()
