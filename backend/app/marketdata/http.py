"""HTTP for data providers: time-outs, retries with back-off, rate limits, and no secrets in errors.

Every provider goes through ``HttpClient``. Error messages name the provider, the path and the
status code only. Query strings and headers are never put in a message or a log line, because
some providers (FRED) only accept the key as a URL parameter.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

from app.core.errors import ProviderError, RateLimited

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER_SECONDS = 30.0


class TokenBucket:
    """Allow ``rate`` calls per second with bursts up to ``capacity``. Thread-safe."""

    def __init__(self, rate: float, capacity: float, *, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.rate, self.capacity = float(rate), float(capacity)
        self._tokens = float(capacity)
        self._clock, self._sleep = clock, sleep
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    def acquire(self, *, max_wait: float = 10.0) -> float:
        """Take one token, waiting up to ``max_wait`` seconds. Returns the time waited."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0 - 1e-9:  # tolerance: refill arithmetic can land a hair under 1
                    self._tokens = max(self._tokens - 1.0, 0.0)
                    return waited
                need = (1.0 - self._tokens) / self.rate if self.rate > 0 else float("inf")
            if waited + need > max_wait:
                raise RateLimited(
                    "The local request budget for this provider is used up for now.",
                    details={"retry_after_seconds": round(need, 2)},
                )
            self._sleep(need)
            waited += need


class HttpClient:
    def __init__(
        self,
        provider: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        rate_limiter: TokenBucket | None = None,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = backoff_base
        self.rate_limiter = rate_limiter
        self._session = session
        self._sleep = sleep

    def _get_session(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "marketvision-research/1.0"})
        return self._session

    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None, expect: str = "json") -> Any:
        """GET ``url``. Returns parsed JSON (or text). Raises ``ProviderError`` or ``RateLimited``."""
        import requests

        path = urlsplit(url).path or "/"
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            try:
                response = self._get_session().get(url, params=params, headers=headers, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = type(exc).__name__  # the text of these exceptions contains the full URL
            except requests.RequestException as exc:
                raise ProviderError(f"{self.provider} request to {path} failed ({type(exc).__name__}).", details={"provider": self.provider}) from None
            else:
                status = response.status_code
                if status < 400:
                    try:
                        return response.json() if expect == "json" else response.text
                    except ValueError:
                        raise ProviderError(f"{self.provider} returned a response that is not valid JSON for {path}.", details={"provider": self.provider}) from None
                if status in (401, 403):
                    raise ProviderError(
                        f"{self.provider} refused the request to {path} (HTTP {status}). Check the API key and plan.",
                        details={"provider": self.provider, "status": status},
                        code="provider_auth",
                    )
                if status not in RETRY_STATUSES:
                    raise ProviderError(f"{self.provider} answered HTTP {status} for {path}.", details={"provider": self.provider, "status": status})
                last_error = f"HTTP {status}"
                if attempt < self.max_retries:
                    retry_after = _retry_after(response)
                    if retry_after is not None:
                        self._sleep(min(retry_after, MAX_RETRY_AFTER_SECONDS))
                        continue
                elif status == 429:
                    raise RateLimited(f"{self.provider} is rate-limiting requests to {path}.", details={"provider": self.provider, "status": 429})
            if attempt < self.max_retries:
                delay = self.backoff_base * (2**attempt) * (1 + random.random() * 0.25)
                log.info("%s %s: %s, retry %d of %d in %.1fs", self.provider, path, last_error, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
        raise ProviderError(
            f"{self.provider} request to {path} failed after {self.max_retries + 1} attempts ({last_error}).",
            details={"provider": self.provider, "last_error": last_error},
        )


def _retry_after(response: Any) -> float | None:
    raw = response.headers.get("Retry-After") if getattr(response, "headers", None) else None
    try:
        return max(float(raw), 0.0) if raw is not None else None
    except (TypeError, ValueError):
        return None
