"""Access control and request limiting for the HTTP API.

Scopes
    read    research routes that cost provider quota (tools, agents). Token optional: if
            ``MARKETVISION_TOOL_TOKEN`` is set it is required, otherwise the route is open.
    admin   routes that change state or spend money: paper orders, ledger resolution, the
            language-model copilot. ``MARKETVISION_ADMIN_TOKEN`` is required in production.
            In development, with no token set, they are open so that a local run needs no setup.

The admin token also satisfies the read scope. Comparisons are constant-time.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import Header, Request

from app.core.config import get_settings
from app.core.errors import AuthRequired, NotConfigured, PermissionDenied, RateLimited


def _bearer(header: str | None) -> str:
    if not header:
        return ""
    parts = header.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def _matches(provided: str, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def authorize(authorization: str | None, scope: str) -> None:
    """Raise unless the bearer token in ``authorization`` grants ``scope``."""
    settings = get_settings()
    provided = _bearer(authorization)
    admin = settings.admin_token.get()
    tool = settings.tool_token.get()

    if scope == "admin":
        if not admin:
            if settings.is_production:
                raise NotConfigured(
                    "This route is disabled until MARKETVISION_ADMIN_TOKEN is set on the server.",
                    details={"scope": "admin"},
                )
            return  # development convenience
        if not provided:
            raise AuthRequired("A bearer token is required for this route.", details={"scope": "admin"})
        if not _matches(provided, admin):
            raise PermissionDenied("The token does not grant access to this route.", details={"scope": "admin"})
        return

    if scope == "read":
        if not tool and not admin:
            return
        if not provided:
            raise AuthRequired("A bearer token is required for this route.", details={"scope": "read"})
        if _matches(provided, tool) or _matches(provided, admin):
            return
        raise PermissionDenied("The token does not grant access to this route.", details={"scope": "read"})

    raise PermissionDenied(f"Unknown scope {scope!r}.")


def require_read(authorization: str | None = Header(default=None)) -> None:
    authorize(authorization, "read")


def require_admin(authorization: str | None = Header(default=None)) -> None:
    authorize(authorization, "admin")


class SlidingWindowLimiter:
    """At most ``limit`` calls per ``per_seconds`` for each key. In-memory, per process."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, per_seconds: float, now: float | None = None) -> float:
        """Record a call. Returns 0 if allowed, otherwise the seconds to wait."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= per_seconds:
                hits.popleft()
            if len(hits) >= limit:
                return max(per_seconds - (now - hits[0]), 0.01)
            hits.append(now)
            return 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def rate_limit(bucket: str, limit: int, per_seconds: float = 60.0) -> Callable[[Request], None]:
    """FastAPI dependency: limit calls to an expensive route per client address."""

    def dependency(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(f"{bucket}:{client}", limit, per_seconds)
        if wait > 0:
            raise RateLimited(
                f"Too many requests to this route. Try again in {wait:.0f} seconds.",
                details={"bucket": bucket, "retry_after_seconds": round(wait, 1)},
            )

    return dependency
