"""One place that answers "what time is it?", so tests can answer for it."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

_now: Callable[[], datetime] | None = None


def utcnow() -> datetime:
    """Timezone-aware current time in UTC."""
    return _now() if _now is not None else datetime.now(timezone.utc)


def set_clock(fn: Callable[[], datetime] | None) -> None:
    """Replace the clock (tests only). ``None`` restores the real one."""
    global _now
    _now = fn


def iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
