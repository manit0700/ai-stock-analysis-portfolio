"""Keeping credentials out of logs, error messages and API responses.

Three layers, because any one of them can be bypassed by a careless call:

1. ``Secret`` wraps a credential so that ``str()``, ``repr()`` and f-strings never show it.
2. ``redact()`` scrubs known secret values and key-looking URL parameters from any text.
3. ``RedactingFilter`` applies ``redact()`` to every log record before it is written.
"""
from __future__ import annotations

import logging
import re
import threading

_MASK = "<redacted>"

# URL/query parameters and header names that carry credentials for the providers this project uses.
_PARAM_PATTERN = re.compile(
    r"(?i)\b(token|api[_-]?key|apikey|access[_-]?token|secret|password|authorization|x-api-key|x-finnhub-token)"
    r"(\s*[=:]\s*)(bearer\s+)?([^\s&\"'<>,;]+)"
)

_registered: set[str] = set()
_lock = threading.Lock()


class Secret:
    """A credential that does not reveal itself when printed, logged or serialised by accident."""

    __slots__ = ("_value",)

    def __init__(self, value: str | None) -> None:
        self._value = (value or "").strip()
        if self._value:
            register_secret(self._value)

    def get(self) -> str:
        """The real value. Call this only at the point where the credential is sent to its provider."""
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "Secret(set)" if self._value else "Secret(unset)"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self._value)


def register_secret(value: str) -> None:
    """Remember a secret value so ``redact`` can remove it wherever it appears."""
    value = (value or "").strip()
    if len(value) >= 6:  # shorter strings would redact ordinary words
        with _lock:
            _registered.add(value)


def redact(text: object) -> str:
    """Return ``text`` with registered secrets and credential-looking parameters masked."""
    out = str(text)
    with _lock:
        known = sorted(_registered, key=len, reverse=True)
    for value in known:
        if value in out:
            out = out.replace(value, _MASK)
    return _PARAM_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", out)


class RedactingFilter(logging.Filter):
    """Scrub every log record. Attached to handlers so that third-party loggers are covered too."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken format string must not break logging
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        if record.exc_info and record.exc_info[1] is not None:
            # The traceback text may hold the URL of a failed request. Keep the type, drop the raw text.
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


def install_log_redaction() -> None:
    """Attach the redacting filter to the root logger and to every handler already configured."""
    root = logging.getLogger()
    targets: list[logging.Filterer] = [root, *root.handlers]
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        targets.extend([logger, *logger.handlers])
    for target in targets:
        if not any(isinstance(f, RedactingFilter) for f in target.filters):
            target.addFilter(RedactingFilter())
