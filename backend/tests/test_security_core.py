"""Secrets stay out of logs and messages; tokens are checked; expensive routes are limited."""
from __future__ import annotations

import logging

import pytest

from app.core import security
from app.core.config import get_settings
from app.core.errors import AuthRequired, NotConfigured, PermissionDenied
from app.core.secrets import RedactingFilter, Secret, redact


def test_secret_never_prints_its_value() -> None:
    secret = Secret("fh_live_key_ABC123xyz")
    assert "fh_live_key_ABC123xyz" not in f"{secret} {secret!r} {[secret]} {dict(key=secret)}"
    assert bool(secret) and secret.get() == "fh_live_key_ABC123xyz"
    assert not Secret("") and not Secret(None)


def test_redact_removes_registered_values_and_key_parameters() -> None:
    Secret("fred_key_SUPERSECRET_42")
    text = (
        "404 Client Error for url: https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key=fred_key_SUPERSECRET_42 "
        "and https://finnhub.io/api/v1/quote?symbol=AAPL&token=abc123def456 and apiKey=zzz999 "
        "Authorization: Bearer tok_live_1234567890"
    )
    cleaned = redact(text)
    for leaked in ("fred_key_SUPERSECRET_42", "abc123def456", "zzz999", "tok_live_1234567890"):
        assert leaked not in cleaned
    assert "series_id=DGS10" in cleaned and "symbol=AAPL" in cleaned  # ordinary parameters survive


def test_log_filter_scrubs_messages_arguments_and_tracebacks(caplog) -> None:
    Secret("news_key_TOPSECRET_77")
    logger = logging.getLogger("marketvision.test.redaction")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.WARNING, logger=logger.name):
        logger.warning("NewsAPI %s failed: %s", "/everything", "HTTPError for url: https://newsapi.org/v2/everything?q=x&apiKey=news_key_TOPSECRET_77")
        try:
            raise RuntimeError("boom https://finnhub.io/api/v1/quote?token=news_key_TOPSECRET_77")
        except RuntimeError:
            logger.exception("request failed")
    text = "\n".join(record.getMessage() + (record.exc_text or "") for record in caplog.records)
    assert "news_key_TOPSECRET_77" not in text and "<redacted>" in text


def test_settings_hold_secrets_wrapped_and_report_capabilities(settings_env) -> None:
    settings_env(FINNHUB_API_KEY="fh_cfg_key_123456", MARKETVISION_OFFLINE="0", MARKETVISION_CORS_ORIGINS="https://app.example.com/, http://localhost:3000")
    settings = get_settings()
    assert "fh_cfg_key_123456" not in repr(settings)
    assert settings.capabilities()["finnhub"] is True and settings.capabilities()["newsapi"] is False
    assert settings.cors_origins == ("https://app.example.com", "http://localhost:3000")
    assert "*" not in settings.cors_origins


def test_read_scope_is_open_without_a_token_and_enforced_with_one(settings_env) -> None:
    security.authorize(None, "read")
    settings_env(MARKETVISION_TOOL_TOKEN="tool-token-123456")
    with pytest.raises(AuthRequired):
        security.authorize(None, "read")
    with pytest.raises(PermissionDenied):
        security.authorize("Bearer wrong-token", "read")
    security.authorize("Bearer tool-token-123456", "read")
    security.authorize("bearer tool-token-123456", "read")


def test_admin_scope_requires_the_admin_token(settings_env) -> None:
    settings_env(MARKETVISION_TOOL_TOKEN="tool-token-123456", MARKETVISION_ADMIN_TOKEN="admin-token-654321")
    with pytest.raises(AuthRequired):
        security.authorize(None, "admin")
    with pytest.raises(PermissionDenied):
        security.authorize("Bearer tool-token-123456", "admin")  # a read token cannot change state
    security.authorize("Bearer admin-token-654321", "admin")
    security.authorize("Bearer admin-token-654321", "read")       # admin implies read


def test_admin_routes_are_closed_in_production_until_a_token_is_set(settings_env) -> None:
    settings_env(MARKETVISION_ENV="production")
    with pytest.raises(NotConfigured):
        security.authorize(None, "admin")
    settings_env(MARKETVISION_ENV="development")
    security.authorize(None, "admin")  # local development needs no setup


def test_sliding_window_limiter() -> None:
    limiter = security.SlidingWindowLimiter()
    assert all(limiter.check("k", 3, 60, now=t) == 0 for t in (0.0, 1.0, 2.0))
    assert limiter.check("k", 3, 60, now=3.0) == pytest.approx(57.0)
    assert limiter.check("other", 3, 60, now=3.0) == 0
    assert limiter.check("k", 3, 60, now=61.0) == 0  # the first call has aged out
