"""Live provider check: make one small real request to each provider and report what came back.

A key being present proves nothing; this module asks each provider for something and records
whether it answered, how long it took, what market time the answer describes, how fresh the
data layer judged it, and whether quality checks passed. It never returns or logs a credential:
every text field passes through ``redact`` before it leaves this module.

Used by ``scripts/check_providers.py`` and ``GET /api/v2/health/providers``. The language-model
check is opt-in because a hosted model may charge per request.
"""
from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Callable

from app.core.clock import iso, utcnow
from app.core.config import Settings, get_settings
from app.core.errors import MarketVisionError
from app.core.secrets import redact
from app.marketdata.cache import TTLCache
from app.marketdata.models import DataEnvelope
from app.marketdata.providers.base import Provider
from app.marketdata.service import MarketData

PROBE_SYMBOL = "AAPL"
PROBE_SERIES = "DGS10"  # 10-year Treasury yield, daily


def _single(provider: Provider) -> MarketData:
    """A data layer holding just ``provider``, with a private cache, so the answer is fresh."""
    return MarketData([provider], cache=TTLCache(), retries=0)


def _summarise(envelope: DataEnvelope, now) -> dict[str, Any]:
    market_ts = envelope.market_timestamp
    return {
        "provider_answered": envelope.provider,
        "freshness": envelope.freshness,
        "market_timestamp": iso(market_ts) if market_ts else None,
        "age_seconds": None if market_ts is None else round((now - market_ts).total_seconds(), 1),
        "quality": envelope.quality.status,
        "quality_issues": [issue.code for issue in envelope.quality.issues],
        "notes": list(envelope.notes),
    }


def _run(check: str, provider_name: str, configured: bool, reason: str, call: Callable[[], dict[str, Any]], clock) -> dict[str, Any]:
    row: dict[str, Any] = {"check": check, "provider": provider_name, "configured": configured}
    if not configured:
        return {**row, "outcome": "not_configured", "detail": redact(reason)}
    started = time.monotonic()
    try:
        result = call()
    except MarketVisionError as exc:
        attempts = (exc.details or {}).get("attempts") or []
        reasons = "; ".join(f"{a.get('provider')}: {a.get('outcome')} {a.get('reason', '')}".strip() for a in attempts)
        return {
            **row,
            "outcome": "failed",
            "latency_ms": round((time.monotonic() - started) * 1000),
            "error_code": exc.code,
            "detail": redact(reasons or exc.message),
        }
    except Exception as exc:  # noqa: BLE001 - a probe must report, never crash
        return {**row, "outcome": "failed", "latency_ms": round((time.monotonic() - started) * 1000), "error_code": type(exc).__name__, "detail": "unexpected error"}
    return {**row, "outcome": "ok", "latency_ms": round((time.monotonic() - started) * 1000), **result}


def probe_providers(
    *,
    settings: Settings | None = None,
    providers: dict[str, Provider] | None = None,
    symbol: str = PROBE_SYMBOL,
    include_llm: bool = False,
    clock=utcnow,
) -> dict[str, Any]:
    """Run every check and return a report. Spends at most one request per check."""
    settings = settings or get_settings()
    started_at = clock()
    if settings.offline:
        return {
            "tested_at": iso(started_at),
            "live": False,
            "summary": "MARKETVISION_OFFLINE=1: no provider was contacted.",
            "checks": [],
            "verified": [],
            "failed": [],
            "not_configured": [],
        }
    if providers is None:
        from app.marketdata.service import build_market_data

        providers = build_market_data(settings).providers

    def check_history(name: str) -> dict[str, Any]:
        now = clock()
        envelope = _single(providers[name]).history(symbol, start=(now - timedelta(days=10)).date(), use_cache=False)
        frame = envelope.data
        return {
            **_summarise(envelope, now),
            "sample": {"bars": int(len(frame)), "last_bar": str(frame.index[-1].date()) if len(frame) else None, "adjusted_close_available": "Adj Close" in frame.columns},
        }

    def check_quote(name: str) -> dict[str, Any]:
        now = clock()
        envelope = _single(providers[name]).quote(symbol, use_cache=False)
        return {**_summarise(envelope, now), "sample": {"price": envelope.data.get("price"), "market_state": envelope.data.get("market_state")}}

    def check_news(name: str) -> dict[str, Any]:
        now = clock()
        envelope = _single(providers[name]).news(symbol=symbol, limit=3)
        items = envelope.data or []
        ages = [i.get("age_hours") for i in items if i.get("age_hours") is not None]
        return {**_summarise(envelope, now), "sample": {"articles": len(items), "newest_age_hours": min(ages) if ages else None}}

    def check_economic(name: str) -> dict[str, Any]:
        now = clock()
        envelope = _single(providers[name]).economic_series(PROBE_SERIES, start=(now - timedelta(days=30)).date())
        valid = envelope.data.dropna()
        return {
            **_summarise(envelope, now),
            "sample": {
                "series": PROBE_SERIES,
                "observations": int(len(valid)),
                "last_observation": str(valid.index[-1].date()) if len(valid) else None,
                "missing_values": int(envelope.data.isna().sum()),
                "route": "json_api" if settings.fred_api_key else "public_csv",
            },
        }

    def availability(name: str) -> tuple[bool, str]:
        provider = providers.get(name)
        if provider is None:
            return False, f"{name} is not set up in this process."
        return provider.available()

    checks: list[dict[str, Any]] = []
    plan = [
        ("yahoo_history", "yahoo", check_history),
        ("yahoo_quote", "yahoo", check_quote),
        ("finnhub_quote", "finnhub", check_quote),
        ("finnhub_company_news", "finnhub", check_news),
        ("fred_series", "fred", check_economic),
        ("newsapi_news", "newsapi", check_news),
    ]
    for check, name, func in plan:
        ok, reason = availability(name)
        checks.append(_run(check, name, ok, reason, lambda func=func, name=name: func(name), clock))

    if include_llm:
        checks.append(_probe_llm(settings))

    verified = sorted({c["provider"] for c in checks if c["outcome"] == "ok"})
    failed = sorted({c["provider"] for c in checks if c["outcome"] == "failed"})
    missing = sorted({c["provider"] for c in checks if c["outcome"] == "not_configured"})
    return {
        "tested_at": iso(started_at),
        "live": True,
        "symbol": symbol,
        "summary": f"{len(verified)} provider(s) answered, {len(failed)} failed, {len(missing)} not configured.",
        "checks": checks,
        "verified": verified,
        "failed": failed,
        "not_configured": missing,
    }


def _probe_llm(settings: Settings) -> dict[str, Any]:
    """One tiny chat request to the configured model. May cost money on a hosted service."""
    from app.services.copilot_service import CopilotService

    row: dict[str, Any] = {"check": "llm_chat", "provider": "llm", "configured": bool(settings.llm_api_key or settings.llm_base_url)}
    service = CopilotService()
    if not service.is_available():
        return {**row, "outcome": "not_configured", "detail": redact(service.status_reason)}
    started = time.monotonic()
    answer = service.chat("Reply with the single word OK.")
    latency = round((time.monotonic() - started) * 1000)
    failed = answer.startswith("The language model could not answer")
    return {**row, "outcome": "failed" if failed else "ok", "latency_ms": latency, "model": service.model, "detail": redact(answer[:120])}
