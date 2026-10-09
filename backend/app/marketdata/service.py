"""The one door to market data.

Every other module asks ``MarketData`` and gets a ``DataEnvelope`` back. Providers are tried in
a fixed order per dataset; a failing provider is skipped (and rested after repeated failures);
if none can answer, ``DataUnavailable`` is raised. Nothing is ever made up to fill a gap.

Freshness labels
    live         the provider states the quote is real-time, the market is open, and the quote
                 is at most ``live_max_age_seconds`` old
    delayed      any other quote while the market is open, an unfinished daily bar, recent
                 intraday bars, and all news
    historical   completed bars, the last close while the market is shut, profiles, economic series
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from app.core.clock import utcnow
from app.core.config import Settings, get_settings
from app.core.errors import DataUnavailable, MarketVisionError, NotSupported, ProviderError, RateLimited, ValidationFailed
from app.marketdata import calendar as cal
from app.marketdata.cache import DiskFrameCache, TTLCache
from app.marketdata.http import HttpClient, TokenBucket
from app.marketdata.models import DataEnvelope, QualityReport
from app.marketdata.providers.base import Provider
from app.marketdata.quality import assess_ohlcv, assess_quote, normalise_ohlcv

log = logging.getLogger(__name__)

DEFAULT_ORDER: dict[str, list[str]] = {
    "history": ["yahoo"],
    "quote": ["finnhub", "yahoo"],
    "profile": ["yahoo", "finnhub"],
    "actions": ["yahoo"],
    "news": ["finnhub", "newsapi", "yahoo"],
    "economic": ["fred"],
    "earnings": ["finnhub"],
}

TTL = {
    "history_daily": 900.0,
    "history_intraday": 60.0,
    "history_disk": 7 * 86400.0,
    "quote": 10.0,
    "profile": 86400.0,
    "actions": 86400.0,
    "news": 600.0,
    "economic": 21600.0,
    "earnings": 21600.0,
}

_PERIOD_CALENDAR_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827, "10y": 3653}
VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo"}
_SYMBOL_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=")


def clean_symbol(symbol: str) -> str:
    text = (symbol or "").strip().upper()
    if not text or len(text) > 12 or any(ch not in _SYMBOL_CHARS for ch in text):
        raise ValidationFailed(f"{symbol!r} is not a valid ticker symbol.", details={"symbol": str(symbol)[:40]})
    return text


def period_start(period: str, today: date) -> date | None:
    """First date of a look-back period such as ``6mo``; ``None`` means all available history."""
    text = (period or "").strip().lower()
    if text in {"max", ""}:
        return None
    if text == "ytd":
        return date(today.year, 1, 1)
    if text in _PERIOD_CALENDAR_DAYS:
        return today - timedelta(days=_PERIOD_CALENDAR_DAYS[text])
    if text.endswith("d") and text[:-1].isdigit():
        return cal.shift_trading_days(today, -max(int(text[:-1]), 1))
    if text.endswith("y") and text[:-1].isdigit():
        return today - timedelta(days=366 * int(text[:-1]))
    if text.endswith("mo") and text[:-2].isdigit():
        return today - timedelta(days=31 * int(text[:-2]))
    raise ValidationFailed(f"{period!r} is not a recognised period.", details={"examples": ["5d", "1mo", "6mo", "1y", "5y", "max"]})


class MarketData:
    def __init__(
        self,
        providers: list[Provider],
        *,
        order: dict[str, list[str]] | None = None,
        cache: TTLCache | None = None,
        disk_cache: DiskFrameCache | None = None,
        clock: Callable[[], datetime] = utcnow,
        limiters: dict[str, TokenBucket] | None = None,
        breaker_threshold: int = 3,
        breaker_cooldown_seconds: float = 60.0,
        live_max_age_seconds: float = 60.0,
        stale_quote_seconds: float = 1200.0,
        retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.providers = {p.name: p for p in providers}
        self.order = {**DEFAULT_ORDER, **(order or {})}
        self.cache = cache if cache is not None else TTLCache()
        self.disk_cache = disk_cache
        self.clock = clock
        self.limiters = limiters or {}
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown_seconds = breaker_cooldown_seconds
        self.live_max_age_seconds = live_max_age_seconds
        self.stale_quote_seconds = stale_quote_seconds
        self.retries = retries
        self._sleep = sleep
        self._failures: dict[str, int] = {}
        self._rest_until: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ provider plumbing
    def _candidates(self, dataset: str) -> list[Provider]:
        named = [self.providers[n] for n in self.order.get(dataset, []) if n in self.providers]
        extra = [p for p in self.providers.values() if p not in named and dataset in p.supports]
        return [p for p in [*named, *extra] if dataset in p.supports]

    def _resting(self, name: str) -> bool:
        with self._lock:
            return time.monotonic() < self._rest_until.get(name, 0.0)

    def _record(self, name: str, ok: bool) -> None:
        with self._lock:
            if ok:
                self._failures[name] = 0
                return
            self._failures[name] = self._failures.get(name, 0) + 1
            if self._failures[name] >= self.breaker_threshold:
                self._rest_until[name] = time.monotonic() + self.breaker_cooldown_seconds
                self._failures[name] = 0
                log.warning("Provider %s is being rested for %.0fs after repeated failures.", name, self.breaker_cooldown_seconds)

    def _ask(self, dataset: str, call: Callable[[Provider], Any]) -> tuple[Provider, Any, list[dict[str, str]]]:
        """Try each provider in order. Returns the first answer and the list of attempts."""
        attempts: list[dict[str, str]] = []
        candidates = self._candidates(dataset)
        if not candidates:
            raise DataUnavailable(f"No data provider is configured for {dataset}.", details={"dataset": dataset, "attempts": attempts})
        for provider in candidates:
            ok, reason = provider.available()
            if not ok:
                attempts.append({"provider": provider.name, "outcome": "skipped", "reason": reason})
                continue
            if self._resting(provider.name):
                attempts.append({"provider": provider.name, "outcome": "skipped", "reason": "resting after repeated failures"})
                continue
            for attempt in range(self.retries + 1):
                try:
                    limiter = self.limiters.get(provider.name)
                    if limiter is not None:
                        limiter.acquire()
                    result = call(provider)
                except NotSupported as exc:
                    attempts.append({"provider": provider.name, "outcome": "not_supported", "reason": exc.message})
                    break
                except RateLimited as exc:
                    attempts.append({"provider": provider.name, "outcome": "rate_limited", "reason": exc.message})
                    self._record(provider.name, False)
                    break
                except ProviderError as exc:
                    definitive = exc.code in {"no_data", "not_configured", "provider_auth", "not_supported", "bad_request"}
                    if not definitive and attempt < self.retries:
                        self._sleep(0.5 * (2**attempt))
                        continue
                    attempts.append({"provider": provider.name, "outcome": exc.code, "reason": exc.message})
                    if not definitive:
                        self._record(provider.name, False)
                    break
                else:
                    self._record(provider.name, True)
                    attempts.append({"provider": provider.name, "outcome": "ok", "reason": ""})
                    return provider, result, attempts
        raise DataUnavailable(
            f"No provider could supply {dataset}. Nothing has been substituted.",
            details={"dataset": dataset, "attempts": attempts},
        )

    def _cached(self, key: str, ttl: float) -> DataEnvelope | None:
        hit = self.cache.get(key, ttl)
        if hit is None:
            return None
        envelope, age = hit
        return DataEnvelope(**{**envelope.__dict__, "cache_hit": True, "cache_age_seconds": round(age, 1)})

    # ------------------------------------------------------------------ datasets
    def history(
        self,
        symbol: str,
        *,
        period: str | None = None,
        start: date | None = None,
        end: date | None = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> DataEnvelope:
        """OHLCV bars, with dividends and splits when the provider has them. Prices are as traded."""
        symbol = clean_symbol(symbol)
        interval = (interval or "1d").lower()
        if interval not in VALID_INTERVALS:
            raise ValidationFailed(f"{interval!r} is not a supported interval.", details={"supported": sorted(VALID_INTERVALS)})
        now = self.clock()
        today = now.astimezone(cal.NEW_YORK).date()
        if start is None and period:
            start = period_start(period, today)
        if start and end and start > end:
            raise ValidationFailed("The start date is after the end date.")

        key = f"history|{symbol}|{start}|{end}|{interval}"
        ttl = TTL["history_daily"] if interval in {"1d", "1wk", "1mo"} else TTL["history_intraday"]
        if use_cache:
            cached = self._cached(key, ttl)
            if cached is not None:
                return cached
            fully_past = end is not None and end < cal.last_completed_session(now)
            if self.disk_cache is not None and interval == "1d" and fully_past:
                stored = self.disk_cache.get(key, TTL["history_disk"])
                if stored is not None:
                    frame, meta, age = stored
                    envelope = self._history_envelope(symbol, frame, interval, now, meta.get("provider", "cache"), [], end)
                    envelope.cache_hit, envelope.cache_age_seconds = True, round(age, 1)
                    return envelope

        provider, raw, attempts = self._ask("history", lambda p: p.history(symbol, start, end, interval))
        envelope = self._history_envelope(symbol, raw, interval, now, provider.name, attempts, end)
        if envelope.quality.status == "failed":
            raise DataUnavailable(
                f"{provider.name} returned unusable price history for {symbol}.",
                details={"symbol": symbol, "quality": envelope.quality.to_dict(), "attempts": attempts},
            )
        self.cache.set(key, envelope)
        if self.disk_cache is not None and interval == "1d" and end is not None and end < cal.last_completed_session(now):
            self.disk_cache.set(key, envelope.data, {"provider": provider.name, "symbol": symbol})
        return envelope

    def _history_envelope(self, symbol: str, raw: pd.DataFrame, interval: str, now: datetime, provider: str, attempts: list, end: date | None) -> DataEnvelope:
        frame, report = normalise_ohlcv(raw)
        historical_request = end is not None and end < cal.last_completed_session(now)
        report = assess_ohlcv(frame, interval=interval, now=now, report=report, check_staleness=not historical_request and not provider.startswith("snapshot"))
        market_ts: datetime | None = None
        freshness = "historical"
        notes: list[str] = []
        if len(frame):
            last = pd.Timestamp(frame.index[-1])
            if interval in {"1d", "1wk", "1mo"}:
                bounds = cal.session_bounds(last.date())
                market_ts = bounds[1] if bounds else datetime.combine(last.date(), datetime.min.time(), tzinfo=timezone.utc)
                state = cal.market_state(now)
                if interval == "1d" and state["is_open"] and state["session_date"] == last.date().isoformat():
                    freshness = "delayed"
                    market_ts = now
                    notes.append("The last bar is the current, unfinished session.")
            else:
                market_ts = last.to_pydatetime().replace(tzinfo=timezone.utc)
                if (now - market_ts).total_seconds() < 3600 and cal.market_state(now)["is_open"]:
                    freshness = "delayed"
        if provider == "yahoo":
            notes.append("Yahoo Finance data is unofficial and may be delayed.")
        return DataEnvelope(
            dataset="history", provider=provider, retrieved_at=now, freshness=freshness, data=frame, symbol=symbol,
            market_timestamp=market_ts, quality=report, attempts=attempts, notes=notes,
        )

    def quote(self, symbol: str, *, use_cache: bool = True) -> DataEnvelope:
        """Latest available price. Labelled live only when that is demonstrably true."""
        symbol = clean_symbol(symbol)
        key = f"quote|{symbol}"
        if use_cache:
            cached = self._cached(key, TTL["quote"])
            if cached is not None:
                return cached
        now = self.clock()
        provider, raw, attempts = self._ask("quote", lambda p: p.quote(symbol))
        price = _to_float(raw.get("price"))
        market_ts: datetime | None = raw.get("market_timestamp")
        state = cal.market_state(now)
        notes: list[str] = []
        if market_ts is None and raw.get("session_date"):
            bounds = cal.session_bounds(raw["session_date"])
            market_ts = bounds[1] if bounds else None
            notes.append("Price is a daily close; the time shown is that session's close.")
        report = assess_quote(price, market_ts, now=now, max_age_seconds=self.stale_quote_seconds)
        age = (now - market_ts).total_seconds() if market_ts else None
        if not state["is_open"]:
            freshness = "historical"
            notes.append("The market is closed; this is the most recent price available.")
        elif raw.get("realtime") and age is not None and 0 <= age <= self.live_max_age_seconds:
            freshness = "live"
        else:
            freshness = "delayed"
        previous = _to_float(raw.get("previous_close"))
        change = _to_float(raw.get("change"))
        change_pct = _to_float(raw.get("change_pct"))
        if price is not None and previous:
            change = price - previous if change is None else change
            change_pct = (price / previous - 1) * 100 if change_pct is None else change_pct
        data = {
            "symbol": symbol,
            "price": price,
            "previous_close": previous,
            "change": None if change is None else round(change, 4),
            "change_pct": None if change_pct is None else round(change_pct, 4),
            "open": _to_float(raw.get("open")),
            "high": _to_float(raw.get("high")),
            "low": _to_float(raw.get("low")),
            "volume": _to_float(raw.get("volume")),
            "basis": raw.get("basis"),
            "age_seconds": None if age is None else round(age, 1),
            "market_state": state["state"],
        }
        envelope = DataEnvelope(
            dataset="quote", provider=provider.name, retrieved_at=now, freshness=freshness, data=data, symbol=symbol,
            market_timestamp=market_ts, quality=report, attempts=attempts, notes=notes,
        )
        if report.status == "failed":
            raise DataUnavailable(f"{provider.name} returned no usable price for {symbol}.", details={"symbol": symbol, "attempts": attempts})
        self.cache.set(key, envelope)
        return envelope

    def profile(self, symbol: str) -> DataEnvelope:
        symbol = clean_symbol(symbol)
        return self._simple("profile", f"profile|{symbol}", TTL["profile"], lambda p: p.profile(symbol), symbol=symbol)

    def corporate_actions(self, symbol: str, *, start: date | None = None, end: date | None = None) -> DataEnvelope:
        symbol = clean_symbol(symbol)
        return self._simple("actions", f"actions|{symbol}|{start}|{end}", TTL["actions"], lambda p: p.corporate_actions(symbol, start, end), symbol=symbol)

    def news(self, *, symbol: str | None = None, query: str | None = None, limit: int = 10) -> DataEnvelope:
        symbol = clean_symbol(symbol) if symbol else None
        query = (query or "").strip()[:200] or None
        limit = max(1, min(int(limit), 50))
        envelope = self._simple(
            "news", f"news|{symbol}|{query}|{limit}", TTL["news"], lambda p: p.news(symbol, query, limit), symbol=symbol, freshness="delayed",
        )
        seen: set[str] = set()
        unique = []
        for item in envelope.data:
            title = (item.get("title") or "").strip().lower()
            if title and title not in seen:
                seen.add(title)
                unique.append(item)
        if len(unique) != len(envelope.data) and not envelope.quality.has("duplicate_articles"):
            envelope.quality.add("duplicate_articles", "info", "Articles with a repeated headline were dropped.", len(envelope.data) - len(unique))
        envelope.data = unique
        stamps = [pd.Timestamp(i["published_at"]) for i in unique if i.get("published_at")]
        if stamps:
            newest = max(s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC") for s in stamps)
            envelope.market_timestamp = newest.to_pydatetime()
        if envelope.provider == "newsapi" and "NewsAPI's free plan delays articles by 24 hours." not in envelope.notes:
            envelope.notes.append("NewsAPI's free plan delays articles by 24 hours.")
        return envelope

    def economic_series(self, series_id: str, *, start: date | None = None, end: date | None = None) -> DataEnvelope:
        series_id = (series_id or "").strip()
        if not series_id or len(series_id) > 40:
            raise ValidationFailed("A series id is required.")
        envelope = self._simple(
            "economic", f"economic|{series_id}|{start}|{end}", TTL["economic"], lambda p: p.economic_series(series_id, start, end), symbol=series_id,
        )
        series: pd.Series = envelope.data
        valid = series.dropna()
        if len(valid):
            envelope.market_timestamp = datetime.combine(valid.index[-1].date(), datetime.min.time(), tzinfo=timezone.utc)
        missing = int(series.isna().sum())
        if missing and not envelope.quality.has("missing_values"):
            envelope.quality.add("missing_values", "info", "Dates on which the series has no value.", missing)
        if "Economic series can be revised after first publication." not in envelope.notes:
            envelope.notes.append("Economic series can be revised after first publication.")
        return envelope

    def earnings_calendar(self, symbol: str, *, days_ahead: int = 90) -> DataEnvelope:
        symbol = clean_symbol(symbol)
        today = self.clock().date()
        horizon = today + timedelta(days=max(1, min(days_ahead, 365)))
        return self._simple("earnings", f"earnings|{symbol}|{today}|{horizon}", TTL["earnings"], lambda p: p.earnings_calendar(symbol, today, horizon), symbol=symbol)

    def market_calendar(self, start: date, end: date) -> DataEnvelope:
        if end < start:
            raise ValidationFailed("The start date is after the end date.")
        if (end - start).days > 3700:
            raise ValidationFailed("A calendar request can cover at most ten years.")
        days = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                bounds = cal.session_bounds(d)
                days.append(
                    {
                        "date": d.isoformat(),
                        "is_trading_day": bounds is not None,
                        "holiday": cal.holiday_name(d),
                        "open": bounds[0].isoformat().replace("+00:00", "Z") if bounds else None,
                        "close": bounds[1].isoformat().replace("+00:00", "Z") if bounds else None,
                        "early_close": cal.is_early_close(d),
                    }
                )
            d += timedelta(days=1)
        now = self.clock()
        return DataEnvelope(
            dataset="calendar", provider="computed:nyse_rules", retrieved_at=now, freshness="historical",
            data={"days": days, "now": cal.market_state(now)}, market_timestamp=now,
            notes=["Computed from NYSE holiday rules. Unscheduled closures announced after this software was written are not included."],
        )

    def _simple(self, dataset: str, key: str, ttl: float, call: Callable[[Provider], Any], *, symbol: str | None, freshness: str = "historical") -> DataEnvelope:
        cached = self._cached(key, ttl)
        if cached is not None:
            return cached
        now = self.clock()
        provider, data, attempts = self._ask(dataset, call)
        envelope = DataEnvelope(dataset=dataset, provider=provider.name, retrieved_at=now, freshness=freshness, data=data, symbol=symbol, attempts=attempts, quality=QualityReport())
        self.cache.set(key, envelope)
        return envelope

    def status(self) -> dict[str, Any]:
        """Which providers are usable right now. No secrets, no network calls."""
        rows = []
        for name, provider in self.providers.items():
            ok, reason = provider.available()
            rows.append({"provider": name, "available": ok and not self._resting(name), "reason": reason or ("resting after repeated failures" if self._resting(name) else ""), "supports": sorted(provider.supports)})
        return {"providers": rows, "order": {k: [n for n in v if n in self.providers] for k, v in self.order.items()}, "market": cal.market_state(self.clock())}


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def adjusted_close(frame: pd.DataFrame) -> pd.Series:
    """Close adjusted for splits and dividends when the provider supplies it, else the raw close."""
    if "Adj Close" in frame.columns and frame["Adj Close"].notna().any():
        return frame["Adj Close"]
    return frame["Close"]


# ---------------------------------------------------------------------- default instance
_instance: MarketData | None = None
_instance_lock = threading.Lock()


def build_market_data(settings: Settings | None = None) -> MarketData:
    """Assemble the providers this process is configured for."""
    import os

    from app.marketdata.providers.finnhub import FinnhubProvider
    from app.marketdata.providers.fred import FredProvider
    from app.marketdata.providers.newsapi import NewsApiProvider
    from app.marketdata.providers.yahoo import YahooProvider
    from app.marketdata.snapshots import SnapshotProvider, SnapshotStore

    settings = settings or get_settings()
    providers: list[Provider] = []
    limiters: dict[str, TokenBucket] = {}
    order = dict(DEFAULT_ORDER)

    def client(name: str, rate: float, capacity: float) -> HttpClient:
        return HttpClient(name, timeout=settings.http_timeout_seconds, max_retries=settings.http_max_retries, rate_limiter=TokenBucket(rate, capacity))

    if not settings.offline:
        providers.append(YahooProvider())
        limiters["yahoo"] = TokenBucket(2.0, 8.0)
        providers.append(FinnhubProvider(settings.finnhub_api_key, client("finnhub", 1.0, 30.0)))
        providers.append(NewsApiProvider(settings.newsapi_key, client("newsapi", 100.0 / 86400.0, 20.0)))
        providers.append(FredProvider(settings.fred_api_key, client("fred", 2.0, 10.0)))

    snapshot_id = os.getenv("MARKETVISION_SNAPSHOT_ID", "").strip()
    if snapshot_id:
        try:
            snapshot = SnapshotProvider(SnapshotStore(settings.snapshot_dir), snapshot_id)
        except MarketVisionError as exc:
            log.warning("Snapshot %s could not be loaded: %s", snapshot_id, exc.message)
        else:
            providers.append(snapshot)
            order["history"] = [*order["history"], snapshot.name]
            order["economic"] = [*order["economic"], snapshot.name]

    return MarketData(providers, order=order, disk_cache=DiskFrameCache(settings.cache_dir), limiters=limiters)


def get_market_data() -> MarketData:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = build_market_data()
        return _instance


def set_market_data(instance: MarketData | None) -> None:
    """Replace the default instance (tests, scripts). ``None`` rebuilds it on next use."""
    global _instance
    with _instance_lock:
        _instance = instance
