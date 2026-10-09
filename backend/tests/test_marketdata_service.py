"""The unified data layer, exercised with stand-in providers and a stand-in HTTP session.

No request leaves the machine. Provider payloads here follow the shapes documented by Yahoo
(yfinance), Finnhub, NewsAPI and FRED; a run against the live services is a separate, manual check.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import pytest
import requests

from app.core.errors import DataUnavailable, ProviderError, RateLimited, ValidationFailed
from app.core.secrets import Secret
from app.marketdata.cache import DiskFrameCache, TTLCache
from app.marketdata.http import HttpClient, TokenBucket
from app.marketdata.providers.base import Provider
from app.marketdata.providers.finnhub import FinnhubProvider
from app.marketdata.providers.fred import FredProvider
from app.marketdata.providers.newsapi import NewsApiProvider
from app.marketdata.providers.yahoo import YahooProvider
from app.marketdata.service import MarketData, clean_symbol, period_start
from app.marketdata.snapshots import SnapshotProvider, SnapshotStore
from tests.helpers import make_ohlcv

OPEN = datetime(2024, 7, 10, 15, 0, tzinfo=timezone.utc)    # Wednesday, mid-session
CLOSED = datetime(2024, 7, 10, 21, 0, tzinfo=timezone.utc)  # same day, after the close


def history_to(day: str) -> pd.DataFrame:
    frame = make_ohlcv(220, start="2024-01-02")
    return frame[frame.index <= pd.Timestamp(day)]


class FakeProvider(Provider):
    def __init__(self, name: str, supports: set[str], *, fail: ProviderError | None = None, history: pd.DataFrame | None = None, quote: dict | None = None, news: list | None = None) -> None:
        self.name = name
        self.supports = frozenset(supports)
        self.fail = fail
        self._history, self._quote, self._news = history, quote, news
        self.calls = 0

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.fail is not None:
            raise self.fail

    def history(self, symbol, start, end, interval):
        self._maybe_fail()
        frame = self._history
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]
        return frame

    def quote(self, symbol):
        self._maybe_fail()
        return dict(self._quote)

    def news(self, symbol, query, limit):
        self._maybe_fail()
        return [dict(item) for item in self._news]


def service(providers, now=CLOSED, **kwargs) -> MarketData:
    return MarketData(providers, clock=lambda: now, sleep=lambda s: None, **kwargs)


# ---------------------------------------------------------------- envelopes and freshness
def test_history_envelope_names_source_times_and_quality() -> None:
    md = service([FakeProvider("yahoo", {"history"}, history=history_to("2024-07-10"))])
    env = md.history("aapl", period="3mo")
    meta = env.meta()
    assert meta["symbol"] == "AAPL" and meta["provider"] == "yahoo" and meta["dataset"] == "history"
    assert meta["freshness"] == "historical"
    assert meta["retrieved_at"] == "2024-07-10T21:00:00Z"
    assert meta["market_timestamp"] == "2024-07-10T20:00:00Z"  # that session's close
    assert meta["quality"]["status"] == "ok"
    assert env.data.index[0] >= pd.Timestamp("2024-04-09")
    payload = env.to_dict()
    assert set(payload["data"][0]) == {"date", "open", "high", "low", "close", "volume"}


def test_unfinished_daily_bar_is_labelled_delayed() -> None:
    md = service([FakeProvider("yahoo", {"history"}, history=history_to("2024-07-10"))], now=OPEN)
    env = md.history("AAPL", period="1mo")
    assert env.freshness == "delayed"
    assert any("unfinished" in note for note in env.notes)


def test_stale_history_is_reported_in_the_envelope() -> None:
    md = service([FakeProvider("yahoo", {"history"}, history=history_to("2024-07-03"))])
    env = md.history("AAPL", period="1mo")
    assert env.quality.status == "degraded" and env.quality.has("stale")


def test_quote_from_a_delayed_source_is_never_labelled_live() -> None:
    quote = {"price": 190.5, "market_timestamp": datetime(2024, 7, 10, 14, 59, 50, tzinfo=timezone.utc), "realtime": False, "previous_close": 188.0}
    env = service([FakeProvider("yahoo", {"quote"}, quote=quote)], now=OPEN).quote("AAPL")
    assert env.freshness == "delayed"
    assert env.data["change"] == pytest.approx(2.5) and env.data["change_pct"] == pytest.approx(1.3298, abs=1e-3)


def test_realtime_quote_is_live_only_when_recent_and_market_open() -> None:
    def quote_at(ts):
        return {"price": 190.5, "market_timestamp": ts, "realtime": True}

    recent = service([FakeProvider("finnhub", {"quote"}, quote=quote_at(datetime(2024, 7, 10, 14, 59, 50, tzinfo=timezone.utc)))], now=OPEN).quote("AAPL")
    assert recent.freshness == "live" and recent.data["age_seconds"] == 10.0
    old = service([FakeProvider("finnhub", {"quote"}, quote=quote_at(datetime(2024, 7, 10, 14, 40, 0, tzinfo=timezone.utc)))], now=OPEN).quote("AAPL")
    assert old.freshness == "delayed"
    shut = service([FakeProvider("finnhub", {"quote"}, quote=quote_at(datetime(2024, 7, 10, 20, 0, 0, tzinfo=timezone.utc)))], now=CLOSED).quote("AAPL")
    assert shut.freshness == "historical" and shut.data["market_state"] == "after_close"


def test_stale_quote_during_market_hours_is_flagged() -> None:
    quote = {"price": 190.5, "market_timestamp": datetime(2024, 7, 10, 14, 0, tzinfo=timezone.utc), "realtime": True}
    env = service([FakeProvider("finnhub", {"quote"}, quote=quote)], now=OPEN).quote("AAPL")
    assert env.quality.has("stale") and env.freshness == "delayed"


def test_quote_without_a_price_is_refused_not_invented() -> None:
    with pytest.raises(DataUnavailable):
        service([FakeProvider("yahoo", {"quote"}, quote={"price": None, "market_timestamp": None, "realtime": False})]).quote("AAPL")


# ---------------------------------------------------------------- fallback, retries, resting
def test_falls_back_to_the_next_provider_and_records_attempts() -> None:
    broken = FakeProvider("finnhub", {"quote"}, fail=ProviderError("finnhub down", details={}))
    working = FakeProvider("yahoo", {"quote"}, quote={"price": 10.0, "market_timestamp": OPEN, "realtime": False})
    env = service([broken, working], now=OPEN).quote("AAPL")
    assert env.provider == "yahoo"
    assert [a["outcome"] for a in env.attempts] == ["provider_error", "ok"]
    assert broken.calls == 2  # one retry before giving up on it


def test_all_providers_failing_raises_data_unavailable_with_attempts() -> None:
    a = FakeProvider("finnhub", {"quote"}, fail=ProviderError("down"))
    b = FakeProvider("yahoo", {"quote"}, fail=ProviderError("no data", code="no_data"))
    with pytest.raises(DataUnavailable) as raised:
        service([a, b]).quote("AAPL")
    attempts = raised.value.details["attempts"]
    assert [x["provider"] for x in attempts] == ["finnhub", "yahoo"]
    assert b.calls == 1  # "no data" is definitive: not retried


def test_repeatedly_failing_provider_is_rested() -> None:
    broken = FakeProvider("finnhub", {"quote"}, fail=ProviderError("down"))
    working = FakeProvider("yahoo", {"quote"}, quote={"price": 10.0, "market_timestamp": OPEN, "realtime": False})
    md = service([broken, working], now=OPEN, breaker_threshold=2, retries=0)
    for _ in range(2):
        md.quote("AAPL", use_cache=False)
    calls_before = broken.calls
    env = md.quote("AAPL", use_cache=False)
    assert broken.calls == calls_before  # not asked again while resting
    assert env.attempts[0]["outcome"] == "skipped"
    status = {row["provider"]: row for row in md.status()["providers"]}
    assert status["finnhub"]["available"] is False


def test_unconfigured_provider_is_skipped_with_a_reason() -> None:
    finnhub = FinnhubProvider(Secret(""), HttpClient("finnhub"))
    working = FakeProvider("yahoo", {"quote"}, quote={"price": 10.0, "market_timestamp": OPEN, "realtime": False})
    env = service([finnhub, working], now=OPEN).quote("AAPL")
    assert env.attempts[0] == {"provider": "finnhub", "outcome": "skipped", "reason": "FINNHUB_API_KEY is not set."}


def test_memory_cache_serves_repeat_requests_and_reports_age() -> None:
    provider = FakeProvider("yahoo", {"history"}, history=history_to("2024-07-10"))
    md = service([provider])
    first = md.history("AAPL", period="1mo")
    second = md.history("AAPL", period="1mo")
    assert provider.calls == 1
    assert first.cache_hit is False and second.cache_hit is True and second.cache_age_seconds is not None


def test_disk_cache_serves_fully_historical_ranges(tmp_path) -> None:
    provider = FakeProvider("yahoo", {"history"}, history=history_to("2024-07-10"))
    disk = DiskFrameCache(tmp_path)
    first = service([provider], disk_cache=disk).history("AAPL", start=date(2024, 3, 1), end=date(2024, 5, 31))
    again = service([provider], disk_cache=disk).history("AAPL", start=date(2024, 3, 1), end=date(2024, 5, 31))
    assert provider.calls == 1 and again.cache_hit is True
    pd.testing.assert_frame_equal(first.data, again.data, check_freq=False, check_names=False)
    assert again.quality.status == "ok"  # an old range is not "stale"


def test_news_is_delayed_deduplicated_and_timestamped() -> None:
    items = [
        {"title": "Chip demand rises", "summary": None, "publisher": "A", "link": "https://example.com/1", "published_at": "2024-07-09T12:00:00Z"},
        {"title": "chip demand rises ", "summary": None, "publisher": "B", "link": "https://example.com/2", "published_at": "2024-07-09T13:00:00Z"},
        {"title": "Rates unchanged", "summary": None, "publisher": "A", "link": "https://example.com/3", "published_at": "2024-07-10T09:00:00Z"},
    ]
    env = service([FakeProvider("newsapi", {"news"}, news=items)]).news(symbol="NVDA", limit=10)
    assert env.freshness == "delayed" and len(env.data) == 2
    assert env.meta()["market_timestamp"] == "2024-07-10T09:00:00Z"
    assert any("24 hours" in note for note in env.notes)


def test_market_calendar_dataset() -> None:
    env = service([]).market_calendar(date(2024, 11, 25), date(2024, 11, 29))
    by_date = {row["date"]: row for row in env.data["days"]}
    assert by_date["2024-11-28"]["is_trading_day"] is False and by_date["2024-11-28"]["holiday"] == "Thanksgiving Day"
    assert by_date["2024-11-29"]["early_close"] is True
    assert env.provider == "computed:nyse_rules"


def test_input_validation() -> None:
    assert clean_symbol(" brk-b ") == "BRK-B"
    for bad in ["", "AAPL; DROP TABLE", "A" * 13, "../etc"]:
        with pytest.raises(ValidationFailed):
            clean_symbol(bad)
    assert period_start("ytd", date(2024, 7, 10)) == date(2024, 1, 1)
    assert period_start("5d", date(2024, 7, 10)) == date(2024, 7, 2)   # five sessions back, skipping 4 July
    assert period_start("max", date(2024, 7, 10)) is None
    with pytest.raises(ValidationFailed):
        period_start("soon", date(2024, 7, 10))
    md = service([FakeProvider("yahoo", {"history"}, history=history_to("2024-07-10"))])
    with pytest.raises(ValidationFailed):
        md.history("AAPL", interval="7m")
    with pytest.raises(ValidationFailed):
        md.history("AAPL", start=date(2024, 5, 1), end=date(2024, 4, 1))


# ---------------------------------------------------------------- HTTP client
class FakeResponse:
    def __init__(self, status: int, payload: Any = None, headers: dict | None = None, text: str = "") -> None:
        self.status_code, self._payload, self.headers, self.text = status, payload, headers or {}, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.headers: dict = {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.requests.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_http_client_retries_then_succeeds_with_backoff() -> None:
    sleeps: list[float] = []
    session = FakeSession([requests.Timeout("slow"), FakeResponse(503), FakeResponse(200, {"ok": True})])
    client = HttpClient("demo", max_retries=2, session=session, sleep=sleeps.append, timeout=3.0)
    assert client.get("https://api.example.com/v1/thing", params={"q": 1}) == {"ok": True}
    assert len(session.requests) == 3 and session.requests[0]["timeout"] == 3.0
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] > 0


def test_http_client_honours_retry_after_and_raises_rate_limited() -> None:
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "2"}), FakeResponse(429)])
    client = HttpClient("demo", max_retries=1, session=session, sleep=sleeps.append)
    with pytest.raises(RateLimited):
        client.get("https://api.example.com/v1/thing")
    assert sleeps == [2.0]


def test_http_client_does_not_retry_client_errors() -> None:
    session = FakeSession([FakeResponse(404)])
    with pytest.raises(ProviderError) as raised:
        HttpClient("demo", max_retries=3, session=session, sleep=lambda s: None).get("https://api.example.com/v1/missing")
    assert len(session.requests) == 1 and raised.value.details["status"] == 404


def test_http_errors_never_contain_the_key_or_query_string() -> None:
    secret = "sk_live_ABCDEF123456"  # not-a-secret (test fixture)
    leaky = requests.ConnectionError(f"Max retries exceeded with url: /fred/series?api_key={secret}")
    session = FakeSession([leaky, leaky])
    with pytest.raises(ProviderError) as raised:
        HttpClient("fred", max_retries=1, session=session, sleep=lambda s: None).get("https://api.example.com/fred/series", params={"api_key": secret})
    text = raised.value.message + str(raised.value.details)
    assert secret not in text and "api_key" not in text
    assert raised.value.__cause__ is None  # the original exception, which holds the URL, is not chained

    denied = FakeSession([FakeResponse(401)])
    with pytest.raises(ProviderError) as raised:
        HttpClient("fred", session=denied).get("https://api.example.com/fred/series", params={"api_key": secret})
    assert raised.value.code == "provider_auth" and secret not in raised.value.message


def test_token_bucket_waits_then_refuses_when_budget_is_gone() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    bucket = TokenBucket(rate=1.0, capacity=2.0, clock=lambda: now[0], sleep=sleep)
    assert bucket.acquire() == 0.0 and bucket.acquire() == 0.0
    assert bucket.acquire() == pytest.approx(1.0)  # waited for one token
    slow = TokenBucket(rate=0.01, capacity=1.0, clock=lambda: now[0], sleep=sleep)
    slow.acquire()
    with pytest.raises(RateLimited):
        slow.acquire(max_wait=5.0)


def test_ttl_cache_expires_and_evicts() -> None:
    now = [100.0]
    cache = TTLCache(maxsize=2, clock=lambda: now[0])
    cache.set("a", 1)
    now[0] += 5
    assert cache.get("a", ttl=10) == (1, 5.0)
    assert cache.get("a", ttl=2) is None
    cache.set("b", 2), cache.set("c", 3), cache.set("d", 4)
    assert cache.get("b", ttl=100) is None and cache.get("d", ttl=100) is not None


# ---------------------------------------------------------------- providers against documented shapes
def test_finnhub_sends_the_key_in_a_header_not_the_url() -> None:
    session = FakeSession([FakeResponse(200, {"c": 261.74, "d": 2.1, "dp": 0.81, "h": 263.3, "l": 260.7, "o": 261.1, "pc": 259.6, "t": 1720623590})])
    provider = FinnhubProvider(Secret("fh_test_key_123456"), HttpClient("finnhub", session=session))
    quote = provider.quote("aapl")
    sent = session.requests[0]
    assert sent["headers"] == {"X-Finnhub-Token": "fh_test_key_123456"}
    assert sent["params"] == {"symbol": "AAPL"} and "token" not in sent["url"]
    assert quote["price"] == 261.74 and quote["realtime"] is True
    assert quote["market_timestamp"] == datetime.fromtimestamp(1720623590, tz=timezone.utc)


def test_finnhub_empty_quote_is_no_data() -> None:
    session = FakeSession([FakeResponse(200, {"c": 0, "d": None, "dp": None, "h": 0, "l": 0, "o": 0, "pc": 0, "t": 0})])
    with pytest.raises(ProviderError) as raised:
        FinnhubProvider(Secret("fh_test_key_123456"), HttpClient("finnhub", session=session)).quote("ZZZZ")
    assert raised.value.code == "no_data"


def test_newsapi_sends_the_key_in_a_header_and_filters_removed_articles() -> None:
    payload = {"status": "ok", "articles": [
        {"title": "[Removed]", "source": {"name": "x"}},
        {"title": "Fed holds rates", "description": "Summary", "source": {"name": "Wire"}, "url": "https://example.com/a", "publishedAt": "2024-07-09T12:00:00Z"},
    ]}
    session = FakeSession([FakeResponse(200, payload)])
    items = NewsApiProvider(Secret("news_test_key_123456"), HttpClient("newsapi", session=session)).news(None, "fed", 5)
    sent = session.requests[0]
    assert sent["headers"] == {"X-Api-Key": "news_test_key_123456"} and "apiKey" not in (sent["params"] or {})
    assert [i["title"] for i in items] == ["Fed holds rates"] and items[0]["publisher"] == "Wire"


def test_fred_api_and_public_csv_paths() -> None:
    api = FakeSession([FakeResponse(200, {"observations": [{"date": "2024-07-08", "value": "4.28"}, {"date": "2024-07-09", "value": "."}, {"date": "2024-07-10", "value": "4.30"}]})])
    series = FredProvider(Secret("fred_test_key_123456"), HttpClient("fred", session=api)).economic_series("dgs10", date(2024, 7, 1), None)
    assert series.name == "DGS10" and series.iloc[0] == 4.28 and pd.isna(series.iloc[1]) and series.iloc[2] == 4.30
    assert api.requests[0]["params"]["observation_start"] == "2024-07-01"

    csv = FakeSession([FakeResponse(200, text="observation_date,VIXCLS\n2024-07-09,12.51\n2024-07-10,\n")])
    keyless = FredProvider(Secret(""), HttpClient("fred", session=csv)).economic_series("VIXCLS", None, None)
    assert csv.requests[0]["url"].endswith("fredgraph.csv") and csv.requests[0]["params"] == {"id": "VIXCLS"}
    assert keyless.iloc[0] == 12.51 and pd.isna(keyless.iloc[1])

    with pytest.raises(ProviderError):
        FredProvider(Secret(""), HttpClient("fred", session=FakeSession([]))).economic_series("bad id!", None, None)


class FakeYahooTicker:
    def __init__(self, symbol: str, daily: pd.DataFrame, minute: pd.DataFrame | None) -> None:
        self.symbol, self._daily, self._minute = symbol, daily, minute
        self.calls: list[dict] = []
        self.info = {"longName": "Example Corp", "sector": "Technology", "marketCap": 1.5e12, "currency": "USD"}
        self.news = [
            {"content": {"title": "New format headline", "summary": "s", "pubDate": "2024-07-10T12:00:00Z", "provider": {"displayName": "Wire"}, "canonicalUrl": {"url": "https://example.com/n"}}},
            {"title": "Old format headline", "publisher": "Desk", "link": "https://example.com/o", "providerPublishTime": 1720612800},
        ]

    def history(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("interval") == "1m":
            if self._minute is None:
                raise RuntimeError("no intraday data")
            return self._minute
        return self._daily

    @property
    def actions(self):
        index = pd.DatetimeIndex(["2024-02-09", "2024-05-10", "2024-06-10"], tz="America/New_York")
        return pd.DataFrame({"Dividends": [0.24, 0.25, 0.0], "Stock Splits": [0.0, 0.0, 4.0]}, index=index)


class FakeYahoo:
    def __init__(self, daily, minute=None):
        self.daily, self.minute = daily, minute
        self.tickers: list[FakeYahooTicker] = []

    def Ticker(self, symbol):  # noqa: N802 - mirrors yfinance
        ticker = FakeYahooTicker(symbol, self.daily, self.minute)
        self.tickers.append(ticker)
        return ticker


def test_yahoo_history_request_and_exclusive_end_date() -> None:
    daily = history_to("2024-07-10")
    daily.index = daily.index.tz_localize("America/New_York")
    fake = FakeYahoo(daily)
    frame = YahooProvider(fake).history("aapl", date(2024, 6, 3), date(2024, 6, 28), "1d")
    call = fake.tickers[0].calls[0]
    assert call["start"] == "2024-06-03" and call["end"] == "2024-06-29"  # yfinance excludes the end date
    assert call["auto_adjust"] is False and call["actions"] is True
    assert len(frame) == len(daily)


def test_yahoo_quote_uses_last_minute_bar_and_previous_close() -> None:
    daily = history_to("2024-07-10")
    daily.index = daily.index.tz_localize("America/New_York")
    minutes = pd.date_range("2024-07-10 09:30", periods=3, freq="1min", tz="America/New_York")
    minute = pd.DataFrame({"Open": [10, 11, 12.0], "High": [11, 12, 13.0], "Low": [9, 10, 11.0], "Close": [10.5, 11.5, 12.5], "Volume": [100, 200, 300.0]}, index=minutes)
    quote = YahooProvider(FakeYahoo(daily, minute)).quote("AAPL")
    assert quote["price"] == 12.5 and quote["realtime"] is False
    assert quote["market_timestamp"] == datetime(2024, 7, 10, 13, 32, tzinfo=timezone.utc)
    assert quote["previous_close"] == pytest.approx(float(daily["Close"].iloc[-2]))
    assert (quote["open"], quote["high"], quote["low"], quote["volume"]) == (10.0, 13.0, 9.0, 600.0)


def test_yahoo_quote_falls_back_to_daily_close_and_service_times_it() -> None:
    daily = history_to("2024-07-10")
    provider = YahooProvider(FakeYahoo(daily, None))
    raw = provider.quote("AAPL")
    assert raw["basis"] == "last daily close" and raw["session_date"] == "2024-07-10" and raw["market_timestamp"] is None
    env = service([provider], now=CLOSED).quote("AAPL")
    assert env.meta()["market_timestamp"] == "2024-07-10T20:00:00Z" and env.freshness == "historical"


def test_yahoo_profile_actions_and_both_news_formats() -> None:
    provider = YahooProvider(FakeYahoo(history_to("2024-07-10")))
    assert provider.profile("AAPL")["name"] == "Example Corp"
    actions = provider.corporate_actions("AAPL", date(2024, 5, 1), None)
    assert list(actions["Stock Splits"]) == [0.0, 4.0] and actions.index.tz is None
    news = provider.news("AAPL", None, 5)
    assert [n["title"] for n in news] == ["New format headline", "Old format headline"]
    assert news[0]["publisher"] == "Wire" and news[1]["link"] == "https://example.com/o"


def test_yahoo_without_the_library_reports_unavailable() -> None:
    class Missing(YahooProvider):
        def _module(self):
            raise ProviderError("The yfinance package is not installed.")

    ok, reason = Missing().available()
    assert ok is False and "yfinance" in reason


# ---------------------------------------------------------------- snapshots
def test_snapshot_roundtrip_identity_and_tamper_detection(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    frames = {"AAA": make_ohlcv(120, seed=1), "BBB": make_ohlcv(120, seed=2)}
    vix = pd.Series(range(120), index=frames["AAA"].index, dtype=float)
    manifest = store.create("unit", frames, {"VIX": vix}, sources=[{"name": "synthetic", "url": None}], description="test")
    again = store.create("unit", frames, {"VIX": vix})
    assert again["id"] == manifest["id"]  # same contents, same id
    changed = store.create("unit", {**frames, "BBB": frames["BBB"] * 1.01})
    assert changed["id"] != manifest["id"]

    assert store.verify(manifest["id"]) == []
    loaded = store.load_ohlcv(manifest["id"], "AAA")
    pd.testing.assert_frame_equal(loaded, frames["AAA"].round(6), check_freq=False, check_names=False, check_exact=False, rtol=1e-9)

    target = store.path(manifest["id"]) / "ohlcv" / "AAA.csv"
    target.write_text(target.read_text().replace("\n", "\n", 1) + "2099-01-01,1,1,1,1,1\n")
    assert store.verify(manifest["id"]) == ["ohlcv/AAA.csv"]


def test_snapshot_provider_serves_history_as_historical(tmp_path) -> None:
    store = SnapshotStore(tmp_path)
    manifest = store.create("unit", {"AAA": history_to("2024-07-10")}, {"VIX": pd.Series([12.0, 13.0], index=pd.to_datetime(["2024-07-09", "2024-07-10"]))})
    provider = SnapshotProvider(store, manifest["id"])
    md = service([provider], order={"history": [provider.name], "economic": [provider.name]}, now=datetime(2026, 1, 5, 21, 0, tzinfo=timezone.utc))
    env = md.history("AAA", start=date(2024, 6, 1), end=date(2024, 6, 28))
    assert env.freshness == "historical" and env.provider.startswith("snapshot:") and not env.quality.has("stale")
    assert md.economic_series("VIX").data.iloc[-1] == 13.0
    with pytest.raises(DataUnavailable):
        md.history("ZZZ", period="1mo")
    with pytest.raises(ValidationFailed):
        store.path("../escape")
