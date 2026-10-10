"""Provider integrations: credentials, shared budgets, caching, honest labels, the live probe.

Everything here runs against stand-in HTTP sessions and stand-in providers. These tests show the
code sends credentials the right way and reports outcomes honestly; they do not show that any
real provider works. That is what ``scripts/check_providers.py`` is for, on a networked machine.
"""
from __future__ import annotations

import json
import logging
import sys
import types
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.core.errors import ProviderError
from app.marketdata.clients import BUDGETS, shared_client
from app.marketdata.http import TokenBucket
from app.marketdata.providers.base import Provider
from app.marketdata.service import MarketData, set_market_data
from tests.test_marketdata_service import FakeProvider, FakeResponse, FakeSession

FINNHUB_KEY = "fh_integration_key_7f3a9c"
NEWS_KEY = "na_integration_key_51be20"


def _install_session(provider: str, outcomes: list) -> FakeSession:
    session = FakeSession(outcomes)
    client = shared_client(provider)
    client._session = session
    client._sleep = lambda s: None
    return session


# ---------------------------------------------------------------- budgets
@pytest.mark.parametrize("provider,limit,window", [("finnhub", 60, 60.0), ("fred", 120, 60.0), ("newsapi", 100, 86400.0)])
def test_budgets_stay_inside_published_free_limits_even_from_a_cold_start(provider, limit, window) -> None:
    rate, burst = BUDGETS[provider]
    assert burst + rate * window <= limit
    clock = [0.0]
    bucket = TokenBucket(rate, burst, clock=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    taken = 0
    while True:
        bucket.acquire(max_wait=window)
        if clock[0] > window:
            break
        taken += 1
    assert taken <= limit


def test_v2_layer_and_old_services_share_one_client_per_provider(settings_env) -> None:
    settings_env(MARKETVISION_OFFLINE="0", FINNHUB_API_KEY=FINNHUB_KEY)
    from app.marketdata.service import build_market_data

    md = build_market_data()
    assert md.providers["finnhub"]._http is shared_client("finnhub")
    assert md.providers["newsapi"]._http is shared_client("newsapi")
    assert md.providers["fred"]._http is shared_client("fred")


# ---------------------------------------------------------------- Finnhub (older dashboard routes)
def test_old_finnhub_service_sends_the_key_in_a_header_and_caches(settings_env) -> None:
    settings_env(MARKETVISION_OFFLINE="0", FINNHUB_API_KEY=FINNHUB_KEY)
    from app.services.finnhub_service import FinnhubService

    session = _install_session("finnhub", [FakeResponse(200, {"c": 190.5, "d": 1.0, "dp": 0.5, "h": 191, "l": 189, "o": 190, "pc": 189.5})])
    service = FinnhubService()
    assert service.is_available()
    first = service.get_quote("aapl")
    second = service.get_quote("AAPL")
    assert first == second and first["current_price"] == 190.5
    assert len(session.requests) == 1  # the second call was served from the cache
    sent = session.requests[0]
    assert sent["headers"] == {"X-Finnhub-Token": FINNHUB_KEY}
    assert "token" not in (sent["params"] or {})
    assert FINNHUB_KEY not in sent["url"] and FINNHUB_KEY not in json.dumps(sent["params"])


def test_old_finnhub_service_failure_returns_none_and_logs_no_key(settings_env, caplog) -> None:
    settings_env(MARKETVISION_OFFLINE="0", FINNHUB_API_KEY=FINNHUB_KEY)
    from app.services.finnhub_service import FinnhubService

    _install_session("finnhub", [FakeResponse(401)])
    with caplog.at_level(logging.WARNING):
        assert FinnhubService().get_company_profile("AAPL") is None
    assert "refused" in caplog.text and FINNHUB_KEY not in caplog.text


def test_old_finnhub_service_is_off_without_a_key_or_offline(settings_env) -> None:
    from app.services.finnhub_service import FinnhubService

    settings_env(MARKETVISION_OFFLINE="0", FINNHUB_API_KEY=None)
    assert not FinnhubService().is_available() and FinnhubService().get_quote("AAPL") is None
    settings_env(MARKETVISION_OFFLINE="1", FINNHUB_API_KEY=FINNHUB_KEY)
    assert not FinnhubService().is_available()


# ---------------------------------------------------------------- NewsAPI
def test_old_newsapi_service_sends_the_key_in_a_header(settings_env) -> None:
    settings_env(MARKETVISION_OFFLINE="0", NEWSAPI_KEY=NEWS_KEY)
    from app.services.newsapi_service import NewsApiService

    article = {"title": "Apple ships a thing", "description": "Up", "source": {"name": "Wire"}, "url": "https://x.test/a", "publishedAt": "2026-10-08T12:00:00Z"}
    session = _install_session("newsapi", [FakeResponse(200, {"status": "ok", "articles": [article, {"title": "[Removed]"}]})])
    items = NewsApiService().get_stock_news("AAPL", limit=5)
    assert [i["title"] for i in items] == ["Apple ships a thing"]
    sent = session.requests[0]
    assert sent["headers"] == {"X-Api-Key": NEWS_KEY}
    assert "apiKey" not in sent["params"] and NEWS_KEY not in sent["url"]


@pytest.mark.parametrize("env", ["production", "staging"])
def test_newsapi_free_plan_is_refused_in_staging_and_production(settings_env, env) -> None:
    from app.core.config import get_settings
    from app.marketdata.providers.newsapi import NewsApiProvider
    from app.services.newsapi_service import NewsApiService

    settings_env(MARKETVISION_OFFLINE="0", NEWSAPI_KEY=NEWS_KEY, MARKETVISION_ENV=env, MARKETVISION_NEWSAPI_PAID_PLAN=None)
    settings = get_settings()
    assert not NewsApiService().is_available()
    assert settings.capabilities()["newsapi"] is False
    provider = NewsApiProvider(settings.newsapi_key, shared_client("newsapi"), blocked_reason=settings.newsapi_blocked_reason)
    ok, reason = provider.available()
    assert not ok and "Developer plan" in reason
    with pytest.raises(ProviderError) as raised:
        provider.news("AAPL", None, 3)
    assert raised.value.code == "not_configured"

    settings_env(MARKETVISION_NEWSAPI_PAID_PLAN="1")
    assert NewsApiService().is_available() and get_settings().capabilities()["newsapi"] is True


def test_newsapi_error_body_code_is_reported_without_the_key(settings_env) -> None:
    from app.core.secrets import Secret
    from app.marketdata.http import HttpClient
    from app.marketdata.providers.newsapi import NewsApiProvider

    session = FakeSession([FakeResponse(200, {"status": "error", "code": "rateLimited", "message": "too many"})])
    provider = NewsApiProvider(Secret(NEWS_KEY), HttpClient("newsapi", session=session))
    with pytest.raises(ProviderError) as raised:
        provider.news("AAPL", None, 3)
    assert "rateLimited" in raised.value.message and NEWS_KEY not in raised.value.message


# ---------------------------------------------------------------- news age
NOW = datetime(2026, 10, 9, 18, 0, tzinfo=timezone.utc)


def test_every_article_carries_its_age_and_old_news_is_flagged() -> None:
    old = [
        {"title": "Week-old story", "published_at": (NOW - timedelta(days=7)).isoformat()},
        {"title": "Three-day-old story", "published_at": (NOW - timedelta(days=3)).isoformat()},
        {"title": "No date", "published_at": None},
    ]
    md = MarketData([FakeProvider("newsapi", {"news"}, news=old)], clock=lambda: NOW, sleep=lambda s: None)
    env = md.news(symbol="AAPL")
    ages = {i["title"]: i["age_hours"] for i in env.data}
    assert ages == {"Week-old story": 168.0, "Three-day-old story": 72.0, "No date": None}
    assert env.quality.has("stale_news") and env.quality.status == "degraded"
    assert env.quality.has("undated_articles")
    assert env.market_timestamp == NOW - timedelta(hours=72)
    assert env.freshness == "delayed"


def test_recent_news_is_not_flagged_stale() -> None:
    fresh = [{"title": "This morning", "published_at": (NOW - timedelta(hours=5)).isoformat()}]
    env = MarketData([FakeProvider("finnhub", {"news"}, news=fresh)], clock=lambda: NOW, sleep=lambda s: None).news(symbol="AAPL")
    assert env.data[0]["age_hours"] == 5.0 and not env.quality.has("stale_news")


# ---------------------------------------------------------------- FRED (older dashboard routes)
class FakeEconomic(Provider):
    name = "fred"
    supports = frozenset({"economic"})

    def __init__(self, data: dict[str, pd.Series] | None = None, fail: bool = False) -> None:
        self.data, self.fail = data or {}, fail

    def economic_series(self, series_id, start, end):
        if self.fail or series_id not in self.data:
            raise ProviderError("fred is unreachable.", details={"provider": "fred"})
        return self.data[series_id]


def test_old_fred_service_reads_through_the_data_layer_and_drops_missing_values(settings_env) -> None:
    settings_env(MARKETVISION_OFFLINE="0", FRED_API_KEY=None)
    from app.services.fred_service import FredService, _get_series

    series = pd.Series([4.1, float("nan"), 4.3], index=pd.to_datetime(["2026-10-05", "2026-10-06", "2026-10-07"]), name="DGS10")
    set_market_data(MarketData([FakeEconomic({"DGS10": series})], sleep=lambda s: None))
    try:
        assert FredService().is_available()  # no key needed: the public CSV route exists
        assert _get_series("DGS10", limit=2) == [{"date": "2026-10-07", "value": 4.3}, {"date": "2026-10-05", "value": 4.1}]
    finally:
        set_market_data(None)


def test_old_fred_service_shows_no_scores_when_nothing_could_be_fetched(settings_env) -> None:
    settings_env(MARKETVISION_OFFLINE="0")
    from app.services.fred_service import FredService

    set_market_data(MarketData([FakeEconomic(fail=True)], sleep=lambda s: None))
    try:
        service = FredService()
        intelligence = service.get_macro_intelligence()
        assert intelligence["available"] is False and "scores" not in intelligence
        assert service.get_market_conditions()["overall_assessment"] == "unavailable"
    finally:
        set_market_data(None)


# ---------------------------------------------------------------- copilot
class _FakeCompletions:
    def __init__(self, owner) -> None:
        self.owner = owner

    def create(self, **request):
        self.owner.requests.append(request)
        if self.owner.fail:
            raise RuntimeError(f"upstream said no; key={self.owner.kwargs.get('api_key')}")
        message = types.SimpleNamespace(content=" Explained. ")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.fixture()
def fake_openai(monkeypatch):
    state = types.SimpleNamespace(kwargs=None, requests=[], fail=False)

    class OpenAI:
        def __init__(self, **kwargs) -> None:
            state.kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=_FakeCompletions(state))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    return state


def test_copilot_is_off_and_says_why_when_nothing_is_configured(settings_env, fake_openai) -> None:
    settings_env(MARKETVISION_LLM_API_KEY=None, MARKETVISION_LLM_BASE_URL=None, OPENAI_API_KEY=None)
    from app.services.copilot_service import CopilotService

    service = CopilotService()
    assert not service.is_available() and fake_openai.kwargs is None
    assert "MARKETVISION_LLM_BASE_URL" in service.chat("hi")
    assert service.explain_prediction_facts({"ticker": "AAPL"})["model"] == "deterministic_fallback"


def test_copilot_uses_the_configured_local_server_and_model(settings_env, fake_openai) -> None:
    settings_env(MARKETVISION_LLM_BASE_URL="http://127.0.0.1:8089/v1", MARKETVISION_LLM_MODEL="qwen2.5-3b-instruct", MARKETVISION_LLM_API_KEY=None, OPENAI_API_KEY=None)
    from app.services.copilot_service import CopilotService

    service = CopilotService()
    assert service.is_available()
    assert fake_openai.kwargs["base_url"] == "http://127.0.0.1:8089/v1"
    assert service.chat("Explain RSI") == "Explained."
    request = fake_openai.requests[0]
    assert request["model"] == "qwen2.5-3b-instruct" and request["temperature"] == 0.4
    assert "Never state one from memory" in request["messages"][0]["content"]


def test_copilot_omits_temperature_for_claudes_openai_compatible_endpoint(settings_env, fake_openai) -> None:
    settings_env(MARKETVISION_LLM_BASE_URL="https://api.anthropic.com/v1/", MARKETVISION_LLM_API_KEY="sk-ant-test-not-real-000000", MARKETVISION_LLM_MODEL="claude-sonnet-5-5")  # not-a-secret: test placeholder
    from app.services.copilot_service import CopilotService

    CopilotService().chat("hi")
    assert "temperature" not in fake_openai.requests[0]


def test_copilot_error_text_never_reaches_the_caller(settings_env, fake_openai) -> None:
    settings_env(MARKETVISION_LLM_API_KEY="sk-test-secret-value-123456", MARKETVISION_LLM_BASE_URL=None)  # not-a-secret: test placeholder
    from app.services.copilot_service import CopilotService

    fake_openai.fail = True
    answer = CopilotService().chat("hi")
    assert answer == "The language model could not answer (RuntimeError). Nothing was generated."


# ---------------------------------------------------------------- live probe
def test_probe_reports_offline_without_contacting_anything(settings_env) -> None:
    settings_env(MARKETVISION_OFFLINE="1")
    from app.marketdata.probe import probe_providers

    report = probe_providers()
    assert report["live"] is False and report["checks"] == []


class _Unavailable(Provider):
    def __init__(self, name, supports, reason) -> None:
        self.name, self.supports, self.reason = name, frozenset(supports), reason

    def available(self):
        return False, self.reason


def test_probe_separates_answered_failed_and_unconfigured_and_never_leaks_a_key(settings_env) -> None:
    settings_env(MARKETVISION_OFFLINE="0", FINNHUB_API_KEY=FINNHUB_KEY)
    from app.marketdata import calendar as cal
    from app.marketdata.probe import probe_providers

    closed = datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc)  # Friday after the close
    history = pd.DataFrame(
        {"Open": [10.0, 10.5], "High": [11.0, 11.0], "Low": [9.5, 10.0], "Close": [10.5, 10.8], "Volume": [1e6, 1e6]},
        index=pd.DatetimeIndex([pd.Timestamp(cal.shift_trading_days(closed.date(), -1)), pd.Timestamp(closed.date())], name="Date"),
    )
    providers = {
        "yahoo": FakeProvider("yahoo", {"history", "quote"}, history=history, quote={"price": 10.8, "market_timestamp": closed - timedelta(hours=2), "realtime": False}),
        "finnhub": FakeProvider("finnhub", {"quote", "news"}, fail=ProviderError(f"finnhub broke while sending token={FINNHUB_KEY}", details={"provider": "finnhub"})),
        "fred": FakeEconomic({"DGS10": pd.Series([4.0, float("nan")], index=pd.to_datetime(["2026-10-07", "2026-10-08"]))}),
        "newsapi": _Unavailable("newsapi", {"news"}, "NEWSAPI_KEY is not set."),
    }
    report = probe_providers(providers=providers, clock=lambda: closed)
    outcomes = {row["check"]: row["outcome"] for row in report["checks"]}
    assert outcomes == {
        "yahoo_history": "ok", "yahoo_quote": "ok", "finnhub_quote": "failed",
        "finnhub_company_news": "failed", "fred_series": "ok", "newsapi_news": "not_configured",
    }
    assert report["verified"] == ["fred", "yahoo"] and report["failed"] == ["finnhub"] and report["not_configured"] == ["newsapi"]
    quote = next(r for r in report["checks"] if r["check"] == "yahoo_quote")
    assert quote["freshness"] == "historical" and quote["age_seconds"] == 7200.0 and quote["sample"]["price"] == 10.8
    fred = next(r for r in report["checks"] if r["check"] == "fred_series")
    assert fred["sample"]["missing_values"] == 1 and fred["sample"]["route"] == "public_csv"
    assert FINNHUB_KEY not in json.dumps(report, default=str)


def test_provider_probe_route_needs_the_read_scope_when_tokens_are_set(settings_env, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import app.marketdata.probe as probe
    from app.main import app

    monkeypatch.setattr(probe, "probe_providers", lambda include_llm=False: {"live": True, "checks": [], "include_llm": include_llm})
    settings_env(MARKETVISION_TOOL_TOKEN="read-token-abcdef")
    client = TestClient(app)
    assert client.get("/api/v2/health/providers").status_code == 401
    answer = client.get("/api/v2/health/providers", headers={"Authorization": "Bearer read-token-abcdef"})
    assert answer.status_code == 200 and answer.json() == {"live": True, "checks": [], "include_llm": False}


# ---------------------------------------------------------------- overview cards
def test_overview_cards_name_the_etf_they_show_and_measure_every_label(settings_env) -> None:
    from tests.helpers import make_ohlcv
    from app.services.market import MarketDataService as MarketService

    frame = make_ohlcv(60, start="2026-08-03")
    frame = frame[frame.index <= pd.Timestamp("2026-10-08")]
    set_market_data(MarketData([FakeProvider("yahoo", {"history"}, history=frame)], clock=lambda: NOW, sleep=lambda s: None))
    try:
        overview = MarketService().get_market_overview()
    finally:
        set_market_data(None)
    names = [c["name"] for c in overview["cards"]]
    assert "S&P 500 ETF (SPY)" in names and "S&P 500" not in names
    assert "Dow Jones ETF (DIA)" in names and "ETF" in overview["note"]
    for card in overview["cards"]:
        assert card["volatility"] in {"Small move", "Medium move", "Large move"}  # measured, never a fixed word
        assert card["as_of"] and card["source"] == "yahoo"
    sector = next(c for c in overview["cards"] if c["name"] == "Top Sector Today")
    assert sector["symbol"] in {"XLK", "XLF", "XLE", "XLV", "XLI", "XLY"}


def test_overview_leaves_out_cards_it_has_no_data_for(settings_env) -> None:
    from app.services.market import MarketDataService as MarketService

    set_market_data(MarketData([FakeProvider("yahoo", {"history"}, fail=ProviderError("down", details={"provider": "yahoo"}))], sleep=lambda s: None))
    try:
        assert MarketService().get_market_overview()["cards"] == []
    finally:
        set_market_data(None)
