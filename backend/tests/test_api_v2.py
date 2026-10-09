"""The v2 routes end to end over the test client, with services wired to a synthetic snapshot."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.core import clock
from app.core.config import reset_settings
from app.marketdata.cache import TTLCache
from app.marketdata.service import MarketData, set_market_data
from app.marketdata.snapshots import SnapshotProvider, SnapshotStore
from app.paper.account import PaperAccountService, PaperConfig, set_paper_service
from tests.helpers import make_snapshot


@pytest.fixture()
def client(tmp_path, settings_env):
    from app.forecast import registry as model_registry
    from app.forecast.service import PredictionService, set_prediction_service

    settings_env(MARKETVISION_DATA_DIR=str(tmp_path / "data"), MARKETVISION_MODEL_DIR=str(tmp_path / "models"))
    snapshot_id, symbols = make_snapshot(tmp_path / "snapshots", symbols=2, periods=460, seed=14)
    provider = SnapshotProvider(SnapshotStore(tmp_path / "snapshots"), snapshot_id)
    market = MarketData([provider], order={"history": [provider.name], "economic": [provider.name]}, cache=TTLCache(), retries=0)
    set_market_data(market)
    model_registry.record_baseline({"from_run": "test", "horizon": 5, "base_rate": 0.54, "uncertainty": {"horizon": 5, "rows": 100, "unconditional": {"0.05": -0.05, "0.10": -0.03, "0.25": -0.01, "0.50": 0.0, "0.75": 0.012, "0.90": 0.032, "0.95": 0.05}, "scaled": {}}, "uncertainty_scaled_supported": False, "historical_mean_return": 0.001}, tmp_path / "models")
    set_prediction_service(PredictionService(market, model_dir=tmp_path / "models"))

    class QuoteAdapter:
        def quote(self, symbol):
            bars = market.history(symbol, interval="1d").data

            class Envelope:
                pass
            env = Envelope()
            env.symbol, env.provider, env.freshness, env.retrieved_at = symbol.upper(), "snapshot", "delayed", clock.utcnow()
            env.data = {"price": float(bars["Close"].iloc[-1]), "age_seconds": 5.0, "market_state": "open"}
            return env
    set_paper_service(PaperAccountService(QuoteAdapter(), db_path=tmp_path / "paper.sqlite3", config=PaperConfig(half_spread_bps=Decimal("0"), slippage_bps=Decimal("0"))))
    last = market.history(symbols[0], interval="1d").data.index[-1]
    clock.set_clock(lambda: datetime(last.year, last.month, last.day, 23, 0, tzinfo=timezone.utc))

    import app.main as main_module

    with TestClient(main_module.app, raise_server_exceptions=False) as tc:
        yield tc, symbols
    set_market_data(None)
    set_prediction_service(None)
    set_paper_service(None)
    reset_settings()


def test_health_names_capabilities_and_the_simulation_banner(client):
    tc, _ = client
    body = tc.get("/api/v2/health").json()
    assert body["status"] == "ok" and body["banner"] == "SIMULATED TRADING — NO REAL MONEY"
    assert body["forecast"]["edge_established"] is False
    assert set(body["capabilities"]) >= {"finnhub", "fred_api", "newsapi", "paper_trading"}


def test_history_carries_provenance_and_errors_are_structured(client):
    tc, symbols = client
    body = tc.get(f"/api/v2/stocks/{symbols[0]}/history", params={"period": "1mo"}).json()
    assert body["provider"].startswith("snapshot:") and body["freshness"] in {"historical", "delayed"} and body["data"]
    missing = tc.get("/api/v2/stocks/NOPE/history")
    assert missing.status_code == 503 and missing.json()["error"]["code"] == "data_unavailable"
    bad = tc.get("/api/v2/stocks/B..D!!/history")
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_failed"


def test_forecast_route_serves_the_labelled_baseline_and_records_it(client):
    tc, symbols = client
    body = tc.get(f"/api/v2/stocks/{symbols[0]}/forecast").json()
    assert body["status"] == "baseline" and body["edge_established"] is False and body["authorises_trade"] is False
    ledger = tc.get("/api/v2/predictions", params={"symbol": symbols[0]}).json()
    assert ledger["performance"]["issued"] == 1 and ledger["history"][0]["probability_up"] == body["probability_up"]


def test_paper_flow_over_http(client):
    tc, symbols = client
    refused = tc.post("/api/v2/paper/orders", json={"side": "buy", "symbol": symbols[0], "notional": 10})
    assert refused.status_code == 422 and "simulation" in refused.json()["error"]["message"]
    order = tc.post("/api/v2/paper/orders", json={"side": "buy", "symbol": symbols[0], "notional": 10, "simulation": True}).json()
    assert order["status"] == "filled"
    account = tc.get("/api/v2/paper/account").json()
    assert account["banner"] == "SIMULATED TRADING — NO REAL MONEY" and len(account["positions"]) == 1
    too_big = tc.post("/api/v2/paper/orders", json={"side": "buy", "symbol": symbols[0], "notional": 45, "simulation": True})
    assert too_big.status_code == 409 and too_big.json()["error"]["code"] in {"position_limit", "insufficient_cash"}
    assert tc.get("/api/v2/paper/performance").json()["starting_cash"] == "50.00"


def test_tool_route_blocks_state_changing_tools(client):
    tc, symbols = client
    listed = tc.get("/api/v2/tools").json()["tools"]
    assert any(t["name"] == "get_stock_history" for t in listed)
    ok = tc.post("/api/v2/tools/get_market_calendar", json={"arguments": {"start": "2026-01-01", "end": "2026-01-10"}})
    assert ok.status_code == 200
    blocked = tc.post("/api/v2/tools/submit_paper_order", json={"arguments": {"side": "buy", "symbol": symbols[0], "notional": 5, "simulation": True}})
    assert blocked.status_code == 422 and "own route" in blocked.json()["error"]["message"]


def test_scopes_are_enforced_when_tokens_are_configured(client, settings_env):
    tc, symbols = client
    settings_env(MARKETVISION_TOOL_TOKEN="tool-token-123456", MARKETVISION_ADMIN_TOKEN="admin-token-123456")
    assert tc.get(f"/api/v2/stocks/{symbols[0]}/history").status_code == 401
    assert tc.get(f"/api/v2/stocks/{symbols[0]}/history", headers={"Authorization": "Bearer wrong-token-000000"}).status_code == 403
    assert tc.get(f"/api/v2/stocks/{symbols[0]}/history", headers={"Authorization": "Bearer tool-token-123456"}).status_code == 200
    # read token is not enough for state changes; admin is, and admin also satisfies read
    order = {"side": "buy", "symbol": symbols[0], "notional": 5, "simulation": True}
    assert tc.post("/api/v2/paper/orders", json=order, headers={"Authorization": "Bearer tool-token-123456"}).status_code == 403
    assert tc.post("/api/v2/paper/orders", json=order, headers={"Authorization": "Bearer admin-token-123456"}).status_code == 200
    assert tc.get(f"/api/v2/stocks/{symbols[0]}/history", headers={"Authorization": "Bearer admin-token-123456"}).status_code == 200
    # the legacy tool route now uses the same scopes (previously: non-constant-time compare, 3 of 36 routes)
    assert tc.get("/api/tools").status_code == 401


def test_the_legacy_health_route_still_answers(client):
    tc, _ = client
    body = tc.get("/health").json()
    assert body["status"] == "ok" and body["ml_model_loaded"] is False
