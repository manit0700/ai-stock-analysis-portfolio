"""The agent tool layer and the MCP server: validated inputs, permissions, the standard transport."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import mcp_server
from app.agent_tools.tools import TOOLS, call_tool, list_tools
from app.core import clock
from app.core.errors import NotFound, ValidationFailed
from app.marketdata.cache import TTLCache
from app.marketdata.service import MarketData
from app.marketdata.snapshots import SnapshotProvider, SnapshotStore
from app.paper.account import PaperAccountService, PaperConfig, set_paper_service
from tests.helpers import make_snapshot

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture()
def wired(tmp_path):
    """Real services wired to a synthetic snapshot: tools run end to end with no network."""
    from app.forecast.service import PredictionService, set_prediction_service
    from app.marketdata.service import set_market_data

    snapshot_id, symbols = make_snapshot(tmp_path / "snapshots", symbols=3, periods=500, seed=9)
    provider = SnapshotProvider(SnapshotStore(tmp_path / "snapshots"), snapshot_id)
    market = MarketData([provider], order={"history": [provider.name], "economic": [provider.name]}, cache=TTLCache(), retries=0)
    set_market_data(market)
    from app.forecast import registry as model_registry

    model_registry.record_baseline({
        "from_run": "test-run", "horizon": 5, "base_rate": 0.55, "data": "snapshot:test",
        "uncertainty": {"horizon": 5, "rows": 1000, "unconditional": {"0.05": -0.06, "0.10": -0.04, "0.25": -0.015, "0.50": 0.001, "0.75": 0.017, "0.90": 0.045, "0.95": 0.062}, "scaled": {}},
        "uncertainty_scaled_supported": False, "historical_mean_return": 0.001, "verdict": "no predictive edge established",
    }, tmp_path / "models")
    set_prediction_service(PredictionService(market, model_dir=tmp_path / "models"))

    class QuoteAdapter:
        def quote(self, symbol):
            bars = market.history(symbol, interval="1d").data

            class Envelope:
                pass
            envelope = Envelope()
            envelope.symbol = symbol.upper()
            envelope.provider = "snapshot"
            envelope.freshness = "delayed"
            envelope.retrieved_at = clock.utcnow()
            envelope.data = {"price": float(bars["Close"].iloc[-1]), "age_seconds": 10.0, "market_state": "open"}
            return envelope
    set_paper_service(PaperAccountService(QuoteAdapter(), db_path=tmp_path / "paper.sqlite3", config=PaperConfig(half_spread_bps=Decimal("0"), slippage_bps=Decimal("0"))))
    last = market.history(symbols[0], interval="1d").data.index[-1]
    clock.set_clock(lambda: datetime(last.year, last.month, last.day, 23, 0, tzinfo=timezone.utc))
    yield {"symbols": symbols, "market": market}
    set_market_data(None)
    set_prediction_service(None)
    set_paper_service(None)


def test_the_catalogue_has_the_fifteen_interfaces_with_schemas_and_permissions():
    names = {t["name"] for t in list_tools()}
    assert {"get_stock_history", "get_market_quote", "get_company_profile", "get_financial_news", "get_economic_indicators", "get_technical_indicators",
            "get_sentiment_analysis", "get_stock_prediction", "get_prediction_history", "run_backtest", "get_portfolio_risk", "get_virtual_account",
            "submit_paper_order", "get_trade_history", "get_portfolio_performance"} <= names
    for tool in list_tools():
        schema = tool["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False and tool["permission"] in {"read", "paper_trade", "admin"}
    assert TOOLS["submit_paper_order"].permission == "paper_trade"
    assert all(TOOLS[n].permission == "read" for n in names - {"submit_paper_order"})
    json.dumps(list_tools())


def test_arguments_are_validated_before_anything_runs():
    with pytest.raises(NotFound):
        call_tool("no_such_tool", {})
    with pytest.raises(ValidationFailed):
        call_tool("get_stock_history", {})  # symbol is required
    with pytest.raises(ValidationFailed):
        call_tool("get_stock_history", {"symbol": "AAPL", "shout": True})  # unknown argument
    with pytest.raises(ValidationFailed):
        call_tool("get_stock_history", {"symbol": "AAPL", "interval": "7m"})  # not in the enum
    with pytest.raises(ValidationFailed):
        call_tool("get_financial_news", {"symbol": "AAPL", "limit": 999})  # over the bound
    with pytest.raises(ValidationFailed):
        call_tool("get_market_calendar", {"start": "soon", "end": "2026-01-31"})  # not a date


def test_research_tools_return_structured_output_with_provenance(wired):
    symbol = wired["symbols"][0]
    history = call_tool("get_stock_history", {"symbol": symbol, "period": "1mo"})
    assert history["tool"] == "get_stock_history" and history["permission"] == "read"
    assert history["result"]["provider"].startswith("snapshot:") and history["result"]["freshness"] in {"historical", "delayed"}
    assert history["result"]["quality"]["status"] in {"ok", "degraded"}
    indicators = call_tool("get_technical_indicators", {"symbol": symbol})["result"]
    assert set(indicators["indicators"]) >= {"rsi_14", "sma_50", "macd", "bollinger", "atr_14"}
    calendar = call_tool("get_market_calendar", {"start": "2026-11-23", "end": "2026-11-27"})["result"]
    thursday = next(day for day in calendar["data"]["days"] if day["date"] == "2026-11-26")
    assert thursday["holiday"] == "Thanksgiving Day" and not thursday["is_trading_day"]


def test_prediction_tool_reports_the_baseline_and_records_to_the_ledger(wired, tmp_path, settings_env):
    settings_env(MARKETVISION_DATA_DIR=str(tmp_path / "data"))
    symbol = wired["symbols"][0]
    out = call_tool("get_stock_prediction", {"symbol": symbol})["result"]
    assert out["status"] in {"baseline", "unavailable"} and out["authorises_trade"] is False
    if out["status"] == "baseline":
        assert out["edge_established"] is False and out.get("recorded") is True
        history = call_tool("get_prediction_history", {"symbol": symbol})["result"]
        assert history["performance"]["issued"] == 1 and history["history"][0]["prediction_id"] == out["prediction_id"]
        again = call_tool("get_stock_prediction", {"symbol": symbol})["result"]
        assert again["prediction_id"] == out["prediction_id"]  # same forecast, same id, no duplicate row
        assert call_tool("get_prediction_history", {"symbol": symbol})["result"]["performance"]["issued"] == 1


def test_paper_tools_enforce_the_simulation_flag_and_share_one_account(wired):
    with pytest.raises(ValidationFailed):
        call_tool("submit_paper_order", {"side": "buy", "symbol": wired["symbols"][0], "notional": 10})
    order = call_tool("submit_paper_order", {"side": "buy", "symbol": wired["symbols"][0], "notional": 10, "simulation": True})["result"]
    assert order["status"] == "filled" and order["banner"] == "SIMULATED TRADING — NO REAL MONEY"
    account = call_tool("get_virtual_account", {})["result"]
    assert account["simulation"] is True and len(account["positions"]) == 1
    trades = call_tool("get_trade_history", {})["result"]["orders"]
    assert trades[0]["order_id"] == order["order_id"]
    performance = call_tool("get_portfolio_performance", {})["result"]
    assert performance["starting_cash"] == "50.00"


# ---------------------------------------------------------------------- MCP
def test_handle_message_speaks_json_rpc():
    init = mcp_server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
    assert init["result"]["protocolVersion"] == "2025-06-18" and init["result"]["serverInfo"]["name"] == "marketvision"
    assert mcp_server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = mcp_server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert {t["name"] for t in tools["result"]["tools"]} == {t["name"] for t in list_tools()}
    bad = mcp_server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    assert bad["error"]["code"] == -32601
    refused = mcp_server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "get_stock_history", "arguments": {}}})
    assert refused["result"]["isError"] is True and refused["result"]["structuredContent"]["error"]["code"] == "validation_failed"


@pytest.mark.timeout(60)
def test_the_official_mcp_client_can_connect_list_and_call():
    """The audit's A1: the old server timed out on initialize with a standard client."""
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=[str(BACKEND / "mcp_server.py")], env={"MARKETVISION_OFFLINE": "1", "MARKETVISION_ENV": "test", "PYTHONPATH": str(BACKEND)})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                with anyio.fail_after(30):
                    info = await session.initialize()
                    assert info.server_info.name == "marketvision"
                    tools = await session.list_tools()
                    assert any(tool.name == "get_market_calendar" for tool in tools.tools)
                    result = await session.call_tool("get_market_calendar", {"start": "2026-07-01", "end": "2026-07-06"})
                    assert not result.is_error
                    days = {d["date"]: d for d in result.structured_content["result"]["data"]["days"]}
                    assert days["2026-07-03"]["holiday"] is not None  # 4 July 2026 is a Saturday, observed Friday
                    refusal = await session.call_tool("submit_paper_order", {"side": "buy", "symbol": "AAPL", "notional": 10, "simulation": False})
                    assert refusal.is_error and "simulation" in json.dumps(refusal.structured_content)
    anyio.run(run)


def test_the_server_is_newline_delimited_not_content_length_framed():
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    proc = subprocess.run(
        [sys.executable, str(BACKEND / "mcp_server.py")], input=request + "\n", text=True, capture_output=True, timeout=30,
        env={"MARKETVISION_OFFLINE": "1", "MARKETVISION_ENV": "test", "PATH": "/usr/bin:/bin", "PYTHONPATH": str(BACKEND)},
    )
    line = proc.stdout.strip().splitlines()[0]
    assert not line.startswith("Content-Length")
    assert json.loads(line)["result"]["protocolVersion"] == "2024-11-05"
