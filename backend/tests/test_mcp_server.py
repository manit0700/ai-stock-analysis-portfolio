from __future__ import annotations

import mcp_server


def test_mcp_initialize_and_tool_list() -> None:
    init = mcp_server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init is not None
    assert init["result"]["serverInfo"]["name"] == "marketvision-ai"
    assert init["result"]["capabilities"]["tools"] == {}

    tools = mcp_server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = {tool["name"] for tool in tools["result"]["tools"]}
    assert "predict_market" in names
    assert "run_scanner" in names
    assert "analyze_portfolio" in names


def test_mcp_tool_call_returns_structured_content(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "_call_tool", lambda name, arguments: {"ticker": arguments["ticker"], "confidence": 82})

    response = mcp_server.handle_message({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "predict_market", "arguments": {"ticker": "NVDA"}},
    })

    assert response is not None
    assert response["result"]["structuredContent"]["ticker"] == "NVDA"
    assert response["result"]["isError"] is False

