"""MarketVision as an MCP server (stdio).

The previous version framed messages with ``Content-Length`` headers, which is LSP framing, not
MCP: the MCP stdio transport is newline-delimited JSON-RPC, so standard clients hung on
``initialize``. This version speaks the standard transport and is tested against the official
``mcp`` client in ``tests/test_mcp_server.py``.

It serves the fifteen tools in ``app.agent_tools`` in-process (no HTTP hop). Run it from the
``backend`` folder:

    python mcp_server.py

Client configuration:

    {"mcpServers": {"marketvision": {"command": "python", "args": ["/path/to/backend/mcp_server.py"]}}}

Research tools are read-only. ``submit_paper_order`` only moves virtual money and requires
``simulation: true``. Nothing here can touch a real brokerage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent_tools.tools import DISCLAIMER, call_tool, list_tools  # noqa: E402
from app.core.errors import MarketVisionError  # noqa: E402
from app.core.secrets import install_log_redaction, redact  # noqa: E402

SERVER_NAME = "marketvision"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18"}
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": redact(message)}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """One JSON-RPC message in, one response out (or ``None`` for a notification)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "Not a JSON-RPC 2.0 message.")
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    is_notification = "id" not in message

    if method == "initialize":
        requested = str(params.get("protocolVersion") or "")
        version = requested if requested in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
        return _result(request_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "MarketVision research and paper trading", "version": SERVER_VERSION},
            "instructions": (
                "Market research, probability forecasts and a simulated $50 trading account. "
                "Forecasts are labelled baseline until a model passes the promotion rule; they never authorise trades. "
                "submit_paper_order moves virtual money only and requires simulation=true. " + DISCLAIMER
            ),
        })
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        tools = [
            {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"], "annotations": {"readOnlyHint": t["permission"] == "read"}}
            for t in list_tools()
        ]
        return _result(request_id, {"tools": tools})
    if method == "tools/call":
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        try:
            payload = call_tool(name, arguments)
        except MarketVisionError as exc:
            # A refusal (bad input, no data, an order the rules reject) is a tool result, not a protocol error.
            body = {"error": {"code": exc.code, "message": redact(exc.message), "details": json.loads(redact(json.dumps(exc.details, default=str)))}}
            return _result(request_id, {"content": [{"type": "text", "text": json.dumps(body, indent=2)}], "structuredContent": body, "isError": True})
        except Exception as exc:  # noqa: BLE001 - never leak a traceback (it may hold a URL with a key)
            return _result(request_id, {"content": [{"type": "text", "text": f"The tool failed unexpectedly ({type(exc).__name__})."}], "isError": True})
        text = json.dumps(payload, indent=2, default=str)
        return _result(request_id, {"content": [{"type": "text", "text": text}], "structuredContent": payload, "isError": False})
    if is_notification:
        return None
    return _error(request_id, METHOD_NOT_FOUND, f"Unsupported method: {method}")


def main() -> None:
    install_log_redaction()
    for raw in sys.stdin.buffer:
        raw = raw.strip()
        if not raw:
            continue
        try:
            message = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            response: dict[str, Any] | None = _error(None, PARSE_ERROR, "The line is not valid JSON.")
        else:
            try:
                response = handle_message(message)
            except Exception as exc:  # noqa: BLE001
                response = _error(message.get("id") if isinstance(message, dict) else None, INTERNAL_ERROR, f"Unhandled server error ({type(exc).__name__}).")
            if isinstance(message, dict) and "id" not in message:
                response = None
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":"), default=str) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
