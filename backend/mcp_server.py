from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests

from app.services.tool_catalog import DISCLAIMER, list_tools

API_BASE_URL = os.getenv("MARKETVISION_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("MARKETVISION_API_TOKEN") or os.getenv("MARKETVISION_TOOL_TOKEN", "")
SERVER_NAME = "marketvision-ai"
PROTOCOL_VERSION = "2024-11-05"


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/tools/{name}",
        headers=_headers(),
        json={"arguments": arguments or {}},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("result", payload)


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": "13.1.0"},
            "instructions": DISCLAIMER,
        })

    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None

    if method == "ping":
        return _result(request_id, {})

    if method == "tools/list":
        tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }
            for tool in list_tools()
        ]
        return _result(request_id, {"tools": tools})

    if method == "tools/call":
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        try:
            structured = _call_tool(name, arguments)
            return _result(request_id, {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(structured, indent=2, default=str),
                    }
                ],
                "structuredContent": structured,
                "isError": False,
            })
        except Exception as exc:
            return _result(request_id, {
                "content": [{"type": "text", "text": f"MarketVision tool call failed: {exc}"}],
                "isError": True,
            })

    return _error(request_id, -32601, f"Unsupported MCP method: {method}")


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line == b"":
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.decode("utf-8").partition(":")
        headers[key.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    raw = sys.stdin.buffer.read(length)
    return json.loads(raw.decode("utf-8"))


def _write_message(message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":"), default=str).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break
        response = handle_message(message)
        if response is not None and "id" in message:
            _write_message(response)


if __name__ == "__main__":
    main()

