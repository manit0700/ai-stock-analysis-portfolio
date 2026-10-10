"""The v2 API: the new capabilities, each route a thin shell over a tested service.

Everything here returns provenance (``data_source`` or equivalent) and honest labels. Routes
that spend provider quota need the read scope when a tool token is configured; routes that
change state (paper orders, pause, reset, resolving predictions) need the admin scope.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from app.agent_tools.tools import call_tool, list_tools
from app.core.config import get_settings
from app.core.errors import ValidationFailed
from app.core.security import rate_limit, require_admin, require_read
from app.forecast.service import get_prediction_service
from app.marketdata.service import get_market_data
from app.paper.account import get_paper_service
from app.predictions.ledger import PredictionLedger

router = APIRouter(prefix="/api/v2")


# ---------------------------------------------------------------------- market data
@router.get("/market/status", dependencies=[Depends(require_read)])
def market_status() -> dict[str, Any]:
    """Provider availability and the market calendar state. No secrets."""
    return get_market_data().status()


@router.get("/market/calendar")
def market_calendar(start: date = Query(...), end: date = Query(...)) -> dict[str, Any]:
    return get_market_data().market_calendar(start, end).to_dict()


@router.get("/stocks/{symbol}/history", dependencies=[Depends(require_read)])
def stock_history(symbol: str, period: str = Query(default="1y"), interval: str = Query(default="1d")) -> dict[str, Any]:
    return get_market_data().history(symbol, period=period, interval=interval).to_dict()


@router.get("/stocks/{symbol}/quote", dependencies=[Depends(require_read)])
def stock_quote(symbol: str) -> dict[str, Any]:
    return get_market_data().quote(symbol).to_dict()


@router.get("/stocks/{symbol}/profile", dependencies=[Depends(require_read)])
def stock_profile(symbol: str) -> dict[str, Any]:
    return get_market_data().profile(symbol).to_dict()


@router.get("/news", dependencies=[Depends(require_read), Depends(rate_limit("news", 30))])
def news(symbol: str | None = Query(default=None), q: str | None = Query(default=None), limit: int = Query(default=10, ge=1, le=50)) -> dict[str, Any]:
    return get_market_data().news(symbol=symbol, query=q, limit=limit).to_dict()


@router.get("/economic/{series}", dependencies=[Depends(require_read)])
def economic(series: str, limit: int = Query(default=24, ge=1, le=120)) -> dict[str, Any]:
    return call_tool("get_economic_indicators", {"series": series, "limit": limit})["result"]


# ---------------------------------------------------------------------- forecasts
@router.get("/forecast/status")
def forecast_status() -> dict[str, Any]:
    return get_prediction_service().status()


@router.get("/stocks/{symbol}/forecast", dependencies=[Depends(require_read), Depends(rate_limit("forecast", 60))])
def forecast(symbol: str, record: bool = Query(default=True)) -> dict[str, Any]:
    return call_tool("get_stock_prediction", {"symbol": symbol, "record": record})["result"]


@router.get("/predictions", dependencies=[Depends(require_read)])
def predictions(symbol: str | None = Query(default=None), horizon: int = Query(default=5, ge=1, le=60), limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    ledger = PredictionLedger(horizon=horizon)
    return {"history": ledger.history(symbol=symbol, limit=limit), "performance": ledger.performance()}


@router.post("/predictions/resolve", dependencies=[Depends(require_admin)])
def resolve_predictions(horizon: int = Query(default=5, ge=1, le=60)) -> dict[str, Any]:
    return PredictionLedger(horizon=horizon).resolve_due(get_market_data())


# ---------------------------------------------------------------------- the virtual account
@router.get("/paper/account", dependencies=[Depends(require_read)])
def paper_account() -> dict[str, Any]:
    return get_paper_service().account()


@router.post("/paper/orders", dependencies=[Depends(require_admin), Depends(rate_limit("paper_orders", 30))])
def paper_order(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    allowed = {"side", "symbol", "quantity", "notional", "client_order_id", "simulation"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValidationFailed(f"Unknown fields: {', '.join(unknown)}.", details={"allowed": sorted(allowed)})
    return get_paper_service().submit_order(
        side=payload.get("side", ""), symbol=payload.get("symbol", ""), quantity=payload.get("quantity"), notional=payload.get("notional"),
        client_order_id=payload.get("client_order_id"), simulation=bool(payload.get("simulation", False)),
    )


@router.get("/paper/orders", dependencies=[Depends(require_read)])
def paper_orders(limit: int = Query(default=50, ge=1, le=500), include_rejected: bool = Query(default=True)) -> dict[str, Any]:
    return {"orders": get_paper_service().history(limit=limit, include_rejected=include_rejected)}


@router.get("/paper/performance", dependencies=[Depends(require_read)])
def paper_performance() -> dict[str, Any]:
    return get_paper_service().performance()


@router.post("/paper/pause", dependencies=[Depends(require_admin)])
def paper_pause(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    return get_paper_service().set_paused(bool(payload.get("paused", True)), reason=str(payload.get("reason", "")))


@router.post("/paper/reset", dependencies=[Depends(require_admin)])
def paper_reset(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    return get_paper_service().reset(reason=str(payload.get("reason", "")))


# ---------------------------------------------------------------------- the tool layer
@router.get("/tools", dependencies=[Depends(require_read)])
def tools() -> dict[str, Any]:
    return {"tools": list_tools(), "note": "The same catalogue the MCP server (backend/mcp_server.py) exposes."}


@router.post("/tools/{name}", dependencies=[Depends(require_read), Depends(rate_limit("tools", 60))])
def tool_call(name: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    from app.agent_tools.tools import TOOLS
    from app.core.security import authorize
    from fastapi import Request  # noqa: F401

    tool = TOOLS.get(name)
    if tool is not None and tool.permission in {"paper_trade", "admin"}:
        # state-changing tools need the admin scope even through the generic tool route
        raise ValidationFailed(f"{name} changes state; call its own route (for orders, POST /api/v2/paper/orders) with the admin token.")
    return call_tool(name, payload.get("arguments", payload))


# ---------------------------------------------------------------------- health
@router.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.env,
        "capabilities": settings.capabilities(),
        "forecast": get_prediction_service().status(),
        "banner": "SIMULATED TRADING — NO REAL MONEY",
        "note": "capabilities show what is configured; /api/v2/health/providers checks what actually answers.",
    }


@router.get("/health/providers", dependencies=[Depends(require_read), Depends(rate_limit("provider_probe", 4))])
def health_providers() -> dict[str, Any]:
    """Make one small live request per provider and report the outcome. No secrets are returned.

    The language model is not called here (a hosted model may charge); use
    ``scripts/check_providers.py --include-llm`` for that.
    """
    from app.marketdata.probe import probe_providers

    return probe_providers(include_llm=False)
