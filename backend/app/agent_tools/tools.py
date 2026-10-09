"""The tool layer for AI agents: fifteen validated, structured interfaces.

Each tool declares a JSON Schema for its input, a permission class, and a handler that returns
plain JSON. The same definitions serve the HTTP API (``/api/tools``) and the MCP server, so an
agent sees one catalogue everywhere.

Permissions
    read         research: market data, features, forecasts, history. No state changes.
    paper_trade  changes the simulated account. Requires the ``simulation: true`` argument in the
                 call itself, and the admin scope on the HTTP API.
    admin        resolves predictions, pauses or resets the account.

Nothing in this layer computes a financial number itself: it validates input, calls the same
services the application uses, and returns their output with its provenance. A forecast returned
by a tool never places an order; placing an order is its own tool with its own permission.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from app.core.errors import NotFound, ValidationFailed

READ, PAPER, ADMIN = "read", "paper_trade", "admin"
DISCLAIMER = "Research data and probability estimates from historical sources. Simulated trading only. Not financial advice."


def _string(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


def _symbol() -> dict[str, Any]:
    return _string("Ticker symbol, e.g. AAPL.", pattern="^[A-Za-z0-9.^=-]{1,12}$")


def _integer(description: str, minimum: int, maximum: int, default: int) -> dict[str, Any]:
    return {"type": "integer", "description": description, "minimum": minimum, "maximum": maximum, "default": default}


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    permission: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "permission": self.permission, "inputSchema": self.input_schema, "returns": "structured_json"}


def _validate(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Check ``args`` against the schema: types, required fields, bounds, enums, no extras."""
    if not isinstance(args, dict):
        raise ValidationFailed("Tool arguments must be a JSON object.")
    properties = schema["properties"]
    unknown = sorted(set(args) - set(properties))
    if unknown:
        raise ValidationFailed(f"Unknown arguments: {', '.join(unknown)}.", details={"allowed": sorted(properties)})
    missing = [name for name in schema.get("required", []) if args.get(name) in (None, "")]
    if missing:
        raise ValidationFailed(f"Missing required arguments: {', '.join(missing)}.")
    out: dict[str, Any] = {}
    for name, spec in properties.items():
        if name not in args or args[name] is None:
            if "default" in spec:
                out[name] = spec["default"]
            continue
        value = args[name]
        kind = spec.get("type")
        if kind == "integer":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise ValidationFailed(f"{name} must be a whole number.")
            value = int(value)
            if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
                raise ValidationFailed(f"{name} must be between {spec.get('minimum')} and {spec.get('maximum')}.")
        elif kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationFailed(f"{name} must be a number.")
            value = float(value)
            if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
                raise ValidationFailed(f"{name} must be between {spec.get('minimum')} and {spec.get('maximum')}.")
        elif kind == "boolean":
            if not isinstance(value, bool):
                raise ValidationFailed(f"{name} must be true or false.")
        elif kind == "string":
            if not isinstance(value, str):
                raise ValidationFailed(f"{name} must be a string.")
            if "enum" in spec and value not in spec["enum"]:
                raise ValidationFailed(f"{name} must be one of: {', '.join(spec['enum'])}.")
            if spec.get("format") == "date":
                try:
                    date.fromisoformat(value)
                except ValueError:
                    raise ValidationFailed(f"{name} must be a date like 2026-01-31.") from None
        elif kind == "array":
            if not isinstance(value, list):
                raise ValidationFailed(f"{name} must be a list.")
            if len(value) > spec.get("maxItems", 100):
                raise ValidationFailed(f"{name} accepts at most {spec['maxItems']} items.")
        out[name] = value
    return out


# ---------------------------------------------------------------------- handlers
def _market():
    from app.marketdata.service import get_market_data

    return get_market_data()


def _history(args: dict[str, Any]) -> Any:
    envelope = _market().history(args["symbol"], period=args.get("period"), interval=args.get("interval", "1d"))
    return envelope.to_dict()


def _quote(args: dict[str, Any]) -> Any:
    return _market().quote(args["symbol"]).to_dict()


def _profile(args: dict[str, Any]) -> Any:
    return _market().profile(args["symbol"]).to_dict()


def _news(args: dict[str, Any]) -> Any:
    return _market().news(symbol=args.get("symbol"), query=args.get("query"), limit=args.get("limit", 10)).to_dict()


def _economic(args: dict[str, Any]) -> Any:
    from app.marketdata.providers.fred import SERIES

    series_id = args["series"]
    resolved = SERIES.get(series_id.lower(), series_id.upper())
    envelope = _market().economic_series(resolved)
    out = envelope.to_dict()
    out["data"] = out["data"][-args.get("limit", 24):]
    out["known_series"] = sorted(SERIES)
    return out


def _calendar(args: dict[str, Any]) -> Any:
    start = date.fromisoformat(args["start"])
    end = date.fromisoformat(args["end"])
    return _market().market_calendar(start, end).to_dict()


def _indicators(args: dict[str, Any]) -> Any:
    from app.features import indicators as ind
    from app.features.pipeline import adjusted_close

    envelope = _market().history(args["symbol"], period=args.get("period", "1y"), interval="1d")
    bars = envelope.data
    close = adjusted_close(bars)
    line, signal, hist = ind.macd(close)
    middle, upper, lower = ind.bollinger(close, 20)
    return {
        "symbol": envelope.symbol,
        "as_of": bars.index[-1].date().isoformat(),
        "indicators": {
            "close": ind.latest(close), "sma_20": ind.latest(ind.sma(close, 20)), "sma_50": ind.latest(ind.sma(close, 50)), "sma_200": ind.latest(ind.sma(close, 200)),
            "ema_20": ind.latest(ind.ema(close, 20)), "rsi_14": ind.latest(ind.rsi(close, 14)),
            "macd": {"line": ind.latest(line), "signal": ind.latest(signal), "histogram": ind.latest(hist)},
            "bollinger": {"middle": ind.latest(middle), "upper": ind.latest(upper), "lower": ind.latest(lower)},
            "atr_14": ind.latest(ind.atr(bars["High"], bars["Low"], bars["Close"], 14)),
            "annualized_volatility_30d": ind.annualised_volatility(close), "max_drawdown": ind.max_drawdown(close),
        },
        "note": "Indicators computed from dividend- and split-adjusted closes by the same code the models use.",
        "data_source": envelope.meta(),
    }


def _sentiment(args: dict[str, Any]) -> Any:
    from app.services.finbert_service import FinBertSentimentService

    envelope = _market().news(symbol=args["symbol"], limit=args.get("limit", 10))
    aggregate = FinBertSentimentService().aggregate(envelope.data, limit=args.get("limit", 10))
    return {"symbol": envelope.symbol, "sentiment": aggregate, "data_source": envelope.meta(),
            "note": "Sentiment of recent headlines. It is a description of the news, not a trading signal."}


def _prediction(args: dict[str, Any]) -> Any:
    from app.forecast.service import get_prediction_service
    from app.predictions.ledger import PredictionLedger

    forecast = get_prediction_service().predict(args["symbol"])
    if args.get("record", True) and forecast.get("probability_up") is not None:
        event = PredictionLedger(horizon=int(forecast["horizon_sessions"])).record_issued(forecast)
        forecast["prediction_id"] = event["prediction_id"]
        forecast["recorded"] = True
    return forecast


def _prediction_history(args: dict[str, Any]) -> Any:
    from app.predictions.ledger import PredictionLedger

    ledger = PredictionLedger(horizon=args.get("horizon", 5))
    return {"history": ledger.history(symbol=args.get("symbol"), limit=args.get("limit", 50)), "performance": ledger.performance()}


def _model_status(args: dict[str, Any]) -> Any:
    from app.forecast.service import get_prediction_service

    return get_prediction_service().status()


def _backtest(args: dict[str, Any]) -> Any:
    from app.services.backtest_service import run_backtest

    result = run_backtest(args["symbol"], strategy=args.get("strategy", "sma_crossover"), period=args.get("period", "3y"), initial_capital=float(args.get("initial_capital", 10000)))
    if not result.get("available"):
        raise NotFound(result.get("error", "The backtest could not be run."), details={"symbol": args["symbol"]})
    return result


def _portfolio_risk(args: dict[str, Any]) -> Any:
    from app.services.portfolio import PortfolioService

    holdings = args["holdings"]
    for row in holdings:
        if not isinstance(row, dict) or not row.get("ticker") or not isinstance(row.get("shares"), (int, float)) or row["shares"] <= 0:
            raise ValidationFailed("Each holding needs a ticker and a positive share count.", details={"holding": row})
    return PortfolioService().analyze_portfolio(holdings=holdings, period=args.get("period", "6mo"))


def _paper():
    from app.paper.account import get_paper_service

    return get_paper_service()


def _virtual_account(args: dict[str, Any]) -> Any:
    return _paper().account()


def _submit_order(args: dict[str, Any]) -> Any:
    return _paper().submit_order(
        side=args["side"], symbol=args["symbol"], quantity=args.get("quantity"), notional=args.get("notional"),
        client_order_id=args.get("client_order_id"), simulation=args.get("simulation", False),
    )


def _trade_history(args: dict[str, Any]) -> Any:
    return {"orders": _paper().history(limit=args.get("limit", 50), include_rejected=args.get("include_rejected", True))}


def _portfolio_performance(args: dict[str, Any]) -> Any:
    return _paper().performance()


# ---------------------------------------------------------------------- the catalogue
TOOLS: dict[str, Tool] = {tool.name: tool for tool in [
    Tool("get_stock_history", "Daily or intraday OHLCV bars with provider, freshness and quality checks.", READ,
         _object({"symbol": _symbol(), "period": _string("Look-back such as 5d, 1mo, 6mo, 1y, 5y, max.", default="1y"),
                  "interval": _string("Bar size.", enum=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], default="1d")}, ["symbol"]), _history),
    Tool("get_market_quote", "The latest available price, labelled live, delayed or historical, with its age.", READ,
         _object({"symbol": _symbol()}, ["symbol"]), _quote),
    Tool("get_company_profile", "Company name, sector, industry, market capitalisation and summary.", READ,
         _object({"symbol": _symbol()}, ["symbol"]), _profile),
    Tool("get_financial_news", "Recent headlines for a symbol or a free-text query, with source and timestamps.", READ,
         _object({"symbol": {**_symbol(), "description": "Ticker symbol (optional if query is given)."}, "query": _string("Free-text search instead of a symbol."),
                  "limit": _integer("Number of articles.", 1, 50, 10)}), _news),
    Tool("get_economic_indicators", "A FRED economic series by friendly name (fed_funds_rate, cpi, unemployment, treasury_10y, vix, ...) or FRED id.", READ,
         _object({"series": _string("Series name or FRED id."), "limit": _integer("Observations to return.", 1, 120, 24)}, ["series"]), _economic),
    Tool("get_market_calendar", "Trading sessions, holidays and early closes between two dates (NYSE rules).", READ,
         _object({"start": _string("First date.", format="date"), "end": _string("Last date.", format="date")}, ["start", "end"]), _calendar),
    Tool("get_technical_indicators", "RSI, moving averages, MACD, Bollinger bands, ATR and volatility from adjusted closes.", READ,
         _object({"symbol": _symbol(), "period": _string("Look-back period.", default="1y")}, ["symbol"]), _indicators),
    Tool("get_sentiment_analysis", "Sentiment of recent headlines for a symbol (FinBERT when available, otherwise VADER).", READ,
         _object({"symbol": _symbol(), "limit": _integer("Headlines to analyse.", 1, 20, 10)}, ["symbol"]), _sentiment),
    Tool("get_stock_prediction", "Probability the stock closes higher after the forecast horizon. Served by the promoted model, or the labelled base rate when none has passed the promotion rule. Never authorises a trade.", READ,
         _object({"symbol": _symbol(), "record": {"type": "boolean", "description": "Record the forecast in the prediction ledger (default true).", "default": True}}, ["symbol"]), _prediction),
    Tool("get_prediction_history", "Issued forecasts with their resolved outcomes, and honest performance summaries.", READ,
         _object({"symbol": {**_symbol(), "description": "Only this symbol (optional)."}, "horizon": _integer("Forecast horizon in sessions.", 1, 60, 5),
                  "limit": _integer("Rows to return.", 1, 500, 50)}), _prediction_history),
    Tool("get_model_status", "Which model (if any) serves predictions, its evidence, and the feature schema in force.", READ,
         _object({}), _model_status),
    Tool("run_backtest", "Backtest an indicator strategy with costs, stops and equal-terms baselines. Research, not advice.", READ,
         _object({"symbol": _symbol(), "strategy": _string("Strategy name.", enum=["sma_crossover", "rsi_mean_reversion", "macd", "bollinger_bands", "buy_and_hold"], default="sma_crossover"),
                  "period": _string("Look-back period.", default="3y"), "initial_capital": {"type": "number", "description": "Virtual starting capital.", "minimum": 100, "maximum": 10_000_000, "default": 10000}}, ["symbol"]), _backtest),
    Tool("get_portfolio_risk", "Risk analysis of a list of holdings: weights, volatility, drawdown, correlation, sector exposure.", READ,
         _object({"holdings": {"type": "array", "description": "Holdings, each {ticker, shares, average_cost?}.", "maxItems": 25, "items": {"type": "object"}},
                  "period": _string("History used for the statistics.", default="6mo")}, ["holdings"]), _portfolio_risk),
    Tool("get_virtual_account", "The simulated $50 account: cash, positions, P&L. SIMULATED TRADING — NO REAL MONEY.", READ,
         _object({}), _virtual_account),
    Tool("submit_paper_order", "Simulate a market order in the virtual account. Requires simulation=true. No real money can move.", PAPER,
         _object({"side": _string("buy or sell.", enum=["buy", "sell"]), "symbol": _symbol(),
                  "quantity": {"type": "number", "description": "Shares (fractional allowed). Give this or notional.", "minimum": 0.000001},
                  "notional": {"type": "number", "description": "Dollar amount to spend instead of a share count.", "minimum": 1},
                  "client_order_id": _string("Caller's id; resubmitting with the same id returns the same order."),
                  "simulation": {"type": "boolean", "description": "Must be true: acknowledges this is virtual money.", "default": False}},
                 ["side", "symbol", "simulation"]), _submit_order),
    Tool("get_trade_history", "Orders in the virtual account, filled and rejected, with reasons.", READ,
         _object({"limit": _integer("Rows to return.", 1, 500, 50), "include_rejected": {"type": "boolean", "description": "Include rejected orders.", "default": True}}), _trade_history),
    Tool("get_portfolio_performance", "Equity, realised and unrealised P&L, fees and win rate of the virtual account.", READ,
         _object({}), _portfolio_performance),
]}


def list_tools() -> list[dict[str, Any]]:
    return [tool.describe() for tool in TOOLS.values()]


def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and run one tool. The result wraps the handler's output with its provenance."""
    tool = TOOLS.get(name)
    if tool is None:
        raise NotFound(f"There is no tool called {name}.", details={"tools": sorted(TOOLS)})
    args = _validate(tool.input_schema, arguments or {})
    result = tool.handler(args)
    return {"tool": name, "permission": tool.permission, "result": result, "disclaimer": DISCLAIMER}
