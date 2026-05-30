from __future__ import annotations

from typing import Any

DISCLAIMER = "Probability-based market simulations and AI-generated financial intelligence, not financial advice."


def _string(default: str, description: str) -> dict[str, Any]:
    return {"type": "string", "default": default, "description": description}


def _integer(default: int, description: str, minimum: int | None = None, maximum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer", "default": default, "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _number(default: float, description: str) -> dict[str, Any]:
    return {"type": "number", "default": default, "description": description}


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "get_live_quote": {
        "description": "Return the latest quote for a ticker.",
        "input_schema": object_schema({"ticker": _string("AAPL", "Ticker symbol.")}, ["ticker"]),
    },
    "get_candles": {
        "description": "Return OHLCV candles for a ticker, period, and interval.",
        "input_schema": object_schema({
            "ticker": _string("AAPL", "Ticker symbol."),
            "period": _string("1y", "History period such as 5d, 1mo, 6mo, 1y."),
            "interval": _string("1d", "Candle interval such as 1m, 5m, 15m, 1h, 1d."),
            "limit": _integer(120, "Maximum candles returned.", 1, 500),
        }, ["ticker"]),
    },
    "calculate_indicators": {
        "description": "Return MarketVision technical indicator analysis for a ticker.",
        "input_schema": object_schema({
            "ticker": _string("AAPL", "Ticker symbol."),
            "period": _string("6mo", "Analysis period."),
            "interval": _string("1d", "Candle interval."),
        }, ["ticker"]),
    },
    "predict_market": {
        "description": "Return compact bullish/bearish/sideways probabilities, confidence, risk, and quality label.",
        "input_schema": object_schema({
            "ticker": _string("NVDA", "Ticker symbol."),
            "period": _string("1y", "Prediction context period."),
            "interval": _string("1d", "Prediction interval."),
            "horizon_steps": _integer(12, "Forecast horizon in candles.", 4, 60),
        }, ["ticker"]),
    },
    "predict_market_scenario": {
        "description": "Return the full MarketVision probability simulation response.",
        "input_schema": object_schema({
            "ticker": _string("NVDA", "Ticker symbol."),
            "period": _string("1y", "Prediction context period."),
            "interval": _string("1d", "Prediction interval."),
            "horizon_steps": _integer(12, "Forecast horizon in candles.", 4, 60),
        }, ["ticker"]),
    },
    "run_scanner": {
        "description": "Run ranked scanner over a custom ticker list or supported universe.",
        "input_schema": object_schema({
            "universe": _string("mega_cap_ai", "Universe name: custom, mega_cap_ai, nasdaq_100_core, sp500_sample."),
            "tickers": _string("", "Comma-separated tickers when universe is custom."),
            "period": _string("5d", "Scanner period."),
            "interval": _string("1d", "Scanner interval."),
            "horizon_steps": _integer(12, "Forecast horizon in candles.", 4, 60),
            "max_symbols": _integer(10, "Maximum symbols to scan from the selected universe.", 1, 25),
        }),
    },
    "run_similarity": {
        "description": "Return historical similarity score and outcome distribution for a ticker.",
        "input_schema": object_schema({
            "ticker": _string("TSLA", "Ticker symbol."),
            "horizon_steps": _integer(12, "Forward horizon in candles.", 4, 60),
        }, ["ticker"]),
    },
    "run_historical_similarity": {
        "description": "Return detailed historical similarity setups for a ticker.",
        "input_schema": object_schema({
            "ticker": _string("TSLA", "Ticker symbol."),
            "horizon_steps": _integer(12, "Forward horizon in candles.", 4, 60),
        }, ["ticker"]),
    },
    "run_monte_carlo": {
        "description": "Return chart-ready Monte Carlo paths, confidence band, and volatility cone.",
        "input_schema": object_schema({
            "ticker": _string("NVDA", "Ticker symbol."),
            "period": _string("1y", "Prediction context period."),
            "interval": _string("1d", "Prediction interval."),
            "horizon_steps": _integer(12, "Forecast horizon in candles.", 4, 60),
        }, ["ticker"]),
    },
    "analyze_portfolio": {
        "description": "Return portfolio-level risk, exposure, correlation, and AI watchlist alerts.",
        "input_schema": object_schema({
            "holdings": {
                "type": "array",
                "description": "Holdings with ticker, shares, and optional average_cost.",
                "items": object_schema({
                    "ticker": _string("AAPL", "Ticker symbol."),
                    "shares": _number(1, "Share quantity."),
                    "average_cost": {"type": ["number", "null"], "description": "Optional cost basis."},
                }, ["ticker", "shares"]),
                "default": [{"ticker": "AAPL", "shares": 1}],
            },
            "period": _string("1y", "Portfolio analysis period."),
        }, ["holdings"]),
    },
    "explain_prediction": {
        "description": "Return a fact-grounded explanation for a MarketVision prediction.",
        "input_schema": object_schema({
            "ticker": _string("NVDA", "Ticker symbol."),
            "period": _string("1y", "Prediction context period."),
            "interval": _string("1d", "Prediction interval."),
            "horizon_steps": _integer(12, "Forecast horizon in candles.", 4, 60),
        }, ["ticker"]),
    },
    "get_performance": {
        "description": "Return signal ledger, calibration, and model drift performance metrics.",
        "input_schema": object_schema({}),
    },
    "get_performance_metrics": {
        "description": "Return signal ledger, calibration, and model drift performance metrics.",
        "input_schema": object_schema({}),
    },
    "get_macro_regime": {
        "description": "Return current macro regime scores from FRED-backed macro intelligence.",
        "input_schema": object_schema({}),
    },
    "get_sentiment": {
        "description": "Return recent news sentiment for a ticker.",
        "input_schema": object_schema({
            "ticker": _string("NVDA", "Ticker symbol."),
            "limit": _integer(10, "Number of headlines to analyze.", 1, 20),
        }, ["ticker"]),
    },
    "get_trade_candidates": {
        "description": "Return promoted or ranked trade candidates from the MarketVision scanner.",
        "input_schema": object_schema({
            "universe": _string("mega_cap_ai", "Universe name."),
            "tickers": _string("", "Comma-separated tickers when universe is custom."),
            "period": _string("5d", "Scanner period."),
            "interval": _string("1d", "Scanner interval."),
            "horizon_steps": _integer(12, "Forecast horizon in candles.", 4, 60),
            "limit": _integer(5, "Maximum candidates returned.", 1, 25),
            "max_symbols": _integer(10, "Maximum symbols to scan from the selected universe.", 1, 25),
        }),
    },
}


def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["input_schema"],
            "returns": "structured_json",
        }
        for name, spec in TOOL_CATALOG.items()
    ]
