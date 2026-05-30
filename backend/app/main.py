from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from app.schemas import PortfolioAnalysisRequest, PredictionRequest, TickerListRequest, ToolCallRequest
from app.services.copilot_service import CopilotService
from app.services.advanced_intelligence import AdvancedMarketIntelligenceEngine
from app.services.finnhub_service import FinnhubService
from app.services.finbert_service import FinBertSentimentService
from app.services.backtest_service import run_backtest, STRATEGY_LABELS
from app.services.fred_service import FredService
from app.services.similarity_service import find_similar_setups
from app.services.market import MarketDataService
from app.services.market import normalize_history
from app.services.model_service import ModelService
from app.services.newsapi_service import NewsApiService
from app.services.portfolio import PortfolioService
from app.services.signal_ledger import record_signal, resolve_pending_signals, summarize_signals
from app.services.simulation import SimulationService

app = FastAPI(
    title="AI Stock Analysis MVP",
    version="0.1.0",
    description=(
        "A practical stock research and portfolio risk API built for investor support. "
        "Responses are informational and not personalized investment advice."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

market_service = MarketDataService()
portfolio_service = PortfolioService()
model_service = ModelService()
finnhub_service = FinnhubService()
copilot_service = CopilotService()
fred_service = FredService()
newsapi_service = NewsApiService()
finbert_service = FinBertSentimentService()
simulation_service = SimulationService(market_service=market_service, model_service=model_service)
advanced_engine = AdvancedMarketIntelligenceEngine(
    market_service=market_service,
    fred_service=fred_service,
    model_service=model_service,
    finnhub_service=finnhub_service,
    newsapi_service=newsapi_service,
    sentiment_service=finbert_service,
)
_SCAN_CACHE: dict[str, tuple[float, dict]] = {}
_SCAN_CACHE_TTL_SECONDS = 120
SCAN_UNIVERSES = {
    "mega_cap_ai": "AAPL,NVDA,MSFT,AMZN,META,GOOGL,TSLA,AVGO,AMD,CRM,PLTR,SMCI,NFLX,ORCL,ADBE",
    "nasdaq_100_core": "AAPL,NVDA,MSFT,AMZN,META,GOOGL,GOOG,AVGO,TSLA,COST,NFLX,AMD,PEP,ADBE,LIN,CSCO,TMUS,INTU,AMAT,ISRG",
    "sp500_sample": "AAPL,MSFT,NVDA,AMZN,META,GOOGL,BRK-B,LLY,JPM,AVGO,XOM,TSLA,UNH,V,MA,COST,PG,HD,NFLX,JNJ",
}

TOOL_DEFINITIONS = {
    "get_live_quote": "Return latest quote for a ticker.",
    "get_candles": "Return OHLCV candles for a ticker, period, and interval.",
    "calculate_indicators": "Return stock overview/analysis indicators.",
    "predict_market_scenario": "Return MarketVision probability simulation.",
    "run_scanner": "Run ranked bot scanner over tickers.",
    "run_historical_similarity": "Return similar historical setups.",
    "run_monte_carlo": "Return chart-ready Monte Carlo paths.",
    "get_performance_metrics": "Return signal ledger, calibration, and drift metrics.",
    "explain_prediction": "Return fact-grounded prediction explanation.",
    "analyze_portfolio": "Return portfolio intelligence.",
}


def _record_bot_signal(ticker: str, simulation: dict, source: str) -> None:
    bot = simulation.get("ai_signal_bot") or {}
    final_signal = simulation.get("final_signal") or {}
    trade_plan = bot.get("trade_plan") or {}
    advanced = simulation.get("advanced_intelligence") or {}
    macro = advanced.get("macro") or {}
    sentiment = advanced.get("sentiment") or {}
    record_signal({
        "source": source,
        "ticker": ticker.upper(),
        "model_version": simulation.get("model_version"),
        "period": simulation.get("period"),
        "interval": simulation.get("interval"),
        "horizon_steps": simulation.get("horizon_steps"),
        "quality_label": bot.get("quality_label") or simulation.get("quality_label"),
        "coverage_level": bot.get("coverage_level") or simulation.get("coverage_level"),
        "action": bot.get("action"),
        "all_gates_passed": bot.get("all_gates_passed", False),
        "failed_gates": bot.get("failed_gates", []),
        "confidence": bot.get("confidence"),
        "risk_score": bot.get("risk_score"),
        "expected_return": bot.get("expected_return"),
        "dominant_scenario": bot.get("dominant_scenario"),
        "final_signal_probability": final_signal.get("probability"),
        "macro_regime": macro.get("macro_regime_label"),
        "sentiment_score": sentiment.get("score"),
        "entry_price": trade_plan.get("entry_price"),
        "stop_loss": trade_plan.get("stop_loss"),
        "target_1": trade_plan.get("target_1"),
        "target_2": trade_plan.get("target_2"),
        "risk_reward_target_1": trade_plan.get("risk_reward_target_1"),
        "outcome": "pending",
        "disclaimer": simulation.get("disclaimer"),
    })


@app.get("/")
def root() -> dict:
    return {
        "name": "AI Stock Analysis MVP",
        "message": "Stock research and portfolio risk API for investor support.",
        "docs": "/docs",
    }


@app.get("/health")
def healthcheck() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "ml_model_loaded": model_service.is_loaded,
        "ml_model_version": model_service.model_version or "none",
        "ml_auxiliary_models_loaded": model_service.has_auxiliary_models,
        "finnhub_connected": finnhub_service.is_available(),
        "copilot_connected": copilot_service.is_available(),
        "fred_connected": fred_service.is_available(),
        "newsapi_connected": newsapi_service.is_available(),
        "finbert_sentiment_available": finbert_service.is_available(),
    }


@app.get("/api/tools")
def list_marketvision_tools() -> dict:
    return {
        "tools": [
            {"name": name, "description": description, "returns": "structured_json"}
            for name, description in TOOL_DEFINITIONS.items()
        ],
        "disclaimer": "MarketVision tools return probability-based market simulations and AI-generated financial intelligence, not financial advice.",
    }


@app.post("/api/tools/{tool_name}")
def call_marketvision_tool(tool_name: str, request: ToolCallRequest) -> dict:
    args = request.arguments
    ticker = str(args.get("ticker", "AAPL")).upper()
    period = str(args.get("period", "1y"))
    interval = str(args.get("interval", "1d"))
    horizon_steps = int(args.get("horizon_steps", 12))

    if tool_name == "get_live_quote":
        return {"tool": tool_name, "result": stock_quote(ticker)}
    if tool_name == "get_candles":
        history = market_service.get_history(ticker=ticker, period=period, interval=interval)
        return {"tool": tool_name, "result": {"ticker": ticker, "period": period, "interval": interval, "candles": normalize_history(history, limit=int(args.get("limit", 120)))}}
    if tool_name == "calculate_indicators":
        return {"tool": tool_name, "result": market_service.build_stock_analysis(ticker=ticker, period=period, interval=interval)}
    if tool_name == "predict_market_scenario":
        return {"tool": tool_name, "result": advanced_engine.build_simulation_response(ticker=ticker, period=period, interval=interval, horizon_steps=horizon_steps)}
    if tool_name == "run_scanner":
        return {"tool": tool_name, "result": scan_signal_bot(tickers=str(args.get("tickers", ticker)), period=period, interval=interval, horizon_steps=horizon_steps)}
    if tool_name == "run_historical_similarity":
        return {"tool": tool_name, "result": bot_historical_similarity(ticker=ticker, horizon_steps=horizon_steps)}
    if tool_name == "run_monte_carlo":
        simulation = advanced_engine.build_simulation_response(ticker=ticker, period=period, interval=interval, horizon_steps=horizon_steps)
        return {"tool": tool_name, "result": simulation.get("monte_carlo_chart")}
    if tool_name == "get_performance_metrics":
        return {"tool": tool_name, "result": bot_performance()}
    if tool_name == "explain_prediction":
        simulation = advanced_engine.build_simulation_response(ticker=ticker, period=period, interval=interval, horizon_steps=horizon_steps)
        return {"tool": tool_name, "result": copilot_service.explain_prediction_facts(simulation)}
    if tool_name == "analyze_portfolio":
        holdings = args.get("holdings") or [{"ticker": ticker, "shares": float(args.get("shares", 1))}]
        return {"tool": tool_name, "result": portfolio_service.analyze_portfolio(holdings=holdings, period=period)}

    raise HTTPException(status_code=404, detail=f"Unknown MarketVision tool: {tool_name}")


@app.get("/api/market/overview")
def market_overview() -> dict:
    return market_service.get_market_overview()


@app.get("/api/stocks/{ticker}/overview")
def stock_overview(
    ticker: str,
    period: str = Query(default="6mo"),
    interval: str = Query(default="1d"),
) -> dict:
    return market_service.build_stock_overview(ticker=ticker, period=period, interval=interval)


@app.get("/api/stocks/{ticker}/analysis")
def stock_analysis(
    ticker: str,
    period: str = Query(default="6mo"),
    interval: str = Query(default="1d"),
) -> dict:
    return market_service.build_stock_analysis(ticker=ticker, period=period, interval=interval)


@app.get("/api/stocks/{ticker}/simulation")
def stock_simulation(
    ticker: str,
    period: str = Query(default="1y"),
    interval: str = Query(default="1d"),
    horizon_steps: int = Query(default=12, ge=4, le=60),
) -> dict:
    return simulation_service.build_prediction_simulation(
        ticker=ticker,
        period=period,
        interval=interval,
        horizon_steps=horizon_steps,
    )


@app.get("/api/stocks/{ticker}/intelligence")
def stock_intelligence(
    ticker: str,
    horizon_steps: int = Query(default=12, ge=4, le=60),
    include_intraday: bool = Query(default=True),
) -> dict:
    return advanced_engine.analyze(
        ticker=ticker,
        horizon_steps=horizon_steps,
        include_intraday=include_intraday,
    )


@app.get("/api/stocks/{ticker}/bot")
def stock_signal_bot(
    ticker: str,
    period: str = Query(default="1y"),
    interval: str = Query(default="1d"),
    horizon_steps: int = Query(default=12, ge=4, le=60),
) -> dict:
    simulation = advanced_engine.build_simulation_response(
        ticker=ticker,
        period=period,
        interval=interval,
        horizon_steps=horizon_steps,
    )
    _record_bot_signal(ticker, simulation, source="single_bot")
    return {
        "ticker": ticker.upper(),
        "bot": simulation.get("ai_signal_bot"),
        "quality_label": simulation.get("quality_label"),
        "coverage_level": simulation.get("coverage_level"),
        "final_signal": simulation.get("final_signal"),
        "probabilities": simulation.get("probabilities"),
        "predicted_candles": simulation.get("predicted_candles"),
        "disclaimer": simulation.get("disclaimer"),
        "as_of": simulation.get("as_of"),
    }


@app.get("/api/bot/scan")
def scan_signal_bot(
    tickers: str = Query(default="AAPL,NVDA,MSFT,AMZN,META,GOOGL,TSLA,AVGO,AMD,CRM"),
    universe: str = Query(default="custom"),
    period: str = Query(default="5d"),
    interval: str = Query(default="1d"),
    horizon_steps: int = Query(default=12, ge=4, le=60),
) -> dict:
    if universe != "custom":
        tickers = SCAN_UNIVERSES.get(universe, tickers)
    cache_key = f"{universe}|{tickers}|{period}|{interval}|{horizon_steps}"
    cached = _SCAN_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached[0] <= _SCAN_CACHE_TTL_SECONDS:
        return {**cached[1], "cache": {"hit": True, "ttl_seconds": _SCAN_CACHE_TTL_SECONDS}}

    symbols = [symbol.strip().upper() for symbol in tickers.split(",") if symbol.strip()]
    results = []
    for symbol in symbols[:25]:
        try:
            simulation = advanced_engine.build_simulation_response(
                ticker=symbol,
                period=period,
                interval=interval,
                horizon_steps=horizon_steps,
            )
            _record_bot_signal(symbol, simulation, source="scanner")
            bot = simulation.get("ai_signal_bot") or {}
            advanced = simulation.get("advanced_intelligence") or {}
            final_signal = simulation.get("final_signal") or {}
            trade_plan = bot.get("trade_plan") or {}
            historical_similarity = advanced.get("historical_similarity") or {}
            sentiment = advanced.get("sentiment") or {}
            macro = advanced.get("macro") or {}
            dominant = bot.get("dominant_scenario") or simulation.get("dominant_scenario")
            failed_gates = list(bot.get("failed_gates", []))
            final_probability = final_signal.get("probability")
            similarity_strength = historical_similarity.get("average_similarity_score")
            sentiment_score = sentiment.get("score")
            macro_regime = macro.get("macro_regime_label")
            if final_probability is not None and float(final_probability) < 0.65 and "confidence_failed" not in failed_gates:
                failed_gates.append("confidence_failed")
            if bot.get("risk_score") is not None and float(bot.get("risk_score") or 0) >= 70 and "risk_failed" not in failed_gates:
                failed_gates.append("risk_failed")
            if trade_plan.get("risk_reward_target_1") is not None and float(trade_plan.get("risk_reward_target_1") or 0) < 1.25 and "rr_failed" not in failed_gates:
                failed_gates.append("rr_failed")
            if sentiment_score is not None and float(sentiment_score) < -0.2 and "sentiment_failed" not in failed_gates:
                failed_gates.append("sentiment_failed")
            if similarity_strength is not None and float(similarity_strength) < 0.55 and "weak_historical_similarity" not in failed_gates:
                failed_gates.append("weak_historical_similarity")
            if macro_regime in {"risk_off", "restrictive_rates"} and dominant == "bullish" and "macro_conflict" not in failed_gates:
                failed_gates.append("macro_conflict")
            results.append({
                "ticker": symbol,
                "quality_label": bot.get("quality_label"),
                "coverage_level": bot.get("coverage_level"),
                "action": bot.get("action"),
                "all_gates_passed": bot.get("all_gates_passed", False),
                "failed_gates": failed_gates,
                "eligible_for_paper_trade": trade_plan.get("eligible_for_paper_trade", False),
                "confidence": bot.get("confidence"),
                "risk_score": bot.get("risk_score"),
                "expected_return": bot.get("expected_return"),
                "final_signal_probability": final_probability,
                "risk_reward_ratio": trade_plan.get("risk_reward_target_1"),
                "historical_similarity_strength": similarity_strength,
                "sentiment_score": sentiment_score,
                "macro_regime": macro_regime,
                "macro_compatible": "macro_conflict" not in failed_gates,
                "intraday_vwap": advanced.get("intraday_vwap"),
                "gates": bot.get("gates"),
                "trade_plan": trade_plan,
            })
        except Exception as exc:
            results.append({"ticker": symbol, "error": str(exc), "all_gates_passed": False})
    quality_order = {
        "high_confidence_trade_candidate": 0,
        "watchlist_candidate": 1,
        "prediction_only": 2,
        "avoid_high_risk": 3,
        "insufficient_data": 4,
    }
    results.sort(key=lambda item: (
        quality_order.get(str(item.get("quality_label")), 9),
        -float(item.get("final_signal_probability") or 0),
        -float(item.get("confidence") or 0),
        float(item.get("risk_score") or 100),
        -float(item.get("risk_reward_ratio") or 0),
        -float(item.get("historical_similarity_strength") or 0),
        -float(item.get("sentiment_score") or 0),
        0 if item.get("macro_compatible", True) else 1,
    ))
    passed = [item for item in results if item.get("all_gates_passed")]
    buckets: dict[str, int] = {}
    for item in results:
        label = str(item.get("quality_label") or "error")
        buckets[label] = buckets.get(label, 0) + 1
    payload = {
        "period": period,
        "interval": interval,
        "universe": universe,
        "available_universes": sorted(SCAN_UNIVERSES.keys()),
        "scanned": len(results),
        "quality_buckets": buckets,
        "passed": passed,
        "results": results,
        "cache": {"hit": False, "ttl_seconds": _SCAN_CACHE_TTL_SECONDS},
        "disclaimer": "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
    }
    _SCAN_CACHE[cache_key] = (now, payload)
    return payload


@app.get("/api/bot/performance")
def bot_performance() -> dict:
    summary = summarize_signals()
    return {
        **summary,
        "v12_walk_forward_proof": model_service.final_hybrid_policy.get("promoted_signal_report", {}),
        "accuracy_policy": model_service.final_hybrid_policy.get("accuracy_claim_policy"),
        "disclaimer": "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
    }


@app.get("/api/bot/historical-similarity")
def bot_historical_similarity(
    ticker: str = Query(..., min_length=1),
    horizon_steps: int = Query(default=12, ge=4, le=60),
) -> dict:
    analysis = advanced_engine.analyze(
        ticker=ticker,
        horizon_steps=horizon_steps,
        include_intraday=True,
    )
    similarity = analysis.get("historical_similarity") or {}
    probabilities = similarity.get("outcome_probabilities") or {}
    setups = similarity.get("similar_setups") or []
    return {
        "ticker": ticker.upper(),
        "available": bool(similarity.get("available")),
        "current_setup_features": similarity.get("current_setup_features", {}),
        "top_similar_historical_dates": setups,
        "similarity_score": similarity.get("average_similarity_score"),
        "average_future_return_pct": similarity.get("average_future_return_pct"),
        "bullish_outcome_pct": probabilities.get("bullish"),
        "bearish_outcome_pct": probabilities.get("bearish"),
        "sideways_outcome_pct": probabilities.get("sideways"),
        "best_case_pct": similarity.get("best_case_pct"),
        "worst_case_pct": similarity.get("worst_case_pct"),
        "average_drawdown_pct": similarity.get("average_drawdown_pct"),
        "reason": similarity.get("reason"),
        "source": "advanced_market_intelligence_historical_similarity",
        "disclaimer": "Probability-based historical comparison for research, not financial advice.",
    }


@app.post("/api/bot/explain")
def explain_bot_prediction(request: PredictionRequest) -> dict:
    simulation = advanced_engine.build_simulation_response(
        ticker=request.ticker,
        period=request.period,
        interval=request.interval,
        horizon_steps=request.horizon_steps,
    )
    explanation = copilot_service.explain_prediction_facts(simulation)
    return {
        "ticker": request.ticker.upper(),
        "quality_label": simulation.get("quality_label"),
        "coverage_level": simulation.get("coverage_level"),
        "final_signal": simulation.get("final_signal"),
        **explanation,
    }


@app.post("/api/bot/resolve-outcomes")
def resolve_bot_outcomes() -> dict:
    result = resolve_pending_signals(market_service=market_service)
    return {
        **result,
        "performance": bot_performance(),
        "disclaimer": "Outcome resolution is for research/paper-trading analytics, not financial advice.",
    }


@app.post("/api/bot/portfolio-overlay")
def bot_portfolio_overlay(request: PortfolioAnalysisRequest) -> dict:
    holdings = [{"ticker": item.ticker.upper(), "shares": item.shares} for item in request.holdings]
    portfolio = portfolio_service.analyze_portfolio(holdings=holdings, period=request.period)
    tickers = [item["ticker"] for item in holdings]
    scan = scan_signal_bot(tickers=",".join(tickers), period="5d", interval="1d", horizon_steps=12)
    weights = {h["ticker"]: h["weight_pct"] for h in portfolio.get("holdings", [])}
    overlays = []
    for row in scan.get("results", []):
        ticker = row.get("ticker")
        weight = float(weights.get(ticker, 0) or 0)
        risk_score = float(row.get("risk_score") or 0)
        quality = row.get("quality_label")
        overlays.append({
            "ticker": ticker,
            "portfolio_weight_pct": weight,
            "scanner_quality": quality,
            "scanner_action": row.get("action"),
            "risk_score": risk_score,
            "exposure_warning": weight > 30 and quality == "avoid_high_risk",
            "candidate_add_warning": weight > 25 and quality == "high_confidence_trade_candidate",
            "failed_gates": row.get("failed_gates", []),
        })
    return {
        "portfolio": portfolio,
        "scanner_overlay": overlays,
        "scan_quality_buckets": scan.get("quality_buckets", {}),
        "warnings": portfolio.get("warnings", []),
        "disclaimer": "Portfolio overlay is risk intelligence for research, not personalized financial advice.",
    }


@app.get("/api/stocks/{ticker}/news")
def stock_news(ticker: str, limit: int = Query(default=5, ge=1, le=20)) -> dict:
    # Use Finnhub when available — includes sentiment scores
    if finnhub_service.is_available():
        items = finnhub_service.get_news(ticker=ticker, limit=limit)
        if items:
            return {"ticker": ticker.upper(), "source": "finnhub", "sentiment_model": "finbert_optional", "items": finbert_service.enrich_items(items)}
    # Fallback to yfinance
    return {"ticker": ticker.upper(), "source": "yfinance", "items": market_service.get_news(ticker=ticker, limit=limit)}


@app.get("/api/stocks/{ticker}/profile")
def stock_profile(ticker: str) -> dict:
    profile = finnhub_service.get_company_profile(ticker) or {}
    financials = finnhub_service.get_basic_financials(ticker) or {}
    recommendations = finnhub_service.get_recommendation_trends(ticker)
    earnings = finnhub_service.get_earnings_calendar(ticker)
    return {
        "ticker": ticker.upper(),
        "profile": profile,
        "financials": financials,
        "analyst_recommendations": recommendations,
        "next_earnings": earnings,
    }


@app.get("/api/stocks/{ticker}/quote")
def stock_quote(ticker: str) -> dict:
    quote = finnhub_service.get_quote(ticker)
    if not quote:
        overview = market_service.build_stock_overview(ticker=ticker, period="5d", interval="1d")
        return {"ticker": ticker.upper(), "source": "yfinance", "quote": {
            "current_price": overview["current_price"],
            "change_pct": overview["daily_change_pct"],
            "prev_close": overview["previous_close"],
        }}
    return {"ticker": ticker.upper(), "source": "finnhub", "quote": quote}


@app.get("/api/market/news")
def market_news(category: str = Query(default="general"), limit: int = Query(default=8, ge=1, le=20)) -> dict:
    # Prefer NewsAPI business headlines; fallback to Finnhub
    if newsapi_service.is_available():
        items = newsapi_service.get_market_headlines(limit=limit)
        if items:
            return {"category": category, "source": "newsapi", "sentiment_model": "finbert_optional", "items": finbert_service.enrich_items(items)}
    items = finnhub_service.get_market_news(category=category, limit=limit)
    return {"category": category, "source": "finnhub", "sentiment_model": "finbert_optional", "items": finbert_service.enrich_items(items)}


@app.get("/api/market/macro")
def market_macro() -> dict:
    if not fred_service.is_available():
        return {"available": False, "message": "FRED API key not configured"}
    return {"available": True, "source": "fred", **fred_service.get_market_conditions()}


@app.get("/api/market/macro/snapshot")
def macro_snapshot() -> dict:
    if not fred_service.is_available():
        return {"available": False, "message": "FRED API key not configured"}
    return fred_service.get_macro_intelligence()


@app.get("/api/market/macro/series/{series_id}")
def macro_series(series_id: str, limit: int = Query(default=12, ge=1, le=60)) -> dict:
    if not fred_service.is_available():
        return {"available": False, "message": "FRED API key not configured"}
    return fred_service.get_indicator(series_id=series_id, limit=limit)


@app.get("/api/backtest/strategies")
def backtest_strategies() -> dict:
    return {"strategies": [{"id": k, "label": v} for k, v in STRATEGY_LABELS.items()]}


@app.get("/api/backtest/{ticker}")
def backtest(
    ticker: str,
    strategy: str = Query(default="sma_crossover"),
    period: str = Query(default="3y"),
    capital: float = Query(default=10_000.0, ge=100),
) -> dict:
    return run_backtest(ticker=ticker, strategy=strategy, period=period, initial_capital=capital)


@app.get("/api/stocks/{ticker}/similarity")
def stock_similarity(ticker: str, top_n: int = Query(default=5, ge=1, le=10)) -> dict:
    return find_similar_setups(ticker=ticker, top_n=top_n)


@app.get("/api/news/search")
def news_search(q: str = Query(..., min_length=2), limit: int = Query(default=10, ge=1, le=20)) -> dict:
    if newsapi_service.is_available():
        items = newsapi_service.search_news(query=q, limit=limit)
        return {"query": q, "source": "newsapi", "sentiment_model": "finbert_optional", "items": finbert_service.enrich_items(items)}
    return {"query": q, "source": "none", "items": []}


@app.get("/api/stocks/{ticker}/news/extended")
def stock_news_extended(ticker: str, limit: int = Query(default=10, ge=1, le=20)) -> dict:
    """Combined news from Finnhub + NewsAPI, deduped by title."""
    finnhub_items = []
    newsapi_items = []

    if finnhub_service.is_available():
        finnhub_items = finnhub_service.get_news(ticker=ticker, limit=limit)

    if newsapi_service.is_available():
        newsapi_items = newsapi_service.get_stock_news(ticker=ticker, limit=limit)

    seen: set[str] = set()
    combined = []
    for item in finnhub_items + newsapi_items:
        title = (item.get("title") or "").strip().lower()
        if title and title not in seen:
            seen.add(title)
            combined.append(item)

    combined.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return {
        "ticker": ticker.upper(),
        "sources": ["finnhub", "newsapi"],
        "sentiment_model": "finbert_optional",
        "sentiment_summary": finbert_service.aggregate(combined[:limit], limit=limit),
        "items": finbert_service.enrich_items(combined[:limit]),
    }


@app.post("/api/stocks/compare")
def compare_stocks(request: TickerListRequest) -> dict:
    return market_service.compare_stocks(
        tickers=request.tickers,
        period=request.period,
        interval=request.interval,
    )


@app.post("/api/stocks/watchlist")
def build_watchlist(request: TickerListRequest) -> dict:
    return market_service.build_watchlist(
        tickers=request.tickers,
        period=request.period,
        interval=request.interval,
    )


@app.post("/api/predict")
def predict_stock(request: PredictionRequest) -> dict:
    try:
        return advanced_engine.build_simulation_response(
            ticker=request.ticker,
            period=request.period,
            interval=request.interval,
            horizon_steps=request.horizon_steps,
        )
    except Exception:
        return simulation_service.build_prediction_simulation(
            ticker=request.ticker,
            period=request.period,
            interval=request.interval,
            horizon_steps=request.horizon_steps,
        )


@app.post("/api/portfolio/analyze")
def analyze_portfolio(request: PortfolioAnalysisRequest) -> dict:
    payload = [{"ticker": item.ticker, "shares": item.shares, "average_cost": item.average_cost} for item in request.holdings]
    return portfolio_service.analyze_portfolio(holdings=payload, period=request.period)


class CopilotChatRequest(BaseModel):
    message: str
    context: dict | None = None
    history: list[dict] | None = None


@app.post("/api/copilot/chat")
def copilot_chat(request: CopilotChatRequest) -> dict:
    reply = copilot_service.chat(
        message=request.message,
        context=request.context,
        history=request.history,
    )
    return {"reply": reply, "model": "gpt-4o-mini"}


@app.get("/api/copilot/explain/{ticker}")
def copilot_explain(ticker: str, period: str = Query(default="1y")) -> dict:
    simulation = simulation_service.build_prediction_simulation(ticker=ticker, period=period)
    explanation = copilot_service.explain_simulation(simulation)
    news = finnhub_service.get_news(ticker, limit=5) if finnhub_service.is_available() else []
    sentiment_summary = copilot_service.summarize_news_sentiment(ticker, news)
    return {
        "ticker": ticker.upper(),
        "explanation": explanation,
        "sentiment_summary": sentiment_summary,
        "simulation_snapshot": {
            "dominant_scenario": simulation["dominant_scenario"],
            "confidence": simulation["confidence"],
            "risk_level": simulation["risk_level"],
            "probabilities": simulation["probabilities"],
        },
    }


@app.get("/api/copilot/analyse/{ticker}")
def copilot_analyse(ticker: str, period: str = Query(default="6mo")) -> dict:
    analysis = market_service.build_stock_analysis(ticker=ticker, period=period)
    explanation = copilot_service.explain_stock_analysis(analysis)
    return {
        "ticker": ticker.upper(),
        "explanation": explanation,
        "analysis_snapshot": {
            "stance": analysis["stance"],
            "score": analysis["score"],
            "current_price": analysis["current_price"],
        },
    }
