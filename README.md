# MarketVision AI - Stock Simulation & Portfolio Intelligence

## Overview

MarketVision AI is an AI-powered market simulation and decision-intelligence platform. The goal is to help traders and investors understand market probabilities, future chart scenarios, portfolio risk, and research context without pretending to guarantee stock predictions.

The current MVP includes:

- a FastAPI backend that pulls market data from `yfinance`
- research endpoints for single stocks, comparisons, news, simulation, and portfolio risk
- a probability-based future chart simulation engine
- bullish, bearish, sideways, and high-volatility scenario paths
- signal promotion, scanner mode, signal ledger, and outcome resolution for paper-signal proof
- AI-style reasoning summaries with risk notes and safety disclaimers
- a Streamlit dashboard for stock buyers to explore the data visually
- a premium Next.js/Tailwind UI concept in `frontend-next/`
- an ML auto-tuning scaffold in `ml/` for local and Kaggle GPU training

## Current MVP Scope

- Stock overview endpoint with price history and company context
- Stock analysis endpoint with moving averages, RSI, volatility, and a simple research stance
- Future chart simulation endpoint with scenario probabilities and confidence bands
- Stock news endpoint
- Stock comparison endpoint for multiple tickers
- Watchlist ranking endpoint
- Portfolio analysis endpoint with weights, concentration checks, volatility, drawdown, and correlations
- Streamlit dashboard with stock, watchlist, compare, and portfolio tabs
- Healthcheck and test coverage for core API structure

## Project Structure

- `backend/`
  FastAPI app, service layer, tests, and Python dependencies
- `frontend/`
  Streamlit dashboard for stock research and portfolio analysis
- `notebooks/`
  Reserved for experiments and model research
- `PROJECT_PLAN.md`
  Broader product roadmap

## Backend Layout

- `backend/app/main.py`
  FastAPI entrypoint
- `backend/app/services/market.py`
  Stock overview, technical indicators, and news fetching
- `backend/app/services/portfolio.py`
  Portfolio-risk and allocation analysis
- `backend/app/services/simulation.py`
  Probability-based future chart simulation and strategy reasoning
- `backend/tests/`
  API and portfolio tests
- `frontend/app.py`
  Streamlit dashboard entrypoint
- `frontend-next/`
  Premium MarketVision AI command-center UI built with Next.js, TypeScript, Tailwind, Framer Motion, and TradingView Lightweight Charts
- `ml/train_autotune.py`
  Kaggle-ready model tuning pipeline for bullish/bearish/sideways market labels

## Run Locally

Install backend dependencies:

```bash
python3 -m pip install -r backend/requirements.txt
```

Install frontend dependencies:

```bash
python3 -m pip install -r frontend/requirements.txt
```

Install the premium Next.js UI dependencies:

```bash
cd frontend-next
npm install
```

Start the API from the repository root:

```bash
uvicorn app.main:app --reload --app-dir backend
```

In a second terminal, start the dashboard:

```bash
streamlit run frontend/app.py
```

Or start the premium MarketVision UI:

```bash
cd frontend-next
npm run dev
```

Open the docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Dashboard: `http://127.0.0.1:8501`
- Premium UI: `http://127.0.0.1:3000`

## Example Endpoints

- `GET /health`
- `GET /`
- `GET /api/stocks/AAPL/overview`
- `GET /api/stocks/AAPL/analysis`
- `GET /api/stocks/AAPL/simulation`
- `GET /api/stocks/AAPL/news`
- `POST /api/predict`
- `GET /api/bot/scan`
- `GET /api/bot/performance`
- `GET /api/bot/historical-similarity?ticker=NVDA`
- `POST /api/bot/explain`
- `POST /api/bot/resolve-outcomes`
- `GET /api/tools`
- `POST /api/tools/{tool_name}`
- `POST /api/stocks/compare`
- `POST /api/stocks/watchlist`
- `POST /api/portfolio/analyze`

Example stock comparison request:

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA"],
  "period": "6mo",
  "interval": "1d"
}
```

Example portfolio request:

```json
{
  "holdings": [
    { "ticker": "AAPL", "shares": 10 },
    { "ticker": "MSFT", "shares": 5 },
    { "ticker": "NVDA", "shares": 2 }
  ],
  "period": "6mo"
}
```

Example prediction request:

```json
{
  "ticker": "NVDA",
  "period": "1y",
  "interval": "1d",
  "horizon_steps": 12
}
```

## Signal Outcome Resolver

MarketVision logs signal-bot outputs to `backend/data/signal_ledger.jsonl`. The resolver evaluates pending paper signals after their prediction window ends and marks each row as:

- `resolved_win`
- `resolved_loss`
- `partial_win`
- `invalid`
- `still_pending`

Resolution compares actual future high/low/close movement against the recorded entry, stop loss, target 1, and target 2. It is for research and paper-trading analytics only.

Run it manually:

```bash
cd backend
python scripts/resolve_signal_outcomes.py
```

Performance is available at `GET /api/bot/performance`, including total predictions, pending/resolved counts, win/loss/partial-win rates, promoted-signal performance, quality-label performance, and confidence-bucket performance.

## Confidence Calibration

The performance endpoint reports live calibration buckets:

- `<50%`
- `50-60%`
- `60-70%`
- `70-80%`
- `80-90%`
- `90%+`

Each bucket includes coverage count, resolved count, accuracy, win rate, loss rate, partial-win rate, and average risk score. The Next.js UI shows this in the Scanner/Proof area so the project can prove whether model confidence is honest over time.

## Historical Similarity Engine

`GET /api/bot/historical-similarity?ticker=NVDA` compares the current setup against past setups using RSI, MACD, trend strength, volume spike, volatility, VWAP distance, market regime, VIX context, and sector strength where available.

It returns similar historical dates, similarity score, average future return, bullish/bearish/sideways outcome rates, best case, worst case, and average drawdown. The simulation UI shows these as “Similar historical setups found” so the prediction has practical context instead of only a confidence number.

## Monte Carlo Simulation Output

Simulation responses include chart-ready Monte Carlo arrays under `monte_carlo_chart`:

- main predicted path
- bullish path
- bearish path
- sideways path
- confidence upper/lower band
- volatility cone p05/p95
- expected final range

The frontend renders historical candles separately from translucent AI-predicted candles and scenario paths, so users can visually distinguish real market history from probability-based future simulations.

## FRED Macro Intelligence

When `FRED_API_KEY` is configured, MarketVision fetches rates, CPI, unemployment, GDP growth, treasury yields, VIX, and yield-curve data. It converts those inputs into:

- risk-on/risk-off score
- inflation pressure score
- rate pressure score
- recession risk score
- volatility pressure score
- macro regime label

Prediction explanations include macro context, for example restrictive rates, inverted yield-curve caution, or risk-off conditions. Macro is an auxiliary risk layer, not a standalone trading signal.

## Intraday VWAP Layer

The advanced engine loads intraday data when available:

- 1-minute
- 5-minute
- 15-minute
- 1-hour

It calculates VWAP, price above/below VWAP, distance from VWAP, VWAP slope, reclaim/rejection state, and intraday volume-curve ratio. This layer is exposed in prediction intelligence and scanner rows without replacing the daily V12 model.

## Scanner Ranking

Scanner rows are ranked by quality label first, then final signal probability, confidence score, lower risk score, risk/reward ratio, historical similarity strength, sentiment score, and macro compatibility.

Failed gates are exposed directly, including `confidence_failed`, `risk_failed`, `sentiment_failed`, `rr_failed`, `insufficient_data`, `macro_conflict`, and `weak_historical_similarity`.

## Explanation Layer

`POST /api/bot/explain` builds a prediction, extracts backend-provided facts, and generates a plain-language explanation. The LLM is only allowed to explain supplied facts; it is not used to invent probabilities, prices, or catalysts. If OpenAI is unavailable, the endpoint returns a deterministic fallback explanation.

Every explanation includes: “This is a probability-based simulation, not financial advice.”

## Execution-Realistic Backtesting

Backtests now include execution assumptions for commission, spread, slippage, max position size, liquidity filtering, stop-loss checks, and target checks. Reports include win rate, profit factor, Sharpe ratio, max drawdown, average reward/risk, false positive rate, invalid trade count, and strategy type.

These are still research fills, not proof of live execution quality.

## Model Drift Monitoring

`GET /api/bot/performance` includes a drift-monitoring block with recent live signal accuracy, high-confidence accuracy, confidence distribution, prediction distribution, feature-drift status, and warnings:

- `model_performance_degraded`
- `confidence_miscalibrated`
- `regime_failure_detected`
- `retraining_recommended`

The drift layer uses resolved signal-ledger outcomes and becomes more useful as more paper signals mature.

## Portfolio Intelligence

Portfolio analysis accepts ticker, share quantity, and optional average cost. It returns total value, sector exposure, correlation risk, volatility risk, concentration risk, portfolio risk score, AI risk summary, unrealized P/L where cost basis is supplied, and suggested watchlist alerts.

## MarketVision Tool Layer

AI agents can call structured JSON tools through:

- `GET /api/tools`
- `POST /api/tools/{tool_name}`

Available tools:

- `get_live_quote`
- `get_candles`
- `calculate_indicators`
- `predict_market`
- `predict_market_scenario`
- `run_scanner`
- `run_similarity`
- `run_historical_similarity`
- `run_monte_carlo`
- `get_performance`
- `get_performance_metrics`
- `explain_prediction`
- `analyze_portfolio`
- `get_macro_regime`
- `get_sentiment`
- `get_trade_candidates`

If `MARKETVISION_TOOL_TOKEN` is set, `/api/tools` and `/api/agents/*` require `Authorization: Bearer <token>`.

## V13.1 MCP Server + Agent Ecosystem

MarketVision now includes a stdio MCP server at `backend/mcp_server.py`. It exposes MarketVision as an AI-agent tool provider for Claude, Cursor, OpenAI Agents, LangGraph, CrewAI, or any MCP-compatible client.

Run the backend first:

```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run the MCP server:

```bash
cd backend
MARKETVISION_API_BASE_URL=http://127.0.0.1:8000 python mcp_server.py
```

Optional auth:

```bash
export MARKETVISION_TOOL_TOKEN=your-local-tool-token
export MARKETVISION_API_TOKEN=your-local-tool-token
```

Example MCP client config:

```json
{
  "mcpServers": {
    "marketvision-ai": {
      "command": "python",
      "args": ["/absolute/path/to/backend/mcp_server.py"],
      "env": {
        "MARKETVISION_API_BASE_URL": "http://127.0.0.1:8000",
        "MARKETVISION_API_TOKEN": ""
      }
    }
  }
}
```

Core MCP tools for agents:

- `predict_market`: compact probabilities, confidence, risk, quality label, and trade plan
- `run_scanner`: ranked scanner over custom or predefined universes
- `run_similarity`: compact historical similarity score and outcome probabilities
- `run_monte_carlo`: chart-ready future path arrays and confidence bands
- `analyze_portfolio`: portfolio value, exposure, correlation, and risk intelligence
- `explain_prediction`: fact-grounded explanation text
- `get_performance`: signal ledger, calibration, and drift metrics
- `get_macro_regime`: FRED-backed macro regime scores
- `get_sentiment`: recent news sentiment for a ticker
- `get_trade_candidates`: top ranked/promoted trade candidates

The backend also exposes an agent orchestration endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/agents/trade-candidates \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Find today'\''s best trade candidates","universe":"mega_cap_ai","max_candidates":5}'
```

The agent flow runs:

- Technical Agent: prediction, RSI/MACD/VWAP/trend/volume context
- Sentiment Agent: news sentiment and macro compatibility flags
- Macro Agent: FRED-backed macro regime
- Risk Agent: risk score, failed gates, risk/reward
- Simulation Agent: historical similarity and Monte Carlo context
- Judge Agent: final candidate decision and quality score

Frontend Copilot includes a “RUN AGENTS” / “FIND BEST SETUPS” action that calls the same orchestration endpoint and shows the tool trace plus Judge Agent candidate scores.

## Product Direction

The real-world version of this project should stay focused on decision support:

- explain stock trends in plain language
- highlight portfolio concentration and volatility
- compare holdings instead of promising winners
- simulate multiple future paths instead of one guaranteed forecast
- train ML models on historical OHLCV, indicators, sentiment, and macro features
- auto-tune model baselines locally or on Kaggle GPU using `ml/train_autotune.py`
- use LLMs such as OpenAI/Grok/Claude for explanation, not as the only prediction engine
- provide useful research context without crossing into personalized investment advice

## Current Platform Capabilities

MarketVision AI currently analyzes tickers with market data, technical indicators, intraday VWAP context, macro regime scores, sentiment, historical similarity, Monte Carlo paths, confidence/risk scoring, and a signal-promotion layer.

Data used today:

- OHLCV candles from market-data providers
- live quotes where configured
- FRED macro indicators
- Finnhub/NewsAPI headlines where configured
- FinBERT sentiment when available
- V12 model artifacts and auxiliary policy outputs
- local signal ledger outcomes

Outputs generated:

- bullish/bearish/sideways probabilities
- confidence score
- risk score
- quality label
- failed gates
- predicted candles
- Monte Carlo scenario paths
- historical similarity outcomes
- LLM explanation facts and explanation text
- portfolio risk intelligence
- calibration and drift metrics

Signal promotion:

Every ticker can receive a probability-based simulation. Only stronger setups are promoted into `high_confidence_trade_candidate`; weaker outputs remain `watchlist_candidate`, `prediction_only`, `avoid_high_risk`, or `insufficient_data`.

Calibration:

The platform tracks confidence buckets and resolved outcomes so confidence can be compared against realized paper-signal results over time.

Historical similarity:

The engine compares the current setup to past setups using technical, volatility, volume, VWAP, regime, sector, and sentiment-style features. It reports similar dates, average future return, bullish/bearish/sideways outcomes, best case, worst case, and drawdown.

Monte Carlo simulation:

The simulation engine returns main, bullish, bearish, sideways, confidence-band, and volatility-cone arrays for chart visualization. These are simulations, not guaranteed forecasts.

Performance tracking:

The signal ledger records paper signals, resolves outcomes after the prediction window, tracks promoted-signal performance, and reports drift warnings when enough resolved data exists.

Still missing or partial:

- true options flow and dark-pool data
- real SEC filing/13F/insider ingestion
- full social sentiment from Reddit/X
- advanced sequence models in production serving
- live brokerage execution
- hosted production MCP gateway; current MCP server is a local stdio server backed by the FastAPI API

All outputs must be framed as: “Probability-based market simulations and AI-generated financial intelligence, not financial advice.”

## Current Workflow

1. Run the FastAPI backend
2. Open the Streamlit dashboard
3. Research a single stock with charts, signals, and news
4. Run a future chart simulation with scenario probabilities
5. Rank a watchlist of candidate stocks
6. Compare several tickers side by side
7. Analyze portfolio concentration, volatility, and correlations

## Repository Link

GitHub: https://github.com/manit0700/ai-stock-analysis-portfolio
