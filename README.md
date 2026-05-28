# MarketVision AI - Stock Simulation & Portfolio Intelligence

## Overview

MarketVision AI is an AI-powered market simulation and decision-intelligence platform. The goal is to help traders and investors understand market probabilities, future chart scenarios, portfolio risk, and research context without pretending to guarantee stock predictions.

The current MVP includes:

- a FastAPI backend that pulls market data from `yfinance`
- research endpoints for single stocks, comparisons, news, simulation, and portfolio risk
- a probability-based future chart simulation engine
- bullish, bearish, sideways, and high-volatility scenario paths
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
