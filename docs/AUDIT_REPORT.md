# MarketVision AI: initial audit

**Date:** 2026-10-09
**Commit audited:** `344f8b5` (2026-05-31), branch `main`
**Method:** every backend file read; frontend read at the level of structure, data flow and types; existing tests run; faults reproduced with small scripts on synthetic inputs before any change was made.

Findings from the earlier reuse map (`marketvision-reuse-map.md`) were treated as leads and re-checked here. Each row in section 3 says how it was confirmed.

## 1. Architecture as found

```
frontend-next/   Next.js 15 + React 19 + Tailwind + lightweight-charts
                 one 2,160-line page.tsx with 10 tabs; lib/api.ts calls 23 backend routes
frontend/        Streamlit dashboard (304 lines), calls the same backend
backend/
  app/main.py            FastAPI, 36 routes, all services created as module globals
  app/services/          15 service modules (5,322 lines)
  mcp_server.py          hand-written MCP server that forwards to the HTTP API
  scripts/               one script: resolve signal outcomes
  tests/                 9 files, 21 tests
ml/              Kaggle training scripts and 29 saved model files (about 350 MB)
render.yaml      Render free-tier blueprint for the backend
```

Data flow for a prediction: `main.py` → `AdvancedMarketIntelligenceEngine` → about 20 separate yfinance requests (the ticker at five intervals, SPY, QQQ, VIX, ten sector ETFs), FRED, Finnhub, NewsAPI → rule-based probabilities blended 55/45 with the saved XGBoost model → Monte Carlo paths → gates → "signal bot" decision → optionally appended to a JSONL ledger.

| Area | What exists |
|---|---|
| API routes | 36: stock overview, analysis, simulation, intelligence, bot, scanner, performance, similarity, explain, news, profile, quote, macro, backtest, compare, watchlist, predict, portfolio, copilot, tools, agents |
| Market data | `yfinance` called directly from five modules; FRED, Finnhub and NewsAPI through `requests` |
| Indicators | RSI, SMA, EMA, MACD, ATR, Bollinger, VWAP, each written several times (five separate RSI functions in the backend, five more in `ml/`) |
| Sentiment | VADER in three modules; FinBERT optional |
| ML models | XGBoost ensembles v5 to v12, trained on Kaggle, loaded from `ml/` by file name |
| Feature engineering | One version in `ml/kaggle_train.py` (158 candidate features), a different one in `backend/app/services/model_service.py` |
| Backtesting | Five indicator strategies in `backtest_service.py` |
| Portfolio analysis | `portfolio.py`, a pure function plus a yfinance wrapper |
| Signal tracking | `signal_ledger.py`, a JSONL file |
| Persistence | No database. One git-ignored JSONL file and in-memory dictionaries |
| Configuration | `os.getenv` at import time in each service; `backend/.env.example` |
| Authentication | One optional bearer token, checked on 3 of 36 routes |
| Tests | 21, all passing; every one replaces the network and the model with a stand-in |

## 2. Baseline before any change

| Check | Result |
|---|---|
| Backend tests | 21 passed in 3.9 s (Python 3.13, current library versions) |
| Frontend type check (`tsc --noEmit`) | **9 errors.** The build hides them: `next.config.ts` sets `ignoreBuildErrors: true` and `ignoreDuringBuilds: true` |
| `npm audit --omit=dev` | 5 advisories (1 critical, 4 high) in `next`, `postcss`, `nanoid`, `sharp`, `source-map-js`; all have a fix available |
| Model file load | Loads, with an XGBoost warning that the file was pickled by an older version |

## 3. Faults, with how each was confirmed

### Security

| # | Finding | Confirmed by |
|---|---|---|
| S1 | **A Finnhub API key is hard-coded** in `frontend-next/lib/useFinnhubWS.ts` line 9 and committed to a public repository. It is also shipped to every browser that loads the page | Read the file |
| S2 | **Provider keys leak into logs on any failed request.** FRED, Finnhub and NewsAPI pass the key as a URL parameter, and each service logs the exception text, which contains the full URL | Read `_get` in all three services; `requests` puts the URL in the message of `HTTPError` and `ConnectionError` |
| S3 | **Open routes spend the owner's keys.** `/api/copilot/chat` forwards any caller's text to OpenAI; `/api/news/search` uses the NewsAPI quota; `/api/bot/scan` makes hundreds of provider requests. None needs a token | Read `main.py`; only lines 144, 154, 253 call `_require_tool_auth` |
| S4 | State-changing routes are unauthenticated: `POST /api/bot/resolve-outcomes` rewrites the ledger; every scan appends to it | Read `main.py` |
| S5 | CORS allows every origin together with credentials | `allow_origins=["*"], allow_credentials=True` |
| S6 | The token comparison is not constant-time, and with no token configured the tool routes are open | `_require_tool_auth` |
| S7 | Dependencies are unpinned in all three requirements files; test tools and `torch` sit in the runtime list | Read the files |

### Machine learning

| # | Finding | Confirmed by |
|---|---|---|
| M1 | **Training and serving compute different features.** The v12 model expects 45 inputs. The serving code computes 24 of them. The other 21 (VIX level, 10-year yield, SPY and QQQ trend, sector returns) are filled with 0.0. Those 21 carry 51.6% of the model's feature importance | Loaded the model, built a synthetic history, listed the model's columns against the columns `_compute_features` returns |
| M2 | **Saved models do not beat a constant guess** on their own holdout: three-class 36.8% against 41.5%; up/down 52.0% against 55.1%; promoted breakout signal 68.4% against 72.4% | Recomputed each baseline from the class counts stored in `model_20260517_044621_metrics.json`. The holdout rows themselves are not in the repository (`*.parquet` is git-ignored), so the scores could not be re-derived from data |
| M3 | Hyperparameter search ran over the whole dataset, including the last 20% later reported as the holdout | `cv_score(X_full, y_full, ...)` at `kaggle_train.py:1586`, holdout split at `:1636` |
| M4 | The holdout was reused to choose between versions | Comment at `kaggle_train.py:118`: "Selected from v7 holdout..." |
| M5 | No gap between training and test rows although labels look 10 days ahead, so boundary rows share outcomes | No purge or embargo anywhere in the script |
| M6 | The "proof" gate that promotes a trade reads the accuracy of the top-confidence bucket, which is mostly confident "do not trade" calls. The signal's precision is 37.5% (6 of 16) | `confidence_bucket_report` uses `proba.max(axis=1)`; precision and recall from the metrics file |
| M7 | "Confidence" is a formula (`45 + margin × 85`, plus bonuses), not a calibrated probability | `_confidence_score` |
| M8 | Calendar features (`month_sin`, `month_cos`) are the two most important inputs of the saved model: a sign of fitting to the sample period | Feature importances read from the model |

### Prediction tracking

| # | Finding | Confirmed by |
|---|---|---|
| L1 | **Outcomes are judged on the wrong prices.** The resolver always asks for the last 5 daily bars, whatever the signal's date or horizon | A signal dated 60 days ago with a 12-day window requested `('TEST', '5d', '1d')` |
| L2 | "Accuracy" counts partial wins, and any positive close counts as a partial win | The same test row: accuracy 1.0, win rate 0.0 |
| L3 | Every scan appends a new row per ticker, so one ticker is logged many times a day | `_record_bot_signal` is called inside the scan loop; a 120-second cache is the only guard |
| L4 | The file is rewritten in place on every resolution; rows are mutated | `write_signals` opens with `"w"` |
| L5 | Stop and target are compared with the window's extreme prices, so order of events is lost | `_resolve_row_outcome` |
| L6 | On Render's free plan the disk is not kept, so the ledger does not survive a restart | `render.yaml` has no disk; plan is `free` |

### Backtesting

| # | Finding | Confirmed by |
|---|---|---|
| B1 | **The trade list and the equity curve disagree.** Stops and targets close trades in the list; the equity curve ignores them | Read `_run_backtest`; on a synthetic series the "buy and hold" strategy logged 36 trades |
| B2 | **Baseline comparison is unequal.** Strategies hold 25% of capital, the buy-and-hold baseline 100% | Same run: strategy "buy and hold" 5.55%, baseline 17.00% |
| B3 | Entry price is the close of the bar after the signal, but that bar's return is already counted | Read the code |
| B4 | No cash account. Returns are multiplied together; nothing prevents a position larger than cash | Read the code |
| B5 | Only indicator strategies are tested. The model's own signals are never backtested | `_STRATEGIES` |
| B6 | Sharpe ratio is annual return over annual volatility with no check on sample length | Read the code |

### Market data

| # | Finding | Confirmed by |
|---|---|---|
| D1 | No retries, no rate limiting, no back-off. Finnhub and NewsAPI have no cache | Read the services |
| D2 | A 20-request fan-out per prediction, repeated for every ticker in a scan, with no shared cache for SPY, QQQ, VIX or sectors | Counted calls in `advanced_intelligence.py` |
| D3 | Responses carry no provider, no retrieval time, no freshness label. Yahoo quotes (delayed for many exchanges) are presented beside a "live" indicator | Read responses and `StockChart.tsx` |
| D4 | Missing bars, duplicate rows and stale data are never checked. `portfolio.py` forward-fills prices silently | Read the code |
| D5 | Failures are swallowed: providers return `None` or `[]` and callers cannot tell "no news" from "provider down" | Read the services |
| D6 | NewsAPI's free plan delays articles by 24 hours; this is not shown | Provider terms; no label in code |

### Agent and tool layer

| # | Finding | Confirmed by |
|---|---|---|
| A1 | **The MCP server cannot talk to a standard MCP client.** It frames messages with `Content-Length`; the MCP stdio transport is newline-delimited | The official Python client timed out on `initialize`; a newline-delimited request got no reply |
| A2 | The "agents" in `agent_orchestrator.py` are dictionaries of copied fields; the judge is a weighted sum | Read the file |
| A3 | `similarity_service.py` scores nearly everything as identical: 99.9% of 1,229 windows of a random walk scored 0.99 or more | Ran it; the cause is cosine similarity on unscaled features |
| A4 | Tool inputs are not validated: `int(args.get(...))` on raw input, and unknown fields are ignored | `_execute_marketvision_tool` |

### Frontend

| # | Finding | Confirmed by |
|---|---|---|
| F1 | **Invented numbers are shown as market data until the backend answers.** Market cards start from `mock-data.ts` ("S&P 500 6,214.88", "Bitcoin $104,820") with no label | `useState<MarketCard[]>(mockMarketCards)` at `page.tsx:1674` |
| F2 | If the backend is down a made-up NVDA forecast is shown. It is labelled "demo data" in a status line only | `page.tsx:1774` |
| F3 | Forecast candles are labelled `exact_model_ohlc_forecast`; in `simulation.py` they are sine curves | Read `_make_path` |
| F4 | **Both charts invent price movement.** When the Finnhub socket is not connected, a timer adds `Math.random()` noise to the last close every second and shows the result as the large current price, under the label "POLLING" | `StockChart.tsx:168-186`; the same block in `SimulationChart.tsx` |
| F5 | 9 type errors hidden by build settings (section 2) | `tsc --noEmit` |

### Code health

| # | Finding |
|---|---|
| C1 | `ml/kaggle_train.py` and `ml/kaggle_kernel/kaggle_train.py` are byte-identical (2,420 lines each) |
| C2 | Two prediction engines overlap: `simulation.py` (older) and `advanced_intelligence.py`. `/api/predict` and the copilot still use the older one |
| C3 | Two similarity implementations; only the one inside `advanced_intelligence.py` standardises its inputs |
| C4 | Duplicate tools: `get_performance` and `get_performance_metrics`; `run_similarity` and `run_historical_similarity` |
| C5 | README lists a `notebooks/` folder that does not exist and a backend layout that is out of date |
| C6 | Services are module-level globals created at import, which makes them hard to test and to configure |
| C7 | `fastapi.HTTPException` is raised from inside service code, tying domain logic to the web framework |
| C8 | 369 MB of binary model files are committed |

## 4. What works and is kept

- `portfolio.py::analyze_portfolio_from_prices`: correct on synthetic prices.
- `fred_service.py`: has caching and a timeout; its macro scoring is covered by a test.
- `finbert_service.py`: sensible fallback to VADER and a cache.
- The calculators inside `advanced_intelligence.py`: technical analysis, intraday VWAP, market regime, sector rotation, standardised similarity, seeded Monte Carlo.
- `copilot_service.py`'s rule that the language model may only explain facts built in code, with a written fallback.
- `tool_catalog.py`: a clean list of tool definitions.
- The whole Next.js interface and its 23 API calls. Routes and response shapes are preserved.

## 5. Limits of this audit

- The build machine cannot reach Yahoo, FRED, Finnhub, NewsAPI or Hugging Face (the network policy returns 403). No provider was called live. Provider code was read and will be tested against recorded response shapes; a live run on the owner's machine is pending.
- The holdout data behind the saved models' metrics is not in the repository, so M2 was checked against the stored counts, not re-scored.
- The copy of this repository at `~/ai-stock-analysis-portfolio` on the owner's Mac is a July 2025 prototype, not this code.
