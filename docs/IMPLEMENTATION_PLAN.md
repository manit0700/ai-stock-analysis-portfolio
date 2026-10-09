# MarketVision AI: repair plan

**Date:** 2026-10-09 · **Branch:** `repair/marketvision-audit` (local only; nothing is pushed) · **Base:** `344f8b5`

The findings this plan answers are in [`AUDIT_REPORT.md`](AUDIT_REPORT.md). Finding numbers (S1, M1, ...) refer to that file.

## Principles

1. Existing routes, response shapes and the Next.js interface are kept. New behaviour is added beside them; a legacy response gains fields (`data_source`, `model_status`), it does not lose any.
2. One implementation of each thing: one data door, one indicator library, one feature pipeline, one accounting engine, one event ledger.
3. Domain code does not import the web framework. The same functions serve the HTTP API, the tool layer, scripts and tests.
4. A number is never invented. Missing data stays missing and is reported; a model that cannot be given its inputs makes no prediction; a decision that cannot be validated is HOLD.
5. Nothing is promoted on a number chosen after the fact. The promotion rule for a model is written to a file and committed before the holdout is evaluated.

## Target layout

```
backend/app/
  core/          configuration, secrets and log redaction, auth, errors, clock
  marketdata/    providers (yahoo, finnhub, fred, newsapi, snapshot), envelope, calendar,
                 quality checks, cache, retries, the unified MarketData service
  features/      indicators, versioned schema, the one pipeline, context loader
  forecast/      dataset, purged walk-forward splits, models, calibration, evaluation,
                 leakage checks, model registry, prediction service, command line
  accounting/    decimal money, the portfolio book (cash, positions, fills, P&L)
  backtesting/   bar-by-bar engine on the same book, strategies, metrics, baselines
  predictions/   append-only, hash-chained prediction and resolution events
  paper/         the $50 virtual account: orders, fills, risk controls (SQLite)
  agent_tools/   15 validated tools with schemas and permissions
  api/           v2 routes; error handling; legacy routes stay in main.py
  services/      the existing services, repaired in place or routed through the above
```

## Phases and what "done" means for each

| Phase | Work | Evidence it is done |
|---|---|---|
| 2 Security | Remove the embedded Finnhub key; quotes come from the backend. Keys in headers, never URLs; redaction on every log handler. Auth scopes (`read`, `admin`), constant-time compare, CORS allow-list, rate limits on costly routes. Secret scanner (script, test, pre-commit, CI). `SECURITY.md` with the rotation steps | Scanner passes on the tree; the built frontend bundle contains no key; tests for redaction and auth |
| 3 Market data | Every legacy service reads through `MarketData`. Responses carry provider, retrieval time, market time, freshness, quality. Retries, back-off, rate limits, caches, circuit breaker | Tests with recorded provider shapes: fallback, stale, duplicate, missing, rate-limited, never "live" when delayed |
| 4 Features | `features.pipeline.build_features` is the only feature code. Schema versioned and hashed. Explicit missing-value policy. A model whose schema hash differs from the running code is refused | Parity test (row from a long history equals row from a truncated one), no-look-ahead test, schema-mismatch test |
| 5 Forecasting | Target: probability that the 5-session forward return is positive. Baselines, logistic regression, random forest, gradient boosting, HistGradientBoosting, LightGBM if installed. Purged walk-forward validation, tuning on development data only, one untouched holdout, calibration, date-block bootstrap intervals, results by market regime. Legacy models quarantined | A model card per run with data snapshot id, schema hash, seeds and every metric beside its baseline. Promotion only under the pre-written rule; otherwise "no predictive edge established" |
| 6 Backtesting | One book: cash, fractional positions, fills at the next open with spread, slippage and commission; stops and targets act on the book, so trades and equity cannot disagree. Cash-only and buy-and-hold baselines on identical terms | Accounting invariants on synthetic paths: cash never negative, equity equals cash plus holdings, P&L reconciles to the cent |
| 7 Prediction tracking | Immutable `prediction_issued` events with a deterministic id (duplicates impossible); separate `prediction_resolved` events computed over the exact forward window from the trading calendar; hash chain; summaries count wins as wins | Tests: a 5-session forecast is scored on sessions t+1..t+5 only; duplicate issue is a no-op; tampering is detected |
| 8 Paper account | $50.00 virtual. SQLite transactions, `Decimal` arithmetic, idempotent orders, no shorting, no margin, no negative cash, stale-quote rejection, position and exposure limits, pause switch, reset that archives history | Tests for each rejection, for idempotency, for valuation and for realised and unrealised P&L |
| 9 Tool interfaces | 15 tools with input validation, structured output, documented JSON Schema and a permission class. Paper tools need `simulation: true`. MCP server speaks the standard transport | Schema export test; permission tests; MCP checked with the official client |
| 10 Frontend | Key removed; invented prices and random ticks removed; source and freshness shown; new panels for the forecast, prediction history, virtual account, holdings, trades, P&L; banner "SIMULATED TRADING — NO REAL MONEY" | `tsc --noEmit` clean; `next build` succeeds; bundle scanned for secrets |
| 11 Free-tier fit | Light default dependencies; heavy ones optional; reduced mode without keys; CI workflow; scheduled-run template (not enabled) | Backend imports and tests pass without torch, xgboost, lightgbm, catboost |
| 12 Verification | Whole suite; independent review of the money code by a reviewer that did not write it | Test counts reported as run |
| 13 Documentation | README, architecture, methodology, security, final report | Files present; every claim traceable to a run |

## Operations that change or move existing things

Nothing here is irreversible, and nothing leaves this machine.

| Operation | Reversible? | Note |
|---|---|---|
| Edit existing source files | Yes (git) | On a local branch only |
| Move the 29 saved model files into `ml/quarantine/` | Yes (`git mv`) | Not deleted. They stop being loaded by default because they fail evaluation (M1, M2) |
| Replace `frontend-next/lib/useFinnhubWS.ts` | Yes (git) | The key stays in git history until history is rewritten, which is **not** done here |
| Create `backend/data/` files at run time | Yes | Git-ignored |

**Not done without the owner:** pushing, merging, deploying, rewriting git history, revoking or rotating the Finnhub key, touching the `autonomous-ai` repository, connecting any brokerage.

## Known limits before starting

- **No live provider can be reached from the build machine** (Yahoo, FRED, Finnhub and NewsAPI are all blocked by its network policy; the linked Mac's sandbox has the same policy). Provider code is tested against recorded response shapes. A live check on the owner's machine is listed as pending in the final report.
- **Empirical evaluation therefore uses public historical files that are reachable** (QuantConnect's sample daily bars with adjustment factors, 1998 to March 2021; a Yahoo extract of 14 technology stocks, 2010 to 2022; a 31-stock Dow Jones extract, 2006 to 2017; CBOE VIX). They are genuine prices, but the stock list was chosen with hindsight (survivorship bias) and ends in 2021. The same pipeline runs on a fresh Yahoo snapshot with one command once the network allows.
- The data files themselves are not committed. The snapshot manifest (sources and SHA-256 of every file) is.
