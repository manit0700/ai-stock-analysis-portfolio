# MarketVision AI repair: final report

**Date:** 2026-10-09 · **Branch:** `repair/marketvision-audit`, 7 commits on top of `344f8b5`.
Nothing was pushed, deployed, published or connected to any financial account. The `autonomous-ai`
repository was not touched.

Companion documents: [AUDIT_REPORT.md](AUDIT_REPORT.md) (every fault, with how it was confirmed),
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) (the plan and its done-tests),
[OPERATIONS.md](OPERATIONS.md) (free-tier operation), [../SECURITY.md](../SECURITY.md).

## 1. Original problems

39 findings; the full table is in the audit report. The ones that shaped the work:

- A Finnhub API key committed to the public repository and shipped to every browser (S1); other
  keys leaking into logs via failed-request URLs (S2); open routes spending the owner's quotas
  and rewriting state (S3-S4); CORS `*` with credentials (S5).
- Training and serving computed different features: 21 of the model's 45 inputs were zero-filled
  in production, carrying 51.6% of feature importance (M1). The saved models scored below a
  constant guess on their own holdout (M2), the tuner saw the holdout (M3), and the holdout was
  reused across versions (M4).
- Prediction outcomes judged on the wrong price window and partial wins counted as accuracy
  (L1-L2); a rewritten, non-append-only ledger (L4).
- The backtester's trade list and equity curve disagreed, and strategies at 25% exposure were
  compared against a 100% buy-and-hold (B1-B2).
- The interface showed invented numbers: mock market cards, a demo forecast, and random noise
  added to the price every second when a websocket was down (F1-F4); nine type errors hidden by
  build flags (F5).
- The MCP server did not speak the MCP transport (A1).

## 2. What was built or repaired (files changed)

| Area | What stands now |
|---|---|
| `backend/app/core/` | Environment-only configuration; `Secret` wrapper + log redaction on every handler; read/admin auth scopes (constant-time); CORS allow-list; structured domain errors |
| `backend/app/marketdata/` | One door to Yahoo/Finnhub/FRED/NewsAPI/snapshots: provider+freshness+quality envelope on every response, NYSE calendar, caches, retries with back-off, per-provider rate limits, circuit breaker, immutable data snapshots. Legacy services now read through it (the older Finnhub, FRED and NewsAPI dashboard services only since the follow-up described in API_INTEGRATIONS.md) |
| `backend/app/features/` | One indicator library (was ten RSI copies), a versioned+hashed feature schema, one pipeline for training/backtesting/serving, explicit missing-value policy (complete rows; never zero-fill) |
| `backend/app/forecast/` | Purged walk-forward evaluation, baselines first, calibration, calendar-week bootstrap intervals, leakage checks (incl. shuffled-label control and recomputing rows from past bars), a promotion rule committed before the holdout was scored, a registry that refuses schema-mismatched or altered model files |
| `backend/app/predictions/` | Append-only hash-chained ledger; deterministic ids (duplicates impossible); resolution on exactly the forecast's window; tail truncation detectable via a head anchor; cross-process file lock |
| `backend/app/accounting/` + `app/backtesting/` | Decimal money; one Book with checked identities; bar-by-bar engine where stops/targets are sales in that book; splits handled; costs, liquidity filter, dividends; cash-only and equal-terms buy-and-hold baselines |
| `backend/app/paper/` | The $50 virtual account: SQLite with BEGIN IMMEDIATE transactions, fractional shares, no margin/shorting/negative cash, stale-and-unknown-age quote rejection, closed-market rejection, position/exposure caps, pause, idempotent orders, reset-with-archive. `simulation=true` required |
| `backend/app/agent_tools/` + `mcp_server.py` + `app/api/` | 15 validated tools, one catalogue for HTTP and MCP; the MCP server now speaks the standard newline-delimited transport (verified against the official client); `/api/v2` routes |
| `frontend-next/` | Key removed; quotes come from the backend with provider/age/freshness labels; all invented numbers removed; `$50·ACCOUNT` tab (forecast, orders, holdings, trades, P&L, ledger); global SIMULATED banner; type errors fixed and no longer hidden; `next build` passes |
| Tooling | Secret scanner (+ tests + bundle scan), pinned split requirements, CI workflow, SECURITY.md, OPERATIONS.md, README |

**Components reused as they were:** `portfolio.py` (pure risk function), `fred_service` macro
scoring, `finbert_service` (VADER fallback), `copilot_service`'s facts-only LLM pattern, the
technical calculators in `advanced_intelligence.py`, `tool_catalog` shapes, the whole Next.js
interface and its legacy routes. **Replaced:** `signal_ledger` (by `app/predictions`),
`backtest_service` internals (by `app/backtesting`), the MCP server, `useFinnhubWS`.
**Quarantined, not deleted:** the v5-v12 model artifacts (`ml/quarantine/README.md` says why).

## 3. Prediction results (genuine prices, honest verdict)

Data: pinned public daily bars, 41 US stocks 1999-2021 plus SPY and VIX (snapshot
`public-daily-6919ac62da5f`; survivorship-biased list, documented). Target: P(adjusted close
higher after 5 sessions). Development 1999-2015, holdout 2016-2021 (34,764 rows), scored once.

| Candidate (holdout, 2016-2021) | Accuracy | Majority class | Balanced acc. | ROC-AUC | Brier | Brier skill vs base rate |
|---|---|---|---|---|---|---|
| Constant base rate | 0.574 | 0.574 | 0.500 | 0.500 | 0.2457 | 0 |
| Trailing-frequency baseline | 0.567 | 0.574 | 0.500 | 0.498 | 0.2459 | −0.0009 |
| Logistic regression | 0.555 | 0.574 | 0.501 | 0.506 | 0.2469 | −0.0046 |
| **Random forest (selected on validation)** | 0.568 | 0.574 | 0.505 | 0.516 | 0.2465 | −0.0031 |
| Gradient boosting | 0.548 | 0.574 | 0.514 | 0.518 | 0.2474 | −0.0069 |
| Hist gradient boosting | 0.563 | 0.574 | 0.504 | 0.500 | 0.2478 | −0.0083 |
| LightGBM | 0.543 | 0.574 | 0.507 | 0.512 | 0.2478 | −0.0083 |

Selected model after calibration: Brier skill **−0.0115** [−0.0274, −0.0000], ROC-AUC **0.500**
[0.481, 0.518] (95% calendar-week bootstrap). Verdict under promotion rule v1 (committed before
scoring): **no predictive edge established.** The API serves the historical base rate, labelled
as a baseline, with `edge_established: false`. Expected-return estimation also failed to beat
the historical mean, so the mean is served and labelled. The volatility-scaled uncertainty range
is used only where it beat the unscaled range out of sample.

The machinery itself is validated both ways on synthetic data: on random walks it refuses to
promote; on a planted predictable signal it finds the effect, calibrates it, passes the rule and
promotes — so "no edge" is a finding about the data, not a broken pipeline.

## 4. Backtesting results

The engine's correctness is established by accounting invariants on synthetic paths (cash never
negative; equity ≡ cash + holdings ≡ starting + realised + unrealised, to the cent; stops and
targets move the trade list and the equity curve together; a buy-and-hold *strategy* now scores
exactly the buy-and-hold *baseline* — the audit's B2 showed 5.55% vs 17.00% for the same thing).
No profitability claim is made for any strategy: runs on the snapshot are research output, and
the indicator strategies are examples, not recommendations.

## 5. Security fixes

S1 key removed from code and bundle (scanner proves it); **rotation still requires the owner**
(SECURITY.md, step 1 — the key lives in git history). S2 closed by header-based keys plus global
log redaction. S3/S4 closed by scopes and rate limits. S5 closed by the allow-list. S6 by
constant-time comparison. S7 by pinned, split requirements. A tracked-tree and built-bundle
secret scan runs in the test suite and CI.

## 6. Test results, as run

- `backend: 219 passed` (pytest, Python 3.13; the suite plants no network calls — providers are
  tested against recorded response shapes, and the paper/forecast stacks run on synthetic
  snapshots). The original 21 tests still pass inside that count.
- Frontend: `tsc --noEmit` clean and `next build` succeeds with error-hiding flags removed;
  the built bundle scanned clean.
- An **independent adversarial review** of the money code (by a separate agent with runnable
  probes) found 5 bugs and 5 sharp edges — splits booked as losses, a crash on sub-fee exits, a
  cross-process double-spend, notional overspend by a cent, and a hash chain broken by two honest
  writers. All are fixed with regression tests (commit `e3aa264`); its probes are reproduced in
  the suite.
- Synthetic tests verify implementation correctness only; they are never cited as market accuracy.

## 7. Remaining limitations

1. **No live provider was called.** This machine cannot reach Yahoo, FRED, Finnhub or NewsAPI
   (network policy). Provider code is tested against recorded shapes; **live verification is
   pending** on the owner's machine: set keys in `backend/.env`, run
   `uvicorn app.main:app`, and open `/api/v2/health`, `/api/v2/stocks/AAPL/quote`,
   `/api/v2/stocks/AAPL/forecast`.
2. The evaluation data ends in March 2021 and its stock list is survivorship-biased (stated in
   the snapshot manifest and on every model card). `scripts/build_yahoo_snapshot.py` is not
   written; building a fresh snapshot needs ~20 lines against `MarketData.history` on a machine
   with Yahoo access, and the pipeline then runs unchanged.
3. No sector-ETF or Treasury-yield features in the evaluation (absent from the public files), so
   the `sector`/`macro` feature groups are untested on real data.
4. The paper account fills at quote ± spread/slippage: no order book, no partial fills, no halts.
   The ledger's hash chain is tamper-*evident*, not tamper-*proof* (an attacker who rewrites the
   whole file and anchor defeats it).
5. The legacy scanner/signal-bot routes still exist for the old UI tabs; their "confidence" is
   the rules engine's and their ledger is the legacy JSONL. They no longer load the quarantined
   models and are rate-limited, but they predate the honest-forecast path and should eventually
   be retired in favour of `/api/v2`.
6. Free-tier Render keeps no disk: ledger and paper DB are lost on restart there (OPERATIONS.md).

## 8. Free-tier readiness and future integration

Verified: the whole application imports and serves `/api/v2/health` from the pinned core
requirements alone (no torch, no boosting libraries, no pytest) in a fresh environment; heavy
pieces are opt-in. CI runs tests, the light-install check, the frontend build and the secret scan
on GitHub Actions' free tier. The evaluation ran on 2 CPU cores in ~19 minutes.

For `autonomous-ai` (later, deliberately not connected now): the 15 interfaces listed in the
plan all exist with validated inputs, structured outputs, documented schemas, permission classes
and deterministic financial arithmetic — as HTTP routes (`/api/v2/tools`) and over standard MCP
(`backend/mcp_server.py`). Research tools are read-only; `submit_paper_order` requires
`simulation: true`; no tool depends on any specific LLM provider.

**Is MarketVision ready for autonomous-agent integration?** The interfaces, the simulated-money
boundary and the honest-labelling guarantees are ready and tested. What an agent would get from
`get_stock_prediction` today is a labelled base rate — which is the truthful state of the
forecasting art in this repository.
