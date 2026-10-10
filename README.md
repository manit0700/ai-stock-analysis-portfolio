# MarketVision AI

Stock research, probability forecasts and a **simulated** $50 trading account, built to be
measurable and reproducible rather than impressive.

**SIMULATED TRADING — NO REAL MONEY.** There is no brokerage integration anywhere in this
codebase. Forecasts are probability estimates from historical data, never guarantees and never
financial advice, and a forecast never places an order.

## What it tells you, honestly

- A forecast endpoint serves the probability that a stock closes higher five sessions later.
  **As of the October 2026 evaluation, no model beats the historical base rate** on out-of-sample
  data, so the API serves the base rate and says so (`"edge_established": false`). A model only
  starts serving after it passes a promotion rule that was committed to git before its holdout
  was scored (`backend/app/forecast/promotion.py`).
- Every response from the data layer names its provider, retrieval time, market time, freshness
  (live / delayed / historical) and quality checks. Missing data stays missing; nothing is
  invented, in the API or the interface.
- Every issued forecast goes into an append-only, hash-chained ledger and is scored later on
  exactly its own five-session window. Open predictions are never counted as wins.

## Layout

```
backend/
  app/core/          configuration, secrets + log redaction, auth scopes, errors, clock
  app/marketdata/    one door to Yahoo/Finnhub/FRED/NewsAPI/snapshots: envelope, calendar,
                     quality checks, caches, retries, rate limits, circuit breaker
  app/features/      indicator library, versioned feature schema, the ONE feature pipeline
  app/forecast/      dataset, purged walk-forward evaluation, calibration, leakage checks,
                     promotion rule, model registry, prediction service   (python -m app.forecast)
  app/predictions/   append-only hash-chained prediction ledger
  app/accounting/    Decimal money and the portfolio book (cash, positions, P&L)
  app/backtesting/   bar-by-bar engine on that book; stops/targets move trades AND equity
  app/paper/         the simulated $50 account (SQLite; no margin, no shorting, no negative cash)
  app/agent_tools/   15 validated tools, one catalogue for HTTP and MCP
  app/api/           /api/v2 routes; legacy routes stay in app/main.py
  mcp_server.py      MCP stdio server (standard newline-delimited transport)
  scripts/           data snapshots, secret scanner, signal-outcome resolver
frontend-next/       the terminal-style dashboard (Next.js); quotes come from the backend
ml/quarantine/       the old v5-v12 models, kept but not loaded (see its README)
docs/                audit report, implementation plan, methodology, final report
```

## Run it

Python 3.11+ and Node 20+.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core; see requirements-{ml,dev,heavy}.txt
cp .env.example .env                     # add free API keys if you have them (all optional)
python scripts/check_providers.py        # which providers actually answer from this machine
uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs

cd ../frontend-next
npm ci && npm run dev                    # http://localhost:3000
```

Tests (no network is ever contacted; provider code is tested against recorded shapes):

```bash
cd backend && pip install -r requirements-dev.txt && python -m pytest
```

## The forecasting workflow

```bash
cd backend
python scripts/build_public_snapshot.py        # pinned public daily bars, 1998-2021 (41 stocks)
# or, on a machine that can reach Yahoo Finance, build a fresh snapshot of your own universe
python -m app.forecast snapshots
python -m app.forecast evaluate --snapshot <id> --stage development   # safe to iterate on
python -m app.forecast evaluate --snapshot <id>                       # scores the holdout ONCE
python -m app.forecast list / show <run id> / promote <run id>        # promote refuses a run that failed the rule
```

Method, in one paragraph: complete-row dataset from the single feature pipeline (schema-versioned
and hashed; a model whose schema hash differs from the running code is refused at serving time);
chronological walk-forward validation with purging and an embargo so no training label window
touches a test period; model and settings chosen on validation Brier score; calibration chosen on
later validation rows; one untouched holdout, scored once, with calendar-week block-bootstrap
intervals; leakage checks including recomputing sampled rows from past bars alone and a
shuffled-label control; promotion only under the pre-written rule, otherwise the baseline stays.
Classification skill is never presented as profitability — that is the backtester's job, with
costs and equal-terms baselines.

## The $50 virtual account

`POST /api/v2/paper/orders` with `{"side": "buy", "symbol": "AAPL", "notional": 5, "simulation": true}`
(the `simulation: true` acknowledgement is required). Market orders, fractional shares,
configurable spread/slippage/commission, stale-quote and closed-market rejection, position and
exposure caps, a pause switch, idempotent `client_order_id`, and a reset that archives history.
Balances are computed by replaying fills through the Decimal accounting book; the arithmetic is
code, never a language model.

## Agent interfaces

Fifteen validated tools (schemas at `GET /api/v2/tools`) serve both the HTTP API and the MCP
server. Research tools are read-only; `submit_paper_order` requires `simulation: true` and the
admin scope. MCP client config:

```json
{"mcpServers": {"marketvision": {"command": "python", "args": ["/path/to/backend/mcp_server.py"]}}}
```

The `autonomous-ai` project is **not** connected; these interfaces are what it would use later.

## Security

See [SECURITY.md](SECURITY.md) — including the step the owner must take: **rotate the Finnhub key**
that an earlier version committed to this repository.

## Documentation

- [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md) — what was wrong, with how each fault was confirmed
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — the repair plan and its done-tests
- [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) — what was repaired, measured results, what remains
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — free-tier deployment and the (not enabled) scheduled jobs
- [docs/API_INTEGRATIONS.md](docs/API_INTEGRATIONS.md) — every provider, its exact key name, free-tier limits, live-check status, cloud readiness
