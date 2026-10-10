# API integrations: inventory, security, configuration, cloud readiness

**Date:** 2026-10-09 · **Branch:** `repair/marketvision-audit` (local; not pushed)

**Live-testing status.** The machine this work was done on cannot reach any market-data
provider (its network policy blocks Yahoo, Finnhub, FRED and NewsAPI). So every provider below is
**implemented and tested against recorded response shapes, but not verified live**. The live
check you run yourself (section C) is what verifies a provider. A key being set is not proof
that a provider works.

## A. API integration inventory

| Provider | Purpose | Exact environment variable | Implemented? | Free tier | Configured here | Live-tested | Limitations |
|---|---|---|---|---|---|---|---|
| Yahoo Finance (`yfinance` 1.7.0) | Daily/intraday OHLCV, splits, dividends, fallback quote, profile, news | none | Yes: `app/marketdata/providers/yahoo.py` | Free, unofficial | n/a | **No.** Network blocked: `ConnectionError`, proxy 403 | Unofficial, "intended for personal use only" (yfinance README); can be delayed; never labelled real-time. Raw prices plus `Adj Close` (`auto_adjust=False`); splits/dividends kept as columns |
| Finnhub | Quotes (primary), profile, company/market news, earnings calendar, basic financials, analyst trends | `FINNHUB_API_KEY` | Yes: provider + older dashboard service | Free key; 60 calls/min | Not set | **No** | Key in `X-Finnhub-Token` header. Budget 0.9/s, burst 5 (≤ 59 in any first minute), shared process-wide. HTTP 429 → `rate_limited`, 401/403 → `provider_auth`. Quote labelled `live` only while the market is open and the quote is ≤ 60 s old |
| FRED | Interest rates, CPI, unemployment, GDP, yields, VIX | `FRED_API_KEY` (optional) | Yes: provider + older macro service | Free key; ~120 req/min | Not set | **No.** Public CSV failed with `ProxyError` | Without a key, FRED's public CSV is used. The key must go in the URL (FRED allows nothing else), so errors drop query strings and logs are redacted. "." observations become missing, never filled. Revisions noted on every response |
| NewsAPI | Headlines and search, sentiment inputs | `NEWSAPI_KEY` (**not** `NEWSAPI_API_KEY`) | Yes: provider + older dashboard service | Developer plan: 100 req/day, 24 h delay, a month of history, **development only** | Not set | **No** | Key in `X-Api-Key` header. **Off automatically in staging/production** unless `MARKETVISION_NEWSAPI_PAID_PLAN=1`. Every article carries `age_hours`; `stale_news` warning if the newest is > 48 h old |
| AI copilot (OpenAI-compatible) | Plain-language explanations of computed facts | `MARKETVISION_LLM_API_KEY` (or `OPENAI_API_KEY`), `MARKETVISION_LLM_BASE_URL`, `MARKETVISION_LLM_MODEL` | Yes, **fixed in this pass** (it used to ignore these three) | Free with a local server; hosted models are paid | Not set | **No** | Needs `pip install -r requirements-heavy.txt` (openai client). Off by default. Generic error messages; instructed never to state numbers absent from its context |
| Anthropic Claude | Possible hosted model behind the copilot | Uses the copilot variables; `ANTHROPIC_API_KEY` is **not read** | Possible through the copilot, not tested | **Paid** per token | Not set | **No** (not activated, per your instruction) | Claude's OpenAI-compatible endpoint is described by Anthropic as "not considered a long-term or production-ready solution" for most uses; the copilot already omits `temperature` for it (recent models reject values below 1) |
| Alpha Vantage | Extra history, indicators | none (not implemented) | **No** adapter | 25 requests/day | n/a | n/a | Too small a quota to add value beyond Yahoo + Finnhub for this app. Future option only |
| Massive (formerly Polygon) | Licensed US stock data | none (not implemented) | **No** adapter | Stocks Basic: 5 calls/min, end-of-day data, 2 years of history | n/a | n/a | Worth it later as an *official* end-of-day backup to Yahoo for building snapshots. Free tier has no intraday/real-time quotes. Check its docs for the credential format before building |

Supporting pieces (no third-party key): NYSE calendar (computed), data snapshots (local files),
the $50 paper account (SQLite), the prediction ledger (append-only file), MCP server (stdio).

## B. Security report

**Exposed-key findings**
- The Finnhub key committed in an earlier `frontend-next/lib/useFinnhubWS.ts` is in the public git
  history. **It must be revoked and replaced; do not reuse it.** This was not done automatically
  (it needs your Finnhub account), and git history was not rewritten.
- Scan of every tracked file on this branch: `python backend/scripts/scan_secrets.py` →
  "No secrets found in the scanned files." The test suite also scans the built frontend bundle
  (`.next`) when one exists: clean.
- No `.env` file is tracked; `.gitignore` excludes `.env`, `.env.*`, `kaggle.json`, `*.pem`,
  `*.key`. `backend/.env.example` holds placeholders only.

**Found and fixed in this pass**
1. The older dashboard routes (company research, market news, macro, the copilot) used their own
   HTTP code that put the Finnhub key (`token=`) and NewsAPI key (`apiKey=`) **in the URL**,
   with no shared rate limit and no cache. `SECURITY.md` wrongly said headers were used
   everywhere; it now carries a correction. Both services now borrow the v2 client
   (`app/marketdata/clients.py`): keys in headers, one free-plan budget per provider, caches.
2. The copilot read only `OPENAI_API_KEY` with a hard-coded model, so the documented
   `MARKETVISION_LLM_*` variables did nothing; and it returned raw exception text to the browser.
   Both fixed.
3. Macro scores were computed from neutral defaults when **no** FRED data arrived, and the
   overview card said "positive" with no data at all. Now: "unavailable", with the missing
   series named.
4. NewsAPI's free plan "cannot be used in a staging or production environment" — now enforced.

**Frontend exposure.** The browser holds no keys. Its only build variable is
`NEXT_PUBLIC_API_BASE_URL`, the backend's address. Quotes and news reach it through the backend.

**Log redaction.** Every log handler runs `RedactingFilter`: registered secret values and
`token=`/`api_key=`/`apiKey=`/`X-Api-Key:`-style parameters are masked, tracebacks included.
Provider errors name the provider, path and HTTP status only. Health responses say whether a key
is *set*, never its value. The live check prints key names only.

**Remaining risks**
- Until the old Finnhub key is revoked, anyone can use your Finnhub quota with it.
- Rate budgets are per process: two uvicorn workers get two budgets. Run one worker on free plans.
- The tool tokens are bearer tokens: anyone holding the admin token can place *simulated* orders
  and reset the paper account. Development runs open when no tokens are set; keep it on localhost.
- Third-party libraries (yfinance, openai) log on their own. They pass through the root
  redaction filter, but their messages are outside this code's control.

## C. Configuration instructions

**Accounts to create** (all free):

| Key | Sign up / get key | Notes |
|---|---|---|
| `FINNHUB_API_KEY` | finnhub.io → register → Dashboard shows the key | If you already have an account, **regenerate** the key (the old one is public) |
| `FRED_API_KEY` | fred.stlouisfed.org → create account → fredaccount.stlouisfed.org/apikeys → request key | Optional: without it the public CSV works |
| `NEWSAPI_KEY` | newsapi.org → "Get API key" | Free plan is for development on your Mac only |
| Copilot (optional, free) | A local OpenAI-compatible server (for example the llama.cpp / Ollama server from your local-models work) | Set `MARKETVISION_LLM_BASE_URL=http://127.0.0.1:<port>/v1` and `MARKETVISION_LLM_MODEL=<model name>`; no key needed |

**Where to store them.** Only in `~/marketvision/backend/.env` on your Mac (copy it from
`.env.example`). For a hosted backend, in the host's secret store (Render "Environment", GitHub
"Actions secrets"), never in a file in the repo and never in the frontend.

```bash
FINNHUB_API_KEY=<new key>
FRED_API_KEY=<key>
NEWSAPI_KEY=<key>
```

**How to verify they work.** On your Mac:

```bash
cd ~/marketvision/backend && source .venv/bin/activate
python scripts/check_providers.py          # one small real request per provider
```

It prints, for each provider: answered / failed / not configured, latency, the market time of
the data, how fresh the data layer judged it, and any quality issues. Exit code 0 means every
configured provider answered. With the backend running you can also open
`http://127.0.0.1:8000/api/v2/health/providers`, and then confirm by hand:

- `http://127.0.0.1:8000/api/v2/stocks/AAPL/quote` — `provider`, `freshness` (live only during
  market hours with a fresh Finnhub quote), `market_timestamp`, `quality`.
- `http://127.0.0.1:8000/api/v2/stocks/AAPL/forecast` — should now return a forecast instead of
  "unavailable", with `"edge_established": false` and the base rate labelled as a baseline.
- `http://127.0.0.1:8000/api/v2/news?symbol=AAPL` — each item has `age_hours`.

Alternatively, after you push the branch: add the three keys as GitHub Actions secrets and start
the **provider-check** workflow by hand (Actions tab → provider-check → Run workflow). It is
manual-only, prints no secret values and places no orders.

## D. Cloud readiness

| Component | Free? | Notes |
|---|---|---|
| CI tests, secret scan, frontend build (GitHub Actions) | **Free** | Standard runners are free on public repos; private repos get 2,000 minutes/month on GitHub Free. Jobs stop at 6 h |
| Manual live provider check (`provider-check.yml`) | **Free** | Manual trigger only |
| Scheduled forecasts + ledger resolution (GitHub Actions cron) | **Free, with an architecture change** | Runners have no persistent disk: the job must commit the ledger and paper database to a (private) repo or another store. Schedules can be delayed under load, run at most every 5 min, and are disabled after 60 days without activity in a public repo. Template in `OPERATIONS.md`, not enabled |
| Backend on Render free | **Free but limited** | Sleeps when idle; **no persistent disk**, so the ledger and paper account reset on restart. A disk is paid |
| Frontend on Cloudflare Pages / Vercel | **Likely free, not tested** | The dashboard fetches from the backend in the browser. It needs a static export or an adapter; `next build` passes, but static export has not been tried |
| Backend on Cloudflare Workers (Python) | **Needs a rewrite** | The free plan allows 10 ms CPU per request and 128 MB of memory, and packages must be pure Python or Pyodide builds. FastAPI is supported, but this backend's pandas/scikit-learn/yfinance stack and its SQLite file are not a fit. Realistic split: a small Worker serving stored results, with the computation elsewhere |
| Paper records on Cloudflare D1 | **Free tier is ample** (5 M rows read and 100 k written per day, 5 GB) | Needs a new storage adapter for the paper account and ledger (they use local SQLite and files today), plus re-testing of the double-spend protections |
| Model evaluation on Kaggle | **Free** | No GPU needed. Upload `backend/` manually and no Kaggle API token is needed; `kaggle.json` is only for automation (git-ignored). Internet and GPUs inside notebooks have required phone verification (check current Kaggle settings; not verified here). Kaggle is not model hosting |
| Hosted language model (Claude, OpenAI) | **Paid** | Not activated. Local open-weight models keep it free but need a machine that is on |
| NewsAPI in production | **Paid** plan required | Free plan is development-only |
| Continuous 24/7 operation without your Mac | **Not available free as-is** | Nothing here keeps a disk-backed process running 24/7 at no cost. Closest free path: daily GitHub Actions job + state committed to a private repo + static dashboard. Claude Code is a development tool, not a hosting service |

## E. Test results (as run on 2026-10-09)

- **Backend suite:** `241 passed` (`python -m pytest`, Python 3.13): the previous 219 plus 22
  new tests in `tests/test_api_integrations.py`. The new tests cover:
  - header-only keys on the older routes, caching, and redacted failures;
  - shared clients and budgets inside the published limits;
  - the NewsAPI production guard, article ages and the stale flag;
  - FRED missing values and the no-data refusal;
  - copilot configuration (local server, Claude endpoint without temperature, generic errors);
  - the live probe's classification and redaction;
  - the probe route's auth scope.

  All use stand-in HTTP sessions: **they show the code is correct, not that any provider works.**
- **Secret scan:** clean. Frontend bundle scan (inside the suite): clean.
- **Live provider check from this machine** (`scripts/check_providers.py`, exit code 2):

  | Check | Outcome | Reason |
  |---|---|---|
  | yahoo_history | failed | `ConnectionError` (network policy; proxy 403) |
  | yahoo_quote | failed | no data (same cause) |
  | finnhub_quote, finnhub_company_news | not configured | no key on this machine |
  | fred_series | failed | public CSV: `ProxyError` |
  | newsapi_news | not configured | no key on this machine |

- **Providers verified live:** none. **Not verified:** Yahoo, Finnhub, FRED, NewsAPI, copilot.

## Agent-tool safety (Part 8), as checked in code

The 17 tools (the 15 planned plus `get_market_calendar` and `get_model_status`) serve both HTTP
(`/api/v2/tools`) and MCP (`backend/mcp_server.py`). 16 have the `read` permission. One,
`submit_paper_order`, has `paper_trade`, needs the admin scope and requires `simulation: true`.

The paper account is long-only, with no margin, leverage or shorting, and never lets cash go
negative. It rejects stale quotes, quotes of unknown age and orders while the market is closed,
and falls back to HOLD whenever data is missing. Every fill and every forecast goes into the
SQLite book or the append-only, hash-chained ledger. No brokerage code exists anywhere.

One nuance: `get_stock_prediction` is classed `read`, but by default (`record: true`) it appends
the issued forecast to the prediction ledger. That write is deliberate, so every prediction an
agent sees can be audited and scored later. It moves no money.

Forecasts report `edge_established: false`, because no model has passed the promotion rule. The
projects are **not** connected; that waits for your explicit go-ahead.

## F. Remaining actions for you

1. Revoke the old Finnhub key and create a new one (finnhub.io).
2. Create free FRED and NewsAPI keys; put all three in `~/marketvision/backend/.env`.
3. Run `python scripts/check_providers.py` on your Mac and send me the output (it contains no keys).
4. Review the branch, then push it yourself when satisfied.
5. Decide later, and only if you want to: a local model for the copilot (free), or a paid hosted
   model; a scheduled GitHub Actions job (needs state stored in a repo); Massive as an official
   end-of-day data source.

Sources: newsapi.org/pricing; platform.claude.com/docs (OpenAI SDK compatibility);
fred.stlouisfed.org/docs/api/api_key.html; github.com/ranaroussi/yfinance; alphavantage.co/premium;
massive.com/pricing; docs.github.com (Actions limits and schedule events);
developers.cloudflare.com (Workers limits, Python packages, D1 pricing). The Finnhub limit is
from third-party write-ups, because Finnhub's own pricing page could not be read here.
