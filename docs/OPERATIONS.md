# Operating MarketVision on free infrastructure

The application is sized for a 16 GB Apple Silicon Mac, GitHub Actions' free tier, and free-tier
hosting. Nothing requires a GPU, a paid API or a paid data subscription.

## Resource profile

| Piece | Footprint |
|---|---|
| Core backend (`requirements.txt`) | ~400 MB of packages, well under 512 MB RAM serving |
| Evaluation (`requirements-ml.txt`, `python -m app.forecast`) | ran in ~19 minutes on 2 CPU cores over 123k rows; no GPU |
| FinBERT (`requirements-heavy.txt`, optional) | ~2 GB of packages; keep `MARKETVISION_ENABLE_FINBERT=0` unless you need it — VADER is the fallback |
| Frontend | static Next.js build, no server state |

Caching keeps the app inside free quotas: history 15 min (plus a disk cache for fully-past
ranges), quotes 10 s, news 10 min, FRED 6 h, plus per-provider token buckets and a circuit
breaker. One prediction makes one history request per context instrument, shared across symbols
in the same run.

## Reduced-capability mode

Every provider key is optional. Missing keys remove capabilities (visible at `/api/v2/health`)
without breaking the app: FRED falls back to its public CSV, news falls back across providers,
sentiment falls back to VADER, and the forecast declines to answer rather than guess when inputs
are missing. `MARKETVISION_OFFLINE=1` cuts the network entirely; data snapshots still work.

## Render (free tier)

`render.yaml` deploys the backend with `requirements-render.txt`. Two things to know:

1. **No persistent disk**: the prediction ledger and the paper-trading SQLite database are lost
   on each restart. Attach a paid disk mounted at `backend/data`, or treat the deployment as a
   stateless research API and keep the ledger on a machine with a disk.
2. Set `MARKETVISION_ADMIN_TOKEN`, `MARKETVISION_TOOL_TOKEN` and `MARKETVISION_CORS_ORIGINS`
   (your Vercel origin). In production the admin routes refuse to run without a token.

Do not promise continuous real-time operation on free infrastructure: free web services sleep,
and NewsAPI's free plan is 24 h delayed (the response says so).

## Scheduled analysis (template — NOT enabled)

A daily job can issue forecasts after the close and resolve due predictions. It is not enabled
because GitHub Actions has no persistent disk for the ledger; run it on a machine that keeps
`backend/data`, or commit the ledger to a private repository from the job.

```yaml
# .github/workflows/daily.yml — template only
# on:
#   schedule: [{cron: "30 22 * * 1-5"}]   # 22:30 UTC, after the NYSE close
# jobs:
#   daily:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with: {python-version: "3.12"}
#       - run: pip install -r backend/requirements.txt
#       - run: |
#           cd backend
#           python - <<'PY'
#           from app.agent_tools.tools import call_tool
#           for symbol in ["AAPL", "MSFT", "NVDA", "SPY"]:
#               print(call_tool("get_stock_prediction", {"symbol": symbol})["result"]["status"])
#           PY
#       - run: cd backend && python -c "from app.predictions.ledger import PredictionLedger; from app.marketdata.service import get_market_data; print(PredictionLedger().resolve_due(get_market_data()))"
#       # then persist backend/data somewhere durable
```

## Kaggle

Model evaluation needs no GPU, but a larger universe or more bootstrap resamples can run on
Kaggle: upload `backend/`, `pip install -r requirements-ml.txt`, build or upload a snapshot, and
run `python -m app.forecast evaluate --snapshot <id>`. Bring back `backend/models/<run>/` —
`card.json` is the record; `promote` still only accepts a run that passed the rule.

## Retraining cadence

A promoted model records `trained_through`. Re-run the evaluation when meaningfully more data
exists (quarterly is plenty for daily bars); the holdout log counts how often a holdout has been
scored, and the count is printed on every card as a multiple-testing warning.
