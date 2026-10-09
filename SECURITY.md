# Security

## Action required: rotate the exposed Finnhub key

A Finnhub API key was committed to this public repository in `frontend-next/lib/useFinnhubWS.ts`
(removed on the `repair/marketvision-audit` branch) and was also served to every browser that
loaded the dashboard. Deleting the file does **not** help by itself: the key remains in the git
history of every clone.

1. Sign in at finnhub.io and regenerate the API key (dashboard → API key → regenerate). The old
   key stops working at that moment.
2. Put the new key only in `backend/.env` as `FINNHUB_API_KEY=...` (never in frontend code).
3. Optional: rewriting the git history (`git filter-repo`) removes the string from the repository,
   but every fork and clone keeps it, so rotation is the fix; history rewriting is cosmetic.

Rotation was **not** performed automatically; it needs the account owner.

## How credentials are handled now

- All keys come from environment variables (`backend/.env.example` lists them); `.env` is
  git-ignored. The browser never receives a key: it talks to the backend, which talks to providers.
- Keys travel in request **headers** (`X-Finnhub-Token`, `X-Api-Key`) wherever the provider allows.
  FRED only accepts a URL parameter, so error messages never include query strings, and a logging
  filter (`app/core/secrets.py`) masks every registered secret and key-shaped parameter in every
  log record, including tracebacks.
- `Secret` wraps each credential so `str()`, `repr()` and accidental serialisation show
  `Secret(set)`, never the value.

## Access control

- Two bearer scopes (`app/core/security.py`), compared in constant time:
  - `MARKETVISION_TOOL_TOKEN` (read): research routes that spend provider quota.
  - `MARKETVISION_ADMIN_TOKEN` (admin): anything that changes state or spends money-like
    resources — paper orders, pause/reset, resolving predictions, the LLM copilot.
  - The admin token also satisfies read. With no tokens set, a development machine runs open;
    in production (`MARKETVISION_ENV=production`) admin routes refuse to run until a token is set.
- CORS is an explicit allow-list (`MARKETVISION_CORS_ORIGINS`), not `*` with credentials.
- In-memory rate limits sit on the scanner, news search, copilot, forecasts and paper orders.

## Secret scanning

- `python backend/scripts/scan_secrets.py` scans every tracked file; `--staged` is made for a
  pre-commit hook:

      echo 'python backend/scripts/scan_secrets.py --staged' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

- The test suite runs the same scanner (`backend/tests/test_security_scan.py`), so CI fails if a
  key-shaped string lands in the tree, and it checks that a built frontend bundle (when present)
  does not contain one either.

## Simulated trading boundary

There is no brokerage integration anywhere in this codebase: no broker SDK, no account
credentials, no order routing. The "trading" surface is `app/paper` (SQLite, virtual $50), whose
orders require `simulation=true` and whose arithmetic is plain code. Keep it that way: a change
that connects a real brokerage should be treated as a new security review, not a feature patch.

## Reporting

This is a personal research project; report problems as GitHub issues on the repository (without
including live keys in the report).
