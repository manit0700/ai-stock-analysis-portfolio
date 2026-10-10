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

- All keys come from environment variables (`backend/.env.example` lists them); `.env` and
  `.env.*` are git-ignored, and the template holds placeholders only. The browser never receives a
  key: it talks to the backend, which talks to providers. The only frontend variable is
  `NEXT_PUBLIC_API_BASE_URL`, the backend's address.
- Finnhub and NewsAPI keys travel in request **headers** (`X-Finnhub-Token`, `X-Api-Key`) on every
  route. *Correction (2026-10-09):* an earlier version of this file said so while the older
  dashboard routes (`app/services/finnhub_service.py`, `newsapi_service.py`) still put the key in
  the URL. Those services now borrow the same client as the v2 data layer, so the statement is
  true everywhere; `tests/test_api_integrations.py` checks it.
- FRED only accepts its key as a URL parameter, so error messages never include query strings,
  and a logging filter (`app/core/secrets.py`) masks every registered secret and key-shaped
  parameter in every log record, including tracebacks.
- `Secret` wraps each credential so `str()`, `repr()` and accidental serialisation show
  `Secret(set)`, never the value. Health responses report only whether a key is set; the live
  check (`scripts/check_providers.py`, `/api/v2/health/providers`) reports key names and
  provider outcomes and passes all text through the redactor.
- The copilot returns a generic message when the language model fails; the raw exception goes to
  the (redacted) log, not to the browser.

## Rotating a credential

Do this for any key that may have been seen by someone else (pasted into a chat, an issue, a
screenshot, a commit or a frontend bundle). Rotation is the fix; deleting the string is not.

| Key | Where to regenerate | Then |
|---|---|---|
| `FINNHUB_API_KEY` | finnhub.io → Dashboard → API key → regenerate | put it in `backend/.env` |
| `FRED_API_KEY` | fredaccount.stlouisfed.org/apikeys → request a new key, delete the old one | `backend/.env` |
| `NEWSAPI_KEY` | newsapi.org → Account → regenerate the API key | `backend/.env` |
| `MARKETVISION_LLM_API_KEY` | the model provider's console (revoke, create new) | `backend/.env` |
| `MARKETVISION_TOOL_TOKEN`, `MARKETVISION_ADMIN_TOKEN` | you invent them: `python -c "import secrets; print(secrets.token_urlsafe(32))"` | `backend/.env`, and any agent that calls the API |

After rotating, restart the backend and run `python scripts/check_providers.py`. If the same key
is stored in a host's secret store (Render, GitHub Actions secrets), update it there too.

## Least privilege

- Use one key per application and per environment (FRED's terms ask for a separate key per
  application). A key for your laptop and a key for a deployment can then be revoked separately.
- The `MARKETVISION_TOOL_TOKEN` (read) is what research agents should get; keep
  `MARKETVISION_ADMIN_TOKEN` for yourself. No key in this project can move real money: there is
  no brokerage integration.
- NewsAPI's free Developer plan may not be used in staging or production. The application turns
  NewsAPI off when `MARKETVISION_ENV` is `staging` or `production` unless
  `MARKETVISION_NEWSAPI_PAID_PLAN=1` declares a paid plan.

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
