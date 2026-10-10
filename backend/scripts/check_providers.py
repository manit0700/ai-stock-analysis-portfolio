"""Check which data providers actually answer from this machine, with real requests.

    cd backend
    python scripts/check_providers.py                 # table
    python scripts/check_providers.py --json          # machine-readable
    python scripts/check_providers.py --include-llm   # also one tiny chat request (may cost money
                                                      # on a hosted model; free on a local one)

Prints only key *names* (never values), what each provider returned, when the data is from and
how fresh the data layer judged it. Exit status: 0 if every configured provider answered,
1 if any configured provider failed, 2 if nothing could be checked (offline or nothing answered).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.secrets import install_log_redaction, redact  # noqa: E402
from app.marketdata.probe import probe_providers  # noqa: E402

KEY_NAMES = ("FINNHUB_API_KEY", "FRED_API_KEY", "NEWSAPI_KEY", "MARKETVISION_LLM_API_KEY", "MARKETVISION_LLM_BASE_URL", "OPENAI_API_KEY")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--include-llm", action="store_true", help="also send one short request to the configured language model")
    args = parser.parse_args()

    install_log_redaction()
    settings = get_settings()  # loads backend/.env
    report = probe_providers(settings=settings, symbol=args.symbol, include_llm=args.include_llm)
    report["environment"] = {
        "MARKETVISION_ENV": settings.env,
        "keys_set": [name for name in KEY_NAMES if (os.getenv(name) or "").strip()],
        "keys_missing": [name for name in KEY_NAMES if not (os.getenv(name) or "").strip()],
    }

    if args.json:
        print(redact(json.dumps(report, indent=2, default=str)))
    else:
        print(redact(render(report)))

    if not report["live"] or not report["verified"]:
        return 2
    return 1 if report["failed"] else 0


def render(report: dict) -> str:
    lines = [
        f"Provider check at {report['tested_at']} (symbol {report.get('symbol', '-')})",
        f"Environment: {report['environment']['MARKETVISION_ENV']}",
        f"Set:     {', '.join(report['environment']['keys_set']) or 'none'}",
        f"Not set: {', '.join(report['environment']['keys_missing']) or 'none'}",
        "",
    ]
    if not report["live"]:
        lines.append(report["summary"])
        return "\n".join(lines)
    header = f"{'check':22} {'outcome':15} {'ms':>6}  {'freshness':10} {'market time (UTC)':26} detail"
    lines += [header, "-" * len(header)]
    for row in report["checks"]:
        sample = row.get("sample") or {}
        detail = row.get("detail") or ", ".join(f"{k}={v}" for k, v in sample.items())
        if row.get("quality_issues"):
            detail += f"  [quality: {row.get('quality')}: {', '.join(row['quality_issues'])}]"
        lines.append(
            f"{row['check']:22} {row['outcome']:15} {str(row.get('latency_ms', '')):>6}  {str(row.get('freshness') or ''):10} "
            f"{str(row.get('market_timestamp') or ''):26} {detail}"
        )
    lines += ["", report["summary"]]
    lines.append("Answered: " + (", ".join(report["verified"]) or "none"))
    if report["failed"]:
        lines.append("Failed:   " + ", ".join(report["failed"]))
    if report["not_configured"]:
        lines.append("Not configured: " + ", ".join(report["not_configured"]))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
