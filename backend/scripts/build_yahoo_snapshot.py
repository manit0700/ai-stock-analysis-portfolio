"""Build a current data snapshot from Yahoo Finance and FRED, for model evaluation.

Run it on a machine that can reach Yahoo Finance (for example your own Mac):

    cd backend && source .venv/bin/activate
    python scripts/build_yahoo_snapshot.py                 # ~100 stocks + context, 1999 to the last close
    python scripts/build_yahoo_snapshot.py --quick         # 10 stocks, for a fast check

It reads every price through the application's one market-data door (``MarketData``), so the
bars get the same normalisation and quality checks as everything else, and writes them with
``SnapshotStore``: CSV files plus a manifest with each file's SHA-256. Nothing is invented: a
symbol Yahoo cannot supply is left out and listed in the manifest.

The stock list (``UNIVERSE``) is approximately the S&P 100 as it stood at the end of 2021,
compiled from memory, not an official constituent file. Choosing it as of the start of the
2022+ holdout keeps hindsight out of that period: no company is included because of what it did
after 2021. Two limits remain and are recorded in the manifest:
    * companies that were delisted, acquired or renamed before 2026 are under-represented,
      because Yahoo serves current tickers only;
    * before 2022 the list is still hindsight-chosen (these were large companies in 2021), so
      development-period results are flattered by survivorship bias.
Yahoo Finance is unofficial, "intended for personal use only" (yfinance README); keep the
snapshot on your own machine.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import MarketVisionError  # noqa: E402
from app.core.secrets import install_log_redaction  # noqa: E402
from app.features.context import KNOWN_SECTORS  # noqa: E402
from app.marketdata import calendar as cal  # noqa: E402
from app.marketdata.service import build_market_data  # noqa: E402
from app.marketdata.snapshots import SnapshotStore  # noqa: E402

#: Approximately the S&P 100 at the end of 2021 (Facebook listed under its current ticker META;
#: GOOG left out in favour of GOOGL; UTX history is served by Yahoo under RTX).
UNIVERSE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMGN", "AMT", "AMZN", "AVGO", "AXP", "BA", "BAC", "BIIB", "BK",
    "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOGL", "GS", "HD",
    "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET",
    "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V",
    "VZ", "WBA", "WFC", "WMT", "XOM",
]
QUICK_UNIVERSE = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "NEE", "LIN", "AMT"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "XLRE"]
CONTEXT = ["SPY", "QQQ", "^VIX", *SECTOR_ETFS]
SERIES = ["DGS10", "T10Y2Y"]
START = date(1999, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="10 stocks only, for a fast check")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    install_log_redaction()

    settings = get_settings()
    if settings.offline:
        print("MARKETVISION_OFFLINE=1 is set; this script needs the network.")
        return 2
    md = build_market_data(settings)
    end = cal.last_completed_session(md.clock())
    universe = QUICK_UNIVERSE if args.quick else UNIVERSE
    name = args.name or ("yahoo-quick" if args.quick else "yahoo-daily")
    missing_sector = [s for s in universe if s not in KNOWN_SECTORS]
    if missing_sector:
        print(f"Every stock needs a sector ETF in app/features/context.py KNOWN_SECTORS; missing: {missing_sector}")
        return 2

    print(f"Downloading {len(universe)} stocks and {len(CONTEXT)} context instruments, {START} to {end} ...")
    frames, failed, quality = {}, {}, {}
    for i, symbol in enumerate([*CONTEXT, *universe], 1):
        for attempt in range(3):
            try:
                envelope = md.history(symbol, start=START, end=end, interval="1d", use_cache=False)
            except MarketVisionError as exc:
                reason = exc.message
                time.sleep(2.0 * (attempt + 1))
                continue
            frames[symbol] = envelope.data
            quality[symbol] = {"status": envelope.quality.status, "issues": [f"{x.code}:{x.count}" for x in envelope.quality.issues if x.code != "zero_volume"]}
            break
        else:
            failed[symbol] = reason
        status = "ok" if symbol in frames else "FAILED"
        rows = len(frames[symbol]) if symbol in frames else 0
        print(f"  [{i:3d}] {symbol:6s} {status:6s} {rows:6d} bars", flush=True)

    for needed in ("SPY", "^VIX"):
        if needed not in frames:
            print(f"{needed} could not be downloaded; a snapshot without it cannot be evaluated. Nothing was written.")
            return 1

    series, series_failed = {}, {}
    for series_id in SERIES:
        try:
            series[series_id] = md.economic_series(series_id, start=START, end=end).data
        except MarketVisionError as exc:
            series_failed[series_id] = exc.message
    stocks = [s for s in universe if s in frames]

    store = SnapshotStore(settings.snapshot_dir)
    manifest = store.create(
        name,
        frames,
        series,
        sources=[
            {"provider": "Yahoo Finance via yfinance (unofficial; personal use)", "symbols": sorted(frames)},
            {"provider": "FRED, Federal Reserve Bank of St. Louis", "route": "json_api" if settings.fred_api_key else "public_csv", "series": sorted(series)},
        ],
        description=f"Yahoo daily bars {START} to {end}: {len(stocks)} large US stocks (approx. S&P 100 at end of 2021), SPY, QQQ, VIX, 11 sector ETFs; FRED DGS10 and T10Y2Y.",
        metadata={
            "context_only": [s for s in CONTEXT if s in frames],
            "universe_rule": "approximately the S&P 100 at the end of 2021, compiled from memory (not an official constituent file)",
            "sectors": {s: KNOWN_SECTORS[s] for s in stocks},
            "quality": quality,
            "failed": {**failed, **series_failed},
            "requested_end": end.isoformat(),
            "limits": [
                "Universe fixed as of end-2021: no hindsight inside a 2022+ holdout, but companies delisted or acquired since are missing (Yahoo serves current tickers only).",
                "Before 2022 the list is hindsight-chosen (large in 2021): development results are flattered by survivorship bias.",
                "Spin-offs show as large one-day moves in Close; Adj Close (used for targets) absorbs them where Yahoo records the distribution (checked: MO 2008-03-31, -69.9% close, -2.5% adjusted).",
                "Sector ETFs XLC (2018) and XLRE (2015) are young: their stocks have no sector features before then.",
                "FRED yields are as published now, after revisions; Treasury yields are rarely revised.",
            ],
        },
    )
    print(f"\nsnapshot {manifest['id']}: {len(stocks)} stocks + {len(manifest['metadata']['context_only'])} context, {manifest['start']} to {manifest['end']}")
    if failed or series_failed:
        print("not downloaded: " + ", ".join(f"{k} ({v})" for k, v in {**failed, **series_failed}.items()))
    flagged = {s: q for s, q in quality.items() if q["status"] != "ok"}
    for symbol, q in sorted(flagged.items()):
        print(f"  quality {symbol:6s} {q['status']:9s} {', '.join(q['issues'])}")
    print(f"written to {store.path(manifest['id'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
