from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.market import MarketDataService
from app.services.signal_ledger import resolve_pending_signals, summarize_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve MarketVision AI pending signal outcomes.")
    parser.add_argument("--ledger", type=Path, default=None, help="Optional JSONL ledger path. Defaults to backend/data/signal_ledger.jsonl.")
    parser.add_argument("--max-rows", type=int, default=1000, help="Maximum pending rows to evaluate.")
    args = parser.parse_args()

    market_service = MarketDataService()
    result = resolve_pending_signals(
        market_service=market_service,
        max_rows=args.max_rows,
        ledger_path=args.ledger,
    )
    result["performance"] = summarize_signals(ledger_path=args.ledger)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
