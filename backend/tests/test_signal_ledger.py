from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.signal_ledger import resolve_pending_signals, summarize_signals


class FakeMarketService:
    def __init__(self, histories: dict[str, pd.DataFrame]) -> None:
        self.histories = histories

    def get_history(self, ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
        return self.histories[ticker.upper()]


def _history(high: list[float], low: list[float], close: list[float], start: datetime) -> pd.DataFrame:
    index = pd.date_range(start.replace(tzinfo=None) + timedelta(days=1), periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": [1_000_000] * len(close),
        },
        index=index,
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _base_row(ticker: str, recorded_at: datetime, confidence: float, quality: str = "high_confidence_trade_candidate") -> dict[str, Any]:
    return {
        "recorded_at": recorded_at.isoformat(),
        "source": "test",
        "ticker": ticker,
        "quality_label": quality,
        "action": "paper_trade_candidate",
        "confidence": confidence,
        "risk_score": 40,
        "dominant_scenario": "bullish",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "target_1": 105.0,
        "target_2": 110.0,
        "interval": "1d",
        "horizon_steps": 1,
        "outcome": "pending",
    }


def test_resolve_pending_signals_marks_win_loss_partial_and_pending(tmp_path: Path) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=5)
    recent = datetime.now(timezone.utc)
    ledger = tmp_path / "signal_ledger.jsonl"
    rows = [
        _base_row("WIN", old, 82.0),
        _base_row("LOSS", old, 72.0, "watchlist_candidate"),
        _base_row("PART", old, 62.0),
        _base_row("WAIT", recent, 92.0),
        {**_base_row("BAD", old, 55.0), "entry_price": None},
    ]
    _write_rows(ledger, rows)

    market = FakeMarketService(
        {
            "WIN": _history([101, 106, 108], [99, 100, 103], [100, 106, 107], old),
            "LOSS": _history([101, 102, 103], [99, 94, 93], [100, 96, 94], old),
            "PART": _history([101, 106, 107], [99, 94, 98], [100, 104, 103], old),
            "WAIT": _history([101, 104, 104], [99, 98, 98], [100, 102, 103], recent),
        }
    )

    result = resolve_pending_signals(market_service=market, ledger_path=ledger)
    summary = summarize_signals(ledger_path=ledger)

    resolved_rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    outcomes = {row["ticker"]: row["outcome"] for row in resolved_rows}
    assert outcomes["WIN"] == "resolved_win"
    assert outcomes["LOSS"] == "resolved_loss"
    assert outcomes["PART"] == "partial_win"
    assert outcomes["WAIT"] == "still_pending"
    assert outcomes["BAD"] == "invalid"
    assert result["updated"] == 5
    assert result["resolved"] == 4
    assert summary["total_predictions"] == 5
    assert summary["pending_signals"] == 1
    assert summary["resolved_signals"] == 4
    assert summary["win_rate"] == 0.25
    assert summary["loss_rate"] == 0.25
    assert summary["partial_win_rate"] == 0.25
    assert summary["promoted_signal_performance"]["resolved"] == 3
    assert summary["quality_performance"]["watchlist_candidate"]["losses"] == 1
    assert summary["confidence_bucket_performance"]["80-90%"]["wins"] == 1
    assert "drift_monitoring" in summary
    assert summary["drift_monitoring"]["recent_resolved_count"] == 3
    assert summary["drift_monitoring"]["feature_drift"]["status"] == "not_enough_feature_history"
