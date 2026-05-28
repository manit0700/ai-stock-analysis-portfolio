from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_ROOT = Path(__file__).resolve().parents[3]
_LEDGER_PATH = _ROOT / "backend" / "data" / "signal_ledger.jsonl"


def _ensure_parent() -> None:
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)


def record_signal(payload: dict[str, Any]) -> None:
    _ensure_parent()
    row = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with _LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def read_signals(limit: int = 5000) -> list[dict[str, Any]]:
    if not _LEDGER_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with _LEDGER_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def write_signals(rows: list[dict[str, Any]]) -> None:
    _ensure_parent()
    with _LEDGER_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def resolve_pending_signals(market_service: Any, max_rows: int = 1000) -> dict[str, Any]:
    rows = read_signals(limit=max_rows)
    changed = 0
    checked = 0
    skipped = 0
    now = datetime.now(timezone.utc)

    for row in rows:
        if row.get("outcome") != "pending":
            continue
        recorded_at_raw = row.get("recorded_at")
        if not recorded_at_raw:
            skipped += 1
            continue
        try:
            recorded_at = datetime.fromisoformat(recorded_at_raw.replace("Z", "+00:00"))
        except ValueError:
            skipped += 1
            continue
        age_hours = (now - recorded_at).total_seconds() / 3600
        if age_hours < 18:
            skipped += 1
            continue

        ticker = row.get("ticker")
        entry = row.get("entry_price")
        target = row.get("target_1")
        stop = row.get("stop_loss")
        if not ticker or entry is None or target is None or stop is None:
            skipped += 1
            continue

        try:
            history = market_service.get_history(ticker=ticker, period="5d", interval="1d")
        except Exception:
            skipped += 1
            continue
        if history.empty:
            skipped += 1
            continue

        checked += 1
        future = history[pd.to_datetime(history.index) >= recorded_at.replace(tzinfo=None)]
        if future.empty:
            future = history.tail(1)
        max_high = float(future["High"].max())
        min_low = float(future["Low"].min())
        latest_close = float(future["Close"].iloc[-1])

        direction = "short" if str(row.get("dominant_scenario")) == "bearish" else "long"
        if direction == "short":
            hit_target = min_low <= float(target)
            hit_stop = max_high >= float(stop)
            pnl_pct = (float(entry) / latest_close - 1) if latest_close else 0
        else:
            hit_target = max_high >= float(target)
            hit_stop = min_low <= float(stop)
            pnl_pct = (latest_close / float(entry) - 1) if entry else 0

        if hit_target and not hit_stop:
            outcome = "win"
        elif hit_stop and not hit_target:
            outcome = "loss"
        elif hit_target and hit_stop:
            outcome = "ambiguous"
        elif age_hours >= 72:
            outcome = "expired_win" if pnl_pct > 0 else "expired_loss"
        else:
            skipped += 1
            continue

        row["outcome"] = outcome
        row["resolved_at"] = now.isoformat()
        row["latest_close_at_resolution"] = round(latest_close, 4)
        row["resolution_pnl_pct"] = round(pnl_pct * 100, 4)
        row["hit_target_1"] = hit_target
        row["hit_stop_loss"] = hit_stop
        changed += 1

    if changed:
        write_signals(rows)
    return {
        "checked_pending": checked,
        "resolved": changed,
        "skipped": skipped,
        "ledger_path": str(_LEDGER_PATH),
    }


def summarize_signals() -> dict[str, Any]:
    rows = read_signals()
    quality_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    quality_performance: dict[str, dict[str, int | float | None]] = {}
    action_performance: dict[str, dict[str, int | float | None]] = {}
    confidence_buckets = {
        ">=60": {"count": 0, "resolved": 0, "wins": 0},
        ">=70": {"count": 0, "resolved": 0, "wins": 0},
        ">=80": {"count": 0, "resolved": 0, "wins": 0},
        ">=85": {"count": 0, "resolved": 0, "wins": 0},
    }
    promoted = 0
    pending = 0
    resolved = 0
    wins = 0

    for row in rows:
        quality = row.get("quality_label") or "unknown"
        action = row.get("action") or "unknown"
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        quality_performance.setdefault(quality, {"count": 0, "resolved": 0, "wins": 0, "win_rate": None})
        action_performance.setdefault(action, {"count": 0, "resolved": 0, "wins": 0, "win_rate": None})
        quality_performance[quality]["count"] = int(quality_performance[quality]["count"] or 0) + 1
        action_performance[action]["count"] = int(action_performance[action]["count"] or 0) + 1
        if quality == "high_confidence_trade_candidate":
            promoted += 1

        outcome = row.get("outcome", "pending")
        if outcome == "pending":
            pending += 1
        else:
            resolved += 1
            quality_performance[quality]["resolved"] = int(quality_performance[quality]["resolved"] or 0) + 1
            action_performance[action]["resolved"] = int(action_performance[action]["resolved"] or 0) + 1
            if outcome in {"win", "expired_win"}:
                wins += 1
                quality_performance[quality]["wins"] = int(quality_performance[quality]["wins"] or 0) + 1
                action_performance[action]["wins"] = int(action_performance[action]["wins"] or 0) + 1

        confidence = float(row.get("confidence") or 0)
        for threshold, bucket in [(60, ">=60"), (70, ">=70"), (80, ">=80"), (85, ">=85")]:
            if confidence >= threshold:
                confidence_buckets[bucket]["count"] += 1
                if outcome != "pending":
                    confidence_buckets[bucket]["resolved"] += 1
                    if outcome in {"win", "expired_win"}:
                        confidence_buckets[bucket]["wins"] += 1

    for bucket in confidence_buckets.values():
        bucket["win_rate"] = round(bucket["wins"] / bucket["resolved"], 4) if bucket["resolved"] else None
    for group in (quality_performance, action_performance):
        for stats in group.values():
            resolved_count = int(stats["resolved"] or 0)
            stats["win_rate"] = round(int(stats["wins"] or 0) / resolved_count, 4) if resolved_count else None

    return {
        "ledger_path": str(_LEDGER_PATH),
        "total_logged_predictions": len(rows),
        "promoted_signals": promoted,
        "pending_outcomes": pending,
        "resolved_outcomes": resolved,
        "resolved_win_rate": round(wins / resolved, 4) if resolved else None,
        "quality_counts": quality_counts,
        "action_counts": action_counts,
        "quality_performance": quality_performance,
        "action_performance": action_performance,
        "confidence_buckets": confidence_buckets,
        "note": "Live outcomes are pending until future prices are evaluated. Do not use pending rows as realized accuracy.",
    }
