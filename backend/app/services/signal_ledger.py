from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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


RESOLVED_WIN_OUTCOMES = {"resolved_win", "win", "expired_win"}
RESOLVED_LOSS_OUTCOMES = {"resolved_loss", "loss", "expired_loss"}
PARTIAL_WIN_OUTCOMES = {"partial_win", "ambiguous"}
PENDING_OUTCOMES = {"pending", "still_pending"}
INVALID_OUTCOMES = {"invalid"}
RESOLVED_OUTCOMES = RESOLVED_WIN_OUTCOMES | RESOLVED_LOSS_OUTCOMES | PARTIAL_WIN_OUTCOMES | INVALID_OUTCOMES


def _ledger_path(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else _LEDGER_PATH


def read_signals(limit: int = 5000, ledger_path: Path | str | None = None) -> list[dict[str, Any]]:
    path = _ledger_path(ledger_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def write_signals(rows: list[dict[str, Any]], ledger_path: Path | str | None = None) -> None:
    path = _ledger_path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def _interval_to_timedelta(interval: str | None) -> timedelta:
    normalized = str(interval or "1d").lower()
    units = {
        "m": 60,
        "min": 60,
        "h": 3600,
        "d": 86400,
        "wk": 604800,
        "mo": 2592000,
    }
    for suffix, seconds in units.items():
        if normalized.endswith(suffix):
            raw = normalized[: -len(suffix)] or "1"
            try:
                return timedelta(seconds=max(float(raw), 1) * seconds)
            except ValueError:
                return timedelta(days=1)
    return timedelta(days=1)


def _resolution_due_at(row: dict[str, Any], recorded_at: datetime) -> datetime:
    explicit = row.get("resolution_due_at")
    if explicit:
        try:
            parsed = datetime.fromisoformat(str(explicit).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    window_hours = row.get("prediction_window_hours")
    if window_hours is not None:
        try:
            return recorded_at + timedelta(hours=max(float(window_hours), 0.0))
        except (TypeError, ValueError):
            pass

    horizon_steps = row.get("horizon_steps")
    interval = row.get("interval")
    if horizon_steps is not None and interval:
        try:
            return recorded_at + _interval_to_timedelta(str(interval)) * max(int(horizon_steps), 1)
        except (TypeError, ValueError):
            pass

    return recorded_at + timedelta(hours=72)


def _is_pending(outcome: Any) -> bool:
    return str(outcome or "pending") in PENDING_OUTCOMES


def _is_win(outcome: Any) -> bool:
    return str(outcome) in RESOLVED_WIN_OUTCOMES


def _is_loss(outcome: Any) -> bool:
    return str(outcome) in RESOLVED_LOSS_OUTCOMES


def _is_partial(outcome: Any) -> bool:
    return str(outcome) in PARTIAL_WIN_OUTCOMES


def _is_resolved(outcome: Any) -> bool:
    return str(outcome) in RESOLVED_OUTCOMES


def _resolve_row_outcome(row: dict[str, Any], future: pd.DataFrame, now: datetime) -> dict[str, Any]:
    entry = float(row["entry_price"])
    target_1 = float(row["target_1"])
    target_2 = row.get("target_2")
    stop = float(row["stop_loss"])
    max_high = float(future["High"].max())
    min_low = float(future["Low"].min())
    latest_close = float(future["Close"].iloc[-1])
    direction = "short" if str(row.get("dominant_scenario")) == "bearish" else "long"

    if direction == "short":
        hit_target_1 = min_low <= target_1
        hit_target_2 = target_2 is not None and min_low <= float(target_2)
        hit_stop = max_high >= stop
        pnl_pct = (entry / latest_close - 1) if latest_close else 0.0
    else:
        hit_target_1 = max_high >= target_1
        hit_target_2 = target_2 is not None and max_high >= float(target_2)
        hit_stop = min_low <= stop
        pnl_pct = (latest_close / entry - 1) if entry else 0.0

    if hit_target_2 or (hit_target_1 and not hit_stop):
        outcome = "resolved_win"
    elif hit_stop and not hit_target_1:
        outcome = "resolved_loss"
    elif hit_target_1 and hit_stop:
        outcome = "partial_win"
    elif pnl_pct > 0:
        outcome = "partial_win"
    else:
        outcome = "resolved_loss"

    return {
        **row,
        "outcome": outcome,
        "resolved_at": now.isoformat(),
        "latest_close_at_resolution": round(latest_close, 4),
        "resolution_pnl_pct": round(pnl_pct * 100, 4),
        "hit_target_1": bool(hit_target_1),
        "hit_target_2": bool(hit_target_2),
        "hit_stop_loss": bool(hit_stop),
    }


def resolve_pending_signals(
    market_service: Any,
    max_rows: int | None = 1000,
    ledger_path: Path | str | None = None,
) -> dict[str, Any]:
    path = _ledger_path(ledger_path)
    rows = read_signals(limit=1_000_000, ledger_path=path)
    changed = 0
    checked = 0
    skipped = 0
    invalid = 0
    still_pending = 0
    now = datetime.now(timezone.utc)
    pending_indexes = [index for index, row in enumerate(rows) if _is_pending(row.get("outcome"))]
    if max_rows is not None:
        pending_indexes = pending_indexes[-max_rows:]

    for index in pending_indexes:
        row = rows[index]
        recorded_at_raw = row.get("recorded_at")
        if not recorded_at_raw:
            rows[index] = {**row, "outcome": "invalid", "resolution_error": "missing_recorded_at", "resolved_at": now.isoformat()}
            invalid += 1
            changed += 1
            continue
        try:
            recorded_at = datetime.fromisoformat(recorded_at_raw.replace("Z", "+00:00"))
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=timezone.utc)
        except ValueError:
            rows[index] = {**row, "outcome": "invalid", "resolution_error": "invalid_recorded_at", "resolved_at": now.isoformat()}
            invalid += 1
            changed += 1
            continue
        due_at = _resolution_due_at(row, recorded_at)
        if now < due_at:
            if row.get("outcome") != "still_pending":
                rows[index] = {**row, "outcome": "still_pending", "resolution_due_at": due_at.isoformat()}
                changed += 1
            still_pending += 1
            skipped += 1
            continue

        ticker = row.get("ticker")
        entry = row.get("entry_price")
        target = row.get("target_1")
        stop = row.get("stop_loss")
        if not ticker or entry is None or target is None or stop is None:
            rows[index] = {**row, "outcome": "invalid", "resolution_error": "missing_trade_plan", "resolved_at": now.isoformat()}
            invalid += 1
            changed += 1
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
        history_index = pd.to_datetime(history.index)
        if history_index.tz is not None:
            history_index = history_index.tz_convert(timezone.utc).tz_localize(None)
        future = history.loc[history_index >= recorded_at.replace(tzinfo=None)]
        if future.empty:
            rows[index] = {**row, "outcome": "still_pending", "resolution_due_at": due_at.isoformat()}
            still_pending += 1
            skipped += 1
            continue

        rows[index] = _resolve_row_outcome(row, future, now)
        changed += 1

    if changed:
        write_signals(rows, ledger_path=path)
    return {
        "checked_pending": checked,
        "updated": changed,
        "resolved": changed - still_pending,
        "skipped": skipped,
        "invalid": invalid,
        "still_pending": still_pending,
        "ledger_path": str(path),
    }


def _new_performance_bucket() -> dict[str, int | float | None]:
    return {
        "count": 0,
        "pending": 0,
        "resolved": 0,
        "wins": 0,
        "losses": 0,
        "partial_wins": 0,
        "invalid": 0,
        "accuracy": None,
        "win_rate": None,
        "loss_rate": None,
        "partial_win_rate": None,
        "average_risk_score": None,
        "_risk_total": 0.0,
        "_risk_count": 0,
    }


def _update_performance_bucket(bucket: dict[str, int | float | None], row: dict[str, Any]) -> None:
    outcome = row.get("outcome", "pending")
    bucket["count"] = int(bucket["count"] or 0) + 1
    risk = row.get("risk_score")
    if risk is not None:
        try:
            bucket["_risk_total"] = float(bucket["_risk_total"] or 0.0) + float(risk)
            bucket["_risk_count"] = int(bucket["_risk_count"] or 0) + 1
        except (TypeError, ValueError):
            pass

    if _is_pending(outcome):
        bucket["pending"] = int(bucket["pending"] or 0) + 1
        return
    if _is_resolved(outcome):
        bucket["resolved"] = int(bucket["resolved"] or 0) + 1
    if _is_win(outcome):
        bucket["wins"] = int(bucket["wins"] or 0) + 1
    elif _is_loss(outcome):
        bucket["losses"] = int(bucket["losses"] or 0) + 1
    elif _is_partial(outcome):
        bucket["partial_wins"] = int(bucket["partial_wins"] or 0) + 1
    elif str(outcome) in INVALID_OUTCOMES:
        bucket["invalid"] = int(bucket["invalid"] or 0) + 1


def _finalize_performance_bucket(bucket: dict[str, int | float | None]) -> None:
    resolved_count = int(bucket["resolved"] or 0)
    wins = int(bucket["wins"] or 0)
    partial_wins = int(bucket["partial_wins"] or 0)
    bucket["accuracy"] = round((wins + partial_wins) / resolved_count, 4) if resolved_count else None
    bucket["win_rate"] = round(int(bucket["wins"] or 0) / resolved_count, 4) if resolved_count else None
    bucket["loss_rate"] = round(int(bucket["losses"] or 0) / resolved_count, 4) if resolved_count else None
    bucket["partial_win_rate"] = round(int(bucket["partial_wins"] or 0) / resolved_count, 4) if resolved_count else None
    risk_count = int(bucket.pop("_risk_count", 0) or 0)
    risk_total = float(bucket.pop("_risk_total", 0.0) or 0.0)
    bucket["average_risk_score"] = round(risk_total / risk_count, 2) if risk_count else None


def _confidence_bucket_label(confidence: float) -> str:
    if confidence >= 90:
        return "90%+"
    if confidence >= 80:
        return "80-90%"
    if confidence >= 70:
        return "70-80%"
    if confidence >= 60:
        return "60-70%"
    if confidence >= 50:
        return "50-60%"
    return "<50%"


def _drift_monitoring(
    rows: list[dict[str, Any]],
    confidence_bucket_performance: dict[str, dict[str, int | float | None]],
    quality_counts: dict[str, int],
) -> dict[str, Any]:
    resolved_rows = [row for row in rows if _is_resolved(row.get("outcome")) and str(row.get("outcome")) not in INVALID_OUTCOMES]
    recent_resolved = resolved_rows[-50:]
    recent_accuracy = None
    if recent_resolved:
        recent_success = sum(1 for row in recent_resolved if _is_win(row.get("outcome")) or _is_partial(row.get("outcome")))
        recent_accuracy = round(recent_success / len(recent_resolved), 4)

    warnings: list[str] = []
    high_bucket = confidence_bucket_performance.get("80-90%", {})
    top_bucket = confidence_bucket_performance.get("90%+", {})
    high_resolved = int(high_bucket.get("resolved") or 0) + int(top_bucket.get("resolved") or 0)
    high_success = int(high_bucket.get("wins") or 0) + int(high_bucket.get("partial_wins") or 0) + int(top_bucket.get("wins") or 0) + int(top_bucket.get("partial_wins") or 0)
    high_conf_accuracy = round(high_success / high_resolved, 4) if high_resolved else None

    if recent_accuracy is not None and len(recent_resolved) >= 20 and recent_accuracy < 0.45:
        warnings.append("model_performance_degraded")
    if high_conf_accuracy is not None and high_resolved >= 10 and high_conf_accuracy < 0.65:
        warnings.append("confidence_miscalibrated")

    total = len(rows) or 1
    quality_distribution = {key: round(value / total, 4) for key, value in quality_counts.items()}
    avoid_share = quality_distribution.get("avoid_high_risk", 0)
    promoted_share = quality_distribution.get("high_confidence_trade_candidate", 0)
    if avoid_share >= 0.6:
        warnings.append("regime_failure_detected")
    if warnings:
        warnings.append("retraining_recommended")

    confidence_distribution = {
        key: round((bucket.get("count") or 0) / total, 4)
        for key, bucket in confidence_bucket_performance.items()
    }
    return {
        "live_signal_accuracy": recent_accuracy,
        "recent_resolved_count": len(recent_resolved),
        "high_confidence_accuracy": high_conf_accuracy,
        "high_confidence_resolved_count": high_resolved,
        "quality_distribution": quality_distribution,
        "confidence_distribution": confidence_distribution,
        "feature_drift": {
            "status": "not_enough_feature_history",
            "tracked_features": ["confidence", "risk_score", "quality_label", "outcome"],
        },
        "prediction_distribution": quality_distribution,
        "regime_specific_performance": {
            "status": "pending_macro_regime_ledger_history",
        },
        "warnings": sorted(set(warnings)),
    }


def summarize_signals(ledger_path: Path | str | None = None) -> dict[str, Any]:
    path = _ledger_path(ledger_path)
    rows = read_signals(ledger_path=path)
    quality_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    quality_performance: dict[str, dict[str, int | float | None]] = {}
    action_performance: dict[str, dict[str, int | float | None]] = {}
    confidence_bucket_performance: dict[str, dict[str, int | float | None]] = {
        label: _new_performance_bucket() for label in ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90%+"]
    }
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
    losses = 0
    partial_wins = 0
    invalid = 0

    for row in rows:
        quality = row.get("quality_label") or "unknown"
        action = row.get("action") or "unknown"
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        quality_performance.setdefault(quality, _new_performance_bucket())
        action_performance.setdefault(action, _new_performance_bucket())
        _update_performance_bucket(quality_performance[quality], row)
        _update_performance_bucket(action_performance[action], row)
        if quality == "high_confidence_trade_candidate":
            promoted += 1

        outcome = row.get("outcome", "pending")
        if _is_pending(outcome):
            pending += 1
        else:
            resolved += 1
            if _is_win(outcome):
                wins += 1
            elif _is_loss(outcome):
                losses += 1
            elif _is_partial(outcome):
                partial_wins += 1
            elif str(outcome) in INVALID_OUTCOMES:
                invalid += 1

        confidence = float(row.get("confidence") or 0)
        _update_performance_bucket(confidence_bucket_performance[_confidence_bucket_label(confidence)], row)
        for threshold, bucket in [(60, ">=60"), (70, ">=70"), (80, ">=80"), (85, ">=85")]:
            if confidence >= threshold:
                confidence_buckets[bucket]["count"] += 1
                if not _is_pending(outcome):
                    confidence_buckets[bucket]["resolved"] += 1
                    if _is_win(outcome):
                        confidence_buckets[bucket]["wins"] += 1

    for bucket in confidence_buckets.values():
        bucket["win_rate"] = round(bucket["wins"] / bucket["resolved"], 4) if bucket["resolved"] else None
    for group in (quality_performance, action_performance, confidence_bucket_performance):
        for stats in group.values():
            _finalize_performance_bucket(stats)

    promoted_signal_performance = quality_performance.get("high_confidence_trade_candidate", _new_performance_bucket())
    if "_risk_count" in promoted_signal_performance:
        _finalize_performance_bucket(promoted_signal_performance)

    return {
        "ledger_path": str(path),
        "total_predictions": len(rows),
        "total_logged_predictions": len(rows),
        "pending_signals": pending,
        "promoted_signals": promoted,
        "pending_outcomes": pending,
        "resolved_signals": resolved,
        "resolved_outcomes": resolved,
        "win_rate": round(wins / resolved, 4) if resolved else None,
        "loss_rate": round(losses / resolved, 4) if resolved else None,
        "partial_win_rate": round(partial_wins / resolved, 4) if resolved else None,
        "resolved_win_rate": round(wins / resolved, 4) if resolved else None,
        "outcome_counts": {
            "pending": pending,
            "resolved_win": wins,
            "resolved_loss": losses,
            "partial_win": partial_wins,
            "invalid": invalid,
        },
        "quality_counts": quality_counts,
        "action_counts": action_counts,
        "promoted_signal_performance": promoted_signal_performance,
        "quality_performance": quality_performance,
        "action_performance": action_performance,
        "confidence_buckets": confidence_buckets,
        "confidence_bucket_performance": confidence_bucket_performance,
        "drift_monitoring": _drift_monitoring(rows, confidence_bucket_performance, quality_counts),
        "note": "Live outcomes are pending until future prices are evaluated. Do not use pending rows as realized accuracy.",
    }
