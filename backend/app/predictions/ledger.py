"""The prediction ledger: what was forecast, and what then happened. Append-only, hash-chained.

Two kinds of event, in one JSONL file per horizon under ``<data dir>/predictions/``:

    prediction_issued     written the moment a forecast is served. Immutable. Its id is a hash of
                          (symbol, as-of date, horizon, model id), so issuing the same forecast
                          twice cannot create a second record.
    prediction_resolved   written once the forecast's window has closed and the prices are in.
                          It names the prediction it resolves and records the adjusted closes of
                          the exact window, the realised return and whether the call was right.

Every event carries the SHA-256 of the previous line, so an edited or deleted line breaks the
chain and ``verify`` reports where. Rows are facts about what the system said and what the market
did; nothing here updates in place.

A five-session forecast issued at the close of day ``t`` is resolved with the adjusted close of
session ``t+5`` against the adjusted close of day ``t`` — never with "the last few days" of
prices (the fault this module replaces).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.clock import iso, utcnow
from app.core.config import get_settings
from app.core.errors import Conflict, MarketVisionError, NotFound, ValidationFailed
from app.features.pipeline import adjusted_close
from app.marketdata import calendar as cal

GENESIS = "0" * 64
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def prediction_id(symbol: str, as_of: str, horizon: int, model_id: str | None) -> str:
    payload = f"{symbol.upper()}|{as_of}|{horizon}|{model_id or 'baseline'}"
    return "pred-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


class PredictionLedger:
    def __init__(self, directory: Path | None = None, *, horizon: int = 5) -> None:
        settings = get_settings()
        self.directory = Path(directory) if directory is not None else settings.data_dir / "predictions"
        self.horizon = int(horizon)
        self.path = self.directory / f"ledger-{self.horizon}d.jsonl"
        key = str(self.path)
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(key, threading.Lock())
        self._lockfile = self.directory / f"ledger-{self.horizon}d.lock"
        self._headfile = self.directory / f"ledger-{self.horizon}d.head.json"

    @contextmanager
    def _file_lock(self):
        """An OS-level lock beside the in-process one, so writers in different processes cannot
        interleave reads of ``prev_hash`` (the reviewer broke the chain with two honest writers)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lockfile.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------------ plumbing
    def _read_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        return [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self._read_lines()]

    def _append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append under the file lock and advance the head anchor (count + hash of the last line).

        The anchor is what lets ``verify`` notice a truncated tail, which an unanchored hash chain
        cannot see.
        """
        lines = self._read_lines()
        event = {**event, "prev_hash": _line_hash(lines[-1]) if lines else GENESIS}
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
        self._headfile.write_text(json.dumps({"events": len(lines) + 1, "last_hash": _line_hash(line)}), encoding="utf-8")
        return event

    def verify(self) -> dict[str, Any]:
        """Walk the chain. Returns how many events are intact and where the first break is."""
        lines = self._read_lines()
        previous = GENESIS
        for number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
            except ValueError:
                return {"intact": False, "events": number - 1, "broken_at_line": number, "problem": "not valid JSON"}
            if not isinstance(event, dict) or event.get("prev_hash") != previous:
                return {"intact": False, "events": number - 1, "broken_at_line": number, "problem": "hash chain broken: a line was edited, removed or reordered"}
            previous = _line_hash(line)
        if self._headfile.exists():
            try:
                head = json.loads(self._headfile.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                head = None
            if head and (head.get("events") != len(lines) or (lines and head.get("last_hash") != _line_hash(lines[-1]))):
                return {"intact": False, "events": len(lines), "problem": f"the head anchor records {head.get('events')} events; the file holds {len(lines)}. Lines were removed from the end or the anchor was altered."}
        return {"intact": True, "events": len(lines)}

    # ------------------------------------------------------------------ issuing
    def record_issued(self, forecast: dict[str, Any]) -> dict[str, Any]:
        """Write a ``prediction_issued`` event from a forecast response. A repeat is a no-op."""
        if forecast.get("probability_up") is None:
            raise ValidationFailed("A forecast with no probability is not recorded.")
        symbol = forecast["symbol"]
        as_of = forecast["as_of"]
        horizon = int(forecast["horizon_sessions"])
        if horizon != self.horizon:
            raise ValidationFailed(f"This ledger records {self.horizon}-session forecasts, not {horizon}.")
        model = forecast.get("model") or {}
        pid = prediction_id(symbol, as_of, horizon, model.get("id"))
        with self._lock, self._file_lock():
            existing = {e["prediction_id"]: e for e in self.events() if e["event"] == "prediction_issued"}
            if pid in existing:
                if abs(float(existing[pid]["probability_up"]) - float(forecast["probability_up"])) > 1e-9:
                    raise Conflict(
                        "This (symbol, date, horizon, model) was already recorded with a different probability. The inputs must have changed; the original record stands.",
                        details={"prediction_id": pid, "recorded": existing[pid]["probability_up"], "offered": forecast["probability_up"]},
                    )
                return {**existing[pid], "duplicate": True}
            event = self._append({
                "event": "prediction_issued",
                "prediction_id": pid,
                "at": iso(utcnow()),
                "symbol": symbol,
                "as_of": as_of,
                "horizon_sessions": horizon,
                "window_last_session": forecast["window"]["last_session"],
                "model_id": model.get("id"),
                "model_kind": model.get("kind"),
                "schema_version": (forecast.get("schema") or {}).get("version"),
                "schema_hash": (forecast.get("schema") or {}).get("hash"),
                "data_snapshot_id": forecast.get("data_snapshot_id"),
                "probability_up": float(forecast["probability_up"]),
                "base_rate": forecast.get("base_rate"),
                "expected_return": (forecast.get("expected_return") or {}).get("value"),
                "range_p10": (forecast.get("range") or {}).get("p10"),
                "range_p90": (forecast.get("range") or {}).get("p90"),
                "status": forecast.get("status"),
            })
        return event

    # ------------------------------------------------------------------ resolving
    def resolve_due(self, market_data: Any, *, today: date | None = None, limit: int = 500) -> dict[str, Any]:
        """Resolve every issued prediction whose window has closed and is not yet resolved."""
        today = today or cal.last_completed_session(utcnow())
        with self._lock, self._file_lock():
            events = self.events()
            issued = {e["prediction_id"]: e for e in events if e["event"] == "prediction_issued"}
            resolved_ids = {e["prediction_id"] for e in events if e["event"] == "prediction_resolved"}
            due = [e for pid, e in issued.items() if pid not in resolved_ids and date.fromisoformat(e["window_last_session"]) <= today]
            outcomes = {"resolved": 0, "still_open": sum(1 for pid, e in issued.items() if pid not in resolved_ids and date.fromisoformat(e["window_last_session"]) > today), "data_missing": 0, "errors": []}
            for event in due[:limit]:
                try:
                    self._resolve_one(event, market_data)
                    outcomes["resolved"] += 1
                except MarketVisionError as exc:
                    outcomes["data_missing"] += 1
                    if len(outcomes["errors"]) < 5:
                        outcomes["errors"].append({"prediction_id": event["prediction_id"], "reason": exc.message})
        return outcomes

    def _resolve_one(self, issued: dict[str, Any], market_data: Any) -> dict[str, Any]:
        symbol = issued["symbol"]
        start = date.fromisoformat(issued["as_of"])
        end = date.fromisoformat(issued["window_last_session"])
        bars = market_data.history(symbol, start=cal.shift_trading_days(start, -3), end=end, interval="1d").data
        close = adjusted_close(bars)
        close = close[close.notna()]
        if pd.Timestamp(start) not in close.index or pd.Timestamp(end) not in close.index:
            missing = [d.isoformat() for d in (start, end) if pd.Timestamp(d) not in close.index]
            raise NotFound(f"No adjusted close for {symbol} on {', '.join(missing)}; the prediction stays open.", details={"symbol": symbol, "missing": missing})
        entry, exit_ = float(close.loc[pd.Timestamp(start)]), float(close.loc[pd.Timestamp(end)])
        window = close.loc[pd.Timestamp(start):pd.Timestamp(end)]
        sessions_in_window = len(window) - 1
        realised = exit_ / entry - 1.0
        went_up = realised > 0
        probability = float(issued["probability_up"])
        return self._append({
            "event": "prediction_resolved",
            "prediction_id": issued["prediction_id"],
            "at": iso(utcnow()),
            "symbol": symbol,
            "as_of": issued["as_of"],
            "window_last_session": issued["window_last_session"],
            "sessions_in_window": sessions_in_window,
            "window_complete": sessions_in_window == int(issued["horizon_sessions"]),
            "entry_adjusted_close": round(entry, 6),
            "exit_adjusted_close": round(exit_, 6),
            "realized_return": round(realised, 6),
            "outcome_up": bool(went_up),
            "called_up": probability >= 0.5,
            "correct": bool((probability >= 0.5) == went_up),
            "brier_contribution": round((probability - float(went_up)) ** 2, 6),
            "within_stated_range": (None if issued.get("range_p10") is None or issued.get("range_p90") is None
                                    else bool(float(issued["range_p10"]) <= realised <= float(issued["range_p90"]))),
        })

    # ------------------------------------------------------------------ reading
    def history(self, *, symbol: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Issued predictions, newest first, each with its resolution when there is one."""
        events = self.events()
        resolved = {e["prediction_id"]: e for e in events if e["event"] == "prediction_resolved"}
        rows = []
        for event in events:
            if event["event"] != "prediction_issued":
                continue
            if symbol and event["symbol"] != symbol.upper():
                continue
            outcome = resolved.get(event["prediction_id"])
            rows.append({
                **{k: event.get(k) for k in ("prediction_id", "at", "symbol", "as_of", "horizon_sessions", "window_last_session", "model_id", "model_kind", "probability_up", "base_rate", "expected_return", "range_p10", "range_p90")},
                "resolved": outcome is not None,
                "outcome": None if outcome is None else {k: outcome.get(k) for k in ("realized_return", "outcome_up", "correct", "window_complete", "within_stated_range", "at")},
            })
        rows.reverse()
        return rows[:limit]

    def performance(self) -> dict[str, Any]:
        """Honest summary: resolved predictions only, wins are wins, Brier beside the base rate."""
        events = self.events()
        issued = {e["prediction_id"]: e for e in events if e["event"] == "prediction_issued"}
        resolutions = [e for e in events if e["event"] == "prediction_resolved"]
        complete = [e for e in resolutions if e.get("window_complete")]
        out: dict[str, Any] = {
            "ledger": str(self.path),
            "chain": self.verify(),
            "issued": len(issued),
            "resolved": len(resolutions),
            "open": len(issued) - len(resolutions),
            "resolved_with_complete_window": len(complete),
            "note": "Open predictions are not counted anywhere below. 'correct' means the side called (probability >= 0.5) matched the outcome.",
        }
        if not complete:
            return {**out, "accuracy": None, "brier": None}
        correct = sum(1 for e in complete if e["correct"])
        up = sum(1 for e in complete if e["outcome_up"])
        brier = sum(e["brier_contribution"] for e in complete) / len(complete)
        base = up / len(complete)
        brier_base = sum((base - float(e["outcome_up"])) ** 2 for e in complete) / len(complete)
        by_kind: dict[str, dict[str, Any]] = {}
        for e in complete:
            kind = issued.get(e["prediction_id"], {}).get("model_kind") or "unknown"
            bucket = by_kind.setdefault(kind, {"n": 0, "correct": 0, "brier_sum": 0.0})
            bucket["n"] += 1
            bucket["correct"] += int(e["correct"])
            bucket["brier_sum"] += e["brier_contribution"]
        stated = [e for e in complete if e.get("within_stated_range") is not None]
        return {
            **out,
            "accuracy": round(correct / len(complete), 4),
            "share_up": round(base, 4),
            "majority_class_accuracy": round(max(base, 1 - base), 4),
            "brier": round(brier, 6),
            "brier_of_base_rate": round(brier_base, 6),
            "brier_skill": round(1 - brier / brier_base, 6) if brier_base > 0 else None,
            "mean_realized_return": round(sum(e["realized_return"] for e in complete) / len(complete), 6),
            "range_coverage": round(sum(1 for e in stated if e["within_stated_range"]) / len(stated), 4) if stated else None,
            "by_model_kind": {kind: {"n": b["n"], "accuracy": round(b["correct"] / b["n"], 4), "brier": round(b["brier_sum"] / b["n"], 6)} for kind, b in sorted(by_kind.items())},
        }
