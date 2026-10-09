"""The prediction ledger: right window, no duplicates, append-only, honest summaries."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from app.core import clock
from app.core.errors import ValidationFailed
from app.marketdata import calendar as cal
from app.predictions.ledger import GENESIS, PredictionLedger, prediction_id
from tests.helpers import make_ohlcv


class OneSymbolMarket:
    def __init__(self, bars: pd.DataFrame):
        self.bars = bars
        self.requests: list[tuple] = []

    def history(self, symbol, *, start=None, end=None, period=None, interval="1d"):
        self.requests.append((symbol, start, end))
        frame = self.bars
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(end)]

        class Envelope:
            data = frame
        return Envelope()


def _forecast(bars: pd.DataFrame, at: int, probability: float = 0.62, model_id: str | None = "mv5d-test", horizon: int = 5) -> dict:
    as_of = bars.index[at].date()
    return {
        "symbol": "TEST", "as_of": as_of.isoformat(), "horizon_sessions": horizon,
        "window": {"first_session": cal.next_trading_day(as_of).isoformat(), "last_session": cal.shift_trading_days(as_of, horizon).isoformat()},
        "model": {"id": model_id, "kind": "model" if model_id else "baseline"},
        "schema": {"version": "1.0.0", "hash": "h"}, "data_snapshot_id": "inputs-x",
        "probability_up": probability, "base_rate": 0.55,
        "expected_return": {"value": 0.004}, "range": {"p10": -0.05, "p90": 0.06}, "status": "model",
    }


@pytest.fixture()
def setup(tmp_path):
    bars = make_ohlcv(60, seed=3, start="2024-01-02")
    return PredictionLedger(tmp_path, horizon=5), OneSymbolMarket(bars), bars


def test_issue_is_idempotent_and_the_id_is_deterministic(setup):
    ledger, market, bars = setup
    first = ledger.record_issued(_forecast(bars, 10))
    again = ledger.record_issued(_forecast(bars, 10))
    assert again.get("duplicate") and first["prediction_id"] == again["prediction_id"]
    assert len([e for e in ledger.events() if e["event"] == "prediction_issued"]) == 1
    assert first["prediction_id"] == prediction_id("TEST", bars.index[10].date().isoformat(), 5, "mv5d-test")
    different_model = ledger.record_issued(_forecast(bars, 10, model_id=None))
    assert different_model["prediction_id"] != first["prediction_id"]


def test_a_forecast_with_no_probability_or_wrong_horizon_is_refused(setup):
    ledger, _, bars = setup
    with pytest.raises(ValidationFailed):
        ledger.record_issued({**_forecast(bars, 10), "probability_up": None})
    with pytest.raises(ValidationFailed):
        ledger.record_issued(_forecast(bars, 10, horizon=10))


def test_resolution_uses_exactly_the_forecast_window(setup):
    ledger, market, bars = setup
    at = 20
    ledger.record_issued(_forecast(bars, at, probability=0.62))
    as_of, end = bars.index[at].date(), date.fromisoformat(_forecast(bars, at)["window"]["last_session"])
    # One session before the window closes: still open, and never judged on a shorter window.
    early = ledger.resolve_due(market, today=cal.previous_trading_day(end))
    assert early == {**early, "resolved": 0, "still_open": 1}
    outcome = None
    result = ledger.resolve_due(market, today=end)
    assert result["resolved"] == 1 and result["still_open"] == 0
    outcome = [e for e in ledger.events() if e["event"] == "prediction_resolved"][0]
    entry, exit_ = float(bars["Close"].loc[pd.Timestamp(as_of)]), float(bars["Close"].loc[pd.Timestamp(end)])
    assert outcome["entry_adjusted_close"] == pytest.approx(entry, rel=1e-6)
    assert outcome["exit_adjusted_close"] == pytest.approx(exit_, rel=1e-6)
    assert outcome["realized_return"] == pytest.approx(exit_ / entry - 1, abs=1e-6)
    assert outcome["sessions_in_window"] == 5 and outcome["window_complete"]
    assert outcome["correct"] == ((exit_ > entry) == (0.62 >= 0.5))
    assert outcome["within_stated_range"] == (-0.05 <= exit_ / entry - 1 <= 0.06)
    # a second resolve pass does nothing
    assert ledger.resolve_due(market, today=end)["resolved"] == 0


def test_an_old_prediction_is_resolved_on_its_own_window_not_recent_prices(setup):
    """The fault this ledger replaces: a 60-day-old signal was judged on the last 5 bars."""
    ledger, market, bars = setup
    ledger.record_issued(_forecast(bars, 5))
    ledger.resolve_due(market, today=bars.index[-1].date())
    outcome = [e for e in ledger.events() if e["event"] == "prediction_resolved"][0]
    assert outcome["window_last_session"] == cal.shift_trading_days(bars.index[5].date(), 5).isoformat()
    assert outcome["exit_adjusted_close"] == pytest.approx(float(bars["Close"].loc[pd.Timestamp(outcome["window_last_session"])]), rel=1e-6)


def test_missing_prices_leave_the_prediction_open_with_a_reason(setup):
    ledger, market, bars = setup
    ledger.record_issued(_forecast(bars, 30))
    gappy = OneSymbolMarket(bars.drop(bars.index[35]))  # the window's last session is missing
    result = gappy_resolve = gappy and ledger.resolve_due(gappy, today=bars.index[-1].date())
    assert result["resolved"] == 0 and result["data_missing"] == 1 and result["errors"][0]["reason"].startswith("No adjusted close")
    assert ledger.resolve_due(market, today=bars.index[-1].date())["resolved"] == 1  # resolvable once the data is there


def test_the_chain_detects_tampering(setup, tmp_path):
    ledger, market, bars = setup
    for at in (10, 12, 14):
        ledger.record_issued(_forecast(bars, at))
    assert ledger.verify() == {"intact": True, "events": 3}
    lines = ledger.path.read_text().splitlines()
    edited = json.loads(lines[1]); edited["probability_up"] = 0.99
    lines[1] = json.dumps(edited, sort_keys=True, separators=(",", ":"))
    ledger.path.write_text("\n".join(lines) + "\n")
    report = PredictionLedger(tmp_path, horizon=5).verify()
    assert not report["intact"] and report["broken_at_line"] == 3  # line 3's prev_hash no longer matches the edited line 2
    # deleting lines from the end leaves a valid chain, but the head anchor records the count
    ledger.path.write_text("\n".join(lines[:1]) + "\n")
    fresh = PredictionLedger(tmp_path, horizon=5)
    truncated = fresh.verify()
    assert not truncated["intact"] and "removed from the end" in truncated["problem"]
    assert fresh.events()[0]["prev_hash"] == GENESIS


def test_history_and_performance_count_only_what_happened(setup):
    ledger, market, bars = setup
    ledger.record_issued(_forecast(bars, 10, probability=0.70))   # will resolve
    ledger.record_issued(_forecast(bars, 12, probability=0.40))   # will resolve
    ledger.record_issued(_forecast(bars, 52, probability=0.80))   # window still open
    ledger.resolve_due(market, today=bars.index[20].date())
    history = ledger.history()
    assert len(history) == 3 and history[0]["as_of"] > history[-1]["as_of"]
    assert sum(1 for row in history if row["resolved"]) == 2
    perf = ledger.performance()
    assert perf["issued"] == 3 and perf["resolved"] == 2 and perf["open"] == 1
    assert perf["resolved_with_complete_window"] == 2
    resolved = [e for e in ledger.events() if e["event"] == "prediction_resolved"]
    expected_accuracy = sum(e["correct"] for e in resolved) / 2
    assert perf["accuracy"] == pytest.approx(expected_accuracy)
    assert perf["brier"] == pytest.approx(sum(e["brier_contribution"] for e in resolved) / 2, abs=1e-6)
    assert "model" in perf["by_model_kind"] and perf["by_model_kind"]["model"]["n"] == 2
    assert perf["chain"]["intact"]


def test_two_ledgers_for_different_horizons_do_not_mix(tmp_path):
    bars = make_ohlcv(40, seed=4)
    five = PredictionLedger(tmp_path, horizon=5)
    ten = PredictionLedger(tmp_path, horizon=10)
    five.record_issued(_forecast(bars, 10))
    assert ten.events() == [] and five.path != ten.path
