from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from app.marketdata import calendar as cal
from app.marketdata.quality import assess_ohlcv, assess_quote, normalise_ohlcv
from tests.helpers import make_ohlcv, session_index


# ---------------------------------------------------------------- calendar
def test_known_holidays_and_weekends_are_closed() -> None:
    closed = [
        date(2024, 1, 1),    # New Year's Day
        date(2024, 1, 15),   # MLK Day
        date(2024, 3, 29),   # Good Friday
        date(2024, 6, 19),   # Juneteenth
        date(2024, 7, 4),
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),
        date(2023, 1, 2),    # New Year's Day 2023 fell on a Sunday, observed Monday
        date(2022, 6, 20),   # Juneteenth 2022 fell on a Sunday, observed Monday
        date(2025, 1, 9),    # unscheduled: day of mourning
        date(2024, 6, 15),   # Saturday
    ]
    for day in closed:
        assert not cal.is_trading_day(day), day


def test_new_years_day_on_saturday_is_not_observed_on_friday() -> None:
    assert date(2022, 1, 1).weekday() == 5
    assert cal.is_trading_day(date(2021, 12, 31))


def test_juneteenth_was_not_a_market_holiday_before_2022() -> None:
    assert cal.is_trading_day(date(2021, 6, 18))


def test_session_counts_match_known_years() -> None:
    # Published NYSE session counts.
    assert len(cal.trading_days(date(2023, 1, 1), date(2023, 12, 31))) == 250
    assert len(cal.trading_days(date(2024, 1, 1), date(2024, 12, 31))) == 252


def test_shift_trading_days_skips_weekends_and_holidays() -> None:
    # Five sessions after Wednesday 2024-11-27: Thanksgiving (28th) and the weekend are skipped.
    assert cal.shift_trading_days(date(2024, 11, 27), 5) == date(2024, 12, 5)
    assert cal.shift_trading_days(date(2024, 12, 5), -5) == date(2024, 11, 27)
    assert cal.shift_trading_days(date(2024, 6, 15), 0) == date(2024, 6, 17)


def test_session_bounds_follow_new_york_time_and_early_closes() -> None:
    opened, closed = cal.session_bounds(date(2024, 7, 10))  # summer: UTC-4
    assert (opened.hour, opened.minute, closed.hour) == (13, 30, 20)
    opened, closed = cal.session_bounds(date(2024, 1, 10))  # winter: UTC-5
    assert (opened.hour, opened.minute, closed.hour) == (14, 30, 21)
    _, early = cal.session_bounds(date(2024, 11, 29))       # day after Thanksgiving
    assert early.hour == 18
    assert cal.session_bounds(date(2024, 11, 28)) is None


def test_market_state_at_different_times() -> None:
    during = cal.market_state(datetime(2024, 7, 10, 15, 0, tzinfo=timezone.utc))
    assert during["is_open"] and during["last_completed_session"] == "2024-07-09"
    after = cal.market_state(datetime(2024, 7, 10, 21, 0, tzinfo=timezone.utc))
    assert after["state"] == "after_close" and after["last_completed_session"] == "2024-07-10"
    weekend = cal.market_state(datetime(2024, 7, 13, 15, 0, tzinfo=timezone.utc))
    assert weekend["state"] == "closed" and weekend["last_completed_session"] == "2024-07-12"
    before = cal.market_state(datetime(2024, 7, 10, 12, 0, tzinfo=timezone.utc))
    assert before["state"] == "pre_open" and before["last_completed_session"] == "2024-07-09"


# ---------------------------------------------------------------- quality
NOW = datetime(2024, 7, 10, 21, 0, tzinfo=timezone.utc)


def _history_ending(day: str, periods: int = 60) -> pd.DataFrame:
    frame = make_ohlcv(220, start="2024-01-02")  # runs past the dates used below
    return frame[frame.index <= pd.Timestamp(day)].tail(periods)


def test_clean_history_is_ok() -> None:
    frame, report = normalise_ohlcv(_history_ending("2024-07-10"))
    report = assess_ohlcv(frame, now=NOW, report=report)
    assert report.status == "ok", report.to_dict()


def test_empty_history_fails() -> None:
    frame, report = normalise_ohlcv(pd.DataFrame())
    assert report.status == "failed" and report.has("empty")


def test_duplicates_are_dropped_and_reported() -> None:
    base = _history_ending("2024-07-10")
    doubled = pd.concat([base, base.iloc[[5, 6]]])
    frame, report = normalise_ohlcv(doubled)
    assert len(frame) == len(base)
    assert report.has("duplicate_bars") and report.has("unsorted")
    assert frame.index.is_monotonic_increasing


def test_missing_sessions_are_reported_not_filled() -> None:
    base = _history_ending("2024-07-10")
    gappy = base.drop(base.index[[10, 11, 25]])
    frame, report = normalise_ohlcv(gappy)
    report = assess_ohlcv(frame, now=NOW, report=report)
    issue = next(i for i in report.issues if i.code == "missing_sessions")
    assert issue.count == 3 and report.status == "degraded"
    assert len(frame) == len(base) - 3  # nothing was invented to fill the gap


def test_missing_prices_stay_missing() -> None:
    base = _history_ending("2024-07-10")
    base.iloc[7, base.columns.get_loc("Close")] = np.nan
    frame, report = normalise_ohlcv(base)
    report = assess_ohlcv(frame, now=NOW, report=report)
    assert report.has("missing_values")
    assert np.isnan(frame["Close"].iloc[7])


def test_stale_history_is_flagged() -> None:
    frame, report = normalise_ohlcv(_history_ending("2024-07-03"))
    report = assess_ohlcv(frame, now=NOW, report=report)
    issue = next(i for i in report.issues if i.code == "stale")
    assert issue.count == 4  # sessions on the 5th, 8th, 9th and 10th are absent (the 4th was a holiday)


def test_inconsistent_bar_and_split_suspect_are_flagged() -> None:
    base = _history_ending("2024-07-10")
    base.iloc[3, base.columns.get_loc("High")] = base["Low"].iloc[3] * 0.9
    base.iloc[30:, :4] = base.iloc[30:, :4] / 4  # an unrecorded 4-for-1 split
    frame, report = normalise_ohlcv(base)
    report = assess_ohlcv(frame, now=NOW, report=report)
    assert report.has("inconsistent_bar") and report.has("possible_unadjusted_split")


def test_timezone_aware_daily_index_keeps_the_calendar_date() -> None:
    base = _history_ending("2024-07-10")
    base.index = base.index.tz_localize("America/New_York")
    frame, _ = normalise_ohlcv(base)
    assert frame.index.tz is None and frame.index[-1] == pd.Timestamp("2024-07-10")


def test_quote_quality() -> None:
    open_now = datetime(2024, 7, 10, 15, 0, tzinfo=timezone.utc)
    fresh = assess_quote(100.0, datetime(2024, 7, 10, 14, 59, 30, tzinfo=timezone.utc), now=open_now, max_age_seconds=1200)
    assert fresh.status == "ok"
    old = assess_quote(100.0, datetime(2024, 7, 10, 14, 0, tzinfo=timezone.utc), now=open_now, max_age_seconds=1200)
    assert old.has("stale")
    assert assess_quote(None, None, now=open_now, max_age_seconds=1200).status == "failed"
    assert assess_quote(0.0, open_now, now=open_now, max_age_seconds=1200).status == "failed"


def test_session_index_helper_uses_trading_days_only() -> None:
    index = session_index("2024-11-27", 4)
    assert [d.date() for d in index] == [date(2024, 11, 27), date(2024, 11, 29), date(2024, 12, 2), date(2024, 12, 3)]
