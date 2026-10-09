"""Checks on price data. They report problems; they never repair prices.

A missing bar stays missing and a bad bar stays flagged. Filling a gap with a neighbouring price
would be inventing data, so the only change made here is removing exact duplicate timestamps
(the later row is kept) and sorting by time, and both are reported.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.marketdata import calendar as cal
from app.marketdata.models import QualityReport

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS = [*PRICE_COLUMNS, "Volume"]
SPLIT_SUSPECT_RETURN = 0.45  # a one-day move this large with no split recorded is flagged


def normalise_ohlcv(frame: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Sort by time, drop exact duplicate timestamps, strip the time zone. Returns the report too."""
    report = QualityReport()
    if frame is None or len(frame) == 0:
        report.add("empty", "error", "The provider returned no rows.")
        return pd.DataFrame(columns=REQUIRED_COLUMNS), report

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add("missing_columns", "error", f"Columns absent from the response: {', '.join(missing)}.")
        return df, report

    index = pd.to_datetime(df.index)
    if getattr(index, "tz", None) is not None:
        # Daily bars carry the exchange's midnight; keep the calendar date. Intraday bars go to UTC.
        intraday = bool(((index.hour != 0) | (index.minute != 0)).any())
        index = index.tz_convert("UTC").tz_localize(None) if intraday else index.tz_localize(None)
    df.index = pd.DatetimeIndex(index).as_unit("ns")  # one resolution everywhere, whatever the source used
    df.index.name = "Date"

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        report.add("unsorted", "info", "Rows were not in time order and have been sorted.")
    duplicated = int(df.index.duplicated(keep="last").sum())
    if duplicated:
        df = df[~df.index.duplicated(keep="last")]
        report.add("duplicate_bars", "warning", "Rows with a repeated timestamp were dropped (the later row kept).", duplicated)
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df, report


def assess_ohlcv(
    frame: pd.DataFrame,
    *,
    interval: str = "1d",
    now: datetime | None = None,
    report: QualityReport | None = None,
    check_staleness: bool = True,
) -> QualityReport:
    """Append findings about ``frame`` (already normalised) to ``report``."""
    report = report or QualityReport()
    if report.status == "failed":
        return report
    if len(frame) == 0:
        report.add("empty", "error", "No rows.")
        return report

    prices = frame[PRICE_COLUMNS]
    missing_rows = int(prices.isna().any(axis=1).sum())
    if missing_rows == len(frame):
        report.add("all_missing", "error", "Every row has a missing price.", missing_rows)
        return report
    if missing_rows:
        report.add("missing_values", "warning", "Rows with a missing open, high, low or close. They are left as missing.", missing_rows)

    valid = prices.dropna()
    non_positive = int((valid <= 0).any(axis=1).sum())
    if non_positive:
        report.add("non_positive_price", "warning", "Rows with a zero or negative price.", non_positive)
    tolerance = 1e-6
    inconsistent = int(
        (
            (valid["High"] + tolerance < valid[["Open", "Close"]].max(axis=1))
            | (valid["Low"] - tolerance > valid[["Open", "Close"]].min(axis=1))
            | (valid["High"] + tolerance < valid["Low"])
        ).sum()
    )
    if inconsistent:
        report.add("inconsistent_bar", "warning", "Rows where the high or low does not bound the open and close.", inconsistent)

    volume = frame["Volume"]
    if int(volume.isna().sum()):
        report.add("missing_volume", "warning", "Rows with no volume.", int(volume.isna().sum()))
    zero_volume = int((volume.fillna(1) == 0).sum())
    if zero_volume:
        report.add("zero_volume", "info", "Rows with zero volume.", zero_volume)

    close = frame["Close"].dropna()
    if len(close) > 1:
        jumps = close.pct_change().abs()
        splits = frame["Stock Splits"] if "Stock Splits" in frame.columns else pd.Series(0.0, index=frame.index)
        suspect = int(((jumps > SPLIT_SUSPECT_RETURN) & (splits.reindex(jumps.index).fillna(0) == 0)).sum())
        if suspect:
            report.add(
                "possible_unadjusted_split",
                "warning",
                f"One-day close changes above {SPLIT_SUSPECT_RETURN:.0%} with no split recorded. Check whether prices are split-adjusted.",
                suspect,
            )

    if interval == "1d":
        first, last = frame.index[0].date(), frame.index[-1].date()
        expected = set(cal.trading_days(first, last))
        present = {ts.date() for ts in frame.index}
        gaps = sorted(expected - present)
        if gaps:
            shown = ", ".join(d.isoformat() for d in gaps[:5]) + (" ..." if len(gaps) > 5 else "")
            report.add("missing_sessions", "warning", f"Trading sessions with no bar: {shown}", len(gaps))
        extra = sorted(present - expected)
        if extra:
            report.add("non_session_bars", "info", "Bars dated on a weekend or market holiday.", len(extra))
        if check_staleness:
            now = now or datetime.now(timezone.utc)
            expected_last = cal.last_completed_session(now)
            if last < expected_last:
                behind = len(cal.trading_days(last, expected_last)) - 1
                report.add(
                    "stale",
                    "warning",
                    f"Latest bar is {last.isoformat()}; the last completed session was {expected_last.isoformat()}.",
                    behind,
                )
    return report


def assess_quote(price: float | None, market_timestamp: datetime | None, *, now: datetime, max_age_seconds: float) -> QualityReport:
    report = QualityReport()
    if price is None or not np.isfinite(price) or price <= 0:
        report.add("invalid_price", "error", "The quote has no usable price.")
        return report
    if market_timestamp is None:
        report.add("no_market_timestamp", "warning", "The provider gave no time for this price.")
        return report
    age = (now - market_timestamp).total_seconds()
    if age < -120:
        report.add("future_timestamp", "warning", "The quote's time is in the future.")
    elif cal.market_state(now)["is_open"] and age > max_age_seconds:
        report.add("stale", "warning", f"The quote is {age:.0f} seconds old while the market is open.", int(age))
    return report
