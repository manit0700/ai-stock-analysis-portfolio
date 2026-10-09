"""Synthetic market data for engineering tests.

These series exist to check that code is correct (accounting adds up, features do not look
ahead, windows are the right length). They say nothing about real markets and are never used as
evidence of prediction accuracy.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.marketdata import calendar as cal


def session_index(start: str | date, periods: int) -> pd.DatetimeIndex:
    """``periods`` consecutive NYSE sessions starting at or after ``start``."""
    days = []
    day = cal.shift_trading_days(start if isinstance(start, date) else date.fromisoformat(start), 0)
    while len(days) < periods:
        days.append(day)
        day = cal.next_trading_day(day)
    return pd.DatetimeIndex(pd.to_datetime(days), name="Date").as_unit("ns")


def make_ohlcv(
    periods: int = 400,
    *,
    start: str = "2022-01-03",
    seed: int = 0,
    start_price: float = 100.0,
    drift: float = 0.0004,
    vol: float = 0.015,
    volume: float = 2_000_000.0,
) -> pd.DataFrame:
    """A random-walk price history with consistent open/high/low/close and plausible volume."""
    rng = np.random.default_rng(seed)
    index = session_index(start, periods)
    returns = rng.normal(drift, vol, periods)
    close = start_price * np.cumprod(1 + returns)
    open_ = np.r_[start_price, close[:-1]] * (1 + rng.normal(0, vol / 4, periods))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, periods)))
    vols = volume * np.exp(rng.normal(0, 0.25, periods))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vols.round()}, index=index)


def flat_ohlcv(prices: list[float], *, start: str = "2024-01-02", volume: float = 5_000_000.0) -> pd.DataFrame:
    """Bars whose open, high, low and close all equal the given price. For exact accounting tests."""
    index = session_index(start, len(prices))
    series = pd.Series(prices, index=index, dtype=float)
    return pd.DataFrame({"Open": series, "High": series, "Low": series, "Close": series, "Volume": volume}, index=index)


def bars(rows: list[tuple[float, float, float, float]], *, start: str = "2024-01-02", volume: float = 5_000_000.0) -> pd.DataFrame:
    """Bars from explicit (open, high, low, close) tuples."""
    index = session_index(start, len(rows))
    frame = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index, dtype=float)
    frame["Volume"] = volume
    return frame
