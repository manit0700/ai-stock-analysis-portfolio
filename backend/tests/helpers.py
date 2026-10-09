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


def make_predictable_ohlcv(periods: int, *, start: str = "2015-01-02", seed: int = 0, vol: float = 0.012, strength: float = 0.6) -> pd.DataFrame:
    """Prices with a slowly changing hidden drift, so that recent returns really do predict the next ones.

    Used to check that the evaluation pipeline can recognise an effect when one exists. It is not
    a model of any market.
    """
    rng = np.random.default_rng(seed)
    index = session_index(start, periods)
    drift = np.zeros(periods)
    for t in range(1, periods):
        drift[t] = 0.97 * drift[t - 1] + rng.normal(0, vol * strength * 0.25)
    returns = drift + rng.normal(0, vol, periods)
    close = 100.0 * np.cumprod(1 + returns)
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, periods)))
    volume = 2_000_000.0 * np.exp(rng.normal(0, 0.25, periods))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.round()}, index=index)


def make_snapshot(directory, *, symbols: int = 6, periods: int = 1700, start: str = "2015-01-02", seed: int = 0, predictable: bool = False, strength: float = 0.6, name: str = "synthetic") -> tuple[str, list[str]]:
    """Write a synthetic data snapshot (stocks, SPY and the VIX). Returns ``(snapshot_id, stock symbols)``."""
    from app.marketdata.snapshots import SnapshotStore

    names = [f"S{k}" for k in range(1, symbols + 1)]
    frames = {}
    for k, symbol in enumerate(names):
        frames[symbol] = (make_predictable_ohlcv(periods, start=start, seed=seed + k, strength=strength) if predictable else make_ohlcv(periods, start=start, seed=seed + k))
    frames["SPY"] = make_ohlcv(periods, start=start, seed=seed + 100, start_price=300, vol=0.009)
    rng = np.random.default_rng(seed + 200)
    vix = 18 + np.clip(np.cumsum(rng.normal(0, 0.35, periods)), -8, 40)
    level = pd.Series(vix, index=frames["SPY"].index)
    frames["^VIX"] = pd.DataFrame({"Open": level, "High": level * 1.02, "Low": level * 0.98, "Close": level, "Volume": 0.0})
    manifest = SnapshotStore(directory).create(name, frames, description="synthetic test data", metadata={"context_only": ["SPY", "^VIX"]})
    return manifest["id"], names
