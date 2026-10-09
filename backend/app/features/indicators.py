"""Technical indicators, written once.

The repository used to carry ten separate RSI functions (five in the backend, five in training
scripts) with small differences between them. Everything now imports from here, so a number
shown in the interface and a number fed to a model come from the same code.

Every function looks backwards only: the value at row ``t`` depends on rows up to and including
``t`` and never on a later row. ``tests/test_features.py`` checks this for all of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pct_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return close / close.shift(periods) - 1.0


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    """Standard deviation of returns over ``window`` rows (not annualised)."""
    return returns.rolling(window, min_periods=window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with simple moving averages of gains and losses (0 to 100).

    A window with no losses gives 100; a window with no movement at all gives 50.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.where(loss > 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(loss > 0, 100.0)
    out = out.where(~((gain == 0) & (loss == 0)), 50.0)
    return out.where(gain.notna() & loss.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    return pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1, skipna=False)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(window, min_periods=window).mean()


def bollinger(close: pd.Series, window: int = 20, width: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Middle, upper and lower Bollinger bands."""
    middle = sma(close, window)
    spread = close.rolling(window, min_periods=window).std() * width
    return middle, middle + spread, middle - spread


def rolling_max_gap(close: pd.Series, window: int) -> pd.Series:
    """Close relative to its highest close in the window: 0 at a new high, negative below it."""
    return close / close.rolling(window, min_periods=window).max() - 1.0


def range_position(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Where the close sits between the window's lowest low (0) and highest high (1)."""
    top = high.rolling(window, min_periods=window).max()
    bottom = low.rolling(window, min_periods=window).min()
    width = (top - bottom).where((top - bottom) > 0)
    return ((close - bottom) / width).clip(0.0, 1.0)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """How far the value is from the mean of the PREVIOUS ``window`` rows, in standard deviations."""
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=window).mean()
    std = prior.rolling(window, min_periods=window).std()
    return (series - mean) / std.where(std > 0)


def rolling_beta(asset_returns: pd.Series, market_returns: pd.Series, window: int) -> pd.Series:
    cov = asset_returns.rolling(window, min_periods=window).cov(market_returns)
    var = market_returns.rolling(window, min_periods=window).var()
    return cov / var.where(var > 0)


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=window).corr(b).clip(-1.0, 1.0)


def up_volume_share(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Share of the window's volume traded on days the close rose (0 to 1)."""
    up = volume.where(close.diff() > 0, 0.0)
    total = volume.rolling(window, min_periods=window).sum()
    return up.rolling(window, min_periods=window).sum() / total.where(total > 0)


def annualised_volatility(close: pd.Series, lookback: int = 30) -> float | None:
    returns = close.pct_change().dropna().tail(lookback)
    if len(returns) < 2:
        return None
    return float(returns.std() * np.sqrt(252))


def max_drawdown(close: pd.Series) -> float | None:
    close = close.dropna()
    if close.empty:
        return None
    return float((close / close.cummax() - 1.0).min())


def latest(series: pd.Series) -> float | None:
    """Last value as a float, or ``None`` when it is missing."""
    if series is None or len(series) == 0:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)
