"""Strategies as functions from bars to a daily "want to be long" series (1 or 0).

The value on day ``t`` is decided with data up to and including day ``t``'s close; the engine
acts on it at the next open. Indicators come from ``app.features.indicators``, the same code the
models and the interface use.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from app.features import indicators as ind

Strategy = Callable[[pd.DataFrame], pd.Series]


def _stateful(enter: pd.Series, leave: pd.Series) -> pd.Series:
    """Long from the day ``enter`` is true until the day ``leave`` is true."""
    out = np.zeros(len(enter))
    held = 0.0
    for i, (go_in, go_out) in enumerate(zip(enter.to_numpy(bool), leave.to_numpy(bool))):
        if go_in:
            held = 1.0
        elif go_out:
            held = 0.0
        out[i] = held
    return pd.Series(out, index=enter.index)


def sma_crossover(bars: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    close = bars["Close"]
    return (ind.sma(close, fast) > ind.sma(close, slow)).astype(float)


def rsi_mean_reversion(bars: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> pd.Series:
    rsi = ind.rsi(bars["Close"], 14)
    return _stateful((rsi < oversold).fillna(False), (rsi > overbought).fillna(False))


def macd_signal(bars: pd.DataFrame) -> pd.Series:
    line, signal, _ = ind.macd(bars["Close"])
    return (line > signal).astype(float)


def bollinger_reversion(bars: pd.DataFrame, window: int = 20, width: float = 2.0) -> pd.Series:
    close = bars["Close"]
    _, upper, lower = ind.bollinger(close, window, width)
    return _stateful((close <= lower).fillna(False), (close >= upper).fillna(False))


def always_long(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=bars.index)


def probability_threshold(probability: pd.Series, enter: float, leave: float | None = None) -> pd.Series:
    """Long when a forecast probability is at or above ``enter``; flat again at or below ``leave``.

    ``probability`` must be out-of-sample: produced for each date by a model that had not seen
    that date. Days with no forecast are treated as "no opinion" and keep the previous state.
    """
    leave = enter if leave is None else leave
    return _stateful((probability >= enter).fillna(False), (probability <= leave).fillna(False) & probability.notna())


STRATEGIES: dict[str, Strategy] = {
    "sma_crossover": sma_crossover,
    "rsi_mean_reversion": rsi_mean_reversion,
    "macd": macd_signal,
    "bollinger_bands": bollinger_reversion,
    "buy_and_hold": always_long,
}

LABELS = {
    "sma_crossover": "SMA Crossover (20/50)",
    "rsi_mean_reversion": "RSI Mean Reversion (30/70)",
    "macd": "MACD Signal Line",
    "bollinger_bands": "Bollinger Bands (20, 2σ)",
    "buy_and_hold": "Buy & Hold (Baseline)",
}

TYPES = {
    "sma_crossover": "trend_following",
    "rsi_mean_reversion": "mean_reversion",
    "macd": "momentum",
    "bollinger_bands": "mean_reversion",
    "buy_and_hold": "baseline",
}
