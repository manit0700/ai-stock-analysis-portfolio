from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_WINDOW = 20          # candles to describe a "setup"
_HISTORY_PERIOD = "5y"
_FORWARD_HORIZON = 10 # candles ahead to measure outcome
_TOP_N = 5            # number of similar setups to return


def _features(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compact 8-feature fingerprint of a 20-bar window."""
    ret = close.pct_change().dropna()
    vol_ret = ret.std() * np.sqrt(252) if len(ret) > 1 else 0.0
    trend = (close.iloc[-1] / close.iloc[0] - 1) if close.iloc[0] != 0 else 0.0
    # RSI proxy
    gains = ret.clip(lower=0).mean()
    losses = (-ret.clip(upper=0)).mean()
    rsi = 100 - 100 / (1 + gains / losses) if losses > 0 else 50.0
    # Position within range
    hi, lo = close.max(), close.min()
    pos_in_range = (close.iloc[-1] - lo) / (hi - lo) if hi != lo else 0.5
    # Volume trend
    vol_trend = (volume.iloc[-5:].mean() / volume.mean() - 1) if volume.mean() > 0 else 0.0
    # Skewness & kurtosis of returns
    skew = float(ret.skew()) if len(ret) > 3 else 0.0
    kurt = float(ret.kurt()) if len(ret) > 3 else 0.0
    # Momentum acceleration
    first_half = ret.iloc[:len(ret)//2].mean()
    second_half = ret.iloc[len(ret)//2:].mean()
    accel = second_half - first_half
    return pd.Series({
        "volatility": vol_ret,
        "trend": trend,
        "rsi": rsi,
        "pos_in_range": pos_in_range,
        "vol_trend": vol_trend,
        "skew": skew,
        "kurt": kurt,
        "accel": accel,
    })


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(a, b) / (n1 * n2))


def find_similar_setups(ticker: str, top_n: int = _TOP_N) -> dict[str, Any]:
    """
    Compare the current 20-bar window against all historical 20-bar windows
    for the same ticker. Return the top-N most similar past setups and what
    happened in the following 10 bars.
    """
    try:
        df = yf.download(ticker.upper(), period=_HISTORY_PERIOD, interval="1d",
                         auto_adjust=True, progress=False)
    except Exception as exc:
        log.warning("Similarity fetch failed for %s: %s", ticker, exc)
        return {"ticker": ticker.upper(), "available": False, "error": str(exc), "setups": []}

    if df is None or len(df) < _WINDOW + _FORWARD_HORIZON + 20:
        return {"ticker": ticker.upper(), "available": False, "error": "Insufficient data", "setups": []}

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"])
    closes = df["Close"].squeeze()
    volumes = df["Volume"].squeeze()

    # Current setup fingerprint (last _WINDOW bars)
    current_window_close = closes.iloc[-_WINDOW:]
    current_window_vol = volumes.iloc[-_WINDOW:]
    current_feat = _features(current_window_close, current_window_vol).values.astype(float)

    # Scan all past windows (leave _FORWARD_HORIZON bars at the end for outcome)
    scan_end = len(closes) - _FORWARD_HORIZON - 1
    candidates: list[dict] = []

    for i in range(_WINDOW, scan_end):
        window_close = closes.iloc[i - _WINDOW:i]
        window_vol = volumes.iloc[i - _WINDOW:i]
        feat = _features(window_close, window_vol).values.astype(float)

        if np.isnan(feat).any() or np.isnan(current_feat).any():
            continue

        sim = _cosine_similarity(current_feat, feat)

        # Measure forward outcome
        entry_price = float(closes.iloc[i])
        exit_price = float(closes.iloc[i + _FORWARD_HORIZON])
        forward_return = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0.0
        outcome = "bullish" if forward_return > 1.5 else "bearish" if forward_return < -1.5 else "sideways"

        date = str(closes.index[i].date())
        candidates.append({
            "date": date,
            "similarity": round(sim * 100, 1),
            "entry_price": round(entry_price, 2),
            "forward_return_pct": round(forward_return, 2),
            "outcome": outcome,
            "forward_bars": _FORWARD_HORIZON,
        })

    # Sort by similarity descending, take top N
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    top = candidates[:top_n]

    # Summarize the outcomes from top matches
    outcomes = [c["outcome"] for c in top]
    bullish_count = outcomes.count("bullish")
    bearish_count = outcomes.count("bearish")
    sideways_count = outcomes.count("sideways")
    avg_return = round(float(np.mean([c["forward_return_pct"] for c in top])), 2) if top else 0.0

    dominant = max(["bullish", "bearish", "sideways"], key=lambda o: outcomes.count(o)) if top else "unknown"

    return {
        "ticker": ticker.upper(),
        "available": True,
        "current_window_bars": _WINDOW,
        "forward_horizon_bars": _FORWARD_HORIZON,
        "setups": top,
        "summary": {
            "dominant_historical_outcome": dominant,
            "avg_forward_return_pct": avg_return,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "sideways_count": sideways_count,
            "total_matches_scanned": len(candidates),
        },
    }
