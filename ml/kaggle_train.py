"""
MarketVision AI — Kaggle GPU Training Script v12 (Final Hybrid)
============================================================
Improvements over v1:
  - 28 features (up from 10): ADX, OBV, Stochastic, Bollinger %B, ATR,
    volume z-score, day-of-week, sector ETF momentum, SPY/VIX regime
  - Dynamic ATR-based labels instead of fixed ±1.5% threshold
  - 10-day prediction horizon (cleaner signal than 5-day)
  - Ensemble-capable: stable Kaggle artifact uses XGBoost first; other models can be enabled later
  - SHAP-based feature pruning to remove noise
  - 50 tickers across sectors + indices
  - Fast Optuna tuning for a stable artifact; increase trials after the pipeline is proven
  - v5: optional recent 5m/15m/1h intraday feature blocks, VWAP reclaim/rejection,
    and multi-timeframe alignment scores
  - v6: strategy-aware setup features and strategy outcome targets for breakout,
    breakdown, pullback continuation, mean reversion, squeeze breakout,
    VWAP reclaim, and RSI divergence
  - v7: setup-only strategy quality training with practical win/loss,
    expected return, MFE/MAE, risk/reward, and backtest-style metrics
  - v8: strategy-filtered opportunity model that only learns from the
    historically positive long setups instead of every market candle
  - v9: specialist models for the strongest individual strategies so each
    setup gets its own probability, expected return, confidence buckets,
    and coverage proof
  - v10: elite trade-quality labels for only breakout long and mean reversion
    long, requiring target-before-stop, positive return, and minimum realized
    risk/reward instead of broad strategy success
  - v11: breakout-only precision optimization with positive-probability
    buckets and class-weight variants for high-confidence trade filtering
  - v12: final hybrid artifact that combines the full market model, auxiliary
    event intelligence, setup quality, specialist models, elite breakout
    proof, and the v11 precision experiment under one honest promotion policy

Kaggle setup:
  - Accelerator: GPU T4 x2
  - Internet: ON
  - Runtime: Python 3
"""

# ── 1. Install dependencies ──────────────────────────────────────────────────
import subprocess, sys

def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip("yfinance", "optuna", "xgboost", "joblib", "scikit-learn", "pyarrow", "shap")

# ── 2. Imports ───────────────────────────────────────────────────────────────
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
import yfinance as yf
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 3. Config ────────────────────────────────────────────────────────────────
CONFIG = {
    "tickers": [
        # Large-cap tech
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "QCOM",
        "AVGO", "MU", "ORCL", "CRM", "ADBE",
        # Finance
        "JPM", "BAC", "GS", "V", "MA",
        # Healthcare
        "UNH", "JNJ", "PFE", "ABBV",
        # Energy / Industrial
        "XOM", "CVX", "CAT", "BA",
        # Consumer
        "AMZN", "WMT", "HD", "NKE",
        # ETFs / indices (market regime context)
        "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV",
    ],
    "regime_tickers": ["SPY", "QQQ", "^VIX", "^TNX"],  # downloaded separately for regime/macro features
    "sector_etfs": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC", "XLU", "XLB", "XLP"],
    "period": "5y",
    "interval": "1d",
    "prediction_horizon": 10,       # 10-day forward return (cleaner than 5)
    "atr_label_multiplier": 0.75,   # dynamic threshold: ±0.75×ATR14
    "n_trials": 20,                 # Fast stable rerun; raise later for exhaustive tuning
    "output_dir": "/kaggle/working/artifacts",
    "use_gpu": True,
    "enable_lightgbm": False,       # LightGBM GPU produces compiler warnings and segfaults on Kaggle final fit
    "enable_catboost": False,       # CatBoost GPU is unstable on Kaggle final fit; keep stable XGB artifact first
    "shap_top_n_features": 45,      # keep top-N features after SHAP pruning
    "intraday_timeframes": ["5m", "15m", "1h"],  # Kaggle/yfinance-safe; 1m is too short/noisy for broad training
    "intraday_period": "60d",
    "strategy_targets": [
        "target_strategy_breakout_long_success",
        "target_strategy_breakdown_short_success",
        "target_strategy_pullback_continuation_success",
        "target_strategy_mean_reversion_long_success",
        "target_strategy_squeeze_breakout_success",
        "target_strategy_vwap_reclaim_success",
        "target_strategy_rsi_divergence_short_success",
    ],
    "strategy_definitions": [
        {"id": "breakout_long", "label": "Breakout long", "setup": "setup_breakout_long", "target": "target_strategy_breakout_long_success", "direction": "long"},
        {"id": "breakdown_short", "label": "Breakdown short", "setup": "setup_breakdown_short", "target": "target_strategy_breakdown_short_success", "direction": "short"},
        {"id": "pullback_continuation", "label": "Pullback continuation", "setup": "setup_pullback_continuation", "target": "target_strategy_pullback_continuation_success", "direction": "long"},
        {"id": "mean_reversion_long", "label": "Mean reversion long", "setup": "setup_mean_reversion_long", "target": "target_strategy_mean_reversion_long_success", "direction": "long"},
        {"id": "squeeze_breakout", "label": "Squeeze breakout", "setup": "setup_squeeze_breakout", "target": "target_strategy_squeeze_breakout_success", "direction": "long"},
        {"id": "vwap_reclaim", "label": "VWAP reclaim", "setup": "setup_vwap_reclaim", "target": "target_strategy_vwap_reclaim_success", "direction": "long"},
        {"id": "rsi_divergence_short", "label": "RSI divergence short", "setup": "setup_rsi_divergence_short", "target": "target_strategy_rsi_divergence_short_success", "direction": "short"},
    ],
    "high_quality_strategy_ids": [
        # Selected from v7 holdout expectancy/profit-factor evidence.
        "breakout_long",
        "mean_reversion_long",
        "squeeze_breakout",
        "vwap_reclaim",
    ],
    "specialist_strategy_ids": [
        # v11 focuses only on breakout precision because v10 showed it is the
        # closest path to 75%+ high-confidence accuracy.
        "breakout_long",
    ],
    "elite_strategy_ids": [
        # v11 removes mean reversion from the elite model and tunes breakout only.
        "breakout_long",
    ],
    "elite_min_return_pct": 0.01,
    "elite_min_risk_reward": 1.5,
    "breakout_precision_class_weights": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    "breakout_precision_min_signals": 10,
    "final_promoted_signal": "target_elite_breakout_long_quality_success",
    "final_model_policy": {
        "primary": "elite_breakout_quality",
        "secondary": ["general_market_regime", "event_probabilities", "strategy_quality", "future_range"],
        "experimental": ["breakout_precision_take"],
        "minimum_claim_threshold": 0.75,
        "disclaimer": "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
    },
}

# Deduplicate tickers
CONFIG["tickers"] = list(dict.fromkeys(CONFIG["tickers"]))

OUTPUT_DIR = Path(CONFIG["output_dir"])
if OUTPUT_DIR.is_absolute() and not OUTPUT_DIR.parent.exists():
    OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 4. Technical indicators ──────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14 = _atr(high, low, close, window)
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr14.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(window).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_pct = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid.replace(0, np.nan)
    return pct_b, width


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda values: np.dot(values, weights) / weights.sum(), raw=True)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    return series.rolling(window).apply(lambda values: float(np.dot(x, values - values.mean()) / denom), raw=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(window).mean()
    md = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    traded_value = typical * df["Volume"]
    session = pd.Series(pd.to_datetime(df.index).date, index=df.index)
    cumulative_value = traded_value.groupby(session).cumsum()
    cumulative_volume = df["Volume"].groupby(session).cumsum().replace(0, np.nan)
    return cumulative_value / cumulative_volume


def build_intraday_daily_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    df = _flatten(df.copy()).dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    close = df["Close"]
    volume = df["Volume"]
    vwap = _session_vwap(df)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ret_3 = close.pct_change(3)
    ret_12 = close.pct_change(12)
    vol_12 = close.pct_change().rolling(12).std()
    vwap_dist = close / vwap.replace(0, np.nan) - 1
    vwap_slope = _rolling_slope(vwap, 6) / close.replace(0, np.nan)
    vol_ratio = volume / volume.rolling(20).mean().replace(0, np.nan)
    price_above_vwap = close > vwap
    previous_above = price_above_vwap.shift(1).astype("boolean").fillna(False).astype(bool)

    raw = pd.DataFrame(index=df.index)
    raw[f"{timeframe}_trend"] = close / ema20.replace(0, np.nan) - 1
    raw[f"{timeframe}_ema9_20"] = ema9 / ema20.replace(0, np.nan) - 1
    raw[f"{timeframe}_ret_3"] = ret_3
    raw[f"{timeframe}_ret_12"] = ret_12
    raw[f"{timeframe}_volatility_12"] = vol_12
    raw[f"{timeframe}_volume_spike"] = vol_ratio
    raw[f"{timeframe}_above_vwap"] = price_above_vwap.astype(int)
    raw[f"{timeframe}_vwap_dist"] = vwap_dist
    raw[f"{timeframe}_vwap_slope"] = vwap_slope
    raw[f"{timeframe}_vwap_reclaim"] = ((~previous_above) & price_above_vwap).astype(int)
    raw[f"{timeframe}_vwap_rejection"] = (previous_above & (~price_above_vwap)).astype(int)

    raw["date"] = pd.to_datetime(raw.index).date
    daily = raw.groupby("date").last()
    daily.index = pd.to_datetime(daily.index)
    return daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ── 5. Feature engineering ───────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _join_context_close(index: pd.Index, context: dict[str, pd.DataFrame], ticker: str) -> pd.Series | None:
    source = context.get(ticker)
    if source is None or source.empty or "Close" not in source:
        return None
    return source["Close"].reindex(index, method="ffill")


def build_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    context: dict[str, pd.DataFrame] | None = None,
    ticker: str | None = None,
    intraday_context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]
    context = context or {}
    intraday_context = intraday_context or {}

    f = pd.DataFrame(index=df.index)

    # ── Returns
    f["ret_1"]  = close.pct_change(1)
    f["ret_3"]  = close.pct_change(3)
    f["ret_5"]  = close.pct_change(5)
    f["ret_10"] = close.pct_change(10)
    f["ret_20"] = close.pct_change(20)

    # ── Volatility
    f["vol_10"] = f["ret_1"].rolling(10).std()
    f["vol_20"] = f["ret_1"].rolling(20).std()
    f["volatility_ratio"] = f["vol_10"] / f["vol_20"].replace(0, np.nan)
    f["vol_expansion_10_20"] = f["vol_10"] / f["vol_20"].replace(0, np.nan) - 1

    # ── SMA / EMA / WMA trend stack
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    wma20 = _wma(close, 20)
    wma50 = _wma(close, 50)

    f["sma10_gap"] = close / sma10 - 1
    f["sma20_gap"] = close / sma20 - 1
    f["sma50_gap"] = close / sma50 - 1
    f["sma200_gap"] = close / sma200 - 1
    f["ema9_gap"] = close / ema9 - 1
    f["ema20_gap"] = close / ema20 - 1
    f["ema50_gap"] = close / ema50 - 1
    f["wma20_gap"] = close / wma20 - 1
    f["wma50_gap"] = close / wma50 - 1
    f["sma10_20_cross"] = sma10 / sma20 - 1
    f["sma20_50_cross"] = sma20 / sma50 - 1
    f["sma50_200_cross"] = sma50 / sma200 - 1
    f["ema9_20_cross"] = ema9 / ema20 - 1
    f["ema20_50_cross"] = ema20 / ema50 - 1
    f["trend_slope_10"] = _rolling_slope(close, 10) / close.replace(0, np.nan)
    f["trend_slope_20"] = _rolling_slope(close, 20) / close.replace(0, np.nan)
    f["trend_slope_50"] = _rolling_slope(close, 50) / close.replace(0, np.nan)

    # ── RSI
    rsi14 = _rsi(close, 14)
    rsi7 = _rsi(close, 7)
    f["rsi_14"] = rsi14
    f["rsi_7"]  = rsi7
    f["rsi_slope_5"] = _rolling_slope(rsi14, 5)
    f["rsi_slope_10"] = _rolling_slope(rsi14, 10)
    f["rsi_overbought"] = (rsi14 >= 70).astype(int)
    f["rsi_oversold"] = (rsi14 <= 30).astype(int)

    # ── MACD histogram
    macd_line, macd_signal, macd_hist = _macd(close)
    f["macd_line"] = macd_line / close.replace(0, np.nan)
    f["macd_signal"] = macd_signal / close.replace(0, np.nan)
    f["macd_hist"] = macd_hist / close.replace(0, np.nan)
    f["macd_hist_slope"] = _rolling_slope(macd_hist, 5) / close.replace(0, np.nan)

    # ── ATR (normalised)
    atr14 = _atr(high, low, close, 14)
    f["atr_norm"] = atr14 / close.replace(0, np.nan)
    f["atr_slope_5"] = _rolling_slope(atr14, 5) / close.replace(0, np.nan)

    # ── ADX
    f["adx"] = _adx(high, low, close, 14)

    # ── OBV momentum
    obv = _obv(close, vol)
    f["obv_mom"] = obv / obv.rolling(20).mean() - 1

    # ── Stochastic
    f["stoch_k"], f["stoch_d"] = _stochastic(high, low, close)
    f["stoch_diff"] = f["stoch_k"] - f["stoch_d"]

    # ── Bollinger
    f["bb_pctb"], f["bb_width"] = _bollinger(close, 20)
    f["bb_width_slope"] = _rolling_slope(f["bb_width"], 5)
    f["bb_squeeze"] = (f["bb_width"] < f["bb_width"].rolling(120).quantile(0.2)).astype(int)

    # ── CCI
    f["cci_20"] = _cci(high, low, close, 20)

    # ── Major volume / sustainability signals
    vol_avg_10 = vol.rolling(10).mean()
    vol_avg_15 = vol.rolling(15).mean()
    vol_avg_20 = vol.rolling(20).mean()
    vol_std_20 = vol.rolling(20).std()
    f["vol_zscore"] = (vol - vol_avg_20) / vol_std_20.replace(0, np.nan)
    f["vol_trend"] = vol / vol_avg_10 - 1
    f["vol_ratio_10"] = vol / vol_avg_10
    f["vol_ratio_15"] = vol / vol_avg_15
    f["vol_ratio_20"] = vol / vol_avg_20
    f["vol_spike_2x_10"] = (f["vol_ratio_10"] >= 2.0).astype(int)
    f["vol_spike_3x_10"] = (f["vol_ratio_10"] >= 3.0).astype(int)
    f["vol_spike_2x_15"] = (f["vol_ratio_15"] >= 2.0).astype(int)
    f["vol_spike_3x_15"] = (f["vol_ratio_15"] >= 3.0).astype(int)
    f["price_volume_pressure"] = f["ret_1"] * f["vol_ratio_20"]
    f["accumulation_pressure"] = ((close > close.shift(1)).astype(int) * f["vol_ratio_20"]).rolling(5).mean()
    f["distribution_pressure"] = ((close < close.shift(1)).astype(int) * f["vol_ratio_20"]).rolling(5).mean()

    # ── Breakout / breakdown with volume confirmation
    resistance_20 = high.shift(1).rolling(20).max()
    support_20 = low.shift(1).rolling(20).min()
    range_20 = (resistance_20 - support_20).replace(0, np.nan)
    f["dist_to_resistance_20"] = close / resistance_20 - 1
    f["dist_to_support_20"] = close / support_20 - 1
    f["range_position_20"] = (close - support_20) / range_20
    breakout_20 = close > resistance_20
    breakdown_20 = close < support_20
    f["breakout_20"] = breakout_20.astype(int)
    f["breakdown_20"] = breakdown_20.astype(int)
    f["breakout_volume_confirmed"] = (breakout_20 & (f["vol_ratio_10"] >= 2.0)).astype(int)
    f["breakdown_volume_confirmed"] = (breakdown_20 & (f["vol_ratio_10"] >= 2.0)).astype(int)
    f["failed_breakout_risk"] = ((high > resistance_20) & (close < resistance_20) & (f["vol_ratio_10"] >= 1.5)).astype(int)
    f["failed_breakdown_reversal"] = ((low < support_20) & (close > support_20) & (f["vol_ratio_10"] >= 1.5)).astype(int)

    # ── RSI divergence: price trend up while RSI trend down, or vice versa
    price_slope_10 = _rolling_slope(close, 10) / close.replace(0, np.nan)
    rsi_slope_10 = _rolling_slope(rsi14, 10) / 100
    f["bearish_rsi_divergence"] = ((price_slope_10 > 0) & (rsi_slope_10 < 0)).astype(int)
    f["bullish_rsi_divergence"] = ((price_slope_10 < 0) & (rsi_slope_10 > 0)).astype(int)
    f["rsi_divergence_strength"] = price_slope_10 - rsi_slope_10

    # ── Day-of-week (cyclical encoding)
    dow = pd.Series(df.index.dayofweek, index=df.index)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # ── Month seasonality
    month = pd.Series(df.index.month, index=df.index)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── SPY regime features
    if spy_df is not None:
        spy_close = spy_df["Close"].reindex(df.index, method="ffill")
        f["spy_ret5"]   = spy_close.pct_change(5)
        f["spy_sma50g"] = spy_close / spy_close.rolling(50).mean() - 1
        f["spy_vol20"]  = spy_close.pct_change().rolling(20).std()
        f["rel_spy5"]   = f["ret_5"] - f["spy_ret5"]   # relative strength

    # ── Market regime / macro proxies available from Yahoo
    qqq_close = _join_context_close(df.index, context, "QQQ")
    vix_close = _join_context_close(df.index, context, "^VIX")
    tnx_close = _join_context_close(df.index, context, "^TNX")
    if qqq_close is not None:
        f["qqq_ret5"] = qqq_close.pct_change(5)
        f["rel_qqq5"] = f["ret_5"] - f["qqq_ret5"]
        f["qqq_sma50g"] = qqq_close / qqq_close.rolling(50).mean() - 1
    if vix_close is not None:
        f["vix_level"] = vix_close / 100
        f["vix_ret5"] = vix_close.pct_change(5)
        f["vix_high_regime"] = (vix_close >= 25).astype(int)
        f["vix_low_regime"] = (vix_close <= 15).astype(int)
    if tnx_close is not None:
        f["ten_year_yield_proxy"] = tnx_close / 100
        f["ten_year_yield_change_20"] = tnx_close.diff(20) / 100

    # ── Sector rotation matrix: ETF momentum and stock relative strength vs sector basket
    sector_returns = []
    for sector in ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLC", "XLU", "XLB", "XLP"]:
        sector_close = _join_context_close(df.index, context, sector)
        if sector_close is None:
            continue
        sector_ret5 = sector_close.pct_change(5)
        sector_ret20 = sector_close.pct_change(20)
        f[f"{sector.lower()}_ret5"] = sector_ret5
        f[f"{sector.lower()}_ret20"] = sector_ret20
        sector_returns.append(sector_ret5)
    if sector_returns:
        sector_matrix = pd.concat(sector_returns, axis=1)
        f["sector_basket_ret5"] = sector_matrix.mean(axis=1)
        f["sector_best_ret5"] = sector_matrix.max(axis=1)
        f["sector_worst_ret5"] = sector_matrix.min(axis=1)
        f["sector_dispersion_5"] = sector_matrix.std(axis=1)
        f["stock_vs_sector_basket_5"] = f["ret_5"] - f["sector_basket_ret5"]

    # ── Intraday/VWAP and multi-timeframe alignment features.
    # yfinance intraday history is limited, so missing older rows are neutral-filled.
    timeframe_trend_cols: list[str] = []
    if ticker and ticker in intraday_context:
        daily_intraday = intraday_context[ticker]
        daily_intraday = daily_intraday.copy()
        daily_intraday.index = pd.to_datetime(daily_intraday.index).tz_localize(None).normalize()
        daily_index = pd.to_datetime([stamp.date() for stamp in df.index])
        aligned = daily_intraday.reindex(daily_index).fillna(0.0)
        aligned.index = df.index
        for col in aligned.columns:
            f[col] = aligned[col].astype(float)
        timeframe_trend_cols = [col for col in aligned.columns if col.endswith("_trend")]
    for timeframe in ["5m", "15m", "1h"]:
        for suffix in [
            "trend", "ema9_20", "ret_3", "ret_12", "volatility_12", "volume_spike",
            "above_vwap", "vwap_dist", "vwap_slope", "vwap_reclaim", "vwap_rejection",
        ]:
            col = f"{timeframe}_{suffix}"
            if col not in f:
                f[col] = 0.0
            if suffix == "trend" and col not in timeframe_trend_cols:
                timeframe_trend_cols.append(col)
    if timeframe_trend_cols:
        trends = pd.concat([(f[col] > 0).astype(int).rename(col) for col in timeframe_trend_cols], axis=1)
        bearish_trends = pd.concat([(f[col] < 0).astype(int).rename(col) for col in timeframe_trend_cols], axis=1)
        f["mtf_bullish_alignment"] = trends.mean(axis=1)
        f["mtf_bearish_alignment"] = bearish_trends.mean(axis=1)
        f["mtf_mixed_alignment"] = 1 - (f["mtf_bullish_alignment"] - f["mtf_bearish_alignment"]).abs()
        f["mtf_trend_score"] = f[timeframe_trend_cols].mean(axis=1)

    # ── Strategy setup detectors. These are not advice; they describe market
    # conditions so the model can learn which setups historically worked.
    f["setup_breakout_long"] = (
        (f["breakout_20"] == 1)
        & (f["vol_ratio_10"] >= 1.5)
        & (f["sma20_gap"] > 0)
        & (f["adx"] >= 18)
    ).astype(int)
    f["setup_breakdown_short"] = (
        (f["breakdown_20"] == 1)
        & (f["vol_ratio_10"] >= 1.5)
        & (f["sma20_gap"] < 0)
        & (f["adx"] >= 18)
    ).astype(int)
    f["setup_pullback_continuation"] = (
        (f["sma50_gap"] > 0)
        & (f["ema20_50_cross"] > 0)
        & (f["ret_5"] < 0)
        & (f["rsi_14"].between(38, 58))
        & (f["mtf_bullish_alignment"] >= 0.5)
    ).astype(int)
    f["setup_mean_reversion_long"] = (
        ((f["rsi_14"] <= 35) | (f["bb_pctb"] <= 0.05))
        & (f["dist_to_support_20"] <= 0.02)
        & (f["volatility_ratio"] <= 1.35)
    ).astype(int)
    f["setup_squeeze_breakout"] = (
        (f["bb_squeeze"] == 1)
        & (f["range_position_20"] >= 0.7)
        & (f["vol_ratio_10"] >= 1.25)
    ).astype(int)
    f["setup_vwap_reclaim"] = (
        ((f["5m_vwap_reclaim"] == 1) | (f["15m_vwap_reclaim"] == 1) | (f["1h_vwap_reclaim"] == 1))
        & (f["mtf_bullish_alignment"] >= 0.5)
    ).astype(int)
    f["setup_rsi_divergence_short"] = (
        (f["bearish_rsi_divergence"] == 1)
        & (f["rsi_14"] >= 55)
        & (f["vol_ratio_10"] >= 1.0)
    ).astype(int)

    return f


def build_labels(df: pd.DataFrame, horizon: int, atr_mult: float) -> pd.Series:
    close = df["Close"]
    atr14 = _atr(df["High"], df["Low"], close, 14)
    fwd_ret = close.shift(-horizon) / close - 1
    thresh = atr_mult * atr14 / close  # dynamic threshold per bar
    labels = np.select(
        [fwd_ret > thresh, fwd_ret < -thresh],
        [2, 0],
        default=1,
    )
    return pd.Series(labels, index=df.index, name="target")


def build_forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    close = df["Close"]
    return (close.shift(-horizon) / close - 1).rename("forward_return")


def build_additional_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    current_atr = _atr(high, low, close, 14)
    forward_return = build_forward_return(df, horizon)
    future_high = high.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    future_low = low.shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    current_resistance = high.shift(1).rolling(20).max()
    current_support = low.shift(1).rolling(20).min()
    current_vol = close.pct_change().rolling(10).std()
    future_vol = close.pct_change().shift(-1).rolling(horizon).std().shift(-(horizon - 1))

    targets = pd.DataFrame(index=df.index)
    targets["target_return_horizon"] = forward_return
    targets["target_future_range_high_pct"] = future_high / close - 1
    targets["target_future_range_low_pct"] = future_low / close - 1
    targets["target_breakout"] = (future_high > current_resistance * 1.002).astype(int)
    targets["target_breakdown"] = (future_low < current_support * 0.998).astype(int)
    targets["target_reversal"] = ((np.sign(close.pct_change(5)) != np.sign(forward_return)) & (close.pct_change(5).abs() > 0.02)).astype(int)
    targets["target_volatility_expansion"] = (future_vol > current_vol * 1.25).astype(int)
    targets["target_risk_adjusted_return"] = forward_return / (current_atr / close).replace(0, np.nan)
    return targets


def build_strategy_targets(df: pd.DataFrame, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    atr_pct = (_atr(high, low, close, 14) / close.replace(0, np.nan)).clip(lower=0.005, upper=0.08)
    forward_return = build_forward_return(df, horizon)
    future_high_pct = high.shift(-1).rolling(horizon).max().shift(-(horizon - 1)) / close - 1
    future_low_pct = low.shift(-1).rolling(horizon).min().shift(-(horizon - 1)) / close - 1
    reward_threshold = (atr_pct * 1.25).clip(lower=0.012)
    risk_threshold = -(atr_pct * 0.9).clip(lower=0.01)
    short_reward_threshold = -reward_threshold
    short_risk_threshold = np.maximum(atr_pct * 0.9, 0.01)

    targets = pd.DataFrame(index=df.index)

    strategy_rules = {
        "breakout_long": {
            "target": "target_strategy_breakout_long_success",
            "setup": "setup_breakout_long",
            "direction": "long",
            "target_hit": future_high_pct >= reward_threshold,
            "stop_hit": future_low_pct <= risk_threshold,
            "return": forward_return,
        },
        "breakdown_short": {
            "target": "target_strategy_breakdown_short_success",
            "setup": "setup_breakdown_short",
            "direction": "short",
            "target_hit": future_low_pct <= short_reward_threshold,
            "stop_hit": future_high_pct >= short_risk_threshold,
            "return": -forward_return,
        },
        "pullback_continuation": {
            "target": "target_strategy_pullback_continuation_success",
            "setup": "setup_pullback_continuation",
            "direction": "long",
            "target_hit": forward_return >= reward_threshold * 0.75,
            "stop_hit": future_low_pct <= risk_threshold,
            "return": forward_return,
        },
        "mean_reversion_long": {
            "target": "target_strategy_mean_reversion_long_success",
            "setup": "setup_mean_reversion_long",
            "direction": "long",
            "target_hit": future_high_pct >= reward_threshold * 0.8,
            "stop_hit": future_low_pct <= risk_threshold * 1.2,
            "return": forward_return,
        },
        "squeeze_breakout": {
            "target": "target_strategy_squeeze_breakout_success",
            "setup": "setup_squeeze_breakout",
            "direction": "long",
            "target_hit": future_high_pct >= reward_threshold * 1.4,
            "stop_hit": future_low_pct <= risk_threshold,
            "return": forward_return,
        },
        "vwap_reclaim": {
            "target": "target_strategy_vwap_reclaim_success",
            "setup": "setup_vwap_reclaim",
            "direction": "long",
            "target_hit": forward_return >= reward_threshold * 0.65,
            "stop_hit": future_low_pct <= risk_threshold,
            "return": forward_return,
        },
        "rsi_divergence_short": {
            "target": "target_strategy_rsi_divergence_short_success",
            "setup": "setup_rsi_divergence_short",
            "direction": "short",
            "target_hit": forward_return <= short_reward_threshold * 0.65,
            "stop_hit": future_high_pct >= short_risk_threshold,
            "return": -forward_return,
        },
    }

    for strategy_id, rule in strategy_rules.items():
        active = features[rule["setup"]].astype(int) == 1
        target_hit = pd.Series(rule["target_hit"], index=df.index).fillna(False)
        stop_hit = pd.Series(rule["stop_hit"], index=df.index).fillna(False)
        strategy_return = pd.Series(rule["return"], index=df.index).astype(float)
        if rule["direction"] == "short":
            mfe = (-future_low_pct).astype(float)
            mae = (-future_high_pct).astype(float)
        else:
            mfe = future_high_pct.astype(float)
            mae = future_low_pct.astype(float)

        success = active & target_hit & ~stop_hit
        adverse = mae.abs().clip(lower=0.002)
        targets[rule["target"]] = success.astype(int)
        targets[f"target_strategy_{strategy_id}_active"] = active.astype(int)
        targets[f"target_strategy_{strategy_id}_return_pct"] = strategy_return.where(active, 0.0)
        targets[f"target_strategy_{strategy_id}_mfe_pct"] = mfe.where(active, 0.0)
        targets[f"target_strategy_{strategy_id}_mae_pct"] = mae.where(active, 0.0)
        targets[f"target_strategy_{strategy_id}_target_hit"] = (active & target_hit).astype(int)
        targets[f"target_strategy_{strategy_id}_stop_hit"] = (active & stop_hit).astype(int)
        targets[f"target_strategy_{strategy_id}_target_without_stop"] = success.astype(int)
        targets[f"target_strategy_{strategy_id}_risk_reward"] = (mfe / adverse).replace([np.inf, -np.inf], np.nan).clip(-10, 10).where(active, 0.0)

    quality_ids = CONFIG.get("high_quality_strategy_ids", [])
    active_cols = [f"target_strategy_{sid}_active" for sid in quality_ids if f"target_strategy_{sid}_active" in targets]
    success_cols = [
        strategy_rules[sid]["target"]
        for sid in quality_ids
        if sid in strategy_rules and strategy_rules[sid]["target"] in targets
    ]
    return_cols = [f"target_strategy_{sid}_return_pct" for sid in quality_ids if f"target_strategy_{sid}_return_pct" in targets]
    rr_cols = [f"target_strategy_{sid}_risk_reward" for sid in quality_ids if f"target_strategy_{sid}_risk_reward" in targets]
    if active_cols:
        active_matrix = targets[active_cols].astype(int)
        active_count = active_matrix.sum(axis=1).replace(0, np.nan)
        active_values = active_matrix.to_numpy()
        active_return_sum = pd.Series(
            (targets[return_cols].to_numpy() * active_values).sum(axis=1),
            index=targets.index,
        )
        active_rr_sum = pd.Series(
            (targets[rr_cols].to_numpy() * active_values).sum(axis=1),
            index=targets.index,
        )
        targets["target_high_quality_strategy_active"] = (active_matrix.sum(axis=1) > 0).astype(int)
        targets["target_high_quality_strategy_success"] = (targets[success_cols].astype(int).sum(axis=1) > 0).astype(int)
        targets["target_high_quality_strategy_return_pct"] = (active_return_sum / active_count).fillna(0.0)
        targets["target_high_quality_strategy_risk_reward"] = (active_rr_sum / active_count).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        targets["target_high_quality_strategy_count"] = active_matrix.sum(axis=1).astype(int)

    elite_ids = CONFIG.get("elite_strategy_ids", [])
    elite_min_return = float(CONFIG.get("elite_min_return_pct", 0.01))
    elite_min_rr = float(CONFIG.get("elite_min_risk_reward", 1.5))
    elite_active_cols = []
    elite_quality_cols = []
    elite_return_cols = []
    elite_rr_cols = []
    for sid in elite_ids:
        if sid not in strategy_rules:
            continue
        active_col = f"target_strategy_{sid}_active"
        return_col = f"target_strategy_{sid}_return_pct"
        rr_col = f"target_strategy_{sid}_risk_reward"
        quality_col = f"target_elite_{sid}_quality_success"
        if active_col not in targets or return_col not in targets or rr_col not in targets:
            continue
        elite_active_cols.append(active_col)
        elite_quality_cols.append(quality_col)
        elite_return_cols.append(return_col)
        elite_rr_cols.append(rr_col)
        active = targets[active_col].astype(int) == 1
        targets[quality_col] = (
            active
            & (targets[strategy_rules[sid]["target"]].astype(int) == 1)
            & (targets[return_col].astype(float) >= elite_min_return)
            & (targets[rr_col].astype(float) >= elite_min_rr)
        ).astype(int)
        targets[f"target_elite_{sid}_return_positive"] = (
            active & (targets[return_col].astype(float) > 0)
        ).astype(int)

    if elite_active_cols:
        elite_active_matrix = targets[elite_active_cols].astype(int)
        elite_count = elite_active_matrix.sum(axis=1).replace(0, np.nan)
        elite_values = elite_active_matrix.to_numpy()
        elite_return_sum = pd.Series(
            (targets[elite_return_cols].to_numpy() * elite_values).sum(axis=1),
            index=targets.index,
        )
        elite_rr_sum = pd.Series(
            (targets[elite_rr_cols].to_numpy() * elite_values).sum(axis=1),
            index=targets.index,
        )
        targets["target_elite_strategy_active"] = (elite_active_matrix.sum(axis=1) > 0).astype(int)
        targets["target_elite_strategy_quality_success"] = (targets[elite_quality_cols].astype(int).sum(axis=1) > 0).astype(int)
        targets["target_elite_strategy_return_pct"] = (elite_return_sum / elite_count).fillna(0.0)
        targets["target_elite_strategy_risk_reward"] = (elite_rr_sum / elite_count).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        targets["target_elite_strategy_count"] = elite_active_matrix.sum(axis=1).astype(int)
    return targets


def build_training_manifest() -> dict[str, dict[str, Any]]:
    return {
        "phase_1_intraday_data": {
            "status": "partial",
            "details": "Adds yfinance-safe recent 5m/15m/1h intraday feature blocks, neutral-filled for older daily rows. Full 1m/4h multi-year coverage still needs Polygon/Alpaca storage.",
        },
        "phase_2_vwap_features": {
            "status": "partial",
            "details": "Adds intraday price-above-VWAP, VWAP distance, VWAP slope, reclaim, and rejection features for available recent intraday windows.",
        },
        "phase_3_multi_timeframe_alignment": {
            "status": "partial",
            "details": "Adds 5m/15m/1h trend features and bullish/bearish/mixed multi-timeframe alignment scores. 4h/daily/weekly stacked alignment is pending.",
        },
        "phase_4_market_regime_detection": {
            "status": "partial",
            "details": "SPY/QQQ trend, VIX regime, volatility, and yield proxy features are included. Full regime classifier is pending.",
        },
        "phase_5_macro_vix_data": {
            "status": "partial",
            "details": "VIX and 10-year yield proxy are included from Yahoo. FRED inflation, GDP, unemployment, Fed/CPI calendars are pending.",
        },
        "phase_6_sector_rotation": {
            "status": "partial",
            "details": "Sector ETF returns, basket strength, dispersion, and stock-vs-sector basket strength are included.",
        },
        "phase_7_news_sentiment": {
            "status": "missing",
            "details": "FinBERT/OpenAI sentiment pipeline is not part of this Kaggle training script yet.",
        },
        "phase_8_event_flags": {
            "status": "missing",
            "details": "Earnings, Fed, CPI, OPEX, and major-event flags require event calendar ingestion.",
        },
        "phase_9_improved_labels": {
            "status": "trained",
            "details": "Adds and trains models for 3-class ATR labels, binary direction, breakout, breakdown, reversal, volatility expansion, future return, future range, and risk-adjusted return targets.",
        },
        "phase_10_confidence_filtering": {
            "status": "trained",
            "details": "Reports all-signal and >=60/70/80/85 confidence bucket accuracy with coverage.",
        },
        "phase_11_historical_similarity": {
            "status": "partial",
            "details": "Backend has a similarity service. Training artifact does not yet ensemble it.",
        },
        "phase_12_monte_carlo_simulation": {
            "status": "partial",
            "details": "Backend simulation exists. Training artifact does not yet save Monte Carlo calibration parameters.",
        },
        "phase_13_sequence_models": {
            "status": "missing",
            "details": "LSTM/GRU/Transformer/TFT should come after feature and label quality improves.",
        },
        "phase_14_ensemble_strategy": {
            "status": "partial",
            "details": "Stable artifact trains multiple XGBoost classifiers/regressors. LightGBM/CatBoost hooks remain disabled because previous Kaggle GPU runs warned/segfaulted.",
        },
        "phase_15_walk_forward_validation": {
            "status": "partial",
            "details": "Optuna uses TimeSeriesSplit and final holdout is chronological. Full rolling walk-forward backtest windows are pending.",
        },
        "phase_16_drift_retraining": {
            "status": "missing",
            "details": "Model/version metadata is saved; automated drift detection and retraining scheduler are pending.",
        },
        "phase_17_llm_explanation_layer": {
            "status": "partial",
            "details": "Metrics and backend provide explanation-ready facts. LLM remains explanation-only, not a direct price predictor.",
        },
        "phase_18_strategy_aware_prediction": {
            "status": "trained",
            "details": "Adds setup detectors and v7 setup-only strategy-quality targets for breakout long, breakdown short, pullback continuation, mean reversion, squeeze breakout, VWAP reclaim, and RSI divergence short.",
        },
        "phase_19_strategy_quality_validation": {
            "status": "trained",
            "details": "Reports strategy quality only on active setup rows: signal count, coverage, win rate, profit factor, average win/loss, max drawdown, Sharpe, MFE/MAE, expected return, and risk/reward.",
        },
        "phase_20_strategy_filtered_prediction": {
            "status": "trained",
            "details": "Adds a v8 high-quality strategy opportunity model trained only on v7-positive long setups instead of all candles, with accuracy and confidence buckets reported against disclosed coverage.",
        },
        "phase_21_specialist_strategy_models": {
            "status": "trained",
            "details": "Adds v9 specialist models for breakout long, mean reversion long, and VWAP reclaim, with separate setup-only probability, expected return, risk/reward, confidence buckets, and coverage proof.",
        },
        "phase_22_elite_trade_quality_models": {
            "status": "trained",
            "details": "Adds v10 elite trade-quality models for breakout long and mean reversion long only, requiring target-before-stop success, positive return, and minimum realized risk/reward.",
        },
        "phase_23_breakout_precision_optimization": {
            "status": "trained",
            "details": "Adds v11 breakout-only class-weight variants and positive-probability precision buckets to optimize high-confidence TAKE-trade filtering.",
        },
        "phase_24_final_hybrid_artifact": {
            "status": "trained",
            "details": "Adds v12 final hybrid policy that combines all model layers but promotes only the best validated signal, elite breakout quality, with all accuracy claims tied to confidence threshold and coverage.",
        },
        "extra_technical_volume_divergence": {
            "status": "trained",
            "details": "Adds SMA/WMA/EMA, RSI, MACD, Bollinger, ATR/ADX, 2x/3x volume spikes, volume-confirmed breakouts/breakdowns, and RSI bullish/bearish divergence features.",
        },
    }


# ── 6. Data download ──────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 100:
            print(f"  [skip] {ticker}: only {len(df)} rows")
            return None
        print(f"  [ok]   {ticker}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"  [err]  {ticker}: {e}")
        return None


def fetch_intraday_context(ticker: str, config: dict) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for timeframe in config.get("intraday_timeframes", []):
        try:
            raw = yf.Ticker(ticker).history(
                period=config.get("intraday_period", "60d"),
                interval=timeframe,
                auto_adjust=True,
            )
            raw = _flatten(raw)
            if raw.empty or len(raw) < 30:
                print(f"  [intraday skip] {ticker} {timeframe}: {len(raw)} rows")
                continue
            engineered = build_intraday_daily_features(raw, timeframe)
            if not engineered.empty:
                frames.append(engineered)
                print(f"  [intraday ok]   {ticker} {timeframe}: {len(engineered)} sessions")
        except Exception as exc:
            print(f"  [intraday err]  {ticker} {timeframe}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_dataset(config: dict) -> tuple[pd.DataFrame, list[str]]:
    print("\n── Downloading regime data ───────────────────────────")
    spy_df = fetch_ticker("SPY", config["period"], config["interval"])
    context = {"SPY": spy_df} if spy_df is not None else {}
    for ctx_ticker in list(dict.fromkeys(config.get("regime_tickers", []) + config.get("sector_etfs", []))):
        if ctx_ticker == "SPY":
            continue
        ctx_df = fetch_ticker(ctx_ticker, config["period"], config["interval"])
        if ctx_df is not None:
            context[ctx_ticker] = ctx_df

    print("\n── Downloading ticker data ───────────────────────────")
    frames = []
    for ticker in config["tickers"]:
        raw = fetch_ticker(ticker, config["period"], config["interval"])
        if raw is None:
            continue
        try:
            intraday_context = {ticker: fetch_intraday_context(ticker, config)}
            feats = build_features(raw, spy_df, context=context, ticker=ticker, intraday_context=intraday_context)
            labels = build_labels(raw, config["prediction_horizon"], config["atr_label_multiplier"])
            forward_return = build_forward_return(raw, config["prediction_horizon"])
            additional_targets = build_additional_targets(raw, config["prediction_horizon"])
            strategy_targets = build_strategy_targets(raw, feats, config["prediction_horizon"])
            combined = feats.copy()
            combined["target"] = labels
            combined["forward_return"] = forward_return
            combined["target_binary"] = (forward_return > 0).astype(int)
            combined = combined.join(additional_targets)
            combined = combined.join(strategy_targets)
            combined["ticker"] = ticker
            combined = combined.dropna()
            frames.append(combined)
        except Exception as e:
            print(f"  [feat err] {ticker}: {e}")

    if not frames:
        raise ValueError("No data built.")

    data = pd.concat(frames).sort_index()
    feature_cols = [
        c for c in data.columns
        if c not in ("target", "forward_return", "ticker") and not c.startswith("target_")
    ]

    print(f"\nDataset: {len(data):,} rows | {data['ticker'].nunique()} tickers | {len(feature_cols)} features")
    dist = data["target"].value_counts().rename({0: "bearish", 1: "sideways", 2: "bullish"})
    print(f"Label distribution:\n{dist}")
    binary_dist = data["target_binary"].value_counts().rename({0: "down_or_flat", 1: "up"})
    print(f"Binary direction distribution:\n{binary_dist}")
    return data, feature_cols


# ── 7. Models ─────────────────────────────────────────────────────────────────

def make_xgb(trial: optuna.Trial | None, use_gpu: bool, params: dict | None = None):
    p = params or _xgb_params(trial)
    return XGBClassifier(
        **p,
        objective="multi:softprob", eval_metric="mlogloss", num_class=3,
        tree_method="hist", device="cuda" if use_gpu else "cpu",
        random_state=42, verbosity=0,
    )


def make_xgb_binary(params: dict, use_gpu: bool):
    return XGBClassifier(
        **params,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device="cuda" if use_gpu else "cpu",
        random_state=42, verbosity=0,
    )


def make_xgb_regressor(params: dict, use_gpu: bool):
    reg_params = params.copy()
    reg_params.pop("scale_pos_weight", None)
    return XGBRegressor(
        **reg_params,
        objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", device="cuda" if use_gpu else "cpu",
        random_state=42, verbosity=0,
    )


def make_lgb(trial: optuna.Trial | None, use_gpu: bool, params: dict | None = None):
    from lightgbm import LGBMClassifier

    p = params or _lgb_params(trial)
    return LGBMClassifier(
        **p,
        objective="multiclass", num_class=3,
        device="cpu",
        random_state=42, verbose=-1,
    )


def make_cat(trial: optuna.Trial | None, use_gpu: bool, params: dict | None = None):
    from catboost import CatBoostClassifier

    p = (params or _cat_params(trial)).copy()
    if use_gpu:
        # CatBoost GPU only supports rsm/colsample_bylevel for pairwise losses.
        # MultiClass training must omit it.
        p.pop("colsample_bylevel", None)
    return CatBoostClassifier(
        **p,
        loss_function="MultiClass", eval_metric="Accuracy",
        bootstrap_type="Bernoulli",
        task_type="GPU" if use_gpu else "CPU",
        random_seed=42, verbose=0,
    )


def _xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("xgb_n_est",   200, 1000),
        "max_depth":         trial.suggest_int("xgb_depth",   2,   8),
        "learning_rate":     trial.suggest_float("xgb_lr",    0.003, 0.15, log=True),
        "subsample":         trial.suggest_float("xgb_sub",   0.5,  1.0),
        "colsample_bytree":  trial.suggest_float("xgb_col",   0.4,  1.0),
        "min_child_weight":  trial.suggest_int("xgb_mcw",     1,    10),
        "gamma":             trial.suggest_float("xgb_gamma", 0.0,  0.5),
        "reg_alpha":         trial.suggest_float("xgb_alpha", 0.0,  1.0),
        "reg_lambda":        trial.suggest_float("xgb_lam",   0.5,  5.0),
    }


def _lgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":       trial.suggest_int("lgb_n_est",  200, 1000),
        "num_leaves":         trial.suggest_int("lgb_leaves", 16,  128),
        "learning_rate":      trial.suggest_float("lgb_lr",   0.003, 0.15, log=True),
        "subsample":          trial.suggest_float("lgb_sub",  0.5,  1.0),
        "colsample_bytree":   trial.suggest_float("lgb_col",  0.4,  1.0),
        "min_child_samples":  trial.suggest_int("lgb_mcs",    5,    50),
        "reg_alpha":          trial.suggest_float("lgb_alpha",0.0,  1.0),
        "reg_lambda":         trial.suggest_float("lgb_lam",  0.0,  5.0),
    }


def _cat_params(trial: optuna.Trial) -> dict:
    return {
        "iterations":    trial.suggest_int("cat_iter",  200, 800),
        "depth":         trial.suggest_int("cat_depth", 4,   8),
        "learning_rate": trial.suggest_float("cat_lr",  0.01, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("cat_l2",  1.0, 10.0),
        "subsample":     trial.suggest_float("cat_sub",  0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("cat_col", 0.5, 1.0),
    }


def _strip_prefix(params: dict, prefix: str, mapping: dict[str, str]) -> dict:
    return {mapping[key]: value for key, value in params.items() if key.startswith(prefix) and key in mapping}


def xgb_best_params(study: optuna.Study) -> dict:
    return _strip_prefix(study.best_params, "xgb_", {
        "xgb_n_est": "n_estimators",
        "xgb_depth": "max_depth",
        "xgb_lr": "learning_rate",
        "xgb_sub": "subsample",
        "xgb_col": "colsample_bytree",
        "xgb_mcw": "min_child_weight",
        "xgb_gamma": "gamma",
        "xgb_alpha": "reg_alpha",
        "xgb_lam": "reg_lambda",
    })


def lgb_best_params(study: optuna.Study) -> dict:
    return _strip_prefix(study.best_params, "lgb_", {
        "lgb_n_est": "n_estimators",
        "lgb_leaves": "num_leaves",
        "lgb_lr": "learning_rate",
        "lgb_sub": "subsample",
        "lgb_col": "colsample_bytree",
        "lgb_mcs": "min_child_samples",
        "lgb_alpha": "reg_alpha",
        "lgb_lam": "reg_lambda",
    })


def cat_best_params(study: optuna.Study) -> dict:
    params = _strip_prefix(study.best_params, "cat_", {
        "cat_iter": "iterations",
        "cat_depth": "depth",
        "cat_lr": "learning_rate",
        "cat_l2": "l2_leaf_reg",
        "cat_sub": "subsample",
        "cat_col": "colsample_bylevel",
    })
    params.pop("colsample_bylevel", None)
    return params


def make_cpu_portable_model(model, model_name: str):
    """
    Kaggle can train with GPU, but the backend usually runs on CPU.
    Switch saved wrappers to CPU-safe inference settings before joblib export.
    """
    try:
        if model_name == "xgboost":
            model.set_params(device="cpu", tree_method="hist")
            model.get_booster().set_param({"device": "cpu", "tree_method": "hist"})
        elif model_name == "lightgbm":
            model.set_params(device="cpu")
            model.booster_.reset_parameter({"device": "cpu"})
        elif model_name == "catboost":
            model.set_params(task_type="CPU")
    except Exception as exc:
        print(f"[warn] Could not switch {model_name} to CPU inference settings: {exc}")
    return model


# ── 8. CV scoring ─────────────────────────────────────────────────────────────

def cv_score(X, y, make_fn, trial, use_gpu, n_splits=5) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        m = make_fn(trial, use_gpu)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = m.predict(X.iloc[va_idx])
        scores.append(f1_score(y.iloc[va_idx], preds, average="macro"))
    return float(np.mean(scores))


def confidence_bucket_report(y_true: pd.Series, proba: np.ndarray, thresholds=(0.60, 0.70, 0.80, 0.85)) -> dict[str, dict[str, float | int | None]]:
    preds = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    y_arr = np.asarray(y_true)
    total = len(y_arr)
    buckets = {}
    for threshold in thresholds:
        mask = confidence >= threshold
        rows = int(mask.sum())
        key = f">={int(threshold * 100)}%"
        if rows == 0:
            buckets[key] = {
                "accuracy": None,
                "coverage": 0.0,
                "signals": 0,
            }
            continue
        buckets[key] = {
            "accuracy": float(accuracy_score(y_arr[mask], preds[mask])),
            "coverage": float(rows / total),
            "signals": rows,
        }
    return buckets


def binary_model_report(y_true: pd.Series, proba: np.ndarray) -> dict[str, Any]:
    preds = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "positive_rate": float(np.mean(preds)),
        "class_counts": y_true.value_counts().sort_index().to_dict(),
        "confidence_buckets": confidence_bucket_report(y_true, proba),
    }


def positive_probability_bucket_report(
    y_true: pd.Series,
    proba: np.ndarray,
    thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
) -> dict[str, dict[str, float | int | None]]:
    y_arr = np.asarray(y_true)
    positive_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.reshape(-1)
    total = len(y_arr)
    buckets: dict[str, dict[str, float | int | None]] = {}
    for threshold in thresholds:
        mask = positive_proba >= threshold
        rows = int(mask.sum())
        key = f">={int(threshold * 100)}%"
        if rows == 0:
            buckets[key] = {
                "precision": None,
                "recall": 0.0,
                "coverage": 0.0,
                "signals": 0,
                "win_rate": None,
            }
            continue
        true_positive = int((y_arr[mask] == 1).sum())
        total_positive = max(int((y_arr == 1).sum()), 1)
        buckets[key] = {
            "precision": float(true_positive / rows),
            "recall": float(true_positive / total_positive),
            "coverage": float(rows / total),
            "signals": rows,
            "win_rate": float(np.mean(y_arr[mask])),
        }
    return buckets


def regression_model_report(y_true: pd.Series, preds: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(preds, dtype=float)
    direction_acc = accuracy_score(y_arr > 0, pred_arr > 0)
    return {
        "mae": float(mean_absolute_error(y_arr, pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_arr, pred_arr))),
        "directional_accuracy": float(direction_acc),
        "prediction_mean": float(np.mean(pred_arr)),
        "actual_mean": float(np.mean(y_arr)),
    }


def strategy_backtest_report(
    data: pd.DataFrame,
    split: int,
    strategy_definitions: list[dict[str, str]],
    horizon: int,
) -> dict[str, Any]:
    test = data.iloc[split:].copy()
    reports: dict[str, Any] = {}
    annualizer = float(np.sqrt(252 / max(horizon, 1)))
    total_rows = max(len(test), 1)

    for definition in strategy_definitions:
        sid = definition["id"]
        setup_col = definition["setup"]
        target_col = definition["target"]
        return_col = f"target_strategy_{sid}_return_pct"
        mfe_col = f"target_strategy_{sid}_mfe_pct"
        mae_col = f"target_strategy_{sid}_mae_pct"
        rr_col = f"target_strategy_{sid}_risk_reward"
        if setup_col not in test or target_col not in test or return_col not in test:
            reports[sid] = {"skipped": True, "reason": "Strategy outcome columns missing."}
            continue

        active = test[test[setup_col].astype(int) == 1]
        if active.empty:
            reports[sid] = {
                "skipped": True,
                "reason": "No active strategy setups in holdout split.",
                "signal_count": 0,
                "coverage": 0.0,
            }
            continue

        returns = active[return_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1 if not equity.empty else pd.Series(dtype=float)
        sharpe = None
        if len(returns) > 1 and float(returns.std()) > 0:
            sharpe = float((returns.mean() / returns.std()) * annualizer)

        reports[sid] = {
            "label": definition.get("label", sid),
            "direction": definition.get("direction"),
            "signal_count": int(len(active)),
            "coverage": float(len(active) / total_rows),
            "win_rate": float(active[target_col].mean()),
            "avg_return": float(returns.mean()) if not returns.empty else None,
            "median_return": float(returns.median()) if not returns.empty else None,
            "avg_win": float(wins.mean()) if not wins.empty else None,
            "avg_loss": float(losses.mean()) if not losses.empty else None,
            "profit_factor": None if gross_loss == 0 else float(gross_profit / gross_loss),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else None,
            "sharpe": sharpe,
            "avg_mfe": float(active[mfe_col].mean()) if mfe_col in active else None,
            "avg_mae": float(active[mae_col].mean()) if mae_col in active else None,
            "avg_risk_reward": float(active[rr_col].mean()) if rr_col in active else None,
            "target_hit_rate": float(active.get(f"target_strategy_{sid}_target_hit", pd.Series(index=active.index, dtype=float)).mean()),
            "stop_hit_rate": float(active.get(f"target_strategy_{sid}_stop_hit", pd.Series(index=active.index, dtype=float)).mean()),
        }

    return reports


def _train_setup_only_classifier(
    *,
    target_name: str,
    setup_col: str,
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    split: int,
    base_params: dict,
    use_gpu: bool,
    min_train: int = 50,
    min_test: int = 20,
) -> tuple[Any | None, dict[str, Any]]:
    active_mask = data[setup_col].astype(int) == 1
    train_mask = active_mask.iloc[:split]
    test_mask = active_mask.iloc[split:]
    y_aux = data[target_name].astype(int)
    y_aux_train = y_aux.iloc[:split][train_mask.to_numpy()]
    y_aux_test = y_aux.iloc[split:][test_mask.to_numpy()]
    X_strategy_train = X_train[train_mask.to_numpy()]
    X_strategy_test = X_test[test_mask.to_numpy()]
    if len(y_aux_train) < min_train or len(y_aux_test) < min_test or y_aux_train.nunique() < 2 or y_aux_test.nunique() < 2:
        return None, {
            "skipped": True,
            "reason": "Not enough active setup rows or class variety in chronological train/test split.",
            "setup": setup_col,
            "train_active_rows": int(len(y_aux_train)),
            "test_active_rows": int(len(y_aux_test)),
            "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
            "train_counts": y_aux_train.value_counts().sort_index().to_dict(),
            "test_counts": y_aux_test.value_counts().sort_index().to_dict(),
        }
    model = make_xgb_binary(base_params, use_gpu)
    model.fit(X_strategy_train, y_aux_train)
    model = make_cpu_portable_model(model, "xgboost")
    proba = model.predict_proba(X_strategy_test)
    report = binary_model_report(y_aux_test, proba)
    report.update({
        "setup": setup_col,
        "train_active_rows": int(len(y_aux_train)),
        "test_active_rows": int(len(y_aux_test)),
        "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
        "baseline_win_rate_test": float(y_aux_test.mean()),
    })
    return model, report


def _train_setup_only_regressor(
    *,
    target_name: str,
    setup_col: str,
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    split: int,
    base_params: dict,
    use_gpu: bool,
    min_train: int = 50,
    min_test: int = 20,
) -> tuple[Any | None, dict[str, Any]]:
    active_mask = data[setup_col].astype(int) == 1
    train_mask = active_mask.iloc[:split]
    test_mask = active_mask.iloc[split:]
    y_aux = data[target_name].astype(float)
    y_aux_train = y_aux.iloc[:split][train_mask.to_numpy()]
    y_aux_test = y_aux.iloc[split:][test_mask.to_numpy()]
    X_strategy_train = X_train[train_mask.to_numpy()]
    X_strategy_test = X_test[test_mask.to_numpy()]
    if len(y_aux_train) < min_train or len(y_aux_test) < min_test or y_aux_train.nunique() < 3:
        return None, {
            "skipped": True,
            "reason": "Not enough active setup rows or target variation in chronological train/test split.",
            "setup": setup_col,
            "train_active_rows": int(len(y_aux_train)),
            "test_active_rows": int(len(y_aux_test)),
            "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
        }
    model = make_xgb_regressor(base_params, use_gpu)
    model.fit(X_strategy_train, y_aux_train)
    model = make_cpu_portable_model(model, "xgboost")
    preds = model.predict(X_strategy_test)
    report = regression_model_report(y_aux_test, preds)
    report.update({
        "setup": setup_col,
        "train_active_rows": int(len(y_aux_train)),
        "test_active_rows": int(len(y_aux_test)),
        "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
    })
    return model, report


def _train_breakout_precision_variants(
    *,
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    split: int,
    base_params: dict,
    use_gpu: bool,
    class_weights: list[float],
    min_signals: int,
) -> tuple[Any | None, dict[str, Any]]:
    target_name = "target_elite_breakout_long_quality_success"
    setup_col = "setup_breakout_long"
    active_mask = data[setup_col].astype(int) == 1
    train_mask = active_mask.iloc[:split]
    test_mask = active_mask.iloc[split:]
    y_train = data[target_name].astype(int).iloc[:split][train_mask.to_numpy()]
    y_test = data[target_name].astype(int).iloc[split:][test_mask.to_numpy()]
    Xb_train = X_train[train_mask.to_numpy()]
    Xb_test = X_test[test_mask.to_numpy()]

    if len(y_train) < 50 or len(y_test) < 20 or y_train.nunique() < 2 or y_test.nunique() < 2:
        return None, {
            "skipped": True,
            "reason": "Not enough breakout elite rows or class variety.",
            "train_active_rows": int(len(y_train)),
            "test_active_rows": int(len(y_test)),
            "train_counts": y_train.value_counts().sort_index().to_dict(),
            "test_counts": y_test.value_counts().sort_index().to_dict(),
        }

    best_model = None
    best_selection: dict[str, Any] | None = None
    variant_reports: dict[str, Any] = {}

    for weight in class_weights:
        params = base_params.copy()
        params["scale_pos_weight"] = float(weight)
        model = make_xgb_binary(params, use_gpu)
        model.fit(Xb_train, y_train)
        model = make_cpu_portable_model(model, "xgboost")
        proba = model.predict_proba(Xb_test)
        report = binary_model_report(y_test, proba)
        positive_buckets = positive_probability_bucket_report(y_test, proba)
        report.update({
            "scale_pos_weight": float(weight),
            "positive_probability_buckets": positive_buckets,
            "train_active_rows": int(len(y_train)),
            "test_active_rows": int(len(y_test)),
            "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
            "baseline_win_rate_test": float(y_test.mean()),
        })
        variant_reports[f"scale_pos_weight_{weight:g}"] = report

        for bucket, stats in positive_buckets.items():
            precision = stats.get("precision")
            signals = int(stats.get("signals") or 0)
            if precision is None or signals < min_signals:
                continue
            candidate = {
                "scale_pos_weight": float(weight),
                "bucket": bucket,
                "precision": float(precision),
                "signals": signals,
                "coverage": float(stats.get("coverage") or 0.0),
                "recall": float(stats.get("recall") or 0.0),
                "model_key": f"scale_pos_weight_{weight:g}",
            }
            if (
                best_selection is None
                or candidate["precision"] > best_selection["precision"]
                or (
                    candidate["precision"] == best_selection["precision"]
                    and candidate["signals"] > best_selection["signals"]
                )
            ):
                best_selection = candidate
                best_model = model

    return best_model, {
        "target": target_name,
        "setup": setup_col,
        "optimization_goal": "maximize positive-class precision for TAKE-breakout signals while disclosing confidence threshold and coverage",
        "minimum_signals_required": int(min_signals),
        "best_selection": best_selection,
        "variants": variant_reports,
        "proof_rule": "Only claim precision for the selected positive-probability bucket, with signal count and setup-filtered coverage disclosed.",
    }


def build_final_hybrid_policy(config: dict, auxiliary_metrics: dict[str, Any]) -> dict[str, Any]:
    promoted_signal = config.get("final_promoted_signal", "target_elite_breakout_long_quality_success")
    elite_metrics = auxiliary_metrics.get("elite_trade_quality", {})
    breakout_precision = auxiliary_metrics.get("breakout_precision", {}).get("target_breakout_precision_take", {})
    promoted_report = elite_metrics.get(promoted_signal, {})
    precision_selection = breakout_precision.get("best_selection")

    return {
        "name": "MarketVision AI Final Hybrid v12",
        "promoted_signal": promoted_signal,
        "promoted_signal_report": promoted_report,
        "breakout_precision_experiment": {
            "target": "target_breakout_precision_take",
            "best_selection": precision_selection,
            "role": "experimental_secondary_evidence_not_promoted",
        },
        "combined_layers": [
            "3-class market regime probability",
            "binary direction confidence proof",
            "breakout/breakdown/reversal/volatility auxiliary models",
            "future return and future range regressors",
            "strategy quality backtest metrics",
            "specialist breakout models",
            "elite breakout quality model",
            "breakout precision variants",
        ],
        "promotion_reason": (
            "v10-style elite breakout quality produced the closest honest high-confidence result; "
            "v11 class-weight precision variants did not improve positive precision enough to promote."
        ),
        "accuracy_claim_policy": (
            "Do not claim 75%+ unless a reported confidence bucket reaches >=75% accuracy/precision "
            "and includes signal count plus setup-filtered coverage."
        ),
        "disclaimer": config.get("final_model_policy", {}).get(
            "disclaimer",
            "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
        ),
    }


# ── 9. SHAP feature selection ─────────────────────────────────────────────────

def shap_top_features(model, X: pd.DataFrame, top_n: int) -> list[str]:
    print(f"\n── SHAP feature selection (top {top_n}) ──────────────")
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.iloc[:2000])
        if isinstance(sv, list):
            importance = np.mean([np.abs(s).mean(0) for s in sv], axis=0)
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:
                importance = np.abs(arr).mean(axis=(0, 2))
            else:
                importance = np.abs(arr).mean(0)
        importance = np.asarray(importance).reshape(-1)
        ranking = pd.Series(importance, index=X.columns).sort_values(ascending=False)
        top = ranking.head(top_n).index.tolist()
        print("Top features:", top[:10], "...")
        return top
    except Exception as e:
        print(f"SHAP failed ({e}), keeping all features")
        return X.columns.tolist()


# ── 10. Main training pipeline ────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("MarketVision AI — v12 Final Hybrid Model Training")
    print("=" * 65)
    print("\n── GPU visibility ─────────────────────────────────────")
    subprocess.run(["nvidia-smi"], check=False)

    data, feature_cols = build_dataset(CONFIG)

    dataset_path = OUTPUT_DIR / "training_dataset.parquet"
    data.to_parquet(dataset_path)
    print(f"\nDataset saved → {dataset_path}")

    X_full = data[feature_cols]
    y_full = data["target"].astype(int)
    use_gpu = CONFIG["use_gpu"]
    n_trials = CONFIG["n_trials"]

    # ── Phase 1: Tune XGBoost, get feature importance via SHAP
    print(f"\n── Phase 1: Tune XGBoost ({n_trials} trials) ─────────────────")
    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(
        lambda t: cv_score(X_full, y_full, make_xgb, t, use_gpu),
        n_trials=n_trials, show_progress_bar=True,
    )
    print(f"XGB best CV F1: {xgb_study.best_value:.4f}")

    # SHAP prune on quick XGB model
    quick_xgb = make_xgb(None, use_gpu, xgb_best_params(xgb_study))
    split = int(len(X_full) * 0.8)
    quick_xgb.fit(X_full.iloc[:split], y_full.iloc[:split])
    selected_features = shap_top_features(quick_xgb, X_full.iloc[:split], CONFIG["shap_top_n_features"])

    X = X_full[selected_features]
    y = y_full

    # ── Phase 2: Retune all three models on pruned features
    print(f"\n── Phase 2: Retune XGB on pruned features ({n_trials} trials) ─")
    xgb_study2 = optuna.create_study(direction="maximize")
    xgb_study2.optimize(
        lambda t: cv_score(X, y, make_xgb, t, use_gpu),
        n_trials=n_trials, show_progress_bar=True,
    )
    print(f"XGB (pruned) best CV F1: {xgb_study2.best_value:.4f}")

    lgb_study = None
    if CONFIG["enable_lightgbm"]:
        print(f"\n── Phase 3: Tune LightGBM ({n_trials} trials) ─────────────────")
        lgb_study = optuna.create_study(direction="maximize")
        lgb_study.optimize(
            lambda t: cv_score(X, y, make_lgb, t, use_gpu),
            n_trials=n_trials, show_progress_bar=True,
        )
        print(f"LGB best CV F1: {lgb_study.best_value:.4f}")
    else:
        print("\n── Phase 3: LightGBM disabled for warning-free stable artifact ─")

    cat_study = None
    if CONFIG["enable_catboost"]:
        print(f"\n── Phase 4: Tune CatBoost ({n_trials} trials) ──────────────────")
        cat_study = optuna.create_study(direction="maximize")
        cat_study.optimize(
            lambda t: cv_score(X, y, make_cat, t, use_gpu),
            n_trials=n_trials, show_progress_bar=True,
        )
        print(f"CAT best CV F1: {cat_study.best_value:.4f}")
    else:
        print("\n── Phase 4: CatBoost disabled for stable Kaggle GPU artifact ───")

    # ── Phase 5: Train final ensemble
    print("\n── Phase 5: Train final ensemble on full data ──────────────")
    split = int(len(X) * 0.82)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    xgb_final = make_xgb(None, use_gpu, xgb_best_params(xgb_study2))
    lgb_final = make_lgb(None, use_gpu, lgb_best_params(lgb_study)) if lgb_study is not None else None
    cat_final = make_cat(None, use_gpu, cat_best_params(cat_study)) if cat_study is not None else None

    xgb_final.fit(X_train, y_train)
    if lgb_final is not None:
        lgb_final.fit(X_train, y_train)
    if cat_final is not None:
        cat_final.fit(X_train, y_train)

    # Make artifacts load cleanly on CPU-only machines after GPU training.
    xgb_final = make_cpu_portable_model(xgb_final, "xgboost")
    if lgb_final is not None:
        lgb_final = make_cpu_portable_model(lgb_final, "lightgbm")
    if cat_final is not None:
        cat_final = make_cpu_portable_model(cat_final, "catboost")

    # Soft voting ensemble
    final_models = {
        "xgboost": xgb_final,
    }
    if lgb_final is not None:
        final_models["lightgbm"] = lgb_final
    if cat_final is not None:
        final_models["catboost"] = cat_final
    probas = [model.predict_proba(X_test) for model in final_models.values()]
    ensemble_proba = np.mean(probas, axis=0)
    ensemble_preds = ensemble_proba.argmax(axis=1)

    acc = accuracy_score(y_test, ensemble_preds)
    f1  = f1_score(y_test, ensemble_preds, average="macro")
    print(f"\nEnsemble holdout accuracy: {acc:.4f}")
    print(f"Ensemble holdout F1-macro: {f1:.4f}")

    # Individual scores
    for name, m in [("XGB", xgb_final), ("LGB", lgb_final), ("CAT", cat_final)]:
        if m is None:
            continue
        p = m.predict(X_test)
        print(f"  {name} accuracy={accuracy_score(y_test,p):.4f} f1={f1_score(y_test,p,average='macro'):.4f}")

    multiclass_buckets = confidence_bucket_report(y_test, ensemble_proba)

    # ── Phase 6: Train direction model for honest high-confidence proof
    # This does not replace the app's 3-class model. It reports whether the
    # system can reach stronger accuracy when it only acts on confident signals.
    print("\n── Phase 6: Train binary direction model for confidence proof ─")
    y_binary = data["target_binary"].astype(int)
    yb_train, yb_test = y_binary.iloc[:split], y_binary.iloc[split:]

    binary_final = make_xgb_binary(xgb_best_params(xgb_study2), use_gpu)
    binary_final.fit(X_train, yb_train)
    binary_final = make_cpu_portable_model(binary_final, "xgboost")
    binary_proba = binary_final.predict_proba(X_test)
    binary_preds = binary_proba.argmax(axis=1)
    binary_acc = accuracy_score(yb_test, binary_preds)
    binary_f1 = f1_score(yb_test, binary_preds)
    binary_precision = precision_score(yb_test, binary_preds, zero_division=0)
    binary_recall = recall_score(yb_test, binary_preds, zero_division=0)
    binary_buckets = confidence_bucket_report(yb_test, binary_proba)

    y_return_test = data["target_return_horizon"].iloc[split:]
    median_abs_train_return = float(data["target_return_horizon"].iloc[:split].abs().median())
    predicted_return_proxy = (ensemble_proba[:, 2] - ensemble_proba[:, 0]) * median_abs_train_return
    return_mae = mean_absolute_error(y_return_test, predicted_return_proxy)
    return_rmse = float(np.sqrt(mean_squared_error(y_return_test, predicted_return_proxy)))

    print(f"Binary direction holdout accuracy: {binary_acc:.4f}")
    print(f"Binary direction holdout F1:       {binary_f1:.4f}")
    for bucket, stats in binary_buckets.items():
        acc_value = stats["accuracy"]
        acc_text = "n/a" if acc_value is None else f"{acc_value:.4f}"
        print(f"  confidence {bucket}: accuracy={acc_text} coverage={stats['coverage']:.2%} signals={stats['signals']}")

    # ── Phase 7: Train auxiliary event/range models
    print("\n── Phase 7: Train auxiliary market-intelligence targets ─")
    strategy_definitions = CONFIG.get("strategy_definitions", [])
    strategy_target_names = [definition["target"] for definition in strategy_definitions]
    strategy_return_targets = [f"target_strategy_{definition['id']}_return_pct" for definition in strategy_definitions]
    strategy_risk_reward_targets = [f"target_strategy_{definition['id']}_risk_reward" for definition in strategy_definitions]
    aux_classification_targets = [
        "target_breakout",
        "target_breakdown",
        "target_reversal",
        "target_volatility_expansion",
    ]
    aux_regression_targets = [
        "target_return_horizon",
        "target_future_range_high_pct",
        "target_future_range_low_pct",
        "target_risk_adjusted_return",
    ]
    auxiliary_models: dict[str, Any] = {}
    auxiliary_metrics: dict[str, Any] = {
        "classification": {},
        "regression": {},
        "strategy_classification": {},
        "strategy_regression": {},
        "strategy_opportunity": {},
        "specialist_strategies": {},
        "elite_trade_quality": {},
        "breakout_precision": {},
    }
    base_params = xgb_best_params(xgb_study2)

    for target_name in aux_classification_targets:
        y_aux = data[target_name].astype(int)
        y_aux_train = y_aux.iloc[:split]
        y_aux_test = y_aux.iloc[split:]
        if y_aux_train.nunique() < 2 or y_aux_test.nunique() < 2:
            auxiliary_metrics["classification"][target_name] = {
                "skipped": True,
                "reason": "Target has fewer than two classes in train or test split.",
                "train_counts": y_aux_train.value_counts().sort_index().to_dict(),
                "test_counts": y_aux_test.value_counts().sort_index().to_dict(),
            }
            print(f"  {target_name}: skipped, insufficient class variety")
            continue
        model = make_xgb_binary(base_params, use_gpu)
        model.fit(X_train, y_aux_train)
        model = make_cpu_portable_model(model, "xgboost")
        proba = model.predict_proba(X_test)
        report = binary_model_report(y_aux_test, proba)
        auxiliary_models[target_name] = model
        auxiliary_metrics["classification"][target_name] = report
        print(f"  {target_name}: accuracy={report['accuracy']:.4f} f1={report['f1']:.4f}")

    for definition in strategy_definitions:
        target_name = definition["target"]
        setup_col = definition["setup"]
        active_mask = data[setup_col].astype(int) == 1
        train_mask = active_mask.iloc[:split]
        test_mask = active_mask.iloc[split:]
        y_aux = data[target_name].astype(int)
        y_aux_train = y_aux.iloc[:split][train_mask.to_numpy()]
        y_aux_test = y_aux.iloc[split:][test_mask.to_numpy()]
        X_strategy_train = X_train[train_mask.to_numpy()]
        X_strategy_test = X_test[test_mask.to_numpy()]
        train_counts = y_aux_train.value_counts().sort_index().to_dict()
        test_counts = y_aux_test.value_counts().sort_index().to_dict()
        if len(y_aux_train) < 50 or len(y_aux_test) < 20 or y_aux_train.nunique() < 2 or y_aux_test.nunique() < 2:
            auxiliary_metrics["strategy_classification"][target_name] = {
                "skipped": True,
                "reason": "Not enough active setup rows or class variety in chronological train/test split.",
                "strategy_id": definition["id"],
                "setup": setup_col,
                "train_active_rows": int(len(y_aux_train)),
                "test_active_rows": int(len(y_aux_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                "train_counts": train_counts,
                "test_counts": test_counts,
            }
            print(f"  {target_name}: skipped setup-only model, active train={len(y_aux_train)} test={len(y_aux_test)}")
            continue
        model = make_xgb_binary(base_params, use_gpu)
        model.fit(X_strategy_train, y_aux_train)
        model = make_cpu_portable_model(model, "xgboost")
        proba = model.predict_proba(X_strategy_test)
        report = binary_model_report(y_aux_test, proba)
        report.update({
            "strategy_id": definition["id"],
            "setup": setup_col,
            "train_active_rows": int(len(y_aux_train)),
            "test_active_rows": int(len(y_aux_test)),
            "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
            "baseline_win_rate_test": float(y_aux_test.mean()),
        })
        auxiliary_models[target_name] = model
        auxiliary_metrics["strategy_classification"][target_name] = report
        print(
            f"  {target_name}: setup-only accuracy={report['accuracy']:.4f} "
            f"f1={report['f1']:.4f} active_test={len(y_aux_test)}"
        )

    for target_name in aux_regression_targets:
        y_aux = data[target_name].astype(float)
        y_aux_train = y_aux.iloc[:split]
        y_aux_test = y_aux.iloc[split:]
        model = make_xgb_regressor(base_params, use_gpu)
        model.fit(X_train, y_aux_train)
        model = make_cpu_portable_model(model, "xgboost")
        preds = model.predict(X_test)
        report = regression_model_report(y_aux_test, preds)
        auxiliary_models[target_name] = model
        auxiliary_metrics["regression"][target_name] = report
        print(f"  {target_name}: mae={report['mae']:.5f} rmse={report['rmse']:.5f} dir_acc={report['directional_accuracy']:.4f}")

    for definition in strategy_definitions:
        setup_col = definition["setup"]
        active_mask = data[setup_col].astype(int) == 1
        train_mask = active_mask.iloc[:split]
        test_mask = active_mask.iloc[split:]
        for target_name in [
            f"target_strategy_{definition['id']}_return_pct",
            f"target_strategy_{definition['id']}_risk_reward",
        ]:
            if target_name not in data:
                continue
            y_aux = data[target_name].astype(float)
            y_aux_train = y_aux.iloc[:split][train_mask.to_numpy()]
            y_aux_test = y_aux.iloc[split:][test_mask.to_numpy()]
            X_strategy_train = X_train[train_mask.to_numpy()]
            X_strategy_test = X_test[test_mask.to_numpy()]
            if len(y_aux_train) < 50 or len(y_aux_test) < 20 or y_aux_train.nunique() < 3:
                auxiliary_metrics["strategy_regression"][target_name] = {
                    "skipped": True,
                    "reason": "Not enough active setup rows or target variation in chronological train/test split.",
                    "strategy_id": definition["id"],
                    "setup": setup_col,
                    "train_active_rows": int(len(y_aux_train)),
                    "test_active_rows": int(len(y_aux_test)),
                    "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                }
                print(f"  {target_name}: skipped setup-only regressor, active train={len(y_aux_train)} test={len(y_aux_test)}")
                continue
            model = make_xgb_regressor(base_params, use_gpu)
            model.fit(X_strategy_train, y_aux_train)
            model = make_cpu_portable_model(model, "xgboost")
            preds = model.predict(X_strategy_test)
            report = regression_model_report(y_aux_test, preds)
            report.update({
                "strategy_id": definition["id"],
                "setup": setup_col,
                "train_active_rows": int(len(y_aux_train)),
                "test_active_rows": int(len(y_aux_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
            })
            auxiliary_models[target_name] = model
            auxiliary_metrics["strategy_regression"][target_name] = report
            print(
                f"  {target_name}: setup-only mae={report['mae']:.5f} "
                f"rmse={report['rmse']:.5f} active_test={len(y_aux_test)}"
            )

    strategy_quality = strategy_backtest_report(
        data=data,
        split=split,
        strategy_definitions=strategy_definitions,
        horizon=CONFIG["prediction_horizon"],
    )

    opportunity_classification_targets: list[str] = []
    opportunity_regression_targets: list[str] = []
    specialist_classification_targets: list[str] = []
    specialist_regression_targets: list[str] = []
    elite_classification_targets: list[str] = []
    elite_regression_targets: list[str] = []
    breakout_precision_targets: list[str] = []
    opportunity_active = "target_high_quality_strategy_active"
    opportunity_target = "target_high_quality_strategy_success"
    opportunity_return = "target_high_quality_strategy_return_pct"
    opportunity_risk_reward = "target_high_quality_strategy_risk_reward"
    if opportunity_active in data and opportunity_target in data:
        print("\n── Phase 8: Train strategy-filtered opportunity model ─")
        active_mask = data[opportunity_active].astype(int) == 1
        train_mask = active_mask.iloc[:split]
        test_mask = active_mask.iloc[split:]
        y_opp = data[opportunity_target].astype(int)
        y_opp_train = y_opp.iloc[:split][train_mask.to_numpy()]
        y_opp_test = y_opp.iloc[split:][test_mask.to_numpy()]
        X_opp_train = X_train[train_mask.to_numpy()]
        X_opp_test = X_test[test_mask.to_numpy()]
        if len(y_opp_train) >= 100 and len(y_opp_test) >= 50 and y_opp_train.nunique() >= 2 and y_opp_test.nunique() >= 2:
            model = make_xgb_binary(base_params, use_gpu)
            model.fit(X_opp_train, y_opp_train)
            model = make_cpu_portable_model(model, "xgboost")
            proba = model.predict_proba(X_opp_test)
            report = binary_model_report(y_opp_test, proba)
            report.update({
                "target": opportunity_target,
                "active_filter": opportunity_active,
                "included_strategy_ids": CONFIG.get("high_quality_strategy_ids", []),
                "train_active_rows": int(len(y_opp_train)),
                "test_active_rows": int(len(y_opp_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                "baseline_win_rate_test": float(y_opp_test.mean()),
                "proof_rule": "Only claim high accuracy for disclosed confidence buckets and setup-filtered coverage.",
            })
            auxiliary_models[opportunity_target] = model
            auxiliary_metrics["strategy_opportunity"][opportunity_target] = report
            opportunity_classification_targets.append(opportunity_target)
            print(
                f"  {opportunity_target}: filtered accuracy={report['accuracy']:.4f} "
                f"f1={report['f1']:.4f} active_test={len(y_opp_test)}"
            )
        else:
            auxiliary_metrics["strategy_opportunity"][opportunity_target] = {
                "skipped": True,
                "reason": "Not enough high-quality active setup rows or class variety in chronological split.",
                "train_active_rows": int(len(y_opp_train)),
                "test_active_rows": int(len(y_opp_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                "train_counts": y_opp_train.value_counts().sort_index().to_dict(),
                "test_counts": y_opp_test.value_counts().sort_index().to_dict(),
            }
            print(f"  {opportunity_target}: skipped, active train={len(y_opp_train)} test={len(y_opp_test)}")

        for target_name in [opportunity_return, opportunity_risk_reward]:
            if target_name not in data:
                continue
            y_opp_reg = data[target_name].astype(float)
            y_opp_reg_train = y_opp_reg.iloc[:split][train_mask.to_numpy()]
            y_opp_reg_test = y_opp_reg.iloc[split:][test_mask.to_numpy()]
            if len(y_opp_reg_train) < 100 or len(y_opp_reg_test) < 50 or y_opp_reg_train.nunique() < 3:
                auxiliary_metrics["strategy_opportunity"][target_name] = {
                    "skipped": True,
                    "reason": "Not enough high-quality active setup rows or target variation in chronological split.",
                    "train_active_rows": int(len(y_opp_reg_train)),
                    "test_active_rows": int(len(y_opp_reg_test)),
                    "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                }
                continue
            model = make_xgb_regressor(base_params, use_gpu)
            model.fit(X_opp_train, y_opp_reg_train)
            model = make_cpu_portable_model(model, "xgboost")
            preds = model.predict(X_opp_test)
            report = regression_model_report(y_opp_reg_test, preds)
            report.update({
                "target": target_name,
                "active_filter": opportunity_active,
                "included_strategy_ids": CONFIG.get("high_quality_strategy_ids", []),
                "train_active_rows": int(len(y_opp_reg_train)),
                "test_active_rows": int(len(y_opp_reg_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
            })
            auxiliary_models[target_name] = model
            auxiliary_metrics["strategy_opportunity"][target_name] = report
            opportunity_regression_targets.append(target_name)
            print(f"  {target_name}: filtered mae={report['mae']:.5f} rmse={report['rmse']:.5f}")

    specialist_ids = set(CONFIG.get("specialist_strategy_ids", []))
    specialist_definitions = [definition for definition in strategy_definitions if definition["id"] in specialist_ids]
    if specialist_definitions:
        print("\n── Phase 9: Train specialist strategy models ─")
    for definition in specialist_definitions:
        sid = definition["id"]
        setup_col = definition["setup"]
        specialist_success_name = f"target_specialist_{sid}_success"
        specialist_return_name = f"target_specialist_{sid}_return_pct"
        specialist_rr_name = f"target_specialist_{sid}_risk_reward"

        model, report = _train_setup_only_classifier(
            target_name=definition["target"],
            setup_col=setup_col,
            data=data,
            X_train=X_train,
            X_test=X_test,
            split=split,
            base_params=base_params,
            use_gpu=use_gpu,
            min_train=50,
            min_test=20,
        )
        report.update({
            "strategy_id": sid,
            "label": definition.get("label", sid),
            "source_target": definition["target"],
            "specialist_target": specialist_success_name,
            "strategy_quality": strategy_quality.get(sid, {}),
            "proof_rule": "Specialist accuracy is only valid on this setup's active rows and must disclose confidence threshold plus coverage.",
        })
        auxiliary_metrics["specialist_strategies"][specialist_success_name] = report
        if model is not None:
            auxiliary_models[specialist_success_name] = model
            specialist_classification_targets.append(specialist_success_name)
            print(
                f"  {specialist_success_name}: accuracy={report['accuracy']:.4f} "
                f"f1={report['f1']:.4f} active_test={report['test_active_rows']}"
            )
        else:
            print(f"  {specialist_success_name}: skipped, active train={report['train_active_rows']} test={report['test_active_rows']}")

        for source_target, specialist_name in [
            (f"target_strategy_{sid}_return_pct", specialist_return_name),
            (f"target_strategy_{sid}_risk_reward", specialist_rr_name),
        ]:
            if source_target not in data:
                continue
            model, report = _train_setup_only_regressor(
                target_name=source_target,
                setup_col=setup_col,
                data=data,
                X_train=X_train,
                X_test=X_test,
                split=split,
                base_params=base_params,
                use_gpu=use_gpu,
                min_train=50,
                min_test=20,
            )
            report.update({
                "strategy_id": sid,
                "label": definition.get("label", sid),
                "source_target": source_target,
                "specialist_target": specialist_name,
                "strategy_quality": strategy_quality.get(sid, {}),
            })
            auxiliary_metrics["specialist_strategies"][specialist_name] = report
            if model is not None:
                auxiliary_models[specialist_name] = model
                specialist_regression_targets.append(specialist_name)
                print(
                    f"  {specialist_name}: mae={report['mae']:.5f} "
                    f"rmse={report['rmse']:.5f} active_test={report['test_active_rows']}"
                )
            else:
                print(f"  {specialist_name}: skipped, active train={report['train_active_rows']} test={report['test_active_rows']}")

    elite_ids = set(CONFIG.get("elite_strategy_ids", []))
    elite_definitions = [definition for definition in strategy_definitions if definition["id"] in elite_ids]
    if elite_definitions:
        print("\n── Phase 10: Train elite trade-quality models ─")
    for definition in elite_definitions:
        sid = definition["id"]
        setup_col = definition["setup"]
        elite_success_name = f"target_elite_{sid}_quality_success"
        elite_return_positive_name = f"target_elite_{sid}_return_positive"
        elite_return_name = f"target_elite_{sid}_return_pct"
        elite_rr_name = f"target_elite_{sid}_risk_reward"

        for source_target, public_name in [
            (elite_success_name, elite_success_name),
            (elite_return_positive_name, elite_return_positive_name),
        ]:
            if source_target not in data:
                continue
            model, report = _train_setup_only_classifier(
                target_name=source_target,
                setup_col=setup_col,
                data=data,
                X_train=X_train,
                X_test=X_test,
                split=split,
                base_params=base_params,
                use_gpu=use_gpu,
                min_train=50,
                min_test=20,
            )
            report.update({
                "strategy_id": sid,
                "label": definition.get("label", sid),
                "source_target": source_target,
                "elite_target": public_name,
                "minimum_return_pct": CONFIG.get("elite_min_return_pct"),
                "minimum_risk_reward": CONFIG.get("elite_min_risk_reward"),
                "strategy_quality": strategy_quality.get(sid, {}),
                "proof_rule": "Elite trade-quality accuracy is valid only on active setup rows and must disclose confidence threshold plus coverage.",
            })
            auxiliary_metrics["elite_trade_quality"][public_name] = report
            if model is not None:
                auxiliary_models[public_name] = model
                elite_classification_targets.append(public_name)
                print(
                    f"  {public_name}: accuracy={report['accuracy']:.4f} "
                    f"f1={report['f1']:.4f} active_test={report['test_active_rows']}"
                )
            else:
                print(f"  {public_name}: skipped, active train={report['train_active_rows']} test={report['test_active_rows']}")

        for source_target, public_name in [
            (f"target_strategy_{sid}_return_pct", elite_return_name),
            (f"target_strategy_{sid}_risk_reward", elite_rr_name),
        ]:
            if source_target not in data:
                continue
            model, report = _train_setup_only_regressor(
                target_name=source_target,
                setup_col=setup_col,
                data=data,
                X_train=X_train,
                X_test=X_test,
                split=split,
                base_params=base_params,
                use_gpu=use_gpu,
                min_train=50,
                min_test=20,
            )
            report.update({
                "strategy_id": sid,
                "label": definition.get("label", sid),
                "source_target": source_target,
                "elite_target": public_name,
                "minimum_return_pct": CONFIG.get("elite_min_return_pct"),
                "minimum_risk_reward": CONFIG.get("elite_min_risk_reward"),
                "strategy_quality": strategy_quality.get(sid, {}),
            })
            auxiliary_metrics["elite_trade_quality"][public_name] = report
            if model is not None:
                auxiliary_models[public_name] = model
                elite_regression_targets.append(public_name)
                print(
                    f"  {public_name}: mae={report['mae']:.5f} "
                    f"rmse={report['rmse']:.5f} active_test={report['test_active_rows']}"
                )
            else:
                print(f"  {public_name}: skipped, active train={report['train_active_rows']} test={report['test_active_rows']}")

    elite_active = "target_elite_strategy_active"
    elite_combined_success = "target_elite_strategy_quality_success"
    elite_combined_return = "target_elite_strategy_return_pct"
    elite_combined_rr = "target_elite_strategy_risk_reward"
    if elite_active in data and elite_combined_success in data:
        print("\n── Phase 11: Train combined elite opportunity model ─")
        active_mask = data[elite_active].astype(int) == 1
        train_mask = active_mask.iloc[:split]
        test_mask = active_mask.iloc[split:]
        y_elite = data[elite_combined_success].astype(int)
        y_elite_train = y_elite.iloc[:split][train_mask.to_numpy()]
        y_elite_test = y_elite.iloc[split:][test_mask.to_numpy()]
        X_elite_train = X_train[train_mask.to_numpy()]
        X_elite_test = X_test[test_mask.to_numpy()]
        if len(y_elite_train) >= 100 and len(y_elite_test) >= 50 and y_elite_train.nunique() >= 2 and y_elite_test.nunique() >= 2:
            model = make_xgb_binary(base_params, use_gpu)
            model.fit(X_elite_train, y_elite_train)
            model = make_cpu_portable_model(model, "xgboost")
            proba = model.predict_proba(X_elite_test)
            report = binary_model_report(y_elite_test, proba)
            report.update({
                "target": elite_combined_success,
                "active_filter": elite_active,
                "included_strategy_ids": CONFIG.get("elite_strategy_ids", []),
                "train_active_rows": int(len(y_elite_train)),
                "test_active_rows": int(len(y_elite_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                "baseline_win_rate_test": float(y_elite_test.mean()),
                "minimum_return_pct": CONFIG.get("elite_min_return_pct"),
                "minimum_risk_reward": CONFIG.get("elite_min_risk_reward"),
            })
            auxiliary_models[elite_combined_success] = model
            auxiliary_metrics["elite_trade_quality"][elite_combined_success] = report
            elite_classification_targets.append(elite_combined_success)
            print(
                f"  {elite_combined_success}: accuracy={report['accuracy']:.4f} "
                f"f1={report['f1']:.4f} active_test={len(y_elite_test)}"
            )
        else:
            auxiliary_metrics["elite_trade_quality"][elite_combined_success] = {
                "skipped": True,
                "reason": "Not enough combined elite setup rows or class variety in chronological split.",
                "train_active_rows": int(len(y_elite_train)),
                "test_active_rows": int(len(y_elite_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                "train_counts": y_elite_train.value_counts().sort_index().to_dict(),
                "test_counts": y_elite_test.value_counts().sort_index().to_dict(),
            }
            print(f"  {elite_combined_success}: skipped, active train={len(y_elite_train)} test={len(y_elite_test)}")

        for target_name in [elite_combined_return, elite_combined_rr]:
            if target_name not in data:
                continue
            y_reg = data[target_name].astype(float)
            y_reg_train = y_reg.iloc[:split][train_mask.to_numpy()]
            y_reg_test = y_reg.iloc[split:][test_mask.to_numpy()]
            if len(y_reg_train) < 100 or len(y_reg_test) < 50 or y_reg_train.nunique() < 3:
                auxiliary_metrics["elite_trade_quality"][target_name] = {
                    "skipped": True,
                    "reason": "Not enough combined elite setup rows or target variation in chronological split.",
                    "train_active_rows": int(len(y_reg_train)),
                    "test_active_rows": int(len(y_reg_test)),
                    "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
                }
                continue
            model = make_xgb_regressor(base_params, use_gpu)
            model.fit(X_elite_train, y_reg_train)
            model = make_cpu_portable_model(model, "xgboost")
            preds = model.predict(X_elite_test)
            report = regression_model_report(y_reg_test, preds)
            report.update({
                "target": target_name,
                "active_filter": elite_active,
                "included_strategy_ids": CONFIG.get("elite_strategy_ids", []),
                "train_active_rows": int(len(y_reg_train)),
                "test_active_rows": int(len(y_reg_test)),
                "setup_coverage_test": float(test_mask.mean()) if len(test_mask) else 0.0,
            })
            auxiliary_models[target_name] = model
            auxiliary_metrics["elite_trade_quality"][target_name] = report
            elite_regression_targets.append(target_name)
            print(f"  {target_name}: mae={report['mae']:.5f} rmse={report['rmse']:.5f}")

    print("\n── Phase 12: Optimize breakout-only precision buckets ─")
    breakout_precision_model, breakout_precision_report = _train_breakout_precision_variants(
        data=data,
        X_train=X_train,
        X_test=X_test,
        split=split,
        base_params=base_params,
        use_gpu=use_gpu,
        class_weights=CONFIG.get("breakout_precision_class_weights", [1.0]),
        min_signals=int(CONFIG.get("breakout_precision_min_signals", 10)),
    )
    auxiliary_metrics["breakout_precision"]["target_breakout_precision_take"] = breakout_precision_report
    if breakout_precision_model is not None:
        auxiliary_models["target_breakout_precision_take"] = breakout_precision_model
        breakout_precision_targets.append("target_breakout_precision_take")
        selection = breakout_precision_report.get("best_selection") or {}
        print(
            "  target_breakout_precision_take: "
            f"best_precision={selection.get('precision')} "
            f"bucket={selection.get('bucket')} signals={selection.get('signals')}"
        )
    else:
        print("  target_breakout_precision_take: skipped")

    final_hybrid_policy = build_final_hybrid_policy(CONFIG, auxiliary_metrics)

    # ── Save
    version = datetime.now(timezone.utc).strftime("model_%Y%m%d_%H%M%S")
    model_path   = OUTPUT_DIR / f"{version}.joblib"
    binary_model_path = OUTPUT_DIR / f"{version}_binary_direction.joblib"
    auxiliary_model_path = OUTPUT_DIR / f"{version}_auxiliary_targets.joblib"
    metrics_path = OUTPUT_DIR / f"{version}_metrics.json"

    joblib.dump({
        "models": final_models,
        "features":  selected_features,
        "ensemble":  "soft_vote_equal",
        "config":    CONFIG,
        "version":   version,
        "training_phase_manifest": build_training_manifest(),
        "final_hybrid_policy": final_hybrid_policy,
        "disclaimer": "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
    }, model_path)

    joblib.dump({
        "models": {"xgboost_binary_direction": binary_final},
        "features": selected_features,
        "target": "binary_direction",
        "positive_class": "forward_return_above_0",
        "prediction_horizon": CONFIG["prediction_horizon"],
        "config": CONFIG,
        "version": version,
        "disclaimer": "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
    }, binary_model_path)

    joblib.dump({
        "models": auxiliary_models,
        "features": selected_features,
        "targets": {
            "classification": [*aux_classification_targets, *strategy_target_names, *opportunity_classification_targets, *specialist_classification_targets, *elite_classification_targets, *breakout_precision_targets],
            "regression": [*aux_regression_targets, *strategy_return_targets, *strategy_risk_reward_targets, *opportunity_regression_targets, *specialist_regression_targets, *elite_regression_targets],
            "strategy_classification": strategy_target_names,
            "strategy_regression": [*strategy_return_targets, *strategy_risk_reward_targets],
            "strategy_opportunity_classification": opportunity_classification_targets,
            "strategy_opportunity_regression": opportunity_regression_targets,
            "specialist_strategy_classification": specialist_classification_targets,
            "specialist_strategy_regression": specialist_regression_targets,
            "elite_trade_quality_classification": elite_classification_targets,
            "elite_trade_quality_regression": elite_regression_targets,
            "breakout_precision_classification": breakout_precision_targets,
        },
        "strategy_definitions": strategy_definitions,
        "high_quality_strategy_ids": CONFIG.get("high_quality_strategy_ids", []),
        "specialist_strategy_ids": CONFIG.get("specialist_strategy_ids", []),
        "elite_strategy_ids": CONFIG.get("elite_strategy_ids", []),
        "breakout_precision_class_weights": CONFIG.get("breakout_precision_class_weights", []),
        "prediction_horizon": CONFIG["prediction_horizon"],
        "config": CONFIG,
        "version": version,
        "final_hybrid_policy": final_hybrid_policy,
        "disclaimer": "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
    }, auxiliary_model_path)

    payload = {
        "version":          version,
        "model_family":     "ensemble_" + "_".join(final_models.keys()),
        "gpu_used":         use_gpu,
        "tickers":          CONFIG["tickers"],
        "features":         selected_features,
        "all_candidate_feature_count": len(feature_cols),
        "n_features":       len(selected_features),
        "prediction_horizon": CONFIG["prediction_horizon"],
        "atr_label_multiplier": CONFIG["atr_label_multiplier"],
        "available_targets": [
            "target",
            "target_binary",
            "target_return_horizon",
            "target_future_range_high_pct",
            "target_future_range_low_pct",
            "target_breakout",
            "target_breakdown",
            "target_reversal",
            "target_volatility_expansion",
            "target_risk_adjusted_return",
            *CONFIG.get("strategy_targets", []),
            *strategy_return_targets,
            *[f"target_strategy_{definition['id']}_mfe_pct" for definition in strategy_definitions],
            *[f"target_strategy_{definition['id']}_mae_pct" for definition in strategy_definitions],
            *strategy_risk_reward_targets,
            "target_high_quality_strategy_active",
            "target_high_quality_strategy_success",
            "target_high_quality_strategy_return_pct",
            "target_high_quality_strategy_risk_reward",
            "target_high_quality_strategy_count",
            *specialist_classification_targets,
            *specialist_regression_targets,
            *[f"target_elite_{sid}_quality_success" for sid in CONFIG.get("elite_strategy_ids", [])],
            *[f"target_elite_{sid}_return_positive" for sid in CONFIG.get("elite_strategy_ids", [])],
            "target_elite_strategy_active",
            "target_elite_strategy_quality_success",
            "target_elite_strategy_return_pct",
            "target_elite_strategy_risk_reward",
            "target_elite_strategy_count",
            *elite_classification_targets,
            *elite_regression_targets,
            *breakout_precision_targets,
        ],
        "best_params": {
            "xgboost":  xgb_best_params(xgb_study2),
            **({"lightgbm": lgb_best_params(lgb_study)} if lgb_study is not None else {}),
            **({"catboost": cat_best_params(cat_study)} if cat_study is not None else {}),
        },
        "raw_optuna_params": {
            "xgboost":  xgb_study2.best_params,
            **({"lightgbm": lgb_study.best_params} if lgb_study is not None else {}),
            **({"catboost": cat_study.best_params} if cat_study is not None else {}),
        },
        "cv_f1_macro": {
            "xgboost":  xgb_study2.best_value,
            **({"lightgbm": lgb_study.best_value} if lgb_study is not None else {}),
            **({"catboost": cat_study.best_value} if cat_study is not None else {}),
        },
        "holdout_metrics": {
            "accuracy":    float(acc),
            "f1_macro":    float(f1),
            "precision_macro": float(precision_score(y_test, ensemble_preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, ensemble_preds, average="macro", zero_division=0)),
            "return_proxy_mae": float(return_mae),
            "return_proxy_rmse": float(return_rmse),
            "test_rows":   int(len(y_test)),
            "class_counts": y_test.value_counts().rename({0:"bearish",1:"sideways",2:"bullish"}).to_dict(),
            "confidence_buckets": multiclass_buckets,
        },
        "binary_high_confidence": {
            "artifact": binary_model_path.name,
            "target": "forward_return > 0 over prediction_horizon",
            "all_signal_accuracy": float(binary_acc),
            "f1": float(binary_f1),
            "precision": float(binary_precision),
            "recall": float(binary_recall),
            "test_rows": int(len(yb_test)),
            "class_counts": yb_test.value_counts().rename({0:"down_or_flat",1:"up"}).to_dict(),
            "confidence_buckets": binary_buckets,
            "proof_rule": "Only claim 75%+ accuracy for confidence buckets whose measured walk-forward accuracy is >= 0.75 and whose coverage is disclosed.",
        },
        "auxiliary_target_models": {
            "artifact": auxiliary_model_path.name,
            "classification_targets": [*aux_classification_targets, *strategy_target_names, *opportunity_classification_targets, *specialist_classification_targets, *elite_classification_targets, *breakout_precision_targets],
            "regression_targets": [*aux_regression_targets, *strategy_return_targets, *strategy_risk_reward_targets, *opportunity_regression_targets, *specialist_regression_targets, *elite_regression_targets],
            "strategy_definitions": strategy_definitions,
            "high_quality_strategy_ids": CONFIG.get("high_quality_strategy_ids", []),
            "specialist_strategy_ids": CONFIG.get("specialist_strategy_ids", []),
            "elite_strategy_ids": CONFIG.get("elite_strategy_ids", []),
            "breakout_precision_targets": breakout_precision_targets,
            "breakout_precision_class_weights": CONFIG.get("breakout_precision_class_weights", []),
            "elite_trade_quality_rules": {
                "minimum_return_pct": CONFIG.get("elite_min_return_pct"),
                "minimum_risk_reward": CONFIG.get("elite_min_risk_reward"),
                "requires_target_before_stop": True,
            },
            "strategy_quality": strategy_quality,
            "metrics": auxiliary_metrics,
            "usage": "These models support probability-based breakout, breakdown, reversal, volatility, strategy quality, future range, and risk-adjusted return intelligence. They do not guarantee outcomes.",
        },
        "final_hybrid_policy": final_hybrid_policy,
        "training_phase_manifest": build_training_manifest(),
        "honesty_policy": "Outputs are probability-based market simulations and AI-generated financial intelligence, not financial advice. Accuracy claims must include validation method, confidence threshold, and coverage.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    print(f"\n✓ Model        → {model_path}")
    print(f"✓ Binary proof → {binary_model_path}")
    print(f"✓ Aux targets  → {auxiliary_model_path}")
    print(f"✓ Metrics      → {metrics_path}")
    print(f"\n{'='*65}")
    print(f"FINAL: accuracy={acc:.4f}  f1_macro={f1:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
