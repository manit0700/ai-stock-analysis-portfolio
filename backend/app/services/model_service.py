from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS_V1 = [
    "return_1", "return_5", "return_10",
    "volatility_10", "volatility_20",
    "volume_change", "sma_20_gap", "sma_50_gap",
    "rsi_14", "macd_hist",
]

FEATURE_COLUMNS_V2 = [
    "return_1", "return_5", "return_10", "return_20",
    "price_accel",
    "sma_20_gap", "sma_50_gap", "sma_200_gap",
    "ema_9_gap", "ema_20_gap",
    "rsi_14", "stoch_k", "stoch_d",
    "williams_r", "cci_14",
    "volatility_10", "volatility_20", "volatility_ratio",
    "atr_pct",
    "volume_change", "obv_signal", "cmf_20",
    "macd_hist", "macd_hist_slope",
    "spy_return_5", "pct_from_52w_high",
]

# Use v2 by default — falls back to v1 for old single-model artifacts
FEATURE_COLUMNS = FEATURE_COLUMNS_V2

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACTS_DIRS = [
    _PROJECT_ROOT / "ml" / "artifacts",
    _PROJECT_ROOT / "ml" / "kaggle_output_latest" / "artifacts",
    _PROJECT_ROOT / "ml" / "kaggle_output_v12" / "artifacts",
]


def _find_latest_model() -> Path | None:
    candidates: list[Path] = []
    for directory in _ARTIFACTS_DIRS:
        if not directory.exists():
            continue
        candidates.extend(
            path for path in directory.glob("model_*.joblib")
            if not path.name.endswith("_binary_direction.joblib")
            and not path.name.endswith("_auxiliary_targets.joblib")
        )
    candidates = sorted(candidates, key=lambda path: path.stem, reverse=True)
    return candidates[0] if candidates else None


def _find_related_artifact(model_path: Path, suffix: str) -> Path | None:
    candidate = model_path.with_name(f"{model_path.stem}_{suffix}.joblib")
    return candidate if candidate.exists() else None


def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda values: np.dot(values, weights) / weights.sum(), raw=True)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    return series.rolling(window).apply(lambda values: float(np.dot(x, values - values.mean()) / denom), raw=True)


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
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
    atr14 = _atr_series(high, low, close, window)
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr14.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(window).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_features(history: pd.DataFrame) -> pd.Series:
    close = history["Close"]
    high = history["High"]
    low = history["Low"]
    volume = history["Volume"]

    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    wma_20 = _wma(close, 20)
    wma_50 = _wma(close, 50)

    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    rsi = _rsi(close, 14)
    rsi_7 = _rsi(close, 7)

    lowest_k = low.rolling(14).min()
    highest_k = high.rolling(14).max()
    stoch_k = 100 * (close - lowest_k) / (highest_k - lowest_k).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    typical = (high + low + close) / 3
    cci_mean_dev = typical.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_14 = (typical - typical.rolling(14).mean()) / (0.015 * cci_mean_dev.replace(0, np.nan))

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = atr / close.replace(0, np.nan)
    adx = _adx(high, low, close, 14)

    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    obv_signal = obv.ewm(span=10, adjust=False).mean().diff()

    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    cmf_20 = (clv * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    return_1 = close.pct_change()
    vol_10 = return_1.rolling(10).std()
    vol_20 = return_1.rolling(20).std()
    rolling_52w_high = close.rolling(252).max()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    vol_avg_10 = volume.rolling(10).mean()
    vol_avg_15 = volume.rolling(15).mean()
    vol_avg_20 = volume.rolling(20).mean()
    vol_ratio_10 = volume / vol_avg_10
    vol_ratio_15 = volume / vol_avg_15
    vol_ratio_20 = volume / vol_avg_20
    resistance_20 = high.shift(1).rolling(20).max()
    support_20 = low.shift(1).rolling(20).min()
    breakout_20 = close > resistance_20
    breakdown_20 = close < support_20
    price_slope_10 = _rolling_slope(close, 10) / close.replace(0, np.nan)
    rsi_slope_10 = _rolling_slope(rsi, 10) / 100

    f = pd.DataFrame({
        # legacy names
        "return_1": return_1,
        "return_5": close.pct_change(5),
        "return_10": close.pct_change(10),
        "return_20": close.pct_change(20),
        "price_accel": return_1 - return_1.shift(1),
        "sma_20_gap": close / sma_20 - 1,
        "sma_50_gap": close / sma_50 - 1,
        "sma_200_gap": close / sma_200 - 1,
        "ema_9_gap": close / ema_9 - 1,
        "ema_20_gap": close / ema_20 - 1,
        "rsi_14": rsi,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "williams_r": -100 * (close.rolling(14).max() - close) / (close.rolling(14).max() - close.rolling(14).min()).replace(0, np.nan),
        "cci_14": cci_14,
        "volatility_10": vol_10,
        "volatility_20": vol_20,
        "volatility_ratio": vol_10 / vol_20.replace(0, np.nan),
        "atr_pct": atr_pct,
        "volume_change": volume.pct_change(),
        "obv_signal": obv_signal,
        "cmf_20": cmf_20,
        "macd_hist": macd_hist,
        "macd_hist_slope": macd_hist.diff(),
        "spy_return_5": pd.Series(np.nan, index=close.index),  # no SPY context at inference time
        "pct_from_52w_high": (close - rolling_52w_high) / rolling_52w_high.replace(0, np.nan),
        # v3/v4 Kaggle feature names
        "ret_1": return_1,
        "ret_3": close.pct_change(3),
        "ret_5": close.pct_change(5),
        "ret_10": close.pct_change(10),
        "ret_20": close.pct_change(20),
        "vol_10": vol_10,
        "vol_20": vol_20,
        "vol_expansion_10_20": vol_10 / vol_20.replace(0, np.nan) - 1,
        "sma10_gap": close / sma_10 - 1,
        "sma20_gap": close / sma_20 - 1,
        "sma50_gap": close / sma_50 - 1,
        "sma200_gap": close / sma_200 - 1,
        "ema9_gap": close / ema_9 - 1,
        "ema20_gap": close / ema_20 - 1,
        "ema50_gap": close / ema_50 - 1,
        "wma20_gap": close / wma_20 - 1,
        "wma50_gap": close / wma_50 - 1,
        "sma10_20_cross": sma_10 / sma_20 - 1,
        "sma20_50_cross": sma_20 / sma_50 - 1,
        "sma50_200_cross": sma_50 / sma_200 - 1,
        "ema9_20_cross": ema_9 / ema_20 - 1,
        "ema20_50_cross": ema_20 / ema_50 - 1,
        "trend_slope_10": _rolling_slope(close, 10) / close.replace(0, np.nan),
        "trend_slope_20": _rolling_slope(close, 20) / close.replace(0, np.nan),
        "trend_slope_50": _rolling_slope(close, 50) / close.replace(0, np.nan),
        "rsi_7": rsi_7,
        "rsi_slope_5": _rolling_slope(rsi, 5),
        "rsi_slope_10": _rolling_slope(rsi, 10),
        "rsi_overbought": (rsi >= 70).astype(int),
        "rsi_oversold": (rsi <= 30).astype(int),
        "macd_line": macd / close.replace(0, np.nan),
        "macd_signal": macd_signal / close.replace(0, np.nan),
        "macd_hist_slope": _rolling_slope(macd_hist, 5) / close.replace(0, np.nan),
        "atr_norm": atr_pct,
        "atr_slope_5": _rolling_slope(atr, 5) / close.replace(0, np.nan),
        "adx": adx,
        "obv_mom": obv / obv.rolling(20).mean() - 1,
        "stoch_diff": stoch_k - stoch_d,
        "bb_pctb": (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan),
        "bb_width": bb_width,
        "bb_width_slope": _rolling_slope(bb_width, 5),
        "bb_squeeze": (bb_width < bb_width.rolling(120).quantile(0.2)).astype(int),
        "cci_20": cci_14,
        "vol_zscore": (volume - vol_avg_20) / volume.rolling(20).std().replace(0, np.nan),
        "vol_trend": volume / vol_avg_10 - 1,
        "vol_ratio_10": vol_ratio_10,
        "vol_ratio_15": vol_ratio_15,
        "vol_ratio_20": vol_ratio_20,
        "vol_spike_2x_10": (vol_ratio_10 >= 2.0).astype(int),
        "vol_spike_3x_10": (vol_ratio_10 >= 3.0).astype(int),
        "vol_spike_2x_15": (vol_ratio_15 >= 2.0).astype(int),
        "vol_spike_3x_15": (vol_ratio_15 >= 3.0).astype(int),
        "price_volume_pressure": return_1 * vol_ratio_20,
        "accumulation_pressure": ((close > close.shift(1)).astype(int) * vol_ratio_20).rolling(5).mean(),
        "distribution_pressure": ((close < close.shift(1)).astype(int) * vol_ratio_20).rolling(5).mean(),
        "dist_to_resistance_20": close / resistance_20 - 1,
        "dist_to_support_20": close / support_20 - 1,
        "range_position_20": (close - support_20) / (resistance_20 - support_20).replace(0, np.nan),
        "breakout_20": breakout_20.astype(int),
        "breakdown_20": breakdown_20.astype(int),
        "breakout_volume_confirmed": (breakout_20 & (vol_ratio_10 >= 2.0)).astype(int),
        "breakdown_volume_confirmed": (breakdown_20 & (vol_ratio_10 >= 2.0)).astype(int),
        "failed_breakout_risk": ((high > resistance_20) & (close < resistance_20) & (vol_ratio_10 >= 1.5)).astype(int),
        "failed_breakdown_reversal": ((low < support_20) & (close > support_20) & (vol_ratio_10 >= 1.5)).astype(int),
        "bearish_rsi_divergence": ((price_slope_10 > 0) & (rsi_slope_10 < 0)).astype(int),
        "bullish_rsi_divergence": ((price_slope_10 < 0) & (rsi_slope_10 > 0)).astype(int),
        "rsi_divergence_strength": price_slope_10 - rsi_slope_10,
    })
    # v5/v6 intraday features require recent intraday context. The live API
    # currently predicts from one OHLCV frame, so neutralize intraday-only
    # fields while still computing daily strategy setup detectors.
    for timeframe in ["5m", "15m", "1h"]:
        for suffix in [
            "trend", "ema9_20", "ret_3", "ret_12", "volatility_12", "volume_spike",
            "above_vwap", "vwap_dist", "vwap_slope", "vwap_reclaim", "vwap_rejection",
        ]:
            f[f"{timeframe}_{suffix}"] = 0.0
    f["mtf_bullish_alignment"] = 0.0
    f["mtf_bearish_alignment"] = 0.0
    f["mtf_mixed_alignment"] = 1.0
    f["mtf_trend_score"] = 0.0
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
    f["setup_vwap_reclaim"] = 0
    f["setup_rsi_divergence_short"] = (
        (f["bearish_rsi_divergence"] == 1)
        & (f["rsi_14"] >= 55)
        & (f["vol_ratio_10"] >= 1.0)
    ).astype(int)
    dow = pd.Series(history.index.dayofweek, index=history.index)
    month = pd.Series(history.index.month, index=history.index)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    return f.iloc[-1]


class ModelService:
    def __init__(self) -> None:
        self._model: Any = None
        self._models: dict[str, Any] | None = None
        self._auxiliary_models: dict[str, Any] = {}
        self._auxiliary_targets: dict[str, list[str]] = {}
        self._model_version: str | None = None
        self._feature_columns: list[str] = FEATURE_COLUMNS_V1
        self._final_hybrid_policy: dict[str, Any] = {}
        self._artifact_disclaimer: str = (
            "Probability-based market simulations and AI-generated financial intelligence, not financial advice."
        )
        self._load()

    def _load(self) -> None:
        model_path = _find_latest_model()
        if model_path is None:
            logger.info("No trained model artifact found — ML blend disabled.")
            return
        try:
            import joblib
            artifact = joblib.load(model_path)
            self._model = artifact.get("model")
            self._models = artifact.get("models")
            if self._model is None and not self._models:
                raise ValueError("Artifact does not contain 'model' or 'models'.")
            self._model_version = model_path.stem
            # Use the feature list the artifact was trained on
            saved_features = artifact.get("features")
            if saved_features:
                self._feature_columns = saved_features
            else:
                self._feature_columns = FEATURE_COLUMNS_V1
            self._final_hybrid_policy = artifact.get("final_hybrid_policy") or {}
            self._artifact_disclaimer = artifact.get("disclaimer", self._artifact_disclaimer)
            aux_path = _find_related_artifact(model_path, "auxiliary_targets")
            if aux_path is not None:
                aux_artifact = joblib.load(aux_path)
                self._auxiliary_models = aux_artifact.get("models") or {}
                self._auxiliary_targets = aux_artifact.get("targets") or {}
                self._final_hybrid_policy = aux_artifact.get("final_hybrid_policy") or self._final_hybrid_policy
            logger.info("Loaded ML model: %s (%d features)", self._model_version, len(self._feature_columns))
        except Exception as exc:
            logger.warning("Failed to load ML model: %s", exc)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or bool(self._models)

    @property
    def model_version(self) -> str | None:
        return self._model_version

    @property
    def has_auxiliary_models(self) -> bool:
        return bool(self._auxiliary_models)

    @property
    def final_hybrid_policy(self) -> dict[str, Any]:
        return self._final_hybrid_policy

    def _build_model_row(self, history: pd.DataFrame) -> pd.DataFrame | None:
        features = _compute_features(history)
        cols = self._feature_columns
        if not cols:
            return None
        row = {feature: features.get(feature, np.nan) for feature in cols}
        # Context features that require SPY/QQQ/VIX/sector/yield data may be absent
        # when predicting a single ticker. Use neutral values instead of skipping
        # the whole model, and expose outputs as probability-based intelligence.
        neutralized = {key: (0.0 if pd.isna(value) else value) for key, value in row.items()}
        return pd.DataFrame([neutralized], columns=cols)

    def predict_probabilities(self, history: pd.DataFrame) -> dict[str, float] | None:
        if not self.is_loaded:
            return None
        try:
            x = self._build_model_row(history)
            if x is None:
                return None
            if self._models:
                probas = [model.predict_proba(x)[0] for model in self._models.values()]
                proba = np.mean(probas, axis=0)
            else:
                proba = self._model.predict_proba(x)[0]
            # class order from training: 0=bearish, 1=sideways, 2=bullish
            if len(proba) == 3:
                return {
                    "bearish": float(proba[0]),
                    "sideways": float(proba[1]),
                    "bullish": float(proba[2]),
                }
            return None
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            return None

    def predict_auxiliary(self, history: pd.DataFrame) -> dict[str, Any] | None:
        if not self._auxiliary_models:
            return None
        try:
            x = self._build_model_row(history)
            if x is None:
                return None
            classification_targets = set(self._auxiliary_targets.get("classification", []))
            regression_targets = set(self._auxiliary_targets.get("regression", []))
            event_probabilities: dict[str, dict[str, float]] = {}
            regression_outputs: dict[str, float] = {}

            for target_name, model in self._auxiliary_models.items():
                public_name = target_name.removeprefix("target_")
                if target_name in classification_targets and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x)[0]
                    positive = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    event_probabilities[public_name] = {
                        "probability": positive,
                        "confidence": max(positive, 1.0 - positive),
                    }
                elif target_name in regression_targets:
                    regression_outputs[public_name] = float(model.predict(x)[0])

            return {
                "event_probabilities": event_probabilities,
                "regression_outputs": regression_outputs,
                "targets": self._auxiliary_targets,
                "model_version": self._model_version,
                "final_hybrid_policy": self._final_hybrid_policy,
                "disclaimer": self._artifact_disclaimer,
            }
        except Exception as exc:
            logger.warning("Auxiliary ML prediction failed: %s", exc)
            return None
