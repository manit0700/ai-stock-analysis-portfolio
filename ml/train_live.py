"""
MarketVision AI — Live Data Ensemble Training Pipeline
======================================================
Fetches fresh daily candles from yfinance, engineers 25+ features,
trains an XGBoost + LightGBM + RandomForest ensemble, and saves a
versioned artifact to ml/artifacts/.

Run locally:
    python ml/train_live.py

Run on Kaggle (GPU):
    python ml/train_live.py --gpu --trials 120 --output /kaggle/working/artifacts

Continuous retraining (cron / GitHub Actions):
    python ml/train_live.py --incremental   # appends new candles, retrains
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Tickers ──────────────────────────────────────────────────────────────────
TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Semiconductors
    "AMD", "INTC", "QCOM", "AVGO", "MU", "TSM",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "UNH", "JNJ", "PFE",
    # Energy / macro proxies
    "XOM", "CVX",
    # Market proxies (very useful context features)
    "SPY", "QQQ", "IWM",
]

FEATURE_COLUMNS = [
    # Price momentum
    "return_1", "return_5", "return_10", "return_20",
    "price_accel",          # 2nd derivative: return_1 - prev return_1
    # Trend
    "sma_20_gap", "sma_50_gap", "sma_200_gap",
    "ema_9_gap", "ema_20_gap",
    # Oscillators
    "rsi_14",
    "stoch_k", "stoch_d",
    "williams_r",
    "cci_14",
    # Volatility
    "volatility_10", "volatility_20", "volatility_ratio",  # 10d/20d
    "atr_pct",              # ATR as % of price
    # Volume
    "volume_change",
    "obv_signal",           # OBV EMA slope
    "cmf_20",               # Chaikin Money Flow
    # MACD
    "macd_hist",
    "macd_hist_slope",      # MACD histogram momentum
    # Market context
    "spy_return_5",         # Broad market direction
    "pct_from_52w_high",    # Distance from 52-week high (mean-reversion signal)
]


# ── Feature engineering ───────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    mean_dev = typical.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - typical.rolling(window).mean()) / (0.015 * mean_dev.replace(0, np.nan))


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = clv * volume
    return mfv.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)


def _obv_signal(close: pd.Series, volume: pd.Series, span: int = 10) -> pd.Series:
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv.ewm(span=span, adjust=False).mean().diff()


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr / close.replace(0, np.nan)


def engineer_features(raw: pd.DataFrame, spy_close: pd.Series | None = None) -> pd.DataFrame:
    close = raw["Close"]
    high = raw["High"]
    low = raw["Low"]
    volume = raw["Volume"]

    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    return_1 = close.pct_change()
    stoch_k, stoch_d = _stochastic(high, low, close)

    rolling_52w_high = close.rolling(252).max()
    pct_from_52w_high = (close - rolling_52w_high) / rolling_52w_high.replace(0, np.nan)

    vol_10 = return_1.rolling(10).std()
    vol_20 = return_1.rolling(20).std()

    if spy_close is not None:
        aligned_spy = spy_close.reindex(close.index, method="ffill")
        spy_return_5 = aligned_spy.pct_change(5)
    else:
        spy_return_5 = pd.Series(np.nan, index=close.index)

    f = pd.DataFrame(index=raw.index)
    f["return_1"] = return_1
    f["return_5"] = close.pct_change(5)
    f["return_10"] = close.pct_change(10)
    f["return_20"] = close.pct_change(20)
    f["price_accel"] = return_1 - return_1.shift(1)
    f["sma_20_gap"] = close / sma_20 - 1
    f["sma_50_gap"] = close / sma_50 - 1
    f["sma_200_gap"] = close / sma_200 - 1
    f["ema_9_gap"] = close / ema_9 - 1
    f["ema_20_gap"] = close / ema_20 - 1
    f["rsi_14"] = _rsi(close)
    f["stoch_k"] = stoch_k
    f["stoch_d"] = stoch_d
    f["williams_r"] = -100 * (close.rolling(14).max() - close) / (close.rolling(14).max() - close.rolling(14).min()).replace(0, np.nan)
    f["cci_14"] = _cci(high, low, close)
    f["volatility_10"] = vol_10
    f["volatility_20"] = vol_20
    f["volatility_ratio"] = vol_10 / vol_20.replace(0, np.nan)
    f["atr_pct"] = _atr_pct(high, low, close)
    f["volume_change"] = volume.pct_change()
    f["obv_signal"] = _obv_signal(close, volume)
    f["cmf_20"] = _cmf(high, low, close, volume)
    f["macd_hist"] = macd_hist
    f["macd_hist_slope"] = macd_hist.diff()
    f["spy_return_5"] = spy_return_5
    f["pct_from_52w_high"] = pct_from_52w_high

    return f


def make_labels(close: pd.Series, horizon: int, atr_pct: pd.Series) -> pd.Series:
    """ATR-adaptive labeling: threshold scales with volatility."""
    future_return = close.shift(-horizon) / close - 1
    # Use 0.5x ATR as dynamic threshold — volatile stocks need bigger moves
    threshold = (atr_pct * 0.5).clip(0.008, 0.025)
    labels = np.select(
        [future_return > threshold, future_return < -threshold],
        [2, 0],
        default=1,
    )
    return pd.Series(labels, index=close.index, name="target")


# ── Dataset builder ───────────────────────────────────────────────────────────

def fetch_and_build(
    tickers: list[str],
    period: str,
    interval: str,
    horizon: int,
    existing_path: Path | None = None,
) -> pd.DataFrame:
    log.info("Downloading market data (%s tickers, period=%s) …", len(tickers), period)

    # Download SPY separately for market-context feature
    try:
        spy_raw = yf.Ticker("SPY").history(period=period, interval=interval, auto_adjust=False)
        spy_close = spy_raw["Close"] if not spy_raw.empty else None
    except Exception:
        spy_close = None

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            if raw.empty or len(raw) < 60:
                log.warning("  skip %-6s — insufficient rows", ticker)
                continue
            raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            feats = engineer_features(raw, spy_close=spy_close)
            feats["ticker"] = ticker
            feats["target"] = make_labels(raw["Close"], horizon, feats["atr_pct"])
            feats = feats.dropna()
            log.info("  ok   %-6s  %d rows", ticker, len(feats))
            frames.append(feats)
        except Exception as exc:
            log.warning("  err  %-6s  %s", ticker, exc)

    if not frames:
        raise ValueError("No data returned for any ticker.")

    fresh = pd.concat(frames).sort_index()

    if existing_path and existing_path.exists():
        old = pd.read_parquet(existing_path)
        combined = pd.concat([old, fresh]).drop_duplicates(subset=["ticker"] + [fresh.index.name or "Date"]).sort_index()
        log.info("Incremental update: %d existing + %d new → %d total rows", len(old), len(fresh), len(combined))
        return combined

    log.info("Dataset built: %d rows, %d tickers", len(fresh), fresh["ticker"].nunique())
    dist = fresh["target"].value_counts().rename({0: "bearish", 1: "sideways", 2: "bullish"})
    log.info("Class distribution:\n%s", dist.to_string())
    return fresh


# ── Model builders ────────────────────────────────────────────────────────────

def _xgb_params(trial: optuna.Trial | None, use_gpu: bool) -> dict:
    t = trial
    return {
        "n_estimators": t.suggest_int("xgb_n", 200, 1000) if t else 600,
        "max_depth": t.suggest_int("xgb_depth", 2, 8) if t else 4,
        "learning_rate": t.suggest_float("xgb_lr", 0.005, 0.15, log=True) if t else 0.03,
        "subsample": t.suggest_float("xgb_ss", 0.6, 1.0) if t else 0.85,
        "colsample_bytree": t.suggest_float("xgb_cs", 0.5, 1.0) if t else 0.85,
        "min_child_weight": t.suggest_int("xgb_mcw", 1, 15) if t else 5,
        "gamma": t.suggest_float("xgb_g", 0.0, 0.5) if t else 0.1,
        "reg_lambda": t.suggest_float("xgb_l2", 0.5, 5.0) if t else 1.5,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 3,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "random_state": 42,
        "verbosity": 0,
    }


def _lgb_params(trial: optuna.Trial | None, use_gpu: bool) -> dict:
    t = trial
    return {
        "n_estimators": t.suggest_int("lgb_n", 200, 1000) if t else 600,
        "num_leaves": t.suggest_int("lgb_nl", 20, 150) if t else 63,
        "learning_rate": t.suggest_float("lgb_lr", 0.005, 0.15, log=True) if t else 0.03,
        "subsample": t.suggest_float("lgb_ss", 0.6, 1.0) if t else 0.85,
        "colsample_bytree": t.suggest_float("lgb_cs", 0.5, 1.0) if t else 0.85,
        "min_child_samples": t.suggest_int("lgb_mcs", 5, 60) if t else 20,
        "reg_lambda": t.suggest_float("lgb_l2", 0.5, 5.0) if t else 1.5,
        "device": "gpu" if use_gpu else "cpu",
        "random_state": 42,
        "verbosity": -1,
        "force_row_wise": True,
    }


def _rf_params(trial: optuna.Trial | None) -> dict:
    t = trial
    return {
        "n_estimators": t.suggest_int("rf_n", 100, 500) if t else 300,
        "max_depth": t.suggest_int("rf_depth", 4, 20) if t else 12,
        "min_samples_leaf": t.suggest_int("rf_msl", 1, 12) if t else 4,
        "max_features": t.suggest_categorical("rf_mf", ["sqrt", "log2", 0.5]) if t else "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }


def cv_score(frame: pd.DataFrame, use_gpu: bool, trial: optuna.Trial) -> float:
    x = frame[FEATURE_COLUMNS]
    y = frame["target"]
    splitter = TimeSeriesSplit(n_splits=5)
    scores: list[float] = []

    for train_idx, val_idx in splitter.split(x):
        estimators = []
        if XGBClassifier:
            estimators.append(("xgb", XGBClassifier(**_xgb_params(trial, use_gpu))))
        if LGBMClassifier:
            estimators.append(("lgb", LGBMClassifier(**_lgb_params(trial, use_gpu))))
        estimators.append(("rf", RandomForestClassifier(**_rf_params(trial))))

        if len(estimators) == 1:
            model = estimators[0][1]
        else:
            model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(x.iloc[val_idx])
        scores.append(f1_score(y.iloc[val_idx], preds, average="macro"))

    return float(np.mean(scores))


def train_full_data_ensemble(frame: pd.DataFrame, best_params: dict, use_gpu: bool):
    """Retrain on 100% of data after architecture is validated — maximises knowledge."""
    x = frame[FEATURE_COLUMNS]
    y = frame["target"]

    estimators = []
    if XGBClassifier:
        estimators.append(("xgb", XGBClassifier(**_xgb_params(None, use_gpu))))
    if LGBMClassifier:
        estimators.append(("lgb", LGBMClassifier(**_lgb_params(None, use_gpu))))
    estimators.append(("rf", RandomForestClassifier(**_rf_params(None))))

    for name, est in estimators:
        filtered = {k[len(name)+1:]: v for k, v in best_params.items() if k.startswith(f"{name}_")}
        if filtered and hasattr(est, "set_params"):
            try:
                est.set_params(**filtered)
            except Exception:
                pass

    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1) if len(estimators) > 1 else estimators[0][1]
    ensemble.fit(x, y)
    return make_cpu_portable_ensemble(ensemble)


def train_final_ensemble(frame: pd.DataFrame, best_params: dict, use_gpu: bool):
    x = frame[FEATURE_COLUMNS]
    y = frame["target"]
    split = int(len(frame) * 0.82)

    estimators = []
    if XGBClassifier:
        estimators.append(("xgb", XGBClassifier(**_xgb_params(None, use_gpu))))
    if LGBMClassifier:
        estimators.append(("lgb", LGBMClassifier(**_lgb_params(None, use_gpu))))
    estimators.append(("rf", RandomForestClassifier(**_rf_params(None))))

    # Apply best params to each sub-model
    for name, est in estimators:
        filtered = {k[len(name)+1:]: v for k, v in best_params.items() if k.startswith(f"{name}_")}
        if filtered and hasattr(est, "set_params"):
            try:
                est.set_params(**filtered)
            except Exception:
                pass

    if len(estimators) == 1:
        ensemble = estimators[0][1]
    else:
        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    ensemble.fit(x.iloc[:split], y.iloc[:split])
    preds = ensemble.predict(x.iloc[split:])
    report = classification_report(
        y.iloc[split:], preds,
        target_names=["bearish", "sideways", "bullish"],
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(y.iloc[split:], preds)),
        "f1_macro": float(f1_score(y.iloc[split:], preds, average="macro")),
        "f1_per_class": {k: round(v["f1-score"], 4) for k, v in report.items() if k in ("bearish", "sideways", "bullish")},
        "test_rows": int(len(y.iloc[split:])),
        "models_in_ensemble": [name for name, _ in estimators],
    }
    return make_cpu_portable_ensemble(ensemble), metrics


def make_cpu_portable_ensemble(ensemble):
    """
    Train with GPU on Kaggle, then save CPU-safe wrappers for local backend inference.
    """
    estimators = getattr(ensemble, "estimators_", None)
    if estimators is None:
        estimators = [ensemble]

    for model in estimators:
        name = model.__class__.__name__.lower()
        try:
            if "xgb" in name:
                model.set_params(device="cpu", tree_method="hist")
                model.get_booster().set_param({"device": "cpu", "tree_method": "hist"})
            elif "lgbm" in name:
                model.set_params(device="cpu")
                model.booster_.reset_parameter({"device": "cpu"})
            elif "catboost" in name:
                model.set_params(task_type="CPU")
        except Exception as exc:
            log.warning("Could not switch %s to CPU inference settings: %s", model.__class__.__name__, exc)
    return ensemble


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days")
    parser.add_argument("--period", default="5y", help="yfinance period string")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for XGBoost/LightGBM")
    parser.add_argument("--incremental", action="store_true", help="Append new candles to existing dataset")
    parser.add_argument("--full-data", action="store_true", help="After validation, retrain on 100%% of data for deployment")
    parser.add_argument("--output", default="ml/artifacts", help="Output directory for artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "training_dataset.parquet"

    log.info("=" * 60)
    log.info("MarketVision AI — Live Ensemble Training")
    log.info("trials=%d  horizon=%d  gpu=%s  incremental=%s  full_data=%s",
             args.trials, args.horizon, args.gpu, args.incremental, args.full_data)
    log.info("=" * 60)

    existing = dataset_path if args.incremental else None
    frame = fetch_and_build(
        tickers=TICKERS,
        period=args.period,
        interval="1d",
        horizon=args.horizon,
        existing_path=existing,
    )
    frame.to_parquet(dataset_path)
    log.info("Dataset saved → %s (%d rows)", dataset_path, len(frame))

    log.info("\nRunning Optuna (%d trials) …", args.trials)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: cv_score(frame, args.gpu, trial),
        n_trials=args.trials,
        show_progress_bar=True,
    )
    log.info("Best CV F1-macro: %.4f", study.best_value)

    log.info("\nTraining validation ensemble (82/18 split) …")
    ensemble, metrics = train_final_ensemble(frame, study.best_params, args.gpu)
    log.info("Holdout accuracy: %.4f  F1-macro: %.4f", metrics["accuracy"], metrics["f1_macro"])

    if args.full_data:
        log.info("\nFull-data mode: retraining on 100%% of data for deployment …")
        ensemble = train_full_data_ensemble(frame, study.best_params, args.gpu)
        metrics["trained_on"] = "100% of data (no holdout — deploy mode)"
    else:
        metrics["trained_on"] = "82% train / 18% holdout (validation mode)"

    version = datetime.now(timezone.utc).strftime("model_%Y%m%d_%H%M%S")
    model_path = output_dir / f"{version}.joblib"
    metrics_path = output_dir / f"{version}_metrics.json"

    joblib.dump({
        "model": ensemble,
        "features": FEATURE_COLUMNS,
        "model_type": "ensemble",
        "tickers_trained_on": TICKERS,
        "horizon": args.horizon,
        "full_data_mode": args.full_data,
    }, model_path)

    payload = {
        "version": version,
        "model_type": "ensemble",
        "models": metrics["models_in_ensemble"],
        "gpu_used": args.gpu,
        "tickers": TICKERS,
        "n_trials": args.trials,
        "horizon_days": args.horizon,
        "cv_f1_macro": study.best_value,
        "holdout": metrics,
        "best_params": study.best_params,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    log.info("\n✓ Model  → %s", model_path)
    log.info("✓ Metrics → %s", metrics_path)
    log.info("\nHoldout results:")
    log.info("  accuracy : %.4f", metrics["accuracy"])
    log.info("  f1_macro : %.4f", metrics["f1_macro"])
    log.info("  per class: %s", metrics["f1_per_class"])
    log.info("\nDrop files in ml/artifacts/ — backend auto-loads latest on restart.")


if __name__ == "__main__":
    main()
