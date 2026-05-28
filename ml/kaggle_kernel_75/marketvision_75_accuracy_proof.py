"""
MarketVision AI - 75% Accuracy Proof Kernel
===========================================

Goal:
  Prove or reject a 75% accuracy claim honestly.

This kernel does NOT claim 75% on every prediction. It measures:
  1. all-signal out-of-sample accuracy
  2. high-confidence out-of-sample accuracy
  3. walk-forward accuracy by future time window
  4. majority-class and naive baselines
  5. a simple long/short backtest with costs
  6. practical confidence explanations for individual high-confidence signals

If 75% appears only in high-confidence buckets, the platform can say:
  "up to 75% accuracy on high-confidence signals"

Every high-confidence signal also includes:
  - ensemble probability margin
  - model agreement across XGBoost, LightGBM, and CatBoost
  - feature-driver facts versus the training distribution
  - actual future result for auditability

Kaggle setup:
  Accelerator: GPU T4 x2 if available
  Internet: ON
  Output: /kaggle/working/artifacts
"""

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path


def pip_install(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


pip_install("yfinance", "xgboost", "lightgbm", "catboost", "joblib", "scikit-learn", "pyarrow")

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


CONFIG = {
    "tickers": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "ORCL",
        "JPM", "BAC", "GS", "V", "MA", "UNH", "JNJ", "LLY", "XOM", "CVX",
        "CAT", "BA", "WMT", "HD", "NKE", "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV",
    ],
    "period": "10y",
    "interval": "1d",
    "prediction_horizon": 5,
    "label_threshold": 0.0075,
    "feature_shift": 1,
    "output_dir": "/kaggle/working/artifacts",
    "use_gpu": True,
    "confidence_thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "explanation_confidence_threshold": 0.75,
    "max_signal_explanations_per_window": 25,
    "top_explanation_features": 8,
    "fee_bps": 2.0,
    "slippage_bps": 3.0,
}

CONFIG["tickers"] = list(dict.fromkeys(CONFIG["tickers"]))
OUTPUT_DIR = Path(CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_DIMENSION_COVERAGE = {
    "technical_analysis": {
        "status": "partial",
        "trained_now": [
            "RSI",
            "MACD histogram",
            "EMA gap",
            "SMA gaps",
            "Bollinger Bands",
            "ATR",
            "momentum returns",
            "volume z-score",
            "stochastic oscillator",
        ],
        "missing_for_full_scope": [
            "VWAP intraday",
            "Fibonacci retracement",
            "volume profile",
            "support/resistance",
            "candlestick patterns",
            "breakout detection",
            "liquidity zones",
            "gap analysis",
            "order blocks",
            "trendlines",
        ],
    },
    "market_structure_analysis": {
        "status": "not_yet_trained",
        "trained_now": [],
        "missing_for_full_scope": [
            "higher highs/lows",
            "lower highs/lows",
            "consolidation zones",
            "accumulation/distribution",
            "liquidity sweeps",
            "trend shifts",
            "range behavior",
        ],
    },
    "multi_timeframe_analysis": {
        "status": "not_yet_trained",
        "trained_now": ["daily timeframe only"],
        "missing_for_full_scope": ["1min", "5min", "15min", "1h", "4h", "weekly alignment"],
    },
    "volume_analysis": {
        "status": "partial",
        "trained_now": ["volume z-score", "volume trend"],
        "missing_for_full_scope": [
            "buying/selling pressure",
            "volume divergence",
            "volume profile",
            "accumulation/distribution volume",
        ],
    },
    "volatility_analysis": {
        "status": "partial",
        "trained_now": ["ATR normalized", "rolling volatility", "Bollinger width"],
        "missing_for_full_scope": ["VIX correlation", "volatility contraction/expansion regimes", "uncertainty model"],
    },
    "momentum_analysis": {
        "status": "partial",
        "trained_now": ["1/2/3/5/10/20-period returns", "MACD histogram", "RSI", "relative SPY strength"],
        "missing_for_full_scope": ["RSI divergence", "MACD divergence", "momentum exhaustion classification"],
    },
    "sentiment_analysis": {
        "status": "not_yet_trained",
        "trained_now": [],
        "missing_for_full_scope": ["news sentiment", "Reddit", "Twitter/X", "analyst sentiment", "earnings tone"],
    },
    "macroeconomic_analysis": {
        "status": "not_yet_trained",
        "trained_now": [],
        "missing_for_full_scope": ["interest rates", "inflation", "GDP", "unemployment", "treasury yields", "Fed events"],
    },
    "sector_rotation_analysis": {
        "status": "partial",
        "trained_now": ["SPY relative strength"],
        "missing_for_full_scope": ["sector ETF momentum matrix", "sector-relative performance", "institutional money flow"],
    },
    "options_flow_analysis": {
        "status": "not_yet_trained",
        "trained_now": [],
        "missing_for_full_scope": ["unusual options", "IV", "gamma exposure", "put/call", "open interest"],
    },
    "fundamental_analysis": {
        "status": "not_yet_trained",
        "trained_now": [],
        "missing_for_full_scope": ["revenue growth", "earnings growth", "debt", "cash flow", "valuation", "earnings surprises"],
    },
    "historical_similarity_analysis": {
        "status": "not_yet_trained_in_this_kernel",
        "trained_now": [],
        "missing_for_full_scope": ["pattern similarity score", "volatility regime similarity", "earnings reaction similarity"],
    },
    "ai_multi_factor_analysis": {
        "status": "partial",
        "trained_now": ["technical", "volume", "volatility", "relative SPY features"],
        "missing_for_full_scope": ["macro", "sentiment", "options", "fundamentals", "events", "institutional activity"],
    },
    "simulation_analysis": {
        "status": "not_trained_in_this_kernel",
        "trained_now": [],
        "missing_for_full_scope": ["Monte Carlo paths", "volatility cone", "future candle path model"],
    },
    "risk_analysis": {
        "status": "partial",
        "trained_now": ["backtest drawdown", "fees", "slippage", "profit factor"],
        "missing_for_full_scope": ["stop-loss zones", "liquidity risk", "portfolio exposure risk"],
    },
    "event_driven_analysis": {
        "status": "not_yet_trained",
        "trained_now": [],
        "missing_for_full_scope": ["earnings", "FDA approvals", "M&A", "Fed meetings", "CPI", "geopolitical news"],
    },
    "quantitative_analysis": {
        "status": "partial",
        "trained_now": ["tree ensemble", "walk-forward validation", "probability buckets", "simple backtest"],
        "missing_for_full_scope": ["factor models", "regression models", "Monte Carlo", "probabilistic price range forecasting"],
    },
    "machine_learning_analysis": {
        "status": "partial",
        "trained_now": ["XGBoost", "LightGBM", "CatBoost"],
        "missing_for_full_scope": ["Random Forest", "LSTM", "GRU", "Transformer", "Temporal Fusion Transformer"],
    },
    "ai_reasoning_engine": {
        "status": "partial",
        "trained_now": ["structured explanation facts in metrics JSON"],
        "missing_for_full_scope": ["LLM-generated institutional narrative using live news/macro/context"],
    },
    "adaptive_learning_system": {
        "status": "not_yet_trained",
        "trained_now": [],
        "missing_for_full_scope": ["model drift tracking", "strategy regime scoring", "prediction feedback loop", "user behavior"],
    },
}


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def macd_hist(close: pd.Series) -> pd.Series:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def bollinger(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / mid.replace(0, np.nan)
    return pct_b, width


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k, d


def build_features(raw: pd.DataFrame, spy: pd.DataFrame | None) -> pd.DataFrame:
    close = raw["Close"]
    high = raw["High"]
    low = raw["Low"]
    volume = raw["Volume"]

    f = pd.DataFrame(index=raw.index)
    f["ret_1"] = close.pct_change(1)
    f["ret_2"] = close.pct_change(2)
    f["ret_3"] = close.pct_change(3)
    f["ret_5"] = close.pct_change(5)
    f["ret_10"] = close.pct_change(10)
    f["ret_20"] = close.pct_change(20)
    f["vol_5"] = f["ret_1"].rolling(5).std()
    f["vol_10"] = f["ret_1"].rolling(10).std()
    f["vol_20"] = f["ret_1"].rolling(20).std()
    f["sma10_gap"] = close / close.rolling(10).mean() - 1
    f["sma20_gap"] = close / close.rolling(20).mean() - 1
    f["sma50_gap"] = close / close.rolling(50).mean() - 1
    f["sma200_gap"] = close / close.rolling(200).mean() - 1
    f["ema20_gap"] = close / close.ewm(span=20, adjust=False).mean() - 1
    f["rsi_7"] = rsi(close, 7)
    f["rsi_14"] = rsi(close, 14)
    f["atr_norm"] = atr(high, low, close, 14) / close.replace(0, np.nan)
    f["macd_hist"] = macd_hist(close)
    f["bb_pctb"], f["bb_width"] = bollinger(close, 20)
    f["stoch_k"], f["stoch_d"] = stochastic(high, low, close)
    f["volume_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)
    f["volume_trend"] = volume / volume.rolling(10).mean() - 1

    dow = pd.Series(raw.index.dayofweek, index=raw.index)
    month = pd.Series(raw.index.month, index=raw.index)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)

    if spy is not None and not spy.empty:
        spy_close = spy["Close"].reindex(raw.index, method="ffill")
        f["spy_ret_5"] = spy_close.pct_change(5)
        f["spy_ret_20"] = spy_close.pct_change(20)
        f["spy_vol_20"] = spy_close.pct_change().rolling(20).std()
        f["rel_spy_5"] = f["ret_5"] - f["spy_ret_5"]
        f["rel_spy_20"] = f["ret_20"] - f["spy_ret_20"]

    # Strong leakage guard: today's target is predicted using features known before today.
    return f.shift(CONFIG["feature_shift"])


def build_labels(raw: pd.DataFrame) -> pd.DataFrame:
    close = raw["Close"]
    horizon = CONFIG["prediction_horizon"]
    fwd_ret = close.shift(-horizon) / close - 1

    # Binary direction label is the honest route to high-confidence 75% buckets.
    binary = (fwd_ret > 0).astype(int)

    # Three-class label remains useful for the app, but 75% is unlikely here.
    th = CONFIG["label_threshold"]
    three = np.select([fwd_ret > th, fwd_ret < -th], [2, 0], default=1)

    return pd.DataFrame({"target_binary": binary, "target_3class": three, "forward_return": fwd_ret}, index=raw.index)


def fetch(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(ticker).history(
            period=CONFIG["period"],
            interval=CONFIG["interval"],
            auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 500:
            print(f"[skip] {ticker}: {len(df)} rows")
            return None
        print(f"[ok] {ticker}: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"[err] {ticker}: {exc}")
        return None


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    print("\nDownloading data")
    spy = fetch("SPY")
    frames = []

    for ticker in CONFIG["tickers"]:
        raw = fetch(ticker)
        if raw is None:
            continue
        try:
            features = build_features(raw, spy)
            labels = build_labels(raw)
            data = features.join(labels).dropna()
            data["ticker"] = ticker
            data["date"] = data.index
            frames.append(data)
        except Exception as exc:
            print(f"[feature error] {ticker}: {exc}")

    if not frames:
        raise RuntimeError("No usable market data was built.")

    data = pd.concat(frames).sort_values(["date", "ticker"]).reset_index(drop=True)
    feature_cols = [
        col for col in data.columns
        if col not in {"target_binary", "target_3class", "forward_return", "ticker", "date"}
    ]

    print(f"\nDataset rows: {len(data):,}")
    print(f"Tickers: {data['ticker'].nunique()}")
    print(f"Features: {len(feature_cols)}")
    print("Binary distribution:")
    print(data["target_binary"].value_counts(normalize=True).sort_index())
    print("Three-class distribution:")
    print(data["target_3class"].value_counts(normalize=True).sort_index())

    data.to_parquet(OUTPUT_DIR / "proof_dataset.parquet")
    return data, feature_cols


def make_xgb(use_gpu: bool, n_classes: int):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    kwargs = {
        "n_estimators": 650,
        "max_depth": 4,
        "learning_rate": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "objective": objective,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "random_state": 42,
        "verbosity": 0,
    }
    if n_classes > 2:
        kwargs["num_class"] = n_classes
        kwargs["eval_metric"] = "mlogloss"
    else:
        kwargs["eval_metric"] = "logloss"
    return XGBClassifier(**kwargs)


def make_lgb(use_gpu: bool, n_classes: int):
    objective = "binary" if n_classes == 2 else "multiclass"
    kwargs = {
        "n_estimators": 700,
        "num_leaves": 48,
        "learning_rate": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 30,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "objective": objective,
        "device": "gpu" if use_gpu else "cpu",
        "random_state": 42,
        "verbose": -1,
    }
    if n_classes > 2:
        kwargs["num_class"] = n_classes
    return LGBMClassifier(**kwargs)


def make_cat(use_gpu: bool, n_classes: int):
    return CatBoostClassifier(
        iterations=650,
        depth=6,
        learning_rate=0.035,
        l2_leaf_reg=5.0,
        loss_function="Logloss" if n_classes == 2 else "MultiClass",
        eval_metric="Accuracy",
        task_type="GPU" if use_gpu else "CPU",
        random_seed=42,
        verbose=0,
    )


def cpu_portable(model, name: str):
    try:
        if name == "xgb":
            model.set_params(device="cpu", tree_method="hist")
            model.get_booster().set_param({"device": "cpu", "tree_method": "hist"})
        elif name == "lgb":
            model.set_params(device="cpu")
            model.booster_.reset_parameter({"device": "cpu"})
        elif name == "cat":
            model.set_params(task_type="CPU")
    except Exception as exc:
        print(f"[warn] CPU portability update failed for {name}: {exc}")
    return model


def ensemble_predict_proba(models: dict, x: pd.DataFrame) -> np.ndarray:
    probas = []
    for model in models.values():
        proba = model.predict_proba(x)
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        probas.append(proba)
    return np.mean(probas, axis=0)


def model_predictions(models: dict, x: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: model.predict(x).astype(int) for name, model in models.items()}


def ensemble_feature_importance(models: dict, feature_cols: list[str]) -> pd.Series:
    parts = []
    for model in models.values():
        values = getattr(model, "feature_importances_", None)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if values.sum() > 0:
            values = values / values.sum()
        parts.append(pd.Series(values, index=feature_cols))
    if not parts:
        return pd.Series(1.0 / len(feature_cols), index=feature_cols)
    return pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)


def human_feature_fact(feature: str, value: float, median: float, std: float) -> str:
    if not np.isfinite(value):
        return f"{feature} was unavailable."
    if not np.isfinite(std) or std == 0:
        return f"{feature}={value:.4f}, near historical median {median:.4f}."
    z = (value - median) / std
    if z >= 1.5:
        relation = "far above"
    elif z >= 0.5:
        relation = "above"
    elif z <= -1.5:
        relation = "far below"
    elif z <= -0.5:
        relation = "below"
    else:
        relation = "near"
    return f"{feature}={value:.4f}, {relation} training median {median:.4f} (z={z:.2f})."


def explain_signal(
    row: pd.Series,
    feature_cols: list[str],
    proba: np.ndarray,
    pred: int,
    individual_preds: dict[str, int],
    train_median: pd.Series,
    train_std: pd.Series,
    importances: pd.Series,
    target_name: str,
) -> dict:
    sorted_proba = np.sort(proba)[::-1]
    margin = float(sorted_proba[0] - sorted_proba[1]) if len(sorted_proba) > 1 else float(sorted_proba[0])
    agreement = float(np.mean([p == pred for p in individual_preds.values()]))
    top_features = importances.head(CONFIG["top_explanation_features"]).index.tolist()
    facts = []
    for feature in top_features:
        facts.append({
            "feature": feature,
            "importance": float(importances.loc[feature]),
            "value": float(row[feature]),
            "training_median": float(train_median.loc[feature]),
            "training_std": float(train_std.loc[feature]),
            "fact": human_feature_fact(
                feature,
                float(row[feature]),
                float(train_median.loc[feature]),
                float(train_std.loc[feature]),
            ),
        })

    return {
        "date": str(pd.Timestamp(row["date"]).date()),
        "ticker": str(row["ticker"]),
        "target": target_name,
        "prediction": int(pred),
        "actual": int(row[target_name]),
        "correct": bool(pred == int(row[target_name])),
        "confidence": float(proba[pred]),
        "probability_margin": margin,
        "model_agreement": agreement,
        "model_votes": {name: int(vote) for name, vote in individual_preds.items()},
        "forward_return": float(row["forward_return"]),
        "why_confident": [
            f"Ensemble assigned {proba[pred]:.2%} probability to class {pred}.",
            f"Top-vs-second probability margin was {margin:.2%}.",
            f"{agreement:.0%} of base models agreed with the ensemble prediction.",
            "Top feature facts are measured against the training window only.",
        ],
        "top_feature_facts": facts,
    }


def explain_high_confidence_signals(
    test: pd.DataFrame,
    feature_cols: list[str],
    proba: np.ndarray,
    models: dict,
    train_median: pd.Series,
    train_std: pd.Series,
    importances: pd.Series,
    target_name: str,
) -> list[dict]:
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    threshold = CONFIG["explanation_confidence_threshold"]
    candidates = np.where(conf >= threshold)[0]
    if len(candidates) == 0:
        return []

    candidates = candidates[np.argsort(conf[candidates])[::-1]]
    candidates = candidates[: CONFIG["max_signal_explanations_per_window"]]
    votes = model_predictions(models, test[feature_cols].iloc[candidates])
    explanations = []
    for local_pos, row_idx in enumerate(candidates):
        row = test.iloc[row_idx]
        individual = {name: values[local_pos] for name, values in votes.items()}
        explanations.append(
            explain_signal(
                row=row,
                feature_cols=feature_cols,
                proba=proba[row_idx],
                pred=int(pred[row_idx]),
                individual_preds=individual,
                train_median=train_median,
                train_std=train_std,
                importances=importances,
                target_name=target_name,
            )
        )
    return explanations


def fit_ensemble(x_train: pd.DataFrame, y_train: pd.Series, n_classes: int, use_gpu: bool) -> dict:
    models = {
        "xgb": make_xgb(use_gpu, n_classes),
        "lgb": make_lgb(use_gpu, n_classes),
        "cat": make_cat(use_gpu, n_classes),
    }
    for name, model in models.items():
        print(f"  fitting {name}")
        model.fit(x_train, y_train)
    return models


def evaluate_predictions(y_true: pd.Series, proba: np.ndarray, returns: pd.Series) -> dict:
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    result = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "confidence_buckets": [],
    }

    for threshold in CONFIG["confidence_thresholds"]:
        mask = conf >= threshold
        if mask.sum() == 0:
            bucket = {"threshold": threshold, "coverage": 0.0, "rows": 0, "accuracy": None, "f1_macro": None}
        else:
            bucket = {
                "threshold": threshold,
                "coverage": float(mask.mean()),
                "rows": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], pred[mask])),
                "f1_macro": float(f1_score(y_true[mask], pred[mask], average="macro")),
            }
        result["confidence_buckets"].append(bucket)

    cost = (CONFIG["fee_bps"] + CONFIG["slippage_bps"]) / 10000.0
    if proba.shape[1] == 2:
        direction = np.where(pred == 1, 1, -1)
        strategy_ret = direction * returns.to_numpy() - cost
        high_conf = conf >= 0.70
        high_conf_ret = np.where(high_conf, strategy_ret, 0.0)
        result["backtest"] = backtest_stats(strategy_ret, "all_binary_signals")
        result["high_conf_backtest"] = backtest_stats(high_conf_ret, "confidence_70_plus")

    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, zero_division=0)
    result["per_class"] = {
        str(i): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(support))
    }
    return result


def backtest_stats(returns: np.ndarray, name: str) -> dict:
    equity = np.cumprod(1 + np.nan_to_num(returns, nan=0.0))
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    wins = returns > 0
    losses = returns < 0
    gross_win = returns[wins].sum() if wins.any() else 0.0
    gross_loss = abs(returns[losses].sum()) if losses.any() else 0.0
    daily_std = returns.std()
    return {
        "name": name,
        "trades": int(np.count_nonzero(returns)),
        "total_return": float(equity[-1] - 1) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "win_rate": float(wins.mean()) if len(returns) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "sharpe_like": float((returns.mean() / daily_std) * np.sqrt(252)) if daily_std > 0 else None,
    }


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    majority = int(y_train.value_counts().idxmax())
    pred = np.full(len(y_test), majority)
    return {
        "majority_class": majority,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }


def walk_forward_windows(data: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.Series(pd.to_datetime(data["date"]).sort_values().unique())
    min_date = dates.iloc[0]
    max_date = dates.iloc[-1]
    windows = []
    for start in pd.date_range("2022-01-01", max_date, freq="12MS"):
        end = start + pd.DateOffset(months=12)
        train_end = start - pd.Timedelta(days=CONFIG["prediction_horizon"] + 3)
        if train_end <= min_date or start >= max_date:
            continue
        windows.append((pd.Timestamp(train_end), pd.Timestamp(start), pd.Timestamp(min(end, max_date))))
    return windows


def run_walk_forward(data: pd.DataFrame, features: list[str], target_col: str, n_classes: int) -> tuple[list[dict], dict]:
    reports = []
    final_models = None
    final_feature_cols = features
    use_gpu = CONFIG["use_gpu"]

    for i, (train_end, test_start, test_end) in enumerate(walk_forward_windows(data), start=1):
        train = data[pd.to_datetime(data["date"]) <= train_end]
        test = data[(pd.to_datetime(data["date"]) >= test_start) & (pd.to_datetime(data["date"]) < test_end)]
        if len(train) < 3000 or len(test) < 300:
            continue

        print(f"\nWindow {i}: train <= {train_end.date()} | test {test_start.date()} to {test_end.date()}")
        x_train = train[final_feature_cols]
        y_train = train[target_col].astype(int)
        x_test = test[final_feature_cols]
        y_test = test[target_col].astype(int)

        models = fit_ensemble(x_train, y_train, n_classes, use_gpu)
        proba = ensemble_predict_proba(models, x_test)
        report = evaluate_predictions(y_test.reset_index(drop=True), proba, test["forward_return"].reset_index(drop=True))
        importances = ensemble_feature_importance(models, final_feature_cols)
        report["algorithm"] = {
            "model_family": "equal-weight soft-vote ensemble",
            "base_models": ["XGBoost", "LightGBM", "CatBoost"],
            "confidence_definition": "maximum ensemble class probability",
            "confidence_margin_definition": "top class probability minus second-highest class probability",
            "agreement_definition": "share of base models voting for the ensemble class",
            "feature_driver_method": "mean normalized tree feature importance across base models",
            "leakage_control": "features are shifted by one day and evaluated only on future walk-forward windows",
        }
        report["global_top_features"] = [
            {"feature": feature, "importance": float(score)}
            for feature, score in importances.head(15).items()
        ]
        report["high_confidence_explanations"] = explain_high_confidence_signals(
            test=test.reset_index(drop=True),
            feature_cols=final_feature_cols,
            proba=proba,
            models=models,
            train_median=x_train.median(),
            train_std=x_train.std().replace(0, np.nan),
            importances=importances,
            target_name=target_col,
        )
        report["window"] = {
            "train_end": train_end.date().isoformat(),
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
        }
        report["baseline"] = majority_baseline(y_train, y_test)
        reports.append(report)
        final_models = models

        best_bucket = max(
            [b for b in report["confidence_buckets"] if b["accuracy"] is not None],
            key=lambda b: (b["accuracy"], b["coverage"]),
        )
        print(
            f"  all accuracy={report['accuracy']:.4f} f1={report['f1_macro']:.4f} | "
            f"best bucket >= {best_bucket['threshold']:.2f}: "
            f"accuracy={best_bucket['accuracy']:.4f}, coverage={best_bucket['coverage']:.2%}"
        )

    if final_models is None:
        raise RuntimeError("No walk-forward windows were trainable.")

    final_models = {name: cpu_portable(model, name) for name, model in final_models.items()}
    artifact = {
        "models": final_models,
        "features": final_feature_cols,
        "target": target_col,
        "classes": n_classes,
        "ensemble": "equal_weight_soft_vote",
        "config": CONFIG,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return reports, artifact


def aggregate_reports(reports: list[dict]) -> dict:
    all_acc = [r["accuracy"] for r in reports]
    all_f1 = [r["f1_macro"] for r in reports]
    bucket_summary = []
    for threshold in CONFIG["confidence_thresholds"]:
        buckets = [b for r in reports for b in r["confidence_buckets"] if b["threshold"] == threshold and b["accuracy"] is not None]
        if not buckets:
            continue
        rows = sum(b["rows"] for b in buckets)
        weighted_acc = sum(b["accuracy"] * b["rows"] for b in buckets) / rows
        weighted_f1 = sum(b["f1_macro"] * b["rows"] for b in buckets) / rows
        coverage = np.mean([b["coverage"] for b in buckets])
        bucket_summary.append({
            "threshold": threshold,
            "rows": int(rows),
            "avg_coverage": float(coverage),
            "weighted_accuracy": float(weighted_acc),
            "weighted_f1_macro": float(weighted_f1),
        })

    accepted_75 = [
        b for b in bucket_summary
        if b["weighted_accuracy"] >= 0.75 and b["avg_coverage"] >= 0.05 and b["rows"] >= 250
    ]

    return {
        "windows": len(reports),
        "mean_accuracy": float(np.mean(all_acc)),
        "min_window_accuracy": float(np.min(all_acc)),
        "mean_f1_macro": float(np.mean(all_f1)),
        "confidence_bucket_summary": bucket_summary,
        "accepted_75_claim": bool(accepted_75),
        "accepted_75_buckets": accepted_75,
        "claim_text": (
            "75% high-confidence accuracy supported by walk-forward holdout buckets."
            if accepted_75
            else "75% claim is not supported by current walk-forward evidence."
        ),
    }


def main() -> None:
    print("=" * 78)
    print("MarketVision AI - 75% Accuracy Proof Kernel")
    print("=" * 78)

    data, features = build_dataset()

    print("\nLeakage guard:")
    print(f"  feature_shift={CONFIG['feature_shift']} day")
    print(f"  prediction_horizon={CONFIG['prediction_horizon']} trading days")
    print("  walk-forward tests use only future test windows")
    print("  training windows are purged before test start")

    print("\nRunning binary direction proof")
    binary_reports, binary_artifact = run_walk_forward(data, features, "target_binary", 2)
    binary_summary = aggregate_reports(binary_reports)

    print("\nRunning three-class app-model proof")
    three_reports, three_artifact = run_walk_forward(data, features, "target_3class", 3)
    three_summary = aggregate_reports(three_reports)

    version = datetime.now(timezone.utc).strftime("proof_%Y%m%d_%H%M%S")
    joblib.dump(binary_artifact, OUTPUT_DIR / f"{version}_binary_high_conf.joblib")
    joblib.dump(three_artifact, OUTPUT_DIR / f"{version}_three_class.joblib")

    payload = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": CONFIG,
        "analysis_dimension_coverage": ANALYSIS_DIMENSION_COVERAGE,
        "prediction_disclaimer": (
            "Outputs are probability-based market simulations and AI-generated financial intelligence, "
            "not guaranteed outcomes or financial advice."
        ),
        "binary_direction": {
            "summary": binary_summary,
            "windows": binary_reports,
        },
        "three_class": {
            "summary": three_summary,
            "windows": three_reports,
        },
        "final_verdict": {
            "all_signal_75_supported": False,
            "high_confidence_75_supported": binary_summary["accepted_75_claim"],
            "recommended_platform_claim": (
                "MarketVision AI reached 75%+ accuracy on high-confidence walk-forward signals."
                if binary_summary["accepted_75_claim"]
                else "MarketVision AI did not yet prove 75% high-confidence accuracy; improve data/features before claiming it."
            ),
        },
    }

    metrics_path = OUTPUT_DIR / f"{version}_proof_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 78)
    print("FINAL PROOF SUMMARY")
    print("=" * 78)
    print("Binary all-signal mean accuracy:", round(binary_summary["mean_accuracy"], 4))
    print("Binary all-signal mean F1:", round(binary_summary["mean_f1_macro"], 4))
    print("Binary 75% high-confidence claim:", binary_summary["claim_text"])
    print("Three-class mean accuracy:", round(three_summary["mean_accuracy"], 4))
    print("Three-class mean F1:", round(three_summary["mean_f1_macro"], 4))
    print(f"Artifacts saved in {OUTPUT_DIR}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
