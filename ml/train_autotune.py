from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_10",
    "volatility_10",
    "volatility_20",
    "volume_change",
    "sma_20_gap",
    "sma_50_gap",
    "rsi_14",
    "macd_hist",
]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_ticker_frame(ticker: str, period: str, interval: str, horizon: int) -> pd.DataFrame:
    raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    frame = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    close = frame["Close"]
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    frame["ticker"] = ticker
    frame["return_1"] = close.pct_change()
    frame["return_5"] = close.pct_change(5)
    frame["return_10"] = close.pct_change(10)
    frame["volatility_10"] = frame["return_1"].rolling(10).std()
    frame["volatility_20"] = frame["return_1"].rolling(20).std()
    frame["volume_change"] = frame["Volume"].pct_change()
    frame["sma_20_gap"] = close / close.rolling(20).mean() - 1
    frame["sma_50_gap"] = close / close.rolling(50).mean() - 1
    frame["rsi_14"] = compute_rsi(close)
    frame["macd_hist"] = macd - signal
    frame["future_return"] = close.shift(-horizon) / close - 1
    frame["target"] = np.select(
        [frame["future_return"] > 0.015, frame["future_return"] < -0.015],
        [2, 0],
        default=1,
    )
    return frame.dropna()


def build_dataset(config: dict[str, Any]) -> pd.DataFrame:
    frames = [
        fetch_ticker_frame(
            ticker=ticker,
            period=config["period"],
            interval=config["interval"],
            horizon=int(config["prediction_horizon"]),
        )
        for ticker in config["tickers"]
    ]
    return pd.concat(frames).sort_index()


def build_model(model_family: str, trial: optuna.Trial | None = None):
    if model_family == "xgboost" and XGBClassifier is not None:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 700) if trial else 350,
            "max_depth": trial.suggest_int("max_depth", 2, 8) if trial else 4,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True) if trial else 0.05,
            "subsample": trial.suggest_float("subsample", 0.6, 1.0) if trial else 0.85,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0) if trial else 0.85,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
        }
        return XGBClassifier(**params)

    if model_family == "lightgbm" and LGBMClassifier is not None:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 700) if trial else 350,
            "num_leaves": trial.suggest_int("num_leaves", 16, 96) if trial else 31,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True) if trial else 0.05,
            "subsample": trial.suggest_float("subsample", 0.6, 1.0) if trial else 0.85,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0) if trial else 0.85,
            "random_state": 42,
        }
        return LGBMClassifier(**params)

    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500) if trial else 250,
        max_depth=trial.suggest_int("max_depth", 3, 14) if trial else 8,
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8) if trial else 3,
        random_state=42,
        n_jobs=-1,
    )


def time_split_score(frame: pd.DataFrame, model_family: str, trial: optuna.Trial | None = None) -> float:
    x = frame[FEATURE_COLUMNS]
    y = frame["target"]
    splitter = TimeSeriesSplit(n_splits=4)
    scores = []
    for train_idx, valid_idx in splitter.split(x):
        model = build_model(model_family, trial)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(x.iloc[valid_idx])
        scores.append(f1_score(y.iloc[valid_idx], preds, average="macro"))
    return float(np.mean(scores))


def train_final(frame: pd.DataFrame, model_family: str, params: dict[str, Any] | None):
    x = frame[FEATURE_COLUMNS]
    y = frame["target"]
    split = int(len(frame) * 0.82)
    model = build_model(model_family, None)
    if params and hasattr(model, "set_params"):
        model.set_params(**params)
    model.fit(x.iloc[:split], y.iloc[:split])
    preds = model.predict(x.iloc[split:])
    return model, {
        "accuracy": float(accuracy_score(y.iloc[split:], preds)),
        "f1_macro": float(f1_score(y.iloc[split:], preds, average="macro")),
        "direction_mae": float(mean_absolute_error(y.iloc[split:], preds)),
        "test_rows": int(len(y.iloc[split:])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.example.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_dataset(config)
    dataset_path = output_dir / "training_dataset.parquet"
    frame.to_parquet(dataset_path)

    model_family = config.get("model_family", "xgboost")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: time_split_score(frame, model_family, trial), n_trials=int(config["n_trials"]))

    model, metrics = train_final(frame, model_family, study.best_params)
    version = datetime.now(timezone.utc).strftime("model_%Y%m%d_%H%M%S")
    model_path = output_dir / f"{version}.joblib"
    metrics_path = output_dir / f"{version}_metrics.json"

    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "config": config}, model_path)
    metrics_payload = {
        "version": version,
        "model_family": model_family,
        "best_params": study.best_params,
        "cross_validation_f1_macro": study.best_value,
        "holdout_metrics": metrics,
        "dataset_path": str(dataset_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
