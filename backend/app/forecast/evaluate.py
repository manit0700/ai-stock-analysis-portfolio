"""Scoring forecasts, always beside a baseline, with honest uncertainty.

Three things this module is careful about:

* **Accuracy is reported but never used to rank.** With about 55% of weeks rising, "always up"
  scores 55% and tells you nothing. Ranking uses the Brier score (mean squared error of the
  probability), and the headline is skill relative to the constant-base-rate baseline.
* **Rows are not independent.** Forty stocks on one date move together, and five-session labels
  overlap. So intervals come from resampling whole calendar weeks, not rows. Treating rows as
  independent would make the intervals several times too narrow.
* **Nothing here says a strategy makes money.** Classification skill ignores the size of moves and
  every cost. Profitability is a backtest question (``app.backtesting``).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Standard metrics for probabilities ``p`` of the positive class against outcomes ``y`` (0/1)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    predicted = (p >= threshold).astype(int)
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    positives, negatives = tp + fn, tn + fp
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    both = len(np.unique(y)) == 2
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": n,
        "base_rate": round(float(y.mean()), 6),
        "majority_class_accuracy": round(float(max(y.mean(), 1 - y.mean())), 6),
        "accuracy": round((tp + tn) / n, 6),
        "balanced_accuracy": round((recall + specificity) / 2, 6) if recall is not None and specificity is not None else None,
        "precision": None if precision is None else round(precision, 6),
        "recall": None if recall is None else round(recall, 6),
        "f1": None if f1 is None else round(f1, 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 6) if both and np.ptp(p) > 0 else (0.5 if both else None),
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "log_loss": round(float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))), 6),
        "predicted_positive_share": round(float(predicted.mean()), 6),
        "mean_probability": round(float(p.mean()), 6),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray, p_reference: np.ndarray) -> float:
    """1 - Brier(model) / Brier(reference). Positive means better than the reference."""
    reference = brier(y, p_reference)
    return float(1.0 - brier(y, p) / reference) if reference > EPS else 0.0


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Calibration table on equal-count bins, and the expected calibration error (ECE)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"ece": None, "bins": []}
    order = np.argsort(p, kind="stable")
    rows, ece = [], 0.0
    for chunk in np.array_split(order, min(bins, max(1, len(y) // 50))):
        if len(chunk) == 0:
            continue
        mean_p, observed = float(p[chunk].mean()), float(y[chunk].mean())
        ece += abs(mean_p - observed) * len(chunk) / len(y)
        rows.append({"n": int(len(chunk)), "p_min": round(float(p[chunk].min()), 4), "p_max": round(float(p[chunk].max()), 4),
                     "mean_predicted": round(mean_p, 4), "observed_rate": round(observed, 4)})
    return {"ece": round(ece, 6), "bins": rows}


def week_blocks(dates: pd.DatetimeIndex) -> np.ndarray:
    """An integer per row identifying its ISO calendar week: the unit that is resampled."""
    iso = pd.DatetimeIndex(dates).isocalendar()
    return (iso["year"].to_numpy(int) * 100 + iso["week"].to_numpy(int)).astype(np.int64)


def block_bootstrap(
    dates: pd.DatetimeIndex,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Interval for ``statistic(row_positions)`` by resampling calendar weeks with replacement."""
    blocks = week_blocks(dates)
    unique, inverse = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    point = float(statistic(np.arange(len(blocks))))
    samples = np.empty(n_resamples)
    for b in range(n_resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        samples[b] = statistic(np.concatenate([members[k] for k in chosen]))
    samples = samples[np.isfinite(samples)]
    alpha = (1 - level) / 2
    return {
        "point": round(point, 6),
        "low": round(float(np.quantile(samples, alpha)), 6) if len(samples) else None,
        "high": round(float(np.quantile(samples, 1 - alpha)), 6) if len(samples) else None,
        "level": level,
        "blocks": int(len(unique)),
        "resamples": int(len(samples)),
        "method": "calendar-week block bootstrap",
        "share_above_zero": round(float((samples > 0).mean()), 4) if len(samples) else None,
    }


def calibration_bands(dates: pd.DatetimeIndex, y: np.ndarray, p: np.ndarray, *, bands: int = 5, tolerance: float = 0.03, n_resamples: int = 500, seed: int = 0) -> dict[str, Any]:
    """Is each band of stated probability matched by what happened, allowing for sampling noise?

    Rows are sorted by probability and cut into ``bands`` equal groups. For each, the gap between
    the observed rate and the mean stated probability gets a calendar-week bootstrap interval. A
    band counts as miscalibrated only if the gap is larger than ``tolerance`` *and* its interval
    excludes zero: a gap that noise alone explains is not held against a model, and neither is a
    gap too small to matter.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    dates = pd.DatetimeIndex(dates)
    order = np.argsort(p, kind="stable")
    rows_out, bad = [], 0
    for k, chunk in enumerate(np.array_split(order, bands)):
        if len(chunk) < 50:
            continue
        chunk = np.sort(chunk)
        gap_values = y[chunk] - p[chunk]
        interval = block_bootstrap(dates[chunk], lambda rows: float(gap_values[rows].mean()), n_resamples=n_resamples, seed=seed + k)
        miscalibrated = abs(interval["point"]) > tolerance and interval["low"] is not None and (interval["low"] > 0 or interval["high"] < 0)
        bad += int(miscalibrated)
        rows_out.append({
            "n": int(len(chunk)), "mean_predicted": round(float(p[chunk].mean()), 4), "observed_rate": round(float(y[chunk].mean()), 4),
            "gap": interval["point"], "gap_low": interval["low"], "gap_high": interval["high"], "miscalibrated": bool(miscalibrated),
        })
    return {"bands": rows_out, "miscalibrated_bands": bad, "tolerance": tolerance,
            "rule": f"a band is miscalibrated when observed and stated rates differ by more than {tolerance} and the 95% interval of the gap excludes zero"}


def skill_intervals(dates: pd.DatetimeIndex, y: np.ndarray, p: np.ndarray, p_reference: np.ndarray, *, n_resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Bootstrap intervals for Brier skill against a reference and for ROC-AUC."""
    y, p, p_reference = np.asarray(y, int), np.asarray(p, float), np.asarray(p_reference, float)

    def skill(rows: np.ndarray) -> float:
        return brier_skill(y[rows], p[rows], p_reference[rows])

    def auc(rows: np.ndarray) -> float:
        return float(roc_auc_score(y[rows], p[rows])) if len(np.unique(y[rows])) == 2 and np.ptp(p[rows]) > 0 else float("nan")

    return {
        "brier_skill": block_bootstrap(dates, skill, n_resamples=n_resamples, seed=seed),
        "roc_auc": block_bootstrap(dates, auc, n_resamples=n_resamples, seed=seed + 1),
    }


def by_segment(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, p_reference: np.ndarray, labels: pd.Series, min_rows: int = 200) -> list[dict[str, Any]]:
    """Metrics for each value of ``labels`` (a regime, a year), with skill against the reference."""
    out = []
    labels = pd.Series(np.asarray(labels), index=frame.index)
    for value in sorted(labels.dropna().unique(), key=str):
        rows = np.flatnonzero((labels == value).to_numpy())
        if len(rows) < min_rows:
            continue
        m = classification_metrics(y[rows], p[rows])
        out.append({
            "segment": str(value), "n": m["n"], "base_rate": m["base_rate"], "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "roc_auc": m["roc_auc"], "brier": m["brier"], "brier_reference": round(brier(y[rows], p_reference[rows]), 6),
            "brier_skill": round(brier_skill(y[rows], p[rows], p_reference[rows]), 6),
        })
    return out


def regime_labels(frame: pd.DataFrame, vix_cuts: tuple[float, float] | None) -> dict[str, pd.Series]:
    """Market-regime labels for rows of a dataset frame. VIX cut points must come from development data."""
    out: dict[str, pd.Series] = {"year": pd.Series(frame.index.year, index=frame.index)}
    if "spy_sma_gap_200" in frame.columns:
        out["market_trend"] = pd.Series(np.where(frame["spy_sma_gap_200"] >= 0, "SPY above 200-session average", "SPY below 200-session average"), index=frame.index)
    if "vix_level" in frame.columns and vix_cuts is not None:
        low, high = vix_cuts
        out["volatility"] = pd.Series(
            np.select([frame["vix_level"] < low, frame["vix_level"] < high], [f"VIX below {low:.1f}", f"VIX {low:.1f} to {high:.1f}"], default=f"VIX above {high:.1f}"),
            index=frame.index,
        )
    return out
