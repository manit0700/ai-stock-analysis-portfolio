"""Checks that no information from the future reaches a model, run on every evaluation.

Unit tests prove the feature code does not look ahead on synthetic prices. These checks look at
the actual dataset of a run, because a leak can also be introduced by how the data is assembled
or split. A failed check blocks promotion.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.features import schema
from app.features.context import load_context
from app.features.pipeline import adjusted_close, build_features
from app.forecast.dataset import META_COLUMNS, Dataset
from app.forecast.splits import Fold, check_fold

SUSPICIOUS_CORRELATION = 0.20


def _result(check: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"check": check, "passed": bool(passed), "detail": detail, **extra}


def columns_are_features(dataset: Dataset) -> dict[str, Any]:
    known = set(schema.all_feature_names())
    foreign = [c for c in dataset.feature_names if c not in known]
    stray = [c for c in dataset.frame.columns if c not in known and c not in META_COLUMNS]
    labelish = [c for c in dataset.feature_names if any(word in c.lower() for word in ("target", "fwd", "future", "label"))]
    ok = not foreign and not stray and not labelish
    return _result("model inputs are schema features only", ok, "every input column is defined in the feature schema" if ok else f"unexpected columns: {foreign + stray + labelish}")


def no_implausible_correlation(dataset: Dataset, limit: float = SUSPICIOUS_CORRELATION) -> dict[str, Any]:
    """A daily feature that correlates strongly with the future return is almost certainly a leak.

    Published return predictors have rank correlations of a few hundredths with future returns.
    The default limit of 0.20 is several times anything credible for real prices.
    """
    frame = dataset.frame
    correlations = frame[dataset.feature_names].corrwith(frame["fwd_return"], method="spearman").abs().dropna()
    worst = correlations.sort_values(ascending=False).head(3)
    ok = bool((correlations < limit).all())
    listing = ", ".join(f"{name} {value:.3f}" for name, value in worst.items())
    return _result("no feature is implausibly correlated with the future return", ok, f"largest absolute rank correlations: {listing} (limit {limit})",
                   max_abs_correlation=round(float(correlations.max()), 4) if len(correlations) else None)


def folds_are_purged(dataset: Dataset, folds: list[Fold], embargo: int) -> dict[str, Any]:
    rows = [check_fold(dataset.frame, fold, dataset.horizon, embargo) for fold in folds]
    ok = all(row["passed"] for row in rows)
    smallest = min((row["sessions_between_last_label_and_test"] or 0) for row in rows) if rows else 0
    return _result("training label windows end before every test period", ok,
                   f"{len(rows)} splits checked; smallest gap between the last training label and the first test date is {smallest} sessions (must exceed {embargo})", splits=rows)


def no_duplicate_rows(dataset: Dataset) -> dict[str, Any]:
    keys = pd.MultiIndex.from_arrays([dataset.frame["symbol"], dataset.frame.index])
    duplicates = int(keys.duplicated().sum())
    return _result("no (symbol, date) appears twice", duplicates == 0, f"{duplicates} duplicated rows")


def rows_recompute_from_the_past(dataset: Dataset, market_data: Any, *, samples: int = 12, seed: int = 0) -> dict[str, Any]:
    """Recompute sampled rows using only bars up to the row's date, and the label from the bars after it."""
    rng = np.random.default_rng(seed)
    frame = dataset.frame
    picks = rng.choice(len(frame), size=min(samples, len(frame)), replace=False)
    worst_feature, worst_label, checked = 0.0, 0.0, 0
    shared: dict[str, Any] = {}
    for position in picks:
        row = frame.iloc[position]
        day = frame.index[position]
        symbol = row["symbol"]
        history = market_data.history(symbol, interval="1d").data
        context = load_context(market_data, symbol, shared=shared)
        past = history.loc[:day]
        truncated = type(context)(
            spy=None if context.spy is None else context.spy.loc[:day],
            qqq=None if context.qqq is None else context.qqq.loc[:day],
            vix=None if context.vix is None else context.vix.loc[:day],
            sector=None if context.sector is None else context.sector.loc[:day],
            sector_symbol=context.sector_symbol,
            us10y=None if context.us10y is None else context.us10y.loc[:day],
            yield_curve=None if context.yield_curve is None else context.yield_curve.loc[:day],
        )
        recomputed = build_features(past, truncated).iloc[-1][dataset.feature_names].to_numpy(float)
        stored = row[dataset.feature_names].to_numpy(float)
        scale = np.maximum(np.abs(stored), 1e-6)
        worst_feature = max(worst_feature, float(np.nanmax(np.abs(recomputed - stored) / scale)))
        close = adjusted_close(history)
        end = pd.Timestamp(row["label_end"])
        label = float(close.loc[end] / close.loc[day] - 1.0)
        worst_label = max(worst_label, abs(label - float(row["fwd_return"])))
        checked += 1
    ok = worst_feature < 1e-6 and worst_label < 1e-9
    return _result("sampled rows are reproducible from past bars alone", ok,
                   f"{checked} rows recomputed with every later bar removed; largest relative feature difference {worst_feature:.2e}, largest label difference {worst_label:.2e}")


def shuffled_label_control(aucs: list[float], *, tolerance: float = 0.02) -> dict[str, Any]:
    """Train on shuffled labels, several times. On average such a model must score like a coin.

    ``aucs`` are walk-forward ROC-AUCs of the selected model (computed within each validation
    block and averaged), one per independent shuffle of its training labels. A single shuffle can land above or below 0.5 by chance (the fit to
    noise is some arbitrary function of the inputs), so the test is on the mean: it must be within
    ``tolerance`` of 0.5, or within three standard errors of it.
    """
    values = np.asarray(list(aucs), dtype=float)
    mean = float(values.mean())
    error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    ok = abs(mean - 0.5) <= max(tolerance, 3 * error)
    listing = ", ".join(f"{v:.4f}" for v in values)
    return _result("a model trained on shuffled labels cannot predict", ok,
                   f"validation ROC-AUC over {len(values)} shuffles of the training labels: {listing}; mean {mean:.4f} (must be within {max(tolerance, 3 * error):.3f} of 0.5)",
                   mean_roc_auc=round(mean, 4), shuffles=[round(float(v), 4) for v in values])
