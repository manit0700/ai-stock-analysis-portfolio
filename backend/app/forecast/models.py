"""The candidate models, simplest first.

Two baselines that use no model at all, then five standard classifiers. Each candidate has a
small, fixed grid of settings. The grid is deliberately coarse: with a signal this weak, a large
search mostly finds noise, and every extra setting tried is another chance to fit the validation
data by luck.

Neural networks are not included. They would be justified only if one of these models showed a
real out-of-sample improvement that a more flexible model might extend; see the evaluation report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PRIOR, TRAILING = "baseline_prior", "baseline_trailing_frequency"
N_JOBS = 2


class PriorClassifier(ClassifierMixin, BaseEstimator):
    """Always predicts the share of positive labels seen in training. The baseline to beat."""

    def fit(self, X, y):  # noqa: N803
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):  # noqa: N803
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])

    def predict(self, X):  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TrailingFrequencyClassifier(ClassifierMixin, BaseEstimator):
    """Predicts each stock's own recent history: the share of its last 252 completed windows that rose.

    Uses the ``trail_up_rate`` column only, shrunk half-way towards the training base rate so that
    one unusual year does not produce an extreme probability.
    """

    def __init__(self, shrink: float = 0.5):
        self.shrink = shrink

    def fit(self, X, y):  # noqa: N803
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):  # noqa: N803
        trail = np.asarray(X["trail_up_rate"] if isinstance(X, pd.DataFrame) else X, dtype=float).reshape(-1)
        trail = np.where(np.isnan(trail), self.rate_, trail)
        p = np.clip(self.shrink * self.rate_ + (1 - self.shrink) * trail, 0.02, 0.98)
        return np.column_stack([1 - p, p])

    def predict(self, X):  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    make: Callable[[dict[str, Any], int], Any]
    grid: tuple[dict[str, Any], ...]
    is_baseline: bool = False
    uses: str = "features"  # "features" or "trail" (the baseline input column)


def _logreg(params: dict, seed: int):
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=params["C"], max_iter=1000, random_state=seed)),
    ])


def _forest(params: dict, seed: int):
    return RandomForestClassifier(
        n_estimators=200, max_depth=params["max_depth"], min_samples_leaf=200, max_features="sqrt",
        max_samples=0.3, n_jobs=N_JOBS, random_state=seed,
    )


def _gbm(params: dict, seed: int):
    return GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=params["max_depth"], subsample=0.3, max_features=0.5,
        min_samples_leaf=200, random_state=seed,
    )


def _hist(params: dict, seed: int):
    return HistGradientBoostingClassifier(
        max_iter=params["max_iter"], learning_rate=params["learning_rate"], max_depth=3, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


def _lightgbm(params: dict, seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300, learning_rate=0.02, num_leaves=params["num_leaves"], min_child_samples=200, subsample=0.5,
        subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_jobs=N_JOBS, random_state=seed, verbose=-1,
    )


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:  # noqa: BLE001 - ImportError, or a missing system library
        return False
    return True


def candidates(*, quick: bool = False, include_lightgbm: bool | None = None) -> list[Candidate]:
    """Every candidate, baselines first. ``quick`` keeps one setting per model (for tests)."""
    out = [
        Candidate(PRIOR, "Majority-class baseline (constant base rate)", lambda p, s: PriorClassifier(), ({},), is_baseline=True),
        Candidate(TRAILING, "Historical-return baseline (stock's trailing 252-session frequency)", lambda p, s: TrailingFrequencyClassifier(), ({},), is_baseline=True, uses="trail"),
        Candidate("logistic_regression", "Logistic regression (standardised inputs)", _logreg, ({"C": 0.001}, {"C": 0.01}, {"C": 0.1}, {"C": 1.0})),
        Candidate("random_forest", "Random forest", _forest, ({"max_depth": 4}, {"max_depth": 8})),
        Candidate("gradient_boosting", "Gradient boosting (scikit-learn)", _gbm, ({"max_depth": 2}, {"max_depth": 3})),
        Candidate(
            "hist_gradient_boosting", "Histogram gradient boosting (scikit-learn)", _hist,
            ({"max_iter": 100, "learning_rate": 0.02}, {"max_iter": 100, "learning_rate": 0.05}, {"max_iter": 300, "learning_rate": 0.02}),
        ),
    ]
    use_lgbm = lightgbm_available() if include_lightgbm is None else include_lightgbm
    if use_lgbm:
        out.append(Candidate("lightgbm", "LightGBM", _lightgbm, ({"num_leaves": 4}, {"num_leaves": 8})))
    if quick:
        out = [Candidate(c.name, c.label, c.make, c.grid[:1], c.is_baseline, c.uses) for c in out]
    return out


def inputs_for(candidate: Candidate, frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """The columns a candidate is allowed to see."""
    return frame[["trail_up_rate"]] if candidate.uses == "trail" else frame[feature_names]


def predict_up(estimator: Any, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
    """Probability of the positive class, clipped away from exactly 0 and 1."""
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    column = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return np.clip(proba[:, column].astype(float), 1e-6, 1 - 1e-6)
