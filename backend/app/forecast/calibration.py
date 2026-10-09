"""Making a stated probability mean what it says.

A calibrator maps a model's raw probability to one that matches observed frequencies. It is
fitted on out-of-fold predictions from the walk-forward validation (never on the holdout, never
on rows the model was trained on), and the choice between methods is itself made on later
validation data than the method was fitted on.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.forecast.evaluate import brier

METHODS = ("none", "sigmoid", "isotonic")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """``none`` returns the input; ``sigmoid`` is Platt scaling; ``isotonic`` is a monotone step fit."""

    def __init__(self, method: str = "none") -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown calibration method {method!r}.")
        self.method = method
        self._model: Any = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "sigmoid":
            out = self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            out = self._model.predict(p)
        else:
            out = p
        return np.clip(out, 1e-6, 1 - 1e-6)


def choose_calibration(p_oof: np.ndarray, y_oof: np.ndarray, order: np.ndarray, *, fit_share: float = 0.6) -> tuple[Calibrator, dict[str, Any]]:
    """Pick a method on time-ordered out-of-fold predictions, then refit it on all of them.

    ``order`` sorts the rows by date. Each method is fitted on the earliest ``fit_share`` and scored
    (Brier) on the rest. A method must beat "none" to be chosen; ties keep "none".
    """
    p, y = np.asarray(p_oof, float)[order], np.asarray(y_oof, int)[order]
    cut = int(len(p) * fit_share)
    scores: dict[str, float] = {}
    for method in METHODS:
        if method != "none" and (cut < 200 or len(np.unique(y[:cut])) < 2):
            continue
        scores[method] = brier(y[cut:], Calibrator(method).fit(p[:cut], y[:cut]).transform(p[cut:]))
    best = min(scores, key=lambda m: (round(scores[m], 7), METHODS.index(m)))
    final = Calibrator(best).fit(p, y)
    return final, {"chosen": best, "brier_on_later_validation_rows": {k: round(v, 6) for k, v in scores.items()}, "fitted_on_rows": int(len(p)), "choice_made_on_rows": int(len(p) - cut)}
