"""How wide the range of outcomes is, stated only as far as the data supports it.

Direction is hard to forecast; the *size* of moves is much more predictable, because volatility
clusters. So the range offered for the next ``horizon`` sessions is the historical distribution of
returns divided by each stock's recent volatility, then scaled back by its current volatility.

Whether that beats the same quantiles with no scaling is measured (pinball loss, and how often
the stated 80% range contains the outcome). If it does not, the unscaled historical range is
used and the response says so.
"""
from __future__ import annotations

from typing import Any

import numpy as np

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _clean(fwd_return: np.ndarray, vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, v = np.asarray(fwd_return, dtype=float), np.asarray(vol, dtype=float)
    ok = np.isfinite(r) & np.isfinite(v) & (v > 0)
    return r[ok], v[ok]


def fit_uncertainty(fwd_return: np.ndarray, vol_21: np.ndarray, horizon: int) -> dict[str, Any]:
    """Quantile tables from development rows: raw returns, and returns in units of recent volatility."""
    r, v = _clean(fwd_return, vol_21)
    scale = v * np.sqrt(horizon)
    return {
        "horizon": horizon,
        "rows": int(len(r)),
        "mean_return": float(r.mean()) if len(r) else None,
        "unconditional": {f"{q:.2f}": float(np.quantile(r, q)) for q in QUANTILES} if len(r) else {},
        "scaled": {f"{q:.2f}": float(np.quantile(r / scale, q)) for q in QUANTILES} if len(r) else {},
        "scaling": "forward return divided by (21-session standard deviation of daily returns x sqrt(horizon))",
    }


def predict_quantiles(table: dict[str, Any], vol_21: float | np.ndarray, *, scaled: bool) -> dict[str, Any]:
    """Return quantiles for the next ``horizon`` sessions given current volatility."""
    if not scaled:
        return dict(table["unconditional"])
    factor = np.asarray(vol_21, dtype=float) * np.sqrt(table["horizon"])
    return {key: value * factor for key, value in table["scaled"].items()}


def pinball_loss(y: np.ndarray, predicted: np.ndarray | float, q: float) -> float:
    diff = np.asarray(y, dtype=float) - predicted
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def evaluate_uncertainty(table: dict[str, Any], fwd_return: np.ndarray, vol_21: np.ndarray) -> dict[str, Any]:
    """Compare the volatility-scaled range with the unscaled one on rows the table was not built from."""
    r, v = _clean(fwd_return, vol_21)
    if len(r) < 500 or not table.get("scaled"):
        return {"rows": int(len(r)), "supported": False, "reason": "too few rows to judge"}
    scaled = predict_quantiles(table, v, scaled=True)
    plain = table["unconditional"]
    keys = [f"{q:.2f}" for q in QUANTILES]
    loss_scaled = float(np.mean([pinball_loss(r, scaled[k], float(k)) for k in keys]))
    loss_plain = float(np.mean([pinball_loss(r, plain[k], float(k)) for k in keys]))

    def coverage(low: str, high: str, table_: dict[str, Any]) -> float:
        return float(np.mean((r >= table_[low]) & (r <= table_[high])))

    cover_scaled_80, cover_plain_80 = coverage("0.10", "0.90", scaled), coverage("0.10", "0.90", plain)
    cover_scaled_90, cover_plain_90 = coverage("0.05", "0.95", scaled), coverage("0.05", "0.95", plain)
    supported = loss_scaled < loss_plain and abs(cover_scaled_80 - 0.80) <= 0.05
    return {
        "rows": int(len(r)),
        "pinball_loss_scaled": round(loss_scaled, 6),
        "pinball_loss_unscaled": round(loss_plain, 6),
        "pinball_improvement": round(1 - loss_scaled / loss_plain, 4) if loss_plain > 0 else None,
        "coverage_of_80pct_range": {"scaled": round(cover_scaled_80, 4), "unscaled": round(cover_plain_80, 4)},
        "coverage_of_90pct_range": {"scaled": round(cover_scaled_90, 4), "unscaled": round(cover_plain_90, 4)},
        "supported": bool(supported),
        "rule": "the volatility-scaled range is used only if its pinball loss is lower than the unscaled range's and its 80% range covers between 75% and 85% of outcomes",
    }
