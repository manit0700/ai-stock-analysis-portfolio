"""Turning price histories into rows a model can learn from.

One row is one (symbol, date): the features known at that date's close, and the label, which is
whether the adjusted close ``horizon`` sessions later is higher. Features come from
``features.pipeline.build_features`` and nowhere else.

Missing-value policy (explicit, and the only one): **complete rows only**. A row is used for
training or evaluation only if every feature the model uses is present. Nothing is imputed. The
number of rows dropped, and why, is recorded in ``Dataset.report``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.core.errors import MarketVisionError, ValidationFailed
from app.features import schema
from app.features.context import load_context
from app.features.pipeline import adjusted_close, availability, build_features
from app.marketdata import calendar as cal

DEFAULT_HORIZON = 5
TARGET_DESCRIPTION = "adjusted close {h} trading sessions after the as-of date is higher than the adjusted close on the as-of date"
#: Columns that are not features. Anything else in ``Dataset.frame`` must be a schema feature.
META_COLUMNS = ("symbol", "label_end", "fwd_return", "target", "trail_up_rate")


@dataclass
class Dataset:
    frame: pd.DataFrame  # index: Date; columns: META_COLUMNS + feature names
    feature_names: list[str]
    groups: list[str]
    horizon: int
    source: str
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - conventional name
        return self.frame[self.feature_names]

    @property
    def y(self) -> pd.Series:
        return self.frame["target"].astype(int)

    def dates(self) -> np.ndarray:
        return np.array(sorted(self.frame.index.unique()))


def forward_return(close: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return over the next ``horizon`` sessions, the date that window ends, and whether it is clean.

    A window is clean when the bar ``horizon`` rows ahead is exactly ``horizon`` trading sessions
    ahead on the exchange calendar. If a bar is missing inside the window, the row is not clean and
    is not used: "five sessions" must mean five sessions.
    """
    index = close.index
    ahead = close.shift(-horizon)
    end = pd.Series(index, index=index).shift(-horizon)
    expected = pd.Series(
        [pd.Timestamp(cal.shift_trading_days(ts.date(), horizon)) for ts in index], index=index
    )
    clean = end.notna() & (end == expected)
    return ahead / close - 1.0, end, clean


def symbol_rows(ohlcv: pd.DataFrame, context: Any, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All rows for one symbol (features for every date; label columns where the future is known)."""
    features = build_features(ohlcv, context)
    close = adjusted_close(ohlcv)
    fwd, end, clean = forward_return(close, horizon)
    rows = features.copy()
    rows["fwd_return"] = fwd.where(clean)
    rows["label_end"] = end.where(clean)
    rows["target"] = (rows["fwd_return"] > 0).astype(float).where(rows["fwd_return"].notna())
    # Baseline input: the share of the last 252 completed windows that ended higher. A window
    # starting at s is complete at s + horizon, so only windows starting at or before t - horizon count.
    past_up = (close / close.shift(horizon) - 1.0 > 0).astype(float).where(close.shift(horizon).notna())
    rows["trail_up_rate"] = past_up.rolling(252, min_periods=126).mean()
    info = {"bars": int(len(ohlcv)), "unclean_windows": int((~clean & end.notna()).sum()), "availability": availability(features)}
    return rows, info


def choose_groups(coverage: dict[str, float], minimum: float) -> list[str]:
    """Feature groups present for at least ``minimum`` of rows, in schema order."""
    return [group for group in schema.GROUPS if coverage.get(group, 0.0) >= minimum]


def build_dataset(
    market_data: Any,
    symbols: Iterable[str],
    *,
    horizon: int = DEFAULT_HORIZON,
    groups: list[str] | None = None,
    min_group_coverage: float = 0.9,
    start: date | None = None,
    end: date | None = None,
    source: str = "",
    include_macro: bool = True,
) -> Dataset:
    """Assemble the panel. ``groups=None`` keeps every feature group the data covers well enough."""
    if horizon < 1 or horizon > 60:
        raise ValidationFailed("The forecast horizon must be between 1 and 60 sessions.")
    shared: dict[str, Any] = {}
    parts: list[pd.DataFrame] = []
    per_symbol: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for symbol in dict.fromkeys(s.upper() for s in symbols):
        try:
            envelope = market_data.history(symbol, start=start, end=end, interval="1d")
            context = load_context(market_data, symbol, start=start, end=end, shared=shared, include_macro=include_macro)
            rows, info = symbol_rows(envelope.data, context, horizon)
        except MarketVisionError as exc:
            skipped[symbol] = exc.message
            continue
        rows.insert(0, "symbol", symbol)
        info["quality"] = envelope.quality.status
        per_symbol[symbol] = info
        parts.append(rows)
    if not parts:
        raise ValidationFailed("No symbol produced any rows.", details={"skipped": skipped})

    panel = pd.concat(parts).sort_index(kind="stable")
    labelled = panel[panel["target"].notna()]
    warm = labelled[labelled[schema.feature_names(["price"])].notna().all(axis=1)]
    coverage = {
        group: float(warm[schema.feature_names([group])].notna().all(axis=1).mean()) if len(warm) else 0.0 for group in schema.GROUPS
    }
    chosen = list(groups) if groups is not None else choose_groups(coverage, min_group_coverage)
    if not chosen:
        raise ValidationFailed("No feature group is covered well enough to build a dataset.", details={"coverage": coverage})
    names = schema.feature_names(chosen)
    complete = warm[warm[names].notna().all(axis=1)]
    frame = complete[["symbol", "label_end", "fwd_return", "target", "trail_up_rate", *names]].copy()
    frame["label_end"] = pd.to_datetime(frame["label_end"])
    frame["target"] = frame["target"].astype(int)

    problems: list[str] = [] if len(frame) else ["no complete rows"]
    for symbol, part in frame.groupby("symbol", sort=True):  # one symbol at a time: dates repeat across symbols
        problems.extend(f"{symbol}: {problem}" for problem in schema.validate_frame(part[names], names, allow_missing=False))
    report = {
        "symbols": sorted(per_symbol),
        "skipped_symbols": skipped,
        "rows_total": int(len(panel)),
        "rows_with_label": int(len(labelled)),
        "rows_after_warmup": int(len(warm)),
        "rows_complete": int(len(frame)),
        "rows_dropped_incomplete_features": int(len(warm) - len(complete)),
        "group_coverage": {k: round(v, 4) for k, v in coverage.items()},
        "groups_used": chosen,
        "groups_absent": [g for g in schema.GROUPS if g not in chosen],
        "missing_value_policy": "complete rows only; nothing imputed",
        "first_date": frame.index.min().date().isoformat() if len(frame) else None,
        "last_date": frame.index.max().date().isoformat() if len(frame) else None,
        "base_rate": round(float(frame["target"].mean()), 6) if len(frame) else None,
        "rows_per_symbol": {k: int(v) for k, v in frame.groupby("symbol").size().items()},
        "schema_problems": problems,
        "context_missing": {k: v for k, v in (per_symbol[next(iter(per_symbol))]["availability"].get("context_missing") or {}).items()},
    }
    return Dataset(frame=frame, feature_names=names, groups=chosen, horizon=horizon, source=source, report=report)


def target_text(horizon: int) -> str:
    return TARGET_DESCRIPTION.format(h=horizon)
