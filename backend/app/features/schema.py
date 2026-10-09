"""The feature schema: names, order, units and valid ranges, under one version.

A model records the schema version, the schema hash and the exact ordered list of features it
was trained on. At prediction time the same list must be producible with the same schema hash,
otherwise the model is refused (``forecast.registry``). That is what stops training and serving
from drifting apart again.

Changing any feature's definition means changing ``SCHEMA_VERSION``. The hash below changes
automatically when a name, group, unit, lookback or range changes; it cannot see a change made
only inside ``pipeline.py``, so the version must be bumped by hand in that case.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SCHEMA_NAME = "marketvision-daily"
SCHEMA_VERSION = "1.0.0"

#: Rows before this position in a history are warm-up: long-window features are not yet defined
#: and exponentially-weighted ones still depend on where the history happened to start.
WARMUP_BARS = 300
#: Minimum bars a caller must supply to get one valid feature row.
MIN_HISTORY_BARS = WARMUP_BARS + 1


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    unit: str
    description: str
    lookback: int
    lower: float | None = None
    upper: float | None = None


def _f(name, group, unit, description, lookback, lower=None, upper=None) -> FeatureSpec:
    return FeatureSpec(name, group, unit, description, lookback, lower, upper)


# Units: "ratio" = fractional change (0.01 is 1%); "stdev" = standard deviation of daily returns;
# "z" = standard deviations; "share" = 0..1; "index" = index points; "percent" = percent as quoted;
# "pp" = percentage points; "coef" = dimensionless coefficient.
FEATURES: tuple[FeatureSpec, ...] = (
    # --- the stock's own price ---------------------------------------------------------------
    _f("ret_1", "price", "ratio", "Return over the last session", 2, -0.95, 5.0),
    _f("ret_5", "price", "ratio", "Return over the last 5 sessions", 6, -0.95, 10.0),
    _f("ret_10", "price", "ratio", "Return over the last 10 sessions", 11, -0.95, 10.0),
    _f("ret_21", "price", "ratio", "Return over the last 21 sessions", 22, -0.99, 20.0),
    _f("ret_63", "price", "ratio", "Return over the last 63 sessions", 64, -0.99, 50.0),
    _f("mom_126_21", "price", "ratio", "Return from 126 to 21 sessions ago (momentum, skipping the last month)", 127, -0.99, 50.0),
    _f("vol_10", "price", "stdev", "Standard deviation of daily returns, 10 sessions", 11, 0.0, 2.0),
    _f("vol_21", "price", "stdev", "Standard deviation of daily returns, 21 sessions", 22, 0.0, 2.0),
    _f("vol_63", "price", "stdev", "Standard deviation of daily returns, 63 sessions", 64, 0.0, 2.0),
    _f("vol_ratio_10_63", "price", "coef", "vol_10 divided by vol_63", 64, 0.0, 50.0),
    _f("rsi_14", "price", "index", "Relative Strength Index, 14 sessions", 15, 0.0, 100.0),
    _f("sma_gap_20", "price", "ratio", "Close relative to its 20-session average", 20, -0.95, 10.0),
    _f("sma_gap_50", "price", "ratio", "Close relative to its 50-session average", 50, -0.95, 10.0),
    _f("sma_gap_200", "price", "ratio", "Close relative to its 200-session average", 200, -0.99, 20.0),
    _f("macd_hist_norm", "price", "ratio", "MACD histogram divided by the close", 35, -1.0, 1.0),
    _f("bb_pctb_20", "price", "coef", "Position within 20-session Bollinger bands (0 = lower band, 1 = upper band)", 20, -5.0, 6.0),
    _f("atr_pct_14", "price", "ratio", "Average true range over 14 sessions divided by the close", 15, 0.0, 2.0),
    _f("dist_52w_high", "price", "ratio", "Close relative to its highest close in 252 sessions", 252, -1.0, 0.0),
    _f("drawdown_63", "price", "ratio", "Close relative to its highest close in 63 sessions", 63, -1.0, 0.0),
    _f("range_pos_20", "price", "share", "Close between the 20-session low (0) and high (1)", 20, 0.0, 1.0),
    # --- volume ------------------------------------------------------------------------------
    _f("volume_z_20", "volume", "z", "Log volume against the previous 20 sessions", 21, -20.0, 20.0),
    _f("volume_ratio_5_20", "volume", "coef", "Average volume of 5 sessions over 20 sessions", 20, 0.0, 100.0),
    _f("up_volume_share_10", "volume", "share", "Share of 10-session volume traded on rising days", 11, 0.0, 1.0),
    # --- the broad market (SPY) -------------------------------------------------------------
    _f("rel_ret_5_spy", "market", "ratio", "5-session return minus SPY's", 6, -10.0, 10.0),
    _f("rel_ret_21_spy", "market", "ratio", "21-session return minus SPY's", 22, -20.0, 20.0),
    _f("rel_ret_63_spy", "market", "ratio", "63-session return minus SPY's", 64, -50.0, 50.0),
    _f("beta_63_spy", "market", "coef", "Beta to SPY over 63 sessions", 64, -20.0, 20.0),
    _f("corr_63_spy", "market", "coef", "Correlation of daily returns with SPY over 63 sessions", 64, -1.0, 1.0),
    _f("spy_ret_5", "market", "ratio", "SPY return over 5 sessions", 6, -0.9, 2.0),
    _f("spy_ret_21", "market", "ratio", "SPY return over 21 sessions", 22, -0.9, 2.0),
    _f("spy_vol_21", "market", "stdev", "SPY daily-return standard deviation, 21 sessions", 22, 0.0, 1.0),
    _f("spy_sma_gap_50", "market", "ratio", "SPY relative to its 50-session average", 50, -0.9, 2.0),
    _f("spy_sma_gap_200", "market", "ratio", "SPY relative to its 200-session average", 200, -0.9, 2.0),
    _f("spy_drawdown_63", "market", "ratio", "SPY relative to its 63-session high", 63, -1.0, 0.0),
    # --- growth stocks (QQQ) ----------------------------------------------------------------
    _f("qqq_ret_5", "qqq", "ratio", "QQQ return over 5 sessions", 6, -0.9, 2.0),
    _f("qqq_ret_21", "qqq", "ratio", "QQQ return over 21 sessions", 22, -0.9, 2.0),
    _f("qqq_rel_spy_21", "qqq", "ratio", "QQQ 21-session return minus SPY's", 22, -2.0, 2.0),
    # --- volatility index -------------------------------------------------------------------
    _f("vix_level", "vix", "index", "VIX close", 1, 5.0, 150.0),
    _f("vix_chg_5", "vix", "ratio", "VIX change over 5 sessions", 6, -0.95, 10.0),
    _f("vix_z_63", "vix", "z", "VIX against its previous 63 sessions", 64, -20.0, 30.0),
    # --- the stock's sector ETF ---------------------------------------------------------------
    _f("sector_ret_5", "sector", "ratio", "Sector ETF return over 5 sessions", 6, -0.9, 2.0),
    _f("sector_ret_21", "sector", "ratio", "Sector ETF return over 21 sessions", 22, -0.9, 2.0),
    _f("sector_rel_spy_21", "sector", "ratio", "Sector ETF 21-session return minus SPY's", 22, -2.0, 2.0),
    _f("rel_ret_21_sector", "sector", "ratio", "21-session return minus the sector ETF's", 22, -20.0, 20.0),
    # --- interest rates (published with a lag; see pipeline.MarketContext) ---------------------
    _f("us10y_level", "macro", "percent", "10-year Treasury yield", 1, -5.0, 25.0),
    _f("us10y_chg_21", "macro", "pp", "Change in the 10-year yield over 21 sessions", 22, -10.0, 10.0),
    _f("yield_curve_10y2y", "macro", "pp", "10-year minus 2-year Treasury yield", 1, -10.0, 10.0),
)

GROUPS: dict[str, str] = {
    "price": "the stock's own daily bars",
    "volume": "the stock's own daily volume",
    "market": "SPY daily bars",
    "qqq": "QQQ daily bars",
    "vix": "VIX daily closes",
    "sector": "daily bars of the stock's sector ETF",
    "macro": "10-year and 2-year Treasury yields",
}

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical": "XLY",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    "Real Estate": "XLRE",
}

_BY_NAME = {spec.name: spec for spec in FEATURES}


def all_feature_names() -> list[str]:
    return [spec.name for spec in FEATURES]


def feature_names(groups: list[str] | tuple[str, ...] | set[str] | None = None) -> list[str]:
    """Feature names in schema order, optionally limited to some groups."""
    if groups is None:
        return all_feature_names()
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    return [spec.name for spec in FEATURES if spec.group in set(groups)]


def spec(name: str) -> FeatureSpec:
    return _BY_NAME[name]


def groups_of(names: list[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        group = _BY_NAME[name].group
        if group not in seen:
            seen.append(group)
    return seen


def schema_hash() -> str:
    payload = {"name": SCHEMA_NAME, "version": SCHEMA_VERSION, "warmup": WARMUP_BARS, "features": [asdict(s) for s in FEATURES]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def describe() -> dict:
    return {
        "name": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "hash": schema_hash(),
        "warmup_bars": WARMUP_BARS,
        "min_history_bars": MIN_HISTORY_BARS,
        "groups": GROUPS,
        "features": [asdict(s) for s in FEATURES],
    }


def validate_frame(frame: pd.DataFrame, names: list[str] | None = None, *, allow_missing: bool = True) -> list[str]:
    """Check a feature frame against the schema. Returns a list of problems (empty = valid).

    Checks: every name known; columns present and in the given order; float dtype; no infinite
    values; values inside each feature's valid range; a sorted, unique, timezone-free date index.
    """
    names = list(names) if names is not None else all_feature_names()
    problems: list[str] = []
    unknown = [n for n in names if n not in _BY_NAME]
    if unknown:
        problems.append(f"not in the schema: {unknown}")
        names = [n for n in names if n in _BY_NAME]
    absent = [n for n in names if n not in frame.columns]
    if absent:
        problems.append(f"missing columns: {absent}")
    present = [n for n in names if n in frame.columns]
    order_in_frame = [c for c in frame.columns if c in set(present)]
    if order_in_frame != present:
        problems.append("columns are not in the expected order")
    if not isinstance(frame.index, pd.DatetimeIndex):
        problems.append("index is not a DatetimeIndex")
    else:
        if frame.index.tz is not None:
            problems.append("index carries a time zone")
        if not frame.index.is_monotonic_increasing:
            problems.append("index is not sorted by date")
        if frame.index.has_duplicates:
            problems.append("index has duplicate dates")
    for name in present:
        column = frame[name]
        if not pd.api.types.is_float_dtype(column):
            problems.append(f"{name}: dtype {column.dtype} is not float")
            continue
        values = column.to_numpy(dtype=float)
        if np.isinf(values).any():
            problems.append(f"{name}: contains infinite values")
        if not allow_missing and np.isnan(values).any():
            problems.append(f"{name}: {int(np.isnan(values).sum())} missing values")
        finite = values[np.isfinite(values)]
        if finite.size:
            feature = _BY_NAME[name]
            if feature.lower is not None and finite.min() < feature.lower - 1e-9:
                problems.append(f"{name}: value {finite.min():.6g} below the valid minimum {feature.lower} ({feature.unit})")
            if feature.upper is not None and finite.max() > feature.upper + 1e-9:
                problems.append(f"{name}: value {finite.max():.6g} above the valid maximum {feature.upper} ({feature.unit})")
    return problems
