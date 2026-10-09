"""One feature pipeline for training, validation, backtesting and live prediction.

``build_features`` is the only place features are computed. Training calls it on a long history
and keeps every row; live prediction calls it on recent history and keeps the last row. Because
both go through the same function, the value for a given date is the same either way, which
``tests/test_features.py`` verifies row by row.

Rules this module holds to
    * A feature for date ``t`` uses information available at the close of ``t`` and nothing later.
    * Unavailable inputs produce ``NaN``. Nothing is replaced by zero or by a guess. What happens
      to a missing value is decided explicitly, by the caller's missing-value policy.
    * Context series (SPY, QQQ, VIX, sector, rates) are aligned to the stock's dates by taking
      the latest value at or before each date, and only if it is recent enough.
    * Interest-rate series are shifted by their publication lag before being used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import FeatureUnavailable, ValidationFailed
from app.features import indicators as ind
from app.features.schema import FEATURES, MIN_HISTORY_BARS, WARMUP_BARS, all_feature_names, groups_of, spec

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass
class MarketContext:
    """Everything outside the stock's own bars that features may use. Any part may be absent."""

    spy: pd.DataFrame | None = None
    qqq: pd.DataFrame | None = None
    vix: pd.Series | None = None
    sector: pd.DataFrame | None = None
    sector_symbol: str | None = None
    us10y: pd.Series | None = None
    yield_curve: pd.Series | None = None
    #: Calendar days after an observation's date before it may be used. Daily Treasury yields are
    #: published the next business day (1). Monthly averages dated the first of the month are
    #: complete only after the month ends (use about 35).
    us10y_lag_days: int = 1
    yield_curve_lag_days: int = 1
    #: A context value older than this many sessions is treated as missing, not carried forward.
    max_gap_sessions: int = 5
    max_macro_gap_sessions: int = 10
    #: Why a part is absent, for reporting. Filled by ``features.context.load_context``.
    missing: dict[str, str] = field(default_factory=dict)


def _prepare(ohlcv: pd.DataFrame, label: str) -> pd.DataFrame:
    absent = [c for c in REQUIRED_COLUMNS if c not in ohlcv.columns]
    if absent:
        raise ValidationFailed(f"{label} is missing columns: {absent}")
    frame = ohlcv.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).as_unit("ns")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    if frame.index.has_duplicates:
        raise ValidationFailed(f"{label} has duplicate dates; pass it through marketdata.quality.normalise_ohlcv first.")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    # Use dividend- and split-adjusted prices when the provider supplies them, so that a split or
    # a dividend is not mistaken for a price move. High and low are scaled by the same factor.
    if "Adj Close" in frame.columns and frame["Adj Close"].notna().any():
        factor = (pd.to_numeric(frame["Adj Close"], errors="coerce") / frame["Close"]).where(frame["Close"] > 0)
        frame["_close"] = frame["Close"] * factor
        frame["_high"] = frame["High"] * factor
        frame["_low"] = frame["Low"] * factor
    else:
        frame["_close"], frame["_high"], frame["_low"] = frame["Close"], frame["High"], frame["Low"]
    return frame


def adjusted_close(ohlcv: pd.DataFrame) -> pd.Series:
    """The close series every return in this project is computed from.

    Features, forecast targets and outcome scoring all call this, so "the 5-session return" means
    the same thing in each of them: dividend- and split-adjusted when the provider supplies an
    adjusted close, otherwise the close as given.
    """
    return _prepare(ohlcv, "The price history")["_close"].rename("adjusted_close")


def _as_of(series: pd.Series, index: pd.DatetimeIndex, max_gap_sessions: int, lag_days: int = 0) -> pd.Series:
    """Latest value of ``series`` known at each date in ``index``; ``NaN`` if too old or absent."""
    values = series.dropna().sort_index()
    if values.empty:
        return pd.Series(np.nan, index=index, dtype=float)
    stamps = pd.DatetimeIndex(pd.to_datetime(values.index)).as_unit("ns")
    if stamps.tz is not None:
        stamps = stamps.tz_localize(None)
    stamps = stamps.normalize() + pd.Timedelta(days=lag_days)
    values = pd.Series(values.to_numpy(dtype=float), index=stamps)
    values = values[~values.index.duplicated(keep="last")]
    # Position of the last observation at or before each target date.
    positions = values.index.searchsorted(index, side="right") - 1
    out = np.full(len(index), np.nan)
    ok = positions >= 0
    out[ok] = values.to_numpy()[positions[ok]]
    # Age in business days between that observation and the target date.
    observed = values.index.to_numpy()[np.clip(positions, 0, None)]
    age = np.busday_count(observed.astype("datetime64[D]"), index.to_numpy().astype("datetime64[D]"))
    out[~ok | (age > max_gap_sessions)] = np.nan
    return pd.Series(out, index=index, dtype=float)


def _context_block(bars: pd.DataFrame | None, label: str) -> dict[str, pd.Series] | None:
    """Features of a context instrument computed on its own calendar."""
    if bars is None or len(bars) == 0:
        return None
    frame = _prepare(bars, label)
    close = frame["_close"]
    daily = ind.pct_return(close, 1)
    return {
        "ret_1": daily,
        "ret_5": ind.pct_return(close, 5),
        "ret_21": ind.pct_return(close, 21),
        "ret_63": ind.pct_return(close, 63),
        "vol_21": ind.rolling_vol(daily, 21),
        "sma_gap_50": close / ind.sma(close, 50) - 1.0,
        "sma_gap_200": close / ind.sma(close, 200) - 1.0,
        "drawdown_63": ind.rolling_max_gap(close, 63),
    }


def build_features(ohlcv: pd.DataFrame, context: MarketContext | None = None) -> pd.DataFrame:
    """Compute every schema feature for every date in ``ohlcv``.

    Returns a float frame with the schema's columns in schema order, indexed by date. The first
    ``WARMUP_BARS`` rows are ``NaN`` in every column. ``frame.attrs["context_missing"]`` records
    which context inputs were absent and why.
    """
    context = context or MarketContext()
    frame = _prepare(ohlcv, "The price history")
    index = frame.index
    close, high, low, volume = frame["_close"], frame["_high"], frame["_low"], frame["Volume"]
    daily = ind.pct_return(close, 1)
    out: dict[str, pd.Series] = {}

    # --- own price ---------------------------------------------------------------------------
    out["ret_1"] = daily
    out["ret_5"] = ind.pct_return(close, 5)
    out["ret_10"] = ind.pct_return(close, 10)
    out["ret_21"] = ind.pct_return(close, 21)
    out["ret_63"] = ind.pct_return(close, 63)
    out["mom_126_21"] = close.shift(21) / close.shift(126) - 1.0
    out["vol_10"] = ind.rolling_vol(daily, 10)
    out["vol_21"] = ind.rolling_vol(daily, 21)
    out["vol_63"] = ind.rolling_vol(daily, 63)
    out["vol_ratio_10_63"] = out["vol_10"] / out["vol_63"].where(out["vol_63"] > 0)
    out["rsi_14"] = ind.rsi(close, 14)
    out["sma_gap_20"] = close / ind.sma(close, 20) - 1.0
    out["sma_gap_50"] = close / ind.sma(close, 50) - 1.0
    out["sma_gap_200"] = close / ind.sma(close, 200) - 1.0
    out["macd_hist_norm"] = ind.macd(close)[2] / close.where(close > 0)
    middle, upper, lower = ind.bollinger(close, 20)
    out["bb_pctb_20"] = (close - lower) / (upper - lower).where((upper - lower) > 0)
    out["atr_pct_14"] = ind.atr(high, low, close, 14) / close.where(close > 0)
    out["dist_52w_high"] = ind.rolling_max_gap(close, 252)
    out["drawdown_63"] = ind.rolling_max_gap(close, 63)
    out["range_pos_20"] = ind.range_position(close, high, low, 20)

    # --- volume ------------------------------------------------------------------------------
    log_volume = np.log(volume.where(volume > 0))
    out["volume_z_20"] = ind.zscore(log_volume, 20)
    out["volume_ratio_5_20"] = ind.sma(volume, 5) / ind.sma(volume, 20).where(ind.sma(volume, 20) > 0)
    out["up_volume_share_10"] = ind.up_volume_share(close, volume, 10)

    # --- broad market ------------------------------------------------------------------------
    gap = context.max_gap_sessions
    spy = _context_block(context.spy, "SPY history")
    spy_ret_21 = None
    if spy is not None:
        spy_daily = spy["ret_1"].reindex(index)  # exact dates only: a return cannot be carried forward
        spy_ret_5 = _as_of(spy["ret_5"], index, gap)
        spy_ret_21 = _as_of(spy["ret_21"], index, gap)
        out["rel_ret_5_spy"] = out["ret_5"] - spy_ret_5
        out["rel_ret_21_spy"] = out["ret_21"] - spy_ret_21
        out["rel_ret_63_spy"] = out["ret_63"] - _as_of(spy["ret_63"], index, gap)
        out["beta_63_spy"] = ind.rolling_beta(daily, spy_daily, 63)
        out["corr_63_spy"] = ind.rolling_corr(daily, spy_daily, 63)
        out["spy_ret_5"] = spy_ret_5
        out["spy_ret_21"] = spy_ret_21
        out["spy_vol_21"] = _as_of(spy["vol_21"], index, gap)
        out["spy_sma_gap_50"] = _as_of(spy["sma_gap_50"], index, gap)
        out["spy_sma_gap_200"] = _as_of(spy["sma_gap_200"], index, gap)
        out["spy_drawdown_63"] = _as_of(spy["drawdown_63"], index, gap)

    # --- QQQ ---------------------------------------------------------------------------------
    qqq = _context_block(context.qqq, "QQQ history")
    if qqq is not None:
        out["qqq_ret_5"] = _as_of(qqq["ret_5"], index, gap)
        out["qqq_ret_21"] = _as_of(qqq["ret_21"], index, gap)
        if spy_ret_21 is not None:
            out["qqq_rel_spy_21"] = out["qqq_ret_21"] - spy_ret_21

    # --- VIX ---------------------------------------------------------------------------------
    if context.vix is not None and len(context.vix.dropna()):
        vix = pd.to_numeric(context.vix, errors="coerce").dropna().sort_index()
        vix = vix[~vix.index.duplicated(keep="last")]
        out["vix_level"] = _as_of(vix, index, gap)
        out["vix_chg_5"] = _as_of(vix / vix.shift(5) - 1.0, index, gap)
        out["vix_z_63"] = _as_of(ind.zscore(vix, 63), index, gap)

    # --- sector ------------------------------------------------------------------------------
    sector = _context_block(context.sector, "Sector ETF history")
    if sector is not None:
        sector_ret_21 = _as_of(sector["ret_21"], index, gap)
        out["sector_ret_5"] = _as_of(sector["ret_5"], index, gap)
        out["sector_ret_21"] = sector_ret_21
        out["rel_ret_21_sector"] = out["ret_21"] - sector_ret_21
        if spy_ret_21 is not None:
            out["sector_rel_spy_21"] = sector_ret_21 - spy_ret_21

    # --- rates -------------------------------------------------------------------------------
    macro_gap = context.max_macro_gap_sessions
    if context.us10y is not None and len(context.us10y.dropna()):
        level = _as_of(pd.to_numeric(context.us10y, errors="coerce"), index, macro_gap, context.us10y_lag_days)
        out["us10y_level"] = level
        out["us10y_chg_21"] = level - level.shift(21)
    if context.yield_curve is not None and len(context.yield_curve.dropna()):
        out["yield_curve_10y2y"] = _as_of(pd.to_numeric(context.yield_curve, errors="coerce"), index, macro_gap, context.yield_curve_lag_days)

    names = all_feature_names()
    result = pd.DataFrame({name: out.get(name, pd.Series(np.nan, index=index)) for name in names}, index=index)[names]
    result = result.astype(float).replace([np.inf, -np.inf], np.nan)
    if len(result):
        result.iloc[: min(WARMUP_BARS, len(result))] = np.nan
    result.index.name = "Date"
    result.attrs["context_missing"] = dict(context.missing)
    result.attrs["bars"] = int(len(result))
    return result


def availability(features: pd.DataFrame) -> dict[str, Any]:
    """Per group: the share of post-warm-up rows where every feature in the group is present."""
    usable = features.iloc[WARMUP_BARS:] if len(features) > WARMUP_BARS else features.iloc[0:0]
    groups: dict[str, dict[str, Any]] = {}
    for group in dict.fromkeys(s.group for s in FEATURES):
        columns = [s.name for s in FEATURES if s.group == group]
        coverage = float(usable[columns].notna().all(axis=1).mean()) if len(usable) else 0.0
        groups[group] = {"coverage": round(coverage, 4), "features": len(columns)}
    return {"rows": int(len(features)), "usable_rows": int(len(usable)), "groups": groups, "context_missing": features.attrs.get("context_missing", {})}


def latest_row(features: pd.DataFrame, names: list[str]) -> pd.Series:
    """The most recent feature row for a model. Raises ``FeatureUnavailable`` rather than fill a gap."""
    if len(features) < MIN_HISTORY_BARS:
        raise FeatureUnavailable(
            f"{len(features)} bars of history were supplied; at least {MIN_HISTORY_BARS} are needed.",
            details={"bars": int(len(features)), "required_bars": MIN_HISTORY_BARS},
        )
    row = features.iloc[-1][names].astype(float)
    absent = [name for name in names if pd.isna(row[name])]
    if absent:
        context_missing = features.attrs.get("context_missing", {})
        raise FeatureUnavailable(
            "Some inputs the model needs could not be computed, so no prediction is made.",
            details={
                "missing_features": absent,
                "missing_groups": groups_of(absent),
                "reasons": {group: context_missing.get(group, "not enough recent data") for group in groups_of(absent)},
                "as_of": features.index[-1].date().isoformat(),
            },
        )
    return row


def feature_units(names: list[str]) -> dict[str, str]:
    return {name: spec(name).unit for name in names}
