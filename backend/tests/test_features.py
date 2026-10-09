"""The feature pipeline: one implementation, no look-ahead, no silent substitutes.

These tests use synthetic prices. They check that the code is correct; they say nothing about
whether any feature predicts anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.errors import FeatureUnavailable, ValidationFailed
from app.features import indicators as ind
from app.features import schema
from app.features.pipeline import MarketContext, availability, build_features, latest_row
from tests.helpers import make_ohlcv

N = 700


def _context(n: int = N, **overrides) -> MarketContext:
    spy = make_ohlcv(n, seed=11, start_price=300, vol=0.01)
    qqq = make_ohlcv(n, seed=12, start_price=250, vol=0.013)
    sector = make_ohlcv(n, seed=13, start_price=120, vol=0.012)
    rng = np.random.default_rng(14)
    vix = pd.Series(18 + np.cumsum(rng.normal(0, 0.4, n)).clip(-8, 40), index=spy.index)
    us10y = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.02, n)), index=spy.index)
    curve = pd.Series(0.5 + np.cumsum(rng.normal(0, 0.01, n)), index=spy.index)
    values = dict(spy=spy, qqq=qqq, vix=vix, sector=sector, sector_symbol="XLK", us10y=us10y, yield_curve=curve)
    values.update(overrides)
    return MarketContext(**values)


def _truncate(context: MarketContext, stop) -> MarketContext:
    cut = lambda x: None if x is None else x.loc[:stop]  # noqa: E731
    return MarketContext(
        spy=cut(context.spy), qqq=cut(context.qqq), vix=cut(context.vix), sector=cut(context.sector), sector_symbol=context.sector_symbol,
        us10y=cut(context.us10y), yield_curve=cut(context.yield_curve),
    )


# ------------------------------------------------------------------ schema
def test_schema_names_are_unique_and_ordered():
    names = schema.all_feature_names()
    assert len(names) == len(set(names)) == len(schema.FEATURES)
    assert names[0] == "ret_1"
    assert set(s.group for s in schema.FEATURES) == set(schema.GROUPS)


def test_schema_hash_is_stable_and_changes_with_the_definition(monkeypatch):
    first = schema.schema_hash()
    assert first == schema.schema_hash() and len(first) == 64
    monkeypatch.setattr(schema, "SCHEMA_VERSION", "9.9.9")
    assert schema.schema_hash() != first


def test_every_feature_declares_a_unit_and_a_lookback_within_the_warmup():
    for feature in schema.FEATURES:
        assert feature.unit and feature.description
        assert 1 <= feature.lookback <= schema.WARMUP_BARS


# ------------------------------------------------------------------ shape and validity
def test_output_has_schema_columns_in_schema_order_and_passes_validation():
    features = build_features(make_ohlcv(N, seed=1), _context())
    assert list(features.columns) == schema.all_feature_names()
    assert all(str(dtype) == "float64" for dtype in features.dtypes)
    assert schema.validate_frame(features) == []
    assert schema.validate_frame(features.iloc[schema.WARMUP_BARS:], allow_missing=False) == []


def test_warmup_rows_are_missing_in_every_column():
    features = build_features(make_ohlcv(N, seed=1), _context())
    assert features.iloc[: schema.WARMUP_BARS].isna().all().all()
    assert features.iloc[schema.WARMUP_BARS:].notna().all().all()


def test_validation_reports_out_of_range_and_wrong_order():
    features = build_features(make_ohlcv(N, seed=1), _context())
    broken = features.copy()
    broken.iloc[-1, broken.columns.get_loc("rsi_14")] = 140.0
    assert any("rsi_14" in problem and "above" in problem for problem in schema.validate_frame(broken))
    swapped = features[list(reversed(features.columns))]
    assert "columns are not in the expected order" in schema.validate_frame(swapped)
    assert any("missing columns" in problem for problem in schema.validate_frame(features.drop(columns=["ret_5"])))


# ------------------------------------------------------------------ no look-ahead, and parity
@pytest.mark.parametrize("cut", [420, 555, 699])
def test_a_row_does_not_change_when_later_data_is_removed(cut):
    """The value for a date must be the same whether or not the future is in the frame."""
    history, context = make_ohlcv(N, seed=2), _context()
    full = build_features(history, context)
    stop = history.index[cut]
    past_only = build_features(history.loc[:stop], _truncate(context, stop))
    pd.testing.assert_series_equal(past_only.iloc[-1], full.loc[stop], check_names=False, rtol=1e-9, atol=1e-12)


def test_future_context_rows_do_not_leak_into_an_earlier_date():
    history, context = make_ohlcv(N, seed=2), _context()
    stop = history.index[500]
    with_future_context = build_features(history.loc[:stop], context)
    without = build_features(history.loc[:stop], _truncate(context, stop))
    pd.testing.assert_frame_equal(with_future_context, without, rtol=1e-12, atol=1e-12)


def test_changing_a_future_bar_changes_no_earlier_row():
    history, context = make_ohlcv(N, seed=3), _context()
    before = build_features(history, context)
    altered = history.copy()
    altered.iloc[600:, altered.columns.get_loc("Close")] *= 1.5
    altered.iloc[600:, altered.columns.get_loc("High")] *= 1.5
    after = build_features(altered, context)
    pd.testing.assert_frame_equal(before.iloc[:600], after.iloc[:600])
    assert not before.iloc[600:].equals(after.iloc[600:])


def test_training_and_live_inference_produce_the_same_row():
    """Training sees the whole history; live prediction sees a recent window. Same date, same numbers."""
    history, context = make_ohlcv(1400, seed=4), _context(1400)
    training = build_features(history, context)
    live = build_features(history.iloc[-520:], _truncate(context, history.index[-1]))
    names = schema.all_feature_names()
    np.testing.assert_allclose(live.iloc[-1][names].to_numpy(float), training.iloc[-1][names].to_numpy(float), rtol=1e-6, atol=1e-9)


def test_every_indicator_looks_backwards_only():
    bars = make_ohlcv(400, seed=5)
    close, high, low, volume = bars["Close"], bars["High"], bars["Low"], bars["Volume"]
    cut = 300
    series = {
        "sma": lambda c, h, l, v: ind.sma(c, 20),
        "ema": lambda c, h, l, v: ind.ema(c, 20),
        "rsi": lambda c, h, l, v: ind.rsi(c, 14),
        "macd": lambda c, h, l, v: ind.macd(c)[2],
        "atr": lambda c, h, l, v: ind.atr(h, l, c, 14),
        "bollinger": lambda c, h, l, v: ind.bollinger(c, 20)[1],
        "zscore": lambda c, h, l, v: ind.zscore(c, 20),
        "range_position": lambda c, h, l, v: ind.range_position(c, h, l, 20),
        "up_volume_share": lambda c, h, l, v: ind.up_volume_share(c, v, 10),
        "rolling_max_gap": lambda c, h, l, v: ind.rolling_max_gap(c, 63),
    }
    for name, fn in series.items():
        whole = fn(close, high, low, volume).iloc[:cut]
        part = fn(close.iloc[:cut], high.iloc[:cut], low.iloc[:cut], volume.iloc[:cut])
        pd.testing.assert_series_equal(whole, part, check_names=False, obj=name)


# ------------------------------------------------------------------ missing inputs are missing, not zero
def test_absent_context_gives_missing_features_never_zero():
    context = _context(vix=None, sector=None, us10y=None, yield_curve=None)
    context.missing = {"vix": "VIX history: provider down"}
    features = build_features(make_ohlcv(N, seed=6), context)
    usable = features.iloc[schema.WARMUP_BARS:]
    for group in ("vix", "sector", "macro"):
        assert usable[schema.feature_names([group])].isna().all().all(), group
    for group in ("price", "volume", "market", "qqq"):
        assert usable[schema.feature_names([group])].notna().all().all(), group
    report = availability(features)
    assert report["groups"]["vix"]["coverage"] == 0.0 and report["groups"]["price"]["coverage"] == 1.0
    assert report["context_missing"] == {"vix": "VIX history: provider down"}


def test_latest_row_refuses_instead_of_filling_a_gap():
    context = _context(vix=None)
    context.missing = {"vix": "VIX history: provider down"}
    features = build_features(make_ohlcv(N, seed=6), context)
    price_only = schema.feature_names(["price", "volume"])
    assert latest_row(features, price_only).notna().all()
    with pytest.raises(FeatureUnavailable) as raised:
        latest_row(features, schema.feature_names(["price", "vix"]))
    details = raised.value.details
    assert details["missing_groups"] == ["vix"] and "vix_level" in details["missing_features"]
    assert details["reasons"]["vix"] == "VIX history: provider down"


def test_latest_row_needs_enough_history():
    features = build_features(make_ohlcv(schema.WARMUP_BARS, seed=7), _context(schema.WARMUP_BARS))
    with pytest.raises(FeatureUnavailable) as raised:
        latest_row(features, ["ret_1"])
    assert raised.value.details["required_bars"] == schema.MIN_HISTORY_BARS


def test_a_stale_context_value_is_not_carried_forward():
    history = make_ohlcv(N, seed=8)
    context = _context()
    stale_from = history.index[-20]
    context.vix = context.vix.loc[: stale_from]  # VIX stops updating 19 sessions before the end
    features = build_features(history, context)
    assert features["vix_level"].loc[stale_from] == pytest.approx(context.vix.iloc[-1])
    assert features["vix_level"].iloc[-(19 - context.max_gap_sessions):].isna().all()
    assert features["vix_level"].loc[history.index[-19]] == pytest.approx(context.vix.iloc[-1])  # one session old: still usable


def test_an_interest_rate_is_not_used_before_its_publication_lag():
    history = make_ohlcv(N, seed=9)
    context = _context()
    features = build_features(history, context)
    day, previous = history.index[-1], history.index[-2]
    # With a one-day lag the value known at ``day`` is the one dated the session before.
    assert features["us10y_level"].loc[day] == pytest.approx(context.us10y.loc[previous])
    assert features["us10y_level"].loc[day] != pytest.approx(context.us10y.loc[day])
    context.us10y_lag_days = 0
    assert build_features(history, context)["us10y_level"].loc[day] == pytest.approx(context.us10y.loc[day])


def test_missing_prices_stay_missing():
    history = make_ohlcv(N, seed=10)
    history.iloc[650, history.columns.get_loc("Close")] = np.nan
    features = build_features(history, _context())
    assert np.isnan(features["ret_1"].iloc[650]) and np.isnan(features["ret_1"].iloc[651])
    assert not (features.iloc[schema.WARMUP_BARS:] == 0.0).all().any()


# ------------------------------------------------------------------ corporate actions and input checks
def test_a_split_is_not_mistaken_for_a_price_move_when_adjusted_prices_are_supplied():
    history = make_ohlcv(N, seed=15)
    adjusted = history["Close"].copy()
    raw = history.copy()
    split_at = 600
    for column in ("Open", "High", "Low", "Close"):
        raw.iloc[:split_at, raw.columns.get_loc(column)] *= 4.0  # prices before a 4-for-1 split, as traded
    raw["Adj Close"] = adjusted
    with_adjustment = build_features(raw, _context())
    reference = build_features(history, _context())
    assert abs(with_adjustment["ret_1"].iloc[split_at]) < 0.2
    for column in ("ret_1", "ret_21", "rsi_14", "sma_gap_50", "atr_pct_14"):
        np.testing.assert_allclose(with_adjustment[column].iloc[schema.WARMUP_BARS:], reference[column].iloc[schema.WARMUP_BARS:], rtol=1e-9, atol=1e-12)
    without_adjustment = build_features(raw.drop(columns=["Adj Close"]), _context())
    assert without_adjustment["ret_1"].iloc[split_at] < -0.7  # shows the test would catch the fault


def test_duplicate_dates_and_missing_columns_are_rejected():
    history = make_ohlcv(N, seed=16)
    with pytest.raises(ValidationFailed):
        build_features(pd.concat([history, history.iloc[[-1]]]))
    with pytest.raises(ValidationFailed):
        build_features(history.drop(columns=["Volume"]))


def test_relative_return_is_the_difference_of_the_two_returns():
    history, context = make_ohlcv(N, seed=17), _context()
    features = build_features(history, context)
    day = history.index[-1]
    own = history["Close"].iloc[-1] / history["Close"].iloc[-22] - 1
    market = context.spy["Close"].iloc[-1] / context.spy["Close"].iloc[-22] - 1
    assert features["ret_21"].loc[day] == pytest.approx(own)
    assert features["spy_ret_21"].loc[day] == pytest.approx(market)
    assert features["rel_ret_21_spy"].loc[day] == pytest.approx(own - market)


def test_rsi_edge_cases():
    rising = pd.Series(np.arange(1.0, 40.0))
    flat = pd.Series(np.full(40, 5.0))
    assert ind.rsi(rising).iloc[-1] == 100.0
    assert ind.rsi(flat).iloc[-1] == 50.0
    assert np.isnan(ind.rsi(rising).iloc[5])
