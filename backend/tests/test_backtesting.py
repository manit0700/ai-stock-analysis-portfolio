"""The backtest engine: one book, so trades, cash and the equity curve agree to the cent.

Synthetic price paths are used throughout. They verify the arithmetic. They are not evidence
that any strategy makes money.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from app.backtesting.engine import FREE, BacktestConfig, CostModel, buy_and_hold, cash_only, run_backtest
from app.backtesting.metrics import max_drawdown, summarise
from app.backtesting.strategies import STRATEGIES, probability_threshold, sma_crossover
from app.core.errors import ValidationFailed
from app.services.backtest_service import _run_backtest
from tests.helpers import bars, flat_ohlcv, make_ohlcv

LONG = lambda frame: pd.Series(1.0, index=frame.index)  # noqa: E731


def _free(**kwargs) -> BacktestConfig:
    return BacktestConfig(costs=FREE, **kwargs)


def _reconciles(result) -> None:
    """Final equity minus starting capital equals realised + dividends + unrealised, to the cent."""
    closed = sum(Decimal(repr(t["pnl"])) for t in result.trades if t["outcome"] != "open")
    assert abs(float(closed) - float(result.book.realized)) < 0.005 * max(1, len(result.trades))
    final_close = {result.config.symbol: Decimal(repr(float(result.invested.iloc[-1]))) / result.book.quantity(result.config.symbol)} if result.book.quantity(result.config.symbol) > 0 else {}
    total = result.book.realized + result.book.dividends + (Decimal(repr(float(result.invested.iloc[-1]))) - result.book.cost_of_open_positions())
    assert abs(float(result.equity.iloc[-1]) - float(result.config.initial_cash) - float(total)) < 0.011
    assert np.allclose(result.equity.to_numpy(), result.cash.to_numpy() + result.invested.to_numpy(), atol=1e-9)
    assert (result.cash >= 0).all()
    assert final_close is not None


# ------------------------------------------------------------------ baselines
def test_cash_only_never_changes():
    frame = make_ohlcv(120, seed=1)
    result = cash_only(frame, BacktestConfig())
    assert (result.equity == 10_000.0).all() and result.trades == [] and (result.invested == 0).all()
    stats = summarise(result)
    assert stats["total_return_pct"] == 0.0 and stats["exposure_pct"] == 0.0 and stats["costs"]["total"] == 0.0


def test_buy_and_hold_without_costs_tracks_the_price_from_the_first_fill():
    frame = flat_ohlcv([100, 100, 110, 121, 121], start="2024-01-02")
    result = buy_and_hold(frame, _free(initial_cash=Decimal("1000")))
    assert result.equity.tolist() == [1000.0, 1000.0, 1100.0, 1210.0, 1210.0]  # decided at close 0, bought at open 1
    assert result.trades[-1]["outcome"] == "open" and result.trades[-1]["entry_date"] == frame.index[1].date().isoformat()
    assert result.book.quantity("ASSET") == Decimal("10") and result.cash.iloc[-1] == 0.0
    _reconciles(result)


def test_a_decision_at_one_close_is_filled_at_the_next_open_never_the_same_bar():
    rows = [(100, 100, 100, 100)] * 3 + [(200, 200, 200, 200)] * 3  # the price doubles overnight before bar 3
    frame = bars(rows)
    target = pd.Series([0, 0, 1, 1, 1, 1], index=frame.index, dtype=float)  # decided at the close of bar 2, at 100
    result = run_backtest(frame, target, _free(initial_cash=Decimal("1000")))
    assert result.trades[0]["entry_price"] == 200.0 and result.trades[0]["entry_date"] == frame.index[3].date().isoformat()
    assert result.equity.tolist() == [1000.0] * 6  # bought at 200, still 200: no gain from the jump it could not have traded


def test_costs_reduce_equity_by_exactly_the_costs_recorded():
    frame = flat_ohlcv([100.0] * 6)
    costs = CostModel(commission_rate=Decimal("0.001"), half_spread_bps=Decimal("5"), slippage_bps=Decimal("5"))
    target = pd.Series([1, 1, 1, 0, 0, 0], index=frame.index, dtype=float)
    result = run_backtest(frame, target, BacktestConfig(initial_cash=Decimal("10000"), costs=costs))
    stats = summarise(result)
    trade = result.trades[0]
    assert trade["entry_price"] == 100.1 and trade["exit_price"] == 99.9 and trade["exit_reason"] == "signal_exit"
    lost = 10_000.0 - result.equity.iloc[-1]
    assert lost == pytest.approx(stats["costs"]["total"], abs=0.02) and lost > 0
    assert stats["costs"]["commission_paid"] == pytest.approx(float(result.book.fees)) and stats["costs"]["commission_paid"] > 19
    _reconciles(result)


# ------------------------------------------------------------------ stops and targets move the equity curve
def _fall_then_recover() -> pd.DataFrame:
    # Bought at 100 on the second bar. On the fifth bar the price opens at 96 and trades down to 92,
    # passing through a 5% stop at 95 during the day. It later recovers to 150.
    rows = [(100, 100, 100, 100), (100, 100.1, 99.9, 100), (99, 99.1, 98.9, 99), (96, 96.1, 95.9, 96), (96, 96, 92, 93)]
    rows += [(p, p * 1.001, p * 0.999, p) for p in (90, 95, 105, 120, 135, 150)]
    return bars(rows)


def test_a_stop_loss_exit_takes_the_account_to_cash_and_the_equity_curve_shows_it():
    frame = _fall_then_recover()
    stopped = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("1000"), stop_loss_pct=Decimal("0.05")))
    held = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("1000")))
    trade = stopped.trades[0]
    assert trade["exit_reason"] == "stop_loss" and trade["exit_price"] == 95.0 and stopped.counters["stop_exits"] == 1
    exit_at = list(frame.index.date.astype(str)).index(trade["exit_date"])
    after = stopped.equity.iloc[exit_at:]
    assert after.nunique() == 1 and after.iloc[0] == 950.0          # flat after the stop: the curve reflects the exit
    assert (stopped.invested.iloc[exit_at:] == 0).all()
    assert held.equity.iloc[-1] == 1500.0                             # without the stop the same account rides the recovery
    assert stopped.equity.iloc[-1] == stopped.cash.iloc[-1] == 950.0
    assert len(stopped.trades) == 1                                   # no re-entry while the signal never reset
    _reconciles(stopped)
    _reconciles(held)


def test_a_gap_through_the_stop_is_filled_at_the_open_not_at_the_stop():
    frame = bars([(100, 100, 100, 100), (100, 100, 100, 100), (80, 82, 79, 81), (81, 81, 81, 81)])
    result = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("1000"), stop_loss_pct=Decimal("0.05")))
    assert result.trades[0]["exit_price"] == 80.0 and result.trades[0]["exit_reason"] == "stop_loss"
    assert result.equity.iloc[-1] == 800.0


def test_take_profit_exits_at_the_target_and_locks_in_the_gain():
    frame = bars([(100, 100, 100, 100), (100, 101, 99, 100), (104, 112, 103, 108), (120, 125, 119, 124)])
    result = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("1000"), take_profit_pct=Decimal("0.10")))
    assert result.trades[0]["exit_reason"] == "take_profit" and result.trades[0]["exit_price"] == 110.0
    assert result.equity.iloc[-1] == 1100.0 and result.counters["target_exits"] == 1


def test_when_stop_and_target_are_both_touched_the_stop_is_assumed_first():
    frame = bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 115, 90, 100), (100, 100, 100, 100)])
    result = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("1000"), stop_loss_pct=Decimal("0.05"), take_profit_pct=Decimal("0.10")))
    assert result.trades[0]["exit_reason"] == "stop_loss" and result.equity.iloc[-1] == 950.0


def test_after_a_stop_the_strategy_waits_for_its_signal_to_reset():
    path = [100, 100, 90, 90, 90, 90, 90, 90]
    frame = bars([(p, p, p, p) for p in path])
    target = pd.Series([1, 1, 1, 0, 1, 1, 1, 1], index=frame.index, dtype=float)
    result = run_backtest(frame, target, _free(initial_cash=Decimal("1000"), stop_loss_pct=Decimal("0.05")))
    assert [t["exit_reason"] for t in result.trades] == ["stop_loss", "open_at_end"]
    assert result.trades[1]["entry_date"] == frame.index[5].date().isoformat()  # signal reset on bar 3, back on at bar 4, filled bar 5


# ------------------------------------------------------------------ comparisons on equal terms
def test_buy_and_hold_as_a_strategy_equals_the_buy_and_hold_baseline():
    """The fault found in the audit: the same strategy scored 5.55% against a 17.00% 'baseline'."""
    frame = make_ohlcv(756, seed=5, drift=0.0004, vol=0.015)
    out = _run_backtest(frame, LONG(frame), stop_loss_pct=None, target_pct=None, min_dollar_volume=None)
    assert out["total_return_pct"] == out["buy_hold_return_pct"]
    assert out["excess_return_vs_buy_hold_pct"] == 0.0 and out["total_trades"] == 0 and out["open_trades"] == 1
    assert out["baselines"]["cash_only"]["total_return_pct"] == 0.0
    assert out["baselines"]["buy_and_hold_fully_invested"]["total_return_pct"] != out["buy_hold_return_pct"]  # 100% is reported separately


def test_position_size_is_a_fixed_share_of_equity():
    frame = flat_ohlcv([100.0] * 5)
    result = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("10000"), position_fraction=Decimal("0.25")))
    assert result.invested.iloc[-1] == 2500.0 and result.cash.iloc[-1] == 7500.0
    assert summarise(result)["average_invested_pct"] == pytest.approx(25.0 * 4 / 5)


def test_whole_share_mode_buys_only_whole_shares_and_fractional_mode_uses_all_the_cash():
    frame = flat_ohlcv([33.0] * 4)
    whole = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("50"), fractional_shares=False))
    assert whole.book.quantity("ASSET") == Decimal("1") and whole.cash.iloc[-1] == 17.0
    fractional = run_backtest(frame, LONG(frame), _free(initial_cash=Decimal("50")))
    assert fractional.book.quantity("ASSET") == Decimal("1.515151") and fractional.cash.iloc[-1] < 0.01


def test_a_minimum_commission_is_respected_and_sizing_terminates():
    frame = flat_ohlcv([200.0] * 4)
    costs = CostModel(commission_rate=Decimal("0"), commission_minimum=Decimal("1.00"), half_spread_bps=Decimal("0"), slippage_bps=Decimal("0"))
    result = run_backtest(frame, LONG(frame), BacktestConfig(initial_cash=Decimal("50"), costs=costs))
    assert result.book.fees == Decimal("1.00") and result.book.quantity("ASSET") == Decimal("0.245") and result.cash.iloc[-1] == 0.0
    tiny = run_backtest(frame, LONG(frame), BacktestConfig(initial_cash=Decimal("0.50"), costs=costs))
    assert tiny.trades == [] and tiny.counters["entries_skipped_no_cash"] >= 1 and tiny.equity.iloc[-1] == 0.5


def test_illiquid_days_are_skipped_and_counted():
    frame = flat_ohlcv([10.0] * 6, volume=1000.0)  # $10,000 a day
    result = run_backtest(frame, LONG(frame), _free(min_dollar_volume=5_000_000))
    assert result.trades == [] and result.counters["entries_skipped_illiquid"] == 5


def test_dividends_are_paid_in_cash_on_the_ex_date_to_shares_already_held():
    frame = flat_ohlcv([100.0] * 5)
    frame["Dividends"] = [0, 0, 0, 0.5, 0]
    result = buy_and_hold(frame, _free(initial_cash=Decimal("1000")))
    assert result.book.dividends == Decimal("5.00") and result.equity.iloc[-1] == 1005.0 and result.counters["dividends_paid"] == 1
    frame["Dividends"] = [0, 0.5, 0, 0, 0]  # ex-date is the entry day: bought at that open, not entitled
    assert buy_and_hold(frame, _free(initial_cash=Decimal("1000"))).book.dividends == Decimal("0")


def test_a_bar_with_no_price_is_not_traded_and_the_last_close_is_carried():
    frame = flat_ohlcv([100.0, 100.0, 100.0, 110.0, 110.0])
    frame.iloc[2, :4] = np.nan
    result = buy_and_hold(frame, _free(initial_cash=Decimal("1000")))
    assert result.counters["bars_without_prices"] == 1 and result.equity.tolist() == [1000.0, 1000.0, 1000.0, 1100.0, 1100.0]


# ------------------------------------------------------------------ no look-ahead
def test_changing_later_bars_does_not_change_earlier_equity():
    frame = make_ohlcv(300, seed=7)
    config = BacktestConfig(stop_loss_pct=Decimal("0.05"), take_profit_pct=Decimal("0.10"))
    first = run_backtest(frame, sma_crossover(frame), config)
    altered = frame.copy()
    altered.iloc[200:, :4] *= 1.7
    second = run_backtest(altered, sma_crossover(altered), config)
    pd.testing.assert_series_equal(first.equity.iloc[:200], second.equity.iloc[:200])
    assert not first.equity.iloc[200:].equals(second.equity.iloc[200:])


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_signals_use_only_past_bars_and_books_reconcile(name):
    frame = make_ohlcv(400, seed=8)
    whole = STRATEGIES[name](frame)
    part = STRATEGIES[name](frame.iloc[:250])
    pd.testing.assert_series_equal(whole.iloc[:250], part, check_names=False)
    assert set(np.unique(whole.dropna())) <= {0.0, 1.0}
    result = run_backtest(frame, whole, BacktestConfig(stop_loss_pct=Decimal("0.05"), take_profit_pct=Decimal("0.10"), position_fraction=Decimal("0.5")))
    _reconciles(result)
    result.book.check()


def test_probability_threshold_strategy():
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    probability = pd.Series([0.50, 0.61, 0.58, np.nan, 0.49, 0.62, 0.55], index=index)
    assert probability_threshold(probability, enter=0.60, leave=0.50).tolist() == [0, 1, 1, 1, 0, 1, 1]


# ------------------------------------------------------------------ metrics and the API shape
def test_metrics_are_only_reported_where_they_mean_something():
    short = summarise(buy_and_hold(make_ohlcv(100, seed=9)))
    assert short["sharpe_ratio"] is None and short["annualized_return_pct"] is None and short["annualized_volatility_pct"] is not None
    long = summarise(buy_and_hold(make_ohlcv(600, seed=9)))
    assert long["sharpe_ratio"] is not None and long["annualized_return_pct"] is not None and long["years"] == pytest.approx(600 / 252, abs=0.01)
    assert max_drawdown(pd.Series([100.0, 120.0, 90.0, 130.0, 117.0])) == pytest.approx(-0.25)


def test_win_rate_and_profit_factor_come_from_the_trade_list():
    path = [100, 100, 110, 110, 110, 100, 100, 100, 104, 104]
    frame = bars([(p, p, p, p) for p in path])
    target = pd.Series([1, 1, 0, 1, 1, 0, 1, 1, 0, 0], index=frame.index, dtype=float)
    result = run_backtest(frame, target, _free(initial_cash=Decimal("1000")))
    stats = summarise(result)
    assert [t["outcome"] for t in result.trades] == ["win", "loss", "win"]
    assert stats["total_trades"] == 3 and stats["win_rate_pct"] == pytest.approx(66.67, abs=0.01)
    wins = sum(t["pnl"] for t in result.trades if t["pnl"] > 0)
    losses = -sum(t["pnl"] for t in result.trades if t["pnl"] <= 0)
    assert stats["profit_factor"] == pytest.approx(wins / losses, abs=1e-3)
    assert result.equity.iloc[-1] == pytest.approx(1000 + wins - losses, abs=0.01)
    _reconciles(result)


def test_api_shape_keeps_every_key_the_interface_reads():
    frame = make_ohlcv(400, seed=10)
    out = _run_backtest(frame, sma_crossover(frame))
    for key in ("total_return_pct", "buy_hold_return_pct", "annualized_return_pct", "annualized_volatility_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades",
                "win_rate_pct", "avg_win_pct", "avg_loss_pct", "profit_factor", "average_reward_risk", "false_positive_rate_pct", "invalid_trades", "final_equity",
                "initial_capital", "trades", "equity_curve", "execution_assumptions", "baselines", "costs", "exposure_pct", "turnover_per_year"):
        assert key in out, key
    assumptions = out["execution_assumptions"]
    assert assumptions["max_position_pct"] == 25.0 and assumptions["stop_loss_pct"] == 5.0 and assumptions["target_pct"] == 10.0
    assert assumptions["one_way_cost_pct"] == pytest.approx(0.33) and assumptions["round_trip_cost_pct"] == pytest.approx(0.66)
    assert out["equity_curve"][-1]["equity"] == out["final_equity"]
    for trade in out["trades"]:
        assert {"entry_date", "exit_date", "entry_price", "exit_price", "pnl_pct", "exit_reason", "outcome"} <= set(trade)


def test_invalid_configuration_is_refused():
    frame = make_ohlcv(30, seed=11)
    for bad in (dict(position_fraction=Decimal("1.5")), dict(position_fraction=Decimal("0")), dict(initial_cash=Decimal("0")), dict(stop_loss_pct=Decimal("1.2"))):
        with pytest.raises(ValidationFailed):
            run_backtest(frame, LONG(frame), BacktestConfig(**bad))
    with pytest.raises(ValidationFailed):
        run_backtest(frame.drop(columns=["Low"]), LONG(frame))
    with pytest.raises(ValidationFailed):
        run_backtest(pd.concat([frame, frame.iloc[[-1]]]), LONG(pd.concat([frame, frame.iloc[[-1]]])))
