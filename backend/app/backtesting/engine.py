"""A bar-by-bar backtester in which the trade list and the equity curve cannot disagree.

Everything happens through one ``Book`` (``app.accounting``): fills change its cash and position,
and the equity curve is that same book valued at each close. A stop-loss or take-profit exit is a
sale in the book like any other, so it moves the equity curve as well as the trade list.

Timing, chosen to rule out look-ahead
    * A strategy decides at the **close** of day ``t`` whether it wants to be long (1) or flat (0).
    * That decision is carried out at the **open** of day ``t + 1``, at the open price moved
      against the trader by half the spread plus slippage, plus commission.
    * While a position is open, each day's low and high are checked against the stop and target.
      If both are touched on the same day the stop is assumed to have come first. If the day
      opens beyond the stop or target, the fill is the open, not the level.
    * After a stop or target exit the strategy stays flat until its signal has gone to 0 and back.

What this does and does not model
    * Cash, fractional or whole shares, a fixed share of equity per position, commission, spread,
      slippage, dividends paid in cash on the ex-date, and a liquidity filter.
    * Prices may be as traded: a ``Stock Splits`` column, when present, scales the share count on
      the ex-date (cost basis unchanged) and rebases the open trade's entry for its stop and
      target, so a split is not booked as a loss. One instrument at a time. Long only. No margin.
    * Fills are assumed at the open price whatever the order size; nothing here models market
      impact, partial fills, halts, or the overnight gap between a decision and its execution
      beyond using the next open.

Synthetic price paths test that the arithmetic is right. They never show that a strategy is
profitable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal
from typing import Any

import numpy as np
import pandas as pd

from app.accounting.book import Book
from app.accounting.money import BPS, CENT, ONE, SHARE_STEP, ZERO, D, cents_down, cents_up, price4, shares
from app.core.errors import ValidationFailed

REQUIRED = ("Open", "High", "Low", "Close")


@dataclass(frozen=True)
class CostModel:
    """Trading costs. Spread and slippage move the fill price; commission is a separate fee."""

    commission_rate: Decimal = Decimal("0")       # fraction of the fill's value
    commission_minimum: Decimal = Decimal("0")    # per fill, in dollars
    half_spread_bps: Decimal = Decimal("2")       # half the bid-ask spread, in basis points
    slippage_bps: Decimal = Decimal("3")          # further adverse move, in basis points

    def _adverse(self) -> Decimal:
        return (D(self.half_spread_bps) + D(self.slippage_bps)) * BPS

    def buy_price(self, reference: Decimal) -> Decimal:
        return price4(reference * (ONE + self._adverse()))

    def sell_price(self, reference: Decimal) -> Decimal:
        return price4(reference * (ONE - self._adverse()))

    def fee(self, value: Decimal) -> Decimal:
        rate, minimum = D(self.commission_rate), D(self.commission_minimum)
        if rate == 0 and minimum == 0:
            return ZERO
        return cents_up(max(value * rate, minimum))

    def describe(self) -> dict[str, Any]:
        return {
            "commission_rate_pct": float(D(self.commission_rate) * 100), "commission_minimum": float(D(self.commission_minimum)),
            "half_spread_bps": float(D(self.half_spread_bps)), "slippage_bps": float(D(self.slippage_bps)),
            "one_way_price_cost_pct": float(self._adverse() * 100),
            "round_trip_cost_pct": float((self._adverse() + D(self.commission_rate)) * 2 * 100),
        }


FREE = CostModel(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"))


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: Decimal = Decimal("10000")
    position_fraction: Decimal = Decimal("1")      # share of equity put into a new position
    stop_loss_pct: Decimal | None = None           # e.g. Decimal("0.05") exits 5% below the entry fill
    take_profit_pct: Decimal | None = None
    costs: CostModel = field(default_factory=CostModel)
    fractional_shares: bool = True
    min_dollar_volume: float | None = None         # skip an entry when the previous day's traded value is below this
    symbol: str = "ASSET"

    def validate(self) -> None:
        if D(self.initial_cash) <= 0:
            raise ValidationFailed("Starting capital must be greater than zero.")
        if not (ZERO < D(self.position_fraction) <= ONE):
            raise ValidationFailed("The position fraction must be above 0 and at most 1 (no leverage).")
        for name in ("stop_loss_pct", "take_profit_pct"):
            value = getattr(self, name)
            if value is not None and not (ZERO < D(value) < ONE):
                raise ValidationFailed(f"{name} must be between 0 and 1.")
        adverse = (D(self.costs.half_spread_bps) + D(self.costs.slippage_bps)) * BPS
        if self.stop_loss_pct is not None and D(self.stop_loss_pct) <= 2 * adverse:
            raise ValidationFailed(
                f"stop_loss_pct ({float(D(self.stop_loss_pct)):.4%}) is inside the round-trip spread and slippage ({float(2 * adverse):.4%}): "
                "entries would stop out instantly at prices the market never traded.",
            )


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series          # account value at each close
    cash: pd.Series
    invested: pd.Series        # market value of the position at each close
    trades: list[dict[str, Any]]
    book: Book
    counters: dict[str, int]
    price_cost: Decimal        # money lost to spread and slippage (difference between fills and reference prices)


def _price(value: float) -> Decimal:
    return D(repr(float(value)))


def run_backtest(bars: pd.DataFrame, target: pd.Series, config: BacktestConfig | None = None) -> BacktestResult:
    """Simulate holding ``config.symbol`` whenever ``target`` (decided at each close) is 1."""
    config = config or BacktestConfig()
    config.validate()
    absent = [c for c in REQUIRED if c not in bars.columns]
    if absent:
        raise ValidationFailed(f"The price history is missing columns: {absent}")
    if len(bars) < 2:
        raise ValidationFailed("At least two bars are needed.")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValidationFailed("Bars must be in date order with no repeated dates.")
    target = target.reindex(bars.index)

    symbol = config.symbol
    book = Book(D(config.initial_cash))
    opens, highs, lows, closes = (bars[c].to_numpy(float) for c in REQUIRED)
    volumes = bars["Volume"].to_numpy(float) if "Volume" in bars.columns else np.full(len(bars), np.nan)
    dividends = bars["Dividends"].to_numpy(float) if "Dividends" in bars.columns else np.zeros(len(bars))
    splits = bars["Stock Splits"].to_numpy(float) if "Stock Splits" in bars.columns else np.zeros(len(bars))
    wants = np.where(np.isfinite(target.to_numpy(float)), target.to_numpy(float) > 0, False)

    n = len(bars)
    equity, cash, invested = np.empty(n), np.empty(n), np.empty(n)
    trades: list[dict[str, Any]] = []
    counters = {"bars": n, "bars_without_prices": 0, "entries_skipped_illiquid": 0, "entries_skipped_no_cash": 0, "stop_exits": 0, "target_exits": 0, "signal_exits": 0, "dividends_paid": 0}
    price_cost = ZERO
    open_trade: dict[str, Any] | None = None
    wanted_yesterday = False
    locked_out = False
    last_close: Decimal | None = None
    stop_pct = None if config.stop_loss_pct is None else D(config.stop_loss_pct)
    target_pct = None if config.take_profit_pct is None else D(config.take_profit_pct)

    def close_trade(day: pd.Timestamp, reference: Decimal, reason: str, index: int) -> None:
        nonlocal open_trade, price_cost
        quantity = book.quantity(symbol)
        fill = config.costs.sell_price(reference)
        value = cents_down(quantity * fill)
        fee = min(config.costs.fee(value), value)  # a forced exit must complete; the fee cannot exceed the proceeds
        entry = book.sell(symbol, quantity, fill, fee=fee, at=day.date().isoformat(), reference=reason)
        price_cost += cents_down(quantity * reference) - value
        cost = open_trade["cost"]
        trades.append({
            "entry_date": open_trade["date"], "exit_date": day.date().isoformat(), "quantity": str(quantity),
            "entry_price": float(open_trade["fill"]), "exit_price": float(fill), "entry_fee": float(open_trade["fee"]), "exit_fee": float(fee),
            "pnl": float(entry.realized), "pnl_pct": float(entry.realized / cost * 100) if cost > 0 else 0.0,
            "gross_pnl_pct": float((reference / open_trade["reference"] - 1) * 100),
            "exit_reason": reason, "bars_held": index - open_trade["index"], "outcome": "win" if entry.realized > 0 else "loss",
        })
        open_trade = None

    for i in range(n):
        day = bars.index[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]  # noqa: E741
        tradable = bool(np.isfinite([o, h, l, c]).all() and min(o, h, l, c) > 0)
        if not tradable:
            counters["bars_without_prices"] += 1
        else:
            # 0. a split scales the share count; the open trade's entry rebases so stops and targets follow
            if np.isfinite(splits[i]) and splits[i] > 0 and splits[i] != 1 and book.quantity(symbol) > 0:
                ratio = _price(splits[i])
                book.apply_split(symbol, ratio, at=day.date().isoformat())
                if open_trade is not None:
                    open_trade["fill"] = price4(open_trade["fill"] / ratio)
                    open_trade["reference"] = price4(open_trade["reference"] / ratio)
                counters["splits_applied"] = counters.get("splits_applied", 0) + 1

            # 1. a dividend goes to whoever held the shares coming into the ex-date
            if np.isfinite(dividends[i]) and dividends[i] > 0 and book.quantity(symbol) > 0:
                if book.credit_dividend(symbol, _price(dividends[i]), at=day.date().isoformat()) is not None:
                    counters["dividends_paid"] += 1

            # 2. carry out, at this open, what was decided at the previous close
            if not wanted_yesterday:
                locked_out = False
            held = book.quantity(symbol)
            entered_today = False
            if wanted_yesterday and not locked_out and held == 0 and i > 0:
                liquid = True
                if config.min_dollar_volume is not None:
                    traded = closes[i - 1] * volumes[i - 1]
                    liquid = bool(np.isfinite(traded) and traded >= config.min_dollar_volume)
                if not liquid:
                    counters["entries_skipped_illiquid"] += 1
                else:
                    reference = _price(o)
                    fill = config.costs.buy_price(reference)
                    budget = cents_down(book.cash * D(config.position_fraction))
                    limit = min(budget, book.cash)

                    def sized(amount: Decimal) -> Decimal:
                        size = shares(amount / fill) if amount > 0 else ZERO
                        return size if config.fractional_shares else size.to_integral_value(rounding=ROUND_FLOOR)

                    # Largest quantity whose cost plus fee fits the budget. The fee depends on the
                    # quantity, so start from an estimate and correct it; this settles in two passes.
                    quantity = sized(limit / (ONE + D(config.costs.commission_rate)))
                    fee, margin = ZERO, ZERO
                    step = SHARE_STEP if config.fractional_shares else ONE
                    for _ in range(8):
                        if quantity <= 0:
                            break
                        value = cents_up(quantity * fill)
                        fee = config.costs.fee(value)
                        if value + fee <= limit:
                            break
                        smaller = sized(limit - fee - margin)
                        quantity = smaller if smaller < quantity else quantity - step
                        margin += CENT  # if rounding still leaves it a cent over, leave a cent spare next time
                    else:
                        quantity = ZERO
                    if quantity > 0:
                        entry = book.buy(symbol, quantity, fill, fee=fee, at=day.date().isoformat(), reference="entry")
                        price_cost += cents_up(quantity * fill) - cents_down(quantity * reference)
                        open_trade = {"date": day.date().isoformat(), "fill": fill, "reference": reference, "fee": fee, "cost": -entry.cash_change, "index": i}
                        entered_today = True
                    else:
                        counters["entries_skipped_no_cash"] += 1
            elif not wanted_yesterday and held > 0 and open_trade is not None:
                close_trade(day, _price(o), "signal_exit", i)
                counters["signal_exits"] += 1

            # 3. stop-loss and take-profit, checked against this bar's range
            if open_trade is not None and book.quantity(symbol) > 0:
                entry_fill = open_trade["fill"]
                exit_reference: Decimal | None = None
                reason = ""
                if stop_pct is not None:
                    stop_level = entry_fill * (ONE - stop_pct)
                    if not entered_today and _price(o) <= stop_level:
                        exit_reference, reason = _price(o), "stop_loss"       # opened below the stop: filled at the open
                    elif _price(l) <= stop_level:
                        exit_reference, reason = price4(stop_level), "stop_loss"
                if exit_reference is None and target_pct is not None:
                    target_level = entry_fill * (ONE + target_pct)
                    if not entered_today and _price(o) >= target_level:
                        exit_reference, reason = _price(o), "take_profit"     # opened above the target: filled at the open
                    elif _price(h) >= target_level:
                        exit_reference, reason = price4(target_level), "take_profit"
                if exit_reference is not None:
                    close_trade(day, exit_reference, reason, i)
                    counters["stop_exits" if reason == "stop_loss" else "target_exits"] += 1
                    locked_out = True
            last_close = _price(c)

        # 4. value the account at this close (the last known close if this bar has no price)
        held = book.quantity(symbol)
        value = cents_down(held * last_close) if held > 0 and last_close is not None else ZERO
        cash[i], invested[i], equity[i] = float(book.cash), float(value), float(book.cash + value)

        # 5. decide for tomorrow, using only what is known at this close
        wanted_yesterday = bool(wants[i]) if tradable else wanted_yesterday

    if open_trade is not None and last_close is not None:
        quantity = book.quantity(symbol)
        value = cents_down(quantity * last_close)
        trades.append({
            "entry_date": open_trade["date"], "exit_date": bars.index[-1].date().isoformat(), "quantity": str(quantity),
            "entry_price": float(open_trade["fill"]), "exit_price": float(last_close), "entry_fee": float(open_trade["fee"]), "exit_fee": 0.0,
            "pnl": float(value - open_trade["cost"]), "pnl_pct": float((value - open_trade["cost"]) / open_trade["cost"] * 100) if open_trade["cost"] > 0 else 0.0,
            "gross_pnl_pct": float((last_close / open_trade["reference"] - 1) * 100),
            "exit_reason": "open_at_end", "bars_held": n - 1 - open_trade["index"], "outcome": "open",
        })
    book.check({symbol: last_close} if last_close is not None else None)
    index = bars.index
    return BacktestResult(config, pd.Series(equity, index=index), pd.Series(cash, index=index), pd.Series(invested, index=index), trades, book, counters, price_cost)


def buy_and_hold(bars: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    """The baseline: always want to be long, on the same capital, sizing and costs, with no stop or target."""
    config = config or BacktestConfig()
    plain = BacktestConfig(config.initial_cash, config.position_fraction, None, None, config.costs, config.fractional_shares, None, config.symbol)
    return run_backtest(bars, pd.Series(1.0, index=bars.index), plain)


def cash_only(bars: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    """The other baseline: never trade."""
    config = config or BacktestConfig()
    return run_backtest(bars, pd.Series(0.0, index=bars.index), config)


__all__ = ["BacktestConfig", "BacktestResult", "CostModel", "FREE", "buy_and_hold", "cash_only", "run_backtest", "CENT"]
