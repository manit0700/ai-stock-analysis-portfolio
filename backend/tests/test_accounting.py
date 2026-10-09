"""The portfolio book: exact money arithmetic and identities that must always hold."""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from app.accounting.book import Book, InsufficientCash, InsufficientShares
from app.accounting.money import D, cents, cents_down, cents_up, money_str, shares
from app.core.errors import OrderRejected, ValidationFailed


def test_decimal_conversion_does_not_inherit_float_error():
    assert D(0.1) == Decimal("0.1") and D("19.99") == Decimal("19.99") and D(3) == Decimal(3)
    assert sum((D(0.1) for _ in range(10)), Decimal(0)) == Decimal("1.0")
    assert D(0.1) + D(0.2) == D(0.3) and 0.1 + 0.2 != 0.3  # the float behaviour this module exists to avoid
    for bad in (float("nan"), float("inf"), "abc", True, None):
        with pytest.raises(ValidationFailed):
            D(bad)


def test_rounding_always_favours_reconciliation_not_the_trader():
    assert cents_up(Decimal("10.001")) == Decimal("10.01") and cents_down(Decimal("10.009")) == Decimal("10.00")
    assert cents(Decimal("10.005")) == Decimal("10.00") and cents(Decimal("10.015")) == Decimal("10.02")
    assert shares(Decimal("1.23456789")) == Decimal("1.234567")
    assert money_str(Decimal("5")) == "5.00"


def test_buy_then_sell_with_fees_gives_the_expected_cash_and_profit():
    book = Book(Decimal("50"))
    book.buy("AAPL", "0.2", "200.00", fee="0.10")          # cost 40.00 + 0.10
    assert book.cash == Decimal("9.90") and book.quantity("AAPL") == Decimal("0.2")
    assert book.positions["AAPL"].cost_basis == Decimal("40.10") and book.positions["AAPL"].average_cost == Decimal("200.5")
    assert book.equity({"AAPL": "210"}) == Decimal("51.90") and book.unrealized({"AAPL": "210"}) == Decimal("1.90")
    sale = book.sell("AAPL", "0.2", "210.00", fee="0.10")   # proceeds 42.00 - 0.10
    assert sale.realized == Decimal("1.80") and book.cash == Decimal("51.80") and book.realized == Decimal("1.80")
    assert book.fees == Decimal("0.20") and "AAPL" not in book.positions
    book.check({})
    assert [e.kind for e in book.entries] == ["buy", "sell"] and book.entries[-1].cash_after == Decimal("51.80")


def test_average_cost_and_partial_sales():
    book = Book(1000)
    book.buy("X", 2, 100)
    book.buy("X", 2, 110)
    assert book.positions["X"].average_cost == Decimal("105")
    first = book.sell("X", 1, 120)
    assert first.realized == Decimal("15.00") and book.positions["X"].cost_basis == Decimal("315.00")
    rest = book.sell("X", 3, 100)
    assert rest.realized == Decimal("-15.00") and book.realized == Decimal("0.00") and book.cash == Decimal("1000.00")
    book.check({})


def test_cash_can_never_go_negative():
    book = Book(Decimal("50"))
    with pytest.raises(InsufficientCash) as refused:
        book.buy("AAPL", 1, "50.00", fee="0.01")
    assert refused.value.code == "insufficient_cash" and refused.value.details["available"] == "50.00"
    assert book.cash == Decimal("50.00") and not book.positions and not book.entries
    book.buy("AAPL", 1, "50.00")  # exactly all the cash is allowed
    assert book.cash == Decimal("0.00")
    book.check({"AAPL": 50})


def test_no_short_selling():
    book = Book(100)
    with pytest.raises(InsufficientShares):
        book.sell("AAPL", 1, 10)
    book.buy("AAPL", 1, 10)
    with pytest.raises(InsufficientShares) as refused:
        book.sell("AAPL", "1.000001", 10)
    assert refused.value.code == "insufficient_shares" and book.quantity("AAPL") == Decimal(1)


@pytest.mark.parametrize("quantity,price,fee,code", [
    (0, 10, 0, "invalid_quantity"), (-1, 10, 0, "invalid_quantity"), ("0.0000001", 10, 0, "invalid_quantity"),
    (1, 0, 0, "invalid_price"), (1, -5, 0, "invalid_price"), (1, 10, -1, "invalid_fee"),
])
def test_invalid_orders_are_rejected_and_change_nothing(quantity, price, fee, code):
    book = Book(100)
    with pytest.raises(OrderRejected) as refused:
        book.buy("X", quantity, price, fee=fee)
    assert refused.value.code == code and book.cash == Decimal("100.00") and not book.entries


def test_a_fee_larger_than_the_proceeds_is_rejected():
    book = Book(100)
    book.buy("X", "0.01", 10)
    with pytest.raises(OrderRejected) as refused:
        book.sell("X", "0.01", 10, fee=5)
    assert refused.value.code == "fee_exceeds_proceeds" and book.quantity("X") == Decimal("0.01")


def test_fractional_shares_are_exact_to_six_places():
    book = Book(Decimal("50"))
    book.buy("NVDA", "0.123456", "123.45")
    assert book.quantity("NVDA") == Decimal("0.123456")
    assert book.cash == Decimal("50") - cents_up(Decimal("0.123456") * Decimal("123.45"))
    book.sell("NVDA", "0.023456", "130")
    assert book.quantity("NVDA") == Decimal("0.100000")
    book.check({"NVDA": "130"})


def test_dividends_are_income_and_keep_the_identities():
    book = Book(1000)
    assert book.credit_dividend("KO", "0.46") is None  # nothing held
    book.buy("KO", 10, 60)
    entry = book.credit_dividend("KO", "0.46")
    assert entry.cash_change == Decimal("4.60") and book.dividends == Decimal("4.60") and book.total_realized() == Decimal("4.60")
    assert book.credit_dividend("KO", "0.0001") is None  # rounds to nothing
    book.check({"KO": 60})


def test_valuation_needs_a_price_for_every_position():
    book = Book(1000)
    book.buy("A", 1, 10)
    book.buy("B", 1, 10)
    with pytest.raises(ValidationFailed):
        book.equity({"A": 10})
    snapshot = book.snapshot({"A": 12})
    assert snapshot["equity"] is None and snapshot["positions"][1]["market_value"] is None
    full = book.snapshot({"A": 12, "B": 9})
    assert full["equity"] == "1001.00" and full["unrealized_pnl"] == "1.00" and full["total_pnl"] == "1.00"


def test_identities_hold_through_a_long_random_sequence_of_orders():
    rng = np.random.default_rng(42)
    book = Book(Decimal("50.00"))
    prices = {"A": Decimal("17.31"), "B": Decimal("243.77"), "C": Decimal("0.83")}
    accepted = rejected = 0
    for _ in range(1500):
        symbol = str(rng.choice(list(prices)))
        prices[symbol] = max(Decimal("0.05"), (prices[symbol] * Decimal(repr(round(float(rng.normal(1.0, 0.03)), 6)))).quantize(Decimal("0.0001")))
        quantity = shares(Decimal(repr(round(float(rng.uniform(0.000001, 0.6)), 6))))
        fee = Decimal(repr(round(float(rng.choice([0, 0, 0.01, 0.35])), 2)))
        try:
            action = rng.random()
            if action < 0.5:
                book.buy(symbol, quantity, prices[symbol], fee=fee)
            elif action < 0.95:
                book.sell(symbol, min(quantity, book.quantity(symbol)) if rng.random() < 0.7 else quantity, prices[symbol], fee=fee)
            else:
                book.credit_dividend(symbol, "0.07")
            accepted += 1
        except OrderRejected:
            rejected += 1
        book.check(prices)
        assert book.cash >= 0 and all(p.quantity > 0 for p in book.positions.values())
    assert accepted > 300 and rejected > 100
    assert book.equity(prices) == book.starting_cash + book.total_realized() + book.unrealized(prices)
