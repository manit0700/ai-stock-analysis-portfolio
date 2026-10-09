"""The portfolio book: cash, positions, fills and profit, in one place.

Both the backtester and the virtual trading account keep their state in a ``Book``. Every change
goes through ``buy``, ``sell`` or ``credit_dividend``; each appends an entry to the book's ledger
and each is refused if it would break a rule. Nothing else may alter cash or positions, which is
why a trade list and an equity curve built from one book cannot disagree.

Rules enforced here
    * cash can never go below zero (no margin, no leverage);
    * a position can never go below zero (no short selling);
    * quantities and prices must be positive; fees cannot be negative.

Cost basis uses the average-cost method and includes purchase fees. Realised profit on a sale is
``proceeds - sale fee - average cost of the shares sold``.

Identities that always hold, checked by ``Book.check``
    cash + cost of open positions == starting cash + realised profit
    equity (cash + market value) == starting cash + realised profit + unrealised profit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from app.accounting.money import CENT, SHARE_STEP, ZERO, D, cents, cents_down, cents_up, money_str, shares
from app.core.errors import OrderRejected, ValidationFailed


class InsufficientCash(OrderRejected):
    code = "insufficient_cash"


class InsufficientShares(OrderRejected):
    code = "insufficient_shares"


@dataclass
class Position:
    symbol: str
    quantity: Decimal = ZERO
    cost_basis: Decimal = ZERO  # total cost of the shares held, purchase fees included

    @property
    def average_cost(self) -> Decimal:
        return (self.cost_basis / self.quantity) if self.quantity > 0 else ZERO


@dataclass(frozen=True)
class Entry:
    """One line of the book's ledger. Never edited after it is written."""

    seq: int
    at: str
    kind: str  # "buy" | "sell" | "dividend"
    symbol: str
    quantity: Decimal
    price: Decimal
    gross: Decimal       # quantity x price, rounded as cash moves
    fee: Decimal
    cash_change: Decimal
    cash_after: Decimal
    realized: Decimal    # profit realised by this entry (0 for a buy)
    reference: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "at": self.at, "kind": self.kind, "symbol": self.symbol, "quantity": str(self.quantity), "price": str(self.price),
            "gross": money_str(self.gross), "fee": money_str(self.fee), "cash_change": money_str(self.cash_change), "cash_after": money_str(self.cash_after),
            "realized": money_str(self.realized), "reference": self.reference, "note": self.note,
        }


@dataclass
class Book:
    starting_cash: Decimal
    cash: Decimal = field(default=None)  # type: ignore[assignment]
    positions: dict[str, Position] = field(default_factory=dict)
    realized: Decimal = ZERO          # trading profit and loss, fees included
    dividends: Decimal = ZERO
    fees: Decimal = ZERO
    entries: list[Entry] = field(default_factory=list)
    keep_entries: bool = True

    def __post_init__(self) -> None:
        self.starting_cash = cents(D(self.starting_cash))
        if self.starting_cash < 0:
            raise ValidationFailed("Starting cash cannot be negative.")
        if self.cash is None:
            self.cash = self.starting_cash

    # ------------------------------------------------------------------ reads
    def quantity(self, symbol: str) -> Decimal:
        position = self.positions.get(symbol)
        return position.quantity if position else ZERO

    def market_value(self, prices: Mapping[str, Any]) -> Decimal:
        total = ZERO
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                if symbol not in prices or prices[symbol] is None:
                    raise ValidationFailed(f"No price was supplied to value the {symbol} position.", details={"symbol": symbol})
                total += cents_down(position.quantity * D(prices[symbol]))
        return total

    def equity(self, prices: Mapping[str, Any]) -> Decimal:
        return self.cash + self.market_value(prices)

    def cost_of_open_positions(self) -> Decimal:
        return sum((p.cost_basis for p in self.positions.values()), ZERO)

    def unrealized(self, prices: Mapping[str, Any]) -> Decimal:
        return self.market_value(prices) - self.cost_of_open_positions()

    def total_realized(self) -> Decimal:
        """Trading profit plus dividends received."""
        return self.realized + self.dividends

    # ------------------------------------------------------------------ changes
    def _append(self, **fields: Any) -> Entry:
        entry = Entry(seq=len(self.entries) + 1 if self.keep_entries else 0, cash_after=self.cash, **fields)
        if self.keep_entries:
            self.entries.append(entry)
        return entry

    @staticmethod
    def _validate(symbol: str, quantity: Decimal, price: Decimal, fee: Decimal) -> None:
        if not symbol:
            raise ValidationFailed("A symbol is required.")
        if quantity <= 0:
            raise OrderRejected("The quantity must be greater than zero.", code="invalid_quantity")
        if quantity != shares(quantity):
            raise OrderRejected(f"Quantities are limited to {abs(SHARE_STEP.as_tuple().exponent)} decimal places.", code="invalid_quantity")
        if price <= 0:
            raise OrderRejected("The price must be greater than zero.", code="invalid_price")
        if fee < 0:
            raise OrderRejected("A fee cannot be negative.", code="invalid_fee")

    def buy(self, symbol: str, quantity: Any, price: Any, *, fee: Any = ZERO, at: str = "", reference: str = "", note: str = "") -> Entry:
        quantity, price, fee = D(quantity), D(price), cents_up(D(fee))
        self._validate(symbol, quantity, price, fee)
        gross = cents_up(quantity * price)
        total = gross + fee
        if total > self.cash:
            raise InsufficientCash(
                f"This purchase costs {money_str(total)} and the account holds {money_str(self.cash)}.",
                details={"required": money_str(total), "available": money_str(self.cash), "symbol": symbol},
            )
        position = self.positions.setdefault(symbol, Position(symbol))
        self.cash -= total
        position.quantity += quantity
        position.cost_basis += total
        self.fees += fee
        return self._append(at=at, kind="buy", symbol=symbol, quantity=quantity, price=price, gross=gross, fee=fee, cash_change=-total, realized=ZERO, reference=reference, note=note)

    def sell(self, symbol: str, quantity: Any, price: Any, *, fee: Any = ZERO, at: str = "", reference: str = "", note: str = "") -> Entry:
        quantity, price, fee = D(quantity), D(price), cents_up(D(fee))
        self._validate(symbol, quantity, price, fee)
        position = self.positions.get(symbol)
        held = position.quantity if position else ZERO
        if quantity > held:
            raise InsufficientShares(
                f"The account holds {held} {symbol} and cannot sell {quantity}. Short selling is not allowed.",
                details={"held": str(held), "requested": str(quantity), "symbol": symbol},
            )
        gross = cents_down(quantity * price)
        if fee > gross:
            raise OrderRejected("The fee is larger than the sale proceeds.", code="fee_exceeds_proceeds", details={"proceeds": money_str(gross), "fee": money_str(fee)})
        proceeds = gross - fee
        # Cost removed is proportional, except that selling everything removes the whole basis so no cent is stranded.
        cost_removed = position.cost_basis if quantity == held else cents(position.cost_basis * quantity / held)
        realized = proceeds - cost_removed
        self.cash += proceeds
        position.quantity -= quantity
        position.cost_basis -= cost_removed
        if position.quantity == 0:
            del self.positions[symbol]
        self.fees += fee
        self.realized += realized
        return self._append(at=at, kind="sell", symbol=symbol, quantity=quantity, price=price, gross=gross, fee=fee, cash_change=proceeds, realized=realized, reference=reference, note=note)

    def credit_dividend(self, symbol: str, per_share: Any, *, at: str = "", reference: str = "") -> Entry | None:
        """Pay a cash dividend on shares held. Returns ``None`` when nothing is held or the amount rounds to zero."""
        per_share = D(per_share)
        held = self.quantity(symbol)
        if per_share <= 0 or held <= 0:
            return None
        amount = cents_down(held * per_share)
        if amount < CENT:
            return None
        self.cash += amount
        self.dividends += amount
        return self._append(at=at, kind="dividend", symbol=symbol, quantity=held, price=per_share, gross=amount, fee=ZERO, cash_change=amount, realized=amount, reference=reference)

    # ------------------------------------------------------------------ verification
    def check(self, prices: Mapping[str, Any] | None = None) -> None:
        """Raise ``AssertionError`` if any accounting identity is broken."""
        assert self.cash >= 0, f"cash is negative: {self.cash}"
        assert self.cash == cents(self.cash), "cash is not a whole number of cents"
        for symbol, position in self.positions.items():
            assert position.quantity > 0, f"{symbol}: non-positive quantity {position.quantity}"
            assert position.cost_basis >= 0, f"{symbol}: negative cost basis"
        left = self.cash + self.cost_of_open_positions()
        right = self.starting_cash + self.total_realized()
        assert left == right, f"cash + open cost ({left}) != starting cash + realised ({right})"
        if self.keep_entries:
            running = self.starting_cash
            for entry in self.entries:
                running += entry.cash_change
                assert running == entry.cash_after, f"ledger entry {entry.seq} does not reconcile"
            assert running == self.cash, "the ledger does not add up to the cash balance"
            assert sum((e.realized for e in self.entries), ZERO) == self.total_realized(), "realised profit does not match the ledger"
            assert sum((e.fee for e in self.entries), ZERO) == self.fees, "fees do not match the ledger"
        if prices is not None:
            assert self.equity(prices) == self.starting_cash + self.total_realized() + self.unrealized(prices), "equity identity broken"

    def snapshot(self, prices: Mapping[str, Any]) -> dict[str, Any]:
        """Balances and positions as plain values, with marks from ``prices``."""
        rows = []
        for symbol in sorted(self.positions):
            position = self.positions[symbol]
            mark = D(prices[symbol]) if symbol in prices and prices[symbol] is not None else None
            value = cents_down(position.quantity * mark) if mark is not None else None
            rows.append({
                "symbol": symbol, "quantity": str(position.quantity), "average_cost": str(position.average_cost.quantize(Decimal("0.0001"))),
                "cost_basis": money_str(position.cost_basis), "price": None if mark is None else str(mark),
                "market_value": None if value is None else money_str(value),
                "unrealized_pnl": None if value is None else money_str(value - position.cost_basis),
            })
        complete = all(row["market_value"] is not None for row in rows)
        market_value = sum((D(row["market_value"]) for row in rows), ZERO) if complete else None
        return {
            "starting_cash": money_str(self.starting_cash), "cash": money_str(self.cash),
            "positions": rows,
            "market_value": None if market_value is None else money_str(market_value),
            "equity": None if market_value is None else money_str(self.cash + market_value),
            "realized_pnl": money_str(self.realized), "dividends": money_str(self.dividends), "fees_paid": money_str(self.fees),
            "unrealized_pnl": None if market_value is None else money_str(market_value - self.cost_of_open_positions()),
            "total_pnl": None if market_value is None else money_str(self.cash + market_value - self.starting_cash),
        }
