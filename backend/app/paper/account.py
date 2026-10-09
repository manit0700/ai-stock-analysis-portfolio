"""The $50 virtual trading account. Entirely simulated; no brokerage, no real money, ever.

State lives in SQLite (``<data dir>/paper.sqlite3``): accounts, orders and an event log, written
in one transaction per order so a crash cannot leave half an order. Balances are computed by the
``Book`` (``app.accounting``) replayed from the fills table — the account's money arithmetic is
code, never a language model's output, and a prediction never places an order by itself.

Rules
    * One default account starts with $50.00 of virtual cash. No deposits, no withdrawals.
    * Market orders only, long only: no margin, no leverage, no short selling, no negative cash.
    * Fractional shares to six decimal places. Configurable execution costs are applied to a
      quote-derived fill price.
    * An order needs a fresh quote: when the newest price is older than ``max_quote_age`` (or the
      market is closed and ``allow_closed`` is off), the order is rejected, not guessed.
    * Risk controls: per-position and whole-portfolio exposure caps, and a trading pause switch.
    * ``client_order_id`` makes submission idempotent: the same id returns the same order.
    * ``reset`` archives the account and opens a fresh one; history is kept, never erased.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.accounting.book import Book
from app.accounting.money import BPS, ONE, ZERO, D, cents, cents_down, cents_up, money_str, price4, shares, to_float
from app.core.clock import iso, utcnow
from app.core.config import get_settings
from app.core.errors import Conflict, MarketVisionError, NotFound, OrderRejected, ValidationFailed

STARTING_CASH = Decimal("50.00")
SIMULATION_BANNER = "SIMULATED TRADING — NO REAL MONEY"


@dataclass(frozen=True)
class PaperConfig:
    starting_cash: Decimal = STARTING_CASH
    commission_rate: Decimal = Decimal("0")          # most retail brokers charge $0 on US equities
    half_spread_bps: Decimal = Decimal("2")
    slippage_bps: Decimal = Decimal("3")
    max_position_share: Decimal = Decimal("0.40")    # one symbol may be at most this share of equity after the order
    max_invested_share: Decimal = Decimal("0.95")    # cash buffer: at most this share of equity may be invested
    max_quote_age_seconds: float = 900.0
    allow_when_market_closed: bool = False
    min_order_value: Decimal = Decimal("1.00")

    def describe(self) -> dict[str, Any]:
        return {
            "starting_cash": money_str(D(self.starting_cash)),
            "commission_rate_pct": to_float(D(self.commission_rate) * 100),
            "half_spread_bps": to_float(D(self.half_spread_bps)),
            "slippage_bps": to_float(D(self.slippage_bps)),
            "max_position_share_pct": to_float(D(self.max_position_share) * 100),
            "max_invested_share_pct": to_float(D(self.max_invested_share) * 100),
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "allow_when_market_closed": self.allow_when_market_closed,
            "min_order_value": money_str(D(self.min_order_value)),
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, starting_cash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', paused INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT, config TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, account_id TEXT NOT NULL REFERENCES accounts(id),
    client_order_id TEXT NOT NULL, at TEXT NOT NULL, side TEXT NOT NULL, symbol TEXT NOT NULL,
    request TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
    quantity TEXT, fill_price TEXT, gross TEXT, fee TEXT, cash_after TEXT, realized TEXT,
    quote TEXT,
    UNIQUE (account_id, client_order_id)
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, account_id TEXT NOT NULL, at TEXT NOT NULL,
    kind TEXT NOT NULL, payload TEXT NOT NULL
);
"""


class PaperAccountService:
    def __init__(self, market_data: Any, *, db_path: Path | None = None, config: PaperConfig | None = None) -> None:
        settings = get_settings()
        if not settings.paper_trading_enabled:
            raise MarketVisionError("Paper trading is switched off (MARKETVISION_PAPER_TRADING=0).", code="not_configured")
        self.market_data = market_data
        self.config = config or PaperConfig()
        self.db_path = Path(db_path) if db_path is not None else settings.data_dir / "paper.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        db = self._connect()
        try:
            db.executescript(_SCHEMA)
        finally:
            db.close()

    # ------------------------------------------------------------------ storage
    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)  # explicit transactions
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 10000")
        return db

    @contextmanager
    def _read(self):
        db = self._connect()
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _write(self):
        """One BEGIN IMMEDIATE transaction: the write lock is taken before any balance is read, so
        two processes cannot both see the same cash and both spend it (the reviewer's race, which
        also left the account unreadable afterwards)."""
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
        finally:
            db.close()

    def _event(self, db: sqlite3.Connection, account_id: str, kind: str, payload: dict[str, Any]) -> None:
        db.execute("INSERT INTO events (account_id, at, kind, payload) VALUES (?, ?, ?, ?)", (account_id, iso(utcnow()), kind, json.dumps(payload, default=str)))

    def _active_account(self, db: sqlite3.Connection) -> sqlite3.Row:
        row = db.execute("SELECT * FROM accounts WHERE status = 'active' ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            account_id = f"paper-{uuid.uuid4().hex[:12]}"
            db.execute(
                "INSERT INTO accounts (id, created_at, starting_cash, config) VALUES (?, ?, ?, ?)",
                (account_id, iso(utcnow()), money_str(D(self.config.starting_cash)), json.dumps(self.config.describe())),
            )
            self._event(db, account_id, "account_created", {"starting_cash": money_str(D(self.config.starting_cash)), "banner": SIMULATION_BANNER})
            row = db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return row

    def _book(self, db: sqlite3.Connection, account: sqlite3.Row) -> Book:
        """Replay the account's fills into a Book. The database stores facts; the Book does arithmetic."""
        book = Book(D(account["starting_cash"]), keep_entries=False)
        rows = db.execute("SELECT * FROM orders WHERE account_id = ? AND status = 'filled' ORDER BY at, rowid", (account["id"],))
        for row in rows:
            if row["side"] == "buy":
                book.buy(row["symbol"], row["quantity"], row["fill_price"], fee=row["fee"], at=row["at"], reference=row["id"])
            else:
                book.sell(row["symbol"], row["quantity"], row["fill_price"], fee=row["fee"], at=row["at"], reference=row["id"])
        return book

    # ------------------------------------------------------------------ quotes and marks
    def _fresh_quote(self, symbol: str) -> dict[str, Any]:
        envelope = self.market_data.quote(symbol)
        data = envelope.data
        age = data.get("age_seconds")
        state = data.get("market_state")
        quote = {
            "symbol": envelope.symbol, "price": data.get("price"), "age_seconds": age, "freshness": envelope.freshness,
            "provider": envelope.provider, "market_state": state, "retrieved_at": iso(envelope.retrieved_at),
        }
        if quote["price"] is None or quote["price"] <= 0:
            raise OrderRejected("No usable price is available, so the order is rejected.", code="no_price", details=quote)
        if state != "open" and not self.config.allow_when_market_closed:
            raise OrderRejected("The market is closed and this account only simulates fills during market hours.", code="market_closed", details=quote)
        if age is None:
            raise OrderRejected("The provider did not say how old this price is, so it is not traded on.", code="unknown_quote_age", details=quote)
        if age > self.config.max_quote_age_seconds:
            raise OrderRejected(
                f"The newest price is {age:.0f} seconds old (limit {self.config.max_quote_age_seconds:.0f}). Stale data is not traded on.",
                code="stale_quote", details=quote,
            )
        return quote

    def _marks(self, book: Book) -> tuple[dict[str, Decimal], list[str]]:
        prices: dict[str, Decimal] = {}
        missing: list[str] = []
        for symbol in book.positions:
            try:
                data = self.market_data.quote(symbol).data
                prices[symbol] = D(data["price"]) if data.get("price") else None
            except MarketVisionError:
                prices[symbol] = None
            if prices[symbol] is None:
                missing.append(symbol)
                prices[symbol] = book.positions[symbol].average_cost  # a labelled fallback for display only
        return prices, missing

    # ------------------------------------------------------------------ public reads
    def account(self) -> dict[str, Any]:
        with self._lock, self._read() as db:
            row = self._active_account(db)
            book = self._book(db, row)
            prices, missing = self._marks(book)
            snapshot = book.snapshot(prices)
            orders = db.execute("SELECT COUNT(*) AS n FROM orders WHERE account_id = ?", (row["id"],)).fetchone()["n"]
            return {
                "simulation": True, "banner": SIMULATION_BANNER,
                "account_id": row["id"], "created_at": row["created_at"], "paused": bool(row["paused"]),
                "config": json.loads(row["config"]), "orders_submitted": orders,
                **snapshot,
                "marks_missing_live_price": missing,
                "mark_note": "Positions in marks_missing_live_price are valued at average cost because no live price was available." if missing else None,
            }

    def history(self, *, limit: int = 100, include_rejected: bool = True) -> list[dict[str, Any]]:
        with self._lock, self._read() as db:
            row = self._active_account(db)
            query = "SELECT * FROM orders WHERE account_id = ?" + ("" if include_rejected else " AND status = 'filled'") + " ORDER BY at DESC, rowid DESC LIMIT ?"
            out = []
            for order in db.execute(query, (row["id"], int(limit))):
                out.append({
                    "order_id": order["id"], "client_order_id": order["client_order_id"], "at": order["at"], "side": order["side"], "symbol": order["symbol"],
                    "status": order["status"], "reason": order["reason"], "quantity": order["quantity"], "fill_price": order["fill_price"],
                    "gross": order["gross"], "fee": order["fee"], "cash_after": order["cash_after"], "realized": order["realized"],
                    "request": json.loads(order["request"]), "quote": json.loads(order["quote"]) if order["quote"] else None,
                })
            return out

    # ------------------------------------------------------------------ orders
    def submit_order(
        self,
        *,
        side: str,
        symbol: str,
        quantity: Any | None = None,
        notional: Any | None = None,
        client_order_id: str | None = None,
        simulation: bool = False,
    ) -> dict[str, Any]:
        """Simulate a market order. ``simulation=True`` is required as an explicit acknowledgement."""
        from app.marketdata.service import clean_symbol

        if simulation is not True:
            raise ValidationFailed("Paper orders require simulation=true, as an explicit acknowledgement that this is virtual money.")
        side = str(side).lower().strip()
        if side not in {"buy", "sell"}:
            raise ValidationFailed("side must be 'buy' or 'sell'.")
        symbol = clean_symbol(symbol)
        if (quantity is None) == (notional is None):
            raise ValidationFailed("Give exactly one of quantity (shares) or notional (dollars).")
        request = {"side": side, "symbol": symbol, "quantity": None if quantity is None else str(D(quantity).normalize()), "notional": None if notional is None else money_str(D(notional))}
        client_order_id = (client_order_id or "").strip() or f"auto-{uuid.uuid4().hex[:16]}"

        with self._lock, self._write() as db:
            account = self._active_account(db)
            existing = db.execute("SELECT * FROM orders WHERE account_id = ? AND client_order_id = ?", (account["id"], client_order_id)).fetchone()
            if existing is not None:
                if json.loads(existing["request"]) != request:
                    raise Conflict(f"client_order_id {client_order_id!r} was already used for a different order.", code="client_order_id_reused")
                return {**self._order_dict(existing), "duplicate": True}
            order_id = f"order-{uuid.uuid4().hex[:16]}"
            at = iso(utcnow())

            def reject(exc: OrderRejected, quote: dict | None = None) -> dict[str, Any]:
                # A rejection is recorded but does not consume the caller's client_order_id:
                # retrying the same id after the market opens (or a cap clears) must be possible.
                db.execute(
                    "INSERT INTO orders (id, account_id, client_order_id, at, side, symbol, request, status, reason, quote) VALUES (?, ?, ?, ?, ?, ?, ?, 'rejected', ?, ?)",
                    (order_id, account["id"], f"rejected-{order_id}", at, side, symbol, json.dumps({**request, "client_order_id": client_order_id}), f"{exc.code}: {exc.message}",
                     json.dumps(quote, default=str) if quote else None),
                )
                self._event(db, account["id"], "order_rejected", {"order_id": order_id, "code": exc.code, "message": exc.message, "request": request})
                db.execute("COMMIT")  # keep the rejection record; the context manager's ROLLBACK must not erase it
                db.execute("BEGIN IMMEDIATE")
                raise exc

            if bool(account["paused"]):
                reject(OrderRejected("Trading on this account is paused.", code="trading_paused"))
            book = self._book(db, account)
            try:
                quote = self._fresh_quote(symbol)
            except OrderRejected as exc:
                reject(exc)

            reference = D(quote["price"])
            adverse = (D(self.config.half_spread_bps) + D(self.config.slippage_bps)) * BPS
            fill = price4(reference * (ONE + adverse) if side == "buy" else reference * (ONE - adverse))
            if quantity is not None:
                amount = shares(D(quantity))
            else:
                budget = cents(D(notional))
                amount = shares(budget / (fill * (ONE + D(self.config.commission_rate))))
                rate = D(self.config.commission_rate)
                for _ in range(8):  # rounding up the cost and the fee can leave the total over the stated notional
                    value = cents_up(amount * fill)
                    fee_estimate = cents_up(value * rate) if rate else ZERO
                    if amount <= 0 or value + fee_estimate <= budget:
                        break
                    smaller = shares((budget - fee_estimate) / fill)
                    amount = smaller if smaller < amount else shares(amount - Decimal("0.000001"))
            if amount <= 0:
                reject(OrderRejected("The order quantity rounds to zero shares.", code="invalid_quantity"), quote)
            value = cents_up(amount * fill) if side == "buy" else cents_down(amount * fill)
            if value < D(self.config.min_order_value):
                reject(OrderRejected(f"Orders below {money_str(D(self.config.min_order_value))} are not accepted.", code="order_too_small"), quote)
            fee = cents_up(value * D(self.config.commission_rate)) if self.config.commission_rate else ZERO

            if side == "buy":
                prices, missing_marks = self._marks(book)
                prices[symbol] = reference
                if missing_marks:
                    reject(OrderRejected(
                        f"No live price for {', '.join(missing_marks)}, so the position and exposure caps cannot be computed. The buy is rejected, not guessed.",
                        code="marks_unavailable", details={"missing": missing_marks},
                    ), quote)
                equity_now = book.cash + book.market_value(prices)
                position_after = cents_down((book.quantity(symbol) + amount) * reference)
                invested_after = book.market_value(prices) + cents_up(amount * reference)
                if equity_now > 0 and position_after / equity_now > D(self.config.max_position_share):
                    reject(OrderRejected(
                        f"After this order {symbol} would be {float(position_after / equity_now) * 100:.1f}% of the account; the limit is {float(D(self.config.max_position_share)) * 100:.0f}%.",
                        code="position_limit", details={"limit_pct": float(D(self.config.max_position_share)) * 100},
                    ), quote)
                if equity_now > 0 and invested_after / equity_now > D(self.config.max_invested_share):
                    reject(OrderRejected(
                        f"After this order {float(invested_after / equity_now) * 100:.1f}% of the account would be invested; the limit is {float(D(self.config.max_invested_share)) * 100:.0f}%.",
                        code="exposure_limit", details={"limit_pct": float(D(self.config.max_invested_share)) * 100},
                    ), quote)
            try:
                entry = (book.buy if side == "buy" else book.sell)(symbol, amount, fill, fee=fee, at=at, reference=order_id)
            except OrderRejected as exc:
                reject(exc, quote)
            db.execute(
                "INSERT INTO orders (id, account_id, client_order_id, at, side, symbol, request, status, quantity, fill_price, gross, fee, cash_after, realized, quote) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'filled', ?, ?, ?, ?, ?, ?, ?)",
                (order_id, account["id"], client_order_id, at, side, symbol, json.dumps(request), str(entry.quantity), str(entry.price), money_str(entry.gross),
                 money_str(entry.fee), money_str(entry.cash_after), money_str(entry.realized), json.dumps(quote, default=str)),
            )
            self._event(db, account["id"], "order_filled", {"order_id": order_id, "side": side, "symbol": symbol, "quantity": str(entry.quantity), "fill_price": str(entry.price)})
            row = db.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
            return self._order_dict(row)

    @staticmethod
    def _order_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "simulation": True, "banner": SIMULATION_BANNER,
            "order_id": row["id"], "client_order_id": row["client_order_id"], "at": row["at"], "side": row["side"], "symbol": row["symbol"],
            "status": row["status"], "reason": row["reason"], "quantity": row["quantity"], "fill_price": row["fill_price"], "gross": row["gross"],
            "fee": row["fee"], "cash_after": row["cash_after"], "realized": row["realized"], "quote": json.loads(row["quote"]) if row["quote"] else None,
        }

    # ------------------------------------------------------------------ controls
    def set_paused(self, paused: bool, *, reason: str = "") -> dict[str, Any]:
        with self._lock, self._write() as db:
            account = self._active_account(db)
            db.execute("UPDATE accounts SET paused = ? WHERE id = ?", (int(bool(paused)), account["id"]))
            self._event(db, account["id"], "trading_paused" if paused else "trading_resumed", {"reason": reason})
        return {"paused": bool(paused), "reason": reason}

    def reset(self, *, reason: str = "") -> dict[str, Any]:
        """Archive the active account (history kept) and open a fresh one with the starting cash."""
        with self._lock, self._write() as db:
            account = self._active_account(db)
            db.execute("UPDATE accounts SET status = 'archived', archived_at = ? WHERE id = ?", (iso(utcnow()), account["id"]))
            self._event(db, account["id"], "account_archived", {"reason": reason})
            fresh = self._active_account(db)
        return {"archived_account_id": account["id"], "account_id": fresh["id"], "starting_cash": money_str(D(self.config.starting_cash)), "banner": SIMULATION_BANNER}

    def performance(self) -> dict[str, Any]:
        """Equity against starting cash, fees paid, realised and unrealised profit, win rate on closed sales."""
        with self._lock, self._read() as db:
            account = self._active_account(db)
            book = self._book(db, account)
            prices, missing = self._marks(book)
            snapshot = book.snapshot(prices)
            sells = db.execute("SELECT realized FROM orders WHERE account_id = ? AND status = 'filled' AND side = 'sell'", (account["id"],)).fetchall()
            wins = sum(1 for row in sells if D(row["realized"]) > 0)
            return {
                "simulation": True, "banner": SIMULATION_BANNER, "account_id": account["id"],
                "starting_cash": snapshot["starting_cash"], "equity": snapshot["equity"],
                "total_pnl": snapshot["total_pnl"], "realized_pnl": snapshot["realized_pnl"], "unrealized_pnl": snapshot["unrealized_pnl"],
                "fees_paid": snapshot["fees_paid"], "dividends": snapshot["dividends"],
                "return_pct": None if snapshot["equity"] is None else round((float(snapshot["equity"]) / float(snapshot["starting_cash"]) - 1) * 100, 4),
                "closed_sales": len(sells), "winning_sales": wins,
                "win_rate_pct": round(wins / len(sells) * 100, 2) if sells else None,
                "marks_missing_live_price": missing,
            }


_default: PaperAccountService | None = None
_default_lock = threading.Lock()


def get_paper_service() -> PaperAccountService:
    global _default
    with _default_lock:
        if _default is None:
            from app.marketdata.service import get_market_data

            _default = PaperAccountService(get_market_data())
        return _default


def set_paper_service(service: PaperAccountService | None) -> None:
    global _default
    with _default_lock:
        _default = service
