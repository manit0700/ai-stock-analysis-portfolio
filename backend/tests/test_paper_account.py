"""The $50 virtual account: every rule that guards it, and the arithmetic behind it.

All simulated. The fake quote source stands in for the market-data service; no network, no broker.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.core import clock
from app.core.errors import Conflict, MarketVisionError, OrderRejected, ValidationFailed
from app.paper.account import SIMULATION_BANNER, PaperAccountService, PaperConfig

OPEN_CLOCK = lambda: datetime(2026, 3, 10, 15, 30, tzinfo=timezone.utc)  # a Tuesday, 10:30 New York  # noqa: E731


class FakeQuotes:
    def __init__(self, prices: dict[str, float], *, age: float = 5.0, state: str = "open"):
        self.prices = dict(prices)
        self.age = age
        self.state = state
        self.calls = 0

    def quote(self, symbol: str):
        self.calls += 1
        symbol = symbol.upper()
        if symbol not in self.prices:
            raise MarketVisionError(f"no quote for {symbol}", code="data_unavailable")

        class Envelope:
            pass

        envelope = Envelope()
        envelope.symbol = symbol
        envelope.provider = "fake"
        envelope.freshness = "live"
        envelope.retrieved_at = clock.utcnow()
        envelope.data = {"price": self.prices[symbol], "age_seconds": self.age, "market_state": self.state}
        return envelope


ZERO_COST = dict(half_spread_bps=Decimal("0"), slippage_bps=Decimal("0"))


@pytest.fixture()
def service(tmp_path):
    clock.set_clock(OPEN_CLOCK)
    quotes = FakeQuotes({"AAPL": 180.0, "F": 11.0, "KO": 60.0})
    svc = PaperAccountService(quotes, db_path=tmp_path / "paper.sqlite3", config=PaperConfig(**ZERO_COST))
    return svc, quotes


def test_account_starts_with_fifty_virtual_dollars_and_says_simulated(service):
    svc, _ = service
    account = svc.account()
    assert account["cash"] == "50.00" and account["starting_cash"] == "50.00" and account["equity"] == "50.00"
    assert account["simulation"] is True and account["banner"] == SIMULATION_BANNER
    assert account["positions"] == [] and account["paused"] is False


def test_simulation_flag_is_required(service):
    svc, _ = service
    with pytest.raises(ValidationFailed):
        svc.submit_order(side="buy", symbol="F", notional="10")
    assert svc.history() == []


def test_buy_then_sell_updates_cash_positions_and_realised_pnl(service):
    svc, quotes = service
    order = svc.submit_order(side="buy", symbol="F", quantity="1.5", simulation=True)
    assert order["status"] == "filled" and order["fill_price"] == "11.0000" and order["gross"] == "16.50"
    account = svc.account()
    assert account["cash"] == "33.50" and account["positions"][0]["quantity"] == "1.500000"
    quotes.prices["F"] = 12.0
    account = svc.account()
    assert account["equity"] == "51.50" and account["unrealized_pnl"] == "1.50" and account["total_pnl"] == "1.50"
    sale = svc.submit_order(side="sell", symbol="F", quantity="1.5", simulation=True)
    assert sale["realized"] == "1.50" and svc.account()["cash"] == "51.50"
    perf = svc.performance()
    assert perf["total_pnl"] == "1.50" and perf["closed_sales"] == 1 and perf["win_rate_pct"] == 100.0 and perf["return_pct"] == 3.0


def test_fractional_notional_order(service):
    svc, _ = service
    order = svc.submit_order(side="buy", symbol="AAPL", notional="18.00", simulation=True)
    assert order["quantity"] == "0.100000" and order["gross"] == "18.00"


def test_cash_can_never_go_negative(service):
    svc, _ = service
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="buy", symbol="KO", quantity="1", simulation=True)  # $60 > $50, and over every limit
    assert refused.value.code in {"insufficient_cash", "position_limit", "exposure_limit"}
    assert svc.account()["cash"] == "50.00"


def test_no_short_selling(service):
    svc, _ = service
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="sell", symbol="F", quantity="1", simulation=True)
    assert refused.value.code == "insufficient_shares"


def test_position_and_exposure_limits(service):
    svc, _ = service
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="buy", symbol="F", notional="25.00", simulation=True)  # 50% of equity > 40% cap
    assert refused.value.code == "position_limit"
    svc.submit_order(side="buy", symbol="F", notional="19.00", simulation=True)
    svc.submit_order(side="buy", symbol="AAPL", notional="19.00", simulation=True)
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="buy", symbol="KO", notional="11.00", simulation=True)  # would take investment over 95%
    assert refused.value.code == "exposure_limit"


def test_stale_quotes_and_closed_market_are_rejected(tmp_path):
    clock.set_clock(OPEN_CLOCK)
    stale = FakeQuotes({"F": 11.0}, age=2000.0)
    svc = PaperAccountService(stale, db_path=tmp_path / "a.sqlite3", config=PaperConfig(**ZERO_COST))
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    assert refused.value.code == "stale_quote"
    closed = FakeQuotes({"F": 11.0}, state="after_close")
    svc2 = PaperAccountService(closed, db_path=tmp_path / "b.sqlite3", config=PaperConfig(**ZERO_COST))
    with pytest.raises(OrderRejected) as refused:
        svc2.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    assert refused.value.code == "market_closed"
    relaxed = PaperAccountService(closed, db_path=tmp_path / "c.sqlite3", config=PaperConfig(allow_when_market_closed=True, **ZERO_COST))
    assert relaxed.submit_order(side="buy", symbol="F", quantity="1", simulation=True)["status"] == "filled"
    # every rejection is in the history with its reason
    rejected = [row for row in svc.history() if row["status"] == "rejected"]
    assert rejected and rejected[0]["reason"].startswith("stale_quote")


def test_execution_costs_move_the_fill_price_against_the_trader(tmp_path):
    clock.set_clock(OPEN_CLOCK)
    quotes = FakeQuotes({"F": 10.0})
    svc = PaperAccountService(quotes, db_path=tmp_path / "paper.sqlite3", config=PaperConfig(half_spread_bps=Decimal("10"), slippage_bps=Decimal("10")))
    buy = svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    assert buy["fill_price"] == "10.0200"
    sale = svc.submit_order(side="sell", symbol="F", quantity="1", simulation=True)
    assert sale["fill_price"] == "9.9800" and Decimal(sale["realized"]) == Decimal("-0.04")  # bought at 10.02, sold at 9.98


def test_client_order_id_makes_submission_idempotent(service):
    svc, _ = service
    first = svc.submit_order(side="buy", symbol="F", quantity="1", client_order_id="abc", simulation=True)
    again = svc.submit_order(side="buy", symbol="F", quantity="1", client_order_id="abc", simulation=True)
    assert again["duplicate"] is True and again["order_id"] == first["order_id"]
    assert svc.account()["positions"][0]["quantity"] == "1.000000"  # filled once
    with pytest.raises(Conflict):
        svc.submit_order(side="buy", symbol="F", quantity="2", client_order_id="abc", simulation=True)


def test_pause_blocks_orders_until_resumed(service):
    svc, _ = service
    svc.set_paused(True, reason="drill")
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    assert refused.value.code == "trading_paused" and svc.account()["paused"] is True
    svc.set_paused(False)
    assert svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)["status"] == "filled"


def test_reset_archives_history_and_starts_fresh(service):
    svc, _ = service
    svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    old_id = svc.account()["account_id"]
    result = svc.reset(reason="new experiment")
    assert result["archived_account_id"] == old_id and result["starting_cash"] == "50.00"
    fresh = svc.account()
    assert fresh["account_id"] != old_id and fresh["cash"] == "50.00" and fresh["positions"] == []
    assert fresh["orders_submitted"] == 0


def test_state_survives_a_restart(tmp_path):
    clock.set_clock(OPEN_CLOCK)
    quotes = FakeQuotes({"F": 11.0})
    config = PaperConfig(max_position_share=Decimal("0.9"), **ZERO_COST)
    first = PaperAccountService(quotes, db_path=tmp_path / "p.sqlite3", config=config)
    first.submit_order(side="buy", symbol="F", quantity="2", simulation=True)
    second = PaperAccountService(quotes, db_path=tmp_path / "p.sqlite3", config=config)
    account = second.account()
    assert account["positions"][0]["quantity"] == "2.000000" and account["cash"] == "28.00"
    assert len(second.history(include_rejected=False)) == 1


def test_a_position_with_no_live_price_is_marked_at_cost_and_labelled(service):
    svc, quotes = service
    svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    del quotes.prices["F"]
    account = svc.account()
    assert account["marks_missing_live_price"] == ["F"] and "average cost" in account["mark_note"]
    assert account["equity"] == "50.00"


def test_concurrent_identical_orders_fill_once(service):
    svc, _ = service
    outcomes = []

    def submit():
        try:
            outcomes.append(svc.submit_order(side="buy", symbol="F", quantity="1", client_order_id="race", simulation=True))
        except Exception as exc:  # noqa: BLE001
            outcomes.append(exc)
    threads = [threading.Thread(target=submit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fills = [o for o in outcomes if isinstance(o, dict)]
    assert len(fills) == 6 and len({o["order_id"] for o in fills}) == 1
    assert svc.account()["positions"][0]["quantity"] == "1.000000"


# ------------------------------------------------------------------ fixes from the independent review
def test_a_quote_with_unknown_age_is_not_traded_on(tmp_path):
    """Review finding 7: an ageless quote used to bypass the staleness gate."""
    clock.set_clock(OPEN_CLOCK)
    ageless = FakeQuotes({"F": 11.0}, age=None)  # type: ignore[arg-type]
    svc = PaperAccountService(ageless, db_path=tmp_path / "a.sqlite3", config=PaperConfig(**ZERO_COST))
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    assert refused.value.code == "unknown_quote_age"


def test_a_buy_is_rejected_when_an_existing_position_has_no_live_mark(service):
    """Review finding 8: the caps used to be computed against average-cost fallbacks."""
    svc, quotes = service
    svc.submit_order(side="buy", symbol="F", quantity="1", simulation=True)
    del quotes.prices["F"]
    with pytest.raises(OrderRejected) as refused:
        svc.submit_order(side="buy", symbol="AAPL", notional="5", simulation=True)
    assert refused.value.code == "marks_unavailable"
    quotes.prices["F"] = 11.0
    assert svc.submit_order(side="buy", symbol="AAPL", notional="5", simulation=True)["status"] == "filled"


def test_a_notional_order_never_costs_more_than_the_notional(tmp_path):
    """Review finding 4: rounding the cost and the fee up could exceed the stated dollar amount."""
    clock.set_clock(OPEN_CLOCK)
    quotes = FakeQuotes({"F": 9.9010})
    svc = PaperAccountService(quotes, db_path=tmp_path / "n.sqlite3", config=PaperConfig(commission_rate=Decimal("0.01"), **ZERO_COST))
    order = svc.submit_order(side="buy", symbol="F", notional="10.00", simulation=True)
    assert Decimal(order["gross"]) + Decimal(order["fee"]) <= Decimal("10.00")
    # and an all-in order for exactly the cash on hand fills rather than bouncing
    all_in = PaperAccountService(FakeQuotes({"F": 9.9010}), db_path=tmp_path / "n2.sqlite3", config=PaperConfig(max_position_share=Decimal("1"), max_invested_share=Decimal("1"), **ZERO_COST))
    filled = all_in.submit_order(side="buy", symbol="F", notional="50.00", simulation=True)
    assert filled["status"] == "filled" and Decimal(filled["gross"]) <= Decimal("50.00")


def test_a_rejected_order_does_not_consume_the_client_order_id(service):
    svc, quotes = service
    quotes.state = "after_close"
    with pytest.raises(OrderRejected):
        svc.submit_order(side="buy", symbol="F", quantity="1", client_order_id="retry-me", simulation=True)
    quotes.state = "open"
    order = svc.submit_order(side="buy", symbol="F", quantity="1", client_order_id="retry-me", simulation=True)
    assert order["status"] == "filled" and "duplicate" not in order
    # equivalent spellings of the same quantity are the same request
    again = svc.submit_order(side="buy", symbol="F", quantity="1.0", client_order_id="retry-me", simulation=True)
    assert again["duplicate"] is True


def test_two_processes_cannot_spend_the_same_cash(tmp_path):
    """Review finding 3: BEGIN IMMEDIATE serialises read-check-write across processes."""
    import multiprocessing as mp

    db = tmp_path / "race.sqlite3"

    def worker(name: str, out):
        from datetime import datetime, timezone

        from app.core import clock as c
        c.set_clock(lambda: datetime(2026, 3, 10, 15, 30, tzinfo=timezone.utc))
        quotes = FakeQuotes({"F": 45.0})
        svc = PaperAccountService(quotes, db_path=db, config=PaperConfig(max_position_share=Decimal("1"), max_invested_share=Decimal("1"), **ZERO_COST))
        try:
            order = svc.submit_order(side="buy", symbol="F", quantity="1", client_order_id=name, simulation=True)
            out.put((name, order["status"]))
        except OrderRejected as exc:
            out.put((name, exc.code))

    # create the account first so both workers race on the same one
    clock.set_clock(OPEN_CLOCK)
    PaperAccountService(FakeQuotes({"F": 45.0}), db_path=db, config=PaperConfig(**ZERO_COST)).account()
    for _ in range(10):
        queue = mp.Queue()
        a = mp.Process(target=worker, args=("a", queue))
        b = mp.Process(target=worker, args=("b", queue))
        a.start(); b.start(); a.join(20); b.join(20)
        outcomes = sorted(queue.get() for _ in range(2))
        statuses = [status for _, status in outcomes]
        assert statuses.count("filled") <= 1, statuses  # never two $45 fills from $50
        service = PaperAccountService(FakeQuotes({"F": 45.0}), db_path=db, config=PaperConfig(max_position_share=Decimal("1"), max_invested_share=Decimal("1"), **ZERO_COST))
        account = service.account()  # and the account stays readable
        if statuses.count("filled") == 1:
            break
    if account["positions"]:
        service.submit_order(side="sell", symbol="F", quantity=account["positions"][0]["quantity"], simulation=True)
