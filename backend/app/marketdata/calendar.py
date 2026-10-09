"""US equity trading calendar (NYSE rules), computed from dates with no extra dependency.

Used to decide which future sessions a forecast covers, whether a bar is missing, whether data
is stale, and whether the market is open for a simulated order.

Limits, stated plainly: holiday rules are those in force since 1998 (Juneteenth from 2022).
Unscheduled closures are listed by hand up to the date this file was written; a closure
announced later is not known until it is added to ``UNSCHEDULED_CLOSURES``. Early closes follow
the usual pattern (13:00 New York time on the day after Thanksgiving, on Christmas Eve, and on
3 July when 4 July is a weekday) and can differ in a given year.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

UNSCHEDULED_CLOSURES: dict[date, str] = {
    date(2001, 9, 11): "September 11 attacks",
    date(2001, 9, 12): "September 11 attacks",
    date(2001, 9, 13): "September 11 attacks",
    date(2001, 9, 14): "September 11 attacks",
    date(2004, 6, 11): "Day of mourning: Ronald Reagan",
    date(2007, 1, 2): "Day of mourning: Gerald Ford",
    date(2012, 10, 29): "Hurricane Sandy",
    date(2012, 10, 30): "Hurricane Sandy",
    date(2018, 12, 5): "Day of mourning: George H. W. Bush",
    date(2025, 1, 9): "Day of mourning: Jimmy Carter",
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date:
    """Saturday holidays are observed on Friday, Sunday holidays on Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=256)
def holidays(year: int) -> dict[date, str]:
    """Full-day market holidays in ``year``."""
    out: dict[date, str] = {}
    new_year = date(year, 1, 1)
    # NYSE rule: New Year's Day on a Saturday is NOT moved to the Friday before.
    if new_year.weekday() == 6:
        out[new_year + timedelta(days=1)] = "New Year's Day (observed)"
    elif new_year.weekday() != 5:
        out[new_year] = "New Year's Day"
    if year >= 1998:
        out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    out[_easter(year) - timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        out[_observed(date(year, 6, 19))] = "Juneteenth"
    out[_observed(date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    out[_observed(date(year, 12, 25))] = "Christmas Day"
    for closed, reason in UNSCHEDULED_CLOSURES.items():
        if closed.year == year:
            out[closed] = reason
    return out


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def is_trading_day(day: date | datetime | str) -> bool:
    d = _as_date(day)
    return d.weekday() < 5 and d not in holidays(d.year)


def holiday_name(day: date | datetime | str) -> str | None:
    d = _as_date(day)
    return holidays(d.year).get(d)


def next_trading_day(day: date | datetime | str) -> date:
    d = _as_date(day) + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def previous_trading_day(day: date | datetime | str) -> date:
    d = _as_date(day) - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def shift_trading_days(day: date | datetime | str, n: int) -> date:
    """The session ``n`` trading days after ``day`` (before, if ``n`` is negative).

    ``day`` itself need not be a session. ``n = 0`` returns ``day`` if it is a session,
    otherwise the next session.
    """
    d = _as_date(day)
    if n == 0:
        return d if is_trading_day(d) else next_trading_day(d)
    step = next_trading_day if n > 0 else previous_trading_day
    for _ in range(abs(n)):
        d = step(d)
    return d


def trading_days(start: date | datetime | str, end: date | datetime | str) -> list[date]:
    """Every session from ``start`` to ``end``, both included."""
    d, last = _as_date(start), _as_date(end)
    out: list[date] = []
    while d <= last:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def is_early_close(day: date | datetime | str) -> bool:
    d = _as_date(day)
    if not is_trading_day(d):
        return False
    if d == _nth_weekday(d.year, 11, 3, 4) + timedelta(days=1):
        return True
    if d.month == 12 and d.day == 24:
        return True
    return d.month == 7 and d.day == 3 and date(d.year, 7, 4).weekday() < 5


def session_bounds(day: date | datetime | str) -> tuple[datetime, datetime] | None:
    """Open and close of the regular session as UTC datetimes, or ``None`` when closed all day."""
    d = _as_date(day)
    if not is_trading_day(d):
        return None
    close = EARLY_CLOSE if is_early_close(d) else REGULAR_CLOSE
    opened = datetime.combine(d, REGULAR_OPEN, tzinfo=NEW_YORK).astimezone(timezone.utc)
    closed = datetime.combine(d, close, tzinfo=NEW_YORK).astimezone(timezone.utc)
    return opened, closed


def market_state(now: datetime) -> dict:
    """Whether the regular session is open at ``now`` and when the neighbouring sessions are."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_day = now.astimezone(NEW_YORK).date()
    bounds = session_bounds(local_day)
    if bounds and bounds[0] <= now < bounds[1]:
        state = "open"
    elif bounds and now < bounds[0]:
        state = "pre_open"
    elif bounds:
        state = "after_close"
    else:
        state = "closed"
    last_session = local_day if bounds and now >= bounds[1] else previous_trading_day(local_day)
    if state == "open":
        last_completed = previous_trading_day(local_day)
        next_open_day = local_day
    else:
        last_completed = last_session
        next_open_day = local_day if state == "pre_open" else next_trading_day(local_day)
    next_bounds = session_bounds(next_open_day)
    return {
        "state": state,
        "is_open": state == "open",
        "session_date": local_day.isoformat() if bounds else None,
        "last_completed_session": last_completed.isoformat(),
        "next_open": next_bounds[0].isoformat().replace("+00:00", "Z") if next_bounds else None,
        "early_close": bool(bounds) and is_early_close(local_day),
        "holiday": holiday_name(local_day),
        "calendar": "NYSE rules (computed)",
    }


def last_completed_session(now: datetime) -> date:
    return date.fromisoformat(market_state(now)["last_completed_session"])
