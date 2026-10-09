"""Exact arithmetic for money and share quantities.

Money is a ``Decimal`` rounded to cents; share quantities are ``Decimal`` to six places, so
fractional shares are exact. Binary floating point is never used for a balance: adding ``0.1``
ten times in floats does not give ``1.0``, and an account that drifts by fractions of a cent
cannot be reconciled.

Rounding rules, applied in one place:
    * a purchase's cost and every fee are rounded **up** to the cent (the account never pays less
      than it owes);
    * sale proceeds and dividends are rounded **down** to the cent;
    * a share quantity is rounded **down** to six places (never buy or sell more than asked).
"""
from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Any

from app.core.errors import ValidationFailed

CENT = Decimal("0.01")
SHARE_STEP = Decimal("0.000001")
ZERO = Decimal("0")
ONE = Decimal("1")
BPS = Decimal("0.0001")


def D(value: Any) -> Decimal:  # noqa: N802 - short on purpose, used everywhere
    """Convert to ``Decimal`` without inheriting a float's binary error. Rejects NaN and infinity."""
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, bool):
        raise ValidationFailed("A number is required, not a true/false value.")
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        number = Decimal(repr(value))
    else:
        try:
            number = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            raise ValidationFailed(f"{value!r} is not a number.") from None
    if not number.is_finite():
        raise ValidationFailed("The number must be finite.")
    return number


def cents_up(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_CEILING)


def cents_down(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_FLOOR)


def cents(value: Decimal) -> Decimal:
    """Round to the nearest cent (banker's rounding). For reporting, not for moving money."""
    return value.quantize(CENT, rounding=ROUND_HALF_EVEN)


def shares(value: Decimal) -> Decimal:
    return value.quantize(SHARE_STEP, rounding=ROUND_DOWN)


def price4(value: Decimal) -> Decimal:
    """Prices are kept to four decimal places (sub-penny quotes exist)."""
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)


def ratio(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    if denominator == 0:
        return None
    with localcontext() as ctx:
        ctx.prec = 28
        return numerator / denominator


def to_float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def money_str(value: Decimal) -> str:
    """Canonical text form of an amount, always with two decimals."""
    return f"{cents(value):.2f}"
