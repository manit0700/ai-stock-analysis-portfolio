"""Loading the market context a feature row needs, from the one market-data door.

Training loads context from a data snapshot; live prediction loads it from the providers. Both go
through ``MarketData``, so both get the same normalisation and the same quality checks.

A part that cannot be loaded is left absent and the reason recorded. The pipeline then leaves the
dependent features missing, and a model that needs them declines to predict.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from app.core.errors import MarketVisionError
from app.features.pipeline import MarketContext
from app.features.schema import SECTOR_ETFS

SPY, QQQ, VIX = "SPY", "QQQ", "^VIX"
US10Y_SERIES, YIELD_CURVE_SERIES = "DGS10", "T10Y2Y"

#: Sector ETF for symbols whose company profile is not looked up (training on a snapshot, or
#: a provider outage). ETFs and indices have no sector and are absent on purpose.
KNOWN_SECTORS: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK", "INTC": "XLK", "CSCO": "XLK",
    "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK", "IBM": "XLK", "QCOM": "XLK", "MU": "XLK",
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "VZ": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "AXP": "XLF", "V": "XLF", "MA": "XLF", "AIG": "XLF", "TRV": "XLF",
    "XOM": "XLE", "CVX": "XLE",
    "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "MRK": "XLV", "LLY": "XLV", "ABBV": "XLV",
    "BA": "XLI", "CAT": "XLI", "GE": "XLI", "MMM": "XLI", "UTX": "XLI", "WM": "XLI",
    "WMT": "XLP", "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP",
}


def sector_etf_for(symbol: str, market_data: Any | None = None) -> tuple[str | None, str]:
    """The sector ETF for ``symbol`` and how it was found. ``None`` when it is not known."""
    symbol = symbol.upper()
    if symbol in KNOWN_SECTORS:
        return KNOWN_SECTORS[symbol], "built-in table"
    if market_data is not None:
        try:
            sector = (market_data.profile(symbol).data or {}).get("sector")
        except MarketVisionError:
            sector = None
        if sector and sector in SECTOR_ETFS:
            return SECTOR_ETFS[sector], "company profile"
    return None, "sector unknown"


def load_context(
    market_data: Any,
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    period: str | None = None,
    sector_symbol: str | None = None,
    include_macro: bool = True,
    shared: dict[str, Any] | None = None,
) -> MarketContext:
    """Fetch SPY, QQQ, VIX, the sector ETF and Treasury yields for a feature computation.

    ``shared`` is a dictionary the caller may pass to several calls so that SPY, QQQ, VIX and the
    yields are fetched once for a whole list of tickers.
    """
    shared = shared if shared is not None else {}
    missing: dict[str, str] = {}

    def bars(name: str) -> pd.DataFrame | None:
        key = f"bars:{name}"
        if key not in shared:
            try:
                shared[key] = market_data.history(name, start=start, end=end, period=period, interval="1d").data
            except MarketVisionError as exc:
                shared[key] = exc
        value = shared[key]
        return None if isinstance(value, Exception) else value

    def reason(name: str) -> str:
        value = shared.get(f"bars:{name}")
        return value.message if isinstance(value, MarketVisionError) else "not available"

    def series(name: str) -> pd.Series | None:
        key = f"series:{name}"
        if key not in shared:
            try:
                shared[key] = market_data.economic_series(name, start=start, end=end).data
            except MarketVisionError as exc:
                shared[key] = exc
        value = shared[key]
        return None if isinstance(value, Exception) else value

    spy = bars(SPY)
    if spy is None:
        missing["market"] = f"SPY history: {reason(SPY)}"
    qqq = bars(QQQ)
    if qqq is None:
        missing["qqq"] = f"QQQ history: {reason(QQQ)}"
    vix_bars = bars(VIX)
    vix = vix_bars["Close"] if vix_bars is not None else None
    if vix is None:
        missing["vix"] = f"VIX history: {reason(VIX)}"

    how = "given"
    if sector_symbol is None:
        sector_symbol, how = sector_etf_for(symbol, market_data)
    sector = bars(sector_symbol) if sector_symbol else None
    if sector is None:
        missing["sector"] = f"{sector_symbol} history: {reason(sector_symbol)}" if sector_symbol else f"no sector ETF for {symbol.upper()} ({how})"

    us10y = curve = None
    if include_macro:
        us10y, curve = series(US10Y_SERIES), series(YIELD_CURVE_SERIES)
        if us10y is None or curve is None:
            value = shared.get(f"series:{US10Y_SERIES if us10y is None else YIELD_CURVE_SERIES}")
            missing["macro"] = f"Treasury yields: {value.message if isinstance(value, MarketVisionError) else 'not available'}"
    else:
        missing["macro"] = "not requested"

    return MarketContext(spy=spy, qqq=qqq, vix=vix, sector=sector, sector_symbol=sector_symbol, us10y=us10y, yield_curve=curve, missing=missing)
