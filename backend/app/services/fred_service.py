from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

_API_KEY = os.getenv("FRED_API_KEY", "")
_BASE = "https://api.stlouisfed.org/fred"
_TIMEOUT = 10

# Series IDs for key macro indicators
_SERIES = {
    "fed_funds_rate": "FEDFUNDS",       # Federal Funds Rate
    "cpi_inflation": "CPIAUCSL",        # CPI All Urban Consumers
    "unemployment": "UNRATE",           # Unemployment Rate
    "gdp_growth": "A191RL1Q225SBEA",    # Real GDP Growth Rate QoQ
    "10y_treasury": "GS10",             # 10-Year Treasury Yield
    "2y_treasury": "GS2",               # 2-Year Treasury Yield
    "vix": "VIXCLS",                    # CBOE VIX Index
    "sp500": "SP500",                   # S&P 500 Index
    "consumer_sentiment": "UMCSENT",    # U of Michigan Consumer Sentiment
    "m2_money_supply": "M2SL",          # M2 Money Supply
}


def _get_series(series_id: str, limit: int = 2) -> list[dict] | None:
    if not _API_KEY:
        return None
    try:
        r = requests.get(
            f"{_BASE}/series/observations",
            params={
                "series_id": series_id,
                "api_key": _API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        obs = data.get("observations", [])
        return [o for o in obs if o.get("value") != "."]
    except Exception as exc:
        log.warning("FRED %s failed: %s", series_id, exc)
        return None


def _latest_value(series_id: str) -> float | None:
    obs = _get_series(series_id, limit=2)
    if not obs:
        return None
    try:
        return float(obs[0]["value"])
    except (ValueError, KeyError):
        return None


def _latest_with_prev(series_id: str) -> tuple[float | None, float | None, str | None]:
    """Returns (current, previous, date)."""
    obs = _get_series(series_id, limit=2)
    if not obs:
        return None, None, None
    try:
        current = float(obs[0]["value"])
        prev = float(obs[1]["value"]) if len(obs) > 1 else None
        date = obs[0].get("date")
        return current, prev, date
    except (ValueError, KeyError, IndexError):
        return None, None, None


def _change_label(current: float | None, prev: float | None) -> str:
    if current is None or prev is None:
        return "neutral"
    if current > prev:
        return "up"
    if current < prev:
        return "down"
    return "neutral"


class FredService:
    def is_available(self) -> bool:
        return bool(_API_KEY)

    def get_macro_snapshot(self) -> dict[str, Any]:
        """Fetch latest readings for all key macro indicators."""
        results: dict[str, Any] = {}

        for name, series_id in _SERIES.items():
            current, prev, date = _latest_with_prev(series_id)
            results[name] = {
                "value": current,
                "previous": prev,
                "date": date,
                "change": round(current - prev, 4) if (current is not None and prev is not None) else None,
                "trend": _change_label(current, prev),
            }

        # Yield curve: 10y - 2y spread
        t10 = results.get("10y_treasury", {}).get("value")
        t2 = results.get("2y_treasury", {}).get("value")
        if t10 is not None and t2 is not None:
            spread = round(t10 - t2, 3)
            results["yield_curve_spread"] = {
                "value": spread,
                "label": "inverted" if spread < 0 else "normal",
                "description": "10Y - 2Y Treasury spread",
            }

        return results

    def get_indicator(self, series_id: str, limit: int = 12) -> dict[str, Any]:
        """Fetch historical observations for a specific FRED series."""
        obs = _get_series(series_id, limit=limit)
        if not obs:
            return {"series_id": series_id, "observations": [], "available": False}
        return {
            "series_id": series_id,
            "observations": [
                {"date": o["date"], "value": float(o["value"])} for o in obs
            ],
            "available": True,
        }

    def get_market_conditions(self) -> dict[str, Any]:
        """
        Summarise macro conditions as a simple assessment card
        useful for the market overview panel.
        """
        fed = _latest_value(_SERIES["fed_funds_rate"])
        cpi = _latest_value(_SERIES["cpi_inflation"])
        unemp = _latest_value(_SERIES["unemployment"])
        t10 = _latest_value(_SERIES["10y_treasury"])
        t2 = _latest_value(_SERIES["2y_treasury"])
        vix = _latest_value(_SERIES["vix"])

        flags: list[str] = []
        risks: list[str] = []

        if fed is not None:
            if fed >= 5.0:
                flags.append(f"High Fed Funds Rate ({fed}%) — restrictive monetary policy")
                risks.append("Rate-sensitive sectors under pressure")
            elif fed <= 1.5:
                flags.append(f"Low Fed Funds Rate ({fed}%) — accommodative policy")

        if t10 is not None and t2 is not None:
            spread = round(t10 - t2, 2)
            if spread < 0:
                flags.append(f"Inverted yield curve ({spread}%) — recession signal")
                risks.append("Historical recession predictor active")
            else:
                flags.append(f"Normal yield curve (spread: {spread}%)")

        if vix is not None:
            if vix > 30:
                flags.append(f"VIX at {vix} — high market fear")
                risks.append("Elevated volatility may trigger forced selling")
            elif vix < 15:
                flags.append(f"VIX at {vix} — market complacency")

        if unemp is not None:
            flags.append(f"Unemployment: {unemp}%")

        overall = "uncertain"
        if len(risks) == 0:
            overall = "positive"
        elif len(risks) >= 2:
            overall = "cautious"
        else:
            overall = "mixed"

        return {
            "fed_funds_rate": fed,
            "cpi": cpi,
            "unemployment": unemp,
            "treasury_10y": t10,
            "treasury_2y": t2,
            "vix": vix,
            "yield_spread": round(t10 - t2, 3) if (t10 and t2) else None,
            "flags": flags,
            "risks": risks,
            "overall_assessment": overall,
        }
