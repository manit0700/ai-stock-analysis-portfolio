"""FRED (Federal Reserve Bank of St. Louis) economic series.

With a free API key the JSON API is used. Without one, the public CSV download behind every
FRED graph is used, so macro data still works in reduced mode. FRED revises some series after
first publication; anything used as a model feature must be lagged (see ``features.pipeline``).
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from app.core.errors import ProviderError
from app.core.secrets import Secret
from app.marketdata.http import HttpClient
from app.marketdata.providers.base import Provider

API_URL = "https://api.stlouisfed.org/fred/series/observations"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: Series the application knows by a friendly name.
SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp_growth": "A191RL1Q225SBEA",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "yield_curve_10y_2y": "T10Y2Y",
    "vix": "VIXCLS",
    "sp500": "SP500",
    "consumer_sentiment": "UMCSENT",
}


class FredProvider(Provider):
    name = "fred"
    supports = frozenset({"economic"})

    def __init__(self, api_key: Secret, http: HttpClient) -> None:
        self._key = api_key
        self._http = http

    def economic_series(self, series_id: str, start: date | None, end: date | None) -> pd.Series:
        series_id = series_id.strip().upper()
        if not series_id.replace("_", "").isalnum():
            raise ProviderError("FRED series ids contain only letters, digits and underscores.", details={"provider": self.name}, code="bad_request")
        if self._key:
            params = {"series_id": series_id, "api_key": self._key.get(), "file_type": "json", "sort_order": "asc"}
            if start:
                params["observation_start"] = start.isoformat()
            if end:
                params["observation_end"] = end.isoformat()
            payload = self._http.get(API_URL, params=params)
            rows = payload.get("observations", []) if isinstance(payload, dict) else []
            data = {pd.Timestamp(r["date"]): r["value"] for r in rows if "date" in r}
            series = pd.to_numeric(pd.Series(data, dtype="object"), errors="coerce")  # "." marks a missing value
        else:
            params = {"id": series_id}
            if start:
                params["cosd"] = start.isoformat()
            if end:
                params["coed"] = end.isoformat()
            text = self._http.get(CSV_URL, params=params, expect="text")
            try:
                frame = pd.read_csv(io.StringIO(text))
            except (ValueError, pd.errors.ParserError):
                raise ProviderError(f"fred returned an unreadable file for {series_id}.", details={"provider": self.name}) from None
            if frame.shape[1] < 2:
                raise ProviderError(f"fred returned no data for {series_id}.", details={"provider": self.name}, code="no_data")
            series = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
            series.index = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
            series = series[series.index.notna()]
        series.name = series_id
        series.index.name = "Date"
        if series.dropna().empty:
            raise ProviderError(f"fred returned no observations for {series_id}.", details={"provider": self.name}, code="no_data")
        return series.sort_index()
