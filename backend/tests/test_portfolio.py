from __future__ import annotations

import pandas as pd

from app.services.portfolio import analyze_portfolio_from_prices


def test_analyze_portfolio_from_prices_flags_concentration_risk() -> None:
    price_frame = pd.DataFrame(
        {
            "AAPL": [100, 101, 103, 102, 105],
            "MSFT": [50, 51, 50, 52, 53],
        },
        index=pd.date_range("2025-01-01", periods=5, freq="D"),
    )
    positions = {"AAPL": 10, "MSFT": 1}

    result = analyze_portfolio_from_prices(price_frame, positions)

    assert result["total_market_value"] == 1103.0
    assert any("concentration risk" in warning.lower() for warning in result["warnings"])
    assert result["holdings"][0]["ticker"] == "AAPL"


def test_analyze_portfolio_from_prices_returns_correlation_matrix() -> None:
    price_frame = pd.DataFrame(
        {
            "AAPL": [100, 101, 99, 102, 104, 106],
            "MSFT": [200, 198, 202, 203, 205, 208],
            "NVDA": [50, 52, 51, 55, 56, 57],
        },
        index=pd.date_range("2025-02-01", periods=6, freq="D"),
    )
    positions = {"AAPL": 4, "MSFT": 2, "NVDA": 3}

    result = analyze_portfolio_from_prices(price_frame, positions)

    assert set(result["correlation_matrix"].keys()) == {"AAPL", "MSFT", "NVDA"}
    assert result["diversification_score"] > 0
