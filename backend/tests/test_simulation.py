from __future__ import annotations

import pandas as pd

from app.services.market import MarketDataService
from app.services.simulation import SimulationService


class FakeMarketDataService(MarketDataService):
    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        dates = pd.date_range("2025-01-01", periods=80, freq="D")
        close = pd.Series([100 + index * 0.45 for index in range(80)], index=dates)
        return pd.DataFrame(
            {
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": [1_000_000 + index * 5_000 for index in range(80)],
            },
            index=dates,
        )


def test_prediction_simulation_returns_required_scenarios() -> None:
    service = SimulationService(market_service=FakeMarketDataService())

    result = service.build_prediction_simulation("AAPL", horizon_steps=8)

    assert result["ticker"] == "AAPL"
    assert result["horizon_steps"] == 8
    assert set(result["scenario_paths"]) == {"bullish", "bearish", "sideways", "high_volatility"}
    assert len(result["predicted_prices"]) == 8
    assert len(result["confidence_band"]["upper"]) == 8
    assert result["dominant_scenario"] in {"bullish", "bearish", "sideways"}
    assert "not financial advice" in result["disclaimer"].lower()


def test_prediction_simulation_probabilities_sum_to_one() -> None:
    service = SimulationService(market_service=FakeMarketDataService())

    result = service.build_prediction_simulation("MSFT", horizon_steps=12)

    assert round(sum(result["probabilities"].values()), 2) == 1.0
    assert result["confidence"] > 0
    assert result["risk_level"] in {"low", "medium", "high"}
