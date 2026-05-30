from __future__ import annotations

import pandas as pd

from app.services.advanced_intelligence import AdvancedMarketIntelligenceEngine
from app.services.market import MarketDataService


class FakeMarketDataService(MarketDataService):
    def get_history(self, ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        if interval == "1m":
            periods = 420
            freq = "min"
        elif interval == "5m":
            periods = 360
            freq = "5min"
        elif interval == "15m":
            periods = 260
            freq = "15min"
        elif interval == "1h":
            periods = 260
            freq = "h"
        else:
            periods = 320
            freq = "D"

        dates = pd.date_range("2025-01-01 09:30", periods=periods, freq=freq)
        base = 100 if ticker.upper() not in {"SPY", "QQQ", "^VIX"} else 400
        if ticker.upper() == "^VIX":
            base = 18
        close = pd.Series([base + i * 0.05 + (i % 7) * 0.02 for i in range(periods)], index=dates)
        volume = pd.Series([1_000_000 + i * 1_000 for i in range(periods)], index=dates)
        return pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.3,
                "Low": close - 0.3,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )


def test_advanced_engine_returns_multifactor_output() -> None:
    engine = AdvancedMarketIntelligenceEngine(market_service=FakeMarketDataService())

    result = engine.analyze("AAPL", horizon_steps=8)

    assert result["ticker"] == "AAPL"
    assert round(sum(result["probabilities"].values()), 2) == 1.0
    assert "vwap" in result["technical_analysis"]
    assert result["intraday_vwap"]["available"] is True
    assert "5m" in result["intraday_vwap"]["timeframes"]
    assert "5m" in result["multi_timeframe_alignment"]["trends"]
    assert result["monte_carlo_simulation"]["horizon_steps"] == 8
    assert len(result["monte_carlo_simulation"]["main_predicted_line"]) == 8
    similarity = result["historical_similarity"]
    assert similarity["available"] is True
    assert "current_setup_features" in similarity
    assert "best_case_pct" in similarity
    assert "worst_case_pct" in similarity
    assert "average_drawdown_pct" in similarity
    assert "not financial advice" in result["disclaimer"].lower()

    simulation = engine.build_simulation_response("AAPL", horizon_steps=8)
    assert len(simulation["monte_carlo_chart"]["main_predicted_path"]) == 8
    assert len(simulation["monte_carlo_chart"]["confidence_upper"]) == 8
    assert simulation["monte_carlo_chart"]["expected_range"]["high"] >= simulation["monte_carlo_chart"]["expected_range"]["low"]
    assert simulation["monte_carlo_chart"]["main_predicted_path"][0]["is_prediction"] is True


def test_advanced_engine_reports_missing_unavailable_layers() -> None:
    engine = AdvancedMarketIntelligenceEngine(market_service=FakeMarketDataService())

    result = engine.analyze("MSFT", include_intraday=False)

    assert "missing" in result["coverage"]
    assert "options flow" in result["coverage"]["missing"]
    assert result["macro"]["available"] is False
