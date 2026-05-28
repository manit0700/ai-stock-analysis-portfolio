from __future__ import annotations

from app.services.market import MarketDataService


def test_compare_stocks_ranks_by_score(monkeypatch) -> None:
    service = MarketDataService()

    fixtures = {
        "AAPL": {"ticker": "AAPL", "company_name": "Apple", "score": 72, "stance": "bullish", "daily_change_pct": 1.2, "reasons": ["A"], "risks": ["R"]},
        "MSFT": {"ticker": "MSFT", "company_name": "Microsoft", "score": 61, "stance": "watch", "daily_change_pct": 0.5, "reasons": ["B"], "risks": ["S"]},
        "TSLA": {"ticker": "TSLA", "company_name": "Tesla", "score": 39, "stance": "cautious", "daily_change_pct": -1.1, "reasons": ["C"], "risks": ["T"]},
    }

    def fake_build_stock_analysis(ticker: str, period: str = "6mo", interval: str = "1d"):
        return fixtures[ticker]

    monkeypatch.setattr(service, "build_stock_analysis", fake_build_stock_analysis)
    result = service.compare_stocks(["TSLA", "AAPL", "MSFT"])

    assert [item["ticker"] for item in result["leaders"]] == ["AAPL", "MSFT", "TSLA"]
    assert result["summary"]["best_ticker"] == "AAPL"


def test_build_watchlist_uses_top_reason(monkeypatch) -> None:
    service = MarketDataService()

    monkeypatch.setattr(
        service,
        "compare_stocks",
        lambda tickers, period="6mo", interval="1d": {
            "leaders": [
                {
                    "ticker": "AAPL",
                    "company_name": "Apple",
                    "stance": "bullish",
                    "score": 70,
                    "daily_change_pct": 1.0,
                    "reasons": ["Price is above the 20-day moving average."],
                    "risks": ["Valuation risk."],
                }
            ]
        },
    )

    result = service.build_watchlist(["AAPL"])

    assert result["watchlist"][0]["headline_reason"] == "Price is above the 20-day moving average."
