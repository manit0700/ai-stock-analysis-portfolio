from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "ml_model_loaded" in payload
    assert "ml_model_version" in payload


def test_predict_endpoint_uses_prediction_request(monkeypatch) -> None:
    def fake_prediction(ticker: str, period: str = "1y", interval: str = "1d", horizon_steps: int = 12):
        return {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "horizon_steps": horizon_steps,
            "dominant_scenario": "bullish",
        }

    monkeypatch.setattr(main_module.simulation_service, "build_prediction_simulation", fake_prediction)

    response = client.post(
        "/api/predict",
        json={"ticker": "nvda", "period": "6mo", "interval": "1d", "horizon_steps": 8},
    )

    assert response.status_code == 200
    assert response.json()["ticker"] == "NVDA"
    assert response.json()["horizon_steps"] == 8


def test_bot_historical_similarity_endpoint(monkeypatch) -> None:
    def fake_analyze(ticker: str, horizon_steps: int = 12, include_intraday: bool = True):
        return {
            "historical_similarity": {
                "available": True,
                "current_setup_features": {"rsi": 55, "macd": 0.1},
                "similar_setups": [
                    {
                        "date": "2025-01-10",
                        "similarity_score": 0.82,
                        "future_10_bar_return_pct": 3.4,
                        "best_case_pct": 5.2,
                        "worst_case_pct": -1.1,
                        "max_drawdown_pct": -0.8,
                        "outcome": "bullish",
                    }
                ],
                "average_similarity_score": 0.82,
                "average_future_return_pct": 3.4,
                "outcome_probabilities": {"bullish": 1.0, "bearish": 0.0, "sideways": 0.0},
                "best_case_pct": 5.2,
                "worst_case_pct": -1.1,
                "average_drawdown_pct": -0.8,
            }
        }

    monkeypatch.setattr(main_module.advanced_engine, "analyze", fake_analyze)

    response = client.get("/api/bot/historical-similarity?ticker=nvda")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "NVDA"
    assert payload["available"] is True
    assert payload["similarity_score"] == 0.82
    assert payload["bullish_outcome_pct"] == 1.0
    assert payload["top_similar_historical_dates"][0]["date"] == "2025-01-10"


def test_scanner_uses_ranking_factors_and_failed_gates(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "record_signal", lambda payload: None)

    def fake_simulation(ticker: str, period: str = "5d", interval: str = "1d", horizon_steps: int = 12):
        strong = ticker.upper() == "AAA"
        return {
            "ticker": ticker.upper(),
            "model_version": "test",
            "period": period,
            "interval": interval,
            "horizon_steps": horizon_steps,
            "dominant_scenario": "bullish",
            "quality_label": "high_confidence_trade_candidate" if strong else "watchlist_candidate",
            "coverage_level": "promoted_signal" if strong else "qualified_prediction",
            "final_signal": {"probability": 0.82 if strong else 0.42},
            "ai_signal_bot": {
                "quality_label": "high_confidence_trade_candidate" if strong else "watchlist_candidate",
                "coverage_level": "promoted_signal" if strong else "qualified_prediction",
                "action": "paper_trade_candidate" if strong else "watch",
                "all_gates_passed": strong,
                "failed_gates": [],
                "confidence": 80 if strong else 62,
                "risk_score": 42 if strong else 75,
                "expected_return": 0.02 if strong else 0.001,
                "dominant_scenario": "bullish",
                "trade_plan": {
                    "eligible_for_paper_trade": strong,
                    "entry_price": 100,
                    "stop_loss": 95,
                    "target_1": 110,
                    "target_2": 115,
                    "risk_reward_target_1": 2.0 if strong else 0.8,
                },
            },
            "advanced_intelligence": {
                "historical_similarity": {"average_similarity_score": 0.8 if strong else 0.4},
                "sentiment": {"score": 0.1 if strong else -0.3},
                "macro": {"macro_regime_label": "mixed_macro" if strong else "risk_off"},
                "intraday_vwap": {"available": True, "alignment": "bullish_vwap_alignment"},
            },
            "disclaimer": "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
        }

    monkeypatch.setattr(main_module.advanced_engine, "build_simulation_response", fake_simulation)

    response = client.get("/api/bot/scan?tickers=BBB,AAA&period=5d&interval=1d&horizon_steps=8")

    assert response.status_code == 200
    rows = response.json()["results"]
    assert rows[0]["ticker"] == "AAA"
    assert rows[0]["final_signal_probability"] == 0.82
    assert rows[0]["historical_similarity_strength"] == 0.8
    assert "risk_failed" in rows[1]["failed_gates"]
    assert "sentiment_failed" in rows[1]["failed_gates"]
    assert "weak_historical_similarity" in rows[1]["failed_gates"]
    assert "macro_conflict" in rows[1]["failed_gates"]


def test_bot_explain_uses_backend_facts(monkeypatch) -> None:
    def fake_simulation(ticker: str, period: str = "1y", interval: str = "1d", horizon_steps: int = 12):
        return {
            "ticker": ticker.upper(),
            "quality_label": "prediction_only",
            "coverage_level": "universal_prediction",
            "dominant_scenario": "sideways",
            "probabilities": {"bullish": 0.4, "bearish": 0.25, "sideways": 0.35},
            "confidence": 55.0,
            "risk_score": 61,
            "risk_level": "medium",
            "final_signal": {"name": "breakout", "probability": 0.4, "confidence": 0.55},
            "ai_signal_bot": {
                "quality_label": "prediction_only",
                "action": "research_only",
                "all_gates_passed": False,
                "failed_gates": ["confidence_failed"],
            },
            "advanced_intelligence": {
                "technical_analysis": {"rsi_14": 52, "macd_hist": 0.02, "above_vwap": True, "vwap_distance_pct": 0.4, "volume_ratio_20": 1.1},
                "intraday_vwap": {"alignment": "mixed_vwap_alignment"},
                "historical_similarity": {"available": True, "average_similarity_score": 0.7, "average_future_return_pct": 1.2},
                "macro": {"macro_regime_label": "mixed_macro", "scores": {"risk_on_score": 50}},
                "sentiment": {"label": "neutral", "score": 0.0},
            },
            "reasons": ["Price is above VWAP."],
            "risks": ["Confidence is not high enough."],
        }

    monkeypatch.setattr(main_module.advanced_engine, "build_simulation_response", fake_simulation)
    monkeypatch.setattr(main_module.copilot_service, "is_available", lambda: False)

    response = client.post("/api/bot/explain", json={"ticker": "nvda", "period": "1y", "interval": "1d", "horizon_steps": 8})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "NVDA"
    assert payload["model"] == "deterministic_fallback"
    assert "confidence failed" in payload["explanation"]
    assert "not financial advice" in payload["explanation"].lower()
    assert payload["facts_used"]["technical"]["rsi_14"] == 52


def test_marketvision_tool_layer_lists_and_calls(monkeypatch) -> None:
    response = client.get("/api/tools")
    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()["tools"]}
    assert "predict_market_scenario" in names
    assert "analyze_portfolio" in names

    monkeypatch.setattr(main_module, "bot_performance", lambda: {"total_predictions": 3})
    call = client.post("/api/tools/get_performance_metrics", json={"arguments": {}})

    assert call.status_code == 200
    assert call.json()["tool"] == "get_performance_metrics"
    assert call.json()["result"]["total_predictions"] == 3
