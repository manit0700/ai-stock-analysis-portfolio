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
