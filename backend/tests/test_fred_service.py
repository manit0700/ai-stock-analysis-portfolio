from __future__ import annotations

from app.services.fred_service import FredService


def test_macro_intelligence_scores_and_regime(monkeypatch) -> None:
    service = FredService()

    def fake_snapshot():
        return {
            "fed_funds_rate": {"value": 5.25, "previous": 5.25, "trend": "neutral"},
            "cpi_inflation": {"value": 315.0, "previous": 314.0, "trend": "up"},
            "unemployment": {"value": 4.2, "previous": 4.0, "trend": "up"},
            "gdp_growth": {"value": 0.6, "previous": 1.2, "trend": "down"},
            "10y_treasury": {"value": 4.6, "previous": 4.5, "trend": "up"},
            "2y_treasury": {"value": 4.9, "previous": 4.8, "trend": "up"},
            "vix": {"value": 28.0, "previous": 22.0, "trend": "up"},
            "yield_curve_spread": {"value": -0.3, "label": "inverted"},
        }

    monkeypatch.setattr(service, "get_macro_snapshot", fake_snapshot)

    result = service.get_macro_intelligence()

    assert result["available"] is True
    assert result["macro_regime_label"] in {"risk_off", "restrictive_rates", "inverted_curve_caution"}
    assert result["scores"]["recession_risk_score"] >= 65
    assert result["scores"]["rate_pressure_score"] >= 70
    assert "Macro regime" in result["summary"]
