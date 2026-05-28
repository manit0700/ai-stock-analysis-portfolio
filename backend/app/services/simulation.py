from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.services.market import MarketDataService, _annualized_volatility, _compute_rsi, _max_drawdown, normalize_history


def _safe_float(value: Any, digits: int = 4) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series) -> dict[str, float | None]:
    if len(close) < 35:
        return {"macd": None, "signal": None, "histogram": None}
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd_line, 9)
    return {
        "macd": _safe_float(macd_line.iloc[-1]),
        "signal": _safe_float(signal.iloc[-1]),
        "histogram": _safe_float((macd_line - signal).iloc[-1]),
    }


def _atr(history: pd.DataFrame, window: int = 14) -> float | None:
    if len(history) < window + 1:
        return None
    high_low = history["High"] - history["Low"]
    high_close = (history["High"] - history["Close"].shift()).abs()
    low_close = (history["Low"] - history["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return _safe_float(true_range.rolling(window=window).mean().iloc[-1])


def _bollinger(close: pd.Series, window: int = 20) -> dict[str, float | None]:
    if len(close) < window:
        return {"upper": None, "middle": None, "lower": None}
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    return {
        "upper": _safe_float((middle + 2 * std).iloc[-1], 2),
        "middle": _safe_float(middle.iloc[-1], 2),
        "lower": _safe_float((middle - 2 * std).iloc[-1], 2),
    }


def _support_resistance(history: pd.DataFrame, lookback: int = 30) -> dict[str, float | None]:
    recent = history.tail(lookback)
    if recent.empty:
        return {"support": None, "resistance": None}
    return {
        "support": _safe_float(recent["Low"].min(), 2),
        "resistance": _safe_float(recent["High"].max(), 2),
    }


def _make_path(start: float, drift: float, volatility: float, steps: int, curve: str) -> list[float]:
    values: list[float] = []
    for step in range(1, steps + 1):
        progress = step / steps
        if curve == "bullish":
            move = drift * progress + volatility * 0.35 * np.sin(progress * np.pi)
        elif curve == "bearish":
            move = -drift * progress - volatility * 0.25 * np.sin(progress * np.pi)
        elif curve == "volatile":
            move = drift * 0.25 * progress + volatility * np.sin(progress * np.pi * 3)
        else:
            move = drift * 0.1 * progress + volatility * 0.2 * np.sin(progress * np.pi * 2)
        values.append(round(start * (1 + move), 2))
    return values


def _interval_seconds(interval: str) -> int:
    return {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "90m": 5400,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
        "5d": 432000,
        "1wk": 604800,
        "1w": 604800,
    }.get(interval.lower(), 86400)


def _make_model_predicted_candles(
    start: float,
    closes: list[float],
    volatility: float,
    last_timestamp: datetime,
    interval: str,
    probabilities: dict[str, float],
    dominant: str,
    confidence: float,
    risk_score: int,
    market_intelligence: dict[str, Any] | None,
) -> list[dict[str, float | str]]:
    candles: list[dict[str, float | str]] = []
    previous_close = start
    events = (market_intelligence or {}).get("event_probabilities", {})
    regression = (market_intelligence or {}).get("expected_price_range", {})
    volatility_expansion = float(events.get("volatility_expansion", {}).get("probability", 0))
    breakout_probability = float(events.get("breakout", {}).get("probability", 0))
    breakdown_probability = float(events.get("breakdown", {}).get("probability", 0))
    confidence_ratio = max(min(confidence / 100, 1), 0)
    risk_ratio = max(min(risk_score / 100, 1), 0)
    directional_edge = float(probabilities.get("bullish", 0)) - float(probabilities.get("bearish", 0))
    range_upper_pct = float(regression.get("upper_pct", 0) or 0)
    range_lower_pct = float(regression.get("lower_pct", 0) or 0)
    interval_delta = timedelta(seconds=_interval_seconds(interval))

    for index, close in enumerate(closes):
        progress = (index + 1) / max(len(closes), 1)
        forecast_time = last_timestamp + interval_delta * (index + 1)
        open_price = previous_close
        body_high = max(open_price, close)
        body_low = min(open_price, close)
        move_pct = abs(close / open_price - 1) if open_price else 0.0

        uncertainty = volatility * (0.45 + risk_ratio + volatility_expansion * 0.6) * (1.15 - confidence_ratio * 0.45)
        trend_wick = move_pct * (0.28 + risk_ratio * 0.35)
        upper_bias = max(directional_edge, 0) * 0.003 + breakout_probability * 0.004
        lower_bias = max(-directional_edge, 0) * 0.003 + breakdown_probability * 0.004
        wick_up = max(uncertainty + trend_wick + upper_bias, 0.0015)
        wick_down = max(uncertainty + trend_wick + lower_bias, 0.0015)

        model_upper = start * (1 + range_upper_pct * progress) if range_upper_pct else None
        model_lower = start * (1 + range_lower_pct * progress) if range_lower_pct else None
        high = body_high * (1 + wick_up)
        low = body_low * (1 - wick_down)
        if model_upper is not None:
            high = max(high, model_upper)
        if model_lower is not None:
            low = min(low, model_lower)

        candles.append({
            "date": forecast_time.isoformat(),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(max(low, 0.01), 2),
            "close": round(close, 2),
            "source": "ml_ensemble" if market_intelligence else "rules_engine",
            "scenario": dominant,
            "confidence": round(confidence_ratio, 3),
            "risk": round(risk_ratio, 3),
            "prediction_type": "exact_model_ohlc_forecast",
        })
        previous_close = close
    return candles


class SimulationService:
    def __init__(
        self,
        market_service: MarketDataService | None = None,
        model_service: Any | None = None,
    ) -> None:
        self.market_service = market_service or MarketDataService()
        self.model_service = model_service

    def build_prediction_simulation(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        horizon_steps: int = 12,
    ) -> dict[str, Any]:
        history = self.market_service.get_history(ticker=ticker, period=period, interval=interval)
        history = history.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        close = history["Close"]
        current_price = float(close.iloc[-1])
        returns = close.pct_change().dropna()

        indicators = self._build_indicators(history)
        rule_probs, reasons, risks = self._score_probabilities(history, indicators)

        ml_probs = None
        auxiliary_ml = None
        model_version = None
        if self.model_service is not None:
            ml_probs = self.model_service.predict_probabilities(history)
            auxiliary_ml = self.model_service.predict_auxiliary(history)
            model_version = self.model_service.model_version

        probabilities = self._blend_probabilities(rule_probs, ml_probs)
        dominant = max(probabilities, key=probabilities.get)

        daily_volatility = float(returns.tail(30).std()) if not returns.empty else 0.02
        drift = float(returns.tail(20).mean()) * horizon_steps if not returns.empty else 0.0
        drift = max(min(drift, 0.08), -0.08)
        band = max(daily_volatility * np.sqrt(max(horizon_steps, 1)), 0.01)

        paths = {
            "bullish": _make_path(current_price, abs(drift) + band * 0.75, band, horizon_steps, "bullish"),
            "bearish": _make_path(current_price, abs(drift) + band * 0.75, band, horizon_steps, "bearish"),
            "sideways": _make_path(current_price, drift, band, horizon_steps, "sideways"),
            "high_volatility": _make_path(current_price, drift, band, horizon_steps, "volatile"),
        }
        main_path = paths[dominant] if dominant in paths else paths["sideways"]

        confidence = round(max(probabilities.values()) * 100, 1)
        risk_score = self._risk_score(indicators)
        market_intelligence = self._build_market_intelligence(auxiliary_ml, current_price)
        last_timestamp = pd.to_datetime(history.index[-1]).to_pydatetime()

        return {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "horizon_steps": horizon_steps,
            "current_price": round(current_price, 2),
            "probabilities": {key: round(value, 3) for key, value in probabilities.items()},
            "dominant_scenario": dominant,
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": self._risk_level(risk_score),
            "ml_enabled": ml_probs is not None,
            "auxiliary_ml_enabled": auxiliary_ml is not None,
            "model_version": model_version,
            "market_intelligence": market_intelligence,
            "indicators": indicators,
            "scenario_paths": paths,
            "predicted_prices": main_path,
            "predicted_candles": _make_model_predicted_candles(
                current_price,
                main_path,
                daily_volatility,
                last_timestamp,
                interval,
                probabilities,
                dominant,
                confidence,
                risk_score,
                market_intelligence,
            ),
            "predicted_candle_model": {
                "source": "ml_ensemble" if ml_probs is not None else "rules_engine",
                "model_version": model_version,
                "uses": [
                    "dominant_scenario",
                    "bullish_bearish_sideways_probabilities",
                    "model_confidence",
                    "risk_score",
                    "volatility_expansion_probability",
                    "breakout_breakdown_probabilities",
                    "expected_model_price_range",
                ],
            },
            "confidence_band": {
                "upper": [round(price * (1 + band), 2) for price in main_path],
                "lower": [round(price * (1 - band), 2) for price in main_path],
            },
            "llm_explanation_facts": {
                "dominant_scenario": dominant,
                "probabilities": {key: round(value, 3) for key, value in probabilities.items()},
                "confidence": confidence,
                "risk_score": risk_score,
                "supporting_reasons": reasons[:4],
                "risk_factors": risks[:4],
                "model_version": model_version,
                "market_intelligence": market_intelligence,
                "required_disclaimer": (
                    "Probability-based market simulations and AI-generated financial intelligence, "
                    "not financial advice."
                ),
            },
            "reasoning": self._build_reasoning(ticker.upper(), dominant, confidence, reasons, risks),
            "reasons": reasons,
            "risks": risks,
            "recent_history": normalize_history(history, limit=90),
            "disclaimer": (
                "Probability-based market simulations and AI-generated financial intelligence, "
                "not financial advice."
            ),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def _build_market_intelligence(self, auxiliary_ml: dict[str, Any] | None, current_price: float) -> dict[str, Any] | None:
        if not auxiliary_ml:
            return None
        events = auxiliary_ml.get("event_probabilities", {})
        regression = auxiliary_ml.get("regression_outputs", {})
        future_high_pct = regression.get("future_range_high_pct")
        future_low_pct = regression.get("future_range_low_pct")
        expected_return = regression.get("return_horizon")
        risk_adjusted_return = regression.get("risk_adjusted_return")

        price_range = None
        if future_high_pct is not None or future_low_pct is not None:
            price_range = {
                "upper": round(current_price * (1 + float(future_high_pct or 0)), 2),
                "lower": round(current_price * (1 + float(future_low_pct or 0)), 2),
                "upper_pct": round(float(future_high_pct or 0), 4),
                "lower_pct": round(float(future_low_pct or 0), 4),
            }

        event_payload = {
            name: {
                "probability": round(float(values.get("probability", 0)), 3),
                "confidence": round(float(values.get("confidence", 0)), 3),
            }
            for name, values in events.items()
        }
        top_event = None
        if event_payload:
            top_event = max(event_payload, key=lambda key: event_payload[key]["probability"])

        return {
            "event_probabilities": event_payload,
            "top_event_signal": top_event,
            "expected_return_pct": round(float(expected_return), 4) if expected_return is not None else None,
            "expected_price_range": price_range,
            "risk_adjusted_return": round(float(risk_adjusted_return), 4) if risk_adjusted_return is not None else None,
            "model_version": auxiliary_ml.get("model_version"),
            "disclaimer": auxiliary_ml.get(
                "disclaimer",
                "Probability-based market simulations and AI-generated financial intelligence, not financial advice.",
            ),
        }

    def _blend_probabilities(
        self,
        rule_probs: dict[str, float],
        ml_probs: dict[str, float] | None,
        ml_weight: float = 0.6,
    ) -> dict[str, float]:
        if ml_probs is None:
            return rule_probs
        rule_w = 1.0 - ml_weight
        blended = {
            "bullish": ml_weight * ml_probs["bullish"] + rule_w * rule_probs["bullish"],
            "bearish": ml_weight * ml_probs["bearish"] + rule_w * rule_probs["bearish"],
            "sideways": ml_weight * ml_probs["sideways"] + rule_w * rule_probs["sideways"],
        }
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}

    def _build_indicators(self, history: pd.DataFrame) -> dict[str, Any]:
        close = history["Close"]
        volume = history["Volume"]
        sma_20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else None
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None
        ema_9 = _ema(close, 9).iloc[-1] if len(close) >= 9 else None
        ema_20 = _ema(close, 20).iloc[-1] if len(close) >= 20 else None
        avg_volume = volume.tail(20).mean() if len(volume) >= 20 else volume.mean()
        volume_ratio = (volume.iloc[-1] / avg_volume) if avg_volume else None
        sr = _support_resistance(history)

        return {
            "sma_20": _safe_float(sma_20, 2),
            "sma_50": _safe_float(sma_50, 2),
            "ema_9": _safe_float(ema_9, 2),
            "ema_20": _safe_float(ema_20, 2),
            "rsi_14": _compute_rsi(close),
            "macd": _macd(close),
            "bollinger": _bollinger(close),
            "atr_14": _atr(history),
            "volume_ratio_20d": _safe_float(volume_ratio, 3),
            "annualized_volatility_30d": _annualized_volatility(close),
            "max_drawdown_period": _max_drawdown(close),
            "support": sr["support"],
            "resistance": sr["resistance"],
        }

    def _score_probabilities(self, history: pd.DataFrame, indicators: dict[str, Any]) -> tuple[dict[str, float], list[str], list[str]]:
        close = history["Close"]
        current = float(close.iloc[-1])
        bullish = 0.34
        bearish = 0.33
        sideways = 0.33
        reasons: list[str] = []
        risks: list[str] = []

        sma_20 = indicators["sma_20"]
        sma_50 = indicators["sma_50"]
        rsi = indicators["rsi_14"]
        macd_hist = indicators["macd"]["histogram"]
        volume_ratio = indicators["volume_ratio_20d"]
        volatility = indicators["annualized_volatility_30d"]

        if sma_20 and current > sma_20:
            bullish += 0.08
            reasons.append("Price is above the 20-period average, which supports trend continuation.")
        elif sma_20:
            bearish += 0.07
            risks.append("Price is below the 20-period average, which weakens short-term momentum.")

        if sma_20 and sma_50 and sma_20 > sma_50:
            bullish += 0.07
            reasons.append("The 20-period average is above the 50-period average.")
        elif sma_20 and sma_50:
            bearish += 0.07
            risks.append("The 20-period average is below the 50-period average.")

        if rsi is not None and 50 <= rsi <= 70:
            bullish += 0.06
            reasons.append("RSI is in a constructive momentum range without being extremely overbought.")
        elif rsi is not None and rsi > 70:
            bearish += 0.05
            risks.append("RSI is overbought, raising pullback risk.")
        elif rsi is not None and rsi < 35:
            sideways += 0.04
            reasons.append("RSI is oversold, so a stabilization or rebound scenario is possible.")

        if macd_hist is not None and macd_hist > 0:
            bullish += 0.05
            reasons.append("MACD histogram is positive.")
        elif macd_hist is not None:
            bearish += 0.05
            risks.append("MACD histogram is negative.")

        if volume_ratio is not None and volume_ratio > 1.4:
            bullish += 0.04
            bearish += 0.03
            reasons.append("Recent volume is elevated, so the move has stronger participation.")
        elif volume_ratio is not None and volume_ratio < 0.75:
            sideways += 0.05
            risks.append("Volume is below its recent average, which can reduce signal quality.")

        if volatility is not None and volatility > 0.45:
            bearish += 0.04
            risks.append("Volatility is elevated, so the projected path has higher uncertainty.")
        elif volatility is not None and volatility < 0.25:
            sideways += 0.03
            reasons.append("Volatility is moderate, which supports tighter scenario ranges.")

        total = bullish + bearish + sideways
        probabilities = {
            "bullish": bullish / total,
            "bearish": bearish / total,
            "sideways": sideways / total,
        }

        if not reasons:
            reasons.append("No strong bullish signals were detected; the model relies on balanced scenario probabilities.")
        if not risks:
            risks.append("No major technical risk flags were triggered, but market risk still applies.")

        return probabilities, reasons, risks

    def _risk_score(self, indicators: dict[str, Any]) -> int:
        score = 35
        volatility = indicators["annualized_volatility_30d"]
        drawdown = indicators["max_drawdown_period"]
        rsi = indicators["rsi_14"]
        volume_ratio = indicators["volume_ratio_20d"]

        if volatility is not None:
            score += int(min(volatility * 70, 35))
        if drawdown is not None:
            score += int(min(abs(drawdown) * 60, 20))
        if rsi is not None and (rsi > 70 or rsi < 30):
            score += 10
        if volume_ratio is not None and volume_ratio > 1.8:
            score += 8

        return max(0, min(100, score))

    def _risk_level(self, risk_score: int) -> str:
        if risk_score >= 70:
            return "high"
        if risk_score >= 45:
            return "medium"
        return "low"

    def _build_reasoning(
        self,
        ticker: str,
        dominant: str,
        confidence: float,
        reasons: list[str],
        risks: list[str],
    ) -> str:
        reason_text = " ".join(reasons[:2])
        risk_text = " ".join(risks[:2])
        return (
            f"{ticker} currently has a {dominant} dominant scenario with {confidence}% model confidence. "
            f"{reason_text} Main risk: {risk_text} "
            "This is a probability-based market simulation and AI-generated financial intelligence, not financial advice."
        )
