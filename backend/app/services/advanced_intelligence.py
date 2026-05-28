from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.services.market import MarketDataService, normalize_history

DISCLAIMER = (
    "Probability-based market simulations and AI-generated financial intelligence, "
    "not financial advice."
)

TIMEFRAME_REQUESTS = {
    "1m": {"period": "7d", "interval": "1m"},
    "5m": {"period": "60d", "interval": "5m"},
    "15m": {"period": "60d", "interval": "15m"},
    "1h": {"period": "730d", "interval": "1h"},
    "1d": {"period": "2y", "interval": "1d"},
}

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLC": "Communication Services",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLP": "Consumer Staples",
}


def _safe_float(value: Any, digits: int = 4) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(history: pd.DataFrame, window: int = 14) -> pd.Series:
    high = history["High"]
    low = history["Low"]
    close = history["Close"]
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _macd_hist(close: pd.Series) -> pd.Series:
    macd = _ema(close, 12) - _ema(close, 26)
    return macd - _ema(macd, 9)


def _normalize_probs(scores: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(float(value), 0.001) for key, value in scores.items()}
    total = sum(cleaned.values()) or 1.0
    return {key: value / total for key, value in cleaned.items()}


def _trend_label(close: pd.Series) -> str:
    if len(close) < 50:
        return "unknown"
    ema_20 = _ema(close, 20).iloc[-1]
    ema_50 = _ema(close, 50).iloc[-1]
    last = close.iloc[-1]
    if last > ema_20 > ema_50:
        return "bullish"
    if last < ema_20 < ema_50:
        return "bearish"
    return "mixed"


@dataclass
class DataCoverage:
    trained: list[str]
    partial: list[str]
    missing: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "trained_or_computed_now": self.trained,
            "partial": self.partial,
            "missing": self.missing,
        }


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


def _predicted_candles_from_path(
    start: float,
    closes: list[float],
    confidence_band: dict[str, list[float]],
    last_timestamp: datetime,
    interval: str,
    dominant: str,
    confidence: float,
    risk_score: int,
    source: str,
) -> list[dict[str, Any]]:
    candles: list[dict[str, Any]] = []
    previous_close = start
    interval_delta = timedelta(seconds=_interval_seconds(interval))
    lower_band = confidence_band.get("lower", [])
    upper_band = confidence_band.get("upper", [])
    confidence_ratio = max(min(confidence / 100, 1), 0)
    risk_ratio = max(min(risk_score / 100, 1), 0)

    for index, close in enumerate(closes):
        forecast_time = last_timestamp + interval_delta * (index + 1)
        open_price = previous_close
        band_low = lower_band[index] if index < len(lower_band) else min(open_price, close)
        band_high = upper_band[index] if index < len(upper_band) else max(open_price, close)
        body_high = max(open_price, close)
        body_low = min(open_price, close)
        high = max(body_high, band_high)
        low = min(body_low, band_low)
        candles.append({
            "date": forecast_time.isoformat(),
            "open": round(float(open_price), 2),
            "high": round(float(high), 2),
            "low": round(max(float(low), 0.01), 2),
            "close": round(float(close), 2),
            "source": source,
            "scenario": dominant,
            "confidence": round(confidence_ratio, 3),
            "risk": round(risk_ratio, 3),
            "prediction_type": "monte_carlo_model_ohlc_forecast",
        })
        previous_close = close
    return candles


class AdvancedMarketIntelligenceEngine:
    def __init__(
        self,
        market_service: MarketDataService | None = None,
        fred_service: Any | None = None,
        model_service: Any | None = None,
        finnhub_service: Any | None = None,
        newsapi_service: Any | None = None,
        sentiment_service: Any | None = None,
    ) -> None:
        self.market_service = market_service or MarketDataService()
        self.fred_service = fred_service
        self.model_service = model_service
        self.finnhub_service = finnhub_service
        self.newsapi_service = newsapi_service
        self.sentiment_service = sentiment_service

    def analyze(
        self,
        ticker: str,
        horizon_steps: int = 12,
        include_intraday: bool = True,
    ) -> dict[str, Any]:
        ticker = ticker.upper()
        histories = self._load_timeframes(ticker, include_intraday=include_intraday)
        base_history = None
        for preferred_timeframe in ("5m", "1h", "1d"):
            candidate = histories.get(preferred_timeframe)
            if candidate is not None and not candidate.empty:
                base_history = candidate
                break
        if base_history is None or base_history.empty:
            base_history = self.market_service.get_history(ticker=ticker, period="1y", interval="1d")

        technical = self._technical_analysis(base_history)
        market_structure = self._market_structure(base_history)
        timeframe = self._multi_timeframe_alignment(histories)
        regime = self._market_regime()
        sector = self._sector_rotation(ticker, histories.get("1d"))
        macro = self._macro_snapshot()
        events = self._event_flags()
        similarity = self._historical_similarity(base_history, technical, regime, sector)
        sentiment = self._sentiment_snapshot(ticker)
        ml_probs = self.model_service.predict_probabilities(base_history) if self.model_service is not None else None
        auxiliary_ml = self.model_service.predict_auxiliary(base_history) if self.model_service is not None else None

        rule_probabilities = self._score_probabilities(
            technical=technical,
            structure=market_structure,
            timeframe=timeframe,
            regime=regime,
            sector=sector,
            macro=macro,
            similarity=similarity,
            sentiment=sentiment,
        )
        probabilities = self._blend_model_probabilities(rule_probabilities, ml_probs)
        risk_score = self._risk_score(technical, regime, macro)
        confidence = self._confidence_score(probabilities, timeframe, regime, similarity)
        final_signal = self._final_signal(auxiliary_ml, technical)
        monte_carlo = self._monte_carlo_paths(
            history=base_history,
            probabilities=probabilities,
            confidence=confidence,
            horizon_steps=horizon_steps,
        )

        return {
            "ticker": ticker,
            "engine": "MarketVision AI multi-factor intelligence engine",
            "probabilities": {key: round(value, 4) for key, value in probabilities.items()},
            "confidence_score": confidence,
            "risk_score": risk_score,
            "risk_level": self._risk_level(risk_score),
            "ml_enabled": ml_probs is not None,
            "auxiliary_ml_enabled": auxiliary_ml is not None,
            "model_version": self.model_service.model_version if self.model_service is not None else None,
            "final_signal": final_signal,
            "technical_analysis": technical,
            "market_structure": market_structure,
            "multi_timeframe_alignment": timeframe,
            "market_regime": regime,
            "sector_rotation": sector,
            "macro": macro,
            "sentiment": sentiment,
            "event_flags": events,
            "historical_similarity": similarity,
            "monte_carlo_simulation": monte_carlo,
            "market_intelligence": self._market_intelligence_payload(auxiliary_ml, final_signal, monte_carlo, base_history),
            "ai_signal_bot": self._bot_decision(probabilities, confidence, risk_score, final_signal, sentiment, monte_carlo, technical),
            "prediction_targets_supported": {
                "bullish_bearish_sideways": "computed",
                "future_price_range": "computed_from_monte_carlo",
                "risk_adjusted_return": "partial",
                "breakout_probability": "partial_rule_score",
                "reversal_probability": "partial_rule_score",
                "volatility_expansion_probability": "partial_rule_score",
            },
            "llm_explanation_facts": self._explanation_facts(
                ticker, probabilities, confidence, risk_score, technical, timeframe, regime, sector, similarity
            ),
            "coverage": self._coverage(include_intraday=include_intraday).as_dict(),
            "disclaimer": DISCLAIMER,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def build_simulation_response(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        horizon_steps: int = 12,
    ) -> dict[str, Any]:
        ticker = ticker.upper()
        history = self.market_service.get_history(ticker=ticker, period=period, interval=interval)
        history = history.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        close = history["Close"]
        current_price = float(close.iloc[-1])

        intelligence = self.analyze(
            ticker=ticker,
            horizon_steps=horizon_steps,
            include_intraday=interval != "1d",
        )
        monte_carlo = intelligence["monte_carlo_simulation"]
        probabilities = intelligence["probabilities"]
        dominant = max(probabilities, key=probabilities.get)
        main_path = monte_carlo["main_predicted_line"]
        confidence_band = monte_carlo["confidence_band"]
        confidence = float(intelligence["confidence_score"])
        risk_score = int(intelligence["risk_score"])
        last_timestamp = pd.to_datetime(history.index[-1]).to_pydatetime()
        source = "v12_final_hybrid_monte_carlo" if intelligence.get("ml_enabled") else "advanced_rules_monte_carlo"

        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "horizon_steps": horizon_steps,
            "current_price": round(current_price, 2),
            "probabilities": probabilities,
            "dominant_scenario": dominant,
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": intelligence["risk_level"],
            "ml_enabled": intelligence["ml_enabled"],
            "auxiliary_ml_enabled": intelligence["auxiliary_ml_enabled"],
            "model_version": intelligence["model_version"],
            "final_signal": intelligence["final_signal"],
            "quality_label": intelligence["ai_signal_bot"].get("quality_label"),
            "coverage_level": intelligence["ai_signal_bot"].get("coverage_level"),
            "market_intelligence": intelligence["market_intelligence"],
            "ai_signal_bot": intelligence["ai_signal_bot"],
            "advanced_intelligence": {
                "technical_analysis": intelligence["technical_analysis"],
                "market_structure": intelligence["market_structure"],
                "multi_timeframe_alignment": intelligence["multi_timeframe_alignment"],
                "market_regime": intelligence["market_regime"],
                "sector_rotation": intelligence["sector_rotation"],
                "macro": intelligence["macro"],
                "sentiment": intelligence["sentiment"],
                "event_flags": intelligence["event_flags"],
                "historical_similarity": intelligence["historical_similarity"],
                "monte_carlo_simulation": {
                    "method": monte_carlo.get("method"),
                    "n_paths": monte_carlo.get("n_paths"),
                    "horizon_steps": monte_carlo.get("horizon_steps"),
                    "expected_range": monte_carlo.get("expected_range"),
                    "confidence_used": monte_carlo.get("confidence_used"),
                    "volatility_cone_terminal": {
                        "p05": (monte_carlo.get("volatility_cone") or {}).get("p05", [None])[-1],
                        "p25": (monte_carlo.get("volatility_cone") or {}).get("p25", [None])[-1],
                        "p50": (monte_carlo.get("volatility_cone") or {}).get("p50", [None])[-1],
                        "p75": (monte_carlo.get("volatility_cone") or {}).get("p75", [None])[-1],
                        "p95": (monte_carlo.get("volatility_cone") or {}).get("p95", [None])[-1],
                    },
                },
                "coverage": intelligence["coverage"],
            },
            "scenario_paths": {
                "bullish": monte_carlo["bullish_path"],
                "bearish": monte_carlo["bearish_path"],
                "sideways": monte_carlo["sideways_path"],
                "high_volatility": monte_carlo["volatility_cone"]["p95"],
            },
            "predicted_prices": main_path,
            "predicted_candles": _predicted_candles_from_path(
                current_price,
                main_path,
                confidence_band,
                last_timestamp,
                interval,
                dominant,
                confidence,
                risk_score,
                source,
            ),
            "predicted_candle_model": {
                "source": source,
                "model_version": intelligence["model_version"],
                "uses": [
                    "v12_final_hybrid_policy",
                    "ml_probabilities_when_loaded",
                    "multi_timeframe_alignment",
                    "historical_similarity",
                    "monte_carlo_return_distribution",
                    "risk_score",
                    "confidence_band",
                ],
            },
            "confidence_band": confidence_band,
            "llm_explanation_facts": intelligence["llm_explanation_facts"],
            "reasoning": self._reasoning_text(ticker, dominant, confidence, risk_score, intelligence),
            "reasons": intelligence["llm_explanation_facts"]["facts"],
            "risks": self._risk_facts(intelligence),
            "recent_history": normalize_history(history, limit=90),
            "disclaimer": DISCLAIMER,
            "as_of": intelligence["as_of"],
        }

    def _load_timeframes(self, ticker: str, include_intraday: bool) -> dict[str, pd.DataFrame]:
        histories: dict[str, pd.DataFrame] = {}
        requests = TIMEFRAME_REQUESTS if include_intraday else {"1d": TIMEFRAME_REQUESTS["1d"]}
        for timeframe, params in requests.items():
            try:
                history = self.market_service.get_history(ticker=ticker, **params)
                histories[timeframe] = history.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            except Exception:
                continue

        if "1h" in histories and "4h" not in histories:
            hourly = histories["1h"].copy()
            hourly.index = pd.to_datetime(hourly.index)
            four_hour = hourly.resample("4h").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna()
            if not four_hour.empty:
                histories["4h"] = four_hour
        return histories

    def _technical_analysis(self, history: pd.DataFrame) -> dict[str, Any]:
        close = history["Close"]
        high = history["High"]
        low = history["Low"]
        volume = history["Volume"]
        current = float(close.iloc[-1])

        typical = (high + low + close) / 3
        session_key = pd.to_datetime(history.index).date
        cumulative_pv = (typical * volume).groupby(session_key).cumsum()
        cumulative_volume = volume.groupby(session_key).cumsum().replace(0, np.nan)
        vwap = cumulative_pv / cumulative_volume
        vwap_distance = current / vwap.iloc[-1] - 1 if not pd.isna(vwap.iloc[-1]) else np.nan
        vwap_slope = vwap.diff().tail(5).mean()

        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_9 = _ema(close, 9)
        ema_20 = _ema(close, 20)
        macd_hist = _macd_hist(close)
        atr = _atr(history)
        returns = close.pct_change()
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        support = low.tail(50).min()
        resistance = high.tail(50).max()
        volume_ratio = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(20).mean() else np.nan
        volatility = returns.tail(30).std()

        was_below_vwap = close.iloc[-2] < vwap.iloc[-2] if len(close) > 2 and not pd.isna(vwap.iloc[-2]) else False
        was_above_vwap = close.iloc[-2] > vwap.iloc[-2] if len(close) > 2 and not pd.isna(vwap.iloc[-2]) else False
        above_vwap = current > vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else False

        breakout_score = 1.0 if current >= resistance * 0.995 and volume_ratio > 1.25 else 0.0
        reversal_score = 1.0 if _safe_float(_rsi(close).iloc[-1]) and (_rsi(close).iloc[-1] < 32 or _rsi(close).iloc[-1] > 72) else 0.0
        volatility_expansion = 1.0 if volatility and returns.tail(5).std() > returns.tail(30).std() * 1.25 else 0.0

        return {
            "current_price": _safe_float(current, 2),
            "rsi_14": _safe_float(_rsi(close).iloc[-1], 2),
            "macd_hist": _safe_float(macd_hist.iloc[-1], 5),
            "macd_hist_slope": _safe_float(macd_hist.diff().iloc[-1], 5),
            "sma_20": _safe_float(sma_20.iloc[-1], 2),
            "sma_50": _safe_float(sma_50.iloc[-1], 2),
            "ema_9": _safe_float(ema_9.iloc[-1], 2),
            "ema_20": _safe_float(ema_20.iloc[-1], 2),
            "bollinger_upper": _safe_float(bb_upper.iloc[-1], 2),
            "bollinger_lower": _safe_float(bb_lower.iloc[-1], 2),
            "atr": _safe_float(atr.iloc[-1], 4),
            "atr_pct": _safe_float(atr.iloc[-1] / current if current else np.nan, 4),
            "vwap": _safe_float(vwap.iloc[-1], 2),
            "above_vwap": bool(above_vwap),
            "vwap_distance_pct": _safe_float(vwap_distance * 100, 3),
            "vwap_slope": _safe_float(vwap_slope, 5),
            "vwap_reclaim": bool(was_below_vwap and above_vwap),
            "vwap_rejection": bool(was_above_vwap and not above_vwap),
            "support": _safe_float(support, 2),
            "resistance": _safe_float(resistance, 2),
            "volume_ratio_20": _safe_float(volume_ratio, 3),
            "momentum_5": _safe_float(close.pct_change(5).iloc[-1] * 100, 3),
            "momentum_20": _safe_float(close.pct_change(20).iloc[-1] * 100, 3),
            "volatility_30": _safe_float(volatility, 5),
            "breakout_probability_proxy": breakout_score,
            "reversal_probability_proxy": reversal_score,
            "volatility_expansion_proxy": volatility_expansion,
        }

    def _market_structure(self, history: pd.DataFrame) -> dict[str, Any]:
        recent = history.tail(40)
        highs = recent["High"]
        lows = recent["Low"]
        close = recent["Close"]
        higher_highs = int((highs.diff() > 0).tail(10).sum())
        higher_lows = int((lows.diff() > 0).tail(10).sum())
        lower_highs = int((highs.diff() < 0).tail(10).sum())
        lower_lows = int((lows.diff() < 0).tail(10).sum())
        range_pct = (highs.max() - lows.min()) / close.iloc[-1] if close.iloc[-1] else np.nan
        consolidation = bool(range_pct < close.pct_change().tail(30).std() * 8) if not pd.isna(range_pct) else False
        if higher_highs + higher_lows > lower_highs + lower_lows + 3:
            structure = "bullish_structure"
        elif lower_highs + lower_lows > higher_highs + higher_lows + 3:
            structure = "bearish_structure"
        elif consolidation:
            structure = "consolidation"
        else:
            structure = "mixed"
        return {
            "structure": structure,
            "higher_high_count_10": higher_highs,
            "higher_low_count_10": higher_lows,
            "lower_high_count_10": lower_highs,
            "lower_low_count_10": lower_lows,
            "range_pct": _safe_float(range_pct * 100, 3),
            "consolidation_zone": consolidation,
        }

    def _multi_timeframe_alignment(self, histories: dict[str, pd.DataFrame]) -> dict[str, Any]:
        trends: dict[str, str] = {}
        score = 0
        for timeframe in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            history = histories.get(timeframe)
            if history is None or history.empty:
                trends[timeframe] = "unavailable"
                continue
            trend = _trend_label(history["Close"])
            trends[timeframe] = trend
            if trend == "bullish":
                score += 1
            elif trend == "bearish":
                score -= 1
        available = sum(1 for value in trends.values() if value != "unavailable") or 1
        normalized = score / available
        if normalized >= 0.45:
            alignment = "bullish_alignment"
        elif normalized <= -0.45:
            alignment = "bearish_alignment"
        else:
            alignment = "mixed_sideways_alignment"
        return {
            "trends": trends,
            "alignment_score": _safe_float(normalized, 3),
            "alignment": alignment,
        }

    def _market_regime(self) -> dict[str, Any]:
        try:
            spy = self.market_service.get_history("SPY", period="1y", interval="1d")
            qqq = self.market_service.get_history("QQQ", period="1y", interval="1d")
        except Exception:
            return {"available": False, "regime": "unknown"}
        try:
            vix = self.market_service.get_history("^VIX", period="6mo", interval="1d")
            vix_level = float(vix["Close"].iloc[-1])
        except Exception:
            vix_level = None

        spy_trend = _trend_label(spy["Close"])
        qqq_trend = _trend_label(qqq["Close"])
        spy_vol = spy["Close"].pct_change().tail(30).std() * np.sqrt(252)
        if vix_level and vix_level >= 28:
            volatility_regime = "high_volatility"
        elif vix_level and vix_level <= 16:
            volatility_regime = "low_volatility"
        elif spy_vol > 0.28:
            volatility_regime = "high_volatility"
        else:
            volatility_regime = "normal_volatility"

        if spy_trend == "bullish" and qqq_trend == "bullish":
            regime = "bull_market"
        elif spy_trend == "bearish" and qqq_trend == "bearish":
            regime = "bear_market"
        else:
            regime = "sideways_market"

        return {
            "available": True,
            "regime": regime,
            "volatility_regime": volatility_regime,
            "spy_trend": spy_trend,
            "qqq_trend": qqq_trend,
            "vix_level": _safe_float(vix_level, 2),
            "spy_annualized_volatility": _safe_float(spy_vol, 4),
        }

    def _sector_rotation(self, ticker: str, stock_daily: pd.DataFrame | None) -> dict[str, Any]:
        sector_rows = []
        for symbol, name in SECTOR_ETFS.items():
            try:
                hist = self.market_service.get_history(symbol, period="3mo", interval="1d")
                close = hist["Close"]
                ret_5 = close.pct_change(5).iloc[-1]
                ret_20 = close.pct_change(20).iloc[-1]
                sector_rows.append({"symbol": symbol, "name": name, "return_5d": ret_5, "return_20d": ret_20})
            except Exception:
                continue
        if not sector_rows:
            return {"available": False}

        ranked = sorted(sector_rows, key=lambda row: row["return_20d"], reverse=True)
        stock_vs_best = None
        if stock_daily is not None and len(stock_daily) >= 21:
            stock_return_20 = stock_daily["Close"].pct_change(20).iloc[-1]
            stock_vs_best = stock_return_20 - ranked[0]["return_20d"]

        return {
            "available": True,
            "leader": ranked[0]["symbol"],
            "laggard": ranked[-1]["symbol"],
            "sector_strength_score": _safe_float(np.mean([row["return_20d"] for row in ranked]) * 100, 3),
            "stock_vs_leading_sector_20d": _safe_float(stock_vs_best * 100, 3) if stock_vs_best is not None else None,
            "rankings": [
                {
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "return_5d_pct": _safe_float(row["return_5d"] * 100, 3),
                    "return_20d_pct": _safe_float(row["return_20d"] * 100, 3),
                }
                for row in ranked
            ],
        }

    def _macro_snapshot(self) -> dict[str, Any]:
        if self.fred_service is None or not getattr(self.fred_service, "is_available", lambda: False)():
            return {
                "available": False,
                "missing": ["interest rates", "inflation", "GDP", "unemployment", "treasury yields", "Fed/CPI calendars"],
            }
        try:
            return {"available": True, "source": "fred", "indicators": self.fred_service.get_macro_snapshot()}
        except Exception:
            return {"available": False, "error": "Unable to fetch FRED macro snapshot."}

    def _event_flags(self) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date()
        return {
            "earnings_today": None,
            "earnings_this_week": None,
            "fed_day": None,
            "cpi_day": None,
            "options_expiration": today.weekday() == 4 and 15 <= today.day <= 21,
            "major_news_event": None,
            "high_volatility_event": None,
            "status": "partial_calendar_only",
        }

    def _sentiment_snapshot(self, ticker: str) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if self.finnhub_service is not None and getattr(self.finnhub_service, "is_available", lambda: False)():
            try:
                items.extend(self.finnhub_service.get_news(ticker=ticker, limit=8))
            except Exception:
                pass
        if self.newsapi_service is not None and getattr(self.newsapi_service, "is_available", lambda: False)():
            try:
                items.extend(self.newsapi_service.get_stock_news(ticker=ticker, limit=8))
            except Exception:
                pass

        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            title = (item.get("title") or "").strip().lower()
            if title and title not in seen:
                seen.add(title)
                deduped.append(item)

        if self.sentiment_service is not None:
            try:
                result = self.sentiment_service.aggregate(deduped, limit=10)
                result["source_count"] = len(deduped)
                result["layer"] = "pretrained_financial_sentiment"
                return result
            except Exception:
                pass

        scores = [float(item.get("sentiment_score", 0) or 0) for item in deduped]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "available": bool(scores),
            "model": "provider_or_vader_fallback",
            "score": round(avg_score, 4),
            "label": "positive" if avg_score >= 0.05 else "negative" if avg_score <= -0.05 else "neutral",
            "confidence": round(abs(avg_score), 4),
            "item_count": len(deduped),
            "source_count": len(deduped),
            "items": deduped[:10],
            "layer": "fallback_news_sentiment",
            "disclaimer": "Sentiment is an auxiliary probability signal, not financial advice.",
        }

    def _historical_similarity(
        self,
        history: pd.DataFrame,
        technical: dict[str, Any],
        regime: dict[str, Any],
        sector: dict[str, Any],
    ) -> dict[str, Any]:
        close = history["Close"]
        if len(close) < 120:
            return {"available": False, "reason": "not enough history"}

        features = pd.DataFrame(index=history.index)
        features["rsi_14"] = _rsi(close)
        features["macd_hist"] = _macd_hist(close)
        features["atr_pct"] = _atr(history) / close.replace(0, np.nan)
        features["momentum_5"] = close.pct_change(5)
        features["momentum_20"] = close.pct_change(20)
        features["volume_ratio"] = history["Volume"] / history["Volume"].rolling(20).mean()
        features = features.dropna()
        if len(features) < 60:
            return {"available": False, "reason": "not enough engineered history"}

        current = features.iloc[-1]
        past = features.iloc[:-10]
        std = past.std().replace(0, np.nan)
        z_current = (current - past.mean()) / std
        z_past = (past - past.mean()) / std
        distances = ((z_past - z_current) ** 2).sum(axis=1) ** 0.5
        matches = distances.sort_values().head(5)

        outcomes = []
        for index, distance in matches.items():
            pos = history.index.get_loc(index)
            if pos + 10 >= len(history):
                continue
            future_return = close.iloc[pos + 10] / close.iloc[pos] - 1
            outcomes.append(
                {
                    "date": pd.Timestamp(index).date().isoformat(),
                    "similarity_score": _safe_float(max(0.0, 1.0 - distance / 10), 4),
                    "future_10_bar_return_pct": _safe_float(future_return * 100, 3),
                    "outcome": "bullish" if future_return > 0.005 else ("bearish" if future_return < -0.005 else "sideways"),
                }
            )

        if not outcomes:
            return {"available": False, "reason": "no valid future outcomes"}
        avg_return = float(np.mean([item["future_10_bar_return_pct"] for item in outcomes]))
        bullish = sum(1 for item in outcomes if item["outcome"] == "bullish") / len(outcomes)
        bearish = sum(1 for item in outcomes if item["outcome"] == "bearish") / len(outcomes)
        sideways = 1 - bullish - bearish
        return {
            "available": True,
            "average_similarity_score": _safe_float(np.mean([item["similarity_score"] for item in outcomes]), 4),
            "average_future_return_pct": _safe_float(avg_return, 3),
            "outcome_probabilities": {
                "bullish": _safe_float(bullish, 3),
                "bearish": _safe_float(bearish, 3),
                "sideways": _safe_float(sideways, 3),
            },
            "similar_setups": outcomes,
            "context_used": {
                "regime": regime.get("regime"),
                "sector_leader": sector.get("leader"),
                "current_rsi": technical.get("rsi_14"),
                "current_vwap_distance_pct": technical.get("vwap_distance_pct"),
            },
        }

    def _score_probabilities(
        self,
        technical: dict[str, Any],
        structure: dict[str, Any],
        timeframe: dict[str, Any],
        regime: dict[str, Any],
        sector: dict[str, Any],
        macro: dict[str, Any],
        similarity: dict[str, Any],
        sentiment: dict[str, Any],
    ) -> dict[str, float]:
        scores = {"bullish": 0.34, "bearish": 0.33, "sideways": 0.33}
        if technical.get("above_vwap"):
            scores["bullish"] += 0.06
        if technical.get("vwap_rejection"):
            scores["bearish"] += 0.06
        if technical.get("rsi_14") and 50 <= technical["rsi_14"] <= 68:
            scores["bullish"] += 0.05
        elif technical.get("rsi_14") and technical["rsi_14"] > 72:
            scores["bearish"] += 0.04
        if technical.get("macd_hist") and technical["macd_hist"] > 0:
            scores["bullish"] += 0.04
        elif technical.get("macd_hist") and technical["macd_hist"] < 0:
            scores["bearish"] += 0.04
        if structure.get("structure") == "bullish_structure":
            scores["bullish"] += 0.07
        elif structure.get("structure") == "bearish_structure":
            scores["bearish"] += 0.07
        elif structure.get("structure") == "consolidation":
            scores["sideways"] += 0.06
        alignment = timeframe.get("alignment")
        if alignment == "bullish_alignment":
            scores["bullish"] += 0.08
        elif alignment == "bearish_alignment":
            scores["bearish"] += 0.08
        else:
            scores["sideways"] += 0.04
        if regime.get("regime") == "bull_market":
            scores["bullish"] += 0.04
        elif regime.get("regime") == "bear_market":
            scores["bearish"] += 0.04
        if regime.get("volatility_regime") == "high_volatility":
            scores["bearish"] += 0.03
            scores["sideways"] -= 0.02
        if sector.get("stock_vs_leading_sector_20d") and sector["stock_vs_leading_sector_20d"] > 0:
            scores["bullish"] += 0.03
        if similarity.get("available"):
            sim_probs = similarity.get("outcome_probabilities", {})
            for key in scores:
                scores[key] += 0.08 * float(sim_probs.get(key, 0) or 0)
        sentiment_score = float(sentiment.get("score", 0) or 0)
        sentiment_confidence = float(sentiment.get("confidence", 0) or 0)
        sentiment_weight = min(0.06, 0.02 + sentiment_confidence * 0.04)
        if sentiment_score > 0.05:
            scores["bullish"] += sentiment_weight
        elif sentiment_score < -0.05:
            scores["bearish"] += sentiment_weight
        else:
            scores["sideways"] += 0.015
        return _normalize_probs(scores)

    def _blend_model_probabilities(
        self,
        rule_probabilities: dict[str, float],
        ml_probabilities: dict[str, float] | None,
        ml_weight: float = 0.55,
    ) -> dict[str, float]:
        if not ml_probabilities:
            return rule_probabilities
        blended = {
            key: ml_weight * float(ml_probabilities.get(key, 0)) + (1 - ml_weight) * float(rule_probabilities.get(key, 0))
            for key in ("bullish", "bearish", "sideways")
        }
        return _normalize_probs(blended)

    def _final_signal(self, auxiliary_ml: dict[str, Any] | None, technical: dict[str, Any]) -> dict[str, Any]:
        policy = (auxiliary_ml or {}).get("final_hybrid_policy") or {}
        events = (auxiliary_ml or {}).get("event_probabilities") or {}
        promoted_key = str(policy.get("promoted_signal", "target_elite_breakout_long_quality_success")).removeprefix("target_")
        promoted = events.get(promoted_key) or {}
        promoted_probability = float(promoted.get("probability", 0.0) or 0.0)
        promoted_confidence = float(promoted.get("confidence", 0.0) or 0.0)
        proof_report = policy.get("promoted_signal_report") or {}
        confidence_buckets = proof_report.get("confidence_buckets") or {}

        signal_state = "inactive"
        if promoted_probability >= 0.65 and bool(technical.get("breakout_probability_proxy")):
            signal_state = "elite_breakout_watch"
        if promoted_probability >= 0.8:
            signal_state = "elite_breakout_high_confidence"

        return {
            "name": promoted_key,
            "state": signal_state,
            "probability": round(promoted_probability, 4),
            "confidence": round(promoted_confidence, 4),
            "proof": {
                "all_active_setup_accuracy": _safe_float(proof_report.get("accuracy"), 4),
                "confidence_buckets": confidence_buckets,
                "setup_coverage_test": _safe_float(proof_report.get("setup_coverage_test"), 4),
                "baseline_win_rate_test": _safe_float(proof_report.get("baseline_win_rate_test"), 4),
                "proof_rule": proof_report.get("proof_rule"),
            },
            "accuracy_claim_policy": policy.get("accuracy_claim_policy"),
            "disclaimer": policy.get("disclaimer", DISCLAIMER),
        }

    def _market_intelligence_payload(
        self,
        auxiliary_ml: dict[str, Any] | None,
        final_signal: dict[str, Any],
        monte_carlo: dict[str, Any],
        history: pd.DataFrame,
    ) -> dict[str, Any] | None:
        if auxiliary_ml is None:
            return {
                "event_probabilities": {},
                "top_event_signal": final_signal.get("name"),
                "expected_return_pct": None,
                "expected_price_range": monte_carlo.get("expected_range"),
                "risk_adjusted_return": None,
                "final_signal": final_signal,
                "pretrained_layers": {
                    "enabled_now": ["FinBERT sentiment layer when available"],
                    "recommended_next": ["time-series foundation model embeddings"],
                },
                "disclaimer": DISCLAIMER,
            }

        events = auxiliary_ml.get("event_probabilities", {})
        regression = auxiliary_ml.get("regression_outputs", {})
        event_payload = {
            name: {
                "probability": round(float(values.get("probability", 0)), 4),
                "confidence": round(float(values.get("confidence", 0)), 4),
            }
            for name, values in events.items()
        }
        top_event = max(event_payload, key=lambda key: event_payload[key]["probability"]) if event_payload else None
        current_price = float(history["Close"].iloc[-1])
        future_high_pct = regression.get("future_range_high_pct")
        future_low_pct = regression.get("future_range_low_pct")
        expected_range = monte_carlo.get("expected_range")
        if future_high_pct is not None or future_low_pct is not None:
            expected_range = {
                "upper": round(current_price * (1 + float(future_high_pct or 0)), 2),
                "lower": round(current_price * (1 + float(future_low_pct or 0)), 2),
                "upper_pct": round(float(future_high_pct or 0), 4),
                "lower_pct": round(float(future_low_pct or 0), 4),
            }

        return {
            "event_probabilities": event_payload,
            "top_event_signal": top_event,
            "expected_return_pct": round(float(regression["return_horizon"]), 4) if regression.get("return_horizon") is not None else None,
            "expected_price_range": expected_range,
            "risk_adjusted_return": round(float(regression["risk_adjusted_return"]), 4) if regression.get("risk_adjusted_return") is not None else None,
            "final_signal": final_signal,
            "model_version": auxiliary_ml.get("model_version"),
            "pretrained_layers": {
                "enabled_now": ["FinBERT sentiment layer when available"],
                "recommended_next": ["time-series foundation model embeddings"],
            },
            "disclaimer": auxiliary_ml.get("disclaimer", DISCLAIMER),
        }

    def _bot_decision(
        self,
        probabilities: dict[str, float],
        confidence: float,
        risk_score: int,
        final_signal: dict[str, Any],
        sentiment: dict[str, Any],
        monte_carlo: dict[str, Any],
        technical: dict[str, Any],
    ) -> dict[str, Any]:
        dominant = max(probabilities, key=probabilities.get)
        promoted_probability = float(final_signal.get("probability", 0) or 0)
        proof80 = ((final_signal.get("proof") or {}).get("confidence_buckets") or {}).get(">=80%", {})
        proof80_accuracy = float(proof80.get("accuracy", 0) or 0)
        sentiment_score = float(sentiment.get("score", 0) or 0)
        expected_range = monte_carlo.get("expected_range") or {}
        current_path = monte_carlo.get("main_predicted_line") or []
        expected_return = None
        entry_price = float(current_path[0]) if current_path else None
        if current_path:
            first = current_path[0]
            last = current_path[-1]
            expected_return = (last / first - 1) if first else None

        action = "watch"
        stance = "neutral"
        gates = {
            "dominant_bullish": dominant == "bullish",
            "confidence_min_65": confidence >= 65,
            "final_signal_min_65": promoted_probability >= 0.65,
            "proof_80_bucket_min_75": proof80_accuracy >= 0.75,
            "risk_below_70": risk_score < 70,
            "sentiment_not_strong_negative": sentiment_score >= -0.25,
        }

        if risk_score >= 75:
            action = "avoid_high_risk"
            stance = "defensive"
        elif all(gates.values()):
            action = "paper_trade_candidate"
            stance = "bullish_selective"
        elif dominant == "bearish" and confidence >= 60:
            action = "avoid_or_hedge"
            stance = "bearish"
        elif dominant == "sideways":
            action = "wait_for_breakout"
            stance = "range_bound"

        if sentiment_score < -0.25 and action == "paper_trade_candidate":
            action = "watch_sentiment_conflict"
            stance = "mixed"

        reasons = [
            f"Dominant scenario is {dominant} with {confidence:.1f}% engine confidence.",
            f"Final signal probability is {promoted_probability * 100:.1f}%.",
            f"Proof bucket >=80% shows {proof80_accuracy * 100:.1f}% historical walk-forward accuracy on {proof80.get('signals', 0)} setup-filtered signals.",
            f"Sentiment is {sentiment.get('label', 'neutral')} with score {sentiment_score}.",
        ]
        risk_controls = [
            "Use this bot only for research or paper trading until live execution is separately validated.",
            "Require liquidity, spread, slippage, and position-size checks before any real execution.",
            "Do not treat the predicted path as guaranteed.",
        ]
        if expected_range:
            risk_controls.append(f"Monte Carlo expected range: {expected_range.get('low')} to {expected_range.get('high')}.")

        trade_plan = self._paper_trade_plan(
            action=action,
            dominant=dominant,
            entry_price=entry_price,
            expected_range=expected_range,
            confidence=confidence,
            risk_score=risk_score,
            volume_ratio=float(technical.get("volume_ratio_20") or 0),
        )
        all_gates_passed = all(gates.values()) and trade_plan.get("eligible_for_paper_trade", False)
        failed_gates = [name for name, passed in gates.items() if not passed]
        quality_label = self._quality_label(
            action=action,
            all_gates_passed=all_gates_passed,
            confidence=confidence,
            risk_score=risk_score,
            has_path=bool(current_path),
        )
        coverage_level = self._coverage_level(quality_label)

        return {
            "name": "MarketVision AI Signal Bot",
            "mode": "research_and_paper_trading_only",
            "action": action,
            "stance": stance,
            "quality_label": quality_label,
            "coverage_level": coverage_level,
            "dominant_scenario": dominant,
            "expected_return": round(float(expected_return), 4) if expected_return is not None else None,
            "confidence": round(confidence, 1),
            "risk_score": risk_score,
            "gates": gates,
            "failed_gates": failed_gates,
            "all_gates_passed": all_gates_passed,
            "inputs_used": [
                "v12 final hybrid probabilities",
                "elite breakout proof policy",
                "pretrained/fallback sentiment layer",
                "Monte Carlo expected range",
                "risk score",
            ],
            "trade_plan": trade_plan,
            "reasons": reasons,
            "risk_controls": risk_controls,
            "disclaimer": DISCLAIMER,
        }

    def _quality_label(
        self,
        action: str,
        all_gates_passed: bool,
        confidence: float,
        risk_score: int,
        has_path: bool,
    ) -> str:
        if not has_path:
            return "insufficient_data"
        if risk_score >= 75 or action in {"avoid_high_risk", "avoid_or_hedge"}:
            return "avoid_high_risk"
        if all_gates_passed:
            return "high_confidence_trade_candidate"
        if confidence >= 60 and risk_score < 70:
            return "watchlist_candidate"
        return "prediction_only"

    def _coverage_level(self, quality_label: str) -> str:
        return {
            "high_confidence_trade_candidate": "promoted_signal",
            "watchlist_candidate": "qualified_prediction",
            "prediction_only": "universal_prediction",
            "avoid_high_risk": "risk_filtered_prediction",
            "insufficient_data": "limited_coverage",
        }.get(quality_label, "universal_prediction")

    def _paper_trade_plan(
        self,
        action: str,
        dominant: str,
        entry_price: float | None,
        expected_range: dict[str, Any],
        confidence: float,
        risk_score: int,
        volume_ratio: float = 0.0,
    ) -> dict[str, Any]:
        if entry_price is None:
            return {"available": False, "reason": "No forecast path available."}

        risk_budget = 100.0
        low = float(expected_range.get("low") or entry_price * 0.97)
        high = float(expected_range.get("high") or entry_price * 1.03)
        risk_cap_pct = 0.018 if risk_score < 45 else 0.012 if risk_score < 70 else 0.008

        if dominant == "bearish":
            stop_loss = round(entry_price * (1 + risk_cap_pct), 2)
            target_1 = round(max(low, entry_price * (1 - risk_cap_pct * 1.5)), 2)
            target_2 = round(max(low, entry_price * (1 - risk_cap_pct * 2.5)), 2)
            risk_per_share = max(stop_loss - entry_price, 0.01)
            reward_per_share = max(entry_price - target_1, 0.01)
            direction = "short_watch"
        else:
            stop_loss = round(max(low, entry_price * (1 - risk_cap_pct)), 2)
            target_1 = round(min(high, entry_price * (1 + risk_cap_pct * 1.5)), 2)
            target_2 = round(min(high, entry_price * (1 + risk_cap_pct * 2.5)), 2)
            risk_per_share = max(entry_price - stop_loss, 0.01)
            reward_per_share = max(target_1 - entry_price, 0.01)
            direction = "long_watch"

        shares = max(int(risk_budget // risk_per_share), 0)
        notional = round(shares * entry_price, 2)
        risk_reward = reward_per_share / risk_per_share if risk_per_share else 0.0
        spread_estimate_pct = 0.0008 if entry_price >= 50 else 0.002
        slippage_estimate_pct = 0.0015 if risk_score < 60 else 0.003
        round_trip_cost_pct = (spread_estimate_pct + slippage_estimate_pct) * 2
        net_reward_pct = (reward_per_share / entry_price) - round_trip_cost_pct if entry_price else 0
        liquidity_pass = volume_ratio >= 0.35
        eligible = (
            action == "paper_trade_candidate"
            and risk_reward >= 1.2
            and confidence >= 65
            and risk_score < 70
            and net_reward_pct > 0
            and liquidity_pass
        )

        return {
            "available": True,
            "eligible_for_paper_trade": eligible,
            "direction": direction,
            "entry_type": "next_candle_research_entry",
            "entry_price": round(entry_price, 2),
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "risk_per_share": round(risk_per_share, 2),
            "reward_per_share_target_1": round(reward_per_share, 2),
            "risk_reward_target_1": round(float(risk_reward), 2),
            "spread_estimate_pct": round(spread_estimate_pct * 100, 4),
            "slippage_estimate_pct": round(slippage_estimate_pct * 100, 4),
            "round_trip_cost_pct": round(round_trip_cost_pct * 100, 4),
            "net_reward_after_cost_pct": round(net_reward_pct * 100, 4),
            "liquidity_filter_passed": liquidity_pass,
            "paper_risk_budget": risk_budget,
            "paper_position_shares": shares,
            "paper_notional": notional,
            "time_stop": "exit/re-score after forecast horizon or if signal state changes",
            "rules": [
                "Only paper trade when eligible_for_paper_trade is true.",
                "Cancel the ticket if live price gaps beyond entry or stop before fill.",
                "Re-score the setup before any new candle interval.",
            ],
        }

    def _reasoning_text(
        self,
        ticker: str,
        dominant: str,
        confidence: float,
        risk_score: int,
        intelligence: dict[str, Any],
    ) -> str:
        final_signal = intelligence.get("final_signal") or {}
        similarity = intelligence.get("historical_similarity") or {}
        timeframe = intelligence.get("multi_timeframe_alignment") or {}
        sentiment = intelligence.get("sentiment") or {}
        proof = (final_signal.get("proof") or {}).get("confidence_buckets") or {}
        bucket = proof.get(">=80%") or {}
        proof_text = ""
        if bucket:
            proof_text = (
                f" The promoted elite breakout proof reached {bucket.get('accuracy', 0) * 100:.1f}% "
                f"walk-forward accuracy at the >=80% confidence bucket on {bucket.get('signals')} setup-filtered signals."
            )
        sim_text = ""
        if similarity.get("available"):
            sim_text = (
                f" Historical similarity shows {similarity.get('average_future_return_pct')}% average future return "
                f"across {len(similarity.get('similar_setups', []))} nearest setups."
            )
        return (
            f"{ticker} currently has a {dominant} dominant probability scenario with {confidence:.1f}% engine confidence "
            f"and {risk_score}/100 risk. Multi-timeframe alignment is {timeframe.get('alignment')}."
            f"{proof_text}{sim_text} Sentiment layer is {sentiment.get('label', 'neutral')} "
            f"with score {sentiment.get('score', 0)} from {sentiment.get('model', 'unknown')}. "
            "Outputs are probability-based market simulations and AI-generated financial intelligence, not financial advice."
        )

    def _risk_facts(self, intelligence: dict[str, Any]) -> list[str]:
        facts: list[str] = []
        risk_score = intelligence.get("risk_score")
        regime = intelligence.get("market_regime") or {}
        technical = intelligence.get("technical_analysis") or {}
        macro = intelligence.get("macro") or {}
        if risk_score is not None:
            facts.append(f"Risk score is {risk_score}/100.")
        if regime.get("volatility_regime"):
            facts.append(f"Volatility regime is {regime.get('volatility_regime')}.")
        if technical.get("atr_pct") is not None:
            facts.append(f"ATR risk is {technical.get('atr_pct')} of price.")
        if not macro.get("available"):
            facts.append("Macro data is partial or unavailable for this request.")
        return facts or ["Market uncertainty still applies even when no major risk flag is active."]

    def _risk_score(self, technical: dict[str, Any], regime: dict[str, Any], macro: dict[str, Any]) -> int:
        score = 35
        atr_pct = technical.get("atr_pct") or 0
        volume_ratio = technical.get("volume_ratio_20") or 1
        if atr_pct > 0.035:
            score += 18
        elif atr_pct > 0.02:
            score += 10
        if volume_ratio > 2:
            score += 8
        if regime.get("volatility_regime") == "high_volatility":
            score += 18
        if technical.get("rsi_14") and (technical["rsi_14"] > 75 or technical["rsi_14"] < 25):
            score += 8
        if not macro.get("available"):
            score += 5
        return max(0, min(100, int(score)))

    def _confidence_score(
        self,
        probabilities: dict[str, float],
        timeframe: dict[str, Any],
        regime: dict[str, Any],
        similarity: dict[str, Any],
    ) -> float:
        ordered = sorted(probabilities.values(), reverse=True)
        margin = ordered[0] - ordered[1]
        confidence = 45 + margin * 85
        if abs(timeframe.get("alignment_score") or 0) > 0.6:
            confidence += 8
        if similarity.get("available") and (similarity.get("average_similarity_score") or 0) > 0.7:
            confidence += 5
        if regime.get("volatility_regime") == "high_volatility":
            confidence -= 7
        return round(float(max(1, min(95, confidence))), 1)

    def _monte_carlo_paths(
        self,
        history: pd.DataFrame,
        probabilities: dict[str, float],
        confidence: float,
        horizon_steps: int,
        n_paths: int = 500,
    ) -> dict[str, Any]:
        close = history["Close"]
        current = float(close.iloc[-1])
        returns = close.pct_change().dropna()
        mu = float(returns.tail(60).mean()) if not returns.empty else 0.0
        sigma = float(returns.tail(60).std()) if not returns.empty else 0.015
        directional_bias = probabilities["bullish"] - probabilities["bearish"]
        adjusted_mu = mu + directional_bias * sigma * 0.25
        rng = np.random.default_rng(42)
        shocks = rng.normal(adjusted_mu, sigma, size=(n_paths, horizon_steps))
        paths = current * np.cumprod(1 + shocks, axis=1)

        median = np.median(paths, axis=0)
        lower = np.percentile(paths, 15, axis=0)
        upper = np.percentile(paths, 85, axis=0)
        bullish = np.percentile(paths, 80, axis=0)
        bearish = np.percentile(paths, 20, axis=0)
        sideways = np.percentile(paths, 50, axis=0)
        volatility_cone = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

        return {
            "method": "Monte Carlo using recent return distribution, model probability bias, and volatility estimate",
            "n_paths": n_paths,
            "horizon_steps": horizon_steps,
            "main_predicted_line": [round(float(x), 2) for x in median],
            "bullish_path": [round(float(x), 2) for x in bullish],
            "bearish_path": [round(float(x), 2) for x in bearish],
            "sideways_path": [round(float(x), 2) for x in sideways],
            "confidence_band": {
                "lower": [round(float(x), 2) for x in lower],
                "upper": [round(float(x), 2) for x in upper],
            },
            "volatility_cone": {
                "p05": [round(float(x), 2) for x in volatility_cone[0]],
                "p25": [round(float(x), 2) for x in volatility_cone[1]],
                "p50": [round(float(x), 2) for x in volatility_cone[2]],
                "p75": [round(float(x), 2) for x in volatility_cone[3]],
                "p95": [round(float(x), 2) for x in volatility_cone[4]],
            },
            "expected_range": {
                "low": round(float(lower[-1]), 2),
                "high": round(float(upper[-1]), 2),
            },
            "confidence_used": confidence,
        }

    def _explanation_facts(
        self,
        ticker: str,
        probabilities: dict[str, float],
        confidence: float,
        risk_score: int,
        technical: dict[str, Any],
        timeframe: dict[str, Any],
        regime: dict[str, Any],
        sector: dict[str, Any],
        similarity: dict[str, Any],
    ) -> dict[str, Any]:
        dominant = max(probabilities, key=probabilities.get)
        facts = [
            f"{ticker} dominant scenario is {dominant} with {confidence}% confidence.",
            f"Price is {'above' if technical.get('above_vwap') else 'below or near'} VWAP; VWAP distance is {technical.get('vwap_distance_pct')}%.",
            f"Multi-timeframe alignment is {timeframe.get('alignment')} with score {timeframe.get('alignment_score')}.",
            f"Market regime is {regime.get('regime')} and volatility regime is {regime.get('volatility_regime')}.",
            f"Risk score is {risk_score}/100.",
        ]
        if sector.get("available"):
            facts.append(f"Sector leader is {sector.get('leader')}; stock vs leading sector 20d is {sector.get('stock_vs_leading_sector_20d')}%.")
        if similarity.get("available"):
            facts.append(
                f"Historical similarity average future return was {similarity.get('average_future_return_pct')}% "
                f"across {len(similarity.get('similar_setups', []))} matched setups."
            )
        return {
            "dominant_scenario": dominant,
            "facts": facts,
            "required_disclaimer": DISCLAIMER,
        }

    def _coverage(self, include_intraday: bool) -> DataCoverage:
        return DataCoverage(
            trained=[
                "technical indicators",
                "VWAP features when intraday candles are available",
                "multi-timeframe trend alignment",
                "market regime from SPY/QQQ/VIX when available",
                "sector ETF rotation",
                "historical similarity",
                "Monte Carlo simulation",
                "risk score",
                "LLM explanation-ready facts",
                "honest probability framing",
            ],
            partial=[
                "event flags without complete earnings/Fed/CPI calendar",
                "macro if FRED is configured",
                "FinBERT sentiment when transformers/torch are installed and news is available",
                "breakout/reversal/volatility expansion proxy labels",
                "options expiration calendar approximation",
            ],
            missing=[
                "options flow",
                "Reddit/Twitter sentiment",
                "analyst upgrade/downgrade feeds",
                "SEC filing embeddings",
                "earnings tone model",
                "LSTM/GRU/Transformer/TFT training",
                "adaptive learning feedback loop",
            ],
        )

    def _risk_level(self, risk_score: int) -> str:
        if risk_score >= 70:
            return "high"
        if risk_score >= 45:
            return "medium"
        return "low"
