from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import HTTPException


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_rsi(close: pd.Series, window: int = 14) -> float | None:
    if len(close) < window + 1:
        return None

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    latest_loss = avg_loss.iloc[-1]
    if latest_loss == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / latest_loss
    return round(float(100 - (100 / (1 + rs))), 2)


def _annualized_volatility(close: pd.Series, lookback: int = 30) -> float | None:
    if len(close) < 2:
        return None
    returns = close.pct_change().dropna().tail(lookback)
    if returns.empty:
        return None
    return round(float(returns.std() * np.sqrt(252)), 4)


def _max_drawdown(close: pd.Series) -> float | None:
    if close.empty:
        return None
    running_peak = close.cummax()
    drawdowns = close / running_peak - 1
    return round(float(drawdowns.min()), 4)


def normalize_history(history: pd.DataFrame, limit: int = 90) -> list[dict[str, Any]]:
    trimmed = history.tail(limit).copy()
    trimmed.index = pd.to_datetime(trimmed.index)

    points: list[dict[str, Any]] = []
    for index, row in trimmed.iterrows():
        points.append(
            {
                "date": index.isoformat() if (index.hour or index.minute or index.second) else index.date().isoformat(),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"] or 0),
            }
        )
    return points


class MarketDataService:
    def _normalize_interval_request(self, period: str, interval: str) -> tuple[str, str, bool]:
        normalized = interval.lower()
        if normalized == "4h":
            return period, "1h", True
        if normalized == "1m" and period not in {"1d", "5d", "7d"}:
            return "5d", "1m", False
        if normalized in {"5m", "15m", "30m"} and period in {"6mo", "1y", "2y", "5y"}:
            return "1mo", normalized, False
        return period, normalized, False

    def get_history(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        yf_period, yf_interval, resample_4h = self._normalize_interval_request(period, interval)
        history = yf.Ticker(ticker.upper()).history(period=yf_period, interval=yf_interval, auto_adjust=False)
        if history.empty:
            raise HTTPException(status_code=404, detail=f"No price history found for {ticker.upper()}")
        history = history.dropna(subset=["Close"])
        if resample_4h:
            history = history.resample("4h").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna(subset=["Open", "High", "Low", "Close"])
        return history

    def _get_company_info(self, ticker: str) -> dict[str, Any]:
        stock = yf.Ticker(ticker.upper())
        info = {}
        fast_info = {}

        try:
            info = stock.info or {}
        except Exception:
            info = {}

        try:
            fast_info = dict(stock.fast_info or {})
        except Exception:
            fast_info = {}

        return {"info": info, "fast_info": fast_info}

    def build_stock_overview(self, ticker: str, period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
        normalized_ticker = ticker.upper()
        history = self.get_history(normalized_ticker, period=period, interval=interval)
        close = history["Close"]
        latest_close = float(close.iloc[-1])
        previous_close = float(close.iloc[-2]) if len(close) > 1 else latest_close
        daily_change_pct = ((latest_close / previous_close) - 1) if previous_close else 0.0

        info_bundle = self._get_company_info(normalized_ticker)
        info = info_bundle["info"]
        fast_info = info_bundle["fast_info"]

        sma_20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else None
        sma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else None

        return {
            "ticker": normalized_ticker,
            "company_name": info.get("longName") or info.get("shortName") or normalized_ticker,
            "currency": info.get("currency") or fast_info.get("currency") or "USD",
            "exchange": info.get("exchange") or fast_info.get("exchange"),
            "current_price": round(latest_close, 2),
            "previous_close": round(previous_close, 2),
            "daily_change_pct": round(daily_change_pct * 100, 2),
            "market_cap": _clean_float(info.get("marketCap") or fast_info.get("market_cap")),
            "trailing_pe": _clean_float(info.get("trailingPE")),
            "forward_pe": _clean_float(info.get("forwardPE")),
            "dividend_yield": _clean_float(info.get("dividendYield")),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "business_summary": info.get("longBusinessSummary"),
            "fifty_two_week_high": _clean_float(info.get("fiftyTwoWeekHigh") or fast_info.get("year_high")),
            "fifty_two_week_low": _clean_float(info.get("fiftyTwoWeekLow") or fast_info.get("year_low")),
            "indicators": {
                "sma_20": round(float(sma_20), 2) if sma_20 is not None and not pd.isna(sma_20) else None,
                "sma_50": round(float(sma_50), 2) if sma_50 is not None and not pd.isna(sma_50) else None,
                "rsi_14": _compute_rsi(close),
                "annualized_volatility_30d": _annualized_volatility(close),
                "max_drawdown_period": _max_drawdown(close),
            },
            "price_history": normalize_history(history),
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def build_stock_analysis(self, ticker: str, period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
        overview = self.build_stock_overview(ticker, period=period, interval=interval)
        indicators = overview["indicators"]
        current_price = overview["current_price"]
        sma_20 = indicators["sma_20"]
        sma_50 = indicators["sma_50"]
        rsi_14 = indicators["rsi_14"]
        volatility = indicators["annualized_volatility_30d"]

        score = 50
        reasons: list[str] = []
        risks: list[str] = []

        if sma_20 is not None and current_price > sma_20:
            score += 10
            reasons.append("Price is trading above the 20-day moving average.")
        elif sma_20 is not None:
            score -= 10
            risks.append("Price is trading below the 20-day moving average.")

        if sma_20 is not None and sma_50 is not None and sma_20 > sma_50:
            score += 10
            reasons.append("Short-term trend is stronger than the medium-term trend.")
        elif sma_20 is not None and sma_50 is not None:
            score -= 10
            risks.append("Short-term trend is weaker than the medium-term trend.")

        if rsi_14 is not None and rsi_14 < 35:
            score += 5
            reasons.append("Momentum is near oversold levels, which can create rebound setups.")
        elif rsi_14 is not None and rsi_14 > 70:
            score -= 8
            risks.append("Momentum is overbought, which raises pullback risk.")
        elif rsi_14 is not None:
            reasons.append("Momentum is in a relatively balanced range.")

        if volatility is not None and volatility > 0.45:
            score -= 10
            risks.append("Recent volatility is elevated, so position sizing matters.")
        elif volatility is not None and volatility < 0.25:
            score += 5
            reasons.append("Recent volatility is moderate compared with many high-beta stocks.")

        score = max(0, min(100, score))
        if score >= 65:
            stance = "bullish"
        elif score <= 40:
            stance = "cautious"
        else:
            stance = "watch"

        if not risks:
            risks.append("No major technical risk flags were triggered in the selected period.")

        return {
            "ticker": overview["ticker"],
            "company_name": overview["company_name"],
            "stance": stance,
            "score": score,
            "current_price": current_price,
            "daily_change_pct": overview["daily_change_pct"],
            "signals": indicators,
            "reasons": reasons,
            "risks": risks,
            "disclaimer": "This analysis is informational and should not be treated as personalized investment advice.",
            "as_of": overview["as_of"],
        }

    def compare_stocks(self, tickers: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
        seen: set[str] = set()
        normalized = []
        for ticker in tickers:
            ticker_upper = ticker.upper().strip()
            if ticker_upper and ticker_upper not in seen:
                normalized.append(ticker_upper)
                seen.add(ticker_upper)

        if not normalized:
            raise HTTPException(status_code=400, detail="At least one valid ticker is required.")

        analyses = [self.build_stock_analysis(ticker, period=period, interval=interval) for ticker in normalized]
        ranked = sorted(analyses, key=lambda item: item["score"], reverse=True)

        return {
            "period": period,
            "interval": interval,
            "leaders": ranked,
            "summary": {
                "best_score": ranked[0]["score"],
                "best_ticker": ranked[0]["ticker"],
                "most_cautious_ticker": ranked[-1]["ticker"],
            },
        }

    def get_news(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
        normalized_ticker = ticker.upper()
        stock = yf.Ticker(normalized_ticker)
        try:
            raw_news = stock.news or []
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Unable to fetch news for {normalized_ticker}: {exc}") from exc

        news_items: list[dict[str, Any]] = []
        for item in raw_news[:limit]:
            provider = item.get("publisher")
            link = item.get("link") or item.get("canonicalUrl", {}).get("url")
            published_at = item.get("providerPublishTime")
            summary = item.get("summary")
            if published_at:
                published_dt = datetime.fromtimestamp(published_at, tz=timezone.utc).isoformat()
            else:
                published_dt = None
            news_items.append(
                {
                    "title": item.get("title", ""),
                    "publisher": provider,
                    "link": link,
                    "published_at": published_dt,
                    "summary": summary,
                }
            )

        return news_items

    def get_market_overview(self) -> dict[str, Any]:
        MARKET_SYMBOLS = {
            "S&P 500": "SPY",
            "Nasdaq": "QQQ",
            "Dow Jones": "DIA",
            "Bitcoin": "BTC-USD",
            "VIX": "^VIX",
        }
        SECTOR_ETFS = {
            "Tech": "XLK", "Financials": "XLF", "Energy": "XLE",
            "Healthcare": "XLV", "Industrials": "XLI", "Consumer": "XLY",
        }

        cards = []
        for label, sym in MARKET_SYMBOLS.items():
            try:
                hist = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=False)
                hist = hist.dropna(subset=["Close"])
                if hist.empty:
                    continue
                price = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                change_pct = round((price / prev - 1) * 100, 2) if prev else 0.0
                points = [round(float(p), 2) for p in hist["Close"].tail(8).tolist()]
                trend = "Bullish" if change_pct > 0.3 else ("Bearish" if change_pct < -0.3 else "Neutral")
                if sym == "^VIX":
                    trend = "Risk-On" if price < 20 else ("Fear" if price > 30 else "Caution")
                if sym == "BTC-USD":
                    label_display = f"${price:,.0f}"
                elif sym == "^VIX":
                    label_display = f"{price:.2f}"
                else:
                    label_display = f"{price:,.2f}"
                cards.append({
                    "name": label,
                    "value": label_display,
                    "change": f"+{change_pct}%" if change_pct >= 0 else f"{change_pct}%",
                    "trend": trend,
                    "volatility": "High" if abs(change_pct) > 1.5 else ("Medium" if abs(change_pct) > 0.5 else "Low"),
                    "points": points,
                })
            except Exception:
                continue

        # top performing sector
        best_sector = None
        best_change = -999.0
        for sector, sym in SECTOR_ETFS.items():
            try:
                hist = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=False).dropna(subset=["Close"])
                if len(hist) < 2:
                    continue
                chg = float((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100)
                if chg > best_change:
                    best_change = chg
                    best_sector = sector
            except Exception:
                continue

        if best_sector:
            cards.append({
                "name": "Top Sector",
                "value": best_sector,
                "change": f"+{round(best_change, 2)}%" if best_change >= 0 else f"{round(best_change, 2)}%",
                "trend": "Leader",
                "volatility": "Medium",
                "points": [],
            })

        return {"cards": cards, "as_of": datetime.now(timezone.utc).isoformat()}

    def build_watchlist(self, tickers: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
        comparison = self.compare_stocks(tickers=tickers, period=period, interval=interval)
        leaders = comparison["leaders"]

        watchlist = []
        for item in leaders:
            watchlist.append(
                {
                    "ticker": item["ticker"],
                    "company_name": item["company_name"],
                    "stance": item["stance"],
                    "score": item["score"],
                    "daily_change_pct": item["daily_change_pct"],
                    "headline_reason": item["reasons"][0] if item["reasons"] else None,
                    "top_risk": item["risks"][0] if item["risks"] else None,
                }
            )

        return {
            "watchlist": watchlist,
            "disclaimer": "This watchlist is an informational ranking and not a personalized recommendation.",
            "period": period,
            "interval": interval,
        }
