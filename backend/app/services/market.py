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
        """Bars from the unified market-data layer: one cache, retries, rate limits and quality checks
        for every legacy service (the old code called yfinance directly from five modules)."""
        from app.marketdata.service import get_market_data

        yf_period, yf_interval, resample_4h = self._normalize_interval_request(period, interval)
        history = get_market_data().history(ticker, period=yf_period, interval=yf_interval).data
        history = history.dropna(subset=["Close"])
        if history.empty:
            raise HTTPException(status_code=404, detail=f"No price history found for {ticker.upper()}")
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
        """Headlines through the unified layer (Finnhub, then NewsAPI, then Yahoo), deduplicated and cached."""
        from app.core.errors import MarketVisionError
        from app.marketdata.service import get_market_data

        try:
            return get_market_data().news(symbol=ticker, limit=limit).data
        except MarketVisionError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.message) from None

    def get_market_overview(self) -> dict[str, Any]:
        """Overview cards. Each card names the instrument it actually shows.

        SPY, QQQ and DIA are exchange-traded funds that track the S&P 500, Nasdaq-100 and Dow
        Jones; their prices are not index levels, so the card says which fund it is. "Up/Down
        today" describes one session's move, nothing more. Data comes through the unified data
        layer (cached, with provenance); a card whose data is unavailable is left out, not filled.
        """
        from app.core.errors import MarketVisionError
        from app.marketdata.service import get_market_data

        MARKET_SYMBOLS = {
            "S&P 500 ETF (SPY)": "SPY",
            "Nasdaq-100 ETF (QQQ)": "QQQ",
            "Dow Jones ETF (DIA)": "DIA",
            "Bitcoin (BTC-USD)": "BTC-USD",
            "VIX": "^VIX",
        }
        SECTOR_ETFS = {
            "Tech": "XLK", "Financials": "XLF", "Energy": "XLE",
            "Healthcare": "XLV", "Industrials": "XLI", "Consumer": "XLY",
        }
        md = get_market_data()

        def closes(sym: str) -> tuple[pd.Series, Any, str] | None:
            try:
                envelope = md.history(sym, period="10d", interval="1d")
            except MarketVisionError:
                return None
            series = envelope.data["Close"].dropna() if "Close" in envelope.data else pd.Series(dtype=float)
            if len(series) < 2:
                return None
            return series, envelope.market_timestamp, envelope.provider

        def day_move(change_pct: float) -> str:
            return "Up today" if change_pct > 0.3 else ("Down today" if change_pct < -0.3 else "Flat today")

        def move_size(change_pct: float) -> str:
            return "Large move" if abs(change_pct) > 1.5 else ("Medium move" if abs(change_pct) > 0.5 else "Small move")

        cards = []
        for label, sym in MARKET_SYMBOLS.items():
            got = closes(sym)
            if got is None:
                continue
            series, market_ts, provider = got
            price, prev = float(series.iloc[-1]), float(series.iloc[-2])
            change_pct = round((price / prev - 1) * 100, 2) if prev else 0.0
            if sym == "^VIX":
                trend = "Low fear" if price < 20 else ("High fear" if price > 30 else "Elevated")
                value = f"{price:.2f}"
            else:
                trend = day_move(change_pct)
                value = f"${price:,.0f}" if sym == "BTC-USD" else f"${price:,.2f}"
            cards.append({
                "name": label,
                "symbol": sym,
                "value": value,
                "change": f"+{change_pct}%" if change_pct >= 0 else f"{change_pct}%",
                "trend": trend,
                "volatility": move_size(change_pct),
                "points": [round(float(v), 2) for v in series.tail(8).tolist()],
                "as_of": series.index[-1].date().isoformat(),
                "market_timestamp": market_ts.isoformat() if market_ts else None,
                "source": provider,
            })

        best: tuple[str, str, float, str, str] | None = None
        for sector, sym in SECTOR_ETFS.items():
            got = closes(sym)
            if got is None:
                continue
            series, _, provider = got
            chg = float((series.iloc[-1] / series.iloc[-2] - 1) * 100)
            if best is None or chg > best[2]:
                best = (sector, sym, chg, series.index[-1].date().isoformat(), provider)
        if best:
            sector, sym, chg, as_of, provider = best
            cards.append({
                "name": "Top Sector Today",
                "symbol": sym,
                "value": sector,
                "change": f"+{round(chg, 2)}%" if chg >= 0 else f"{round(chg, 2)}%",
                "trend": f"Best of 6 sector ETFs ({sym})",
                "volatility": move_size(chg),
                "points": [],
                "as_of": as_of,
                "source": provider,
            })

        return {
            "cards": cards,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "note": "SPY, QQQ and DIA are ETFs tracking the indices; their prices are not index levels. Moves are one session.",
        }

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
