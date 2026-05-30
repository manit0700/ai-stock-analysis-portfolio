from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import HTTPException


SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "AVGO": "Technology",
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "LLY": "Health Care",
    "UNH": "Health Care",
    "JNJ": "Health Care",
}


def _validate_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    if price_frame.empty:
        raise HTTPException(status_code=404, detail="No price history available for the requested portfolio.")
    if isinstance(price_frame.columns, pd.MultiIndex):
        price_frame = price_frame["Close"]
    return price_frame.dropna(how="all")


def analyze_portfolio_from_prices(
    price_frame: pd.DataFrame,
    positions: dict[str, float],
    average_costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    close_prices = _validate_price_frame(price_frame).ffill().dropna()
    if close_prices.empty:
        raise HTTPException(status_code=404, detail="Portfolio price history is incomplete.")

    latest_prices = close_prices.iloc[-1]
    latest_values = {ticker: round(float(latest_prices[ticker]) * shares, 2) for ticker, shares in positions.items()}
    total_value = float(sum(latest_values.values()))
    if total_value <= 0:
        raise HTTPException(status_code=400, detail="Portfolio value must be greater than zero.")

    weights = {ticker: value / total_value for ticker, value in latest_values.items()}
    returns = close_prices.pct_change().dropna()
    if returns.empty:
        raise HTTPException(status_code=404, detail="Not enough price history to calculate portfolio returns.")

    ordered_tickers = list(positions.keys())
    weight_vector = np.array([weights[ticker] for ticker in ordered_tickers])
    portfolio_returns = returns[ordered_tickers].mul(weight_vector, axis=1).sum(axis=1)

    annual_return = float(portfolio_returns.mean() * 252)
    annual_volatility = float(portfolio_returns.std() * np.sqrt(252))
    sharpe_ratio = (annual_return / annual_volatility) if annual_volatility else 0.0
    max_drawdown = float((1 + portfolio_returns).cumprod().div((1 + portfolio_returns).cumprod().cummax()).sub(1).min())
    herfindahl_index = float(sum(weight**2 for weight in weights.values()))
    diversification_score = max(0, round((1 - herfindahl_index) * 100, 2))
    largest_position = max(weights.values())

    warnings: list[str] = []
    if largest_position > 0.4:
        warnings.append("One holding exceeds 40% of the portfolio, which creates concentration risk.")
    if annual_volatility > 0.35:
        warnings.append("Estimated volatility is high for a long-only retail portfolio.")
    if len(ordered_tickers) < 3:
        warnings.append("The portfolio has fewer than three holdings, which limits diversification.")
    if not warnings:
        warnings.append("No major portfolio-level risk flags were triggered by the current allocation.")

    holdings = []
    for ticker in ordered_tickers:
        avg_cost = (average_costs or {}).get(ticker)
        unrealized_pnl = ((float(latest_prices[ticker]) / avg_cost - 1) * 100) if avg_cost else None
        holdings.append(
            {
                "ticker": ticker,
                "shares": positions[ticker],
                "average_cost": round(avg_cost, 2) if avg_cost else None,
                "latest_price": round(float(latest_prices[ticker]), 2),
                "market_value": latest_values[ticker],
                "weight_pct": round(weights[ticker] * 100, 2),
                "unrealized_pnl_pct": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
                "sector": SECTOR_MAP.get(ticker, "Unknown"),
            }
        )

    correlation_matrix = returns[ordered_tickers].corr().round(3).fillna(0.0)
    average_correlation = 0.0
    if len(ordered_tickers) > 1:
        corr_values = correlation_matrix.where(~np.eye(len(correlation_matrix), dtype=bool)).stack()
        average_correlation = float(corr_values.mean()) if not corr_values.empty else 0.0

    sector_exposure: dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = SECTOR_MAP.get(ticker, "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight * 100
    sector_exposure = {sector: round(value, 2) for sector, value in sorted(sector_exposure.items(), key=lambda item: item[1], reverse=True)}

    concentration_risk = min(100, round(largest_position * 160, 1))
    volatility_risk = min(100, round(annual_volatility / 0.45 * 100, 1)) if annual_volatility else 0.0
    correlation_risk = min(100, round(max(average_correlation, 0) * 100, 1))
    sector_concentration = max(sector_exposure.values()) if sector_exposure else 0.0
    portfolio_risk_score = round((concentration_risk * 0.35) + (volatility_risk * 0.3) + (correlation_risk * 0.2) + (sector_concentration * 0.15), 1)
    suggested_watchlist_alerts = []
    for holding in holdings:
        if holding["weight_pct"] >= 30:
            suggested_watchlist_alerts.append(f"{holding['ticker']}: position exceeds 30% portfolio weight.")
        if holding.get("unrealized_pnl_pct") is not None and holding["unrealized_pnl_pct"] <= -10:
            suggested_watchlist_alerts.append(f"{holding['ticker']}: unrealized loss exceeds 10%.")
    if sector_concentration >= 50:
        suggested_watchlist_alerts.append(f"{next(iter(sector_exposure))}: sector exposure exceeds 50%.")

    ai_risk_summary = (
        f"Portfolio risk score is {portfolio_risk_score}/100. "
        f"Largest position is {largest_position * 100:.1f}%, average correlation is {average_correlation:.2f}, "
        f"and top sector exposure is {sector_concentration:.1f}%."
    )

    return {
        "total_market_value": round(total_value, 2),
        "annualized_return_pct": round(annual_return * 100, 2),
        "annualized_volatility_pct": round(annual_volatility * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "diversification_score": diversification_score,
        "holdings": holdings,
        "sector_exposure": sector_exposure,
        "correlation_risk": round(correlation_risk, 1),
        "volatility_risk": round(volatility_risk, 1),
        "concentration_risk": round(concentration_risk, 1),
        "portfolio_risk_score": portfolio_risk_score,
        "ai_risk_summary": ai_risk_summary,
        "suggested_watchlist_alerts": suggested_watchlist_alerts,
        "correlation_matrix": correlation_matrix.to_dict(),
        "warnings": warnings,
    }


class PortfolioService:
    def analyze_portfolio(self, holdings: list[dict[str, float]], period: str = "6mo") -> dict[str, Any]:
        positions = {item["ticker"].upper(): item["shares"] for item in holdings}
        average_costs = {
            item["ticker"].upper(): float(item["average_cost"])
            for item in holdings
            if item.get("average_cost") is not None
        }
        tickers = list(positions.keys())
        if not tickers:
            raise HTTPException(status_code=400, detail="At least one holding is required.")

        price_frame = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if isinstance(price_frame.columns, pd.MultiIndex):
            close_prices = price_frame["Close"]
        else:
            close_prices = price_frame.rename(columns={"Close": tickers[0]})[tickers]

        return analyze_portfolio_from_prices(close_prices, positions, average_costs=average_costs)
