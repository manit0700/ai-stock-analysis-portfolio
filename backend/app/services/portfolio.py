from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import HTTPException


def _validate_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    if price_frame.empty:
        raise HTTPException(status_code=404, detail="No price history available for the requested portfolio.")
    if isinstance(price_frame.columns, pd.MultiIndex):
        price_frame = price_frame["Close"]
    return price_frame.dropna(how="all")


def analyze_portfolio_from_prices(price_frame: pd.DataFrame, positions: dict[str, float]) -> dict[str, Any]:
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
        holdings.append(
            {
                "ticker": ticker,
                "shares": positions[ticker],
                "latest_price": round(float(latest_prices[ticker]), 2),
                "market_value": latest_values[ticker],
                "weight_pct": round(weights[ticker] * 100, 2),
            }
        )

    correlation_matrix = returns[ordered_tickers].corr().round(3).fillna(0.0)

    return {
        "total_market_value": round(total_value, 2),
        "annualized_return_pct": round(annual_return * 100, 2),
        "annualized_volatility_pct": round(annual_volatility * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "diversification_score": diversification_score,
        "holdings": holdings,
        "correlation_matrix": correlation_matrix.to_dict(),
        "warnings": warnings,
    }


class PortfolioService:
    def analyze_portfolio(self, holdings: list[dict[str, float]], period: str = "6mo") -> dict[str, Any]:
        positions = {item["ticker"].upper(): item["shares"] for item in holdings}
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

        return analyze_portfolio_from_prices(close_prices, positions)
