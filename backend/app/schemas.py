from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class PricePoint(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class HoldingInput(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    shares: float = Field(gt=0)


class TickerListRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=25)
    period: str = Field(default="6mo", min_length=2, max_length=10)
    interval: str = Field(default="1d", min_length=2, max_length=10)


class PortfolioAnalysisRequest(BaseModel):
    holdings: list[HoldingInput]
    period: str = Field(default="6mo", min_length=2, max_length=10)


class PredictionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    period: str = Field(default="1y", min_length=2, max_length=10)
    interval: str = Field(default="1d", min_length=2, max_length=10)
    horizon_steps: int = Field(default=12, ge=4, le=60)


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    link: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
