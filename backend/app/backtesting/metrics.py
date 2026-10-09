"""Performance measures for a backtest, each stated only where it means something.

Annualised return and the Sharpe ratio are reported only when the test covers at least a year of
bars; on a few weeks of data they are noise with a decimal point. Everything is after costs,
because the equity curve is after costs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.backtesting.engine import BacktestResult

TRADING_DAYS = 252
MIN_BARS_FOR_ANNUAL = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min()) if len(equity) else 0.0


def summarise(result: BacktestResult) -> dict[str, Any]:
    equity = result.equity
    initial = float(result.config.initial_cash)
    n = len(equity)
    years = n / TRADING_DAYS
    final = float(equity.iloc[-1])
    total_return = final / initial - 1.0
    daily = equity.pct_change().dropna()
    enough = n >= MIN_BARS_FOR_ANNUAL
    volatility = float(daily.std() * np.sqrt(TRADING_DAYS)) if len(daily) > 20 else None
    sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if enough and daily.std() > 0 else None
    annualised = float((final / initial) ** (1 / years) - 1.0) if enough and final > 0 else None

    closed = [t for t in result.trades if t["outcome"] != "open"]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    gross_profit = float(sum(t["pnl"] for t in wins))
    gross_loss = float(-sum(t["pnl"] for t in losses))
    mean_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else None
    mean_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else None
    reward_risk = round(mean_win / abs(mean_loss), 4) if mean_win is not None and mean_loss is not None and mean_loss < 0 else None
    fills_value = sum(float(e.gross) for e in result.book.entries if e.kind in {"buy", "sell"})
    mean_equity = float(equity.mean()) if n else initial
    fees = float(result.book.fees)
    price_cost = float(result.price_cost)
    return {
        "bars": n,
        "years": round(years, 2),
        "initial_capital": initial,
        "final_equity": round(final, 2),
        "total_return_pct": round(total_return * 100, 4),
        "annualized_return_pct": None if annualised is None else round(annualised * 100, 4),
        "annualized_volatility_pct": None if volatility is None else round(volatility * 100, 4),
        "sharpe_ratio": None if sharpe is None else round(sharpe, 4),
        "sharpe_note": "risk-free rate taken as zero; reported only for tests of a year or longer",
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 4),
        "total_trades": len(closed),
        "open_trades": len(result.trades) - len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 2) if closed else None,
        "avg_win_pct": round(float(np.mean([t["pnl_pct"] for t in wins])), 4) if wins else None,
        "avg_loss_pct": round(float(np.mean([t["pnl_pct"] for t in losses])), 4) if losses else None,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "average_reward_risk": reward_risk,
        "turnover_per_year": round(fills_value / mean_equity / years, 4) if years > 0 and mean_equity > 0 else None,
        "exposure_pct": round(float((result.invested > 0).mean() * 100), 2),
        "average_invested_pct": round(float((result.invested / equity.where(equity > 0)).mean() * 100), 2),
        "costs": {
            "commission_paid": round(fees, 2),
            "spread_and_slippage": round(price_cost, 2),
            "total": round(fees + price_cost, 2),
            "share_of_initial_capital_pct": round((fees + price_cost) / initial * 100, 4),
        },
        "dividends_received": round(float(result.book.dividends), 2),
        "realized_pnl": round(float(result.book.realized), 2),
        "counters": result.counters,
    }


def monthly_equity(equity: pd.Series) -> list[dict[str, Any]]:
    if len(equity) == 0:
        return []
    sampled = equity.groupby([equity.index.year, equity.index.month]).tail(1)
    return [{"date": ix.date().isoformat(), "equity": round(float(value), 2)} for ix, value in sampled.items()]
