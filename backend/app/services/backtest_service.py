"""Backtests for the API and the interface, on the single accounting engine.

This module used to compute an equity curve from signal-times-return arithmetic and a separate
trade list with stops and targets, so the two could (and did) disagree, and it compared a 25%
position against a 100% buy-and-hold. It is now a thin adapter over ``app.backtesting``:

* one ``Book`` holds cash and the position, so trades and equity are the same thing seen twice;
* stops and targets are sales in that book;
* the buy-and-hold baseline uses the same capital, the same position size and the same costs,
  and a cash-only baseline is reported beside it.

Response keys used by the interface are unchanged. New keys are additions.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pandas as pd

from app.accounting.money import D
from app.backtesting.engine import BacktestConfig, BacktestResult, CostModel, buy_and_hold, run_backtest as _engine_run
from app.backtesting.metrics import monthly_equity, summarise
from app.backtesting.strategies import LABELS as STRATEGY_LABELS
from app.backtesting.strategies import STRATEGIES as _STRATEGIES
from app.backtesting.strategies import TYPES as STRATEGY_TYPES
from app.core.errors import MarketVisionError

log = logging.getLogger(__name__)

DISCLAIMER = "A simulation on historical prices with assumed costs. Past results do not predict future results. Not financial advice."
MIN_BARS = 60


def _config(initial_capital: float, commission: float, spread: float, slippage: float, max_position_pct: float, min_dollar_volume: float | None,
            stop_loss_pct: float | None, target_pct: float | None, symbol: str = "ASSET") -> BacktestConfig:
    costs = CostModel(
        commission_rate=D(commission), commission_minimum=Decimal("0"),
        half_spread_bps=D(spread) * 10_000, slippage_bps=D(slippage) * 10_000,
    )
    return BacktestConfig(
        initial_cash=D(initial_capital), position_fraction=D(max_position_pct),
        stop_loss_pct=None if stop_loss_pct is None else D(stop_loss_pct), take_profit_pct=None if target_pct is None else D(target_pct),
        costs=costs, fractional_shares=True, min_dollar_volume=min_dollar_volume, symbol=symbol,
    )


def _zero(value: float | None) -> float:
    return 0.0 if value is None else value


def _legacy_shape(result: BacktestResult, baseline: BacktestResult, fully_invested: BacktestResult, config: BacktestConfig) -> dict[str, Any]:
    stats, base_stats, full_stats = summarise(result), summarise(baseline), summarise(fully_invested)
    costs = config.costs.describe()
    closed = [t for t in result.trades if t["outcome"] != "open"]
    losses = [t for t in closed if t["outcome"] == "loss"]
    trades = [
        {
            "entry_date": t["entry_date"], "exit_date": t["exit_date"], "entry_price": round(t["entry_price"], 2), "exit_price": round(t["exit_price"], 2),
            "pnl_pct": round(t["pnl_pct"], 2), "gross_pnl_pct": round(t["gross_pnl_pct"], 2), "pnl": round(t["pnl"], 2), "quantity": t["quantity"],
            "estimated_round_trip_cost_pct": round(costs["round_trip_cost_pct"], 3), "exit_reason": t["exit_reason"], "bars_held": t["bars_held"], "outcome": t["outcome"],
        }
        for t in result.trades
    ]
    return {
        "total_return_pct": round(stats["total_return_pct"], 2),
        "buy_hold_return_pct": round(base_stats["total_return_pct"], 2),
        "annualized_return_pct": stats["annualized_return_pct"] if stats["annualized_return_pct"] is None else round(stats["annualized_return_pct"], 2),
        "annualized_volatility_pct": stats["annualized_volatility_pct"] if stats["annualized_volatility_pct"] is None else round(stats["annualized_volatility_pct"], 2),
        "sharpe_ratio": stats["sharpe_ratio"] if stats["sharpe_ratio"] is None else round(stats["sharpe_ratio"], 3),
        "max_drawdown_pct": round(stats["max_drawdown_pct"], 2),
        "total_trades": stats["total_trades"],
        "open_trades": stats["open_trades"],
        "win_rate_pct": _zero(stats["win_rate_pct"]),
        "avg_win_pct": round(_zero(stats["avg_win_pct"]), 2),
        "avg_loss_pct": round(_zero(stats["avg_loss_pct"]), 2),
        "profit_factor": stats["profit_factor"],
        "average_reward_risk": stats["average_reward_risk"],
        "false_positive_rate_pct": round(len(losses) / len(closed) * 100, 1) if closed else 0.0,
        "invalid_trades": result.counters["entries_skipped_illiquid"],
        "final_equity": stats["final_equity"],
        "initial_capital": stats["initial_capital"],
        "turnover_per_year": stats["turnover_per_year"],
        "exposure_pct": stats["exposure_pct"],
        "average_invested_pct": stats["average_invested_pct"],
        "costs": stats["costs"],
        "dividends_received": stats["dividends_received"],
        "bars": stats["bars"],
        "years": stats["years"],
        "trades": trades[-20:],
        "equity_curve": monthly_equity(result.equity),
        "baselines": {
            "cash_only": {"total_return_pct": 0.0, "final_equity": stats["initial_capital"], "note": "never trades"},
            "buy_and_hold_same_terms": {
                "total_return_pct": round(base_stats["total_return_pct"], 2), "max_drawdown_pct": round(base_stats["max_drawdown_pct"], 2), "final_equity": base_stats["final_equity"],
                "exposure_pct": base_stats["exposure_pct"], "note": "same capital, same position size, same costs, no stop or target",
            },
            "buy_and_hold_fully_invested": {
                "total_return_pct": round(full_stats["total_return_pct"], 2), "max_drawdown_pct": round(full_stats["max_drawdown_pct"], 2), "final_equity": full_stats["final_equity"],
                "note": "100% of capital, same costs; shown for scale, not a like-for-like comparison",
            },
        },
        "excess_return_vs_buy_hold_pct": round(stats["total_return_pct"] - base_stats["total_return_pct"], 2),
        "equity_curve_buy_hold": monthly_equity(baseline.equity),
        "metric_notes": {
            "annualized": "annualised return and Sharpe ratio are null for tests shorter than 252 bars",
            "sharpe": stats["sharpe_note"],
            "buy_hold_return_pct": "buy and hold on the same terms as the strategy (see baselines)",
        },
        "execution_assumptions": {
            "commission_pct_per_trade": round(costs["commission_rate_pct"], 4),
            "spread_estimate_pct": round(costs["half_spread_bps"] / 100, 4),
            "slippage_estimate_pct": round(costs["slippage_bps"] / 100, 4),
            "one_way_cost_pct": round(costs["commission_rate_pct"] + costs["one_way_price_cost_pct"], 4),
            "round_trip_cost_pct": round(costs["round_trip_cost_pct"], 4),
            "max_position_pct": round(float(config.position_fraction) * 100, 2),
            "min_dollar_volume": config.min_dollar_volume,
            "stop_loss_pct": None if config.stop_loss_pct is None else round(float(config.stop_loss_pct) * 100, 2),
            "target_pct": None if config.take_profit_pct is None else round(float(config.take_profit_pct) * 100, 2),
            "entry": "decided at a close, filled at the next open",
            "fractional_shares": config.fractional_shares,
            "note": "One accounting book drives both the trade list and the equity curve. Stops and targets are checked against each day's range; if both are touched the stop is assumed first.",
        },
        "disclaimer": DISCLAIMER,
    }


def _run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    spread: float = 0.0008,
    slippage: float = 0.0015,
    max_position_pct: float = 0.25,
    min_dollar_volume: float | None = 5_000_000,
    stop_loss_pct: float | None = 0.05,
    target_pct: float | None = 0.10,
) -> dict[str, Any]:
    """Backtest ``signal`` (1 = want to be long, decided at each close) on ``df``. Returns the API shape."""
    config = _config(initial_capital, commission, spread, slippage, max_position_pct, min_dollar_volume, stop_loss_pct, target_pct)
    target = (pd.Series(signal.to_numpy(), index=df.index) > 0).astype(float)  # short signals are treated as flat: no short selling
    result = _engine_run(df, target, config)
    baseline = buy_and_hold(df, config)
    fully_invested = buy_and_hold(df, _config(initial_capital, commission, spread, slippage, 1.0, None, None, None))
    return _legacy_shape(result, baseline, fully_invested, config)


def run_backtest(ticker: str, strategy: str = "sma_crossover", period: str = "3y", initial_capital: float = 10_000.0) -> dict[str, Any]:
    if strategy not in _STRATEGIES:
        return {"available": False, "error": f"Unknown strategy: {strategy}. Valid: {list(_STRATEGIES.keys())}"}
    from app.marketdata.service import get_market_data

    try:
        envelope = get_market_data().history(ticker, period=period, interval="1d")
    except MarketVisionError as exc:
        return {"available": False, "ticker": str(ticker).upper(), "error": exc.message, "error_code": exc.code}
    bars = envelope.data.dropna(subset=["Open", "High", "Low", "Close"])
    if len(bars) < MIN_BARS:
        return {"available": False, "ticker": envelope.symbol, "error": f"Insufficient data (need at least {MIN_BARS} bars)", "data_source": envelope.meta()}
    try:
        results = _run_backtest(bars, _STRATEGIES[strategy](bars), initial_capital=initial_capital)
    except MarketVisionError as exc:
        return {"available": False, "ticker": envelope.symbol, "error": exc.message, "error_code": exc.code}
    except Exception as exc:  # noqa: BLE001 - report, never crash the route
        log.warning("Backtest error %s/%s: %s", ticker, strategy, type(exc).__name__)
        return {"available": False, "ticker": envelope.symbol, "error": f"The backtest could not be completed ({type(exc).__name__})."}
    return {
        "available": True,
        "ticker": envelope.symbol,
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "strategy_type": STRATEGY_TYPES[strategy],
        "period": period,
        "data_source": envelope.meta(),
        "survivorship_note": "A test on one ticker chosen today cannot include companies that failed; results across today's large companies are flattered by that.",
        **results,
    }
