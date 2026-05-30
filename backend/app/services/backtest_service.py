from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


# ── Strategy implementations ─────────────────────────────────────────────────

def _strategy_sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Buy when fast SMA crosses above slow SMA, sell when it crosses below."""
    fast_sma = _sma(df["Close"], fast)
    slow_sma = _sma(df["Close"], slow)
    signal = pd.Series(0, index=df.index)
    signal[fast_sma > slow_sma] = 1
    signal[fast_sma < slow_sma] = -1
    return signal


def _strategy_rsi_mean_reversion(df: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """Buy when RSI < oversold, sell when RSI > overbought."""
    rsi = _rsi(df["Close"])
    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(rsi)):
        if pd.isna(rsi.iloc[i]):
            signal.iloc[i] = position
            continue
        if rsi.iloc[i] < oversold:
            position = 1
        elif rsi.iloc[i] > overbought:
            position = 0
        signal.iloc[i] = position
    return signal


def _strategy_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Always long — baseline comparison."""
    return pd.Series(1, index=df.index)


def _strategy_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.Series:
    """Buy when MACD crosses above signal line, sell when it crosses below."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    sig = pd.Series(0, index=df.index)
    sig[macd > signal_line] = 1
    return sig


def _strategy_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Buy when price touches lower band, sell at upper band."""
    mid = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    signal = pd.Series(0, index=df.index)
    position = 0
    for i in range(len(df)):
        if pd.isna(lower.iloc[i]):
            signal.iloc[i] = position
            continue
        if df["Close"].iloc[i] <= lower.iloc[i]:
            position = 1
        elif df["Close"].iloc[i] >= upper.iloc[i]:
            position = 0
        signal.iloc[i] = position
    return signal


_STRATEGIES = {
    "sma_crossover": _strategy_sma_crossover,
    "rsi_mean_reversion": _strategy_rsi_mean_reversion,
    "macd": _strategy_macd,
    "bollinger_bands": _strategy_bollinger_bands,
    "buy_and_hold": _strategy_buy_and_hold,
}

STRATEGY_LABELS = {
    "sma_crossover": "SMA Crossover (20/50)",
    "rsi_mean_reversion": "RSI Mean Reversion (30/70)",
    "macd": "MACD Signal Line",
    "bollinger_bands": "Bollinger Bands (20, 2σ)",
    "buy_and_hold": "Buy & Hold (Baseline)",
}

STRATEGY_TYPES = {
    "sma_crossover": "trend_following",
    "rsi_mean_reversion": "mean_reversion",
    "macd": "momentum",
    "bollinger_bands": "mean_reversion",
    "buy_and_hold": "baseline",
}


# ── Core backtest engine ─────────────────────────────────────────────────────

def _run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,   # 0.1% per trade
    spread: float = 0.0008,      # 0.08% estimated spread
    slippage: float = 0.0015,    # 0.15% estimated market impact/slippage
    max_position_pct: float = 0.25,
    min_dollar_volume: float = 5_000_000,
    stop_loss_pct: float = 0.05,
    target_pct: float = 0.10,
) -> dict[str, Any]:
    df = df.copy()
    df["dollar_volume"] = df["Close"] * df.get("Volume", 0)
    df["liquidity_pass"] = df["dollar_volume"] >= min_dollar_volume
    df["signal"] = signal.values
    invalid_trade_bars = int(((df["signal"] != 0) & (~df["liquidity_pass"])).sum())
    df.loc[~df["liquidity_pass"], "signal"] = 0
    df["position"] = df["signal"].shift(1).fillna(0)  # enter next bar
    df["daily_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["daily_return"] * max_position_pct
    df["strategy_return_after_costs"] = df["strategy_return"]

    # Apply execution realism on position changes.
    position_changes = df["position"].diff().abs()
    one_way_cost = commission + spread + slippage
    df.loc[position_changes > 0, "strategy_return_after_costs"] -= one_way_cost * max_position_pct

    # Equity curve
    df["equity"] = initial_capital * (1 + df["strategy_return_after_costs"]).cumprod()
    df["bh_equity"] = initial_capital * (1 + df["daily_return"]).cumprod()

    # Detect trades
    trades: list[dict] = []
    in_trade = False
    entry_price = 0.0
    entry_date = ""
    stop_price = 0.0
    target_price = 0.0
    for i, (date, row) in enumerate(df.iterrows()):
        if row["position"] == 1 and not in_trade:
            in_trade = True
            entry_price = float(row["Close"])
            entry_date = str(date.date())
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + target_pct)
        elif row["position"] == 1 and in_trade:
            exit_reason = None
            exit_price = None
            if float(row.get("Low", row["Close"])) <= stop_price:
                exit_reason = "stop_loss"
                exit_price = stop_price
            elif float(row.get("High", row["Close"])) >= target_price:
                exit_reason = "target"
                exit_price = target_price
            if exit_reason and exit_price is not None:
                in_trade = False
                pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0.0
                net_pnl_pct = pnl_pct - (one_way_cost * 2 * 100)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": str(date.date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(net_pnl_pct, 2),
                    "gross_pnl_pct": round(pnl_pct, 2),
                    "estimated_round_trip_cost_pct": round(one_way_cost * 2 * 100, 3),
                    "exit_reason": exit_reason,
                    "outcome": "win" if net_pnl_pct > 0 else "loss",
                })
        elif row["position"] == 0 and in_trade:
            in_trade = False
            exit_price = float(row["Close"])
            pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0.0
            net_pnl_pct = pnl_pct - (one_way_cost * 2 * 100)
            trades.append({
                "entry_date": entry_date,
                "exit_date": str(date.date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(net_pnl_pct, 2),
                "gross_pnl_pct": round(pnl_pct, 2),
                "estimated_round_trip_cost_pct": round(one_way_cost * 2 * 100, 3),
                "exit_reason": "signal_exit",
                "outcome": "win" if net_pnl_pct > 0 else "loss",
            })

    # Close any open trade at end
    if in_trade and len(df) > 0:
        exit_price = float(df["Close"].iloc[-1])
        pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0.0
        net_pnl_pct = pnl_pct - (one_way_cost * 2 * 100)
        trades.append({
            "entry_date": entry_date,
            "exit_date": str(df.index[-1].date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(net_pnl_pct, 2),
            "gross_pnl_pct": round(pnl_pct, 2),
            "estimated_round_trip_cost_pct": round(one_way_cost * 2 * 100, 3),
            "exit_reason": "open_at_end",
            "outcome": "win" if net_pnl_pct > 0 else "open",
        })

    # Metrics
    strat_returns = df["strategy_return_after_costs"].dropna()
    bh_returns = df["daily_return"].dropna()

    final_equity = float(df["equity"].iloc[-1]) if not df["equity"].isna().all() else initial_capital
    bh_final = float(df["bh_equity"].iloc[-1]) if not df["bh_equity"].isna().all() else initial_capital

    total_return = (final_equity / initial_capital - 1) * 100
    bh_total_return = (bh_final / initial_capital - 1) * 100

    ann_return = float((1 + total_return / 100) ** (252 / max(len(strat_returns), 1)) - 1) * 100
    ann_vol = float(strat_returns.std() * np.sqrt(252)) * 100 if len(strat_returns) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100

    wins = [t for t in trades if t["outcome"] == "win"]
    losses_list = [t for t in trades if t["outcome"] == "loss"]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses_list])) if losses_list else 0.0
    gross_profit = float(sum(max(t["pnl_pct"], 0) for t in trades))
    gross_loss = abs(float(sum(min(t["pnl_pct"], 0) for t in trades)))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    avg_reward_risk = avg_win / abs(avg_loss) if avg_loss < 0 else 0.0
    false_positive_rate = len(losses_list) / len(trades) * 100 if trades else 0.0

    # Equity curve (sampled monthly for chart)
    equity_curve = []
    sampled = df["equity"].resample("ME").last().dropna()
    for date, val in sampled.items():
        equity_curve.append({"date": str(date.date()), "equity": round(float(val), 2)})

    return {
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(bh_total_return, 2),
        "annualized_return_pct": round(ann_return, 2),
        "annualized_volatility_pct": round(ann_vol, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "average_reward_risk": round(avg_reward_risk, 3),
        "false_positive_rate_pct": round(false_positive_rate, 1),
        "invalid_trades": invalid_trade_bars,
        "final_equity": round(final_equity, 2),
        "initial_capital": initial_capital,
        "trades": trades[-20:],   # last 20 trades
        "equity_curve": equity_curve,
        "execution_assumptions": {
            "commission_pct_per_trade": round(commission * 100, 4),
            "spread_estimate_pct": round(spread * 100, 4),
            "slippage_estimate_pct": round(slippage * 100, 4),
            "one_way_cost_pct": round(one_way_cost * 100, 4),
            "round_trip_cost_pct": round(one_way_cost * 2 * 100, 4),
            "max_position_pct": round(max_position_pct * 100, 2),
            "min_dollar_volume": min_dollar_volume,
            "stop_loss_pct": round(stop_loss_pct * 100, 2),
            "target_pct": round(target_pct * 100, 2),
            "note": "Backtest applies estimated costs, liquidity filtering, max position sizing, and simplified daily stop/target checks.",
        },
    }


def run_backtest(
    ticker: str,
    strategy: str = "sma_crossover",
    period: str = "3y",
    initial_capital: float = 10_000.0,
) -> dict[str, Any]:
    if strategy not in _STRATEGIES:
        return {"available": False, "error": f"Unknown strategy: {strategy}. Valid: {list(_STRATEGIES.keys())}"}

    try:
        df = yf.download(ticker.upper(), period=period, interval="1d", auto_adjust=True, progress=False)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    if df is None or len(df) < 60:
        return {"available": False, "error": "Insufficient data (need at least 60 bars)"}

    df = _flatten(df).dropna(subset=["Close"])

    try:
        signal = _STRATEGIES[strategy](df)
        results = _run_backtest(df, signal, initial_capital=initial_capital)
    except Exception as exc:
        log.warning("Backtest error %s/%s: %s", ticker, strategy, exc)
        return {"available": False, "error": str(exc)}

    return {
        "available": True,
        "ticker": ticker.upper(),
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "strategy_type": STRATEGY_TYPES[strategy],
        "period": period,
        "initial_capital": initial_capital,
        **results,
    }
