from __future__ import annotations

import pandas as pd

from app.services.backtest_service import _run_backtest


def test_backtest_reports_execution_realism_metrics() -> None:
    index = pd.date_range("2025-01-01", periods=80, freq="D")
    close = pd.Series([100 + i * 0.2 for i in range(80)], index=index)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.02,
            "Low": close * 0.99,
            "Close": close,
            "Volume": [1_000_000] * 80,
        },
        index=index,
    )
    signal = pd.Series(1, index=index)

    result = _run_backtest(df, signal, min_dollar_volume=5_000_000)

    assert "profit_factor" in result
    assert "average_reward_risk" in result
    assert "false_positive_rate_pct" in result
    assert "invalid_trades" in result
    assert result["execution_assumptions"]["max_position_pct"] == 25.0
    assert result["execution_assumptions"]["stop_loss_pct"] == 5.0
    assert result["execution_assumptions"]["target_pct"] == 10.0
    assert result["trades"]
    assert "exit_reason" in result["trades"][-1]
