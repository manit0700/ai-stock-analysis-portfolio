from __future__ import annotations

from typing import Any, Callable

DISCLAIMER = "Probability-based market simulations and AI-generated financial intelligence, not financial advice."


ToolCaller = Callable[[str, dict[str, Any]], dict[str, Any]]


def build_trade_candidate_report(
    *,
    prompt: str,
    universe: str,
    tickers: str | None,
    period: str,
    interval: str,
    horizon_steps: int,
    max_candidates: int,
    call_tool: ToolCaller,
) -> dict[str, Any]:
    scanner_args = {
        "universe": universe,
        "tickers": tickers or "",
        "period": period,
        "interval": interval,
        "horizon_steps": horizon_steps,
        "max_symbols": max(max_candidates, 5),
    }
    scanner = call_tool("run_scanner", scanner_args).get("result", {})
    rows = scanner.get("results", [])[: max(max_candidates, 1)]
    macro = call_tool("get_macro_regime", {}).get("result", {})
    performance = call_tool("get_performance", {}).get("result", {})

    agent_runs = []
    ranked = []
    for row in rows:
        symbol = row.get("ticker")
        if not symbol:
            continue
        similarity = call_tool("run_similarity", {"ticker": symbol, "horizon_steps": horizon_steps}).get("result", {})
        prediction = call_tool("predict_market", {"ticker": symbol, "period": "1y", "interval": interval, "horizon_steps": horizon_steps}).get("result", {})
        explanation = call_tool("explain_prediction", {"ticker": symbol, "period": "1y", "interval": interval, "horizon_steps": horizon_steps}).get("result", {})

        technical = {
            "agent": "technical_agent",
            "ticker": symbol,
            "stance": prediction.get("dominant_scenario"),
            "confidence": prediction.get("confidence"),
            "risk_score": prediction.get("risk_score"),
            "reason": "Uses RSI, MACD, VWAP, trend, and volume features from MarketVision prediction facts.",
        }
        simulation_agent = {
            "agent": "simulation_agent",
            "ticker": symbol,
            "similarity_score": similarity.get("similarity_score"),
            "average_future_return_pct": similarity.get("average_future_return_pct"),
            "bullish_outcome_pct": similarity.get("bullish_outcome_pct"),
            "bearish_outcome_pct": similarity.get("bearish_outcome_pct"),
        }
        risk_agent = {
            "agent": "risk_agent",
            "ticker": symbol,
            "risk_score": row.get("risk_score"),
            "failed_gates": row.get("failed_gates", []),
            "risk_reward_ratio": row.get("risk_reward_ratio"),
        }
        sentiment_agent = {
            "agent": "sentiment_agent",
            "ticker": symbol,
            "sentiment_score": row.get("sentiment_score"),
            "macro_regime": row.get("macro_regime"),
        }

        judge = _judge_candidate(row=row, prediction=prediction, similarity=similarity, macro=macro)
        agent_runs.extend([technical, simulation_agent, risk_agent, sentiment_agent, judge])
        ranked.append({
            "ticker": symbol,
            "quality_label": row.get("quality_label"),
            "action": row.get("action"),
            "confidence": row.get("confidence"),
            "risk_score": row.get("risk_score"),
            "final_signal_probability": row.get("final_signal_probability"),
            "risk_reward_ratio": row.get("risk_reward_ratio"),
            "historical_similarity_strength": row.get("historical_similarity_strength"),
            "failed_gates": row.get("failed_gates", []),
            "trade_plan": row.get("trade_plan"),
            "judge": judge,
            "explanation": explanation.get("explanation"),
        })

    return {
        "version": "v13.1",
        "prompt": prompt,
        "universe": universe,
        "period": period,
        "interval": interval,
        "horizon_steps": horizon_steps,
        "summary": _build_summary(ranked=ranked, macro=macro, performance=performance),
        "top_candidates": ranked,
        "agent_runs": agent_runs,
        "tool_trace": [
            {"tool": "run_scanner", "purpose": "Rank the requested universe."},
            {"tool": "get_macro_regime", "purpose": "Check macro compatibility."},
            {"tool": "get_performance", "purpose": "Check calibration and signal proof."},
            {"tool": "run_similarity", "purpose": "Compare candidates with historical setups."},
            {"tool": "predict_market", "purpose": "Get probabilities, confidence, risk, and quality labels."},
            {"tool": "explain_prediction", "purpose": "Generate fact-grounded explanation text."},
        ],
        "performance_snapshot": {
            "total_predictions": performance.get("total_predictions"),
            "resolved_signals": performance.get("resolved_signals"),
            "win_rate": performance.get("win_rate"),
            "drift_warnings": (performance.get("drift_monitoring") or {}).get("warnings", []),
        },
        "macro_snapshot": macro,
        "disclaimer": DISCLAIMER,
    }


def _judge_candidate(row: dict[str, Any], prediction: dict[str, Any], similarity: dict[str, Any], macro: dict[str, Any]) -> dict[str, Any]:
    confidence = float(row.get("confidence") or prediction.get("confidence") or 0)
    risk = float(row.get("risk_score") or prediction.get("risk_score") or 100)
    probability = float(row.get("final_signal_probability") or 0)
    similarity_score = float(row.get("historical_similarity_strength") or similarity.get("similarity_score") or 0)
    macro_label = str((macro.get("macro_regime_label") or macro.get("macro_intelligence", {}).get("macro_regime_label") or "unknown"))
    failed_gates = row.get("failed_gates", [])

    score = confidence * 0.35 + probability * 100 * 0.25 + max(0, 100 - risk) * 0.2 + similarity_score * 100 * 0.2
    if failed_gates:
        score -= min(25, len(failed_gates) * 6)
    if macro_label in {"risk_off", "restrictive_rates"} and row.get("quality_label") == "high_confidence_trade_candidate":
        score -= 5

    if row.get("all_gates_passed"):
        decision = "promote_candidate"
    elif score >= 62 and risk < 70:
        decision = "watchlist_candidate"
    elif risk >= 70:
        decision = "avoid_high_risk"
    else:
        decision = "prediction_only"

    return {
        "agent": "judge_agent",
        "decision": decision,
        "final_confidence_score": round(max(0, min(100, score)), 1),
        "final_risk_score": risk,
        "trade_quality": row.get("quality_label"),
        "macro_regime": macro_label,
        "failed_gates": failed_gates,
        "reason": "Combines technical probability, scanner gates, risk, historical similarity, and macro context.",
    }


def _build_summary(ranked: list[dict[str, Any]], macro: dict[str, Any], performance: dict[str, Any]) -> str:
    promoted = [row for row in ranked if row.get("judge", {}).get("decision") == "promote_candidate"]
    watch = [row for row in ranked if row.get("judge", {}).get("decision") == "watchlist_candidate"]
    macro_label = macro.get("macro_regime_label") or macro.get("macro_intelligence", {}).get("macro_regime_label") or "unknown"
    drift_warnings = (performance.get("drift_monitoring") or {}).get("warnings", [])
    if promoted:
        tickers = ", ".join(str(row.get("ticker")) for row in promoted[:3])
        lead = f"Judge Agent promoted {tickers} as the strongest current candidates."
    elif watch:
        tickers = ", ".join(str(row.get("ticker")) for row in watch[:3])
        lead = f"No full promotions; {tickers} are watchlist candidates."
    else:
        lead = "No high-confidence trade candidates passed the current proof and risk gates."
    warning = f" Drift warnings: {', '.join(drift_warnings)}." if drift_warnings else ""
    return f"{lead} Macro regime: {macro_label}.{warning} {DISCLAIMER}"
