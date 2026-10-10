"""The AI copilot: plain-language explanations of numbers the application has already computed.

Any OpenAI-compatible chat server works, configured only through the environment:

    MARKETVISION_LLM_BASE_URL   e.g. http://127.0.0.1:8089/v1 for a free local model
                                (empty = OpenAI's own endpoint)
    MARKETVISION_LLM_API_KEY    the server's key (OPENAI_API_KEY is also read); a local server
                                usually needs none
    MARKETVISION_LLM_MODEL      the model name that server expects

Nothing is enabled by default and no paid service is called unless the owner configures one.
The model explains; it never supplies prices, balances, trades or returns, and the arithmetic
stays in code. Errors returned to the browser are generic: the raw exception text is logged
(through the redacting filter), not sent.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlsplit

from app.core.config import get_settings

log = logging.getLogger(__name__)

#: Sent instead of a key to local servers that need none; such servers ignore it.
_NO_KEY_PLACEHOLDER = "no-key-required"

SYSTEM_PROMPT = """You are MarketVision AI Copilot — a financial intelligence assistant built into a stock analysis platform.

You help traders and investors by:
- Explaining AI model predictions and simulation scenarios in plain language
- Summarizing news sentiment and market context
- Answering stock research questions using the data provided
- Highlighting risks and probabilities — never promising guaranteed outcomes

Rules:
- Always remind users this is for research, not personalized financial advice
- Be concise — 2-4 sentences per answer unless a detailed breakdown is requested
- Use real numbers from the context provided
- Speak like a smart analyst, not a chatbot
- Every number you state must appear in the context you were given. If a price, balance, trade,
  return or probability is not in that context, say it is not available. Never state one from memory.
- Trading in this application is simulated with virtual money; never suggest real-money orders.
"""


class CopilotService:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = None
        self._model = settings.llm_model
        self._base_url = settings.llm_base_url
        host = (urlsplit(self._base_url).hostname or "") if self._base_url else ""
        # Claude's OpenAI-compatible endpoint rejects a temperature below 1 on recent models.
        self._send_temperature = not host.endswith("anthropic.com")
        self.status_reason = ""
        if not (settings.llm_api_key or settings.llm_base_url):
            self.status_reason = "No language model is configured (MARKETVISION_LLM_BASE_URL or MARKETVISION_LLM_API_KEY)."
            return
        try:
            from openai import OpenAI
        except ImportError:
            self.status_reason = "The openai client package is not installed (pip install -r requirements-heavy.txt)."
            log.warning("Copilot disabled: %s", self.status_reason)
            return
        try:
            self._client = OpenAI(
                api_key=settings.llm_api_key.get() or _NO_KEY_PLACEHOLDER,
                base_url=self._base_url or None,
                timeout=30.0,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001 - the client raises several types on bad config
            self.status_reason = f"The language-model client could not start ({type(exc).__name__})."
            log.warning("Copilot init failed: %s", exc)
            return
        log.info("Copilot ready (model=%s, endpoint=%s)", self._model, host or "api.openai.com")

    def is_available(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if not self.is_available():
            return f"AI Copilot is not available. {self.status_reason}".strip()

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            ctx_text = _build_context_block(context)
            messages.append({"role": "system", "content": f"Current market context:\n{ctx_text}"})

        if history:
            messages.extend(history[-6:])  # keep last 3 turns

        messages.append({"role": "user", "content": message})

        request: dict[str, Any] = {"model": self._model, "messages": messages, "max_tokens": 400}
        if self._send_temperature:
            request["temperature"] = 0.4
        try:
            response = self._client.chat.completions.create(**request)
            content = response.choices[0].message.content or ""
            return content.strip() or "The language model returned an empty answer."
        except Exception as exc:  # noqa: BLE001 - network, auth, quota and model errors all land here
            log.warning("Copilot chat failed: %s", exc)
            return f"The language model could not answer ({type(exc).__name__}). Nothing was generated."

    def explain_simulation(self, simulation: dict[str, Any]) -> str:
        prompt = (
            f"Explain the AI simulation for {simulation.get('ticker')}. "
            f"The dominant scenario is {simulation.get('dominant_scenario')} with "
            f"{simulation.get('confidence')}% confidence. "
            f"Bullish probability: {round(simulation['probabilities'].get('bullish', 0) * 100, 1)}%, "
            f"Bearish: {round(simulation['probabilities'].get('bearish', 0) * 100, 1)}%, "
            f"Sideways: {round(simulation['probabilities'].get('sideways', 0) * 100, 1)}%. "
            f"Risk level: {simulation.get('risk_level')} (score {simulation.get('risk_score')}). "
            f"Key reasons: {'; '.join(simulation.get('reasons', [])[:2])}. "
            f"Main risks: {'; '.join(simulation.get('risks', [])[:2])}. "
            "Explain what this means for a trader in 3-4 sentences."
        )
        return self.chat(prompt)

    def explain_prediction_facts(self, simulation: dict[str, Any]) -> dict[str, Any]:
        facts = _extract_explanation_facts(simulation)
        if not self.is_available():
            return {
                "available": False,
                "model": "deterministic_fallback",
                "explanation": _deterministic_prediction_explanation(facts),
                "facts_used": facts,
                "disclaimer": "This is a probability-based simulation, not financial advice.",
            }

        prompt = (
            "Explain this MarketVision AI prediction using only the JSON facts below. "
            "Do not invent numbers, prices, catalysts, or certainty. Explain why the setup is bullish, bearish, or sideways; "
            "why it was promoted or rejected; key risks; relevant indicators; historical similarity; and macro/sentiment context. "
            "End with exactly: This is a probability-based simulation, not financial advice.\n\n"
            f"FACTS:\n{json.dumps(facts, indent=2, default=str)}"
        )
        explanation = self.chat(prompt)
        if "not financial advice" not in explanation.lower():
            explanation = explanation.rstrip() + "\n\nThis is a probability-based simulation, not financial advice."
        return {
            "available": True,
            "model": self._model,
            "explanation": explanation,
            "facts_used": facts,
            "disclaimer": "This is a probability-based simulation, not financial advice.",
        }

    def summarize_news_sentiment(self, ticker: str, news: list[dict[str, Any]]) -> str:
        if not news:
            return "No recent news available to summarize."
        headlines = "\n".join(
            f"- [{n.get('sentiment_label', 'neutral').upper()} {n.get('sentiment_score', 0):+.2f}] {n['title']}"
            for n in news[:6]
        )
        prompt = (
            f"Summarize the news sentiment for {ticker} based on these recent headlines:\n"
            f"{headlines}\n\n"
            "Give a 2-3 sentence market sentiment summary. What is the overall tone and does it support or contradict the technical picture?"
        )
        return self.chat(prompt)

    def explain_stock_analysis(self, analysis: dict[str, Any]) -> str:
        prompt = (
            f"Explain the technical analysis for {analysis.get('ticker')} ({analysis.get('company_name')}). "
            f"Current price: ${analysis.get('current_price')}, daily change: {analysis.get('daily_change_pct')}%. "
            f"Stance: {analysis.get('stance')} with score {analysis.get('score')}/100. "
            f"RSI: {analysis.get('signals', {}).get('rsi_14')}. "
            f"Key signals: {'; '.join(analysis.get('reasons', [])[:2])}. "
            "Summarise in 2-3 sentences what this means and what a trader should watch for."
        )
        return self.chat(prompt)


def _build_context_block(context: dict[str, Any]) -> str:
    lines = []
    if "ticker" in context:
        lines.append(f"Ticker: {context['ticker']}")
    if "current_price" in context:
        lines.append(f"Price: ${context['current_price']}")
    if "dominant_scenario" in context:
        lines.append(f"Dominant scenario: {context['dominant_scenario']} ({context.get('confidence')}% confidence)")
    if "risk_level" in context:
        lines.append(f"Risk level: {context['risk_level']} (score {context.get('risk_score')})")
    if "stance" in context:
        lines.append(f"Technical stance: {context['stance']} (score {context.get('score')})")
    if "reasons" in context:
        lines.append(f"Bullish signals: {'; '.join(context['reasons'][:2])}")
    if "risks" in context:
        lines.append(f"Risk signals: {'; '.join(context['risks'][:2])}")
    return "\n".join(lines)


def _extract_explanation_facts(simulation: dict[str, Any]) -> dict[str, Any]:
    advanced = simulation.get("advanced_intelligence") or {}
    bot = simulation.get("ai_signal_bot") or {}
    final_signal = simulation.get("final_signal") or {}
    similarity = advanced.get("historical_similarity") or {}
    macro = advanced.get("macro") or {}
    sentiment = advanced.get("sentiment") or {}
    technical = advanced.get("technical_analysis") or {}
    intraday_vwap = advanced.get("intraday_vwap") or {}
    return {
        "ticker": simulation.get("ticker"),
        "dominant_scenario": simulation.get("dominant_scenario"),
        "probabilities": simulation.get("probabilities"),
        "confidence": simulation.get("confidence"),
        "risk_score": simulation.get("risk_score"),
        "risk_level": simulation.get("risk_level"),
        "quality_label": bot.get("quality_label") or simulation.get("quality_label"),
        "action": bot.get("action"),
        "all_gates_passed": bot.get("all_gates_passed"),
        "failed_gates": bot.get("failed_gates", []),
        "final_signal": {
            "name": final_signal.get("name"),
            "probability": final_signal.get("probability"),
            "confidence": final_signal.get("confidence"),
        },
        "technical": {
            "rsi_14": technical.get("rsi_14"),
            "macd_hist": technical.get("macd_hist"),
            "above_vwap": technical.get("above_vwap"),
            "vwap_distance_pct": technical.get("vwap_distance_pct"),
            "volume_ratio_20": technical.get("volume_ratio_20"),
            "momentum_20": technical.get("momentum_20"),
        },
        "intraday_vwap_alignment": intraday_vwap.get("alignment"),
        "historical_similarity": {
            "available": similarity.get("available"),
            "average_similarity_score": similarity.get("average_similarity_score"),
            "average_future_return_pct": similarity.get("average_future_return_pct"),
            "outcome_probabilities": similarity.get("outcome_probabilities"),
            "average_drawdown_pct": similarity.get("average_drawdown_pct"),
        },
        "macro": {
            "regime": macro.get("macro_regime_label"),
            "scores": macro.get("scores"),
            "summary": macro.get("summary"),
        },
        "sentiment": {
            "label": sentiment.get("label"),
            "score": sentiment.get("score"),
            "source_count": sentiment.get("source_count"),
        },
        "reasons": simulation.get("reasons", [])[:6],
        "risks": simulation.get("risks", [])[:6],
    }


def _deterministic_prediction_explanation(facts: dict[str, Any]) -> str:
    ticker = facts.get("ticker") or "This ticker"
    scenario = facts.get("dominant_scenario") or "unknown"
    confidence = facts.get("confidence")
    risk = facts.get("risk_score")
    quality = str(facts.get("quality_label") or "prediction_only").replace("_", " ")
    action = str(facts.get("action") or "research_only").replace("_", " ")
    failed = facts.get("failed_gates") or []
    technical = facts.get("technical") or {}
    similarity = facts.get("historical_similarity") or {}
    macro = facts.get("macro") or {}
    sentiment = facts.get("sentiment") or {}
    lines = [
        f"{ticker} is currently classified as {scenario} with {confidence}% confidence and risk score {risk}/100.",
        f"The signal quality is {quality}; bot action is {action}."
    ]
    if failed:
        lines.append(f"It was not fully promoted because these gates failed: {', '.join(str(x).replace('_', ' ') for x in failed)}.")
    lines.append(
        "Key model facts: "
        f"RSI {technical.get('rsi_14')}, MACD histogram {technical.get('macd_hist')}, "
        f"VWAP distance {technical.get('vwap_distance_pct')}%, and volume ratio {technical.get('volume_ratio_20')}."
    )
    if similarity.get("available"):
        lines.append(
            f"Historical similarity average future return is {similarity.get('average_future_return_pct')}% "
            f"with similarity score {similarity.get('average_similarity_score')}."
        )
    if macro.get("regime") or sentiment.get("label"):
        lines.append(f"Macro regime is {macro.get('regime')}; sentiment is {sentiment.get('label')} with score {sentiment.get('score')}.")
    lines.append("This is a probability-based simulation, not financial advice.")
    return " ".join(lines)
