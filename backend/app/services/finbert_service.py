from __future__ import annotations

import logging
import os
import importlib.util
from hashlib import sha1
from functools import lru_cache
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Pretrained sentiment is an auxiliary signal for probability-based market simulations, "
    "not financial advice."
)

_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
_ENABLE_FINBERT = os.getenv("MARKETVISION_ENABLE_FINBERT", "1").lower() not in {"0", "false", "no"}
_vader = SentimentIntensityAnalyzer()
_score_cache: dict[str, dict[str, Any]] = {}


def _label_from_score(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def _vader_score(text: str) -> float:
    return round(float(_vader.polarity_scores(text)["compound"]), 4)


@lru_cache(maxsize=1)
def _load_finbert_pipeline() -> Any | None:
    if not _ENABLE_FINBERT:
        return None
    try:
        from transformers import pipeline

        return pipeline(
            task="text-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            top_k=None,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        log.warning("FinBERT unavailable, using VADER fallback: %s", exc)
        return None


class FinBertSentimentService:
    def is_available(self) -> bool:
        return (
            _ENABLE_FINBERT
            and importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def score_text(self, text: str) -> dict[str, Any]:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return {
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "model": "none",
                "available": False,
            }
        cache_key = sha1(cleaned[:2000].encode("utf-8")).hexdigest()
        if cache_key in _score_cache:
            return dict(_score_cache[cache_key])

        model = _load_finbert_pipeline()
        if model is None:
            score = _vader_score(cleaned)
            result = {
                "score": score,
                "label": _label_from_score(score),
                "confidence": abs(score),
                "model": "vader_fallback",
                "available": False,
            }
            _score_cache[cache_key] = result
            return dict(result)

        raw = model(cleaned[:2000])
        rows = raw[0] if raw and isinstance(raw[0], list) else raw
        scores = {str(item["label"]).lower(): float(item["score"]) for item in rows}
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        neutral = scores.get("neutral", 0.0)
        score = positive - negative
        label = "positive" if positive >= negative and positive >= neutral else "negative" if negative >= neutral else "neutral"
        confidence = max(positive, negative, neutral)
        result = {
            "score": round(score, 4),
            "label": label,
            "confidence": round(confidence, 4),
            "model": _MODEL_NAME,
            "available": True,
            "raw": {
                "positive": round(positive, 4),
                "negative": round(negative, 4),
                "neutral": round(neutral, 4),
            },
        }
        _score_cache[cache_key] = result
        return dict(result)

    def enrich_items(self, items: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
        selected = items[:limit] if limit else items
        enriched: list[dict[str, Any]] = []
        for item in selected:
            title = item.get("title") or ""
            summary = item.get("summary") or ""
            sentiment = self.score_text(f"{title}. {summary}" if summary else title)
            next_item = dict(item)
            next_item["sentiment_score"] = sentiment["score"]
            next_item["sentiment_label"] = sentiment["label"]
            next_item["sentiment_confidence"] = sentiment["confidence"]
            next_item["sentiment_model"] = sentiment["model"]
            enriched.append(next_item)
        return enriched

    def aggregate(self, items: list[dict[str, Any]], limit: int = 12) -> dict[str, Any]:
        if not items:
            return {
                "available": False,
                "model": "none",
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "item_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "disclaimer": DISCLAIMER,
            }

        enriched = self.enrich_items(items, limit=limit)
        scores = [float(item.get("sentiment_score", 0) or 0) for item in enriched]
        confidences = [float(item.get("sentiment_confidence", 0) or 0) for item in enriched]
        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)
        labels = [item.get("sentiment_label", "neutral") for item in enriched]
        model = enriched[0].get("sentiment_model", "unknown")
        return {
            "available": model != "vader_fallback",
            "model": model,
            "score": round(avg_score, 4),
            "label": _label_from_score(avg_score),
            "confidence": round(avg_confidence, 4),
            "item_count": len(enriched),
            "positive_count": labels.count("positive"),
            "negative_count": labels.count("negative"),
            "neutral_count": labels.count("neutral"),
            "items": enriched,
            "disclaimer": DISCLAIMER,
        }
