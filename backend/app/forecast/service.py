"""Live forecasts, through the same feature pipeline that training used.

What a caller gets depends on what has been established:

* A promoted model exists: its calibrated probability, with the model id, schema version and a
  fingerprint of the inputs.
* No model is promoted: the historical base rate, labelled as a baseline, with a plain statement
  that no predictive edge has been established. This is the normal state until a model passes
  the promotion rule; it is not an error.
* The inputs cannot be computed (a provider is down, history is too short, the latest bar is
  stale): no probability at all, and the reason. Nothing is substituted.

A forecast never authorises a trade. See ``app.paper``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.core.clock import iso, utcnow
from app.core.errors import FeatureUnavailable, MarketVisionError, ModelUnavailable
from app.features import schema
from app.features.context import load_context
from app.features.pipeline import build_features, latest_row
from app.forecast import registry
from app.forecast.dataset import target_text
from app.forecast.uncertainty import predict_quantiles
from app.marketdata import calendar as cal

DISCLAIMER = "A probability estimate from historical data. Not a guarantee and not financial advice."
HISTORY_PERIOD = "3y"  # comfortably more than the 301 bars the features need
MAX_STALE_SESSIONS = 1


def _fingerprint(symbol: str, as_of: date, names: list[str], row: pd.Series, providers: dict[str, str]) -> str:
    payload = {"symbol": symbol, "as_of": as_of.isoformat(), "schema": schema.schema_hash(), "features": {n: round(float(row[n]), 10) for n in names}, "providers": providers}
    return "inputs-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class PredictionService:
    def __init__(self, market_data: Any, *, model_dir: Any | None = None) -> None:
        self.market_data = market_data
        self.model_dir = model_dir
        self._bundle: dict[str, Any] | None = None
        self._bundle_id: str | None = None
        self._bundle_error: MarketVisionError | None = None

    # ------------------------------------------------------------------ the model in force
    def _champion(self) -> dict[str, Any] | None:
        champion = registry.champion(self.model_dir)
        if champion is None:
            self._bundle, self._bundle_id, self._bundle_error = None, None, None
            return None
        if champion != self._bundle_id:
            self._bundle_id = champion
            try:
                self._bundle, self._bundle_error = registry.load_bundle(champion, self.model_dir), None
            except MarketVisionError as exc:
                self._bundle, self._bundle_error = None, exc
        if self._bundle_error is not None:
            raise self._bundle_error
        return self._bundle

    def status(self) -> dict[str, Any]:
        """What is serving predictions right now, and on what evidence."""
        reg = registry.read_registry(self.model_dir)
        out: dict[str, Any] = {
            "champion": reg.get("champion"),
            "baseline": {k: v for k, v in (reg.get("baseline") or {}).items() if k not in {"uncertainty"}} or None,
            "schema": {"name": schema.SCHEMA_NAME, "version": schema.SCHEMA_VERSION, "hash": schema.schema_hash()},
            "edge_established": False,
        }
        if reg.get("champion"):
            try:
                self._champion()
                out["edge_established"] = True
                out["model_state"] = "loaded"
            except MarketVisionError as exc:
                out["model_state"] = f"refused: {exc.message}"
                out["model_error"] = exc.code
        else:
            out["model_state"] = "no model has passed the promotion rule; the baseline is served"
        return out

    # ------------------------------------------------------------------ inputs
    def _inputs(self, symbol: str, names: list[str], as_of: date | None) -> tuple[pd.Series, date, dict[str, Any]]:
        now = utcnow()
        last_complete = cal.last_completed_session(now)
        end = min(as_of, last_complete) if as_of else last_complete
        envelope = self.market_data.history(symbol, period=HISTORY_PERIOD if as_of is None else None, start=None if as_of is None else cal.shift_trading_days(end, -760), end=end, interval="1d")
        bars = envelope.data
        bars = bars[bars.index <= pd.Timestamp(end)]  # an unfinished session is never used
        if len(bars) == 0:
            raise FeatureUnavailable("No completed daily bars are available.", details={"symbol": symbol})
        latest = bars.index[-1].date()
        behind = len(cal.trading_days(latest, end)) - 1
        if behind > MAX_STALE_SESSIONS:
            raise FeatureUnavailable(
                f"The latest bar for {symbol} is {latest.isoformat()}, {behind} sessions behind {end.isoformat()}. A forecast from stale data is not made.",
                details={"symbol": symbol, "latest_bar": latest.isoformat(), "expected": end.isoformat(), "sessions_behind": behind}, code="stale_data",
            )
        context = load_context(self.market_data, symbol, period=HISTORY_PERIOD if as_of is None else None, start=None if as_of is None else cal.shift_trading_days(end, -760), end=end,
                               include_macro=any(schema.spec(n).group == "macro" for n in names))
        for part in ("spy", "qqq", "sector"):
            frame = getattr(context, part)
            if frame is not None:
                setattr(context, part, frame[frame.index <= pd.Timestamp(latest)])
        if context.vix is not None:
            context.vix = context.vix[context.vix.index <= pd.Timestamp(latest)]
        features = build_features(bars, context)
        row = latest_row(features, names)
        problems = schema.validate_frame(features.iloc[[-1]][names], names, allow_missing=False)
        if problems:
            raise FeatureUnavailable("The computed inputs fail the feature schema's checks, so no forecast is made.", details={"problems": problems[:5]}, code="schema_validation_failed")
        data_source = envelope.meta()
        data_source.pop("attempts", None)
        return row, latest, data_source

    # ------------------------------------------------------------------ the forecast
    def predict(self, symbol: str, *, as_of: date | None = None) -> dict[str, Any]:
        """Probability that ``symbol`` closes higher ``horizon`` sessions after the as-of date."""
        from app.marketdata.service import clean_symbol

        symbol = clean_symbol(symbol)
        reg = registry.read_registry(self.model_dir)
        baseline = reg.get("baseline") or {}
        try:
            bundle = self._champion()
            model_error = None
        except MarketVisionError as exc:
            bundle, model_error = None, exc

        horizon = int(bundle["horizon"]) if bundle else int(baseline.get("horizon", 5))
        base: dict[str, Any] = {
            "symbol": symbol,
            "horizon_sessions": horizon,
            "target": target_text(horizon),
            "issued_at": iso(utcnow()),
            "schema": {"name": schema.SCHEMA_NAME, "version": schema.SCHEMA_VERSION, "hash": schema.schema_hash()},
            "disclaimer": DISCLAIMER,
            "authorises_trade": False,
        }
        names = list(bundle["feature_names"]) if bundle else schema.feature_names(["price"])
        try:
            row, as_of_date, data_source = self._inputs(symbol, names, as_of)
        except MarketVisionError as exc:
            return {
                **base, "status": "unavailable", "probability_up": None, "reason": {"code": exc.code, "message": exc.message, "details": exc.details},
                "decision_default": "HOLD", "model": {"id": bundle["model_id"] if bundle else None, "kind": "model" if bundle else "baseline"},
            }
        window_end = cal.shift_trading_days(as_of_date, horizon)
        common = {
            **base, "as_of": as_of_date.isoformat(), "window": {"first_session": cal.next_trading_day(as_of_date).isoformat(), "last_session": window_end.isoformat()},
            "data_source": data_source, "data_snapshot_id": _fingerprint(symbol, as_of_date, names, row, {"history": data_source.get("provider", "")}),
        }
        volatility = float(row["vol_21"]) if "vol_21" in row.index else None

        if bundle is None:
            table = baseline.get("uncertainty") or {}
            scaled = bool(baseline.get("uncertainty_scaled_supported")) and volatility is not None and bool(table.get("scaled"))
            message = "No model has passed the promotion rule, so the historical base rate is reported. No predictive edge has been established."
            if model_error is not None:
                message = f"The promoted model was refused ({model_error.message}) The historical base rate is reported instead."
            return {
                **common, "status": "baseline", "probability_up": baseline.get("base_rate"), "base_rate": baseline.get("base_rate"), "edge_established": False,
                "model": {"id": None, "kind": "baseline", "method": "historical base rate of the development data", "from_run": baseline.get("from_run"), "error": model_error.code if model_error else None},
                "expected_return": {"value": baseline.get("historical_mean_return"), "method": "historical mean"} if baseline else None,
                "range": _range(table, volatility, scaled) if table else None,
                "message": message if baseline else "No evaluation has been run yet, so there is no base rate to report.",
            }

        X = pd.DataFrame([row[names].to_numpy(float)], columns=names)  # noqa: N806
        raw = float(np.clip(bundle["estimator"].predict_proba(X)[0, list(bundle["estimator"].classes_).index(1)], 1e-6, 1 - 1e-6))
        probability = float(bundle["calibrator"].transform(np.array([raw]))[0])
        expected = bundle.get("expected_return") or {}
        expected_value = float(expected["regressor"].predict(X)[0]) if expected.get("supported") and expected.get("regressor") is not None else expected.get("historical_mean")
        scaled = bool(bundle.get("uncertainty_scaled_supported")) and volatility is not None
        return {
            **common, "status": "model", "probability_up": round(probability, 6), "probability_up_uncalibrated": round(raw, 6), "base_rate": round(float(bundle["base_rate"]), 6),
            "edge_established": True,
            "model": {"id": bundle["model_id"], "kind": "model", "trained_through": bundle.get("trained_through"), "feature_groups": bundle.get("groups"), "calibration": bundle["calibrator"].method},
            "expected_return": {"value": expected_value, "method": "model estimate" if expected.get("supported") else "historical mean (no model beat it)"},
            "range": _range(bundle["uncertainty"], volatility, scaled),
            "message": "Calibrated probability from the promoted model. It passed the promotion rule on out-of-sample data; that is evidence of a small statistical edge, not of profit.",
        }


def _range(table: dict[str, Any], volatility: float | None, scaled: bool) -> dict[str, Any] | None:
    if not table or not table.get("unconditional"):
        return None
    quantiles = predict_quantiles(table, volatility, scaled=scaled) if scaled else dict(table["unconditional"])
    return {
        "method": "historical returns scaled by the stock's 21-session volatility" if scaled else "historical returns, not adjusted for current volatility",
        "p10": round(float(quantiles["0.10"]), 6), "p50": round(float(quantiles["0.50"]), 6), "p90": round(float(quantiles["0.90"]), 6),
        "p05": round(float(quantiles["0.05"]), 6), "p95": round(float(quantiles["0.95"]), 6),
        "meaning": "share of past outcomes below each value; an 80% range runs from p10 to p90",
    }


_default: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _default
    if _default is None:
        from app.marketdata.service import get_market_data

        _default = PredictionService(get_market_data())
    return _default


def set_prediction_service(service: PredictionService | None) -> None:
    global _default
    _default = service
