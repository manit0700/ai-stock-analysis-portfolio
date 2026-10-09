# Quarantined model artifacts

These are the v5-v12 XGBoost ensembles trained on Kaggle in May 2026. They are kept for the
record and are no longer loaded by the backend, because the October 2026 audit found:

1. **Training and serving computed different features.** The serving code filled 21 of the
   models' 45 inputs with 0.0 (VIX, yields, SPY/QQQ trend, sector returns) — 51.6% of the
   models' feature importance (audit finding M1).
2. **Their own stored holdout metrics are below a constant guess** (M2): three-class accuracy
   36.8% vs 41.5% for "always bullish"; direction 52.0% vs 55.1%; the promoted breakout signal
   68.4% vs 72.4% for "always no".
3. The hyperparameter search ran over data that included the holdout (M3), and the holdout was
   reused to choose between versions (M4).

The replacement pipeline lives in `backend/app/forecast/`. It found no predictive edge either —
and says so. Do not point `backend/app/services/model_service.py` back at these files.
