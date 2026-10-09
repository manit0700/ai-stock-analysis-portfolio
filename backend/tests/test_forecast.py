"""The forecasting pipeline: splits, scoring, calibration, the evaluation run and the promotion gate.

Synthetic prices are used to check that the machinery is correct: that it refuses to promote a
model when there is nothing to find, and that it can recognise an effect that was deliberately
planted. Neither result says anything about real markets.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.core import clock
from app.core.errors import Conflict, ModelUnavailable, SchemaMismatch
from app.features import schema
from app.forecast import leakage, registry
from app.forecast.calibration import Calibrator, choose_calibration
from app.forecast.dataset import Dataset, forward_return
from app.forecast.evaluate import block_bootstrap, brier, brier_skill, classification_metrics, reliability, week_blocks
from app.forecast.models import PRIOR, TRAILING, candidates
from app.forecast.promotion import evaluate_rule
from app.forecast.service import PredictionService
from app.forecast.splits import check_fold, holdout_fold, masks, train_end_before, walk_forward_folds
from app.forecast.train import EvalConfig, dataset_from_snapshot, run_evaluation
from app.forecast.uncertainty import evaluate_uncertainty, fit_uncertainty, pinball_loss, predict_quantiles
from tests.helpers import make_ohlcv, make_snapshot, session_index

QUICK = dict(quick=True, n_folds=3, n_bootstrap=200, first_validation_start="2017-06-01", holdout_start="2020-01-02", include_lightgbm=False, leakage_samples=4, shuffle_repeats=3)


# ------------------------------------------------------------------ splits
def test_purge_leaves_the_label_window_and_embargo_before_the_test_period():
    sessions = session_index("2020-01-02", 300)
    test_start = sessions[200]
    end = train_end_before(sessions, test_start, horizon=5, embargo=5)
    assert end == sessions[200 - 1 - 5 - 5]
    # the last training row's label window ends 5 sessions later, still 6 sessions before the test period
    assert sessions.get_loc(test_start) - (sessions.get_loc(end) + 5) == 6


def test_walk_forward_folds_are_consecutive_and_only_train_on_the_past():
    sessions = session_index("2015-01-02", 1500)
    folds = walk_forward_folds(sessions, first_test_start=sessions[500], last_test_end=sessions[-1], n_folds=4, horizon=5, embargo=5)
    assert len(folds) == 4
    for earlier, later in zip(folds, folds[1:]):
        assert sessions.get_loc(later.test_start) == sessions.get_loc(earlier.test_end) + 1
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert sessions.get_loc(fold.test_start) - sessions.get_loc(fold.train_end) == 11
    assert folds[0].test_start == sessions[500] and folds[-1].test_end == sessions[-1]


def test_check_fold_catches_a_training_label_that_reaches_the_test_period():
    sessions = session_index("2020-01-02", 120)
    frame = pd.DataFrame({"symbol": "A", "label_end": sessions.to_series().shift(-5).to_numpy(), "target": 1}, index=sessions).dropna()
    good = holdout_fold(frame.index, holdout_start=sessions[80], horizon=5, embargo=5)
    assert check_fold(frame, good, 5, 5)["passed"]
    leaky = type(good)("leaky", sessions[78], sessions[80], sessions[-6])  # label of the last training row ends inside the test period
    report = check_fold(frame, leaky, 5, 5)
    assert not report["passed"] and report["latest_train_label_end"] > "2020-04-27"
    train, test = masks(frame, good)
    assert not (train & test).any()


# ------------------------------------------------------------------ labels
def test_forward_return_uses_exactly_h_sessions_and_rejects_windows_with_a_missing_bar():
    bars = make_ohlcv(60, seed=1, start="2024-01-02")
    fwd, end, clean = forward_return(bars["Close"], 5)
    assert fwd.iloc[0] == pytest.approx(bars["Close"].iloc[5] / bars["Close"].iloc[0] - 1)
    assert end.iloc[0] == bars.index[5] and clean.iloc[:-5].all() and not clean.iloc[-5:].any()
    gappy = bars.drop(bars.index[12])
    _, _, clean_gappy = forward_return(gappy["Close"], 5)
    affected = [i for i in range(len(gappy) - 5) if gappy.index[i] < bars.index[12] <= gappy.index[i + 5] or gappy.index[i] < bars.index[12] < gappy.index[i + 5]]
    assert affected and not clean_gappy.iloc[affected].any()
    assert clean_gappy.iloc[20:-5].all()


# ------------------------------------------------------------------ scoring
def test_metrics_on_a_small_known_example():
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.7, 0.2, 0.55])
    m = classification_metrics(y, p)
    assert m["confusion"] == {"tp": 4, "fp": 1, "tn": 2, "fn": 1}
    assert m["accuracy"] == pytest.approx(6 / 8) and m["precision"] == pytest.approx(4 / 5) and m["recall"] == pytest.approx(4 / 5)
    assert m["balanced_accuracy"] == pytest.approx((4 / 5 + 2 / 3) / 2, abs=1e-6)
    assert m["majority_class_accuracy"] == pytest.approx(5 / 8)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2), abs=1e-6)
    assert m["roc_auc"] == pytest.approx(13 / 15, abs=1e-6)  # 13 of the 15 (up, down) pairs are ranked correctly


def test_always_up_has_majority_accuracy_and_no_skill():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.56).astype(int)
    constant = np.full(5000, y.mean())
    m = classification_metrics(y, constant)
    assert m["accuracy"] == m["majority_class_accuracy"] and m["balanced_accuracy"] == 0.5 and m["roc_auc"] == 0.5
    assert brier_skill(y, constant, constant) == 0.0
    informed = np.clip(y * 0.2 + 0.46, 0, 1)
    assert brier_skill(y, informed, constant) > 0 and brier(y, informed) < brier(y, constant)


def test_reliability_is_near_zero_for_calibrated_probabilities_and_large_for_overconfident_ones():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.3, 0.8, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert reliability(y, p)["ece"] < 0.02
    overconfident = np.clip(0.5 + (p - 0.5) * 3, 0.01, 0.99)
    assert reliability(y, overconfident)["ece"] > 0.08


def test_block_bootstrap_is_reproducible_and_wider_than_a_row_bootstrap_for_clustered_data():
    dates = pd.DatetimeIndex(np.repeat(session_index("2020-01-02", 250), 30))
    rng = np.random.default_rng(2)
    weekly_effect = pd.Series(rng.normal(0, 1, 60), index=np.unique(week_blocks(dates))[:60] if len(np.unique(week_blocks(dates))) >= 60 else None) if False else None
    blocks = week_blocks(dates)
    effect = {b: rng.normal(0, 1) for b in np.unique(blocks)}
    values = np.array([effect[b] for b in blocks]) + rng.normal(0, 0.2, len(blocks))
    first = block_bootstrap(dates, lambda rows: float(values[rows].mean()), n_resamples=300, seed=5)
    again = block_bootstrap(dates, lambda rows: float(values[rows].mean()), n_resamples=300, seed=5)
    assert first == again and first["blocks"] == len(effect)
    naive_width = 2 * 1.96 * values.std() / np.sqrt(len(values))
    assert (first["high"] - first["low"]) > 3 * naive_width


# ------------------------------------------------------------------ calibration
def test_calibration_repairs_overconfident_probabilities_and_leaves_good_ones_alone():
    rng = np.random.default_rng(3)
    truth = rng.uniform(0.35, 0.75, 8000)
    y = (rng.random(8000) < truth).astype(int)
    order = np.arange(8000)
    overconfident = np.clip(0.5 + (truth - 0.5) * 3, 0.01, 0.99)
    calibrator, report = choose_calibration(overconfident, y, order)
    assert report["chosen"] in {"sigmoid", "isotonic"}
    assert brier(y, calibrator.transform(overconfident)) < brier(y, overconfident)
    _, kept = choose_calibration(truth, y, order)
    assert kept["brier_on_later_validation_rows"]["none"] <= min(kept["brier_on_later_validation_rows"].values()) + 5e-4
    assert Calibrator("none").transform(np.array([0.3]))[0] == pytest.approx(0.3)


# ------------------------------------------------------------------ uncertainty
def test_volatility_scaled_range_is_used_only_when_it_is_better():
    rng = np.random.default_rng(4)
    vol = np.exp(rng.normal(np.log(0.012), 0.6, 20000))
    returns = rng.normal(0, 1, 20000) * vol * np.sqrt(5)
    table = fit_uncertainty(returns[:10000], vol[:10000], 5)
    report = evaluate_uncertainty(table, returns[10000:], vol[10000:])
    assert report["supported"] and report["pinball_loss_scaled"] < report["pinball_loss_unscaled"]
    assert abs(report["coverage_of_80pct_range"]["scaled"] - 0.8) < 0.03
    scaled = predict_quantiles(table, 0.02, scaled=True)
    assert scaled["0.90"] > 0 > scaled["0.10"] and scaled["0.90"] == pytest.approx(table["scaled"]["0.90"] * 0.02 * np.sqrt(5))
    unrelated = evaluate_uncertainty(table, returns[10000:], rng.permutation(vol[10000:]))
    assert not unrelated["supported"]
    assert pinball_loss(np.array([1.0, 3.0]), 2.0, 0.5) == pytest.approx(0.5)


# ------------------------------------------------------------------ candidates
def test_baselines_come_first_and_the_prior_predicts_the_training_base_rate():
    pool = candidates(include_lightgbm=False)
    assert [c.name for c in pool[:2]] == [PRIOR, TRAILING] and all(c.is_baseline for c in pool[:2])
    assert {"logistic_regression", "random_forest", "gradient_boosting", "hist_gradient_boosting"} <= {c.name for c in pool}
    prior = pool[0].make({}, 0).fit(pd.DataFrame({"x": range(10)}), np.array([1] * 7 + [0] * 3))
    assert prior.predict_proba(pd.DataFrame({"x": range(4)}))[:, 1].tolist() == [0.7] * 4


# ------------------------------------------------------------------ the promotion rule
GOOD_BANDS = {"bands": [{"gap": 0.004}, {"gap": -0.011}], "miscalibrated_bands": 0}


def _intervals(skill_low, auc_low, skill=0.01):
    return {"brier_skill": {"point": skill, "low": skill_low, "high": 0.02}, "roc_auc": {"point": 0.55, "low": auc_low, "high": 0.6}}


def test_promotion_rule_needs_every_requirement():
    years = [{"segment": str(y), "n": 5000, "brier_skill": s} for y, s in zip(range(2016, 2022), [0.01, 0.02, -0.001, 0.01, 0.005, 0.02])]
    checks = [{"check": "x", "passed": True}]
    good = evaluate_rule(holdout_intervals=_intervals(0.002, 0.52), holdout_years=years, validation_skill=0.004, calibration=GOOD_BANDS, leakage_checks=checks, is_baseline=False)
    assert good["eligible"] and good["verdict"] == "eligible for promotion"
    cases = {
        "interval includes zero": dict(holdout_intervals=_intervals(-0.001, 0.52)),
        "auc interval reaches 0.5": dict(holdout_intervals=_intervals(0.002, 0.499)),
        "validation disagrees": dict(validation_skill=-0.0001),
        "badly calibrated": dict(calibration={"bands": [{"gap": 0.08}], "miscalibrated_bands": 1}),
        "leak": dict(leakage_checks=[{"check": "x", "passed": False}]),
        "baseline": dict(is_baseline=True),
        "too few good years": dict(holdout_years=[{**row, "brier_skill": -0.01} for row in years[:3]] + years[3:]),
    }
    base = dict(holdout_intervals=_intervals(0.002, 0.52), holdout_years=years, validation_skill=0.004, calibration=GOOD_BANDS, leakage_checks=checks, is_baseline=False)
    for name, change in cases.items():
        verdict = evaluate_rule(**{**base, **change})
        assert not verdict["eligible"], name
        assert "no predictive edge established" in verdict["verdict"]


# ------------------------------------------------------------------ the evaluation run
@pytest.fixture(scope="module")
def noise_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("noise")
    snapshot_id, symbols = make_snapshot(root / "snapshots", seed=10)
    config = EvalConfig(**QUICK)
    dataset, market_data, manifest = dataset_from_snapshot(snapshot_id, config, snapshot_dir=root / "snapshots")
    messages: list[str] = []
    card = run_evaluation(dataset, market_data, config, snapshot=manifest, model_dir=root / "models", progress=messages.append)
    return {"root": root, "card": card, "dataset": dataset, "market_data": market_data, "manifest": manifest, "config": config, "messages": messages, "symbols": symbols}


@pytest.fixture(scope="module")
def signal_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("signal")
    # A deliberately strong effect, far beyond anything in real prices, so the leak detector's
    # correlation limit is raised for this run only (and the card records that it was).
    snapshot_id, symbols = make_snapshot(root / "snapshots", seed=20, predictable=True, strength=0.4, symbols=12, periods=2100)
    config = EvalConfig(**{**QUICK, "holdout_start": "2020-07-01", "suspicious_correlation": 0.7})
    dataset, market_data, manifest = dataset_from_snapshot(snapshot_id, config, snapshot_dir=root / "snapshots")
    card = run_evaluation(dataset, market_data, config, snapshot=manifest, model_dir=root / "models", progress=lambda _: None)
    return {"root": root, "card": card, "dataset": dataset, "market_data": market_data, "symbols": symbols, "snapshot_id": snapshot_id}


def test_dataset_keeps_complete_rows_only_and_reports_what_it_dropped(noise_run):
    dataset: Dataset = noise_run["dataset"]
    report = dataset.report
    assert dataset.groups == ["price", "volume", "market", "vix"] and report["groups_absent"] == ["qqq", "sector", "macro"]
    assert dataset.X.notna().all().all() and set(dataset.y.unique()) == {0, 1}
    assert report["missing_value_policy"] == "complete rows only; nothing imputed"
    assert report["rows_complete"] == len(dataset.frame) and report["schema_problems"] == []
    assert list(dataset.X.columns) == schema.feature_names(dataset.groups)
    assert (dataset.frame["label_end"] > dataset.frame.index).all()


def test_no_model_is_promoted_when_prices_are_a_random_walk(noise_run):
    card = noise_run["card"]
    assert card["status"] == "evaluated" and card["promotion"]["eligible"] is False
    assert "no predictive edge established" in card["promotion"]["verdict"]
    assert all(check["passed"] for check in card["leakage_checks"]), card["leakage_checks"]
    with pytest.raises(Conflict) as refused:
        registry.promote(card["model_id"], model_dir=noise_run["root"] / "models")
    assert refused.value.code == "not_eligible"
    reg = registry.read_registry(noise_run["root"] / "models")
    assert reg["champion"] is None and reg["baseline"]["from_run"] == card["model_id"]
    assert 0.4 < reg["baseline"]["base_rate"] < 0.7


def test_the_card_is_complete_and_every_score_sits_beside_its_baseline(noise_run):
    card = noise_run["card"]
    json.dumps(card)  # plain JSON all the way down
    names = [row["candidate"] for row in card["holdout"]["candidates"]]
    assert names[:2] == [PRIOR, TRAILING] and card["selection"]["candidate"] in names[2:]
    for row in card["holdout"]["candidates"]:
        metrics = row["metrics"]
        for key in ("accuracy", "majority_class_accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss", "base_rate"):
            assert key in metrics
        assert "brier_skill_vs_prior" in row
    prior = card["holdout"]["candidates"][0]
    assert prior["brier_skill_vs_prior"] == 0.0 and prior["metrics"]["balanced_accuracy"] == 0.5
    assert card["schema"]["hash"] == schema.schema_hash() and card["data"]["snapshot_id"] == noise_run["manifest"]["id"]
    assert card["config"]["seed"] == noise_run["config"].seed and "scikit-learn" in card["environment"]
    assert card["selection"]["written_at"] <= card["holdout"]["evaluated_at"]
    prior_row = next(row for row in card["validation"] if row["candidate"] == PRIOR)
    assert prior_row["mean_fold_roc_auc"] == 0.5  # a constant cannot rank; pooled AUC across blocks would not show that
    predictions = pd.read_csv(noise_run["root"] / "models" / card["model_id"] / "predictions.csv.gz")
    assert set(predictions["split"]) == {"validation", "holdout"} and predictions["probability"].between(0, 1).all()
    assert len(predictions[predictions["split"] == "holdout"]) == card["splits"]["holdout"]["rows"]
    assert card["holdout"]["times_this_holdout_was_scored_before"] == 0
    assert set(card["holdout"]["by_regime"]) == {"year", "market_trend", "volatility"}
    assert card["splits"]["development"]["last"] < card["splits"]["holdout"]["first"]
    assert card["expected_return"]["served_as"].startswith("historical mean") or card["expected_return"]["supported"]


def test_a_second_run_on_the_same_holdout_is_counted(noise_run):
    card = run_evaluation(noise_run["dataset"], noise_run["market_data"], noise_run["config"], snapshot=noise_run["manifest"], model_dir=noise_run["root"] / "models", progress=lambda _: None)
    assert card["holdout"]["times_this_holdout_was_scored_before"] == 1
    assert card["holdout"]["earlier_runs"] == [noise_run["card"]["model_id"]]
    # same data, same seed: the same selection and the same holdout numbers
    assert card["selection"]["candidate"] == noise_run["card"]["selection"]["candidate"]
    assert card["holdout"]["selected"]["metrics"]["brier"] == noise_run["card"]["holdout"]["selected"]["metrics"]["brier"]


def test_development_stage_never_scores_the_holdout(noise_run, tmp_path):
    card = run_evaluation(noise_run["dataset"], noise_run["market_data"], noise_run["config"], snapshot=noise_run["manifest"], stage="development", model_dir=tmp_path, progress=lambda _: None)
    assert card["status"] == "development_only" and "holdout" not in card and "promotion" not in card
    assert not (tmp_path / "holdout_log.jsonl").exists()
    with pytest.raises(Conflict):
        registry.promote(card["model_id"], model_dir=tmp_path)


def test_a_planted_effect_is_found_and_the_model_can_then_be_promoted(signal_run):
    card = signal_run["card"]
    selected = card["holdout"]["selected"]
    assert selected["intervals"]["brier_skill"]["low"] > 0 and selected["intervals"]["roc_auc"]["low"] > 0.5
    assert card["promotion"]["eligible"], card["promotion"]["requirements"]
    registry.promote(card["model_id"], model_dir=signal_run["root"] / "models")
    assert registry.champion(signal_run["root"] / "models") == card["model_id"]


# ------------------------------------------------------------------ leakage detectors detect leaks
def test_leakage_checks_fail_when_a_leak_is_planted(noise_run):
    dataset: Dataset = noise_run["dataset"]
    leaky_frame = dataset.frame.copy()
    leaky_frame["ret_1"] = leaky_frame["fwd_return"] * 0.5 + leaky_frame["ret_1"] * 0.01  # a feature built from the future
    leaky = Dataset(leaky_frame, dataset.feature_names, dataset.groups, dataset.horizon, dataset.source)
    assert not leakage.no_implausible_correlation(leaky)["passed"]
    assert not leakage.rows_recompute_from_the_past(leaky, noise_run["market_data"], samples=3)["passed"]
    stray = dataset.frame.copy()
    stray["tomorrow_close"] = 1.0
    assert not leakage.columns_are_features(Dataset(stray, dataset.feature_names, dataset.groups, dataset.horizon, dataset.source))["passed"]
    doubled = Dataset(pd.concat([dataset.frame, dataset.frame.iloc[:5]]), dataset.feature_names, dataset.groups, dataset.horizon, dataset.source)
    assert not leakage.no_duplicate_rows(doubled)["passed"]
    assert not leakage.shuffled_label_control([0.71, 0.70, 0.72])["passed"]
    assert leakage.shuffled_label_control([0.505])["passed"] and leakage.shuffled_label_control([0.46, 0.53, 0.51, 0.47, 0.52])["passed"]
    assert leakage.rows_recompute_from_the_past(dataset, noise_run["market_data"], samples=3)["passed"]


# ------------------------------------------------------------------ serving
def _at_snapshot_end(market_data, symbol):
    last = market_data.history(symbol, interval="1d").data.index[-1].date()
    clock.set_clock(lambda: datetime(last.year, last.month, last.day, 23, 0, tzinfo=timezone.utc))
    return last


def test_without_a_promoted_model_the_baseline_is_served_and_says_so(noise_run):
    last = _at_snapshot_end(noise_run["market_data"], noise_run["symbols"][0])
    service = PredictionService(noise_run["market_data"], model_dir=noise_run["root"] / "models")
    out = service.predict(noise_run["symbols"][0])
    assert out["status"] == "baseline" and out["edge_established"] is False and out["model"]["kind"] == "baseline"
    assert out["probability_up"] == registry.read_registry(noise_run["root"] / "models")["baseline"]["base_rate"]
    assert "No predictive edge has been established" in out["message"] and out["authorises_trade"] is False
    assert out["as_of"] == last.isoformat() and out["window"]["last_session"] > out["window"]["first_session"] > out["as_of"]
    assert out["range"]["p10"] < out["range"]["p50"] < out["range"]["p90"]
    assert service.status()["edge_established"] is False


def test_a_promoted_model_serves_through_the_same_pipeline_as_training(signal_run):
    model_dir = signal_run["root"] / "models"
    if registry.champion(model_dir) is None:
        registry.promote(signal_run["card"]["model_id"], model_dir=model_dir)
    symbol = signal_run["symbols"][0]
    last = _at_snapshot_end(signal_run["market_data"], symbol)
    service = PredictionService(signal_run["market_data"], model_dir=model_dir)
    out = service.predict(symbol)
    assert out["status"] == "model" and out["model"]["id"] == signal_run["card"]["model_id"] and out["as_of"] == last.isoformat()
    assert 0 < out["probability_up"] < 1 and out["schema"]["hash"] == schema.schema_hash() and out["data_snapshot_id"].startswith("inputs-")
    assert out["data_source"]["provider"].startswith("snapshot:") and out["data_source"]["freshness"] == "historical"
    # an earlier as-of date: the live answer equals the model applied to the training row for that date
    frame = signal_run["dataset"].frame
    rows = frame[frame["symbol"] == symbol]
    day = rows.index[-30]
    bundle = registry.load_bundle(signal_run["card"]["model_id"], model_dir)
    training_row = rows.loc[[day], bundle["feature_names"]]
    expected = float(bundle["calibrator"].transform(bundle["estimator"].predict_proba(training_row)[:, 1])[0])
    assert service.predict(symbol, as_of=day.date())["probability_up"] == pytest.approx(expected, abs=1e-6)
    first, second = service.predict(symbol), service.predict(symbol)
    assert first["probability_up"] == second["probability_up"] and first["data_snapshot_id"] == second["data_snapshot_id"]  # deterministic; only cache age differs


def test_a_model_with_a_different_feature_schema_is_refused(signal_run, monkeypatch):
    model_dir = signal_run["root"] / "models"
    if registry.champion(model_dir) is None:
        registry.promote(signal_run["card"]["model_id"], model_dir=model_dir)
    monkeypatch.setattr(schema, "SCHEMA_VERSION", "2.0.0")  # the code's features changed after training
    with pytest.raises(SchemaMismatch):
        registry.load_bundle(signal_run["card"]["model_id"], model_dir)
    symbol = signal_run["symbols"][0]
    _at_snapshot_end(signal_run["market_data"], symbol)
    out = PredictionService(signal_run["market_data"], model_dir=model_dir).predict(symbol)
    assert out["status"] == "baseline" and out["model"]["error"] == "feature_schema_mismatch" and out["edge_established"] is False
    status = PredictionService(signal_run["market_data"], model_dir=model_dir).status()
    assert status["edge_established"] is False and status["model_error"] == "feature_schema_mismatch"


def test_a_changed_model_file_is_refused(signal_run):
    model_dir = signal_run["root"] / "models"
    path = model_dir / signal_run["card"]["model_id"] / "model.joblib"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"x")
        with pytest.raises(ModelUnavailable) as refused:
            registry.load_bundle(signal_run["card"]["model_id"], model_dir)
        assert refused.value.code == "model_file_changed"
    finally:
        path.write_bytes(original)
    assert registry.load_bundle(signal_run["card"]["model_id"], model_dir)["model_id"] == signal_run["card"]["model_id"]


def test_no_forecast_from_stale_or_short_history(noise_run):
    symbol = noise_run["symbols"][0]
    last = noise_run["market_data"].history(symbol, interval="1d").data.index[-1].date()
    service = PredictionService(noise_run["market_data"], model_dir=noise_run["root"] / "models")
    clock.set_clock(lambda: datetime(last.year + 1, last.month, min(last.day, 28), 23, 0, tzinfo=timezone.utc))  # a year after the data ends
    out = service.predict(symbol)
    assert out["status"] == "unavailable" and out["probability_up"] is None and out["decision_default"] == "HOLD"
    assert out["reason"]["code"] in {"stale_data", "data_unavailable"}
    first = noise_run["market_data"].history(symbol, interval="1d").data.index[150].date()
    clock.set_clock(lambda: datetime(first.year, first.month, first.day, 23, 0, tzinfo=timezone.utc))
    short = service.predict(symbol)
    assert short["status"] == "unavailable" and short["reason"]["code"] == "feature_unavailable"
