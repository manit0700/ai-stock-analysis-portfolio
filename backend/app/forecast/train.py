"""The evaluation run: development first, then the holdout, once.

Order of work, which is the point of this module:

 1. Split off the holdout by date. Nothing below touches it until step 7.
 2. Walk-forward validation of every candidate on development data, with purged splits.
 3. Choose one candidate and its settings by pooled validation Brier score.
 4. Choose a calibration method on validation predictions.
 5. Run the leakage checks.
 6. Write the selection to disk.
 7. Fit each candidate once on all development data and score the holdout.
 8. Apply the pre-written promotion rule to the selected candidate.

Every number in the resulting model card is beside its baseline. The card says which data
snapshot, feature schema, random seed and library versions produced it, and how many times this
holdout has been scored before.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.core.clock import iso, utcnow
from app.core.config import get_settings
from app.features import schema
from app.forecast import leakage, registry
from app.forecast.calibration import choose_calibration
from app.forecast.dataset import Dataset, build_dataset, target_text
from app.forecast.evaluate import (
    brier, brier_skill, block_bootstrap, by_segment, calibration_bands, classification_metrics, regime_labels, reliability, skill_intervals,
)
from app.forecast.models import PRIOR, Candidate, candidates, inputs_for, predict_up
from app.forecast.promotion import evaluate_rule
from app.forecast.splits import Fold, holdout_fold, masks, walk_forward_folds
from app.forecast.uncertainty import evaluate_uncertainty, fit_uncertainty

LIMITATIONS = [
    "Skill in forecasting direction is not profit. Costs, the size of moves and position sizing are not part of these scores.",
    "Rows are correlated across stocks and overlap in time; intervals resample calendar weeks to allow for this, which is an approximation.",
    "The model was fitted through the end of the development period and has not been refitted on later data.",
]


@dataclass
class EvalConfig:
    horizon: int = 5
    embargo: int = 5
    first_validation_start: str = "2008-01-01"
    holdout_start: str = "2016-01-01"
    n_folds: int = 8
    seed: int = 20261009
    n_bootstrap: int = 1000
    quick: bool = False
    min_group_coverage: float = 0.9
    groups: list[str] | None = None
    include_lightgbm: bool | None = None
    leakage_samples: int = 12
    shuffle_repeats: int = 5
    #: Largest rank correlation between a feature and the future return that is not treated as a leak.
    suspicious_correlation: float = 0.20
    notes: list[str] = field(default_factory=list)


def jsonable(value: Any) -> Any:
    """Plain Python values only, so a card can be written as JSON."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def snapshot_market_data(snapshot_id: str, snapshot_dir: Path | None = None):
    """A ``MarketData`` that serves one snapshot and nothing else: what training reads from."""
    from app.marketdata.cache import TTLCache
    from app.marketdata.service import MarketData
    from app.marketdata.snapshots import SnapshotProvider, SnapshotStore

    store = SnapshotStore(snapshot_dir or get_settings().snapshot_dir)
    provider = SnapshotProvider(store, snapshot_id)
    order = {"history": [provider.name], "economic": [provider.name], "quote": [], "profile": [], "actions": [], "news": [], "earnings": []}
    return MarketData([provider], order=order, cache=TTLCache(maxsize=4096), disk_cache=None, retries=0), store.manifest(snapshot_id)


def dataset_from_snapshot(snapshot_id: str, config: EvalConfig, *, symbols: list[str] | None = None, snapshot_dir: Path | None = None) -> tuple[Dataset, Any, dict[str, Any]]:
    from app.marketdata.snapshots import SnapshotStore

    market_data, manifest = snapshot_market_data(snapshot_id, snapshot_dir)
    changed = SnapshotStore(snapshot_dir or get_settings().snapshot_dir).verify(snapshot_id)
    if changed:
        raise ValueError(f"Snapshot {snapshot_id} has files that no longer match its manifest: {changed[:5]}")
    context_only = set((manifest.get("metadata") or {}).get("context_only", []))
    universe = symbols or [s for s in manifest["symbols"] if s not in context_only]
    include_macro = "DGS10" in manifest.get("series", [])
    dataset = build_dataset(
        market_data, universe, horizon=config.horizon, groups=config.groups, min_group_coverage=config.min_group_coverage,
        source=f"snapshot:{snapshot_id}", include_macro=include_macro,
    )
    return dataset, market_data, manifest


def _fit_predict(candidate: Candidate, params: dict, seed: int, train: pd.DataFrame, test: pd.DataFrame, names: list[str], y_train: np.ndarray | None = None) -> np.ndarray:
    estimator = candidate.make(params, seed)
    estimator.fit(inputs_for(candidate, train, names), train["target"].to_numpy(int) if y_train is None else y_train)
    return predict_up(estimator, inputs_for(candidate, test, names))


def walk_forward(candidate: Candidate, params: dict, frame: pd.DataFrame, folds: list[Fold], names: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities for every row in a validation block. Returns ``(positions, probabilities)``."""
    positions, probabilities = [], []
    for fold in folds:
        train_mask, test_mask = masks(frame, fold)
        probabilities.append(_fit_predict(candidate, params, seed, frame[train_mask], frame[test_mask], names))
        positions.append(np.flatnonzero(test_mask))
    return np.concatenate(positions), np.concatenate(probabilities)


def fold_mean_auc(y: np.ndarray, p: np.ndarray, sizes: list[int]) -> float | None:
    """ROC-AUC computed inside each validation block, then averaged by rows.

    Pooling blocks first would be wrong: a model whose average output differs from block to block
    (even a constant refitted per block) gets a pooled AUC away from 0.5 with no ability to rank
    stocks within any block.
    """
    from sklearn.metrics import roc_auc_score

    total, weight, offset = 0.0, 0, 0
    for n in sizes:
        rows = slice(offset, offset + n)
        offset += n
        if len(np.unique(y[rows])) < 2:
            continue
        total += (float(roc_auc_score(y[rows], p[rows])) if np.ptp(p[rows]) > 0 else 0.5) * n
        weight += n
    return round(total / weight, 6) if weight else None


def _environment() -> dict[str, str]:
    import sklearn

    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    try:
        import lightgbm

        env["lightgbm"] = lightgbm.__version__
    except Exception:  # noqa: BLE001
        env["lightgbm"] = "not installed"
    return env


def _holdout_log(model_dir: Path) -> Path:
    return model_dir / "holdout_log.jsonl"


def _previous_holdout_runs(model_dir: Path, key: str) -> list[dict[str, Any]]:
    path = _holdout_log(model_dir)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("holdout_key") == key]


def expected_return_check(frame: pd.DataFrame, folds: list[Fold], names: list[str], seed: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Can a model beat "the average return" at predicting the size of the 5-session return? Walk-forward R²."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    positions, predicted, naive = [], [], []
    for fold in folds:
        train_mask, test_mask = masks(frame, fold)
        train, test = frame[train_mask], frame[test_mask]
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.02, max_depth=3, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=seed)
        model.fit(train[names], train["fwd_return"])
        predicted.append(model.predict(test[names]))
        naive.append(np.full(len(test), float(train["fwd_return"].mean())))
        positions.append(np.flatnonzero(test_mask))
    positions, predicted, naive = np.concatenate(positions), np.concatenate(predicted), np.concatenate(naive)
    actual = frame["fwd_return"].to_numpy(float)[positions]
    r2 = 1.0 - float(np.sum((actual - predicted) ** 2) / np.sum((actual - naive) ** 2))
    return {"validation_r2_vs_historical_mean": round(r2, 6), "rows": int(len(positions))}, positions, predicted, naive


def run_evaluation(
    dataset: Dataset,
    market_data: Any,
    config: EvalConfig,
    *,
    snapshot: dict[str, Any] | None = None,
    stage: str = "all",
    model_dir: Path | None = None,
    progress: Callable[[str], None] = print,
    save: bool = True,
) -> dict[str, Any]:
    """Run the evaluation described at the top of this module and return the model card."""
    if stage not in {"development", "all"}:
        raise ValueError("stage must be 'development' or 'all'")
    started = time.time()
    model_dir = Path(model_dir) if model_dir is not None else get_settings().model_dir
    frame, names, horizon = dataset.frame, dataset.feature_names, dataset.horizon
    snapshot_id = (snapshot or {}).get("id")

    # 1 ---------------------------------------------------------------- split off the holdout
    hold = holdout_fold(frame.index, holdout_start=config.holdout_start, horizon=horizon, embargo=config.embargo)
    dev_mask, hold_mask = masks(frame, hold)
    dev, holdout = frame[dev_mask], frame[hold_mask]
    progress(f"development rows {len(dev):,} ({dev.index.min().date()} to {dev.index.max().date()}); holdout rows {len(holdout):,} ({holdout.index.min().date()} to {holdout.index.max().date()}); "
             f"{int((~dev_mask & ~hold_mask).sum()):,} rows in the purge gap are unused")

    # 2 ---------------------------------------------------------------- walk-forward validation
    folds = walk_forward_folds(dev.index, first_test_start=config.first_validation_start, last_test_end=dev.index.max(), n_folds=config.n_folds, horizon=horizon, embargo=config.embargo)
    pool = candidates(quick=config.quick, include_lightgbm=config.include_lightgbm)
    y_dev = dev["target"].to_numpy(int)
    prior = next(c for c in pool if c.name == PRIOR)
    oof_positions, prior_oof = walk_forward(prior, {}, dev, folds, names, config.seed)
    y_oof = y_dev[oof_positions]
    oof_dates = dev.index[oof_positions]
    fold_sizes = [int(masks(dev, fold)[1].sum()) for fold in folds]

    validation: list[dict[str, Any]] = []
    oof_store: dict[tuple[str, str], np.ndarray] = {}
    for candidate in pool:
        for params in candidate.grid:
            t0 = time.time()
            positions, p = walk_forward(candidate, params, dev, folds, names, config.seed)
            assert np.array_equal(positions, oof_positions)
            key = (candidate.name, json.dumps(params, sort_keys=True))
            oof_store[key] = p
            per_fold = []
            offset = 0
            for fold in folds:
                n = int(masks(dev, fold)[1].sum())
                rows = slice(offset, offset + n)
                per_fold.append({"fold": fold.name, "n": n, "brier": round(brier(y_oof[rows], p[rows]), 6), "brier_skill_vs_prior": round(brier_skill(y_oof[rows], p[rows], prior_oof[rows]), 6)})
                offset += n
            metrics = classification_metrics(y_oof, p)
            row = {
                "candidate": candidate.name, "label": candidate.label, "params": params, "is_baseline": candidate.is_baseline,
                "pooled": metrics, "mean_fold_roc_auc": fold_mean_auc(y_oof, p, fold_sizes), "brier_skill_vs_prior": round(brier_skill(y_oof, p, prior_oof), 6),
                "folds_with_positive_skill": int(sum(f["brier_skill_vs_prior"] > 0 for f in per_fold)), "folds": per_fold, "seconds": round(time.time() - t0, 1),
            }
            validation.append(row)
            progress(f"  validation {candidate.name:26s} {json.dumps(params):42s} brier {metrics['brier']:.5f}  skill {row['brier_skill_vs_prior']:+.5f}  auc(within folds) {row['mean_fold_roc_auc']:.4f}  ({row['seconds']}s)")

    # 3 ---------------------------------------------------------------- choose one candidate
    best_per_candidate: dict[str, dict[str, Any]] = {}
    for row in validation:
        current = best_per_candidate.get(row["candidate"])
        if current is None or row["pooled"]["brier"] < current["pooled"]["brier"]:
            best_per_candidate[row["candidate"]] = row
    model_rows = [row for row in best_per_candidate.values() if not row["is_baseline"]]
    chosen_row = min(model_rows, key=lambda row: row["pooled"]["brier"])
    chosen = next(c for c in pool if c.name == chosen_row["candidate"])
    chosen_oof = oof_store[(chosen.name, json.dumps(chosen_row["params"], sort_keys=True))]
    progress(f"selected on development data: {chosen.name} {chosen_row['params']} (validation Brier skill {chosen_row['brier_skill_vs_prior']:+.5f})")

    # 4 ---------------------------------------------------------------- calibration
    order = np.argsort(oof_dates.to_numpy(), kind="stable")
    calibrator, calibration_report = choose_calibration(chosen_oof, y_oof, order)
    validation_interval = block_bootstrap(oof_dates, lambda rows: brier_skill(y_oof[rows], chosen_oof[rows], prior_oof[rows]), n_resamples=config.n_bootstrap, seed=config.seed)

    # 5 ---------------------------------------------------------------- leakage checks
    dev_dataset = Dataset(dev, names, dataset.groups, horizon, dataset.source)
    shuffle_rng = np.random.default_rng(config.seed)
    shuffled_aucs = []
    for _ in range(max(1, config.shuffle_repeats)):  # the selected model, walk-forward, with its training labels shuffled
        parts = []
        for fold in folds:
            train_mask, test_mask = masks(dev, fold)
            train = dev[train_mask]
            parts.append(_fit_predict(chosen, chosen_row["params"], config.seed, train, dev[test_mask], names, y_train=shuffle_rng.permutation(train["target"].to_numpy(int))))
        shuffled_aucs.append(fold_mean_auc(y_oof, np.concatenate(parts), fold_sizes) or 0.5)

    full_dataset = Dataset(frame, names, dataset.groups, horizon, dataset.source)
    checks = [
        leakage.columns_are_features(dataset),
        leakage.no_implausible_correlation(dev_dataset, config.suspicious_correlation),
        leakage.folds_are_purged(full_dataset, [*folds, hold], config.embargo),
        leakage.no_duplicate_rows(dataset),
        leakage.rows_recompute_from_the_past(dev_dataset, market_data, samples=config.leakage_samples, seed=config.seed),
        leakage.shuffled_label_control(shuffled_aucs),
    ]
    for check in checks:
        progress(f"  leakage check {'PASS' if check['passed'] else 'FAIL'}: {check['check']} ({check['detail'][:110]})")

    # uncertainty and expected return, from development data only
    uncertainty_table = fit_uncertainty(dev["fwd_return"].to_numpy(float), dev["vol_21"].to_numpy(float) if "vol_21" in dev else np.full(len(dev), np.nan), horizon)
    expected_report, er_positions, er_predicted, er_naive = expected_return_check(dev, folds, names, config.seed)
    vix_cuts = tuple(float(v) for v in np.quantile(dev["vix_level"], [1 / 3, 2 / 3])) if "vix_level" in dev else None

    # 6 ---------------------------------------------------------------- write the selection before the holdout
    config_dict = asdict(config)
    fingerprint = hashlib.sha256(json.dumps({"config": config_dict, "snapshot": snapshot_id, "schema": schema.schema_hash(), "features": names}, sort_keys=True).encode()).hexdigest()[:10]
    model_id = f"mv{horizon}d-{utcnow().strftime('%Y%m%d-%H%M%S')}-{fingerprint}"
    selection = {
        "candidate": chosen.name, "label": chosen.label, "params": chosen_row["params"], "criterion": "lowest pooled walk-forward validation Brier score among models",
        "validation_brier": chosen_row["pooled"]["brier"], "validation_brier_skill_vs_prior": chosen_row["brier_skill_vs_prior"],
        "validation_brier_skill_interval": validation_interval, "folds_with_positive_skill": chosen_row["folds_with_positive_skill"], "n_folds": len(folds),
        "calibration": calibration_report, "written_at": iso(utcnow()),
    }
    card: dict[str, Any] = {
        "model_id": model_id,
        "created_at": iso(utcnow()),
        "status": "development_only",
        "target": {"horizon_sessions": horizon, "definition": target_text(horizon), "positive_class": "5-session forward return above zero" if horizon == 5 else f"{horizon}-session forward return above zero"},
        "data": {"source": dataset.source, "snapshot_id": snapshot_id, "snapshot_content_sha256": (snapshot or {}).get("content_sha256"), "snapshot_limits": ((snapshot or {}).get("metadata") or {}).get("limits", []), **dataset.report},
        "schema": {"name": schema.SCHEMA_NAME, "version": schema.SCHEMA_VERSION, "hash": schema.schema_hash()},
        "features": names,
        "feature_groups": dataset.groups,
        "config": config_dict,
        "splits": {
            "development": {"first": dev.index.min(), "last": dev.index.max(), "rows": len(dev), "base_rate": round(float(dev["target"].mean()), 6)},
            "holdout": {"first": holdout.index.min(), "last": holdout.index.max(), "rows": len(holdout), "start": hold.test_start},
            "purge": f"training rows must have their {horizon}-session label window end more than {config.embargo} sessions before the test period starts",
            "validation_folds": [{"fold": f.name, "train_end": f.train_end, "test_start": f.test_start, "test_end": f.test_end} for f in folds],
        },
        "validation": validation,
        "selection": selection,
        "leakage_checks": checks,
        "uncertainty": {"table": uncertainty_table},
        "expected_return": expected_report,
        "regime_definition": {"vix_tercile_cuts_from_development": vix_cuts},
        "environment": _environment(),
        "limitations": [*LIMITATIONS, *config.notes],
    }
    if save:
        pending = model_dir / "_selections"
        pending.mkdir(parents=True, exist_ok=True)
        (pending / f"{model_id}.json").write_text(json.dumps(jsonable(selection), indent=2), encoding="utf-8")
    if stage == "development":
        card["seconds"] = round(time.time() - started, 1)
        card = jsonable(card)
        if save:
            registry.save_run(model_id, card, None, model_dir)
        return card

    # 7 ---------------------------------------------------------------- the holdout, once
    holdout_key = hashlib.sha256(json.dumps({"snapshot": snapshot_id, "source": dataset.source, "start": str(hold.test_start.date()), "horizon": horizon}, sort_keys=True).encode()).hexdigest()[:16]
    earlier = _previous_holdout_runs(model_dir, holdout_key) if save else []
    y_hold = holdout["target"].to_numpy(int)
    hold_results: list[dict[str, Any]] = []
    hold_probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    for candidate in pool:
        row = best_per_candidate[candidate.name]
        estimator = candidate.make(row["params"], config.seed)
        estimator.fit(inputs_for(candidate, dev, names), y_dev)
        hold_probabilities[candidate.name] = predict_up(estimator, inputs_for(candidate, holdout, names))
        fitted[candidate.name] = estimator
    p_prior = hold_probabilities[PRIOR]
    p_selected_raw = hold_probabilities[chosen.name]
    p_selected = calibrator.transform(p_selected_raw)
    for candidate in pool:
        p = hold_probabilities[candidate.name]
        metrics = classification_metrics(y_hold, p)
        entry = {"candidate": candidate.name, "label": candidate.label, "params": best_per_candidate[candidate.name]["params"], "is_baseline": candidate.is_baseline,
                 "selected": candidate.name == chosen.name, "metrics": metrics, "brier_skill_vs_prior": round(brier_skill(y_hold, p, p_prior), 6)}
        if candidate.name != PRIOR:
            entry["intervals"] = skill_intervals(holdout.index, y_hold, p, p_prior, n_resamples=config.n_bootstrap, seed=config.seed)
        hold_results.append(entry)
        progress(f"  holdout    {candidate.name:26s} acc {metrics['accuracy']:.4f} (majority {metrics['majority_class_accuracy']:.4f})  bal-acc {metrics['balanced_accuracy']:.4f}  auc {metrics['roc_auc']:.4f}  brier {metrics['brier']:.5f}  skill {entry['brier_skill_vs_prior']:+.5f}")

    selected_metrics = classification_metrics(y_hold, p_selected)
    selected_intervals = skill_intervals(holdout.index, y_hold, p_selected, p_prior, n_resamples=config.n_bootstrap, seed=config.seed)
    selected_reliability = reliability(y_hold, p_selected)
    selected_calibration = calibration_bands(holdout.index, y_hold, p_selected, n_resamples=min(config.n_bootstrap, 500), seed=config.seed + 11)
    labels = regime_labels(holdout, vix_cuts)
    regimes = {name: by_segment(holdout, y_hold, p_selected, p_prior, series) for name, series in labels.items()}

    uncertainty_eval = evaluate_uncertainty(uncertainty_table, holdout["fwd_return"].to_numpy(float), holdout["vol_21"].to_numpy(float)) if "vol_21" in holdout else {"supported": False, "reason": "vol_21 is not among the features"}
    from sklearn.ensemble import HistGradientBoostingRegressor

    regressor = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.02, max_depth=3, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=config.seed)
    regressor.fit(dev[names], dev["fwd_return"])
    actual = holdout["fwd_return"].to_numpy(float)
    predicted_return = regressor.predict(holdout[names])
    naive_return = np.full(len(holdout), float(dev["fwd_return"].mean()))
    r2_interval = block_bootstrap(
        holdout.index, lambda rows: 1.0 - float(np.sum((actual[rows] - predicted_return[rows]) ** 2) / np.sum((actual[rows] - naive_return[rows]) ** 2)),
        n_resamples=config.n_bootstrap, seed=config.seed + 7,
    )
    expected_supported = bool(expected_report["validation_r2_vs_historical_mean"] > 0 and r2_interval["low"] is not None and r2_interval["low"] > 0)
    expected_report = {
        **expected_report, "holdout_r2_vs_historical_mean": r2_interval, "supported": expected_supported,
        "served_as": "model estimate" if expected_supported else "historical mean of development data (no model beat it)",
        "historical_mean_return": round(float(dev["fwd_return"].mean()), 6),
        "rule": "a model's expected return is served only if its R-squared against the historical mean is above 0 in validation and its 95% holdout interval is above 0",
    }

    # 8 ---------------------------------------------------------------- the promotion rule
    promotion = evaluate_rule(
        holdout_intervals=selected_intervals, holdout_years=regimes["year"], validation_skill=chosen_row["brier_skill_vs_prior"],
        calibration=selected_calibration, leakage_checks=checks, is_baseline=chosen.is_baseline,
    )
    progress(f"holdout, selected model after calibration ({calibration_report['chosen']}): brier skill {selected_intervals['brier_skill']['point']:+.5f} "
             f"[{selected_intervals['brier_skill']['low']:+.5f}, {selected_intervals['brier_skill']['high']:+.5f}], auc {selected_intervals['roc_auc']['point']:.4f} "
             f"[{selected_intervals['roc_auc']['low']:.4f}, {selected_intervals['roc_auc']['high']:.4f}]")
    progress(f"verdict: {promotion['verdict']}")

    card.update({
        "status": "evaluated",
        "holdout": {
            "evaluated_at": iso(utcnow()),
            "times_this_holdout_was_scored_before": len(earlier),
            "earlier_runs": [row.get("model_id") for row in earlier],
            "candidates": hold_results,
            "selected": {"candidate": chosen.name, "calibration": calibration_report["chosen"], "metrics": selected_metrics, "intervals": selected_intervals,
                         "brier_skill_vs_prior": round(brier_skill(y_hold, p_selected, p_prior), 6), "reliability": selected_reliability, "calibration_bands": selected_calibration},
            "by_regime": regimes,
        },
        "uncertainty": {"table": uncertainty_table, "holdout_evaluation": uncertainty_eval},
        "expected_return": expected_report,
        "promotion": promotion,
        "trained_through": dev.index.max(),
        "seconds": round(time.time() - started, 1),
    })
    card = jsonable(card)
    bundle = {
        "model_id": model_id, "estimator": fitted[chosen.name], "calibrator": calibrator, "feature_names": list(names), "groups": list(dataset.groups),
        "schema": card["schema"], "horizon": horizon, "base_rate": float(dev["target"].mean()), "trained_through": card["trained_through"],
        "uncertainty": uncertainty_table, "uncertainty_scaled_supported": bool(uncertainty_eval.get("supported")),
        "expected_return": {"supported": expected_supported, "historical_mean": float(dev["fwd_return"].mean()), "regressor": regressor if expected_supported else None},
        "snapshot_id": snapshot_id,
    }
    if save:
        folder = registry.save_run(model_id, card, bundle, model_dir)
        # Out-of-sample probabilities of the selected model, row by row, for backtests and audits.
        rows_validation = pd.DataFrame({
            "date": oof_dates.strftime("%Y-%m-%d"), "symbol": dev["symbol"].to_numpy()[oof_positions], "split": "validation",
            "probability": calibrator.transform(chosen_oof), "probability_raw": chosen_oof, "base_rate": prior_oof,
            "target": y_oof, "fwd_return": dev["fwd_return"].to_numpy(float)[oof_positions],
        })
        rows_holdout = pd.DataFrame({
            "date": holdout.index.strftime("%Y-%m-%d"), "symbol": holdout["symbol"].to_numpy(), "split": "holdout",
            "probability": p_selected, "probability_raw": p_selected_raw, "base_rate": p_prior, "target": y_hold, "fwd_return": holdout["fwd_return"].to_numpy(float),
        })
        pd.concat([rows_validation, rows_holdout]).to_csv(folder / "predictions.csv.gz", index=False, float_format="%.6f")
        with _holdout_log(model_dir).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": iso(utcnow()), "holdout_key": holdout_key, "model_id": model_id, "snapshot": snapshot_id, "holdout_start": str(hold.test_start.date()), "selected": chosen.name, "eligible": promotion["eligible"]}) + "\n")
        registry.record_baseline({
            "from_run": model_id, "horizon": horizon, "base_rate": round(float(dev["target"].mean()), 6), "data": dataset.source,
            "development_period": [card["splits"]["development"]["first"], card["splits"]["development"]["last"]],
            "uncertainty": uncertainty_table, "uncertainty_scaled_supported": bool(uncertainty_eval.get("supported")),
            "historical_mean_return": float(dev["fwd_return"].mean()), "verdict": promotion["verdict"],
        }, model_dir)
    return card
