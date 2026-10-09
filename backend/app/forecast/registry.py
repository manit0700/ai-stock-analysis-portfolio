"""Where trained models live, and the gate between "trained" and "allowed to serve".

    <model dir>/registry.json            which model serves (the champion), the baseline, and a log
    <model dir>/<model id>/card.json     everything about one evaluation run
    <model dir>/<model id>/model.joblib  the fitted estimator, calibrator and uncertainty tables

Three refusals are built in:

* ``promote`` refuses a model whose card does not say it passed the pre-written promotion rule.
* ``load_bundle`` refuses a model whose feature schema hash differs from the running code's
  (``SchemaMismatch``). This is the guard against training/serving drift.
* ``load_bundle`` refuses a file whose SHA-256 differs from the one recorded in its card, and a
  model trained with a different scikit-learn release.

Model files are Python pickles. Only files written by this code into the configured model
directory are ever loaded; never load a model file obtained from somewhere else.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.core.clock import iso, utcnow
from app.core.config import get_settings
from app.core.errors import Conflict, ModelUnavailable, NotFound, SchemaMismatch, ValidationFailed
from app.features import schema

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


def _dir(model_dir: Path | None) -> Path:
    return Path(model_dir) if model_dir is not None else get_settings().model_dir


def _safe(model_id: str) -> str:
    if not _ID.match(model_id or ""):
        raise ValidationFailed(f"{model_id!r} is not a valid model id.")
    return model_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_registry(model_dir: Path | None = None) -> dict[str, Any]:
    path = _dir(model_dir) / "registry.json"
    if not path.exists():
        return {"champion": None, "baseline": None, "events": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(data: dict[str, Any], model_dir: Path | None) -> None:
    directory = _dir(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "registry.json.tmp"
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(directory / "registry.json")


def save_run(model_id: str, card: dict[str, Any], bundle: dict[str, Any] | None, model_dir: Path | None = None) -> Path:
    """Write a run's card and, if there is one, its fitted bundle. Never overwrites a run."""
    import joblib

    folder = _dir(model_dir) / _safe(model_id)
    if folder.exists():
        raise Conflict(f"A run called {model_id} already exists.")
    folder.mkdir(parents=True)
    if bundle is not None:
        joblib.dump(bundle, folder / "model.joblib", compress=3)
        card = {**card, "artifact": {"file": "model.joblib", "sha256": _sha256(folder / "model.joblib"), "bytes": (folder / "model.joblib").stat().st_size}}
    (folder / "card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return folder


def read_card(model_id: str, model_dir: Path | None = None) -> dict[str, Any]:
    path = _dir(model_dir) / _safe(model_id) / "card.json"
    if not path.exists():
        raise NotFound(f"There is no model run called {model_id}.")
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(model_dir: Path | None = None) -> list[dict[str, Any]]:
    directory = _dir(model_dir)
    champion = read_registry(model_dir).get("champion")
    rows = []
    for path in sorted(directory.glob("*/card.json")) if directory.exists() else []:
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append({
            "model_id": card.get("model_id"), "created_at": card.get("created_at"), "selected": (card.get("selection") or {}).get("candidate"),
            "eligible": (card.get("promotion") or {}).get("eligible"), "verdict": (card.get("promotion") or {}).get("verdict"),
            "snapshot": (card.get("data") or {}).get("snapshot_id"), "champion": card.get("model_id") == champion,
        })
    return rows


def check_schema(recorded: dict[str, Any], feature_names: list[str]) -> None:
    """Raise ``SchemaMismatch`` unless the model's schema is exactly the one this code computes."""
    current = schema.schema_hash()
    if recorded.get("hash") != current:
        raise SchemaMismatch(
            "The model was trained on a different feature schema from the one this code computes, so it is not used.",
            details={"model_schema": {"version": recorded.get("version"), "hash": recorded.get("hash")}, "running_schema": {"version": schema.SCHEMA_VERSION, "hash": current}},
        )
    known = schema.all_feature_names()
    unknown = [name for name in feature_names if name not in known]
    if unknown:
        raise SchemaMismatch("The model expects features this code does not compute.", details={"unknown_features": unknown})
    positions = [known.index(name) for name in feature_names]
    if positions != sorted(positions) or len(set(feature_names)) != len(feature_names):
        raise SchemaMismatch("The model's feature order differs from the schema's order.", details={"features": feature_names})


def load_bundle(model_id: str, model_dir: Path | None = None) -> dict[str, Any]:
    """Load a fitted model after verifying its file hash, library version and feature schema."""
    import joblib
    import sklearn

    card = read_card(model_id, model_dir)
    artifact = card.get("artifact")
    path = _dir(model_dir) / _safe(model_id) / "model.joblib"
    if not artifact or not path.exists():
        raise ModelUnavailable(f"Run {model_id} has no fitted model file.")
    if _sha256(path) != artifact["sha256"]:
        raise ModelUnavailable(f"The model file for {model_id} does not match the hash in its card and is not loaded.", code="model_file_changed")
    check_schema(card["schema"], card["features"])
    trained_with = (card.get("environment") or {}).get("scikit-learn", "")
    if trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ModelUnavailable(
            f"Run {model_id} was trained with scikit-learn {trained_with}; this process runs {sklearn.__version__}. Retrain, or install the matching release.",
            code="library_version_mismatch",
        )
    bundle = joblib.load(path)
    if list(bundle.get("feature_names", [])) != list(card["features"]):
        raise SchemaMismatch("The model file and its card list different features.")
    return bundle


def promote(model_id: str, *, by: str = "owner", model_dir: Path | None = None) -> dict[str, Any]:
    """Make ``model_id`` the champion. Refused unless its card records that it passed the rule."""
    card = read_card(model_id, model_dir)
    promotion = card.get("promotion") or {}
    if not promotion.get("eligible"):
        raise Conflict(
            "This model did not pass the promotion rule and cannot serve predictions.",
            details={"model_id": model_id, "verdict": promotion.get("verdict"), "failed": [r["requirement"] for r in promotion.get("requirements", []) if not r["passed"]]},
            code="not_eligible",
        )
    load_bundle(model_id, model_dir)  # must be loadable by the running code
    registry = read_registry(model_dir)
    registry["events"].append({"at": iso(utcnow()), "event": "promote", "model_id": model_id, "previous": registry.get("champion"), "by": by})
    registry["champion"] = model_id
    _write_registry(registry, model_dir)
    return registry


def demote(*, by: str = "owner", reason: str = "", model_dir: Path | None = None) -> dict[str, Any]:
    """Remove the champion. Predictions fall back to the baseline."""
    registry = read_registry(model_dir)
    registry["events"].append({"at": iso(utcnow()), "event": "demote", "model_id": registry.get("champion"), "by": by, "reason": reason})
    registry["champion"] = None
    _write_registry(registry, model_dir)
    return registry


def record_baseline(baseline: dict[str, Any], model_dir: Path | None = None) -> None:
    """Remember the latest evaluation's base rate and uncertainty tables, used when no model is promoted."""
    registry = read_registry(model_dir)
    registry["baseline"] = baseline
    registry["events"].append({"at": iso(utcnow()), "event": "baseline_recorded", "model_id": baseline.get("from_run")})
    _write_registry(registry, model_dir)


def champion(model_dir: Path | None = None) -> str | None:
    return read_registry(model_dir).get("champion")
