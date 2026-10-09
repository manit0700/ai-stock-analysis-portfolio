"""Immutable data snapshots: the exact prices a model was trained and tested on.

A snapshot is a folder of CSV files plus a manifest that records where each file came from and
its SHA-256. The snapshot id ends with a hash of all contents, so two runs can be shown to have
used the same data, and a changed file is detected by ``verify``.

    <snapshot_dir>/<id>/manifest.json
    <snapshot_dir>/<id>/ohlcv/<SYMBOL>.csv     Date,Open,High,Low,Close[,Adj Close],Volume[,Dividends,Stock Splits]
    <snapshot_dir>/<id>/series/<NAME>.csv      Date,value

Only what is needed for reproducibility is stored. This is not a general market database.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.clock import iso, utcnow
from app.core.errors import NotFound, ProviderError, ValidationFailed
from app.marketdata.providers.base import Provider

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._^=-]{0,40}$")


def _safe_name(name: str) -> str:
    if not _SAFE.match(name):
        raise ValidationFailed(f"{name!r} cannot be used as a file name in a snapshot.")
    return name


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SnapshotStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path(self, snapshot_id: str) -> Path:
        return self.directory / _safe_name(snapshot_id)

    def list(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        out = []
        for manifest in sorted(self.directory.glob("*/manifest.json")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({k: data.get(k) for k in ("id", "name", "created_at", "description", "start", "end", "n_symbols", "n_series")})
        return out

    def create(
        self,
        name: str,
        ohlcv: dict[str, pd.DataFrame],
        series: dict[str, pd.Series] | None = None,
        *,
        sources: list[dict[str, Any]] | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a new snapshot and return its manifest. Never overwrites an existing snapshot."""
        _safe_name(name)
        series = series or {}
        if not ohlcv:
            raise ValidationFailed("A snapshot needs at least one price history.")
        staging = self.directory / f".staging-{name}"
        if staging.exists():
            for old in sorted(staging.rglob("*"), reverse=True):
                old.unlink() if old.is_file() else old.rmdir()
        (staging / "ohlcv").mkdir(parents=True, exist_ok=True)
        (staging / "series").mkdir(parents=True, exist_ok=True)

        files: dict[str, dict[str, Any]] = {}
        starts, ends = [], []
        for symbol in sorted(ohlcv):
            frame = ohlcv[symbol].sort_index()
            frame.index = pd.to_datetime(frame.index)
            frame.index.name = "Date"
            target = staging / "ohlcv" / f"{_safe_name(symbol)}.csv"
            frame.to_csv(target, float_format="%.6f", date_format="%Y-%m-%d")
            files[f"ohlcv/{symbol}.csv"] = {"sha256": _file_hash(target), "rows": int(len(frame)), "start": frame.index[0].date().isoformat(), "end": frame.index[-1].date().isoformat()}
            starts.append(frame.index[0].date())
            ends.append(frame.index[-1].date())
        for key in sorted(series):
            values = series[key].sort_index().dropna()
            values.index = pd.to_datetime(values.index)
            target = staging / "series" / f"{_safe_name(key)}.csv"
            values.rename("value").rename_axis("Date").to_csv(target, float_format="%.6f", date_format="%Y-%m-%d")
            files[f"series/{key}.csv"] = {"sha256": _file_hash(target), "rows": int(len(values)), "start": values.index[0].date().isoformat(), "end": values.index[-1].date().isoformat()}

        content = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in sorted(files.items())}, sort_keys=True).encode()).hexdigest()
        snapshot_id = f"{name}-{content[:12]}"
        final = self.directory / snapshot_id
        manifest = {
            "id": snapshot_id,
            "name": name,
            "description": description,
            "created_at": iso(utcnow()),
            "content_sha256": content,
            "start": min(starts).isoformat(),
            "end": max(ends).isoformat(),
            "n_symbols": len(ohlcv),
            "n_series": len(series),
            "symbols": sorted(ohlcv),
            "series": sorted(series),
            "sources": sources or [],
            "metadata": metadata or {},
            "files": files,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if final.exists():
            # Same content already stored: keep the existing copy, discard the staging folder.
            for old in sorted(staging.rglob("*"), reverse=True):
                old.unlink() if old.is_file() else old.rmdir()
            staging.rmdir()
            return json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        staging.rename(final)
        return manifest

    def manifest(self, snapshot_id: str) -> dict[str, Any]:
        path = self.path(snapshot_id) / "manifest.json"
        if not path.exists():
            raise NotFound(f"There is no data snapshot called {snapshot_id}.")
        return json.loads(path.read_text(encoding="utf-8"))

    def verify(self, snapshot_id: str) -> list[str]:
        """Return the files whose contents no longer match the manifest (empty list = intact)."""
        manifest = self.manifest(snapshot_id)
        root = self.path(snapshot_id)
        changed = []
        for rel, info in manifest["files"].items():
            target = root / rel
            if not target.exists() or _file_hash(target) != info["sha256"]:
                changed.append(rel)
        return changed

    def load_ohlcv(self, snapshot_id: str, symbol: str) -> pd.DataFrame:
        path = self.path(snapshot_id) / "ohlcv" / f"{_safe_name(symbol)}.csv"
        if not path.exists():
            raise NotFound(f"Snapshot {snapshot_id} has no prices for {symbol}.")
        frame = pd.read_csv(path, index_col="Date", parse_dates=True)
        frame.index = pd.DatetimeIndex(frame.index).as_unit("ns")
        return frame

    def load_series(self, snapshot_id: str, name: str) -> pd.Series:
        path = self.path(snapshot_id) / "series" / f"{_safe_name(name)}.csv"
        if not path.exists():
            raise NotFound(f"Snapshot {snapshot_id} has no series called {name}.")
        series = pd.read_csv(path, index_col="Date", parse_dates=True)["value"].rename(name)
        series.index = pd.DatetimeIndex(series.index).as_unit("ns")
        return series


class SnapshotProvider(Provider):
    """Serve history and series from one snapshot. Everything it returns is historical."""

    supports = frozenset({"history", "economic"})

    def __init__(self, store: SnapshotStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id
        self.name = f"snapshot:{snapshot_id}"
        self._manifest = store.manifest(snapshot_id)

    def history(self, symbol: str, start: date | None, end: date | None, interval: str) -> pd.DataFrame:
        if interval != "1d":
            raise ProviderError("Snapshots hold daily bars only.", details={"provider": self.name}, code="not_supported")
        symbol = symbol.upper()
        if symbol not in self._manifest["symbols"]:
            raise ProviderError(f"Snapshot {self.snapshot_id} has no prices for {symbol}.", details={"provider": self.name}, code="no_data")
        frame = self.store.load_ohlcv(self.snapshot_id, symbol)
        if start:
            frame = frame[frame.index >= pd.Timestamp(start)]
        if end:
            frame = frame[frame.index <= pd.Timestamp(end)]
        if frame.empty:
            raise ProviderError(f"Snapshot {self.snapshot_id} has no {symbol} prices in that range.", details={"provider": self.name}, code="no_data")
        return frame

    def economic_series(self, series_id: str, start: date | None, end: date | None) -> pd.Series:
        if series_id not in self._manifest["series"]:
            raise ProviderError(f"Snapshot {self.snapshot_id} has no series {series_id}.", details={"provider": self.name}, code="no_data")
        series = self.store.load_series(self.snapshot_id, series_id)
        if start:
            series = series[series.index >= pd.Timestamp(start)]
        if end:
            series = series[series.index <= pd.Timestamp(end)]
        return series
