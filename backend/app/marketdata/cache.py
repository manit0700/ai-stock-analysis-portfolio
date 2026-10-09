"""Caches for market data: a small in-memory one, and files on disk for price history.

Caching is what lets the application stay inside free API quotas. Every cached value keeps the
time it was stored, so a response can say how old it is.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class TTLCache:
    """Thread-safe least-recently-used cache whose entries expire."""

    def __init__(self, maxsize: int = 512, clock: Callable[[], float] = time.time) -> None:
        self.maxsize = maxsize
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float) -> tuple[Any, float] | None:
        """Return ``(value, age_seconds)`` if present and younger than ``ttl``."""
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            age = self._clock() - stored_at
            if age > ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value, age

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class DiskFrameCache:
    """Price history on disk, one CSV per request key with a small JSON beside it."""

    def __init__(self, directory: Path, clock: Callable[[], float] = time.time) -> None:
        self.directory = Path(directory)
        self._clock = clock

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.directory / f"{digest}.csv", self.directory / f"{digest}.json"

    def get(self, key: str, ttl: float) -> tuple[pd.DataFrame, dict, float] | None:
        data_path, meta_path = self._paths(key)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            age = self._clock() - float(meta["stored_at"])
            if age > ttl or meta.get("key") != key:
                return None
            frame = pd.read_csv(data_path, index_col=0, parse_dates=True)
            frame.index = pd.DatetimeIndex(frame.index).as_unit("ns")
        except (OSError, ValueError, KeyError):
            return None
        return frame, meta, age

    def set(self, key: str, frame: pd.DataFrame, meta: dict) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            data_path, meta_path = self._paths(key)
            tmp = data_path.with_suffix(".tmp")
            frame.to_csv(tmp)
            tmp.replace(data_path)
            meta_path.write_text(json.dumps({**meta, "key": key, "stored_at": self._clock()}, default=str), encoding="utf-8")
        except OSError:
            return  # a cache that cannot be written is not an error
