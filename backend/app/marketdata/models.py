"""The envelope every market-data response travels in.

A number without its source and age is not usable for research or for a simulated trade. So
nothing leaves the data layer bare: every response says which provider it came from, when it was
fetched, what market time it describes, how fresh it is and whether quality checks passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd

from app.core.clock import iso

Freshness = Literal["live", "delayed", "historical"]
QualityStatus = Literal["ok", "degraded", "failed"]
Severity = Literal["info", "warning", "error"]


@dataclass
class QualityIssue:
    code: str
    severity: Severity
    detail: str
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "detail": self.detail, "count": self.count}


@dataclass
class QualityReport:
    status: QualityStatus = "ok"
    issues: list[QualityIssue] = field(default_factory=list)

    def add(self, code: str, severity: Severity, detail: str, count: int = 0) -> None:
        self.issues.append(QualityIssue(code, severity, detail, count))
        if severity == "error":
            self.status = "failed"
        elif severity == "warning" and self.status == "ok":
            self.status = "degraded"

    def has(self, code: str) -> bool:
        return any(issue.code == code for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "issues": [issue.to_dict() for issue in self.issues]}


@dataclass
class DataEnvelope:
    dataset: str
    provider: str
    retrieved_at: datetime
    freshness: Freshness
    data: Any
    symbol: str | None = None
    market_timestamp: datetime | None = None
    quality: QualityReport = field(default_factory=QualityReport)
    cache_hit: bool = False
    cache_age_seconds: float | None = None
    attempts: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def meta(self) -> dict[str, Any]:
        """Everything except the payload. Attached to legacy responses as ``data_source``."""
        return {
            "symbol": self.symbol,
            "dataset": self.dataset,
            "provider": self.provider,
            "retrieved_at": iso(self.retrieved_at),
            "market_timestamp": iso(self.market_timestamp),
            "freshness": self.freshness,
            "quality": self.quality.to_dict(),
            "cache": {"hit": self.cache_hit, "age_seconds": self.cache_age_seconds},
            "attempts": self.attempts,
            "notes": self.notes,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.meta(), "data": serialise(self.data)}


def serialise(value: Any) -> Any:
    """Turn frames, series and timestamps into plain JSON-ready values. NaN becomes ``None``."""
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.index = [_stamp(ix) for ix in frame.index]
        records = []
        for ix, row in frame.iterrows():
            item = {"date": ix}
            for key, cell in row.items():
                item[_snake(str(key))] = _scalar(cell)
            records.append(item)
        return records
    if isinstance(value, pd.Series):
        return [{"date": _stamp(ix), "value": _scalar(v)} for ix, v in value.items()]
    if isinstance(value, dict):
        return {str(k): serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise(v) for v in value]
    return _scalar(value)


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _stamp(ix: Any) -> str:
    ts = pd.Timestamp(ix)
    if ts.hour or ts.minute or ts.second:
        return ts.isoformat()
    return ts.date().isoformat()


def _scalar(cell: Any) -> Any:
    if cell is None:
        return None
    if isinstance(cell, (pd.Timestamp, datetime)):
        return iso(pd.Timestamp(cell).to_pydatetime())
    if hasattr(cell, "item") and not isinstance(cell, (str, bytes)):
        try:
            cell = cell.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(cell, float) and cell != cell:  # NaN
        return None
    return cell
