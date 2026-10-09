"""Chronological splits with purging, so no training row shares an outcome with a test row.

A row dated ``t`` is labelled with the price at ``t + horizon``. If a row dated just before the
test period were used for training, its label would have been decided by prices inside the test
period. So training rows must have their whole label window finished before the test period
starts (the purge), and a further ``embargo`` of sessions is left empty between them.

Only walk-forward splits are produced: a model is always tested on dates later than every date it
was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp   # last feature date allowed in training
    test_start: pd.Timestamp  # first feature date tested
    test_end: pd.Timestamp    # last feature date tested


def _sessions(dates) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))


def train_end_before(sessions: pd.DatetimeIndex, test_start: pd.Timestamp, horizon: int, embargo: int) -> pd.Timestamp | None:
    """Last session whose label window ends at least ``embargo`` + 1 sessions before ``test_start``."""
    position = int(sessions.searchsorted(test_start, side="left"))  # index of the first test session
    last = position - 1 - horizon - embargo
    return sessions[last] if last >= 0 else None


def walk_forward_folds(
    dates,
    *,
    first_test_start: str | pd.Timestamp,
    last_test_end: str | pd.Timestamp,
    n_folds: int,
    horizon: int,
    embargo: int = 5,
    min_train_sessions: int = 252,
) -> list[Fold]:
    """Split ``[first_test_start, last_test_end]`` into ``n_folds`` consecutive test blocks.

    Each fold trains on everything before its own test block (expanding window), minus the purge
    and embargo.
    """
    sessions = _sessions(dates)
    first, last = pd.Timestamp(first_test_start), pd.Timestamp(last_test_end)
    inside = sessions[(sessions >= first) & (sessions <= last)]
    if len(inside) < n_folds * 20:
        raise ValueError("Not enough sessions in the requested range for that many folds.")
    edges = np.linspace(0, len(inside), n_folds + 1).astype(int)
    folds: list[Fold] = []
    for k in range(n_folds):
        block = inside[edges[k]: edges[k + 1]]
        train_end = train_end_before(sessions, block[0], horizon, embargo)
        if train_end is None or int(sessions.searchsorted(train_end, side="right")) < min_train_sessions:
            raise ValueError(f"Fold {k + 1} would have fewer than {min_train_sessions} training sessions.")
        folds.append(Fold(f"fold{k + 1}:{block[0].date()}..{block[-1].date()}", train_end, block[0], block[-1]))
    return folds


def holdout_fold(dates, *, holdout_start: str | pd.Timestamp, horizon: int, embargo: int = 5) -> Fold:
    """Everything from ``holdout_start`` on is the holdout; development ends before the purge gap."""
    sessions = _sessions(dates)
    start = sessions[int(sessions.searchsorted(pd.Timestamp(holdout_start), side="left"))]
    train_end = train_end_before(sessions, start, horizon, embargo)
    if train_end is None:
        raise ValueError("There is no data before the holdout.")
    return Fold(f"holdout:{start.date()}..{sessions[-1].date()}", train_end, start, sessions[-1])


def masks(frame: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Boolean row masks ``(train, test)`` for ``fold``. ``frame`` is indexed by feature date."""
    index = frame.index
    train = np.asarray(index <= fold.train_end)
    test = np.asarray((index >= fold.test_start) & (index <= fold.test_end))
    return train, test


def check_fold(frame: pd.DataFrame, fold: Fold, horizon: int, embargo: int) -> dict:
    """Verify the purge on the actual rows: no training label window reaches the test period."""
    train, test = masks(frame, fold)
    sessions = _sessions(frame.index)
    latest_label_end = pd.Timestamp(frame.loc[train, "label_end"].max()) if train.any() else None
    gap = None
    if latest_label_end is not None and test.any():
        gap = int(sessions.searchsorted(fold.test_start, side="left")) - int(sessions.searchsorted(latest_label_end, side="left"))
    return {
        "fold": fold.name,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "overlap_rows": int((train & test).sum()),
        "latest_train_label_end": latest_label_end.date().isoformat() if latest_label_end is not None else None,
        "sessions_between_last_label_and_test": gap,
        "passed": bool(train.any() and test.any() and not (train & test).any() and latest_label_end < fold.test_start and gap is not None and gap > embargo),
    }
