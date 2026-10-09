"""The rule a model must pass before it may serve predictions. Written before any holdout is scored.

The rule is code, in version control, and is copied into every model card together with the
numbers it was applied to. Changing the rule means changing ``RULE_VERSION`` in a commit made
before the holdout it will judge is evaluated.

Why these requirements
    * Skill is measured with the Brier score against a constant base rate, because accuracy
      rewards always saying "up" in a market that mostly rises.
    * The interval must exclude zero, because a positive point estimate on a few years of
      overlapping, correlated rows is easily luck.
    * Skill must also be positive in most individual years, because one unusual year can carry an
      average.
    * Development and holdout must agree in sign, because a result that appears only on the holdout
      is as suspect as one that disappears there.

Passing this rule establishes a statistical edge in forecasting direction. It does **not**
establish that a trading strategy built on the forecast is profitable after costs.
"""
from __future__ import annotations

from typing import Any

RULE_VERSION = 1
MIN_SEGMENT_ROWS = 1000
MIN_SHARE_OF_YEARS = 2 / 3
CALIBRATION_TOLERANCE = 0.03

RULE_TEXT = [
    "The candidate is the one model chosen on development data (lowest pooled validation Brier score). Only it can be promoted.",
    "Holdout Brier skill against the constant-base-rate baseline is above 0, and the lower end of its 95% calendar-week bootstrap interval is above 0.",
    "The lower end of the 95% interval for holdout ROC-AUC is above 0.50.",
    f"Brier skill is above 0 in at least two thirds of the holdout calendar years that have {MIN_SEGMENT_ROWS} or more rows.",
    "Pooled validation Brier skill (development data) is above 0 as well.",
    f"On the holdout, no fifth of the stated probabilities is off by more than {CALIBRATION_TOLERANCE} with a 95% interval that excludes zero.",
    "Every leakage check passes.",
]


def evaluate_rule(*, holdout_intervals: dict[str, Any], holdout_years: list[dict[str, Any]], validation_skill: float, calibration: dict[str, Any], leakage_checks: list[dict[str, Any]], is_baseline: bool) -> dict[str, Any]:
    """Apply the rule to one candidate's numbers. Returns each requirement with its verdict."""
    skill = holdout_intervals["brier_skill"]
    auc = holdout_intervals["roc_auc"]
    years = [row for row in holdout_years if row["n"] >= MIN_SEGMENT_ROWS]
    positive_years = [row for row in years if row["brier_skill"] > 0]
    requirements = [
        {"requirement": "candidate is a model, not a baseline", "passed": not is_baseline, "value": "baseline" if is_baseline else "model"},
        {"requirement": "holdout Brier skill > 0 with 95% interval above 0", "passed": bool(skill["point"] > 0 and skill["low"] is not None and skill["low"] > 0),
         "value": {"point": skill["point"], "low": skill["low"], "high": skill["high"]}},
        {"requirement": "holdout ROC-AUC 95% interval above 0.50", "passed": bool(auc["low"] is not None and auc["low"] > 0.5),
         "value": {"point": auc["point"], "low": auc["low"], "high": auc["high"]}},
        {"requirement": "positive skill in at least two thirds of holdout years", "passed": bool(years) and len(positive_years) / len(years) >= MIN_SHARE_OF_YEARS,
         "value": {"years_positive": len(positive_years), "years_counted": len(years), "by_year": {row["segment"]: row["brier_skill"] for row in years}}},
        {"requirement": "validation Brier skill > 0", "passed": bool(validation_skill > 0), "value": round(float(validation_skill), 6)},
        {"requirement": "no miscalibrated probability band on the holdout", "passed": bool(calibration.get("bands")) and calibration.get("miscalibrated_bands") == 0,
         "value": {"miscalibrated_bands": calibration.get("miscalibrated_bands"), "largest_gap": max((abs(b["gap"]) for b in calibration.get("bands", [])), default=None)}},
        {"requirement": "all leakage checks pass", "passed": all(check["passed"] for check in leakage_checks), "value": {check["check"]: check["passed"] for check in leakage_checks}},
    ]
    eligible = all(item["passed"] for item in requirements)
    return {
        "rule_version": RULE_VERSION,
        "rule": RULE_TEXT,
        "requirements": requirements,
        "eligible": eligible,
        "verdict": "eligible for promotion" if eligible else "not promoted: no predictive edge established under the pre-written rule",
    }
