"""Command line for the forecasting pipeline.

    python -m app.forecast snapshots                          list data snapshots
    python -m app.forecast evaluate --snapshot <id>           development, then the holdout, once
    python -m app.forecast evaluate --snapshot <id> --stage development    stop before the holdout
    python -m app.forecast list                               every run and its verdict
    python -m app.forecast show <model id>                    the summary of one run
    python -m app.forecast promote <model id>                 refused unless the run passed the rule
    python -m app.forecast demote --reason "..."              go back to the baseline
    python -m app.forecast status                             what serves predictions now

Run from the ``backend`` folder. Nothing here places a trade or contacts a broker.
"""
from __future__ import annotations

import argparse
import json
import sys

from app.core.config import get_settings
from app.core.errors import MarketVisionError
from app.forecast import registry
from app.forecast.train import EvalConfig, dataset_from_snapshot, run_evaluation


def _summary(card: dict) -> str:
    lines = [f"run {card['model_id']}  ({card['status']})", f"  data      {card['data']['source']}  rows {card['data']['rows_complete']:,}  groups {', '.join(card['feature_groups'])}"]
    selection = card["selection"]
    lines.append(f"  selected  {selection['candidate']} {selection['params']}  validation Brier skill {selection['validation_brier_skill_vs_prior']:+.5f}")
    if card.get("holdout"):
        lines.append("  holdout   candidate                    accuracy  majority  bal.acc   AUC     Brier    skill vs base rate")
        for row in card["holdout"]["candidates"]:
            m = row["metrics"]
            mark = "*" if row["selected"] else " "
            lines.append(f"          {mark} {row['candidate']:28s} {m['accuracy']:.4f}    {m['majority_class_accuracy']:.4f}    {m['balanced_accuracy']:.4f}  {m['roc_auc']:.4f}  {m['brier']:.5f}  {row['brier_skill_vs_prior']:+.5f}")
        chosen = card["holdout"]["selected"]
        skill, auc = chosen["intervals"]["brier_skill"], chosen["intervals"]["roc_auc"]
        lines.append(f"  selected model, calibrated ({chosen['calibration']}): skill {skill['point']:+.5f} [{skill['low']:+.5f}, {skill['high']:+.5f}]   AUC {auc['point']:.4f} [{auc['low']:.4f}, {auc['high']:.4f}]   ECE {chosen['reliability']['ece']}")
        lines.append(f"  this holdout had been scored {card['holdout']['times_this_holdout_was_scored_before']} time(s) before")
        for item in card["promotion"]["requirements"]:
            lines.append(f"    [{'pass' if item['passed'] else 'FAIL'}] {item['requirement']}")
        lines.append(f"  verdict   {card['promotion']['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.forecast", description="Train, evaluate and promote forecasting models.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshots")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--snapshot", required=True)
    ev.add_argument("--stage", choices=["development", "all"], default="all")
    ev.add_argument("--holdout-start", default=EvalConfig.holdout_start)
    ev.add_argument("--first-validation-start", default=EvalConfig.first_validation_start)
    ev.add_argument("--folds", type=int, default=EvalConfig.n_folds)
    ev.add_argument("--horizon", type=int, default=EvalConfig.horizon)
    ev.add_argument("--seed", type=int, default=EvalConfig.seed)
    ev.add_argument("--bootstrap", type=int, default=EvalConfig.n_bootstrap)
    ev.add_argument("--quick", action="store_true", help="one setting per model; for a fast smoke run")
    ev.add_argument("--no-lightgbm", action="store_true")
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("model_id")
    show.add_argument("--json", action="store_true")
    promote = sub.add_parser("promote")
    promote.add_argument("model_id")
    demote = sub.add_parser("demote")
    demote.add_argument("--reason", default="")
    sub.add_parser("status")
    args = parser.parse_args(argv)

    try:
        if args.command == "snapshots":
            from app.marketdata.snapshots import SnapshotStore

            for row in SnapshotStore(get_settings().snapshot_dir).list():
                print(f"{row['id']:40s} {row['start']} to {row['end']}  {row['n_symbols']} symbols  {row['description']}")
        elif args.command == "evaluate":
            config = EvalConfig(
                horizon=args.horizon, holdout_start=args.holdout_start, first_validation_start=args.first_validation_start, n_folds=args.folds,
                seed=args.seed, n_bootstrap=args.bootstrap, quick=args.quick, include_lightgbm=False if args.no_lightgbm else None,
            )
            dataset, market_data, manifest = dataset_from_snapshot(args.snapshot, config)
            report = dataset.report
            print(f"dataset: {report['rows_complete']:,} complete rows from {len(report['symbols'])} symbols, {report['first_date']} to {report['last_date']}; "
                  f"groups used {report['groups_used']}; absent {report['groups_absent']}; base rate {report['base_rate']}")
            card = run_evaluation(dataset, market_data, config, snapshot=manifest, stage=args.stage)
            print()
            print(_summary(card))
            print(f"\ncard: {get_settings().model_dir / card['model_id'] / 'card.json'}")
        elif args.command == "list":
            for row in registry.list_runs():
                flag = "CHAMPION" if row["champion"] else ("eligible" if row["eligible"] else "")
                print(f"{row['model_id']:44s} {row['selected'] or '':26s} {flag:9s} {row['verdict'] or 'development only'}")
        elif args.command == "show":
            card = registry.read_card(args.model_id)
            print(json.dumps(card, indent=2) if args.json else _summary(card))
        elif args.command == "promote":
            registry.promote(args.model_id)
            print(f"{args.model_id} now serves predictions.")
        elif args.command == "demote":
            registry.demote(reason=args.reason)
            print("No model serves predictions; the baseline is reported.")
        elif args.command == "status":
            from app.forecast.service import PredictionService

            print(json.dumps(PredictionService(None).status(), indent=2))
    except MarketVisionError as exc:
        print(f"refused: {exc.message}", file=sys.stderr)
        if exc.details:
            print(json.dumps(exc.details, indent=2, default=str), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
