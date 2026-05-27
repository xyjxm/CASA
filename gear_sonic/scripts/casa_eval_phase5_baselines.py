"""Evaluate CASA Phase 5 conformal and baseline gates on Phase 4 predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import (  # noqa: E402
    ALPHA_DEFAULT,
    MAIN_SKILLS,
    METHOD_ORDER,
    budget_reject_metrics,
    evaluate_method,
    read_json,
    read_prediction_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--thresholds-json", type=Path)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_csv = args.predictions_csv or args.phase4_root / "raw_critic" / "predictions.csv"
    thresholds_json = args.thresholds_json or args.phase5_root / "conformal_thresholds.json"
    rows = read_prediction_rows(predictions_csv)
    thresholds = read_json(thresholds_json)["thresholds"]
    split_rows = [row for row in rows if str(row.get("phase4_split")) == args.split]
    if not split_rows:
        raise SystemExit(f"No rows found for split {args.split!r} in {predictions_csv}")

    baseline_rows = [evaluate_method(split_rows, method, thresholds, split=args.split) for method in METHOD_ORDER]
    per_skill_rows = []
    for method in METHOD_ORDER:
        for skill in MAIN_SKILLS:
            skill_rows = [row for row in split_rows if row.get("skill_name") == skill]
            per_skill_rows.append(evaluate_method(skill_rows, method, thresholds, split=args.split, skill_name=skill))
    curve_rows = _rejection_risk_curve(split_rows, args.split)
    summary = _summary(baseline_rows, per_skill_rows, args.alpha)

    args.phase5_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.phase5_root / "baseline_results.csv", baseline_rows)
    write_json(args.phase5_root / "baseline_results.json", {"split": args.split, "methods": baseline_rows})
    write_csv(args.phase5_root / "per_skill_fnr.csv", per_skill_rows)
    write_csv(args.phase5_root / "rejection_risk_curve.csv", curve_rows)
    write_json(args.phase5_root / "baseline_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _rejection_risk_curve(rows: list[dict], split: str) -> list[dict]:
    output = []
    for budget in [round(value, 2) for value in np.linspace(0.0, 0.50, 21)]:
        output.append(
            budget_reject_metrics(
                rows,
                lambda row: float(row["raw_critic_risk_float"]),
                method="raw_critic_budgeted",
                split=split,
                reject_budget=budget,
            )
        )
        output.append(
            budget_reject_metrics(
                rows,
                lambda row: float(row["hard_contract_score_float"]),
                method="hard_contract_budgeted",
                split=split,
                reject_budget=budget,
            )
        )
    return output


def _summary(baseline_rows: list[dict], per_skill_rows: list[dict], alpha: float) -> dict:
    by_method = {row["method"]: row for row in baseline_rows}
    casa = by_method["casa_a_per_skill"]
    sonic = by_method["sonic_only"]
    hard = by_method["hard_contract"]
    global_conf = by_method["global_conformal"]
    per_skill = {
        method: {row["skill_name"]: row for row in per_skill_rows if row["method"] == method}
        for method in METHOD_ORDER
    }
    casa_vs_global_better = []
    for skill in MAIN_SKILLS:
        casa_dist = abs(float(per_skill["casa_a_per_skill"][skill]["fnr"]) - alpha)
        global_dist = abs(float(per_skill["global_conformal"][skill]["fnr"]) - alpha)
        if casa_dist < global_dist:
            casa_vs_global_better.append(skill)
    return {
        "phase": "CASA Phase5 baseline evaluation",
        "split": casa["split"],
        "alpha": alpha,
        "test_count": casa["count"],
        "unsafe_total": casa["unsafe_total"],
        "methods": by_method,
        "casa_vs_sonic_unsafe_reduction": _reduction(
            sonic["unsafe_invocation_count"], casa["unsafe_invocation_count"]
        ),
        "casa_vs_hard_unsafe_reduction": _reduction(
            hard["unsafe_invocation_count"], casa["unsafe_invocation_count"]
        ),
        "casa_vs_global_unsafe_reduction": _reduction(
            global_conf["unsafe_invocation_count"], casa["unsafe_invocation_count"]
        ),
        "casa_task_success_drop_abs_vs_sonic": sonic["task_success_proxy"] - casa["task_success_proxy"],
        "casa_task_success_drop_rel_vs_sonic": (
            (sonic["task_success_proxy"] - casa["task_success_proxy"]) / sonic["task_success_proxy"]
            if sonic["task_success_proxy"]
            else 0.0
        ),
        "casa_vs_global_fnr_closer_skills": casa_vs_global_better,
        "casa_vs_global_fnr_closer_skill_count": len(casa_vs_global_better),
        "per_skill": per_skill,
    }


def _reduction(baseline_unsafe: int, method_unsafe: int) -> float:
    if baseline_unsafe <= 0:
        return 0.0
    return (baseline_unsafe - method_unsafe) / baseline_unsafe


if __name__ == "__main__":
    main()
