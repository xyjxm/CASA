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
    relative_reduction,
    safe_reject_budget_metrics,
    validate_phase5_prediction_rows,
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
    validate_phase5_prediction_rows(rows, min_unsafe_calibration_per_skill=0)
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
    summary = _summary(split_rows, baseline_rows, per_skill_rows, args.alpha)

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


def _summary(split_rows: list[dict], baseline_rows: list[dict], per_skill_rows: list[dict], alpha: float) -> dict:
    by_method = {row["method"]: row for row in baseline_rows}
    casa = by_method["casa_a_per_skill"]
    sonic = by_method["sonic_only"]
    hard = by_method["hard_contract"]
    global_conf = by_method["global_conformal"]
    per_skill = {
        method: {row["skill_name"]: row for row in per_skill_rows if row["method"] == method}
        for method in METHOD_ORDER
    }
    casa_vs_global_diagnostics = {}
    casa_vs_global_better = []
    for skill in MAIN_SKILLS:
        casa_fnr = float(per_skill["casa_a_per_skill"][skill]["fnr"])
        global_fnr = float(per_skill["global_conformal"][skill]["fnr"])
        casa_dist = abs(casa_fnr - alpha)
        global_dist = abs(global_fnr - alpha)
        casa_closer = casa_dist < global_dist
        if casa_closer:
            casa_vs_global_better.append(skill)
        casa_vs_global_diagnostics[skill] = {
            "casa_fnr": casa_fnr,
            "global_fnr": global_fnr,
            "casa_distance_to_alpha": casa_dist,
            "global_distance_to_alpha": global_dist,
            "casa_closer_to_alpha": casa_closer,
            "casa_more_conservative": casa_fnr < global_fnr,
        }
    casa_safe_acceptance = float(casa["safe_acceptance_rate"])
    sonic_safe_acceptance = float(sonic["safe_acceptance_rate"])
    safe_drop_abs = sonic_safe_acceptance - casa_safe_acceptance
    safe_drop_rel = safe_drop_abs / sonic_safe_acceptance if sonic_safe_acceptance else 0.0
    matched_hard = safe_reject_budget_metrics(
        split_rows,
        lambda row: float(row["hard_contract_score_float"]),
        method="hard_contract_matched_safe_budget",
        split=casa["split"],
        max_safe_reject_rate=1.0 - casa_safe_acceptance,
    )
    fixed_hard_safe_acceptance = float(hard["safe_acceptance_rate"])
    fixed_hard_comparison_mode = "blocking"
    if hard["unsafe_invocation_count"] <= 0 or fixed_hard_safe_acceptance < casa_safe_acceptance - 1e-12:
        fixed_hard_comparison_mode = "diagnostic"
    return {
        "phase": "CASA Phase5 baseline evaluation",
        "split": casa["split"],
        "alpha": alpha,
        "test_count": casa["count"],
        "unsafe_total": casa["unsafe_total"],
        "methods": by_method,
        "casa_vs_sonic_unsafe_reduction": relative_reduction(
            sonic["unsafe_invocation_count"], casa["unsafe_invocation_count"]
        ),
        "casa_vs_hard_unsafe_reduction": relative_reduction(
            hard["unsafe_invocation_count"], casa["unsafe_invocation_count"]
        ),
        "casa_vs_global_unsafe_reduction": relative_reduction(
            global_conf["unsafe_invocation_count"], casa["unsafe_invocation_count"]
        ),
        "hard_contract_matched_safe_budget": matched_hard,
        "casa_vs_matched_hard_unsafe_reduction": relative_reduction(
            matched_hard["unsafe_invocation_count"], casa["unsafe_invocation_count"]
        ),
        "fixed_hard_comparison_mode": fixed_hard_comparison_mode,
        "fixed_hard_comparison_reason": (
            "undefined_relative_reduction"
            if hard["unsafe_invocation_count"] <= 0
            else "fixed_hard_over_rejects_safe_samples"
            if fixed_hard_comparison_mode == "diagnostic"
            else "fixed_hard_safe_acceptance_not_lower_than_casa"
        ),
        "casa_safe_acceptance_drop_abs_vs_sonic": safe_drop_abs,
        "casa_safe_acceptance_drop_rel_vs_sonic": safe_drop_rel,
        "casa_task_success_drop_abs_vs_sonic": safe_drop_abs,
        "casa_task_success_drop_rel_vs_sonic": safe_drop_rel,
        "casa_vs_global_fnr_closer_skills": casa_vs_global_better,
        "casa_vs_global_fnr_closer_skill_count": len(casa_vs_global_better),
        "casa_vs_global_fnr_diagnostics": casa_vs_global_diagnostics,
        "per_skill": per_skill,
    }


if __name__ == "__main__":
    main()
