"""Audit CASA Phase 5 conformal baseline acceptance criteria."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import ALPHA_DEFAULT, MAIN_SKILLS, read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = read_json(args.phase5_root / "calibration_audit.json")
    baseline = read_json(args.phase5_root / "baseline_summary.json")
    thresholds = read_json(args.phase5_root / "conformal_thresholds.json")
    checks = _checks(calibration, baseline, args.alpha)
    blocking_reasons = [name for name, passed in checks.items() if not passed]
    warning_reasons = _warnings(calibration, baseline)
    go = not blocking_reasons
    audit = {
        "phase": "CASA Phase5 acceptance audit",
        "status": "PASS_STRICT" if go else "STRICT_NO_GO",
        "go": go,
        "alpha": args.alpha,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "thresholds": thresholds.get("thresholds", {}),
        "calibration_summary": {
            "go_criteria_passed": calibration.get("go_criteria_passed"),
            "calibration_distribution": calibration.get("calibration_distribution"),
            "per_skill_calibration": calibration.get("per_skill_calibration"),
            "hard_contract_filtered_calibration_diagnostic": calibration.get(
                "hard_contract_filtered_calibration_diagnostic"
            ),
            "hard_contract_filtered_calibration": calibration.get("hard_contract_filtered_calibration"),
        },
        "baseline_summary": _compact_baseline(baseline),
    }
    go_no_go = {
        "phase": "CASA Phase5",
        "go": go,
        "status": audit["status"],
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "alpha": args.alpha,
        "thresholds": thresholds.get("thresholds", {}),
        "baseline_summary": audit["baseline_summary"],
    }
    write_json(args.phase5_root / "phase5_acceptance_audit.json", audit)
    write_json(args.phase5_root / "phase5_go_no_go.json", go_no_go)
    (args.phase5_root / "phase5_acceptance_audit.md").write_text(_markdown(audit) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.strict and not go:
        raise SystemExit(2)


def _checks(calibration: dict[str, Any], baseline: dict[str, Any], alpha: float) -> dict[str, bool]:
    per_skill_cal = calibration.get("per_skill_calibration", {})
    per_skill_test = baseline.get("per_skill", {}).get("casa_a_per_skill", {})
    fixed_hard_mode = baseline.get("fixed_hard_comparison_mode", "blocking")
    fixed_hard_reduction = baseline.get("casa_vs_hard_unsafe_reduction")
    matched_hard_reduction = baseline.get("casa_vs_matched_hard_unsafe_reduction")
    hc_required = bool(calibration.get("require_hard_contract_filtered_calibration"))
    hc_main = calibration.get("hard_contract_filtered_calibration", {})
    checks = {
        "thresholds_file_valid": bool(calibration.get("thresholds") or calibration.get("thresholds", {}))
        or bool(calibration.get("per_skill_calibration")),
        "calibration_each_skill_dangerous_ge_200": all(
            per_skill_cal.get(skill, {}).get("unsafe_total", 0) >= 200 for skill in MAIN_SKILLS
        ),
        "calibration_each_skill_fnr_le_alpha": all(
            per_skill_cal.get(skill, {}).get("fnr", 1.0) <= alpha + 1e-12 for skill in MAIN_SKILLS
        ),
    }
    if hc_required:
        checks["calibration_hard_contract_filtered"] = bool(
            hc_main.get("all_calibration_rows_hard_contract_accepted")
        )
        checks["hard_contract_filtered_calibration_each_skill_dangerous_ge_200"] = all(
            hc_main.get("by_skill", {}).get(skill, {}).get("unsafe", 0) >= 200 for skill in MAIN_SKILLS
        )
    checks.update(
        {
            "test_each_skill_fnr_le_alpha_plus_0p03": all(
                per_skill_test.get(skill, {}).get("fnr", 1.0) <= alpha + 0.03 + 1e-12 for skill in MAIN_SKILLS
            ),
            "casa_vs_sonic_unsafe_reduction_ge_40pct": baseline.get("casa_vs_sonic_unsafe_reduction", 0.0)
            >= 0.40,
            "casa_vs_fixed_hard_unsafe_reduction_ge_20pct_or_diagnostic": (
                fixed_hard_mode == "diagnostic"
                or fixed_hard_reduction is None
                or fixed_hard_reduction >= 0.20
            ),
            "casa_vs_matched_hard_unsafe_reduction_ge_20pct": (
                matched_hard_reduction is None or matched_hard_reduction >= 0.20
            ),
            "casa_safe_acceptance_drop_abs_le_10pp": _metric(
                baseline, "casa_safe_acceptance_drop_abs_vs_sonic", "casa_task_success_drop_abs_vs_sonic", 1.0
            )
            <= 0.10,
            "casa_safe_acceptance_drop_rel_le_30pct": _metric(
                baseline, "casa_safe_acceptance_drop_rel_vs_sonic", "casa_task_success_drop_rel_vs_sonic", 1.0
            )
            <= 0.30,
        }
    )
    return checks


def _warnings(calibration: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    warnings = []
    hc_diag = calibration.get("hard_contract_filtered_calibration_diagnostic", {})
    hc_main = calibration.get("hard_contract_filtered_calibration", {})
    if not (hc_main.get("usable_as_main_calibration") or hc_diag.get("usable_as_main_calibration", False)):
        warnings.append("hard_contract_filtered_calibration_subset_dangerous_insufficient")
    hard = baseline.get("methods", {}).get("hard_contract", {})
    if hard.get("safe_acceptance_rate", hard.get("task_success_proxy", 1.0)) < 0.50:
        warnings.append("hard_contract_fixed_baseline_over_rejects")
    if baseline.get("fixed_hard_comparison_mode") == "diagnostic":
        warnings.append(f"fixed_hard_comparison_diagnostic:{baseline.get('fixed_hard_comparison_reason')}")
    if baseline.get("casa_vs_global_fnr_closer_skill_count", 0) < 2:
        warnings.append("casa_vs_global_fnr_closeness_diagnostic_only")
    return warnings


def _compact_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    methods = baseline.get("methods", {})
    return {
        "split": baseline.get("split"),
        "test_count": baseline.get("test_count"),
        "unsafe_total": baseline.get("unsafe_total"),
        "casa_vs_sonic_unsafe_reduction": baseline.get("casa_vs_sonic_unsafe_reduction"),
        "casa_vs_hard_unsafe_reduction": baseline.get("casa_vs_hard_unsafe_reduction"),
        "casa_vs_matched_hard_unsafe_reduction": baseline.get("casa_vs_matched_hard_unsafe_reduction"),
        "fixed_hard_comparison_mode": baseline.get("fixed_hard_comparison_mode"),
        "fixed_hard_comparison_reason": baseline.get("fixed_hard_comparison_reason"),
        "casa_vs_global_unsafe_reduction": baseline.get("casa_vs_global_unsafe_reduction"),
        "casa_safe_acceptance_drop_abs_vs_sonic": _metric(
            baseline, "casa_safe_acceptance_drop_abs_vs_sonic", "casa_task_success_drop_abs_vs_sonic", None
        ),
        "casa_safe_acceptance_drop_rel_vs_sonic": _metric(
            baseline, "casa_safe_acceptance_drop_rel_vs_sonic", "casa_task_success_drop_rel_vs_sonic", None
        ),
        "casa_vs_global_fnr_closer_skills": baseline.get("casa_vs_global_fnr_closer_skills"),
        "casa_vs_global_fnr_diagnostics": baseline.get("casa_vs_global_fnr_diagnostics"),
        "methods": methods,
    }


def _metric(data: dict[str, Any], preferred: str, fallback: str, default: Any) -> Any:
    if preferred in data:
        return data[preferred]
    return data.get(fallback, default)


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# CASA Phase5 Acceptance Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- go: `{audit['go']}`",
        f"- alpha: `{audit['alpha']}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if audit["blocking_reasons"]:
        lines.extend(f"- {reason}" for reason in audit["blocking_reasons"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if audit["warning_reasons"]:
        lines.extend(f"- {reason}" for reason in audit["warning_reasons"])
    else:
        lines.append("- none")
    lines.extend(["", "## Checks", ""])
    for name, passed in audit["checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Baseline Summary", "", "```json"])
    lines.append(json.dumps(audit["baseline_summary"], indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
