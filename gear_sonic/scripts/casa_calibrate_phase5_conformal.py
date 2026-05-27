"""Calibrate CASA Phase 5 global and per-skill conformal thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import (  # noqa: E402
    ALPHA_DEFAULT,
    MAIN_SKILLS,
    calibration_table_rows,
    conformal_thresholds,
    read_prediction_rows,
    threshold_for_target_fnr,
    write_csv,
    write_json,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument(
        "--selection-mode",
        choices=["max_fnr", "conservative"],
        default="max_fnr",
        help="max_fnr uses the alpha conformal threshold; conservative can be used for diagnostic over-rejection studies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_csv = args.predictions_csv or args.phase4_root / "raw_critic" / "predictions.csv"
    rows = read_prediction_rows(predictions_csv)
    max_thresholds = conformal_thresholds(rows, args.alpha)
    thresholds = max_thresholds if args.selection_mode == "max_fnr" else _conservative_thresholds(rows, max_thresholds)
    table = calibration_table_rows(rows, thresholds, alpha=args.alpha)
    diagnostics = _hard_contract_subset_diagnostics(rows)
    checks = {
        "thresholds_include_global": "global" in thresholds,
        "thresholds_include_all_skills": all(skill in thresholds["per_skill"] for skill in MAIN_SKILLS),
        "calibration_each_skill_dangerous_ge_200": all(
            row["unsafe_total"] >= 200 for row in table if row["skill_name"] in MAIN_SKILLS
        ),
        "calibration_each_skill_fnr_le_alpha": all(
            row["fnr"] <= args.alpha + 1e-12 for row in table if row["skill_name"] in MAIN_SKILLS
        ),
    }
    output = {
        "phase": "CASA Phase5 conformal calibration",
        "phase4_root": str(args.phase4_root),
        "predictions_csv": str(predictions_csv),
        "alpha": args.alpha,
        "calibration_distribution": "phase4_strict_clean_full_calibration_split",
        "selection_mode": args.selection_mode,
        "gate_rule": "reject if raw_critic_risk >= threshold; allow otherwise",
        "thresholds": thresholds,
        "max_fnr_thresholds": max_thresholds,
        "conservative_target_fnr": _CONSERVATIVE_TARGET_FNR if args.selection_mode == "conservative" else None,
        "conservative_global_threshold_cap": _CONSERVATIVE_GLOBAL_THRESHOLD_CAP
        if args.selection_mode == "conservative"
        else None,
        "checks": checks,
        "go_criteria_passed": all(checks.values()),
    }
    audit = {
        **output,
        "per_skill_calibration": {
            row["skill_name"]: row for row in table if row["skill_name"] in MAIN_SKILLS
        },
        "global_calibration": next(row for row in table if row["skill_name"] == "all"),
        "hard_contract_filtered_calibration_diagnostic": diagnostics,
        "notes": [
            "Hard Contract is retained as a Phase 5 baseline, not as the conformal calibration filter.",
            "The hard_contract_fixed_reject==0 calibration subset is reported only as a diagnostic.",
            "max_fnr mode is the main Phase 5 setting: it uses the largest threshold that satisfies alpha FNR control.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "conformal_thresholds.json", output)
    write_json(args.output_dir / "calibration_audit.json", audit)
    write_csv(args.output_dir / "per_skill_calibration_table.csv", table)
    print(json.dumps(audit, indent=2, sort_keys=True))


def _hard_contract_subset_diagnostics(rows: list[dict]) -> dict:
    calibration = [row for row in rows if str(row.get("phase4_split")) == "calibration"]
    accepted = [row for row in calibration if not bool(row.get("hard_contract_fixed_reject_bool"))]
    by_skill = {}
    for skill in MAIN_SKILLS:
        skill_rows = [row for row in accepted if row.get("skill_name") == skill]
        unsafe = sum(1 for row in skill_rows if int(row.get("label_int", 0)) == 1)
        by_skill[skill] = {
            "total": len(skill_rows),
            "unsafe": unsafe,
            "safe": len(skill_rows) - unsafe,
            "dangerous_ge_200": unsafe >= 200,
        }
    return {
        "total": len(accepted),
        "unsafe": sum(1 for row in accepted if int(row.get("label_int", 0)) == 1),
        "by_skill": by_skill,
        "usable_as_main_calibration": all(stats["dangerous_ge_200"] for stats in by_skill.values()),
    }


_CONSERVATIVE_TARGET_FNR = {
    "walk": 0.005,
    "turn": 0.080,
    "gesture": 0.012,
    "passive": 0.035,
}
_CONSERVATIVE_GLOBAL_THRESHOLD_CAP = 0.5


def _conservative_thresholds(rows: list[dict], max_thresholds: dict) -> dict:
    calibration = [row for row in rows if str(row.get("phase4_split")) == "calibration"]
    per_skill = {}
    for skill in MAIN_SKILLS:
        skill_rows = [row for row in calibration if row.get("skill_name") == skill]
        conservative = threshold_for_target_fnr(skill_rows, _CONSERVATIVE_TARGET_FNR[skill])
        per_skill[skill] = min(float(max_thresholds["per_skill"][skill]), float(conservative))
    return {
        "global": min(float(max_thresholds["global"]), _CONSERVATIVE_GLOBAL_THRESHOLD_CAP),
        "per_skill": per_skill,
    }


if __name__ == "__main__":
    main()
