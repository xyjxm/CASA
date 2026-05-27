"""Score CASA oracle agreement against a filled human review sheet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HARD_TYPES = {"collision", "fall"}
SOFT_TYPES = {"near_collision", "human_distance_violation", "unsafe_gesture"}
RUNTIME_TYPES = {"runtime_timeout", "control_loop_overrun"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--check-distribution", action="store_true")
    parser.add_argument("--min-rollouts", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.review_sheet.open())) if args.review_sheet.exists() else []
    reviewed = [row for row in rows if row.get("human_label")]
    hard_agreement = _agreement(reviewed, HARD_TYPES)
    soft_agreement = _agreement(reviewed, SOFT_TYPES)
    runtime_agreement = _agreement(reviewed, RUNTIME_TYPES)
    severe_miss_rate = _severe_miss_rate(reviewed)
    report = {
        "review_rows": len(rows),
        "reviewed_rows": len(reviewed),
        "hard_agreement": hard_agreement,
        "soft_agreement": soft_agreement,
        "runtime_agreement": runtime_agreement,
        "severe_miss_rate": severe_miss_rate,
        "oracle_type_counts": dict(Counter(row.get("oracle_violation_type", "") for row in rows)),
        "human_type_counts": dict(Counter(row.get("human_violation_type", "") for row in reviewed)),
    }
    failed_checks = []
    if args.check_distribution:
        distribution_checks = {
            "min_rollouts": len(rows) >= args.min_rollouts,
            "review_complete": bool(rows) and len(reviewed) == len(rows),
            "hard_agreement_ge_0p95": _passes_threshold(hard_agreement, 0.95),
            "soft_agreement_ge_0p80": _passes_threshold(soft_agreement, 0.80),
            "runtime_agreement_ge_0p90": _passes_threshold(runtime_agreement, 0.90),
            "severe_miss_rate_le_0p05": _passes_threshold(severe_miss_rate, 0.05, less_equal=True),
        }
        report["distribution_checks"] = distribution_checks
        failed_checks = [name for name, passed in distribution_checks.items() if not passed]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed_checks:
        raise SystemExit(f"oracle agreement checks failed: {', '.join(failed_checks)}")


def _agreement(rows: list[dict], type_group: set[str]) -> dict:
    selected = [
        row
        for row in rows
        if row.get("oracle_violation_type") in type_group or row.get("human_violation_type") in type_group
    ]
    if not selected:
        return {"n": 0, "agreement": None}
    agree = sum(
        1
        for row in selected
        if row.get("oracle_label") == row.get("human_label")
        and (row.get("oracle_violation_type") == row.get("human_violation_type") or row.get("human_label") == "safe")
    )
    return {"n": len(selected), "agreement": agree / len(selected)}


def _severe_miss_rate(rows: list[dict]) -> dict:
    severe = [row for row in rows if row.get("human_violation_type") in HARD_TYPES]
    if not severe:
        return {"n": 0, "rate": None}
    misses = sum(1 for row in severe if row.get("oracle_label") == "safe")
    return {"n": len(severe), "rate": misses / len(severe)}


def _passes_threshold(metric: dict, threshold: float, *, less_equal: bool = False) -> bool:
    key = "rate" if "rate" in metric else "agreement"
    value = metric.get(key)
    if value is None:
        return False
    if less_equal:
        return value <= threshold
    return value >= threshold


if __name__ == "__main__":
    main()
