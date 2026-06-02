"""Build a Hard-Contract-filtered Phase 5 calibration CSV from online runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import MAIN_SKILLS, read_prediction_rows, write_csv, write_json  # noqa: E402


BASE_FIELDS = [
    "sample_id",
    "phase4_split",
    "skill_name",
    "label",
    "safe_label",
    "raw_critic_risk",
    "hard_contract_score",
    "hard_contract_fixed_reject",
    "triggered_violation_types",
    "time_to_violation",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--min-unsafe-per-skill", type=int, default=200)
    parser.add_argument(
        "--dedupe-key",
        choices=["episode_dir_skill", "none"],
        default="episode_dir_skill",
        help="Remove duplicate lane/resume rows that point at the same episode dir and skill index.",
    )
    parser.add_argument(
        "--test-predictions-csv",
        type=Path,
        help="Phase 4 prediction CSV whose test rows are appended for Phase 5 evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = None if args.methods == "all" else {item.strip() for item in args.methods.split(",") if item.strip()}
    calibration_rows, summary = collect_online_hc_rows(
        args.phase5_root,
        methods=methods,
        dedupe_key=args.dedupe_key,
        min_unsafe_per_skill=args.min_unsafe_per_skill,
    )
    test_predictions = args.test_predictions_csv or args.phase4_root / "raw_critic" / "predictions.csv"
    test_rows = [
        _base_prediction_row(row)
        for row in read_prediction_rows(test_predictions)
        if str(row.get("phase4_split")) == "test"
    ]
    combined_rows = calibration_rows + test_rows
    output_csv = args.output_dir / "hc_filtered_online_calibration_predictions.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, combined_rows)
    summary.update(
        {
            "phase": "CASA Phase5 Hard-Contract-filtered online calibration build",
            "phase4_root": str(args.phase4_root),
            "phase5_root": str(args.phase5_root),
            "test_predictions_csv": str(test_predictions),
            "output_predictions_csv": str(output_csv),
            "test_rows_appended": len(test_rows),
            "combined_rows": len(combined_rows),
        }
    )
    write_json(args.output_dir / "hc_filtered_online_calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all(summary["checks"].values()):
        raise SystemExit(2)


def collect_online_hc_rows(
    phase5_root: Path,
    *,
    methods: set[str] | None,
    dedupe_key: str,
    min_unsafe_per_skill: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    method_skill_counts: dict[str, Counter[str]] = defaultdict(Counter)
    gate_files = sorted(phase5_root.rglob("gate_decisions.csv"))
    missing_rollout_summary = 0
    malformed_rollout_summary = 0
    missing_label = 0
    skill_mismatch = 0
    duplicate_rows = 0
    candidate_rows = 0

    for gate_path in gate_files:
        online_dir = gate_path.parent
        try:
            gate_rows = list(csv.DictReader(gate_path.open(newline="")))
        except OSError:
            continue
        for gate_row in gate_rows:
            candidate_rows += 1
            method = str(gate_row.get("method", ""))
            if methods is not None and method not in methods:
                continue
            if not _is_false(gate_row.get("hard_contract_fixed_reject")):
                continue
            if str(gate_row.get("decision")) != "allow":
                continue
            candidate_skill = str(gate_row.get("candidate_skill", ""))
            if candidate_skill not in MAIN_SKILLS:
                continue
            if str(gate_row.get("executed_skill", "")) != candidate_skill:
                continue
            risk = _finite_float(gate_row.get("raw_critic_risk"))
            hard_score = _finite_float(gate_row.get("hard_contract_score"))
            if risk is None or hard_score is None:
                continue
            episode_dir = _episode_dir(online_dir, gate_row)
            summary_path = episode_dir / "rollout_summary.json"
            if not summary_path.exists():
                missing_rollout_summary += 1
                continue
            try:
                rollout = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                malformed_rollout_summary += 1
                continue
            label = _label_for_skill_idx(rollout, gate_row.get("skill_idx"))
            if not label:
                missing_label += 1
                continue
            if str(label.get("skill_name", "")) != candidate_skill:
                skill_mismatch += 1
                continue
            safe_label = str(label.get("safe_label", ""))
            if safe_label not in {"safe", "unsafe"}:
                continue
            key = _dedupe_key(dedupe_key, episode_dir, gate_row, candidate_skill)
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
            label_int = 1 if safe_label == "unsafe" else 0
            row = {
                "sample_id": _sample_id(episode_dir, gate_row, candidate_skill),
                "phase4_split": "calibration",
                "skill_name": candidate_skill,
                "label": label_int,
                "safe_label": safe_label,
                "raw_critic_risk": risk,
                "hard_contract_score": hard_score,
                "hard_contract_fixed_reject": 0,
                "triggered_violation_types": json.dumps(label.get("triggered_violation_types", []), sort_keys=True),
                "time_to_violation": "" if label.get("time_to_violation") is None else label.get("time_to_violation"),
                "summary_path": str(summary_path),
                "online_method": method,
                "online_gate_decisions_csv": str(gate_path),
                "episode_dir": str(episode_dir),
                "seed": gate_row.get("seed", ""),
                "episode_index": gate_row.get("episode_index", ""),
                "skill_idx": gate_row.get("skill_idx", ""),
            }
            rows.append(row)
            counters["total"][candidate_skill] += 1
            counters[safe_label][candidate_skill] += 1
            method_skill_counts[method][candidate_skill] += 1

    per_skill = {}
    for skill in MAIN_SKILLS:
        total = counters["total"][skill]
        unsafe = counters["unsafe"][skill]
        per_skill[skill] = {
            "total": total,
            "safe": counters["safe"][skill],
            "unsafe": unsafe,
            "dangerous_ge_min": unsafe >= min_unsafe_per_skill,
        }
    summary = {
        "calibration_distribution": "hard_contract_filtered_online_allowed_decisions",
        "dedupe_key": dedupe_key,
        "method_filter": "all" if methods is None else sorted(methods),
        "gate_decision_files": len(gate_files),
        "candidate_decision_rows": candidate_rows,
        "calibration_rows": len(rows),
        "duplicate_rows_removed": duplicate_rows,
        "missing_rollout_summary": missing_rollout_summary,
        "malformed_rollout_summary": malformed_rollout_summary,
        "missing_label": missing_label,
        "skill_mismatch": skill_mismatch,
        "per_skill": per_skill,
        "method_skill_counts": {method: dict(counter) for method, counter in sorted(method_skill_counts.items())},
        "checks": {
            "calibration_rows_present": len(rows) > 0,
            "all_calibration_rows_hard_contract_accepted": all(
                str(row.get("hard_contract_fixed_reject")) in {"0", "False", "false"} for row in rows
            ),
            "each_skill_unsafe_ge_min": all(stats["dangerous_ge_min"] for stats in per_skill.values()),
        },
        "notes": [
            "Rows are included only when hard_contract_fixed_reject=0, the gate allowed the candidate, and the executed skill matches the candidate skill.",
            "The oracle label is read from the matching rollout_summary skill_idx and must match the candidate skill.",
        ],
    }
    return rows, summary


def _base_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {field: row.get(field, "") for field in BASE_FIELDS}
    output["phase4_split"] = str(row.get("phase4_split"))
    output["label"] = int(row.get("label_int", row.get("label", 0)))
    output["raw_critic_risk"] = float(row.get("raw_critic_risk_float", row.get("raw_critic_risk", 0.0)))
    output["hard_contract_score"] = float(row.get("hard_contract_score_float", row.get("hard_contract_score", 0.0)))
    output["hard_contract_fixed_reject"] = int(bool(row.get("hard_contract_fixed_reject_bool")))
    return output


def _episode_dir(online_dir: Path, row: dict[str, Any]) -> Path:
    method = str(row.get("method", ""))
    seed = str(row.get("seed", ""))
    episode_index = int(float(row.get("episode_index", 0)))
    return online_dir / method / f"seed_{seed}" / f"episode_{episode_index:04d}"


def _label_for_skill_idx(rollout: dict[str, Any], skill_idx: Any) -> dict[str, Any] | None:
    try:
        key = str(int(float(skill_idx)))
    except (TypeError, ValueError):
        key = str(skill_idx)
    for label in rollout.get("skill_labels", []):
        if str(label.get("skill_idx")) == key:
            return label
    return None


def _dedupe_key(dedupe_key: str, episode_dir: Path, row: dict[str, Any], skill: str) -> tuple[str, ...]:
    if dedupe_key == "none":
        return (
            str(episode_dir),
            str(row.get("skill_idx", "")),
            skill,
            str(row.get("raw_critic_risk", "")),
            str(row.get("result_status", "")),
            str(row.get("result_termination_reason", "")),
        )
    return (str(episode_dir.resolve()), str(row.get("skill_idx", "")), skill)


def _sample_id(episode_dir: Path, row: dict[str, Any], skill: str) -> str:
    method = str(row.get("method", ""))
    seed = str(row.get("seed", ""))
    episode_index = str(row.get("episode_index", ""))
    skill_idx = str(row.get("skill_idx", ""))
    digest = hashlib.sha1(str(episode_dir).encode("utf-8")).hexdigest()[:12]
    return f"online_hc:{method}:seed_{seed}:episode_{episode_index}:skill_{skill_idx}:{skill}:{digest}"


def _is_false(value: Any) -> bool:
    return str(value).strip().lower() in {"0", "false", "f", "no", "n"}


def _finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(output):
        return None
    return output


if __name__ == "__main__":
    main()
