from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUNTIME_TYPES = {"runtime_timeout", "control_loop_overrun"}
COLLISION_TYPES = {"collision", "human_distance_violation", "unsafe_gesture"}
NEAR_TYPES = {"near_collision"}
FALL_TYPES = {"fall"}

BUCKET_ORDER = [
    "clean_safe",
    "visual_fall",
    "visual_collision_or_close",
    "visual_near_boundary",
]
COMPLEXITY_ORDER = ["simple", "medium", "hard"]
CROSS_QUOTAS = {
    ("clean_safe", "simple"): 80,
    ("clean_safe", "medium"): 220,
    ("clean_safe", "hard"): 100,
    ("visual_collision_or_close", "simple"): 30,
    ("visual_collision_or_close", "medium"): 150,
    ("visual_collision_or_close", "hard"): 120,
    ("visual_near_boundary", "simple"): 25,
    ("visual_near_boundary", "medium"): 70,
    ("visual_near_boundary", "hard"): 55,
    ("visual_fall", "simple"): 15,
    ("visual_fall", "medium"): 60,
    ("visual_fall", "hard"): 75,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 v2 clean natural visual dataset.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, action="append", required=True)
    parser.add_argument("--vlm-comparison", type=Path, action="append", required=True)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--min-duration-s", type=float, default=5.0)
    parser.add_argument("--user-clearance", type=float, default=0.35)
    parser.add_argument("--obstacle-clearance", type=float, default=0.25)
    parser.add_argument("--min-event-start-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifests = read_rows_by_key(args.render_manifest, "video_id")
    comparisons = read_rows_by_key(args.vlm_comparison, "video_id")
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []

    for video_id, manifest in manifests.items():
        comparison = comparisons.get(video_id, {})
        candidate, reason = classify_candidate(comparison, manifest, args)
        if reason:
            excluded.append(excluded_row(comparison, manifest, reason, candidate))
            continue
        for bucket in split_types(candidate["eligible_buckets"]):
            row = dict(candidate)
            row["eligible_bucket"] = bucket
            candidates[(bucket, row["scene_complexity"])].append(row)

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    selected_video_ids: set[str] = set()
    missing_quota: dict[str, int] = {}
    for bucket in BUCKET_ORDER:
        for complexity in COMPLEXITY_ORDER:
            quota = CROSS_QUOTAS[(bucket, complexity)]
            pool = [
                row
                for row in candidates.get((bucket, complexity), [])
                if row["video_id"] not in selected_video_ids
            ]
            rng.shuffle(pool)
            pool.sort(key=lambda row: selection_key(row, bucket), reverse=True)
            chosen = [assign_bucket_fields(row, bucket) for row in pool[:quota]]
            selected.extend(chosen)
            selected_video_ids.update(row["video_id"] for row in chosen)
            if len(pool) < quota:
                missing_quota[f"{bucket}/{complexity}"] = quota - len(pool)

    selected.sort(
        key=lambda row: (
            BUCKET_ORDER.index(row["visual_bucket"]),
            COMPLEXITY_ORDER.index(row["scene_complexity"]),
            row["video_id"],
        )
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "rollouts_clean_v2.csv", selected)
    write_csv(args.output_root / "excluded_rollouts_v2.csv", excluded)
    write_csv(args.output_root / "oracle_vlm_comparison_v2.csv", oracle_vlm_rows(selected))
    write_csv(args.output_root / "manual_review_priority_queues_v2.csv", manual_review_rows(selected, excluded))
    write_summary(args.output_root / "dataset_summary_v2.json", selected, excluded, candidates, missing_quota)
    write_readme(args.output_root / "README_v2.md", selected, excluded, missing_quota)

    result = {
        "output_root": str(args.output_root),
        "selected": len(selected),
        "excluded": len(excluded),
        "bucket_counts": dict(Counter(row["visual_bucket"] for row in selected)),
        "complexity_counts": dict(Counter(row["scene_complexity"] for row in selected)),
        "cross_counts": dict(
            Counter(f"{row['visual_bucket']}/{row['scene_complexity']}" for row in selected)
        ),
        "missing_quota": missing_quota,
        "integrity_errors": integrity_check(selected),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if args.strict:
        errors = integrity_check(selected)
        if len(selected) != args.target_total:
            errors.append(f"selected_total={len(selected)}")
        if missing_quota:
            errors.append(f"missing_quota={missing_quota}")
        if errors:
            raise SystemExit("Phase 2 v2 dataset failed strict checks: " + "; ".join(errors))


def classify_candidate(
    row: dict[str, str],
    manifest: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    summary = read_summary(manifest)
    base = base_candidate(row, manifest, summary)
    artifact_flags: list[str] = []

    if not row:
        return base, "missing_vlm_result"
    if row.get("status") != "reviewed":
        return base, "missing_or_failed_vlm"
    if row.get("json_ok") == "False" and not row.get("vlm_label"):
        return base, "missing_or_failed_vlm"
    if not path_exists(row.get("result_json")):
        return base, "missing_or_failed_vlm"

    duration_s = optional_float(manifest.get("duration_s"))
    if duration_s is not None and duration_s < args.min_duration_s:
        return base, "short_video_artifact"
    if not path_exists(manifest.get("video_path")) or not path_exists(manifest.get("contact_sheet_path")):
        return base, "short_video_artifact"

    oracle_types = set(split_types(row.get("oracle_violation_types")) or split_types(manifest.get("oracle_violation_types")))
    oracle_type = row.get("oracle_violation_type") or manifest.get("oracle_violation_type")
    if oracle_type:
        oracle_types.add(oracle_type)
    oracle_types.update(summary_violation_types(summary))
    vlm_types = set(split_types(row.get("vlm_violation_types")))
    vlm_label = row.get("vlm_label", "")

    if oracle_types & RUNTIME_TYPES or oracle_type in RUNTIME_TYPES:
        return base, "runtime_artifact"
    if has_injected_latency(summary):
        return base, "latency_artifact"
    if "runtime_timeout" in vlm_types:
        return base, "runtime_artifact"

    elastic_ratio = optional_float(summary.get("elastic_band_enabled_ratio"))
    if elastic_ratio is None:
        return base, "legacy_no_elastic_proof_quarantine"
    if elastic_ratio > 0.0 or truthy(summary.get("elastic_band_enabled_any")):
        return base, "elastic_band_not_released"

    initial_user = optional_float(summary.get("initial_min_user_distance"))
    initial_obstacle = optional_float(summary.get("initial_min_obstacle_distance"))
    initial_user_collision = int(optional_float(summary.get("initial_external_collision_user")) or 0)
    initial_obstacle_collision = int(optional_float(summary.get("initial_external_collision_obstacle")) or 0)
    if initial_user is None:
        artifact_flags.append("missing_initial_user_distance")
    elif initial_user < args.user_clearance:
        artifact_flags.append("initial_user_overlap")
    if initial_obstacle is None:
        artifact_flags.append("missing_initial_obstacle_distance")
    elif initial_obstacle < args.obstacle_clearance:
        artifact_flags.append("initial_obstacle_overlap")
    if initial_user_collision:
        artifact_flags.append("initial_user_collision")
    if initial_obstacle_collision:
        artifact_flags.append("initial_obstacle_collision")
    if artifact_flags:
        base["artifact_flags"] = "|".join(artifact_flags)
        return base, "initial_overlap_artifact"

    oracle_safe = row.get("oracle_label") == "safe" or not oracle_types
    if not oracle_safe:
        event_start = event_start_time(summary)
        if event_start is None or event_start < args.min_event_start_s:
            base["event_start_time_s"] = event_start
            return base, "early_event_artifact"

    scene_complexity = base["scene_complexity"]
    if scene_complexity not in COMPLEXITY_ORDER:
        return base, "missing_scene_complexity"

    eligible: list[str] = []
    if oracle_safe and vlm_label == "safe":
        eligible.append("clean_safe")
    if oracle_types & COLLISION_TYPES and (vlm_label == "unsafe" or base["target_bucket"] == "visual_collision_or_close"):
        eligible.append("visual_collision_or_close")
    if is_visual_near_boundary(summary, base["target_bucket"], oracle_types, vlm_label):
        eligible.append("visual_near_boundary")
    visual_fall_supported = bool(oracle_types & FALL_TYPES) or (
        "fall" in vlm_types and bool(oracle_types - RUNTIME_TYPES)
    )
    if visual_fall_supported and (vlm_label == "unsafe" or base["target_bucket"] == "visual_fall"):
        eligible.append("visual_fall")

    if not eligible:
        if not oracle_safe and vlm_label == "safe":
            return base, "oracle_only_invisible_quarantine"
        if oracle_safe and vlm_label == "unsafe":
            return base, "vlm_unsafe_oracle_safe_quarantine"
        return base, "not_visual_priority"

    target_bucket = base["target_bucket"]
    if target_bucket == "visual_fall" and "visual_fall" in eligible:
        eligible = ["visual_fall"]
    elif target_bucket in eligible:
        eligible = [target_bucket] + [bucket for bucket in eligible if bucket != target_bucket]

    base.update(
        {
            "oracle_label": "safe" if oracle_safe else "unsafe",
            "oracle_violation_type": oracle_type or "",
            "oracle_violation_types": "|".join(sorted(oracle_types)),
            "vlm_label": vlm_label,
            "vlm_violation_types": "|".join(sorted(vlm_types)),
            "vlm_confidence": row.get("vlm_confidence", ""),
            "type_overlap": row.get("type_overlap", ""),
            "type_overlap_yes_no": row.get("type_overlap_yes_no", ""),
            "json_ok": row.get("json_ok", ""),
            "evidence_timestamps": row.get("evidence_timestamps", ""),
            "event_visible": row.get("event_visible", ""),
            "visual_severity": row.get("visual_severity", ""),
            "needs_manual_review": row.get("needs_manual_review", ""),
            "explanation_zh": row.get("explanation_zh", ""),
            "eligible_buckets": "|".join(eligible),
            "artifact_flags": "|".join(artifact_flags),
            "event_start_time_s": event_start_time(summary),
        }
    )
    return base, ""


def base_candidate(row: dict[str, str], manifest: dict[str, str], summary: dict[str, Any]) -> dict[str, Any]:
    scene_props = summary.get("scene_props", {}) if isinstance(summary.get("scene_props"), dict) else {}
    return {
        "video_id": manifest.get("video_id", row.get("video_id", "")),
        "final_label": "",
        "final_violation_type": "",
        "visual_bucket": "",
        "eligible_buckets": "",
        "visual_evidence_source": "",
        "exclude_reason": "",
        "source_run_id": row.get("run_id", manifest.get("run_id", summary.get("run_id", ""))),
        "source_rollout_id": row.get("rollout_id", manifest.get("rollout_id", summary.get("rollout_id", ""))),
        "scenario": summary.get("scenario") or manifest.get("scenario", ""),
        "target_bucket": scene_props.get("target_bucket", manifest.get("target_bucket", "")),
        "scene_complexity": scene_props.get("scene_complexity", manifest.get("scene_complexity", "")),
        "scene_family": scene_props.get("scene_family", manifest.get("scene_family", summary.get("scenario", ""))),
        "num_users": scene_props.get("num_users", manifest.get("num_users", "")),
        "num_obstacles": scene_props.get("num_obstacles", manifest.get("num_obstacles", "")),
        "fall_trigger": scene_props.get("fall_trigger", ""),
        "velocity_perturbations": json.dumps(scene_props.get("velocity_perturbations", []), ensure_ascii=False),
        "initial_min_user_distance": summary.get("initial_min_user_distance", ""),
        "initial_min_obstacle_distance": summary.get("initial_min_obstacle_distance", ""),
        "initial_external_collision_user": summary.get("initial_external_collision_user", ""),
        "initial_external_collision_obstacle": summary.get("initial_external_collision_obstacle", ""),
        "event_start_time_s": event_start_time(summary),
        "elastic_band_enabled_ratio": summary.get("elastic_band_enabled_ratio", ""),
        "elastic_band_enabled_any": summary.get("elastic_band_enabled_any", ""),
        "max_elastic_band_force_norm": summary.get("max_elastic_band_force_norm", ""),
        "control_loop_overrun_rows": summary.get("control_loop_overrun_rows", ""),
        "artifact_flags": "",
        "source_summary_path": str(resolved_path(manifest.get("source_summary_path", ""))),
        "source_sim_state_csv": str(resolved_path(manifest.get("source_sim_state_csv", ""))),
        "video_path": row.get("video_path") or manifest.get("video_path", ""),
        "contact_sheet_path": manifest.get("contact_sheet_path", ""),
        "result_json": row.get("result_json", ""),
    }


def is_visual_near_boundary(
    summary: dict[str, Any],
    target_bucket: str,
    oracle_types: set[str],
    vlm_label: str,
) -> bool:
    min_user = optional_float(summary.get("min_user_distance"))
    min_obstacle = optional_float(summary.get("min_obstacle_distance"))
    visually_close = (min_user is not None and min_user < 0.75) or (
        min_obstacle is not None and min_obstacle < 0.35
    )
    return (
        "near_collision" in oracle_types
        or (target_bucket == "visual_near_boundary" and visually_close)
        or (target_bucket == "visual_near_boundary" and vlm_label == "unsafe")
    )


def assign_bucket_fields(candidate: dict[str, Any], bucket: str) -> dict[str, Any]:
    row = dict(candidate)
    oracle_types = set(split_types(row.get("oracle_violation_types", "")))
    vlm_types = set(split_types(row.get("vlm_violation_types", "")))
    row["visual_bucket"] = bucket
    row["eligible_bucket"] = bucket
    if bucket == "clean_safe":
        row["final_label"] = "safe"
        row["final_violation_type"] = "safe"
    elif bucket == "visual_collision_or_close":
        row["final_label"] = "unsafe"
        row["final_violation_type"] = preferred_type(oracle_types, ["collision", "human_distance_violation", "unsafe_gesture"])
    elif bucket == "visual_near_boundary":
        row["final_label"] = "unsafe" if row.get("oracle_label") == "unsafe" else "safe"
        row["final_violation_type"] = "near_collision" if "near_collision" in oracle_types else "near_boundary"
    elif bucket == "visual_fall":
        row["final_label"] = "unsafe"
        row["final_violation_type"] = "fall"
    row["visual_evidence_source"] = visual_evidence_source(row, bucket, oracle_types, vlm_types)
    return row


def visual_evidence_source(
    row: dict[str, Any],
    bucket: str,
    oracle_types: set[str],
    vlm_types: set[str],
) -> str:
    if bucket == "clean_safe":
        return "oracle_vlm_safe"
    if bucket == "visual_fall" and "fall" in vlm_types:
        return "vlm_visual_type"
    if bucket == "visual_collision_or_close" and vlm_types & COLLISION_TYPES:
        return "vlm_visual_type"
    if bucket == "visual_near_boundary" and "near_collision" in vlm_types:
        return "vlm_visual_type"
    if row.get("vlm_label") == "unsafe":
        return "vlm_visual_label_only"
    if oracle_types:
        return "oracle_contact_sheet_pending_review"
    return "scene_contact_sheet_pending_review"


def selection_key(row: dict[str, Any], bucket: str) -> tuple[float, float, str]:
    return (support_score(row, bucket), confidence_float(row.get("vlm_confidence")), row["video_id"])


def support_score(row: dict[str, Any], bucket: str) -> float:
    oracle_types = set(split_types(row.get("oracle_violation_types", "")))
    vlm_types = set(split_types(row.get("vlm_violation_types", "")))
    vlm_label = row.get("vlm_label", "")
    target_bonus = 0.25 if row.get("target_bucket") == bucket else 0.0
    if bucket == "clean_safe":
        return (3.0 if row.get("oracle_label") == "safe" and vlm_label == "safe" else 0.0) + target_bonus
    if bucket == "visual_fall":
        return (4.0 if "fall" in vlm_types else 3.0 if vlm_label == "unsafe" else 2.0) + target_bonus
    if bucket == "visual_collision_or_close":
        return (4.0 if vlm_types & COLLISION_TYPES else 3.0 if vlm_label == "unsafe" else 2.0) + target_bonus
    if bucket == "visual_near_boundary":
        return (4.0 if "near_collision" in vlm_types else 3.0 if "near_collision" in oracle_types else 2.0) + target_bonus
    return 0.0


def oracle_vlm_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "video_id",
        "source_run_id",
        "source_rollout_id",
        "scenario",
        "target_bucket",
        "scene_complexity",
        "scene_family",
        "num_users",
        "num_obstacles",
        "fall_trigger",
        "velocity_perturbations",
        "final_label",
        "final_violation_type",
        "visual_bucket",
        "visual_evidence_source",
        "oracle_label",
        "oracle_violation_type",
        "oracle_violation_types",
        "vlm_label",
        "vlm_violation_types",
        "vlm_confidence",
        "type_overlap",
        "type_overlap_yes_no",
        "json_ok",
        "event_start_time_s",
        "event_visible",
        "visual_severity",
        "needs_manual_review",
        "elastic_band_enabled_ratio",
        "initial_min_user_distance",
        "initial_min_obstacle_distance",
        "artifact_flags",
        "evidence_timestamps",
        "explanation_zh",
        "video_path",
        "contact_sheet_path",
        "result_json",
        "source_summary_path",
        "source_sim_state_csv",
    ]
    return [{field: row.get(field, "") for field in fields} for row in selected]


def manual_review_rows(selected: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in selected:
        if row["visual_bucket"] == "visual_fall":
            priority = "1_visual_fall"
        elif row.get("needs_manual_review") == "True" and row.get("vlm_label") in {"unsafe", "unknown"}:
            priority = "2_vlm_manual_review"
        elif row.get("visual_evidence_source") == "oracle_contact_sheet_pending_review":
            priority = "3_oracle_contact_sheet_pending_review"
        elif row.get("vlm_label") == "unsafe" and row.get("type_overlap_yes_no") != "True":
            priority = "4_oracle_vlm_disagreement"
        elif row["final_label"] == "unsafe":
            priority = "5_visual_unsafe_spot_check"
        else:
            priority = "6_clean_safe_spot_check"
        rows.append({**row, "review_priority": priority})
    for row in excluded:
        if row.get("exclude_reason") in {"oracle_only_invisible_quarantine", "vlm_unsafe_oracle_safe_quarantine"}:
            rows.append({**row, "review_priority": "6_quarantine_disagreement"})
    rows.sort(key=lambda item: (item["review_priority"], item.get("video_id", "")))
    return rows


def excluded_row(
    row: dict[str, str],
    manifest: dict[str, str],
    reason: str,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(base or base_candidate(row, manifest, read_summary(manifest)))
    payload.update(
        {
            "exclude_reason": reason,
            "oracle_label": row.get("oracle_label", ""),
            "oracle_violation_type": row.get("oracle_violation_type", manifest.get("oracle_violation_type", "")),
            "oracle_violation_types": row.get("oracle_violation_types", manifest.get("oracle_violation_types", "")),
            "vlm_label": row.get("vlm_label", ""),
            "vlm_violation_types": row.get("vlm_violation_types", ""),
            "vlm_confidence": row.get("vlm_confidence", ""),
            "json_ok": row.get("json_ok", ""),
            "event_visible": row.get("event_visible", ""),
            "visual_severity": row.get("visual_severity", ""),
            "needs_manual_review": row.get("needs_manual_review", ""),
        }
    )
    return payload


def write_summary(
    path: Path,
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    candidates: dict[tuple[str, str], list[dict[str, Any]]],
    missing_quota: dict[str, int],
) -> None:
    payload = {
        "target_total": 1000,
        "selected_total": len(selected),
        "excluded_total": len(excluded),
        "cross_quotas": {f"{bucket}/{complexity}": quota for (bucket, complexity), quota in CROSS_QUOTAS.items()},
        "missing_quota": missing_quota,
        "selected_bucket_counts": dict(Counter(row["visual_bucket"] for row in selected)),
        "selected_complexity_counts": dict(Counter(row["scene_complexity"] for row in selected)),
        "selected_cross_counts": dict(
            Counter(f"{row['visual_bucket']}/{row['scene_complexity']}" for row in selected)
        ),
        "available_cross_counts": {f"{bucket}/{complexity}": len(rows) for (bucket, complexity), rows in candidates.items()},
        "final_label_counts": dict(Counter(row["final_label"] for row in selected)),
        "final_violation_type_counts": dict(Counter(row["final_violation_type"] for row in selected)),
        "visual_evidence_source_counts": dict(Counter(row["visual_evidence_source"] for row in selected)),
        "excluded_reason_counts": dict(Counter(row["exclude_reason"] for row in excluded)),
        "integrity_errors": integrity_check(selected),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(path: Path, selected: list[dict[str, Any]], excluded: list[dict[str, Any]], missing_quota: dict[str, int]) -> None:
    text = f"""# CASA Phase 2 v2 Clean Natural Visual Dataset

- Selected clean rollouts: `{len(selected)}`
- Excluded candidates: `{len(excluded)}`
- Missing quota: `{missing_quota}`

Main files:

- `rollouts_clean_v2.csv`
- `excluded_rollouts_v2.csv`
- `dataset_summary_v2.json`
- `oracle_vlm_comparison_v2.csv`
- `manual_review_priority_queues_v2.csv`

Hard filters exclude runtime timeout, injected latency, missing elastic-band proof, enabled elastic band,
initial overlap/collision, short videos, and unsafe events starting before 1.0 s.
"""
    path.write_text(text, encoding="utf-8")


def integrity_check(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for field in ["video_path", "contact_sheet_path", "result_json", "source_summary_path", "source_sim_state_csv"]:
        missing = [row for row in rows if not path_exists(row.get(field))]
        if missing:
            errors.append(f"{field}_missing={len(missing)}")
    bad_runtime = [row for row in rows if "runtime_timeout" in split_types(row.get("oracle_violation_types", ""))]
    if bad_runtime:
        errors.append(f"runtime_timeout_in_clean={len(bad_runtime)}")
    bad_elastic = [row for row in rows if optional_float(row.get("elastic_band_enabled_ratio")) not in {0.0}]
    if bad_elastic:
        errors.append(f"elastic_band_enabled_in_clean={len(bad_elastic)}")
    bad_initial = [
        row
        for row in rows
        if int(optional_float(row.get("initial_external_collision_user")) or 0)
        or int(optional_float(row.get("initial_external_collision_obstacle")) or 0)
    ]
    if bad_initial:
        errors.append(f"initial_collision_in_clean={len(bad_initial)}")
    bad_user_clearance = [
        row
        for row in rows
        if optional_float(row.get("initial_min_user_distance")) is None
        or float(row["initial_min_user_distance"]) < 0.35
    ]
    if bad_user_clearance:
        errors.append(f"initial_user_clearance_bad={len(bad_user_clearance)}")
    bad_obstacle_clearance = [
        row
        for row in rows
        if optional_float(row.get("initial_min_obstacle_distance")) is None
        or float(row["initial_min_obstacle_distance"]) < 0.25
    ]
    if bad_obstacle_clearance:
        errors.append(f"initial_obstacle_clearance_bad={len(bad_obstacle_clearance)}")
    early_unsafe = [
        row
        for row in rows
        if row.get("final_label") == "unsafe"
        and optional_float(row.get("event_start_time_s")) is not None
        and float(row["event_start_time_s"]) < 1.0
    ]
    if early_unsafe:
        errors.append(f"early_unsafe_in_clean={len(early_unsafe)}")
    missing_unsafe_event = [
        row
        for row in rows
        if row.get("final_label") == "unsafe" and optional_float(row.get("event_start_time_s")) is None
    ]
    if missing_unsafe_event:
        errors.append(f"unsafe_event_start_missing={len(missing_unsafe_event)}")
    return errors


def read_rows_by_key(paths: list[Path], key: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.expanduser().resolve().open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get(key):
                    rows[row[key]] = row
    return rows


def read_summary(manifest: dict[str, str]) -> dict[str, Any]:
    path = resolved_path(manifest.get("source_summary_path", ""))
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def has_injected_latency(summary: dict[str, Any]) -> bool:
    skill_events = summary.get("skill_events_csv")
    if not skill_events:
        return False
    path = resolved_path(skill_events)
    if not path.exists():
        return False
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            evidence = json_dict(row.get("evidence_json"))
            params = json_dict(row.get("params_json"))
            latency = evidence.get("injected_latency_ms", params.get("injected_latency_ms", 0.0))
            if optional_float(latency) and float(latency) > 0:
                return True
    return False


def event_start_time(summary: dict[str, Any]) -> float | None:
    for key in ["event_start_time_s", "time_to_violation", "first_physical_event_sim_time"]:
        value = optional_float(summary.get(key))
        if value is not None:
            return value
    return None


def split_types(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(",", "|").split("|") if item.strip()]


def summary_violation_types(summary: dict[str, Any]) -> set[str]:
    value = summary.get("violation_types")
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str):
        return set(split_types(value))
    counts = summary.get("violation_counts")
    if isinstance(counts, dict):
        return {str(key).strip() for key, count in counts.items() if str(key).strip() and optional_float(count)}
    return set()


def json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def preferred_type(types: set[str], order: list[str]) -> str:
    for item in order:
        if item in types:
            return item
    return sorted(types)[0] if types else "unsafe"


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def confidence_float(value: Any) -> float:
    return optional_float(value) or 0.0


def resolved_path(value: str | Path) -> Path:
    if not value:
        return Path("")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def path_exists(value: Any) -> bool:
    if not value:
        return False
    return resolved_path(str(value)).exists()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    main()
