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


PHYSICAL_RUNTIME_TYPES = {"runtime_timeout", "control_loop_overrun"}
COLLISION_TYPES = {"collision", "human_distance_violation", "unsafe_gesture"}
NEAR_TYPES = {"near_collision"}
FALL_TYPES = {"fall"}
NEAR_BOUNDARY_SCENARIOS = {
    "safe_near_user",
    "safe_near_obstacle",
    "safe_gesture_user",
    "future_user_center",
    "future_obstacle_offset",
}
BUCKET_ORDER = [
    "clean_safe",
    "visual_fall",
    "visual_near_boundary",
    "visual_collision_or_close",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean visual-priority CASA rollout dataset.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, action="append", required=True)
    parser.add_argument("--vlm-comparison", type=Path, action="append", required=True)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--quota-clean-safe", type=int, default=400)
    parser.add_argument("--quota-visual-collision-or-close", type=int, default=300)
    parser.add_argument("--quota-visual-near-boundary", type=int, default=150)
    parser.add_argument("--quota-visual-fall", type=int, default=150)
    parser.add_argument("--min-duration-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    quotas = {
        "clean_safe": args.quota_clean_safe,
        "visual_collision_or_close": args.quota_visual_collision_or_close,
        "visual_near_boundary": args.quota_visual_near_boundary,
        "visual_fall": args.quota_visual_fall,
    }
    if sum(quotas.values()) != args.target_total:
        raise SystemExit(f"bucket quotas sum to {sum(quotas.values())}, expected {args.target_total}")

    manifests = read_manifests(args.render_manifest)
    comparisons = read_comparisons(args.vlm_comparison)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    for video_id, row in comparisons.items():
        manifest = manifests.get(video_id)
        if manifest is None:
            excluded.append(excluded_row(row, {}, "missing_render_manifest"))
            continue
        candidate, reason = classify_candidate(row, manifest, min_duration_s=args.min_duration_s)
        if reason:
            excluded.append(excluded_row(row, manifest, reason))
            continue
        for bucket in split_types(candidate["eligible_buckets"]):
            candidates[bucket].append(candidate)

    selected: list[dict[str, Any]] = []
    selected_video_ids: set[str] = set()
    missing_quota: dict[str, int] = {}
    for bucket in BUCKET_ORDER:
        pool = [item for item in candidates.get(bucket, []) if item["video_id"] not in selected_video_ids]
        rng.shuffle(pool)
        pool.sort(key=lambda item: selection_key(item, bucket), reverse=True)
        quota = quotas[bucket]
        chosen = [assign_bucket_fields(item, bucket) for item in pool[:quota]]
        selected.extend(chosen)
        selected_video_ids.update(item["video_id"] for item in chosen)
        if len(pool) < quota:
            missing_quota[bucket] = quota - len(pool)
    selected.sort(key=lambda item: (BUCKET_ORDER.index(item["visual_bucket"]), item["video_id"]))

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "rollouts_clean.csv", selected)
    write_csv(args.output_root / "oracle_vlm_comparison.csv", oracle_vlm_rows(selected))
    write_csv(args.output_root / "excluded_rollouts.csv", excluded)
    write_csv(args.output_root / "manual_review_priority_queues.csv", manual_review_rows(selected, excluded))
    write_summary(args.output_root / "dataset_summary.json", selected, excluded, candidates, quotas, missing_quota)
    write_readme(args.output_root / "README.md", args, selected, excluded, quotas, missing_quota)

    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "selected": len(selected),
                "excluded": len(excluded),
                "bucket_counts": dict(Counter(row["visual_bucket"] for row in selected)),
                "missing_quota": missing_quota,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict:
        if len(selected) != args.target_total or missing_quota:
            raise SystemExit(f"clean dataset is incomplete: selected={len(selected)} missing={missing_quota}")
        integrity_errors = integrity_check(selected)
        if integrity_errors:
            raise SystemExit("clean dataset integrity failed: " + "; ".join(integrity_errors))


def read_manifests(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.expanduser().resolve().open(newline="") as file:
            for row in csv.DictReader(file):
                rows[row["video_id"]] = row
    return rows


def read_comparisons(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.expanduser().resolve().open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows[row["video_id"]] = row
    return rows


def classify_candidate(
    row: dict[str, str],
    manifest: dict[str, str],
    *,
    min_duration_s: float,
) -> tuple[dict[str, Any], str]:
    summary = read_summary(manifest)
    oracle_types = set(split_types(row.get("oracle_violation_types")) or split_types(manifest.get("oracle_violation_types")))
    oracle_type = row.get("oracle_violation_type") or manifest.get("oracle_violation_type")
    if oracle_type:
        oracle_types.add(oracle_type)
    vlm_types = set(split_types(row.get("vlm_violation_types")))
    vlm_label = row.get("vlm_label", "")

    if oracle_types & PHYSICAL_RUNTIME_TYPES or oracle_type in PHYSICAL_RUNTIME_TYPES:
        return {}, "runtime_artifact"
    if has_injected_latency(summary):
        return {}, "runtime_artifact_injected_latency"
    duration_s = optional_float(manifest.get("duration_s"))
    if duration_s is not None and duration_s < min_duration_s:
        return {}, "short_video_artifact"
    if row.get("status") != "reviewed":
        return {}, "missing_or_failed_vlm"
    if row.get("json_ok") == "False" and not row.get("vlm_label"):
        return {}, "unusable_vlm_json"
    if "runtime_timeout" in vlm_types:
        return {}, "vlm_runtime_timeout_quarantine"

    oracle_safe = row.get("oracle_label") == "safe" or not oracle_types
    scenario = row.get("scenario", manifest.get("scenario", ""))
    eligible_buckets = []
    if oracle_safe and vlm_label == "safe":
        eligible_buckets.append("clean_safe")
    if oracle_types & FALL_TYPES or vlm_types & FALL_TYPES:
        eligible_buckets.append("visual_fall")
    if is_near_boundary(summary, scenario, oracle_types):
        eligible_buckets.append("visual_near_boundary")
    if oracle_types & COLLISION_TYPES:
        eligible_buckets.append("visual_collision_or_close")

    if oracle_safe and vlm_label == "unsafe":
        return {}, "vlm_unsafe_oracle_safe_quarantine"
    if not eligible_buckets:
        if not oracle_safe and vlm_label == "safe":
            return {}, "oracle_only_quarantine"
        return {}, "not_visual_priority"

    summary_path = resolved_path(manifest.get("source_summary_path", ""))
    sim_path = resolved_path(manifest.get("source_sim_state_csv", ""))
    return {
        "video_id": row.get("video_id", manifest.get("video_id", "")),
        "final_label": "",
        "final_violation_type": "",
        "visual_bucket": "",
        "eligible_buckets": "|".join(eligible_buckets),
        "visual_evidence_source": "",
        "exclude_reason": "",
        "source_run_id": row.get("run_id", manifest.get("run_id", "")),
        "source_rollout_id": row.get("rollout_id", manifest.get("rollout_id", "")),
        "scenario": scenario,
        "oracle_label": row.get("oracle_label", ""),
        "oracle_violation_type": oracle_type or "",
        "oracle_violation_types": "|".join(sorted(oracle_types)),
        "vlm_label": vlm_label,
        "vlm_violation_types": "|".join(sorted(vlm_types)),
        "vlm_confidence": row.get("vlm_confidence", ""),
        "type_overlap": row.get("type_overlap", ""),
        "type_overlap_yes_no": row.get("type_overlap_yes_no", ""),
        "json_ok": row.get("json_ok", ""),
        "explanation_zh": row.get("explanation_zh", ""),
        "source_summary_path": str(summary_path),
        "source_sim_state_csv": str(sim_path),
        "video_path": row.get("video_path") or manifest.get("video_path", ""),
        "contact_sheet_path": manifest.get("contact_sheet_path", ""),
        "result_json": row.get("result_json", ""),
    }, ""


def excluded_row(row: dict[str, str], manifest: dict[str, str], reason: str) -> dict[str, Any]:
    return {
        "video_id": row.get("video_id", manifest.get("video_id", "")),
        "exclude_reason": reason,
        "source_run_id": row.get("run_id", manifest.get("run_id", "")),
        "source_rollout_id": row.get("rollout_id", manifest.get("rollout_id", "")),
        "scenario": row.get("scenario", manifest.get("scenario", "")),
        "oracle_label": row.get("oracle_label", ""),
        "oracle_violation_type": row.get("oracle_violation_type", manifest.get("oracle_violation_type", "")),
        "oracle_violation_types": row.get("oracle_violation_types", manifest.get("oracle_violation_types", "")),
        "vlm_label": row.get("vlm_label", ""),
        "vlm_violation_types": row.get("vlm_violation_types", ""),
        "vlm_confidence": row.get("vlm_confidence", ""),
        "json_ok": row.get("json_ok", ""),
        "source_summary_path": manifest.get("source_summary_path", ""),
        "video_path": row.get("video_path", manifest.get("video_path", "")),
        "contact_sheet_path": manifest.get("contact_sheet_path", ""),
        "result_json": row.get("result_json", ""),
    }


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
            try:
                if float(latency or 0.0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def is_near_boundary(summary: dict[str, Any], scenario: str, oracle_types: set[str]) -> bool:
    if scenario in NEAR_BOUNDARY_SCENARIOS:
        return True
    if "near_collision" in oracle_types:
        return True
    min_user = optional_float(summary.get("min_user_distance"))
    min_obstacle = optional_float(summary.get("min_obstacle_distance"))
    return (min_user is not None and min_user < 0.75) or (min_obstacle is not None and min_obstacle < 0.35)


def manual_review_rows(selected: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in selected:
        if row["visual_bucket"] == "visual_fall":
            priority = "1_visual_fall"
        elif row.get("visual_evidence_source") == "oracle_contact_sheet_pending_review":
            priority = "2_oracle_vlm_disagreement"
        elif row["final_label"] == "unsafe" and row.get("type_overlap_yes_no") == "True":
            priority = "3_oracle_vlm_type_overlap"
        elif row["final_label"] == "unsafe":
            priority = "4_visual_unsafe_type_check"
        else:
            priority = "5_clean_safe_spot_check"
        rows.append({**row, "review_priority": priority})
    for row in excluded:
        if row.get("exclude_reason") in {"oracle_only_quarantine", "vlm_unsafe_oracle_safe_quarantine"}:
            rows.append({**row, "review_priority": "4_quarantine_disagreement"})
    rows.sort(key=lambda item: (item["review_priority"], item.get("video_id", "")))
    return rows


def assign_bucket_fields(candidate: dict[str, Any], bucket: str) -> dict[str, Any]:
    row = dict(candidate)
    oracle_types = set(split_types(row.get("oracle_violation_types", "")))
    vlm_types = set(split_types(row.get("vlm_violation_types", "")))
    row["visual_bucket"] = bucket
    if bucket == "clean_safe":
        row["final_label"] = "safe"
        row["final_violation_type"] = "safe"
    elif bucket == "visual_fall":
        row["final_label"] = "unsafe"
        row["final_violation_type"] = "fall"
    elif bucket == "visual_near_boundary":
        row["final_label"] = "unsafe" if row.get("oracle_label") == "unsafe" else "safe"
        row["final_violation_type"] = "near_collision" if "near_collision" in oracle_types else "near_boundary"
    elif bucket == "visual_collision_or_close":
        row["final_label"] = "unsafe"
        row["final_violation_type"] = preferred_type(
            oracle_types,
            ["collision", "human_distance_violation", "unsafe_gesture"],
        )
    row["visual_evidence_source"] = visual_evidence_source(row, bucket, oracle_types, vlm_types)
    return row


def visual_evidence_source(
    row: dict[str, Any],
    bucket: str,
    oracle_types: set[str],
    vlm_types: set[str],
) -> str:
    vlm_label = row.get("vlm_label", "")
    if bucket == "clean_safe":
        return "oracle_vlm_safe"
    if bucket == "visual_fall" and "fall" in vlm_types:
        return "vlm_visual_type"
    if bucket == "visual_collision_or_close" and vlm_types & COLLISION_TYPES:
        return "vlm_visual_type"
    if bucket == "visual_near_boundary" and "near_collision" in vlm_types:
        return "vlm_visual_type"
    if vlm_label == "unsafe":
        return "vlm_visual_label_only"
    if oracle_types:
        return "oracle_contact_sheet_pending_review"
    return "scenario_contact_sheet_pending_review"


def selection_key(item: dict[str, Any], bucket: str) -> tuple[float, float, str]:
    return (
        bucket_support_score(item, bucket),
        confidence_float(item.get("vlm_confidence")),
        item["video_id"],
    )


def bucket_support_score(item: dict[str, Any], bucket: str) -> float:
    oracle_types = set(split_types(item.get("oracle_violation_types", "")))
    vlm_types = set(split_types(item.get("vlm_violation_types", "")))
    vlm_label = item.get("vlm_label", "")
    if bucket == "clean_safe":
        return 3.0 if item.get("oracle_label") == "safe" and vlm_label == "safe" else 0.0
    if bucket == "visual_fall":
        if "fall" in vlm_types:
            return 4.0
        if "fall" in oracle_types and vlm_label == "unsafe":
            return 3.0
        if "fall" in oracle_types:
            return 2.0
        return 1.0
    if bucket == "visual_collision_or_close":
        if vlm_types & COLLISION_TYPES:
            return 4.0
        if vlm_label == "unsafe":
            return 3.0
        if oracle_types & COLLISION_TYPES:
            return 2.0
        return 1.0
    if bucket == "visual_near_boundary":
        if "near_collision" in vlm_types:
            return 4.0
        if "near_collision" in oracle_types:
            return 3.0
        if vlm_label == "unsafe":
            return 2.0
        return 1.0
    return 0.0


def oracle_vlm_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "video_id",
        "source_run_id",
        "source_rollout_id",
        "scenario",
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
        "explanation_zh",
        "video_path",
        "contact_sheet_path",
        "result_json",
    ]
    return [{field: row.get(field, "") for field in fields} for row in selected]


def write_summary(
    path: Path,
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    missing_quota: dict[str, int],
) -> None:
    payload = {
        "selected_total": len(selected),
        "excluded_total": len(excluded),
        "quotas": quotas,
        "missing_quota": missing_quota,
        "selected_bucket_counts": dict(Counter(row["visual_bucket"] for row in selected)),
        "available_bucket_counts": {bucket: len(rows) for bucket, rows in candidates.items()},
        "final_label_counts": dict(Counter(row["final_label"] for row in selected)),
        "final_violation_type_counts": dict(Counter(row["final_violation_type"] for row in selected)),
        "visual_evidence_source_counts": dict(Counter(row["visual_evidence_source"] for row in selected)),
        "excluded_reason_counts": dict(Counter(row["exclude_reason"] for row in excluded)),
        "integrity_errors": integrity_check(selected),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(
    path: Path,
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    quotas: dict[str, int],
    missing_quota: dict[str, int],
) -> None:
    text = f"""# CASA Clean Visual Dataset

- Target total: `{args.target_total}`
- Selected: `{len(selected)}`
- Excluded: `{len(excluded)}`
- Quotas: `{quotas}`
- Missing quota: `{missing_quota}`

Files:

- `rollouts_clean.csv`: accepted clean visual-priority dataset rows.
- `oracle_vlm_comparison.csv`: final clean rows with oracle, VLM, bucket, and evidence-source fields.
- `excluded_rollouts.csv`: rejected candidates with explicit reasons.
- `dataset_summary.json`: aggregate counts and integrity checks.
- `manual_review_priority_queues.csv`: review queue for visible unsafe and disagreements.

Runtime artifacts are excluded if the oracle contains `runtime_timeout` or any skill evidence has
`injected_latency_ms > 0`. Short videos and videos the VLM marks as visible runtime stalls are quarantined
before dataset selection.
"""
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
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


def integrity_check(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for field in ["video_path", "contact_sheet_path", "result_json", "source_summary_path"]:
        missing = [row["video_id"] for row in rows if not row.get(field) or not Path(row[field]).exists()]
        if missing:
            errors.append(f"{field}_missing={len(missing)}")
    runtime = [row for row in rows if row.get("final_violation_type") == "runtime_timeout"]
    if runtime:
        errors.append(f"runtime_in_clean={len(runtime)}")
    vlm_runtime = [row for row in rows if "runtime_timeout" in split_types(row.get("vlm_violation_types", ""))]
    if vlm_runtime:
        errors.append(f"vlm_runtime_in_clean={len(vlm_runtime)}")
    return errors


def split_types(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(",", "|").split("|") if item.strip()]


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


def confidence_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolved_path(value: str | Path) -> Path:
    if not value:
        return Path("")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
