"""Build CASA Phase 4 ID Dataset v1 splits from rollout logs.

Phase 4 has stricter dataset acceptance criteria than the Phase 3 feasibility
dataset. This builder reuses the same state-skill-label feature extractor, but
emits Phase 4 artifacts with train/calibration/test splits and an explicit
strict-vs-bootstrap status.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.dataset import build_invocation_dataset, write_dataset_outputs
from gear_sonic.casa.dataset.invocation_dataset import DatasetBuildResult


MAIN_SKILLS = ("walk", "turn", "gesture", "passive")
DEFAULT_PHASE3_MANIFEST = (
    REPO_ROOT
    / "outputs/casa/phase3_feasibility/phase3_hybrid_phase2v2_manifest_20260521/source_manifest.json"
)
CRITICAL_ROW_FIELDS = (
    "sample_id",
    "source_root",
    "run_id",
    "rollout_id",
    "skill_idx",
    "skill_name",
    "safe_label",
    "params_json",
    "start_wall_time",
    "pre_state_wall_time",
    "sim_state_csv",
    "skill_events_csv",
    "summary_path",
    "split_group",
)
DEFAULT_SKILL_QUOTA = 12_500
DEFAULT_DANGEROUS_PER_SKILL_TARGET = 1_500
ARTIFACT_SOURCE_KEYWORDS = ("latency", "runtime_timeout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--episodes-root",
        action="append",
        type=Path,
        default=[],
        help="Root containing episode_*/rollout_summary.json. Can be repeated.",
    )
    parser.add_argument(
        "--source-manifest",
        action="append",
        type=Path,
        default=[],
        help="Manifest with a sources list, for example Phase 3 source_manifest.json.",
    )
    parser.add_argument("--target-samples", type=int, default=50_000)
    parser.add_argument("--target-samples-ideal", type=int, default=80_000)
    parser.add_argument("--min-per-skill", type=int, default=10_000)
    parser.add_argument("--skill-quota", type=int, default=DEFAULT_SKILL_QUOTA)
    parser.add_argument("--dangerous-per-skill-min", type=int, default=600)
    parser.add_argument("--dangerous-per-skill-target", type=int, default=DEFAULT_DANGEROUS_PER_SKILL_TARGET)
    parser.add_argument("--calibration-dangerous-per-skill-min", type=int, default=200)
    parser.add_argument("--positive-rate-min", type=float, default=0.10)
    parser.add_argument("--positive-rate-max", type=float, default=0.50)
    parser.add_argument("--pre-window-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--current-violation-grace-seconds", type=float, default=0.05)
    parser.add_argument(
        "--include-current-violations",
        action="store_true",
        help="Keep unsafe samples whose violation is already present at skill start.",
    )
    parser.add_argument(
        "--allow-runtime-artifacts",
        action="store_true",
        help="Allow runtime_timeout/injected_latency sources into the main dataset. Default is clean strict mode.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when Phase 4 strict dataset criteria are not met.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (
        REPO_ROOT / "outputs/casa" / f"phase4_dataset_v1_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    dataset_dir = output_root / "dataset_v1"
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifests = list(args.source_manifest)
    if not source_manifests and not args.episodes_root and DEFAULT_PHASE3_MANIFEST.exists():
        source_manifests.append(DEFAULT_PHASE3_MANIFEST)
    source_roots = _resolve_sources(args.episodes_root, source_manifests)
    if not source_roots:
        raise SystemExit("No Phase 4 source roots were provided or discovered.")

    full_result = build_invocation_dataset(
        episodes_roots=source_roots,
        target_samples=0,
        pre_window_seconds=args.pre_window_seconds,
        exclude_unverified=True,
        min_per_skill=0,
        positive_rate_min=args.positive_rate_min,
        positive_rate_max=args.positive_rate_max,
        seed=args.seed,
        exclude_current_violations=not args.include_current_violations,
        current_violation_grace_seconds=args.current_violation_grace_seconds,
    )

    full_splits = np.asarray(
        ["calibration" if str(split) == "val" else str(split) for split in full_result.splits],
        dtype="<U11",
    )
    clean_indices, excluded_rows, runtime_quarantine = _partition_clean_indices(
        full_result.rows,
        allow_runtime_artifacts=args.allow_runtime_artifacts,
    )
    selected_indices, selection_summary = _select_phase4_indices(
        rows=full_result.rows,
        splits=full_splits,
        clean_indices=clean_indices,
        args=args,
    )
    result = _subset_result(full_result, selected_indices, full_splits)
    phase4_splits = np.asarray(
        [str(split) for split in result.splits],
        dtype="<U11",
    )
    for row, split in zip(result.rows, phase4_splits):
        row["phase4_split"] = str(split)
        row["clean_filter_status"] = "accepted"
        row["exclude_reason"] = ""

    write_dataset_outputs(result, dataset_dir)
    _write_split_outputs(dataset_dir, result.rows, result.features, result.labels, result.skill_type_ids, phase4_splits, result.feature_schema)
    _write_csv(dataset_dir / "excluded_invocations.csv", excluded_rows)
    _write_csv(dataset_dir / "runtime_stress_quarantine.csv", runtime_quarantine)

    split_audit = _split_audit(result.rows, phase4_splits)
    source_manifest = {
        "phase": "CASA Phase4 Dataset v1",
        "sources": [str(path) for path in source_roots],
        "source_manifests": [str(path) for path in source_manifests],
        "output_root": str(output_root),
        "dataset_dir": str(dataset_dir),
        "seed": args.seed,
    }
    phase4_summary = _phase4_summary(
        args=args,
        builder_summary=result.summary,
        rows=result.rows,
        all_rows=full_result.rows_all,
        features=result.features,
        splits=phase4_splits,
        split_audit=split_audit,
        source_manifest=source_manifest,
        clean_indices=clean_indices,
        excluded_rows=excluded_rows,
        runtime_quarantine=runtime_quarantine,
        selection_summary=selection_summary,
    )

    (output_root / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    (dataset_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    (dataset_dir / "builder_summary.json").write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n")
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(phase4_summary, indent=2, sort_keys=True) + "\n")
    (dataset_dir / "split_audit.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True) + "\n")
    (dataset_dir / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2, sort_keys=True) + "\n")
    (output_root / "dataset_summary.json").write_text(json.dumps(phase4_summary, indent=2, sort_keys=True) + "\n")
    (output_root / "phase4_dataset_build.md").write_text(_dataset_markdown(phase4_summary) + "\n")

    print(json.dumps(phase4_summary, indent=2, sort_keys=True))
    if args.strict and not phase4_summary["strict_go"]:
        raise SystemExit(2)


def _resolve_sources(roots: list[Path], manifests: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        _append_source(resolved, seen, root)
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        sources = manifest.get("sources") or manifest.get("seed_pools") or []
        if not isinstance(sources, list):
            raise SystemExit(f"Manifest {manifest_path} must contain a list field named sources or seed_pools.")
        for source in sources:
            _append_source(resolved, seen, Path(str(source)))
    return resolved


def _append_source(output: list[Path], seen: set[str], path: Path) -> None:
    if not path.is_absolute():
        path = REPO_ROOT / path
    key = str(path.resolve()) if path.exists() else str(path)
    if key in seen:
        return
    seen.add(key)
    output.append(path)


def _write_split_outputs(
    dataset_dir: Path,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    labels: np.ndarray,
    skill_type_ids: np.ndarray,
    splits: np.ndarray,
    feature_schema: dict[str, Any],
) -> None:
    split_to_filename = {
        "train": "train_invocations.csv",
        "calibration": "calibration_invocations.csv",
        "test": "test_invocations.csv",
    }
    feature_names = np.asarray(feature_schema["feature_names"])
    for split, filename in split_to_filename.items():
        indices = np.where(splits == split)[0]
        split_rows = [rows[int(index)] for index in indices]
        _write_csv(dataset_dir / filename, split_rows)
        np.savez_compressed(
            dataset_dir / f"features_{split}.npz",
            X=features[indices],
            y=labels[indices],
            skill_type=skill_type_ids[indices],
            feature_names=feature_names,
        )


def _partition_clean_indices(
    rows: list[dict[str, Any]],
    *,
    allow_runtime_artifacts: bool,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    clean_indices: list[int] = []
    excluded_rows: list[dict[str, Any]] = []
    runtime_quarantine: list[dict[str, Any]] = []
    skill_event_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for index, row in enumerate(rows):
        reason = "" if allow_runtime_artifacts else _artifact_reason(row, skill_event_cache)
        if reason:
            excluded = dict(row)
            excluded["exclude_reason"] = reason
            excluded["clean_filter_status"] = "excluded"
            excluded_rows.append(excluded)
            if reason in {
                "runtime_timeout",
                "injected_latency",
                "runtime_artifact_source",
                "latency_artifact_source",
            }:
                runtime_quarantine.append(excluded)
            continue
        clean_indices.append(index)
    return clean_indices, excluded_rows, runtime_quarantine


def _artifact_reason(
    row: dict[str, Any],
    skill_event_cache: dict[str, dict[str, dict[str, Any]]],
) -> str:
    if _row_has_runtime_timeout(row):
        return "runtime_timeout"
    if _row_has_injected_latency(row, skill_event_cache):
        return "injected_latency"
    source_text = " ".join(
        str(row.get(field, ""))
        for field in ["source_root", "run_id", "summary_path", "skill_events_csv", "sim_state_csv"]
    ).lower()
    if "runtime_timeout" in source_text:
        return "runtime_artifact_source"
    if "latency" in source_text:
        return "latency_artifact_source"
    return ""


def _row_has_runtime_timeout(row: dict[str, Any]) -> bool:
    violation_types = _json_list(row.get("triggered_violation_types"))
    if any(str(item) == "runtime_timeout" for item in violation_types):
        return True
    text = " ".join(str(row.get(field, "")) for field in ["termination_reason", "status"]).lower()
    return "runtime_timeout" in text


def _row_has_injected_latency(
    row: dict[str, Any],
    skill_event_cache: dict[str, dict[str, dict[str, Any]]],
) -> bool:
    params = _json_dict(row.get("params_json"))
    if _float(params.get("injected_latency_ms"), 0.0) > 0:
        return True
    skill_events_csv = str(row.get("skill_events_csv") or "")
    skill_idx = str(row.get("skill_idx") or "")
    if not skill_events_csv or not skill_idx:
        return False
    events = skill_event_cache.get(skill_events_csv)
    if events is None:
        events = _read_skill_events_by_idx(Path(skill_events_csv))
        skill_event_cache[skill_events_csv] = events
    event = events.get(skill_idx, {})
    event_params = _json_dict(event.get("params_json"))
    evidence = _json_dict(event.get("evidence_json"))
    return (
        _float(event_params.get("injected_latency_ms"), 0.0) > 0
        or _float(evidence.get("injected_latency_ms"), 0.0) > 0
    )


def _read_skill_events_by_idx(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        return {str(row.get("skill_idx")): row for row in csv.DictReader(file)}


def _select_phase4_indices(
    *,
    rows: list[dict[str, Any]],
    splits: np.ndarray,
    clean_indices: list[int],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    rng = random.Random(args.seed)
    selected: list[int] = []
    selected_set: set[int] = set()
    clean_set = set(clean_indices)
    per_skill_summary: dict[str, dict[str, Any]] = {}

    for skill in MAIN_SKILLS:
        skill_indices = [index for index in clean_indices if rows[index].get("skill_name") == skill]
        if len(skill_indices) <= args.skill_quota:
            chosen = list(skill_indices)
        else:
            chosen = _select_skill_quota(
                rows=rows,
                splits=splits,
                indices=skill_indices,
                quota=args.skill_quota,
                dangerous_target=args.dangerous_per_skill_target,
                calibration_dangerous_target=args.calibration_dangerous_per_skill_min,
                rng=rng,
            )
        selected.extend(chosen)
        selected_set.update(chosen)
        per_skill_summary[skill] = {
            "available": len(skill_indices),
            "selected": len(chosen),
            "available_unsafe": sum(1 for index in skill_indices if rows[index].get("safe_label") == "unsafe"),
            "selected_unsafe": sum(1 for index in chosen if rows[index].get("safe_label") == "unsafe"),
            "selected_calibration_unsafe": sum(
                1
                for index in chosen
                if rows[index].get("safe_label") == "unsafe" and str(splits[index]) == "calibration"
            ),
        }

    if len(selected) < args.target_samples:
        fill_pool = [index for index in clean_indices if index not in selected_set]
        fill_pool = _prefer_safe_then_unsafe(rows, fill_pool, rng)
        for index in fill_pool:
            if len(selected) >= args.target_samples:
                break
            selected.append(index)
            selected_set.add(index)

    selected = selected[: args.target_samples]
    selected.sort(key=lambda index: rows[index].get("sample_id", ""))
    summary = {
        "target_samples": args.target_samples,
        "skill_quota": args.skill_quota,
        "dangerous_per_skill_target": args.dangerous_per_skill_target,
        "clean_available": len(clean_indices),
        "selected": len(selected),
        "per_skill": per_skill_summary,
        "cross_skill_fill_count": max(0, len(selected) - sum(min(stats["available"], args.skill_quota) for stats in per_skill_summary.values())),
        "notes": [
            "Selection first tries a balanced per-skill quota, then cross-fills from remaining clean candidates only if needed for 50k.",
            "Unsafe selection is split-aware: calibration unsafe samples are prioritized before other unsafe samples.",
        ],
    }
    return selected, summary


def _select_skill_quota(
    *,
    rows: list[dict[str, Any]],
    splits: np.ndarray,
    indices: list[int],
    quota: int,
    dangerous_target: int,
    calibration_dangerous_target: int,
    rng: random.Random,
) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()

    calibration_unsafe = [
        index for index in indices if rows[index].get("safe_label") == "unsafe" and str(splits[index]) == "calibration"
    ]
    _add_sample(selected, selected_set, calibration_unsafe, min(calibration_dangerous_target, quota), rng)

    unsafe_remaining = [
        index for index in indices if rows[index].get("safe_label") == "unsafe" and index not in selected_set
    ]
    unsafe_need = max(0, min(dangerous_target, quota) - len(selected))
    _add_sample(selected, selected_set, unsafe_remaining, unsafe_need, rng)

    safe_remaining = [
        index for index in indices if rows[index].get("safe_label") == "safe" and index not in selected_set
    ]
    _add_sample(selected, selected_set, safe_remaining, quota - len(selected), rng)

    remaining = [index for index in indices if index not in selected_set]
    _add_sample(selected, selected_set, remaining, quota - len(selected), rng)
    return selected[:quota]


def _prefer_safe_then_unsafe(rows: list[dict[str, Any]], indices: list[int], rng: random.Random) -> list[int]:
    safe = [index for index in indices if rows[index].get("safe_label") == "safe"]
    unsafe = [index for index in indices if rows[index].get("safe_label") == "unsafe"]
    rng.shuffle(safe)
    rng.shuffle(unsafe)
    return safe + unsafe


def _add_sample(
    selected: list[int],
    selected_set: set[int],
    candidates: list[int],
    count: int,
    rng: random.Random,
) -> None:
    if count <= 0:
        return
    candidates = [index for index in candidates if index not in selected_set]
    rng.shuffle(candidates)
    for index in candidates[:count]:
        selected.append(index)
        selected_set.add(index)


def _subset_result(full_result: DatasetBuildResult, indices: list[int], splits: np.ndarray) -> DatasetBuildResult:
    rows = [dict(full_result.rows[index]) for index in indices]
    features = full_result.features[np.asarray(indices, dtype=np.int64)] if indices else full_result.features[:0]
    labels = full_result.labels[np.asarray(indices, dtype=np.int64)] if indices else full_result.labels[:0]
    skill_type_ids = (
        full_result.skill_type_ids[np.asarray(indices, dtype=np.int64)] if indices else full_result.skill_type_ids[:0]
    )
    subset_splits = splits[np.asarray(indices, dtype=np.int64)] if indices else splits[:0]
    return DatasetBuildResult(
        rows_all=full_result.rows_all,
        rows=rows,
        features=features,
        labels=labels,
        skill_type_ids=skill_type_ids,
        splits=subset_splits,
        feature_schema=full_result.feature_schema,
        summary=full_result.summary,
    )


def _phase4_summary(
    *,
    args: argparse.Namespace,
    builder_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    features: np.ndarray,
    splits: np.ndarray,
    split_audit: dict[str, Any],
    source_manifest: dict[str, Any],
    clean_indices: list[int],
    excluded_rows: list[dict[str, Any]],
    runtime_quarantine: list[dict[str, Any]],
    selection_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_count = len(rows)
    label_counts = Counter(str(row.get("safe_label", "")) for row in rows)
    positive_rate = label_counts.get("unsafe", 0) / selected_count if selected_count else 0.0
    per_skill = _per_skill_counts(rows)
    per_split = _per_split_counts(rows, splits)
    missing_rate = _critical_field_missing_rate(rows)
    nonfinite_rate = _nonfinite_feature_rate(features)
    label_missing_rate = 1.0 - (
        builder_summary.get("label_complete_samples", 0) / max(builder_summary.get("raw_samples", len(all_rows)), 1)
    )
    calibration_rows = [row for row, split in zip(rows, splits) if str(split) == "calibration"]
    calibration_per_skill = _per_skill_counts(calibration_rows)
    excluded_reason_counts = Counter(str(row.get("exclude_reason", "")) for row in excluded_rows)
    clean_checks = {
        "runtime_timeout_zero": not _selected_has_runtime_timeout(rows),
        "injected_latency_zero": not _selected_has_injected_latency(rows),
        "runtime_artifact_source_zero": not any(_artifact_source_row(row) for row in rows),
        "excluded_or_quarantined_artifacts_recorded": len(excluded_rows) == 0 or len(runtime_quarantine) > 0,
    }

    per_skill_sample_checks = {
        skill: per_skill[skill]["total"] >= args.min_per_skill for skill in MAIN_SKILLS
    }
    per_skill_dangerous_checks = {
        skill: per_skill[skill]["unsafe"] >= args.dangerous_per_skill_min for skill in MAIN_SKILLS
    }
    calibration_dangerous_checks = {
        skill: calibration_per_skill[skill]["unsafe"] >= args.calibration_dangerous_per_skill_min
        for skill in MAIN_SKILLS
    }
    checks = {
        "total_samples_ge_50k": selected_count >= args.target_samples,
        "target_80k_met": selected_count >= args.target_samples_ideal,
        "per_skill_samples_ge_10k": all(per_skill_sample_checks.values()),
        "per_skill_dangerous_ge_600": all(per_skill_dangerous_checks.values()),
        "calibration_per_skill_dangerous_ge_200": all(calibration_dangerous_checks.values()),
        "split_group_no_leakage": split_audit["split_group_cross_split_count"] == 0,
        "sim_state_csv_no_leakage": all(value == 0 for value in split_audit["sim_state_csv_cross_split_intersections"].values()),
        "label_missing_rate_le_0p02": label_missing_rate <= 0.02,
        "critical_field_missing_rate_le_0p02": missing_rate <= 0.02,
        "nonfinite_feature_rate_le_0p02": nonfinite_rate <= 0.02,
        "unsafe_positive_rate_in_10_50pct": args.positive_rate_min <= positive_rate <= args.positive_rate_max,
        **clean_checks,
    }
    strict_go = all(passed for name, passed in checks.items() if name != "target_80k_met")
    deficits = _deficits(
        selected_count=selected_count,
        args=args,
        per_skill=per_skill,
        calibration_per_skill=calibration_per_skill,
    )
    return {
        "phase": "CASA Phase4 Dataset v1",
        "status": "PASS_STRICT" if strict_go else "BOOTSTRAP_ONLY",
        "strict_go": strict_go,
        "bootstrap_usable": selected_count > 0 and checks["split_group_no_leakage"],
        "selected_samples": selected_count,
        "target_samples_min": args.target_samples,
        "target_samples_ideal": args.target_samples_ideal,
        "raw_samples": len(all_rows),
        "usable_samples": builder_summary.get("usable_samples"),
        "clean_available_samples": len(clean_indices),
        "excluded_samples": len(excluded_rows),
        "runtime_stress_quarantine_samples": len(runtime_quarantine),
        "excluded_reason_counts": dict(excluded_reason_counts),
        "label_complete_samples": builder_summary.get("label_complete_samples"),
        "label_missing_rate": label_missing_rate,
        "critical_field_missing_rate": missing_rate,
        "nonfinite_feature_rate": nonfinite_rate,
        "positive_rate": positive_rate,
        "label_counts": dict(label_counts),
        "per_skill": per_skill,
        "per_split": per_split,
        "calibration_per_skill": calibration_per_skill,
        "checks": checks,
        "per_skill_sample_checks": per_skill_sample_checks,
        "per_skill_dangerous_checks": per_skill_dangerous_checks,
        "calibration_dangerous_checks": calibration_dangerous_checks,
        "deficits": deficits,
        "selection_warnings": builder_summary.get("selection_warnings", []),
        "selection_summary": selection_summary,
        "source_manifest": source_manifest,
        "split_audit_path": "split_audit.json",
        "notes": [
            "Phase 4 strict acceptance requires >=50k selected samples and >=10k samples per main skill.",
            "The calibration split is the deterministic 15% rollout-level split formerly named val in Phase 3 utilities.",
            "When strict_go is false, outputs are valid for bootstrap/training smoke tests but not final Phase 4 acceptance.",
            "Strict clean mode excludes runtime_timeout, injected_latency_ms, and latency/runtime artifact sources from the main dataset.",
        ],
    }


def _per_skill_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for skill in MAIN_SKILLS:
        skill_rows = [row for row in rows if row.get("skill_name") == skill]
        counts = Counter(str(row.get("safe_label", "")) for row in skill_rows)
        total = len(skill_rows)
        output[skill] = {
            "total": total,
            "safe": counts.get("safe", 0),
            "unsafe": counts.get("unsafe", 0),
            "positive_rate": counts.get("unsafe", 0) / total if total else 0.0,
        }
    return output


def _per_split_counts(rows: list[dict[str, Any]], splits: np.ndarray) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for split in ("train", "calibration", "test"):
        split_rows = [row for row, row_split in zip(rows, splits) if str(row_split) == split]
        counts = Counter(str(row.get("safe_label", "")) for row in split_rows)
        total = len(split_rows)
        output[split] = {
            "total": total,
            "safe": counts.get("safe", 0),
            "unsafe": counts.get("unsafe", 0),
            "positive_rate": counts.get("unsafe", 0) / total if total else 0.0,
        }
    return output


def _split_audit(rows: list[dict[str, Any]], splits: np.ndarray) -> dict[str, Any]:
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    sim_by_split: dict[str, set[str]] = defaultdict(set)
    for row, split_value in zip(rows, splits):
        split = str(split_value)
        group = str(row.get("split_group", ""))
        sim_path = str(row.get("sim_state_csv", ""))
        groups_by_split[split].add(group)
        split_by_group[group].add(split)
        sim_by_split[split].add(sim_path)

    split_names = sorted(groups_by_split)
    group_intersections = {}
    sim_intersections = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            group_intersections[f"{left}_vs_{right}"] = len(groups_by_split[left] & groups_by_split[right])
            sim_intersections[f"{left}_vs_{right}"] = len(sim_by_split[left] & sim_by_split[right])
    return {
        "protocol": "rollout-level deterministic hash split using split_group=source_root:run_id:rollout_id",
        "split_counts": dict(Counter(str(item) for item in splits)),
        "split_group_count": len(split_by_group),
        "split_group_cross_split_count": sum(1 for values in split_by_group.values() if len(values) > 1),
        "split_group_intersections": group_intersections,
        "sim_state_csv_cross_split_intersections": sim_intersections,
    }


def _deficits(
    *,
    selected_count: int,
    args: argparse.Namespace,
    per_skill: dict[str, dict[str, Any]],
    calibration_per_skill: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "samples_to_50k": max(0, args.target_samples - selected_count),
        "samples_to_80k": max(0, args.target_samples_ideal - selected_count),
        "per_skill_samples_to_10k": {
            skill: max(0, args.min_per_skill - per_skill[skill]["total"]) for skill in MAIN_SKILLS
        },
        "per_skill_dangerous_to_600": {
            skill: max(0, args.dangerous_per_skill_min - per_skill[skill]["unsafe"]) for skill in MAIN_SKILLS
        },
        "calibration_dangerous_to_200": {
            skill: max(0, args.calibration_dangerous_per_skill_min - calibration_per_skill[skill]["unsafe"])
            for skill in MAIN_SKILLS
        },
    }


def _selected_has_runtime_timeout(rows: list[dict[str, Any]]) -> bool:
    return any(_row_has_runtime_timeout(row) for row in rows)


def _selected_has_injected_latency(rows: list[dict[str, Any]]) -> bool:
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    return any(_row_has_injected_latency(row, cache) for row in rows)


def _artifact_source_row(row: dict[str, Any]) -> bool:
    source_text = " ".join(
        str(row.get(field, ""))
        for field in ["source_root", "run_id", "summary_path", "skill_events_csv", "sim_state_csv"]
    ).lower()
    return any(keyword in source_text for keyword in ARTIFACT_SOURCE_KEYWORDS)


def _critical_field_missing_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    total = len(rows) * len(CRITICAL_ROW_FIELDS)
    missing = sum(1 for row in rows for field in CRITICAL_ROW_FIELDS if row.get(field) in (None, ""))
    return missing / total


def _nonfinite_feature_rate(features: np.ndarray) -> float:
    if features.size == 0:
        return 1.0
    return float((~np.isfinite(features)).sum() / features.size)


def _dataset_markdown(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    lines = [
        "# CASA Phase4 Dataset v1 Build",
        "",
        f"- status: `{summary['status']}`",
        f"- strict_go: `{summary['strict_go']}`",
        f"- selected_samples: `{summary['selected_samples']}`",
        f"- clean_available_samples: `{summary['clean_available_samples']}`",
        f"- excluded_samples: `{summary['excluded_samples']}`",
        f"- runtime_stress_quarantine_samples: `{summary['runtime_stress_quarantine_samples']}`",
        f"- positive_rate: `{summary['positive_rate']:.4f}`",
        f"- label_missing_rate: `{summary['label_missing_rate']:.4f}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Per Skill", ""])
    for skill, stats in summary["per_skill"].items():
        lines.append(
            f"- {skill}: total `{stats['total']}`, unsafe `{stats['unsafe']}`, positive_rate `{stats['positive_rate']:.4f}`"
        )
    lines.extend(["", "## Deficits", "", "```json", json.dumps(summary["deficits"], indent=2, sort_keys=True), "```"])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON {path}: {exc}") from exc


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)] if str(value) else []
    return parsed if isinstance(parsed, list) else [parsed]


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
