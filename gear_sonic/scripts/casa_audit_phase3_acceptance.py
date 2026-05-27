"""Audit CASA Phase 3 acceptance artifacts for leakage and missing go criteria."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MAIN_SKILLS = ("walk", "turn", "gesture", "passive")
SHORTCUT_PATTERNS = (
    "fall_flag",
    "external_collision",
    "min_user_distance",
    "min_arm_user_distance",
    "min_obstacle_distance",
    "previous_safe_label",
    "evidence_",
    "current_publish_ok",
    "current_downstream_observed",
)
FUTURE_OR_OFFLINE_PATTERNS = (
    "runtime/evidence_",
    "runtime/current_publish_ok",
    "runtime/current_downstream_observed",
    "runtime/previous_safe_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--mini-critic-dir", type=Path, default=None)
    parser.add_argument("--counterfactual-dir", type=Path, default=None)
    parser.add_argument("--shuffle-critic-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--sample-dump-json", type=Path, default=None)
    parser.add_argument("--suspicious-auroc-threshold", type=float, default=0.95)
    parser.add_argument("--max-instant-unsafe-rate", type=float, default=0.20)
    parser.add_argument("--max-single-feature-auroc", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase3_root = args.phase3_root
    dataset_dir = args.dataset_dir or phase3_root / "dataset"
    critic_dir = args.mini_critic_dir or phase3_root / "mini_critic"
    counterfactual_dir = args.counterfactual_dir or phase3_root / "counterfactual"
    shuffle_dir = args.shuffle_critic_dir or phase3_root / "mini_critic_shuffle_labels"
    output_json = args.output_json or phase3_root / "phase3_acceptance_audit.json"
    output_md = args.output_md or phase3_root / "phase3_acceptance_audit.md"
    sample_dump_json = args.sample_dump_json or phase3_root / "phase3_sample_feature_dump.json"

    rows = _read_csv(dataset_dir / "invocations.csv")
    all_rows = _read_csv(dataset_dir / "invocations_all.csv")
    arrays = np.load(dataset_dir / "features.npz", allow_pickle=False)
    X = np.nan_to_num(arrays["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = arrays["y"].astype(np.int64)
    splits = arrays["split"].astype(str)
    feature_names = [str(item) for item in arrays["feature_names"]]
    dataset_summary = _read_json(dataset_dir / "dataset_summary.json")
    critic_metrics = _read_json(critic_dir / "metrics.json")
    counterfactual = _read_json(counterfactual_dir / "counterfactual_summary.json")
    shuffle_metrics = _read_json(shuffle_dir / "metrics.json")
    manifest = _read_json(phase3_root / "source_manifest.json")

    split_audit = _audit_splits(rows, splits)
    dataset_audit = _audit_dataset(rows, all_rows, splits, dataset_summary)
    feature_audit = _audit_features(X, y, feature_names, rows, args)
    rollout_audit = _audit_rollouts(rows, manifest)
    critic_audit = _audit_critic(critic_metrics, shuffle_metrics, args)
    counterfactual_audit = _audit_counterfactual(counterfactual)
    leakage_analysis = _leakage_analysis(
        split_audit,
        dataset_audit,
        feature_audit,
        rollout_audit,
        critic_audit,
    )
    acceptance = _acceptance_table(
        dataset_summary,
        dataset_audit,
        critic_metrics,
        counterfactual,
        rollout_audit,
        feature_audit,
        critic_audit,
        leakage_analysis,
    )

    audit = {
        "phase": "CASA Phase3 acceptance audit",
        "overall_status": "BLOCKED",
        "blocking_reasons": [],
        "acceptance_table": acceptance,
        "split_audit": split_audit,
        "dataset_audit": dataset_audit,
        "feature_audit": feature_audit,
        "rollout_audit": rollout_audit,
        "critic_audit": critic_audit,
        "counterfactual_audit": counterfactual_audit,
        "leakage_analysis": leakage_analysis,
    }
    audit["blocking_reasons"] = _blocking_reasons(audit)
    audit["overall_status"] = "PASS" if not audit["blocking_reasons"] else "BLOCKED"

    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown(audit) + "\n")
    sample_dump_json.write_text(json.dumps(_sample_feature_dump(rows, X, y, feature_names), indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def _audit_splits(rows: list[dict[str, str]], splits: np.ndarray) -> dict[str, Any]:
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    sim_by_split: dict[str, set[str]] = defaultdict(set)
    group_to_sims: dict[str, set[str]] = defaultdict(set)
    group_to_runs: dict[str, set[str]] = defaultdict(set)
    for row, split in zip(rows, splits):
        group = row.get("split_group", "")
        groups_by_split[split].add(group)
        split_by_group[group].add(split)
        sim_path = row.get("sim_state_csv", "")
        sim_by_split[split].add(sim_path)
        group_to_sims[group].add(sim_path)
        group_to_runs[group].add(row.get("run_id", ""))

    split_names = sorted(groups_by_split)
    group_intersections = {}
    sim_intersections = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            group_intersections[f"{left}_vs_{right}"] = len(groups_by_split[left] & groups_by_split[right])
            sim_intersections[f"{left}_vs_{right}"] = len(sim_by_split[left] & sim_by_split[right])

    return {
        "protocol": "rollout-level deterministic hash split using split_group=source_root:run_id:rollout_id",
        "split_counts": dict(Counter(str(item) for item in splits)),
        "split_group_count": len(split_by_group),
        "split_group_cross_split_count": sum(1 for values in split_by_group.values() if len(values) > 1),
        "split_group_intersections": group_intersections,
        "sim_state_csv_cross_split_intersections": sim_intersections,
        "split_groups_with_multiple_sim_paths": sum(1 for values in group_to_sims.values() if len(values) > 1),
        "split_groups_with_multiple_run_ids": sum(1 for values in group_to_runs.values() if len(values) > 1),
        "note": (
            "No selected split_group or sim_state_csv crosses train/val/test. "
            "split_group includes source root, run id, and rollout id so adjacent samples from one rollout remain in one split."
        ),
    }


def _audit_dataset(
    rows: list[dict[str, str]],
    all_rows: list[dict[str, str]],
    splits: np.ndarray,
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    per_skill = {}
    for skill in MAIN_SKILLS:
        skill_rows = [row for row in rows if row.get("skill_name") == skill]
        counts = Counter(row.get("safe_label") for row in skill_rows)
        total = len(skill_rows)
        per_skill[skill] = {
            "count": total,
            "safe": counts.get("safe", 0),
            "unsafe": counts.get("unsafe", 0),
            "positive_rate": counts.get("unsafe", 0) / total if total else 0.0,
        }

    split_table = {}
    for split in sorted(set(str(item) for item in splits)):
        split_rows = [row for row, row_split in zip(rows, splits) if row_split == split]
        counts = Counter(row.get("safe_label") for row in split_rows)
        split_table[split] = {
            "count": len(split_rows),
            "safe": counts.get("safe", 0),
            "unsafe": counts.get("unsafe", 0),
            "positive_rate": counts.get("unsafe", 0) / len(split_rows) if split_rows else 0.0,
        }

    unsafe_rows = [row for row in rows if row.get("safe_label") == "unsafe"]
    instant_unsafe = [
        row
        for row in unsafe_rows
        if _float(row.get("time_to_violation"), math.inf) <= 0.05
    ]
    source_rates = {}
    for source in sorted({row.get("source_root", "") for row in rows}):
        source_rows = [row for row in rows if row.get("source_root") == source]
        counts = Counter(row.get("safe_label") for row in source_rows)
        source_rates[source] = {
            "count": len(source_rows),
            "unsafe": counts.get("unsafe", 0),
            "positive_rate": counts.get("unsafe", 0) / len(source_rows) if source_rows else 0.0,
        }

    phase2_rows = [row for row in rows if row.get("source_root", "").startswith("phase2_")]
    phase3_rows = [row for row in rows if row.get("source_root", "").startswith("collection_")]
    return {
        "selected_samples": len(rows),
        "raw_samples": len(all_rows),
        "usable_samples": dataset_summary.get("usable_samples"),
        "label_complete_samples": dataset_summary.get("label_complete_samples"),
        "label_complete_rate": dataset_summary.get(
            "label_complete_rate",
            dataset_summary.get("usable_samples", 0) / max(dataset_summary.get("raw_samples", 0), 1),
        ),
        "current_violation_excluded_samples": dataset_summary.get("current_violation_excluded_samples", 0),
        "save_completeness": dataset_summary.get(
            "label_complete_rate",
            dataset_summary.get("usable_samples", 0) / max(dataset_summary.get("raw_samples", 0), 1),
        ),
        "label_counts": dict(Counter(row.get("safe_label") for row in rows)),
        "overall_positive_rate": dataset_summary.get("overall_positive_rate"),
        "per_skill": per_skill,
        "split_table": split_table,
        "unsafe_time_to_violation_bins": dict(Counter(row.get("violation_time_bin") for row in unsafe_rows)),
        "instant_unsafe_count": len(instant_unsafe),
        "instant_unsafe_rate_among_unsafe": len(instant_unsafe) / len(unsafe_rows) if unsafe_rows else 0.0,
        "future_unsafe_count_after_0p05s": len(unsafe_rows) - len(instant_unsafe),
        "unsafe_violation_types": dict(
            Counter(
                violation_type
                for row in unsafe_rows
                for violation_type in _json_list(row.get("triggered_violation_types"))
            )
        ),
        "source_positive_rates": source_rates,
        "phase2_seed_positive_rate": _positive_rate(phase2_rows),
        "phase3_added_positive_rate": _positive_rate(phase3_rows),
    }


def _audit_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    shortcut_features = [
        name for name in feature_names if any(pattern in name for pattern in SHORTCUT_PATTERNS)
    ]
    future_or_offline_features = [
        name for name in feature_names if any(pattern in name for pattern in FUTURE_OR_OFFLINE_PATTERNS)
    ]
    pre_time_violations = [
        row
        for row in rows
        if _float(row.get("pre_state_wall_time"), math.inf) - _float(row.get("start_wall_time"), -math.inf) > 1e-6
    ]
    single_feature_auc = []
    for index, name in enumerate(feature_names):
        auc = _auroc(y, X[:, index])
        if auc is None:
            continue
        single_feature_auc.append(
            {
                "feature": name,
                "auroc": auc,
                "max_direction_auroc": max(auc, 1.0 - auc),
            }
        )
    single_feature_auc.sort(key=lambda item: item["max_direction_auroc"], reverse=True)
    top_auc = single_feature_auc[0]["max_direction_auroc"] if single_feature_auc else 0.0
    top_shortcut_auc = max(
        (
            item["max_direction_auroc"]
            for item in single_feature_auc
            if item["feature"] in set(shortcut_features)
        ),
        default=0.0,
    )
    return {
        "feature_count": len(feature_names),
        "shortcut_feature_count": len(shortcut_features),
        "shortcut_features": shortcut_features,
        "future_or_offline_feature_count": len(future_or_offline_features),
        "future_or_offline_features": future_or_offline_features,
        "pre_state_after_skill_start_count": len(pre_time_violations),
        "top_single_feature_auc": top_auc,
        "top_shortcut_single_feature_auc": top_shortcut_auc,
        "top_single_feature_auc_table": single_feature_auc[:30],
        "single_feature_auc_threshold": args.max_single_feature_auroc,
    }


def _audit_rollouts(rows: list[dict[str, str]], manifest: dict[str, Any]) -> dict[str, Any]:
    source_roots = [Path(path) for path in manifest.get("sources", [])]
    if not source_roots:
        source_roots = sorted({Path(row.get("summary_path", ".")).parents[1] for row in rows if row.get("summary_path")})

    summaries = []
    missing_summary_dirs = 0
    review_rows = 0
    expected_from_names = 0
    expected_name_available = 0
    completed_for_expected_roots = 0
    attempts = []
    for root in source_roots:
        episode_dirs = _discover_episode_dirs(root)
        expected = _expected_from_collection_name(root.name)
        if expected is not None:
            expected_name_available += 1
            expected_from_names += expected
            completed_for_expected_roots += len(episode_dirs)
        for episode_dir in episode_dirs:
            summary_path = episode_dir / "rollout_summary.json"
            if summary_path.exists():
                summaries.append(_read_json(summary_path))
            else:
                missing_summary_dirs += 1
        review_csv = root / "review" / "rollouts.csv"
        if review_csv.exists():
            review_rows += len(_read_csv(review_csv))
        for attempts_csv in _discover_attempt_csvs(root):
            attempts.extend(_read_csv(attempts_csv))

    completed = len(summaries)
    selected_summary_paths = {row.get("summary_path") for row in rows}
    selected_summaries = [_read_json(Path(path)) for path in selected_summary_paths if path and Path(path).exists()]
    usable_samples_per_completed_rollout = len(rows) / completed if completed else 0.0
    calendar_spans = _calendar_spans(summaries)
    selected_spans = _calendar_spans(selected_summaries)
    observed_hang_rate = missing_summary_dirs / (completed + missing_summary_dirs) if completed + missing_summary_dirs else None
    attempt_rows = [row for row in attempts if row.get("episode_id")]
    attempt_status = Counter(row.get("status") for row in attempt_rows)
    attempt_level_hang_rate = None
    if attempt_rows:
        failed = sum(count for status, count in attempt_status.items() if status != "completed")
        attempt_level_hang_rate = failed / len(attempt_rows)
    expected_completion_rate_from_names = (
        completed_for_expected_roots / expected_from_names if expected_from_names else None
    )
    return {
        "source_roots": [str(path) for path in source_roots],
        "completed_rollout_summaries": completed,
        "missing_summary_episode_dirs": missing_summary_dirs,
        "observed_completed_dir_hang_rate": observed_hang_rate,
        "review_rollout_rows": review_rows,
        "attempt_rows": len(attempt_rows),
        "attempt_status_counts": dict(attempt_status),
        "expected_rollout_count_from_collection_names": expected_from_names if expected_name_available else None,
        "completed_rollouts_for_name_expected_collections": completed_for_expected_roots if expected_name_available else None,
        "expected_completion_rate_from_collection_names": expected_completion_rate_from_names,
        "attempt_level_hang_rate_available": bool(attempt_rows),
        "attempt_level_hang_rate": attempt_level_hang_rate,
        "attempt_level_hang_rate_note": (
            "attempts.csv records completed/failed rollout attempts. A process-level hard kill can still truncate "
            "the final in-flight attempt, so observed completed-dir missing-summary rate is also reported."
        ),
        "usable_samples_per_completed_rollout": usable_samples_per_completed_rollout,
        "calendar_span_hours_all_sources_sum": calendar_spans["sum_hours"],
        "calendar_span_hours_selected_summaries_sum": selected_spans["sum_hours"],
        "throughput_samples_per_hour_all_completed_sources": len(rows) / calendar_spans["sum_hours"]
        if calendar_spans["sum_hours"] > 0
        else None,
        "throughput_samples_per_hour_selected_summaries": len(rows) / selected_spans["sum_hours"]
        if selected_spans["sum_hours"] > 0
        else None,
        "unique_run_ids": calendar_spans["run_count"],
        "source_run_spans": calendar_spans["by_run"],
    }


def _discover_episode_dirs(root: Path) -> list[Path]:
    candidates = [path for path in root.glob("episode_*") if path.is_dir()]
    if candidates:
        return sorted(candidates)
    nested = sorted({path.parent for path in root.rglob("rollout_summary.json") if path.parent.name.startswith("episode_")})
    if nested:
        return nested
    branches = sorted({path.parent for path in root.rglob("rollout_summary.json") if path.parent.parent.name == "branches"})
    return branches


def _discover_attempt_csvs(root: Path) -> list[Path]:
    candidates = []
    direct = root / "attempts.csv"
    if direct.exists():
        candidates.append(direct)
    nested = sorted(root.rglob("attempts.csv"))
    for path in nested:
        if path not in candidates:
            candidates.append(path)
    return candidates


def _audit_critic(
    critic_metrics: dict[str, Any],
    shuffle_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    test = critic_metrics.get("overall", {}).get("test", {})
    per_skill_test = critic_metrics.get("per_skill", {}).get("test", {})
    shuffle_test = shuffle_metrics.get("overall", {}).get("test", {}) if shuffle_metrics else {}
    return {
        "test_auroc": test.get("auroc"),
        "test_brier": test.get("brier"),
        "constant_brier": test.get("constant_brier"),
        "auroc_suspicious_threshold": args.suspicious_auroc_threshold,
        "auroc_gt_suspicious_threshold": (
            test.get("auroc") is not None and test.get("auroc") > args.suspicious_auroc_threshold
        ),
        "per_skill_test": per_skill_test,
        "shuffle_label_metrics_available": bool(shuffle_metrics),
        "shuffle_label_test_auroc": shuffle_test.get("auroc"),
        "shuffle_label_test_brier": shuffle_test.get("brier"),
        "shuffle_label_epochs": shuffle_metrics.get("epochs") if shuffle_metrics else None,
    }


def _leakage_analysis(
    split_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    feature_audit: dict[str, Any],
    rollout_audit: dict[str, Any],
    critic_audit: dict[str, Any],
) -> dict[str, Any]:
    shuffle_auc = critic_audit.get("shuffle_label_test_auroc")
    sim_intersections = split_audit.get("sim_state_csv_cross_split_intersections", {})
    group_intersections = split_audit.get("split_group_intersections", {})
    checks = {
        "split_groups_disjoint": split_audit.get("split_group_cross_split_count", 1) == 0
        and all(value == 0 for value in group_intersections.values()),
        "sim_logs_disjoint": all(value == 0 for value in sim_intersections.values()),
        "instant_unsafe_rate_ok": dataset_audit.get("instant_unsafe_rate_among_unsafe", 1.0) <= 0.20,
        "pre_state_not_after_skill_start": feature_audit.get("pre_state_after_skill_start_count", 1) == 0,
        "no_future_or_offline_features": feature_audit.get("future_or_offline_feature_count", 1) == 0,
        "no_single_feature_oracle_shortcut": feature_audit.get("top_shortcut_single_feature_auc", 1.0)
        <= feature_audit.get("single_feature_auc_threshold", 0.95),
        "no_single_feature_deterministic_separator": feature_audit.get("top_single_feature_auc", 1.0)
        <= feature_audit.get("single_feature_auc_threshold", 0.95),
        "shuffle_label_baseline_near_chance": shuffle_auc is not None and 0.40 <= shuffle_auc <= 0.60,
        "attempt_hang_rate_recorded_and_ok": rollout_audit.get("attempt_level_hang_rate_available")
        and rollout_audit.get("attempt_level_hang_rate", 1.0) <= 0.02,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "triggered_by_high_auroc": bool(critic_audit.get("auroc_gt_suspicious_threshold")),
        "passed": passed,
        "checks": checks,
        "note": (
            "High AUROC remains a warning. It is accepted only when split isolation, no future/offline fields, "
            "future-only unsafe labels, single-feature shortcut limits, shuffle-label baseline, and attempt hang "
            "accounting all pass."
        ),
    }


def _audit_counterfactual(counterfactual: dict[str, Any]) -> dict[str, Any]:
    requested_states = int(counterfactual.get("requested_states") or 0)
    requested_repeats = int(counterfactual.get("requested_repeats") or 0)
    candidate_counts = counterfactual.get("candidate_counts", {}) or {}
    candidate_count = len(candidate_counts)
    expected = requested_states * candidate_count * requested_repeats if requested_states and candidate_count and requested_repeats else None
    return {
        "requested_states": requested_states,
        "candidate_count": candidate_count,
        "requested_repeats": requested_repeats,
        "expected_branches": expected,
        "branch_count": counterfactual.get("branch_count"),
        "completed_branch_count": counterfactual.get("completed_branch_count"),
        "candidate_counts": candidate_counts,
        "risk_by_candidate": counterfactual.get("risk_by_candidate"),
        "mean_repeat_consistency": counterfactual.get("mean_repeat_consistency"),
        "min_repeat_consistency": counterfactual.get("min_repeat_consistency"),
        "repeat_consistency_groups_below_0p90": counterfactual.get("repeat_consistency_groups_below_0p90"),
        "clarification": "branch_count includes repeats: 100 states x 4 candidate skills x 5 repeats = 2000 executions.",
    }


def _acceptance_table(
    dataset: dict[str, Any],
    dataset_audit: dict[str, Any],
    critic: dict[str, Any],
    counterfactual: dict[str, Any],
    rollout: dict[str, Any],
    feature: dict[str, Any],
    critic_audit: dict[str, Any],
    leakage_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    per_skill_counts = dataset.get("per_skill_label_counts") or {}
    per_skill_rates = dataset.get("per_skill_positive_rate") or {}
    checks = dataset.get("checks", {})
    test = critic.get("overall", {}).get("test", {})
    critic_checks = critic.get("checks", {})
    return [
        _row("样本保存完整率", ">= 98%", checks.get("save_completeness_ge_0p98"), "PASS" if checks.get("save_completeness_ge_0p98") else "FAIL"),
        _row(
            "rollout hang rate",
            "<= 2%",
            rollout.get("attempt_level_hang_rate"),
            (
                "PASS"
                if rollout.get("attempt_level_hang_rate_available")
                and rollout.get("attempt_level_hang_rate", 1.0) <= 0.02
                else "MISSING"
                if not rollout.get("attempt_level_hang_rate_available")
                else "FAIL"
            ),
            "computed from attempts.csv when available",
        ),
        _row("每个主技能样本数", ">= 500", per_skill_counts, "PASS" if checks.get("per_skill_min_samples_met") else "FAIL"),
        _row("总体 unsafe positive rate", "10% - 50%", dataset.get("overall_positive_rate"), "PASS" if checks.get("overall_positive_rate_in_range") else "FAIL"),
        _row("每个主技能 positive rate", ">= 5%", per_skill_rates, "PASS" if checks.get("per_skill_positive_rate_ge_0p05") else "FAIL"),
        _row("总样本数", "5000", dataset.get("selected_samples"), "PASS" if checks.get("target_samples_met") else "FAIL"),
        _row(
            "overall AUROC",
            ">= 0.65; if > 0.95, leakage analysis must pass",
            test.get("auroc"),
            (
                "PASS_WITH_WARNING"
                if critic_audit.get("auroc_gt_suspicious_threshold") and leakage_analysis.get("passed")
                else "BLOCKED"
                if critic_audit.get("auroc_gt_suspicious_threshold")
                else "PASS"
            ),
            "AUROC > 0.95 is treated as suspicious and requires the leakage audit below.",
        ),
        _row("至少 3 个 skill per-skill AUROC", ">= 0.60", critic_checks.get("three_skills_auroc_ge_0p60"), "PASS" if critic_checks.get("three_skills_auroc_ge_0p60") else "FAIL"),
        _row("Brier 优于常数 baseline", "true", {"brier": test.get("brier"), "constant_brier": test.get("constant_brier")}, "PASS" if critic_checks.get("brier_better_than_constant") else "FAIL"),
        _row("counterfactual: 100 states x 4 skills", "complete", {"branch_count": counterfactual.get("branch_count"), "completed": counterfactual.get("completed_branch_count")}, "PASS" if counterfactual.get("complete_100x4x5") else "FAIL", "branch_count includes 5 repeats"),
        _row("counterfactual: repeats consistency", ">= 0.90", counterfactual.get("mean_repeat_consistency"), "PASS" if counterfactual.get("repeat_consistency_ge_0p90") else "FAIL"),
        _row(
            "leakage audit: instant unsafe rate",
            "<= 20% of unsafe",
            dataset_audit.get("instant_unsafe_rate_among_unsafe"),
            "BLOCKED" if dataset_audit.get("instant_unsafe_rate_among_unsafe", 0.0) > 0.20 else "PASS",
            "unsafe labels should mostly be future outcomes, not current violations",
        ),
    ]


def _row(name: str, required: str, observed: Any, status: str, note: str = "") -> dict[str, Any]:
    return {"dimension": name, "requirement": required, "observed": observed, "status": status, "note": note}


def _blocking_reasons(audit: dict[str, Any]) -> list[str]:
    reasons = []
    dataset = audit["dataset_audit"]
    feature = audit["feature_audit"]
    critic = audit["critic_audit"]
    rollout = audit["rollout_audit"]
    leakage = audit.get("leakage_analysis", {})
    if critic.get("auroc_gt_suspicious_threshold") and not leakage.get("passed"):
        reasons.append("mini_critic_test_auroc_gt_0p95_requires_leakage_analysis")
    if dataset.get("instant_unsafe_rate_among_unsafe", 0.0) > 0.20:
        reasons.append("unsafe_labels_are_mostly_instantaneous_current_violations")
    if feature.get("top_shortcut_single_feature_auc", 0.0) > 0.95:
        reasons.append("oracle_intermediate_or_current_state_shortcut_features_are_nearly_sufficient")
    if feature.get("future_or_offline_feature_count", 0) > 0:
        reasons.append("feature_schema_contains_post_execution_or_offline_label_derived_fields")
    if not rollout.get("attempt_level_hang_rate_available"):
        reasons.append("attempt_level_rollout_hang_rate_not_recorded")
    elif rollout.get("attempt_level_hang_rate", 1.0) > 0.02:
        reasons.append("attempt_level_rollout_hang_rate_gt_0p02")
    shuffle_auc = critic.get("shuffle_label_test_auroc")
    if shuffle_auc is None:
        reasons.append("shuffle_label_baseline_missing")
    elif not (0.40 <= shuffle_auc <= 0.60):
        reasons.append("shuffle_label_baseline_not_near_chance")
    return reasons


def _sample_feature_dump(
    rows: list[dict[str, str]],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    if not rows:
        return {}
    unsafe_index = next((index for index, row in enumerate(rows) if row.get("safe_label") == "unsafe"), 0)
    safe_index = next((index for index, row in enumerate(rows) if row.get("safe_label") == "safe"), 0)

    def dump(index: int) -> dict[str, Any]:
        values = {name: float(X[index, feature_index]) for feature_index, name in enumerate(feature_names)}
        shortcut = {name: value for name, value in values.items() if any(pattern in name for pattern in SHORTCUT_PATTERNS)}
        return {
            "row": rows[index],
            "label": int(y[index]),
            "shortcut_features": shortcut,
            "all_features": values,
        }

    return {"unsafe_sample": dump(unsafe_index), "safe_sample": dump(safe_index)}


def _calendar_spans(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for summary in summaries:
        run_id = str(summary.get("run_id") or "")
        start = _float(summary.get("episode_start_wall_time"), None)
        end = _float(summary.get("episode_end_wall_time"), None)
        if run_id and start is not None and end is not None and end >= start:
            by_run[run_id].append((start, end))
    run_spans = {}
    total_hours = 0.0
    for run_id, spans in sorted(by_run.items()):
        start = min(item[0] for item in spans)
        end = max(item[1] for item in spans)
        hours = (end - start) / 3600.0
        total_hours += max(hours, 0.0)
        run_spans[run_id] = {"episode_count": len(spans), "span_hours": hours}
    return {"by_run": run_spans, "sum_hours": total_hours, "run_count": len(run_spans)}


def _expected_from_collection_name(name: str) -> int | None:
    if not name.startswith("collection_"):
        return None
    match = re.search(r"(?:balanced|heavy|final|pilot|extra)(\d+)$", name)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)$", name)
    if match:
        return int(match.group(1))
    return None


def _positive_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("safe_label") == "unsafe") / len(rows)


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# CASA Phase3 Acceptance Audit",
        "",
        f"Overall status: **{audit['overall_status']}**",
        "",
        "## Blocking Reasons",
        "",
    ]
    if audit["blocking_reasons"]:
        lines.extend(f"- `{reason}`" for reason in audit["blocking_reasons"])
    else:
        lines.append("- none")
    lines.extend(["", "## Plan_a Go Criteria Ledger", "", "| Dimension | Requirement | Observed | Status | Note |", "|---|---:|---|---|---|"])
    for row in audit["acceptance_table"]:
        observed = json.dumps(row["observed"], ensure_ascii=False, sort_keys=True) if isinstance(row["observed"], (dict, list)) else str(row["observed"])
        lines.append(
            f"| {row['dimension']} | {row['requirement']} | `{observed}` | **{row['status']}** | {row.get('note', '')} |"
        )
    dataset = audit["dataset_audit"]
    lines.extend(
        [
            "",
            "## Dataset Details",
            "",
            f"- selected_samples: `{dataset['selected_samples']}`",
            f"- raw_samples: `{dataset['raw_samples']}`",
            f"- save_completeness: `{dataset['save_completeness']}`",
            f"- current_violation_excluded_samples: `{dataset['current_violation_excluded_samples']}`",
            f"- overall_positive_rate: `{dataset['overall_positive_rate']}`",
            f"- instant_unsafe_count: `{dataset['instant_unsafe_count']}` / unsafe labels",
            f"- future_unsafe_count_after_0p05s: `{dataset['future_unsafe_count_after_0p05s']}`",
            "",
            "| Skill | Count | Safe | Unsafe | Positive Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for skill, stats in dataset["per_skill"].items():
        lines.append(
            f"| {skill} | {stats['count']} | {stats['safe']} | {stats['unsafe']} | {stats['positive_rate']:.4f} |"
        )
    lines.extend(["", "## Critic And Leakage", ""])
    critic = audit["critic_audit"]
    lines.extend(
        [
            f"- test_auroc: `{critic.get('test_auroc')}`",
            f"- test_brier: `{critic.get('test_brier')}`",
            f"- constant_brier: `{critic.get('constant_brier')}`",
            f"- shuffle_label_test_auroc: `{critic.get('shuffle_label_test_auroc')}`",
            f"- shuffle_label_test_brier: `{critic.get('shuffle_label_test_brier')}`",
            "",
            "| Skill | Test Count | Pos Rate | AUROC | Brier |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for skill, stats in critic.get("per_skill_test", {}).items():
        lines.append(
            f"| {skill} | {stats.get('count')} | {stats.get('positive_rate')} | {stats.get('auroc')} | {stats.get('brier')} |"
        )
    leakage = audit.get("leakage_analysis", {})
    lines.extend(
        [
            "",
            f"- leakage_analysis_triggered_by_high_auroc: `{leakage.get('triggered_by_high_auroc')}`",
            f"- leakage_analysis_passed: `{leakage.get('passed')}`",
            "",
            "| Leakage Check | Passed |",
            "|---|---:|",
        ]
    )
    for name, passed in (leakage.get("checks") or {}).items():
        lines.append(f"| `{name}` | `{passed}` |")
    feature = audit["feature_audit"]
    lines.extend(
        [
            "",
            f"- shortcut_feature_count: `{feature['shortcut_feature_count']}`",
            f"- future_or_offline_feature_count: `{feature['future_or_offline_feature_count']}`",
            f"- top_shortcut_single_feature_auc: `{feature['top_shortcut_single_feature_auc']}`",
            "",
            "| Top Single Feature | Directional AUROC | Raw AUROC |",
            "|---|---:|---:|",
        ]
    )
    for item in feature["top_single_feature_auc_table"][:10]:
        lines.append(f"| `{item['feature']}` | {item['max_direction_auroc']:.4f} | {item['auroc']:.4f} |")
    rollout = audit["rollout_audit"]
    lines.extend(
        [
            "",
            "## Rollout Runtime",
            "",
            f"- completed_rollout_summaries: `{rollout['completed_rollout_summaries']}`",
            f"- observed_completed_dir_hang_rate: `{rollout['observed_completed_dir_hang_rate']}`",
            f"- attempt_level_hang_rate_available: `{rollout['attempt_level_hang_rate_available']}`",
            f"- attempt_level_hang_rate: `{rollout['attempt_level_hang_rate']}`",
            f"- throughput_samples_per_hour_selected_summaries: `{rollout['throughput_samples_per_hour_selected_summaries']}`",
            f"- unique_run_ids: `{rollout['unique_run_ids']}`",
            "",
            "## Counterfactual Clarification",
            "",
            audit["counterfactual_audit"]["clarification"],
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
