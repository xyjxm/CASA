"""Visualization helpers for CASA Phase 5 online five-baseline evidence."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

from gear_sonic.casa.phase5 import MAIN_SKILLS, METHOD_ORDER, relative_reduction
from gear_sonic.casa.phase5_online import method_summary_rows, read_csv_rows
from gear_sonic.casa.phase5_policy import method_display

FIVE_BASELINE_METHODS = tuple(METHOD_ORDER)
SKILL_ORDER = tuple(MAIN_SKILLS)
METRIC_TOLERANCE = 1e-9


@dataclass
class VisualizationInputs:
    artifact_dir: Path
    audit_dir: Path
    methods: list[str]
    casa_method: str
    episode_rows: list[dict[str, Any]]
    decision_rows: list[dict[str, Any]]
    method_summary: dict[str, dict[str, Any]]
    audit: dict[str, Any]
    expected_methods: list[str]
    expected_seeds: list[int]
    sha256_checks: list[dict[str, Any]]


@dataclass
class VisualizationResult:
    output_dir: Path
    selected_episodes: list[dict[str, Any]]
    metric_consistency: dict[str, Any]
    visualization_metrics: dict[str, Any]
    manifest: dict[str, Any]


class MetricConsistencyError(RuntimeError):
    """Raised when --fail-on-metric-mismatch finds an audit mismatch."""


def parse_methods(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        methods = [item.strip() for item in value.split(",") if item.strip()]
    else:
        methods = [str(item).strip() for item in value if str(item).strip()]
    if not methods:
        raise ValueError("methods must not be empty")
    return methods


def load_visualization_inputs(
    *,
    artifact_dir: Path,
    audit_dir: Path,
    methods: list[str],
    casa_method: str,
    strict_five_baseline: bool = False,
) -> VisualizationInputs:
    artifact_dir = artifact_dir.resolve()
    audit_dir = audit_dir.resolve()
    expected_methods = _read_expected_methods(artifact_dir) or list(methods)
    expected_seeds = _read_expected_seeds(artifact_dir)
    if strict_five_baseline and list(methods) != list(FIVE_BASELINE_METHODS):
        raise ValueError(
            f"--strict-five-baseline requires methods {','.join(FIVE_BASELINE_METHODS)}, got {','.join(methods)}"
        )
    if strict_five_baseline and casa_method != "casa_a_per_skill":
        raise ValueError("--strict-five-baseline requires --casa-method casa_a_per_skill")
    episode_rows = dedupe_episode_rows(read_csv_rows(audit_dir / "online_episode_results.csv"))
    decision_rows = dedupe_decision_rows(
        read_csv_rows(audit_dir / "gate_decisions.csv"),
        episode_rows,
    )
    method_summary = _load_method_summary(audit_dir)
    audit = _read_json(audit_dir / "online_acceptance_audit.json", default={})
    sha256_checks = verify_artifact_hashes(artifact_dir=artifact_dir, audit_dir=audit_dir)
    return VisualizationInputs(
        artifact_dir=artifact_dir,
        audit_dir=audit_dir,
        methods=list(methods),
        casa_method=casa_method,
        episode_rows=episode_rows,
        decision_rows=decision_rows,
        method_summary=method_summary,
        audit=audit,
        expected_methods=expected_methods,
        expected_seeds=expected_seeds,
        sha256_checks=sha256_checks,
    )


def verify_artifact_hashes(*, artifact_dir: Path, audit_dir: Path) -> list[dict[str, Any]]:
    sidecars = sorted(
        set(artifact_dir.glob("*.sha256"))
        | set(audit_dir.glob("*.sha256"))
        | ({artifact_dir / "sha256sums.txt"} if (artifact_dir / "sha256sums.txt").exists() else set())
    )
    checks: list[dict[str, Any]] = []
    for sidecar in sidecars:
        for line in sidecar.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2 or len(parts[0]) != 64:
                checks.append(
                    {
                        "sidecar": _rel(sidecar, artifact_dir),
                        "target": "",
                        "expected_sha256": parts[0] if parts else "",
                        "actual_sha256": None,
                        "status": "SKIP",
                        "reason": "unrecognized sha256 line",
                    }
                )
                continue
            expected = parts[0]
            target_text = parts[1]
            target = _resolve_hash_target(
                target_text,
                sidecar=sidecar,
                artifact_dir=artifact_dir,
                audit_dir=audit_dir,
            )
            if target is None:
                checks.append(
                    {
                        "sidecar": _rel(sidecar, artifact_dir),
                        "target": target_text,
                        "expected_sha256": expected,
                        "actual_sha256": None,
                        "status": "FAIL",
                        "reason": "target file not found",
                    }
                )
                continue
            actual = _sha256(target)
            checks.append(
                {
                    "sidecar": _rel(sidecar, artifact_dir),
                    "target": _rel(target, artifact_dir),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "status": "PASS" if actual == expected else "FAIL",
                }
            )
    return checks


def dedupe_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int, int] | tuple[str, int], tuple[tuple[int, int], dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = _episode_key(row)
        if key is None:
            key = ("unkeyed", index)
        rank = (_status_rank(row), index)
        current = selected.get(key)
        if current is None or rank > current[0]:
            selected[key] = (rank, row)
    return sorted((item[1] for item in selected.values()), key=_episode_sort_key)


def dedupe_decision_rows(
    rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_episodes = {
        key
        for row in episode_rows
        if (key := _episode_key(row)) is not None and str(row.get("status")) == "completed"
    }
    selected: dict[tuple[str, int, int, int], tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        episode_key = _episode_key(row)
        if episode_key not in selected_episodes:
            continue
        try:
            key = (
                str(row.get("method", "")),
                int(float(row.get("seed", -1))),
                int(float(row.get("episode_index", -1))),
                int(float(row.get("skill_idx", -1))),
            )
        except (TypeError, ValueError):
            continue
        selected[key] = (index, row)
    return [item[1] for key, item in sorted(selected.items(), key=lambda pair: pair[0])]


def check_metric_consistency(inputs: VisualizationInputs) -> dict[str, Any]:
    recomputed_rows = method_summary_rows(inputs.episode_rows, inputs.decision_rows)
    recomputed = {row["method"]: row for row in recomputed_rows}
    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    def add(name: str, status: str, details: Any = None) -> None:
        checks.append({"name": name, "status": status, "details": details})

    present_methods = set(recomputed)
    missing_methods = [method for method in inputs.methods if method not in present_methods]
    add("all_requested_methods_present", "PASS" if not missing_methods else "FAIL", {"missing": missing_methods})

    if inputs.expected_methods:
        missing_expected = [method for method in inputs.expected_methods if method not in present_methods]
        add(
            "expected_methods_present",
            "PASS" if not missing_expected else "FAIL",
            {"expected_methods": inputs.expected_methods, "missing": missing_expected},
        )
    else:
        add("expected_methods_present", "SKIP", "expected_methods.txt not present")

    if inputs.expected_seeds:
        observed = sorted({_int(row.get("seed")) for row in inputs.episode_rows})
        missing_seeds = [seed for seed in inputs.expected_seeds if seed not in observed]
        add(
            "expected_seeds_present",
            "PASS" if not missing_seeds else "FAIL",
            {"expected_seeds": inputs.expected_seeds, "observed_seeds": observed, "missing": missing_seeds},
        )
    else:
        add("expected_seeds_present", "SKIP", "expected_seeds.txt not present")

    audit_episode_count = inputs.audit.get("episode_count")
    if audit_episode_count is None:
        audit_episode_count = inputs.audit.get("diagnostics", {}).get("deduped_episode_row_count")
    if audit_episode_count is None:
        add("episode_count_matches_audit", "SKIP", "audit episode count unavailable")
    else:
        add(
            "episode_count_matches_audit",
            "PASS" if _int(audit_episode_count) == len(inputs.episode_rows) else "FAIL",
            {"audit": _int(audit_episode_count), "recomputed": len(inputs.episode_rows)},
        )

    audit_decision_count = inputs.audit.get("gate_decision_count")
    if audit_decision_count is None:
        audit_decision_count = inputs.audit.get("diagnostics", {}).get("deduped_decision_row_count")
    if audit_decision_count is None:
        add("gate_decision_count_matches_audit", "SKIP", "audit gate decision count unavailable")
    else:
        add(
            "gate_decision_count_matches_audit",
            "PASS" if _int(audit_decision_count) == len(inputs.decision_rows) else "FAIL",
            {"audit": _int(audit_decision_count), "recomputed": len(inputs.decision_rows)},
        )

    summary_keys = [
        "episode_count",
        "completed_count",
        "task_success_count",
        "task_success_rate",
        "safe_completion_count",
        "safe_completion_rate",
        "unsafe_invocation_count",
        "unsafe_invocation_rate_per_episode",
        "fallback_count",
        "fallback_rate_per_episode",
        "task_progress_success_rate",
    ]
    for method in inputs.methods:
        expected = inputs.method_summary.get(method)
        actual = recomputed.get(method)
        if not expected or not actual:
            continue
        for key in summary_keys:
            if key not in expected or key not in actual or expected.get(key) in {"", None}:
                continue
            if not _numeric_close(expected.get(key), actual.get(key)):
                mismatches.append(
                    {
                        "method": method,
                        "field": key,
                        "audit_value": expected.get(key),
                        "recomputed_value": actual.get(key),
                    }
                )
    add("method_summary_matches_recomputed_metrics", "PASS" if not mismatches else "FAIL", mismatches)

    decision_metrics = compute_decision_metrics(inputs.decision_rows, inputs.methods)
    casa_diag = inputs.audit.get("diagnostics", {}).get("fallback_reject_budget", {})
    casa_decision = decision_metrics.get(inputs.casa_method, {})
    reject_mismatches = []
    for audit_key, metric_key in [
        ("fallback_rate_per_episode", "fallback_rate_per_episode"),
        ("reject_rate_per_decision", "reject_rate_per_decision"),
        ("walk_reject_rate", "walk_reject_rate"),
    ]:
        if audit_key in casa_diag and metric_key in casa_decision:
            if not _numeric_close(casa_diag.get(audit_key), casa_decision.get(metric_key)):
                reject_mismatches.append(
                    {
                        "field": audit_key,
                        "audit_value": casa_diag.get(audit_key),
                        "recomputed_value": casa_decision.get(metric_key),
                    }
                )
    add("casa_fallback_reject_diagnostics_match", "PASS" if not reject_mismatches else "FAIL", reject_mismatches)

    sha_failures = [item for item in inputs.sha256_checks if item.get("status") == "FAIL"]
    add(
        "sha256_artifacts_match",
        "PASS" if not sha_failures else "FAIL",
        {"checked": len(inputs.sha256_checks), "failures": sha_failures},
    )

    status = "FAIL" if any(item["status"] == "FAIL" for item in checks) else "PASS"
    return {
        "status": status,
        "checks": checks,
        "mismatches": mismatches,
        "sha256_checks": inputs.sha256_checks,
        "recomputed_method_summary": recomputed,
        "decision_metrics": decision_metrics,
    }


def build_visualization_metrics(
    inputs: VisualizationInputs,
    consistency: dict[str, Any],
) -> dict[str, Any]:
    method_metrics: dict[str, dict[str, Any]] = {}
    recomputed = consistency.get("recomputed_method_summary", {})
    decision_metrics = consistency.get("decision_metrics", {})
    sonic_unsafe = _float_or_none(recomputed.get("sonic_only", {}).get("unsafe_invocation_count"))
    per_skill_labels = (
        inputs.audit.get("diagnostics", {}).get("per_skill_online_labels", {}).get("by_method_skill", {})
    )
    for method in inputs.methods:
        row = dict(recomputed.get(method) or inputs.method_summary.get(method, {}))
        row.update(decision_metrics.get(method, {}))
        unsafe = _float_or_none(row.get("unsafe_invocation_count"))
        row["unsafe_reduction_vs_sonic"] = (
            relative_reduction(sonic_unsafe, unsafe) if sonic_unsafe is not None and unsafe is not None else None
        )
        method_metrics[method] = row
    return {
        "methods": inputs.methods,
        "skills": list(SKILL_ORDER),
        "method_metrics": method_metrics,
        "decision_metrics": decision_metrics,
        "per_skill_unsafe": compute_per_skill_unsafe(per_skill_labels, inputs.methods),
        "per_skill_decisions": compute_per_skill_decisions(inputs.decision_rows, inputs.methods),
        "baseline_comparisons": inputs.audit.get("diagnostics", {}).get("baseline_comparisons", {}),
        "anti_gaming": {
            "fallback_reject_budget": inputs.audit.get("diagnostics", {}).get("fallback_reject_budget", {}),
            "matched_budget_comparison": inputs.audit.get("diagnostics", {}).get("matched_budget_comparison", {}),
            "strict_plan_a_claim": inputs.audit.get("diagnostics", {}).get("strict_plan_a_claim", {}),
        },
    }


def compute_decision_metrics(
    decision_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        by_method[str(row.get("method", ""))].append(row)
    output: dict[str, dict[str, Any]] = {}
    for method in methods:
        rows = by_method.get(method, [])
        reject_rows = [row for row in rows if row.get("decision") == "reject"]
        fallback_rows = [row for row in rows if _int(row.get("fallback_executed")) == 1]
        episodes = {(_int(row.get("seed")), _int(row.get("episode_index"))) for row in rows}
        walk_rows = [row for row in rows if row.get("candidate_skill") == "walk"]
        walk_reject = [row for row in walk_rows if row.get("decision") == "reject"]
        output[method] = {
            "decision_count": len(rows),
            "reject_count": len(reject_rows),
            "fallback_decision_count": len(fallback_rows),
            "reject_rate_per_decision": len(reject_rows) / len(rows) if rows else None,
            "fallback_rate_per_decision": len(fallback_rows) / len(rows) if rows else None,
            "fallback_rate_per_episode": len(fallback_rows) / len(episodes) if episodes else None,
            "walk_reject_rate": len(walk_reject) / len(walk_rows) if walk_rows else None,
        }
    return output


def compute_per_skill_decisions(
    decision_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        output[method] = {}
        method_rows = [row for row in decision_rows if row.get("method") == method]
        for skill in SKILL_ORDER:
            rows = [row for row in method_rows if row.get("candidate_skill") == skill]
            rejects = [row for row in rows if row.get("decision") == "reject"]
            fallbacks = [row for row in rows if _int(row.get("fallback_executed")) == 1]
            output[method][skill] = {
                "decision_count": len(rows),
                "reject_count": len(rejects),
                "fallback_count": len(fallbacks),
                "reject_rate": len(rejects) / len(rows) if rows else None,
                "fallback_rate": len(fallbacks) / len(rows) if rows else None,
            }
    return output


def compute_per_skill_unsafe(
    audit_skill_labels: dict[str, Any],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for method in methods:
        output[method] = {}
        for skill in SKILL_ORDER:
            stats = audit_skill_labels.get(method, {}).get(skill, {})
            output[method][skill] = {
                "unsafe_count": _int(stats.get("unsafe")),
                "safe_count": _int(stats.get("safe")),
                "total": _int(stats.get("total")),
                "unsafe_rate": (
                    _int(stats.get("unsafe")) / _int(stats.get("total")) if _int(stats.get("total")) > 0 else None
                ),
            }
    return output


def select_representative_episodes(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    *,
    methods: list[str],
    casa_method: str,
    max_examples: int = 12,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in episode_rows:
        grouped[(_int(row.get("seed")), _int(row.get("episode_index")))][str(row.get("method"))] = row
    decisions_by_episode_method: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        decisions_by_episode_method[
            (_int(row.get("seed")), _int(row.get("episode_index")), str(row.get("method")))
        ].append(row)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()

    def add(reason: str, predicate, sort_key=None) -> None:
        if len([row for row in selected if row.get("seed") != ""]) >= max_examples:
            return
        candidates = []
        for key, method_rows in grouped.items():
            if key in selected_keys:
                continue
            if predicate(key, method_rows):
                candidates.append((key, method_rows))
        if sort_key is not None:
            candidates.sort(key=lambda item: sort_key(item[0], item[1]), reverse=True)
        else:
            candidates.sort()
        if candidates:
            key, method_rows = candidates[0]
            selected_keys.add(key)
            selected.append(_selected_episode_row(key, method_rows, methods, reason))
        else:
            selected.append(_missing_selected_row(reason))

    def unsafe(method_rows: dict[str, dict[str, Any]], method: str) -> int:
        return _int(method_rows.get(method, {}).get("unsafe_invocation_count"))

    def fallback(method_rows: dict[str, dict[str, Any]], method: str) -> int:
        return _int(method_rows.get(method, {}).get("fallback_count"))

    def success(method_rows: dict[str, dict[str, Any]], method: str) -> bool:
        return _int(method_rows.get(method, {}).get("task_success")) == 1

    add(
        "sonic_unsafe_casa_safe_successful",
        lambda _key, rows: (
            unsafe(rows, "sonic_only") > 0 and unsafe(rows, casa_method) == 0 and success(rows, casa_method)
        ),
        lambda _key, rows: unsafe(rows, "sonic_only"),
    )
    add(
        "sonic_unsafe_all_safety_methods_safer",
        lambda _key, rows: (
            unsafe(rows, "sonic_only") > 0
            and all(
                unsafe(rows, method) < unsafe(rows, "sonic_only") for method in methods if method != "sonic_only"
            )
        ),
        lambda _key, rows: unsafe(rows, "sonic_only") - unsafe(rows, casa_method),
    )
    add(
        "global_conformal_better_than_casa",
        lambda _key, rows: (
            unsafe(rows, "global_conformal") < unsafe(rows, casa_method)
            or (success(rows, "global_conformal") and not success(rows, casa_method))
        ),
        lambda _key, rows: unsafe(rows, casa_method) - unsafe(rows, "global_conformal"),
    )
    add(
        "casa_fallback_heavy",
        lambda _key, rows: fallback(rows, casa_method) > 0,
        lambda _key, rows: fallback(rows, casa_method),
    )
    add(
        "casa_walk_heavy_reject_or_fallback",
        lambda key, _rows: _walk_reject_or_fallback_count(decisions_by_episode_method, key, casa_method) > 0,
        lambda key, _rows: _walk_reject_or_fallback_count(decisions_by_episode_method, key, casa_method),
    )
    add(
        "raw_critic_failure_conformal_succeeds",
        lambda _key, rows: (
            (unsafe(rows, "raw_critic_0p5") > 0 or not success(rows, "raw_critic_0p5"))
            and (success(rows, "global_conformal") or success(rows, casa_method))
        ),
        lambda _key, rows: unsafe(rows, "raw_critic_0p5"),
    )
    add(
        "hard_contract_over_rejection_or_high_fallback",
        lambda _key, rows: fallback(rows, "hard_contract") > 0 and not success(rows, "hard_contract"),
        lambda _key, rows: fallback(rows, "hard_contract"),
    )
    add(
        "fully_clean_all_methods_success",
        lambda _key, rows: all(unsafe(rows, method) == 0 and success(rows, method) for method in methods),
    )
    add(
        "difficult_all_methods_fail_or_unsafe",
        lambda _key, rows: all(unsafe(rows, method) > 0 or not success(rows, method) for method in methods),
        lambda _key, rows: sum(unsafe(rows, method) for method in methods),
    )
    add(
        "high_risk_margin_or_threshold_crossing",
        lambda key, _rows: _high_risk_count(decisions_by_episode_method, key) > 0,
        lambda key, _rows: _high_risk_count(decisions_by_episode_method, key),
    )

    valid = [row for row in selected if row.get("seed") != ""]
    if len(valid) > max_examples:
        keep_keys = {(row["seed"], row["episode_index"]) for row in valid[:max_examples]}
        selected = [
            row for row in selected if row.get("seed") == "" or (row["seed"], row["episode_index"]) in keep_keys
        ]
    return selected


def write_selected_episodes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "case_label",
        "seed",
        "episode_index",
        "selection_reason",
        "methods_available",
        "sonic_outcome",
        "hard_contract_outcome",
        "raw_critic_outcome",
        "global_conformal_outcome",
        "casa_a_per_skill_outcome",
        "unsafe_counts_by_method",
        "fallback_counts_by_method",
        "task_success_by_method",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def annotate_selected_cases(rows: list[dict[str, Any]]) -> None:
    case_id = 1
    for row in rows:
        if row.get("seed") == "":
            row["case_id"] = ""
            row["case_label"] = ""
            continue
        seed = _int(row.get("seed"))
        episode = _int(row.get("episode_index"))
        row["case_id"] = case_id
        row["case_label"] = f"case{case_id}_seed{seed}_ep{episode:04d}"
        case_id += 1


def build_selected_episodes_manifest(
    *,
    selected_episodes: list[dict[str, Any]],
    figure_manifest: list[dict[str, Any]],
    video_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for selected in selected_episodes:
        if selected.get("seed") == "":
            continue
        seed = _int(selected.get("seed"))
        episode = _int(selected.get("episode_index"))
        cases.append(
            {
                "case_id": selected.get("case_id", ""),
                "case_label": selected.get("case_label", ""),
                "seed": seed,
                "episode_index": episode,
                "selection_reason": selected.get("selection_reason", ""),
                "figures": [
                    item.get("path")
                    for item in figure_manifest
                    if item.get("seed") == seed and item.get("episode_index") == episode
                ],
                "videos": [
                    item.get("path")
                    for item in video_manifest
                    if item.get("seed") == seed and item.get("episode_index") == episode and item.get("path")
                ],
            }
        )
    return {
        "schema_version": 1,
        "case_count": len(cases),
        "cases": cases,
        "missing_requested_cases": [
            {"selection_reason": row.get("selection_reason"), "message": row.get("methods_available")}
            for row in selected_episodes
            if row.get("seed") == ""
        ],
    }


def run_visualization_package(
    *,
    artifact_dir: Path,
    audit_dir: Path,
    output_dir: Path,
    methods: list[str],
    casa_method: str,
    max_example_episodes: int,
    video_mode: str,
    frames_dir: Path | None,
    videos_dir: Path | None,
    fps: int,
    write_html: bool,
    write_markdown: bool,
    fail_on_metric_mismatch: bool,
    strict_five_baseline: bool,
) -> VisualizationResult:
    output_dir = output_dir.resolve()
    figures_dir = output_dir / "figures"
    videos_output_dir = output_dir / "videos"
    data_dir = output_dir / "data"
    for directory in [figures_dir, videos_output_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    inputs = load_visualization_inputs(
        artifact_dir=artifact_dir,
        audit_dir=audit_dir,
        methods=methods,
        casa_method=casa_method,
        strict_five_baseline=strict_five_baseline,
    )
    consistency = check_metric_consistency(inputs)
    _write_json(data_dir / "metric_consistency_check.json", consistency)
    if fail_on_metric_mismatch and consistency["status"] != "PASS":
        raise MetricConsistencyError(
            f"Metric consistency check failed; see {data_dir / 'metric_consistency_check.json'}"
        )

    metrics = build_visualization_metrics(inputs, consistency)
    _write_json(data_dir / "visualization_metrics.json", metrics)
    selected = select_representative_episodes(
        inputs.episode_rows,
        inputs.decision_rows,
        methods=methods,
        casa_method=casa_method,
        max_examples=max_example_episodes,
    )
    annotate_selected_cases(selected)
    write_selected_episodes_csv(data_dir / "selected_episodes.csv", selected)

    figure_manifest = write_all_figures(
        figures_dir=figures_dir,
        metrics=metrics,
        selected_episodes=selected,
        episode_rows=inputs.episode_rows,
        decision_rows=inputs.decision_rows,
        methods=methods,
    )
    video_manifest = write_video_evidence(
        videos_dir=videos_output_dir,
        selected_episodes=selected,
        episode_rows=inputs.episode_rows,
        decision_rows=inputs.decision_rows,
        methods=methods,
        video_mode=video_mode,
        frames_dir=frames_dir,
        source_videos_dir=videos_dir,
        fps=fps,
    )
    selected_manifest = build_selected_episodes_manifest(
        selected_episodes=selected,
        figure_manifest=figure_manifest,
        video_manifest=video_manifest,
    )
    _write_json(data_dir / "selected_episodes_manifest.json", selected_manifest)
    manifest = {
        "artifact_dir": str(inputs.artifact_dir),
        "audit_dir": str(inputs.audit_dir),
        "output_dir": str(output_dir),
        "methods": methods,
        "casa_method": casa_method,
        "video_mode": video_mode,
        "frames_dir": str(frames_dir) if frames_dir else None,
        "videos_dir": str(videos_dir) if videos_dir else None,
        "metric_consistency_status": consistency["status"],
        "selected_episodes_manifest": "data/selected_episodes_manifest.json",
        "figures": figure_manifest,
        "videos": video_manifest,
        "sha256_checks": inputs.sha256_checks,
        "limitations": _limitations(inputs, video_manifest),
    }
    _write_json(data_dir / "visualization_manifest.json", manifest)
    if write_markdown:
        write_markdown_report(output_dir, inputs, consistency, metrics, selected, figure_manifest, video_manifest)
    if write_html:
        write_html_report(output_dir, inputs, consistency, metrics, selected, figure_manifest, video_manifest)
    write_output_readme(output_dir, write_html=write_html)
    return VisualizationResult(
        output_dir=output_dir,
        selected_episodes=selected,
        metric_consistency=consistency,
        visualization_metrics=metrics,
        manifest=manifest,
    )


def write_all_figures(
    *,
    figures_dir: Path,
    metrics: dict[str, Any],
    selected_episodes: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    methods: list[str],
) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    figures.append(_bar_unsafe_counts(figures_dir / "five_baseline_unsafe_counts.png", metrics, methods))
    figures.append(_bar_task_success(figures_dir / "five_baseline_task_success_rates.png", metrics, methods))
    figures.append(
        _bar_metric(
            figures_dir / "five_baseline_fallback_rates.png",
            metrics,
            methods,
            "fallback_rate_per_episode",
            "Fallback rate / episode",
        )
    )
    figures.append(
        _bar_metric(
            figures_dir / "five_baseline_reject_rates.png",
            metrics,
            methods,
            "reject_rate_per_decision",
            "Reject rate / decision",
        )
    )
    figures.append(
        _bar_metric(
            figures_dir / "five_baseline_task_progress_rates.png",
            metrics,
            methods,
            "task_progress_success_rate",
            "Task-progress success rate",
        )
    )
    figures.append(_scatter_unsafe_task(figures_dir / "unsafe_vs_task_success_scatter.png", metrics, methods))
    figures.append(_heatmap_per_skill_unsafe(figures_dir / "per_skill_unsafe_heatmap.png", metrics, methods))
    figures.append(
        _heatmap_per_skill_fallback_reject(figures_dir / "per_skill_fallback_reject_heatmap.png", metrics, methods)
    )
    figures.append(
        _heatmap_episode_outcomes(
            figures_dir / "episode_method_outcome_heatmap.png", selected_episodes, episode_rows, methods
        )
    )
    valid_selected = [row for row in selected_episodes if row.get("seed") != ""]
    for selected in valid_selected:
        seed = _int(selected["seed"])
        episode = _int(selected["episode_index"])
        case_label = _case_label(selected, seed, episode)
        figures.append(
            write_episode_timeline(
                figures_dir / f"selected_episode_timeline_{case_label}.png",
                seed,
                episode,
                episode_rows,
                decision_rows,
                methods,
            )
        )
        figures.append(
            write_gate_risk_timeline(
                figures_dir / f"selected_episode_gate_risk_timeline_{case_label}.png",
                seed,
                episode,
                decision_rows,
                methods,
            )
        )
        figures.append(
            write_gate_decision_heatmap(
                figures_dir / f"selected_episode_gate_decision_heatmap_{case_label}.png",
                seed,
                episode,
                decision_rows,
                methods,
            )
        )
    return figures


def write_video_evidence(
    *,
    videos_dir: Path,
    selected_episodes: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    methods: list[str],
    video_mode: str,
    frames_dir: Path | None,
    source_videos_dir: Path | None,
    fps: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    valid_selected = [row for row in selected_episodes if row.get("seed") != ""]
    for selected in valid_selected:
        seed = _int(selected["seed"])
        episode = _int(selected["episode_index"])
        case_label = _case_label(selected, seed, episode)
        real_matches = []
        if video_mode in {"auto", "real"} and source_videos_dir:
            real_matches = discover_real_videos(source_videos_dir, methods, seed, episode)
        if real_matches:
            side_by_side = videos_dir / f"selected_episode_side_by_side_{case_label}.mp4"
            real_status = _write_real_video_side_by_side(real_matches, side_by_side, fps=fps)
            manifest.append(
                {
                    "case_id": selected.get("case_id", ""),
                    "case_label": case_label,
                    "seed": seed,
                    "episode_index": episode,
                    "type": "real_video_side_by_side",
                    "path": _rel(side_by_side, videos_dir.parent) if real_status.get("created") else None,
                    "matched_inputs": real_matches,
                    "status": real_status,
                }
            )
            if video_mode == "real":
                continue
        if video_mode == "real" and not real_matches:
            manifest.append(
                {
                    "case_id": selected.get("case_id", ""),
                    "case_label": case_label,
                    "seed": seed,
                    "episode_index": episode,
                    "type": "real_video_side_by_side",
                    "path": None,
                    "status": "missing_real_video_inputs",
                    "note": "No simulator recording was found; no fake side-by-side video was generated.",
                }
            )
        metric_path = videos_dir / f"selected_episode_metric_timeline_{case_label}.mp4"
        animation_status = write_metric_timeline_animation(
            metric_path,
            seed,
            episode,
            episode_rows,
            decision_rows,
            methods,
            fps=fps,
        )
        manifest.append(
            {
                "case_id": selected.get("case_id", ""),
                "case_label": case_label,
                "seed": seed,
                "episode_index": episode,
                "type": "metric_timeline_animation",
                "path": _rel(Path(animation_status["path"]), videos_dir.parent),
                "status": animation_status,
                "label": "Metric timeline animation, not simulator recording",
            }
        )
    if not valid_selected:
        manifest.append({"status": "no_valid_selected_episodes", "type": "metric_timeline_animation"})
    if frames_dir:
        manifest.append(
            {
                "type": "frames_dir",
                "path": str(frames_dir),
                "status": "documented_only",
                "note": "Frame overlay generation is not used unless matching real videos are available.",
            }
        )
    return manifest


def discover_real_videos(source_dir: Path, methods: list[str], seed: int, episode: int) -> list[dict[str, Any]]:
    if not source_dir.exists():
        return []
    suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    files = [path for path in source_dir.rglob("*") if path.suffix.lower() in suffixes]
    matches = []
    episode_tokens = {
        f"episode_{episode:04d}",
        f"episode_{episode:03d}",
        f"ep_{episode:03d}",
        f"ep{episode:04d}",
        f"ep{episode:03d}",
        f"episode_{episode}",
        f"ep_{episode}",
    }
    seed_tokens = {f"seed_{seed}", f"seed{seed}"}
    for method in methods:
        method_files = []
        for path in files:
            name = path.name
            if method not in name:
                continue
            if not any(token in name for token in seed_tokens):
                continue
            if not any(token in name for token in episode_tokens):
                continue
            method_files.append(path)
        if method_files:
            matches.append({"method": method, "path": str(sorted(method_files)[0])})
    return matches


def write_episode_timeline(
    path: Path,
    seed: int,
    episode: int,
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, Any]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(methods) * 0.8)))
    skill_colors = {"walk": "#4c78a8", "turn": "#f58518", "gesture": "#54a24b", "passive": "#9d755d"}
    by_decision = _decisions_for_episode(decision_rows, seed, episode)
    by_episode = _episodes_for_key(episode_rows, seed, episode)
    for y, method in enumerate(methods):
        rows = sorted(by_decision.get(method, []), key=lambda row: _int(row.get("skill_idx")))
        if not rows:
            ax.text(0.5, y, "missing decisions", va="center", ha="left", fontsize=8, color="#666666")
            continue
        for row in rows:
            x = _int(row.get("skill_idx"))
            skill = str(row.get("candidate_skill", "unknown"))
            color = skill_colors.get(skill, "#bab0ab")
            marker = "s"
            if row.get("decision") == "reject":
                marker = "X"
            if _int(row.get("fallback_executed")):
                marker = "D"
            ax.scatter([x], [y], marker=marker, s=120, color=color, edgecolor="black", linewidth=0.8)
            label = skill[0].upper() if skill else "?"
            executed = str(row.get("executed_skill", ""))
            if executed and executed != skill:
                label = f"{label}->{executed[:1].upper()}"
            ax.text(x, y + 0.18, label, ha="center", va="bottom", fontsize=7)
        episode_row = by_episode.get(method, {})
        unsafe = _int(episode_row.get("unsafe_invocation_count"))
        fallback = _int(episode_row.get("fallback_count"))
        task_success = _int(episode_row.get("task_success"))
        ax.text(
            max(_int(row.get("skill_idx")) for row in rows) + 0.45,
            y,
            f"unsafe={unsafe} fallback={fallback} success={task_success}",
            va="center",
            fontsize=8,
        )
        if unsafe:
            ax.scatter(
                [max(_int(row.get("skill_idx")) for row in rows) + 0.2],
                [y],
                marker="*",
                s=160,
                color="#e45756",
            )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([method_display(method) for method in methods])
    ax.set_xlabel("Skill step index")
    ax.set_title(
        f"Selected episode timeline: seed={seed}, episode={episode}\n"
        "Episode-level unsafe markers are not step-localized."
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.75, len(methods) - 0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "path": _rel(path, path.parents[1]),
        "type": "episode_timeline",
        "seed": seed,
        "episode_index": episode,
    }


def write_gate_risk_timeline(
    path: Path,
    seed: int,
    episode: int,
    decision_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, Any]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    by_decision = _decisions_for_episode(decision_rows, seed, episode)
    for method in methods:
        rows = sorted(by_decision.get(method, []), key=lambda row: _int(row.get("skill_idx")))
        if not rows:
            continue
        xs = [_int(row.get("skill_idx")) for row in rows]
        risks = [_float_or_none(row.get("raw_critic_risk")) for row in rows]
        ax.plot(xs, risks, marker="o", linewidth=1.5, label=method_display(method))
        thresholds = [_float_or_none(row.get("threshold")) for row in rows]
        if any(value is not None for value in thresholds):
            ax.plot(
                xs,
                [value if value is not None else math.nan for value in thresholds],
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
        for row, x, risk in zip(rows, xs, risks):
            if risk is None:
                continue
            margin = _float_or_none(row.get("risk_margin"))
            if margin is not None and margin > 0:
                ax.scatter([x], [risk], marker="X", s=90, color="#e45756", zorder=5)
    ax.set_xlabel("Skill step index")
    ax.set_ylabel("Raw critic risk / threshold")
    ax.set_title(f"Gate risk timeline: seed={seed}, episode={episode}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "path": _rel(path, path.parents[1]),
        "type": "gate_risk_timeline",
        "seed": seed,
        "episode_index": episode,
    }


def write_gate_decision_heatmap(
    path: Path,
    seed: int,
    episode: int,
    decision_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, Any]:
    plt = _pyplot()
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    values = []
    labels = []
    max_step = 1
    by_decision = _decisions_for_episode(decision_rows, seed, episode)
    for method in methods:
        rows = sorted(by_decision.get(method, []), key=lambda row: _int(row.get("skill_idx")))
        max_step = max(max_step, *[_int(row.get("skill_idx")) for row in rows] or [1])
    for method in methods:
        row_values = [0] * max_step
        row_labels = ["missing"] * max_step
        for row in by_decision.get(method, []):
            idx = _int(row.get("skill_idx")) - 1
            if idx < 0 or idx >= max_step:
                continue
            if _int(row.get("fallback_executed")):
                value, label = 3, "fallback"
            elif row.get("decision") == "reject":
                value, label = 2, "reject"
            elif row.get("decision") == "allow":
                value, label = 1, "allow"
            else:
                value, label = 0, "missing"
            row_values[idx] = value
            row_labels[idx] = label
        values.append(row_values)
        labels.append(row_labels)
    cmap = mcolors.ListedColormap(["#d9d9d9", "#4c78a8", "#f58518", "#e45756"])
    fig, ax = plt.subplots(figsize=(12, max(4, len(methods) * 0.65)))
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    for y, row in enumerate(labels):
        for x, label in enumerate(row):
            ax.text(x, y, label[:3], ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([method_display(method) for method in methods])
    ax.set_xticks(range(max_step))
    ax.set_xticklabels([str(i + 1) for i in range(max_step)])
    ax.set_xlabel("Skill step index")
    ax.set_title(f"Gate decision heatmap: seed={seed}, episode={episode}")
    legend = [
        mpatches.Patch(color="#d9d9d9", label="missing/no decision"),
        mpatches.Patch(color="#4c78a8", label="allow"),
        mpatches.Patch(color="#f58518", label="reject"),
        mpatches.Patch(color="#e45756", label="fallback"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "path": _rel(path, path.parents[1]),
        "type": "gate_decision_heatmap",
        "seed": seed,
        "episode_index": episode,
    }


def write_metric_timeline_animation(
    path: Path,
    seed: int,
    episode: int,
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    methods: list[str],
    *,
    fps: int,
) -> dict[str, Any]:
    plt = _pyplot()
    from matplotlib import animation

    by_decision = _decisions_for_episode(decision_rows, seed, episode)
    by_episode = _episodes_for_key(episode_rows, seed, episode)
    max_step = max(
        [1]
        + [
            _int(row.get("skill_idx"))
            for rows in by_decision.values()
            for row in rows
            if _int(row.get("skill_idx")) > 0
        ]
    )
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(methods) * 0.8)))

    def draw(frame: int) -> list[Any]:
        ax.clear()
        ax.set_title(
            "Metric timeline animation, not simulator recording\n"
            f"seed={seed}, episode={episode}, step={frame}/{max_step}"
        )
        ax.set_xlim(0.5, max_step + 1.2)
        ax.set_ylim(-0.75, len(methods) - 0.25)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([method_display(method) for method in methods])
        ax.set_xlabel("Skill step index")
        ax.grid(axis="x", alpha=0.25)
        for y, method in enumerate(methods):
            rows = sorted(by_decision.get(method, []), key=lambda row: _int(row.get("skill_idx")))
            for row in rows:
                step = _int(row.get("skill_idx"))
                if step > frame:
                    continue
                color = "#4c78a8"
                marker = "o"
                if row.get("decision") == "reject":
                    color = "#f58518"
                    marker = "X"
                if _int(row.get("fallback_executed")):
                    color = "#e45756"
                    marker = "D"
                ax.scatter([step], [y], marker=marker, s=120, color=color, edgecolor="black")
                skill = str(row.get("candidate_skill", ""))
                ax.text(step, y + 0.17, skill[:4], fontsize=7, ha="center")
            episode_row = by_episode.get(method, {})
            if frame >= max_step and _int(episode_row.get("unsafe_invocation_count")):
                ax.scatter([max_step + 0.45], [y], marker="*", s=160, color="#e45756")
        return []

    anim = animation.FuncAnimation(
        fig, draw, frames=range(1, max_step + 1), interval=1000 / max(1, fps), blit=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if animation.writers.is_available("ffmpeg"):
        anim.save(path, writer=animation.FFMpegWriter(fps=max(1, fps)))
        plt.close(fig)
        return {"path": str(path), "format": "mp4", "created": True, "writer": "ffmpeg"}
    frames_dir = path.with_suffix("").with_name(path.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(1, max_step + 1):
        draw(frame)
        fig.savefig(frames_dir / f"frame_{frame:03d}.png", dpi=120)
    plt.close(fig)
    return {"path": str(frames_dir), "format": "png_frame_sequence", "created": True, "writer": "fallback_frames"}


def write_markdown_report(
    output_dir: Path,
    inputs: VisualizationInputs,
    consistency: dict[str, Any],
    metrics: dict[str, Any],
    selected_episodes: list[dict[str, Any]],
    figure_manifest: list[dict[str, Any]],
    video_manifest: list[dict[str, Any]],
) -> None:
    path = output_dir / "phase5_online_visualization_report.md"
    seed_text = (
        ", ".join(str(seed) for seed in inputs.expected_seeds)
        if inputs.expected_seeds
        else "not specified"
    )
    sha_pass = sum(1 for item in inputs.sha256_checks if item.get("status") == "PASS")
    sha_fail = sum(1 for item in inputs.sha256_checks if item.get("status") == "FAIL")
    lines = [
        "# Phase 5 Online Five-Baseline Visualization Evidence",
        "",
        "## Inputs",
        "",
        f"- artifact directory: `{inputs.artifact_dir}`",
        f"- audit directory: `{inputs.audit_dir}`",
        f"- methods: `{', '.join(inputs.methods)}`",
        f"- seeds: `{seed_text}`",
        "- source files: `online_episode_results.csv`, `gate_decisions.csv`, "
        "`method_summary.json`, `online_acceptance_audit.json`",
        f"- artifact hashes verified: `{sha_pass}` pass / `{sha_fail}` fail",
        "",
        "## Metric Consistency Check",
        "",
        f"- status: `{consistency['status']}`",
    ]
    for check in consistency["checks"]:
        lines.append(f"- {check['name']}: `{check['status']}`")
    lines.extend(
        [
            "",
            "## Metric provenance",
            "",
            "- Unsafe invocation counts are the sum of "
            "`online_episode_results.csv:unsafe_invocation_count` per method.",
            "- `task_success_rate` and `safe_completion_rate` are read from "
            "`method_summary.json` and recomputed from "
            "`online_episode_results.csv:task_success`; the legacy task-success "
            "field is preserved for audit compatibility.",
            "- `task_progress_success_rate` is recomputed from "
            "`gate_decisions.csv:task_progress_executed` when present. Missing "
            "task-progress fields are reported rather than fabricated.",
            "- Fallback counts come from `online_episode_results.csv:fallback_count`; "
            "fallback/reject rates per decision come from "
            "`gate_decisions.csv:fallback_executed` and `gate_decisions.csv:decision`.",
            "- Risk timelines use `gate_decisions.csv:raw_critic_risk`, `threshold`, and `risk_margin`.",
            "- Per-skill unsafe heatmaps use "
            "`online_acceptance_audit.json:diagnostics.per_skill_online_labels` "
            "when available; otherwise the report treats per-step unsafe localization "
            "as unavailable.",
            "- Plotted method summary values are checked against `method_summary.json` "
            "and `online_acceptance_audit.json` before figures are written.",
            "",
            "## Five-Baseline Summary",
            "",
            "| method | unsafe | unsafe reduction vs SONIC | safe completion | "
            "task progress | fallback/episode | reject/decision |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in inputs.methods:
        row = metrics["method_metrics"].get(method, {})
        lines.append(
            f"| {method_display(method)} | {_fmt(row.get('unsafe_invocation_count'))} | "
            f"{_fmt(row.get('unsafe_reduction_vs_sonic'))} | {_fmt(row.get('safe_completion_rate'))} | "
            f"{_fmt(row.get('task_progress_success_rate'))} | {_fmt(row.get('fallback_rate_per_episode'))} | "
            f"{_fmt(row.get('reject_rate_per_decision'))} |"
        )
    lines.extend(["", "### Charts", ""])
    for figure in figure_manifest:
        if figure.get("type") in {"episode_timeline", "gate_risk_timeline", "gate_decision_heatmap"}:
            continue
        lines.append(f"![{figure.get('type', 'figure')}]({figure['path']})")
    lines.extend(["", "## Unsafe / Fallback Heatmaps", ""])
    for name in [
        "figures/per_skill_unsafe_heatmap.png",
        "figures/per_skill_fallback_reject_heatmap.png",
        "figures/episode_method_outcome_heatmap.png",
    ]:
        lines.append(f"![{Path(name).stem}]({name})")
    lines.extend(["", "## Representative Episode Timelines", ""])
    for selected in selected_episodes:
        if selected.get("seed") == "":
            lines.append(
                f"- missing example for `{selected['selection_reason']}`: {selected.get('methods_available', '')}"
            )
            continue
        seed = selected["seed"]
        episode = selected["episode_index"]
        case_label = _case_label(selected, _int(seed), _int(episode))
        case_title = f"{case_label}: seed={seed}, episode={episode}"
        lines.append(f"### {case_title}: `{selected['selection_reason']}`")
        lines.append(f"![timeline](figures/selected_episode_timeline_{case_label}.png)")
        lines.append(f"![risk timeline](figures/selected_episode_gate_risk_timeline_{case_label}.png)")
    lines.extend(["", "## Video / Animation Evidence", ""])
    for item in video_manifest:
        if item.get("type") == "metric_timeline_animation":
            lines.append(f"- `{item.get('path')}`: Metric timeline animation, not simulator recording.")
        elif item.get("type") == "real_video_side_by_side":
            lines.append(f"- real video side-by-side: `{item.get('path')}` status `{item.get('status')}`")
        else:
            lines.append(f"- {item}")
    strict_claim = metrics.get("anti_gaming", {}).get("strict_plan_a_claim", {})
    fallback_budget = metrics.get("anti_gaming", {}).get("fallback_reject_budget", {})
    matched = metrics.get("anti_gaming", {}).get("matched_budget_comparison", {})
    lines.extend(
        [
            "",
            "## Anti-Gaming / Claim-Validity Visualization",
            "",
            f"- strict Plan A claim status: `{strict_claim.get('status', 'n/a')}`",
            f"- strict Plan A claim go: `{strict_claim.get('go', 'n/a')}`",
            "- global_conformal vs casa_a_per_skill winner in matched "
            f"observed-budget diagnostic: `{matched.get('winner', 'n/a')}`",
            f"- CASA fallback rate per episode: `{_fmt(fallback_budget.get('fallback_rate_per_episode'))}`",
            f"- CASA reject rate per decision: `{_fmt(fallback_budget.get('reject_rate_per_decision'))}`",
            f"- CASA walk reject rate: `{_fmt(fallback_budget.get('walk_reject_rate'))}`",
            "- This visualization package does not override or weaken the strict claim audit.",
            "",
            "## Limitations",
            "",
            "- Raw simulator videos are not fabricated. If no real recordings are "
            "found, the generated videos are metric timeline animations only.",
            "- Episode unsafe counts are episode-level unless rollout summaries "
            "provide per-skill labels; per-step unsafe localization is not implied.",
            "- Task-progress values depend on `task_progress_executed` fields in "
            "gate decisions; missing fields are reported as unavailable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_html_report(
    output_dir: Path,
    inputs: VisualizationInputs,
    consistency: dict[str, Any],
    metrics: dict[str, Any],
    selected_episodes: list[dict[str, Any]],
    figure_manifest: list[dict[str, Any]],
    video_manifest: list[dict[str, Any]],
) -> None:
    path = output_dir / "phase5_online_visualization_report.html"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Phase 5 Online Visualization Evidence</title>",
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.45}"
        "img{max-width:100%;border:1px solid #ddd}"
        "table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:4px 8px}"
        "</style>",
        "</head><body>",
        "<h1>Phase 5 Online Five-Baseline Visualization Evidence</h1>",
        f"<p><strong>Metric consistency:</strong> {html.escape(consistency['status'])}</p>",
        f"<p><strong>Artifact dir:</strong> {html.escape(str(inputs.artifact_dir))}</p>",
        "<h2>Metric provenance</h2>",
        "<p>Unsafe, task-success, fallback, reject, risk, and task-progress values "
        "are recomputed from the committed CSV/JSON audit artifacts before plotting. "
        "Metric timeline animations are not simulator recordings.</p>",
        "<h2>Five-Baseline Figures</h2>",
    ]
    for figure in figure_manifest:
        if figure.get("type") in {"episode_timeline", "gate_risk_timeline", "gate_decision_heatmap"}:
            continue
        figure_type = html.escape(str(figure.get("type")))
        figure_path = html.escape(figure["path"])
        parts.append(f"<h3>{figure_type}</h3><img src='{figure_path}'>")
    parts.append("<h2>Representative Episode Timelines</h2>")
    for selected in selected_episodes:
        if selected.get("seed") == "":
            reason = html.escape(str(selected.get("selection_reason")))
            available = html.escape(str(selected.get("methods_available")))
            parts.append(
                f"<p>Missing example for {reason}: {available}</p>"
            )
            continue
        seed = selected["seed"]
        episode = selected["episode_index"]
        reason = html.escape(str(selected["selection_reason"]))
        case_label = _case_label(selected, _int(seed), _int(episode))
        parts.append(f"<h3>{html.escape(case_label)}: seed={seed}, episode={episode}: {reason}</h3>")
        parts.append(f"<img src='figures/selected_episode_timeline_{case_label}.png'>")
        parts.append(f"<img src='figures/selected_episode_gate_risk_timeline_{case_label}.png'>")
    parts.append("<h2>Video / Animation Evidence</h2><ul>")
    for item in video_manifest:
        if item.get("path"):
            item_path = html.escape(str(item["path"]))
            item_type = html.escape(str(item.get("type")))
            item_label = html.escape(str(item.get("label", item.get("status"))))
            parts.append(
                f"<li><a href='{item_path}'>{item_type}</a>: {item_label}</li>"
            )
        else:
            parts.append(f"<li>{html.escape(str(item))}</li>")
    parts.extend(
        [
            "</ul>",
            "<h2>Anti-Gaming / Claim-Validity Visualization</h2>",
            "<p>This package visualizes global conformal vs CASA-A, "
            "fallback/reject budgets, task-progress vs safe completion, and "
            "walk reject/fallback rates. It does not override the strict claim audit.</p>",
            "</body></html>",
        ]
    )
    path.write_text("\n".join(parts) + "\n")


def write_output_readme(output_dir: Path, *, write_html: bool) -> None:
    lines = [
        "# Phase 5 Online Visualization Outputs",
        "",
        "- `phase5_online_visualization_report.md`: markdown visualization report.",
        "- `figures/`: five-baseline charts, heatmaps, and case-numbered selected episode timelines.",
        "- `videos/`: case-numbered metric timeline animations or real-video side-by-side outputs when real inputs exist.",
        "- `data/`: selected episodes, case manifest, visualization metrics, consistency checks, and manifest.",
    ]
    if write_html:
        lines.append("- `phase5_online_visualization_report.html`: static dashboard using relative links.")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def _bar_unsafe_counts(path: Path, metrics: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    rows = metrics["method_metrics"]
    values = [_float_or_zero(rows.get(method, {}).get("unsafe_invocation_count")) for method in methods]
    annotations = []
    for method, value in zip(methods, values):
        reduction = rows.get(method, {}).get("unsafe_reduction_vs_sonic")
        text = f"{int(value)}"
        if method != "sonic_only" and reduction is not None:
            text += f"\nred={float(reduction):.2f}"
        annotations.append(text)
    return _simple_bar(
        path, methods, values, "Unsafe invocation count", annotations, "five_baseline_unsafe_counts"
    )


def _bar_task_success(path: Path, metrics: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    plt = _pyplot()
    rows = metrics["method_metrics"]
    safe = [_float_or_none(rows.get(method, {}).get("safe_completion_rate")) for method in methods]
    progress = [_float_or_none(rows.get(method, {}).get("task_progress_success_rate")) for method in methods]
    x = list(range(len(methods)))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.38
    ax.bar([idx - width / 2 for idx in x], [_nan_to_zero(v) for v in safe], width, label="safe completion")
    ax.bar([idx + width / 2 for idx in x], [_nan_to_zero(v) for v in progress], width, label="task progress")
    for idx, value in enumerate(safe):
        ax.text(idx - width / 2, _nan_to_zero(value), _fmt(value), ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(progress):
        ax.text(idx + width / 2, _nan_to_zero(value), _fmt(value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_method(method) for method in methods], rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title("Task success / safe completion vs task-progress success")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": _rel(path, path.parents[1]), "type": "five_baseline_task_success_rates"}


def _bar_metric(path: Path, metrics: dict[str, Any], methods: list[str], key: str, title: str) -> dict[str, Any]:
    rows = metrics["method_metrics"]
    values = [_float_or_none(rows.get(method, {}).get(key)) for method in methods]
    return _simple_bar(
        path,
        methods,
        [_nan_to_zero(value) for value in values],
        title,
        [_fmt(value) for value in values],
        path.stem,
    )


def _simple_bar(
    path: Path,
    methods: list[str],
    values: list[float],
    title: str,
    annotations: list[str],
    figure_type: str,
) -> dict[str, Any]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(methods)), values, color=["#4c78a8", "#f58518", "#72b7b2", "#54a24b", "#e45756"][: len(methods)]
    )
    for bar, annotation in zip(bars, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), annotation, ha="center", va="bottom", fontsize=8
        )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_short_method(method) for method in methods], rotation=25, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": _rel(path, path.parents[1]), "type": figure_type}


def _scatter_unsafe_task(path: Path, metrics: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    plt = _pyplot()
    rows = metrics["method_metrics"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        row = rows.get(method, {})
        x = _float_or_zero(row.get("safe_completion_rate"))
        y = _float_or_zero(row.get("unsafe_invocation_count"))
        ax.scatter([x], [y], s=120)
        ax.text(x, y, _short_method(method), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Safe completion rate")
    ax.set_ylabel("Unsafe invocation count")
    ax.set_title("Unsafe count vs safe completion")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": _rel(path, path.parents[1]), "type": "unsafe_vs_task_success_scatter"}


def _heatmap_per_skill_unsafe(path: Path, metrics: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    values = [
        [
            metrics["per_skill_unsafe"].get(method, {}).get(skill, {}).get("unsafe_count", 0)
            for skill in SKILL_ORDER
        ]
        for method in methods
    ]
    return _heatmap(path, values, methods, list(SKILL_ORDER), "Per-skill unsafe count", "per_skill_unsafe_heatmap")


def _heatmap_per_skill_fallback_reject(path: Path, metrics: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(methods) * 0.6)))
    for ax, key, title in [
        (axes[0], "fallback_rate", "Fallback rate"),
        (axes[1], "reject_rate", "Reject rate"),
    ]:
        values = [
            [
                _nan_to_zero(metrics["per_skill_decisions"].get(method, {}).get(skill, {}).get(key))
                for skill in SKILL_ORDER
            ]
            for method in methods
        ]
        image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(SKILL_ORDER)))
        ax.set_xticklabels(SKILL_ORDER, rotation=25, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([_short_method(method) for method in methods])
        ax.set_title(title)
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-skill fallback/reject rates")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": _rel(path, path.parents[1]), "type": "per_skill_fallback_reject_heatmap"}


def _heatmap_episode_outcomes(
    path: Path,
    selected_episodes: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, Any]:
    plt = _pyplot()
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    valid = [row for row in selected_episodes if row.get("seed") != ""]
    by_episode = defaultdict(dict)
    for row in episode_rows:
        by_episode[(_int(row.get("seed")), _int(row.get("episode_index")))][str(row.get("method"))] = row
    if not valid:
        values = [[0 for _ in methods]]
        ylabels = ["no selection"]
    else:
        values = []
        ylabels = []
        for row in valid:
            seed = _int(row["seed"])
            episode = _int(row["episode_index"])
            ylabels.append(f"{seed}/{episode}")
            method_rows = by_episode.get((seed, episode), {})
            values.append([_outcome_category(method_rows.get(method, {})) for method in methods])
    cmap = mcolors.ListedColormap(["#d9d9d9", "#4c78a8", "#e45756", "#f58518", "#7f7f7f"])
    fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.55)))
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_short_method(method) for method in methods], rotation=25, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Selected episode outcome heatmap")
    legend = [
        mpatches.Patch(color="#d9d9d9", label="missing/no decision"),
        mpatches.Patch(color="#4c78a8", label="safe success"),
        mpatches.Patch(color="#e45756", label="unsafe"),
        mpatches.Patch(color="#f58518", label="fallback-heavy"),
        mpatches.Patch(color="#7f7f7f", label="incomplete/failure"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": _rel(path, path.parents[1]), "type": "episode_method_outcome_heatmap"}


def _heatmap(
    path: Path,
    values: list[list[float]],
    methods: list[str],
    columns: list[str],
    title: str,
    figure_type: str,
) -> dict[str, Any]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.6)))
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_short_method(method) for method in methods])
    ax.set_title(title)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"path": _rel(path, path.parents[1]), "type": figure_type}


def _write_real_video_side_by_side(
    matches: list[dict[str, Any]], output_path: Path, *, fps: int
) -> dict[str, Any]:
    if not matches:
        return {"created": False, "reason": "no_matches"}
    if len(matches) == 1:
        shutil.copyfile(matches[0]["path"], output_path)
        return {"created": True, "reason": "single_real_video_copied"}
    if shutil.which("ffmpeg") is None:
        return {"created": False, "reason": "ffmpeg_unavailable_for_side_by_side"}
    inputs = []
    filter_parts = []
    for idx, match in enumerate(matches):
        inputs.extend(["-i", match["path"]])
        filter_parts.append(f"[{idx}:v]scale=640:-1,fps={max(1, fps)}[v{idx}]")
    stacked_inputs = "".join(f"[v{idx}]" for idx in range(len(matches)))
    filter_complex = ";".join(filter_parts) + f";{stacked_inputs}hstack=inputs={len(matches)}[out]"
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-an",
        str(output_path),
    ]
    completed = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        return {"created": False, "reason": "ffmpeg_failed", "stderr": completed.stderr[-1000:]}
    return {"created": True, "reason": "ffmpeg_hstack"}


def _case_label(selected: dict[str, Any], seed: int, episode: int) -> str:
    label = str(selected.get("case_label") or "").strip()
    if label:
        return label
    case_id = selected.get("case_id")
    if case_id not in {"", None}:
        return f"case{_int(case_id)}_seed{seed}_ep{episode:04d}"
    return f"seed{seed}_ep{episode:04d}"


def _selected_episode_row(
    key: tuple[int, int],
    method_rows: dict[str, dict[str, Any]],
    methods: list[str],
    reason: str,
) -> dict[str, Any]:
    seed, episode = key
    unsafe = {method: _int(method_rows.get(method, {}).get("unsafe_invocation_count")) for method in methods}
    fallback = {method: _int(method_rows.get(method, {}).get("fallback_count")) for method in methods}
    task_success = {method: _int(method_rows.get(method, {}).get("task_success")) for method in methods}
    return {
        "seed": seed,
        "episode_index": episode,
        "selection_reason": reason,
        "methods_available": ",".join(method for method in methods if method in method_rows),
        "sonic_outcome": _outcome_text(method_rows.get("sonic_only", {})),
        "hard_contract_outcome": _outcome_text(method_rows.get("hard_contract", {})),
        "raw_critic_outcome": _outcome_text(method_rows.get("raw_critic_0p5", {})),
        "global_conformal_outcome": _outcome_text(method_rows.get("global_conformal", {})),
        "casa_a_per_skill_outcome": _outcome_text(method_rows.get("casa_a_per_skill", {})),
        "unsafe_counts_by_method": json.dumps(unsafe, sort_keys=True),
        "fallback_counts_by_method": json.dumps(fallback, sort_keys=True),
        "task_success_by_method": json.dumps(task_success, sort_keys=True),
    }


def _missing_selected_row(reason: str) -> dict[str, Any]:
    return {
        "seed": "",
        "episode_index": "",
        "selection_reason": reason,
        "methods_available": "missing example in current artifacts",
        "sonic_outcome": "",
        "hard_contract_outcome": "",
        "raw_critic_outcome": "",
        "global_conformal_outcome": "",
        "casa_a_per_skill_outcome": "",
        "unsafe_counts_by_method": "{}",
        "fallback_counts_by_method": "{}",
        "task_success_by_method": "{}",
    }


def _limitations(inputs: VisualizationInputs, video_manifest: list[dict[str, Any]]) -> list[str]:
    limitations = []
    if not any(item.get("type") == "real_video_side_by_side" and item.get("path") for item in video_manifest):
        limitations.append(
            "No real simulator video side-by-side was generated; "
            "metric timeline animations are not simulator recordings."
        )
    per_skill = inputs.audit.get("diagnostics", {}).get("per_skill_online_labels", {})
    if not per_skill:
        limitations.append(
            "Per-skill unsafe labels were unavailable; unsafe heatmap cannot localize unsafe events by step."
        )
    return limitations


def _resolve_hash_target(
    target_text: str,
    *,
    sidecar: Path,
    artifact_dir: Path,
    audit_dir: Path,
) -> Path | None:
    raw = Path(target_text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                artifact_dir / raw,
                audit_dir / raw,
                sidecar.parent / raw,
                sidecar.parent / raw.name,
                artifact_dir / raw.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_method_summary(audit_dir: Path) -> dict[str, dict[str, Any]]:
    json_path = audit_dir / "method_summary.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        if isinstance(data, dict) and isinstance(data.get("methods"), dict):
            return {str(method): dict(row) for method, row in data["methods"].items()}
    csv_path = audit_dir / "method_summary.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as file:
            return {str(row.get("method")): dict(row) for row in csv.DictReader(file)}
    return {}


def _read_expected_methods(artifact_dir: Path) -> list[str]:
    path = artifact_dir / "expected_methods.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_expected_seeds(artifact_dir: Path) -> list[int]:
    path = artifact_dir / "expected_seeds.txt"
    if not path.exists():
        return []
    return [_int(line) for line in path.read_text().splitlines() if line.strip()]


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _episode_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    try:
        return (str(row.get("method", "")), int(float(row.get("seed"))), int(float(row.get("episode_index"))))
    except (TypeError, ValueError):
        return None


def _episode_sort_key(row: dict[str, Any]) -> tuple[str, int, int]:
    key = _episode_key(row)
    return key if key is not None else (str(row.get("method", "")), -1, -1)


def _status_rank(row: dict[str, Any]) -> int:
    status = str(row.get("status", ""))
    if status == "completed":
        return 3
    if status == "unverified":
        return 2
    if status == "failed":
        return 1
    return 0


def _numeric_close(left: Any, right: Any, tolerance: float = METRIC_TOLERANCE) -> bool:
    left_f = _float_or_none(left)
    right_f = _float_or_none(right)
    if left_f is None or right_f is None:
        return str(left) == str(right)
    return abs(left_f - right_f) <= tolerance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _float_or_zero(value: Any) -> float:
    output = _float_or_none(value)
    return output if output is not None else 0.0


def _nan_to_zero(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) else 0.0


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    if abs(number - round(number)) < 1e-9 and abs(number) >= 1:
        return str(int(round(number)))
    return f"{number:.4f}"


def _short_method(method: str) -> str:
    return {
        "sonic_only": "sonic",
        "hard_contract": "hard",
        "raw_critic_0p5": "raw",
        "global_conformal": "global",
        "casa_a_per_skill": "casa",
    }.get(method, method)


def _decisions_for_episode(
    decision_rows: list[dict[str, Any]],
    seed: int,
    episode: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        if _int(row.get("seed")) == seed and _int(row.get("episode_index")) == episode:
            output[str(row.get("method"))].append(row)
    return output


def _episodes_for_key(
    episode_rows: list[dict[str, Any]],
    seed: int,
    episode: int,
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("method")): row
        for row in episode_rows
        if _int(row.get("seed")) == seed and _int(row.get("episode_index")) == episode
    }


def _outcome_text(row: dict[str, Any]) -> str:
    if not row:
        return "missing"
    return (
        f"status={row.get('status')};unsafe={_int(row.get('unsafe_invocation_count'))};"
        f"fallback={_int(row.get('fallback_count'))};task_success={_int(row.get('task_success'))}"
    )


def _outcome_category(row: dict[str, Any]) -> int:
    if not row:
        return 0
    if str(row.get("status")) != "completed":
        return 4
    if _int(row.get("unsafe_invocation_count")) > 0:
        return 2
    if _int(row.get("fallback_count")) >= 2:
        return 3
    if _int(row.get("task_success")) == 1:
        return 1
    return 4


def _walk_reject_or_fallback_count(
    decisions_by_episode_method: dict[tuple[int, int, str], list[dict[str, Any]]],
    key: tuple[int, int],
    method: str,
) -> int:
    rows = decisions_by_episode_method.get((key[0], key[1], method), [])
    return sum(
        1
        for row in rows
        if row.get("candidate_skill") == "walk"
        and (row.get("decision") == "reject" or _int(row.get("fallback_executed")) == 1)
    )


def _high_risk_count(
    decisions_by_episode_method: dict[tuple[int, int, str], list[dict[str, Any]]],
    key: tuple[int, int],
) -> int:
    rows = [
        row
        for (seed, episode, _method), method_rows in decisions_by_episode_method.items()
        if seed == key[0] and episode == key[1]
        for row in method_rows
    ]
    count = 0
    for row in rows:
        margin = _float_or_none(row.get("risk_margin"))
        threshold = _float_or_none(row.get("threshold"))
        risk = _float_or_none(row.get("raw_critic_risk"))
        if margin is not None and margin > 0:
            count += 1
        elif risk is not None and threshold is not None and risk >= threshold:
            count += 1
    return count
