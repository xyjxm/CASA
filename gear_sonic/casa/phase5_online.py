"""Shared CASA Phase 5 online audit and validation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

from gear_sonic.casa.phase5 import MAIN_SKILLS, METHOD_DISPLAY, METHOD_ORDER, relative_reduction

REQUIRED_EPISODE_FIELDS = (
    "method",
    "seed",
    "episode_index",
    "episode_id",
    "status",
    "task_success",
    "fallback_count",
    "unsafe_invocation_count",
    "violation_count",
    "completion_time_s",
    "episode_dir",
    "initial_upright_ok",
)
REQUIRED_DECISION_FIELDS = (
    "method",
    "candidate_skill",
    "raw_critic_risk",
    "hard_contract_score",
    "hard_contract_fixed_reject",
    "decision",
    "episode_id",
    "episode_index",
    "seed",
    "skill_idx",
    "executed_skill",
    "fallback_executed",
    "result_status",
)
STATUS_VALUES = {"completed", "failed", "unverified", "initial_upright_failed"}
DECISION_VALUES = {"allow", "reject"}
RESULT_STATUS_VALUES = {"success", "failed", "unverified", ""}


class OnlineValidationError(ValueError):
    """Raised when Phase 5 online artifacts are malformed."""


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise OnlineValidationError(f"Missing required CSV artifact: {path}")
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({str(row.get("method", "")) for row in rows if row.get("method")})
    output = []
    for method in methods:
        method_rows = [row for row in rows if row.get("method") == method]
        count = len(method_rows)
        if count == 0:
            continue
        unsafe = sum(_int(row.get("unsafe_invocation_count")) for row in method_rows)
        fallback = sum(_int(row.get("fallback_count")) for row in method_rows)
        completed = sum(1 for row in method_rows if row.get("status") == "completed")
        task_success = sum(_int(row.get("task_success")) for row in method_rows)
        output.append(
            {
                "method": method,
                "method_display": method_rows[0].get("method_display", METHOD_DISPLAY.get(method, method)),
                "episode_count": count,
                "completed_count": completed,
                "failed_or_unverified": count - completed,
                "task_success_count": task_success,
                "task_success_rate": task_success / count,
                "unsafe_invocation_count": unsafe,
                "unsafe_invocation_rate_per_episode": unsafe / count,
                "fallback_count": fallback,
                "fallback_rate_per_episode": fallback / count,
                "mean_completion_time_s": sum(_float(row.get("completion_time_s")) for row in method_rows)
                / count,
            }
        )
    return output


def build_online_audit(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    *,
    expected_episodes: int,
    expected_methods: list[str] | None = None,
    expected_seeds: list[int] | None = None,
    episodes_per_seed: int | None = None,
    skills_per_episode: int = 8,
    lane_summaries: list[dict[str, Any]] | None = None,
    raw_episode_row_count: int | None = None,
    raw_decision_row_count: int | None = None,
    casa_method: str = "casa_a_per_skill",
    sonic_method: str = "sonic_only",
    hard_method: str = "hard_contract",
    min_sonic_unsafe_reduction: float = 0.40,
    min_hard_unsafe_reduction: float = 0.20,
    max_task_success_drop_abs: float = 0.10,
    max_task_success_drop_rel: float = 0.30,
) -> dict[str, Any]:
    expected_methods = expected_methods or list(METHOD_ORDER)
    lane_summaries = lane_summaries or []
    validation = validate_online_artifacts(
        episode_rows,
        decision_rows,
        expected_methods=expected_methods,
        expected_seeds=expected_seeds,
        episodes_per_seed=episodes_per_seed,
        skills_per_episode=skills_per_episode,
    )
    method_rows = method_summary_rows(episode_rows)
    method_summary = {row["method"]: row for row in method_rows}
    decision_summary = decision_summary_by_method_and_skill(decision_rows)
    skill_label_summary = skill_label_summary_from_rollouts(episode_rows)
    confidence = confidence_intervals(method_summary)
    comparisons = baseline_comparisons(
        method_summary,
        casa_method=casa_method,
        sonic_method=sonic_method,
        hard_method=hard_method,
    )
    checks = online_checks(
        episode_rows=episode_rows,
        decision_rows=decision_rows,
        validation=validation,
        method_summary=method_summary,
        comparisons=comparisons,
        expected_episodes=expected_episodes,
        expected_methods=expected_methods,
        min_sonic_unsafe_reduction=min_sonic_unsafe_reduction,
        min_hard_unsafe_reduction=min_hard_unsafe_reduction,
        max_task_success_drop_abs=max_task_success_drop_abs,
        max_task_success_drop_rel=max_task_success_drop_rel,
    )
    warnings = online_warnings(
        method_summary,
        comparisons,
        validation,
        raw_episode_row_count,
        raw_decision_row_count,
    )
    diagnostics = {
        "artifact_validation": validation,
        "baseline_comparisons": comparisons,
        "confidence_intervals": confidence,
        "per_skill_decisions": decision_summary,
        "per_skill_online_labels": skill_label_summary,
        "lane_summary_count": len(lane_summaries),
        "raw_episode_row_count": raw_episode_row_count,
        "raw_decision_row_count": raw_decision_row_count,
        "deduped_episode_row_count": len(episode_rows),
        "deduped_decision_row_count": len(decision_rows),
        "criteria": {
            "expected_episodes": expected_episodes,
            "expected_methods": expected_methods,
            "expected_seeds": expected_seeds,
            "episodes_per_seed": episodes_per_seed,
            "skills_per_episode": skills_per_episode,
            "min_sonic_unsafe_reduction": min_sonic_unsafe_reduction,
            "min_hard_unsafe_reduction": min_hard_unsafe_reduction,
            "max_task_success_drop_abs": max_task_success_drop_abs,
            "max_task_success_drop_rel": max_task_success_drop_rel,
        },
    }
    blocking = [name for name, passed in checks.items() if not passed]
    actionable = [
        actionable_blocker(name, method_summary, comparisons, validation) for name in blocking
    ]
    return {
        "phase": "CASA Phase5 online main experiment",
        "status": "PASS_STRICT_ONLINE" if not blocking else "ONLINE_NO_GO",
        "go": not blocking,
        "expected_episodes": expected_episodes,
        "episode_count": len(episode_rows),
        "completed_count": sum(1 for row in episode_rows if row.get("status") == "completed"),
        "gate_decision_count": len(decision_rows),
        "methods": sorted(method_summary),
        "method_summary": method_summary,
        "checks": checks,
        "blocking_reasons": blocking,
        "warning_reasons": warnings,
        "diagnostics": diagnostics,
        "actionable_blockers": actionable,
        "lane_summaries": lane_summaries,
    }


def validate_online_artifacts(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    *,
    expected_methods: list[str],
    expected_seeds: list[int] | None,
    episodes_per_seed: int | None,
    skills_per_episode: int,
) -> dict[str, Any]:
    if not episode_rows:
        raise OnlineValidationError("online_episode_results.csv contains no rows.")
    if not decision_rows:
        raise OnlineValidationError("gate_decisions.csv contains no rows.")
    expected_method_set = set(expected_methods)
    episode_keys: set[tuple[str, int, int]] = set()
    duplicate_episodes: list[dict[str, Any]] = []
    for index, row in enumerate(episode_rows, start=2):
        context = f"online_episode_results.csv:{index}"
        _require_fields(row, REQUIRED_EPISODE_FIELDS, context)
        method = str(row["method"])
        if method not in expected_method_set:
            raise OnlineValidationError(f"{context}: unexpected method {method!r}")
        seed = _parse_int(row["seed"], "seed", context, minimum=0)
        episode_index = _parse_int(row["episode_index"], "episode_index", context, minimum=0)
        _validate_episode_id(row["episode_id"], method, seed, episode_index, context)
        status = str(row["status"])
        if status not in STATUS_VALUES:
            raise OnlineValidationError(f"{context}: invalid status {status!r}")
        _parse_binary(row["task_success"], "task_success", context)
        _parse_int(row["fallback_count"], "fallback_count", context, minimum=0)
        _parse_int(row["unsafe_invocation_count"], "unsafe_invocation_count", context, minimum=0)
        _parse_int(row["violation_count"], "violation_count", context, minimum=0)
        _parse_finite_float(row["completion_time_s"], "completion_time_s", context, minimum=0.0)
        _parse_binary(row["initial_upright_ok"], "initial_upright_ok", context)
        key = (method, seed, episode_index)
        if key in episode_keys:
            duplicate_episodes.append({"method": method, "seed": seed, "episode_index": episode_index})
        episode_keys.add(key)
    if duplicate_episodes:
        raise OnlineValidationError(f"Duplicate episode rows found: {duplicate_episodes[:5]}")

    decision_keys: set[tuple[str, int, int, int]] = set()
    duplicate_decisions: list[dict[str, Any]] = []
    decision_counts_by_episode: Counter[tuple[str, int, int]] = Counter()
    per_skill_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(decision_rows, start=2):
        context = f"gate_decisions.csv:{index}"
        _require_fields(row, REQUIRED_DECISION_FIELDS, context)
        method = str(row["method"])
        if method not in expected_method_set:
            raise OnlineValidationError(f"{context}: unexpected method {method!r}")
        seed = _parse_int(row["seed"], "seed", context, minimum=0)
        episode_index = _parse_int(row["episode_index"], "episode_index", context, minimum=0)
        skill_idx = _parse_int(row["skill_idx"], "skill_idx", context, minimum=1)
        _validate_episode_id(row["episode_id"], method, seed, episode_index, context)
        candidate_skill = str(row["candidate_skill"])
        if candidate_skill not in MAIN_SKILLS:
            raise OnlineValidationError(f"{context}: invalid candidate_skill {candidate_skill!r}")
        executed_skill = str(row["executed_skill"])
        if executed_skill not in MAIN_SKILLS:
            raise OnlineValidationError(f"{context}: invalid executed_skill {executed_skill!r}")
        decision = str(row["decision"])
        if decision not in DECISION_VALUES:
            raise OnlineValidationError(f"{context}: invalid decision {decision!r}")
        result_status = str(row["result_status"])
        if result_status not in RESULT_STATUS_VALUES:
            raise OnlineValidationError(f"{context}: invalid result_status {result_status!r}")
        _parse_finite_float(row["raw_critic_risk"], "raw_critic_risk", context, minimum=0.0, maximum=1.0)
        _parse_finite_float(row["hard_contract_score"], "hard_contract_score", context, minimum=0.0)
        _parse_binary(row["hard_contract_fixed_reject"], "hard_contract_fixed_reject", context)
        _parse_binary(row["fallback_executed"], "fallback_executed", context)
        threshold = str(row.get("threshold", "")).strip()
        if threshold:
            _parse_finite_float(threshold, "threshold", context)
        episode_key = (method, seed, episode_index)
        key = (method, seed, episode_index, skill_idx)
        if key in decision_keys:
            duplicate_decisions.append(
                {"method": method, "seed": seed, "episode_index": episode_index, "skill_idx": skill_idx}
            )
        decision_keys.add(key)
        decision_counts_by_episode[episode_key] += 1
        per_skill_counts[method][candidate_skill] += 1
    if duplicate_decisions:
        raise OnlineValidationError(f"Duplicate gate decision rows found: {duplicate_decisions[:5]}")

    expected_keys = _expected_episode_keys(expected_methods, expected_seeds, episodes_per_seed)
    missing_expected = sorted(expected_keys - episode_keys) if expected_keys else []
    extra_expected = sorted(episode_keys - expected_keys) if expected_keys else []
    completed_episode_keys = {
        (
            str(row["method"]),
            _parse_int(row["seed"], "seed", "episode row", minimum=0),
            _parse_int(row["episode_index"], "episode_index", "episode row", minimum=0),
        )
        for row in episode_rows
        if row.get("status") == "completed"
    }
    missing_decisions = [
        {"method": method, "seed": seed, "episode_index": episode_index, "decision_count": count}
        for method, seed, episode_index in sorted(completed_episode_keys)
        for count in [decision_counts_by_episode[(method, seed, episode_index)]]
        if count < skills_per_episode
    ]
    missing_skill_coverage = [
        {"method": method, "skill": skill}
        for method in expected_methods
        for skill in MAIN_SKILLS
        if per_skill_counts[method][skill] <= 0
    ]
    return {
        "episode_rows_valid": True,
        "decision_rows_valid": True,
        "episode_count": len(episode_rows),
        "decision_count": len(decision_rows),
        "expected_grid_available": bool(expected_keys),
        "expected_grid_count": len(expected_keys),
        "missing_expected_episode_count": len(missing_expected),
        "missing_expected_episode_examples": _key_examples(missing_expected),
        "extra_expected_episode_count": len(extra_expected),
        "extra_expected_episode_examples": _key_examples(extra_expected),
        "completed_episode_count": len(completed_episode_keys),
        "missing_decision_episode_count": len(missing_decisions),
        "missing_decision_episode_examples": missing_decisions[:10],
        "missing_per_skill_coverage": missing_skill_coverage,
        "per_skill_decision_counts": {
            method: {skill: per_skill_counts[method][skill] for skill in MAIN_SKILLS}
            for method in expected_methods
        },
    }


def online_checks(
    *,
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    validation: dict[str, Any],
    method_summary: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    expected_episodes: int,
    expected_methods: list[str],
    min_sonic_unsafe_reduction: float,
    min_hard_unsafe_reduction: float,
    max_task_success_drop_abs: float,
    max_task_success_drop_rel: float,
) -> dict[str, bool]:
    casa_vs_sonic = comparisons.get("casa_vs_sonic", {})
    casa_vs_hard = comparisons.get("casa_vs_hard", {})
    hard_reduction = casa_vs_hard.get("unsafe_reduction")
    return {
        "online_artifacts_schema_valid": True,
        "expected_episode_count_met": len(episode_rows) == expected_episodes,
        "expected_method_seed_episode_grid_complete": validation["missing_expected_episode_count"] == 0,
        "all_expected_methods_present": set(expected_methods).issubset(method_summary),
        "all_episodes_completed": all(row.get("status") == "completed" for row in episode_rows),
        "all_initial_upright_ok": all(_int(row.get("initial_upright_ok")) == 1 for row in episode_rows),
        "gate_decisions_present": len(decision_rows) >= len(episode_rows) * 1,
        "each_completed_episode_has_expected_decisions": validation["missing_decision_episode_count"] == 0,
        "per_skill_decision_coverage_present": not validation["missing_per_skill_coverage"],
        "casa_vs_sonic_unsafe_reduction_ge_40pct": (
            casa_vs_sonic.get("unsafe_reduction") is not None
            and casa_vs_sonic.get("unsafe_reduction", -math.inf) >= min_sonic_unsafe_reduction
        ),
        "casa_vs_hard_unsafe_reduction_ge_20pct_or_undefined": (
            hard_reduction is None or hard_reduction >= min_hard_unsafe_reduction
        ),
        "casa_task_success_drop_abs_le_10pp": (
            casa_vs_sonic.get("task_success_drop_abs", math.inf) <= max_task_success_drop_abs
        ),
        "casa_task_success_drop_rel_le_30pct": (
            casa_vs_sonic.get("task_success_drop_rel", math.inf) <= max_task_success_drop_rel
        ),
    }


def online_warnings(
    method_summary: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    validation: dict[str, Any],
    raw_episode_row_count: int | None,
    raw_decision_row_count: int | None,
) -> list[str]:
    warnings: list[str] = []
    sonic = method_summary.get("sonic_only", {})
    if sonic and float(sonic.get("task_success_rate", 0.0)) < 0.05:
        warnings.append("sonic_task_success_rate_low_relative_drop_check_is_unstable")
    hard = method_summary.get("hard_contract", {})
    casa = method_summary.get("casa_a_per_skill", {})
    if hard and casa and float(hard.get("fallback_rate_per_episode", 0.0)) > float(
        casa.get("fallback_rate_per_episode", 0.0)
    ):
        warnings.append("hard_contract_intervenes_more_than_casa_diagnostic")
    if raw_episode_row_count is not None and raw_episode_row_count > validation["episode_count"]:
        warnings.append("raw_lane_episode_duplicates_were_deduped")
    if raw_decision_row_count is not None and raw_decision_row_count > validation["decision_count"]:
        warnings.append("raw_lane_decision_duplicates_were_deduped")
    hard_reduction = comparisons.get("casa_vs_hard", {}).get("unsafe_reduction")
    if hard_reduction is None:
        warnings.append("hard_contract_relative_reduction_undefined")
    return warnings


def baseline_comparisons(
    method_summary: dict[str, dict[str, Any]],
    *,
    casa_method: str,
    sonic_method: str,
    hard_method: str,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    casa = method_summary.get(casa_method)
    if not casa:
        return comparisons
    for baseline in [sonic_method, hard_method, "raw_critic_0p5", "global_conformal"]:
        base = method_summary.get(baseline)
        if not base:
            continue
        task_drop_abs = float(base["task_success_rate"]) - float(casa["task_success_rate"])
        task_drop_rel = (
            task_drop_abs / float(base["task_success_rate"]) if float(base["task_success_rate"]) > 0 else 0.0
        )
        comparisons[f"casa_vs_{_short_method(baseline)}"] = {
            "baseline_method": baseline,
            "method": casa_method,
            "unsafe_reduction": relative_reduction(
                base["unsafe_invocation_count"],
                casa["unsafe_invocation_count"],
            ),
            "baseline_unsafe_invocation_count": base["unsafe_invocation_count"],
            "method_unsafe_invocation_count": casa["unsafe_invocation_count"],
            "task_success_drop_abs": task_drop_abs,
            "task_success_drop_rel": task_drop_rel,
            "baseline_task_success_rate": base["task_success_rate"],
            "method_task_success_rate": casa["task_success_rate"],
            "baseline_fallback_rate_per_episode": base["fallback_rate_per_episode"],
            "method_fallback_rate_per_episode": casa["fallback_rate_per_episode"],
        }
    return comparisons


def decision_summary_by_method_and_skill(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("method")), str(row.get("candidate_skill")))].append(row)
    for method in sorted({key[0] for key in grouped}):
        summary[method] = {}
        for skill in MAIN_SKILLS:
            skill_rows = grouped.get((method, skill), [])
            if not skill_rows:
                summary[method][skill] = {"decision_count": 0}
                continue
            reject_count = sum(1 for row in skill_rows if row.get("decision") == "reject")
            fallback_count = sum(_int(row.get("fallback_executed")) for row in skill_rows)
            hard_reject = sum(_int(row.get("hard_contract_fixed_reject")) for row in skill_rows)
            result_status = Counter(str(row.get("result_status", "")) for row in skill_rows)
            risks = [_float(row.get("raw_critic_risk")) for row in skill_rows]
            summary[method][skill] = {
                "decision_count": len(skill_rows),
                "reject_count": reject_count,
                "reject_rate": reject_count / len(skill_rows),
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / len(skill_rows),
                "hard_contract_fixed_reject_count": hard_reject,
                "result_status_counts": dict(result_status),
                "raw_critic_risk_mean": sum(risks) / len(risks),
                "raw_critic_risk_min": min(risks),
                "raw_critic_risk_max": max(risks),
            }
    return summary


def skill_label_summary_from_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "safe": 0, "unsafe": 0}))
    loaded = 0
    missing = 0
    malformed = 0
    for row in rows:
        episode_dir = Path(str(row.get("episode_dir", "")))
        summary_path = episode_dir / "rollout_summary.json"
        if not summary_path.exists():
            missing += 1
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            malformed += 1
            continue
        loaded += 1
        method = str(row.get("method"))
        for label in summary.get("skill_labels", []):
            skill = str(label.get("skill_name", ""))
            if skill not in MAIN_SKILLS:
                continue
            stats = output[method][skill]
            stats["total"] += 1
            safe_label = str(label.get("safe_label", ""))
            if safe_label == "unsafe":
                stats["unsafe"] += 1
            elif safe_label == "safe":
                stats["safe"] += 1
    return {
        "loaded_rollout_summary_count": loaded,
        "missing_rollout_summary_count": missing,
        "malformed_rollout_summary_count": malformed,
        "by_method_skill": {method: dict(skill_map) for method, skill_map in output.items()},
    }


def confidence_intervals(method_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for method, row in method_summary.items():
        count = int(row.get("episode_count", 0))
        task_success = int(row.get("task_success_count", 0))
        output[method] = {
            "task_success_rate_wilson_95": _wilson_interval(task_success, count),
            "unsafe_invocation_rate_per_episode": row.get("unsafe_invocation_rate_per_episode"),
        }
    return output


def online_report_markdown(audit: dict[str, Any]) -> str:
    method_summary = audit.get("method_summary", {})
    lines = [
        "# CASA Phase5 Online Main Experiment Report",
        "",
        f"- status: `{audit['status']}`",
        f"- go: `{audit['go']}`",
        f"- episodes: `{audit['episode_count']}` / expected `{audit['expected_episodes']}`",
        f"- completed: `{audit['completed_count']}`",
        f"- gate_decisions: `{audit['gate_decision_count']}`",
        "",
        "## Blockers",
        "",
    ]
    if audit["actionable_blockers"]:
        for blocker in audit["actionable_blockers"]:
            lines.append(f"- `{blocker['check']}`: {blocker['evidence']}")
            lines.append(f"  Next: {blocker['next_step']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if audit["warning_reasons"]:
        lines.extend(f"- {warning}" for warning in audit["warning_reasons"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Method Summary",
            "",
            "| method | episodes | unsafe | unsafe/episode | fallback/episode | task_success_rate | mean_time_s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_ORDER:
        row = method_summary.get(method)
        if not row:
            continue
        lines.append(
            f"| {row.get('method_display', method)} | {row.get('episode_count')} | "
            f"{row.get('unsafe_invocation_count')} | {_fmt(row.get('unsafe_invocation_rate_per_episode'))} | "
            f"{_fmt(row.get('fallback_rate_per_episode'))} | {_fmt(row.get('task_success_rate'))} | "
            f"{_fmt(row.get('mean_completion_time_s'))} |"
        )
    lines.extend(["", "## Checks", ""])
    for name, passed in audit["checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Baseline Comparisons", "", "```json"])
    lines.append(json.dumps(audit["diagnostics"].get("baseline_comparisons", {}), indent=2, sort_keys=True))
    lines.extend(["```", "", "## Artifact Validation", "", "```json"])
    lines.append(json.dumps(audit["diagnostics"].get("artifact_validation", {}), indent=2, sort_keys=True))
    lines.extend(["```", "", "## Per-skill Diagnostics", "", "```json"])
    lines.append(json.dumps(audit["diagnostics"].get("per_skill_decisions", {}), indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines)


def online_go_no_go(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "CASA Phase5 online",
        "go": audit["go"],
        "status": audit["status"],
        "blocking_reasons": audit["blocking_reasons"],
        "warning_reasons": audit["warning_reasons"],
        "actionable_blockers": audit["actionable_blockers"],
        "method_summary": audit["method_summary"],
        "checks": audit["checks"],
    }


def actionable_blocker(
    check: str,
    method_summary: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, str]:
    casa = method_summary.get("casa_a_per_skill", {})
    sonic = method_summary.get("sonic_only", {})
    if check == "casa_vs_sonic_unsafe_reduction_ge_40pct":
        comp = comparisons.get("casa_vs_sonic", {})
        return {
            "check": check,
            "evidence": (
                f"CASA unsafe={casa.get('unsafe_invocation_count')} vs "
                f"SONIC unsafe={sonic.get('unsafe_invocation_count')}; "
                f"reduction={_fmt(comp.get('unsafe_reduction'))}, required >=0.4000."
            ),
            "next_step": (
                "Inspect per-skill online labels and gate decisions; run threshold/intervention sweeps "
                "or hybrid hard_contract OR CASA online experiments."
            ),
        }
    if check == "casa_task_success_drop_rel_le_30pct":
        comp = comparisons.get("casa_vs_sonic", {})
        return {
            "check": check,
            "evidence": (
                f"CASA task_success_rate={_fmt(casa.get('task_success_rate'))} vs "
                f"SONIC={_fmt(sonic.get('task_success_rate'))}; "
                f"relative_drop={_fmt(comp.get('task_success_drop_rel'))}, required <=0.3000."
            ),
            "next_step": (
                "Separate true task failure from safety fallback behavior and inspect low-success episodes; "
                "the SONIC baseline success rate is also reported as a warning when very low."
            ),
        }
    if check == "expected_method_seed_episode_grid_complete":
        return {
            "check": check,
            "evidence": f"Missing expected episodes: {validation.get('missing_expected_episode_examples')}",
            "next_step": "Resume the missing method/seed/episode ranges before making a strict online claim.",
        }
    if check == "each_completed_episode_has_expected_decisions":
        examples = validation.get("missing_decision_episode_examples")
        return {
            "check": check,
            "evidence": f"Completed episodes with too few decisions: {examples}",
            "next_step": "Regenerate or repair gate_decisions.csv for those episodes.",
        }
    if check == "per_skill_decision_coverage_present":
        return {
            "check": check,
            "evidence": f"Missing per-skill coverage: {validation.get('missing_per_skill_coverage')}",
            "next_step": "Ensure the online skill schedule covers walk/turn/gesture/passive for every method.",
        }
    return {
        "check": check,
        "evidence": "See online_acceptance_audit.json for the failed check and diagnostics.",
        "next_step": "Fix the corresponding artifact or rerun the affected online lane.",
    }


def _expected_episode_keys(
    methods: list[str],
    seeds: list[int] | None,
    episodes_per_seed: int | None,
) -> set[tuple[str, int, int]]:
    if not seeds or episodes_per_seed is None:
        return set()
    return {
        (method, seed, episode_index)
        for method in methods
        for seed in seeds
        for episode_index in range(episodes_per_seed)
    }


def _key_examples(keys: list[tuple[str, int, int]], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"method": method, "seed": seed, "episode_index": episode_index}
        for method, seed, episode_index in keys[:limit]
    ]


def _require_fields(row: dict[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in row or str(row.get(field, "")).strip() == ""]
    if missing:
        raise OnlineValidationError(f"{context}: missing required fields: {', '.join(missing)}")


def _validate_episode_id(value: Any, method: str, seed: int, episode_index: int, context: str) -> None:
    expected = f"{method}__seed_{seed}__episode_{episode_index:04d}"
    if str(value) != expected:
        raise OnlineValidationError(f"{context}: episode_id {value!r} does not match expected {expected!r}")


def _parse_int(value: Any, field: str, context: str, *, minimum: int | None = None) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OnlineValidationError(f"{context}: {field} must be an integer, got {value!r}") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise OnlineValidationError(f"{context}: {field} must be an integer, got {value!r}")
    output = int(parsed)
    if minimum is not None and output < minimum:
        raise OnlineValidationError(f"{context}: {field} must be >= {minimum}, got {output}")
    return output


def _parse_binary(value: Any, field: str, context: str) -> int:
    text = str(value).strip().lower()
    if text in {"0", "0.0", "false", "f", "no", "n"}:
        return 0
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    raise OnlineValidationError(f"{context}: {field} must be boolean or 0/1, got {value!r}")


def _parse_finite_float(
    value: Any,
    field: str,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise OnlineValidationError(f"{context}: {field} must be finite, got {value!r}") from exc
    if not math.isfinite(output):
        raise OnlineValidationError(f"{context}: {field} must be finite, got {value!r}")
    if minimum is not None and output < minimum:
        raise OnlineValidationError(f"{context}: {field} must be >= {minimum}, got {output}")
    if maximum is not None and output > maximum:
        raise OnlineValidationError(f"{context}: {field} must be <= {maximum}, got {output}")
    return output


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    phat = successes / total
    denom = 1.0 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z**2 / (4 * total)) / total) / denom
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _short_method(method: str) -> str:
    return {
        "sonic_only": "sonic",
        "hard_contract": "hard",
        "raw_critic_0p5": "raw_critic",
        "global_conformal": "global",
    }.get(method, method)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
