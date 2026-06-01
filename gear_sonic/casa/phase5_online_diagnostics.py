"""Failure decomposition for CASA Phase 5 online artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

from gear_sonic.casa.phase5 import MAIN_SKILLS, relative_reduction
from gear_sonic.casa.phase5_online import method_summary_rows
from gear_sonic.casa.phase5_policy import method_display

RETRY_DIAGNOSTIC_FIELDS = (
    "attempt_type",
    "attempt_index",
    "parent_skill_idx",
    "original_candidate_skill",
    "original_candidate_params_json",
    "recovery_attempt_count",
    "retry_decision",
    "retry_reject_reason",
    "retry_executed",
    "task_progress_executed",
    "final_segment_outcome",
)


def build_failure_breakdown(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    *,
    sonic_method: str = "sonic_only",
    casa_method: str = "casa_a_per_skill",
) -> dict[str, Any]:
    labels, episode_meta = _load_rollout_evidence(episode_rows)
    method_summary = {row["method"]: row for row in method_summary_rows(episode_rows)}
    per_method = _per_method_breakdown(episode_rows, decision_rows, labels)
    per_skill = _per_skill_breakdown(decision_rows, labels)
    missing_retry_fields = _missing_retry_fields(decision_rows)
    target_buckets = _bucket_breakdown(episode_rows, episode_meta, "target_bucket")
    scene_complexity = _bucket_breakdown(episode_rows, episode_meta, "scene_complexity")
    clusters = _top_failure_clusters(decision_rows, labels, episode_meta)
    recommendations = _recommendations(per_method, per_skill, clusters)
    casa = method_summary.get(casa_method, {})
    sonic = method_summary.get(sonic_method, {})
    return {
        "phase": "CASA Phase5 online failure decomposition",
        "episode_count": len(episode_rows),
        "decision_count": len(decision_rows),
        "sonic_method": sonic_method,
        "casa_method": casa_method,
        "summary": {
            "casa_vs_sonic_unsafe_reduction": relative_reduction(
                sonic.get("unsafe_invocation_count", 0),
                casa.get("unsafe_invocation_count", 0),
            ),
            "casa_task_success_rate": casa.get("task_success_rate"),
            "sonic_task_success_rate": sonic.get("task_success_rate"),
            "casa_task_success_drop_rel": _relative_drop(
                sonic.get("task_success_rate"),
                casa.get("task_success_rate"),
            ),
        },
        "per_method": per_method,
        "per_skill": per_skill,
        "target_bucket": target_buckets,
        "scene_complexity": scene_complexity,
        "top_failure_clusters": clusters,
        "recommendations": recommendations,
        "diagnostics": {
            "missing_retry_fields": missing_retry_fields,
        },
        "evidence_loading": {
            "rollout_summary_loaded": sum(1 for meta in episode_meta.values() if meta.get("loaded")),
            "rollout_summary_missing": sum(1 for meta in episode_meta.values() if meta.get("missing")),
            "rollout_summary_malformed": sum(1 for meta in episode_meta.values() if meta.get("malformed")),
            "skill_label_count": len(labels),
        },
    }


def failure_breakdown_markdown(breakdown: dict[str, Any]) -> str:
    lines = [
        "# CASA Phase5 Online Failure Decomposition",
        "",
        f"- episodes: `{breakdown['episode_count']}`",
        f"- gate decisions: `{breakdown['decision_count']}`",
        f"- CASA method: `{breakdown['casa_method']}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(breakdown["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Dominant Recommendations",
        "",
    ]
    for item in breakdown["recommendations"]:
        lines.append(f"- `{item['category']}`: {item['evidence']} Next: {item['next_step']}")
    lines.extend(["", "## Per-method Breakdown", ""])
    lines.append(
        "| method | episodes | task_success | unsafe | reject | allow_then_unsafe | "
        "reject_but_still_unsafe | retry_success | final_reject |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method, row in sorted(breakdown["per_method"].items()):
        lines.append(
            f"| {method_display(method)} | {row['episode_count']} | {_fmt(row['task_success_rate'])} | "
            f"{row['unsafe_invocation_count']} | {row['reject_count']} | {row['allow_then_unsafe']} | "
            f"{row['reject_but_still_unsafe']} | {_fmt_count(row.get('reject_then_recovery_then_retry'))} | "
            f"{_fmt_count(row.get('reject_then_all_retries_rejected'))} |"
        )
    lines.extend(["", "## Per-skill Breakdown", "", "```json"])
    lines.append(json.dumps(breakdown["per_skill"], indent=2, sort_keys=True))
    lines.extend(["```", "", "## Top Failure Clusters", "", "```json"])
    lines.append(json.dumps(breakdown["top_failure_clusters"], indent=2, sort_keys=True))
    lines.extend(["```", "", "## Target Buckets", "", "```json"])
    lines.append(json.dumps(breakdown["target_bucket"], indent=2, sort_keys=True))
    lines.extend(["```", "", "## Scene Complexity", "", "```json"])
    lines.append(json.dumps(breakdown["scene_complexity"], indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines)


def _per_method_breakdown(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    labels: dict[tuple[str, int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for method, rows in _group_by(episode_rows, "method").items():
        episode_count = len(rows)
        unsafe = sum(_int(row.get("unsafe_invocation_count")) for row in rows)
        fallback = sum(_int(row.get("fallback_count")) for row in rows)
        task_success = sum(_int(row.get("task_success")) for row in rows)
        by_method[method] = {
            "episode_count": episode_count,
            "task_success_rate": task_success / episode_count if episode_count else 0.0,
            "unsafe_invocation_count": unsafe,
            "unsafe_rate_per_episode": unsafe / episode_count if episode_count else 0.0,
            "fallback_count": fallback,
            "fallback_rate_per_episode": fallback / episode_count if episode_count else 0.0,
            "incomplete_rollout_count": sum(1 for row in rows if row.get("status") not in {"completed"}),
            "environment_failure_count": sum(1 for row in rows if _is_environment_failure(row)),
            "initial_upright_failure_count": sum(1 for row in rows if _initial_upright_failed(row)),
        }
    for method, rows in _group_by(decision_rows, "method").items():
        stats = by_method.setdefault(method, {"episode_count": 0})
        _add_decision_counts(stats, rows, labels)
    return by_method


def _per_skill_breakdown(
    decision_rows: list[dict[str, Any]],
    labels: dict[tuple[str, int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (method, skill), rows in _group_by_pair(decision_rows, "method", "candidate_skill").items():
        stats: dict[str, Any] = {"method": method, "skill": skill}
        _add_decision_counts(stats, rows, labels)
        safe_risks = []
        unsafe_risks = []
        margins = []
        thresholds = []
        for row in rows:
            key = _decision_key(row)
            label = labels.get(key, {})
            risk = _float_or_none(row.get("raw_critic_risk"))
            if risk is None:
                continue
            if label.get("safe_label") == "unsafe":
                unsafe_risks.append(risk)
            else:
                safe_risks.append(risk)
            threshold = _float_or_none(row.get("threshold"))
            if threshold is not None:
                thresholds.append(threshold)
                margins.append(risk - threshold)
        stats["raw_risk_safe"] = _distribution(safe_risks)
        stats["raw_risk_unsafe"] = _distribution(unsafe_risks)
        stats["threshold"] = _distribution(thresholds)
        stats["risk_margin"] = _distribution(margins)
        output.setdefault(method, {})[skill] = stats
    for method in output:
        for skill in MAIN_SKILLS:
            output[method].setdefault(skill, {"method": method, "skill": skill, "decision_count": 0})
    return output


def _add_decision_counts(
    stats: dict[str, Any],
    rows: list[dict[str, Any]],
    labels: dict[tuple[str, int, int, int], dict[str, Any]],
) -> None:
    decision_count = len(rows)
    reject_count = sum(1 for row in rows if row.get("decision") == "reject")
    fallback_count = sum(_int(row.get("fallback_executed")) for row in rows)
    hard_reject_count = sum(_int(row.get("hard_contract_fixed_reject")) for row in rows)
    allow_then_unsafe = 0
    reject_but_still_unsafe = 0
    unsafe_after_allow = 0
    unsafe_after_reject = 0
    low_risk_unsafe = 0
    allow_then_unsafe_segment = 0
    reject_then_recovery_only = 0
    reject_then_recovery_then_retry = 0
    reject_then_retry_allowed_but_unsafe = 0
    reject_then_all_retries_rejected = 0
    recovery_only_final_reject = 0
    no_task_progress_after_reject = 0
    unsafe_after_recovery = 0
    hard_contract_caught_casa_missed = 0
    over_rejection_safe_segments = 0
    has_retry_fields = _has_retry_fields(rows)
    for row in rows:
        label = labels.get(_decision_key(row), {})
        has_label = bool(label)
        unsafe = label.get("safe_label") == "unsafe"
        risk = _float_or_none(row.get("raw_critic_risk"))
        threshold = _float_or_none(row.get("threshold"))
        if unsafe and risk is not None and risk < 0.5:
            low_risk_unsafe += 1
        if unsafe and row.get("decision") == "allow":
            allow_then_unsafe += 1
            unsafe_after_allow += 1
        if unsafe and row.get("decision") == "reject":
            reject_but_still_unsafe += 1
            unsafe_after_reject += 1
        if not has_retry_fields:
            continue
        attempt_type = str(row.get("attempt_type", ""))
        outcome = str(row.get("final_segment_outcome", ""))
        if unsafe and attempt_type == "candidate_allow":
            allow_then_unsafe_segment += 1
        if attempt_type == "recovery_only":
            reject_then_recovery_only += 1
        if attempt_type == "retry_executed" or outcome == "recovery_then_retry":
            reject_then_recovery_then_retry += 1
        if unsafe and attempt_type == "retry_executed":
            reject_then_retry_allowed_but_unsafe += 1
        if attempt_type == "final_reject" or outcome == "recovery_only_final_reject":
            reject_then_all_retries_rejected += 1
        if outcome == "recovery_only_final_reject":
            recovery_only_final_reject += 1
        if row.get("decision") == "reject" and _int(row.get("task_progress_executed")) == 0:
            no_task_progress_after_reject += 1
        recovery_attempt_types = {"recovery_attempt", "recovery_only", "retry_executed"}
        if unsafe and (_int(row.get("fallback_executed")) or attempt_type in recovery_attempt_types):
            unsafe_after_recovery += 1
        if (
            row.get("decision") == "reject"
            and _int(row.get("hard_contract_fixed_reject"))
            and risk is not None
            and threshold is not None
            and risk < threshold
        ):
            hard_contract_caught_casa_missed += 1
        if row.get("decision") == "reject" and has_label and not unsafe:
            over_rejection_safe_segments += 1
    stats.update(
        {
            "decision_count": decision_count,
            "reject_count": reject_count,
            "reject_rate": reject_count / decision_count if decision_count else 0.0,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / decision_count if decision_count else 0.0,
            "hard_contract_fixed_reject_count": hard_reject_count,
            "allow_then_unsafe": allow_then_unsafe,
            "reject_but_still_unsafe": reject_but_still_unsafe,
            "unsafe_after_allow": unsafe_after_allow,
            "unsafe_after_reject": unsafe_after_reject,
            "low_risk_unsafe_count": low_risk_unsafe,
            "allow_then_unsafe_segment": allow_then_unsafe_segment if has_retry_fields else None,
            "reject_then_recovery_only": reject_then_recovery_only if has_retry_fields else None,
            "reject_then_recovery_then_retry": (
                reject_then_recovery_then_retry if has_retry_fields else None
            ),
            "reject_then_retry_allowed_but_unsafe": (
                reject_then_retry_allowed_but_unsafe if has_retry_fields else None
            ),
            "reject_then_all_retries_rejected": (
                reject_then_all_retries_rejected if has_retry_fields else None
            ),
            "recovery_only_final_reject": recovery_only_final_reject if has_retry_fields else None,
            "no_task_progress_after_reject": no_task_progress_after_reject if has_retry_fields else None,
            "unsafe_after_recovery": unsafe_after_recovery if has_retry_fields else None,
            "hard_contract_caught_casa_missed": (
                hard_contract_caught_casa_missed if has_retry_fields else None
            ),
            "over_rejection_safe_segments": over_rejection_safe_segments if has_retry_fields else None,
        }
    )


def _load_rollout_evidence(
    episode_rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int, int, int], dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]]]:
    labels: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    meta: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in episode_rows:
        episode_key = _episode_key(row)
        summary_path = Path(str(row.get("episode_dir", ""))) / "rollout_summary.json"
        if not summary_path.exists():
            meta[episode_key] = {"missing": True}
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta[episode_key] = {"malformed": True}
            continue
        scene_props = summary.get("scene_props", {}) if isinstance(summary.get("scene_props"), dict) else {}
        meta[episode_key] = {
            "loaded": True,
            "target_bucket": scene_props.get("target_bucket"),
            "scene_complexity": scene_props.get("scene_complexity"),
            "scene_family": scene_props.get("scene_family"),
        }
        for label in summary.get("skill_labels", []):
            try:
                skill_idx = int(label.get("skill_idx"))
            except (TypeError, ValueError):
                continue
            labels[(*episode_key, skill_idx)] = dict(label)
    return labels, meta


def _bucket_breakdown(
    episode_rows: list[dict[str, Any]],
    episode_meta: dict[tuple[str, int, int], dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    output: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: {"episodes": 0, "unsafe": 0, "task_success": 0})
    )
    for row in episode_rows:
        key = _episode_key(row)
        bucket = episode_meta.get(key, {}).get(field) or "unknown"
        method = str(row.get("method", ""))
        stats = output[method][str(bucket)]
        stats["episodes"] += 1
        stats["unsafe"] += _int(row.get("unsafe_invocation_count"))
        stats["task_success"] += _int(row.get("task_success"))
    return {method: dict(buckets) for method, buckets in output.items()}


def _top_failure_clusters(
    decision_rows: list[dict[str, Any]],
    labels: dict[tuple[str, int, int, int], dict[str, Any]],
    episode_meta: dict[tuple[str, int, int], dict[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    examples: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in decision_rows:
        key = _decision_key(row)
        label = labels.get(key, {})
        if label.get("safe_label") != "unsafe":
            continue
        episode_key = key[:3]
        meta = episode_meta.get(episode_key, {})
        cluster_key = (
            str(row.get("method", "")),
            str(row.get("candidate_skill", "")),
            str(row.get("decision", "")),
            str(meta.get("target_bucket") or "unknown"),
            str(meta.get("scene_complexity") or "unknown"),
            ",".join(label.get("triggered_violation_types", []) or ["unknown"]),
        )
        counts[cluster_key] += 1
        examples.setdefault(
            cluster_key,
            {
                "episode_id": row.get("episode_id"),
                "skill_idx": row.get("skill_idx"),
                "raw_critic_risk": row.get("raw_critic_risk"),
                "threshold": row.get("threshold"),
            },
        )
    output = []
    for key, count in counts.most_common(limit):
        method, skill, decision, target_bucket, complexity, violation_types = key
        output.append(
            {
                "count": count,
                "method": method,
                "skill": skill,
                "decision": decision,
                "target_bucket": target_bucket,
                "scene_complexity": complexity,
                "violation_types": violation_types,
                "example": examples[key],
            }
        )
    return output


def _recommendations(
    per_method: dict[str, Any],
    per_skill: dict[str, Any],
    clusters: list[dict[str, Any]],
) -> list[dict[str, str]]:
    totals = {
        "allow_then_unsafe": sum(row.get("allow_then_unsafe", 0) for row in per_method.values()),
        "reject_but_still_unsafe": sum(row.get("reject_but_still_unsafe", 0) for row in per_method.values()),
        "low_risk_unsafe": sum(row.get("low_risk_unsafe_count", 0) for row in per_method.values()),
        "environment": sum(row.get("environment_failure_count", 0) for row in per_method.values()),
        "fallback": sum(row.get("fallback_count", 0) for row in per_method.values()),
    }
    output = []
    if totals["allow_then_unsafe"] >= totals["reject_but_still_unsafe"] and totals["allow_then_unsafe"] > 0:
        output.append(
            {
                "category": "gate_recall_or_receding_check",
                "evidence": f"allow_then_unsafe={totals['allow_then_unsafe']}",
                "next_step": "Pilot hard-OR-CASA and receding segment checks before long skills.",
            }
        )
    if totals["reject_but_still_unsafe"] > 0:
        output.append(
            {
                "category": "fallback_quality",
                "evidence": f"reject_but_still_unsafe={totals['reject_but_still_unsafe']}",
                "next_step": "Use adaptive recovery and inspect rejected unsafe clusters.",
            }
        )
    if totals["fallback"] > 0:
        output.append(
            {
                "category": "task_success_over_rejection",
                "evidence": f"fallback_count={totals['fallback']}",
                "next_step": "Sweep threshold scales and adaptive_retry count against task success.",
            }
        )
    if totals["low_risk_unsafe"] > 0:
        output.append(
            {
                "category": "online_distribution_shift",
                "evidence": f"low_risk_unsafe_count={totals['low_risk_unsafe']}",
                "next_step": "Consider an online adapter using only dev/tuning online data.",
            }
        )
    if totals["environment"] > 0:
        output.append(
            {
                "category": "evaluation_hygiene",
                "evidence": f"environment_failure_count={totals['environment']}",
                "next_step": "Fix reset/upright/rollout completeness before strict final runs.",
            }
        )
    if clusters:
        top = clusters[0]
        output.append(
            {
                "category": "top_cluster",
                "evidence": (
                    f"{top['count']} unsafe labels for {top['method']}/{top['skill']} "
                    f"in {top['target_bucket']}/{top['scene_complexity']}"
                ),
                "next_step": "Target this cluster in the next pilot policy sweep.",
            }
        )
    if not output:
        output.append(
            {
                "category": "no_dominant_failure_found",
                "evidence": "No unsafe decision labels were available.",
                "next_step": "Check rollout_summary.json loading and skill label generation.",
            }
        )
    return output


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": values[0],
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    index = q * (len(values) - 1)
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row.get(key, ""))].append(row)
    return output


def _group_by_pair(rows: list[dict[str, Any]], a: str, b: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[(str(row.get(a, "")), str(row.get(b, "")))].append(row)
    return output


def _missing_retry_fields(rows: list[dict[str, Any]]) -> list[str]:
    available: set[str] = set()
    for row in rows:
        available.update(row)
    return [field for field in RETRY_DIAGNOSTIC_FIELDS if field not in available]


def _has_retry_fields(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and not _missing_retry_fields(rows)


def _decision_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row.get("method", "")),
        _int(row.get("seed")),
        _int(row.get("episode_index")),
        _int(row.get("skill_idx")),
    )


def _episode_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("method", "")), _int(row.get("seed")), _int(row.get("episode_index")))


def _relative_drop(baseline: Any, method: Any) -> float:
    baseline_float = _float_or_none(baseline) or 0.0
    method_float = _float_or_none(method) or 0.0
    if baseline_float <= 0:
        return 0.0
    return (baseline_float - method_float) / baseline_float


def _is_environment_failure(row: dict[str, Any]) -> bool:
    text = f"{row.get('status', '')} {row.get('error', '')}".lower()
    return any(token in text for token in ["sim_state", "environment", "rollout", "timeout", "zmq"])


def _initial_upright_failed(row: dict[str, Any]) -> bool:
    return row.get("status") == "initial_upright_failed" or _int(row.get("initial_upright_ok")) == 0


def _fmt(value: Any) -> str:
    parsed = _float_or_none(value)
    return "n/a" if parsed is None else f"{parsed:.4f}"


def _fmt_count(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
