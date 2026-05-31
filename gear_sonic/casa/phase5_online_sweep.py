"""Pilot sweep reporting helpers for Phase 5 online policy candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gear_sonic.casa.phase5 import relative_reduction
from gear_sonic.casa.phase5_online import method_summary_rows, read_csv_rows
from gear_sonic.casa.phase5_policy import method_display


def build_sweep_grid(
    *,
    fallback_policies: list[str],
    hard_or_values: list[bool],
    segmentation_values: list[bool],
    max_segment_durations: list[float],
    global_threshold_scales: list[float],
    per_skill_threshold_scales: list[str],
    recovery_retry_counts: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for fallback_policy in fallback_policies:
        for hard_or in hard_or_values:
            for segmentation in segmentation_values:
                for max_segment_duration in max_segment_durations:
                    for global_scale in global_threshold_scales:
                        for by_skill in per_skill_threshold_scales:
                            for retry_count in recovery_retry_counts:
                                rows.append(
                                    {
                                        "candidate_id": f"candidate_{len(rows):04d}",
                                        "fallback_policy": fallback_policy,
                                        "hard_or_casa": hard_or,
                                        "segment_long_skills": segmentation,
                                        "max_segment_duration": max_segment_duration,
                                        "threshold_scale_global": global_scale,
                                        "threshold_scale_by_skill": by_skill,
                                        "recovery_retry_count": retry_count,
                                        "status": "planned",
                                    }
                                )
    return rows


def evaluate_sweep_candidates(
    candidates: list[tuple[str, Path]],
    *,
    sonic_method: str = "sonic_only",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id, path in candidates:
        online_dir = path / "merged" if (path / "merged" / "online_episode_results.csv").exists() else path
        episodes = read_csv_rows(online_dir / "online_episode_results.csv")
        summary = {row["method"]: row for row in method_summary_rows(episodes)}
        sonic = summary.get(sonic_method)
        if not sonic:
            continue
        for method, row in sorted(summary.items()):
            if method == sonic_method:
                continue
            sonic_success_rate = float(sonic["task_success_rate"])
            task_drop_abs = sonic_success_rate - float(row["task_success_rate"])
            task_drop_rel = task_drop_abs / sonic_success_rate if sonic_success_rate > 0 else 0.0
            unsafe_reduction = relative_reduction(
                sonic["unsafe_invocation_count"],
                row["unsafe_invocation_count"],
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_path": str(path),
                    "method": method,
                    "method_display": method_display(method),
                    "episodes": row["episode_count"],
                    "unsafe_invocation_count": row["unsafe_invocation_count"],
                    "unsafe_reduction_vs_sonic": unsafe_reduction,
                    "task_success_rate": row["task_success_rate"],
                    "task_success_drop_rel_vs_sonic": task_drop_rel,
                    "fallback_rate_per_episode": row["fallback_rate_per_episode"],
                    "pilot_pass": (
                        unsafe_reduction is not None
                        and unsafe_reduction >= 0.45
                        and task_drop_rel <= 0.20
                    ),
                    "status": "evaluated",
                }
            )
    return rows


def pareto_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    output = []
    for row in evaluated:
        reduction = _float(row.get("unsafe_reduction_vs_sonic"))
        drop = _float(row.get("task_success_drop_rel_vs_sonic"))
        fallback = _float(row.get("fallback_rate_per_episode"))
        dominated = False
        for other in evaluated:
            if other is row:
                continue
            other_reduction = _float(other.get("unsafe_reduction_vs_sonic"))
            other_drop = _float(other.get("task_success_drop_rel_vs_sonic"))
            other_fallback = _float(other.get("fallback_rate_per_episode"))
            if (
                other_reduction >= reduction
                and other_drop <= drop
                and other_fallback <= fallback
                and (other_reduction > reduction or other_drop < drop or other_fallback < fallback)
            ):
                dominated = True
                break
        if not dominated:
            output.append(row)
    return sorted(
        output,
        key=lambda row: (
            not bool(row.get("pilot_pass")),
            -_float(row.get("unsafe_reduction_vs_sonic")),
            _float(row.get("task_success_drop_rel_vs_sonic")),
            _float(row.get("fallback_rate_per_episode")),
        ),
    )


def sweep_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pareto = pareto_candidates(rows)
    return {
        "phase": "CASA Phase5 online policy pilot sweep",
        "candidate_count": len(rows),
        "evaluated_count": sum(1 for row in rows if row.get("status") == "evaluated"),
        "planned_count": sum(1 for row in rows if row.get("status") == "planned"),
        "pilot_pass_count": sum(1 for row in rows if row.get("pilot_pass")),
        "pareto_candidates": pareto,
        "rows": rows,
        "promotion_rule": {
            "unsafe_reduction_vs_sonic": ">= 0.45",
            "task_success_drop_rel_vs_sonic": "<= 0.20",
            "final_claim": "fresh held-out 2500-episode strict run only",
        },
    }


def sweep_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CASA Phase5 Online Policy Sweep",
        "",
        f"- candidates: `{report['candidate_count']}`",
        f"- evaluated: `{report['evaluated_count']}`",
        f"- planned: `{report['planned_count']}`",
        f"- pilot pass: `{report['pilot_pass_count']}`",
        "",
        "## Pareto Candidates",
        "",
        "| candidate | method | unsafe_reduction | task_success_drop_rel | fallback/episode | pilot_pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["pareto_candidates"]:
        lines.append(
            f"| {row.get('candidate_id')} | {row.get('method_display', row.get('method'))} | "
            f"{_fmt(row.get('unsafe_reduction_vs_sonic'))} | "
            f"{_fmt(row.get('task_success_drop_rel_vs_sonic'))} | "
            f"{_fmt(row.get('fallback_rate_per_episode'))} | {row.get('pilot_pass')} |"
        )
    if not report["pareto_candidates"]:
        lines.append("| none |  |  |  |  |  |")
    lines.extend(["", "## Full Report", "", "```json"])
    lines.append(json.dumps(report, indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"
