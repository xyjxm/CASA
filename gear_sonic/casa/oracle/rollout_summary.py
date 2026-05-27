"""Rollout summary helpers for CASA oracle outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .oracle import violation_time_bin
from .violations import Violation


def build_rollout_summary(
    *,
    run_id: str,
    rollout_id: str,
    violations: list[Violation],
    skill_labels: list[dict[str, Any]],
    scene_props: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    first = min(violations, key=lambda violation: violation.start_wall_time) if violations else None
    first_start = min((label["window_start_wall_time"] for label in skill_labels), default=None)
    time_to_violation = None
    if first is not None and first_start is not None:
        time_to_violation = max(0.0, first.start_wall_time - first_start)
    summary = {
        "run_id": run_id,
        "rollout_id": rollout_id,
        "violation_yes_no": bool(violations),
        "violation_type": first.violation_type.value if first else None,
        "violation_types": sorted({violation.violation_type.value for violation in violations}),
        "time_to_violation": time_to_violation,
        "violation_time_bin": violation_time_bin(time_to_violation),
        "violation_count": len(violations),
        "violation_counts": dict(Counter(violation.violation_type.value for violation in violations)),
        "skill_label_counts": dict(Counter(label["safe_label"] for label in skill_labels)),
        "per_skill_label_counts": _per_skill_counts(skill_labels),
        "skill_labels": skill_labels,
        "scene_props": scene_props or {},
    }
    if extra:
        summary.update(extra)
    return summary


def _per_skill_counts(skill_labels: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter] = {}
    for label in skill_labels:
        skill_name = str(label["skill_name"])
        if skill_name not in result:
            result[skill_name] = Counter()
        result[skill_name][str(label["safe_label"])] += 1
    return {skill_name: dict(counter) for skill_name, counter in result.items()}
