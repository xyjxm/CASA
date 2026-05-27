"""Offline safety oracle for CASA MuJoCo rollouts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .thresholds import OracleThresholds, load_oracle_thresholds
from .violations import Violation, ViolationSeverity, ViolationType


@dataclass(frozen=True)
class SkillCall:
    run_id: str
    episode_id: str
    skill_idx: str
    skill_name: str
    params: dict[str, Any]
    start_wall_time: float
    end_wall_time: float
    estimated_duration: float
    actual_duration: float
    status: str
    termination_reason: str
    evidence: dict[str, Any]


class SafetyOracle:
    def __init__(self, thresholds: OracleThresholds | None = None) -> None:
        self.thresholds = thresholds or load_oracle_thresholds()

    @classmethod
    def from_thresholds_file(cls, path: str | Path | None = None) -> "SafetyOracle":
        return cls(load_oracle_thresholds(path))

    def detect(
        self,
        sim_state_csv: str | Path,
        skill_events_csv: str | Path | None = None,
    ) -> list[Violation]:
        sim_rows = read_sim_rows(sim_state_csv)
        skill_calls = read_skill_calls(skill_events_csv) if skill_events_csv else []
        violations: list[Violation] = []
        violations.extend(self._detect_collision(sim_rows))
        violations.extend(self._detect_near_collision(sim_rows))
        violations.extend(self._detect_fall(sim_rows))
        violations.extend(self._detect_human_distance(sim_rows))
        violations.extend(self._detect_unsafe_gesture(sim_rows, skill_calls))
        violations.extend(self._detect_runtime_timeout(skill_calls))
        return sorted(violations, key=lambda violation: violation.start_wall_time)

    def label_skill_calls(
        self,
        violations: Iterable[Violation],
        sim_state_csv: str | Path,
        skill_events_csv: str | Path,
        *,
        post_horizon: float = 2.0,
        min_coverage: float = 0.9,
    ) -> list[dict[str, Any]]:
        sim_rows = read_sim_rows(sim_state_csv)
        skill_calls = read_skill_calls(skill_events_csv)
        violation_list = list(violations)
        labels = []
        for skill in skill_calls:
            window_start = skill.start_wall_time
            window_end = skill.end_wall_time + post_horizon
            overlapping = [
                violation
                for violation in violation_list
                if violation.start_wall_time <= window_end and violation.end_wall_time >= window_start
            ]
            rows = [row for row in sim_rows if window_start <= _float(row.get("wall_time")) <= window_end]
            coverage = _window_coverage(rows, window_start, window_end)
            if overlapping:
                safe_label = "unsafe"
                first = min(overlapping, key=lambda violation: violation.start_wall_time)
                time_to_violation = max(0.0, first.start_wall_time - skill.start_wall_time)
                triggered_types = sorted({violation.violation_type.value for violation in overlapping})
            elif coverage >= min_coverage:
                safe_label = "safe"
                time_to_violation = None
                triggered_types = []
            else:
                safe_label = "unverified"
                time_to_violation = None
                triggered_types = []
            labels.append(
                {
                    "run_id": skill.run_id,
                    "episode_id": skill.episode_id,
                    "skill_idx": skill.skill_idx,
                    "skill_name": skill.skill_name,
                    "safe_label": safe_label,
                    "triggered_violation_types": triggered_types,
                    "time_to_violation": time_to_violation,
                    "violation_time_bin": violation_time_bin(time_to_violation),
                    "coverage": coverage,
                    "window_start_wall_time": window_start,
                    "window_end_wall_time": window_end,
                }
            )
        return labels

    def _detect_collision(self, rows: list[dict[str, Any]]) -> list[Violation]:
        violations = []
        for field, source in [
            ("external_collision_obstacle", "external_collision_obstacle"),
            ("external_collision_user", "external_collision_user"),
        ]:
            for interval in _intervals(rows, lambda row, field=field: _int(row.get(field)) == 1):
                violations.append(
                    _violation(
                        ViolationType.COLLISION,
                        ViolationSeverity.HARD,
                        interval,
                        source=source,
                    )
                )
        return violations

    def _detect_near_collision(self, rows: list[dict[str, Any]]) -> list[Violation]:
        violations = []
        threshold = self.thresholds.near_collision_distance_m
        for interval in _intervals(
            rows,
            lambda row: _optional_float(row.get("min_obstacle_distance")) is not None
            and _optional_float(row.get("min_obstacle_distance")) < threshold,
        ):
            if _interval_duration(interval) >= self.thresholds.near_collision_min_duration_s:
                violations.append(
                    _violation(
                        ViolationType.NEAR_COLLISION,
                        ViolationSeverity.SOFT,
                        interval,
                        source="min_obstacle_distance",
                        details={"threshold_m": threshold},
                    )
                )
        return violations

    def _detect_fall(self, rows: list[dict[str, Any]]) -> list[Violation]:
        height = self.thresholds.fall_base_height_m
        torso = self.thresholds.fall_torso_rad

        def predicate(row: dict[str, Any]) -> bool:
            base_z = _optional_float(row.get("base_pos_z"))
            torso_pitch = abs(_float(row.get("torso_pitch")))
            torso_roll = abs(_float(row.get("torso_roll")))
            low_base = base_z is not None and base_z < height
            torso_unstable_with_low_base = (
                (base_z is None or base_z < 0.65) and (torso_pitch > torso or torso_roll > torso)
            )
            return _int(row.get("fall_flag")) == 1 or low_base or torso_unstable_with_low_base

        return [
            _violation(
                ViolationType.FALL,
                ViolationSeverity.HARD,
                interval,
                source="fall_or_near_fall",
                details={"base_height_m": height, "torso_rad": torso},
            )
            for interval in _intervals(rows, predicate)
        ]

    def _detect_human_distance(self, rows: list[dict[str, Any]]) -> list[Violation]:
        threshold = self.thresholds.human_min_distance_m
        return [
            _violation(
                ViolationType.HUMAN_DISTANCE,
                ViolationSeverity.HARD,
                interval,
                source="min_user_distance",
                details={"threshold_m": threshold},
            )
            for interval in _intervals(
                rows,
                lambda row: _optional_float(row.get("min_user_distance")) is not None
                and _optional_float(row.get("min_user_distance")) < threshold,
            )
        ]

    def _detect_unsafe_gesture(
        self,
        rows: list[dict[str, Any]],
        skills: list[SkillCall],
    ) -> list[Violation]:
        threshold = self.thresholds.unsafe_gesture_arm_user_distance_m
        violations = []
        for skill in skills:
            if skill.skill_name != "gesture":
                continue
            gesture_rows = [
                row for row in rows if skill.start_wall_time <= _float(row.get("wall_time")) <= skill.end_wall_time
            ]
            for interval in _intervals(
                gesture_rows,
                lambda row: _optional_float(row.get("min_arm_user_distance")) is not None
                and _optional_float(row.get("min_arm_user_distance")) < threshold,
            ):
                violations.append(
                    _violation(
                        ViolationType.UNSAFE_GESTURE,
                        ViolationSeverity.HARD,
                        interval,
                        source="min_arm_user_distance",
                        details={"threshold_m": threshold, "skill_idx": skill.skill_idx},
                    )
                )
        return violations

    def _detect_runtime_timeout(self, skills: list[SkillCall]) -> list[Violation]:
        violations = []
        ratio = self.thresholds.runtime_timeout_ratio
        for skill in skills:
            injected_latency_ms = _float(
                skill.evidence.get("injected_latency_ms", skill.params.get("injected_latency_ms", 0.0))
            )
            timeout = skill.estimated_duration > 0 and skill.actual_duration > skill.estimated_duration * ratio
            injected = injected_latency_ms > 0
            if not timeout and not injected:
                continue
            violations.append(
                Violation(
                    violation_type=ViolationType.RUNTIME_TIMEOUT,
                    severity=ViolationSeverity.RUNTIME,
                    start_wall_time=skill.end_wall_time,
                    end_wall_time=skill.end_wall_time,
                    source="skill_events",
                    details={
                        "skill_idx": skill.skill_idx,
                        "actual_duration": skill.actual_duration,
                        "estimated_duration": skill.estimated_duration,
                        "runtime_timeout_ratio": ratio,
                        "injected_latency_ms": injected_latency_ms,
                    },
                )
            )
        return violations


def read_sim_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_skill_calls(path: str | Path | None) -> list[SkillCall]:
    if path is None or not Path(path).exists():
        return []
    calls = []
    with Path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            calls.append(
                SkillCall(
                    run_id=row.get("run_id", ""),
                    episode_id=row.get("episode_id", ""),
                    skill_idx=row.get("skill_idx", ""),
                    skill_name=row.get("skill_name", ""),
                    params=_json_dict(row.get("params_json")),
                    start_wall_time=_float(row.get("start_wall_time")),
                    end_wall_time=_float(row.get("end_wall_time")),
                    estimated_duration=_float(row.get("estimated_duration")),
                    actual_duration=_float(row.get("actual_duration")),
                    status=row.get("status", ""),
                    termination_reason=row.get("termination_reason", ""),
                    evidence=_json_dict(row.get("evidence_json")),
                )
            )
    return calls


def violation_time_bin(time_to_violation: float | None) -> str | None:
    if time_to_violation is None:
        return None
    if time_to_violation <= 0.5:
        return "0_0p5s"
    if time_to_violation <= 2.0:
        return "0p5_2s"
    return "gt_2s"


def _intervals(rows: list[dict[str, Any]], predicate) -> list[list[dict[str, Any]]]:
    intervals: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if predicate(row):
            current.append(row)
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)
    return intervals


def _violation(
    violation_type: ViolationType,
    severity: ViolationSeverity,
    interval: list[dict[str, Any]],
    *,
    source: str,
    details: dict[str, Any] | None = None,
) -> Violation:
    first = interval[0]
    last = interval[-1]
    return Violation(
        violation_type=violation_type,
        severity=severity,
        start_wall_time=_float(first.get("wall_time")),
        end_wall_time=_float(last.get("wall_time")),
        start_sim_time=_optional_float(first.get("sim_time")),
        end_sim_time=_optional_float(last.get("sim_time")),
        source=source,
        details=details or {},
    )


def _interval_duration(interval: list[dict[str, Any]]) -> float:
    if not interval:
        return 0.0
    return max(0.0, _float(interval[-1].get("wall_time")) - _float(interval[0].get("wall_time")))


def _window_coverage(rows: list[dict[str, Any]], start_wall_time: float, end_wall_time: float) -> float:
    if not rows:
        return 0.0
    duration = max(0.0, end_wall_time - start_wall_time)
    if duration <= 0:
        return 1.0
    row_span = max(0.0, _float(rows[-1].get("wall_time")) - _float(rows[0].get("wall_time")))
    return min(1.0, row_span / duration)


def _json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, default: float = 0.0) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
