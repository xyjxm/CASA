"""Read MuJoCo episode logs and compute CASA skill evidence."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEFT_SHOULDER_PITCH_29DOF = 15
RIGHT_SHOULDER_PITCH_29DOF = 22
RIGHT_SHOULDER_PITCH_43DOF = 29


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quat_wxyz(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class SkillEvaluationConfig:
    """Thresholds used to infer skill success from MuJoCo state logs."""

    walk_speed_threshold: float = 0.1
    walk_min_ratio: float = 0.5
    passive_speed_threshold: float = 0.08
    passive_min_ratio: float = 0.7
    passive_tail_fraction: float = 0.5
    turn_yaw_threshold_rad: float = 0.3
    gesture_min_rom: float = 0.05
    middle_trim_fraction: float = 0.2


@dataclass(frozen=True)
class SkillEvidence:
    success: bool
    downstream_state_observed: bool
    status: str
    termination_reason: str
    details: dict[str, Any]


class EpisodeLogReader:
    """Small CSV reader for sim_state.csv windows keyed by wall time."""

    def __init__(self, sim_state_csv: Path, config: SkillEvaluationConfig | None = None) -> None:
        self.sim_state_csv = Path(sim_state_csv)
        self.config = config or SkillEvaluationConfig()
        self.rows: list[dict[str, Any]] = []
        self.fieldnames: list[str] = []
        self._load()

    @classmethod
    def from_log_dir(cls, sim_log_dir: Path, config: SkillEvaluationConfig | None = None) -> "EpisodeLogReader":
        return cls(Path(sim_log_dir) / "sim_state.csv", config=config)

    def _load(self) -> None:
        if not self.sim_state_csv.exists():
            return
        with self.sim_state_csv.open(newline="") as file:
            reader = csv.DictReader(file)
            self.fieldnames = list(reader.fieldnames or [])
            for raw in reader:
                row: dict[str, Any] = dict(raw)
                row["wall_time"] = _to_float(raw.get("wall_time"))
                row["sim_time"] = _to_float(raw.get("sim_time"))
                row["base_speed_xy"] = math.hypot(
                    _to_float(raw.get("base_lin_vel_x")),
                    _to_float(raw.get("base_lin_vel_y")),
                )
                row["base_yaw"] = _yaw_from_quat_wxyz(
                    _to_float(raw.get("base_quat_w"), 1.0),
                    _to_float(raw.get("base_quat_x")),
                    _to_float(raw.get("base_quat_y")),
                    _to_float(raw.get("base_quat_z")),
                )
                self.rows.append(row)
        self.rows.sort(key=lambda row: row["wall_time"])

    @property
    def available(self) -> bool:
        return bool(self.rows)

    def window(self, start_wall_time: float, end_wall_time: float) -> list[dict[str, Any]]:
        if end_wall_time < start_wall_time:
            start_wall_time, end_wall_time = end_wall_time, start_wall_time
        return [row for row in self.rows if start_wall_time <= row["wall_time"] <= end_wall_time]

    def middle_window(
        self,
        start_wall_time: float,
        end_wall_time: float,
        trim_fraction: float | None = None,
    ) -> list[dict[str, Any]]:
        if trim_fraction is None:
            trim_fraction = self.config.middle_trim_fraction
        duration = max(0.0, end_wall_time - start_wall_time)
        return self.window(
            start_wall_time + duration * trim_fraction,
            end_wall_time - duration * trim_fraction,
        )

    def shoulder_pitch_columns(self) -> tuple[str, str]:
        max_body_q = self._max_body_q_index()
        left = f"body_q_{LEFT_SHOULDER_PITCH_29DOF}"
        if max_body_q >= 42:
            right = f"body_q_{RIGHT_SHOULDER_PITCH_43DOF}"
        else:
            right = f"body_q_{RIGHT_SHOULDER_PITCH_29DOF}"
        return left, right

    def _max_body_q_index(self) -> int:
        max_index = -1
        for field in self.fieldnames:
            if not field.startswith("body_q_"):
                continue
            try:
                max_index = max(max_index, int(field.removeprefix("body_q_")))
            except ValueError:
                continue
        return max_index

    def evaluate(
        self,
        skill_name: str,
        params: dict[str, Any],
        start_wall_time: float,
        end_wall_time: float,
    ) -> SkillEvidence:
        rows = self.window(start_wall_time, end_wall_time)
        middle_rows = self.middle_window(start_wall_time, end_wall_time)
        if not rows:
            return SkillEvidence(
                success=False,
                downstream_state_observed=False,
                status="unverified",
                termination_reason="sim_log_window_empty",
                details={"row_count": 0, "sim_state_csv": str(self.sim_state_csv)},
            )

        if skill_name == "walk":
            return self._evaluate_walk(rows, middle_rows)
        if skill_name == "turn":
            return self._evaluate_turn(rows, params)
        if skill_name == "gesture":
            return self._evaluate_gesture(rows, middle_rows, params)
        if skill_name == "passive":
            return self._evaluate_passive(rows, middle_rows)

        return SkillEvidence(
            success=False,
            downstream_state_observed=True,
            status="unverified",
            termination_reason=f"unknown_skill:{skill_name}",
            details={"row_count": len(rows)},
        )

    def _evaluate_walk(self, rows: list[dict[str, Any]], middle_rows: list[dict[str, Any]]) -> SkillEvidence:
        speed_rows = middle_rows or rows
        speed_threshold = self.config.walk_speed_threshold
        min_ratio = self.config.walk_min_ratio
        ratio = _ratio(speed_rows, lambda row: _to_float(row.get("base_speed_xy")) > speed_threshold)
        legacy_ratio = _ratio(speed_rows, lambda row: _to_float(row.get("base_speed_xy")) > 0.1)
        success = ratio > min_ratio
        return SkillEvidence(
            success=success,
            downstream_state_observed=True,
            status="success" if success else "failed",
            termination_reason="" if success else "walk_speed_threshold_not_met",
            details={
                "row_count": len(rows),
                "middle_row_count": len(speed_rows),
                "speed_threshold": speed_threshold,
                "min_ratio": min_ratio,
                "speed_gt_threshold_ratio": ratio,
                "speed_gt_0p1_ratio": legacy_ratio,
                "max_base_speed_xy": _max(speed_rows, "base_speed_xy"),
            },
        )

    def _evaluate_passive(self, rows: list[dict[str, Any]], middle_rows: list[dict[str, Any]]) -> SkillEvidence:
        speed_rows = _tail_rows(rows, self.config.passive_tail_fraction) or middle_rows or rows
        speed_threshold = self.config.passive_speed_threshold
        min_ratio = self.config.passive_min_ratio
        ratio = _ratio(speed_rows, lambda row: _to_float(row.get("base_speed_xy")) < speed_threshold)
        legacy_ratio = _ratio(speed_rows, lambda row: _to_float(row.get("base_speed_xy")) < 0.05)
        success = ratio > min_ratio
        return SkillEvidence(
            success=success,
            downstream_state_observed=True,
            status="success" if success else "failed",
            termination_reason="" if success else "passive_stillness_threshold_not_met",
            details={
                "row_count": len(rows),
                "middle_row_count": len(middle_rows or rows),
                "eval_row_count": len(speed_rows),
                "eval_window": "tail",
                "tail_fraction": self.config.passive_tail_fraction,
                "speed_threshold": speed_threshold,
                "min_ratio": min_ratio,
                "speed_lt_threshold_ratio": ratio,
                "speed_lt_0p05_ratio": legacy_ratio,
                "max_base_speed_xy": _max(speed_rows, "base_speed_xy"),
            },
        )

    def _evaluate_turn(self, rows: list[dict[str, Any]], params: dict[str, Any]) -> SkillEvidence:
        target_rad = math.radians(_to_float(params.get("face_yaw_deg")))
        yaw_start = _to_float(rows[0].get("base_yaw"))
        yaw_end = _to_float(rows[-1].get("base_yaw"))
        yaw_error = abs(_angle_wrap(yaw_end - target_rad))
        yaw_threshold = self.config.turn_yaw_threshold_rad
        success = yaw_error < yaw_threshold
        return SkillEvidence(
            success=success,
            downstream_state_observed=True,
            status="success" if success else "failed",
            termination_reason="" if success else "turn_yaw_error_threshold_not_met",
            details={
                "row_count": len(rows),
                "target_yaw_rad": target_rad,
                "yaw_start": yaw_start,
                "yaw_end": yaw_end,
                "yaw_delta": _angle_wrap(yaw_end - yaw_start),
                "yaw_error_rad": yaw_error,
                "yaw_threshold_rad": yaw_threshold,
            },
        )

    def _evaluate_gesture(
        self,
        rows: list[dict[str, Any]],
        middle_rows: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> SkillEvidence:
        selected = middle_rows or rows
        left_col, right_col = self.shoulder_pitch_columns()
        side = str(params.get("side", "both"))
        roms: dict[str, float | None] = {
            "left_shoulder_pitch_rom": _rom(selected, left_col),
            "right_shoulder_pitch_rom": _rom(selected, right_col),
        }
        if side == "left":
            measured_rom = roms["left_shoulder_pitch_rom"]
        elif side == "right":
            measured_rom = roms["right_shoulder_pitch_rom"]
        else:
            measured_rom = max(value or 0.0 for value in roms.values())
        min_rom = self.config.gesture_min_rom
        success = measured_rom is not None and measured_rom > min_rom
        return SkillEvidence(
            success=success,
            downstream_state_observed=True,
            status="success" if success else "failed",
            termination_reason="" if success else "gesture_rom_threshold_not_met",
            details={
                "row_count": len(rows),
                "middle_row_count": len(selected),
                "side": side,
                "measured_rom": measured_rom,
                "min_rom": min_rom,
                "left_shoulder_column": left_col,
                "right_shoulder_column": right_col,
                **roms,
            },
        )


def _ratio(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def _max(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return max(_to_float(row.get(field)) for row in rows)


def _tail_rows(rows: list[dict[str, Any]], tail_fraction: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    tail_fraction = min(max(tail_fraction, 0.0), 1.0)
    if tail_fraction <= 0.0:
        return []
    start_wall_time = _to_float(rows[0].get("wall_time"))
    end_wall_time = _to_float(rows[-1].get("wall_time"))
    cutoff = start_wall_time + max(0.0, end_wall_time - start_wall_time) * (1.0 - tail_fraction)
    return [row for row in rows if _to_float(row.get("wall_time")) >= cutoff]


def _rom(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_to_float(row.get(field), default=float("nan")) for row in rows if field in row]
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return None
    return max(values) - min(values)
