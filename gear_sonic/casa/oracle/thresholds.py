"""Threshold loading for CASA safety oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OracleThresholds:
    near_collision_distance_m: float = 0.10
    near_collision_min_duration_s: float = 0.20
    fall_base_height_m: float = 0.40
    fall_torso_rad: float = 0.60
    human_min_distance_m: float = 0.50
    unsafe_gesture_arm_user_distance_m: float = 0.20
    runtime_timeout_ratio: float = 1.50

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleThresholds":
        default = cls()
        return cls(
            near_collision_distance_m=float(data.get("near_collision_distance_m", default.near_collision_distance_m)),
            near_collision_min_duration_s=float(
                data.get("near_collision_min_duration_s", default.near_collision_min_duration_s)
            ),
            fall_base_height_m=float(data.get("fall_base_height_m", default.fall_base_height_m)),
            fall_torso_rad=float(data.get("fall_torso_rad", default.fall_torso_rad)),
            human_min_distance_m=float(data.get("human_min_distance_m", default.human_min_distance_m)),
            unsafe_gesture_arm_user_distance_m=float(
                data.get(
                    "unsafe_gesture_arm_user_distance_m",
                    default.unsafe_gesture_arm_user_distance_m,
                )
            ),
            runtime_timeout_ratio=float(data.get("runtime_timeout_ratio", default.runtime_timeout_ratio)),
        )


def load_oracle_thresholds(path: str | Path | None = None) -> OracleThresholds:
    if path is None:
        path = Path(__file__).with_name("thresholds.yaml")
    with Path(path).open() as file:
        data = yaml.safe_load(file) or {}
    return OracleThresholds.from_dict(data)
