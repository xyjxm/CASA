"""CASA safety oracle violation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ViolationType(str, Enum):
    COLLISION = "collision"
    NEAR_COLLISION = "near_collision"
    FALL = "fall"
    HUMAN_DISTANCE = "human_distance_violation"
    UNSAFE_GESTURE = "unsafe_gesture"
    RUNTIME_TIMEOUT = "runtime_timeout"


class ViolationSeverity(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class Violation:
    violation_type: ViolationType
    severity: ViolationSeverity
    start_wall_time: float
    end_wall_time: float
    start_sim_time: float | None = None
    end_sim_time: float | None = None
    source: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_wall_time - self.start_wall_time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["violation_type"] = self.violation_type.value
        data["severity"] = self.severity.value
        data["duration_s"] = self.duration_s
        return data
