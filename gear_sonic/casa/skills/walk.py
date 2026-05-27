"""Walk skill wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gear_sonic.casa.io.zmq_publisher import LOCOMOTION_WALK

from .base import PlannerCommand, Skill
from .utils import facing_from_yaw, normalize_xy


@dataclass(frozen=True)
class WalkSkill(Skill):
    vx: float
    vy: float
    facing_yaw_deg: float
    duration: float
    speed: float = -1.0
    height: float = -1.0

    name = "walk"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("WalkSkill duration must be positive")

    def params(self) -> dict[str, Any]:
        return {
            "vx": self.vx,
            "vy": self.vy,
            "facing_yaw_deg": self.facing_yaw_deg,
            "duration": self.duration,
            "speed": self.speed,
            "height": self.height,
        }

    def command_at(self, elapsed_s: float, state: dict[str, Any]) -> PlannerCommand:
        del elapsed_s, state
        return PlannerCommand(
            mode=LOCOMOTION_WALK,
            movement=normalize_xy(self.vx, self.vy),
            facing=facing_from_yaw(self.facing_yaw_deg),
            speed=self.speed,
            height=self.height,
        )

