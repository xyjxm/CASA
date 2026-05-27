"""Turn/face skill wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gear_sonic.casa.io.zmq_publisher import LOCOMOTION_IDLE

from .base import PlannerCommand, Skill
from .utils import facing_from_yaw


@dataclass(frozen=True)
class TurnSkill(Skill):
    face_yaw_deg: float
    duration: float

    name = "turn"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("TurnSkill duration must be positive")

    def params(self) -> dict[str, Any]:
        return {"face_yaw_deg": self.face_yaw_deg, "duration": self.duration}

    def command_at(self, elapsed_s: float, state: dict[str, Any]) -> PlannerCommand:
        del elapsed_s, state
        return PlannerCommand(
            mode=LOCOMOTION_IDLE,
            movement=(0.0, 0.0, 0.0),
            facing=facing_from_yaw(self.face_yaw_deg),
        )

