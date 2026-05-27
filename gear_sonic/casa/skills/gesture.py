"""Upper-body gesture skill wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gear_sonic.casa.io.zmq_publisher import LOCOMOTION_IDLE

from .base import PlannerCommand, Skill
from .utils import coerce_facing


def gesture_upper_body(
    elapsed_s: float,
    amplitude: float,
    frequency: float,
    side: str = "both",
) -> tuple[list[float], list[float]]:
    omega = 2.0 * math.pi * frequency
    position = [0.0] * 17
    velocity = [0.0] * 17
    pitch = amplitude * math.sin(omega * elapsed_s)
    pitch_dot = amplitude * omega * math.cos(omega * elapsed_s)
    if side in {"left", "both"}:
        position[3] = pitch
        velocity[3] = pitch_dot
    if side in {"right", "both"}:
        position[4] = -pitch
        velocity[4] = -pitch_dot
    return position, velocity


@dataclass(frozen=True)
class GestureSkill(Skill):
    amplitude: float
    frequency: float
    duration: float
    side: str = "both"

    name = "gesture"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("GestureSkill duration must be positive")
        if self.frequency <= 0:
            raise ValueError("GestureSkill frequency must be positive")
        if self.side not in {"left", "right", "both"}:
            raise ValueError("GestureSkill side must be one of left, right, both")

    def params(self) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "side": self.side,
            "duration": self.duration,
        }

    def command_at(self, elapsed_s: float, state: dict[str, Any]) -> PlannerCommand:
        position, velocity = gesture_upper_body(elapsed_s, self.amplitude, self.frequency, self.side)
        return PlannerCommand(
            mode=LOCOMOTION_IDLE,
            movement=(0.0, 0.0, 0.0),
            facing=coerce_facing(state.get("last_facing")),
            upper_body_position=position,
            upper_body_velocity=velocity,
        )
