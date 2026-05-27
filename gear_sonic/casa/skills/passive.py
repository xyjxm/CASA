"""Passive stop/wait skill wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gear_sonic.casa.io.zmq_publisher import LOCOMOTION_IDLE

from .base import PlannerCommand, Skill
from .utils import coerce_facing


@dataclass(frozen=True)
class PassiveSkill(Skill):
    duration: float
    mode: str = "stop"

    name = "passive"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("PassiveSkill duration must be positive")
        if self.mode not in {"stop", "wait"}:
            raise ValueError("PassiveSkill mode must be stop or wait")

    def params(self) -> dict[str, Any]:
        return {"duration": self.duration, "mode": self.mode}

    def command_at(self, elapsed_s: float, state: dict[str, Any]) -> PlannerCommand:
        del elapsed_s
        return PlannerCommand(
            mode=LOCOMOTION_IDLE,
            movement=(0.0, 0.0, 0.0),
            facing=coerce_facing(state.get("last_facing")),
        )

