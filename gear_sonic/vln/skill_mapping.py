"""Map no-CASA VLN actions onto executable high-level SONIC skill intents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .actions import ActionDecision, VLNAction


@dataclass(frozen=True)
class SonicSkill:
    name: str
    duration: float

    def params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WalkSkill(SonicSkill):
    vx: float
    vy: float
    facing_yaw_deg: float
    step_target_m: float

    def __init__(
        self,
        *,
        vx: float,
        vy: float,
        facing_yaw_deg: float,
        duration: float,
        step_target_m: float,
        name: str = "walk",
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "vx", vx)
        object.__setattr__(self, "vy", vy)
        object.__setattr__(self, "facing_yaw_deg", facing_yaw_deg)
        object.__setattr__(self, "step_target_m", step_target_m)


@dataclass(frozen=True)
class TurnSkill(SonicSkill):
    delta_yaw_deg: float
    face_yaw_deg: float

    def __init__(
        self,
        *,
        delta_yaw_deg: float,
        face_yaw_deg: float,
        duration: float,
        name: str = "turn",
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "delta_yaw_deg", delta_yaw_deg)
        object.__setattr__(self, "face_yaw_deg", face_yaw_deg)


@dataclass(frozen=True)
class PassiveSkill(SonicSkill):
    mode: str

    def __init__(self, *, duration: float, mode: str = "stop", name: str = "passive") -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class SkillMappingConfig:
    forward_duration: float = 0.7
    turn_duration: float = 0.45
    backoff_duration: float = 0.4
    stop_duration: float = 0.3
    turn_degrees: float = 30.0
    forward_step_m: float = 0.50
    backoff_step_m: float = 0.20


def _unit_from_yaw(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


class ActionSkillMapper:
    """Stateful mapper for relative action enums."""

    def __init__(self, config: SkillMappingConfig | None = None, initial_yaw_deg: float = 0.0) -> None:
        self.config = config or SkillMappingConfig()
        self.heading_yaw_deg = float(initial_yaw_deg)

    def reset(self, initial_yaw_deg: float = 0.0) -> None:
        self.heading_yaw_deg = float(initial_yaw_deg)

    def update_heading(self, yaw_deg: float | None) -> None:
        if yaw_deg is not None and math.isfinite(yaw_deg):
            self.heading_yaw_deg = float(yaw_deg)

    def action_to_skill(self, decision: ActionDecision | VLNAction) -> SonicSkill:
        action = decision.action if isinstance(decision, ActionDecision) else decision
        cfg = self.config

        if action is VLNAction.TURN_LEFT:
            self.heading_yaw_deg = wrap_degrees(self.heading_yaw_deg + cfg.turn_degrees)
            return TurnSkill(
                delta_yaw_deg=cfg.turn_degrees,
                face_yaw_deg=self.heading_yaw_deg,
                duration=cfg.turn_duration,
            )
        if action is VLNAction.TURN_RIGHT:
            self.heading_yaw_deg = wrap_degrees(self.heading_yaw_deg - cfg.turn_degrees)
            return TurnSkill(
                delta_yaw_deg=-cfg.turn_degrees,
                face_yaw_deg=self.heading_yaw_deg,
                duration=cfg.turn_duration,
            )
        if action is VLNAction.FORWARD:
            vx, vy = _unit_from_yaw(self.heading_yaw_deg)
            return WalkSkill(
                vx=vx,
                vy=vy,
                facing_yaw_deg=self.heading_yaw_deg,
                duration=cfg.forward_duration,
                step_target_m=cfg.forward_step_m,
            )
        if action is VLNAction.BACKOFF:
            vx, vy = _unit_from_yaw(self.heading_yaw_deg + 180.0)
            return WalkSkill(
                vx=vx,
                vy=vy,
                facing_yaw_deg=self.heading_yaw_deg,
                duration=cfg.backoff_duration,
                step_target_m=cfg.backoff_step_m,
                name="backoff_walk",
            )
        if action is VLNAction.STOP:
            return PassiveSkill(duration=cfg.stop_duration, mode="stop")
        raise ValueError(f"unsupported VLNAction: {action}")
