"""Base types for CASA skill execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class PlannerCommand:
    mode: int
    movement: Vector3
    facing: Vector3
    speed: float = -1.0
    height: float = -1.0
    upper_body_position: list[float] | None = None
    upper_body_velocity: list[float] | None = None


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    episode_id: str
    skill_idx: int
    skill_name: str
    params: dict[str, Any]
    start_wall_time: float
    end_wall_time: float
    start_monotonic: float
    end_monotonic: float
    estimated_duration: float
    actual_duration: float
    planner_publishes: int
    publish_ok: bool
    downstream_state_observed: bool
    status: str
    termination_reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


class SkillExecutorProtocol(Protocol):
    def execute_one(
        self,
        skill: "Skill",
        *,
        episode_id: str = "0",
        skill_idx: int | None = None,
    ) -> ExecutionResult:
        ...


class Skill:
    name = "skill"

    @property
    def estimated_duration(self) -> float:
        return float(getattr(self, "duration"))

    def params(self) -> dict[str, Any]:
        raise NotImplementedError

    def command_at(self, elapsed_s: float, state: dict[str, Any]) -> PlannerCommand:
        raise NotImplementedError

    def execute(
        self,
        executor: SkillExecutorProtocol,
        *,
        episode_id: str = "0",
        skill_idx: int | None = None,
    ) -> ExecutionResult:
        return executor.execute_one(self, episode_id=episode_id, skill_idx=skill_idx)


def monotonic_and_wall_time() -> tuple[float, float]:
    return time.monotonic(), time.time()
