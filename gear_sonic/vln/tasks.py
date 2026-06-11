"""Navigation task schemas for no-CASA MuJoCo maze VLN runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .actions import VLNAction
from .maze import Cell, MazeRoute


@dataclass(frozen=True)
class NavigationTask:
    episode_id: str
    instruction: str
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw_deg: float
    success_radius: float = 0.8
    max_steps: int = 96
    visual_cues: list[str] = field(default_factory=list)
    split: str = "heldout"
    start_cell: Cell | None = None
    goal_cell: Cell | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_xy"] = list(self.start_xy)
        data["goal_xy"] = list(self.goal_xy)
        if self.start_cell is not None:
            data["start_cell"] = list(self.start_cell)
        if self.goal_cell is not None:
            data["goal_cell"] = list(self.goal_cell)
        return data


@dataclass(frozen=True)
class TeacherTask(NavigationTask):
    route_actions: tuple[str, ...] = field(default_factory=tuple)
    path_cells: tuple[Cell, ...] = field(default_factory=tuple)
    stop_xy: tuple[float, float] | None = None

    def teacher_metadata(self) -> dict[str, Any]:
        return {
            "path_cells": [list(cell) for cell in self.path_cells],
            "route_actions": list(self.route_actions),
            "stop_xy": list(self.stop_xy) if self.stop_xy else None,
            "start_cell": list(self.start_cell) if self.start_cell else None,
            "goal_cell": list(self.goal_cell) if self.goal_cell else None,
        }

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["route_actions"] = list(self.route_actions)
        data["path_cells"] = [list(cell) for cell in self.path_cells]
        data["stop_xy"] = list(self.stop_xy) if self.stop_xy else None
        return data


def task_from_route(route: MazeRoute, *, episode_id: str, split: str, max_steps: int = 96) -> TeacherTask:
    return TeacherTask(
        episode_id=episode_id,
        instruction=route.instruction,
        start_xy=route.start_xy,
        goal_xy=route.goal_xy,
        start_yaw_deg=route.start_yaw_deg,
        success_radius=0.8,
        max_steps=max_steps,
        visual_cues=route.visual_cues,
        split=split,
        start_cell=route.start_cell,
        goal_cell=route.goal_cell,
        route_actions=tuple(action.value if isinstance(action, VLNAction) else str(action) for action in route.route_actions),
        path_cells=tuple(route.path_cells),
        stop_xy=route.stop_xy,
    )
