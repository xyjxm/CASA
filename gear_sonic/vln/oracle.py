"""Training-only oracle teacher for no-CASA MuJoCo maze VLN."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .actions import ActionDecision, VLNAction
from .maze import Cell, MazeMap, build_route
from .tasks import TeacherTask, task_from_route


@dataclass(frozen=True)
class RobotPose2D:
    x: float
    y: float
    yaw_deg: float = 0.0


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


class AStarTeacher:
    """Offline teacher. Do not instantiate from final policy code."""

    def __init__(self, maze: MazeMap, *, stop_radius: float = 0.8) -> None:
        self.maze = maze
        self.stop_radius = stop_radius

    def build_task(self, start_cell: Cell, goal_cell: Cell, *, episode_id: str, split: str) -> TeacherTask:
        route = build_route(self.maze, start_cell, goal_cell)
        return task_from_route(route, episode_id=episode_id, split=split)

    def teacher_action_for_step(self, task: TeacherTask, step_idx: int) -> ActionDecision:
        if step_idx < len(task.route_actions):
            action = VLNAction(task.route_actions[step_idx])
            should_stop = False
            reason = "route_step"
        else:
            action = VLNAction.STOP
            should_stop = True
            reason = "near_goal_visual"
        return ActionDecision(
            action=action,
            raw_output=f"teacher:{action.value}",
            source="training_only_astar_teacher",
            metadata={
                "should_stop": should_stop,
                "stop_label_reason": reason,
                "teacher_metadata": task.teacher_metadata(),
            },
        )

    def next_action_from_pose(self, task: TeacherTask, pose: RobotPose2D) -> ActionDecision:
        distance = math.hypot(task.goal_xy[0] - pose.x, task.goal_xy[1] - pose.y)
        if distance <= self.stop_radius:
            return ActionDecision(
                VLNAction.STOP,
                "teacher:stop",
                "training_only_astar_teacher",
                metadata={"distance": distance, "stop_label_reason": "near_goal_visual"},
            )
        cell = self.maze.xy_to_cell(pose.x, pose.y)
        path = self.maze.astar(cell, task.goal_cell) if task.goal_cell else []
        metadata: dict[str, Any] = {"distance": distance, "cell": cell, "path_cells": path}
        if len(path) < 2:
            return ActionDecision(VLNAction.STOP, "teacher:stop_no_path", "training_only_astar_teacher", metadata=metadata)
        next_xy = self.maze.cell_to_xy(path[1])
        desired_yaw = math.degrees(math.atan2(next_xy[1] - pose.y, next_xy[0] - pose.x))
        yaw_error = wrap_degrees(desired_yaw - pose.yaw_deg)
        if yaw_error > 18.0:
            action = VLNAction.TURN_LEFT
        elif yaw_error < -18.0:
            action = VLNAction.TURN_RIGHT
        else:
            action = VLNAction.FORWARD
        metadata.update({"desired_yaw_deg": desired_yaw, "yaw_error_deg": yaw_error})
        return ActionDecision(action, f"teacher:{action.value}", "training_only_astar_teacher", metadata=metadata)
