"""Metrics for no-CASA online SONIC VLN episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from .actions import VLNAction
from .maze import MazeMap
from .oracle import RobotPose2D
from .tasks import NavigationTask


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_id: str
    instruction: str
    success: bool
    failure_reason: str
    steps: int
    policy_issued_stop: bool
    stop_distance: float | None
    final_distance: float
    start_distance: float
    min_distance: float
    shortest_path_progress: float
    progress_m: float
    fall_count: int
    collision_count: int
    stuck: bool
    timeout: bool
    executed_motion_skills: int
    meaningful_motion: bool
    stop_fallback_ratio: float
    action_distribution: dict[str, int]
    stop_failure_type: str | None
    privileged_policy_usage_count: int
    video_paths: list[str]
    frames_dir: str
    decision_log: str
    trajectory_log: str
    final_stop_frame: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def distance_to_goal(task: NavigationTask, pose: RobotPose2D) -> float:
    return math.hypot(task.goal_xy[0] - pose.x, task.goal_xy[1] - pose.y)


def _pose_from_row(row: dict[str, Any]) -> RobotPose2D:
    return RobotPose2D(float(row["pose_x"]), float(row["pose_y"]), float(row["pose_yaw_deg"]))


def summarize_episode(
    *,
    task: NavigationTask,
    maze: MazeMap,
    decision_records: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    video_paths: list[Path],
    frames_dir: Path,
    decision_log: Path,
    trajectory_log: Path,
) -> EpisodeMetrics:
    poses = [_pose_from_row(row) for row in trajectory_rows]
    start_pose = poses[0] if poses else RobotPose2D(task.start_xy[0], task.start_xy[1], task.start_yaw_deg)
    final_pose = poses[-1] if poses else start_pose
    start_distance = distance_to_goal(task, start_pose)
    final_distance = distance_to_goal(task, final_pose)
    min_distance = min([distance_to_goal(task, pose) for pose in poses] or [final_distance])

    start_remaining = maze.shortest_path_distance_xy((start_pose.x, start_pose.y), task.goal_xy)
    final_remaining = maze.shortest_path_distance_xy((final_pose.x, final_pose.y), task.goal_xy)
    progress_m = max(0.0, start_remaining - final_remaining)
    shortest_path_progress = progress_m / start_remaining if math.isfinite(start_remaining) and start_remaining > 1e-6 else 0.0

    actions = [str(record["final_action"]) for record in decision_records]
    distribution = {action.value: 0 for action in VLNAction}
    distribution.update(Counter(actions))
    stop_records = [record for record in decision_records if record["final_action"] == VLNAction.STOP.value]
    policy_issued_stop = bool(stop_records)
    stop_record = stop_records[0] if stop_records else None
    stop_distance = float(stop_record["evaluator_distance_to_goal"]) if stop_record else None
    final_stop_frame = stop_record.get("image_path") if stop_record else None

    fall_count = sum(int(row.get("fall", 0)) for row in trajectory_rows)
    collision_count = sum(int(row.get("wall_collision", 0)) for row in trajectory_rows)
    displacement = math.hypot(final_pose.x - start_pose.x, final_pose.y - start_pose.y)
    executed_motion_skills = sum(1 for action in actions if action in {VLNAction.FORWARD.value, VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value, VLNAction.BACKOFF.value})
    meaningful_motion = displacement >= 0.25 and executed_motion_skills >= 2
    stop_fallback_ratio = (
        (distribution[VLNAction.STOP.value] + distribution[VLNAction.BACKOFF.value]) / max(1, len(actions))
    )
    timeout = not policy_issued_stop and len(decision_records) >= task.max_steps
    stuck = displacement < 0.10 and len(decision_records) >= 3
    privileged_policy_usage_count = sum(1 for record in decision_records if record.get("privileged_policy_usage"))

    reached_goal_at_stop = stop_distance is not None and stop_distance <= task.success_radius
    no_fall = fall_count == 0
    no_wall_collision = collision_count == 0
    sufficient_progress = shortest_path_progress >= 0.50
    bounded_stop_fallback = stop_fallback_ratio <= 0.50
    success = (
        reached_goal_at_stop
        and policy_issued_stop
        and no_fall
        and no_wall_collision
        and meaningful_motion
        and sufficient_progress
        and bounded_stop_fallback
        and privileged_policy_usage_count == 0
    )

    stop_failure_type = None
    if policy_issued_stop and not reached_goal_at_stop:
        stop_failure_type = "premature_stop"
    elif not policy_issued_stop and min_distance <= task.success_radius:
        stop_failure_type = "late_stop"
    elif distribution[VLNAction.STOP.value] > 1 and stop_fallback_ratio > 0.50:
        stop_failure_type = "infinite_stop"

    failure_reason = ""
    if not success:
        if privileged_policy_usage_count:
            failure_reason = "privileged_policy_usage"
        elif not policy_issued_stop:
            failure_reason = "missing_policy_stop"
        elif not reached_goal_at_stop:
            failure_reason = "stop_outside_goal_radius"
        elif not no_fall:
            failure_reason = "fall_detected"
        elif not no_wall_collision:
            failure_reason = "wall_collision"
        elif not meaningful_motion:
            failure_reason = "insufficient_motion"
        elif not sufficient_progress:
            failure_reason = "insufficient_shortest_path_progress"
        elif not bounded_stop_fallback:
            failure_reason = "stop_fallback_ratio_too_high"
        else:
            failure_reason = "unknown"

    return EpisodeMetrics(
        episode_id=task.episode_id,
        instruction=task.instruction,
        success=success,
        failure_reason=failure_reason,
        steps=len(decision_records),
        policy_issued_stop=policy_issued_stop,
        stop_distance=stop_distance,
        final_distance=final_distance,
        start_distance=start_distance,
        min_distance=min_distance,
        shortest_path_progress=shortest_path_progress,
        progress_m=progress_m,
        fall_count=fall_count,
        collision_count=collision_count,
        stuck=stuck,
        timeout=timeout,
        executed_motion_skills=executed_motion_skills,
        meaningful_motion=meaningful_motion,
        stop_fallback_ratio=stop_fallback_ratio,
        action_distribution=distribution,
        stop_failure_type=stop_failure_type,
        privileged_policy_usage_count=privileged_policy_usage_count,
        video_paths=[str(path) for path in video_paths],
        frames_dir=str(frames_dir),
        decision_log=str(decision_log),
        trajectory_log=str(trajectory_log),
        final_stop_frame=final_stop_frame,
    )


def aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    success_count = sum(1 for episode in episodes if episode["success"])
    action_distribution = {action.value: 0 for action in VLNAction}
    for episode in episodes:
        for action, count in episode.get("action_distribution", {}).items():
            action_distribution[action] = action_distribution.get(action, 0) + int(count)

    stop_distances = [
        float(episode["stop_distance"])
        for episode in episodes
        if episode.get("policy_issued_stop") and episode.get("stop_distance") is not None
    ]
    true_stop_positive = sum(
        1
        for episode in episodes
        if episode.get("policy_issued_stop") and (episode.get("stop_distance") or float("inf")) <= 0.8
    )
    false_stop_positive = sum(1 for episode in episodes if episode.get("stop_failure_type") == "premature_stop")
    false_stop_negative = sum(1 for episode in episodes if episode.get("stop_failure_type") == "late_stop")
    infinite_stop = sum(1 for episode in episodes if episode.get("stop_failure_type") == "infinite_stop")

    return {
        "total_episodes": total,
        "success_count": success_count,
        "success_rate": success_count / total if total else 0.0,
        "fall_count": sum(int(episode["fall_count"]) for episode in episodes),
        "collision_count": sum(int(episode["collision_count"]) for episode in episodes),
        "stuck_count": sum(1 for episode in episodes if episode["stuck"]),
        "timeout_count": sum(1 for episode in episodes if episode["timeout"]),
        "mean_final_distance": _mean([episode["final_distance"] for episode in episodes]),
        "median_final_distance": median([episode["final_distance"] for episode in episodes]) if episodes else 0.0,
        "mean_shortest_path_progress": _mean([episode["shortest_path_progress"] for episode in episodes]),
        "mean_executed_motion_skills": _mean([episode["executed_motion_skills"] for episode in episodes]),
        "stop_fallback_ratio": _mean([episode["stop_fallback_ratio"] for episode in episodes]),
        "stop_precision": true_stop_positive / max(1, true_stop_positive + false_stop_positive),
        "stop_recall": true_stop_positive / max(1, true_stop_positive + false_stop_negative),
        "premature_stop_rate": false_stop_positive / total if total else 0.0,
        "late_stop_rate": false_stop_negative / total if total else 0.0,
        "infinite_stop_rate": infinite_stop / total if total else 0.0,
        "mean_stop_distance": _mean(stop_distances),
        "privileged_policy_usage_count": sum(int(episode["privileged_policy_usage_count"]) for episode in episodes),
        "action_distribution": action_distribution,
    }


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
