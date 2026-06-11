from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gear_sonic.vln.actions import VLNAction
from gear_sonic.vln.casa_bridge import (
    CASA_SKILL_BY_VLN_ACTION,
    CasaGateDecision,
    RecoveryCandidate,
    WallRiskFeatures,
    compute_wall_risk_features,
    is_policy_stop_source,
    policy_internal_guard_summary,
)
from gear_sonic.vln.casa_runner import (
    _metrics_action,
    choose_replan_candidate,
    skill_from_action,
    summarize_casa_episode,
)
from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.no_casa_runner import MAP_METADATA
from gear_sonic.vln.oracle import RobotPose2D
from gear_sonic.vln.skill_mapping import PassiveSkill, TurnSkill, WalkSkill
from gear_sonic.vln.tasks import TeacherTask


def test_vln_to_casa_skill_mapping_keeps_backoff_as_walk_threshold() -> None:
    assert CASA_SKILL_BY_VLN_ACTION[VLNAction.FORWARD] == "walk"
    assert CASA_SKILL_BY_VLN_ACTION[VLNAction.BACKOFF] == "walk"
    assert CASA_SKILL_BY_VLN_ACTION[VLNAction.TURN_LEFT] == "turn"
    assert CASA_SKILL_BY_VLN_ACTION[VLNAction.TURN_RIGHT] == "turn"
    assert CASA_SKILL_BY_VLN_ACTION[VLNAction.STOP] == "passive"

    backoff = skill_from_action(VLNAction.BACKOFF, 0.0)
    assert isinstance(backoff, WalkSkill)
    assert backoff.name == "backoff_walk"
    assert _metrics_action("short_forward_segment") == "forward"
    assert _metrics_action("stop_as_last_resort") == "stop"


def test_wall_risk_features_detect_blocking_forward_candidate() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose, skill = _find_forward_into_wall_case(maze)
    features = compute_wall_risk_features(skill, pose=pose, maze=maze, robot_radius=0.28)

    as_dict = features.to_dict()
    assert "vln/candidate_min_clearance" in as_dict
    assert "vln/candidate_would_block" in as_dict
    assert "env/min_obstacle_distance/current" in as_dict
    assert "env/external_collision_obstacle/current" in as_dict
    assert features.candidate_would_block is True
    assert features.candidate_min_clearance < 0.10


def test_stop_sources_separate_policy_success_from_casa_safety_stop() -> None:
    assert is_policy_stop_source("vln_policy")
    assert is_policy_stop_source("policy_internal_guard")
    assert not is_policy_stop_source("casa_reject_only_stop")
    assert not is_policy_stop_source("casa_replan_last_resort_stop")

    applied, names = policy_internal_guard_summary({"blocked_forward_backoff_guard_applied": True})
    assert applied is True
    assert names == "blocked_forward_backoff_guard_applied"


def test_replan_candidate_selection_uses_allowed_non_stop_before_last_resort() -> None:
    bridge = _FakeBridge({"backoff": True, "turn_left": True, "turn_right": False, "short_forward_segment": True})
    pose = RobotPose2D(0.0, 0.0, 0.0)
    selected, rows = choose_replan_candidate(
        bridge=bridge,
        task=_task(),
        method="vln_casa_replan",
        step_idx=0,
        pose=pose,
        maze=MazeMap.from_metadata(MAP_METADATA),
        robot_radius=0.28,
    )

    assert selected.action == "turn_right"
    assert selected.stop_as_last_resort is False
    assert any(row["candidate_action"] == "stop_as_last_resort" for row in rows)
    assert sum(int(row["selected_by_replan"]) for row in rows) == 1


def test_casa_stop_does_not_count_as_policy_success_even_at_goal() -> None:
    task = _task()
    trajectory = [
        {"pose_x": 0.0, "pose_y": 0.0, "pose_yaw_deg": 0.0, "wall_collision": 0, "fall": 0},
        {"pose_x": 1.0, "pose_y": 0.0, "pose_yaw_deg": 0.0, "wall_collision": 0, "fall": 0},
    ]
    metrics = summarize_casa_episode(
        method="vln_casa_reject_only",
        task=task,
        maze=MazeMap.from_metadata(MAP_METADATA),
        decision_records=[
            {
                "metrics_action": "forward",
                "evaluator_distance_to_goal": 1.0,
                "privileged_policy_usage": False,
            },
            {
                "metrics_action": "stop",
                "stop_source": "casa_reject_only_stop",
                "evaluator_distance_to_goal": 0.0,
                "privileged_policy_usage": False,
            },
        ],
        trajectory_rows=trajectory,
        video_paths=[],
        frames_dir=Path("frames"),
        decision_log=Path("decision.jsonl"),
        gate_log=Path("gate.jsonl"),
        replan_log=Path("replan.jsonl"),
        trajectory_log=Path("trajectory.jsonl"),
    )

    assert metrics["policy_issued_stop"] is False
    assert metrics["success"] is False
    assert metrics["failure_reason"] == "missing_policy_stop"
    assert metrics["casa_stop_count"] == 1


def _find_forward_into_wall_case(maze: MazeMap) -> tuple[RobotPose2D, WalkSkill]:
    directions = [
        ((0, 1), 0.0),
        ((-1, 0), 90.0),
        ((0, -1), 180.0),
        ((1, 0), -90.0),
    ]
    for cell in maze.free_cells:
        x, y = maze.cell_to_xy(cell)
        for (dr, dc), yaw in directions:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if neighbor in maze.blocking_cells():
                vx, vy = _unit_from_yaw(yaw)
                skill = WalkSkill(vx=vx, vy=vy, facing_yaw_deg=yaw, duration=0.7, step_target_m=1.5)
                pose = RobotPose2D(x, y, yaw)
                if compute_wall_risk_features(skill, pose=pose, maze=maze, robot_radius=0.28).candidate_would_block:
                    return pose, skill
    raise AssertionError("could not find a free-cell candidate pointing into a wall")


def _unit_from_yaw(degrees: float) -> tuple[float, float]:
    import math

    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def _task() -> TeacherTask:
    return TeacherTask(
        episode_id="heldout_test",
        instruction="Follow the corridor and stop at the gallery wall painting.",
        start_xy=(0.0, 0.0),
        goal_xy=(1.0, 0.0),
        start_yaw_deg=0.0,
        success_radius=0.8,
        max_steps=8,
        route_actions=("forward", "stop"),
        path_cells=((0, 0), (0, 1)),
        stop_xy=(1.0, 0.0),
    )


@dataclass
class _FakeBridge:
    reject_by_action: dict[str, bool]

    def recovery_candidates(self, *, pose: RobotPose2D) -> list[RecoveryCandidate]:
        del pose
        return [
            RecoveryCandidate("backoff", WalkSkill(vx=-1.0, vy=0.0, facing_yaw_deg=0.0, duration=0.2, step_target_m=0.1, name="backoff_walk"), "walk", 0),
            RecoveryCandidate("turn_left", TurnSkill(delta_yaw_deg=30.0, face_yaw_deg=30.0, duration=0.2), "turn", 1),
            RecoveryCandidate("turn_right", TurnSkill(delta_yaw_deg=-30.0, face_yaw_deg=-30.0, duration=0.2), "turn", 2),
            RecoveryCandidate("short_forward_segment", WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=0.0, duration=0.2, step_target_m=0.1), "walk", 3),
            RecoveryCandidate("stop_as_last_resort", PassiveSkill(duration=0.2, mode="stop"), "passive", 4, True),
        ]

    def decide(self, *, action, skill, pose, maze, robot_radius) -> CasaGateDecision:
        del skill, pose, maze, robot_radius
        action_value = action.value
        if action_value == "forward":
            action_value = "short_forward_segment"
        reject = bool(self.reject_by_action.get(action_value, False))
        return CasaGateDecision(
            raw_risk={"backoff": 0.8, "turn_left": 0.7, "turn_right": 0.2, "short_forward_segment": 0.9, "stop": 0.0}.get(action_value, 0.1),
            threshold=0.5,
            risk_margin=0.1 if reject else -0.1,
            hard_contract_score=1.0 if reject else 0.0,
            hard_contract_fixed_reject=reject,
            casa_reject=reject,
            reject_reason="hard_contract_or_casa_threshold" if reject else "allow",
            candidate_casa_skill_name="passive" if action_value == "stop" else "walk",
            wall_features=WallRiskFeatures(
                candidate_min_clearance=-0.1 if reject else 1.0,
                candidate_would_block=reject,
                current_min_obstacle_distance=1.0,
                current_external_collision_obstacle=False,
                nearest_obstacle_rel_x=0.0,
                nearest_obstacle_rel_y=0.0,
                sampled_points=1,
            ),
            risk_source="real_casa_critic",
        )
