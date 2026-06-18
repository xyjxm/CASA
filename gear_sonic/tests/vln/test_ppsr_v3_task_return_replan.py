from __future__ import annotations

from pathlib import Path

from PIL import Image

from gear_sonic.vln.actions import VLNAction
from gear_sonic.vln.casa_bridge import is_policy_stop_source
from gear_sonic.vln.casa_runner import METHODS, aggregate_method, audit_final_status, method_list
from gear_sonic.vln.dmps_mpc_cbf_replan import (
    DMPS_METHOD_NAME,
    PPSR_V2_METHOD_NAME,
    PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE,
    PPSR_V3_LAST_RESORT_STOP_SOURCE,
    PPSR_V3_METHOD_NAME,
    build_ppsr_v2_candidate_sequences,
    build_ppsr_v3_candidate_sequences,
    is_ppsr_v2_method,
    is_ppsr_v3_method,
    rollout_candidate_sequence,
    select_ppsr_v3_replan,
    selected_replan_row,
)
from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.no_casa_runner import MAP_METADATA
from gear_sonic.vln.oracle import RobotPose2D
from gear_sonic.vln.progress_monitor import ProgressMonitor


def test_ppsr_v3_method_registration_is_independent() -> None:
    assert PPSR_V3_METHOD_NAME in METHODS
    assert method_list(PPSR_V3_METHOD_NAME) == (PPSR_V3_METHOD_NAME,)
    assert method_list("vln_task_return_replan") == (PPSR_V3_METHOD_NAME,)
    assert is_ppsr_v3_method(PPSR_V3_METHOD_NAME)
    assert not is_ppsr_v2_method(PPSR_V3_METHOD_NAME)


def test_ppsr_v3_does_not_change_v2_candidate_generation() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose = _find_safe_forward_pose(maze)
    monitor = ProgressMonitor(episode_id="ep")
    candidates, signal = build_ppsr_v2_candidate_sequences(
        pose=pose,
        nominal_action=VLNAction.FORWARD,
        progress_monitor=monitor,
    )
    assert signal["stuck_mode"] is False
    assert {candidate.sequence_id for candidate in candidates} >= {
        "short_forward",
        "backoff",
        "turn_left",
        "turn_right",
        "stop_as_last_resort",
    }
    assert not any("task_return" in candidate.sequence_id for candidate in candidates)


def test_ppsr_v3_candidate_generation_includes_translating_escape_sequences() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose = _find_safe_forward_pose(maze)
    candidates, signal = build_ppsr_v3_candidate_sequences(
        pose=pose,
        nominal_action=VLNAction.FORWARD,
        progress_monitor=ProgressMonitor(episode_id="ep"),
    )
    non_stop = [candidate for candidate in candidates if not candidate.stop_as_last_resort]
    assert signal["horizon_used"] >= 2
    assert all(2 <= len(candidate.actions) <= 5 for candidate in non_stop)
    assert {candidate.sequence_id for candidate in non_stop} >= {
        "backoff_wide_turn_left_forward",
        "backoff_wide_turn_right_forward",
        "wide_arc_left_forward",
        "wide_arc_right_forward",
        "wall_follow_left_forward",
        "wall_follow_right_forward",
        "open_space_seek_forward",
        "target_reacquire_left_forward",
    }
    assert all(any(action in candidate.actions for action in {"backoff", "short_forward", "open_space_seek", "wide_arc_left", "wide_arc_right", "wall_follow_left", "wall_follow_right"}) for candidate in non_stop)


def test_ppsr_v3_unsafe_candidate_sequences_are_filtered(tmp_path: Path) -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose = _find_pose_with_unsafe_v3_candidate(maze)
    selection = _select_v3(tmp_path, pose=pose)
    unsafe_scores = [
        score
        for score in selection.candidate_scores
        if not score.safety_feasible and score.candidate_sequence[0] != "stop_as_last_resort"
    ]
    assert unsafe_scores
    assert selection.selected_score.safety_feasible
    assert selection.selected_score.candidate_sequence[0] != "stop_as_last_resort"


def test_ppsr_v3_policy_ready_score_prefers_next_vln_action_allowed(tmp_path: Path) -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose = _find_pose_with_mixed_next_allowed(maze)
    selection = _select_v3(tmp_path, pose=pose)
    allowed_scores = [score.total_score for score in selection.candidate_scores if score.next_vln_action_allowed]
    blocked_scores = [
        score.total_score
        for score in selection.candidate_scores
        if not score.next_vln_action_allowed and score.candidate_sequence[0] != "stop_as_last_resort"
    ]
    assert allowed_scores and blocked_scores
    assert max(allowed_scores) > max(blocked_scores)
    assert selection.selected_score.next_vln_action_allowed


def test_ppsr_v3_loop_penalty_suppresses_repeated_turn_backoff_loops(tmp_path: Path) -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose = _find_safe_forward_pose(maze)
    monitor = ProgressMonitor(episode_id="ep")
    for idx, action in enumerate(["turn_left", "turn_right", "turn_left", "turn_right", "backoff", "backoff"]):
        monitor.record_step(
            step_idx=idx,
            executed_action=action,
            skill_status="ok",
            stop_source="",
            selected_recovery_action=action,
        )
    selection = _select_v3(tmp_path, pose=pose, monitor=monitor)
    loop_scores = [score for score in selection.candidate_scores if score.recent_loop_flag]
    assert loop_scores
    assert any(score.loop_penalty > 0.0 for score in loop_scores)
    assert not selection.selected_score.turn_only_sequence


def test_ppsr_v3_safety_and_recovery_stop_cannot_count_as_success() -> None:
    assert not is_policy_stop_source(PPSR_V3_LAST_RESORT_STOP_SOURCE)
    assert not is_policy_stop_source(PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE)
    status = audit_final_status(
        [
            _summary(DMPS_METHOD_NAME, safe_success_rate=0.30, loop_rate=0.40),
            _summary(PPSR_V2_METHOD_NAME, safe_success_rate=0.30, loop_rate=0.40),
            _summary(
                PPSR_V3_METHOD_NAME,
                safe_success_rate=0.40,
                loop_rate=0.10,
                reject_to_policy_ready_rate=0.50,
                next_vln_action_allowed_rate=0.50,
                recovery_stop_success_leakage_count=1,
            ),
        ],
        _minimal_audits(v3_recovery_leak=1),
    )
    assert status == "FAILED_STOP_LEAKAGE"


def test_ppsr_v3_online_audit_excludes_privileged_inputs(tmp_path: Path) -> None:
    selection = _select_v3(tmp_path)
    assert all(score.v3_no_privileged_online_inputs for score in selection.candidate_scores)
    assert all(score.next_vln_action_source == "nominal_action_requery_proxy" for score in selection.candidate_scores)


def test_ppsr_v3_selected_replan_row_logs_required_policy_ready_fields(tmp_path: Path) -> None:
    selection = _select_v3(tmp_path)
    row = selected_replan_row(
        method=PPSR_V3_METHOD_NAME,
        episode_id="ep",
        step_idx=2,
        nominal_action="forward",
        reject_reason="test",
        selection=selection,
        executed_first_skill=selection.selected_sequence.skills[0],
        stop_source="",
        post_reject_progress_evaluator_only=0.0,
        committed_second_action="wide_turn_left",
        committed_second_step_executed=True,
        second_action_safety_feasible=True,
        max_commit_steps=2,
    )
    required = {
        "terminal_front_clearance",
        "terminal_side_clearance",
        "next_vln_action",
        "next_vln_action_allowed",
        "predicted_reject_drop",
        "recent_loop_flag",
        "visual_novelty",
        "landmark_retained_or_reacquired",
        "policy_ready_score",
        "language_progress_score",
        "v3_no_privileged_online_inputs",
    }
    assert required.issubset(row)


def test_ppsr_v3_final_status_detects_dev_improved_but_locked_no_improvement() -> None:
    current_locked = [
        _summary(DMPS_METHOD_NAME, safe_success_rate=0.30, loop_rate=0.40),
        _summary(PPSR_V2_METHOD_NAME, safe_success_rate=0.35, loop_rate=0.40),
        _summary(
            PPSR_V3_METHOD_NAME,
            safe_success_rate=0.35,
            loop_rate=0.20,
            reject_to_policy_ready_rate=0.30,
            next_vln_action_allowed_rate=0.30,
        ),
    ]
    dev_improved = [
        _summary(DMPS_METHOD_NAME, safe_success_rate=0.30, loop_rate=0.40),
        _summary(PPSR_V2_METHOD_NAME, safe_success_rate=0.35, loop_rate=0.40),
        _summary(
            PPSR_V3_METHOD_NAME,
            safe_success_rate=0.45,
            loop_rate=0.10,
            reject_to_policy_ready_rate=0.50,
            next_vln_action_allowed_rate=0.50,
        ),
    ]
    audits = _minimal_audits()
    audits["ppsr_v3_dev_method_summary"] = dev_improved
    audits["ppsr_v3_locked_method_summary"] = current_locked
    assert audit_final_status(current_locked, audits) == "DEV_IMPROVED_BUT_LOCKED_NO_IMPROVEMENT"


def test_ppsr_v3_required_metrics_aggregate_from_episode_rows() -> None:
    summary = aggregate_method(
        [
            {
                "method": PPSR_V3_METHOD_NAME,
                "success": True,
                "safe_success": True,
                "unsafe_violation": False,
                "steps": 4,
                "casa_reject_count": 2,
                "casa_replan_count": 2,
                "unsafe_wall_collision_count": 0,
                "collision_count": 0,
                "fall_count": 0,
                "wall_contact_steps": 0,
                "stuck": False,
                "timeout": False,
                "final_distance": 0.2,
                "shortest_path_progress": 0.8,
                "executed_motion_skills": 4,
                "stop_fallback_ratio": 0.0,
                "policy_stop_count": 1,
                "last_resort_stop_count": 0,
                "non_stop_recovery_count": 2,
                "fallback_to_stop_count": 0,
                "dmps_recovery_stuck_count": 0,
                "ppsr_v2_replan_count": 0,
                "ppsr_v3_replan_count": 2,
                "ppsr_v3_repeated_replan_count": 1,
                "escape_macro_selected_count": 0,
                "escape_macro_commitment_count": 0,
                "escape_macro_commitment_success_count": 0,
                "turn_only_recovery_count": 0,
                "small_turn_recovery_count": 0,
                "post_reject_clearance_gain": 0.2,
                "post_reject_forward_reenabled_count": 2,
                "post_reject_unblocked_count": 2,
                "post_reject_visual_novelty_count": 2,
                "reject_to_policy_ready_count": 1,
                "next_vln_action_allowed_count": 2,
                "repeat_reject_count": 1,
                "loop_count": 0,
                "mean_steps_after_recovery_before_next_reject": 2.0,
                "landmark_retention_or_reacquisition_count": 2,
                "safety_stop_success_leakage_count": 0,
                "recovery_stop_success_leakage_count": 0,
                "readiness_replan_count": 2,
                "mean_post_reject_progress": 0.1,
                "rejected_candidate_skill_executed_count": 0,
                "policy_internal_guard_steps": 0,
                "action_distribution": {"forward": 3, "stop": 1},
                "recovery_action_distribution": {"backoff": 1, "wide_arc_left": 1},
                "stop_source_distribution": {"vln_policy": 1},
                "privileged_policy_usage_count": 0,
            }
        ]
    )
    assert summary["reject_to_policy_ready_rate"] == 0.5
    assert summary["next_vln_action_allowed_rate"] == 1.0
    assert summary["repeat_reject_rate"] == 0.5
    assert summary["landmark_retention_or_reacquisition_rate"] == 1.0
    assert summary["safety_stop_success_leakage_count"] == 0
    assert summary["recovery_stop_success_leakage_count"] == 0


def _select_v3(tmp_path: Path, pose: RobotPose2D | None = None, monitor: ProgressMonitor | None = None):
    image = tmp_path / "frame.png"
    Image.new("RGB", (96, 72), (180, 180, 180)).save(image)
    maze = MazeMap.from_metadata(MAP_METADATA)
    return select_ppsr_v3_replan(
        bridge=None,
        pose=pose or _find_safe_forward_pose(maze),
        maze=maze,
        robot_radius=0.28,
        nominal_action=VLNAction.FORWARD,
        image_path=image,
        progress_monitor=monitor or ProgressMonitor(episode_id="ep"),
        step_idx=2,
        horizon=5,
        safety_margin=0.05,
    )


def _find_safe_forward_pose(maze: MazeMap) -> RobotPose2D:
    for cell in maze.free_cells:
        x, y = maze.cell_to_xy(cell)
        if maze.is_xy_safe(x + 0.16, y, robot_radius=0.28):
            return RobotPose2D(x, y, 0.0)
    raise AssertionError("no safe forward pose found")


def _find_pose_with_unsafe_v3_candidate(maze: MazeMap) -> RobotPose2D:
    for cell in maze.free_cells:
        cx, cy = maze.cell_to_xy(cell)
        for dx in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
            for dy in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
                x, y = cx + dx, cy + dy
                if not maze.is_xy_safe(x, y, robot_radius=0.28):
                    continue
                for yaw in range(-180, 180, 15):
                    pose = RobotPose2D(x, y, float(yaw))
                    candidates, _ = build_ppsr_v3_candidate_sequences(
                        pose=pose,
                        nominal_action=VLNAction.FORWARD,
                        progress_monitor=ProgressMonitor(episode_id="ep"),
                    )
                    rollouts = [
                        rollout_candidate_sequence(
                            candidate=candidate,
                            start_pose=pose,
                            maze=maze,
                            robot_radius=0.28,
                            safety_margin=0.05,
                        )
                        for candidate in candidates
                        if not candidate.stop_as_last_resort
                    ]
                    if any(not rollout.safety_feasible for rollout in rollouts) and any(
                        rollout.safety_feasible for rollout in rollouts
                    ):
                        return pose
    raise AssertionError("no pose with mixed v3 safety feasibility found")


def _find_pose_with_mixed_next_allowed(maze: MazeMap) -> RobotPose2D:
    for cell in maze.free_cells:
        cx, cy = maze.cell_to_xy(cell)
        for dx in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
            for dy in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
                x, y = cx + dx, cy + dy
                if not maze.is_xy_safe(x, y, robot_radius=0.28):
                    continue
                for yaw in range(-180, 180, 15):
                    pose = RobotPose2D(x, y, float(yaw))
                    selection = select_ppsr_v3_replan(
                        bridge=None,
                        pose=pose,
                        maze=maze,
                        robot_radius=0.28,
                        nominal_action=VLNAction.FORWARD,
                        image_path=None,
                        progress_monitor=ProgressMonitor(episode_id="ep"),
                        step_idx=0,
                        horizon=5,
                        safety_margin=0.05,
                    )
                    allowed = [
                        score
                        for score in selection.candidate_scores
                        if score.next_vln_action_allowed and score.candidate_sequence[0] != "stop_as_last_resort"
                    ]
                    blocked = [
                        score
                        for score in selection.candidate_scores
                        if not score.next_vln_action_allowed and score.candidate_sequence[0] != "stop_as_last_resort"
                    ]
                    if allowed and blocked:
                        return pose
    raise AssertionError("no pose with mixed next-action allowed scores found")


def _summary(method: str, **overrides):
    base = {
        "method": method,
        "safe_success_rate": 0.30,
        "success_rate": 0.30,
        "unsafe_violation_rate": 0.0,
        "reject_to_policy_ready_rate": 0.0,
        "next_vln_action_allowed_rate": 0.0,
        "loop_rate": 0.40,
        "safety_stop_success_leakage_count": 0,
        "recovery_stop_success_leakage_count": 0,
    }
    base.update(overrides)
    return base


def _minimal_audits(*, v3_safety_leak: int = 0, v3_recovery_leak: int = 0):
    return {
        "casa_vln_bridge_audit": {"loaded_real_casa_artifacts": True},
        "privileged_leakage_audit": {
            "used_goal_distance_for_online_replan": False,
            "used_shortest_path_for_online_replan": False,
            "used_a_star_for_online_replan": False,
            "used_oracle_waypoint_for_online_replan": False,
            "used_evaluator_success_for_online_replan": False,
            "used_map_cell_progress_for_online_replan": False,
            "privileged_online_leakage": False,
        },
        "stop_source_audit": {"casa_stop_rows_counted_as_policy_stop": 0},
        "v3_stop_source_audit": {
            "safety_stop_success_leakage_count": v3_safety_leak,
            "recovery_stop_success_leakage_count": v3_recovery_leak,
        },
        "readiness_progress_metrics": {"post_reject_clearance_gain": 0.1},
    }
