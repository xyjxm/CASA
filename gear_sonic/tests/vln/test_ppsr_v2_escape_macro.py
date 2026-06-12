from __future__ import annotations

from pathlib import Path

from PIL import Image

from gear_sonic.vln.actions import VLNAction
from gear_sonic.vln.backends import NaVidBackend
from gear_sonic.vln.casa_bridge import is_policy_stop_source
from gear_sonic.vln.casa_runner import aggregate_method, audit_final_status
from gear_sonic.vln.dmps_mpc_cbf_replan import (
    DMPS_METHOD_NAME,
    PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE,
    PPSR_V2_LAST_RESORT_STOP_SOURCE,
    PPSR_V2_METHOD_NAME,
    build_ppsr_v2_candidate_sequences,
    select_ppsr_v2_replan,
    selected_replan_row,
    terminal_short_forward_probe,
)
from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.no_casa_runner import MAP_METADATA
from gear_sonic.vln.oracle import RobotPose2D
from gear_sonic.vln.progress_monitor import ProgressMonitor


def test_ppsr_v2_generates_escape_macros_only_in_stuck_mode() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose = _find_safe_forward_pose(maze)
    monitor = ProgressMonitor(episode_id="ep")
    normal_candidates, normal_signal = build_ppsr_v2_candidate_sequences(
        pose=pose,
        nominal_action=VLNAction.FORWARD,
        progress_monitor=monitor,
    )
    assert normal_signal["stuck_mode"] is False
    assert not any(candidate.escape_macro for candidate in normal_candidates)

    monitor.record_reject(step_idx=0, nominal_action="forward", reject_reason="test")
    monitor.record_reject(step_idx=1, nominal_action="forward", reject_reason="test")
    stuck_candidates, stuck_signal = build_ppsr_v2_candidate_sequences(
        pose=pose,
        nominal_action=VLNAction.FORWARD,
        progress_monitor=monitor,
    )
    assert stuck_signal["stuck_mode"] is True
    assert {candidate.sequence_id for candidate in stuck_candidates if candidate.escape_macro} >= {
        "escape_left",
        "escape_right",
        "wide_turn_left_forward",
        "wide_turn_right_forward",
    }


def test_ppsr_v2_hard_mask_blocks_turn_only_after_turn_loop(tmp_path: Path) -> None:
    selection = _select_with_monitor(tmp_path, _turn_loop_monitor())
    masked_turn_only = [
        score
        for score in selection.candidate_scores
        if score.turn_only_sequence and score.candidate_sequence[0] != "stop_as_last_resort"
    ]
    assert masked_turn_only
    assert all(score.hard_mask_applied for score in masked_turn_only)
    assert any(score.repeated_turn_loop_blocked for score in masked_turn_only)


def test_ppsr_v2_translation_required_after_two_turn_recoveries(tmp_path: Path) -> None:
    monitor = ProgressMonitor(episode_id="ep")
    monitor.record_step(step_idx=0, executed_action="turn_left", skill_status="ok", stop_source="", selected_recovery_action="turn_left")
    monitor.record_step(step_idx=1, executed_action="turn_right", skill_status="ok", stop_source="", selected_recovery_action="turn_right")
    selection = _select_with_monitor(tmp_path, monitor)
    blocked = [
        score
        for score in selection.candidate_scores
        if score.translation_required and score.turn_only_candidate_blocked
    ]
    assert blocked
    assert all(not score.contains_translation for score in blocked)
    assert selection.safe_translation_candidate_exists is True


def test_ppsr_v2_terminal_probe_and_scoring_fields_exist(tmp_path: Path) -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose = _find_safe_forward_pose(maze)
    probe = terminal_short_forward_probe(pose=pose, maze=maze, robot_radius=0.28, safety_margin=0.05)
    assert set(probe) == {
        "terminal_can_short_forward",
        "terminal_short_forward_min_clearance",
        "terminal_short_forward_would_block",
    }
    selection = _select_with_monitor(tmp_path, _stuck_monitor())
    assert any(score.terminal_can_short_forward for score in selection.candidate_scores)
    assert any(score.escape_macro_candidate for score in selection.candidate_scores)
    assert all(hasattr(score, "clearance_gain") for score in selection.candidate_scores)


def test_ppsr_v2_stuck_selection_prefers_escape_macro_when_safe(tmp_path: Path) -> None:
    selection = _select_with_monitor(tmp_path, _stuck_monitor())
    assert any(score.escape_macro_candidate and score.safety_feasible for score in selection.candidate_scores)
    assert selection.selected_sequence.escape_macro is True
    assert selection.selected_score.contains_translation is True


def test_ppsr_v2_selected_replan_row_records_commitment_and_abort(tmp_path: Path) -> None:
    selection = _select_with_monitor(tmp_path, _stuck_monitor())
    row = selected_replan_row(
        method=PPSR_V2_METHOD_NAME,
        episode_id="ep",
        step_idx=3,
        nominal_action="forward",
        reject_reason="test",
        selection=selection,
        executed_first_skill=selection.selected_sequence.skills[0],
        stop_source="",
        post_reject_progress_evaluator_only=0.0,
        committed_second_action="short_forward",
        committed_second_step_executed=True,
        committed_second_step_aborted=False,
        second_action_safety_feasible=True,
        abort_reason="",
        max_commit_steps=2,
    )
    assert row["committed_second_step_executed"] == 1
    assert row["control_returned_to_vln"] == 1
    assert row["commitment_used_goal_or_astar"] == 0

    aborted = dict(row)
    aborted.update({"committed_second_step_executed": 0, "committed_second_step_aborted": 1, "abort_reason": "would_block"})
    assert aborted["committed_second_step_aborted"] == 1
    assert aborted["abort_reason"] == "would_block"


def test_ppsr_v2_stop_sources_never_count_as_policy_success() -> None:
    assert not is_policy_stop_source(PPSR_V2_LAST_RESORT_STOP_SOURCE)
    assert not is_policy_stop_source(PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE)


def test_navid_backend_records_explicit_cuda_visible_devices(tmp_path: Path) -> None:
    repo = tmp_path / "navid_repo"
    repo.mkdir()
    (repo / "agent_navid.py").write_text("# probe only\n")
    (repo / "navid").mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "pytorch_model.bin.index.json").write_text('{"weight_map": {"a": "pytorch_model-00001-of-00001.bin"}}')
    (model / "pytorch_model-00001-of-00001.bin").write_bytes(b"x")
    vision = tmp_path / "eva_vit_g.pth"
    vision.write_bytes(b"x")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    backend = NaVidBackend(
        repo_path=repo,
        result_dir=tmp_path / "probe",
        model_path=model,
        python_executable=python,
        vision_tower_path=vision,
        strict_model=False,
        cuda_visible_devices="GPU-test-uuid",
    )
    assert backend.availability["cuda_visible_devices"] == "GPU-test-uuid"


def test_ppsr_v2_final_status_detects_safety_regression() -> None:
    audits = _minimal_audits()
    status = audit_final_status(
        [
            _summary(DMPS_METHOD_NAME, safe_success_rate=0.30, unsafe_violation_rate=0.0),
            _summary(PPSR_V2_METHOD_NAME, safe_success_rate=0.35, unsafe_violation_rate=0.20),
        ],
        audits,
    )
    assert status == "FAILED_SAFETY_REGRESSION"


def test_ppsr_v2_final_status_requires_escape_macro_use() -> None:
    audits = _minimal_audits()
    status = audit_final_status(
        [
            _summary(DMPS_METHOD_NAME, safe_success_rate=0.30, unsafe_violation_rate=0.0),
            _summary(PPSR_V2_METHOD_NAME, safe_success_rate=0.35, unsafe_violation_rate=0.0, escape_macro_selected_count=0),
        ],
        audits,
    )
    assert status == "FAILED_ESCAPE_MACRO_NOT_USED"


def test_ppsr_v2_readiness_metrics_aggregate_from_episode_rows() -> None:
    episodes = [
        {
            "method": PPSR_V2_METHOD_NAME,
            "success": False,
            "safe_success": False,
            "unsafe_violation": False,
            "steps": 2,
            "casa_reject_count": 1,
            "casa_replan_count": 1,
            "unsafe_wall_collision_count": 0,
            "collision_count": 0,
            "fall_count": 0,
            "wall_contact_steps": 0,
            "stuck": False,
            "timeout": True,
            "final_distance": 2.0,
            "shortest_path_progress": 0.1,
            "executed_motion_skills": 2,
            "stop_fallback_ratio": 0.0,
            "policy_stop_count": 0,
            "last_resort_stop_count": 0,
            "non_stop_recovery_count": 1,
            "fallback_to_stop_count": 0,
            "dmps_recovery_stuck_count": 1,
            "ppsr_v2_replan_count": 1,
            "escape_macro_selected_count": 1,
            "escape_macro_commitment_count": 1,
            "escape_macro_commitment_success_count": 1,
            "turn_only_recovery_count": 0,
            "small_turn_recovery_count": 0,
            "post_reject_clearance_gain": 0.2,
            "post_reject_forward_reenabled_count": 1,
            "post_reject_unblocked_count": 1,
            "post_reject_visual_novelty_count": 1,
            "mean_post_reject_progress": 0.0,
            "rejected_candidate_skill_executed_count": 0,
            "policy_internal_guard_steps": 0,
            "action_distribution": {"forward": 1},
            "recovery_action_distribution": {"backoff": 1},
            "stop_source_distribution": {},
            "privileged_policy_usage_count": 0,
        }
    ]
    summary = aggregate_method(episodes)
    assert summary["escape_macro_selected_count"] == 1
    assert summary["post_reject_forward_reenabled_rate"] == 1.0
    assert summary["small_turn_recovery_rate"] == 0.0


def _select_with_monitor(tmp_path: Path, monitor: ProgressMonitor):
    image = tmp_path / "frame.png"
    Image.new("RGB", (96, 72), (180, 180, 180)).save(image)
    maze = MazeMap.from_metadata(MAP_METADATA)
    return select_ppsr_v2_replan(
        bridge=None,
        pose=_find_safe_forward_pose(maze),
        maze=maze,
        robot_radius=0.28,
        nominal_action=VLNAction.FORWARD,
        image_path=image,
        progress_monitor=monitor,
        step_idx=2,
        horizon_default=2,
        horizon_stuck=3,
        safety_margin=0.05,
    )


def _stuck_monitor() -> ProgressMonitor:
    monitor = ProgressMonitor(episode_id="ep")
    monitor.record_reject(step_idx=0, nominal_action="forward", reject_reason="test")
    monitor.record_reject(step_idx=1, nominal_action="forward", reject_reason="test")
    return monitor


def _turn_loop_monitor() -> ProgressMonitor:
    monitor = ProgressMonitor(episode_id="ep")
    for idx, action in enumerate(["turn_left", "turn_right", "turn_left", "turn_right"]):
        monitor.record_step(step_idx=idx, executed_action=action, skill_status="ok", stop_source="", selected_recovery_action=action)
    return monitor


def _summary(method: str, **overrides):
    base = {
        "method": method,
        "safe_success_rate": 0.30,
        "unsafe_violation_rate": 0.0,
        "small_turn_recovery_rate": 0.5,
        "dmps_recovery_stuck_count": 10,
        "escape_macro_selected_count": 1,
        "escape_macro_commitment_count": 1,
        "post_reject_forward_reenabled_rate": 0.5,
        "post_reject_clearance_gain": 0.1,
        "positive_post_reject_progress_rate": 0.1,
    }
    base.update(overrides)
    return base


def _minimal_audits():
    return {
        "casa_vln_bridge_audit": {"loaded_real_casa_artifacts": True},
        "privileged_leakage_audit": {
            "used_goal_distance_for_online_replan": False,
            "used_shortest_path_for_online_replan": False,
            "used_a_star_for_online_replan": False,
            "used_oracle_waypoint_for_online_replan": False,
            "used_evaluator_success_for_online_replan": False,
            "used_map_cell_progress_for_online_replan": False,
        },
        "stop_source_audit": {"casa_stop_rows_counted_as_policy_stop": 0},
        "readiness_progress_metrics": {"post_reject_clearance_gain": 0.1},
    }


def _find_safe_forward_pose(maze: MazeMap) -> RobotPose2D:
    for cell in maze.free_cells:
        x, y = maze.cell_to_xy(cell)
        if maze.is_xy_safe(x + 0.16, y, robot_radius=0.28):
            return RobotPose2D(x, y, 0.0)
    raise AssertionError("no safe forward pose found")
