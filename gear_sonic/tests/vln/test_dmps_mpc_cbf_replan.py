from __future__ import annotations

from pathlib import Path

from PIL import Image

from gear_sonic.vln.actions import VLNAction
from gear_sonic.vln.casa_runner import write_table_csv
from gear_sonic.vln.dmps_mpc_cbf_replan import (
    DMPS_LAST_RESORT_STOP_SOURCE,
    build_candidate_sequences,
    candidate_scores_to_rows,
    compute_visual_free_space,
    intent_consistency_score,
    rollout_candidate_sequence,
    select_dmps_replan,
)
from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.no_casa_runner import MAP_METADATA
from gear_sonic.vln.oracle import RobotPose2D
from gear_sonic.vln.progress_monitor import ProgressMonitor
from gear_sonic.vln.skill_mapping import WalkSkill


def test_dmps_candidate_generation_includes_required_single_and_two_step_sequences() -> None:
    candidates = build_candidate_sequences(pose=RobotPose2D(0.0, 0.0, 0.0), nominal_action=VLNAction.FORWARD)
    sequences = {candidate.actions for candidate in candidates}
    assert ("short_forward",) in sequences
    assert ("backoff",) in sequences
    assert ("turn_left",) in sequences
    assert ("turn_right",) in sequences
    assert ("stop_as_last_resort",) in sequences
    assert ("backoff", "turn_left") in sequences
    assert ("backoff", "turn_right") in sequences
    assert ("turn_left", "short_forward") in sequences
    assert ("turn_right", "short_forward") in sequences
    assert ("backoff", "short_forward") in sequences


def test_safety_hard_constraint_blocks_wall_candidate() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose, unsafe_skill = _find_forward_into_wall_case(maze)
    candidate = build_candidate_sequences(pose=pose, nominal_action=VLNAction.FORWARD)[0]
    candidate = type(candidate)("unsafe_forward", ("short_forward",), (unsafe_skill,), False)
    rollout = rollout_candidate_sequence(
        candidate=candidate,
        start_pose=pose,
        maze=maze,
        robot_radius=0.28,
        safety_margin=0.05,
    )
    assert rollout.would_block
    assert not rollout.safety_feasible


def test_stop_last_resort_not_selected_when_non_stop_safe_candidate_exists(tmp_path) -> None:
    image = tmp_path / "frame.png"
    Image.new("RGB", (96, 72), (180, 180, 180)).save(image)
    selection = select_dmps_replan(
        bridge=None,
        pose=_find_safe_forward_pose(MazeMap.from_metadata(MAP_METADATA)),
        maze=MazeMap.from_metadata(MAP_METADATA),
        robot_radius=0.28,
        nominal_action=VLNAction.FORWARD,
        image_path=image,
        progress_monitor=ProgressMonitor(episode_id="ep"),
        step_idx=0,
        horizon=2,
        safety_margin=0.05,
    )
    assert selection.selected_sequence.actions[0] != "stop_as_last_resort"


def test_no_evaluator_leakage_in_progress_monitor_and_selection(tmp_path) -> None:
    image = tmp_path / "frame.png"
    Image.new("RGB", (96, 72), (140, 140, 140)).save(image)
    monitor = ProgressMonitor(episode_id="ep")
    monitor.pre_step(step_idx=0, image_path=image)
    snapshot = monitor.snapshot(step_idx=0)
    forbidden = {"goal", "distance", "shortest", "astar", "waypoint"}
    assert all(token not in str(snapshot).lower() for token in forbidden)


def test_repeated_forward_reject_penalizes_forward_selection(tmp_path) -> None:
    monitor = ProgressMonitor(episode_id="ep")
    monitor.record_reject(step_idx=0, nominal_action="forward", reject_reason="test")
    monitor.record_reject(step_idx=1, nominal_action="forward", reject_reason="test")
    forward_penalty = monitor.penalties_for_candidate(nominal_action="forward", candidate_sequence=["short_forward"])
    turn_penalty = monitor.penalties_for_candidate(nominal_action="forward", candidate_sequence=["turn_left"])
    assert forward_penalty["repeated_reject_penalty"] >= turn_penalty["repeated_reject_penalty"]


def test_turn_loop_is_detected_and_penalized() -> None:
    monitor = ProgressMonitor(episode_id="ep")
    for idx, action in enumerate(["turn_left", "turn_right", "turn_left"]):
        monitor.record_step(step_idx=idx, executed_action=action, skill_status="ok", stop_source="")
    penalties = monitor.penalties_for_candidate(nominal_action="turn_right", candidate_sequence=["turn_right"])
    assert penalties["recovery_loop_penalty"] > 0.0


def test_visual_free_space_scores_left_center_right(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (96, 72), (30, 30, 30))
    for x in range(32, 64):
        for y in range(72):
            image.putpixel((x, y), (220, 220, 220))
    image.save(image_path)
    scores = compute_visual_free_space(image_path)
    assert scores["visual_free_space_score_available"] is True
    assert scores["center_score"] > scores["left_score"]
    assert scores["center_score"] > scores["right_score"]


def test_intent_score_prefers_forward_over_stop_for_forward_nominal() -> None:
    assert intent_consistency_score(nominal_action=VLNAction.FORWARD, candidate_sequence=["short_forward"]) > intent_consistency_score(
        nominal_action=VLNAction.FORWARD,
        candidate_sequence=["stop_as_last_resort"],
    )


def test_dmps_control_return_flag_and_stop_source(tmp_path) -> None:
    image = tmp_path / "frame.png"
    Image.new("RGB", (96, 72), (128, 128, 128)).save(image)
    selection = select_dmps_replan(
        bridge=None,
        pose=_find_safe_forward_pose(MazeMap.from_metadata(MAP_METADATA)),
        maze=MazeMap.from_metadata(MAP_METADATA),
        robot_radius=0.28,
        nominal_action=VLNAction.FORWARD,
        image_path=image,
        progress_monitor=ProgressMonitor(episode_id="ep"),
        step_idx=0,
        horizon=2,
        safety_margin=0.05,
    )
    assert selection.control_returned_to_vln_next_step is True
    assert DMPS_LAST_RESORT_STOP_SOURCE != "vln_policy"
    assert DMPS_LAST_RESORT_STOP_SOURCE != "policy_internal_guard"


def test_dmps_log_schema_contains_required_columns(tmp_path) -> None:
    image = tmp_path / "frame.png"
    Image.new("RGB", (96, 72), (128, 128, 128)).save(image)
    selection = select_dmps_replan(
        bridge=None,
        pose=_find_safe_forward_pose(MazeMap.from_metadata(MAP_METADATA)),
        maze=MazeMap.from_metadata(MAP_METADATA),
        robot_radius=0.28,
        nominal_action=VLNAction.FORWARD,
        image_path=image,
        progress_monitor=ProgressMonitor(episode_id="ep"),
        step_idx=0,
        horizon=2,
        safety_margin=0.05,
    )
    rows = candidate_scores_to_rows(
        method="vln_dmps_mpc_cbf_progress",
        episode_id="ep",
        step_idx=0,
        nominal_action="forward",
        reject_reason="test",
        selection=selection,
    )
    csv_path = tmp_path / "dmps_candidate_sequences.csv"
    write_table_csv(csv_path, rows)
    header = csv_path.read_text().splitlines()[0].split(",")
    for column in [
        "episode_id",
        "step_id",
        "nominal_action",
        "candidate_sequence_id",
        "candidate_sequence",
        "sequence_length",
        "min_clearance",
        "min_barrier_h",
        "cbf_violation",
        "would_block",
        "predicted_wall_contact_steps",
        "safety_feasible",
        "intent_consistency_score",
        "visual_free_space_score",
        "total_score",
        "selected",
    ]:
        assert column in header


def _find_forward_into_wall_case(maze: MazeMap) -> tuple[RobotPose2D, WalkSkill]:
    import math

    for cell in maze.free_cells:
        x, y = maze.cell_to_xy(cell)
        for (dr, dc), yaw in [((0, 1), 0.0), ((-1, 0), 90.0), ((0, -1), 180.0), ((1, 0), -90.0)]:
            if (cell[0] + dr, cell[1] + dc) in maze.blocking_cells():
                radians = math.radians(yaw)
                skill = WalkSkill(
                    vx=math.cos(radians),
                    vy=math.sin(radians),
                    facing_yaw_deg=yaw,
                    duration=0.7,
                    step_target_m=1.5,
                )
                return RobotPose2D(x, y, yaw), skill
    raise AssertionError("no wall case found")


def _find_safe_forward_pose(maze: MazeMap) -> RobotPose2D:
    for cell in maze.free_cells:
        x, y = maze.cell_to_xy(cell)
        if maze.is_xy_safe(x + 0.16, y, robot_radius=0.28):
            return RobotPose2D(x, y, 0.0)
    raise AssertionError("no safe forward pose found")
