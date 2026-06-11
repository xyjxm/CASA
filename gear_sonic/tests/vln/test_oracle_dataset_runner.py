from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from pathlib import Path

import pytest

from gear_sonic.vln.actions import ActionDecision, VLNAction
from gear_sonic.vln.backends import BackendResult
from gear_sonic.vln.dataset import AutoVLNDemoRecord, read_jsonl, validate_demo_record, write_jsonl
from gear_sonic.vln.maze import MazeMap, generate_routes
from gear_sonic.vln.metrics import aggregate_metrics, summarize_episode
from gear_sonic.vln.no_casa_policy import InferenceInput, detect_gallery_wall_painting, parse_route_actions_from_instruction
from gear_sonic.vln.no_casa_runner import MAP_METADATA, RealNaVidVisualAdapterPolicy, RunnerConfig, run, run_episode
from gear_sonic.vln.oracle import AStarTeacher
from gear_sonic.vln.tasks import task_from_route
from gear_sonic.vln.real_navid_online import _select_final_action, parse_args as parse_strict_real_navid_args
from gear_sonic.vln.visual_adapter import (
    VisualActionAdapterPolicy,
    VisualAdapterConfig,
    assert_no_privileged_inference_schema,
    train_visual_action_adapter,
)


def test_training_only_astar_teacher_avoids_occupied_cells() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    route = generate_routes(maze, count=1, seed=3, min_edges=3, max_edges=5)[0]
    teacher = AStarTeacher(maze)
    task = teacher.build_task(route.start_cell, route.goal_cell, episode_id="teacher", split="train")
    assert task.path_cells
    for cell in task.path_cells:
        assert maze.is_free(cell)
    assert task.route_actions
    assert all(action in {item.value for item in VLNAction} for action in task.route_actions)


def test_generate_routes_can_enforce_long_range_euclidean_distance() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    routes = generate_routes(maze, count=5, seed=13, min_edges=4, max_edges=8, min_euclidean_m=6.0)
    assert len(routes) == 5
    for route in routes:
        distance = maze.euclidean_distance_cells(route.start_cell, route.goal_cell)
        assert distance >= 6.0


def test_furniture_obstacles_block_teacher_routes_and_xy_safety(tmp_path) -> None:
    base = json.loads(Path(MAP_METADATA).read_text())
    base["furniture_obstacles"] = [
        {"name": "test_table", "kind": "table", "cell": [3, 3], "size": [0.58, 0.58, 0.42]},
        {"name": "test_chair", "kind": "chair", "cell": [6, 5], "size": [0.42, 0.42, 0.38]},
    ]
    metadata = tmp_path / "furnished.json"
    metadata.write_text(json.dumps(base))
    maze = MazeMap.from_metadata(metadata)

    assert (3, 3) not in maze.free_cells
    assert (6, 5) not in maze.free_cells
    assert not maze.is_xy_safe(*maze.cell_to_xy((3, 3)))

    routes = generate_routes(maze, count=5, seed=19, min_edges=4, max_edges=10, require_furniture_detour=True)
    for route in routes:
        assert (3, 3) not in route.path_cells
        assert (6, 5) not in route.path_cells
        assert maze.route_requires_furniture_detour(route.start_cell, route.goal_cell, list(route.path_cells))
        assert " x" not in route.instruction.lower()
        assert "wall painting" in route.instruction.lower()
        assert "red target ball" not in route.instruction.lower()
        assert "furniture" in route.instruction.lower()
        assert "table" in route.instruction.lower()


def test_xy_cell_mapping_round_trip() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    for cell in maze.free_cells:
        assert maze.xy_to_cell(*maze.cell_to_xy(cell)) == cell


def test_route_instruction_parser() -> None:
    instruction = "move forward x4; then turn right x3; then move forward x2; stop near the red ball"
    assert parse_route_actions_from_instruction(instruction) == [
        VLNAction.FORWARD,
        VLNAction.FORWARD,
        VLNAction.FORWARD,
        VLNAction.FORWARD,
        VLNAction.TURN_RIGHT,
        VLNAction.TURN_RIGHT,
        VLNAction.TURN_RIGHT,
        VLNAction.FORWARD,
        VLNAction.FORWARD,
    ]


def test_inference_schema_has_no_privileged_fields() -> None:
    schema = InferenceInput("ep", 0, "go", "frame.png").to_schema_dict()
    forbidden = {"map", "pose", "goal", "path", "waypoint", "cell", "occupancy"}
    assert forbidden.isdisjoint(schema)


def test_final_online_inference_path_does_not_call_teacher_planner() -> None:
    visual_module = __import__(
        "gear_sonic.vln.visual_adapter",
        fromlist=["VisualActionAdapter", "VisualActionAdapterPolicy", "build_inference_features"],
    )
    policy_source = "\n".join(
        [
            inspect.getsource(visual_module.VisualActionAdapterPolicy.next_action),
            inspect.getsource(visual_module.VisualActionAdapter.next_action),
            inspect.getsource(visual_module.VisualActionAdapter.predict_proba),
            inspect.getsource(visual_module.build_inference_features),
        ]
    )
    episode_source = inspect.getsource(run_episode)
    forbidden = ["AStarTeacher", ".astar(", "teacher_action", "teacher_metadata", "parse_route_actions_from_instruction"]
    assert all(token not in policy_source for token in forbidden)
    assert all(token not in episode_source for token in forbidden)


def test_auto_dataset_schema_validation(tmp_path) -> None:
    record = AutoVLNDemoRecord(
        episode_id="demo",
        step_idx=0,
        instruction="move forward x1 and stop at the red ball",
        image_path="frame.png",
        history_image_paths=[],
        previous_actions=[],
        previous_skill_status=[],
        teacher_action="forward",
        sonic_skill={"name": "walk", "params": {"duration": 1.0}},
        should_stop=False,
        stop_label_reason="far_from_goal_negative",
        teacher_metadata={"path_cells": [[1, 1], [1, 2]]},
        goal_xy=[-3.0, 5.0],
        pose={"x": -5.0, "y": 5.0, "yaw_deg": 0.0},
    )
    validate_demo_record(asdict(record))
    path = tmp_path / "auto_vln_dataset.jsonl"
    write_jsonl(path, [record])
    assert read_jsonl(path)[0]["teacher_action"] == "forward"

    invalid = json.loads(record.to_json())
    invalid.pop("teacher_metadata")
    with pytest.raises(ValueError):
        validate_demo_record(invalid)


def test_success_metric_rejects_missing_stop_and_stop_only() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    route = generate_routes(maze, count=1, seed=9, min_edges=2, max_edges=3)[0]
    task = task_from_route(route, episode_id="metric", split="heldout")
    trajectory = [
        {"pose_x": task.start_xy[0], "pose_y": task.start_xy[1], "pose_yaw_deg": task.start_yaw_deg, "wall_collision": 0, "fall": 0},
        {"pose_x": task.goal_xy[0], "pose_y": task.goal_xy[1], "pose_yaw_deg": task.start_yaw_deg, "wall_collision": 0, "fall": 0},
    ]
    no_stop = summarize_episode(
        task=task,
        maze=maze,
        decision_records=[{"final_action": "forward", "evaluator_distance_to_goal": 0.0, "privileged_policy_usage": False}],
        trajectory_rows=trajectory,
        video_paths=[],
        frames_dir=Path("frames"),
        decision_log=Path("decision.jsonl"),
        trajectory_log=Path("trajectory.jsonl"),
    )
    assert not no_stop.success
    assert no_stop.failure_reason == "missing_policy_stop"

    stop_only = summarize_episode(
        task=task,
        maze=maze,
        decision_records=[{"final_action": "stop", "evaluator_distance_to_goal": 0.0, "image_path": "frame.png", "privileged_policy_usage": False}],
        trajectory_rows=[trajectory[-1]],
        video_paths=[],
        frames_dir=Path("frames"),
        decision_log=Path("decision.jsonl"),
        trajectory_log=Path("trajectory.jsonl"),
    )
    assert not stop_only.success
    assert stop_only.failure_reason == "insufficient_motion"


def test_stop_metrics_precision_recall_rates() -> None:
    episodes = [
        {
            "success": True,
            "fall_count": 0,
            "collision_count": 0,
            "stuck": False,
            "timeout": False,
            "final_distance": 0.4,
            "shortest_path_progress": 0.9,
            "executed_motion_skills": 4,
            "stop_fallback_ratio": 0.2,
            "policy_issued_stop": True,
            "stop_distance": 0.4,
            "stop_failure_type": None,
            "privileged_policy_usage_count": 0,
            "action_distribution": {"forward": 3, "turn_left": 0, "turn_right": 0, "backoff": 0, "stop": 1},
        },
        {
            "success": False,
            "fall_count": 0,
            "collision_count": 0,
            "stuck": False,
            "timeout": False,
            "final_distance": 2.0,
            "shortest_path_progress": 0.2,
            "executed_motion_skills": 2,
            "stop_fallback_ratio": 0.33,
            "policy_issued_stop": True,
            "stop_distance": 2.0,
            "stop_failure_type": "premature_stop",
            "privileged_policy_usage_count": 0,
            "action_distribution": {"forward": 2, "turn_left": 0, "turn_right": 0, "backoff": 0, "stop": 1},
        },
    ]
    metrics = aggregate_metrics(episodes)
    assert metrics["stop_precision"] == 0.5
    assert metrics["premature_stop_rate"] == 0.5


def test_mock_smoke_runner(tmp_path) -> None:
    report = run(
        RunnerConfig(
            run_dir=tmp_path,
            smoke_only=True,
            auto_demo_count=2,
            heldout_episodes=1,
            frame_width=160,
            frame_height=120,
            visual_adapter_epochs=8,
        )
    )
    assert report["episode_metrics"]["total_episodes"] == 1
    assert report["policy_name"] == "visual_action_adapter"
    assert not report["route_parser_used_by_test_time_policy"]
    assert (tmp_path / "navigation_manifest.json").exists()
    assert (tmp_path / "figures/head_camera_check.png").exists()


def test_visual_action_adapter_training_and_inference_contract(tmp_path) -> None:
    from PIL import Image

    assert_no_privileged_inference_schema()
    records = []
    for idx, (name, color) in enumerate(
        [
            ("forward", (20, 20, 20)),
            ("turn_left", (20, 80, 20)),
            ("turn_right", (20, 20, 80)),
            ("backoff", (80, 20, 20)),
            ("stop", (230, 20, 20)),
        ]
    ):
        image_path = tmp_path / f"{name}.png"
        Image.new("RGB", (32, 24), color).save(image_path)
        records.append(
            AutoVLNDemoRecord(
                episode_id=f"demo_{idx}",
                step_idx=idx,
                instruction="Follow the corridor and stop when the red target is close.",
                image_path=str(image_path),
                history_image_paths=[],
                previous_actions=["forward"] * idx,
                previous_skill_status=["ok"] * idx,
                teacher_action=name,
                sonic_skill={"name": "skill", "params": {}},
                should_stop=name == "stop",
                stop_label_reason="near_goal_visual" if name == "stop" else "far_from_goal_negative",
                teacher_metadata={"training_only": True},
            )
        )
    dataset_path = tmp_path / "auto_vln_dataset.jsonl"
    write_jsonl(dataset_path, records)
    adapter_path = tmp_path / "visual_action_adapter.json"
    adapter = train_visual_action_adapter(
        dataset_path,
        adapter_path,
        config=VisualAdapterConfig(epochs=20, validation_fraction=0.0, image_width=8, image_height=6),
    )
    policy = VisualActionAdapterPolicy(adapter)
    obs = InferenceInput(
        episode_id="heldout",
        step_idx=0,
        instruction="Follow the corridor and stop when the red target is close.",
        image_path=str(tmp_path / "stop.png"),
        history_image_paths=[],
        previous_actions=[],
        previous_skill_status=[],
    )
    decision = policy.next_action(obs)
    assert decision.metadata["privileged_policy_usage"] is False
    assert decision.metadata["test_time_astar_used"] is False
    assert decision.metadata["test_time_map_pose_goal_path_used"] is False
    assert adapter.train_records == 5


def test_visual_action_adapter_masks_zero_support_actions(tmp_path) -> None:
    from PIL import Image

    records = []
    for idx, (name, color) in enumerate(
        [
            ("forward", (20, 20, 20)),
            ("turn_left", (20, 80, 20)),
            ("turn_right", (20, 20, 80)),
            ("stop", (230, 20, 20)),
        ]
    ):
        image_path = tmp_path / f"{name}.png"
        Image.new("RGB", (32, 24), color).save(image_path)
        records.append(
            AutoVLNDemoRecord(
                episode_id=f"demo_{idx}",
                step_idx=idx,
                instruction="Follow the corridor.",
                image_path=str(image_path),
                history_image_paths=[],
                previous_actions=[],
                previous_skill_status=[],
                teacher_action=name,
                sonic_skill={"name": "skill", "params": {}},
                should_stop=name == "stop",
                stop_label_reason="near_goal_visual" if name == "stop" else "far_from_goal_negative",
                teacher_metadata={"training_only": True},
            )
        )
    dataset_path = tmp_path / "auto_vln_dataset.jsonl"
    write_jsonl(dataset_path, records)
    adapter = train_visual_action_adapter(
        dataset_path,
        tmp_path / "visual_action_adapter.json",
        config=VisualAdapterConfig(epochs=8, validation_fraction=0.0, image_width=8, image_height=6),
    )
    obs = InferenceInput(
        episode_id="heldout",
        step_idx=0,
        instruction="Follow the corridor.",
        image_path=str(tmp_path / "forward.png"),
        history_image_paths=[],
        previous_actions=[],
        previous_skill_status=[],
    )
    probs = adapter.predict_proba(obs)
    assert adapter.metrics["label_counts"]["backoff"] == 0
    assert probs[adapter.action_values.index("backoff")] == 0.0


class _FakeRealNaVidBackend:
    name = "navid_real_model_inference"

    def reset(self, episode_id: str) -> None:
        del episode_id

    def next_action(self, observation) -> BackendResult:
        del observation
        decision = ActionDecision(VLNAction.TURN_LEFT, "left", self.name, metadata={"navid_model_loaded": True})
        return BackendResult(
            decision=decision,
            backend_name=self.name,
            raw_output="left",
            available=True,
            metadata={"navid_model_loaded": True},
        )

    def close(self) -> None:
        pass


class _ConstantVisualPolicy:
    name = "visual_action_adapter"

    def __init__(self, action: VLNAction) -> None:
        self.action = action

    def reset(self, episode_id: str) -> None:
        del episode_id

    def next_action(self, obs: InferenceInput) -> ActionDecision:
        del obs
        return ActionDecision(
            self.action,
            {"action": self.action.value},
            self.name,
            confidence=0.99,
            metadata={
                "privileged_policy_usage": False,
                "test_time_astar_used": False,
                "test_time_map_pose_goal_path_used": False,
                "probabilities": {action.value: 1.0 if action is self.action else 0.0 for action in VLNAction},
            },
        )


def test_gallery_wall_painting_detection_and_stop_override(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "near_wall_painting.png"
    Image.new("RGB", (64, 48), (10, 210, 240)).save(image_path)
    evidence = detect_gallery_wall_painting(image_path)
    assert evidence.visible
    assert evidence.target_ratio and evidence.target_ratio > 0.90

    policy = RealNaVidVisualAdapterPolicy(
        _FakeRealNaVidBackend(),
        _ConstantVisualPolicy(VLNAction.FORWARD),
    )
    decision = policy.next_action(
        InferenceInput(
            episode_id="heldout",
            step_idx=3,
            instruction="Stop near the gallery wall painting.",
            image_path=str(image_path),
            previous_actions=[VLNAction.FORWARD.value] * 3,
            previous_skill_status=["ok"] * 3,
        )
    )
    assert decision.action is VLNAction.STOP
    assert decision.metadata["near_wall_painting_stop_guard_applied"] is True
    assert decision.metadata["gallery_wall_painting_evidence"]["target_kind"] == "gallery_wall_painting"


def test_premature_visual_stop_is_suppressed_without_wall_painting(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "plain_corridor.png"
    Image.new("RGB", (64, 48), (38, 38, 38)).save(image_path)
    policy = RealNaVidVisualAdapterPolicy(
        _FakeRealNaVidBackend(),
        _ConstantVisualPolicy(VLNAction.STOP),
    )

    moving_decision = policy.next_action(
        InferenceInput(
            episode_id="heldout",
            step_idx=8,
            instruction="Walk through the corridor and stop at the gallery painting.",
            image_path=str(image_path),
            previous_actions=[VLNAction.FORWARD.value] * 8,
            previous_skill_status=["ok"] * 8,
        )
    )
    assert moving_decision.action is VLNAction.FORWARD
    assert moving_decision.metadata["premature_visual_stop_suppressed"] is True
    assert moving_decision.metadata["near_wall_painting_stop_guard_applied"] is False

    blocked_decision = policy.next_action(
        InferenceInput(
            episode_id="heldout",
            step_idx=9,
            instruction="Walk through the corridor and stop at the gallery painting.",
            image_path=str(image_path),
            previous_actions=[VLNAction.FORWARD.value] * 9,
            previous_skill_status=["ok"] * 8 + ["blocked"],
        )
    )
    assert blocked_decision.action is VLNAction.BACKOFF
    assert blocked_decision.metadata["premature_visual_stop_suppressed"] is True


def test_repeated_turn_cycle_recovery_uses_forward_burst_and_opposite_turn(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "corridor.png"
    Image.new("RGB", (64, 48), (35, 35, 35)).save(image_path)
    policy = RealNaVidVisualAdapterPolicy(
        _FakeRealNaVidBackend(),
        _ConstantVisualPolicy(VLNAction.TURN_LEFT),
        turn_cycle_forward_burst=3,
    )

    burst_decision = policy.next_action(
        InferenceInput(
            episode_id="heldout",
            step_idx=6,
            instruction="Follow the corridor.",
            image_path=str(image_path),
            previous_actions=[
                VLNAction.TURN_LEFT.value,
                VLNAction.TURN_LEFT.value,
                VLNAction.TURN_LEFT.value,
                VLNAction.FORWARD.value,
                VLNAction.FORWARD.value,
            ],
            previous_skill_status=["ok"] * 5,
        )
    )
    assert burst_decision.action is VLNAction.FORWARD
    assert burst_decision.metadata["turn_cycle_forward_burst_guard_applied"] is True

    opposite_decision = policy.next_action(
        InferenceInput(
            episode_id="heldout",
            step_idx=7,
            instruction="Follow the corridor.",
            image_path=str(image_path),
            previous_actions=[
                VLNAction.TURN_LEFT.value,
                VLNAction.TURN_LEFT.value,
                VLNAction.TURN_LEFT.value,
                VLNAction.FORWARD.value,
                VLNAction.FORWARD.value,
                VLNAction.FORWARD.value,
            ],
            previous_skill_status=["ok"] * 6,
        )
    )
    assert opposite_decision.action is VLNAction.TURN_RIGHT
    assert opposite_decision.metadata["turn_cycle_opposite_turn_guard_applied"] is True


def test_strict_real_navid_runner_visual_adapter_override_contract(tmp_path) -> None:
    adapter_path = tmp_path / "visual_action_adapter.json"
    adapter_path.write_text("{}\n")
    config = parse_strict_real_navid_args(["--visual-adapter-path", str(adapter_path)])
    assert config.visual_adapter_mode == "override"

    final_action, final_source, metadata = _select_final_action(
        navid_decision=ActionDecision(VLNAction.TURN_LEFT, "left", "navid_real_model_inference"),
        visual_decision=ActionDecision(
            VLNAction.FORWARD,
            {"action": "forward"},
            "visual_action_adapter",
            confidence=0.9,
            metadata={"privileged_policy_usage": False, "test_time_astar_used": False},
        ),
        previous_actions=[],
        visual_adapter_mode="override",
        visual_adapter_confidence_threshold=0.0,
        visual_adapter_min_stop_step=0,
        visual_adapter_min_backoff_step=0,
        force_forward_after_turn_streak=0,
    )
    assert final_action is VLNAction.FORWARD
    assert final_source == "navid_real_model_inference+visual_action_adapter_override"
    assert metadata["navid_action"] == "turn_left"
    assert metadata["visual_adapter_action"] == "forward"
    assert metadata["visual_adapter_used_for_final_action"] is True

    guarded_action, guarded_source, guarded_metadata = _select_final_action(
        navid_decision=ActionDecision(VLNAction.TURN_LEFT, "left", "navid_real_model_inference"),
        visual_decision=ActionDecision(VLNAction.STOP, {"action": "stop"}, "visual_action_adapter", confidence=1.0),
        previous_actions=[],
        visual_adapter_mode="override",
        visual_adapter_confidence_threshold=0.0,
        visual_adapter_min_stop_step=2,
        visual_adapter_min_backoff_step=0,
        force_forward_after_turn_streak=0,
    )
    assert guarded_action is VLNAction.FORWARD
    assert guarded_source.endswith("+strict_runner_early_stop_guard")
    assert guarded_metadata["strict_runner_early_stop_guard_applied"] is True

    backoff_guarded_action, backoff_guarded_source, backoff_guarded_metadata = _select_final_action(
        navid_decision=ActionDecision(VLNAction.TURN_LEFT, "left", "navid_real_model_inference"),
        visual_decision=ActionDecision(VLNAction.BACKOFF, {"action": "backoff"}, "visual_action_adapter", confidence=1.0),
        previous_actions=[],
        visual_adapter_mode="override",
        visual_adapter_confidence_threshold=0.0,
        visual_adapter_min_stop_step=0,
        visual_adapter_min_backoff_step=2,
        force_forward_after_turn_streak=0,
    )
    assert backoff_guarded_action is VLNAction.FORWARD
    assert backoff_guarded_source.endswith("+strict_runner_early_backoff_guard")
    assert backoff_guarded_metadata["strict_runner_early_backoff_guard_applied"] is True


def test_strict_real_navid_runner_visual_adapter_stop_only_contract(tmp_path) -> None:
    adapter_path = tmp_path / "visual_action_adapter.json"
    adapter_path.write_text("{}\n")
    config = parse_strict_real_navid_args(
        ["--visual-adapter-path", str(adapter_path), "--visual-adapter-mode", "stop_only"]
    )
    assert config.visual_adapter_mode == "stop_only"

    navid_turn = ActionDecision(VLNAction.TURN_RIGHT, "right", "navid_real_model_inference")
    visual_forward = ActionDecision(
        VLNAction.FORWARD,
        {"action": "forward"},
        "visual_action_adapter",
        confidence=1.0,
        metadata={"privileged_policy_usage": False, "test_time_astar_used": False},
    )
    final_action, final_source, metadata = _select_final_action(
        navid_decision=navid_turn,
        visual_decision=visual_forward,
        previous_actions=[],
        visual_adapter_mode="stop_only",
        visual_adapter_confidence_threshold=0.0,
        visual_adapter_min_stop_step=0,
        visual_adapter_min_backoff_step=0,
        force_forward_after_turn_streak=0,
    )
    assert final_action is VLNAction.TURN_RIGHT
    assert final_source == "navid_real_model_inference"
    assert metadata["visual_adapter_used_for_final_action"] is False
    assert metadata["visual_adapter_selection_reason"] == "visual_adapter_stop_only_non_stop_ignored"

    visual_stop = ActionDecision(VLNAction.STOP, {"action": "stop"}, "visual_action_adapter", confidence=1.0)
    early_action, early_source, early_metadata = _select_final_action(
        navid_decision=navid_turn,
        visual_decision=visual_stop,
        previous_actions=["forward"] * 2,
        visual_adapter_mode="stop_only",
        visual_adapter_confidence_threshold=0.0,
        visual_adapter_min_stop_step=5,
        visual_adapter_min_backoff_step=0,
        force_forward_after_turn_streak=0,
    )
    assert early_action is VLNAction.TURN_RIGHT
    assert early_source == "navid_real_model_inference"
    assert early_metadata["visual_adapter_used_for_final_action"] is False
    assert early_metadata["visual_adapter_selection_reason"] == "visual_adapter_stop_only_early_stop_reject"

    stop_action, stop_source, stop_metadata = _select_final_action(
        navid_decision=navid_turn,
        visual_decision=visual_stop,
        previous_actions=["forward"] * 5,
        visual_adapter_mode="stop_only",
        visual_adapter_confidence_threshold=0.0,
        visual_adapter_min_stop_step=5,
        visual_adapter_min_backoff_step=0,
        force_forward_after_turn_streak=0,
    )
    assert stop_action is VLNAction.STOP
    assert stop_source == "navid_real_model_inference+visual_action_adapter_stop_only"
    assert stop_metadata["visual_adapter_used_for_final_action"] is True
    assert stop_metadata["visual_adapter_selection_reason"] == "visual_adapter_stop_only_stop_pass"
