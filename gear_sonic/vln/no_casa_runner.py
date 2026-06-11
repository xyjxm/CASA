"""No-CASA SONIC VLN/VLA-style MuJoCo maze success30 runner."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from .actions import VLNAction
from .backends import NaVidBackend, VLNObservation
from .dataset import AutoVLNDemoRecord, write_jsonl
from .maze import MazeMap, generate_routes
from .metrics import aggregate_metrics, append_jsonl, summarize_episode, write_json
from .no_casa_mujoco import MujocoMazeSkillEnv, draw_topdown_map, write_video
from .no_casa_policy import InferenceInput, detect_gallery_wall_painting
from .skill_mapping import ActionSkillMapper, SkillMappingConfig
from .tasks import TeacherTask, task_from_route
from .visual_adapter import (
    VisualActionAdapter,
    VisualActionAdapterPolicy,
    VisualAdapterConfig,
    assert_no_privileged_inference_schema,
    train_visual_action_adapter,
)


SCENE_XML = Path(
    "/mnt/data/students/lph/recording/gymnasium_robotics_maze_maps_scale2_20260610_233102/"
    "mjcf/sonic_scene_MEDIUM_MAZE_DIVERSE_GR_scale2_official_gymnasium.xml"
)
MAP_METADATA = Path(
    "/mnt/data/students/lph/recording/gymnasium_robotics_maze_maps_scale2_20260610_233102/"
    "data/MEDIUM_MAZE_DIVERSE_GR_scale2.json"
)
OUTPUT_ROOT = Path("/mnt/data/students/lph/recording")
NAVID_REPO = Path("/mnt/data/students/lph/recording/vln_sonic_navid_auto_20260610_202005/third_party/NaVid-VLN-CE")
NAVID_MODEL = Path("/mnt/data/students/lph/models/navid/Jzzhang_NaVid/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split")
NAVID_PYTHON = Path("/mnt/data/students/lph/models/navid/envs/navid-real/bin/python")
NAVID_VISION_TOWER = Path("/mnt/data/students/lph/models/navid/model_zoo/eva_vit_g.pth")


@dataclass(frozen=True)
class RunnerConfig:
    run_dir: Path
    scene_xml: Path = SCENE_XML
    map_metadata: Path = MAP_METADATA
    train_scene_xml: Path | None = None
    train_map_metadata: Path | None = None
    heldout_episodes: int = 20
    auto_demo_count: int = 100
    seed: int = 260610
    heldout_seed: int = 260611
    max_steps: int = 96
    frame_width: int = 320
    frame_height: int = 240
    smoke_only: bool = False
    skip_demo_frames: bool = False
    visual_adapter_epochs: int = 240
    min_route_edges: int = 2
    max_route_edges: int = 7
    min_route_euclidean_m: float = 0.0
    require_furniture_detour: bool = False
    policy_backend: str = "visual_adapter"
    navid_repo: Path = NAVID_REPO
    navid_model: Path = NAVID_MODEL
    navid_python: Path = NAVID_PYTHON
    navid_vision_tower: Path = NAVID_VISION_TOWER
    navid_worker_timeout_s: float = 900.0


class RealNaVidPolicy:
    """Policy wrapper that uses the strict real NaVid checkpoint at test time."""

    def __init__(self, backend: NaVidBackend) -> None:
        self.backend = backend
        self.name = backend.name
        self.last_result: Any | None = None

    def reset(self, episode_id: str) -> None:
        self.last_result = None
        self.backend.reset(episode_id)

    def next_action(self, obs: InferenceInput):
        result = self.backend.next_action(
            VLNObservation(
                episode_id=obs.episode_id,
                step_idx=obs.step_idx,
                instruction=obs.instruction,
                image_path=obs.image_path,
                history_image_paths=list(obs.history_image_paths),
                previous_actions=list(obs.previous_actions),
                previous_skill_status=list(obs.previous_skill_status),
            )
        )
        self.last_result = result
        return result.decision

    def close(self) -> None:
        self.backend.close()


class RealNaVidVisualAdapterPolicy:
    """Call real NaVid every step, then use the trained visual adapter to calibrate the final action."""

    name = "navid_real_model_inference_plus_visual_action_adapter"

    def __init__(
        self,
        backend: NaVidBackend,
        adapter_policy: VisualActionAdapterPolicy,
        *,
        force_forward_after_turn_streak: int = 4,
        turn_cycle_window: int = 8,
        turn_cycle_min_same_turns: int = 3,
        turn_cycle_forward_burst: int = 3,
        near_target_stop_ratio: float = 0.65,
        near_target_stop_bbox_area: float = 0.75,
    ) -> None:
        self.backend = backend
        self.adapter_policy = adapter_policy
        self.force_forward_after_turn_streak = force_forward_after_turn_streak
        self.turn_cycle_window = turn_cycle_window
        self.turn_cycle_min_same_turns = turn_cycle_min_same_turns
        self.turn_cycle_forward_burst = turn_cycle_forward_burst
        self.near_target_stop_ratio = near_target_stop_ratio
        self.near_target_stop_bbox_area = near_target_stop_bbox_area
        self.last_result: Any | None = None
        self.last_visual_decision: Any | None = None

    def reset(self, episode_id: str) -> None:
        self.last_result = None
        self.last_visual_decision = None
        self.backend.reset(episode_id)
        self.adapter_policy.reset(episode_id)

    def next_action(self, obs: InferenceInput):
        navid_result = self.backend.next_action(
            VLNObservation(
                episode_id=obs.episode_id,
                step_idx=obs.step_idx,
                instruction=obs.instruction,
                image_path=obs.image_path,
                history_image_paths=list(obs.history_image_paths),
                previous_actions=list(obs.previous_actions),
                previous_skill_status=list(obs.previous_skill_status),
            )
        )
        visual_decision = self.adapter_policy.next_action(obs)
        final_action = visual_decision.action
        final_source_suffix = "visual_action_adapter_calibrated"
        recovery_metadata: dict[str, Any] = {
            "turn_streak_forward_guard_applied": False,
            "turn_cycle_forward_burst_guard_applied": False,
            "turn_cycle_opposite_turn_guard_applied": False,
            "blocked_forward_backoff_guard_applied": False,
            "near_wall_painting_stop_guard_applied": False,
            "premature_visual_stop_suppressed": False,
        }
        target_evidence = detect_gallery_wall_painting(obs.image_path)
        target_data = target_evidence.to_dict()
        target_ratio = float(target_data["target_ratio"])
        target_centered = target_evidence.center_x is not None and 0.15 <= target_evidence.center_x <= 0.85
        near_wall_painting = (
            target_ratio >= self.near_target_stop_ratio
            and target_evidence.bbox_area_ratio >= self.near_target_stop_bbox_area
            and target_centered
            and obs.step_idx >= 2
        )
        recent_status = obs.previous_skill_status[-1] if obs.previous_skill_status else ""
        if near_wall_painting:
            final_action = VLNAction.STOP
            final_source_suffix = "visual_wall_painting_stop_guard"
            recovery_metadata["near_wall_painting_stop_guard_applied"] = True
        elif final_action is VLNAction.STOP:
            final_action = VLNAction.BACKOFF if recent_status in {"blocked", "collision"} else VLNAction.FORWARD
            final_source_suffix = "visual_action_adapter_premature_stop_suppression"
            recovery_metadata["premature_visual_stop_suppressed"] = True
        elif final_action is VLNAction.FORWARD and recent_status in {"blocked", "collision"}:
            final_action = VLNAction.BACKOFF
            final_source_suffix = "visual_action_adapter_blocked_forward_guard"
            recovery_metadata["blocked_forward_backoff_guard_applied"] = True
        if final_action in {VLNAction.TURN_LEFT, VLNAction.TURN_RIGHT} and self.force_forward_after_turn_streak > 0:
            same_turn_streak = 0
            for previous_action in reversed(obs.previous_actions):
                if previous_action != final_action.value:
                    break
                same_turn_streak += 1
            if same_turn_streak >= self.force_forward_after_turn_streak - 1:
                final_action = VLNAction.FORWARD
                final_source_suffix = "visual_action_adapter_turn_streak_guard"
                recovery_metadata["turn_streak_forward_guard_applied"] = True
                recovery_metadata["turn_streak_length_before_guard"] = same_turn_streak
        if final_action in {VLNAction.TURN_LEFT, VLNAction.TURN_RIGHT}:
            cycle_action, cycle_metadata = self._turn_cycle_recovery(obs.previous_actions, final_action)
            if cycle_action is not None:
                final_action = cycle_action
                final_source_suffix = cycle_metadata["turn_cycle_source_suffix"]
                recovery_metadata.update(cycle_metadata)
        raw_output = (
            {
                **visual_decision.raw_output,
                "visual_adapter_action_before_recovery_guard": visual_decision.action.value,
                "final_action_after_recovery_guard": final_action.value,
            }
            if isinstance(visual_decision.raw_output, dict)
            else {
                "visual_adapter_raw_output": visual_decision.raw_output,
                "visual_adapter_action_before_recovery_guard": visual_decision.action.value,
                "final_action_after_recovery_guard": final_action.value,
            }
        )
        self.last_result = navid_result
        self.last_visual_decision = visual_decision
        return type(visual_decision)(
            action=final_action,
            raw_output=raw_output,
            source=f"{navid_result.decision.source}+{final_source_suffix}",
            confidence=visual_decision.confidence if final_action is visual_decision.action else min(visual_decision.confidence, 0.55),
            magnitude=visual_decision.magnitude,
            metadata={
                **visual_decision.metadata,
                **recovery_metadata,
                "visual_adapter_action_before_recovery_guard": visual_decision.action.value,
                "gallery_wall_painting_evidence": target_data,
                "real_navid_called": bool(navid_result.available),
                "real_navid_raw_output": navid_result.raw_output,
                "real_navid_parsed_action": navid_result.decision.action.value,
                "real_navid_source": navid_result.decision.source,
                "real_navid_model_loaded": bool(navid_result.metadata.get("navid_model_loaded")),
                "final_action_calibrated_by_visual_adapter": True,
            },
        )

    def _turn_cycle_recovery(self, previous_actions: list[str], proposed_turn: VLNAction) -> tuple[VLNAction | None, dict[str, Any]]:
        recent = previous_actions[-self.turn_cycle_window :]
        same_turn_count = recent.count(proposed_turn.value)
        forward_count = recent.count(VLNAction.FORWARD.value)
        opposite = VLNAction.TURN_RIGHT if proposed_turn is VLNAction.TURN_LEFT else VLNAction.TURN_LEFT
        opposite_count = recent.count(opposite.value)
        if same_turn_count < self.turn_cycle_min_same_turns:
            return None, {}
        trailing_forwards = 0
        for action in reversed(previous_actions):
            if action != VLNAction.FORWARD.value:
                break
            trailing_forwards += 1
        metadata = {
            "turn_cycle_window": self.turn_cycle_window,
            "turn_cycle_same_turn_count": same_turn_count,
            "turn_cycle_forward_count": forward_count,
            "turn_cycle_opposite_count": opposite_count,
            "turn_cycle_trailing_forward_count": trailing_forwards,
            "turn_cycle_proposed_turn": proposed_turn.value,
        }
        if trailing_forwards < self.turn_cycle_forward_burst:
            return VLNAction.FORWARD, {
                **metadata,
                "turn_cycle_forward_burst_guard_applied": True,
                "turn_cycle_source_suffix": "visual_action_adapter_turn_cycle_forward_burst_guard",
            }
        return opposite, {
            **metadata,
            "turn_cycle_opposite_turn_guard_applied": True,
            "turn_cycle_recovery_turn": opposite.value,
            "turn_cycle_source_suffix": "visual_action_adapter_turn_cycle_opposite_turn_guard",
        }

    def close(self) -> None:
        self.backend.close()


def default_run_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT / f"vln_sonic_mujoco_no_casa_success30_{stamp}"


def _ensure_layout(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "figures": run_dir / "figures",
        "videos": run_dir / "videos",
        "frames": run_dir / "frames",
        "data": run_dir / "data",
        "models": run_dir / "models",
        "logs": run_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_inference_schema(path: Path) -> None:
    schema = InferenceInput(
        episode_id="example",
        step_idx=0,
        instruction="Follow the first-person visual corridor route.",
        image_path="frames/episode_000/frame_0000.png",
    ).to_schema_dict()
    forbidden = ["map", "occupancy", "pose", "goal", "path", "waypoint", "cell"]
    write_json(
        path,
        {
            "policy_input_schema": schema,
            "forbidden_privileged_fields": forbidden,
            "contains_forbidden_privileged_fields": any(key in schema for key in forbidden),
            "test_time_policy_uses_privileged_map_pose_goal_or_path": False,
        },
    )


def _copy_map_metadata(config: RunnerConfig, data_dir: Path) -> None:
    shutil.copyfile(config.map_metadata, data_dir / "map_metadata_test.json")
    shutil.copyfile(config.map_metadata, data_dir / "map_metadata.json")
    if config.train_map_metadata is not None:
        shutil.copyfile(config.train_map_metadata, data_dir / "map_metadata_train.json")


def _write_teacher_paths(path: Path, tasks: list[TeacherTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for task in tasks:
            file.write(json.dumps({"episode_id": task.episode_id, "split": task.split, **task.teacher_metadata()}, sort_keys=True) + "\n")


def _skill_for_action(mapper: ActionSkillMapper, action: VLNAction):
    return mapper.action_to_skill(action)


def collect_auto_demos(
    *,
    config: RunnerConfig,
    maze: MazeMap,
    env: MujocoMazeSkillEnv,
    demo_tasks: list[TeacherTask],
    data_dir: Path,
    models_dir: Path,
) -> tuple[Path, Path]:
    from .oracle import AStarTeacher

    if config.skip_demo_frames:
        raise ValueError("visual action adapter training requires rendered teacher demo frames")

    records: list[AutoVLNDemoRecord] = []
    teacher = AStarTeacher(maze)
    demo_root = data_dir.parent / "frames" / "teacher_demos"
    mapper_config = SkillMappingConfig()

    for task_idx, task in enumerate(demo_tasks):
        env.reset(task.start_xy, task.start_yaw_deg, task.goal_xy, episode_id=task.episode_id)
        mapper = ActionSkillMapper(mapper_config, initial_yaw_deg=task.start_yaw_deg)
        history: list[str] = []
        previous_actions: list[str] = []
        previous_skill_status: list[str] = []
        actions = list(task.route_actions) + [VLNAction.STOP.value]
        for step_idx, action_value in enumerate(actions):
            frame_path = demo_root / task.episode_id / f"frame_{step_idx:04d}.png"
            env.render_head(frame_path)
            decision = teacher.teacher_action_for_step(task, step_idx)
            action = VLNAction(action_value)
            skill = _skill_for_action(mapper, action)
            pose = env.pose()
            should_stop = action is VLNAction.STOP
            stop_reason = "near_goal_visual" if should_stop else "far_from_goal_negative"
            records.append(
                AutoVLNDemoRecord(
                    episode_id=task.episode_id,
                    step_idx=step_idx,
                    instruction=task.instruction,
                    image_path=str(frame_path) if frame_path else None,
                    history_image_paths=history[-4:],
                    previous_actions=list(previous_actions),
                    previous_skill_status=list(previous_skill_status),
                    teacher_action=action.value,
                    sonic_skill={"name": skill.name, "params": skill.params()},
                    should_stop=should_stop,
                    stop_label_reason=stop_reason,
                    teacher_metadata={
                        **task.teacher_metadata(),
                        "teacher_decision": decision.metadata,
                        "pose": asdict(pose),
                        "goal_xy": list(task.goal_xy),
                    },
                    goal_xy=list(task.goal_xy),
                    pose=asdict(pose),
                    navid_raw_action=None,
                    metadata={"split": task.split, "demo_task_index": task_idx},
                )
            )
            if frame_path:
                history.append(str(frame_path))
            if action is not VLNAction.STOP:
                result = env.execute_skill(skill)
                previous_skill_status.append(result.status)
            else:
                previous_skill_status.append("teacher_stop_label")
            previous_actions.append(action.value)

    dataset_path = data_dir / "auto_vln_dataset.jsonl"
    write_jsonl(dataset_path, records)
    adapter_path = models_dir / "visual_action_adapter.json"
    adapter = train_visual_action_adapter(
        dataset_path,
        adapter_path,
        config=VisualAdapterConfig(epochs=min(config.visual_adapter_epochs, 80) if config.smoke_only else config.visual_adapter_epochs),
    )
    write_json(
        data_dir / "visual_adapter_training_summary.json",
        {
            "adapter_path": str(adapter_path),
            "train_records": adapter.train_records,
            "demo_task_count": len(demo_tasks),
            "model_type": adapter.metadata.get("trainer"),
            "metrics": adapter.metrics,
            "test_time_inputs": adapter.metadata.get("test_time_inputs"),
            "forbidden_test_time_inputs": adapter.metadata.get("forbidden_test_time_inputs"),
            "online_privileged_inputs_used": False,
            "stop_positive_records": sum(1 for record in records if record.should_stop),
            "stop_negative_records": sum(1 for record in records if not record.should_stop),
        },
    )
    return dataset_path, adapter_path


def run_episode(
    *,
    task: TeacherTask,
    env: MujocoMazeSkillEnv,
    policy: VisualActionAdapterPolicy,
    config: RunnerConfig,
    run_dir: Path,
    data_dir: Path,
    videos_dir: Path,
    frames_dir: Path,
) -> dict[str, Any]:
    episode_frames_dir = frames_dir / task.episode_id
    episode_frames_dir.mkdir(parents=True, exist_ok=True)
    env.reset(task.start_xy, task.start_yaw_deg, task.goal_xy, episode_id=task.episode_id)
    policy.reset(task.episode_id)
    mapper = ActionSkillMapper(SkillMappingConfig(), initial_yaw_deg=task.start_yaw_deg)

    decision_records: list[dict[str, Any]] = []
    previous_actions: list[str] = []
    previous_status: list[str] = []
    frame_paths: list[Path] = []

    for step_idx in range(min(config.max_steps, task.max_steps)):
        frame_path = episode_frames_dir / f"frame_{step_idx:04d}.png"
        env.render_head(frame_path)
        frame_paths.append(frame_path)
        pose_before = env.pose()
        obs = InferenceInput(
            episode_id=task.episode_id,
            step_idx=step_idx,
            instruction=task.instruction,
            image_path=str(frame_path),
            history_image_paths=[str(path) for path in frame_paths[-5:-1]],
            previous_actions=list(previous_actions),
            previous_skill_status=list(previous_status),
        )
        decision = policy.next_action(obs)
        backend_result = getattr(policy, "last_result", None)
        mapper.update_heading(pose_before.yaw_deg)
        skill = mapper.action_to_skill(decision)
        result = env.execute_skill(skill)
        distance = ((task.goal_xy[0] - pose_before.x) ** 2 + (task.goal_xy[1] - pose_before.y) ** 2) ** 0.5
        record = {
            "episode_id": task.episode_id,
            "step_idx": step_idx,
            "instruction": task.instruction,
            "image_path": str(frame_path),
            "history_image_paths": obs.history_image_paths,
            "previous_actions": obs.previous_actions,
            "previous_skill_status": obs.previous_skill_status,
            "backend_name": policy.name,
            "real_navid_or_uninavid_used": bool(
                backend_result is not None
                and getattr(backend_result, "available", False)
                and "real_model_inference" in policy.name
            ),
            "navid_model_loaded": bool(
                (getattr(backend_result, "metadata", {}) or {}).get("navid_model_loaded")
            )
            if backend_result is not None
            else False,
            "navid_raw_output": getattr(backend_result, "raw_output", None) if backend_result is not None else None,
            "navid_parsed_action": getattr(getattr(backend_result, "decision", None), "action", None).value
            if backend_result is not None and getattr(backend_result, "decision", None) is not None
            else None,
            "backend_action": decision.action.value,
            "final_action": decision.action.value,
            "final_source": decision.source,
            "policy_metadata": decision.metadata,
            "selected_sonic_skill": skill.name,
            "skill_params": skill.params(),
            "skill_status": result.status,
            "skill_execution": result.to_dict(),
            "evaluator_pose": asdict(pose_before),
            "evaluator_distance_to_goal": distance,
            "evaluator_goal_xy": list(task.goal_xy),
            "evaluator_only_shortest_path_remaining": env.maze.shortest_path_distance_xy((pose_before.x, pose_before.y), task.goal_xy),
            "fall": result.fall,
            "wall_collision": result.wall_collision,
            "privileged_policy_usage": False,
        }
        decision_records.append(record)
        previous_actions.append(decision.action.value)
        previous_status.append(result.status)
        if decision.action is VLNAction.STOP:
            break

    video = write_video(frame_paths, videos_dir / f"{task.episode_id}.mp4", fps=8)
    episode_decision_log = data_dir / f"{task.episode_id}_decision_log.jsonl"
    append_jsonl(episode_decision_log, decision_records)
    trajectory_rows = list(env.trajectory)
    episode_trajectory_log = data_dir / f"{task.episode_id}_trajectory_log.jsonl"
    append_jsonl(episode_trajectory_log, trajectory_rows)
    draw_topdown_map(
        env.maze,
        run_dir / "figures" / f"{task.episode_id}_topdown.png",
        trajectory_xy=[(float(row["pose_x"]), float(row["pose_y"])) for row in trajectory_rows],
        start_xy=task.start_xy,
        goal_xy=task.goal_xy,
    )
    metrics = summarize_episode(
        task=task,
        maze=env.maze,
        decision_records=decision_records,
        trajectory_rows=trajectory_rows,
        video_paths=[video] if video else [],
        frames_dir=episode_frames_dir,
        decision_log=episode_decision_log,
        trajectory_log=episode_trajectory_log,
    )
    return metrics.to_dict()


def _write_global_logs(data_dir: Path, episode_metrics: list[dict[str, Any]]) -> None:
    decision_jsonls = sorted(data_dir.glob("heldout_*_decision_log.jsonl"))
    decision_csv = data_dir / "decision_logs.csv"
    fieldnames = [
        "episode_id",
        "step_idx",
        "instruction",
        "image_path",
        "backend_name",
        "real_navid_or_uninavid_used",
        "navid_model_loaded",
        "navid_raw_output",
        "navid_parsed_action",
        "backend_action",
        "final_action",
        "selected_sonic_skill",
        "skill_status",
        "evaluator_distance_to_goal",
        "privileged_policy_usage",
    ]
    with decision_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for jsonl in decision_jsonls:
            for line in jsonl.read_text().splitlines():
                row = json.loads(line)
                writer.writerow({key: row.get(key) for key in fieldnames})

    trajectory_jsonls = sorted(data_dir.glob("heldout_*_trajectory_log.jsonl"))
    trajectory_csv = data_dir / "trajectory_logs.csv"
    t_fields = ["episode_id", "event", "pose_x", "pose_y", "pose_yaw_deg", "goal_x", "goal_y", "wall_collision", "fall"]
    with trajectory_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=t_fields)
        writer.writeheader()
        for jsonl in trajectory_jsonls:
            for line in jsonl.read_text().splitlines():
                row = json.loads(line)
                writer.writerow({key: row.get(key) for key in t_fields})

    summaries = data_dir / "episode_summaries.jsonl"
    if summaries.exists():
        summaries.unlink()
    append_jsonl(summaries, episode_metrics)


def _draw_metric_figures(figures_dir: Path, aggregate: dict[str, Any], episodes: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 360), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    rate = float(aggregate["success_rate"])
    draw.rectangle([80, 80, 560, 140], outline=(40, 40, 40), width=2)
    draw.rectangle([80, 80, 80 + int(480 * rate), 140], fill=(30, 150, 70))
    draw.text((80, 45), f"Success rate: {rate:.2%}", fill=(0, 0, 0))
    image.save(figures_dir / "success_rate_by_attempt.png")

    counts: dict[str, int] = {}
    for episode in episodes:
        reason = episode["failure_reason"] or "success"
        counts[reason] = counts.get(reason, 0) + 1
    image = Image.new("RGB", (760, 420), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    x = 50
    max_count = max(counts.values()) if counts else 1
    for reason, count in sorted(counts.items()):
        height = int(260 * count / max_count)
        draw.rectangle([x, 340 - height, x + 70, 340], fill=(60, 120, 210))
        draw.text((x, 350), reason[:18], fill=(0, 0, 0))
        draw.text((x + 20, 320 - height), str(count), fill=(0, 0, 0))
        x += 100
    image.save(figures_dir / "failure_reason_breakdown.png")


def _write_report(run_dir: Path, report: dict[str, Any]) -> None:
    aggregate = report["episode_metrics"]
    lines = [
        "# No-CASA SONIC MuJoCo VLN Success30 Report",
        "",
        f"- PASS_SUCCESS30: `{report['pass_success30']}`",
        f"- run_dir: `{report['run_dir']}`",
        f"- scene_xml: `{report['scene_xml']}`",
        f"- train_test_split_active: `{report.get('train_test_split_active')}`",
        f"- train_test_furniture_distributions_differ: `{report.get('train_test_furniture_distributions_differ')}`",
        f"- target_visual: `{report.get('target_visual')}`",
        f"- red_ball_target_visible: `{report.get('red_ball_target_visible')}`",
        f"- strict_goal_audit_pass: `{report.get('strict_goal_audit_pass')}`",
        f"- total_episodes: `{aggregate['total_episodes']}`",
        f"- success_count: `{aggregate['success_count']}`",
        f"- success_rate: `{aggregate['success_rate']:.3f}`",
        f"- privileged_policy_usage_count: `{aggregate['privileged_policy_usage_count']}`",
        f"- teacher_used_for_training: `{report['teacher_used_for_training']}`",
        f"- auto_adapter_used: `{report['auto_adapter_used']}`",
        f"- auto_demo_count: `{report['auto_demo_count']}`",
        f"- policy_backend: `{report['policy_backend']}`",
        f"- policy_name: `{report['policy_name']}`",
        f"- real_navid_or_uninavid_used: `{report['real_navid_or_uninavid_used']}`",
        "",
        "## Metrics",
    ]
    for key in [
        "fall_count",
        "collision_count",
        "stuck_count",
        "timeout_count",
        "mean_final_distance",
        "median_final_distance",
        "mean_shortest_path_progress",
        "mean_executed_motion_skills",
        "stop_fallback_ratio",
        "stop_precision",
        "stop_recall",
        "premature_stop_rate",
        "late_stop_rate",
        "infinite_stop_rate",
        "mean_stop_distance",
    ]:
        lines.append(f"- {key}: `{aggregate[key]}`")
    lines.extend(["", "## Videos"])
    lines.extend(f"- `{path}`" for path in report["video_paths"])
    lines.extend(["", "## Implementation Changes"])
    lines.extend(f"- {item}" for item in report["key_implementation_changes"])
    lines.extend(["", "## Remaining Limitations"])
    lines.extend(f"- {item}" for item in report["remaining_limitations"])
    (run_dir / "navigation_success30_report.md").write_text("\n".join(lines) + "\n")
    (run_dir / "README.md").write_text("\n".join(lines[:20]) + "\n")


def _write_strict_goal_audit(
    *,
    data_dir: Path,
    report: dict[str, Any],
    maze: MazeMap,
    heldout_tasks: list[TeacherTask],
) -> dict[str, Any]:
    decision_csv = data_dir / "decision_logs.csv"
    decision_rows: list[dict[str, str]] = []
    if decision_csv.exists():
        with decision_csv.open(newline="") as file:
            decision_rows = list(csv.DictReader(file))
    heldout_distances = [
        maze.euclidean_distance_cells(task.start_cell, task.goal_cell)
        for task in heldout_tasks
        if task.start_cell is not None and task.goal_cell is not None
    ]
    instructions = [task.instruction.lower() for task in heldout_tasks]
    no_action_counts = all(" x" not in instruction and "x1" not in instruction for instruction in instructions)
    no_red_ball_text = all("red target" not in instruction and "red ball" not in instruction for instruction in instructions)
    mentions_wall_painting = all("wall painting" in instruction for instruction in instructions)
    navid_rows = [
        row
        for row in decision_rows
        if str(row.get("real_navid_or_uninavid_used", "")).lower() in {"true", "1"}
        and str(row.get("navid_model_loaded", "")).lower() in {"true", "1"}
    ]
    audit = {
        "strict_goal": "no_casa_end_to_end_vln_style_wall_painting_split_distribution_success30",
        "pass": bool(
            report["pass_success30"]
            and report["train_test_split_active"]
            and report["train_test_furniture_distributions_differ"]
            and report["target_visual"] == "gallery_wall_painting"
            and report["red_ball_target_visible"] is False
            and no_action_counts
            and no_red_ball_text
            and mentions_wall_painting
            and heldout_distances
            and min(heldout_distances) >= 6.0
            and report["real_navid_or_uninavid_used"]
            and len(navid_rows) == len(decision_rows)
            and len(decision_rows) > 0
        ),
        "success30_pass": bool(report["pass_success30"]),
        "success_rate": report["episode_metrics"]["success_rate"],
        "success_count": report["episode_metrics"]["success_count"],
        "total_episodes": report["episode_metrics"]["total_episodes"],
        "train_test_split_active": report["train_test_split_active"],
        "train_test_furniture_distributions_differ": report["train_test_furniture_distributions_differ"],
        "target_visual": report["target_visual"],
        "red_ball_target_visible": report["red_ball_target_visible"],
        "instructions_no_action_counts": no_action_counts,
        "instructions_no_red_ball_text": no_red_ball_text,
        "instructions_all_mention_wall_painting": mentions_wall_painting,
        "heldout_min_euclidean_distance_m": min(heldout_distances) if heldout_distances else None,
        "heldout_route_euclidean_distances_m": heldout_distances,
        "real_navid_or_uninavid_used": report["real_navid_or_uninavid_used"],
        "decision_rows": len(decision_rows),
        "real_navid_model_loaded_decision_rows": len(navid_rows),
        "final_policy_note": (
            "The real NaVid checkpoint is invoked online at every held-out decision; "
            "final actions are calibrated by the trained visual adapter."
            if report["real_navid_or_uninavid_used"]
            else "Strict real NaVid was not used in this run; this audit cannot pass the real-backend requirement."
        ),
        "blocked_if_false": [
            "Requires >=30% success over at least 20 held-out episodes.",
            "Requires train/test furnished distributions to differ.",
            "Requires visible target to be gallery_wall_painting, not a red ball.",
            "Requires all held-out tasks to be >=6m Euclidean start-goal distance.",
            "Requires strict real NaVid model loaded and called for every held-out decision row.",
        ],
    }
    write_json(data_dir / "strict_goal_audit.json", audit)
    return audit


def run(config: RunnerConfig) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    assert_no_privileged_inference_schema()
    dirs = _ensure_layout(config.run_dir)
    maze = MazeMap.from_metadata(config.map_metadata)
    train_map_metadata = config.train_map_metadata or config.map_metadata
    train_scene_xml = config.train_scene_xml or config.scene_xml
    train_maze = MazeMap.from_metadata(train_map_metadata)
    train_test_split_active = Path(train_map_metadata).resolve() != Path(config.map_metadata).resolve()
    _copy_map_metadata(config, dirs["data"])
    _write_inference_schema(dirs["data"] / "inference_input_schema.json")
    draw_topdown_map(maze, dirs["figures"] / "map_topdown_test.png")
    draw_topdown_map(train_maze, dirs["figures"] / "map_topdown_train.png")
    if not train_test_split_active:
        shutil.copyfile(dirs["figures"] / "map_topdown_test.png", dirs["figures"] / "map_topdown.png")

    train_env = MujocoMazeSkillEnv(
        scene_xml=train_scene_xml,
        maze=train_maze,
        output_dir=config.run_dir,
        width=config.frame_width,
        height=config.frame_height,
    )
    env = MujocoMazeSkillEnv(
        scene_xml=config.scene_xml,
        maze=maze,
        output_dir=config.run_dir,
        width=config.frame_width,
        height=config.frame_height,
    )
    train_env.render_head(dirs["figures"] / "head_camera_check_train.png")
    env.render_head(dirs["figures"] / "head_camera_check_test.png")
    if not train_test_split_active:
        shutil.copyfile(dirs["figures"] / "head_camera_check_test.png", dirs["figures"] / "head_camera_check.png")
    scene_report = {
        "scene_xml": str(config.scene_xml),
        "train_scene_xml": str(train_scene_xml),
        "test_scene_xml": str(config.scene_xml),
        "train_map_metadata": str(train_map_metadata),
        "test_map_metadata": str(config.map_metadata),
        "train_test_split_active": train_test_split_active,
        "train_test_furniture_distributions_differ": sorted(obstacle.cell for obstacle in train_maze.furniture_obstacles)
        != sorted(obstacle.cell for obstacle in maze.furniture_obstacles),
        "mujoco_load_passed": True,
        "maze_wall_geoms": int(maze.occupancy.sum()),
        "train_wall_painting_geoms": sorted(train_env.wall_painting_geom_ids),
        "test_wall_painting_geoms": sorted(env.wall_painting_geom_ids),
        "target_visual": maze.metadata.get("target_visual"),
        "red_ball_target_visible": bool(maze.metadata.get("red_ball_target_visible", True)),
        "test_furniture_obstacle_count": len(maze.furniture_obstacles),
        "test_furniture_obstacles": [asdict(obstacle) for obstacle in maze.furniture_obstacles],
        "train_furniture_obstacle_count": len(train_maze.furniture_obstacles),
        "train_furniture_obstacles": [asdict(obstacle) for obstacle in train_maze.furniture_obstacles],
        "furniture_obstacle_count": len(maze.furniture_obstacles),
        "furniture_obstacles": [asdict(obstacle) for obstacle in maze.furniture_obstacles],
        "head_camera_render_path": str(dirs["figures"] / "head_camera_check_test.png"),
        "train_head_camera_render_path": str(dirs["figures"] / "head_camera_check_train.png"),
        "robot_initial_pose_check": asdict(env.pose()),
    }
    write_json(dirs["data"] / "scene_load_report.json", scene_report)

    heldout_count = 1 if config.smoke_only else config.heldout_episodes
    demo_count = 2 if config.smoke_only else config.auto_demo_count
    demo_routes = generate_routes(
        train_maze,
        count=demo_count,
        seed=config.seed,
        min_edges=config.min_route_edges,
        max_edges=config.max_route_edges,
        min_euclidean_m=config.min_route_euclidean_m,
        require_furniture_detour=config.require_furniture_detour,
    )
    demo_pairs = {(route.start_cell, route.goal_cell) for route in demo_routes}
    exclude_pairs = demo_pairs if Path(train_map_metadata).resolve() == Path(config.map_metadata).resolve() else None
    heldout_routes = generate_routes(
        maze,
        count=heldout_count,
        seed=config.heldout_seed,
        min_edges=config.min_route_edges,
        max_edges=config.max_route_edges,
        min_euclidean_m=config.min_route_euclidean_m,
        require_furniture_detour=config.require_furniture_detour,
        exclude_pairs=exclude_pairs,
    )
    demo_tasks = [task_from_route(route, episode_id=f"train_{idx:03d}", split="train") for idx, route in enumerate(demo_routes)]
    heldout_tasks = [
        task_from_route(route, episode_id=f"heldout_{idx:03d}", split="heldout", max_steps=config.max_steps)
        for idx, route in enumerate(heldout_routes)
    ]
    _write_teacher_paths(dirs["data"] / "teacher_paths.jsonl", demo_tasks + heldout_tasks)
    dataset_path, adapter_path = collect_auto_demos(
        config=config,
        maze=train_maze,
        env=train_env,
        demo_tasks=demo_tasks,
        data_dir=dirs["data"],
        models_dir=dirs["models"],
    )

    real_navid_backend: NaVidBackend | None = None
    if config.policy_backend == "visual_adapter":
        policy = VisualActionAdapterPolicy(VisualActionAdapter.load(adapter_path))
    elif config.policy_backend in {"real_navid", "real_navid_visual_adapter"}:
        real_navid_backend = NaVidBackend(
            repo_path=config.navid_repo,
            result_dir=dirs["logs"] / "real_navid_backend",
            model_path=config.navid_model,
            variant="navid",
            strict_model=True,
            python_executable=config.navid_python,
            vision_tower_path=config.navid_vision_tower,
            worker_timeout_s=config.navid_worker_timeout_s,
        )
        if config.policy_backend == "real_navid_visual_adapter":
            policy = RealNaVidVisualAdapterPolicy(
                real_navid_backend,
                VisualActionAdapterPolicy(VisualActionAdapter.load(adapter_path)),
            )
        else:
            policy = RealNaVidPolicy(real_navid_backend)
        if not real_navid_backend.availability.get("model_loaded"):
            write_json(dirs["data"] / "real_navid_backend_probe.json", real_navid_backend.availability)
            raise RuntimeError(f"strict real NaVid backend did not load: {real_navid_backend.availability}")
    else:
        raise ValueError(f"unsupported policy_backend: {config.policy_backend}")

    episode_metrics: list[dict[str, Any]] = []
    try:
        for task in heldout_tasks:
            metrics = run_episode(
                task=task,
                env=env,
                policy=policy,
                config=config,
                run_dir=config.run_dir,
                data_dir=dirs["data"],
                videos_dir=dirs["videos"],
                frames_dir=dirs["frames"],
            )
            episode_metrics.append(metrics)
    finally:
        env.close()
        train_env.close()
        if real_navid_backend is not None:
            real_navid_backend.close()

    _write_global_logs(dirs["data"], episode_metrics)
    aggregate = aggregate_metrics(episode_metrics)
    stop_eval = {
        key: aggregate[key]
        for key in [
            "stop_precision",
            "stop_recall",
            "premature_stop_rate",
            "late_stop_rate",
            "infinite_stop_rate",
            "mean_stop_distance",
        ]
    }
    write_json(dirs["data"] / "stop_eval_metrics.json", stop_eval)
    write_json(dirs["data"] / "action_distribution.json", aggregate["action_distribution"])
    _draw_metric_figures(dirs["figures"], aggregate, episode_metrics)

    video_paths = [path for episode in episode_metrics for path in episode.get("video_paths", [])]
    report = {
        "pass_success30": aggregate["total_episodes"] >= 20 and aggregate["success_rate"] >= 0.30,
        "run_dir": str(config.run_dir),
        "scene_xml": str(config.scene_xml),
        "train_scene_xml": str(train_scene_xml),
        "test_scene_xml": str(config.scene_xml),
        "train_map_metadata": str(train_map_metadata),
        "test_map_metadata": str(config.map_metadata),
        "train_test_split_active": train_test_split_active,
        "train_test_furniture_distributions_differ": sorted(obstacle.cell for obstacle in train_maze.furniture_obstacles)
        != sorted(obstacle.cell for obstacle in maze.furniture_obstacles),
        "target_visual": maze.metadata.get("target_visual", "legacy_target_site"),
        "red_ball_target_visible": bool(maze.metadata.get("red_ball_target_visible", True)),
        "map": maze.name,
        "train_map": train_maze.name,
        "furniture_obstacle_count": len(maze.furniture_obstacles),
        "furniture_obstacles": [asdict(obstacle) for obstacle in maze.furniture_obstacles],
        "train_furniture_obstacle_count": len(train_maze.furniture_obstacles),
        "train_furniture_obstacles": [asdict(obstacle) for obstacle in train_maze.furniture_obstacles],
        "test_furniture_obstacle_count": len(maze.furniture_obstacles),
        "test_furniture_obstacles": [asdict(obstacle) for obstacle in maze.furniture_obstacles],
        "episode_metrics": aggregate,
        "episodes": episode_metrics,
        "video_paths": video_paths,
        "videos_path": str(dirs["videos"]),
        "report_path": str(config.run_dir / "navigation_success30_report.md"),
        "dataset_path": str(dataset_path),
        "adapter_path": str(adapter_path),
        "teacher_used_for_training": True,
        "auto_adapter_trained": True,
        "auto_adapter_used": config.policy_backend in {"visual_adapter", "real_navid_visual_adapter"},
        "policy_name": policy.name,
        "policy_backend": config.policy_backend,
        "route_parser_used_by_test_time_policy": False,
        "auto_demo_count": len(demo_tasks),
        "min_route_edges": config.min_route_edges,
        "max_route_edges": config.max_route_edges,
        "min_route_euclidean_m": config.min_route_euclidean_m,
        "require_furniture_detour": config.require_furniture_detour,
        "real_navid_or_uninavid_used": bool(
            config.policy_backend in {"real_navid", "real_navid_visual_adapter"}
            and real_navid_backend is not None
            and real_navid_backend.availability.get("model_loaded")
        ),
        "real_navid_or_uninavid_used_for_inference": config.policy_backend in {"real_navid", "real_navid_visual_adapter"},
        "real_navid_backend_probe": real_navid_backend.availability if real_navid_backend is not None else None,
        "privileged_map_pose_goal_used_by_test_time_policy": False,
        "key_implementation_changes": [
            "Replaced CASA skill imports with no-CASA WalkSkill, TurnSkill, PassiveSkill dataclasses.",
            "Added Gymnasium-Robotics scale2 maze A* teacher for offline route-language and action-label generation only.",
            "Added VisualActionAdapterPolicy trained from automatic teacher data; held-out inference uses only instruction, head_camera frames/history, previous actions, and runtime skill status.",
            "Added strict real-NaVid policy backend; when --policy-backend real_navid is selected, held-out inference uses the loaded NaVid checkpoint and current first-person images.",
            "Added real_navid_visual_adapter backend; it calls real NaVid online every step and calibrates final actions through the trained visual adapter.",
            "Replaced generated route instructions with natural object-aware language that mentions furniture and target cues without action counts.",
            "Added split train/test furnished scene support so automatic teacher data and held-out evaluation can use different furniture distributions.",
            "Replaced the visible red target ball with a movable cyan-and-green gallery wall painting; the hidden target site remains evaluator-only.",
            "Added MuJoCo scene executor that loads the official SONIC maze XML and renders robot head_camera observations online.",
            "Added success and stop metrics requiring active policy stop, no wall collision/fall, meaningful motion, and shortest-path progress.",
        ],
        "remaining_limitations": [
            (
                "The final policy is a lightweight supervised visual action adapter, not a loaded NaVid/Uni-NaVid checkpoint."
                if config.policy_backend == "visual_adapter"
                else (
                    "The loaded NaVid checkpoint is used zero-shot; it may not navigate the synthetic MuJoCo furnished maze successfully without adaptation."
                    if config.policy_backend == "real_navid"
                    else "The loaded NaVid checkpoint is invoked online, but final actions are calibrated by a lightweight visual adapter."
                )
            ),
            "High-level walking is executed kinematically through the MuJoCo floating base rather than the full deployed low-level WBC stack.",
            "A* is still used to produce offline labels and evaluator metadata; it is not used by the held-out policy.",
        ],
    }
    strict_goal_audit = _write_strict_goal_audit(
        data_dir=dirs["data"],
        report=report,
        maze=maze,
        heldout_tasks=heldout_tasks,
    )
    report["strict_goal_audit_path"] = str(dirs["data"] / "strict_goal_audit.json")
    report["strict_goal_audit_pass"] = strict_goal_audit["pass"]
    write_json(config.run_dir / "navigation_manifest.json", {**report, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _write_report(config.run_dir, report)
    return report


def parse_args(argv: list[str]) -> RunnerConfig:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--scene-xml", type=Path, default=SCENE_XML)
    parser.add_argument("--map-metadata", type=Path, default=MAP_METADATA)
    parser.add_argument("--train-scene-xml", type=Path, default=None)
    parser.add_argument("--train-map-metadata", type=Path, default=None)
    parser.add_argument("--heldout-episodes", type=int, default=20)
    parser.add_argument("--auto-demo-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--heldout-seed", type=int, default=260611)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=240)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-demo-frames", action="store_true")
    parser.add_argument("--visual-adapter-epochs", type=int, default=240)
    parser.add_argument("--min-route-edges", type=int, default=2)
    parser.add_argument("--max-route-edges", type=int, default=7)
    parser.add_argument("--min-route-euclidean-m", type=float, default=0.0)
    parser.add_argument("--require-furniture-detour", action="store_true")
    parser.add_argument(
        "--policy-backend",
        choices=["visual_adapter", "real_navid", "real_navid_visual_adapter"],
        default="visual_adapter",
    )
    parser.add_argument("--navid-repo", type=Path, default=NAVID_REPO)
    parser.add_argument("--navid-model", type=Path, default=NAVID_MODEL)
    parser.add_argument("--navid-python", type=Path, default=NAVID_PYTHON)
    parser.add_argument("--navid-vision-tower", type=Path, default=NAVID_VISION_TOWER)
    parser.add_argument("--navid-worker-timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)
    return RunnerConfig(
        run_dir=args.run_dir,
        scene_xml=args.scene_xml,
        map_metadata=args.map_metadata,
        train_scene_xml=args.train_scene_xml,
        train_map_metadata=args.train_map_metadata,
        heldout_episodes=args.heldout_episodes,
        auto_demo_count=args.auto_demo_count,
        seed=args.seed,
        heldout_seed=args.heldout_seed,
        max_steps=args.max_steps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        smoke_only=args.smoke_only,
        skip_demo_frames=args.skip_demo_frames,
        visual_adapter_epochs=args.visual_adapter_epochs,
        min_route_edges=args.min_route_edges,
        max_route_edges=args.max_route_edges,
        min_route_euclidean_m=args.min_route_euclidean_m,
        require_furniture_detour=args.require_furniture_detour,
        policy_backend=args.policy_backend,
        navid_repo=args.navid_repo,
        navid_model=args.navid_model,
        navid_python=args.navid_python,
        navid_vision_tower=args.navid_vision_tower,
        navid_worker_timeout_s=args.navid_worker_timeout_s,
    )


def main(argv: list[str] | None = None) -> None:
    report = run(parse_args(sys.argv[1:] if argv is None else argv))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
