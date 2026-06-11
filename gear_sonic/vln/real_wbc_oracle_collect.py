"""Training-only real-WBC VLN frame/action collector.

This module may use the privileged MuJoCo pose and an A* teacher to label
automatic demonstrations.  It is intentionally separate from the final online
runner: generated adapters must still be evaluated by
``real_navid_online.py`` without map/pose/goal/path inputs at test time.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.casa.skills import PassiveSkill, SkillExecutor
from gear_sonic.vln.actions import VLNAction
from gear_sonic.vln.dataset import AutoVLNDemoRecord, write_jsonl
from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.no_casa_policy import InferenceInput
from gear_sonic.vln.oracle import AStarTeacher, RobotPose2D
from gear_sonic.vln.real_navid_online import (
    ContinuousCameraRecorder,
    OnlineConfig,
    RealSonicSkillMapper,
    _allocate_ports,
    _distance_xy,
    _latest_sim_row,
    _make_dirs,
    _robot_unstable,
    _safe_float,
    _sim_summary,
    _wait_for_camera_frame,
    _wait_for_elastic_disabled,
    _write_video,
    write_target_scene_variant,
)
from gear_sonic.vln.stack import launch_sonic_stack, wait_for_file, wait_for_log_contains
from gear_sonic.vln.tasks import TeacherTask
from gear_sonic.vln.visual_adapter import VisualAdapterConfig, train_visual_action_adapter


OUTPUT_ROOT = Path("/mnt/data/students/lph/recording")
SCENE_XML = Path(
    "/mnt/data/students/lph/recording/gymnasium_robotics_maze_maps_scale2_20260610_233102/"
    "mjcf/sonic_scene_MEDIUM_MAZE_DIVERSE_GR_scale2_official_gymnasium.xml"
)
MAP_METADATA = Path(
    "/mnt/data/students/lph/recording/gymnasium_robotics_maze_maps_scale2_20260610_233102/"
    "data/MEDIUM_MAZE_DIVERSE_GR_scale2.json"
)


@dataclass(frozen=True)
class RealWBCOracleCollectConfig:
    run_dir: Path
    scene_xml: Path = SCENE_XML
    map_metadata: Path = MAP_METADATA
    instruction: str = (
        "Use the first-person view to move through the maze corridor to the far red target ball. "
        "Stop only when the red ball is close and centered."
    )
    episode_id: str = "real_wbc_oracle_collect_000"
    max_steps: int = 48
    camera_port: int = 0
    zmq_port: int = 0
    domain_id: int = 0
    target_x: float = 5.0
    target_y: float = -5.0
    target_z: float = 0.5
    head_camera_pos: str = "0.18 0 0.42"
    head_camera_euler: str = "0 -1.0 -1.57"
    publish_fps: float = 8.0
    warmup_seconds: float = 4.0
    drop_on_start: bool = False
    drop_after_seconds: float | None = 35.0
    disable_reset_on_fall: bool = True
    keep_stack_on_exit: bool = False
    frame_width: int = 480
    frame_height: int = 360
    forward_duration: float = 1.4
    turn_duration: float = 1.6
    backoff_duration: float = 0.8
    stop_duration: float = 0.5
    walk_speed: float = 0.25
    turn_degrees: float = 30.0
    success_radius: float = 0.8
    teacher_stop_radius: float = 0.75
    min_motion_m: float = 2.0
    min_start_distance: float = 6.0
    record_continuous_video: bool = True
    continuous_frame_save_fps: float = 8.0
    continuous_video_fps: float = 8.0
    train_adapter: bool = True
    visual_adapter_epochs: int = 240
    visual_adapter_validation_fraction: float = 0.0


def default_run_dir() -> Path:
    return OUTPUT_ROOT / f"vln_real_wbc_oracle_collect_{time.strftime('%Y%m%d_%H%M%S')}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def _yaw_degrees(row: dict[str, Any]) -> float:
    yaw = _safe_float(row.get("torso_yaw"))
    if yaw is None:
        return 0.0
    # MuJoCo sim_state stores torso_yaw in radians.
    return math.degrees(yaw)


def _pose_from_row(row: dict[str, Any]) -> RobotPose2D:
    return RobotPose2D(
        x=float(_safe_float(row.get("base_pos_x")) or 0.0),
        y=float(_safe_float(row.get("base_pos_y")) or 0.0),
        yaw_deg=_yaw_degrees(row),
    )


def _build_teacher_task(
    *,
    config: RealWBCOracleCollectConfig,
    maze: MazeMap,
    start_row: dict[str, Any],
) -> TeacherTask:
    start_x = float(_safe_float(start_row.get("base_pos_x")) or 0.0)
    start_y = float(_safe_float(start_row.get("base_pos_y")) or 0.0)
    start_cell = maze.xy_to_cell(start_x, start_y)
    goal_cell = maze.xy_to_cell(config.target_x, config.target_y)
    path_cells = tuple(maze.astar(start_cell, goal_cell))
    return TeacherTask(
        episode_id=config.episode_id,
        instruction=config.instruction,
        start_xy=(start_x, start_y),
        goal_xy=(config.target_x, config.target_y),
        start_yaw_deg=_yaw_degrees(start_row),
        success_radius=config.success_radius,
        max_steps=config.max_steps,
        visual_cues=["corridor openings", "wall turns", "red target ball"],
        split="train_real_wbc",
        start_cell=start_cell,
        goal_cell=goal_cell,
        route_actions=tuple(),
        path_cells=path_cells,
        stop_xy=(config.target_x, config.target_y),
    )


def _eval_collection(
    *,
    config: RealWBCOracleCollectConfig,
    decision_records: list[dict[str, Any]],
    sim_state_csv: Path,
) -> dict[str, Any]:
    summary = _sim_summary(sim_state_csv)
    if not decision_records:
        return {
            "teacher_rollout_success": False,
            "failure_reason": "no_decision_records",
            "start_distance_requirement_met": False,
        }
    first = decision_records[0]
    last = decision_records[-1]
    start_row = {
        "base_pos_x": first.get("sim_base_pos_x_before"),
        "base_pos_y": first.get("sim_base_pos_y_before"),
    }
    final_row = {"base_pos_x": last.get("sim_base_pos_x"), "base_pos_y": last.get("sim_base_pos_y")}
    start_distance = _distance_xy(start_row, target_x=config.target_x, target_y=config.target_y)
    final_distance = _distance_xy(final_row, target_x=config.target_x, target_y=config.target_y)
    start_x = _safe_float(start_row.get("base_pos_x"))
    start_y = _safe_float(start_row.get("base_pos_y"))
    final_x = _safe_float(final_row.get("base_pos_x"))
    final_y = _safe_float(final_row.get("base_pos_y"))
    motion_m = None
    if None not in {start_x, start_y, final_x, final_y}:
        motion_m = math.hypot(float(final_x) - float(start_x), float(final_y) - float(start_y))
    fall_rows = int(summary.get("fall_rows") or 0)
    final_action = str(last.get("teacher_action") or "")

    reasons: list[str] = []
    if fall_rows:
        reasons.append("fall_rows_nonzero")
    if start_distance is None or start_distance < config.min_start_distance:
        reasons.append("post_drop_first_decision_start_distance_below_min_start_distance")
    if final_distance is None or final_distance > config.success_radius:
        reasons.append("final_distance_above_success_radius")
    if motion_m is None or motion_m < config.min_motion_m:
        reasons.append("motion_below_min_motion_m")
    if final_action != VLNAction.STOP.value:
        reasons.append("teacher_did_not_issue_stop")

    return {
        "teacher_rollout_success": not reasons,
        "failure_reason": ",".join(reasons),
        "action_sequence": [str(record.get("teacher_action") or "") for record in decision_records],
        "start_distance_reference": "first_decision_pre_skill_after_elastic_drop",
        "post_drop_first_decision_start_distance_to_target": start_distance,
        "post_drop_first_decision_start_xy": [start_x, start_y],
        "target_xy": [config.target_x, config.target_y],
        "start_distance_requirement_met": start_distance is not None and start_distance >= config.min_start_distance,
        "final_distance_to_target": final_distance,
        "progress_m": start_distance - final_distance if start_distance is not None and final_distance is not None else None,
        "motion_m": motion_m,
        "success_radius": config.success_radius,
        "min_start_distance": config.min_start_distance,
        "min_motion_m": config.min_motion_m,
        "final_action": final_action,
        "fall_rows": fall_rows,
        "privileged_teacher_used_for_collection": True,
        "eligible_as_final_policy_success": False,
    }


def run(config: RealWBCOracleCollectConfig) -> dict[str, Any]:
    if config.camera_port <= 0 or config.zmq_port <= 0 or config.domain_id <= 0:
        camera_port, zmq_port, domain_id = _allocate_ports()
        config = RealWBCOracleCollectConfig(
            **{**asdict(config), "camera_port": camera_port, "zmq_port": zmq_port, "domain_id": domain_id}
        )

    dirs = _make_dirs(config.run_dir)
    models_dir = config.run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir = dirs["data"]
    scene_variant = write_target_scene_variant(
        config.scene_xml,
        dirs["scenes"] / "episode_000_target_scene.xml",
        target_xyz=(config.target_x, config.target_y, config.target_z),
        head_camera_pos=config.head_camera_pos,
        head_camera_euler=config.head_camera_euler,
    )

    stack = None
    client = None
    executor = None
    recorder = None
    records: list[AutoVLNDemoRecord] = []
    decision_records: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    continuous_frame_paths: list[Path] = []
    blocker = ""
    status = "BLOCKED_REAL_WBC_ORACLE_COLLECT"
    started_at = time.time()

    try:
        stack = launch_sonic_stack(
            run_dir=config.run_dir / "stack",
            robot_scene=scene_variant,
            camera_port=config.camera_port,
            zmq_port=config.zmq_port,
            domain_id=config.domain_id,
            episode_name=config.episode_id,
            image_publish_fps=max(4, int(math.ceil(config.continuous_frame_save_fps))),
            episode_log_fps=20,
            offscreen_camera="head_camera",
            offscreen_camera_width=config.frame_width,
            offscreen_camera_height=config.frame_height,
            drop_on_start=config.drop_on_start,
            drop_after_seconds=config.drop_after_seconds,
            disable_reset_on_fall=config.disable_reset_on_fall,
        )
        sim_state_csv = stack.sim_log_dir / "sim_state.csv"
        if not wait_for_file(sim_state_csv, timeout_s=45.0, min_size=64):
            blocker = "sim_state_not_created"
            return _finalize(config, status, blocker, records, decision_records, frame_paths, dirs, None, started_at)
        if not wait_for_log_contains(stack.run_dir / "logs/deploy_stdout.log", ["Init Done"], timeout_s=120.0):
            blocker = "deploy_not_ready"
            return _finalize(config, status, blocker, records, decision_records, frame_paths, dirs, sim_state_csv, started_at)

        client = ComposedCameraClientSensor(server_ip="localhost", port=config.camera_port, verbose=False)
        executor = SkillExecutor.from_paths(
            output_dir=dirs["skill"],
            sim_log_dir=stack.sim_log_dir,
            run_id=config.episode_id,
            zmq_host="localhost",
            zmq_port=config.zmq_port,
            publish_fps=config.publish_fps,
            dry_run=False,
        )
        executor.start_policy()
        if config.warmup_seconds > 0:
            executor.execute_one(PassiveSkill(duration=config.warmup_seconds, mode="wait"), episode_id=config.episode_id, skill_idx=0)
        if config.drop_after_seconds is not None and not config.drop_on_start:
            if not _wait_for_elastic_disabled(sim_state_csv, executor, timeout_s=90.0):
                blocker = "elastic_band_drop_not_observed"
                return _finalize(config, status, blocker, records, decision_records, frame_paths, dirs, sim_state_csv, started_at)

        pre_decision_row = _latest_sim_row(sim_state_csv)
        if _robot_unstable(pre_decision_row):
            blocker = "pre_decision_robot_unstable"
            return _finalize(config, status, blocker, records, decision_records, frame_paths, dirs, sim_state_csv, started_at)

        start_distance = _distance_xy(pre_decision_row, target_x=config.target_x, target_y=config.target_y)
        if start_distance is None or start_distance < config.min_start_distance:
            blocker = "post_drop_first_decision_start_distance_below_min_start_distance"
            return _finalize(config, status, blocker, records, decision_records, frame_paths, dirs, sim_state_csv, started_at)

        if config.record_continuous_video:
            recorder = ContinuousCameraRecorder(
                client,
                output_dir=dirs["continuous_frames"] / config.episode_id,
                save_fps=config.continuous_frame_save_fps,
            )
            recorder.start()

        maze = MazeMap.from_metadata(config.map_metadata)
        teacher = AStarTeacher(maze, stop_radius=config.teacher_stop_radius)
        task = _build_teacher_task(config=config, maze=maze, start_row=pre_decision_row)
        _write_json(data_dir / "teacher_task.json", task.to_dict())

        mapper = RealSonicSkillMapper(
            turn_degrees=config.turn_degrees,
            forward_duration=config.forward_duration,
            turn_duration=config.turn_duration,
            backoff_duration=config.backoff_duration,
            stop_duration=config.stop_duration,
            speed=config.walk_speed,
        )
        history: list[str] = []
        previous_actions: list[str] = []
        previous_status: list[str] = []
        last_recorder_index = -1

        for step_idx in range(config.max_steps):
            frame_path = dirs["frames"] / config.episode_id / f"frame_{step_idx:04d}.jpg"
            if recorder is not None:
                camera_name, camera_metadata, last_recorder_index = recorder.wait_for_next(
                    output_path=frame_path,
                    after_index=last_recorder_index,
                    timeout_s=30.0,
                )
            else:
                camera_name, camera_metadata = _wait_for_camera_frame(client, output_path=frame_path, timeout_s=30.0)
            frame_paths.append(frame_path)

            sim_row_before = _latest_sim_row(sim_state_csv)
            pose = _pose_from_row(sim_row_before)
            decision = teacher.next_action_from_pose(task, pose)
            mapper.heading_yaw_deg = pose.yaw_deg
            skill = mapper.action_to_skill(decision.action)
            execution = executor.execute_one(skill, episode_id=config.episode_id, skill_idx=step_idx + 1)
            sim_row_after = _latest_sim_row(sim_state_csv)

            should_stop = decision.action is VLNAction.STOP
            stop_reason = "near_goal_visual" if should_stop else "far_from_goal_negative"
            records.append(
                AutoVLNDemoRecord(
                    episode_id=config.episode_id,
                    step_idx=step_idx,
                    instruction=config.instruction,
                    image_path=str(frame_path),
                    history_image_paths=history[-4:],
                    previous_actions=list(previous_actions),
                    previous_skill_status=list(previous_status),
                    teacher_action=decision.action.value,
                    sonic_skill={"name": skill.name, "params": skill.params()},
                    should_stop=should_stop,
                    stop_label_reason=stop_reason,
                    teacher_metadata={
                        **task.teacher_metadata(),
                        "teacher_decision": decision.metadata,
                        "pose": asdict(pose),
                        "collection_source": "real_wbc_training_only_astar_teacher",
                    },
                    goal_xy=[config.target_x, config.target_y],
                    pose=asdict(pose),
                    source="real_wbc_oracle_teacher",
                    metadata={"camera_name": camera_name, "camera_metadata": camera_metadata},
                )
            )
            record = {
                "episode_id": config.episode_id,
                "step_idx": step_idx,
                "instruction": config.instruction,
                "image_path": str(frame_path),
                "history_image_paths": list(history[-4:]),
                "previous_actions": list(previous_actions),
                "previous_skill_status": list(previous_status),
                "teacher_action": decision.action.value,
                "teacher_raw_output": decision.raw_output,
                "teacher_metadata": decision.metadata,
                "selected_sonic_skill": skill.name,
                "skill_params": skill.params(),
                "skill_status": execution.status,
                "skill_termination_reason": execution.termination_reason,
                "skill_publish_ok": execution.publish_ok,
                "skill_downstream_state_observed": execution.downstream_state_observed,
                "sim_base_pos_x_before": sim_row_before.get("base_pos_x"),
                "sim_base_pos_y_before": sim_row_before.get("base_pos_y"),
                "sim_base_pos_z_before": sim_row_before.get("base_pos_z"),
                "sim_torso_yaw_before": sim_row_before.get("torso_yaw"),
                "sim_base_pos_x": sim_row_after.get("base_pos_x"),
                "sim_base_pos_y": sim_row_after.get("base_pos_y"),
                "sim_base_pos_z": sim_row_after.get("base_pos_z"),
                "sim_torso_yaw": sim_row_after.get("torso_yaw"),
                "sim_fall_flag": sim_row_after.get("fall_flag"),
                "privileged_policy_usage": True,
                "test_time_astar_used": True,
                "eligible_as_final_policy_success": False,
            }
            decision_records.append(record)
            history.append(str(frame_path))
            previous_actions.append(decision.action.value)
            previous_status.append(execution.status)

            if decision.action is VLNAction.STOP or _robot_unstable(sim_row_after):
                break

        status = "REAL_WBC_ORACLE_COLLECT_RAN" if records else "BLOCKED_REAL_WBC_ORACLE_COLLECT"
        blocker = "" if records else "no_records"
        if recorder is not None:
            recorder.stop()
            continuous_frame_paths = recorder.frame_paths_snapshot()
            recorder = None
        return _finalize(
            config,
            status,
            blocker,
            records,
            decision_records,
            frame_paths,
            dirs,
            sim_state_csv,
            started_at,
            continuous_frame_paths=recorder.frame_paths_snapshot() if recorder is not None else [],
        )
    except Exception as exc:  # noqa: BLE001
        blocker = f"{type(exc).__name__}:{exc}"
        sim_state_csv = stack.sim_log_dir / "sim_state.csv" if stack else None
        return _finalize(config, status, blocker, records, decision_records, frame_paths, dirs, sim_state_csv, started_at)
    finally:
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass
        if executor is not None:
            try:
                executor.stop_policy()
                executor.close()
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if stack is not None and not config.keep_stack_on_exit:
            stack.close()


def _finalize(
    config: RealWBCOracleCollectConfig,
    status: str,
    blocker: str,
    records: list[AutoVLNDemoRecord],
    decision_records: list[dict[str, Any]],
    frame_paths: list[Path],
    dirs: dict[str, Path],
    sim_state_csv: Path | None,
    started_at: float,
    *,
    continuous_frame_paths: list[Path] | None = None,
) -> dict[str, Any]:
    dataset_path = dirs["data"] / "auto_vln_real_wbc_dataset.jsonl"
    if records:
        write_jsonl(dataset_path, records)
    else:
        dataset_path.write_text("")
    decisions_path = dirs["data"] / "real_wbc_oracle_decisions.jsonl"
    _append_jsonl(decisions_path, decision_records)
    video_path = _write_video(frame_paths, dirs["videos"] / f"{config.episode_id}_head_camera.mp4", fps=4)
    continuous_video_path = None
    if continuous_frame_paths:
        continuous_video_path = _write_video(
            continuous_frame_paths,
            dirs["videos"] / f"{config.episode_id}_head_camera_continuous.mp4",
            fps=config.continuous_video_fps,
        )
    adapter_path = None
    adapter_metrics = None
    if records and config.train_adapter:
        adapter = train_visual_action_adapter(
            dataset_path,
            config.run_dir / "models/visual_action_adapter_real_wbc.json",
            config=VisualAdapterConfig(
                epochs=config.visual_adapter_epochs,
                validation_fraction=config.visual_adapter_validation_fraction,
            ),
        )
        adapter_path = str(config.run_dir / "models/visual_action_adapter_real_wbc.json")
        adapter_metrics = adapter.metrics

    collection_eval = (
        _eval_collection(config=config, decision_records=decision_records, sim_state_csv=sim_state_csv)
        if sim_state_csv is not None
        else {"teacher_rollout_success": False, "failure_reason": "sim_state_missing"}
    )
    manifest = {
        "status": status,
        "blocker": blocker,
        "run_dir": str(config.run_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": time.time() - started_at,
        "episode_id": config.episode_id,
        "scene_xml": str(config.scene_xml),
        "map_metadata": str(config.map_metadata),
        "target_xyz": [config.target_x, config.target_y, config.target_z],
        "instruction": config.instruction,
        "record_count": len(records),
        "decision_count": len(decision_records),
        "dataset_path": str(dataset_path),
        "adapter_path": adapter_path,
        "adapter_metrics": adapter_metrics,
        "decision_log": str(decisions_path),
        "frames": [str(path) for path in frame_paths],
        "video_path": video_path,
        "continuous_recording_start_reference": (
            "post_elastic_drop_pre_first_decision" if config.record_continuous_video else None
        ),
        "continuous_frames": [str(path) for path in (continuous_frame_paths or [])],
        "continuous_frame_count": len(continuous_frame_paths or []),
        "continuous_video_path": continuous_video_path,
        "sim_summary": _sim_summary(sim_state_csv) if sim_state_csv is not None else None,
        "collection_eval": collection_eval,
        "privileged_teacher_used_for_collection": True,
        "eligible_as_final_policy_success": False,
        "final_policy_must_use": (
            "real_navid_online.py with adapter inference only; no map/pose/goal/path/A* at test time"
        ),
    }
    _write_json(config.run_dir / "real_wbc_oracle_collect_manifest.json", manifest)
    _write_report(config.run_dir / "real_wbc_oracle_collect_report.md", manifest)
    return manifest


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    eval_payload = manifest.get("collection_eval") or {}
    lines = [
        "# Real WBC Oracle Collection",
        "",
        f"- status: `{manifest['status']}`",
        f"- blocker: `{manifest['blocker']}`",
        f"- record_count: `{manifest['record_count']}`",
        f"- adapter_path: `{manifest['adapter_path']}`",
        f"- teacher_rollout_success: `{eval_payload.get('teacher_rollout_success')}`",
        f"- start_distance: `{eval_payload.get('post_drop_first_decision_start_distance_to_target')}`",
        f"- start_distance_requirement_met: `{eval_payload.get('start_distance_requirement_met')}`",
        f"- final_distance: `{eval_payload.get('final_distance_to_target')}`",
        f"- failure_reason: `{eval_payload.get('failure_reason')}`",
        f"- privileged_teacher_used_for_collection: `{manifest['privileged_teacher_used_for_collection']}`",
        f"- eligible_as_final_policy_success: `{manifest['eligible_as_final_policy_success']}`",
        f"- dataset_path: `{manifest['dataset_path']}`",
        f"- video_path: `{manifest['video_path']}`",
        f"- continuous_video_path: `{manifest['continuous_video_path']}`",
        "",
        "## Collection Eval",
        "",
        "```json",
        json.dumps(eval_payload, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str]) -> RealWBCOracleCollectConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--scene-xml", type=Path, default=SCENE_XML)
    parser.add_argument("--map-metadata", type=Path, default=MAP_METADATA)
    parser.add_argument("--instruction", default=RealWBCOracleCollectConfig(Path(".")).instruction)
    parser.add_argument("--episode-id", default="real_wbc_oracle_collect_000")
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--camera-port", type=int, default=0)
    parser.add_argument("--zmq-port", type=int, default=0)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--target-x", type=float, default=5.0)
    parser.add_argument("--target-y", type=float, default=-5.0)
    parser.add_argument("--target-z", type=float, default=0.5)
    parser.add_argument("--head-camera-pos", default="0.18 0 0.42")
    parser.add_argument("--head-camera-euler", default="0 -1.0 -1.57")
    parser.add_argument("--publish-fps", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=4.0)
    parser.add_argument("--drop-on-start", action="store_true")
    parser.add_argument("--drop-after-seconds", type=float, default=35.0)
    parser.add_argument("--disable-reset-on-fall", action="store_true", default=True)
    parser.add_argument("--keep-stack-on-exit", action="store_true")
    parser.add_argument("--frame-width", type=int, default=480)
    parser.add_argument("--frame-height", type=int, default=360)
    parser.add_argument("--forward-duration", type=float, default=1.4)
    parser.add_argument("--turn-duration", type=float, default=1.6)
    parser.add_argument("--backoff-duration", type=float, default=0.8)
    parser.add_argument("--stop-duration", type=float, default=0.5)
    parser.add_argument("--walk-speed", type=float, default=0.25)
    parser.add_argument("--turn-degrees", type=float, default=30.0)
    parser.add_argument("--success-radius", type=float, default=0.8)
    parser.add_argument("--teacher-stop-radius", type=float, default=0.75)
    parser.add_argument("--min-motion-m", type=float, default=2.0)
    parser.add_argument("--min-start-distance", type=float, default=6.0)
    parser.add_argument("--record-continuous-video", action="store_true", default=True)
    parser.add_argument("--continuous-frame-save-fps", type=float, default=8.0)
    parser.add_argument("--continuous-video-fps", type=float, default=8.0)
    parser.add_argument("--no-train-adapter", action="store_true")
    parser.add_argument("--visual-adapter-epochs", type=int, default=240)
    parser.add_argument("--visual-adapter-validation-fraction", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.drop_after_seconds is not None and args.drop_after_seconds < 0:
        args.drop_after_seconds = None
    payload = vars(args)
    payload["train_adapter"] = not payload.pop("no_train_adapter")
    return RealWBCOracleCollectConfig(**payload)


def main(argv: list[str] | None = None) -> None:
    manifest = run(parse_args(sys.argv[1:] if argv is None else argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
