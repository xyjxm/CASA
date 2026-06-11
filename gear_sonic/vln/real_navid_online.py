"""Strict Real-NaVid online runner through the SONIC skill executor.

This module intentionally avoids the legacy route parser / red-detector runner.
It is meant to establish the causal online chain:

head_camera frame -> real NaVid inference -> parsed VLN action ->
CASA/SONIC skill wrapper -> SkillExecutor/ZMQ/deploy-WBC.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

import cv2

from gear_sonic.casa.skills.base import PlannerCommand, Skill
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.casa.skills import PassiveSkill, SkillExecutor, WalkSkill
from gear_sonic.vln.actions import ActionDecision, VLNAction
from gear_sonic.vln.backends import NaVidBackend, VLNObservation
from gear_sonic.vln.no_casa_policy import InferenceInput
from gear_sonic.vln.stack import launch_sonic_stack, wait_for_file, wait_for_log_contains
from gear_sonic.vln.visual_adapter import VisualActionAdapterPolicy


SCENE_XML = Path(
    "/mnt/data/students/lph/recording/gymnasium_robotics_maze_maps_scale2_20260610_233102/"
    "mjcf/sonic_scene_MEDIUM_MAZE_DIVERSE_GR_scale2_official_gymnasium.xml"
)
OUTPUT_ROOT = Path("/mnt/data/students/lph/recording")
NAVID_REPO = Path("/mnt/data/students/lph/recording/vln_sonic_navid_auto_20260610_202005/third_party/NaVid-VLN-CE")
NAVID_MODEL = Path("/mnt/data/students/lph/models/navid/Jzzhang_NaVid/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split")
NAVID_PYTHON = Path("/mnt/data/students/lph/models/navid/envs/navid-real/bin/python")
NAVID_VISION_TOWER = Path("/mnt/data/students/lph/models/navid/model_zoo/eva_vit_g.pth")


@dataclass(frozen=True)
class OnlineConfig:
    run_dir: Path
    scene_xml: Path = SCENE_XML
    navid_repo: Path = NAVID_REPO
    navid_model: Path = NAVID_MODEL
    navid_python: Path = NAVID_PYTHON
    navid_vision_tower: Path = NAVID_VISION_TOWER
    instruction: str = (
        "Use the first-person view to move through the open corridor toward the red target ball. "
        "Stop only when the red target ball is close and centered."
    )
    episode_id: str = "real_navid_online_smoke_000"
    max_steps: int = 3
    camera_port: int = 0
    zmq_port: int = 0
    domain_id: int = 0
    target_x: float = 3.0
    target_y: float = -1.0
    target_z: float = 0.5
    head_camera_pos: str = "0.18 0 0.42"
    head_camera_euler: str = "0 -1.0 -1.57"
    publish_fps: float = 8.0
    warmup_seconds: float = 4.0
    drop_on_start: bool = False
    drop_after_seconds: float | None = 35.0
    disable_reset_on_fall: bool = True
    prime_streamed_motion_seconds: float = 0.7
    keep_stack_on_exit: bool = False
    worker_timeout_s: float = 900.0
    frame_width: int = 480
    frame_height: int = 360
    dry_run: bool = False
    force_forward_after_turn_streak: int = 0
    visual_adapter_path: Path | None = None
    visual_adapter_mode: str = "off"
    visual_adapter_confidence_threshold: float = 0.0
    visual_adapter_min_stop_step: int = 0
    visual_adapter_min_backoff_step: int = 0
    forward_duration: float = 0.9
    turn_duration: float = 2.0
    backoff_duration: float = 0.5
    stop_duration: float = 0.5
    walk_speed: float = 0.25
    turn_degrees: float = 30.0
    success_radius: float = 0.8
    min_motion_m: float = 0.25
    min_start_distance: float = 0.0
    require_turn_forward_stop: bool = False
    max_initial_red_fraction: float | None = None
    record_continuous_video: bool = False
    continuous_frame_save_fps: float = 8.0
    continuous_video_fps: float = 8.0


@dataclass
class CommandDeltaHeadingTurnSkill(Skill):
    """Turn by streaming the deploy command-topic delta heading.

    ZMQManager applies delta_heading through the streamed-motion heading state;
    planner-topic facing changes alone do not rotate the simulated base.
    """

    start_yaw_deg: float
    face_yaw_deg: float
    duration: float

    name = "turn"

    def params(self) -> dict[str, Any]:
        return {
            "start_yaw_deg": self.start_yaw_deg,
            "face_yaw_deg": self.face_yaw_deg,
            "duration": self.duration,
            "turn_protocol": "command_delta_heading_streamed",
        }

    def command_at(self, elapsed_s: float, state: dict[str, Any]) -> PlannerCommand:
        del state
        ratio = max(0.0, min(1.0, elapsed_s / max(self.duration, 1e-6)))
        delta = _wrap_degrees(self.face_yaw_deg - self.start_yaw_deg)
        yaw_deg = _wrap_degrees(self.start_yaw_deg + delta * ratio)
        return PlannerCommand(
            mode=0,
            movement=(0.0, 0.0, 0.0),
            facing=(1.0, 0.0, 0.0),
            command_start=True,
            command_stop=False,
            command_planner=False,
            command_delta_heading=math.radians(yaw_deg),
        )


@dataclass
class RealSonicSkillMapper:
    """Map relative five-action VLN outputs to existing SONIC skill wrappers."""

    heading_yaw_deg: float = 0.0
    turn_degrees: float = 30.0
    forward_duration: float = 0.9
    turn_duration: float = 2.0
    backoff_duration: float = 0.5
    stop_duration: float = 0.5
    speed: float = 0.25

    def action_to_skill(self, action: VLNAction):
        if action is VLNAction.TURN_LEFT:
            start_yaw_deg = self.heading_yaw_deg
            self.heading_yaw_deg = _wrap_degrees(self.heading_yaw_deg + self.turn_degrees)
            return CommandDeltaHeadingTurnSkill(
                start_yaw_deg=start_yaw_deg,
                face_yaw_deg=self.heading_yaw_deg,
                duration=self.turn_duration,
            )
        if action is VLNAction.TURN_RIGHT:
            start_yaw_deg = self.heading_yaw_deg
            self.heading_yaw_deg = _wrap_degrees(self.heading_yaw_deg - self.turn_degrees)
            return CommandDeltaHeadingTurnSkill(
                start_yaw_deg=start_yaw_deg,
                face_yaw_deg=self.heading_yaw_deg,
                duration=self.turn_duration,
            )
        if action is VLNAction.FORWARD:
            vx, vy = _unit_from_yaw(self.heading_yaw_deg)
            return WalkSkill(vx=vx, vy=vy, facing_yaw_deg=self.heading_yaw_deg, duration=self.forward_duration, speed=self.speed)
        if action is VLNAction.BACKOFF:
            vx, vy = _unit_from_yaw(self.heading_yaw_deg + 180.0)
            return WalkSkill(vx=vx, vy=vy, facing_yaw_deg=self.heading_yaw_deg, duration=self.backoff_duration, speed=self.speed)
        if action is VLNAction.STOP:
            return PassiveSkill(duration=self.stop_duration, mode="stop")
        raise ValueError(f"unsupported action: {action}")


def default_run_dir() -> Path:
    return OUTPUT_ROOT / f"vln_sonic_real_navid_online_front_{time.strftime('%Y%m%d_%H%M%S')}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _value_counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _unit_from_yaw(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def _allocate_ports(seed: int | None = None) -> tuple[int, int, int]:
    rng = random.Random(seed if seed is not None else int(time.time()))
    base = rng.randrange(6200, 7200)
    return base, base + 1, rng.randrange(20, 220)


def _make_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "frames": run_dir / "frames",
        "videos": run_dir / "videos",
        "data": run_dir / "data",
        "logs": run_dir / "logs",
        "scenes": run_dir / "scenes",
        "skill": run_dir / "skill",
        "navid_backend": run_dir / "navid_backend",
        "continuous_frames": run_dir / "continuous_frames",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_target_scene_variant(
    source_xml: Path,
    output_xml: Path,
    *,
    target_xyz: tuple[float, float, float],
    head_camera_pos: str,
    head_camera_euler: str,
) -> Path:
    """Copy the official topology scene, target, and a front-facing head camera.

    The runtime runner never teleports qpos.  Start/target/camera changes are
    encoded before simulator launch as scene assets so they are auditable.
    """

    tree = ET.parse(source_xml)
    root = tree.getroot()
    include = root.find("include")
    if include is None or not include.get("file"):
        raise ValueError(f"scene has no robot include file: {source_xml}")
    robot_source = Path(include.get("file", ""))
    if not robot_source.is_absolute():
        robot_source = source_xml.parent / robot_source
    robot_tree = ET.parse(robot_source)
    robot_root = robot_tree.getroot()
    camera = robot_root.find(".//camera[@name='head_camera']")
    if camera is None:
        raise ValueError(f"robot xml has no camera named head_camera: {robot_source}")
    camera.set("pos", head_camera_pos)
    camera.set("euler", head_camera_euler)
    robot_output = output_xml.with_name(output_xml.stem + "_robot.xml")
    robot_tree.write(robot_output, encoding="unicode")
    include.set("file", str(robot_output))

    target = root.find(".//site[@name='target']")
    if target is None:
        raise ValueError(f"scene has no site named target: {source_xml}")
    target.set("pos", f"{target_xyz[0]:.4f} {target_xyz[1]:.4f} {target_xyz[2]:.4f}")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="unicode")
    return output_xml


def _wait_for_camera_frame(
    client: ComposedCameraClientSensor,
    *,
    output_path: Path,
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    last_seen: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        message = client.read(blocking=False)
        if message and message.get("images"):
            last_seen = message
            for name, image in sorted(message["images"].items()):
                if image is None:
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(output_path), corrected)
                return name, {
                    "camera_name": name,
                    "timestamps": message.get("timestamps", {}),
                    "image_shape": list(image.shape),
                    "frame_path": str(output_path),
                    "saved_color_correction": "bgr_to_rgb",
                }
        time.sleep(0.05)
    raise TimeoutError(f"no camera frame received within {timeout_s:.1f}s; last_seen={bool(last_seen)}")


@dataclass(frozen=True)
class RecordedCameraFrame:
    index: int
    path: Path
    metadata: dict[str, Any]


class ContinuousCameraRecorder:
    """Continuously save first-person camera frames while skills execute."""

    def __init__(
        self,
        client: ComposedCameraClientSensor,
        *,
        output_dir: Path,
        save_fps: float,
    ) -> None:
        self.client = client
        self.output_dir = output_dir
        self.save_fps = max(float(save_fps), 0.1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="continuous-camera-recorder", daemon=True)
        self._frame_paths: list[Path] = []
        self._latest: RecordedCameraFrame | None = None
        self._error: str | None = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def frame_paths_snapshot(self) -> list[Path]:
        with self._lock:
            return list(self._frame_paths)

    def wait_for_next(
        self,
        *,
        output_path: Path,
        after_index: int,
        timeout_s: float,
    ) -> tuple[str, dict[str, Any], int]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                latest = self._latest
                error = self._error
            if latest is not None and latest.index > after_index:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(latest.path, output_path)
                metadata = dict(latest.metadata)
                metadata.update(
                    {
                        "frame_path": str(output_path),
                        "continuous_frame_path": str(latest.path),
                        "continuous_frame_index": latest.index,
                    }
                )
                return str(metadata.get("camera_name") or "unknown"), metadata, latest.index
            if error and latest is None:
                raise RuntimeError(f"continuous camera recorder failed before any frame: {error}")
            time.sleep(0.02)
        raise TimeoutError(f"no continuous camera frame received within {timeout_s:.1f}s")

    def _run(self) -> None:
        min_period = 1.0 / self.save_fps
        next_save_at = 0.0
        index = 0
        while not self._stop.is_set():
            try:
                message = self.client.read(blocking=False)
                now = time.monotonic()
                if not message or not message.get("images") or now < next_save_at:
                    time.sleep(0.01)
                    continue
                for name, image in sorted(message["images"].items()):
                    if image is None:
                        continue
                    frame_path = self.output_dir / f"frame_{index:06d}.jpg"
                    corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(frame_path), corrected)
                    metadata = {
                        "camera_name": name,
                        "timestamps": message.get("timestamps", {}),
                        "image_shape": list(image.shape),
                        "saved_color_correction": "bgr_to_rgb",
                    }
                    record = RecordedCameraFrame(index=index, path=frame_path, metadata=metadata)
                    with self._lock:
                        self._frame_paths.append(frame_path)
                        self._latest = record
                    index += 1
                    next_save_at = now + min_period
                    break
            except Exception as exc:  # noqa: BLE001 - surface recorder errors in manifest/logs.
                with self._lock:
                    self._error = f"{type(exc).__name__}:{exc}"
                time.sleep(0.05)


def _latest_sim_row(sim_state_csv: Path) -> dict[str, Any]:
    if not sim_state_csv.exists():
        return {}
    with sim_state_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[-1] if rows else {}


def _wait_for_elastic_disabled(
    sim_state_csv: Path,
    executor: SkillExecutor,
    *,
    timeout_s: float = 60.0,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        row = _latest_sim_row(sim_state_csv)
        if row and row.get("elastic_band_enabled") == "0":
            return True
        executor.publisher.send_command(start=True, stop=False, planner=True)
        time.sleep(0.25)
    return False


def _sim_summary(sim_state_csv: Path) -> dict[str, Any]:
    if not sim_state_csv.exists():
        return {"sim_state_csv": str(sim_state_csv), "exists": False}
    with sim_state_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {"sim_state_csv": str(sim_state_csv), "exists": True, "rows": 0}
    fall_rows = sum(int(float(row.get("fall_flag") or 0)) for row in rows)
    elastic_rows = sum(int(float(row.get("elastic_band_enabled") or 0)) for row in rows)
    last = rows[-1]
    return {
        "sim_state_csv": str(sim_state_csv),
        "exists": True,
        "rows": len(rows),
        "fall_rows": fall_rows,
        "fall_count_last": int(float(last.get("fall_count") or 0)),
        "elastic_band_enabled_rows": elastic_rows,
        "elastic_band_enabled_ratio": elastic_rows / len(rows),
        "last_base_pos": [
            _safe_float(last.get("base_pos_x")),
            _safe_float(last.get("base_pos_y")),
            _safe_float(last.get("base_pos_z")),
        ],
        "last_torso_rpy": [
            _safe_float(last.get("torso_roll")),
            _safe_float(last.get("torso_pitch")),
            _safe_float(last.get("torso_yaw")),
        ],
    }


def _distance_xy(row: dict[str, Any], *, target_x: float, target_y: float) -> float | None:
    x = _safe_float(row.get("base_pos_x"))
    y = _safe_float(row.get("base_pos_y"))
    if x is None or y is None:
        return None
    return math.hypot(target_x - x, target_y - y)


def _red_fraction(image_path: str | Path | None) -> float | None:
    if image_path is None:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 0.50) & (red > green * 1.35 + 0.08) & (red > blue * 1.35 + 0.08)
    return float(mask.mean())


def _has_turn_forward_stop_subsequence(actions: list[str]) -> bool:
    seen_turn = False
    seen_forward_after_turn = False
    for action in actions:
        if action in {VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value}:
            seen_turn = True
        elif action == VLNAction.FORWARD.value and seen_turn:
            seen_forward_after_turn = True
        elif action == VLNAction.STOP.value and seen_forward_after_turn:
            return True
    return False


def _navigation_eval(
    *,
    config: OnlineConfig,
    decision_records: list[dict[str, Any]],
    sim_state_csv: Path | None,
) -> dict[str, Any]:
    sim_summary = _sim_summary(sim_state_csv) if sim_state_csv is not None else None
    if not decision_records:
        return {
            "strict_episode_success": False,
            "failure_reason": "no_decision_records",
            "action_sequence": [],
            "turn_forward_stop_subsequence": False,
            "initial_red_fraction": None,
            "final_red_fraction": None,
            "start_distance_to_target": None,
            "start_distance_reference": "first_decision_pre_skill_after_elastic_drop_and_prime",
            "post_drop_first_decision_start_distance_to_target": None,
            "post_drop_first_decision_start_xy": None,
            "target_xy": [config.target_x, config.target_y],
            "start_distance_requirement_met": False,
            "final_distance_to_target": None,
            "progress_m": None,
            "motion_m": None,
            "success_radius": config.success_radius,
            "min_motion_m": config.min_motion_m,
            "min_start_distance": config.min_start_distance,
            "require_turn_forward_stop": config.require_turn_forward_stop,
            "max_initial_red_fraction": config.max_initial_red_fraction,
            "requires_policy_stop": True,
            "final_action": "",
            "real_model_used_for_all_decisions": False,
            "no_privileged_test_time_policy": False,
            "fall_rows": None,
        }
    first = decision_records[0]
    last = decision_records[-1]
    start_row = {
        "base_pos_x": first.get("sim_base_pos_x_before"),
        "base_pos_y": first.get("sim_base_pos_y_before"),
    }
    final_row = {
        "base_pos_x": last.get("sim_base_pos_x"),
        "base_pos_y": last.get("sim_base_pos_y"),
    }
    start_distance = _distance_xy(start_row, target_x=config.target_x, target_y=config.target_y)
    final_distance = _distance_xy(final_row, target_x=config.target_x, target_y=config.target_y)
    start_x = _safe_float(start_row.get("base_pos_x"))
    start_y = _safe_float(start_row.get("base_pos_y"))
    final_x = _safe_float(final_row.get("base_pos_x"))
    final_y = _safe_float(final_row.get("base_pos_y"))
    motion_m = None
    if None not in {start_x, start_y, final_x, final_y}:
        motion_m = math.hypot(float(final_x) - float(start_x), float(final_y) - float(start_y))
    progress_m = None
    if start_distance is not None and final_distance is not None:
        progress_m = start_distance - final_distance

    fall_rows = int((sim_summary or {}).get("fall_rows") or 0)
    final_action = str(last.get("final_action") or "")
    action_sequence = [str(record.get("final_action") or "") for record in decision_records]
    initial_red_fraction = _red_fraction(first.get("image_path"))
    final_red_fraction = _red_fraction(last.get("image_path"))
    real_model_used = all(bool(record.get("real_navid_or_uninavid_used")) for record in decision_records)
    no_privileged_policy = all(
        not bool(record.get("privileged_policy_usage"))
        and not bool(record.get("test_time_route_parser_used"))
        and not bool(record.get("test_time_red_detector_used"))
        and not bool(record.get("test_time_astar_used"))
        and not bool(record.get("test_time_map_pose_goal_path_used"))
        and not bool(record.get("test_time_direct_qpos_or_teleport_used"))
        for record in decision_records
    )

    failure_reasons: list[str] = []
    if not real_model_used:
        failure_reasons.append("real_navid_or_uninavid_not_used_for_all_decisions")
    if not no_privileged_policy:
        failure_reasons.append("privileged_or_route_policy_used")
    if fall_rows:
        failure_reasons.append("fall_rows_nonzero")
    if final_distance is None or final_distance > config.success_radius:
        failure_reasons.append("final_distance_above_success_radius")
    if start_distance is None or start_distance < config.min_start_distance:
        failure_reasons.append("post_drop_first_decision_start_distance_below_min_start_distance")
    if motion_m is None or motion_m < config.min_motion_m:
        failure_reasons.append("motion_below_min_motion_m")
    if final_action != VLNAction.STOP.value:
        failure_reasons.append("policy_did_not_issue_stop")
    turn_forward_stop = _has_turn_forward_stop_subsequence(action_sequence)
    if config.require_turn_forward_stop and not turn_forward_stop:
        failure_reasons.append("turn_forward_stop_subsequence_missing")
    if (
        config.max_initial_red_fraction is not None
        and (initial_red_fraction is None or initial_red_fraction > config.max_initial_red_fraction)
    ):
        failure_reasons.append("initial_red_fraction_above_max")

    return {
        "strict_episode_success": not failure_reasons,
        "failure_reason": ",".join(failure_reasons),
        "action_sequence": action_sequence,
        "turn_forward_stop_subsequence": turn_forward_stop,
        "initial_red_fraction": initial_red_fraction,
        "final_red_fraction": final_red_fraction,
        "start_distance_to_target": start_distance,
        "start_distance_reference": "first_decision_pre_skill_after_elastic_drop_and_prime",
        "post_drop_first_decision_start_distance_to_target": start_distance,
        "post_drop_first_decision_start_xy": [start_x, start_y],
        "target_xy": [config.target_x, config.target_y],
        "start_distance_requirement_met": start_distance is not None and start_distance >= config.min_start_distance,
        "final_distance_to_target": final_distance,
        "progress_m": progress_m,
        "motion_m": motion_m,
        "success_radius": config.success_radius,
        "min_motion_m": config.min_motion_m,
        "min_start_distance": config.min_start_distance,
        "require_turn_forward_stop": config.require_turn_forward_stop,
        "max_initial_red_fraction": config.max_initial_red_fraction,
        "requires_policy_stop": True,
        "final_action": final_action,
        "real_model_used_for_all_decisions": real_model_used,
        "no_privileged_test_time_policy": no_privileged_policy,
        "fall_rows": fall_rows,
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _robot_unstable(row: dict[str, Any], *, min_base_z: float = 0.5) -> bool:
    if not row:
        return True
    try:
        fall_flag = int(float(row.get("fall_flag") or 0))
        base_z = float(row.get("base_pos_z") or 0.0)
    except (TypeError, ValueError):
        return True
    return bool(fall_flag) or base_z < min_base_z


def _adapt_action_history(
    action: VLNAction,
    previous_actions: list[str],
    *,
    force_forward_after_turn_streak: int,
) -> tuple[VLNAction, dict[str, Any]]:
    if force_forward_after_turn_streak <= 0:
        return action, {"adapter_used": False}
    if action not in {VLNAction.TURN_LEFT, VLNAction.TURN_RIGHT}:
        return action, {"adapter_used": False}
    turn_actions = {VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value}
    streak = 0
    for previous in reversed(previous_actions):
        if previous in turn_actions:
            streak += 1
        else:
            break
    if streak >= force_forward_after_turn_streak:
        return VLNAction.FORWARD, {
            "adapter_used": True,
            "adapter_name": "action_history_antispin",
            "turn_streak": streak,
            "force_forward_after_turn_streak": force_forward_after_turn_streak,
            "pre_adapter_action": action.value,
        }
    return action, {
        "adapter_used": False,
        "turn_streak": streak,
        "force_forward_after_turn_streak": force_forward_after_turn_streak,
    }


def _load_visual_adapter_policy(config: OnlineConfig) -> VisualActionAdapterPolicy | None:
    if config.visual_adapter_mode == "off":
        return None
    if config.visual_adapter_path is None:
        raise ValueError("visual_adapter_mode is enabled but --visual-adapter-path was not provided")
    if not config.visual_adapter_path.exists():
        raise FileNotFoundError(f"visual adapter not found: {config.visual_adapter_path}")
    return VisualActionAdapterPolicy.from_path(config.visual_adapter_path)


def _select_final_action(
    *,
    navid_decision: ActionDecision,
    visual_decision: ActionDecision | None,
    previous_actions: list[str],
    visual_adapter_mode: str,
    visual_adapter_confidence_threshold: float,
    visual_adapter_min_stop_step: int,
    visual_adapter_min_backoff_step: int,
    force_forward_after_turn_streak: int,
) -> tuple[VLNAction, str, dict[str, Any]]:
    if visual_adapter_mode not in {"off", "override", "confidence_gate", "stop_only"}:
        raise ValueError(f"unsupported visual_adapter_mode: {visual_adapter_mode}")

    selected_action = navid_decision.action
    selected_source = navid_decision.source
    visual_used_for_final = False
    reason = "navid_default"

    if visual_decision is not None and visual_adapter_mode == "override":
        selected_action = visual_decision.action
        selected_source = f"{navid_decision.source}+visual_action_adapter_override"
        visual_used_for_final = True
        reason = "visual_adapter_override"
    elif visual_decision is not None and visual_adapter_mode == "confidence_gate":
        if visual_decision.confidence >= visual_adapter_confidence_threshold:
            selected_action = visual_decision.action
            selected_source = f"{navid_decision.source}+visual_action_adapter_confidence_gate"
            visual_used_for_final = True
            reason = "visual_adapter_confidence_gate_pass"
        else:
            reason = "visual_adapter_confidence_gate_reject"
    elif visual_decision is not None and visual_adapter_mode == "stop_only":
        if visual_decision.action is not VLNAction.STOP:
            reason = "visual_adapter_stop_only_non_stop_ignored"
        elif visual_decision.confidence < visual_adapter_confidence_threshold:
            reason = "visual_adapter_stop_only_confidence_reject"
        elif len(previous_actions) < visual_adapter_min_stop_step:
            reason = "visual_adapter_stop_only_early_stop_reject"
        else:
            selected_action = VLNAction.STOP
            selected_source = f"{navid_decision.source}+visual_action_adapter_stop_only"
            visual_used_for_final = True
            reason = "visual_adapter_stop_only_stop_pass"

    early_stop_guard_applied = False
    if (
        visual_used_for_final
        and selected_action is VLNAction.STOP
        and len(previous_actions) < visual_adapter_min_stop_step
    ):
        selected_action = VLNAction.FORWARD
        selected_source = f"{selected_source}+strict_runner_early_stop_guard"
        early_stop_guard_applied = True
        reason = f"{reason}_early_stop_guard_forward"

    early_backoff_guard_applied = False
    if (
        visual_used_for_final
        and selected_action is VLNAction.BACKOFF
        and len(previous_actions) < visual_adapter_min_backoff_step
    ):
        selected_action = VLNAction.FORWARD
        selected_source = f"{selected_source}+strict_runner_early_backoff_guard"
        early_backoff_guard_applied = True
        reason = f"{reason}_early_backoff_guard_forward"

    selected_action, history_metadata = _adapt_action_history(
        selected_action,
        previous_actions,
        force_forward_after_turn_streak=force_forward_after_turn_streak,
    )
    if history_metadata.get("adapter_used"):
        selected_source = f"{selected_source}+action_history_antispin"

    metadata = {
        "visual_adapter_mode": visual_adapter_mode,
        "visual_adapter_used_for_final_action": visual_used_for_final,
        "visual_adapter_selection_reason": reason,
        "navid_action": navid_decision.action.value,
        "navid_source": navid_decision.source,
        "visual_adapter_action": visual_decision.action.value if visual_decision else None,
        "visual_adapter_source": visual_decision.source if visual_decision else None,
        "visual_adapter_confidence": visual_decision.confidence if visual_decision else None,
        "visual_adapter_min_stop_step": visual_adapter_min_stop_step,
        "visual_adapter_min_backoff_step": visual_adapter_min_backoff_step,
        "strict_runner_early_stop_guard_applied": early_stop_guard_applied,
        "strict_runner_early_backoff_guard_applied": early_backoff_guard_applied,
        "visual_adapter_metadata": visual_decision.metadata if visual_decision else None,
        "history_adapter_metadata": history_metadata,
    }
    return selected_action, selected_source, metadata


def _write_video(frame_paths: list[Path], output_path: Path, *, fps: float = 4) -> str | None:
    images = [cv2.imread(str(path)) for path in frame_paths if path.exists()]
    images = [image for image in images if image is not None]
    if not images:
        return None
    height, width = images[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for image in images:
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        writer.write(image)
    writer.release()
    return str(output_path)


def run(config: OnlineConfig) -> dict[str, Any]:
    if config.camera_port <= 0 or config.zmq_port <= 0 or config.domain_id <= 0:
        camera_port, zmq_port, domain_id = _allocate_ports()
        config = OnlineConfig(**{**asdict(config), "camera_port": camera_port, "zmq_port": zmq_port, "domain_id": domain_id})

    dirs = _make_dirs(config.run_dir)
    scene_variant = write_target_scene_variant(
        config.scene_xml,
        dirs["scenes"] / "episode_000_target_scene.xml",
        target_xyz=(config.target_x, config.target_y, config.target_z),
        head_camera_pos=config.head_camera_pos,
        head_camera_euler=config.head_camera_euler,
    )

    backend = NaVidBackend(
        repo_path=config.navid_repo,
        result_dir=dirs["navid_backend"],
        model_path=config.navid_model,
        variant="navid",
        strict_model=True,
        python_executable=config.navid_python,
        vision_tower_path=config.navid_vision_tower,
        worker_timeout_s=config.worker_timeout_s,
    )
    visual_policy = _load_visual_adapter_policy(config)
    stack = None
    client = None
    executor = None
    recorder = None
    decision_records: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    continuous_frame_paths: list[Path] = []
    status = "BLOCKED_REAL_NAVID_ONLINE"
    blocker = ""
    video_path = None
    continuous_video_path = None
    started_at = time.time()
    try:
        if not backend.availability.get("model_loaded"):
            blocker = "real_navid_model_not_loaded"
            return _finalize(config, status, blocker, backend, decision_records, frame_paths, video_path, dirs, None, started_at)

        stack = launch_sonic_stack(
            run_dir=config.run_dir / "stack",
            robot_scene=scene_variant,
            camera_port=config.camera_port,
            zmq_port=config.zmq_port,
            domain_id=config.domain_id,
            episode_name=config.episode_id,
            image_publish_fps=max(4, int(math.ceil(config.continuous_frame_save_fps)))
            if config.record_continuous_video
            else 4,
            episode_log_fps=20,
            offscreen_camera="head_camera",
            offscreen_camera_width=config.frame_width,
            offscreen_camera_height=config.frame_height,
            drop_on_start=config.drop_on_start,
            drop_after_seconds=config.drop_after_seconds,
            disable_reset_on_fall=config.disable_reset_on_fall,
        )
        sim_state_csv = stack.sim_log_dir / "sim_state.csv"
        sim_ready = wait_for_file(sim_state_csv, timeout_s=45.0, min_size=64)
        deploy_ready = wait_for_log_contains(
            stack.run_dir / "logs/deploy_stdout.log",
            ["Init Done"],
            timeout_s=120.0,
        )
        if not sim_ready:
            blocker = "sim_state_not_created"
            return _finalize(config, status, blocker, backend, decision_records, frame_paths, video_path, dirs, sim_state_csv, started_at)
        if not deploy_ready:
            blocker = "deploy_not_ready"
            return _finalize(config, status, blocker, backend, decision_records, frame_paths, video_path, dirs, sim_state_csv, started_at)

        client = ComposedCameraClientSensor(server_ip="localhost", port=config.camera_port, verbose=False)
        executor = SkillExecutor.from_paths(
            output_dir=dirs["skill"],
            sim_log_dir=stack.sim_log_dir,
            run_id=config.episode_id,
            zmq_host="localhost",
            zmq_port=config.zmq_port,
            publish_fps=config.publish_fps,
            dry_run=config.dry_run,
        )
        executor.start_policy()
        if config.warmup_seconds > 0:
            executor.execute_one(PassiveSkill(duration=config.warmup_seconds, mode="wait"), episode_id=config.episode_id, skill_idx=0)
        elastic_drop_seen = True
        if config.drop_after_seconds is not None and not config.drop_on_start:
            elastic_drop_seen = _wait_for_elastic_disabled(sim_state_csv, executor, timeout_s=90.0)
            if not elastic_drop_seen:
                blocker = "elastic_band_drop_not_observed"
                return _finalize(config, status, blocker, backend, decision_records, frame_paths, video_path, dirs, sim_state_csv, started_at)
        prime_result = None
        if config.prime_streamed_motion_seconds > 0 and not config.dry_run:
            prime_result = executor.execute_one(
                CommandDeltaHeadingTurnSkill(0.0, 0.0, config.prime_streamed_motion_seconds),
                episode_id=config.episode_id,
                skill_idx=-1,
            )
        pre_decision_row = _latest_sim_row(sim_state_csv)
        if _robot_unstable(pre_decision_row):
            blocker = "pre_decision_robot_unstable"
            return _finalize(config, status, blocker, backend, decision_records, frame_paths, video_path, dirs, sim_state_csv, started_at)
        if config.record_continuous_video:
            recorder = ContinuousCameraRecorder(
                client,
                output_dir=dirs["continuous_frames"] / config.episode_id,
                save_fps=config.continuous_frame_save_fps,
            )
            recorder.start()

        backend.reset(config.episode_id)
        mapper = RealSonicSkillMapper(
            turn_degrees=config.turn_degrees,
            forward_duration=config.forward_duration,
            turn_duration=config.turn_duration,
            backoff_duration=config.backoff_duration,
            stop_duration=config.stop_duration,
            speed=config.walk_speed,
        )
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
            observation = VLNObservation(
                episode_id=config.episode_id,
                step_idx=step_idx,
                instruction=config.instruction,
                image_path=str(frame_path),
                history_image_paths=[str(path) for path in frame_paths[-5:-1]],
                previous_actions=list(previous_actions),
                previous_skill_status=list(previous_status),
            )
            backend_result = backend.next_action(observation)
            visual_decision = None
            if visual_policy is not None:
                visual_observation = InferenceInput(
                    episode_id=config.episode_id,
                    step_idx=step_idx,
                    instruction=config.instruction,
                    image_path=str(frame_path),
                    history_image_paths=[str(path) for path in frame_paths[-5:-1]],
                    previous_actions=list(previous_actions),
                    previous_skill_status=list(previous_status),
                )
                visual_decision = visual_policy.next_action(visual_observation)
            final_action, final_action_source, adapter_metadata = _select_final_action(
                navid_decision=backend_result.decision,
                visual_decision=visual_decision,
                previous_actions=previous_actions,
                visual_adapter_mode=config.visual_adapter_mode,
                visual_adapter_confidence_threshold=config.visual_adapter_confidence_threshold,
                visual_adapter_min_stop_step=config.visual_adapter_min_stop_step,
                visual_adapter_min_backoff_step=config.visual_adapter_min_backoff_step,
                force_forward_after_turn_streak=config.force_forward_after_turn_streak,
            )
            skill = mapper.action_to_skill(final_action)
            sim_row_before = _latest_sim_row(sim_state_csv)
            execution = executor.execute_one(skill, episode_id=config.episode_id, skill_idx=step_idx + 1)
            sim_row_after = _latest_sim_row(sim_state_csv)
            record = {
                "episode_id": config.episode_id,
                "step_idx": step_idx,
                "instruction": config.instruction,
                "image_path": str(frame_path),
                "camera_name": camera_name,
                "camera_metadata": camera_metadata,
                "backend_name": backend_result.backend_name,
                "real_navid_or_uninavid_used": bool(backend_result.available and backend.availability.get("model_loaded")),
                "navid_model_loaded": bool(backend_result.metadata.get("navid_model_loaded")),
                "navid_raw_output": backend_result.raw_output,
                "parsed_action": backend_result.decision.action.value,
                "visual_adapter_action": visual_decision.action.value if visual_decision else None,
                "visual_adapter_raw_output": visual_decision.raw_output if visual_decision else None,
                "visual_adapter_confidence": visual_decision.confidence if visual_decision else None,
                "final_action": final_action.value,
                "final_action_source": final_action_source,
                "action_adapter_metadata": adapter_metadata,
                "navid_latency_ms": backend_result.metadata.get("navid_latency_ms"),
                "request_id": backend_result.metadata.get("request_id"),
                "navid_response_id": backend_result.metadata.get("navid_response_id"),
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
                "sim_fall_count": sim_row_after.get("fall_count"),
                "sim_elastic_band_enabled": sim_row_after.get("elastic_band_enabled"),
                "sim_low_cmd_count": sim_row_after.get("low_cmd_count"),
                "elastic_drop_observed_before_decision": elastic_drop_seen,
                "prime_streamed_motion_status": prime_result.status if prime_result else None,
                "prime_streamed_motion_seconds": config.prime_streamed_motion_seconds,
                "privileged_policy_usage": False,
                "test_time_route_parser_used": False,
                "test_time_red_detector_used": False,
                "test_time_astar_used": False,
                "test_time_map_pose_goal_path_used": False,
                "test_time_direct_qpos_or_teleport_used": False,
            }
            decision_records.append(record)
            previous_actions.append(record["final_action"])
            previous_status.append(execution.status)
            if final_action is VLNAction.STOP:
                break

        video_path = _write_video(frame_paths, dirs["videos"] / f"{config.episode_id}_head_camera.mp4")
        if recorder is not None:
            recorder.stop()
            continuous_frame_paths = recorder.frame_paths_snapshot()
            recorder = None
            continuous_video_path = _write_video(
                continuous_frame_paths,
                dirs["videos"] / f"{config.episode_id}_head_camera_continuous.mp4",
                fps=config.continuous_video_fps,
            )
        status = "REAL_NAVID_ONLINE_SMOKE_RAN" if decision_records else "BLOCKED_REAL_NAVID_ONLINE"
        blocker = "" if decision_records else "no_decision_records"
        return _finalize(
            config,
            status,
            blocker,
            backend,
            decision_records,
            frame_paths,
            video_path,
            dirs,
            sim_state_csv,
            started_at,
            continuous_frame_paths=continuous_frame_paths,
            continuous_video_path=continuous_video_path,
        )
    except Exception as exc:  # noqa: BLE001 - report exact blocker in artifact.
        blocker = f"{type(exc).__name__}:{exc}"
        return _finalize(config, status, blocker, backend, decision_records, frame_paths, video_path, dirs, stack.sim_log_dir / "sim_state.csv" if stack else None, started_at)
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
        backend.close()
        if stack is not None and not config.keep_stack_on_exit:
            stack.close()


def _finalize(
    config: OnlineConfig,
    status: str,
    blocker: str,
    backend: NaVidBackend,
    decision_records: list[dict[str, Any]],
    frame_paths: list[Path],
    video_path: str | None,
    dirs: dict[str, Path],
    sim_state_csv: Path | None,
    started_at: float,
    *,
    continuous_frame_paths: list[Path] | None = None,
    continuous_video_path: str | None = None,
) -> dict[str, Any]:
    decisions_path = dirs["data"] / "real_navid_online_decisions.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("w") as file:
        for record in decision_records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    navigation_eval = _navigation_eval(config=config, decision_records=decision_records, sim_state_csv=sim_state_csv)
    sim_summary = _sim_summary(sim_state_csv) if sim_state_csv is not None else None
    manifest = {
        "status": status,
        "blocker": blocker,
        "run_dir": str(config.run_dir),
        "episode_id": config.episode_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": time.time() - started_at,
        "scene_xml": str(config.scene_xml),
        "target_xyz": [config.target_x, config.target_y, config.target_z],
        "head_camera_pos": config.head_camera_pos,
        "head_camera_euler": config.head_camera_euler,
        "drop_on_start": config.drop_on_start,
        "drop_after_seconds": config.drop_after_seconds,
        "disable_reset_on_fall": config.disable_reset_on_fall,
        "prime_streamed_motion_seconds": config.prime_streamed_motion_seconds,
        "force_forward_after_turn_streak": config.force_forward_after_turn_streak,
        "forward_duration": config.forward_duration,
        "turn_duration": config.turn_duration,
        "backoff_duration": config.backoff_duration,
        "stop_duration": config.stop_duration,
        "walk_speed": config.walk_speed,
        "turn_degrees": config.turn_degrees,
        "success_radius": config.success_radius,
        "min_motion_m": config.min_motion_m,
        "min_start_distance": config.min_start_distance,
        "require_turn_forward_stop": config.require_turn_forward_stop,
        "max_initial_red_fraction": config.max_initial_red_fraction,
        "record_continuous_video": config.record_continuous_video,
        "continuous_recording_start_reference": (
            "post_elastic_drop_pre_first_decision" if config.record_continuous_video else None
        ),
        "continuous_frame_save_fps": config.continuous_frame_save_fps,
        "continuous_video_fps": config.continuous_video_fps,
        "visual_adapter_path": str(config.visual_adapter_path) if config.visual_adapter_path else None,
        "visual_adapter_mode": config.visual_adapter_mode,
        "visual_adapter_confidence_threshold": config.visual_adapter_confidence_threshold,
        "visual_adapter_min_stop_step": config.visual_adapter_min_stop_step,
        "visual_adapter_min_backoff_step": config.visual_adapter_min_backoff_step,
        "visual_adapter_enabled": config.visual_adapter_mode != "off",
        "instruction": config.instruction,
        "backend_name": backend.name,
        "backend_availability": backend.availability,
        "real_navid_or_uninavid_used": bool(decision_records) and all(
            bool(record.get("real_navid_or_uninavid_used")) for record in decision_records
        ),
        "decision_count": len(decision_records),
        "visual_adapter_decision_count": sum(1 for record in decision_records if record.get("visual_adapter_action")),
        "visual_adapter_final_action_count": sum(
            1
            for record in decision_records
            if (record.get("action_adapter_metadata") or {}).get("visual_adapter_used_for_final_action")
        ),
        "strict_runner_early_stop_guard_count": sum(
            1
            for record in decision_records
            if (record.get("action_adapter_metadata") or {}).get("strict_runner_early_stop_guard_applied")
        ),
        "strict_runner_early_backoff_guard_count": sum(
            1
            for record in decision_records
            if (record.get("action_adapter_metadata") or {}).get("strict_runner_early_backoff_guard_applied")
        ),
        "final_action_source_counts": _value_counts(
            record.get("final_action_source") for record in decision_records
        ),
        "required_final_episodes_met": False,
        "strict_episode_navigation_eval": navigation_eval,
        "pass_success30_real_navid": False,
        "route_parser_used": False,
        "red_detector_used": False,
        "direct_qpos_or_teleport_used_by_runner": False,
        "decision_log": str(decisions_path),
        "frames": [str(path) for path in frame_paths],
        "video_path": video_path,
        "continuous_frames": [str(path) for path in (continuous_frame_paths or [])],
        "continuous_frame_count": len(continuous_frame_paths or []),
        "continuous_video_path": continuous_video_path,
        "sim_summary": sim_summary,
        "notes": [
            "Smoke runner only: this proves the online model-to-skill chain when decisions exist.",
            "It does not claim the required >=20 held-out episodes or >=30% success.",
        ],
    }
    _write_json(config.run_dir / "real_navid_online_manifest.json", manifest)
    _write_report(config.run_dir / "real_navid_online_report.md", manifest)
    return manifest


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Real-NaVid Online SONIC Skill Smoke",
        "",
        f"- status: `{manifest['status']}`",
        f"- blocker: `{manifest['blocker']}`",
        f"- decision_count: `{manifest['decision_count']}`",
        f"- real_navid_or_uninavid_used: `{manifest['real_navid_or_uninavid_used']}`",
        f"- pass_success30_real_navid: `{manifest['pass_success30_real_navid']}`",
        f"- visual_adapter_enabled: `{manifest['visual_adapter_enabled']}`",
        f"- visual_adapter_path: `{manifest['visual_adapter_path']}`",
        f"- visual_adapter_mode: `{manifest['visual_adapter_mode']}`",
        f"- record_continuous_video: `{manifest['record_continuous_video']}`",
        f"- continuous_recording_start_reference: `{manifest['continuous_recording_start_reference']}`",
        f"- continuous_frame_count: `{manifest['continuous_frame_count']}`",
        f"- visual_adapter_min_stop_step: `{manifest['visual_adapter_min_stop_step']}`",
        f"- visual_adapter_min_backoff_step: `{manifest['visual_adapter_min_backoff_step']}`",
        f"- visual_adapter_decision_count: `{manifest['visual_adapter_decision_count']}`",
        f"- visual_adapter_final_action_count: `{manifest['visual_adapter_final_action_count']}`",
        f"- strict_runner_early_stop_guard_count: `{manifest['strict_runner_early_stop_guard_count']}`",
        f"- strict_runner_early_backoff_guard_count: `{manifest['strict_runner_early_backoff_guard_count']}`",
        f"- strict_episode_success: `{manifest['strict_episode_navigation_eval']['strict_episode_success']}`",
        f"- action_sequence: `{manifest['strict_episode_navigation_eval']['action_sequence']}`",
        f"- turn_forward_stop_subsequence: `{manifest['strict_episode_navigation_eval']['turn_forward_stop_subsequence']}`",
        f"- start_distance_reference: `{manifest['strict_episode_navigation_eval']['start_distance_reference']}`",
        f"- post_drop_first_decision_start_distance_to_target: `{manifest['strict_episode_navigation_eval']['post_drop_first_decision_start_distance_to_target']}`",
        f"- start_distance_requirement_met: `{manifest['strict_episode_navigation_eval']['start_distance_requirement_met']}`",
        f"- initial_red_fraction: `{manifest['strict_episode_navigation_eval']['initial_red_fraction']}`",
        f"- final_distance_to_target: `{manifest['strict_episode_navigation_eval']['final_distance_to_target']}`",
        f"- progress_m: `{manifest['strict_episode_navigation_eval']['progress_m']}`",
        f"- navigation_failure_reason: `{manifest['strict_episode_navigation_eval']['failure_reason']}`",
        f"- final_action_source_counts: `{manifest['final_action_source_counts']}`",
        f"- route_parser_used: `{manifest['route_parser_used']}`",
        f"- red_detector_used: `{manifest['red_detector_used']}`",
        f"- direct_qpos_or_teleport_used_by_runner: `{manifest['direct_qpos_or_teleport_used_by_runner']}`",
        f"- decision_log: `{manifest['decision_log']}`",
        f"- video_path: `{manifest['video_path']}`",
        f"- continuous_video_path: `{manifest['continuous_video_path']}`",
        "",
        "## Sim Summary",
        "",
        "```json",
        json.dumps(manifest.get("sim_summary"), indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str]) -> OnlineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--scene-xml", type=Path, default=SCENE_XML)
    parser.add_argument("--navid-repo", type=Path, default=NAVID_REPO)
    parser.add_argument("--navid-model", type=Path, default=NAVID_MODEL)
    parser.add_argument("--navid-python", type=Path, default=NAVID_PYTHON)
    parser.add_argument("--navid-vision-tower", type=Path, default=NAVID_VISION_TOWER)
    parser.add_argument("--instruction", default=OnlineConfig(Path(".")).instruction)
    parser.add_argument("--episode-id", default="real_navid_online_smoke_000")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--camera-port", type=int, default=0)
    parser.add_argument("--zmq-port", type=int, default=0)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--target-x", type=float, default=3.0)
    parser.add_argument("--target-y", type=float, default=-1.0)
    parser.add_argument("--target-z", type=float, default=0.5)
    parser.add_argument("--head-camera-pos", default="0.18 0 0.42")
    parser.add_argument("--head-camera-euler", default="0 -1.0 -1.57")
    parser.add_argument("--publish-fps", type=float, default=8.0)
    parser.add_argument("--warmup-seconds", type=float, default=4.0)
    parser.add_argument("--drop-on-start", action="store_true")
    parser.add_argument("--drop-after-seconds", type=float, default=35.0)
    parser.add_argument("--disable-reset-on-fall", action="store_true", default=True)
    parser.add_argument("--prime-streamed-motion-seconds", type=float, default=0.7)
    parser.add_argument("--keep-stack-on-exit", action="store_true")
    parser.add_argument("--worker-timeout-s", type=float, default=900.0)
    parser.add_argument("--frame-width", type=int, default=480)
    parser.add_argument("--frame-height", type=int, default=360)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-forward-after-turn-streak", type=int, default=0)
    parser.add_argument("--visual-adapter-path", type=Path, default=None)
    parser.add_argument(
        "--visual-adapter-mode",
        choices=["off", "override", "confidence_gate", "stop_only"],
        default="off",
        help="Use a trained visual action adapter in the strict runner. If a path is provided with mode off, mode becomes override.",
    )
    parser.add_argument("--visual-adapter-confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--visual-adapter-min-stop-step",
        type=int,
        default=0,
        help="Optional non-privileged guard: before this many prior actions, visual-adapter stop is converted to forward and logged.",
    )
    parser.add_argument(
        "--visual-adapter-min-backoff-step",
        type=int,
        default=0,
        help="Optional non-privileged guard: before this many prior actions, visual-adapter backoff is converted to forward and logged.",
    )
    parser.add_argument("--forward-duration", type=float, default=0.9)
    parser.add_argument("--turn-duration", type=float, default=2.0)
    parser.add_argument("--backoff-duration", type=float, default=0.5)
    parser.add_argument("--stop-duration", type=float, default=0.5)
    parser.add_argument("--walk-speed", type=float, default=0.25)
    parser.add_argument("--turn-degrees", type=float, default=30.0)
    parser.add_argument("--success-radius", type=float, default=0.8)
    parser.add_argument("--min-motion-m", type=float, default=0.25)
    parser.add_argument("--min-start-distance", type=float, default=0.0)
    parser.add_argument("--require-turn-forward-stop", action="store_true")
    parser.add_argument("--max-initial-red-fraction", type=float, default=None)
    parser.add_argument(
        "--record-continuous-video",
        action="store_true",
        help="Continuously save head-camera frames during skill execution and write a second normal video.",
    )
    parser.add_argument("--continuous-frame-save-fps", type=float, default=8.0)
    parser.add_argument("--continuous-video-fps", type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.drop_after_seconds is not None and args.drop_after_seconds < 0:
        args.drop_after_seconds = None
    if args.visual_adapter_path is not None and args.visual_adapter_mode == "off":
        args.visual_adapter_mode = "override"
    return OnlineConfig(**vars(args))


def main(argv: list[str] | None = None) -> None:
    manifest = run(parse_args(sys.argv[1:] if argv is None else argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
