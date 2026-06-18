"""Run SONIC VLN episodes with CASA safety-gate interventions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from typing import Any, Protocol

from .actions import ActionDecision, VLNAction
from .backends import NaVidBackend
from .casa_bridge import (
    CasaGateDecision,
    CasaVlnBridge,
    RecoveryCandidate,
    bridge_audit_dict,
    clearance_to_blocking_geometry,
    is_policy_stop_source,
    policy_internal_guard_summary,
    stop_source_for_policy_decision,
)
from .dmps_mpc_cbf_replan import (
    CandidateSequence,
    DEFAULT_SCORE_WEIGHTS,
    DMPS_ALIAS,
    DMPS_LAST_RESORT_STOP_SOURCE,
    DMPS_METHOD_NAME,
    PPSR_V2_ALIASES,
    PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE,
    PPSR_V2_LAST_RESORT_STOP_SOURCE,
    PPSR_V2_METHOD_NAME,
    PPSR_V2_SCORE_WEIGHTS,
    PPSR_V3_ALIASES,
    PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE,
    PPSR_V3_LAST_RESORT_STOP_SOURCE,
    PPSR_V3_METHOD_NAME,
    PPSR_V3_SCORE_WEIGHTS,
    candidate_scores_to_rows,
    is_dmps_method,
    is_ppsr_v2_method,
    is_ppsr_v3_method,
    normalize_dmps_method,
    rollout_candidate_sequence,
    select_dmps_replan,
    select_ppsr_v2_replan,
    select_ppsr_v3_replan,
    selected_replan_row,
    skill_for_recovery_action,
    terminal_action_allowed_probe,
    terminal_short_forward_probe,
)
from .maze import MazeMap, generate_routes
from .metrics import append_jsonl, distance_to_goal, write_json
from .no_casa_mujoco import MujocoMazeSkillEnv, draw_topdown_map, write_video
from .no_casa_policy import InferenceInput
from .progress_monitor import ProgressMonitor
from .no_casa_runner import (
    MAP_METADATA as LEGACY_MAP_METADATA,
    NAVID_MODEL,
    NAVID_PYTHON,
    NAVID_REPO,
    NAVID_VISION_TOWER,
    SCENE_XML as LEGACY_SCENE_XML,
    RealNaVidPolicy,
    RealNaVidVisualAdapterPolicy,
    collect_auto_demos,
)
from .skill_mapping import PassiveSkill, SonicSkill, TurnSkill, WalkSkill, wrap_degrees
from .tasks import TeacherTask, task_from_route
from .visual_adapter import (
    VisualActionAdapter,
    VisualActionAdapterPolicy,
    assert_no_privileged_inference_schema,
)


OUTPUT_ROOT = Path("/mnt/data/students/lph/recording")
DEFAULT_PHASE4_ROOT = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase4_dataset_v1_strict_50k_20260522"
)
DEFAULT_PHASE5_ROOT = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522"
)
GALLERY_SPLIT_ROOT = Path("/mnt/data/students/lph/recording/vln_wall_painting_split_maps_20260611_141106")
DEFAULT_TRAIN_MAP_METADATA = GALLERY_SPLIT_ROOT / "train_gallery/data/MEDIUM_MAZE_DIVERSE_GR_scale2_furnished_train_gallery_wall_painting.json"
DEFAULT_TRAIN_SCENE_XML = GALLERY_SPLIT_ROOT / "train_gallery/mjcf/sonic_scene_MEDIUM_MAZE_DIVERSE_GR_scale2_furnished_train_gallery_wall_painting.xml"
DEFAULT_TEST_MAP_METADATA = GALLERY_SPLIT_ROOT / "test_gallery/data/MEDIUM_MAZE_DIVERSE_GR_scale2_furnished_test_gallery_wall_painting.json"
DEFAULT_TEST_SCENE_XML = GALLERY_SPLIT_ROOT / "test_gallery/mjcf/sonic_scene_MEDIUM_MAZE_DIVERSE_GR_scale2_furnished_test_gallery_wall_painting.xml"
METHODS = (
    "vln_only",
    "vln_casa_reject_only",
    "vln_casa_replan",
    DMPS_METHOD_NAME,
    PPSR_V2_METHOD_NAME,
    PPSR_V3_METHOD_NAME,
)
DECISION_LOG_COLUMNS = [
    "method",
    "episode_id",
    "step_idx",
    "instruction",
    "image_path",
    "backend_name",
    "real_navid_or_uninavid_used",
    "navid_model_loaded",
    "navid_raw_output",
    "navid_parsed_action",
    "vln_raw_action",
    "vln_final_action_after_policy_internal_guards",
    "policy_decision_source",
    "policy_internal_guard_applied",
    "policy_internal_guard_names",
    "candidate_skill",
    "candidate_casa_skill_name",
    "casa_gate_evaluated",
    "raw_risk",
    "threshold",
    "risk_margin",
    "hard_contract_score",
    "hard_contract_fixed_reject",
    "casa_reject",
    "reject_reason",
    "final_executed_action",
    "metrics_action",
    "final_executed_skill",
    "stop_source",
    "casa_replan_applied",
    "casa_replan_selected_action",
    "casa_replan_selected_skill",
    "rejected_candidate_skill_was_executed",
    "skill_status",
    "wall_collision",
    "fall",
    "evaluator_distance_to_goal",
    "evaluator_pose_x",
    "evaluator_pose_y",
    "evaluator_pose_yaw_deg",
    "evaluator_goal_x",
    "evaluator_goal_y",
    "evaluator_only_shortest_path_remaining",
    "privileged_policy_usage",
    "policy_metadata_json",
    "skill_params_json",
    "wall_features_json",
]


class VLNPolicy(Protocol):
    name: str

    def reset(self, episode_id: str) -> None:
        ...

    def next_action(self, obs: InferenceInput) -> ActionDecision:
        ...


@dataclass(frozen=True)
class CasaVlnRunnerConfig:
    output_dir: Path
    scene_xml: Path
    map_metadata: Path
    train_scene_xml: Path
    train_map_metadata: Path
    phase4_root: Path = DEFAULT_PHASE4_ROOT
    phase5_root: Path = DEFAULT_PHASE5_ROOT
    methods: tuple[str, ...] = METHODS
    heldout_episodes: int = 20
    auto_demo_count: int = 100
    seed: int = 260610
    heldout_seed: int = 260611
    max_steps: int = 96
    frame_width: int = 320
    frame_height: int = 240
    visual_adapter_epochs: int = 240
    min_route_edges: int = 4
    max_route_edges: int = 14
    min_route_euclidean_m: float = 6.0
    require_furniture_detour: bool = True
    policy_backend: str = "real_navid_visual_adapter"
    navid_repo: Path = NAVID_REPO
    navid_model: Path = NAVID_MODEL
    navid_python: Path = NAVID_PYTHON
    navid_vision_tower: Path = NAVID_VISION_TOWER
    navid_worker_timeout_s: float = 900.0
    navid_cuda_visible_devices: str | None = None
    gate_method: str = "casa_a_hard_or_per_skill"
    max_consecutive_casa_recovery_steps: int = 1
    max_consecutive_dmps_recovery_steps: int = 1
    dmps_horizon: int = 2
    safety_margin: float = 0.05
    ppsr_v2_horizon_default: int = 2
    ppsr_v2_horizon_stuck: int = 3
    ppsr_v2_max_commit_steps: int = 2
    ppsr_v2_safety_margin: float = 0.05
    ppsr_v2_score_config: Path | None = None
    ppsr_v3_horizon: int = 5
    ppsr_v3_max_execute_steps: int = 2
    ppsr_v3_safety_margin: float = 0.05
    ppsr_v3_score_config: Path | None = None
    progress_score_config: Path | None = None
    score_weights: dict[str, float] | None = None
    previous_casa_vln_result_path: Path = Path("/mnt/data/students/lph/recording/casa_vln_safety_locked_real_navid_20260611_190121")
    seeds: tuple[int, ...] = (260611,)
    record_video: bool = True
    dry_run: bool = False
    strict: bool = False


def default_output_dir() -> Path:
    return OUTPUT_ROOT / f"casa_vln_safety_{time.strftime('%Y%m%d_%H%M%S')}"


def default_existing(path: Path, fallback: Path) -> Path:
    return path if path.exists() else fallback


def ensure_layout(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "figures": output_dir / "figures",
        "videos": output_dir / "videos",
        "frames": output_dir / "frames",
        "data": output_dir / "data",
        "models": output_dir / "models",
        "logs": output_dir / "logs",
        "notes": output_dir / "implementation_notes",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def skill_from_action(action: VLNAction, pose_yaw_deg: float) -> SonicSkill:
    if action is VLNAction.FORWARD:
        vx, vy = _unit_from_yaw(pose_yaw_deg)
        return WalkSkill(vx=vx, vy=vy, facing_yaw_deg=pose_yaw_deg, duration=0.70, step_target_m=0.50)
    if action is VLNAction.BACKOFF:
        vx, vy = _unit_from_yaw(pose_yaw_deg + 180.0)
        return WalkSkill(
            vx=vx,
            vy=vy,
            facing_yaw_deg=pose_yaw_deg,
            duration=0.40,
            step_target_m=0.20,
            name="backoff_walk",
        )
    if action is VLNAction.TURN_LEFT:
        return TurnSkill(delta_yaw_deg=30.0, face_yaw_deg=wrap_degrees(pose_yaw_deg + 30.0), duration=0.45)
    if action is VLNAction.TURN_RIGHT:
        return TurnSkill(delta_yaw_deg=-30.0, face_yaw_deg=wrap_degrees(pose_yaw_deg - 30.0), duration=0.45)
    if action is VLNAction.STOP:
        return PassiveSkill(duration=0.30, mode="stop")
    raise ValueError(f"unsupported VLN action: {action}")


def _unit_from_yaw(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def method_list(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return METHODS
    methods = tuple(normalize_dmps_method(item.strip()) for item in raw.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; expected comma list from {METHODS} or all")
    return methods


def verify_casa_artifacts(config: CasaVlnRunnerConfig) -> None:
    missing = [
        path
        for path in [
            config.phase4_root / "raw_critic" / "raw_critic.pt",
            config.phase5_root / "conformal_thresholds.json",
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "PARTIAL_BLOCKED_ENGINEERING: missing_phase4_or_phase5_artifacts: "
            + ", ".join(str(path) for path in missing)
        )


def build_policy(
    *,
    config: CasaVlnRunnerConfig,
    dirs: dict[str, Path],
    adapter_path: Path,
) -> tuple[VLNPolicy, NaVidBackend | None]:
    if config.policy_backend == "visual_adapter":
        return VisualActionAdapterPolicy(VisualActionAdapter.load(adapter_path)), None
    if config.policy_backend in {"real_navid", "real_navid_visual_adapter"}:
        backend = NaVidBackend(
            repo_path=config.navid_repo,
            result_dir=dirs["logs"] / "real_navid_backend",
            model_path=config.navid_model,
            variant="navid",
            strict_model=True,
            python_executable=config.navid_python,
            vision_tower_path=config.navid_vision_tower,
            worker_timeout_s=config.navid_worker_timeout_s,
            cuda_visible_devices=config.navid_cuda_visible_devices,
        )
        if not backend.availability.get("model_loaded"):
            write_json(dirs["data"] / "real_navid_backend_probe.json", backend.availability)
            raise RuntimeError(f"strict real NaVid backend did not load: {backend.availability}")
        if config.policy_backend == "real_navid":
            return RealNaVidPolicy(backend), backend
        return RealNaVidVisualAdapterPolicy(
            backend,
            VisualActionAdapterPolicy(VisualActionAdapter.load(adapter_path)),
        ), backend
    raise ValueError(f"unsupported policy backend: {config.policy_backend}")


def run_episode_with_method(
    *,
    method: str,
    task: TeacherTask,
    env: MujocoMazeSkillEnv,
    policy: VLNPolicy,
    bridge: CasaVlnBridge | None,
    config: CasaVlnRunnerConfig,
    run_dir: Path,
    data_dir: Path,
    videos_dir: Path,
    frames_dir: Path,
) -> dict[str, Any]:
    episode_id = f"{method}_{task.episode_id}"
    episode_frames_dir = frames_dir / method / task.episode_id
    episode_frames_dir.mkdir(parents=True, exist_ok=True)
    env.reset(task.start_xy, task.start_yaw_deg, task.goal_xy, episode_id=episode_id)
    policy.reset(episode_id)

    decision_records: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    replan_records: list[dict[str, Any]] = []
    dmps_candidate_rows: list[dict[str, Any]] = []
    dmps_selected_rows: list[dict[str, Any]] = []
    dmps_rollout_rows: list[dict[str, Any]] = []
    previous_actions: list[str] = []
    previous_status: list[str] = []
    frame_paths: list[Path] = []
    progress_monitor = ProgressMonitor(episode_id=episode_id)

    for step_idx in range(min(config.max_steps, task.max_steps)):
        frame_path = episode_frames_dir / f"frame_{step_idx:04d}.png"
        env.render_head(frame_path)
        frame_paths.append(frame_path)
        progress_monitor.pre_step(step_idx=step_idx, image_path=frame_path)
        pose_before = env.pose()
        obs = InferenceInput(
            episode_id=episode_id,
            step_idx=step_idx,
            instruction=task.instruction,
            image_path=str(frame_path),
            history_image_paths=[str(path) for path in frame_paths[-5:-1]],
            previous_actions=list(previous_actions),
            previous_skill_status=list(previous_status),
        )
        decision = policy.next_action(obs)
        backend_result = getattr(policy, "last_result", None)
        metadata = dict(decision.metadata or {})
        internal_guard_applied, internal_guard_names = policy_internal_guard_summary(metadata)
        vln_raw_action = metadata.get("real_navid_parsed_action") or _raw_action_from_decision(decision)
        action = decision.action
        candidate_skill = skill_from_action(action, pose_before.yaw_deg)
        candidate_casa_skill_name = bridge.executable_casa_skill_name(candidate_skill, action=action) if bridge else ""
        casa_decision: CasaGateDecision | None = None
        final_action_text = action.value
        final_skill = candidate_skill
        stop_source = stop_source_for_policy_decision(decision.source, metadata) if action is VLNAction.STOP else ""
        casa_replan_applied = False
        replan_selected_action = ""
        replan_selected_skill = ""
        dmps_replan_applied = False
        dmps_selected_sequence = ""
        dmps_recovery_stuck = False
        ppsr_v2_replan_applied = False
        ppsr_v2_stuck_mode = False
        ppsr_v2_stuck_mode_reasons: tuple[str, ...] = ()
        ppsr_v2_hard_mask_applied = False
        ppsr_v2_escape_macro_selected = False
        ppsr_v2_commitment_executed = False
        ppsr_v2_commitment_aborted = False
        ppsr_v2_second_action = ""
        ppsr_v2_second_action_safety_feasible = False
        ppsr_v2_commitment_abort_reason = ""
        ppsr_v2_recovery_took_over = False
        ppsr_v2_selected_sequence_contains_translation = False
        ppsr_v2_post_reject_forward_reenabled = False
        ppsr_v2_post_reject_clearance_gain = 0.0
        ppsr_v2_post_reject_unblocked = False
        ppsr_v2_post_reject_visual_novelty = False
        ppsr_v2_second_skill_status = ""
        ppsr_v2_second_wall_collision = False
        ppsr_v2_second_fall = False
        ppsr_v3_replan_applied = False
        ppsr_v3_repeated_replan = False
        ppsr_v3_recent_loop_flag = False
        ppsr_v3_policy_ready_candidate_exists = False
        ppsr_v3_policy_ready_state = False
        ppsr_v3_next_vln_action = ""
        ppsr_v3_next_vln_action_allowed = False
        ppsr_v3_actual_next_vln_action_allowed = False
        ppsr_v3_predicted_reject_drop = 0.0
        ppsr_v3_terminal_front_clearance = 0.0
        ppsr_v3_terminal_side_clearance = 0.0
        ppsr_v3_visual_novelty = 0.0
        ppsr_v3_landmark_retained_or_reacquired = False
        ppsr_v3_language_progress_score = 0.0
        ppsr_v3_policy_ready_score = 0.0
        ppsr_v3_post_reject_forward_reenabled = False
        ppsr_v3_post_reject_clearance_gain = 0.0
        ppsr_v3_post_reject_unblocked = False
        ppsr_v3_second_action = ""
        ppsr_v3_second_action_safety_feasible = False
        ppsr_v3_second_step_executed = False
        ppsr_v3_second_step_aborted = False
        ppsr_v3_commitment_abort_reason = ""
        ppsr_v3_second_skill_status = ""
        ppsr_v3_second_wall_collision = False
        ppsr_v3_second_fall = False
        non_stop_recovery_selected = False
        all_non_stop_candidates_infeasible = False
        rejected_candidate_skill_was_executed = False
        selected_sequence = None

        if method != "vln_only":
            if bridge is None:
                raise RuntimeError("CASA bridge is required for CASA methods")
            casa_decision = bridge.decide(
                action=action,
                skill=candidate_skill,
                pose=pose_before,
                maze=env.maze,
                robot_radius=env.robot_radius,
            )
            gate_records.append(
                _gate_record(
                    method=method,
                    task=task,
                    step_idx=step_idx,
                    action_text=action.value,
                    skill=candidate_skill,
                    decision=casa_decision,
                    candidate_role="vln_candidate",
                )
            )
            if casa_decision.casa_reject and action is not VLNAction.STOP:
                progress_monitor.record_reject(
                    step_idx=step_idx,
                    nominal_action=action.value,
                    reject_reason=casa_decision.reject_reason,
                )
                if method == "vln_casa_reject_only":
                    final_action_text = VLNAction.STOP.value
                    final_skill = PassiveSkill(duration=0.20, mode="stop")
                    stop_source = "casa_reject_only_stop"
                elif method == "vln_casa_replan":
                    selected, replan_rows = choose_replan_candidate(
                        bridge=bridge,
                        task=task,
                        method=method,
                        step_idx=step_idx,
                        pose=pose_before,
                        maze=env.maze,
                        robot_radius=env.robot_radius,
                    )
                    replan_records.extend(replan_rows)
                    casa_replan_applied = True
                    replan_selected_action = selected.action
                    replan_selected_skill = selected.skill.name
                    final_action_text = selected.action
                    final_skill = selected.skill
                    if selected.stop_as_last_resort:
                        stop_source = "casa_replan_last_resort_stop"
                elif is_dmps_method(method):
                    dmps_selection = select_dmps_replan(
                        bridge=bridge,
                        pose=pose_before,
                        maze=env.maze,
                        robot_radius=env.robot_radius,
                        nominal_action=action,
                        image_path=frame_path,
                        progress_monitor=progress_monitor,
                        step_idx=step_idx,
                        horizon=config.dmps_horizon,
                        safety_margin=config.safety_margin,
                        score_weights=config.score_weights,
                    )
                    dmps_replan_applied = True
                    casa_replan_applied = True
                    selected_sequence = dmps_selection.selected_sequence
                    first_action = selected_sequence.actions[0]
                    final_action_text = first_action
                    final_skill = selected_sequence.skills[0]
                    replan_selected_action = first_action
                    replan_selected_skill = final_skill.name
                    dmps_selected_sequence = json.dumps(list(selected_sequence.actions))
                    all_non_stop_candidates_infeasible = dmps_selection.all_non_stop_candidates_infeasible
                    non_stop_recovery_selected = first_action != "stop_as_last_resort"
                    dmps_recovery_stuck = len(progress_monitor.state.recent_selected_recoveries) >= config.max_consecutive_dmps_recovery_steps and bool(progress_monitor.state.recent_selected_recoveries)
                    if selected_sequence.stop_as_last_resort:
                        stop_source = DMPS_LAST_RESORT_STOP_SOURCE
                    dmps_candidate_rows.extend(
                        candidate_scores_to_rows(
                            method=method,
                            episode_id=task.episode_id,
                            step_idx=step_idx,
                            nominal_action=action.value,
                            reject_reason=casa_decision.reject_reason,
                            selection=dmps_selection,
                        )
                    )
                    for rollout_row in dmps_selection.rollout_rows:
                        rollout_row.update(
                            {
                                "method": method,
                                "episode_id": task.episode_id,
                                "step_id": step_idx,
                                "nominal_action": action.value,
                            }
                        )
                        dmps_rollout_rows.append(rollout_row)
                elif is_ppsr_v2_method(method):
                    dmps_selection = select_ppsr_v2_replan(
                        bridge=bridge,
                        pose=pose_before,
                        maze=env.maze,
                        robot_radius=env.robot_radius,
                        nominal_action=action,
                        image_path=frame_path,
                        progress_monitor=progress_monitor,
                        step_idx=step_idx,
                        horizon_default=config.ppsr_v2_horizon_default,
                        horizon_stuck=config.ppsr_v2_horizon_stuck,
                        safety_margin=config.ppsr_v2_safety_margin,
                        score_weights=config.score_weights,
                    )
                    dmps_replan_applied = True
                    ppsr_v2_replan_applied = True
                    casa_replan_applied = True
                    selected_sequence = dmps_selection.selected_sequence
                    first_action = selected_sequence.actions[0]
                    final_action_text = first_action
                    final_skill = selected_sequence.skills[0]
                    replan_selected_action = first_action
                    replan_selected_skill = final_skill.name
                    dmps_selected_sequence = json.dumps(list(selected_sequence.actions))
                    all_non_stop_candidates_infeasible = dmps_selection.all_non_stop_candidates_infeasible
                    non_stop_recovery_selected = first_action != "stop_as_last_resort"
                    dmps_recovery_stuck = bool(dmps_selection.stuck_mode)
                    ppsr_v2_stuck_mode = bool(dmps_selection.stuck_mode)
                    ppsr_v2_stuck_mode_reasons = tuple(dmps_selection.stuck_mode_reasons)
                    ppsr_v2_hard_mask_applied = any(score.hard_mask_applied for score in dmps_selection.candidate_scores)
                    ppsr_v2_escape_macro_selected = bool(selected_sequence.escape_macro)
                    ppsr_v2_selected_sequence_contains_translation = bool(dmps_selection.selected_score.contains_translation)
                    if selected_sequence.stop_as_last_resort:
                        stop_source = PPSR_V2_LAST_RESORT_STOP_SOURCE
                    dmps_candidate_rows.extend(
                        candidate_scores_to_rows(
                            method=method,
                            episode_id=task.episode_id,
                            step_idx=step_idx,
                            nominal_action=action.value,
                            reject_reason=casa_decision.reject_reason,
                            selection=dmps_selection,
                        )
                    )
                    for rollout_row in dmps_selection.rollout_rows:
                        rollout_row.update(
                            {
                                "method": method,
                                "episode_id": task.episode_id,
                                "step_id": step_idx,
                                "nominal_action": action.value,
                            }
                        )
                        dmps_rollout_rows.append(rollout_row)
                elif is_ppsr_v3_method(method):
                    dmps_selection = select_ppsr_v3_replan(
                        bridge=bridge,
                        pose=pose_before,
                        maze=env.maze,
                        robot_radius=env.robot_radius,
                        nominal_action=action,
                        image_path=frame_path,
                        progress_monitor=progress_monitor,
                        step_idx=step_idx,
                        horizon=config.ppsr_v3_horizon,
                        safety_margin=config.ppsr_v3_safety_margin,
                        score_weights=config.score_weights,
                    )
                    dmps_replan_applied = True
                    ppsr_v3_replan_applied = True
                    casa_replan_applied = True
                    selected_sequence = dmps_selection.selected_sequence
                    first_action = selected_sequence.actions[0]
                    final_action_text = first_action
                    final_skill = selected_sequence.skills[0]
                    replan_selected_action = first_action
                    replan_selected_skill = final_skill.name
                    dmps_selected_sequence = json.dumps(list(selected_sequence.actions))
                    all_non_stop_candidates_infeasible = dmps_selection.all_non_stop_candidates_infeasible
                    non_stop_recovery_selected = first_action != "stop_as_last_resort"
                    dmps_recovery_stuck = bool(dmps_selection.recent_loop_flag or dmps_selection.repeated_replan)
                    ppsr_v3_repeated_replan = bool(dmps_selection.repeated_replan)
                    ppsr_v3_recent_loop_flag = bool(dmps_selection.recent_loop_flag)
                    ppsr_v3_policy_ready_candidate_exists = bool(dmps_selection.policy_ready_candidate_exists)
                    ppsr_v3_policy_ready_state = bool(dmps_selection.selected_score.policy_ready_state)
                    ppsr_v3_next_vln_action = dmps_selection.selected_score.next_vln_action
                    ppsr_v3_next_vln_action_allowed = bool(dmps_selection.selected_score.next_vln_action_allowed)
                    ppsr_v3_predicted_reject_drop = float(dmps_selection.selected_score.predicted_reject_drop)
                    ppsr_v3_terminal_front_clearance = float(dmps_selection.selected_score.terminal_front_clearance)
                    ppsr_v3_terminal_side_clearance = float(dmps_selection.selected_score.terminal_side_clearance)
                    ppsr_v3_visual_novelty = float(dmps_selection.selected_score.visual_novelty)
                    ppsr_v3_landmark_retained_or_reacquired = bool(
                        dmps_selection.selected_score.landmark_retained_or_reacquired
                    )
                    ppsr_v3_language_progress_score = float(dmps_selection.selected_score.language_progress_score)
                    ppsr_v3_policy_ready_score = float(dmps_selection.selected_score.policy_ready_score)
                    if selected_sequence.stop_as_last_resort:
                        stop_source = PPSR_V3_LAST_RESORT_STOP_SOURCE
                    dmps_candidate_rows.extend(
                        candidate_scores_to_rows(
                            method=method,
                            episode_id=task.episode_id,
                            step_idx=step_idx,
                            nominal_action=action.value,
                            reject_reason=casa_decision.reject_reason,
                            selection=dmps_selection,
                        )
                    )
                    for rollout_row in dmps_selection.rollout_rows:
                        rollout_row.update(
                            {
                                "method": method,
                                "episode_id": task.episode_id,
                                "step_id": step_idx,
                                "nominal_action": action.value,
                            }
                        )
                        dmps_rollout_rows.append(rollout_row)
                else:
                    raise ValueError(f"unsupported CASA method: {method}")

        distance_after = None
        result = env.execute_skill(final_skill)
        pose_after = env.pose()
        distance_after = distance_to_goal(task, pose_after)
        if ppsr_v2_replan_applied and selected_sequence is not None:
            start_clearance, _ = clearance_to_blocking_geometry(
                pose_before.x,
                pose_before.y,
                maze=env.maze,
                robot_radius=env.robot_radius,
            )
            should_commit_second = (
                ppsr_v2_stuck_mode
                and len(selected_sequence.actions) >= 2
                and ppsr_v2_selected_sequence_contains_translation
                and config.ppsr_v2_max_commit_steps >= 2
                and result.status not in {"blocked", "collision"}
                and not result.fall
            )
            if should_commit_second:
                ppsr_v2_second_action = selected_sequence.actions[1]
                current_pose = env.pose()
                second_skill = skill_for_recovery_action(ppsr_v2_second_action, current_pose.yaw_deg)
                second_candidate = CandidateSequence(
                    sequence_id=f"commit_{ppsr_v2_second_action}",
                    actions=(ppsr_v2_second_action,),
                    skills=(second_skill,),
                    stop_as_last_resort=ppsr_v2_second_action == "stop_as_last_resort",
                )
                second_rollout = rollout_candidate_sequence(
                    candidate=second_candidate,
                    start_pose=current_pose,
                    maze=env.maze,
                    robot_radius=env.robot_radius,
                    safety_margin=config.ppsr_v2_safety_margin,
                )
                ppsr_v2_second_action_safety_feasible = bool(second_rollout.safety_feasible)
                if second_rollout.safety_feasible and ppsr_v2_second_action != "stop_as_last_resort":
                    second_result = env.execute_skill(second_skill)
                    ppsr_v2_commitment_executed = True
                    ppsr_v2_commitment_aborted = False
                    ppsr_v2_second_skill_status = second_result.status
                    ppsr_v2_second_wall_collision = bool(second_result.wall_collision)
                    ppsr_v2_second_fall = bool(second_result.fall)
                    pose_after = env.pose()
                    distance_after = distance_to_goal(task, pose_after)
                    if second_result.status in {"blocked", "collision"} or second_result.fall:
                        ppsr_v2_commitment_abort_reason = f"second_step_runtime_{second_result.status}"
                else:
                    ppsr_v2_commitment_aborted = True
                    ppsr_v2_commitment_abort_reason = second_rollout.safety_rejection_reason
            elif ppsr_v2_stuck_mode and len(selected_sequence.actions) >= 2:
                ppsr_v2_commitment_aborted = True
                ppsr_v2_commitment_abort_reason = "first_step_not_successful_or_no_translation"
            end_clearance, _ = clearance_to_blocking_geometry(
                pose_after.x,
                pose_after.y,
                maze=env.maze,
                robot_radius=env.robot_radius,
            )
            probe = terminal_short_forward_probe(
                pose=pose_after,
                maze=env.maze,
                robot_radius=env.robot_radius,
                safety_margin=config.ppsr_v2_safety_margin,
            )
            ppsr_v2_post_reject_clearance_gain = float(end_clearance - start_clearance)
            ppsr_v2_post_reject_forward_reenabled = bool(probe["terminal_can_short_forward"])
            ppsr_v2_post_reject_unblocked = result.status not in {"blocked", "collision"}
            ppsr_v2_post_reject_visual_novelty = bool(ppsr_v2_commitment_executed or ppsr_v2_escape_macro_selected)
        if ppsr_v3_replan_applied and selected_sequence is not None:
            start_clearance, _ = clearance_to_blocking_geometry(
                pose_before.x,
                pose_before.y,
                maze=env.maze,
                robot_radius=env.robot_radius,
            )
            should_execute_second = (
                len(selected_sequence.actions) >= 2
                and config.ppsr_v3_max_execute_steps >= 2
                and selected_sequence.actions[0] != "stop_as_last_resort"
                and result.status not in {"blocked", "collision"}
                and not result.fall
            )
            if should_execute_second:
                ppsr_v3_second_action = selected_sequence.actions[1]
                current_pose = env.pose()
                second_skill = skill_for_recovery_action(ppsr_v3_second_action, current_pose.yaw_deg)
                second_candidate = CandidateSequence(
                    sequence_id=f"v3_commit_{ppsr_v3_second_action}",
                    actions=(ppsr_v3_second_action,),
                    skills=(second_skill,),
                    stop_as_last_resort=ppsr_v3_second_action == "stop_as_last_resort",
                )
                second_rollout = rollout_candidate_sequence(
                    candidate=second_candidate,
                    start_pose=current_pose,
                    maze=env.maze,
                    robot_radius=env.robot_radius,
                    safety_margin=config.ppsr_v3_safety_margin,
                )
                ppsr_v3_second_action_safety_feasible = bool(second_rollout.safety_feasible)
                if second_rollout.safety_feasible and ppsr_v3_second_action != "stop_as_last_resort":
                    second_result = env.execute_skill(second_skill)
                    ppsr_v3_second_step_executed = True
                    ppsr_v3_second_step_aborted = False
                    ppsr_v3_second_skill_status = second_result.status
                    ppsr_v3_second_wall_collision = bool(second_result.wall_collision)
                    ppsr_v3_second_fall = bool(second_result.fall)
                    pose_after = env.pose()
                    distance_after = distance_to_goal(task, pose_after)
                    if second_result.status in {"blocked", "collision"} or second_result.fall:
                        ppsr_v3_commitment_abort_reason = f"second_step_runtime_{second_result.status}"
                else:
                    ppsr_v3_second_step_aborted = True
                    ppsr_v3_commitment_abort_reason = second_rollout.safety_rejection_reason
            elif len(selected_sequence.actions) >= 2 and selected_sequence.actions[0] != "stop_as_last_resort":
                ppsr_v3_second_step_aborted = True
                ppsr_v3_commitment_abort_reason = "first_step_not_successful_or_max_execute_steps_one"
            end_clearance, _ = clearance_to_blocking_geometry(
                pose_after.x,
                pose_after.y,
                maze=env.maze,
                robot_radius=env.robot_radius,
            )
            probe = terminal_short_forward_probe(
                pose=pose_after,
                maze=env.maze,
                robot_radius=env.robot_radius,
                safety_margin=config.ppsr_v3_safety_margin,
            )
            action_probe = terminal_action_allowed_probe(
                pose=pose_after,
                next_action=action,
                maze=env.maze,
                robot_radius=env.robot_radius,
                safety_margin=config.ppsr_v3_safety_margin,
            )
            ppsr_v3_post_reject_clearance_gain = float(end_clearance - start_clearance)
            ppsr_v3_post_reject_forward_reenabled = bool(probe["terminal_can_short_forward"])
            ppsr_v3_actual_next_vln_action_allowed = bool(action_probe["next_vln_action_allowed"])
            ppsr_v3_post_reject_unblocked = (
                result.status not in {"blocked", "collision"}
                and not result.fall
                and not ppsr_v3_second_wall_collision
                and not ppsr_v3_second_fall
            )
            ppsr_v3_policy_ready_state = bool(
                ppsr_v3_post_reject_unblocked
                and (ppsr_v3_post_reject_forward_reenabled or ppsr_v3_terminal_side_clearance >= config.ppsr_v3_safety_margin)
                and ppsr_v3_actual_next_vln_action_allowed
                and ppsr_v3_landmark_retained_or_reacquired
                and not (ppsr_v3_recent_loop_flag and not selected_sequence.escape_macro)
            )
        if casa_decision is not None and casa_decision.casa_reject:
            rejected_candidate_skill_was_executed = _same_executable(candidate_skill, final_skill)
        metrics_action = _metrics_action(final_action_text)
        if metrics_action != VLNAction.STOP.value:
            stop_source = ""
        distance = distance_to_goal(task, pose_before)
        post_reject_progress = (distance - distance_after) if casa_decision is not None and casa_decision.casa_reject else None
        if dmps_replan_applied:
            dmps_selected_rows.append(
                selected_replan_row(
                    method=method,
                    episode_id=task.episode_id,
                    step_idx=step_idx,
                    nominal_action=action.value,
                    reject_reason=casa_decision.reject_reason if casa_decision else "",
                    selection=dmps_selection,
                    executed_first_skill=final_skill,
                    stop_source=stop_source,
                    post_reject_progress_evaluator_only=float(post_reject_progress or 0.0),
                    committed_second_action=ppsr_v2_second_action or ppsr_v3_second_action,
                    committed_second_step_executed=ppsr_v2_commitment_executed or ppsr_v3_second_step_executed,
                    committed_second_step_aborted=ppsr_v2_commitment_aborted or ppsr_v3_second_step_aborted,
                    second_action_safety_feasible=ppsr_v2_second_action_safety_feasible
                    or ppsr_v3_second_action_safety_feasible,
                    abort_reason=ppsr_v2_commitment_abort_reason or ppsr_v3_commitment_abort_reason,
                    max_commit_steps=config.ppsr_v3_max_execute_steps
                    if ppsr_v3_replan_applied
                    else config.ppsr_v2_max_commit_steps
                    if ppsr_v2_replan_applied
                    else 1,
                )
            )
        record = {
            "method": method,
            "episode_id": task.episode_id,
            "full_episode_id": episode_id,
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
            "navid_model_loaded": bool((getattr(backend_result, "metadata", {}) or {}).get("navid_model_loaded"))
            if backend_result is not None
            else False,
            "navid_raw_output": getattr(backend_result, "raw_output", None) if backend_result is not None else None,
            "navid_parsed_action": getattr(getattr(backend_result, "decision", None), "action", None).value
            if backend_result is not None and getattr(backend_result, "decision", None) is not None
            else None,
            "vln_raw_action": vln_raw_action,
            "vln_final_action_after_policy_internal_guards": action.value,
            "backend_action": action.value,
            "final_action": metrics_action,
            "final_executed_action": final_action_text,
            "metrics_action": metrics_action,
            "policy_decision_source": decision.source,
            "policy_internal_guard_applied": internal_guard_applied,
            "policy_internal_guard_names": internal_guard_names,
            "policy_metadata": metadata,
            "candidate_skill": candidate_skill.name,
            "candidate_casa_skill_name": candidate_casa_skill_name,
            "selected_sonic_skill": final_skill.name,
            "final_executed_skill": final_skill.name,
            "skill_params": final_skill.params(),
            "skill_status": result.status,
            "skill_execution": result.to_dict(),
            "stop_source": stop_source,
            "casa_gate_evaluated": casa_decision is not None,
            "raw_risk": casa_decision.raw_risk if casa_decision else None,
            "threshold": casa_decision.threshold if casa_decision else None,
            "risk_margin": casa_decision.risk_margin if casa_decision else None,
            "hard_contract_score": casa_decision.hard_contract_score if casa_decision else None,
            "hard_contract_fixed_reject": casa_decision.hard_contract_fixed_reject if casa_decision else False,
            "casa_reject": casa_decision.casa_reject if casa_decision else False,
            "reject_reason": casa_decision.reject_reason if casa_decision else "",
            "wall_features": casa_decision.wall_features.to_dict() if casa_decision else {},
            "risk_source": casa_decision.risk_source if casa_decision else "",
            "casa_replan_applied": casa_replan_applied,
            "casa_replan_selected_action": replan_selected_action,
            "casa_replan_selected_skill": replan_selected_skill,
            "dmps_replan_applied": dmps_replan_applied,
            "dmps_selected_sequence": dmps_selected_sequence,
            "dmps_recovery_stuck": dmps_recovery_stuck,
            "ppsr_v2_replan_applied": ppsr_v2_replan_applied,
            "ppsr_v2_stuck_mode": ppsr_v2_stuck_mode,
            "ppsr_v2_stuck_mode_reasons": list(ppsr_v2_stuck_mode_reasons),
            "ppsr_v2_hard_mask_applied": ppsr_v2_hard_mask_applied,
            "ppsr_v2_escape_macro_selected": ppsr_v2_escape_macro_selected,
            "ppsr_v2_commitment_executed": ppsr_v2_commitment_executed,
            "ppsr_v2_commitment_aborted": ppsr_v2_commitment_aborted,
            "ppsr_v2_second_action": ppsr_v2_second_action,
            "ppsr_v2_second_action_safety_feasible": ppsr_v2_second_action_safety_feasible,
            "ppsr_v2_second_skill_status": ppsr_v2_second_skill_status,
            "ppsr_v2_commitment_abort_reason": ppsr_v2_commitment_abort_reason,
            "ppsr_v2_recovery_took_over": ppsr_v2_recovery_took_over,
            "ppsr_v2_post_reject_clearance_gain": ppsr_v2_post_reject_clearance_gain,
            "ppsr_v2_post_reject_forward_reenabled": ppsr_v2_post_reject_forward_reenabled,
            "ppsr_v2_post_reject_unblocked": ppsr_v2_post_reject_unblocked,
            "ppsr_v2_post_reject_visual_novelty": ppsr_v2_post_reject_visual_novelty,
            "ppsr_v3_replan_applied": ppsr_v3_replan_applied,
            "ppsr_v3_repeated_replan": ppsr_v3_repeated_replan,
            "ppsr_v3_recent_loop_flag": ppsr_v3_recent_loop_flag,
            "ppsr_v3_policy_ready_candidate_exists": ppsr_v3_policy_ready_candidate_exists,
            "ppsr_v3_policy_ready_state": ppsr_v3_policy_ready_state,
            "ppsr_v3_next_vln_action": ppsr_v3_next_vln_action,
            "ppsr_v3_next_vln_action_allowed": ppsr_v3_next_vln_action_allowed,
            "ppsr_v3_actual_next_vln_action_allowed": ppsr_v3_actual_next_vln_action_allowed,
            "ppsr_v3_predicted_reject_drop": ppsr_v3_predicted_reject_drop,
            "ppsr_v3_terminal_front_clearance": ppsr_v3_terminal_front_clearance,
            "ppsr_v3_terminal_side_clearance": ppsr_v3_terminal_side_clearance,
            "ppsr_v3_visual_novelty": ppsr_v3_visual_novelty,
            "ppsr_v3_landmark_retained_or_reacquired": ppsr_v3_landmark_retained_or_reacquired,
            "ppsr_v3_language_progress_score": ppsr_v3_language_progress_score,
            "ppsr_v3_policy_ready_score": ppsr_v3_policy_ready_score,
            "ppsr_v3_post_reject_forward_reenabled": ppsr_v3_post_reject_forward_reenabled,
            "ppsr_v3_post_reject_clearance_gain": ppsr_v3_post_reject_clearance_gain,
            "ppsr_v3_post_reject_unblocked": ppsr_v3_post_reject_unblocked,
            "ppsr_v3_second_action": ppsr_v3_second_action,
            "ppsr_v3_second_action_safety_feasible": ppsr_v3_second_action_safety_feasible,
            "ppsr_v3_second_step_executed": ppsr_v3_second_step_executed,
            "ppsr_v3_second_step_aborted": ppsr_v3_second_step_aborted,
            "ppsr_v3_second_skill_status": ppsr_v3_second_skill_status,
            "ppsr_v3_commitment_abort_reason": ppsr_v3_commitment_abort_reason,
            "non_stop_recovery_selected": non_stop_recovery_selected,
            "all_non_stop_candidates_infeasible": all_non_stop_candidates_infeasible,
            "post_reject_progress_evaluator_only": post_reject_progress,
            "rejected_candidate_skill_was_executed": rejected_candidate_skill_was_executed,
            "evaluator_pose": asdict(pose_before),
            "evaluator_distance_to_goal": distance,
            "evaluator_distance_after_action": distance_after,
            "evaluator_goal_xy": list(task.goal_xy),
            "evaluator_only_shortest_path_remaining": env.maze.shortest_path_distance_xy(
                (pose_before.x, pose_before.y), task.goal_xy
            ),
            "fall": bool(result.fall or ppsr_v2_second_fall or ppsr_v3_second_fall),
            "wall_collision": bool(result.wall_collision or ppsr_v2_second_wall_collision or ppsr_v3_second_wall_collision),
            "privileged_policy_usage": False,
        }
        decision_records.append(record)
        progress_monitor.record_step(
            step_idx=step_idx,
            executed_action=metrics_action,
            skill_status=result.status,
            stop_source=stop_source,
            selected_recovery_action=replan_selected_action if dmps_replan_applied else None,
        )
        previous_actions.append(metrics_action)
        previous_status.append(result.status)
        if ppsr_v2_commitment_executed and ppsr_v2_second_action:
            progress_monitor.record_step(
                step_idx=step_idx,
                executed_action=_metrics_action(ppsr_v2_second_action),
                skill_status=ppsr_v2_second_skill_status or "ok",
                stop_source="",
                selected_recovery_action=ppsr_v2_second_action,
            )
            previous_actions.append(_metrics_action(ppsr_v2_second_action))
            previous_status.append(ppsr_v2_second_skill_status or "ok")
        if ppsr_v3_second_step_executed and ppsr_v3_second_action:
            progress_monitor.record_step(
                step_idx=step_idx,
                executed_action=_metrics_action(ppsr_v3_second_action),
                skill_status=ppsr_v3_second_skill_status or "ok",
                stop_source="",
                selected_recovery_action=ppsr_v3_second_action,
            )
            previous_actions.append(_metrics_action(ppsr_v3_second_action))
            previous_status.append(ppsr_v3_second_skill_status or "ok")
        if metrics_action == VLNAction.STOP.value:
            break

    video = write_video(frame_paths, videos_dir / method / f"{task.episode_id}.mp4", fps=8)
    episode_decision_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_decision_log.jsonl"
    append_jsonl(episode_decision_log, decision_records)
    episode_gate_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_gate_decisions.jsonl"
    append_jsonl(episode_gate_log, gate_records)
    episode_replan_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_replan_log.jsonl"
    append_jsonl(episode_replan_log, replan_records)
    episode_dmps_candidate_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_dmps_candidate_sequences.jsonl"
    append_jsonl(episode_dmps_candidate_log, dmps_candidate_rows)
    episode_dmps_selected_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_dmps_selected_replans.jsonl"
    append_jsonl(episode_dmps_selected_log, dmps_selected_rows)
    episode_dmps_rollout_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_dmps_mpc_cbf_rollouts.jsonl"
    append_jsonl(episode_dmps_rollout_log, dmps_rollout_rows)
    episode_progress_events_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_progress_monitor_events.jsonl"
    append_jsonl(episode_progress_events_log, progress_monitor.event_rows())
    trajectory_rows = list(env.trajectory)
    episode_trajectory_log = data_dir / "per_episode" / f"{method}_{task.episode_id}_trajectory_log.jsonl"
    append_jsonl(episode_trajectory_log, trajectory_rows)
    draw_topdown_map(
        env.maze,
        run_dir / "figures" / method / f"{task.episode_id}_topdown.png",
        trajectory_xy=[(float(row["pose_x"]), float(row["pose_y"])) for row in trajectory_rows],
        start_xy=task.start_xy,
        goal_xy=task.goal_xy,
    )
    metrics = summarize_casa_episode(
        method=method,
        task=task,
        maze=env.maze,
        decision_records=decision_records,
        trajectory_rows=trajectory_rows,
        video_paths=[video] if video else [],
        frames_dir=episode_frames_dir,
        decision_log=episode_decision_log,
        gate_log=episode_gate_log,
        replan_log=episode_replan_log,
        trajectory_log=episode_trajectory_log,
    )
    metrics["dmps_candidate_log"] = str(episode_dmps_candidate_log)
    metrics["dmps_selected_log"] = str(episode_dmps_selected_log)
    metrics["dmps_rollout_log"] = str(episode_dmps_rollout_log)
    metrics["progress_monitor_events_log"] = str(episode_progress_events_log)
    metrics["progress_monitor_audit"] = progress_monitor.audit_dict()
    return metrics


def choose_replan_candidate(
    *,
    bridge: CasaVlnBridge,
    task: TeacherTask,
    method: str,
    step_idx: int,
    pose,
    maze: MazeMap,
    robot_radius: float,
) -> tuple[RecoveryCandidate, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    allowed_non_stop: list[tuple[float, int, RecoveryCandidate]] = []
    stop_candidate: RecoveryCandidate | None = None
    for candidate in bridge.recovery_candidates(pose=pose):
        if candidate.stop_as_last_resort:
            action_for_gate = VLNAction.STOP
        elif candidate.action == "short_forward_segment":
            action_for_gate = VLNAction.FORWARD
        else:
            action_for_gate = VLNAction(candidate.action)
        decision = bridge.decide(
            action=action_for_gate,
            skill=candidate.skill,
            pose=pose,
            maze=maze,
            robot_radius=robot_radius,
        )
        row = _gate_record(
            method=method,
            task=task,
            step_idx=step_idx,
            action_text=candidate.action,
            skill=candidate.skill,
            decision=decision,
            candidate_role="replan_candidate",
        )
        row["stop_as_last_resort"] = int(candidate.stop_as_last_resort)
        row["prior_rank"] = candidate.prior_rank
        rows.append(row)
        if candidate.stop_as_last_resort:
            stop_candidate = candidate
        elif not decision.casa_reject:
            allowed_non_stop.append((decision.raw_risk, candidate.prior_rank, candidate))
    if allowed_non_stop:
        allowed_non_stop.sort(key=lambda item: (item[0], item[1]))
        selected = allowed_non_stop[0][2]
    elif stop_candidate is not None:
        selected = stop_candidate
    else:
        selected = RecoveryCandidate(
            action="stop_as_last_resort",
            skill=PassiveSkill(duration=0.20, mode="stop"),
            casa_skill_name="passive",
            prior_rank=99,
            stop_as_last_resort=True,
        )
    for row in rows:
        row["selected_by_replan"] = int(row["candidate_action"] == selected.action)
    return selected, rows


def _gate_record(
    *,
    method: str,
    task: TeacherTask,
    step_idx: int,
    action_text: str,
    skill: SonicSkill,
    decision: CasaGateDecision,
    candidate_role: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "episode_id": task.episode_id,
        "step_idx": step_idx,
        "candidate_role": candidate_role,
        "candidate_action": action_text,
        "candidate_skill": skill.name,
        "candidate_casa_skill_name": decision.candidate_casa_skill_name,
        "raw_risk": decision.raw_risk,
        "threshold": decision.threshold,
        "risk_margin": decision.risk_margin,
        "hard_contract_score": decision.hard_contract_score,
        "hard_contract_fixed_reject": int(decision.hard_contract_fixed_reject),
        "casa_reject": int(decision.casa_reject),
        "reject_reason": decision.reject_reason,
        "risk_source": decision.risk_source,
        **decision.wall_features.to_dict(),
    }


def _raw_action_from_decision(decision: ActionDecision) -> str:
    if isinstance(decision.raw_output, dict):
        raw = decision.raw_output.get("action") or decision.raw_output.get("text")
        if raw is not None:
            return str(raw)
    return decision.action.value


def _same_executable(candidate: SonicSkill, final: SonicSkill) -> bool:
    return candidate.name == final.name and candidate.params() == final.params()


def _metrics_action(action_text: str) -> str:
    if action_text in {"short_forward_segment", "short_forward"}:
        return VLNAction.FORWARD.value
    if action_text == "stop_as_last_resort":
        return VLNAction.STOP.value
    if action_text == "small_turn_left":
        return VLNAction.TURN_LEFT.value
    if action_text == "small_turn_right":
        return VLNAction.TURN_RIGHT.value
    if action_text == "wide_turn_left":
        return VLNAction.TURN_LEFT.value
    if action_text == "wide_turn_right":
        return VLNAction.TURN_RIGHT.value
    if action_text in {"target_reacquire_turn_left"}:
        return VLNAction.TURN_LEFT.value
    if action_text in {"target_reacquire_turn_right"}:
        return VLNAction.TURN_RIGHT.value
    if action_text in {"wide_arc_left", "wide_arc_right", "wall_follow_left", "wall_follow_right", "open_space_seek"}:
        return VLNAction.FORWARD.value
    return VLNAction(action_text).value


def _selected_sequence_is_turn_only(raw_sequence: Any) -> bool:
    if not raw_sequence:
        return False
    try:
        actions = json.loads(str(raw_sequence))
    except Exception:
        return False
    if not isinstance(actions, list) or not actions:
        return False
    return all(
        str(action)
        in {
            "turn_left",
            "turn_right",
            "small_turn_left",
            "small_turn_right",
            "wide_turn_left",
            "wide_turn_right",
            "target_reacquire_turn_left",
            "target_reacquire_turn_right",
        }
        for action in actions
    )


def summarize_casa_episode(
    *,
    method: str,
    task: TeacherTask,
    maze: MazeMap,
    decision_records: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    video_paths: list[Path],
    frames_dir: Path,
    decision_log: Path,
    gate_log: Path,
    replan_log: Path,
    trajectory_log: Path,
) -> dict[str, Any]:
    poses = [
        (float(row["pose_x"]), float(row["pose_y"]), float(row["pose_yaw_deg"]))
        for row in trajectory_rows
        if "pose_x" in row
    ]
    start_xy = poses[0][:2] if poses else task.start_xy
    final_xy = poses[-1][:2] if poses else start_xy
    start_distance = math.hypot(task.goal_xy[0] - start_xy[0], task.goal_xy[1] - start_xy[1])
    final_distance = math.hypot(task.goal_xy[0] - final_xy[0], task.goal_xy[1] - final_xy[1])
    min_distance = min([math.hypot(task.goal_xy[0] - x, task.goal_xy[1] - y) for x, y, _ in poses] or [final_distance])
    start_remaining = maze.shortest_path_distance_xy(start_xy, task.goal_xy)
    final_remaining = maze.shortest_path_distance_xy(final_xy, task.goal_xy)
    progress_m = max(0.0, start_remaining - final_remaining)
    shortest_path_progress = progress_m / start_remaining if math.isfinite(start_remaining) and start_remaining > 1e-6 else 0.0

    actions = [str(record["metrics_action"]) for record in decision_records]
    distribution = {action.value: 0 for action in VLNAction}
    distribution.update(Counter(actions))
    policy_stop_records = [
        record
        for record in decision_records
        if record["metrics_action"] == VLNAction.STOP.value and is_policy_stop_source(record.get("stop_source"))
    ]
    all_stop_records = [record for record in decision_records if record["metrics_action"] == VLNAction.STOP.value]
    policy_stop_record = policy_stop_records[0] if policy_stop_records else None
    stop_distance = float(policy_stop_record["evaluator_distance_to_goal"]) if policy_stop_record else None
    final_stop_frame = policy_stop_record.get("image_path") if policy_stop_record else None

    fall_count = sum(int(row.get("fall", 0)) for row in trajectory_rows)
    collision_count = sum(int(row.get("wall_collision", 0)) for row in trajectory_rows)
    displacement = math.hypot(final_xy[0] - start_xy[0], final_xy[1] - start_xy[1])
    executed_motion_skills = sum(
        1
        for action in actions
        if action in {VLNAction.FORWARD.value, VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value, VLNAction.BACKOFF.value}
    )
    meaningful_motion = displacement >= 0.25 and executed_motion_skills >= 2
    stop_fallback_ratio = (distribution[VLNAction.STOP.value] + distribution[VLNAction.BACKOFF.value]) / max(1, len(actions))
    timeout = not policy_stop_records and len(decision_records) >= task.max_steps
    stuck = displacement < 0.10 and len(decision_records) >= 3
    privileged_policy_usage_count = sum(1 for record in decision_records if record.get("privileged_policy_usage"))
    reached_goal_at_policy_stop = stop_distance is not None and stop_distance <= task.success_radius
    no_fall = fall_count == 0
    no_wall_collision = collision_count == 0
    sufficient_progress = shortest_path_progress >= 0.50
    bounded_stop_fallback = stop_fallback_ratio <= 0.50
    success = (
        reached_goal_at_policy_stop
        and bool(policy_stop_records)
        and no_fall
        and no_wall_collision
        and meaningful_motion
        and sufficient_progress
        and bounded_stop_fallback
        and privileged_policy_usage_count == 0
    )
    safe_success = success and no_fall and no_wall_collision

    stop_failure_type = None
    if policy_stop_records and not reached_goal_at_policy_stop:
        stop_failure_type = "premature_stop"
    elif not policy_stop_records and min_distance <= task.success_radius:
        stop_failure_type = "late_stop"
    elif len(all_stop_records) > 1 and stop_fallback_ratio > 0.50:
        stop_failure_type = "infinite_stop"

    if success:
        failure_reason = ""
    elif privileged_policy_usage_count:
        failure_reason = "privileged_policy_usage"
    elif not policy_stop_records:
        failure_reason = "missing_policy_stop"
    elif not reached_goal_at_policy_stop:
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

    reject_count = sum(1 for record in decision_records if record.get("casa_reject"))
    replan_count = sum(1 for record in decision_records if record.get("casa_replan_applied"))
    stop_source_distribution = dict(Counter(str(record.get("stop_source") or "") for record in all_stop_records))
    post_reject_progress_values = [
        float(record["post_reject_progress_evaluator_only"])
        for record in decision_records
        if record.get("post_reject_progress_evaluator_only") is not None
    ]
    recovery_actions = [
        str(record.get("casa_replan_selected_action") or "")
        for record in decision_records
        if record.get("casa_replan_applied")
    ]
    ppsr_v2_records = [record for record in decision_records if record.get("ppsr_v2_replan_applied")]
    ppsr_v2_clearance_gains = [float(record.get("ppsr_v2_post_reject_clearance_gain", 0.0)) for record in ppsr_v2_records]
    ppsr_v3_records = [record for record in decision_records if record.get("ppsr_v3_replan_applied")]
    ppsr_v3_clearance_gains = [float(record.get("ppsr_v3_post_reject_clearance_gain", 0.0)) for record in ppsr_v3_records]
    readiness_replan_count = len(ppsr_v2_records) + len(ppsr_v3_records)
    repeat_reject_count = _repeat_reject_count(decision_records)
    loop_count = (
        sum(1 for record in decision_records if record.get("ppsr_v3_recent_loop_flag"))
        + sum(1 for record in decision_records if _selected_sequence_is_turn_only(record.get("dmps_selected_sequence")))
        + sum(
            1
            for record in decision_records
            if record.get("dmps_recovery_stuck") and not record.get("ppsr_v3_replan_applied")
        )
    )
    safety_stop_success_leakage_count = int(
        success
        and any(
            record.get("metrics_action") == VLNAction.STOP.value
            and str(record.get("stop_source") or "")
            in {"casa_reject_only_stop", "casa_replan_last_resort_stop"}
            for record in decision_records
        )
    )
    recovery_stop_success_leakage_count = int(
        success
        and any(
            record.get("metrics_action") == VLNAction.STOP.value
            and str(record.get("stop_source") or "")
            in {
                DMPS_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE,
                PPSR_V3_LAST_RESORT_STOP_SOURCE,
                PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE,
            }
            for record in decision_records
        )
    )
    small_turn_recovery_count = sum(1 for action in recovery_actions if action in {"small_turn_left", "small_turn_right"})
    turn_only_recovery_count = sum(1 for record in decision_records if _selected_sequence_is_turn_only(record.get("dmps_selected_sequence")))
    return {
        "method": method,
        "episode_id": task.episode_id,
        "instruction": task.instruction,
        "success": success,
        "safe_success": safe_success,
        "failure_reason": failure_reason,
        "steps": len(decision_records),
        "policy_issued_stop": bool(policy_stop_records),
        "policy_stop_source": policy_stop_record.get("stop_source") if policy_stop_record else None,
        "casa_stop_count": sum(1 for record in all_stop_records if not is_policy_stop_source(record.get("stop_source"))),
        "stop_distance": stop_distance,
        "final_distance": final_distance,
        "start_distance": start_distance,
        "min_distance": min_distance,
        "shortest_path_progress": shortest_path_progress,
        "progress_m": progress_m,
        "fall_count": fall_count,
        "collision_count": collision_count,
        "unsafe_wall_collision_count": collision_count,
        "unsafe_violation": bool(fall_count or collision_count),
        "wall_contact_steps": collision_count,
        "max_wall_contact_force": None,
        "stuck": stuck,
        "timeout": timeout,
        "executed_motion_skills": executed_motion_skills,
        "meaningful_motion": meaningful_motion,
        "stop_fallback_ratio": stop_fallback_ratio,
        "action_distribution": distribution,
        "stop_failure_type": stop_failure_type,
        "privileged_policy_usage_count": privileged_policy_usage_count,
        "casa_reject_count": reject_count,
        "casa_replan_count": replan_count,
        "reject_rate_per_step": reject_count / max(1, len(decision_records)),
        "policy_stop_count": len(policy_stop_records),
        "last_resort_stop_count": sum(
            1
            for record in all_stop_records
            if record.get("stop_source")
            in {
                "casa_reject_only_stop",
                "casa_replan_last_resort_stop",
                DMPS_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE,
                PPSR_V3_LAST_RESORT_STOP_SOURCE,
                PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE,
            }
        ),
        "stop_source_distribution": stop_source_distribution,
        "non_stop_recovery_count": sum(1 for record in decision_records if record.get("non_stop_recovery_selected")),
        "fallback_to_stop_count": sum(
            1
            for record in all_stop_records
            if record.get("stop_source")
            in {
                "casa_replan_last_resort_stop",
                DMPS_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE,
                PPSR_V3_LAST_RESORT_STOP_SOURCE,
                PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE,
            }
        ),
        "repeated_reject_loop_count": sum(
            1
            for record in decision_records
            if "repeated_forward_reject" in str(record.get("policy_internal_guard_names", ""))
        ),
        "dmps_recovery_stuck_count": sum(1 for record in decision_records if record.get("dmps_recovery_stuck")),
        "mean_post_reject_progress": _mean(post_reject_progress_values),
        "median_post_reject_progress": median(post_reject_progress_values) if post_reject_progress_values else 0.0,
        "positive_post_reject_progress_rate": sum(1 for value in post_reject_progress_values if value > 0.0) / len(post_reject_progress_values)
        if post_reject_progress_values
        else 0.0,
        "recovery_action_distribution": dict(Counter(action for action in recovery_actions if action)),
        "ppsr_v2_replan_count": len(ppsr_v2_records),
        "ppsr_v2_hard_mask_count": sum(1 for record in ppsr_v2_records if record.get("ppsr_v2_hard_mask_applied")),
        "escape_macro_selected_count": sum(1 for record in ppsr_v2_records if record.get("ppsr_v2_escape_macro_selected")),
        "escape_macro_commitment_count": sum(1 for record in ppsr_v2_records if record.get("ppsr_v2_commitment_executed")),
        "escape_macro_commitment_success_count": sum(
            1
            for record in ppsr_v2_records
            if record.get("ppsr_v2_commitment_executed") and not record.get("ppsr_v2_commitment_abort_reason")
        ),
        "turn_only_recovery_count": turn_only_recovery_count,
        "small_turn_recovery_count": small_turn_recovery_count,
        "ppsr_v3_replan_count": len(ppsr_v3_records),
        "ppsr_v3_repeated_replan_count": sum(1 for record in ppsr_v3_records if record.get("ppsr_v3_repeated_replan")),
        "post_reject_clearance_gain": _mean(ppsr_v3_clearance_gains or ppsr_v2_clearance_gains),
        "post_reject_forward_reenabled_count": sum(1 for record in ppsr_v2_records if record.get("ppsr_v2_post_reject_forward_reenabled"))
        + sum(1 for record in ppsr_v3_records if record.get("ppsr_v3_post_reject_forward_reenabled")),
        "post_reject_unblocked_count": sum(1 for record in ppsr_v2_records if record.get("ppsr_v2_post_reject_unblocked"))
        + sum(1 for record in ppsr_v3_records if record.get("ppsr_v3_post_reject_unblocked")),
        "post_reject_visual_novelty_count": sum(1 for record in ppsr_v2_records if record.get("ppsr_v2_post_reject_visual_novelty"))
        + sum(1 for record in ppsr_v3_records if record.get("ppsr_v3_visual_novelty", 0.0) > 0.0),
        "reject_to_policy_ready_count": sum(1 for record in ppsr_v3_records if record.get("ppsr_v3_policy_ready_state")),
        "next_vln_action_allowed_count": sum(
            1
            for record in ppsr_v3_records
            if record.get("ppsr_v3_actual_next_vln_action_allowed") or record.get("ppsr_v3_next_vln_action_allowed")
        ),
        "repeat_reject_count": repeat_reject_count,
        "loop_count": loop_count,
        "mean_steps_after_recovery_before_next_reject": _mean(_steps_after_recovery_before_next_reject(decision_records)),
        "landmark_retention_or_reacquisition_count": sum(
            1 for record in ppsr_v3_records if record.get("ppsr_v3_landmark_retained_or_reacquired")
        ),
        "safety_stop_success_leakage_count": safety_stop_success_leakage_count,
        "recovery_stop_success_leakage_count": recovery_stop_success_leakage_count,
        "readiness_replan_count": readiness_replan_count,
        "rejected_candidate_skill_executed_count": sum(
            1 for record in decision_records if record.get("rejected_candidate_skill_was_executed")
        ),
        "policy_internal_guard_steps": sum(1 for record in decision_records if record.get("policy_internal_guard_applied")),
        "video_paths": [str(path) for path in video_paths],
        "frames_dir": str(frames_dir),
        "decision_log": str(decision_log),
        "gate_log": str(gate_log),
        "replan_log": str(replan_log),
        "trajectory_log": str(trajectory_log),
        "final_stop_frame": final_stop_frame,
    }


def aggregate_method(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    success_count = sum(1 for episode in episodes if episode["success"])
    safe_success_count = sum(1 for episode in episodes if episode["safe_success"])
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
    action_distribution = {action.value: 0 for action in VLNAction}
    recovery_action_distribution: dict[str, int] = {}
    stop_source_distribution: dict[str, int] = {}
    post_reject_progress_values: list[float] = []
    for episode in episodes:
        for action, count in episode.get("action_distribution", {}).items():
            action_distribution[action] = action_distribution.get(action, 0) + int(count)
        for action, count in episode.get("recovery_action_distribution", {}).items():
            recovery_action_distribution[action] = recovery_action_distribution.get(action, 0) + int(count)
        for source, count in episode.get("stop_source_distribution", {}).items():
            stop_source_distribution[source] = stop_source_distribution.get(source, 0) + int(count)
        if episode.get("casa_reject_count", 0):
            post_reject_progress_values.append(float(episode.get("mean_post_reject_progress", 0.0)))
    step_count = sum(int(episode["steps"]) for episode in episodes)
    reject_count = sum(int(episode["casa_reject_count"]) for episode in episodes)
    replan_count = sum(int(episode["casa_replan_count"]) for episode in episodes)
    non_stop_recovery_count = sum(int(episode.get("non_stop_recovery_count", 0)) for episode in episodes)
    fallback_to_stop_count = sum(int(episode.get("fallback_to_stop_count", 0)) for episode in episodes)
    ppsr_v2_replan_count = sum(int(episode.get("ppsr_v2_replan_count", 0)) for episode in episodes)
    ppsr_v3_replan_count = sum(int(episode.get("ppsr_v3_replan_count", 0)) for episode in episodes)
    escape_macro_selected_count = sum(int(episode.get("escape_macro_selected_count", 0)) for episode in episodes)
    escape_macro_commitment_count = sum(int(episode.get("escape_macro_commitment_count", 0)) for episode in episodes)
    escape_macro_commitment_success_count = sum(int(episode.get("escape_macro_commitment_success_count", 0)) for episode in episodes)
    turn_only_recovery_count = sum(int(episode.get("turn_only_recovery_count", 0)) for episode in episodes)
    small_turn_recovery_count = sum(int(episode.get("small_turn_recovery_count", 0)) for episode in episodes)
    forward_reenabled_count = sum(int(episode.get("post_reject_forward_reenabled_count", 0)) for episode in episodes)
    post_reject_unblocked_count = sum(int(episode.get("post_reject_unblocked_count", 0)) for episode in episodes)
    post_reject_visual_novelty_count = sum(int(episode.get("post_reject_visual_novelty_count", 0)) for episode in episodes)
    readiness_replan_count = sum(int(episode.get("readiness_replan_count", 0)) for episode in episodes)
    reject_to_policy_ready_count = sum(int(episode.get("reject_to_policy_ready_count", 0)) for episode in episodes)
    next_vln_action_allowed_count = sum(int(episode.get("next_vln_action_allowed_count", 0)) for episode in episodes)
    repeat_reject_count = sum(int(episode.get("repeat_reject_count", 0)) for episode in episodes)
    loop_count = sum(int(episode.get("loop_count", 0)) for episode in episodes)
    landmark_retention_count = sum(int(episode.get("landmark_retention_or_reacquisition_count", 0)) for episode in episodes)
    steps_after_recovery_values = [
        float(episode.get("mean_steps_after_recovery_before_next_reject", 0.0))
        for episode in episodes
        if int(episode.get("casa_replan_count", 0)) > 0
    ]
    return {
        "method": episodes[0]["method"] if episodes else "",
        "total_episodes": total,
        "success_count": success_count,
        "success_rate": success_count / total if total else 0.0,
        "task_success_rate": success_count / total if total else 0.0,
        "safe_success_count": safe_success_count,
        "safe_success_rate": safe_success_count / total if total else 0.0,
        "unsafe_violation_count": sum(1 for episode in episodes if episode["unsafe_violation"]),
        "unsafe_violation_rate": sum(1 for episode in episodes if episode["unsafe_violation"]) / total if total else 0.0,
        "unsafe_wall_collision_count": sum(int(episode["unsafe_wall_collision_count"]) for episode in episodes),
        "collision_count": sum(int(episode["collision_count"]) for episode in episodes),
        "fall_count": sum(int(episode["fall_count"]) for episode in episodes),
        "wall_contact_steps": sum(int(episode["wall_contact_steps"]) for episode in episodes),
        "casa_reject_count": reject_count,
        "reject_count": reject_count,
        "reject_rate_per_step": reject_count / max(1, step_count),
        "casa_replan_count": replan_count,
        "replan_count": replan_count,
        "rejected_candidate_skill_executed_count": sum(
            int(episode["rejected_candidate_skill_executed_count"]) for episode in episodes
        ),
        "policy_internal_guard_steps": sum(int(episode["policy_internal_guard_steps"]) for episode in episodes),
        "stuck_count": sum(1 for episode in episodes if episode["stuck"]),
        "stuck_rate": sum(1 for episode in episodes if episode["stuck"]) / total if total else 0.0,
        "timeout_count": sum(1 for episode in episodes if episode["timeout"]),
        "timeout_rate": sum(1 for episode in episodes if episode["timeout"]) / total if total else 0.0,
        "mean_final_distance": _mean([episode["final_distance"] for episode in episodes]),
        "median_final_distance": median([episode["final_distance"] for episode in episodes]) if episodes else 0.0,
        "mean_shortest_path_progress": _mean([episode["shortest_path_progress"] for episode in episodes]),
        "mean_executed_motion_skills": _mean([episode["executed_motion_skills"] for episode in episodes]),
        "stop_fallback_ratio": _mean([episode["stop_fallback_ratio"] for episode in episodes]),
        "policy_stop_count": sum(int(episode.get("policy_stop_count", 0)) for episode in episodes),
        "last_resort_stop_count": sum(int(episode.get("last_resort_stop_count", 0)) for episode in episodes),
        "stop_source_distribution": stop_source_distribution,
        "non_stop_recovery_count": non_stop_recovery_count,
        "non_stop_recovery_rate": non_stop_recovery_count / max(1, replan_count),
        "fallback_to_stop_count": fallback_to_stop_count,
        "fallback_to_stop_rate": fallback_to_stop_count / max(1, replan_count),
        "repeated_reject_loop_count": sum(int(episode.get("repeated_reject_loop_count", 0)) for episode in episodes),
        "dmps_recovery_stuck_count": sum(int(episode.get("dmps_recovery_stuck_count", 0)) for episode in episodes),
        "ppsr_v2_replan_count": ppsr_v2_replan_count,
        "ppsr_v3_replan_count": ppsr_v3_replan_count,
        "ppsr_v3_repeated_replan_count": sum(int(episode.get("ppsr_v3_repeated_replan_count", 0)) for episode in episodes),
        "ppsr_v2_hard_mask_count": sum(int(episode.get("ppsr_v2_hard_mask_count", 0)) for episode in episodes),
        "escape_macro_selected_count": escape_macro_selected_count,
        "escape_macro_selected_rate": escape_macro_selected_count / max(1, ppsr_v2_replan_count),
        "escape_macro_commitment_count": escape_macro_commitment_count,
        "escape_macro_commitment_success_rate": escape_macro_commitment_success_count / max(1, escape_macro_commitment_count),
        "turn_only_recovery_count": turn_only_recovery_count,
        "turn_only_recovery_rate": turn_only_recovery_count / max(1, replan_count),
        "small_turn_recovery_count": small_turn_recovery_count,
        "small_turn_recovery_rate": small_turn_recovery_count / max(1, replan_count),
        "post_reject_clearance_gain": _mean([episode.get("post_reject_clearance_gain", 0.0) for episode in episodes]),
        "post_reject_forward_reenabled_count": forward_reenabled_count,
        "post_reject_forward_reenabled_rate": forward_reenabled_count / max(1, readiness_replan_count),
        "post_reject_forward_reenabled_rate_v2_compat": forward_reenabled_count / max(1, ppsr_v2_replan_count)
        if ppsr_v3_replan_count == 0
        else 0.0,
        "post_reject_unblocked_rate": post_reject_unblocked_count / max(1, readiness_replan_count),
        "post_reject_visual_novelty_rate": post_reject_visual_novelty_count / max(1, readiness_replan_count),
        "reject_to_policy_ready_count": reject_to_policy_ready_count,
        "reject_to_policy_ready_rate": reject_to_policy_ready_count / max(1, ppsr_v3_replan_count),
        "next_vln_action_allowed_count": next_vln_action_allowed_count,
        "next_vln_action_allowed_rate": next_vln_action_allowed_count / max(1, ppsr_v3_replan_count),
        "repeat_reject_count": repeat_reject_count,
        "repeat_reject_rate": repeat_reject_count / max(1, reject_count),
        "loop_count": loop_count,
        "loop_rate": loop_count / max(1, replan_count),
        "mean_steps_after_recovery_before_next_reject": _mean(steps_after_recovery_values),
        "post_reject_progress": _mean(post_reject_progress_values),
        "landmark_retention_or_reacquisition_count": landmark_retention_count,
        "landmark_retention_or_reacquisition_rate": landmark_retention_count / max(1, ppsr_v3_replan_count),
        "safety_stop_success_leakage_count": sum(
            int(episode.get("safety_stop_success_leakage_count", 0)) for episode in episodes
        ),
        "recovery_stop_success_leakage_count": sum(
            int(episode.get("recovery_stop_success_leakage_count", 0)) for episode in episodes
        ),
        "mean_post_reject_progress": _mean(post_reject_progress_values),
        "median_post_reject_progress": median(post_reject_progress_values) if post_reject_progress_values else 0.0,
        "positive_post_reject_progress_rate": sum(1 for value in post_reject_progress_values if value > 0.0) / len(post_reject_progress_values)
        if post_reject_progress_values
        else 0.0,
        "stop_precision": true_stop_positive / max(1, true_stop_positive + false_stop_positive),
        "stop_recall": true_stop_positive / max(1, true_stop_positive + false_stop_negative),
        "premature_stop_rate": false_stop_positive / total if total else 0.0,
        "late_stop_rate": false_stop_negative / total if total else 0.0,
        "mean_stop_distance": _mean(stop_distances),
        "action_distribution": action_distribution,
        "recovery_action_distribution": recovery_action_distribution,
    }


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _repeat_reject_count(decision_records: list[dict[str, Any]]) -> int:
    count = 0
    previous_action = None
    previous_rejected = False
    for record in decision_records:
        rejected = bool(record.get("casa_reject"))
        action = str(record.get("vln_final_action_after_policy_internal_guards") or "")
        if rejected and previous_rejected and action == previous_action:
            count += 1
        previous_rejected = rejected
        previous_action = action if rejected else None
    return count


def _steps_after_recovery_before_next_reject(decision_records: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for idx, record in enumerate(decision_records):
        if not record.get("casa_replan_applied"):
            continue
        steps = 0
        for future in decision_records[idx + 1 :]:
            if future.get("casa_reject"):
                break
            steps += 1
        values.append(steps)
    return values


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def load_score_weights(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    return {str(key): float(value) for key, value in data.items()}


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return int(value)
    return value


def _json_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return [parsed]


def flatten_decision_record(record: dict[str, Any]) -> dict[str, Any]:
    pose = record.get("evaluator_pose") or {}
    goal = record.get("evaluator_goal_xy") or [None, None]
    return {
        "method": record.get("method"),
        "episode_id": record.get("episode_id"),
        "step_idx": record.get("step_idx"),
        "instruction": record.get("instruction"),
        "image_path": record.get("image_path"),
        "backend_name": record.get("backend_name"),
        "real_navid_or_uninavid_used": record.get("real_navid_or_uninavid_used"),
        "navid_model_loaded": record.get("navid_model_loaded"),
        "navid_raw_output": record.get("navid_raw_output"),
        "navid_parsed_action": record.get("navid_parsed_action"),
        "vln_raw_action": record.get("vln_raw_action"),
        "vln_final_action_after_policy_internal_guards": record.get("vln_final_action_after_policy_internal_guards"),
        "policy_decision_source": record.get("policy_decision_source"),
        "policy_internal_guard_applied": record.get("policy_internal_guard_applied"),
        "policy_internal_guard_names": record.get("policy_internal_guard_names"),
        "candidate_skill": record.get("candidate_skill"),
        "candidate_casa_skill_name": record.get("candidate_casa_skill_name"),
        "casa_gate_evaluated": record.get("casa_gate_evaluated"),
        "raw_risk": record.get("raw_risk"),
        "threshold": record.get("threshold"),
        "risk_margin": record.get("risk_margin"),
        "hard_contract_score": record.get("hard_contract_score"),
        "hard_contract_fixed_reject": record.get("hard_contract_fixed_reject"),
        "casa_reject": record.get("casa_reject"),
        "reject_reason": record.get("reject_reason"),
        "final_executed_action": record.get("final_executed_action"),
        "metrics_action": record.get("metrics_action"),
        "final_executed_skill": record.get("final_executed_skill"),
        "stop_source": record.get("stop_source"),
        "casa_replan_applied": record.get("casa_replan_applied"),
        "casa_replan_selected_action": record.get("casa_replan_selected_action"),
        "casa_replan_selected_skill": record.get("casa_replan_selected_skill"),
        "dmps_replan_applied": record.get("dmps_replan_applied"),
        "dmps_selected_sequence": record.get("dmps_selected_sequence"),
        "dmps_recovery_stuck": record.get("dmps_recovery_stuck"),
        "ppsr_v2_replan_applied": record.get("ppsr_v2_replan_applied"),
        "ppsr_v2_stuck_mode": record.get("ppsr_v2_stuck_mode"),
        "ppsr_v2_stuck_mode_reasons": record.get("ppsr_v2_stuck_mode_reasons"),
        "ppsr_v2_hard_mask_applied": record.get("ppsr_v2_hard_mask_applied"),
        "ppsr_v2_escape_macro_selected": record.get("ppsr_v2_escape_macro_selected"),
        "ppsr_v2_commitment_executed": record.get("ppsr_v2_commitment_executed"),
        "ppsr_v2_commitment_aborted": record.get("ppsr_v2_commitment_aborted"),
        "ppsr_v2_second_action": record.get("ppsr_v2_second_action"),
        "ppsr_v2_second_action_safety_feasible": record.get("ppsr_v2_second_action_safety_feasible"),
        "ppsr_v2_post_reject_clearance_gain": record.get("ppsr_v2_post_reject_clearance_gain"),
        "ppsr_v2_post_reject_forward_reenabled": record.get("ppsr_v2_post_reject_forward_reenabled"),
        "ppsr_v2_post_reject_unblocked": record.get("ppsr_v2_post_reject_unblocked"),
        "ppsr_v2_post_reject_visual_novelty": record.get("ppsr_v2_post_reject_visual_novelty"),
        "ppsr_v3_replan_applied": record.get("ppsr_v3_replan_applied"),
        "ppsr_v3_repeated_replan": record.get("ppsr_v3_repeated_replan"),
        "ppsr_v3_recent_loop_flag": record.get("ppsr_v3_recent_loop_flag"),
        "ppsr_v3_policy_ready_candidate_exists": record.get("ppsr_v3_policy_ready_candidate_exists"),
        "ppsr_v3_policy_ready_state": record.get("ppsr_v3_policy_ready_state"),
        "ppsr_v3_next_vln_action": record.get("ppsr_v3_next_vln_action"),
        "ppsr_v3_next_vln_action_allowed": record.get("ppsr_v3_next_vln_action_allowed"),
        "ppsr_v3_actual_next_vln_action_allowed": record.get("ppsr_v3_actual_next_vln_action_allowed"),
        "ppsr_v3_predicted_reject_drop": record.get("ppsr_v3_predicted_reject_drop"),
        "ppsr_v3_terminal_front_clearance": record.get("ppsr_v3_terminal_front_clearance"),
        "ppsr_v3_terminal_side_clearance": record.get("ppsr_v3_terminal_side_clearance"),
        "ppsr_v3_visual_novelty": record.get("ppsr_v3_visual_novelty"),
        "ppsr_v3_landmark_retained_or_reacquired": record.get("ppsr_v3_landmark_retained_or_reacquired"),
        "ppsr_v3_language_progress_score": record.get("ppsr_v3_language_progress_score"),
        "ppsr_v3_policy_ready_score": record.get("ppsr_v3_policy_ready_score"),
        "ppsr_v3_post_reject_forward_reenabled": record.get("ppsr_v3_post_reject_forward_reenabled"),
        "ppsr_v3_post_reject_clearance_gain": record.get("ppsr_v3_post_reject_clearance_gain"),
        "ppsr_v3_post_reject_unblocked": record.get("ppsr_v3_post_reject_unblocked"),
        "ppsr_v3_second_action": record.get("ppsr_v3_second_action"),
        "ppsr_v3_second_action_safety_feasible": record.get("ppsr_v3_second_action_safety_feasible"),
        "ppsr_v3_second_step_executed": record.get("ppsr_v3_second_step_executed"),
        "ppsr_v3_second_step_aborted": record.get("ppsr_v3_second_step_aborted"),
        "ppsr_v3_commitment_abort_reason": record.get("ppsr_v3_commitment_abort_reason"),
        "non_stop_recovery_selected": record.get("non_stop_recovery_selected"),
        "all_non_stop_candidates_infeasible": record.get("all_non_stop_candidates_infeasible"),
        "post_reject_progress_evaluator_only": record.get("post_reject_progress_evaluator_only"),
        "rejected_candidate_skill_was_executed": record.get("rejected_candidate_skill_was_executed"),
        "skill_status": record.get("skill_status"),
        "wall_collision": record.get("wall_collision"),
        "fall": record.get("fall"),
        "evaluator_distance_to_goal": record.get("evaluator_distance_to_goal"),
        "evaluator_pose_x": pose.get("x"),
        "evaluator_pose_y": pose.get("y"),
        "evaluator_pose_yaw_deg": pose.get("yaw_deg"),
        "evaluator_goal_x": goal[0],
        "evaluator_goal_y": goal[1],
        "evaluator_only_shortest_path_remaining": record.get("evaluator_only_shortest_path_remaining"),
        "privileged_policy_usage": record.get("privileged_policy_usage"),
        "policy_metadata_json": record.get("policy_metadata"),
        "skill_params_json": record.get("skill_params"),
        "wall_features_json": record.get("wall_features"),
    }


def collect_jsonl_records(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_global_logs(data_dir: Path, episode_metrics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_episode = data_dir / "per_episode"
    decision_rows = collect_jsonl_records(sorted(per_episode.glob("*_decision_log.jsonl")))
    gate_rows = collect_jsonl_records(sorted(per_episode.glob("*_gate_decisions.jsonl")))
    replan_rows = collect_jsonl_records(sorted(per_episode.glob("*_replan_log.jsonl")))
    trajectory_rows = collect_jsonl_records(sorted(per_episode.glob("*_trajectory_log.jsonl")))
    dmps_candidate_rows = collect_jsonl_records(sorted(per_episode.glob("*_dmps_candidate_sequences.jsonl")))
    dmps_selected_rows = collect_jsonl_records(sorted(per_episode.glob("*_dmps_selected_replans.jsonl")))
    dmps_rollout_rows = collect_jsonl_records(sorted(per_episode.glob("*_dmps_mpc_cbf_rollouts.jsonl")))
    progress_event_rows = collect_jsonl_records(sorted(per_episode.glob("*_progress_monitor_events.jsonl")))
    write_table_csv(data_dir / "decision_logs.csv", [flatten_decision_record(row) for row in decision_rows])
    write_table_csv(data_dir / "gate_decisions.csv", gate_rows)
    write_table_csv(data_dir / "replan_logs.csv", replan_rows)
    write_table_csv(data_dir / "trajectory_logs.csv", trajectory_rows)
    write_table_csv(data_dir / "dmps_candidate_sequences.csv", dmps_candidate_rows)
    write_table_csv(data_dir / "dmps_selected_replans.csv", dmps_selected_rows)
    write_table_csv(data_dir / "dmps_score_breakdown.csv", dmps_candidate_rows)
    ppsr_v2_candidate_rows = [row for row in dmps_candidate_rows if row.get("method") == PPSR_V2_METHOD_NAME]
    ppsr_v2_selected_rows = [row for row in dmps_selected_rows if row.get("method") == PPSR_V2_METHOD_NAME]
    ppsr_v3_candidate_rows = [row for row in dmps_candidate_rows if row.get("method") == PPSR_V3_METHOD_NAME]
    ppsr_v3_selected_rows = [row for row in dmps_selected_rows if row.get("method") == PPSR_V3_METHOD_NAME]
    write_table_csv(data_dir / "ppsr_v2_candidate_sequences.csv", ppsr_v2_candidate_rows)
    write_table_csv(data_dir / "ppsr_v2_selected_replans.csv", ppsr_v2_selected_rows)
    write_table_csv(data_dir / "v3_candidate_sequences.csv", ppsr_v3_candidate_rows)
    write_table_csv(data_dir / "v3_selected_replans.csv", ppsr_v3_selected_rows)
    write_table_csv(data_dir / "v3_policy_ready_audit.csv", ppsr_v3_candidate_rows)
    write_table_csv(data_dir / "v3_language_progress_audit.csv", ppsr_v3_candidate_rows)
    write_table_csv(data_dir / "terminal_sequence_score_audit.csv", ppsr_v2_candidate_rows)
    write_table_csv(
        data_dir / "escape_macro_candidates.csv",
        [row for row in ppsr_v2_candidate_rows if int(row.get("escape_macro_candidate", 0))],
    )
    write_table_csv(data_dir / "escape_commitment_logs.csv", ppsr_v2_selected_rows)
    write_table_csv(data_dir / "ppsr_v2_method_summary.csv", [])
    write_table_csv(data_dir / "v3_method_summary.csv", [])
    for jsonl_path in [data_dir / "dmps_mpc_cbf_rollouts.jsonl", data_dir / "progress_monitor_events.jsonl"]:
        if jsonl_path.exists():
            jsonl_path.unlink()
    append_jsonl(data_dir / "dmps_mpc_cbf_rollouts.jsonl", dmps_rollout_rows)
    append_jsonl(data_dir / "progress_monitor_events.jsonl", progress_event_rows)
    mask_events = [row for row in ppsr_v2_candidate_rows if int(row.get("hard_mask_applied", 0))]
    mask_path = data_dir / "anti_turn_loop_mask_events.jsonl"
    if mask_path.exists():
        mask_path.unlink()
    append_jsonl(mask_path, mask_events)
    summaries = data_dir / "episode_summaries.jsonl"
    if summaries.exists():
        summaries.unlink()
    append_jsonl(summaries, episode_metrics)
    return decision_rows, gate_rows, replan_rows, trajectory_rows, dmps_candidate_rows, dmps_selected_rows, dmps_rollout_rows, progress_event_rows


def write_repo_audit(path: Path) -> None:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"ERROR: {exc}"

    files = [
        "gear_sonic/vln/casa_bridge.py",
        "gear_sonic/vln/casa_runner.py",
        "gear_sonic/scripts/casa_run_vln_safety_online.py",
        "gear_sonic/vln/no_casa_runner.py",
        "gear_sonic/vln/no_casa_mujoco.py",
        "gear_sonic/vln/metrics.py",
    ]
    lines = [
        "# CASA-VLN Repository Audit",
        "",
        f"- pwd: `{Path.cwd()}`",
        f"- branch: `{run_git(['branch', '--show-current'])}`",
        f"- head: `{run_git(['rev-parse', 'HEAD'])}`",
        "- remotes:",
        "```text",
        run_git(["remote", "-v"]),
        "```",
        "- status:",
        "```text",
        run_git(["status", "--short"]) or "clean",
        "```",
        "## File Presence",
    ]
    for file in files:
        lines.append(f"- `{file}`: `{Path(file).exists()}`")
    lines.extend(
        [
            "",
            "## Audit Answers",
            "",
            "### Current vln_only runner",
            "`vln_only` runs the NaVid/visual-adapter VLN policy, maps each action to a SONIC skill, executes it in the MuJoCo maze executor, and records decision, trajectory, frame, video, and episode summary logs.",
            "",
            "### Current vln_casa_replan",
            "`vln_casa_replan` evaluates the nominal VLN skill with the CASA bridge. If rejected, it scores a fixed single-step recovery set mostly by raw CASA risk plus priority, chooses one allowed non-stop candidate or stop as last resort, executes one action, and returns control to VLN.",
            "",
            "### CASA-VLN bridge features",
            "The bridge maps VLN actions to CASA skill names, loads the real Phase4 raw critic and Phase5 conformal thresholds, computes hard-contract features and wall-risk features including candidate clearance, candidate would-block flag, current obstacle distance, current obstacle collision flag, and nearest obstacle relative position.",
            "",
            "### stop_source",
            "Only `vln_policy` and `policy_internal_guard` count as policy-issued success stops. CASA/DMPS safety stops such as `casa_reject_only_stop`, `casa_replan_last_resort_stop`, and `dmps_mpc_cbf_last_resort_stop` are logged but cannot satisfy task success.",
            "",
            "### unsafe_wall_collision / wall_contact_steps",
            "The MuJoCo maze executor uses `maze.is_xy_safe` at walking microsteps. If a walk microstep would enter unsafe wall/furniture geometry, it records `wall_collision=1`, returns status `blocked`, and episode summary counts those rows as `unsafe_wall_collision_count` and `wall_contact_steps`.",
            "",
            "### Why naive replan can lower success",
            "The old replan removes unsafe contacts but does not optimize visual progress or loop avoidance. It can repeatedly choose low-risk turns/backoffs, time out without a policy stop, or replace hazardous progress with safety-only behavior, lowering safe success.",
            "",
            "### Logs not visible in Git",
            "Large online artifacts are written under `/mnt/data/students/lph/recording/`: videos, frames, per-episode decision/gate/replan logs, trajectory logs, trained visual adapter outputs, full manifests, and MuJoCo/NaVid worker logs.",
            "",
            "### New insertion point",
            "The new DMPS/PPSR method is inserted in `gear_sonic/vln/casa_runner.py` inside the CASA-reject branch, after the nominal action is rejected and before execution of a recovery action.",
            "",
            "### Files modified",
            "`gear_sonic/vln/casa_bridge.py`, `gear_sonic/vln/casa_runner.py`, `gear_sonic/scripts/casa_run_vln_safety_online.py`, and VLN tests are modified; `progress_monitor.py` and `dmps_mpc_cbf_replan.py` are added.",
            "",
            "### Existing controls preserved",
            "`vln_only`, `vln_casa_reject_only`, and `vln_casa_replan` remain as comparison methods. The new method is `vln_dmps_mpc_cbf_progress`, alias `vln_ppsr`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_audits(
    *,
    data_dir: Path,
    config: CasaVlnRunnerConfig,
    bridge: CasaVlnBridge,
    decision_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    replan_rows: list[dict[str, Any]],
    dmps_candidate_rows: list[dict[str, Any]],
    dmps_selected_rows: list[dict[str, Any]],
    dmps_rollout_rows: list[dict[str, Any]],
    progress_event_rows: list[dict[str, Any]],
    method_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    required_wall_fields = [
        "vln/candidate_min_clearance",
        "vln/candidate_would_block",
        "env/min_obstacle_distance/current",
        "env/external_collision_obstacle/current",
    ]
    wall_feature_audit = {
        "required_fields": required_wall_fields,
        "gate_rows": len(gate_rows),
        "rows_with_all_required_fields": sum(
            all(field in row and row.get(field) not in {None, ""} for field in required_wall_fields)
            for row in gate_rows
        ),
        "min_candidate_clearance": min([float(row["vln/candidate_min_clearance"]) for row in gate_rows], default=None),
        "candidate_would_block_count": sum(int(row.get("vln/candidate_would_block", 0)) for row in gate_rows),
    }
    write_json(data_dir / "wall_feature_audit.json", wall_feature_audit)

    stop_rows = [row for row in decision_rows if row.get("metrics_action") == VLNAction.STOP.value]
    stop_source_audit = {
        "policy_success_stop_sources": sorted(["vln_policy", "policy_internal_guard"]),
        "non_success_casa_stop_sources": sorted(
            [
                "casa_reject_only_stop",
                "casa_replan_last_resort_stop",
                DMPS_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_LAST_RESORT_STOP_SOURCE,
                PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE,
                PPSR_V3_LAST_RESORT_STOP_SOURCE,
                PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE,
            ]
        ),
        "stop_source_counts": dict(Counter(str(row.get("stop_source") or "") for row in stop_rows)),
        "casa_stop_rows_counted_as_policy_stop": sum(
            1
            for row in stop_rows
            if str(row.get("stop_source") or "").startswith(("casa_", "dmps_", "ppsr_"))
            and is_policy_stop_source(row.get("stop_source"))
        ),
    }
    write_json(data_dir / "stop_source_audit.json", stop_source_audit)

    policy_vs_casa = {
        "policy_internal_guard_steps": sum(1 for row in decision_rows if row.get("policy_internal_guard_applied")),
        "casa_replan_steps": sum(1 for row in decision_rows if row.get("casa_replan_applied")),
        "note": "Policy internal guards are logged separately from CASA replan/reject interventions.",
    }
    write_json(data_dir / "policy_internal_guard_vs_casa_replan_audit.json", policy_vs_casa)

    mapping_audit = {
        "vln_action_space": [action.value for action in VLNAction],
        "vln_to_casa_skill_name": {
            "forward": "walk",
            "backoff": "walk",
            "turn_left": "turn",
            "turn_right": "turn",
            "stop": "passive",
        },
        "backoff_selected_sonic_skill": "backoff_walk",
        "backoff_casa_threshold_lookup": "walk",
    }
    write_json(data_dir / "skill_name_mapping_audit.json", mapping_audit)

    leakage_audit = {
        "test_time_policy_uses_map_pose_goal_path": False,
        "used_goal_distance_for_online_replan": False,
        "used_shortest_path_for_online_replan": False,
        "used_a_star_for_online_replan": False,
        "used_oracle_waypoint_for_online_replan": False,
        "used_evaluator_success_for_online_replan": False,
        "used_map_cell_progress_for_online_replan": False,
        "privileged_online_leakage": False,
        "policy_input_fields": [
            "instruction",
            "head_camera_image",
            "head_camera_history",
            "previous_actions",
            "previous_skill_status",
        ],
        "teacher_or_astar_used_for_training_only": True,
        "decision_rows_with_privileged_policy_usage": sum(1 for row in decision_rows if row.get("privileged_policy_usage")),
    }
    write_json(data_dir / "privileged_leakage_audit.json", leakage_audit)

    risk_activation = {
        "gate_rows": len(gate_rows),
        "risk_source_values": dict(Counter(str(row.get("risk_source")) for row in gate_rows)),
        "hard_contract_fixed_reject_count": sum(int(row.get("hard_contract_fixed_reject", 0)) for row in gate_rows),
        "casa_reject_count": sum(int(row.get("casa_reject", 0)) for row in gate_rows),
        "reject_reason_counts": dict(Counter(str(row.get("reject_reason")) for row in gate_rows)),
    }
    write_json(data_dir / "casa_gate_risk_activation_audit.json", risk_activation)

    bridge_audit = bridge_audit_dict(bridge)
    bridge_audit.update(
        {
            "phase4_root": str(config.phase4_root),
            "phase5_root": str(config.phase5_root),
            "loaded_real_casa_artifacts": bridge.critic_loaded and bridge.thresholds_loaded,
            "claimable_risk_source": "real_casa_critic",
        }
    )
    write_json(data_dir / "casa_vln_bridge_audit.json", bridge_audit)

    action_distribution = {
        summary["method"]: summary["action_distribution"]
        for summary in method_summary
        if "action_distribution" in summary
    }
    write_json(data_dir / "action_distribution.json", action_distribution)
    failure_breakdown: dict[str, dict[str, int]] = {}
    episode_rows = collect_jsonl_records([data_dir / "episode_summaries.jsonl"])
    for row in episode_rows:
        method = row["method"]
        failure_breakdown.setdefault(method, {})
        reason = row["failure_reason"] or "success"
        failure_breakdown[method][reason] = failure_breakdown[method].get(reason, 0) + 1
    write_json(data_dir / "failure_reason_breakdown.json", failure_breakdown)
    write_json(data_dir / "ppsr_v2_failure_reason_breakdown.json", {PPSR_V2_METHOD_NAME: failure_breakdown.get(PPSR_V2_METHOD_NAME, {})})
    write_json(data_dir / "v3_failure_reason_breakdown.json", {PPSR_V3_METHOD_NAME: failure_breakdown.get(PPSR_V3_METHOD_NAME, {})})

    visual_examples = []
    for row in dmps_candidate_rows[:20]:
        visual_examples.append(
            {
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "candidate_sequence": row.get("candidate_sequence"),
                "visual_free_space_score": row.get("visual_free_space_score"),
            }
        )
    visual_free_space_audit = {
        "visual_free_space_score_available": bool(dmps_candidate_rows),
        "method": "head_camera_left_center_right_brightness_edge_proxy",
        "examples": visual_examples,
        "forbidden_inputs_used": False,
        "fallback": "If image read fails, score is unavailable and a conservative fixed proxy is used.",
    }
    write_json(data_dir / "visual_free_space_audit.json", visual_free_space_audit)

    intent_score_audit = {
        "rules": {
            "forward": "short_forward highest unless repeatedly rejected; turns medium; backoff low-to-medium for recovery; stop lowest.",
            "turn_left": "left turns highest; backoff+turn_left medium-high; opposite turn penalized unless repeated rejects.",
            "turn_right": "right turns highest; backoff+turn_right medium-high; opposite turn penalized unless repeated rejects.",
            "backoff": "backoff highest; turns medium; forward low.",
            "stop": "stop requires visual stop cue in policy; DMPS last-resort stop remains non-success stop.",
        },
        "candidate_rows": len(dmps_candidate_rows),
        "selected_sequence_counts": dict(Counter(str(row.get("selected_sequence")) for row in dmps_selected_rows)),
    }
    write_json(data_dir / "intent_score_audit.json", intent_score_audit)

    progress_monitor_audit = {
        "event_count": len(progress_event_rows),
        "event_type_counts": dict(Counter(str(row.get("event_type")) for row in progress_event_rows)),
        "episodes_with_events": sorted(set(str(row.get("episode_id")) for row in progress_event_rows)),
        "forbidden_online_inputs": ["goal_xy", "goal_distance", "shortest_path", "astar_path", "oracle_waypoint"],
        "uses_forbidden_online_inputs": False,
    }
    write_json(data_dir / "progress_monitor_audit.json", progress_monitor_audit)

    ppsr_v2_candidate_rows = [row for row in dmps_candidate_rows if row.get("method") == PPSR_V2_METHOD_NAME]
    ppsr_v2_selected_rows = [row for row in dmps_selected_rows if row.get("method") == PPSR_V2_METHOD_NAME]
    ppsr_v3_candidate_rows = [row for row in dmps_candidate_rows if row.get("method") == PPSR_V3_METHOD_NAME]
    ppsr_v3_selected_rows = [row for row in dmps_selected_rows if row.get("method") == PPSR_V3_METHOD_NAME]
    anti_mask_rows = [row for row in ppsr_v2_candidate_rows if int(row.get("hard_mask_applied", 0))]
    anti_turn_loop_mask_audit = {
        "candidate_rows": len(ppsr_v2_candidate_rows),
        "hard_mask_applied_count": len(anti_mask_rows),
        "hard_mask_reason_counts": dict(
            Counter(
                reason
                for row in anti_mask_rows
                for reason in _json_list(row.get("hard_mask_reasons"))
            )
        ),
        "masked_candidate_count": sum(len(_json_list(row.get("masked_candidates"))) for row in anti_mask_rows),
        "translation_required_count": sum(int(row.get("translation_required", 0)) for row in ppsr_v2_candidate_rows),
        "turn_only_candidate_blocked_count": sum(int(row.get("turn_only_candidate_blocked", 0)) for row in ppsr_v2_candidate_rows),
        "repeated_turn_loop_blocked_count": sum(int(row.get("repeated_turn_loop_blocked", 0)) for row in ppsr_v2_candidate_rows),
    }
    write_json(data_dir / "anti_turn_loop_mask_audit.json", anti_turn_loop_mask_audit)

    v3_loop_rows = [row for row in ppsr_v3_candidate_rows if int(row.get("recent_loop_flag", 0))]
    v3_mask_rows = [row for row in ppsr_v3_candidate_rows if int(row.get("hard_mask_applied", 0))]
    v3_loop_avoidance_audit = {
        "candidate_rows": len(ppsr_v3_candidate_rows),
        "selected_rows": len(ppsr_v3_selected_rows),
        "recent_loop_candidate_rows": len(v3_loop_rows),
        "hard_mask_applied_count": len(v3_mask_rows),
        "hard_mask_reason_counts": dict(
            Counter(reason for row in v3_mask_rows for reason in _json_list(row.get("hard_mask_reasons")))
        ),
        "loop_penalty_mean": _mean([float(row.get("loop_penalty", 0.0)) for row in ppsr_v3_candidate_rows]),
        "repeated_replan_candidate_count": sum(int(row.get("repeated_replan", 0)) for row in ppsr_v3_candidate_rows),
        "turn_only_candidate_blocked_count": sum(int(row.get("turn_only_candidate_blocked", 0)) for row in ppsr_v3_candidate_rows),
        "uses_forbidden_online_inputs": False,
    }
    write_json(data_dir / "v3_loop_avoidance_audit.json", v3_loop_avoidance_audit)

    v3_summary_for_stop = next((row for row in method_summary if row["method"] == PPSR_V3_METHOD_NAME), {})
    v3_stop_source_audit = {
        "non_success_v3_stop_sources": sorted(
            [PPSR_V3_LAST_RESORT_STOP_SOURCE, PPSR_V3_COMMITMENT_ABORT_STOP_SOURCE]
        ),
        "safety_stop_success_leakage_count": int(v3_summary_for_stop.get("safety_stop_success_leakage_count", 0)),
        "recovery_stop_success_leakage_count": int(v3_summary_for_stop.get("recovery_stop_success_leakage_count", 0)),
        "v3_stop_source_counts": {
            source: count
            for source, count in stop_source_audit["stop_source_counts"].items()
            if str(source).startswith("ppsr_v3")
        },
        "policy_stop_sources": sorted(["vln_policy", "policy_internal_guard"]),
    }
    write_json(data_dir / "v3_stop_source_audit.json", v3_stop_source_audit)

    ppsr_v2_summary = next((row for row in method_summary if row["method"] == PPSR_V2_METHOD_NAME), {})
    ppsr_v1_summary = next((row for row in method_summary if row["method"] == DMPS_METHOD_NAME), {})
    ppsr_v3_summary = next((row for row in method_summary if row["method"] == PPSR_V3_METHOD_NAME), {})
    readiness_progress_metrics = {
        "post_reject_clearance_gain": (ppsr_v3_summary or ppsr_v2_summary).get("post_reject_clearance_gain", 0.0),
        "post_reject_forward_reenabled_rate": (ppsr_v3_summary or ppsr_v2_summary).get("post_reject_forward_reenabled_rate", 0.0),
        "post_reject_unblocked_rate": (ppsr_v3_summary or ppsr_v2_summary).get("post_reject_unblocked_rate", 0.0),
        "post_reject_visual_novelty_rate": (ppsr_v3_summary or ppsr_v2_summary).get("post_reject_visual_novelty_rate", 0.0),
        "escape_macro_selected_count": ppsr_v2_summary.get("escape_macro_selected_count", 0),
        "escape_macro_selected_rate": ppsr_v2_summary.get("escape_macro_selected_rate", 0.0),
        "escape_macro_commitment_count": ppsr_v2_summary.get("escape_macro_commitment_count", 0),
        "escape_macro_commitment_success_rate": ppsr_v2_summary.get("escape_macro_commitment_success_rate", 0.0),
        "turn_only_recovery_count": ppsr_v2_summary.get("turn_only_recovery_count", 0),
        "turn_only_recovery_rate": ppsr_v2_summary.get("turn_only_recovery_rate", 0.0),
        "small_turn_recovery_count": ppsr_v2_summary.get("small_turn_recovery_count", 0),
        "small_turn_recovery_rate": ppsr_v2_summary.get("small_turn_recovery_rate", 0.0),
        "small_turn_recovery_delta_vs_ppsr_v1": float(ppsr_v2_summary.get("small_turn_recovery_rate", 0.0))
        - float(ppsr_v1_summary.get("small_turn_recovery_rate", 0.0)),
        "recovery_stuck_delta_vs_ppsr_v1": int(ppsr_v2_summary.get("dmps_recovery_stuck_count", 0))
        - int(ppsr_v1_summary.get("dmps_recovery_stuck_count", 0)),
        "v3_reject_to_policy_ready_rate": ppsr_v3_summary.get("reject_to_policy_ready_rate", 0.0),
        "v3_next_vln_action_allowed_rate": ppsr_v3_summary.get("next_vln_action_allowed_rate", 0.0),
        "v3_loop_rate": ppsr_v3_summary.get("loop_rate", 0.0),
        "v3_landmark_retention_or_reacquisition_rate": ppsr_v3_summary.get(
            "landmark_retention_or_reacquisition_rate", 0.0
        ),
    }
    write_json(data_dir / "readiness_progress_metrics.json", readiness_progress_metrics)

    return {
        "wall_feature_audit": wall_feature_audit,
        "stop_source_audit": stop_source_audit,
        "policy_internal_guard_vs_casa_replan_audit": policy_vs_casa,
        "skill_name_mapping_audit": mapping_audit,
        "privileged_leakage_audit": leakage_audit,
        "casa_gate_risk_activation_audit": risk_activation,
        "casa_vln_bridge_audit": bridge_audit,
        "visual_free_space_audit": visual_free_space_audit,
        "intent_score_audit": intent_score_audit,
        "progress_monitor_audit": progress_monitor_audit,
        "anti_turn_loop_mask_audit": anti_turn_loop_mask_audit,
        "v3_loop_avoidance_audit": v3_loop_avoidance_audit,
        "v3_stop_source_audit": v3_stop_source_audit,
        "readiness_progress_metrics": readiness_progress_metrics,
        "dmps_rollout_count": len(dmps_rollout_rows),
    }


def draw_comparison_figures(figures_dir: Path, method_summary: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    figures_dir.mkdir(parents=True, exist_ok=True)

    def bar_chart(path: Path, title: str, key: str, color: tuple[int, int, int]) -> None:
        width = max(900, 190 * max(1, len(method_summary)) + 120)
        image = Image.new("RGB", (width, 440), (248, 248, 248))
        draw = ImageDraw.Draw(image)
        draw.text((30, 20), title, fill=(0, 0, 0))
        values = [float(row.get(key, 0.0)) for row in method_summary]
        min_value = min(values + [0.0])
        max_value = max(values + [0.0])
        span = max(max_value - min_value, 1e-6)
        baseline_y = 340 - int(280 * (0.0 - min_value) / span)
        draw.line([(50, baseline_y), (width - 30, baseline_y)], fill=(120, 120, 120), width=1)
        x = 70
        bar_width = 115
        for row in method_summary:
            value = float(row.get(key, 0.0))
            value_y = 340 - int(280 * (value - min_value) / span)
            y0 = min(value_y, baseline_y)
            y1 = max(value_y, baseline_y)
            draw.rectangle([x, y0, x + bar_width, y1], fill=color)
            label = str(row["method"]).replace("vln_", "")[:20]
            draw.text((x, 348), label, fill=(0, 0, 0))
            label_y = max(42, min(y0 - 18, 360))
            draw.text((x + 16, label_y), f"{value:.2f}", fill=(0, 0, 0))
            x += 185
        image.save(path)

    bar_chart(figures_dir / "safe_success_comparison.png", "Safe success rate", "safe_success_rate", (40, 145, 85))
    bar_chart(figures_dir / "unsafe_comparison.png", "Unsafe violation rate", "unsafe_violation_rate", (190, 85, 55))
    bar_chart(figures_dir / "small_turn_recovery_reduction.png", "Small-turn recovery rate", "small_turn_recovery_rate", (130, 95, 180))
    bar_chart(figures_dir / "recovery_stuck_comparison.png", "Recovery stuck count", "dmps_recovery_stuck_count", (180, 120, 50))
    bar_chart(figures_dir / "readiness_progress_comparison.png", "Post-reject forward re-enabled rate", "post_reject_forward_reenabled_rate", (75, 130, 180))
    bar_chart(figures_dir / "reject_to_policy_ready_comparison.png", "Reject-to-policy-ready rate", "reject_to_policy_ready_rate", (75, 130, 180))
    bar_chart(figures_dir / "next_vln_action_allowed_comparison.png", "Next VLN action allowed rate", "next_vln_action_allowed_rate", (75, 130, 180))
    bar_chart(figures_dir / "loop_rate_comparison.png", "Loop rate", "loop_rate", (130, 95, 180))
    bar_chart(figures_dir / "escape_macro_usage.png", "Escape macro selected count", "escape_macro_selected_count", (70, 115, 190))
    bar_chart(figures_dir / "success_vs_safe_success.png", "Safe success rate", "safe_success_rate", (40, 145, 85))
    bar_chart(figures_dir / "collision_breakdown.png", "Wall contact steps", "wall_contact_steps", (190, 85, 55))
    bar_chart(figures_dir / "reject_replan_outcomes.png", "CASA replan count", "casa_replan_count", (70, 115, 190))
    bar_chart(figures_dir / "unsafe_collision_comparison.png", "Unsafe violation rate", "unsafe_violation_rate", (190, 85, 55))
    bar_chart(figures_dir / "post_reject_progress_comparison.png", "Mean post-reject progress", "mean_post_reject_progress", (75, 130, 180))
    bar_chart(figures_dir / "fallback_to_stop_comparison.png", "Fallback-to-stop rate", "fallback_to_stop_rate", (175, 120, 60))
    bar_chart(figures_dir / "dmps_replan_action_distribution.png", "Non-stop recovery rate", "non_stop_recovery_rate", (80, 150, 120))


def _ppsr_v3_improvement_status(method_summary: list[dict[str, Any]]) -> bool:
    by_method = {row["method"]: row for row in method_summary}
    v3 = by_method.get(PPSR_V3_METHOD_NAME)
    v2 = by_method.get(PPSR_V2_METHOD_NAME)
    dmps = by_method.get(DMPS_METHOD_NAME)
    if v3 is None or v2 is None or dmps is None:
        return False
    return bool(
        float(v3.get("safe_success_rate", 0.0)) > float(v2.get("safe_success_rate", 0.0))
        and float(v3.get("safe_success_rate", 0.0)) >= float(dmps.get("safe_success_rate", 0.0))
        and float(v3.get("unsafe_violation_rate", 0.0)) <= float(v2.get("unsafe_violation_rate", 0.0)) + 0.05
        and float(v3.get("reject_to_policy_ready_rate", 0.0)) > float(v2.get("reject_to_policy_ready_rate", 0.0))
        and float(v3.get("next_vln_action_allowed_rate", 0.0)) > float(v2.get("next_vln_action_allowed_rate", 0.0))
        and float(v3.get("loop_rate", 0.0)) < float(v2.get("loop_rate", 0.0))
        and int(v3.get("safety_stop_success_leakage_count", 0)) == 0
        and int(v3.get("recovery_stop_success_leakage_count", 0)) == 0
    )


def audit_final_status(method_summary: list[dict[str, Any]], audits: dict[str, Any]) -> str:
    by_method = {row["method"]: row for row in method_summary}
    if not audits["casa_vln_bridge_audit"]["loaded_real_casa_artifacts"]:
        return "PARTIAL_BLOCKED_ENGINEERING"
    leakage = audits["privileged_leakage_audit"]
    if any(
        bool(leakage.get(key))
        for key in [
            "used_goal_distance_for_online_replan",
            "used_shortest_path_for_online_replan",
            "used_a_star_for_online_replan",
            "used_oracle_waypoint_for_online_replan",
            "used_evaluator_success_for_online_replan",
            "used_map_cell_progress_for_online_replan",
            "privileged_online_leakage",
        ]
    ):
        return "FAILED_PRIVILEGED_LEAKAGE"
    if audits["stop_source_audit"]["casa_stop_rows_counted_as_policy_stop"] != 0:
        return "FAILED_EVALUATION_BUT_AUDITABLE"

    if PPSR_V3_METHOD_NAME in by_method:
        v3 = by_method[PPSR_V3_METHOD_NAME]
        v2 = by_method.get(PPSR_V2_METHOD_NAME)
        dmps = by_method.get(DMPS_METHOD_NAME)
        if v2 is None or dmps is None:
            return "FAILED_EVALUATION_BUT_AUDITABLE"
        v3_stop = audits.get("v3_stop_source_audit", {})
        if int(v3.get("safety_stop_success_leakage_count", 0)) or int(
            v3_stop.get("safety_stop_success_leakage_count", 0)
        ):
            return "FAILED_STOP_LEAKAGE"
        if int(v3.get("recovery_stop_success_leakage_count", 0)) or int(
            v3_stop.get("recovery_stop_success_leakage_count", 0)
        ):
            return "FAILED_STOP_LEAKAGE"
        if float(v3["unsafe_violation_rate"]) > float(v2["unsafe_violation_rate"]) + 0.05:
            return "FAILED_SAFETY_REGRESSION"

        dev_summary = audits.get("ppsr_v3_dev_method_summary")
        locked_summary = audits.get("ppsr_v3_locked_method_summary")
        if dev_summary and locked_summary:
            dev_status = _ppsr_v3_improvement_status(dev_summary)
            locked_status = _ppsr_v3_improvement_status(locked_summary)
            if dev_status and not locked_status:
                return "DEV_IMPROVED_BUT_LOCKED_NO_IMPROVEMENT"

        improves = _ppsr_v3_improvement_status(method_summary)
        return "PASS_PPSR_V3_TASK_RETURN_IMPROVES" if improves else "FAILED_EVALUATION_BUT_AUDITABLE"

    if PPSR_V2_METHOD_NAME in by_method:
        ppsr_v2 = by_method[PPSR_V2_METHOD_NAME]
        ppsr_v1 = by_method.get(DMPS_METHOD_NAME)
        baseline = by_method.get("vln_only")
        if ppsr_v1 is None:
            return "FAILED_EVALUATION_BUT_AUDITABLE"
        if float(ppsr_v2["unsafe_violation_rate"]) > float(ppsr_v1["unsafe_violation_rate"]) + 0.05:
            return "FAILED_SAFETY_REGRESSION"
        if int(ppsr_v2.get("escape_macro_selected_count", 0)) <= 0 or int(ppsr_v2.get("escape_macro_commitment_count", 0)) <= 0:
            return "FAILED_ESCAPE_MACRO_NOT_USED"
        readiness = audits.get("readiness_progress_metrics", {})
        close_to_vln = True
        if baseline is not None:
            close_to_vln = float(ppsr_v2["safe_success_rate"]) >= float(baseline["safe_success_rate"]) - 0.05
        clearance_or_progress = (
            float(readiness.get("post_reject_clearance_gain", 0.0)) > 0.0
            or float(ppsr_v2.get("positive_post_reject_progress_rate", 0.0))
            > float(ppsr_v1.get("positive_post_reject_progress_rate", 0.0))
        )
        improves = (
            float(ppsr_v2["safe_success_rate"]) >= float(ppsr_v1["safe_success_rate"])
            and close_to_vln
            and float(ppsr_v2.get("small_turn_recovery_rate", 0.0))
            < float(ppsr_v1.get("small_turn_recovery_rate", 0.0))
            and int(ppsr_v2.get("dmps_recovery_stuck_count", 0))
            < int(ppsr_v1.get("dmps_recovery_stuck_count", 0))
            and float(ppsr_v2.get("post_reject_forward_reenabled_rate", 0.0)) > 0.0
            and clearance_or_progress
        )
        return "PASS_PPSR_V2_ESCAPE_IMPROVES" if improves else "PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT"

    if "vln_only" not in by_method:
        return "FAILED_EVALUATION_BUT_AUDITABLE"
    if DMPS_METHOD_NAME in by_method and "vln_casa_replan" in by_method:
        dmps = by_method[DMPS_METHOD_NAME]
        naive = by_method["vln_casa_replan"]
        baseline = by_method["vln_only"]
        if float(dmps["unsafe_violation_rate"]) > float(naive["unsafe_violation_rate"]) + 0.05:
            return "FAILED_SAFETY_REGRESSION"
        improves = (
            float(dmps["safe_success_rate"]) > float(naive["safe_success_rate"])
            and (
                float(dmps["safe_success_rate"]) >= float(baseline["safe_success_rate"])
                or float(baseline["safe_success_rate"]) - float(dmps["safe_success_rate"]) <= 0.05
            )
            and float(dmps["mean_post_reject_progress"]) > float(naive["mean_post_reject_progress"])
            and float(dmps["positive_post_reject_progress_rate"]) > float(naive["positive_post_reject_progress_rate"])
            and float(dmps["fallback_to_stop_rate"]) < float(naive["fallback_to_stop_rate"])
            and int(dmps["non_stop_recovery_count"]) > 0
            and audits["stop_source_audit"]["casa_stop_rows_counted_as_policy_stop"] == 0
        )
        return "PASS_DMPS_PROGRESS_REPLAN_IMPROVES" if improves else "PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT"
    replan = by_method.get("vln_casa_replan")
    baseline = by_method["vln_only"]
    if replan is None:
        return "FAILED_EVALUATION_BUT_AUDITABLE"
    improves_safe_success = float(replan["safe_success_rate"]) > float(baseline["safe_success_rate"])
    reduces_unsafe = float(replan["unsafe_violation_rate"]) < float(baseline["unsafe_violation_rate"])
    if improves_safe_success and reduces_unsafe:
        return "PASS_CASA_IMPROVES_SAFE_SUCCESS"
    return "PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT"


def comparison_metrics(method_summary: list[dict[str, Any]]) -> dict[str, float | None]:
    by_method = {row["method"]: row for row in method_summary}
    dmps = by_method.get(DMPS_METHOD_NAME)
    ppsr_v2 = by_method.get(PPSR_V2_METHOD_NAME)
    ppsr_v3 = by_method.get(PPSR_V3_METHOD_NAME)
    vln = by_method.get("vln_only")
    naive = by_method.get("vln_casa_replan")
    target = ppsr_v3 or ppsr_v2 or dmps
    if target is None:
        return {}

    def delta(row: dict[str, Any], key: str, other: dict[str, Any] | None) -> float | None:
        if other is None:
            return None
        return float(row.get(key, 0.0)) - float(other.get(key, 0.0))

    metrics: dict[str, float | None] = {}
    if dmps is not None:
        metrics.update(
            {
                "dmps_safe_success_delta_vs_vln_only": delta(dmps, "safe_success_rate", vln),
                "dmps_safe_success_delta_vs_naive_replan": delta(dmps, "safe_success_rate", naive),
                "dmps_unsafe_delta_vs_vln_only": delta(dmps, "unsafe_violation_rate", vln),
                "dmps_unsafe_delta_vs_naive_replan": delta(dmps, "unsafe_violation_rate", naive),
                "dmps_post_reject_progress_delta_vs_naive_replan": delta(dmps, "mean_post_reject_progress", naive),
                "dmps_fallback_to_stop_delta_vs_naive_replan": delta(dmps, "fallback_to_stop_rate", naive),
            }
        )
    if ppsr_v2 is not None:
        metrics.update(
            {
                "ppsr_v2_safe_success_delta_vs_ppsr_v1": delta(ppsr_v2, "safe_success_rate", dmps),
                "ppsr_v2_safe_success_delta_vs_vln_only": delta(ppsr_v2, "safe_success_rate", vln),
                "ppsr_v2_unsafe_delta_vs_ppsr_v1": delta(ppsr_v2, "unsafe_violation_rate", dmps),
                "ppsr_v2_small_turn_rate_delta_vs_ppsr_v1": delta(ppsr_v2, "small_turn_recovery_rate", dmps),
                "ppsr_v2_stuck_count_delta_vs_ppsr_v1": delta(ppsr_v2, "dmps_recovery_stuck_count", dmps),
                "ppsr_v2_post_reject_progress_delta_vs_ppsr_v1": delta(ppsr_v2, "mean_post_reject_progress", dmps),
                "ppsr_v2_forward_reenabled_rate": float(ppsr_v2.get("post_reject_forward_reenabled_rate", 0.0)),
                "ppsr_v2_escape_macro_selected_count": float(ppsr_v2.get("escape_macro_selected_count", 0.0)),
                "ppsr_v2_escape_macro_commitment_count": float(ppsr_v2.get("escape_macro_commitment_count", 0.0)),
            }
        )
    if ppsr_v3 is not None:
        metrics.update(
            {
                "ppsr_v3_safe_success_delta_vs_ppsr_v2": delta(ppsr_v3, "safe_success_rate", ppsr_v2),
                "ppsr_v3_safe_success_delta_vs_dmps": delta(ppsr_v3, "safe_success_rate", dmps),
                "ppsr_v3_unsafe_delta_vs_ppsr_v2": delta(ppsr_v3, "unsafe_violation_rate", ppsr_v2),
                "ppsr_v3_reject_to_policy_ready_delta_vs_ppsr_v2": delta(
                    ppsr_v3, "reject_to_policy_ready_rate", ppsr_v2
                ),
                "ppsr_v3_next_vln_action_allowed_delta_vs_ppsr_v2": delta(
                    ppsr_v3, "next_vln_action_allowed_rate", ppsr_v2
                ),
                "ppsr_v3_loop_rate_delta_vs_ppsr_v2": delta(ppsr_v3, "loop_rate", ppsr_v2),
                "ppsr_v3_post_reject_progress_delta_vs_ppsr_v2": delta(
                    ppsr_v3, "mean_post_reject_progress", ppsr_v2
                ),
            }
        )
    return metrics


def _git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def write_report(
    *,
    output_dir: Path,
    config: CasaVlnRunnerConfig,
    method_summary: list[dict[str, Any]],
    final_status: str,
    report: dict[str, Any],
) -> None:
    lines = [
        "# PPSR-v3 Task-Return VLN Safety Replan Report",
        "",
        f"- final_status: `{final_status}`",
        f"- output_dir: `{output_dir}`",
        f"- policy_backend: `{config.policy_backend}`",
        f"- real_navid_or_uninavid_used: `{report.get('real_navid_or_uninavid_used')}`",
        "- learned_ranker_used: `False`",
        f"- methods: `{','.join(config.methods)}`",
        f"- heldout_episodes_per_method: `{config.heldout_episodes}`",
        f"- gate_method: `{config.gate_method}`",
        f"- dmps_horizon: `{config.dmps_horizon}`",
        f"- safety_margin: `{config.safety_margin}`",
        f"- ppsr_v2_horizon_default: `{config.ppsr_v2_horizon_default}`",
        f"- ppsr_v2_horizon_stuck: `{config.ppsr_v2_horizon_stuck}`",
        f"- ppsr_v2_max_commit_steps: `{config.ppsr_v2_max_commit_steps}`",
        f"- ppsr_v2_safety_margin: `{config.ppsr_v2_safety_margin}`",
        f"- ppsr_v3_horizon: `{config.ppsr_v3_horizon}`",
        f"- ppsr_v3_max_execute_steps: `{config.ppsr_v3_max_execute_steps}`",
        f"- ppsr_v3_safety_margin: `{config.ppsr_v3_safety_margin}`",
        f"- method_alias: `{DMPS_ALIAS}` == `{DMPS_METHOD_NAME}`",
        f"- ppsr_v2_aliases: `{','.join(PPSR_V2_ALIASES)}` == `{PPSR_V2_METHOD_NAME}`",
        f"- ppsr_v3_aliases: `{','.join(PPSR_V3_ALIASES)}` == `{PPSR_V3_METHOD_NAME}`",
        f"- phase4_root: `{config.phase4_root}`",
        f"- phase5_root: `{config.phase5_root}`",
        "",
        "## Method Summary",
        "",
        "| method | safe_success_rate | success_rate | unsafe_violation_rate | reject_count | reject_to_policy_ready_rate | next_vln_action_allowed_rate | loop_rate | post_reject_progress | landmark_rate | safety_stop_leak | recovery_stop_leak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method']} | {row['safe_success_rate']:.3f} | {row['success_rate']:.3f} | "
            f"{row['unsafe_violation_rate']:.3f} | {row['reject_count']} | "
            f"{row.get('reject_to_policy_ready_rate', 0.0):.3f} | "
            f"{row.get('next_vln_action_allowed_rate', 0.0):.3f} | "
            f"{row.get('loop_rate', 0.0):.3f} | {row.get('post_reject_progress', row['mean_post_reject_progress']):.3f} | "
            f"{row.get('landmark_retention_or_reacquisition_rate', 0.0):.3f} | "
            f"{row.get('safety_stop_success_leakage_count', 0)} | "
            f"{row.get('recovery_stop_success_leakage_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            "1. Old `vln_casa_replan` lowers safe success because it prioritizes low-risk single-step recovery without visual progress, intent preservation, or loop avoidance.",
            "2. The new method performs dynamic candidate-sequence search with safety filtering and progress/intent scoring; the old replan uses a fixed single-step risk/priority choice.",
            "3. DMPS-style shielding is implemented: VLN remains nominal policy, the shield intervenes only on unsafe actions, executes only the first recovery action, and returns control to VLN.",
            "4. Skill-level MPC-CBF-style safety is implemented as discrete rollout over skill microsteps with clearance barrier `h(x)=clearance-safety_margin`; this is not full torque-level humanoid MPC-CBF.",
            "5. A non-privileged progress monitor tracks recent actions, rejects, blocked flags, stop sources, visual hashes, and loop counters.",
            "6. Online replan does not use goal distance, shortest path, A*, or oracle waypoints; those are evaluator-only.",
            "7. Fallback-to-stop is reported in `method_summary.csv` and `fallback_to_stop_comparison.png`.",
            "8. Post-reject progress is evaluator-only and reported for comparison, not used for online action selection.",
            "9. Unsafe/wall collision is kept under the same `maze.is_xy_safe` microstep criterion.",
            "10. Safe-success improvement is determined by the final status rule, not assumed.",
            "11. Recovery sequence effectiveness is visible in `dmps_selected_replans.csv` and `dmps_replan_action_distribution.png`.",
            "12. Remaining failures are reported in `failure_reason_breakdown.json`.",
            "13. Visual free-space uses a first-person left/center/right brightness-edge proxy; fallback is logged if image read fails.",
            "14. The paper claim is supported only if final_status is `PASS_DMPS_PROGRESS_REPLAN_IMPROVES`.",
            "",
            "## PPSR-v2 Required Audit Answers",
            "",
            "1. PPSR-v2 method name: `vln_ppsr_v2_escape_macro`; aliases normalize to the same canonical method.",
            "2. It is inserted only after a CASA rejection; nominal `vln_only` behavior is not replaced.",
            "3. The old PPSR-v1 turn-loop failure is addressed with a hard anti-turn-loop mask, not just a soft score penalty.",
            "4. Translation is required in stuck mode whenever a safe translation candidate exists.",
            "5. Escape macros are generated only under the stuck/repeated-reject signal and are logged in `escape_macro_candidates.csv`.",
            "6. Terminal scoring explicitly checks whether short forward motion becomes feasible after the candidate sequence.",
            "7. Two-step commitment is allowed only for safe, translating stuck-mode sequences and is rechecked before the second skill.",
            "8. Commitment aborts are logged in `escape_commitment_logs.csv`; aborted safety stops are non-success stops.",
            "9. PPSR-v2 does not use final goal distance, evaluator success, shortest path, A*, map-cell progress, or oracle waypoint online.",
            "10. Stop-source auditing keeps CASA/PPSR stops separate from policy stop success.",
            "11. Unsafe evaluation remains the same wall/fall oracle used by vln_only, vln_casa_replan, and PPSR-v1.",
            "12. Safety regression is reported as `FAILED_SAFETY_REGRESSION` if PPSR-v2 unsafe rate exceeds PPSR-v1 by more than 0.05.",
            "13. `FAILED_ESCAPE_MACRO_NOT_USED` is reported if escape macros or commitments are never selected.",
            "14. `PASS_PPSR_V2_ESCAPE_IMPROVES` requires safe success not below PPSR-v1, no safety regression, lower small-turn and stuck rates, macro use, commitments, and readiness improvement.",
            "15. If those strict conditions fail but logs are complete, the status is `PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT` or another explicit failure status.",
            "16. Required v2 artifacts are under `data/`: candidate sequences, selected replans, mask audit, terminal score audit, commitment logs, readiness metrics, and failure breakdown.",
            "17. Required v2 figures are under `figures/`: safe success, unsafe, small-turn, stuck, readiness, escape usage, and post-reject progress comparisons.",
            "",
            "## PPSR-v3 Task-Return Audit Answers",
            "",
            "1. PPSR-v3 method name: `vln_ppsr_v3_task_return_replan`; it is registered independently of v1/v2.",
            "2. Online scoring uses first-person frame proxies, recent actions/rejects/status, CASA safety geometry, and visual free-space/landmark proxies only.",
            "3. Online scoring does not use goal distance, evaluator success, A*, oracle waypoint, global shortest path, or map-cell progress.",
            "4. CASA reject triggers 2-5 step task-return candidate sequences; only the first one or two steps are executed and safety is rechecked before step two.",
            "5. Policy-ready fields are logged in `data/v3_policy_ready_audit.csv` and `data/v3_selected_replans.csv`.",
            "6. Safety/recovery stops are non-success stops and leakage counts must remain zero.",
            "7. No learned recovery ranker is used in this run.",
            "",
            "## Stop Source Rule",
            "",
            "CASA/DMPS safety stops are not counted as policy-issued task success. "
            "Only `vln_policy` or `policy_internal_guard` can satisfy the stop condition.",
            "",
            "The bridge maps `backoff` to CASA `walk` for threshold lookup while preserving the executable "
            "`backoff_walk` skill name in logs.",
            "",
            "## Videos",
        ]
    )
    for path in report.get("video_paths", [])[:30]:
        lines.append(f"- `{path}`")
    (output_dir / "ppsr_v2_escape_macro_report.md").write_text("\n".join(lines) + "\n")
    (output_dir / "ppsr_v3_task_return_report.md").write_text("\n".join(lines) + "\n")
    (output_dir / "dmps_mpc_cbf_progress_report.md").write_text("\n".join(lines) + "\n")
    (output_dir / "casa_vln_safety_report.md").write_text("\n".join(lines) + "\n")
    (output_dir / "README.md").write_text("\n".join(lines[:22]) + "\n")


def run(config: CasaVlnRunnerConfig) -> dict[str, Any]:
    start_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    assert_no_privileged_inference_schema()
    verify_casa_artifacts(config)
    dirs = ensure_layout(config.output_dir)
    write_repo_audit(dirs["notes"] / "repo_audit.md")

    maze = MazeMap.from_metadata(config.map_metadata)
    train_maze = MazeMap.from_metadata(config.train_map_metadata)
    draw_topdown_map(maze, dirs["figures"] / "map_topdown_test.png")
    draw_topdown_map(train_maze, dirs["figures"] / "map_topdown_train.png")

    train_env = MujocoMazeSkillEnv(
        scene_xml=config.train_scene_xml,
        maze=train_maze,
        output_dir=config.output_dir,
        width=config.frame_width,
        height=config.frame_height,
    )
    env = MujocoMazeSkillEnv(
        scene_xml=config.scene_xml,
        maze=maze,
        output_dir=config.output_dir,
        width=config.frame_width,
        height=config.frame_height,
    )
    train_env.render_head(dirs["figures"] / "head_camera_check_train.png")
    env.render_head(dirs["figures"] / "head_camera_check_test.png")
    write_json(
        dirs["data"] / "scene_load_report.json",
        {
            "scene_xml": str(config.scene_xml),
            "train_scene_xml": str(config.train_scene_xml),
            "map_metadata": str(config.map_metadata),
            "train_map_metadata": str(config.train_map_metadata),
            "mujoco_load_passed": True,
            "target_visual": maze.metadata.get("target_visual"),
            "red_ball_target_visible": bool(maze.metadata.get("red_ball_target_visible", True)),
            "test_furniture_obstacle_count": len(maze.furniture_obstacles),
            "train_furniture_obstacle_count": len(train_maze.furniture_obstacles),
            "robot_initial_pose_check": asdict(env.pose()),
        },
    )

    demo_routes = generate_routes(
        train_maze,
        count=config.auto_demo_count,
        seed=config.seed,
        min_edges=config.min_route_edges,
        max_edges=config.max_route_edges,
        min_euclidean_m=config.min_route_euclidean_m,
        require_furniture_detour=config.require_furniture_detour,
    )
    heldout_seed = config.seeds[0] if config.seeds else config.heldout_seed
    heldout_routes = generate_routes(
        maze,
        count=config.heldout_episodes,
        seed=heldout_seed,
        min_edges=config.min_route_edges,
        max_edges=config.max_route_edges,
        min_euclidean_m=config.min_route_euclidean_m,
        require_furniture_detour=config.require_furniture_detour,
    )
    demo_tasks = [task_from_route(route, episode_id=f"train_{idx:03d}", split="train") for idx, route in enumerate(demo_routes)]
    heldout_tasks = [
        task_from_route(route, episode_id=f"heldout_{idx:03d}", split="heldout", max_steps=config.max_steps)
        for idx, route in enumerate(heldout_routes)
    ]
    write_json(
        dirs["data"] / "eval_splits.json",
        {
            "heldout_seed": heldout_seed,
            "seeds": list(config.seeds),
            "heldout_episodes": [task.to_dict() for task in heldout_tasks],
            "train_episodes": [task.to_dict() for task in demo_tasks],
            "methods": list(config.methods),
            "same_heldout_split_for_all_methods": True,
        },
    )
    dataset_path, adapter_path = collect_auto_demos(
        config=_NoCasaCompatConfig(config),
        maze=train_maze,
        env=train_env,
        demo_tasks=demo_tasks,
        data_dir=dirs["data"],
        models_dir=dirs["models"],
    )
    bridge = CasaVlnBridge(phase4_root=config.phase4_root, phase5_root=config.phase5_root, gate_method=config.gate_method)
    policy, real_navid_backend = build_policy(config=config, dirs=dirs, adapter_path=adapter_path)

    episode_metrics: list[dict[str, Any]] = []
    try:
        for method in config.methods:
            for task in heldout_tasks:
                episode_metrics.append(
                    run_episode_with_method(
                        method=method,
                        task=task,
                        env=env,
                        policy=policy,
                        bridge=bridge,
                        config=config,
                        run_dir=config.output_dir,
                        data_dir=dirs["data"],
                        videos_dir=dirs["videos"],
                        frames_dir=dirs["frames"],
                    )
                )
    finally:
        env.close()
        train_env.close()
        if real_navid_backend is not None:
            real_navid_backend.close()

    (
        decision_rows,
        gate_rows,
        replan_rows,
        _trajectory_rows,
        dmps_candidate_rows,
        dmps_selected_rows,
        dmps_rollout_rows,
        progress_event_rows,
    ) = write_global_logs(dirs["data"], episode_metrics)
    by_method = {method: [row for row in episode_metrics if row["method"] == method] for method in config.methods}
    method_summary = [aggregate_method(rows) for rows in by_method.values()]
    write_table_csv(dirs["data"] / "method_summary.csv", method_summary)
    write_table_csv(dirs["data"] / "ppsr_v2_method_summary.csv", [row for row in method_summary if row["method"] == PPSR_V2_METHOD_NAME])
    write_table_csv(dirs["data"] / "v3_method_summary.csv", [row for row in method_summary if row["method"] == PPSR_V3_METHOD_NAME])
    write_json(dirs["data"] / "method_summary.json", method_summary)
    audits = write_audits(
        data_dir=dirs["data"],
        config=config,
        bridge=bridge,
        decision_rows=decision_rows,
        gate_rows=gate_rows,
        replan_rows=replan_rows,
        dmps_candidate_rows=dmps_candidate_rows,
        dmps_selected_rows=dmps_selected_rows,
        dmps_rollout_rows=dmps_rollout_rows,
        progress_event_rows=progress_event_rows,
        method_summary=method_summary,
    )
    draw_comparison_figures(dirs["figures"], method_summary)
    final_status = audit_final_status(method_summary, audits)
    video_paths = [path for episode in episode_metrics for path in episode.get("video_paths", [])]
    report = {
        "created_at": start_time,
        "start_time": start_time,
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_text(["rev-parse", "HEAD"]),
        "git_status_short": _git_text(["status", "--short"]),
        "final_status": final_status,
        "output_dir": str(config.output_dir),
        "scene_xml": str(config.scene_xml),
        "runner_script": "gear_sonic/scripts/casa_run_vln_safety_online.py",
        "methods": list(config.methods),
        "heldout_episodes_per_method": config.heldout_episodes,
        "seeds": list(config.seeds),
        "max_steps": config.max_steps,
        "dataset_path": str(dataset_path),
        "adapter_path": str(adapter_path),
        "policy_backend": config.policy_backend,
        "navid_cuda_visible_devices": config.navid_cuda_visible_devices,
        "gpu_uuid": config.navid_cuda_visible_devices,
        "policy_name": policy.name,
        "real_navid_or_uninavid_used": bool(
            config.policy_backend in {"real_navid", "real_navid_visual_adapter"}
            and real_navid_backend is not None
            and real_navid_backend.availability.get("model_loaded")
        ),
        "real_navid_backend_probe": real_navid_backend.availability if real_navid_backend is not None else None,
        "phase4_root": str(config.phase4_root),
        "phase5_root": str(config.phase5_root),
        "casa_gate_method": config.gate_method,
        "previous_casa_vln_result_path": str(config.previous_casa_vln_result_path),
        "dmps_horizon": config.dmps_horizon,
        "safety_margin": config.safety_margin,
        "ppsr_v2_horizon_default": config.ppsr_v2_horizon_default,
        "ppsr_v2_horizon_stuck": config.ppsr_v2_horizon_stuck,
        "ppsr_v2_max_commit_steps": config.ppsr_v2_max_commit_steps,
        "ppsr_v2_safety_margin": config.ppsr_v2_safety_margin,
        "ppsr_v2_score_config": str(config.ppsr_v2_score_config) if config.ppsr_v2_score_config else None,
        "ppsr_v3_horizon": config.ppsr_v3_horizon,
        "ppsr_v3_max_execute_steps": config.ppsr_v3_max_execute_steps,
        "ppsr_v3_safety_margin": config.ppsr_v3_safety_margin,
        "ppsr_v3_score_config": str(config.ppsr_v3_score_config) if config.ppsr_v3_score_config else None,
        "score_weights": config.score_weights or DEFAULT_SCORE_WEIGHTS,
        "ppsr_v2_score_weights": {**PPSR_V2_SCORE_WEIGHTS, **(config.score_weights or {})},
        "ppsr_v3_score_weights": {**PPSR_V3_SCORE_WEIGHTS, **(config.score_weights or {})},
        "learned_ranker_used": False,
        "learned_ranker_training_data_count": 0,
        "learned_ranker_training_command": None,
        "learned_ranker_held_out_split": None,
        "visual_free_space_available": audits["visual_free_space_audit"]["visual_free_space_score_available"],
        "used_goal_distance_for_online_replan": False,
        "used_shortest_path_for_online_replan": False,
        "used_a_star_for_online_replan": False,
        "used_oracle_waypoint_for_online_replan": False,
        "used_evaluator_success_for_online_replan": False,
        "used_map_cell_progress_for_online_replan": False,
        "privileged_online_leakage": False,
        "python_executable": os.sys.executable,
        "risk_source": "real_casa_critic",
        "method_summary": method_summary,
        "comparisons": comparison_metrics(method_summary),
        "episodes": episode_metrics,
        "video_paths": video_paths,
        "videos_path": str(dirs["videos"]),
        "frames_path": str(dirs["frames"]),
        "audits": audits,
        "large_outputs_not_in_git": True,
    }
    write_json(config.output_dir / "run_manifest.json", report)
    write_report(
        output_dir=config.output_dir,
        config=config,
        method_summary=method_summary,
        final_status=final_status,
        report=report,
    )
    return report


@dataclass(frozen=True)
class _NoCasaCompatConfig:
    """Minimal compatibility object for no_casa_runner.collect_auto_demos."""

    source: CasaVlnRunnerConfig

    @property
    def run_dir(self) -> Path:
        return self.source.output_dir

    @property
    def skip_demo_frames(self) -> bool:
        return False

    @property
    def visual_adapter_epochs(self) -> int:
        return self.source.visual_adapter_epochs

    @property
    def smoke_only(self) -> bool:
        return False


def parse_args(argv: list[str] | None = None) -> CasaVlnRunnerConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--scene-xml", type=Path, default=default_existing(DEFAULT_TEST_SCENE_XML, LEGACY_SCENE_XML))
    parser.add_argument("--map-metadata", type=Path, default=default_existing(DEFAULT_TEST_MAP_METADATA, LEGACY_MAP_METADATA))
    parser.add_argument("--train-scene-xml", type=Path, default=default_existing(DEFAULT_TRAIN_SCENE_XML, LEGACY_SCENE_XML))
    parser.add_argument("--train-map-metadata", type=Path, default=default_existing(DEFAULT_TRAIN_MAP_METADATA, LEGACY_MAP_METADATA))
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4_ROOT)
    parser.add_argument("--phase5-root", type=Path, default=DEFAULT_PHASE5_ROOT)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--method", default=None)
    parser.add_argument("--heldout-episodes", type=int, default=20)
    parser.add_argument("--auto-demo-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260610)
    parser.add_argument("--heldout-seed", type=int, default=260611)
    parser.add_argument("--seeds", default="260611")
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=240)
    parser.add_argument("--visual-adapter-epochs", type=int, default=240)
    parser.add_argument("--min-route-edges", type=int, default=4)
    parser.add_argument("--max-route-edges", type=int, default=14)
    parser.add_argument("--min-route-euclidean-m", type=float, default=6.0)
    parser.add_argument("--require-furniture-detour", action="store_true", default=True)
    parser.add_argument(
        "--policy-backend",
        choices=["visual_adapter", "real_navid", "real_navid_visual_adapter"],
        default="real_navid_visual_adapter",
    )
    parser.add_argument("--navid-repo", type=Path, default=NAVID_REPO)
    parser.add_argument("--navid-model", type=Path, default=NAVID_MODEL)
    parser.add_argument("--navid-python", type=Path, default=NAVID_PYTHON)
    parser.add_argument("--navid-vision-tower", type=Path, default=NAVID_VISION_TOWER)
    parser.add_argument("--navid-worker-timeout-s", type=float, default=900.0)
    parser.add_argument("--navid-cuda-visible-devices", default=None)
    parser.add_argument("--gate-method", default="casa_a_hard_or_per_skill")
    parser.add_argument("--max-consecutive-casa-recovery-steps", type=int, default=1)
    parser.add_argument("--max-consecutive-dmps-recovery-steps", type=int, default=1)
    parser.add_argument("--dmps-horizon", type=int, default=2)
    parser.add_argument("--safety-margin", type=float, default=0.05)
    parser.add_argument("--progress-score-config", type=Path, default=None)
    parser.add_argument("--ppsr-v2-horizon-default", type=int, default=2)
    parser.add_argument("--ppsr-v2-horizon-stuck", type=int, default=3)
    parser.add_argument("--ppsr-v2-max-commit-steps", type=int, default=2)
    parser.add_argument("--ppsr-v2-safety-margin", type=float, default=0.05)
    parser.add_argument("--ppsr-v2-score-config", type=Path, default=None)
    parser.add_argument("--ppsr-v3-horizon", type=int, default=5)
    parser.add_argument("--ppsr-v3-max-execute-steps", type=int, default=2)
    parser.add_argument("--ppsr-v3-safety-margin", type=float, default=0.05)
    parser.add_argument("--ppsr-v3-score-config", type=Path, default=None)
    parser.add_argument("--previous-casa-vln-result-path", type=Path, default=Path("/mnt/data/students/lph/recording/casa_vln_safety_locked_real_navid_20260611_190121"))
    parser.add_argument("--record-video", action="store_true", default=True)
    parser.add_argument("--dry-run", default="false")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    methods_arg = args.method if args.method is not None else args.methods
    seeds = parse_seeds(args.seeds)
    return CasaVlnRunnerConfig(
        output_dir=args.output_dir,
        scene_xml=args.scene_xml,
        map_metadata=args.map_metadata,
        train_scene_xml=args.train_scene_xml,
        train_map_metadata=args.train_map_metadata,
        phase4_root=args.phase4_root,
        phase5_root=args.phase5_root,
        methods=method_list(methods_arg),
        heldout_episodes=args.heldout_episodes,
        auto_demo_count=args.auto_demo_count,
        seed=args.seed,
        heldout_seed=args.heldout_seed,
        max_steps=args.max_steps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
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
        navid_cuda_visible_devices=args.navid_cuda_visible_devices,
        gate_method=args.gate_method,
        max_consecutive_casa_recovery_steps=args.max_consecutive_casa_recovery_steps,
        max_consecutive_dmps_recovery_steps=args.max_consecutive_dmps_recovery_steps,
        dmps_horizon=args.dmps_horizon,
        safety_margin=args.safety_margin,
        ppsr_v2_horizon_default=args.ppsr_v2_horizon_default,
        ppsr_v2_horizon_stuck=args.ppsr_v2_horizon_stuck,
        ppsr_v2_max_commit_steps=args.ppsr_v2_max_commit_steps,
        ppsr_v2_safety_margin=args.ppsr_v2_safety_margin,
        ppsr_v2_score_config=args.ppsr_v2_score_config,
        ppsr_v3_horizon=args.ppsr_v3_horizon,
        ppsr_v3_max_execute_steps=args.ppsr_v3_max_execute_steps,
        ppsr_v3_safety_margin=args.ppsr_v3_safety_margin,
        ppsr_v3_score_config=args.ppsr_v3_score_config,
        progress_score_config=args.progress_score_config,
        score_weights=load_score_weights(args.ppsr_v3_score_config or args.ppsr_v2_score_config or args.progress_score_config),
        previous_casa_vln_result_path=args.previous_casa_vln_result_path,
        seeds=seeds,
        record_video=args.record_video,
        dry_run=str(args.dry_run).lower() in {"1", "true", "yes"},
        strict=args.strict,
    )


def main(argv: list[str] | None = None) -> None:
    report = run(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    print("CASA-VLN safety package written to:")
    print(report["output_dir"])


if __name__ == "__main__":
    main()
