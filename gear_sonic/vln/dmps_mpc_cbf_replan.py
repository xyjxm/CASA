"""DMPS-style progress-preserving skill-level MPC-CBF replan for VLN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .actions import VLNAction
from .casa_bridge import CasaVlnBridge, clearance_to_blocking_geometry
from .maze import MazeMap
from .no_casa_policy import detect_gallery_wall_painting
from .oracle import RobotPose2D
from .progress_monitor import ProgressMonitor
from .skill_mapping import PassiveSkill, SonicSkill, TurnSkill, WalkSkill, wrap_degrees


DMPS_METHOD_NAME = "vln_dmps_mpc_cbf_progress"
DMPS_ALIAS = "vln_ppsr"
DMPS_LAST_RESORT_STOP_SOURCE = "dmps_mpc_cbf_last_resort_stop"


DEFAULT_SCORE_WEIGHTS = {
    "safety_margin": 0.50,
    "intent": 0.40,
    "visual": 0.25,
    "target": 0.20,
    "loop": 0.35,
    "repeat": 0.40,
    "switch": 0.10,
    "stop": 0.75,
    "length": 0.08,
}


@dataclass(frozen=True)
class CandidateSequence:
    sequence_id: str
    actions: tuple[str, ...]
    skills: tuple[SonicSkill, ...]
    stop_as_last_resort: bool = False


@dataclass(frozen=True)
class RolloutResult:
    sequence_candidate_id: str
    candidate_sequence: tuple[str, ...]
    predicted_microsteps: int
    min_clearance: float
    min_barrier_h: float
    cbf_violation: bool
    would_block: bool
    predicted_wall_contact_steps: int
    safety_feasible: bool
    safety_rejection_reason: str
    min_discrete_cbf_value: float
    discrete_cbf_violation_count: int
    final_pose: RobotPose2D
    sampled_points: list[dict[str, float]]


@dataclass(frozen=True)
class CandidateScore:
    sequence_candidate_id: str
    candidate_sequence: tuple[str, ...]
    sequence_length: int
    min_clearance: float
    min_barrier_h: float
    cbf_violation: bool
    would_block: bool
    predicted_wall_contact_steps: int
    safety_feasible: bool
    safety_rejection_reason: str
    intent_consistency_score: float
    visual_free_space_score: float
    target_or_stop_cue_score: float
    recovery_loop_penalty: float
    repeated_reject_penalty: float
    unnecessary_stop_penalty: float
    action_switching_penalty: float
    candidate_sequence_length_penalty: float
    normalized_clearance: float
    total_score: float
    selected: bool = False
    selection_reason: str = ""


@dataclass(frozen=True)
class DmpsSelection:
    selected_sequence: CandidateSequence
    selected_score: CandidateScore
    candidate_scores: list[CandidateScore]
    rollout_rows: list[dict[str, Any]]
    visual_free_space: dict[str, Any]
    progress_monitor_snapshot: dict[str, Any]
    all_non_stop_candidates_infeasible: bool
    control_returned_to_vln_next_step: bool = True


def normalize_dmps_method(method: str) -> str:
    return DMPS_METHOD_NAME if method == DMPS_ALIAS else method


def is_dmps_method(method: str) -> bool:
    return normalize_dmps_method(method) == DMPS_METHOD_NAME


def build_candidate_sequences(
    *,
    pose: RobotPose2D,
    nominal_action: VLNAction | str,
    horizon: int = 2,
) -> list[CandidateSequence]:
    del nominal_action
    heading = float(pose.yaw_deg)
    candidates = [
        ("short_forward", ("short_forward",)),
        ("backoff", ("backoff",)),
        ("turn_left", ("turn_left",)),
        ("turn_right", ("turn_right",)),
        ("backoff_turn_left", ("backoff", "turn_left")),
        ("backoff_turn_right", ("backoff", "turn_right")),
        ("turn_left_short_forward", ("turn_left", "short_forward")),
        ("turn_right_short_forward", ("turn_right", "short_forward")),
        ("backoff_short_forward", ("backoff", "short_forward")),
        ("small_turn_left", ("small_turn_left",)),
        ("small_turn_right", ("small_turn_right",)),
        ("stop_as_last_resort", ("stop_as_last_resort",)),
    ]
    output: list[CandidateSequence] = []
    for sequence_id, actions in candidates:
        clipped = tuple(actions[: max(1, horizon)])
        output.append(
            CandidateSequence(
                sequence_id=sequence_id,
                actions=clipped,
                skills=tuple(_skills_for_sequence(clipped, heading)),
                stop_as_last_resort=sequence_id == "stop_as_last_resort",
            )
        )
    return output


def _skills_for_sequence(actions: tuple[str, ...], heading: float) -> list[SonicSkill]:
    skills: list[SonicSkill] = []
    current_heading = float(heading)
    for action in actions:
        skill = skill_for_recovery_action(action, current_heading)
        skills.append(skill)
        if isinstance(skill, TurnSkill):
            current_heading = skill.face_yaw_deg
    return skills


def skill_for_recovery_action(action: str, heading: float) -> SonicSkill:
    if action in {"short_forward", "short_forward_segment"}:
        vx, vy = _unit_from_yaw(heading)
        return WalkSkill(vx=vx, vy=vy, facing_yaw_deg=heading, duration=0.25, step_target_m=0.16)
    if action == "backoff":
        vx, vy = _unit_from_yaw(heading + 180.0)
        return WalkSkill(
            vx=vx,
            vy=vy,
            facing_yaw_deg=heading,
            duration=0.25,
            step_target_m=0.18,
            name="backoff_walk",
        )
    if action == "turn_left":
        return TurnSkill(delta_yaw_deg=30.0, face_yaw_deg=wrap_degrees(heading + 30.0), duration=0.30)
    if action == "turn_right":
        return TurnSkill(delta_yaw_deg=-30.0, face_yaw_deg=wrap_degrees(heading - 30.0), duration=0.30)
    if action == "small_turn_left":
        return TurnSkill(delta_yaw_deg=15.0, face_yaw_deg=wrap_degrees(heading + 15.0), duration=0.20)
    if action == "small_turn_right":
        return TurnSkill(delta_yaw_deg=-15.0, face_yaw_deg=wrap_degrees(heading - 15.0), duration=0.20)
    if action == "stop_as_last_resort":
        return PassiveSkill(duration=0.20, mode="stop")
    raise ValueError(f"unknown recovery action: {action}")


def select_dmps_replan(
    *,
    bridge: CasaVlnBridge,
    pose: RobotPose2D,
    maze: MazeMap,
    robot_radius: float,
    nominal_action: VLNAction,
    image_path: str | Path | None,
    progress_monitor: ProgressMonitor,
    step_idx: int,
    horizon: int,
    safety_margin: float,
    score_weights: dict[str, float] | None = None,
) -> DmpsSelection:
    del bridge  # Real CASA gate is already used to trigger DMPS; rollout uses the simulator safety geometry.
    weights = {**DEFAULT_SCORE_WEIGHTS, **(score_weights or {})}
    visual = compute_visual_free_space(image_path)
    target_cue = target_or_stop_cue_score(image_path)
    snapshot = progress_monitor.snapshot(step_idx=step_idx)
    candidates = build_candidate_sequences(pose=pose, nominal_action=nominal_action, horizon=horizon)
    scored: list[CandidateScore] = []
    rollout_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rollout = rollout_candidate_sequence(
            candidate=candidate,
            start_pose=pose,
            maze=maze,
            robot_radius=robot_radius,
            safety_margin=safety_margin,
        )
        rollout_rows.append(asdict(rollout))
        penalties = progress_monitor.penalties_for_candidate(
            nominal_action=nominal_action.value,
            candidate_sequence=list(candidate.actions),
        )
        score = score_candidate(
            candidate=candidate,
            rollout=rollout,
            nominal_action=nominal_action,
            visual_free_space=visual,
            target_cue_score=target_cue,
            penalties=penalties,
            weights=weights,
            safety_margin=safety_margin,
        )
        scored.append(score)

    non_stop_safe = [item for item in scored if item.safety_feasible and item.candidate_sequence[0] != "stop_as_last_resort"]
    stop_scores = [item for item in scored if item.candidate_sequence[0] == "stop_as_last_resort"]
    all_non_stop_candidates_infeasible = len(non_stop_safe) == 0
    if non_stop_safe:
        selected_score = sorted(non_stop_safe, key=lambda item: item.total_score, reverse=True)[0]
        reason = "best_safe_progress_preserving_non_stop"
    elif stop_scores:
        selected_score = stop_scores[0]
        reason = "all_non_stop_candidates_infeasible"
    else:
        raise RuntimeError("DMPS candidate generation produced no selectable candidate")

    selected_scores = [
        CandidateScore(
            **{
                **asdict(score),
                "selected": score.sequence_candidate_id == selected_score.sequence_candidate_id,
                "selection_reason": reason if score.sequence_candidate_id == selected_score.sequence_candidate_id else "",
            }
        )
        for score in scored
    ]
    selected_sequence = next(
        candidate for candidate in candidates if candidate.sequence_id == selected_score.sequence_candidate_id
    )
    selected_score_with_flag = next(score for score in selected_scores if score.selected)
    return DmpsSelection(
        selected_sequence=selected_sequence,
        selected_score=selected_score_with_flag,
        candidate_scores=selected_scores,
        rollout_rows=rollout_rows,
        visual_free_space=visual,
        progress_monitor_snapshot=snapshot,
        all_non_stop_candidates_infeasible=all_non_stop_candidates_infeasible,
    )


def rollout_candidate_sequence(
    *,
    candidate: CandidateSequence,
    start_pose: RobotPose2D,
    maze: MazeMap,
    robot_radius: float,
    safety_margin: float,
    microsteps_per_skill: int = 8,
    cbf_alpha: float = 0.50,
) -> RolloutResult:
    pose = start_pose
    clearances: list[float] = []
    h_values: list[float] = []
    sampled: list[dict[str, float]] = []
    would_block = False
    predicted_wall_contact_steps = 0
    discrete_values: list[float] = []

    for skill_idx, skill in enumerate(candidate.skills):
        points = _skill_rollout_points(skill, pose=pose, microsteps=microsteps_per_skill)
        last_h = None
        for micro_idx, next_pose in enumerate(points):
            clearance, _ = clearance_to_blocking_geometry(next_pose.x, next_pose.y, maze=maze, robot_radius=robot_radius)
            safe_xy = maze.is_xy_safe(next_pose.x, next_pose.y, robot_radius=robot_radius)
            h_value = float(clearance - safety_margin)
            clearances.append(float(clearance))
            h_values.append(h_value)
            sampled.append(
                {
                    "skill_idx": float(skill_idx),
                    "micro_idx": float(micro_idx),
                    "x": float(next_pose.x),
                    "y": float(next_pose.y),
                    "yaw_deg": float(next_pose.yaw_deg),
                    "clearance": float(clearance),
                    "barrier_h": h_value,
                    "safe_xy": float(safe_xy),
                }
            )
            if not safe_xy or clearance < safety_margin:
                would_block = True
                predicted_wall_contact_steps += 1
            if last_h is not None:
                discrete_values.append(h_value - last_h + cbf_alpha * last_h)
            last_h = h_value
            pose = next_pose

    if not clearances:
        clearance, _ = clearance_to_blocking_geometry(pose.x, pose.y, maze=maze, robot_radius=robot_radius)
        clearances = [float(clearance)]
        h_values = [float(clearance - safety_margin)]
    min_clearance = min(clearances)
    min_barrier_h = min(h_values)
    min_discrete = min(discrete_values) if discrete_values else min_barrier_h
    discrete_violation_count = sum(1 for value in discrete_values if value < -1e-6)
    cbf_violation = min_barrier_h < -1e-6 or discrete_violation_count > 0
    safety_feasible = not candidate.stop_as_last_resort and not would_block and min_clearance >= safety_margin and not cbf_violation
    if candidate.stop_as_last_resort:
        safety_feasible = True
        reason = "last_resort_stop"
    elif would_block:
        reason = "would_block"
    elif min_clearance < safety_margin:
        reason = "min_clearance_below_safety_margin"
    elif cbf_violation:
        reason = "cbf_violation"
    else:
        reason = "safe"
    return RolloutResult(
        sequence_candidate_id=candidate.sequence_id,
        candidate_sequence=candidate.actions,
        predicted_microsteps=len(sampled),
        min_clearance=float(min_clearance),
        min_barrier_h=float(min_barrier_h),
        cbf_violation=bool(cbf_violation),
        would_block=bool(would_block),
        predicted_wall_contact_steps=int(predicted_wall_contact_steps),
        safety_feasible=bool(safety_feasible),
        safety_rejection_reason=reason,
        min_discrete_cbf_value=float(min_discrete),
        discrete_cbf_violation_count=int(discrete_violation_count),
        final_pose=pose,
        sampled_points=sampled,
    )


def score_candidate(
    *,
    candidate: CandidateSequence,
    rollout: RolloutResult,
    nominal_action: VLNAction,
    visual_free_space: dict[str, Any],
    target_cue_score: float,
    penalties: dict[str, float],
    weights: dict[str, float],
    safety_margin: float,
) -> CandidateScore:
    first = candidate.actions[0]
    normalized_clearance = max(0.0, min(1.0, (rollout.min_clearance - safety_margin) / max(0.50, safety_margin)))
    intent = intent_consistency_score(nominal_action=nominal_action, candidate_sequence=list(candidate.actions), penalties=penalties)
    visual_score = visual_score_for_action(first, visual_free_space)
    length_penalty = max(0, len(candidate.actions) - 1)
    target_score = target_cue_score if first == "stop_as_last_resort" else 0.0
    total = (
        weights["safety_margin"] * normalized_clearance
        + weights["intent"] * intent
        + weights["visual"] * visual_score
        + weights["target"] * target_score
        - weights["loop"] * penalties["recovery_loop_penalty"]
        - weights["repeat"] * penalties["repeated_reject_penalty"]
        - weights["switch"] * penalties["action_switching_penalty"]
        - weights["stop"] * penalties["unnecessary_stop_penalty"]
        - weights["length"] * length_penalty
    )
    if not rollout.safety_feasible and first != "stop_as_last_resort":
        total = -1e6
    return CandidateScore(
        sequence_candidate_id=candidate.sequence_id,
        candidate_sequence=candidate.actions,
        sequence_length=len(candidate.actions),
        min_clearance=rollout.min_clearance,
        min_barrier_h=rollout.min_barrier_h,
        cbf_violation=rollout.cbf_violation,
        would_block=rollout.would_block,
        predicted_wall_contact_steps=rollout.predicted_wall_contact_steps,
        safety_feasible=rollout.safety_feasible,
        safety_rejection_reason=rollout.safety_rejection_reason,
        intent_consistency_score=float(intent),
        visual_free_space_score=float(visual_score),
        target_or_stop_cue_score=float(target_score),
        recovery_loop_penalty=float(penalties["recovery_loop_penalty"]),
        repeated_reject_penalty=float(penalties["repeated_reject_penalty"]),
        unnecessary_stop_penalty=float(penalties["unnecessary_stop_penalty"]),
        action_switching_penalty=float(penalties["action_switching_penalty"]),
        candidate_sequence_length_penalty=float(length_penalty),
        normalized_clearance=float(normalized_clearance),
        total_score=float(total),
    )


def intent_consistency_score(
    *,
    nominal_action: VLNAction,
    candidate_sequence: list[str],
    penalties: dict[str, float] | None = None,
) -> float:
    first = candidate_sequence[0] if candidate_sequence else "stop_as_last_resort"
    second = candidate_sequence[1] if len(candidate_sequence) > 1 else ""
    rejected_many = (penalties or {}).get("repeated_reject_counter", 0.0) >= 2.0
    turn_loop = (penalties or {}).get("turn_loop_counter", 0.0) >= 2.0
    recovery_stuck = (penalties or {}).get("recovery_stuck_counter", 0.0) >= 2.0
    if nominal_action is VLNAction.FORWARD:
        scores = {
            "short_forward": 1.0 if not rejected_many else 0.45,
            "turn_left": 0.62,
            "turn_right": 0.62,
            "small_turn_left": 0.70,
            "small_turn_right": 0.70,
            "backoff": 0.38 if second not in {"turn_left", "turn_right", "short_forward"} else 0.58,
            "stop_as_last_resort": 0.0,
        }
        if turn_loop or recovery_stuck:
            scores.update(
                {
                    "short_forward": 0.42,
                    "turn_left": 0.28,
                    "turn_right": 0.28,
                    "small_turn_left": 0.22,
                    "small_turn_right": 0.22,
                    "backoff": 0.86 if second in {"turn_left", "turn_right", "short_forward"} else 0.72,
                }
            )
    elif nominal_action is VLNAction.TURN_LEFT:
        scores = {
            "turn_left": 1.0,
            "small_turn_left": 1.0,
            "backoff": 0.75 if second == "turn_left" else 0.52,
            "short_forward": 0.35,
            "turn_right": 0.20 if not rejected_many else 0.55,
            "small_turn_right": 0.25 if not rejected_many else 0.55,
            "stop_as_last_resort": 0.0,
        }
        if turn_loop or recovery_stuck:
            scores.update({"turn_left": 0.35, "small_turn_left": 0.28, "backoff": 0.88 if second == "turn_left" else 0.70})
    elif nominal_action is VLNAction.TURN_RIGHT:
        scores = {
            "turn_right": 1.0,
            "small_turn_right": 1.0,
            "backoff": 0.75 if second == "turn_right" else 0.52,
            "short_forward": 0.35,
            "turn_left": 0.20 if not rejected_many else 0.55,
            "small_turn_left": 0.25 if not rejected_many else 0.55,
            "stop_as_last_resort": 0.0,
        }
        if turn_loop or recovery_stuck:
            scores.update({"turn_right": 0.35, "small_turn_right": 0.28, "backoff": 0.88 if second == "turn_right" else 0.70})
    elif nominal_action is VLNAction.BACKOFF:
        scores = {
            "backoff": 1.0 if not rejected_many else 0.50,
            "turn_left": 0.55,
            "turn_right": 0.55,
            "small_turn_left": 0.60,
            "small_turn_right": 0.60,
            "short_forward": 0.20,
            "stop_as_last_resort": 0.0,
        }
    else:
        scores = {
            "stop_as_last_resort": 0.20,
            "short_forward": 0.25,
            "turn_left": 0.20,
            "turn_right": 0.20,
            "backoff": 0.20,
            "small_turn_left": 0.20,
            "small_turn_right": 0.20,
        }
    return float(scores.get(first, 0.0))


def compute_visual_free_space(image_path: str | Path | None) -> dict[str, Any]:
    if image_path is None or not Path(image_path).exists():
        return {
            "visual_free_space_score_available": False,
            "left_score": 0.5,
            "center_score": 0.5,
            "right_score": 0.5,
            "fallback_reason": "missing_image",
        }
    try:
        image = Image.open(image_path).convert("L").resize((96, 72))
    except Exception as exc:
        return {
            "visual_free_space_score_available": False,
            "left_score": 0.5,
            "center_score": 0.5,
            "right_score": 0.5,
            "fallback_reason": f"image_error:{exc}",
        }
    arr = np.asarray(image, dtype=np.float32) / 255.0
    edges = np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    thirds = {
        "left": (0, arr.shape[1] // 3),
        "center": (arr.shape[1] // 3, 2 * arr.shape[1] // 3),
        "right": (2 * arr.shape[1] // 3, arr.shape[1]),
    }
    scores: dict[str, float] = {}
    for name, (x0, x1) in thirds.items():
        region = arr[:, x0:x1]
        edge_region = edges[:, x0:x1]
        brightness = float(region.mean())
        edge_penalty = float(edge_region.mean())
        dark_penalty = float((region < 0.16).mean())
        scores[f"{name}_score"] = max(0.0, min(1.0, 0.65 * brightness + 0.35 * (1.0 - edge_penalty) - 0.20 * dark_penalty))
    return {
        "visual_free_space_score_available": True,
        **scores,
        "fallback_reason": "",
    }


def visual_score_for_action(action: str, visual_free_space: dict[str, Any]) -> float:
    if not visual_free_space.get("visual_free_space_score_available"):
        return 0.45
    if action in {"short_forward", "short_forward_segment"}:
        return float(visual_free_space.get("center_score", 0.5))
    if action in {"turn_left", "small_turn_left"}:
        return float(visual_free_space.get("left_score", 0.5))
    if action in {"turn_right", "small_turn_right"}:
        return float(visual_free_space.get("right_score", 0.5))
    if action == "backoff":
        return 0.45
    return 0.0


def target_or_stop_cue_score(image_path: str | Path | None) -> float:
    if image_path is None:
        return 0.0
    try:
        evidence = detect_gallery_wall_painting(image_path)
    except Exception:
        return 0.0
    data = evidence.to_dict()
    centered = evidence.center_x is not None and 0.15 <= evidence.center_x <= 0.85
    return float(min(1.0, data.get("target_ratio", 0.0) + data.get("bbox_area_ratio", 0.0) + (0.2 if centered else 0.0)))


def candidate_scores_to_rows(
    *,
    method: str,
    episode_id: str,
    step_idx: int,
    nominal_action: str,
    reject_reason: str,
    selection: DmpsSelection,
) -> list[dict[str, Any]]:
    rows = []
    for score in selection.candidate_scores:
        row = asdict(score)
        row.update(
            {
                "method": method,
                "episode_id": episode_id,
                "step_id": step_idx,
                "nominal_action": nominal_action,
                "reject_reason": reject_reason,
                "candidate_sequence_id": score.sequence_candidate_id,
                "candidate_sequence": json.dumps(list(score.candidate_sequence)),
                "selected": int(score.selected),
                "selection_reason": score.selection_reason,
            }
        )
        rows.append(row)
    return rows


def selected_replan_row(
    *,
    method: str,
    episode_id: str,
    step_idx: int,
    nominal_action: str,
    reject_reason: str,
    selection: DmpsSelection,
    executed_first_skill: SonicSkill,
    stop_source: str,
    post_reject_progress_evaluator_only: float,
) -> dict[str, Any]:
    first = selection.selected_sequence.actions[0]
    return {
        "method": method,
        "episode_id": episode_id,
        "step_id": step_idx,
        "nominal_action": nominal_action,
        "reject_reason": reject_reason,
        "selected_sequence": json.dumps(list(selection.selected_sequence.actions)),
        "executed_first_action": first,
        "executed_first_skill": executed_first_skill.name,
        "stop_source": stop_source,
        "non_stop_recovery_selected": int(first != "stop_as_last_resort"),
        "all_non_stop_candidates_infeasible": int(selection.all_non_stop_candidates_infeasible),
        "post_reject_progress_evaluator_only": post_reject_progress_evaluator_only,
        "control_returned_to_vln_next_step": int(selection.control_returned_to_vln_next_step),
        "selected_total_score": selection.selected_score.total_score,
        "selected_min_clearance": selection.selected_score.min_clearance,
        "selected_visual_free_space_score": selection.selected_score.visual_free_space_score,
        "selected_intent_consistency_score": selection.selected_score.intent_consistency_score,
    }


def _skill_rollout_points(skill: SonicSkill, *, pose: RobotPose2D, microsteps: int) -> list[RobotPose2D]:
    if isinstance(skill, PassiveSkill):
        return [pose]
    if isinstance(skill, TurnSkill):
        delta = skill.delta_yaw_deg / float(max(1, microsteps))
        return [RobotPose2D(pose.x, pose.y, wrap_degrees(pose.yaw_deg + delta * idx)) for idx in range(1, microsteps + 1)]
    if isinstance(skill, WalkSkill):
        step = skill.step_target_m / float(max(1, microsteps))
        return [
            RobotPose2D(pose.x + skill.vx * step * idx, pose.y + skill.vy * step * idx, skill.facing_yaw_deg)
            for idx in range(1, microsteps + 1)
        ]
    raise ValueError(f"unsupported skill type: {type(skill)!r}")


def _unit_from_yaw(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)
