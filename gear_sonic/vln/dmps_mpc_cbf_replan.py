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
PPSR_V2_METHOD_NAME = "vln_ppsr_v2_escape_macro"
PPSR_V2_ALIASES = ("vln_escape_macro_progress", "vln_dmps_escape_macro_progress")
PPSR_V2_LAST_RESORT_STOP_SOURCE = "ppsr_v2_last_resort_stop"
PPSR_V2_COMMITMENT_ABORT_STOP_SOURCE = "ppsr_v2_commitment_abort_stop"


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

PPSR_V2_SCORE_WEIGHTS = {
    **DEFAULT_SCORE_WEIGHTS,
    "terminal_forward": 1.20,
    "clearance_gain": 0.60,
    "end_front_clearance": 0.30,
    "translation": 0.55,
    "escape_macro": 0.45,
    "visual_novelty": 0.20,
    "turn_only": 1.25,
    "small_turn_loop": 1.50,
}


@dataclass(frozen=True)
class CandidateSequence:
    sequence_id: str
    actions: tuple[str, ...]
    skills: tuple[SonicSkill, ...]
    stop_as_last_resort: bool = False
    escape_macro: bool = False


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
    start_clearance: float
    end_clearance: float
    clearance_gain: float
    end_front_clearance: float
    terminal_can_short_forward: bool
    terminal_short_forward_min_clearance: float
    terminal_short_forward_would_block: bool
    terminal_visual_novelty_score: float
    terminal_heading_change_abs: float
    sequence_translation_distance: float
    turn_only_sequence: bool
    contains_translation: bool
    escape_macro_completed_candidate: bool


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
    start_clearance: float
    end_clearance: float
    clearance_gain: float
    end_front_clearance: float
    terminal_can_short_forward: bool
    terminal_short_forward_min_clearance: float
    terminal_short_forward_would_block: bool
    terminal_visual_novelty_score: float
    terminal_heading_change_abs: float
    sequence_translation_distance: float
    turn_only_sequence: bool
    contains_translation: bool
    escape_macro_candidate: bool
    escape_macro_completed_candidate: bool
    hard_mask_applied: bool
    hard_mask_reasons: tuple[str, ...]
    masked_candidates: tuple[str, ...]
    translation_required: bool
    turn_only_candidate_blocked: bool
    repeated_turn_loop_blocked: bool
    small_turn_loop_risk: float
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
    stuck_mode: bool = False
    stuck_mode_reasons: tuple[str, ...] = ()
    horizon_used: int = 2
    safe_translation_candidate_exists: bool = False


def normalize_dmps_method(method: str) -> str:
    if method == DMPS_ALIAS:
        return DMPS_METHOD_NAME
    if method in PPSR_V2_ALIASES:
        return PPSR_V2_METHOD_NAME
    return method


def is_dmps_method(method: str) -> bool:
    return normalize_dmps_method(method) == DMPS_METHOD_NAME


def is_ppsr_v2_method(method: str) -> bool:
    return normalize_dmps_method(method) == PPSR_V2_METHOD_NAME


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


def build_ppsr_v2_candidate_sequences(
    *,
    pose: RobotPose2D,
    nominal_action: VLNAction | str,
    progress_monitor: ProgressMonitor,
    horizon_default: int = 2,
    horizon_stuck: int = 3,
) -> tuple[list[CandidateSequence], dict[str, Any]]:
    signal = ppsr_v2_stuck_signal(progress_monitor)
    horizon = horizon_stuck if signal["stuck_mode"] else horizon_default
    candidates = build_candidate_sequences(pose=pose, nominal_action=nominal_action, horizon=horizon)
    if signal["stuck_mode"]:
        heading = float(pose.yaw_deg)
        macros = [
            ("escape_left", ("backoff", "turn_left", "short_forward")),
            ("escape_right", ("backoff", "turn_right", "short_forward")),
            ("wide_turn_left_forward", ("wide_turn_left", "short_forward")),
            ("wide_turn_right_forward", ("wide_turn_right", "short_forward")),
            ("turn_left_forward", ("turn_left", "short_forward")),
            ("turn_right_forward", ("turn_right", "short_forward")),
            ("reverse_and_open_left", ("backoff", "wide_turn_left", "short_forward")),
            ("reverse_and_open_right", ("backoff", "wide_turn_right", "short_forward")),
        ]
        for sequence_id, actions in macros:
            clipped = tuple(actions[: max(1, horizon)])
            candidates.append(
                CandidateSequence(
                    sequence_id=sequence_id,
                    actions=clipped,
                    skills=tuple(_skills_for_sequence(clipped, heading)),
                    stop_as_last_resort=False,
                    escape_macro=True,
                )
            )
    signal["horizon_used"] = horizon
    return candidates, signal


def ppsr_v2_stuck_signal(progress_monitor: ProgressMonitor) -> dict[str, Any]:
    state = progress_monitor.state
    recent_recoveries = list(state.recent_selected_recoveries)
    recent_visual_hashes = [item for item in state.recent_visual_hashes if item]
    reasons: list[str] = []
    if state.recovery_stuck_counter >= 1:
        reasons.append("recovery_stuck_counter")
    if state.turn_loop_counter >= 1:
        reasons.append("turn_loop_counter")
    if state.repeated_reject_counter >= 2:
        reasons.append("repeated_forward_reject")
    if len(recent_visual_hashes) >= 3 and len(set(recent_visual_hashes[-3:])) == 1:
        reasons.append("same_visual_hash_repeated")
    if last_two_recoveries_are_turn_family(progress_monitor):
        reasons.append("last_two_recoveries_are_turn_family")
    if _small_turn_count(recent_recoveries[-3:]) >= 2:
        reasons.append("small_turns_in_recent_recoveries")
    return {
        "stuck_mode": bool(reasons),
        "stuck_mode_reasons": tuple(reasons),
        "recent_selected_recoveries": tuple(recent_recoveries),
        "last_two_recoveries_are_turn_family": last_two_recoveries_are_turn_family(progress_monitor),
        "recent_small_turn_count": _small_turn_count(recent_recoveries[-3:]),
    }


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
    if action == "wide_turn_left":
        return TurnSkill(delta_yaw_deg=45.0, face_yaw_deg=wrap_degrees(heading + 45.0), duration=0.40)
    if action == "wide_turn_right":
        return TurnSkill(delta_yaw_deg=-45.0, face_yaw_deg=wrap_degrees(heading - 45.0), duration=0.40)
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


def select_ppsr_v2_replan(
    *,
    bridge: CasaVlnBridge,
    pose: RobotPose2D,
    maze: MazeMap,
    robot_radius: float,
    nominal_action: VLNAction,
    image_path: str | Path | None,
    progress_monitor: ProgressMonitor,
    step_idx: int,
    horizon_default: int,
    horizon_stuck: int,
    safety_margin: float,
    score_weights: dict[str, float] | None = None,
) -> DmpsSelection:
    del bridge
    weights = {**PPSR_V2_SCORE_WEIGHTS, **(score_weights or {})}
    visual = compute_visual_free_space(image_path)
    target_cue = target_or_stop_cue_score(image_path)
    snapshot = progress_monitor.snapshot(step_idx=step_idx)
    candidates, stuck_signal = build_ppsr_v2_candidate_sequences(
        pose=pose,
        nominal_action=nominal_action,
        progress_monitor=progress_monitor,
        horizon_default=horizon_default,
        horizon_stuck=horizon_stuck,
    )
    raw_scores: list[CandidateScore] = []
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
        raw_scores.append(
            score_candidate(
                candidate=candidate,
                rollout=rollout,
                nominal_action=nominal_action,
                visual_free_space=visual,
                target_cue_score=target_cue,
                penalties=penalties,
                weights=weights,
                safety_margin=safety_margin,
            )
        )

    safe_translation_candidate_exists = any(
        score.safety_feasible and score.contains_translation and score.candidate_sequence[0] != "stop_as_last_resort"
        for score in raw_scores
    )
    scored = [
        apply_ppsr_v2_hard_mask(
            score=score,
            progress_monitor=progress_monitor,
            nominal_action=nominal_action,
            stuck_signal=stuck_signal,
            safe_translation_candidate_exists=safe_translation_candidate_exists,
        )
        for score in raw_scores
    ]

    non_stop_safe = [
        item
        for item in scored
        if item.safety_feasible
        and item.candidate_sequence[0] != "stop_as_last_resort"
        and not item.hard_mask_applied
    ]
    stop_scores = [item for item in scored if item.candidate_sequence[0] == "stop_as_last_resort"]
    all_non_stop_candidates_infeasible = len(non_stop_safe) == 0
    if non_stop_safe:
        selected_score = sorted(non_stop_safe, key=lambda item: item.total_score, reverse=True)[0]
        reason = "best_safe_escape_macro_progress_candidate"
    elif stop_scores:
        selected_score = stop_scores[0]
        reason = "all_non_stop_candidates_masked_or_infeasible"
    else:
        raise RuntimeError("PPSR-v2 candidate generation produced no selectable candidate")

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
        stuck_mode=bool(stuck_signal["stuck_mode"]),
        stuck_mode_reasons=tuple(stuck_signal["stuck_mode_reasons"]),
        horizon_used=int(stuck_signal["horizon_used"]),
        safe_translation_candidate_exists=bool(safe_translation_candidate_exists),
    )


def apply_ppsr_v2_hard_mask(
    *,
    score: CandidateScore,
    progress_monitor: ProgressMonitor,
    nominal_action: VLNAction,
    stuck_signal: dict[str, Any],
    safe_translation_candidate_exists: bool,
) -> CandidateScore:
    del nominal_action
    first = score.candidate_sequence[0] if score.candidate_sequence else "stop_as_last_resort"
    reasons: list[str] = []
    translation_required = False
    turn_only_candidate_blocked = False
    repeated_turn_loop_blocked = False
    state = progress_monitor.state
    recent_recoveries = list(state.recent_selected_recoveries)

    if first in {"small_turn_left", "small_turn_right"} and state.recovery_stuck_counter >= 1:
        reasons.append("mask_small_turn_after_recovery_stuck")
    if stuck_signal.get("last_two_recoveries_are_turn_family"):
        translation_required = True
        if not score.contains_translation and first != "stop_as_last_resort":
            reasons.append("translation_required_after_two_turn_recoveries")
            turn_only_candidate_blocked = score.turn_only_sequence
    if state.turn_loop_counter >= 1 and score.turn_only_sequence and first != "stop_as_last_resort":
        reasons.append("turn_only_blocked_after_turn_loop")
        turn_only_candidate_blocked = True
        repeated_turn_loop_blocked = True
    if _small_turn_count(recent_recoveries[-3:]) >= 2 and score.turn_only_sequence:
        reasons.append("turn_only_blocked_after_recent_small_turns")
        turn_only_candidate_blocked = True
    if state.repeated_reject_counter >= 2 and first == "short_forward" and not score.terminal_can_short_forward:
        reasons.append("plain_short_forward_blocked_after_repeated_forward_reject")
    if stuck_signal.get("stuck_mode") and safe_translation_candidate_exists and score.turn_only_sequence:
        reasons.append("turn_only_blocked_when_safe_translation_exists")
        turn_only_candidate_blocked = True

    masked = bool(reasons)
    total = -1e6 if masked and first != "stop_as_last_resort" else score.total_score
    return CandidateScore(
        **{
            **asdict(score),
            "hard_mask_applied": masked,
            "hard_mask_reasons": tuple(reasons),
            "masked_candidates": (score.sequence_candidate_id,) if masked else (),
            "translation_required": translation_required,
            "turn_only_candidate_blocked": turn_only_candidate_blocked,
            "repeated_turn_loop_blocked": repeated_turn_loop_blocked,
            "total_score": total,
        }
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
    start_clearance, _ = clearance_to_blocking_geometry(start_pose.x, start_pose.y, maze=maze, robot_radius=robot_radius)

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
    end_clearance, _ = clearance_to_blocking_geometry(pose.x, pose.y, maze=maze, robot_radius=robot_radius)
    terminal_probe = terminal_short_forward_probe(
        pose=pose,
        maze=maze,
        robot_radius=robot_radius,
        safety_margin=safety_margin,
        microsteps=microsteps_per_skill,
    )
    translation_distance = _sequence_translation_distance(candidate.actions)
    contains_translation = _contains_translation(candidate.actions)
    turn_only_sequence = _is_turn_only_sequence(candidate.actions)
    heading_change = abs(wrap_degrees(pose.yaw_deg - start_pose.yaw_deg))
    terminal_visual_novelty_score = min(1.0, translation_distance / 0.35 + heading_change / 120.0)
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
        start_clearance=float(start_clearance),
        end_clearance=float(end_clearance),
        clearance_gain=float(end_clearance - start_clearance),
        end_front_clearance=float(terminal_probe["terminal_short_forward_min_clearance"]),
        terminal_can_short_forward=bool(terminal_probe["terminal_can_short_forward"]),
        terminal_short_forward_min_clearance=float(terminal_probe["terminal_short_forward_min_clearance"]),
        terminal_short_forward_would_block=bool(terminal_probe["terminal_short_forward_would_block"]),
        terminal_visual_novelty_score=float(terminal_visual_novelty_score),
        terminal_heading_change_abs=float(heading_change),
        sequence_translation_distance=float(translation_distance),
        turn_only_sequence=bool(turn_only_sequence),
        contains_translation=bool(contains_translation),
        escape_macro_completed_candidate=bool(candidate.escape_macro and contains_translation and len(candidate.actions) >= 2),
    )


def terminal_short_forward_probe(
    *,
    pose: RobotPose2D,
    maze: MazeMap,
    robot_radius: float,
    safety_margin: float,
    microsteps: int = 8,
) -> dict[str, Any]:
    skill = skill_for_recovery_action("short_forward", pose.yaw_deg)
    points = _skill_rollout_points(skill, pose=pose, microsteps=microsteps)
    clearances: list[float] = []
    would_block = False
    for next_pose in points:
        clearance, _ = clearance_to_blocking_geometry(next_pose.x, next_pose.y, maze=maze, robot_radius=robot_radius)
        clearances.append(float(clearance))
        if not maze.is_xy_safe(next_pose.x, next_pose.y, robot_radius=robot_radius) or clearance < safety_margin:
            would_block = True
    min_clearance = min(clearances) if clearances else 0.0
    return {
        "terminal_can_short_forward": bool((not would_block) and min_clearance >= safety_margin),
        "terminal_short_forward_min_clearance": float(min_clearance),
        "terminal_short_forward_would_block": bool(would_block),
    }


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
    small_turn_loop_risk = float(
        first in {"small_turn_left", "small_turn_right"}
        and (penalties.get("turn_loop_counter", 0.0) >= 1.0 or penalties.get("recovery_stuck_counter", 0.0) >= 1.0)
    )
    total = (
        weights["safety_margin"] * normalized_clearance
        + weights["intent"] * intent
        + weights["visual"] * visual_score
        + weights["target"] * target_score
        + weights.get("terminal_forward", 0.0) * float(rollout.terminal_can_short_forward)
        + weights.get("clearance_gain", 0.0) * rollout.clearance_gain
        + weights.get("end_front_clearance", 0.0) * rollout.end_front_clearance
        + weights.get("translation", 0.0) * rollout.sequence_translation_distance
        + weights.get("escape_macro", 0.0) * float(candidate.escape_macro and rollout.contains_translation)
        + weights.get("visual_novelty", 0.0) * rollout.terminal_visual_novelty_score
        - weights["loop"] * penalties["recovery_loop_penalty"]
        - weights["repeat"] * penalties["repeated_reject_penalty"]
        - weights["switch"] * penalties["action_switching_penalty"]
        - weights["stop"] * penalties["unnecessary_stop_penalty"]
        - weights["length"] * length_penalty
        - weights.get("turn_only", 0.0) * float(rollout.turn_only_sequence)
        - weights.get("small_turn_loop", 0.0) * small_turn_loop_risk
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
        start_clearance=rollout.start_clearance,
        end_clearance=rollout.end_clearance,
        clearance_gain=rollout.clearance_gain,
        end_front_clearance=rollout.end_front_clearance,
        terminal_can_short_forward=rollout.terminal_can_short_forward,
        terminal_short_forward_min_clearance=rollout.terminal_short_forward_min_clearance,
        terminal_short_forward_would_block=rollout.terminal_short_forward_would_block,
        terminal_visual_novelty_score=rollout.terminal_visual_novelty_score,
        terminal_heading_change_abs=rollout.terminal_heading_change_abs,
        sequence_translation_distance=rollout.sequence_translation_distance,
        turn_only_sequence=rollout.turn_only_sequence,
        contains_translation=rollout.contains_translation,
        escape_macro_candidate=candidate.escape_macro,
        escape_macro_completed_candidate=rollout.escape_macro_completed_candidate,
        hard_mask_applied=False,
        hard_mask_reasons=(),
        masked_candidates=(),
        translation_required=False,
        turn_only_candidate_blocked=False,
        repeated_turn_loop_blocked=False,
        small_turn_loop_risk=small_turn_loop_risk,
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
    if action in {"turn_left", "small_turn_left", "wide_turn_left"}:
        return float(visual_free_space.get("left_score", 0.5))
    if action in {"turn_right", "small_turn_right", "wide_turn_right"}:
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
    committed_second_action: str = "",
    committed_second_step_executed: bool = False,
    committed_second_step_aborted: bool = False,
    second_action_safety_feasible: bool = False,
    abort_reason: str = "",
    max_commit_steps: int = 1,
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
        "stuck_mode": int(selection.stuck_mode),
        "stuck_mode_reasons": json.dumps(list(selection.stuck_mode_reasons)),
        "horizon_used": selection.horizon_used,
        "safe_translation_candidate_exists": int(selection.safe_translation_candidate_exists),
        "escape_macro_selected": int(selection.selected_sequence.escape_macro),
        "selected_terminal_can_short_forward": int(selection.selected_score.terminal_can_short_forward),
        "selected_clearance_gain": selection.selected_score.clearance_gain,
        "selected_sequence_translation_distance": selection.selected_score.sequence_translation_distance,
        "selected_turn_only_sequence": int(selection.selected_score.turn_only_sequence),
        "selected_hard_mask_applied": int(selection.selected_score.hard_mask_applied),
        "committed_second_action": committed_second_action,
        "committed_second_step_executed": int(committed_second_step_executed),
        "committed_second_step_aborted": int(committed_second_step_aborted),
        "second_action_safety_feasible": int(second_action_safety_feasible),
        "abort_reason": abort_reason,
        "max_commit_steps": max_commit_steps,
        "control_returned_to_vln": 1,
        "commitment_used_goal_or_astar": 0,
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


def action_family(action: str) -> str:
    if action in {"turn_left", "small_turn_left", "wide_turn_left"}:
        return "turn_left"
    if action in {"turn_right", "small_turn_right", "wide_turn_right"}:
        return "turn_right"
    if action in {"short_forward", "short_forward_segment"}:
        return "forward"
    return action


def is_turn_family(action: str) -> bool:
    return action_family(action) in {"turn_left", "turn_right"}


def last_two_recoveries_are_turn_family(progress_monitor: ProgressMonitor) -> bool:
    recent = list(progress_monitor.state.recent_selected_recoveries)
    return len(recent) >= 2 and all(is_turn_family(item) for item in recent[-2:])


def _small_turn_count(actions: list[str]) -> int:
    return sum(1 for action in actions if action in {"small_turn_left", "small_turn_right"})


def _contains_translation(actions: tuple[str, ...]) -> bool:
    return any(action in {"short_forward", "short_forward_segment", "backoff"} for action in actions)


def _is_turn_only_sequence(actions: tuple[str, ...]) -> bool:
    return bool(actions) and all(is_turn_family(action) for action in actions)


def _sequence_translation_distance(actions: tuple[str, ...]) -> float:
    distance = 0.0
    for action in actions:
        if action in {"short_forward", "short_forward_segment"}:
            distance += 0.16
        elif action == "backoff":
            distance += 0.18
    return distance
