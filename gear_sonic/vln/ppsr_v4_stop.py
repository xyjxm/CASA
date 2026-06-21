"""PPSR-v4 non-privileged stop verification and audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

from .actions import ActionDecision, VLNAction
from .no_casa_policy import InferenceInput, detect_gallery_wall_painting, parse_route_actions_from_instruction


PPSR_V4_METHOD_NAME = "vln_ppsr_v4_zero_unsafe_success60"
PPSR_V4_ALIASES = ("vln_ppsr_v4", "vln_ppsr_zero_unsafe_success60")
PPSR_V4_LAST_RESORT_STOP_SOURCE = "ppsr_v4_last_resort_stop"
PPSR_V4_COMMITMENT_ABORT_STOP_SOURCE = "ppsr_v4_commitment_abort_stop"

V4_PASS_STATUS = "PASS_ZERO_UNSAFE_SUCCESS60"
V4_FAILED_UNSAFE_STATUS = "FAILED_UNSAFE_NOT_ZERO"
V4_FAILED_SUCCESS_STATUS = "FAILED_SUCCESS_BELOW_60"
V4_FAILED_STOP_LEAKAGE_STATUS = "FAILED_STOP_LEAKAGE"
V4_FAILED_PRIVILEGED_STATUS = "FAILED_PRIVILEGED_LEAKAGE"
V4_BLOCKED_STATUS = "PARTIAL_BLOCKED_ENGINEERING"


@dataclass(frozen=True)
class PpsrV4StopVerifierConfig:
    min_motion_actions: int = 2
    min_step_for_visual_stop: int = 2
    min_strong_visual_motion_actions: int = 8
    min_route_motion_actions: int = 8
    min_late_recovery_motion_actions: int = 8
    min_calibrated_route_motion_actions: int = 20
    min_late_calibrated_route_motion_actions: int = 70
    min_late_route_low_visual_motion_actions: int = 999
    min_late_route_forward_motion_actions: int = 110
    min_late_route_forward_recent_turns: int = 4
    min_late_forward_momentum_motion_actions: int = 999
    min_late_forward_momentum_streak: int = 3
    min_late_forward_momentum_recent_turns: int = 2
    min_ultra_late_route_motion_actions: int = 115
    min_ultra_late_route_recent_turns: int = 3
    min_ultra_late_forward_streak_stop_motion_actions: int = 999
    min_ultra_late_forward_streak_stop_streak: int = 8
    min_recent_visual_memory_motion_actions: int = 120
    min_recent_visual_memory_recent_turns: int = 4
    min_recent_visual_memory_forward_streak: int = 1
    min_late_forward_memory_motion_actions: int = 100
    min_late_forward_memory_streak: int = 7
    min_borderline_visual_stop_holdoff_motion_actions: int = 70
    min_route_cadence_motion_actions: int = 999
    min_route_cadence_recent_turns: int = 5
    min_low_conf_corridor_forward_streak: int = 3
    min_low_conf_route_complete_motion_actions: int = 20
    min_visual_goal_tracking_motion_actions: int = 180
    strong_visual_ratio: float = 0.65
    strong_visual_bbox_area: float = 0.75
    policy_stop_visual_ratio: float = 0.30
    route_complete_visual_ratio: float = 0.65
    late_recovery_visual_ratio: float = 0.45
    calibrated_route_visual_ratio: float = 0.385
    calibrated_route_bbox_area: float = 0.55
    calibrated_route_visual_growth_ratio: float = 0.15
    late_calibrated_route_visual_ratio: float = 0.375
    late_calibrated_route_bbox_area: float = 0.70
    late_calibrated_route_visual_growth_ratio: float = 0.15
    late_route_low_visual_ratio: float = 1.01
    late_route_low_visual_bbox_area: float = 0.90
    late_route_forward_visual_ratio: float = 0.03
    late_forward_momentum_visual_ratio: float = 0.06
    ultra_late_route_visual_ratio: float = 0.34
    ultra_late_route_bbox_area: float = 0.70
    ultra_late_forward_streak_stop_visual_ratio: float = 0.035
    ultra_late_forward_streak_stop_bbox_area: float = 0.95
    recent_visual_memory_ratio: float = 0.20
    recent_visual_memory_bbox_area: float = 0.95
    recent_visual_memory_current_max_ratio: float = 0.01
    recent_visual_memory_current_max_bbox_area: float = 0.60
    late_forward_memory_min_recent_ratio: float = 0.07
    late_forward_memory_max_recent_ratio: float = 0.18
    late_forward_memory_recent_bbox_area: float = 0.90
    late_forward_memory_current_max_ratio: float = 0.02
    late_forward_memory_current_bbox_area: float = 0.90
    min_low_ratio_full_bbox_memory_motion_actions: int = 55
    min_low_ratio_full_bbox_memory_forward_streak: int = 3
    min_low_ratio_full_bbox_memory_recent_turns: int = 3
    low_ratio_full_bbox_memory_min_ratio: float = 0.005
    low_ratio_full_bbox_memory_max_ratio: float = 0.020
    low_ratio_full_bbox_memory_bbox_area: float = 0.70
    low_ratio_full_bbox_memory_stop_probability: float = 0.95
    borderline_visual_stop_holdoff_ratio: float = 0.40
    max_borderline_visual_stop_holdoff_forward_streak: int = 1
    max_low_conf_route_turn_probability: float = 0.80
    min_low_conf_route_forward_probability: float = 0.20
    max_low_conf_corridor_turn_visual_ratio: float = 0.08
    max_late_route_forward_streak: int = 3
    max_ultra_late_forward_streak_stop_recent_turns: int = 1
    max_late_forward_memory_recent_turns: int = 2
    max_route_cadence_forward_streak: int = 3
    max_route_cadence_visual_ratio: float = 0.04
    visual_goal_tracking_ratio: float = 0.035
    max_visual_goal_tracking_ratio: float = 0.12
    max_visual_goal_tracking_growth_ratio: float = 0.08
    min_visual_goal_tracking_bbox_area: float = 0.90
    visual_goal_center_deadband: float = 0.08
    visual_growth_ratio: float = 0.025
    centered_min_x: float = 0.08
    centered_max_x: float = 0.92
    max_recent_stop_count: int = 0


def normalize_ppsr_v4_method(method: str) -> str:
    return PPSR_V4_METHOD_NAME if method in PPSR_V4_ALIASES else method


def is_ppsr_v4_method(method: str) -> bool:
    return normalize_ppsr_v4_method(method) == PPSR_V4_METHOD_NAME


def apply_ppsr_v4_stop_verifier(
    *,
    obs: InferenceInput,
    decision: ActionDecision,
    config: PpsrV4StopVerifierConfig | None = None,
) -> ActionDecision:
    """Calibrate stop recall with only policy-facing language, image, and history.

    This deliberately does not read pose, goal distance, shortest paths, A*, map
    cell progress, evaluator success, or oracle waypoints.
    """

    cfg = config or PpsrV4StopVerifierConfig()
    metadata = dict(decision.metadata or {})
    evidence = _safe_wall_painting_evidence(obs.image_path)
    history_evidence = [_safe_wall_painting_evidence(path) for path in obs.history_image_paths[-3:]]
    history_best_ratio = max([float(item.get("target_ratio", 0.0)) for item in history_evidence] or [0.0])
    history_best_bbox_area = max([float(item.get("bbox_area_ratio", 0.0)) for item in history_evidence] or [0.0])
    history_recent_strong_visual = any(
        _evidence_centered(item, cfg)
        and bool(item.get("visible"))
        and float(item.get("target_ratio", 0.0)) >= cfg.recent_visual_memory_ratio
        and float(item.get("bbox_area_ratio", 0.0)) >= cfg.recent_visual_memory_bbox_area
        for item in history_evidence
    )
    history_recent_forward_memory_visual = any(
        _evidence_centered(item, cfg)
        and float(item.get("target_ratio", 0.0)) >= cfg.late_forward_memory_min_recent_ratio
        and float(item.get("target_ratio", 0.0)) <= cfg.late_forward_memory_max_recent_ratio
        and float(item.get("bbox_area_ratio", 0.0)) >= cfg.late_forward_memory_recent_bbox_area
        for item in history_evidence
    )
    target_ratio = float(evidence.get("target_ratio", 0.0))
    bbox_area = float(evidence.get("bbox_area_ratio", 0.0))
    center_x = evidence.get("center_x")
    centered = _evidence_centered(evidence, cfg)
    visible = bool(evidence.get("visible")) and centered
    visual_growth = max(0.0, target_ratio - history_best_ratio)
    motion_count = _motion_action_count(obs.previous_actions)
    recent_stop_count = _recent_count(obs.previous_actions, VLNAction.STOP.value)
    recent_forward_streak = _recent_count(obs.previous_actions, VLNAction.FORWARD.value)
    recent_turn_count = _recent_turn_count(obs.previous_actions[-10:])
    recent_blocked = bool(obs.previous_skill_status and obs.previous_skill_status[-1] in {"blocked", "collision"})
    route_actions, route_mode = _route_actions_for_instruction(obs.instruction)
    route_index = _route_progress(route_actions, obs.previous_actions, turn_only=route_mode == "natural_turns")
    route_complete = bool(route_actions) and route_index >= len(route_actions)
    route_near_complete = bool(route_actions) and route_index >= max(0, len(route_actions) - 1)
    recent_recovery = any(action in {VLNAction.BACKOFF.value, VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value} for action in obs.previous_actions[-4:])
    navid_stop_hint = str(metadata.get("real_navid_parsed_action") or "").lower() == VLNAction.STOP.value
    model_stop_hint = decision.action is VLNAction.STOP or navid_stop_hint
    raw_text = decision.raw_output if isinstance(decision.raw_output, str) else ""
    raw_stop_hint = "stop" in raw_text.lower() or "done" in raw_text.lower()
    probabilities = metadata.get("probabilities") if isinstance(metadata.get("probabilities"), dict) else {}
    turn_probability = max(
        _float_probability(probabilities.get(VLNAction.TURN_LEFT.value)),
        _float_probability(probabilities.get(VLNAction.TURN_RIGHT.value)),
    )
    forward_probability = _float_probability(probabilities.get(VLNAction.FORWARD.value))
    stop_probability = _float_probability(probabilities.get(VLNAction.STOP.value))

    strong_visual_stop = (
        visible
        and obs.step_idx >= cfg.min_step_for_visual_stop
        and motion_count >= cfg.min_strong_visual_motion_actions
        and (model_stop_hint or raw_stop_hint or route_complete)
        and target_ratio >= cfg.strong_visual_ratio
        and bbox_area >= cfg.strong_visual_bbox_area
    )
    policy_verified_stop = (
        visible
        and motion_count >= cfg.min_motion_actions
        and (model_stop_hint or raw_stop_hint)
        and target_ratio >= cfg.policy_stop_visual_ratio
        and bbox_area >= cfg.strong_visual_bbox_area * 0.5
    )
    route_verified_stop = (
        visible
        and motion_count >= cfg.min_route_motion_actions
        and route_complete
        and target_ratio >= cfg.route_complete_visual_ratio
        and bbox_area >= cfg.strong_visual_bbox_area
    )
    calibrated_route_visual_stop = (
        visible
        and motion_count >= cfg.min_calibrated_route_motion_actions
        and route_complete
        and target_ratio >= cfg.calibrated_route_visual_ratio
        and bbox_area >= cfg.calibrated_route_bbox_area
        and visual_growth >= cfg.calibrated_route_visual_growth_ratio
    )
    late_calibrated_route_visual_stop = (
        visible
        and motion_count >= cfg.min_late_calibrated_route_motion_actions
        and route_complete
        and target_ratio >= cfg.late_calibrated_route_visual_ratio
        and bbox_area >= cfg.late_calibrated_route_bbox_area
        and visual_growth >= cfg.late_calibrated_route_visual_growth_ratio
    )
    late_route_low_visual_stop = (
        visible
        and motion_count >= cfg.min_late_route_low_visual_motion_actions
        and route_complete
        and target_ratio >= cfg.late_route_low_visual_ratio
        and bbox_area >= cfg.late_route_low_visual_bbox_area
    )
    ultra_late_route_visual_stop = (
        visible
        and motion_count >= cfg.min_ultra_late_route_motion_actions
        and route_complete
        and recent_turn_count >= cfg.min_ultra_late_route_recent_turns
        and target_ratio >= cfg.ultra_late_route_visual_ratio
        and bbox_area >= cfg.ultra_late_route_bbox_area
    )
    ultra_late_forward_streak_visual_stop = (
        visible
        and motion_count >= cfg.min_ultra_late_forward_streak_stop_motion_actions
        and route_complete
        and recent_forward_streak >= cfg.min_ultra_late_forward_streak_stop_streak
        and recent_turn_count <= cfg.max_ultra_late_forward_streak_stop_recent_turns
        and target_ratio >= cfg.ultra_late_forward_streak_stop_visual_ratio
        and bbox_area >= cfg.ultra_late_forward_streak_stop_bbox_area
    )
    recent_visual_memory_stop = (
        route_complete
        and motion_count >= cfg.min_recent_visual_memory_motion_actions
        and recent_turn_count >= cfg.min_recent_visual_memory_recent_turns
        and recent_forward_streak >= cfg.min_recent_visual_memory_forward_streak
        and history_recent_strong_visual
        and centered
        and target_ratio <= cfg.recent_visual_memory_current_max_ratio
        and 0.0 < bbox_area <= cfg.recent_visual_memory_current_max_bbox_area
    )
    late_forward_visual_memory_stop = (
        route_complete
        and motion_count >= cfg.min_late_forward_memory_motion_actions
        and recent_forward_streak >= cfg.min_late_forward_memory_streak
        and recent_turn_count <= cfg.max_late_forward_memory_recent_turns
        and history_recent_forward_memory_visual
        and centered
        and target_ratio <= cfg.late_forward_memory_current_max_ratio
        and bbox_area >= cfg.late_forward_memory_current_bbox_area
    )
    low_ratio_full_bbox_memory_stop = (
        route_complete
        and motion_count >= cfg.min_low_ratio_full_bbox_memory_motion_actions
        and recent_forward_streak >= cfg.min_low_ratio_full_bbox_memory_forward_streak
        and recent_turn_count >= cfg.min_low_ratio_full_bbox_memory_recent_turns
        and history_recent_forward_memory_visual
        and centered
        and stop_probability >= cfg.low_ratio_full_bbox_memory_stop_probability
        and cfg.low_ratio_full_bbox_memory_min_ratio <= target_ratio <= cfg.low_ratio_full_bbox_memory_max_ratio
        and bbox_area >= cfg.low_ratio_full_bbox_memory_bbox_area
    )
    late_recovery_stop = (
        visible
        and motion_count >= cfg.min_late_recovery_motion_actions
        and recent_recovery
        and route_near_complete
        and target_ratio >= cfg.late_recovery_visual_ratio
        and (bbox_area >= cfg.strong_visual_bbox_area * 0.65 or visual_growth >= cfg.visual_growth_ratio)
    )
    verifier_accepts_stop = bool(
        not recent_blocked
        and recent_stop_count <= cfg.max_recent_stop_count
        and (
            strong_visual_stop
            or policy_verified_stop
            or route_verified_stop
            or calibrated_route_visual_stop
            or late_calibrated_route_visual_stop
            or late_route_low_visual_stop
            or ultra_late_route_visual_stop
            or ultra_late_forward_streak_visual_stop
            or recent_visual_memory_stop
            or late_forward_visual_memory_stop
            or low_ratio_full_bbox_memory_stop
            or late_recovery_stop
        )
    )
    borderline_visual_stop_holdoff = bool(
        verifier_accepts_stop
        and decision.action is not VLNAction.STOP
        and route_complete
        and not recent_blocked
        and motion_count >= cfg.min_borderline_visual_stop_holdoff_motion_actions
        and recent_forward_streak <= cfg.max_borderline_visual_stop_holdoff_forward_streak
        and target_ratio < cfg.borderline_visual_stop_holdoff_ratio
        and (calibrated_route_visual_stop or late_calibrated_route_visual_stop)
        and not (
            strong_visual_stop
            or policy_verified_stop
            or route_verified_stop
            or late_route_low_visual_stop
            or ultra_late_route_visual_stop
            or ultra_late_forward_streak_visual_stop
            or recent_visual_memory_stop
            or late_forward_visual_memory_stop
            or low_ratio_full_bbox_memory_stop
            or late_recovery_stop
        )
    )
    late_route_forward_stabilizer = bool(
        not verifier_accepts_stop
        and not recent_blocked
        and route_complete
        and decision.action in {VLNAction.TURN_LEFT, VLNAction.TURN_RIGHT}
        and motion_count >= cfg.min_late_route_forward_motion_actions
        and target_ratio <= cfg.late_route_forward_visual_ratio
        and recent_turn_count >= cfg.min_late_route_forward_recent_turns
        and recent_forward_streak <= cfg.max_late_route_forward_streak
    )
    late_forward_momentum_lock = bool(
        not verifier_accepts_stop
        and not recent_blocked
        and route_complete
        and decision.action in {VLNAction.TURN_LEFT, VLNAction.TURN_RIGHT}
        and motion_count >= cfg.min_late_forward_momentum_motion_actions
        and target_ratio <= cfg.late_forward_momentum_visual_ratio
        and recent_forward_streak >= cfg.min_late_forward_momentum_streak
        and recent_turn_count >= cfg.min_late_forward_momentum_recent_turns
    )
    visual_goal_action = _visual_goal_tracking_action(
        center_x=center_x,
        cfg=cfg,
        route_complete=route_complete,
        visible=visible,
        recent_blocked=recent_blocked,
        motion_count=motion_count,
        target_ratio=target_ratio,
        bbox_area=bbox_area,
        visual_growth=visual_growth,
        verifier_accepts_stop=verifier_accepts_stop,
    )
    visual_goal_tracker_applied = bool(visual_goal_action is not None and visual_goal_action is not decision.action)
    low_conf_corridor_turn_filter = bool(
        not verifier_accepts_stop
        and not recent_blocked
        and bool(route_actions)
        and (
            (not route_complete and route_index > 0)
            or (route_complete and motion_count >= cfg.min_low_conf_route_complete_motion_actions)
        )
        and decision.action in {VLNAction.TURN_LEFT, VLNAction.TURN_RIGHT}
        and recent_forward_streak >= cfg.min_low_conf_corridor_forward_streak
        and target_ratio <= cfg.max_low_conf_corridor_turn_visual_ratio
        and turn_probability < cfg.max_low_conf_route_turn_probability
        and forward_probability >= cfg.min_low_conf_route_forward_probability
    )
    route_cadence_turn_filter = bool(
        not verifier_accepts_stop
        and not recent_blocked
        and bool(route_actions)
        and not route_complete
        and route_index > 0
        and decision.action in {VLNAction.TURN_LEFT, VLNAction.TURN_RIGHT}
        and motion_count >= cfg.min_route_cadence_motion_actions
        and recent_turn_count >= cfg.min_route_cadence_recent_turns
        and recent_forward_streak <= cfg.max_route_cadence_forward_streak
        and target_ratio <= cfg.max_route_cadence_visual_ratio
    )

    suppress_premature_stop = bool(
        decision.action is VLNAction.STOP
        and not verifier_accepts_stop
        and (recent_blocked or motion_count < cfg.min_motion_actions or not visible)
    )

    audit = {
        "ppsr_v4_stop_verifier_checked": True,
        "ppsr_v4_stop_verifier_accept": verifier_accepts_stop,
        "ppsr_v4_stop_verifier_applied": False,
        "ppsr_v4_premature_stop_suppressed": suppress_premature_stop,
        "ppsr_v4_late_stop_recovery_applied": late_recovery_stop and verifier_accepts_stop,
        "ppsr_v4_calibrated_route_visual_stop": calibrated_route_visual_stop,
        "ppsr_v4_late_calibrated_route_visual_stop": late_calibrated_route_visual_stop,
        "ppsr_v4_late_route_low_visual_stop": late_route_low_visual_stop,
        "ppsr_v4_ultra_late_route_visual_stop": ultra_late_route_visual_stop,
        "ppsr_v4_ultra_late_forward_streak_visual_stop": ultra_late_forward_streak_visual_stop,
        "ppsr_v4_recent_visual_memory_stop": recent_visual_memory_stop,
        "ppsr_v4_late_forward_visual_memory_stop": late_forward_visual_memory_stop,
        "ppsr_v4_low_ratio_full_bbox_memory_stop": low_ratio_full_bbox_memory_stop,
        "ppsr_v4_borderline_visual_stop_holdoff": borderline_visual_stop_holdoff,
        "ppsr_v4_late_route_forward_stabilizer_applied": late_route_forward_stabilizer,
        "ppsr_v4_late_forward_momentum_lock_applied": late_forward_momentum_lock,
        "ppsr_v4_low_conf_corridor_turn_filter_applied": low_conf_corridor_turn_filter,
        "ppsr_v4_route_cadence_turn_filter_applied": route_cadence_turn_filter,
        "ppsr_v4_turn_probability": turn_probability,
        "ppsr_v4_forward_probability": forward_probability,
        "ppsr_v4_stop_probability": stop_probability,
        "ppsr_v4_visual_goal_tracker_applied": visual_goal_tracker_applied,
        "ppsr_v4_visual_goal_tracker_action": visual_goal_action.value if visual_goal_action is not None else "",
        "ppsr_v4_visual_stop_cue_detected": visible,
        "ppsr_v4_visual_target_ratio": target_ratio,
        "ppsr_v4_history_best_visual_target_ratio": history_best_ratio,
        "ppsr_v4_history_best_visual_bbox_area_ratio": history_best_bbox_area,
        "ppsr_v4_history_recent_strong_visual": history_recent_strong_visual,
        "ppsr_v4_history_recent_forward_memory_visual": history_recent_forward_memory_visual,
        "ppsr_v4_visual_bbox_area_ratio": bbox_area,
        "ppsr_v4_visual_centered": centered,
        "ppsr_v4_visual_growth_ratio": visual_growth,
        "ppsr_v4_navid_stop_hint": navid_stop_hint,
        "ppsr_v4_model_stop_hint": model_stop_hint,
        "ppsr_v4_route_complete": route_complete,
        "ppsr_v4_route_near_complete": route_near_complete,
        "ppsr_v4_route_mode": route_mode,
        "ppsr_v4_route_index": route_index,
        "ppsr_v4_route_length": len(route_actions),
        "ppsr_v4_motion_action_count": motion_count,
        "ppsr_v4_recent_turn_count": recent_turn_count,
        "ppsr_v4_recent_forward_streak": recent_forward_streak,
        "ppsr_v4_recent_recovery": recent_recovery,
        "ppsr_v4_recent_blocked": recent_blocked,
        "ppsr_v4_config": asdict(cfg),
        "privileged_policy_usage": False,
    }

    if borderline_visual_stop_holdoff:
        return ActionDecision(
            action=VLNAction.FORWARD,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_borderline_visual_stop_holdoff",
            confidence=max(float(decision.confidence or 0.0), 0.81),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit},
        )

    if verifier_accepts_stop and decision.action is not VLNAction.STOP:
        return ActionDecision(
            action=VLNAction.STOP,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_policy_stop_guard",
            confidence=max(float(decision.confidence or 0.0), 0.88),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit, "ppsr_v4_stop_verifier_applied": True},
        )

    if suppress_premature_stop:
        fallback = VLNAction.BACKOFF if recent_blocked else VLNAction.FORWARD
        return ActionDecision(
            action=fallback,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_premature_stop_filter_guard",
            confidence=min(float(decision.confidence or 0.0), 0.55),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit},
        )

    if late_route_forward_stabilizer:
        return ActionDecision(
            action=VLNAction.FORWARD,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_late_route_forward_stabilizer",
            confidence=max(float(decision.confidence or 0.0), 0.80),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit},
        )

    if late_forward_momentum_lock:
        return ActionDecision(
            action=VLNAction.FORWARD,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_late_forward_momentum_lock",
            confidence=max(float(decision.confidence or 0.0), 0.84),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit},
        )

    if route_cadence_turn_filter:
        return ActionDecision(
            action=VLNAction.FORWARD,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_route_cadence_turn_filter",
            confidence=max(float(decision.confidence or 0.0), 0.67),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit},
        )

    if low_conf_corridor_turn_filter:
        return ActionDecision(
            action=VLNAction.FORWARD,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_low_conf_corridor_turn_filter",
            confidence=min(max(float(decision.confidence or 0.0), 0.62), 0.78),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit},
        )

    if visual_goal_tracker_applied and visual_goal_action is not None:
        return ActionDecision(
            action=visual_goal_action,
            raw_output={
                "base_raw_output": decision.raw_output,
                "base_action": decision.action.value,
                "ppsr_v4_stop_verifier": audit,
            },
            source=f"{decision.source}+ppsr_v4_visual_goal_tracker",
            confidence=max(float(decision.confidence or 0.0), 0.82),
            magnitude=decision.magnitude,
            metadata={**metadata, **audit},
        )

    return ActionDecision(
        action=decision.action,
        raw_output=decision.raw_output,
        source=decision.source,
        confidence=decision.confidence,
        magnitude=decision.magnitude,
        metadata={**metadata, **audit},
    )


def _route_actions_for_instruction(instruction: str) -> tuple[list[VLNAction], str]:
    explicit = parse_route_actions_from_instruction(instruction)
    if explicit:
        return explicit, "explicit_counts"
    turns = _natural_turn_route_actions(instruction)
    if turns:
        return turns, "natural_turns"
    return [], "none"


def _natural_turn_route_actions(instruction: str) -> list[VLNAction]:
    actions: list[VLNAction] = []
    for match in re.finditer(r"\bturn\s+(left|right)\b", instruction.lower()):
        actions.append(VLNAction.TURN_LEFT if match.group(1) == "left" else VLNAction.TURN_RIGHT)
    return actions


def _visual_goal_tracking_action(
    *,
    center_x: Any,
    cfg: PpsrV4StopVerifierConfig,
    route_complete: bool,
    visible: bool,
    recent_blocked: bool,
    motion_count: int,
    target_ratio: float,
    bbox_area: float,
    visual_growth: float,
    verifier_accepts_stop: bool,
) -> VLNAction | None:
    if (
        verifier_accepts_stop
        or recent_blocked
        or not route_complete
        or not visible
        or motion_count < cfg.min_visual_goal_tracking_motion_actions
        or target_ratio < cfg.visual_goal_tracking_ratio
        or target_ratio > cfg.max_visual_goal_tracking_ratio
        or bbox_area < cfg.min_visual_goal_tracking_bbox_area
        or visual_growth > cfg.max_visual_goal_tracking_growth_ratio
        or not isinstance(center_x, (float, int))
    ):
        return None
    cx = float(center_x)
    if cx < 0.5 - cfg.visual_goal_center_deadband:
        return VLNAction.TURN_LEFT
    if cx > 0.5 + cfg.visual_goal_center_deadband:
        return VLNAction.TURN_RIGHT
    return VLNAction.FORWARD


def _float_probability(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_v4_stop_audit(
    *,
    method_summary: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    leakage_audit: dict[str, Any],
) -> dict[str, Any]:
    summary = next((row for row in method_summary if row.get("method") == PPSR_V4_METHOD_NAME), {})
    episodes = [row for row in episode_rows if row.get("method") == PPSR_V4_METHOD_NAME]
    decisions = [row for row in decision_rows if row.get("method") == PPSR_V4_METHOD_NAME]
    verifier_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_stop_verifier_checked") is True
    ]
    applied_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_stop_verifier_applied") is True
    ]
    suppressed_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_premature_stop_suppressed") is True
    ]
    late_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_late_stop_recovery_applied") is True
    ]
    calibrated_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_calibrated_route_visual_stop") is True
    ]
    late_calibrated_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_late_calibrated_route_visual_stop") is True
    ]
    late_low_visual_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_late_route_low_visual_stop") is True
    ]
    ultra_late_visual_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_ultra_late_route_visual_stop") is True
    ]
    ultra_late_forward_streak_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_ultra_late_forward_streak_visual_stop") is True
    ]
    recent_visual_memory_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_recent_visual_memory_stop") is True
    ]
    late_forward_visual_memory_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_late_forward_visual_memory_stop") is True
    ]
    borderline_holdoff_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_borderline_visual_stop_holdoff") is True
    ]
    late_route_forward_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_late_route_forward_stabilizer_applied") is True
    ]
    late_forward_momentum_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_late_forward_momentum_lock_applied") is True
    ]
    low_conf_corridor_turn_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_low_conf_corridor_turn_filter_applied") is True
    ]
    route_cadence_turn_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_route_cadence_turn_filter_applied") is True
    ]
    visual_goal_rows = [
        row
        for row in decisions
        if _metadata_value(row, "ppsr_v4_visual_goal_tracker_applied") is True
    ]
    return {
        "method": PPSR_V4_METHOD_NAME,
        "total_episodes": int(summary.get("total_episodes", len(episodes) or 0)),
        "safe_success_rate": float(summary.get("safe_success_rate", 0.0)),
        "unsafe_violation_count": int(summary.get("unsafe_violation_count", 0)),
        "unsafe_violation_rate": float(summary.get("unsafe_violation_rate", 0.0)),
        "stop_precision": float(summary.get("stop_precision", 0.0)),
        "stop_recall": float(summary.get("stop_recall", 0.0)),
        "premature_stop_rate": float(summary.get("premature_stop_rate", 0.0)),
        "late_stop_rate": float(summary.get("late_stop_rate", 0.0)),
        "premature_stop_count": sum(1 for row in episodes if row.get("stop_failure_type") == "premature_stop"),
        "late_stop_count": sum(1 for row in episodes if row.get("stop_failure_type") == "late_stop"),
        "missing_policy_stop_count": sum(1 for row in episodes if row.get("failure_reason") == "missing_policy_stop"),
        "stop_verifier_checked_count": len(verifier_rows),
        "stop_verifier_applied_count": len(applied_rows),
        "premature_stop_suppressed_count": len(suppressed_rows),
        "late_stop_recovery_applied_count": len(late_rows),
        "calibrated_route_visual_stop_count": len(calibrated_rows),
        "late_calibrated_route_visual_stop_count": len(late_calibrated_rows),
        "late_route_low_visual_stop_count": len(late_low_visual_rows),
        "ultra_late_route_visual_stop_count": len(ultra_late_visual_rows),
        "ultra_late_forward_streak_visual_stop_count": len(ultra_late_forward_streak_rows),
        "recent_visual_memory_stop_count": len(recent_visual_memory_rows),
        "late_forward_visual_memory_stop_count": len(late_forward_visual_memory_rows),
        "borderline_visual_stop_holdoff_count": len(borderline_holdoff_rows),
        "late_route_forward_stabilizer_count": len(late_route_forward_rows),
        "late_forward_momentum_lock_count": len(late_forward_momentum_rows),
        "low_conf_corridor_turn_filter_count": len(low_conf_corridor_turn_rows),
        "route_cadence_turn_filter_count": len(route_cadence_turn_rows),
        "visual_goal_tracker_applied_count": len(visual_goal_rows),
        "visual_stop_cue_detected_count": sum(
            1 for row in verifier_rows if _metadata_value(row, "ppsr_v4_visual_stop_cue_detected") is True
        ),
        "navid_stop_hint_count": sum(1 for row in verifier_rows if _metadata_value(row, "ppsr_v4_navid_stop_hint") is True),
        "route_complete_count": sum(1 for row in verifier_rows if _metadata_value(row, "ppsr_v4_route_complete") is True),
        "safety_stop_success_leakage_count": int(summary.get("safety_stop_success_leakage_count", 0)),
        "recovery_stop_success_leakage_count": int(summary.get("recovery_stop_success_leakage_count", 0)),
        "privileged_online_leakage": bool(leakage_audit.get("privileged_online_leakage", False)),
        "forbidden_online_findings": leakage_audit.get("forbidden_online_findings", []),
    }


def detect_privileged_online_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        for field in ("policy_metadata", "policy_metadata_json", "v4_policy_metadata"):
            metadata = row.get(field)
            if metadata is None or metadata == "":
                continue
            parsed = _parse_json_if_needed(metadata)
            _collect_forbidden_findings(
                parsed,
                findings=findings,
                row_idx=row_idx,
                root_field=field,
                path=(),
            )
    bool_by_key = {str(item["key"]) for item in findings}
    return {
        "used_goal_distance_for_online_replan": bool({"goal_distance", "distance_to_goal", "evaluator_distance_to_goal"}.intersection(bool_by_key)),
        "used_shortest_path_for_online_replan": bool({"shortest_path", "shortest_path_remaining", "evaluator_only_shortest_path_remaining"}.intersection(bool_by_key)),
        "used_a_star_for_online_replan": bool({"astar", "a_star", "astar_path", "a_star_path", "test_time_astar_used"}.intersection(bool_by_key)),
        "used_oracle_waypoint_for_online_replan": bool({"oracle_waypoint", "waypoint"}.intersection(bool_by_key)),
        "used_evaluator_success_for_online_replan": bool({"evaluator_success", "success"}.intersection(bool_by_key)),
        "used_map_cell_progress_for_online_replan": bool({"map_cell_progress", "cell_progress"}.intersection(bool_by_key)),
        "privileged_online_leakage": bool(findings),
        "forbidden_online_findings": findings,
    }


def _safe_wall_painting_evidence(image_path: str | Path) -> dict[str, Any]:
    try:
        return detect_gallery_wall_painting(image_path).to_dict()
    except Exception as exc:
        return {
            "red_ratio": 0.0,
            "bbox_area_ratio": 0.0,
            "center_x": None,
            "center_y": None,
            "visible": False,
            "target_kind": "gallery_wall_painting",
            "target_ratio": 0.0,
            "error": str(exc),
        }


def _evidence_centered(evidence: dict[str, Any], cfg: PpsrV4StopVerifierConfig) -> bool:
    center_x = evidence.get("center_x")
    return isinstance(center_x, (float, int)) and cfg.centered_min_x <= float(center_x) <= cfg.centered_max_x


def _route_progress(route: list[VLNAction], previous_actions: list[str], *, turn_only: bool = False) -> int:
    progress = 0
    for action_value in previous_actions:
        if action_value == VLNAction.STOP.value:
            break
        if turn_only and action_value not in {VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value}:
            continue
        if action_value == VLNAction.BACKOFF.value:
            continue
        if progress < len(route) and action_value == route[progress].value:
            progress += 1
    return progress


def _motion_action_count(actions: list[str]) -> int:
    return sum(
        1
        for action in actions
        if action in {VLNAction.FORWARD.value, VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value, VLNAction.BACKOFF.value}
    )


def _recent_count(actions: list[str], value: str) -> int:
    count = 0
    for action in reversed(actions):
        if action != value:
            break
        count += 1
    return count


def _recent_turn_count(actions: list[str]) -> int:
    return sum(1 for action in actions if action in {VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value})


def _metadata_value(row: dict[str, Any], key: str) -> Any:
    metadata = _parse_json_if_needed(row.get("policy_metadata") or row.get("policy_metadata_json") or {})
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _parse_json_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_FORBIDDEN_ALWAYS = {
    "goal_distance",
    "distance_to_goal",
    "evaluator_distance_to_goal",
    "shortest_path",
    "shortest_path_remaining",
    "evaluator_only_shortest_path_remaining",
    "astar",
    "a_star",
    "astar_path",
    "a_star_path",
    "oracle_waypoint",
    "waypoint",
    "evaluator_success",
    "map_cell_progress",
    "cell_progress",
    "goal_xy",
    "goal",
    "path_cells",
    "path",
    "map",
    "occupancy",
    "cell",
    "pose",
}

_FORBIDDEN_TRUTHY = {
    "test_time_astar_used",
    "test_time_map_pose_goal_path_used",
    "used_goal_distance_for_online_replan",
    "used_shortest_path_for_online_replan",
    "used_a_star_for_online_replan",
    "used_oracle_waypoint_for_online_replan",
    "used_evaluator_success_for_online_replan",
    "used_map_cell_progress_for_online_replan",
}


def _collect_forbidden_findings(
    value: Any,
    *,
    findings: list[dict[str, Any]],
    row_idx: int,
    root_field: str,
    path: tuple[str, ...],
) -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_path = (*path, key)
            if normalized in _FORBIDDEN_ALWAYS and _meaningful_forbidden_value(child):
                findings.append(
                    {
                        "row_idx": row_idx,
                        "root_field": root_field,
                        "path": ".".join(child_path),
                        "key": normalized,
                    }
                )
            elif normalized in _FORBIDDEN_TRUTHY and bool(child):
                findings.append(
                    {
                        "row_idx": row_idx,
                        "root_field": root_field,
                        "path": ".".join(child_path),
                        "key": normalized,
                    }
                )
            _collect_forbidden_findings(child, findings=findings, row_idx=row_idx, root_field=root_field, path=child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _collect_forbidden_findings(child, findings=findings, row_idx=row_idx, root_field=root_field, path=(*path, str(idx)))


def _meaningful_forbidden_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True
