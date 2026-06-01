"""Adaptive recovery and segmentation helpers for Phase 5 online runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from gear_sonic.casa.skills import GestureSkill, PassiveSkill, TurnSkill, WalkSkill


@dataclass(frozen=True)
class RecoveryPlan:
    skill: Any
    policy: str
    action: str
    reason: str
    retry_count: int = 0


def choose_fallback_policy(method_default: str, requested: str) -> str:
    if requested == "auto":
        return method_default
    return requested


def plan_recovery(
    candidate_skill: Any,
    *,
    fallback_policy: str,
    fallback_duration: float,
    max_retry_count: int = 0,
    reject_reason: str = "",
    raw_risk: float | None = None,
    hard_contract_fixed_reject: bool = False,
) -> RecoveryPlan:
    duration = max(0.05, float(fallback_duration))
    if fallback_policy == "stop":
        return RecoveryPlan(
            skill=PassiveSkill(duration=duration, mode="stop"),
            policy=fallback_policy,
            action="stop",
            reason="generic_stop_fallback",
        )
    if fallback_policy not in {"adaptive", "adaptive_retry"}:
        raise ValueError(f"Unknown fallback policy: {fallback_policy}")

    risk_text = "unknown risk" if raw_risk is None else f"risk={raw_risk:.4f}"
    context = f"{reject_reason or 'gate_reject'}; {risk_text}; hard_fixed={int(bool(hard_contract_fixed_reject))}"
    retry_count = max(0, int(max_retry_count)) if fallback_policy == "adaptive_retry" else 0

    if isinstance(candidate_skill, WalkSkill):
        speed = candidate_skill.speed if candidate_skill.speed > 0 else 0.35
        return RecoveryPlan(
            skill=WalkSkill(
                vx=0.5 * candidate_skill.vx,
                vy=0.5 * candidate_skill.vy,
                facing_yaw_deg=candidate_skill.facing_yaw_deg,
                duration=min(candidate_skill.duration, duration),
                speed=max(0.05, 0.5 * speed),
                height=candidate_skill.height,
            ),
            policy=fallback_policy,
            action="short_walk_recovery",
            reason=f"shorten and slow walk after {context}",
            retry_count=retry_count,
        )
    if isinstance(candidate_skill, TurnSkill):
        yaw = max(-30.0, min(30.0, candidate_skill.face_yaw_deg))
        return RecoveryPlan(
            skill=TurnSkill(face_yaw_deg=yaw, duration=min(candidate_skill.duration, duration)),
            policy=fallback_policy,
            action="small_turn_recovery",
            reason=f"split/reduce turn after {context}",
            retry_count=retry_count,
        )
    if isinstance(candidate_skill, GestureSkill):
        side = "left" if candidate_skill.side == "both" else candidate_skill.side
        return RecoveryPlan(
            skill=GestureSkill(
                amplitude=max(0.01, 0.5 * candidate_skill.amplitude),
                frequency=max(0.05, 0.5 * candidate_skill.frequency),
                side=side,
                duration=min(candidate_skill.duration, duration),
            ),
            policy=fallback_policy,
            action="low_amplitude_gesture_recovery",
            reason=f"reduce gesture amplitude/frequency after {context}",
            retry_count=retry_count,
        )
    if isinstance(candidate_skill, PassiveSkill):
        return RecoveryPlan(
            skill=PassiveSkill(duration=min(candidate_skill.duration, duration), mode="stop"),
            policy=fallback_policy,
            action="passive_stabilize_recheck",
            reason=f"passive is treated as rechecked stabilization after {context}",
            retry_count=retry_count,
        )
    return RecoveryPlan(
        skill=PassiveSkill(duration=duration, mode="stop"),
        policy=fallback_policy,
        action="unknown_skill_stop",
        reason=f"unknown skill fallback after {context}",
        retry_count=retry_count,
    )


def rewrite_skill_for_retry(
    candidate_skill: Any,
    *,
    attempt_index: int,
    max_retry_count: int,
    fallback_duration: float,
    scene_props: dict[str, Any] | None = None,
) -> Any:
    """Return a safer retry candidate after a recovery execution.

    The rewrite is intentionally monotone with respect to the original command:
    duration and commanded magnitudes never increase, while absolute target
    facing for turn skills is preserved.
    """

    del max_retry_count
    attempt = max(0, int(attempt_index))
    scale = max(0.20, 0.70 - 0.15 * attempt)
    duration = _shortened_duration(candidate_skill, fallback_duration, scale)
    scene_props = scene_props or {}

    if isinstance(candidate_skill, WalkSkill):
        vx = _scaled_signed(candidate_skill.vx, scale)
        vy = _scaled_signed(candidate_skill.vy, scale)
        if _scene_prefers_backing_away(scene_props):
            vx = -vx
            vy = -vy
        speed = candidate_skill.speed
        if speed > 0:
            speed = min(speed, max(0.05, speed * scale))
        return WalkSkill(
            vx=vx,
            vy=vy,
            facing_yaw_deg=candidate_skill.facing_yaw_deg,
            duration=duration,
            speed=speed,
            height=candidate_skill.height,
        )
    if isinstance(candidate_skill, TurnSkill):
        return TurnSkill(face_yaw_deg=candidate_skill.face_yaw_deg, duration=duration)
    if isinstance(candidate_skill, GestureSkill):
        amplitude = min(candidate_skill.amplitude, max(0.01, candidate_skill.amplitude * scale))
        frequency = min(candidate_skill.frequency, max(0.05, candidate_skill.frequency * scale))
        side = "left" if candidate_skill.side == "both" else candidate_skill.side
        return GestureSkill(
            amplitude=amplitude,
            frequency=frequency,
            side=side,
            duration=duration,
        )
    if isinstance(candidate_skill, PassiveSkill):
        return PassiveSkill(duration=duration, mode="stop")
    return PassiveSkill(duration=max(0.05, min(float(fallback_duration), 0.25)), mode="stop")


def split_skill(skill: Any, *, max_segment_duration: float) -> list[Any]:
    max_duration = float(max_segment_duration)
    if not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError("--max-segment-duration must be finite and > 0")
    duration = float(getattr(skill, "duration", 0.0))
    if duration <= max_duration:
        return [skill]
    count = int(math.ceil(duration / max_duration))
    segment_duration = duration / count
    if isinstance(skill, WalkSkill):
        return [
            WalkSkill(
                vx=skill.vx,
                vy=skill.vy,
                facing_yaw_deg=skill.facing_yaw_deg,
                duration=segment_duration,
                speed=skill.speed,
                height=skill.height,
            )
            for _ in range(count)
        ]
    if isinstance(skill, TurnSkill):
        return [TurnSkill(face_yaw_deg=skill.face_yaw_deg, duration=segment_duration) for _ in range(count)]
    if isinstance(skill, GestureSkill):
        return [
            GestureSkill(
                amplitude=skill.amplitude,
                frequency=skill.frequency,
                side=skill.side,
                duration=segment_duration,
            )
            for _ in range(count)
        ]
    if isinstance(skill, PassiveSkill):
        return [PassiveSkill(duration=segment_duration, mode=skill.mode) for _ in range(count)]
    return [skill]


def _shortened_duration(skill: Any, fallback_duration: float, scale: float) -> float:
    original = max(0.05, float(getattr(skill, "duration", fallback_duration)))
    bounded = min(original, max(0.05, float(fallback_duration)))
    return max(0.05, min(original, bounded * scale))


def _scaled_signed(value: float, scale: float) -> float:
    return math.copysign(min(abs(value), abs(value) * scale), value)


def _scene_prefers_backing_away(scene_props: dict[str, Any]) -> bool:
    text = " ".join(str(scene_props.get(key, "")).lower() for key in ("target_bucket", "scene_family"))
    return "collision" in text or "close" in text or "fall" in text
