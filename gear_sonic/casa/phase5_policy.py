"""Online Phase 5 policy helpers for CASA candidate methods."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from gear_sonic.casa.phase5 import METHOD_DISPLAY, METHOD_ORDER

ONLINE_CANDIDATE_METHODS = (
    "casa_a_hard_or_per_skill",
    "casa_a_recovery_per_skill",
    "casa_a_hard_or_recovery",
    "casa_a_receding_recovery",
)
ONLINE_METHOD_ORDER = (*METHOD_ORDER, *ONLINE_CANDIDATE_METHODS)
ONLINE_METHOD_DISPLAY = {
    **METHOD_DISPLAY,
    "casa_a_hard_or_per_skill": "SONIC + CASA-A hard-OR per-skill",
    "casa_a_recovery_per_skill": "SONIC + CASA-A adaptive recovery",
    "casa_a_hard_or_recovery": "SONIC + CASA-A hard-OR adaptive recovery",
    "casa_a_receding_recovery": "SONIC + CASA-A receding recovery",
}


@dataclass(frozen=True)
class OnlineDecision:
    method: str
    reject: bool
    threshold: float | None
    reject_reason: str
    risk_margin: float | None


@dataclass(frozen=True)
class MethodBehavior:
    fallback_policy: str
    hard_or_casa: bool
    segment_long_skills: bool


def method_display(method: str) -> str:
    return ONLINE_METHOD_DISPLAY.get(method, method)


def method_behavior(method: str) -> MethodBehavior:
    if method in {"casa_a_recovery_per_skill", "casa_a_hard_or_recovery"}:
        return MethodBehavior(
            fallback_policy="adaptive",
            hard_or_casa="hard_or" in method,
            segment_long_skills=False,
        )
    if method == "casa_a_receding_recovery":
        return MethodBehavior(fallback_policy="adaptive_retry", hard_or_casa=False, segment_long_skills=True)
    if method == "casa_a_hard_or_per_skill":
        return MethodBehavior(fallback_policy="stop", hard_or_casa=True, segment_long_skills=False)
    return MethodBehavior(fallback_policy="stop", hard_or_casa=False, segment_long_skills=False)


def evaluate_online_method(
    *,
    method: str,
    skill_name: str,
    raw_risk: float,
    hard_contract_fixed_reject: bool,
    thresholds: dict[str, Any],
) -> OnlineDecision:
    threshold: float | None
    if method == "sonic_only":
        return OnlineDecision(method, False, None, "sonic_only_allow", None)
    if method == "hard_contract":
        reject = bool(hard_contract_fixed_reject)
        return OnlineDecision(method, reject, None, "hard_contract_fixed_reject" if reject else "allow", None)
    if method == "raw_critic_0p5":
        threshold = 0.5
        reject = raw_risk >= threshold
        return OnlineDecision(
            method,
            reject,
            threshold,
            "raw_critic_threshold" if reject else "allow",
            raw_risk - threshold,
        )
    if method == "global_conformal":
        threshold = float(thresholds["global"])
        reject = raw_risk >= threshold
        return OnlineDecision(
            method,
            reject,
            threshold,
            "global_conformal_threshold" if reject else "allow",
            raw_risk - threshold,
        )
    if method in {
        "casa_a_per_skill",
        "casa_a_recovery_per_skill",
        "casa_a_receding_recovery",
        "casa_a_hard_or_per_skill",
        "casa_a_hard_or_recovery",
    }:
        threshold = float(thresholds["per_skill"][skill_name])
        casa_reject = raw_risk >= threshold
        hard_or = method_behavior(method).hard_or_casa
        reject = casa_reject or (hard_or and bool(hard_contract_fixed_reject))
        if bool(hard_contract_fixed_reject) and hard_or:
            reason = "hard_contract_or_casa_threshold"
        elif casa_reject:
            reason = "per_skill_conformal_threshold"
        else:
            reason = "allow"
        return OnlineDecision(method, reject, threshold, reason, raw_risk - threshold)
    raise ValueError(f"Unknown Phase 5 online method: {method}")


def parse_threshold_scale_by_skill(raw: str) -> dict[str, float]:
    if not raw.strip():
        return {}
    output: dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid threshold scale item {item!r}; expected skill=scale")
        skill, value = [part.strip() for part in item.split("=", 1)]
        scale = float(value)
        if not skill:
            raise ValueError(f"Invalid threshold scale item {item!r}; skill must not be empty")
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"Invalid threshold scale for {skill!r}: {value!r}")
        output[skill] = scale
    return output


def scale_thresholds(
    thresholds: dict[str, Any],
    *,
    global_scale: float = 1.0,
    by_skill: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(global_scale) or global_scale < 0:
        raise ValueError("--threshold-scale-global must be finite and >= 0")
    by_skill = by_skill or {}
    scaled = {
        "global": float(thresholds["global"]) * global_scale,
        "per_skill": {},
    }
    for skill, threshold in thresholds.get("per_skill", {}).items():
        scaled["per_skill"][skill] = float(threshold) * by_skill.get(skill, global_scale)
    return scaled
