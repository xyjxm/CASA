"""First-slice CASA-adapted SOTA baseline implementations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from gear_sonic.casa.baselines.sota_adapters.api import (
    GateDecision,
    SotaBaselineAdapter,
    SotaDecisionContext,
    finite_bool,
    finite_float,
)
from gear_sonic.casa.phase5 import MAIN_SKILLS, split_role


class PCBFAdapted(SotaBaselineAdapter):
    name = "pcbf_adapted"
    source_method = (
        "Reinforcement Learning with Probabilistically Safe Control Barrier Functions "
        "for Ramp Merging"
    )
    implementation_fidelity = "paper_faithful_proxy"
    mode = "gate"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thresholds = _default_per_skill(0.95)
        self.uncertainty_by_skill = _default_per_skill(self.config.uncertainty_floor)

    def calibrate(
        self,
        calibration_data: Sequence[Mapping[str, Any]],
        config: Any | None = None,
    ) -> None:
        super().calibrate(calibration_data, config)
        rows = [row for row in calibration_data if _has_label(row)]
        self.uncertainty_by_skill = _per_skill_value(
            rows,
            lambda skill_rows: _quantile(
                [
                    abs(_label(row) - max(_raw(row), _hard(row)))
                    for row in skill_rows
                ],
                1.0 - self.config.epsilon,
                self.config.uncertainty_floor,
            ),
            self.config.uncertainty_floor,
        )
        self.thresholds = _per_skill_value(
            rows,
            lambda skill_rows: _threshold_for_fnr(
                [_pcbf_row_score(row, self.uncertainty_by_skill, self.config.chance_z) for row in skill_rows],
                [_label(row) for row in skill_rows],
                self.config.alpha,
                default=0.95,
            ),
            0.95,
        )
        self.calibration_summary.update(
            {
                "thresholds": self.thresholds,
                "uncertainty_by_skill": self.uncertainty_by_skill,
                "epsilon": self.config.epsilon,
                "chance_z": self.config.chance_z,
                "score": "max(raw_critic_risk, hard_contract_score) + z_epsilon * calibrated_uncertainty",
            }
        )

    def decide(self, context: SotaDecisionContext) -> GateDecision:
        uncertainty = float(self.uncertainty_by_skill.get(context.skill_name, self.config.uncertainty_floor))
        base_risk = max(float(context.raw_critic_risk), float(context.hard_contract_score))
        risk_score = base_risk + float(self.config.chance_z) * uncertainty
        threshold = float(self.thresholds.get(context.skill_name, self.thresholds["global"]))
        allow = not bool(context.hard_contract_fixed_reject) and risk_score < threshold
        reason = "allow"
        if context.hard_contract_fixed_reject:
            reason = "hard_contract_fixed_reject"
        elif not allow:
            reason = "pcbf_chance_constraint"
        return GateDecision(
            method_name=self.name,
            source_method=self.source_method,
            implementation_fidelity=self.implementation_fidelity,
            mode=self.mode,
            allow=allow,
            risk_score=risk_score,
            threshold=threshold,
            fallback_mode=None if allow else self.config.fallback_mode,
            reject_reason=reason,
            solver_status="chance_constraint_satisfied" if allow else "chance_constraint_violated",
            diagnostics={
                "base_risk": base_risk,
                "uncertainty_margin": uncertainty,
                "chance_epsilon": self.config.epsilon,
                "chance_z": self.config.chance_z,
                "threshold_source": "calibration_split_per_skill",
            },
        )


class CRCCBFAdapted(SotaBaselineAdapter):
    name = "crc_cbf_adapted"
    source_method = (
        "Safe Probabilistic Planning for Human-Robot Interaction using "
        "Conformal Risk Control"
    )
    implementation_fidelity = "paper_faithful_proxy"
    mode = "gate"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_by_skill = _default_per_skill(0.05)
        self.risk_budget = self.config.alpha

    def calibrate(
        self,
        calibration_data: Sequence[Mapping[str, Any]],
        config: Any | None = None,
    ) -> None:
        super().calibrate(calibration_data, config)
        rows = [row for row in calibration_data if _has_label(row)]
        self.lambda_by_skill = _per_skill_value(
            rows,
            lambda skill_rows: _quantile(
                [max(0.0, _label(row) - max(_raw(row), _hard(row))) for row in skill_rows],
                1.0 - self.config.alpha,
                0.05,
            ),
            0.05,
        )
        self.risk_budget = self.config.alpha
        self.calibration_summary.update(
            {
                "lambda_by_skill": self.lambda_by_skill,
                "risk_budget": self.risk_budget,
                "score": "max(0, max(raw_critic_risk, hard_contract_score) + conformal_lambda - 1)",
                "alpha": self.config.alpha,
            }
        )

    def decide(self, context: SotaDecisionContext) -> GateDecision:
        lambda_hat = float(self.lambda_by_skill.get(context.skill_name, self.lambda_by_skill["global"]))
        base_risk = max(float(context.raw_critic_risk), float(context.hard_contract_score))
        calibrated_risk = 1.0 if context.hard_contract_fixed_reject else max(0.0, base_risk + lambda_hat - 1.0)
        threshold = float(self.risk_budget)
        allow = calibrated_risk <= threshold and not bool(context.hard_contract_fixed_reject)
        reason = "allow"
        if context.hard_contract_fixed_reject:
            reason = "hard_contract_fixed_reject"
        elif not allow:
            reason = "crc_cbf_calibrated_risk_budget"
        return GateDecision(
            method_name=self.name,
            source_method=self.source_method,
            implementation_fidelity=self.implementation_fidelity,
            mode=self.mode,
            allow=allow,
            risk_score=calibrated_risk,
            threshold=threshold,
            fallback_mode=None if allow else self.config.fallback_mode,
            reject_reason=reason,
            solver_status="crc_risk_budget_satisfied" if allow else "crc_risk_budget_violated",
            diagnostics={
                "base_risk": base_risk,
                "conformal_lambda": lambda_hat,
                "risk_budget_alpha": self.config.alpha,
                "threshold_source": "calibration_split_conformal_residual",
            },
        )


class MPCCBFHumanoidAdapted(SotaBaselineAdapter):
    name = "mpc_cbf_humanoid_adapted"
    source_method = (
        "Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson "
        "Safety Functions to CBF Constrained MPC"
    )
    implementation_fidelity = "lightweight_proxy"
    mode = "gate"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thresholds = _default_per_skill(-0.05)
        self.safety_buffer_by_skill = _default_per_skill(0.05)

    def calibrate(
        self,
        calibration_data: Sequence[Mapping[str, Any]],
        config: Any | None = None,
    ) -> None:
        super().calibrate(calibration_data, config)
        rows = [row for row in calibration_data if _has_label(row)]
        self.thresholds = _per_skill_value(
            rows,
            lambda skill_rows: _threshold_for_fnr(
                [_mpc_row_risk(row) for row in skill_rows],
                [_label(row) for row in skill_rows],
                self.config.alpha,
                default=-0.05,
            ),
            -0.05,
        )
        self.safety_buffer_by_skill = {
            skill: -float(value)
            for skill, value in self.thresholds.items()
        }
        self.calibration_summary.update(
            {
                "thresholds": self.thresholds,
                "safety_buffer_by_skill": self.safety_buffer_by_skill,
                "score": "- predicted_min_cbf_margin under reduced-order skill-level rollout",
                "proxy_scope": "gate_only_reduced_order_mpc_cbf_no_poisson_solver",
            }
        )

    def decide(self, context: SotaDecisionContext) -> GateDecision:
        margin, diagnostics = _predict_skill_margin(context, self.config.mpc_horizon_s)
        risk_score = -margin
        threshold = float(self.thresholds.get(context.skill_name, self.thresholds["global"]))
        allow = not bool(context.hard_contract_fixed_reject) and risk_score < threshold
        reason = "allow"
        if context.hard_contract_fixed_reject:
            reason = "hard_contract_fixed_reject"
        elif not allow:
            reason = "mpc_cbf_skill_horizon_infeasible"
        diagnostics.update(
            {
                "mpc_horizon_s": self.config.mpc_horizon_s,
                "mpc_step_s": self.config.mpc_step_s,
                "threshold_source": "calibration_split_skill_level_feasibility",
                "proxy_scope": "reduced_order_skill_level_feasibility_surrogate",
            }
        )
        return GateDecision(
            method_name=self.name,
            source_method=self.source_method,
            implementation_fidelity=self.implementation_fidelity,
            mode=self.mode,
            allow=allow,
            risk_score=risk_score,
            threshold=threshold,
            fallback_mode=None if allow else self.config.fallback_mode,
            reject_reason=reason,
            solver_status="feasible" if allow else "infeasible",
            diagnostics=diagnostics,
        )


def _default_per_skill(value: float) -> dict[str, float]:
    output = {"global": float(value)}
    output.update({skill: float(value) for skill in MAIN_SKILLS})
    return output


def _per_skill_value(
    rows: Sequence[Mapping[str, Any]],
    fn: Any,
    default: float,
) -> dict[str, float]:
    output = {"global": float(fn(rows)) if rows else float(default)}
    for skill in MAIN_SKILLS:
        skill_rows = [row for row in rows if _skill(row) == skill]
        output[skill] = float(fn(skill_rows)) if skill_rows else float(output["global"])
    return output


def _quantile(values: Sequence[float], q: float, default: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return float(default)
    q = min(1.0, max(0.0, float(q)))
    index = min(len(finite) - 1, int(math.ceil(q * len(finite))) - 1)
    return float(finite[max(0, index)])


def _threshold_for_fnr(
    scores: Sequence[float],
    labels: Sequence[int],
    alpha: float,
    *,
    default: float,
) -> float:
    positives = sorted(float(score) for score, label in zip(scores, labels) if int(label) == 1)
    if not positives:
        return float(default)
    allowed_false_negatives = max(0, min(len(positives) - 1, int(math.floor(float(alpha) * len(positives)))))
    best = positives[0]
    for threshold in sorted(set(positives)):
        false_negatives = sum(1 for score in positives if score < threshold)
        if false_negatives <= allowed_false_negatives:
            best = threshold
        else:
            break
    return float(best)


def _pcbf_row_score(
    row: Mapping[str, Any],
    uncertainty_by_skill: Mapping[str, float],
    chance_z: float,
) -> float:
    skill = _skill(row)
    uncertainty = float(uncertainty_by_skill.get(skill, uncertainty_by_skill.get("global", 0.02)))
    return max(_raw(row), _hard(row)) + float(chance_z) * uncertainty


def _mpc_row_risk(row: Mapping[str, Any]) -> float:
    margin = 1.0 - max(0.85 * _raw(row), _hard(row))
    return -margin


def _predict_skill_margin(context: SotaDecisionContext, horizon_s: float) -> tuple[float, dict[str, Any]]:
    hard_margin = 1.0 - max(float(context.hard_contract_score), 0.85 * float(context.raw_critic_risk))
    min_user = context.feature("env/min_user_distance/current", 10.0)
    min_arm_user = context.feature("env/min_arm_user_distance/current", 10.0)
    min_obstacle = context.feature("env/min_obstacle_distance/current", 10.0)
    torso_roll = abs(context.feature("robot/torso_roll/current", 0.0))
    torso_pitch = abs(context.feature("robot/torso_pitch/current", 0.0))

    distance_margin = min(
        min_user - 0.50,
        min_arm_user - 0.20,
        min_obstacle - 0.10,
        0.60 - max(torso_roll, torso_pitch),
        hard_margin,
    )
    duration = max(0.0, context.param("duration", 1.0))
    effective_horizon = min(max(0.0, horizon_s), duration)
    speed = math.hypot(context.param("vx", 0.0), context.param("vy", 0.0))
    yaw_cost = abs(context.param("facing_yaw_deg", context.param("face_yaw_deg", 0.0))) / 180.0
    gesture_cost = 0.0
    if context.skill_name == "gesture":
        gesture_cost = 0.10 + 0.10 * context.param("amplitude", 0.35) + 0.03 * context.param("frequency", 0.75)
    travel_erosion = 0.10 * speed * effective_horizon
    yaw_erosion = 0.05 * yaw_cost
    passive_credit = 0.05 if context.skill_name == "passive" else 0.0
    predicted_margin = distance_margin - travel_erosion - yaw_erosion - gesture_cost + passive_credit
    return predicted_margin, {
        "current_distance_margin": distance_margin,
        "hard_margin": hard_margin,
        "min_user_distance": min_user,
        "min_arm_user_distance": min_arm_user,
        "min_obstacle_distance": min_obstacle,
        "travel_erosion": travel_erosion,
        "yaw_erosion": yaw_erosion,
        "gesture_cost": gesture_cost,
        "passive_credit": passive_credit,
        "predicted_min_cbf_margin": predicted_margin,
    }


def _has_label(row: Mapping[str, Any]) -> bool:
    return str(row.get("label", row.get("label_int", ""))).strip() in {"0", "1", "0.0", "1.0"}


def _label(row: Mapping[str, Any]) -> int:
    return int(finite_float(row.get("label_int", row.get("label")), 0.0))


def _raw(row: Mapping[str, Any]) -> float:
    return finite_float(row.get("raw_critic_risk_float", row.get("raw_critic_risk")), 0.0)


def _hard(row: Mapping[str, Any]) -> float:
    fixed = finite_bool(row.get("hard_contract_fixed_reject_bool", row.get("hard_contract_fixed_reject")))
    return max(
        finite_float(row.get("hard_contract_score_float", row.get("hard_contract_score")), 0.0),
        1.0 if fixed else 0.0,
    )


def _skill(row: Mapping[str, Any]) -> str:
    return str(row.get("skill_name", row.get("candidate_skill", ""))).strip()


def calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
