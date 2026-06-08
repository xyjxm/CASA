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


class SafeDPAAdapted(SotaBaselineAdapter):
    name = "safedpa_adapted"
    source_method = "Safe Deep Policy Adaptation"
    implementation_fidelity = "paper_faithful_proxy"
    mode = "gate"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thresholds = _default_per_skill(0.92)
        self.adaptation_residual_by_skill = _default_per_skill(0.04)
        self.fit_residual_bias_by_skill = _default_per_skill(0.0)

    def fit(
        self,
        train_data: Sequence[Mapping[str, Any]],
        val_data: Sequence[Mapping[str, Any]],
        config: Any | None = None,
    ) -> None:
        super().fit(train_data, val_data, config)
        rows = [row for row in [*train_data, *val_data] if _has_label(row)]
        self.fit_residual_bias_by_skill = _per_skill_value(
            rows,
            lambda skill_rows: _mean(
                [
                    max(0.0, _label(row) - max(_raw(row), _hard(row)))
                    + 0.02 * _row_skill_demand(row)
                    for row in skill_rows
                ],
                0.0,
            ),
            0.0,
        )
        self.calibration_summary["fit"].update(
            {
                "residual_bias_by_skill": self.fit_residual_bias_by_skill,
                "fit_feature_source": "phase4_feature_splits_when_available_else_prediction_scores",
            }
        )

    def calibrate(
        self,
        calibration_data: Sequence[Mapping[str, Any]],
        config: Any | None = None,
    ) -> None:
        super().calibrate(calibration_data, config)
        rows = [row for row in calibration_data if _has_label(row)]
        self.adaptation_residual_by_skill = _per_skill_value(
            rows,
            lambda skill_rows: _quantile(
                [_safedpa_adaptation_residual(row) for row in skill_rows],
                1.0 - self.config.epsilon,
                0.04,
            ),
            0.04,
        )
        for skill, bias in self.fit_residual_bias_by_skill.items():
            self.adaptation_residual_by_skill[skill] = max(
                float(self.adaptation_residual_by_skill.get(skill, 0.04)),
                float(bias),
            )
        self.thresholds = _per_skill_value(
            rows,
            lambda skill_rows: _threshold_for_fnr(
                [
                    _safedpa_row_score(row, self.adaptation_residual_by_skill)
                    for row in skill_rows
                ],
                [_label(row) for row in skill_rows],
                self.config.alpha,
                default=0.92,
            ),
            0.92,
        )
        self.calibration_summary.update(
            {
                "thresholds": self.thresholds,
                "adaptation_residual_by_skill": self.adaptation_residual_by_skill,
                "epsilon": self.config.epsilon,
                "score": (
                    "max(raw_critic_risk, hard_contract_score) plus calibrated "
                    "dynamics-adaptation residual and CBF margin erosion"
                ),
                "proxy_scope": (
                    "SafeDPA-style residual dynamics adaptation over frozen CASA "
                    "skill commands with gate-only CBF filtering"
                ),
            }
        )

    def decide(self, context: SotaDecisionContext) -> GateDecision:
        base_risk = max(float(context.raw_critic_risk), float(context.hard_contract_score))
        nominal_margin, margin_diagnostics = _predict_skill_margin(context, self.config.mpc_horizon_s)
        skill_demand = _context_skill_demand(context)
        residual = float(
            self.adaptation_residual_by_skill.get(
                context.skill_name,
                self.adaptation_residual_by_skill["global"],
            )
        )
        adaptation_scale = 1.0 + 0.5 * skill_demand
        adapted_margin = nominal_margin - residual * adaptation_scale
        cbf_pressure = max(0.0, 0.05 - adapted_margin)
        risk_score = _clip(base_risk + 0.35 * cbf_pressure + 0.05 * skill_demand, 0.0, 1.5)
        threshold = float(self.thresholds.get(context.skill_name, self.thresholds["global"]))
        allow = not bool(context.hard_contract_fixed_reject) and risk_score < threshold
        reason = "allow"
        if context.hard_contract_fixed_reject:
            reason = "hard_contract_fixed_reject"
        elif not allow:
            reason = "safedpa_adapted_cbf_filter"
        diagnostics = {
            "base_risk": base_risk,
            "nominal_cbf_margin": nominal_margin,
            "adaptation_residual": residual,
            "adaptation_scale": adaptation_scale,
            "adapted_cbf_margin": adapted_margin,
            "cbf_pressure": cbf_pressure,
            "skill_demand": skill_demand,
            "threshold_source": "phase4_calibrated_residual_cbf_gate",
            "proxy_scope": "gate_only_safe_dpa_residual_dynamics_adapter",
        }
        diagnostics.update(margin_diagnostics)
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
            solver_status="adapted_cbf_satisfied" if allow else "adapted_cbf_violated",
            diagnostics=diagnostics,
        )


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


class SAFERSplatCBFAdapted(SotaBaselineAdapter):
    name = "safer_splat_cbf_adapted"
    source_method = "SAFER-Splat: A Control Barrier Function for Safe Navigation with Online Gaussian Splatting Maps"
    implementation_fidelity = "paper_faithful_proxy"
    mode = "gate"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thresholds = _default_per_skill(0.65)
        self.margin_buffer_by_skill = _default_per_skill(0.10)

    def calibrate(
        self,
        calibration_data: Sequence[Mapping[str, Any]],
        config: Any | None = None,
    ) -> None:
        super().calibrate(calibration_data, config)
        rows = [row for row in calibration_data if _has_label(row)]
        self.margin_buffer_by_skill = _per_skill_value(
            rows,
            lambda skill_rows: _quantile(
                [max(0.0, 0.15 - _safer_row_margin(row)) for row in skill_rows],
                1.0 - self.config.epsilon,
                0.10,
            ),
            0.10,
        )
        self.thresholds = _per_skill_value(
            rows,
            lambda skill_rows: _threshold_for_fnr(
                [
                    _safer_row_score(row, self.margin_buffer_by_skill)
                    for row in skill_rows
                ],
                [_label(row) for row in skill_rows],
                self.config.alpha,
                default=0.65,
            ),
            0.65,
        )
        self.calibration_summary.update(
            {
                "thresholds": self.thresholds,
                "margin_buffer_by_skill": self.margin_buffer_by_skill,
                "score": (
                    "Gaussian/ellipsoid map CBF pressure plus CASA risk score; "
                    "offline calibration falls back to Phase4 distance features"
                ),
                "map_scope": "oracle_map_adaptation_from_mujoco_scene_props_or_distance_features",
            }
        )

    def decide(self, context: SotaDecisionContext) -> GateDecision:
        map_result = _safer_context_map_cbf(context, self.config.mpc_horizon_s)
        margin_buffer = float(
            self.margin_buffer_by_skill.get(context.skill_name, self.margin_buffer_by_skill["global"])
        )
        cbf_pressure = max(0.0, margin_buffer - map_result["min_cbf_margin"])
        base_risk = max(float(context.raw_critic_risk), float(context.hard_contract_score))
        risk_score = _clip(0.55 * base_risk + 0.85 * cbf_pressure + 0.20 * map_result["max_occupancy"], 0.0, 1.5)
        threshold = float(self.thresholds.get(context.skill_name, self.thresholds["global"]))
        allow = not bool(context.hard_contract_fixed_reject) and risk_score < threshold
        reason = "allow"
        if context.hard_contract_fixed_reject:
            reason = "hard_contract_fixed_reject"
        elif not allow:
            reason = "safer_splat_map_cbf_violation"
        diagnostics = {
            "base_risk": base_risk,
            "cbf_pressure": cbf_pressure,
            "margin_buffer": margin_buffer,
            "oracle_map_adaptation": True,
            "map_source": map_result["map_source"],
            "hazard_count": map_result["hazard_count"],
            "closest_hazard_kind": map_result["closest_hazard_kind"],
            "closest_hazard_name": map_result["closest_hazard_name"],
            "min_gaussian_cbf_margin": map_result["min_cbf_margin"],
            "max_gaussian_occupancy": map_result["max_occupancy"],
            "query_point_count": map_result["query_point_count"],
            "threshold_source": "phase4_calibrated_map_cbf_pressure",
        }
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
            solver_status="map_cbf_satisfied" if allow else "map_cbf_violated",
            diagnostics=diagnostics,
        )


class CLBFLBACAdapted(SotaBaselineAdapter):
    name = "clbf_lbac_adapted"
    source_method = "Reinforcement Learning for Safe Robot Control using Control Lyapunov Barrier Functions"
    implementation_fidelity = "paper_faithful_proxy"
    mode = "gate"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thresholds = _default_per_skill(0.75)
        self.weights = {name: value for name, value in _default_clbf_weights().items()}
        self.intercept = -2.0
        self.skill_bias = _default_per_skill(0.0)

    def fit(
        self,
        train_data: Sequence[Mapping[str, Any]],
        val_data: Sequence[Mapping[str, Any]],
        config: Any | None = None,
    ) -> None:
        super().fit(train_data, val_data, config)
        rows = [row for row in [*train_data, *val_data] if _has_label(row)]
        if not rows:
            self.calibration_summary["fit"].update(
                {"model_source": "default_weights_no_phase4_train_rows"}
            )
            return
        features = [_clbf_row_features(row) for row in rows]
        labels = [_label(row) for row in rows]
        positives = [features[index] for index, label in enumerate(labels) if label == 1]
        negatives = [features[index] for index, label in enumerate(labels) if label == 0]
        if positives and negatives:
            learned: dict[str, float] = {}
            for name in CLBF_FEATURES:
                pos_mean = _mean([item[name] for item in positives], 0.0)
                neg_mean = _mean([item[name] for item in negatives], 0.0)
                std = _std([item[name] for item in features], 1.0)
                learned[name] = _clip((pos_mean - neg_mean) / max(std, 1e-6), -4.0, 4.0)
            self.weights = _blend_clbf_weights(self.weights, learned, 0.65)
        global_rate = _clip(sum(labels) / max(1, len(labels)), 1e-4, 1.0 - 1e-4)
        global_mean = {name: _mean([item[name] for item in features], 0.0) for name in CLBF_FEATURES}
        self.intercept = _logit(global_rate) - sum(
            self.weights[name] * global_mean[name] for name in CLBF_FEATURES
        )
        for skill in MAIN_SKILLS:
            skill_labels = [_label(row) for row in rows if _skill(row) == skill]
            if not skill_labels:
                self.skill_bias[skill] = 0.0
                continue
            skill_rate = _clip(sum(skill_labels) / len(skill_labels), 1e-4, 1.0 - 1e-4)
            self.skill_bias[skill] = _clip(_logit(skill_rate) - _logit(global_rate), -2.0, 2.0)
        self.calibration_summary["fit"].update(
            {
                "model_source": "phase4_train_plus_critic_val_linear_clbf_proxy",
                "feature_names": list(CLBF_FEATURES),
                "weights": self.weights,
                "intercept": self.intercept,
                "skill_bias": self.skill_bias,
                "train_positive_count": int(sum(labels)),
                "train_negative_count": int(len(labels) - sum(labels)),
            }
        )

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
                [_clbf_row_probability(row, self.weights, self.intercept, self.skill_bias) for row in skill_rows],
                [_label(row) for row in skill_rows],
                self.config.alpha,
                default=0.75,
            ),
            0.75,
        )
        self.calibration_summary.update(
            {
                "thresholds": self.thresholds,
                "score": (
                    "learned CLBF-style unsafe probability from Phase4 features, "
                    "calibrated by per-skill FNR on the conformal split"
                ),
                "proxy_scope": "LBAC-inspired learned barrier/progress gate_over_casa_skills",
            }
        )

    def decide(self, context: SotaDecisionContext) -> GateDecision:
        feature_map = _clbf_context_features(context)
        probability = _clbf_probability(feature_map, self.weights, self.intercept, self.skill_bias, context.skill_name)
        threshold = float(self.thresholds.get(context.skill_name, self.thresholds["global"]))
        allow = not bool(context.hard_contract_fixed_reject) and probability < threshold
        reason = "allow"
        if context.hard_contract_fixed_reject:
            reason = "hard_contract_fixed_reject"
        elif not allow:
            reason = "clbf_barrier_progress_gate"
        clbf_value = 1.0 - probability
        lyapunov_progress = feature_map["progress_credit"] - probability
        return GateDecision(
            method_name=self.name,
            source_method=self.source_method,
            implementation_fidelity=self.implementation_fidelity,
            mode=self.mode,
            allow=allow,
            risk_score=probability,
            threshold=threshold,
            fallback_mode=None if allow else self.config.fallback_mode,
            reject_reason=reason,
            solver_status="clbf_decrease_condition_satisfied" if allow else "clbf_barrier_condition_violated",
            diagnostics={
                "clbf_value": clbf_value,
                "lyapunov_progress_score": lyapunov_progress,
                "unsafe_probability": probability,
                "feature_map": feature_map,
                "skill_bias": self.skill_bias.get(context.skill_name, self.skill_bias["global"]),
                "threshold_source": "phase4_trained_clbf_proxy_per_skill_calibration",
                "top_weighted_features": _top_weighted_features(feature_map, self.weights, 5),
            },
        )


CLBF_FEATURES = (
    "raw_risk",
    "hard_risk",
    "hard_fixed",
    "user_pressure",
    "arm_user_pressure",
    "obstacle_pressure",
    "posture_pressure",
    "base_height_deficit",
    "skill_demand",
    "progress_credit",
)


def _mean(values: Sequence[float], default: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float(default)
    return float(sum(finite) / len(finite))


def _std(values: Sequence[float], default: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return float(default)
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return float(math.sqrt(max(variance, 0.0)))


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(float(upper), max(float(lower), float(value))))


def _logit(probability: float) -> float:
    probability = _clip(probability, 1e-6, 1.0 - 1e-6)
    return float(math.log(probability / (1.0 - probability)))


def _sigmoid(value: float) -> float:
    value = _clip(value, -60.0, 60.0)
    return float(1.0 / (1.0 + math.exp(-value)))


def _row_feature(row: Mapping[str, Any], name: str, default: float) -> float:
    if name in row:
        return finite_float(row.get(name), default)
    vector = row.get("_feature_vector")
    names = row.get("_feature_names")
    if vector is None or names is None:
        return default
    try:
        index = list(names).index(name)
    except ValueError:
        return default
    try:
        return finite_float(vector[index], default)
    except (IndexError, TypeError):
        return default


def _row_skill_demand(row: Mapping[str, Any]) -> float:
    skill = _skill(row)
    duration = max(0.0, _row_feature(row, "skill/duration", 1.0))
    speed = math.hypot(
        _row_feature(row, "skill/walk_vx", 0.0),
        _row_feature(row, "skill/walk_vy", 0.0),
    )
    yaw_cost = abs(_row_feature(row, "skill/facing_yaw_deg", 0.0)) / 180.0
    yaw_cost = max(yaw_cost, abs(_row_feature(row, "skill/turn_face_yaw_deg", 0.0)) / 180.0)
    gesture_cost = 0.0
    if skill == "gesture" or _row_feature(row, "skill/is_gesture", 0.0) > 0.5:
        gesture_cost = (
            0.4 * _row_feature(row, "skill/gesture_amplitude", 0.35)
            + 0.2 * _row_feature(row, "skill/gesture_frequency", 0.75)
        )
    passive_credit = -0.15 if skill == "passive" or _row_feature(row, "skill/is_passive", 0.0) > 0.5 else 0.0
    return _clip(0.12 * duration + speed + yaw_cost + gesture_cost + passive_credit, 0.0, 2.5)


def _context_skill_demand(context: SotaDecisionContext) -> float:
    duration = max(0.0, context.param("duration", context.feature("skill/duration", 1.0)))
    speed = math.hypot(
        context.param("vx", context.feature("skill/walk_vx", 0.0)),
        context.param("vy", context.feature("skill/walk_vy", 0.0)),
    )
    yaw_cost = abs(
        context.param(
            "facing_yaw_deg",
            context.param("face_yaw_deg", context.feature("skill/facing_yaw_deg", 0.0)),
        )
    ) / 180.0
    gesture_cost = 0.0
    if context.skill_name == "gesture":
        gesture_cost = (
            0.4 * context.param("amplitude", context.feature("skill/gesture_amplitude", 0.35))
            + 0.2 * context.param("frequency", context.feature("skill/gesture_frequency", 0.75))
        )
    passive_credit = -0.15 if context.skill_name == "passive" else 0.0
    return _clip(0.12 * duration + speed + yaw_cost + gesture_cost + passive_credit, 0.0, 2.5)


def _row_predict_skill_margin(row: Mapping[str, Any]) -> float:
    hard_margin = 1.0 - max(_hard(row), 0.85 * _raw(row))
    min_user = _row_feature(row, "env/min_user_distance/current", 10.0)
    min_user = min(min_user, _row_feature(row, "env/min_user_distance/min", min_user))
    min_arm_user = _row_feature(row, "env/min_arm_user_distance/current", 10.0)
    min_arm_user = min(min_arm_user, _row_feature(row, "env/min_arm_user_distance/min", min_arm_user))
    min_obstacle = _row_feature(row, "env/min_obstacle_distance/current", 10.0)
    min_obstacle = min(min_obstacle, _row_feature(row, "env/min_obstacle_distance/min", min_obstacle))
    torso_roll = abs(_row_feature(row, "robot/torso_roll/current", 0.0))
    torso_pitch = abs(_row_feature(row, "robot/torso_pitch/current", 0.0))
    base_z = _row_feature(row, "robot/base_pos_z/current", 0.75)
    distance_margin = min(
        min_user - 0.50,
        min_arm_user - 0.20,
        min_obstacle - 0.10,
        0.60 - max(torso_roll, torso_pitch),
        base_z - 0.55,
        hard_margin,
    )
    skill_demand = _row_skill_demand(row)
    return float(distance_margin - 0.05 * skill_demand)


def _safedpa_adaptation_residual(row: Mapping[str, Any]) -> float:
    base_risk = max(_raw(row), _hard(row))
    nominal_margin = _row_predict_skill_margin(row)
    label_gap = max(0.0, float(_label(row)) - base_risk)
    margin_gap = max(0.0, 0.05 - nominal_margin)
    return float(label_gap + 0.20 * margin_gap + 0.02 * _row_skill_demand(row))


def _safedpa_row_score(
    row: Mapping[str, Any],
    residual_by_skill: Mapping[str, float],
) -> float:
    skill = _skill(row)
    base_risk = max(_raw(row), _hard(row))
    residual = float(residual_by_skill.get(skill, residual_by_skill.get("global", 0.04)))
    adapted_margin = _row_predict_skill_margin(row) - residual * (1.0 + 0.5 * _row_skill_demand(row))
    cbf_pressure = max(0.0, 0.05 - adapted_margin)
    return _clip(base_risk + 0.35 * cbf_pressure + 0.05 * _row_skill_demand(row), 0.0, 1.5)


def _safer_row_margin(row: Mapping[str, Any]) -> float:
    min_user = _row_feature(row, "env/min_user_distance/current", 10.0)
    min_user = min(min_user, _row_feature(row, "env/min_user_distance/min", min_user))
    min_arm_user = _row_feature(row, "env/min_arm_user_distance/current", 10.0)
    min_arm_user = min(min_arm_user, _row_feature(row, "env/min_arm_user_distance/min", min_arm_user))
    min_obstacle = _row_feature(row, "env/min_obstacle_distance/current", 10.0)
    min_obstacle = min(min_obstacle, _row_feature(row, "env/min_obstacle_distance/min", min_obstacle))
    return float(min(min_user - 0.55, min_arm_user - 0.25, min_obstacle - 0.18))


def _safer_row_score(
    row: Mapping[str, Any],
    margin_buffer_by_skill: Mapping[str, float],
) -> float:
    skill = _skill(row)
    margin = _safer_row_margin(row)
    buffer = float(margin_buffer_by_skill.get(skill, margin_buffer_by_skill.get("global", 0.10)))
    cbf_pressure = max(0.0, buffer - margin)
    occupancy = _clip(math.exp(-0.5 * max(0.0, margin) ** 2), 0.0, 1.0)
    base_risk = max(_raw(row), _hard(row))
    return _clip(0.55 * base_risk + 0.85 * cbf_pressure + 0.20 * occupancy, 0.0, 1.5)


def _safer_context_map_cbf(context: SotaDecisionContext, horizon_s: float) -> dict[str, Any]:
    hazards = _scene_hazards(context.scene_props)
    points = _candidate_path_points(context, horizon_s)
    if not hazards:
        min_user = context.feature("env/min_user_distance/current", 10.0)
        min_arm_user = context.feature("env/min_arm_user_distance/current", 10.0)
        min_obstacle = context.feature("env/min_obstacle_distance/current", 10.0)
        margin = min(min_user - 0.55, min_arm_user - 0.25, min_obstacle - 0.18)
        occupancy = _clip(math.exp(-0.5 * max(0.0, margin) ** 2), 0.0, 1.0)
        return {
            "map_source": "phase4_distance_feature_fallback",
            "hazard_count": int(min_user < 9.0) + int(min_obstacle < 9.0),
            "closest_hazard_kind": "distance_feature",
            "closest_hazard_name": "nearest_user_or_obstacle",
            "min_cbf_margin": margin,
            "max_occupancy": occupancy,
            "query_point_count": len(points),
        }

    best_margin = float("inf")
    best_hazard = hazards[0]
    best_occupancy = 0.0
    for point in points:
        for hazard in hazards:
            dx = point[0] - hazard["x"]
            dy = point[1] - hazard["y"]
            ellipsoid_norm = math.sqrt((dx / hazard["radius_x"]) ** 2 + (dy / hazard["radius_y"]) ** 2)
            margin = ellipsoid_norm - 1.0
            occupancy = math.exp(-0.5 * ellipsoid_norm**2)
            if margin < best_margin:
                best_margin = margin
                best_hazard = hazard
            best_occupancy = max(best_occupancy, occupancy)
    return {
        "map_source": "mujoco_scene_props_oracle_gaussian_ellipsoids",
        "hazard_count": len(hazards),
        "closest_hazard_kind": best_hazard["kind"],
        "closest_hazard_name": best_hazard["name"],
        "min_cbf_margin": best_margin,
        "max_occupancy": _clip(best_occupancy, 0.0, 1.0),
        "query_point_count": len(points),
    }


def _scene_hazards(scene_props: Mapping[str, Any]) -> list[dict[str, Any]]:
    placements = scene_props.get("placements") if isinstance(scene_props, Mapping) else None
    if not isinstance(placements, Mapping):
        nested = scene_props.get("scene_props") if isinstance(scene_props, Mapping) else None
        placements = nested.get("placements") if isinstance(nested, Mapping) else None
    if not isinstance(placements, Mapping):
        return []
    hazards: list[dict[str, Any]] = []
    for name, raw in placements.items():
        if not isinstance(raw, Mapping):
            continue
        if not finite_bool(raw.get("enabled", False)):
            continue
        position = raw.get("position", [])
        if not isinstance(position, Sequence) or len(position) < 2:
            continue
        x = finite_float(position[0], 0.0)
        y = finite_float(position[1], 0.0)
        z = finite_float(position[2], 0.0) if len(position) > 2 else 0.0
        if z < -5.0:
            continue
        kind = "user" if str(name).startswith("user_proxy_") else "obstacle"
        radius = (0.55, 0.45) if kind == "user" else (0.34, 0.28)
        hazards.append(
            {
                "name": str(name),
                "kind": kind,
                "x": x,
                "y": y,
                "radius_x": radius[0],
                "radius_y": radius[1],
            }
        )
    return hazards


def _candidate_path_points(context: SotaDecisionContext, horizon_s: float) -> list[tuple[float, float]]:
    base_x = context.feature("robot/base_pos_x/current", 0.0)
    base_y = context.feature("robot/base_pos_y/current", 0.0)
    duration = max(0.0, context.param("duration", context.feature("skill/duration", 1.0)))
    effective_horizon = min(max(0.0, horizon_s), duration)
    vx = context.param("vx", context.feature("skill/walk_vx", 0.0)) if context.skill_name == "walk" else 0.0
    vy = context.param("vy", context.feature("skill/walk_vy", 0.0)) if context.skill_name == "walk" else 0.0
    samples = max(2, min(8, int(math.ceil(effective_horizon / 0.35)) + 1))
    points = []
    for index in range(samples):
        fraction = index / max(1, samples - 1)
        t = effective_horizon * fraction
        points.append((base_x + 0.45 * vx * t, base_y + 0.45 * vy * t))
    if context.skill_name == "gesture":
        amplitude = context.param("amplitude", context.feature("skill/gesture_amplitude", 0.35))
        points.extend([(base_x, base_y + 0.35 + 0.3 * amplitude), (base_x, base_y - 0.35 - 0.3 * amplitude)])
    return points


def _default_clbf_weights() -> dict[str, float]:
    return {
        "raw_risk": 1.8,
        "hard_risk": 1.2,
        "hard_fixed": 2.0,
        "user_pressure": 1.0,
        "arm_user_pressure": 0.8,
        "obstacle_pressure": 0.9,
        "posture_pressure": 1.1,
        "base_height_deficit": 1.2,
        "skill_demand": 0.35,
        "progress_credit": -0.30,
    }


def _blend_clbf_weights(
    default: Mapping[str, float],
    learned: Mapping[str, float],
    learned_weight: float,
) -> dict[str, float]:
    output = {}
    for name in CLBF_FEATURES:
        output[name] = (1.0 - learned_weight) * float(default.get(name, 0.0)) + learned_weight * float(
            learned.get(name, 0.0)
        )
    return output


def _clbf_row_features(row: Mapping[str, Any]) -> dict[str, float]:
    min_user = min(
        _row_feature(row, "env/min_user_distance/current", 10.0),
        _row_feature(row, "env/min_user_distance/min", 10.0),
    )
    min_arm_user = min(
        _row_feature(row, "env/min_arm_user_distance/current", 10.0),
        _row_feature(row, "env/min_arm_user_distance/min", 10.0),
    )
    min_obstacle = min(
        _row_feature(row, "env/min_obstacle_distance/current", 10.0),
        _row_feature(row, "env/min_obstacle_distance/min", 10.0),
    )
    torso_tilt = max(
        abs(_row_feature(row, "robot/torso_roll/current", 0.0)),
        abs(_row_feature(row, "robot/torso_pitch/current", 0.0)),
    )
    base_z = _row_feature(row, "robot/base_pos_z/current", 0.75)
    demand = _row_skill_demand(row)
    return {
        "raw_risk": _clip(_raw(row), 0.0, 1.0),
        "hard_risk": _clip(_hard(row), 0.0, 1.0),
        "hard_fixed": 1.0 if _hard(row) >= 1.0 else 0.0,
        "user_pressure": _clip((0.75 - min_user) / 0.75, 0.0, 2.0),
        "arm_user_pressure": _clip((0.35 - min_arm_user) / 0.35, 0.0, 2.0),
        "obstacle_pressure": _clip((0.40 - min_obstacle) / 0.40, 0.0, 2.0),
        "posture_pressure": _clip((torso_tilt - 0.35) / 0.25, 0.0, 2.0),
        "base_height_deficit": _clip((0.68 - base_z) / 0.25, 0.0, 2.0),
        "skill_demand": demand,
        "progress_credit": _clbf_progress_credit(_skill(row), demand),
    }


def _clbf_context_features(context: SotaDecisionContext) -> dict[str, float]:
    min_user = context.feature("env/min_user_distance/current", 10.0)
    min_arm_user = context.feature("env/min_arm_user_distance/current", 10.0)
    min_obstacle = context.feature("env/min_obstacle_distance/current", 10.0)
    torso_tilt = max(
        abs(context.feature("robot/torso_roll/current", 0.0)),
        abs(context.feature("robot/torso_pitch/current", 0.0)),
    )
    base_z = context.feature("robot/base_pos_z/current", 0.75)
    demand = _context_skill_demand(context)
    return {
        "raw_risk": _clip(float(context.raw_critic_risk), 0.0, 1.0),
        "hard_risk": _clip(float(context.hard_contract_score), 0.0, 1.0),
        "hard_fixed": 1.0 if context.hard_contract_fixed_reject else 0.0,
        "user_pressure": _clip((0.75 - min_user) / 0.75, 0.0, 2.0),
        "arm_user_pressure": _clip((0.35 - min_arm_user) / 0.35, 0.0, 2.0),
        "obstacle_pressure": _clip((0.40 - min_obstacle) / 0.40, 0.0, 2.0),
        "posture_pressure": _clip((torso_tilt - 0.35) / 0.25, 0.0, 2.0),
        "base_height_deficit": _clip((0.68 - base_z) / 0.25, 0.0, 2.0),
        "skill_demand": demand,
        "progress_credit": _clbf_progress_credit(context.skill_name, demand),
    }


def _clbf_progress_credit(skill: str, demand: float) -> float:
    if skill == "passive":
        return 0.20
    if skill == "gesture":
        return _clip(0.45 - 0.10 * demand, 0.0, 0.5)
    return _clip(0.55 - 0.12 * demand, 0.0, 0.6)


def _clbf_row_probability(
    row: Mapping[str, Any],
    weights: Mapping[str, float],
    intercept: float,
    skill_bias: Mapping[str, float],
) -> float:
    return _clbf_probability(_clbf_row_features(row), weights, intercept, skill_bias, _skill(row))


def _clbf_probability(
    features: Mapping[str, float],
    weights: Mapping[str, float],
    intercept: float,
    skill_bias: Mapping[str, float],
    skill: str,
) -> float:
    score = float(intercept) + float(skill_bias.get(skill, skill_bias.get("global", 0.0)))
    for name in CLBF_FEATURES:
        score += float(weights.get(name, 0.0)) * float(features.get(name, 0.0))
    return _sigmoid(score)


def _top_weighted_features(
    features: Mapping[str, float],
    weights: Mapping[str, float],
    count: int,
) -> list[dict[str, float | str]]:
    weighted = [
        {
            "feature": name,
            "value": float(features.get(name, 0.0)),
            "weight": float(weights.get(name, 0.0)),
            "contribution": float(features.get(name, 0.0)) * float(weights.get(name, 0.0)),
        }
        for name in CLBF_FEATURES
    ]
    weighted.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
    return weighted[:count]


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
