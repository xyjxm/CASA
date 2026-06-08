"""CASA-Hybrid Phase 5 gates built from CASA-A and adapted SOTA anchors."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

from gear_sonic.casa.baselines.sota_adapters import (
    GateDecision,
    SotaBaselineRegistry,
    SotaDecisionContext,
)

MPC_ANCHOR = "mpc_cbf_humanoid_adapted"
SAFEDPA_ANCHOR = "safedpa_adapted"
CASA_ANCHOR = "casa_a_per_skill"
HYBRID_SOURCE_METHOD = "CASA-Hybrid(MPC-CBF-Humanoid + SafeDPA + CASA-A refine)"

HYBRID_METHOD_STRATEGIES = {
    "casa_h_mpc_safedpa_anchor_or": "anchor_or",
    "casa_h_mpc_safedpa_consensus": "consensus",
    "casa_h_mpc_safedpa_gray_refine": "gray_refine",
    "casa_h_mpc_safedpa_budgeted_selector": "budgeted_selector",
    "casa_h_mpc_safedpa_calibrated_ensemble": "calibrated_ensemble",
    "casa_h_mpc_safedpa_casa_refine": "gray_refine",
}
HYBRID_METHOD_ORDER = tuple(HYBRID_METHOD_STRATEGIES)
HYBRID_METHOD_DISPLAY = {
    "casa_h_mpc_safedpa_anchor_or": "CASA-Hybrid anchor-OR",
    "casa_h_mpc_safedpa_consensus": "CASA-Hybrid consensus",
    "casa_h_mpc_safedpa_gray_refine": "CASA-Hybrid gray refine",
    "casa_h_mpc_safedpa_budgeted_selector": "CASA-Hybrid budgeted selector",
    "casa_h_mpc_safedpa_calibrated_ensemble": "CASA-Hybrid calibrated ensemble",
    "casa_h_mpc_safedpa_casa_refine": "CASA-Hybrid CASA refine",
}


@dataclass(frozen=True)
class Phase5HybridConfig:
    """Runtime configuration for the CASA-Hybrid gate family."""

    default_strategy: str = "gray_refine"
    method_strategies: dict[str, str] = field(
        default_factory=lambda: dict(HYBRID_METHOD_STRATEGIES)
    )
    anchors: tuple[str, str] = (MPC_ANCHOR, SAFEDPA_ANCHOR)
    hard_contract_veto: bool = True
    gray_margin: float = 0.06
    gray_anchor_reject_votes: int = 2
    outside_gray_anchor_veto_votes: int = 2
    consensus_reject_votes: int = 2
    budget_high_confidence_margin: float = 0.12
    budget_rescue_margin: float = 0.04
    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {CASA_ANCHOR: 0.50, MPC_ANCHOR: 0.25, SAFEDPA_ANCHOR: 0.25}
    )
    ensemble_threshold: float = 0.0
    ensemble_margin_scale: float = 1.0

    def strategy_for_method(self, method: str) -> str:
        return self.method_strategies.get(method, HYBRID_METHOD_STRATEGIES.get(method, self.default_strategy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_strategy": self.default_strategy,
            "method_strategies": dict(self.method_strategies),
            "anchors": list(self.anchors),
            "hard_contract_veto": self.hard_contract_veto,
            "gray_refine": {
                "gray_margin": self.gray_margin,
                "gray_anchor_reject_votes": self.gray_anchor_reject_votes,
                "outside_gray_anchor_veto_votes": self.outside_gray_anchor_veto_votes,
            },
            "consensus": {"reject_votes": self.consensus_reject_votes},
            "budgeted_selector": {
                "high_confidence_margin": self.budget_high_confidence_margin,
                "rescue_margin": self.budget_rescue_margin,
            },
            "calibrated_ensemble": {
                "weights": dict(self.ensemble_weights),
                "threshold": self.ensemble_threshold,
                "margin_scale": self.ensemble_margin_scale,
            },
        }


@dataclass(frozen=True)
class AnchorDecision:
    name: str
    allow: bool
    risk_score: float | None
    threshold: float | None
    margin: float | None
    reject_reason: str
    runtime_ms: float | None = None
    solver_status: str = ""
    source_method: str = ""

    @property
    def reject(self) -> bool:
        return not self.allow


def is_hybrid_method(method: str) -> bool:
    return method in HYBRID_METHOD_STRATEGIES


def load_phase5_hybrid_config(path: Path | None = None) -> Phase5HybridConfig:
    if path is None:
        return Phase5HybridConfig()
    if not path.exists():
        raise FileNotFoundError(f"Hybrid config does not exist: {path}")
    try:
        import yaml  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load --hybrid-config") from exc
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Hybrid config must be a YAML mapping: {path}")
    return _config_from_mapping(data)


def decide_phase5_hybrid(
    *,
    method: str,
    context: SotaDecisionContext,
    casa_decision: Any,
    sota_registry: SotaBaselineRegistry,
    config: Phase5HybridConfig,
) -> GateDecision:
    if not is_hybrid_method(method):
        raise ValueError(f"Unknown CASA-Hybrid method: {method}")
    anchors = _anchor_decisions(context, casa_decision, sota_registry, config)
    strategy = config.strategy_for_method(method)
    reject, risk_score, threshold, reason, details = _decide_strategy(strategy, anchors, config, context)
    diagnostics = {
        "hybrid_method": method,
        "hybrid_strategy": strategy,
        "anchor_methods": [CASA_ANCHOR, *config.anchors],
        "anchor_decisions": {name: _anchor_to_dict(anchor) for name, anchor in anchors.items()},
        "hard_contract_veto": bool(config.hard_contract_veto),
        "strategy_details": details,
        "config": config.to_dict(),
    }
    return GateDecision(
        method_name=method,
        source_method=HYBRID_SOURCE_METHOD,
        implementation_fidelity="exact_code",
        mode="gate",
        allow=not reject,
        risk_score=risk_score,
        threshold=threshold,
        fallback_mode=None if not reject else "passive_stop",
        reject_reason=reason,
        runtime_ms=sum(anchor.runtime_ms or 0.0 for anchor in anchors.values()),
        solver_status=f"hybrid_{strategy}_{'reject' if reject else 'allow'}",
        diagnostics=diagnostics,
    )


def _config_from_mapping(data: Mapping[str, Any]) -> Phase5HybridConfig:
    strategies = dict(HYBRID_METHOD_STRATEGIES)
    strategies.update(_dict(data.get("method_strategies")))
    strategies.update(_dict(data.get("aliases")))
    anchors = tuple(str(item) for item in data.get("anchors", [MPC_ANCHOR, SAFEDPA_ANCHOR]))
    if len(anchors) != 2:
        raise ValueError("Hybrid config anchors must contain exactly two SOTA method names")
    gray = _dict(data.get("gray_refine"))
    consensus = _dict(data.get("consensus"))
    budget = _dict(data.get("budgeted_selector"))
    ensemble = _dict(data.get("calibrated_ensemble"))
    weights = dict(Phase5HybridConfig().ensemble_weights)
    weights.update({str(key): _finite_float(value, weights.get(str(key), 0.0)) for key, value in _dict(ensemble.get("weights")).items()})
    config = Phase5HybridConfig(
        default_strategy=str(data.get("default_strategy", "gray_refine")),
        method_strategies={str(key): str(value) for key, value in strategies.items()},
        anchors=(anchors[0], anchors[1]),
        hard_contract_veto=_bool(data.get("hard_contract_veto", True)),
        gray_margin=_finite_float(gray.get("gray_margin"), 0.06),
        gray_anchor_reject_votes=_finite_int(gray.get("gray_anchor_reject_votes"), 2),
        outside_gray_anchor_veto_votes=_finite_int(gray.get("outside_gray_anchor_veto_votes"), 2),
        consensus_reject_votes=_finite_int(consensus.get("reject_votes"), 2),
        budget_high_confidence_margin=_finite_float(budget.get("high_confidence_margin"), 0.12),
        budget_rescue_margin=_finite_float(budget.get("rescue_margin"), 0.04),
        ensemble_weights=weights,
        ensemble_threshold=_finite_float(ensemble.get("threshold"), 0.0),
        ensemble_margin_scale=max(1e-6, _finite_float(ensemble.get("margin_scale"), 1.0)),
    )
    _validate_strategies(config)
    return config


def _validate_strategies(config: Phase5HybridConfig) -> None:
    valid = {"anchor_or", "consensus", "gray_refine", "budgeted_selector", "calibrated_ensemble"}
    unknown = sorted({strategy for strategy in config.method_strategies.values() if strategy not in valid})
    if config.default_strategy not in valid:
        unknown.append(config.default_strategy)
    if unknown:
        raise ValueError(f"Unknown CASA-Hybrid strategies: {unknown}; valid strategies are {sorted(valid)}")


def _anchor_decisions(
    context: SotaDecisionContext,
    casa_decision: Any,
    sota_registry: SotaBaselineRegistry,
    config: Phase5HybridConfig,
) -> dict[str, AnchorDecision]:
    anchors: dict[str, AnchorDecision] = {
        CASA_ANCHOR: AnchorDecision(
            name=CASA_ANCHOR,
            allow=not casa_decision.reject,
            risk_score=float(context.raw_critic_risk),
            threshold=casa_decision.threshold,
            margin=casa_decision.risk_margin,
            reject_reason=casa_decision.reject_reason,
            source_method="CASA-A per-skill conformal",
        )
    }
    for method in config.anchors:
        decision = sota_registry.decide(method, context)
        anchors[method] = AnchorDecision(
            name=method,
            allow=decision.allow,
            risk_score=decision.risk_score,
            threshold=decision.threshold,
            margin=decision.risk_margin,
            reject_reason=decision.reject_reason,
            runtime_ms=decision.runtime_ms,
            solver_status=decision.solver_status,
            source_method=decision.source_method,
        )
    return anchors


def _decide_strategy(
    strategy: str,
    anchors: dict[str, AnchorDecision],
    config: Phase5HybridConfig,
    context: SotaDecisionContext,
) -> tuple[bool, float, float, str, dict[str, Any]]:
    if config.hard_contract_veto and bool(context.hard_contract_fixed_reject):
        return True, 1.0, 0.0, "hybrid_hard_contract_veto", {"hard_contract_fixed_reject": True}
    if strategy == "anchor_or":
        return _anchor_or(anchors)
    if strategy == "consensus":
        return _consensus(anchors, config.consensus_reject_votes)
    if strategy == "gray_refine":
        return _gray_refine(anchors, config)
    if strategy == "budgeted_selector":
        return _budgeted_selector(anchors, config)
    if strategy == "calibrated_ensemble":
        return _calibrated_ensemble(anchors, config)
    raise ValueError(f"Unknown CASA-Hybrid strategy: {strategy}")


def _anchor_or(anchors: dict[str, AnchorDecision]) -> tuple[bool, float, float, str, dict[str, Any]]:
    reject_names = [name for name, anchor in anchors.items() if anchor.reject]
    margin = max(_margin(anchor) for anchor in anchors.values())
    reject = bool(reject_names)
    reason = "allow" if not reject else "hybrid_anchor_or_" + "_".join(reject_names)
    return reject, margin, 0.0, reason, {"reject_votes": len(reject_names), "reject_anchors": reject_names}


def _consensus(
    anchors: dict[str, AnchorDecision],
    reject_votes_required: int,
) -> tuple[bool, float, float, str, dict[str, Any]]:
    reject_names = [name for name, anchor in anchors.items() if anchor.reject]
    reject = len(reject_names) >= max(1, reject_votes_required)
    score = float(len(reject_names) - max(1, reject_votes_required))
    reason = "allow" if not reject else "hybrid_consensus_reject"
    return reject, score, 0.0, reason, {
        "reject_votes": len(reject_names),
        "reject_votes_required": max(1, reject_votes_required),
        "reject_anchors": reject_names,
    }


def _gray_refine(
    anchors: dict[str, AnchorDecision],
    config: Phase5HybridConfig,
) -> tuple[bool, float, float, str, dict[str, Any]]:
    casa = anchors[CASA_ANCHOR]
    sota_names = list(config.anchors)
    sota = [anchors[name] for name in sota_names]
    casa_margin = _margin(casa)
    gray = abs(casa_margin) <= max(0.0, config.gray_margin)
    sota_reject_names = [anchor.name for anchor in sota if anchor.reject]
    sota_allow_names = [anchor.name for anchor in sota if anchor.allow]
    details = {
        "casa_margin": casa_margin,
        "gray_margin": config.gray_margin,
        "is_gray_zone": gray,
        "sota_reject_votes": len(sota_reject_names),
        "sota_reject_anchors": sota_reject_names,
        "sota_allow_anchors": sota_allow_names,
    }
    if gray and len(sota_reject_names) >= max(1, config.gray_anchor_reject_votes):
        return True, max(_margin(anchor) for anchor in sota), 0.0, "hybrid_gray_refine_anchor_reject", details
    if gray and len(sota_allow_names) == len(sota):
        return False, min(_margin(anchor) for anchor in sota), 0.0, "allow", details
    if not gray and casa.allow and len(sota_reject_names) >= max(1, config.outside_gray_anchor_veto_votes):
        return True, max(_margin(anchor) for anchor in sota), 0.0, "hybrid_gray_refine_dual_anchor_veto", details
    reject = casa.reject
    return reject, casa_margin, 0.0, casa.reject_reason if reject else "allow", details


def _budgeted_selector(
    anchors: dict[str, AnchorDecision],
    config: Phase5HybridConfig,
) -> tuple[bool, float, float, str, dict[str, Any]]:
    casa = anchors[CASA_ANCHOR]
    sota = [anchors[name] for name in config.anchors]
    casa_margin = _margin(casa)
    sota_rejects = [anchor.name for anchor in sota if anchor.reject]
    details = {
        "casa_margin": casa_margin,
        "sota_reject_votes": len(sota_rejects),
        "high_confidence_margin": config.budget_high_confidence_margin,
        "rescue_margin": config.budget_rescue_margin,
        "policy": "reject only on high-confidence CASA or dual-anchor evidence; allow low-margin CASA rejects when both anchors allow",
    }
    if len(sota_rejects) == len(sota):
        return True, max(_margin(anchor) for anchor in sota), 0.0, "hybrid_budgeted_dual_anchor_reject", details
    if casa.reject and casa_margin >= max(0.0, config.budget_high_confidence_margin):
        return True, casa_margin, 0.0, "hybrid_budgeted_high_confidence_casa_reject", details
    if casa.reject and not sota_rejects and casa_margin <= max(0.0, config.budget_rescue_margin):
        return False, casa_margin, 0.0, "allow", details
    return casa.reject, casa_margin, 0.0, casa.reject_reason if casa.reject else "allow", details


def _calibrated_ensemble(
    anchors: dict[str, AnchorDecision],
    config: Phase5HybridConfig,
) -> tuple[bool, float, float, str, dict[str, Any]]:
    total_weight = sum(max(0.0, float(weight)) for weight in config.ensemble_weights.values())
    if total_weight <= 0:
        total_weight = 1.0
    terms = {}
    score = 0.0
    for name, anchor in anchors.items():
        weight = max(0.0, float(config.ensemble_weights.get(name, 0.0))) / total_weight
        margin = _margin(anchor) / config.ensemble_margin_scale
        terms[name] = {"weight": weight, "margin": margin}
        score += weight * margin
    reject = score >= float(config.ensemble_threshold)
    reason = "hybrid_calibrated_ensemble_threshold" if reject else "allow"
    return reject, score, float(config.ensemble_threshold), reason, {
        "terms": terms,
        "score": score,
        "threshold": config.ensemble_threshold,
        "margin_scale": config.ensemble_margin_scale,
    }


def _anchor_to_dict(anchor: AnchorDecision) -> dict[str, Any]:
    return {
        "method": anchor.name,
        "allow": anchor.allow,
        "decision": "allow" if anchor.allow else "reject",
        "risk_score": anchor.risk_score,
        "threshold": anchor.threshold,
        "risk_margin": anchor.margin,
        "reject_reason": anchor.reject_reason,
        "runtime_ms": anchor.runtime_ms,
        "solver_status": anchor.solver_status,
        "source_method": anchor.source_method,
    }


def _margin(anchor: AnchorDecision) -> float:
    if anchor.margin is not None and math.isfinite(float(anchor.margin)):
        return float(anchor.margin)
    if anchor.reject:
        return 1.0
    return -1.0


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _finite_float(value: Any, default: float) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def _finite_int(value: Any, default: int) -> int:
    try:
        output = int(value)
    except (TypeError, ValueError):
        return default
    return output


def decision_diagnostics_json(decision: GateDecision) -> str:
    return json.dumps(decision.diagnostics, sort_keys=True)
