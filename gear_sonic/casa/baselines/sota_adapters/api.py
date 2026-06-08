"""Shared adapter API for CASA-adapted external safety baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import time
from typing import Any, Literal, Mapping, Sequence

ImplementationFidelity = Literal["exact_code", "paper_faithful_proxy", "lightweight_proxy"]
AdapterMode = Literal["gate", "filter"]


@dataclass(frozen=True)
class SotaAdapterConfig:
    """Configuration shared by first-slice adapted SOTA baselines."""

    alpha: float = 0.10
    epsilon: float = 0.05
    fallback_mode: str = "passive_stop"
    chance_z: float = 1.6448536269514722
    uncertainty_floor: float = 0.02
    mpc_horizon_s: float = 2.0
    mpc_step_s: float = 0.5
    calibration_mode: str = "phase4_conformal_calibration_only"


@dataclass(frozen=True)
class SotaDecisionContext:
    """Inputs available to a SOTA adapter for one CASA skill candidate."""

    skill_name: str
    skill_params: Mapping[str, Any] = field(default_factory=dict)
    raw_critic_risk: float = 0.0
    hard_contract_score: float = 0.0
    hard_contract_fixed_reject: bool = False
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    feature_vector: Sequence[float] | None = None
    feature_names: Sequence[str] = field(default_factory=tuple)
    scene_props: Mapping[str, Any] = field(default_factory=dict)
    row: Mapping[str, Any] = field(default_factory=dict)

    def feature(self, name: str, default: float) -> float:
        if name in self.row:
            return finite_float(self.row.get(name), default)
        if self.feature_vector is None or not self.feature_names:
            return default
        try:
            index = list(self.feature_names).index(name)
        except ValueError:
            return default
        try:
            return finite_float(self.feature_vector[index], default)
        except IndexError:
            return default

    def param(self, name: str, default: float) -> float:
        return finite_float(self.skill_params.get(name), default)


@dataclass(frozen=True)
class GateDecision:
    """Adapter decision normalized for online and offline CASA logging."""

    method_name: str
    source_method: str
    implementation_fidelity: ImplementationFidelity
    mode: AdapterMode
    allow: bool
    risk_score: float | None
    threshold: float | None
    corrected_skill: dict[str, Any] | None = None
    fallback_mode: str | None = None
    reject_reason: str = "allow"
    runtime_ms: float | None = None
    solver_status: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_margin(self) -> float | None:
        if self.risk_score is None or self.threshold is None:
            return None
        return float(self.risk_score) - float(self.threshold)


class SotaBaselineAdapter(ABC):
    """Base class for adapted SOTA gates.

    First-slice adapters operate as CASA-compatible gate-only safety filters.
    Controller-modifying variants can reuse the same API later by setting
    ``mode = "filter"`` and filling ``corrected_skill``.
    """

    name: str
    source_method: str
    implementation_fidelity: ImplementationFidelity
    mode: AdapterMode = "gate"

    def __init__(self, config: SotaAdapterConfig | None = None) -> None:
        self.config = config or SotaAdapterConfig()
        self.calibration_summary: dict[str, Any] = {}

    def fit(
        self,
        train_data: Sequence[Mapping[str, Any]],
        val_data: Sequence[Mapping[str, Any]],
        config: SotaAdapterConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        self.calibration_summary["fit"] = {
            "train_rows": len(train_data),
            "val_rows": len(val_data),
            "note": "adapter_default_fit_noop",
        }

    def calibrate(
        self,
        calibration_data: Sequence[Mapping[str, Any]],
        config: SotaAdapterConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        self.calibration_summary["calibration_rows"] = len(calibration_data)

    def timed_decide(self, context: SotaDecisionContext) -> GateDecision:
        start = time.perf_counter()
        decision = self.decide(context)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        diagnostics = dict(decision.diagnostics)
        diagnostics.setdefault("calibration_mode", self.config.calibration_mode)
        diagnostics.setdefault("adapter_runtime_scope", "adapter_decide_only")
        return GateDecision(
            method_name=decision.method_name,
            source_method=decision.source_method,
            implementation_fidelity=decision.implementation_fidelity,
            mode=decision.mode,
            allow=decision.allow,
            risk_score=decision.risk_score,
            threshold=decision.threshold,
            corrected_skill=decision.corrected_skill,
            fallback_mode=decision.fallback_mode,
            reject_reason=decision.reject_reason,
            runtime_ms=runtime_ms,
            solver_status=decision.solver_status,
            diagnostics=diagnostics,
        )

    @abstractmethod
    def decide(self, context: SotaDecisionContext) -> GateDecision:
        """Return an allow/reject decision for a CASA candidate skill."""


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(output):
        return default
    return output


def finite_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "t", "yes", "y"}
