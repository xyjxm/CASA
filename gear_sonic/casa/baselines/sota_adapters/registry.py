"""Registry for CASA-adapted external SOTA baselines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gear_sonic.casa.baselines.sota_adapters.api import (
    GateDecision,
    SotaAdapterConfig,
    SotaBaselineAdapter,
    SotaDecisionContext,
)
from gear_sonic.casa.baselines.sota_adapters.methods import (
    CLBFLBACAdapted,
    MPCCBFHumanoidAdapted,
    PCBFAdapted,
    SAFERSplatCBFAdapted,
    SafeDPAAdapted,
)

SOTA_METHOD_ORDER = (
    "safedpa_adapted",
    "pcbf_adapted",
    "safer_splat_cbf_adapted",
    "mpc_cbf_humanoid_adapted",
    "clbf_lbac_adapted",
)
SOTA_METHOD_DISPLAY = {
    "safedpa_adapted": "SafeDPA-adapted",
    "pcbf_adapted": "PCBF-adapted",
    "safer_splat_cbf_adapted": "SAFER-Splat-CBF-adapted",
    "mpc_cbf_humanoid_adapted": "MPC-CBF-Humanoid-adapted",
    "clbf_lbac_adapted": "CLBF-LBAC-adapted",
}


class SotaBaselineRegistry:
    def __init__(self, adapters: Sequence[SotaBaselineAdapter]) -> None:
        self.adapters = {adapter.name: adapter for adapter in adapters}

    def __contains__(self, method: str) -> bool:
        return method in self.adapters

    def names(self) -> tuple[str, ...]:
        return tuple(name for name in SOTA_METHOD_ORDER if name in self.adapters)

    def calibrate(
        self,
        calibration_data: Sequence[Mapping[str, Any]],
        config: SotaAdapterConfig | None = None,
    ) -> None:
        for adapter in self.adapters.values():
            adapter.calibrate(calibration_data, config)

    def fit(
        self,
        train_data: Sequence[Mapping[str, Any]],
        val_data: Sequence[Mapping[str, Any]],
        config: SotaAdapterConfig | None = None,
    ) -> None:
        for adapter in self.adapters.values():
            adapter.fit(train_data, val_data, config)

    def decide(self, method: str, context: SotaDecisionContext) -> GateDecision:
        try:
            adapter = self.adapters[method]
        except KeyError as exc:
            raise ValueError(f"Unknown SOTA baseline adapter: {method}") from exc
        return adapter.timed_decide(context)

    def metadata(self) -> dict[str, Any]:
        return {
            name: {
                "method": adapter.name,
                "source_method": adapter.source_method,
                "implementation_fidelity": adapter.implementation_fidelity,
                "mode": adapter.mode,
                "calibration_summary": adapter.calibration_summary,
            }
            for name, adapter in self.adapters.items()
        }


def build_sota_registry(config: SotaAdapterConfig | None = None) -> SotaBaselineRegistry:
    return SotaBaselineRegistry(
        [
            SafeDPAAdapted(config),
            PCBFAdapted(config),
            SAFERSplatCBFAdapted(config),
            MPCCBFHumanoidAdapted(config),
            CLBFLBACAdapted(config),
        ]
    )


def is_sota_method(method: str) -> bool:
    return method in SOTA_METHOD_ORDER
