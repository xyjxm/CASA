"""Adapted external SOTA baseline gates for CASA Phase 5."""

from gear_sonic.casa.baselines.sota_adapters.api import (
    GateDecision,
    SotaAdapterConfig,
    SotaBaselineAdapter,
    SotaDecisionContext,
)
from gear_sonic.casa.baselines.sota_adapters.registry import (
    SOTA_METHOD_DISPLAY,
    SOTA_METHOD_ORDER,
    SotaBaselineRegistry,
    build_sota_registry,
    is_sota_method,
)

__all__ = [
    "GateDecision",
    "SOTA_METHOD_DISPLAY",
    "SOTA_METHOD_ORDER",
    "SotaAdapterConfig",
    "SotaBaselineAdapter",
    "SotaBaselineRegistry",
    "SotaDecisionContext",
    "build_sota_registry",
    "is_sota_method",
]
