from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gear_sonic.casa.baselines.sota_adapters import GateDecision, SotaDecisionContext
from gear_sonic.casa.phase5_hybrid import (
    Phase5HybridConfig,
    decide_phase5_hybrid,
    load_phase5_hybrid_config,
)
from gear_sonic.casa.phase5_policy import ONLINE_METHOD_ORDER, method_display
from gear_sonic.scripts.casa_audit_phase5_hybrid import build_hybrid_audit
from gear_sonic.scripts.casa_run_phase5_online_main_lowmem import lane_cmd


def test_hybrid_methods_are_online_registered() -> None:
    assert "casa_h_mpc_safedpa_casa_refine" in ONLINE_METHOD_ORDER
    assert method_display("casa_h_mpc_safedpa_casa_refine") != "casa_h_mpc_safedpa_casa_refine"


def test_config_maps_final_method_to_gray_refine() -> None:
    config = load_phase5_hybrid_config(Path("configs/phase5_hybrid_mpc_safedpa.yaml"))
    assert config.strategy_for_method("casa_h_mpc_safedpa_casa_refine") == "gray_refine"


def test_gray_refine_rescues_gray_casa_reject_when_anchors_allow() -> None:
    context = SotaDecisionContext(skill_name="walk", raw_critic_risk=0.52, hard_contract_score=0.0)
    casa = SimpleNamespace(
        reject=True,
        threshold=0.50,
        risk_margin=0.02,
        reject_reason="per_skill_conformal_threshold",
    )
    decision = decide_phase5_hybrid(
        method="casa_h_mpc_safedpa_casa_refine",
        context=context,
        casa_decision=casa,
        sota_registry=_FakeRegistry({"mpc_cbf_humanoid_adapted": True, "safedpa_adapted": True}),
        config=Phase5HybridConfig(gray_margin=0.06),
    )
    assert decision.allow is True
    assert decision.diagnostics["hybrid_strategy"] == "gray_refine"


def test_gray_refine_dual_anchor_vetoes_confident_casa_allow() -> None:
    context = SotaDecisionContext(skill_name="walk", raw_critic_risk=0.30, hard_contract_score=0.0)
    casa = SimpleNamespace(reject=False, threshold=0.50, risk_margin=-0.20, reject_reason="allow")
    decision = decide_phase5_hybrid(
        method="casa_h_mpc_safedpa_casa_refine",
        context=context,
        casa_decision=casa,
        sota_registry=_FakeRegistry({"mpc_cbf_humanoid_adapted": False, "safedpa_adapted": False}),
        config=Phase5HybridConfig(gray_margin=0.06, outside_gray_anchor_veto_votes=2),
    )
    assert decision.allow is False
    assert decision.reject_reason == "hybrid_gray_refine_dual_anchor_veto"


def test_anchor_or_consensus_and_ensemble_variants() -> None:
    context = SotaDecisionContext(skill_name="turn", raw_critic_risk=0.20, hard_contract_score=0.0)
    casa = SimpleNamespace(reject=False, threshold=0.50, risk_margin=-0.30, reject_reason="allow")
    registry = _FakeRegistry({"mpc_cbf_humanoid_adapted": False, "safedpa_adapted": True})

    anchor_or = decide_phase5_hybrid(
        method="casa_h_mpc_safedpa_anchor_or",
        context=context,
        casa_decision=casa,
        sota_registry=registry,
        config=Phase5HybridConfig(),
    )
    consensus = decide_phase5_hybrid(
        method="casa_h_mpc_safedpa_consensus",
        context=context,
        casa_decision=casa,
        sota_registry=registry,
        config=Phase5HybridConfig(consensus_reject_votes=2),
    )
    ensemble = decide_phase5_hybrid(
        method="casa_h_mpc_safedpa_calibrated_ensemble",
        context=context,
        casa_decision=casa,
        sota_registry=registry,
        config=Phase5HybridConfig(ensemble_threshold=0.0),
    )

    assert anchor_or.allow is False
    assert consensus.allow is True
    assert ensemble.diagnostics["hybrid_strategy"] == "calibrated_ensemble"


def test_lowmem_lane_cmd_passes_hybrid_config() -> None:
    args = SimpleNamespace(
        phase4_root="phase4",
        phase5_root="phase5",
        startup_seconds=0.0,
        startup_timeout_seconds=1.0,
        pre_episode_settle_seconds=0.0,
        episode_boundary_stop_seconds=0.0,
        post_reset_idle_seconds=0.0,
        episode_upright_timeout_seconds=1.0,
        episode_upright_min_z=0.65,
        episode_upright_retries=0,
        initial_warmup_resets=0,
        initial_warmup_idle_seconds=0.0,
        episode_start_command_seconds=0.0,
        fallback_policy="auto",
        adaptive_retry_count=1,
        max_segment_duration=0.5,
        threshold_scale_global=1.0,
        threshold_scale_by_skill="",
        hybrid_config=Path("configs/phase5_hybrid_mpc_safedpa.yaml"),
        method_order_seed=0,
        segment_long_skills=False,
        recheck_before_segment=False,
        randomize_method_order=False,
    )
    job = {
        "method": "casa_h_mpc_safedpa_casa_refine",
        "seed": 3001,
        "start": 0,
        "end": 1,
        "output_dir": Path("/tmp/out"),
        "domain": 180,
        "port": 7800,
    }
    cmd = lane_cmd(args, job, "0")
    assert "--hybrid-config" in cmd
    assert "configs/phase5_hybrid_mpc_safedpa.yaml" in cmd


def test_hybrid_audit_reports_more_intervention_and_matched_budget() -> None:
    rows = [
        _summary("casa_h_mpc_safedpa_casa_refine", unsafe=0.10, progress=0.80, fallback=0.20, reject=0.30),
        _summary("mpc_cbf_humanoid_adapted", unsafe=0.20, progress=0.80, fallback=0.20, reject=0.30),
        _summary("safedpa_adapted", unsafe=0.30, progress=0.75, fallback=0.10, reject=0.10),
        _summary("casa_a_per_skill", unsafe=0.25, progress=0.70, fallback=0.20, reject=0.30),
    ]
    report = build_hybrid_audit(
        rows,
        hybrid_method="casa_h_mpc_safedpa_casa_refine",
        anchor_methods=["mpc_cbf_humanoid_adapted", "safedpa_adapted", "casa_a_per_skill"],
        intervention_tolerance_rel=0.05,
    )

    assert report["hybrid_beats_mpc_cbf_humanoid_adapted"] is True
    assert report["hybrid_beats_safedpa_adapted"] is True
    assert report["hybrid_beats_casa_a_per_skill"] is True
    assert report["matched_intervention_budget_passes_all_anchors"] is False
    assert report["hybrid_win_depends_on_more_fallback_or_reject"] is True
    assert report["pareto_frontier"]


class _FakeRegistry:
    def __init__(self, allow_by_method: dict[str, bool]) -> None:
        self.allow_by_method = allow_by_method

    def decide(self, method: str, context: SotaDecisionContext) -> GateDecision:
        allow = self.allow_by_method[method]
        return GateDecision(
            method_name=method,
            source_method=f"fake {method}",
            implementation_fidelity="lightweight_proxy",
            mode="gate",
            allow=allow,
            risk_score=-0.10 if allow else 0.10,
            threshold=0.0,
            fallback_mode=None if allow else "passive_stop",
            reject_reason="allow" if allow else f"{method}_reject",
            runtime_ms=0.01,
            solver_status="fake_ok" if allow else "fake_reject",
            diagnostics={"skill": context.skill_name},
        )


def _summary(method: str, *, unsafe: float, progress: float, fallback: float, reject: float) -> dict[str, object]:
    return {
        "method": method,
        "episode_count": 10,
        "unsafe_invocation_rate_per_episode": unsafe,
        "unsafe_invocation_count": int(unsafe * 10),
        "task_progress_success_rate": progress,
        "task_success_rate": progress,
        "fallback_rate_per_episode": fallback,
        "fallback_count": int(fallback * 10),
        "reject_rate_per_decision": reject,
        "decision_count": 80,
        "reject_count": int(reject * 80),
    }
