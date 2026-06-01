from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gear_sonic.casa.phase5_online import build_online_audit, read_csv_rows
from gear_sonic.casa.phase5_online_diagnostics import build_failure_breakdown
from gear_sonic.casa.phase5_online_sweep import evaluate_sweep_candidates, pareto_candidates, sweep_report
from gear_sonic.casa.phase5_policy import evaluate_online_method, parse_threshold_scale_by_skill, scale_thresholds
from gear_sonic.casa.phase5_recovery import plan_recovery, rewrite_skill_for_retry, split_skill
from gear_sonic.casa.skills import GestureSkill, PassiveSkill, TurnSkill, WalkSkill
from gear_sonic.scripts import casa_run_phase5_online_main_lowmem as lowmem
from gear_sonic.scripts.casa_run_phase5_online_experiment import _execute_segment_with_policy

EPISODE_FIELDS = [
    "method",
    "method_display",
    "seed",
    "episode_index",
    "episode_id",
    "status",
    "error",
    "task_success",
    "fallback_count",
    "unsafe_invocation_count",
    "violation_count",
    "violation_types",
    "completion_time_s",
    "episode_dir",
    "initial_upright_ok",
    "reset_attempts",
]
DECISION_FIELDS = [
    "method",
    "candidate_skill",
    "candidate_params_json",
    "raw_critic_risk",
    "hard_contract_score",
    "hard_contract_fixed_reject",
    "threshold",
    "risk_margin",
    "decision",
    "reject_reason",
    "episode_id",
    "episode_index",
    "seed",
    "skill_idx",
    "original_skill_idx",
    "segment_idx",
    "segment_count",
    "executed_skill",
    "executed_params_json",
    "fallback_policy",
    "fallback_executed",
    "recovery_action",
    "recovery_reason",
    "recovery_retry_count",
    "result_status",
    "result_termination_reason",
]


def test_adaptive_retry_executes_recovery_then_retry() -> None:
    segment = WalkSkill(vx=0.8, vy=0.0, facing_yaw_deg=0.0, duration=1.0, speed=0.4)
    decisions: list[dict[str, object]] = []
    gate = _FakeGate([_gate_decision(segment, decision="allow", risk=0.2)])
    executor = _FakeExecutor()

    fallback_delta, next_skill_idx = _execute_segment_with_policy(
        executor=executor,
        gate=gate,
        method="casa_a_hard_or_receding_recovery",
        segment=segment,
        original_skill_idx=1,
        segment_idx=1,
        segment_count=1,
        initial_decision=_gate_decision(segment, decision="reject", risk=0.9),
        episode_id="casa_a_hard_or_receding_recovery__seed_1__episode_0000",
        episode_index=0,
        seed=1,
        sim_log_dir=None,
        scene_props={},
        fallback_policy="adaptive_retry",
        fallback_duration=0.5,
        adaptive_retry_count=1,
        next_skill_idx=1,
        decisions=decisions,
    )

    assert fallback_delta == 1
    assert next_skill_idx == 4
    assert [item["skill"].name for item in executor.executed] == ["walk", "walk"]
    assert [row["attempt_type"] for row in decisions] == [
        "recovery_attempt",
        "retry_decision",
        "retry_executed",
    ]
    assert decisions[-1]["retry_executed"] == 1
    assert decisions[-1]["final_segment_outcome"] == "recovery_then_retry"
    assert len(gate.remembered) == 2


def test_adaptive_retry_final_reject_after_max_attempts() -> None:
    segment = TurnSkill(face_yaw_deg=90.0, duration=1.0)
    decisions: list[dict[str, object]] = []
    gate = _FakeGate(
        [
            _gate_decision(segment, decision="reject", risk=0.8),
            _gate_decision(segment, decision="reject", risk=0.7),
        ]
    )
    executor = _FakeExecutor()

    fallback_delta, _next_skill_idx = _execute_segment_with_policy(
        executor=executor,
        gate=gate,
        method="casa_a_hard_or_receding_recovery",
        segment=segment,
        original_skill_idx=1,
        segment_idx=1,
        segment_count=1,
        initial_decision=_gate_decision(segment, decision="reject", risk=0.9),
        episode_id="casa_a_hard_or_receding_recovery__seed_1__episode_0000",
        episode_index=0,
        seed=1,
        sim_log_dir=None,
        scene_props={},
        fallback_policy="adaptive_retry",
        fallback_duration=0.5,
        adaptive_retry_count=1,
        next_skill_idx=1,
        decisions=decisions,
    )

    assert fallback_delta == 2
    assert [item["skill"].name for item in executor.executed] == ["turn", "turn"]
    assert "retry_executed" not in {row["attempt_type"] for row in decisions}
    assert decisions[-1]["attempt_type"] == "final_reject"
    assert decisions[-1]["final_segment_outcome"] == "recovery_only_final_reject"


def test_hard_or_receding_method_rejects_on_hard_contract() -> None:
    decision = evaluate_online_method(
        method="casa_a_hard_or_receding_recovery",
        skill_name="walk",
        raw_risk=0.1,
        hard_contract_fixed_reject=True,
        thresholds={"global": 0.8, "per_skill": {"walk": 0.6}},
    )

    assert decision.reject is True
    assert decision.reject_reason == "hard_contract_or_casa_threshold"


def test_hard_or_receding_method_rejects_on_casa_threshold() -> None:
    decision = evaluate_online_method(
        method="casa_a_hard_or_receding_recovery",
        skill_name="walk",
        raw_risk=0.61,
        hard_contract_fixed_reject=False,
        thresholds={"global": 0.8, "per_skill": {"walk": 0.6}},
    )

    assert decision.reject is True
    assert decision.reject_reason == "per_skill_conformal_threshold"


def test_unknown_threshold_scale_skill_fails_fast() -> None:
    thresholds = {"global": 0.8, "per_skill": {"walk": 0.6, "turn": 0.7, "gesture": 0.5}}

    scaled = scale_thresholds(
        thresholds,
        global_scale=1.0,
        by_skill=parse_threshold_scale_by_skill("walk=0.8,turn=0.9"),
    )

    assert scaled["per_skill"]["walk"] == pytest.approx(0.48)
    assert scaled["per_skill"]["turn"] == pytest.approx(0.63)
    with pytest.raises(ValueError, match="gestuer"):
        scale_thresholds(thresholds, by_skill=parse_threshold_scale_by_skill("gestuer=0.7"))
    for raw in ["walk=nan", "walk=inf", "walk=-0.1", "=0.5"]:
        with pytest.raises(ValueError):
            parse_threshold_scale_by_skill(raw)
    with pytest.raises(ValueError, match="per_skill"):
        scale_thresholds({"global": 0.8}, by_skill={})


def test_turn_segmentation_preserves_absolute_facing() -> None:
    turn = TurnSkill(face_yaw_deg=90.0, duration=2.0)
    segments = split_skill(turn, max_segment_duration=0.5)

    assert len(segments) == 4
    assert sum(segment.duration for segment in segments) == pytest.approx(2.0)
    assert {segment.face_yaw_deg for segment in segments} == {90.0}
    facings = [segment.command_at(0.0, {}).facing for segment in segments]
    assert facings == [facings[0]] * 4


def test_lowmem_performance_preset_sets_new_candidate() -> None:
    args = _lowmem_args(performance_preset="hard_or_receding_adaptive")

    lowmem.apply_performance_preset(args)

    assert args.methods == "sonic_only,hard_contract,casa_a_hard_or_receding_recovery"
    assert args.casa_method == "casa_a_hard_or_receding_recovery"
    assert args.fallback_policy == "auto"
    assert args.adaptive_retry_count == 2
    assert args.segment_long_skills is True
    assert args.recheck_before_segment is True
    assert args.threshold_scale_by_skill == "walk=0.8,turn=0.9,gesture=0.7,passive=1.0"
    assert args.randomize_method_order is True
    assert args.method_order_seed == 20260531


def test_lane_cmd_contains_effective_performance_flags() -> None:
    args = _lowmem_args(performance_preset="hard_or_receding_adaptive")
    lowmem.apply_performance_preset(args)
    job = {
        "method": "casa_a_hard_or_receding_recovery",
        "seed": 2001,
        "start": 0,
        "end": 10,
        "domain": 180,
        "port": 7800,
        "output_dir": Path("unused"),
    }

    cmd = lowmem.lane_cmd(args, job, "0")

    assert "--methods" in cmd
    assert cmd[cmd.index("--methods") + 1] == "casa_a_hard_or_receding_recovery"
    assert cmd[cmd.index("--fallback-policy") + 1] == "auto"
    assert cmd[cmd.index("--adaptive-retry-count") + 1] == "2"
    assert cmd[cmd.index("--threshold-scale-by-skill") + 1] == "walk=0.8,turn=0.9,gesture=0.7,passive=1.0"
    assert "--segment-long-skills" in cmd
    assert "--recheck-before-segment" in cmd
    assert "--randomize-method-order" in cmd


def test_hard_or_casa_and_threshold_scaling() -> None:
    thresholds = scale_thresholds(
        {"global": 0.8, "per_skill": {"walk": 0.6, "turn": 0.7}},
        global_scale=0.5,
        by_skill=parse_threshold_scale_by_skill("turn=0.25"),
    )

    assert thresholds["global"] == 0.4
    assert thresholds["per_skill"]["walk"] == 0.3
    assert thresholds["per_skill"]["turn"] == 0.175

    decision = evaluate_online_method(
        method="casa_a_hard_or_per_skill",
        skill_name="walk",
        raw_risk=0.1,
        hard_contract_fixed_reject=True,
        thresholds=thresholds,
    )

    assert decision.reject is True
    assert decision.reject_reason == "hard_contract_or_casa_threshold"


def test_adaptive_recovery_selection_and_segmentation() -> None:
    walk = WalkSkill(vx=0.8, vy=0.2, facing_yaw_deg=10.0, duration=3.0, speed=0.4)
    plan = plan_recovery(
        walk,
        fallback_policy="adaptive_retry",
        fallback_duration=0.75,
        max_retry_count=2,
        reject_reason="per_skill_conformal_threshold",
        raw_risk=0.9,
    )

    assert isinstance(plan.skill, WalkSkill)
    assert plan.skill.duration == 0.75
    assert plan.skill.speed == 0.2
    assert plan.retry_count == 2
    assert "shorten and slow walk" in plan.reason
    retry_walk = rewrite_skill_for_retry(
        walk,
        attempt_index=1,
        max_retry_count=2,
        fallback_duration=0.75,
        scene_props={},
    )
    assert isinstance(retry_walk, WalkSkill)
    assert abs(retry_walk.vx) <= abs(walk.vx)
    assert retry_walk.speed <= walk.speed
    assert retry_walk.duration <= walk.duration

    assert len(split_skill(walk, max_segment_duration=0.5)) == 6
    assert len(split_skill(TurnSkill(face_yaw_deg=90.0, duration=2.0), max_segment_duration=0.5)) == 4
    gesture_segments = split_skill(
        GestureSkill(amplitude=0.5, frequency=1.0, duration=1.0),
        max_segment_duration=2.0,
    )
    passive_recovery = plan_recovery(
        PassiveSkill(duration=1.0),
        fallback_policy="adaptive",
        fallback_duration=0.5,
    )
    assert len(gesture_segments) == 1
    assert isinstance(passive_recovery.skill, PassiveSkill)


def test_failure_breakdown_identifies_gate_and_fallback_failures(tmp_path: Path) -> None:
    online_dir = _write_perf_online_dir(tmp_path)
    breakdown = build_failure_breakdown(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        casa_method="casa_a_hard_or_recovery",
    )

    casa = breakdown["per_method"]["casa_a_hard_or_recovery"]
    assert casa["allow_then_unsafe"] == 1
    assert casa["reject_but_still_unsafe"] == 1
    assert casa["low_risk_unsafe_count"] == 1
    assert "attempt_type" in breakdown["diagnostics"]["missing_retry_fields"]
    categories = {item["category"] for item in breakdown["recommendations"]}
    assert "fallback_quality" in categories
    assert "online_distribution_shift" in categories
    assert breakdown["target_bucket"]["casa_a_hard_or_recovery"]["visual_collision_or_close"]["episodes"] == 2


def test_diagnose_cli_writes_breakdown_outputs(tmp_path: Path) -> None:
    online_dir = _write_perf_online_dir(tmp_path)
    output_dir = tmp_path / "diagnostics"

    subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/casa_diagnose_phase5_online_failures.py",
            "--online-dir",
            str(online_dir),
            "--output-dir",
            str(output_dir),
            "--casa-method",
            "casa_a_hard_or_recovery",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )

    assert (output_dir / "phase5_online_failure_breakdown.json").exists()
    report = (output_dir / "phase5_online_failure_breakdown.md").read_text()
    assert "Dominant Recommendations" in report


def test_sweep_report_finds_pareto_candidate(tmp_path: Path) -> None:
    online_dir = _write_perf_online_dir(tmp_path)
    rows = evaluate_sweep_candidates([("pilot_a", online_dir)])
    report = sweep_report(rows)

    assert report["evaluated_count"] == 1
    assert rows[0]["method"] == "casa_a_hard_or_recovery"
    assert rows[0]["unsafe_reduction_vs_sonic"] > 0
    assert pareto_candidates(rows)[0]["candidate_id"] == "pilot_a"


def test_sweep_cli_writes_planned_grid(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/casa_sweep_phase5_online_policy.py",
            "--output-dir",
            str(output_dir),
            "--fallback-policies",
            "adaptive",
            "--hard-or-casa",
            "true",
            "--segment-long-skills",
            "true",
            "--max-segment-durations",
            "0.5",
            "--threshold-scale-global",
            "0.8",
            "--threshold-scale-by-skill",
            "walk=0.8,turn=0.9",
            "--recovery-retry-counts",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )

    report = json.loads((output_dir / "phase5_online_policy_sweep.json").read_text())
    assert report["planned_count"] == 1
    assert (output_dir / "phase5_online_policy_sweep.csv").exists()


def test_lowmem_runner_clamps_cyclonedds_domain_pool() -> None:
    assert lowmem.safe_domain_pool_size(base_domain_id=180, requested_pool_size=70) == 53

    old_methods = list(lowmem.METHODS)
    old_seeds = list(lowmem.SEEDS)
    try:
        lowmem.METHODS = ["sonic_only"]
        lowmem.SEEDS = list(range(60))
        args = argparse.Namespace(
            online_root=Path("unused"),
            episodes_per_seed=1,
            chunk_size=1,
            base_zmq_port=8800,
            base_domain_id=180,
            domain_pool_size=lowmem.safe_domain_pool_size(180, 70),
            randomize_method_order=False,
            method_order_seed=0,
        )
        jobs = lowmem.build_jobs(args, defaultdict(set))
    finally:
        lowmem.METHODS = old_methods
        lowmem.SEEDS = old_seeds

    domains = [job["domain"] for job in jobs]
    assert max(domains) == 232
    assert 233 not in domains
    assert domains[52] == 232
    assert domains[53] == 180


def test_strict_online_audit_thresholds_not_weakened(tmp_path: Path) -> None:
    online_dir = _write_perf_online_dir(tmp_path, casa_success=0, casa_unsafe=3)
    audit = build_online_audit(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        expected_episodes=4,
        expected_methods=["sonic_only", "casa_a_hard_or_recovery"],
        expected_seeds=[1],
        episodes_per_seed=2,
        casa_method="casa_a_hard_or_recovery",
    )

    assert audit["status"] == "ONLINE_NO_GO"
    assert "casa_task_success_drop_rel_le_30pct" in audit["blocking_reasons"]


class _FakeGate:
    def __init__(self, decisions: list[dict[str, object]]) -> None:
        self.decisions = list(decisions)
        self.remembered: list[object] = []

    def decide(
        self,
        *,
        method: str,
        skill: object,
        sim_log_dir: Path | None,
        scene_props: dict[str, object],
    ) -> dict[str, object]:
        del method, sim_log_dir, scene_props
        decision = dict(self.decisions.pop(0))
        decision["candidate_skill"] = skill.name
        decision["candidate_params_json"] = json.dumps(skill.params(), sort_keys=True)
        return decision

    def remember_result(self, result: object) -> None:
        self.remembered.append(result)


class _FakeExecutor:
    def __init__(self) -> None:
        self.executed: list[dict[str, object]] = []

    def execute_one(self, skill: object, *, episode_id: str, skill_idx: int) -> SimpleNamespace:
        self.executed.append({"skill": skill, "episode_id": episode_id, "skill_idx": skill_idx})
        return SimpleNamespace(
            status="success",
            termination_reason="",
            skill_name=skill.name,
            params=skill.params(),
        )


def _gate_decision(skill: object, *, decision: str, risk: float) -> dict[str, object]:
    return {
        "method": "casa_a_hard_or_receding_recovery",
        "candidate_skill": skill.name,
        "candidate_params_json": json.dumps(skill.params(), sort_keys=True),
        "raw_critic_risk": risk,
        "hard_contract_score": 0.0,
        "hard_contract_fixed_reject": 0,
        "threshold": 0.5,
        "risk_margin": risk - 0.5,
        "decision": decision,
        "reject_reason": "per_skill_conformal_threshold" if decision == "reject" else "allow",
    }


def _lowmem_args(**overrides: object) -> argparse.Namespace:
    values = {
        "performance_preset": "custom",
        "phase4_root": "phase4",
        "phase5_root": "phase5",
        "online_root": Path("online"),
        "max_parallel": 1,
        "chunk_size": 10,
        "episodes_per_seed": 20,
        "methods": "sonic_only,hard_contract",
        "casa_method": "casa_a_per_skill",
        "seeds": "2001,2002",
        "base_domain_id": 180,
        "domain_pool_size": 2,
        "base_zmq_port": 7800,
        "cuda_devices": "0",
        "startup_seconds": 20.0,
        "startup_timeout_seconds": 180.0,
        "pre_episode_settle_seconds": 1.0,
        "episode_boundary_stop_seconds": 0.75,
        "post_reset_idle_seconds": 1.5,
        "episode_upright_timeout_seconds": 3.0,
        "episode_upright_min_z": 0.65,
        "episode_upright_retries": 2,
        "initial_warmup_resets": 1,
        "initial_warmup_idle_seconds": 4.0,
        "episode_start_command_seconds": 0.75,
        "max_sweeps": 1,
        "fallback_policy": "adaptive",
        "adaptive_retry_count": 1,
        "segment_long_skills": False,
        "max_segment_duration": 0.75,
        "recheck_before_segment": False,
        "threshold_scale_global": 1.0,
        "threshold_scale_by_skill": "",
        "randomize_method_order": False,
        "method_order_seed": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_perf_online_dir(
    tmp_path: Path,
    *,
    casa_success: int = 1,
    casa_unsafe: int = 2,
) -> Path:
    online_dir = tmp_path / "online"
    episode_rows = []
    decision_rows = []
    methods = {
        "sonic_only": {"unsafe": [2, 2], "success": [1, 1], "fallback": [0, 0]},
        "casa_a_hard_or_recovery": {"unsafe": [casa_unsafe, 0], "success": [casa_success, 0], "fallback": [1, 1]},
    }
    for method, values in methods.items():
        for episode_index in range(2):
            episode_dir = online_dir / method / f"episode_{episode_index:04d}"
            episode_rows.append(
                {
                    "method": method,
                    "method_display": method,
                    "seed": 1,
                    "episode_index": episode_index,
                    "episode_id": f"{method}__seed_1__episode_{episode_index:04d}",
                    "status": "completed",
                    "error": "",
                    "task_success": values["success"][episode_index],
                    "fallback_count": values["fallback"][episode_index],
                    "unsafe_invocation_count": values["unsafe"][episode_index],
                    "violation_count": values["unsafe"][episode_index],
                    "violation_types": "[]",
                    "completion_time_s": 10.0,
                    "episode_dir": str(episode_dir),
                    "initial_upright_ok": 1,
                    "reset_attempts": 1,
                }
            )
            decision_rows.extend(_decision_rows(method, episode_index, episode_dir))
    _write_csv(online_dir / "online_episode_results.csv", episode_rows, EPISODE_FIELDS)
    _write_csv(online_dir / "gate_decisions.csv", decision_rows, DECISION_FIELDS)
    return online_dir


def _decision_rows(method: str, episode_index: int, episode_dir: Path) -> list[dict[str, object]]:
    episode_id = f"{method}__seed_1__episode_{episode_index:04d}"
    labels = []
    rows = []
    for skill_idx, skill in enumerate(["walk", "turn"], start=1):
        reject = method != "sonic_only" and skill_idx == 2
        unsafe = method == "sonic_only" or (method != "sonic_only" and episode_index == 0)
        risk = 0.2 if skill_idx == 1 else 0.9
        rows.append(
            {
                "method": method,
                "candidate_skill": skill,
                "candidate_params_json": "{}",
                "raw_critic_risk": risk,
                "hard_contract_score": 1.0 if reject else 0.0,
                "hard_contract_fixed_reject": int(reject),
                "threshold": "" if method == "sonic_only" else 0.5,
                "risk_margin": "" if method == "sonic_only" else risk - 0.5,
                "decision": "reject" if reject else "allow",
                "reject_reason": "hard_contract_or_casa_threshold" if reject else "allow",
                "episode_id": episode_id,
                "episode_index": episode_index,
                "seed": 1,
                "skill_idx": skill_idx,
                "original_skill_idx": skill_idx,
                "segment_idx": 1,
                "segment_count": 1,
                "executed_skill": "passive" if reject else skill,
                "executed_params_json": "{}",
                "fallback_policy": "adaptive",
                "fallback_executed": int(reject),
                "recovery_action": "small_turn_recovery" if reject else "",
                "recovery_reason": "test" if reject else "",
                "recovery_retry_count": 1 if reject else 0,
                "result_status": "success",
                "result_termination_reason": "",
            }
        )
        labels.append(
            {
                "skill_idx": str(skill_idx),
                "skill_name": skill,
                "safe_label": "unsafe" if unsafe and (method == "sonic_only" or episode_index == 0) else "safe",
                "triggered_violation_types": ["near_collision"] if unsafe else [],
            }
        )
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "rollout_summary.json").write_text(
        json.dumps(
            {
                "skill_labels": labels,
                "scene_props": {
                    "target_bucket": "visual_collision_or_close",
                    "scene_complexity": "hard",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
