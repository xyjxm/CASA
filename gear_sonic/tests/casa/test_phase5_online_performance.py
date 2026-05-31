from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

from gear_sonic.scripts import casa_run_phase5_online_main_lowmem as lowmem
from gear_sonic.casa.phase5_online import build_online_audit, read_csv_rows
from gear_sonic.casa.phase5_online_diagnostics import build_failure_breakdown
from gear_sonic.casa.phase5_online_sweep import evaluate_sweep_candidates, pareto_candidates, sweep_report
from gear_sonic.casa.phase5_policy import evaluate_online_method, parse_threshold_scale_by_skill, scale_thresholds
from gear_sonic.casa.phase5_recovery import plan_recovery, split_skill
from gear_sonic.casa.skills import GestureSkill, PassiveSkill, TurnSkill, WalkSkill

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


def test_strict_online_audit_thresholds_are_not_weakened(tmp_path: Path) -> None:
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
