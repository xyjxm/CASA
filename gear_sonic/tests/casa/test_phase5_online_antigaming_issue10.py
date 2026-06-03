from __future__ import annotations

import csv
import json
from pathlib import Path

from gear_sonic.casa.phase5 import METHOD_ORDER
from gear_sonic.casa.phase5_online import build_online_audit, online_report_markdown, read_csv_rows

SKILL_SEQUENCE = ["walk", "turn", "passive", "gesture", "walk", "turn", "passive", "walk"]


def test_strict_plan_a_requires_all_five_methods(tmp_path: Path) -> None:
    online_dir = _write_online_dir(tmp_path, methods=["sonic_only", "hard_contract", "casa_a_per_skill"])

    audit = _strict_audit(online_dir, expected_methods=["sonic_only", "hard_contract", "casa_a_per_skill"])

    claim = audit["diagnostics"]["strict_plan_a_claim"]
    assert audit["status"] == "STRICT_PLAN_A_NO_GO"
    assert "strict_plan_a_missing_original_five_methods" in claim["blocking_reasons"]


def test_strict_plan_a_requires_original_casa_method(tmp_path: Path) -> None:
    methods = ["sonic_only", "hard_contract", "raw_critic_0p5", "global_conformal", "casa_a_hard_or_recovery"]
    online_dir = _write_online_dir(tmp_path, methods=methods, casa_method="casa_a_hard_or_recovery")

    audit = _strict_audit(online_dir, expected_methods=methods, casa_method="casa_a_hard_or_recovery")

    claim = audit["diagnostics"]["strict_plan_a_claim"]
    assert claim["method_kind"] == "plan_a_plus_variant"
    assert "strict_plan_a_requires_original_casa_a_per_skill" in claim["blocking_reasons"]


def test_casa_vs_global_blocks_when_global_better(tmp_path: Path) -> None:
    online_dir = _write_online_dir(
        tmp_path,
        totals={
            "global_conformal": {"unsafe": 4, "success": 2, "fallback": 2},
            "casa_a_per_skill": {"unsafe": 8, "success": 2, "fallback": 2},
        },
    )

    audit = _strict_audit(online_dir)

    claim = audit["diagnostics"]["strict_plan_a_claim"]
    assert not claim["checks"]["casa_vs_global_advantage"]
    assert "casa_a_per_skill_does_not_outperform_global_conformal" in claim["blocking_reasons"]


def test_casa_vs_global_passes_when_casa_better(tmp_path: Path) -> None:
    online_dir = _write_online_dir(
        tmp_path,
        totals={
            "global_conformal": {"unsafe": 8, "success": 2, "fallback": 2},
            "casa_a_per_skill": {"unsafe": 4, "success": 2, "fallback": 2},
        },
    )

    audit = _strict_audit(online_dir)

    claim = audit["diagnostics"]["strict_plan_a_claim"]
    assert claim["checks"]["casa_vs_global_advantage"]
    assert "casa_a_per_skill_does_not_outperform_global_conformal" not in claim["blocking_reasons"]


def test_fallback_budget_blocks_high_fallback(tmp_path: Path) -> None:
    online_dir = _write_online_dir(
        tmp_path,
        totals={"casa_a_per_skill": {"unsafe": 4, "success": 2, "fallback": 8}},
        reject_plan={"casa_a_per_skill": {0: [1, 2, 3, 4], 1: [1, 2, 3, 4]}},
    )

    audit = _strict_audit(online_dir)

    claim = audit["diagnostics"]["strict_plan_a_claim"]
    assert not claim["checks"]["fallback_rate_budget"]
    assert "casa_fallback_rate_exceeds_strict_plan_a_budget" in claim["blocking_reasons"]


def test_walk_reject_budget_blocks_high_walk_reject(tmp_path: Path) -> None:
    online_dir = _write_online_dir(
        tmp_path,
        totals={"casa_a_per_skill": {"unsafe": 4, "success": 2, "fallback": 6}},
        reject_plan={"casa_a_per_skill": {0: [1, 5, 8], 1: [1, 5, 8]}},
    )

    audit = _strict_audit(
        online_dir,
        max_fallback_rate_per_episode=3.0,
        max_reject_rate=0.8,
        max_walk_reject_rate=0.75,
    )

    claim = audit["diagnostics"]["strict_plan_a_claim"]
    assert not claim["checks"]["walk_reject_rate_budget"]
    assert "casa_walk_reject_rate_exceeds_strict_plan_a_budget" in claim["blocking_reasons"]


def test_task_success_split_fields_present(tmp_path: Path) -> None:
    online_dir = _write_online_dir(tmp_path)

    audit = _strict_audit(online_dir)

    for row in audit["method_summary"].values():
        assert "safe_completion_rate" in row
        assert "task_progress_success_rate" in row
        assert "semantic_task_success_rate" in row
    assert audit["diagnostics"]["strict_plan_a_claim"]["checks"]["task_success_split_fields_present"]


def test_current_task_success_backward_compatibility(tmp_path: Path) -> None:
    online_dir = _write_online_dir(tmp_path)

    audit = _strict_audit(online_dir)

    casa = audit["method_summary"]["casa_a_per_skill"]
    assert "task_success_rate" in casa
    assert casa["safe_completion_rate"] == casa["task_success_rate"]
    assert casa["safe_completion_count"] == casa["task_success_count"]


def test_antigaming_report_section_written(tmp_path: Path) -> None:
    online_dir = _write_online_dir(tmp_path)
    audit = _strict_audit(online_dir)

    report = online_report_markdown(audit)

    assert "## Anti-gaming / Claim-validity Checks" in report
    assert "strict_plan_a_claim_status" in report
    assert "matched_budget_winner" in report


def _strict_audit(
    online_dir: Path,
    *,
    expected_methods: list[str] | None = None,
    casa_method: str = "casa_a_per_skill",
    max_fallback_rate_per_episode: float = 2.0,
    max_reject_rate: float = 0.50,
    max_walk_reject_rate: float = 0.75,
) -> dict[str, object]:
    expected_methods = expected_methods or list(METHOD_ORDER)
    return build_online_audit(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        expected_episodes=len(expected_methods) * 2,
        expected_methods=expected_methods,
        expected_seeds=[1],
        episodes_per_seed=2,
        casa_method=casa_method,
        strict_plan_a_claim=True,
        max_fallback_rate_per_episode=max_fallback_rate_per_episode,
        max_reject_rate=max_reject_rate,
        max_walk_reject_rate=max_walk_reject_rate,
    )


def _write_online_dir(
    tmp_path: Path,
    *,
    methods: list[str] | None = None,
    casa_method: str = "casa_a_per_skill",
    totals: dict[str, dict[str, int]] | None = None,
    reject_plan: dict[str, dict[int, list[int]]] | None = None,
) -> Path:
    online_dir = tmp_path / "online"
    methods = methods or list(METHOD_ORDER)
    base_totals = {
        "sonic_only": {"unsafe": 16, "success": 2, "fallback": 0},
        "hard_contract": {"unsafe": 12, "success": 2, "fallback": 2},
        "raw_critic_0p5": {"unsafe": 10, "success": 2, "fallback": 2},
        "global_conformal": {"unsafe": 8, "success": 2, "fallback": 2},
        casa_method: {"unsafe": 4, "success": 2, "fallback": 2},
    }
    for method, values in (totals or {}).items():
        base_totals[method] = values
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for method in methods:
        values = base_totals[method]
        fallback_per_episode = values["fallback"] // 2
        for episode_index in range(2):
            episode_dir = online_dir / method / f"episode_{episode_index:04d}"
            rows.append(
                _episode_row(
                    method,
                    seed=1,
                    episode_index=episode_index,
                    unsafe=values["unsafe"] // 2,
                    task_success=int(episode_index < values["success"]),
                    fallback=fallback_per_episode,
                    episode_dir=episode_dir,
                )
            )
            rejected = (reject_plan or {}).get(method, {}).get(
                episode_index,
                list(range(1, fallback_per_episode + 1)),
            )
            decisions.extend(_decision_rows(method, seed=1, episode_index=episode_index, rejected=rejected))
            _write_rollout_summary(episode_dir)
    _write_csv(online_dir / "online_episode_results.csv", rows)
    _write_csv(online_dir / "gate_decisions.csv", decisions)
    return online_dir


def _episode_row(
    method: str,
    *,
    seed: int,
    episode_index: int,
    unsafe: int,
    task_success: int,
    fallback: int,
    episode_dir: Path,
) -> dict[str, object]:
    return {
        "method": method,
        "method_display": method,
        "seed": seed,
        "episode_index": episode_index,
        "episode_id": f"{method}__seed_{seed}__episode_{episode_index:04d}",
        "status": "completed",
        "error": "",
        "task_success": task_success,
        "fallback_count": fallback,
        "unsafe_invocation_count": unsafe,
        "violation_count": unsafe,
        "violation_types": "[]",
        "completion_time_s": 10.0,
        "episode_dir": str(episode_dir),
        "initial_upright_ok": 1,
        "reset_attempts": 1,
    }


def _decision_rows(
    method: str,
    *,
    seed: int,
    episode_index: int,
    rejected: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    episode_id = f"{method}__seed_{seed}__episode_{episode_index:04d}"
    rejected_set = set(rejected)
    for index, skill in enumerate(SKILL_SEQUENCE, start=1):
        reject = int(index in rejected_set)
        rows.append(
            {
                "method": method,
                "candidate_skill": skill,
                "candidate_params_json": "{}",
                "raw_critic_risk": 0.8 if reject else 0.1,
                "hard_contract_score": 0.8 if reject else 0.1,
                "hard_contract_fixed_reject": reject,
                "threshold": "" if method in {"sonic_only", "hard_contract"} else 0.5,
                "risk_margin": "" if method in {"sonic_only", "hard_contract"} else (0.3 if reject else -0.4),
                "decision": "reject" if reject else "allow",
                "reject_reason": "per_skill_conformal_threshold" if reject else "allow",
                "episode_id": episode_id,
                "episode_index": episode_index,
                "seed": seed,
                "skill_idx": index,
                "original_skill_idx": index,
                "segment_idx": 1,
                "segment_count": 1,
                "executed_skill": "passive" if reject else skill,
                "executed_params_json": "{}",
                "fallback_policy": "stop",
                "fallback_executed": reject,
                "task_progress_executed": int(not reject and skill in {"walk", "turn", "gesture"}),
                "result_status": "success",
                "result_termination_reason": "",
            }
        )
    return rows


def _write_rollout_summary(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    labels = [
        {"skill_idx": index, "skill_name": skill, "safe_label": "safe", "triggered_violation_types": []}
        for index, skill in enumerate(SKILL_SEQUENCE, start=1)
    ]
    (path / "rollout_summary.json").write_text(json.dumps({"skill_labels": labels}) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
