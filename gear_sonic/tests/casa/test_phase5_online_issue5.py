from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gear_sonic.casa.phase5 import METHOD_ORDER
from gear_sonic.casa.phase5_online import (
    OnlineValidationError,
    build_online_audit,
    online_report_markdown,
    read_csv_rows,
)

SKILL_SEQUENCE = ["walk", "turn", "passive", "gesture", "walk", "turn", "passive", "walk"]
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
    "decision",
    "episode_id",
    "episode_index",
    "seed",
    "skill_idx",
    "executed_skill",
    "executed_params_json",
    "fallback_executed",
    "result_status",
    "result_termination_reason",
]


def test_online_audit_accepts_synthetic_go_case(tmp_path: Path) -> None:
    online_dir = _write_synthetic_online_dir(tmp_path, casa_unsafe=6, casa_task_success=2)
    audit = build_online_audit(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        expected_episodes=10,
        expected_methods=list(METHOD_ORDER),
        expected_seeds=[1],
        episodes_per_seed=2,
    )

    assert audit["go"] is True
    assert audit["status"] == "PASS_STRICT_ONLINE"
    assert not audit["blocking_reasons"]


def test_online_audit_rejects_actionable_no_go_case(tmp_path: Path) -> None:
    online_dir = _write_synthetic_online_dir(tmp_path, casa_unsafe=15, casa_task_success=0)
    audit = build_online_audit(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        expected_episodes=10,
        expected_methods=list(METHOD_ORDER),
        expected_seeds=[1],
        episodes_per_seed=2,
    )

    assert audit["go"] is False
    assert audit["status"] == "ONLINE_NO_GO"
    assert "casa_vs_sonic_unsafe_reduction_ge_40pct" in audit["blocking_reasons"]
    assert "casa_task_success_drop_rel_le_30pct" in audit["blocking_reasons"]
    assert audit["actionable_blockers"][0]["next_step"]


def test_online_audit_fails_fast_on_malformed_decisions(tmp_path: Path) -> None:
    online_dir = _write_synthetic_online_dir(tmp_path)
    decision_rows = read_csv_rows(online_dir / "gate_decisions.csv")
    decision_rows[0]["raw_critic_risk"] = "nan"
    _write_csv(online_dir / "gate_decisions.csv", decision_rows, DECISION_FIELDS)

    with pytest.raises(OnlineValidationError, match="raw_critic_risk must be finite"):
        build_online_audit(
            read_csv_rows(online_dir / "online_episode_results.csv"),
            read_csv_rows(online_dir / "gate_decisions.csv"),
            expected_episodes=10,
            expected_methods=list(METHOD_ORDER),
            expected_seeds=[1],
            episodes_per_seed=2,
        )


def test_missing_per_skill_coverage_blocks_strict_acceptance(tmp_path: Path) -> None:
    online_dir = _write_synthetic_online_dir(tmp_path)
    decision_rows = [
        row for row in read_csv_rows(online_dir / "gate_decisions.csv") if row["candidate_skill"] != "gesture"
    ]
    _write_csv(online_dir / "gate_decisions.csv", decision_rows, DECISION_FIELDS)

    audit = build_online_audit(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        expected_episodes=10,
        expected_methods=list(METHOD_ORDER),
        expected_seeds=[1],
        episodes_per_seed=2,
        skills_per_episode=7,
    )

    assert not audit["checks"]["per_skill_decision_coverage_present"]
    assert "per_skill_decision_coverage_present" in audit["blocking_reasons"]


def test_baseline_over_rejection_is_diagnostic_warning(tmp_path: Path) -> None:
    online_dir = _write_synthetic_online_dir(tmp_path, hard_fallback=8, casa_fallback=2)
    audit = build_online_audit(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        expected_episodes=10,
        expected_methods=list(METHOD_ORDER),
        expected_seeds=[1],
        episodes_per_seed=2,
    )

    assert "hard_contract_intervenes_more_than_casa_diagnostic" in audit["warning_reasons"]


def test_online_report_separates_blockers_warnings_and_diagnostics(tmp_path: Path) -> None:
    online_dir = _write_synthetic_online_dir(tmp_path, casa_unsafe=15, casa_task_success=0)
    audit = build_online_audit(
        read_csv_rows(online_dir / "online_episode_results.csv"),
        read_csv_rows(online_dir / "gate_decisions.csv"),
        expected_episodes=10,
        expected_methods=list(METHOD_ORDER),
        expected_seeds=[1],
        episodes_per_seed=2,
    )
    report = online_report_markdown(audit)

    assert "## Blockers" in report
    assert "## Warnings" in report
    assert "## Artifact Validation" in report
    assert "## Per-skill Diagnostics" in report


def test_online_audit_cli_writes_expected_outputs(tmp_path: Path) -> None:
    online_dir = _write_synthetic_online_dir(tmp_path, casa_unsafe=6, casa_task_success=2)
    output_dir = tmp_path / "audit"

    subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/casa_audit_phase5_online.py",
            "--online-dir",
            str(online_dir),
            "--output-dir",
            str(output_dir),
            "--expected-episodes",
            "10",
            "--expected-methods",
            ",".join(METHOD_ORDER),
            "--expected-seeds",
            "1",
            "--episodes-per-seed",
            "2",
            "--strict",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )

    assert (output_dir / "online_acceptance_audit.json").exists()
    assert (output_dir / "online_go_no_go.json").exists()
    assert (output_dir / "online_report.md").exists()
    go_no_go = json.loads((output_dir / "online_go_no_go.json").read_text())
    assert go_no_go["go"] is True


def _write_synthetic_online_dir(
    tmp_path: Path,
    *,
    casa_unsafe: int = 6,
    casa_task_success: int = 1,
    hard_fallback: int = 4,
    casa_fallback: int = 2,
) -> Path:
    online_dir = tmp_path / "online"
    rows = []
    decisions = []
    totals = {
        "sonic_only": {"unsafe": 16, "success": 2, "fallback": 0},
        "hard_contract": {"unsafe": 12, "success": 2, "fallback": hard_fallback},
        "raw_critic_0p5": {"unsafe": 8, "success": 2, "fallback": 2},
        "global_conformal": {"unsafe": 7, "success": 2, "fallback": 2},
        "casa_a_per_skill": {"unsafe": casa_unsafe, "success": casa_task_success, "fallback": casa_fallback},
    }
    for method in METHOD_ORDER:
        for episode_index in range(2):
            rows.append(
                _episode_row(
                    method,
                    seed=1,
                    episode_index=episode_index,
                    unsafe=totals[method]["unsafe"] // 2,
                    task_success=int(episode_index < totals[method]["success"]),
                    fallback=totals[method]["fallback"] // 2,
                    episode_dir=online_dir / method / f"episode_{episode_index:04d}",
                )
            )
            decisions.extend(
                _decision_rows(
                    method,
                    seed=1,
                    episode_index=episode_index,
                    rejected_count=totals[method]["fallback"] // 2,
                )
            )
    _write_csv(online_dir / "online_episode_results.csv", rows, EPISODE_FIELDS)
    _write_csv(online_dir / "gate_decisions.csv", decisions, DECISION_FIELDS)
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
    rejected_count: int,
) -> list[dict[str, object]]:
    rows = []
    episode_id = f"{method}__seed_{seed}__episode_{episode_index:04d}"
    for index, skill in enumerate(SKILL_SEQUENCE, start=1):
        reject = int(index <= rejected_count)
        rows.append(
            {
                "method": method,
                "candidate_skill": skill,
                "candidate_params_json": "{}",
                "raw_critic_risk": 0.8 if reject else 0.1,
                "hard_contract_score": 0.8 if reject else 0.1,
                "hard_contract_fixed_reject": reject,
                "threshold": "" if method in {"sonic_only", "hard_contract"} else 0.5,
                "decision": "reject" if reject else "allow",
                "episode_id": episode_id,
                "episode_index": episode_index,
                "seed": seed,
                "skill_idx": index,
                "executed_skill": "passive" if reject else skill,
                "executed_params_json": "{}",
                "fallback_executed": reject,
                "result_status": "success",
                "result_termination_reason": "",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
