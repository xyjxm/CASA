from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gear_sonic.casa.phase5 import METHOD_ORDER
from gear_sonic.casa.phase5_online import method_summary_rows
from gear_sonic.casa.phase5_visualization import (
    MetricConsistencyError,
    check_metric_consistency,
    load_visualization_inputs,
    run_visualization_package,
    select_representative_episodes,
)

SKILL_SEQUENCE = ["walk", "turn", "gesture", "passive"]


def test_visualization_loads_minimal_artifacts(tmp_path: Path) -> None:
    artifact_dir, audit_dir = _write_artifacts(tmp_path)

    loaded = load_visualization_inputs(
        artifact_dir=artifact_dir,
        audit_dir=audit_dir,
        methods=list(METHOD_ORDER),
        casa_method="casa_a_per_skill",
        strict_five_baseline=True,
    )

    assert len(loaded.episode_rows) == len(METHOD_ORDER) * 3
    assert len(loaded.decision_rows) == len(METHOD_ORDER) * 3 * len(SKILL_SEQUENCE)
    assert set(loaded.method_summary) == set(METHOD_ORDER)


def test_metric_consistency_check_passes(tmp_path: Path) -> None:
    artifact_dir, audit_dir = _write_artifacts(tmp_path)
    loaded = _load(artifact_dir, audit_dir)

    result = check_metric_consistency(loaded)

    assert result["status"] == "PASS"


def test_metric_consistency_check_fails_on_mismatch(tmp_path: Path) -> None:
    artifact_dir, audit_dir = _write_artifacts(tmp_path)
    summary_path = audit_dir / "method_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["methods"]["sonic_only"]["unsafe_invocation_count"] = 999
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with pytest.raises(MetricConsistencyError):
        run_visualization_package(
            artifact_dir=artifact_dir,
            audit_dir=audit_dir,
            output_dir=tmp_path / "viz",
            methods=list(METHOD_ORDER),
            casa_method="casa_a_per_skill",
            max_example_episodes=1,
            video_mode="timeline-only",
            frames_dir=None,
            videos_dir=None,
            fps=5,
            write_html=False,
            write_markdown=True,
            fail_on_metric_mismatch=True,
            strict_five_baseline=True,
        )
    check_path = tmp_path / "viz" / "data" / "metric_consistency_check.json"
    assert json.loads(check_path.read_text())["status"] == "FAIL"


def test_select_representative_episodes(tmp_path: Path) -> None:
    artifact_dir, audit_dir = _write_artifacts(tmp_path)
    loaded = _load(artifact_dir, audit_dir)

    selected = select_representative_episodes(
        loaded.episode_rows,
        loaded.decision_rows,
        methods=list(METHOD_ORDER),
        casa_method="casa_a_per_skill",
        max_examples=5,
    )
    reasons = {row["selection_reason"] for row in selected}

    assert "sonic_unsafe_casa_safe_successful" in reasons
    assert "global_conformal_better_than_casa" in reasons


def test_bar_chart_outputs_created(tmp_path: Path) -> None:
    artifact_dir, audit_dir = _write_artifacts(tmp_path)
    output_dir = tmp_path / "viz_cli"
    completed = subprocess.run(
        [
            sys.executable,
            "gear_sonic/scripts/casa_visualize_phase5_online.py",
            "--artifact-dir",
            str(artifact_dir),
            "--audit-dir",
            str(audit_dir),
            "--output-dir",
            str(output_dir),
            "--methods",
            ",".join(METHOD_ORDER),
            "--casa-method",
            "casa_a_per_skill",
            "--max-example-episodes",
            "1",
            "--video-mode",
            "timeline-only",
            "--write-markdown",
            "--fail-on-metric-mismatch",
            "--strict-five-baseline",
            "--fps",
            "5",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "figures" / "five_baseline_unsafe_counts.png").exists()
    assert (output_dir / "figures" / "five_baseline_task_success_rates.png").exists()
    assert (output_dir / "figures" / "five_baseline_fallback_rates.png").exists()


def test_heatmap_outputs_created(tmp_path: Path) -> None:
    output_dir = _run_package(tmp_path)

    assert (output_dir / "figures" / "per_skill_unsafe_heatmap.png").exists()
    assert (output_dir / "figures" / "per_skill_fallback_reject_heatmap.png").exists()
    assert (output_dir / "figures" / "episode_method_outcome_heatmap.png").exists()


def test_timeline_animation_fallback_created(tmp_path: Path) -> None:
    output_dir = _run_package(tmp_path)

    animations = list((output_dir / "videos").glob("selected_episode_metric_timeline_*"))
    assert animations
    report = (output_dir / "phase5_online_visualization_report.md").read_text()
    assert "Metric timeline animation, not simulator recording" in report


def test_report_contains_metric_provenance(tmp_path: Path) -> None:
    output_dir = _run_package(tmp_path)

    report = (output_dir / "phase5_online_visualization_report.md").read_text()
    assert "## Metric provenance" in report
    assert "online_episode_results.csv:unsafe_invocation_count" in report


def test_report_contains_antigaming_visualization_section(tmp_path: Path) -> None:
    output_dir = _run_package(tmp_path)

    report = (output_dir / "phase5_online_visualization_report.md").read_text()
    assert "## Anti-Gaming / Claim-Validity Visualization" in report
    assert "does not override or weaken the strict claim audit" in report


def test_no_fake_video_claim(tmp_path: Path) -> None:
    output_dir = _run_package(tmp_path)

    report = (output_dir / "phase5_online_visualization_report.md").read_text()
    assert "Metric timeline animation, not simulator recording" in report
    assert "real simulator recording" not in report


def _run_package(tmp_path: Path) -> Path:
    artifact_dir, audit_dir = _write_artifacts(tmp_path)
    output_dir = tmp_path / "viz"
    run_visualization_package(
        artifact_dir=artifact_dir,
        audit_dir=audit_dir,
        output_dir=output_dir,
        methods=list(METHOD_ORDER),
        casa_method="casa_a_per_skill",
        max_example_episodes=1,
        video_mode="timeline-only",
        frames_dir=None,
        videos_dir=None,
        fps=5,
        write_html=True,
        write_markdown=True,
        fail_on_metric_mismatch=True,
        strict_five_baseline=True,
    )
    return output_dir


def _load(artifact_dir: Path, audit_dir: Path):
    return load_visualization_inputs(
        artifact_dir=artifact_dir,
        audit_dir=audit_dir,
        methods=list(METHOD_ORDER),
        casa_method="casa_a_per_skill",
        strict_five_baseline=True,
    )


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    artifact_dir = tmp_path / "artifact"
    audit_dir = artifact_dir / "audit"
    artifact_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    (artifact_dir / "expected_methods.txt").write_text("\n".join(METHOD_ORDER) + "\n")
    (artifact_dir / "expected_seeds.txt").write_text("1\n")

    episode_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    totals = {
        0: {
            "sonic_only": (4, 0, 0),
            "hard_contract": (1, 0, 3),
            "raw_critic_0p5": (2, 0, 1),
            "global_conformal": (1, 1, 1),
            "casa_a_per_skill": (0, 1, 1),
        },
        1: {
            "sonic_only": (3, 0, 0),
            "hard_contract": (2, 0, 3),
            "raw_critic_0p5": (2, 0, 1),
            "global_conformal": (0, 1, 1),
            "casa_a_per_skill": (2, 0, 3),
        },
        2: {
            "sonic_only": (0, 1, 0),
            "hard_contract": (0, 1, 0),
            "raw_critic_0p5": (0, 1, 0),
            "global_conformal": (0, 1, 0),
            "casa_a_per_skill": (0, 1, 0),
        },
    }
    for episode_index, method_totals in totals.items():
        for method in METHOD_ORDER:
            unsafe, task_success, fallback = method_totals[method]
            episode_rows.append(_episode_row(method, episode_index, unsafe, task_success, fallback))
            decision_rows.extend(_decision_rows(method, episode_index, fallback))
    _write_csv(audit_dir / "online_episode_results.csv", episode_rows)
    _write_csv(audit_dir / "gate_decisions.csv", decision_rows)
    summary_rows = method_summary_rows(episode_rows, decision_rows)
    summary = {"methods": {row["method"]: row for row in summary_rows}}
    (audit_dir / "method_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(audit_dir / "method_summary.csv", summary_rows)
    audit = {
        "status": "PASS_STRICT_ONLINE",
        "go": True,
        "episode_count": len(episode_rows),
        "gate_decision_count": len(decision_rows),
        "diagnostics": {
            "deduped_episode_row_count": len(episode_rows),
            "deduped_decision_row_count": len(decision_rows),
            "baseline_comparisons": {},
            "fallback_reject_budget": {
                "fallback_rate_per_episode": summary["methods"]["casa_a_per_skill"]["fallback_rate_per_episode"],
                "reject_rate_per_decision": 4 / 12,
                "walk_reject_rate": 2 / 3,
            },
            "matched_budget_comparison": {"winner": "global_conformal"},
            "strict_plan_a_claim": {"status": "STRICT_PLAN_A_NO_GO", "go": False},
            "per_skill_online_labels": {
                "by_method_skill": {
                    method: {
                        skill: {"total": 3, "safe": 2, "unsafe": int(method == "sonic_only" and skill == "walk")}
                        for skill in ["walk", "turn", "gesture", "passive"]
                    }
                    for method in METHOD_ORDER
                }
            },
        },
    }
    (audit_dir / "online_acceptance_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return artifact_dir, audit_dir


def _episode_row(
    method: str, episode_index: int, unsafe: int, task_success: int, fallback: int
) -> dict[str, object]:
    return {
        "method": method,
        "method_display": method,
        "seed": 1,
        "episode_index": episode_index,
        "episode_id": f"{method}__seed_1__episode_{episode_index:04d}",
        "status": "completed",
        "error": "",
        "task_success": task_success,
        "fallback_count": fallback,
        "unsafe_invocation_count": unsafe,
        "violation_count": unsafe,
        "violation_types": "[]",
        "completion_time_s": 10.0 + episode_index,
        "episode_dir": "",
        "initial_upright_ok": 1,
    }


def _decision_rows(method: str, episode_index: int, fallback_count: int) -> list[dict[str, object]]:
    rows = []
    rejected = set(range(1, fallback_count + 1))
    for skill_idx, skill in enumerate(SKILL_SEQUENCE, start=1):
        reject = int(skill_idx in rejected)
        rows.append(
            {
                "method": method,
                "candidate_skill": skill,
                "candidate_params_json": "{}",
                "raw_critic_risk": 0.8 if reject else 0.2,
                "hard_contract_score": 0.7 if reject else 0.1,
                "hard_contract_fixed_reject": reject,
                "threshold": "" if method == "sonic_only" else 0.5,
                "risk_margin": "" if method == "sonic_only" else (0.3 if reject else -0.3),
                "decision": "reject" if reject else "allow",
                "reject_reason": "synthetic_reject" if reject else "allow",
                "episode_id": f"{method}__seed_1__episode_{episode_index:04d}",
                "episode_index": episode_index,
                "seed": 1,
                "skill_idx": skill_idx,
                "executed_skill": "passive" if reject else skill,
                "fallback_executed": reject,
                "task_progress_executed": int(not reject and skill in {"walk", "turn", "gesture"}),
                "result_status": "success",
            }
        )
    return rows


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
