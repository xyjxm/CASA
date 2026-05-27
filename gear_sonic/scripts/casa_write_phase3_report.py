"""Write a compact CASA Phase 3 go/no-go report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--mini-critic-dir", type=Path, default=None)
    parser.add_argument("--counterfactual-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or args.phase3_root / "dataset"
    critic_dir = args.mini_critic_dir or args.phase3_root / "mini_critic"
    counterfactual_dir = args.counterfactual_dir or args.phase3_root / "counterfactual"
    dataset = _read_json(dataset_dir / "dataset_summary.json")
    critic = _read_json(critic_dir / "metrics.json")
    counterfactual = _read_json(counterfactual_dir / "counterfactual_summary.json")
    audit = _read_json(args.phase3_root / "phase3_acceptance_audit.json")

    checks = {
        "dataset_go": bool(dataset.get("go_criteria_passed")),
        "mini_critic_go": bool(critic.get("go_criteria_passed")),
        "counterfactual_complete": bool(counterfactual.get("complete_100x4x5")),
        "counterfactual_consistent": bool(counterfactual.get("repeat_consistency_ge_0p90")),
    }
    if audit:
        checks["leakage_audit_pass"] = audit.get("overall_status") == "PASS"
    go_no_go = {
        "phase": "CASA Phase3",
        "go": all(checks.values()),
        "checks": checks,
        "dataset_summary": _compact_dataset(dataset),
        "mini_critic_summary": _compact_critic(critic),
        "counterfactual_summary": _compact_counterfactual(counterfactual),
        "acceptance_audit_summary": _compact_audit(audit),
    }
    args.phase3_root.mkdir(parents=True, exist_ok=True)
    (args.phase3_root / "go_no_go.json").write_text(json.dumps(go_no_go, indent=2, sort_keys=True) + "\n")
    (args.phase3_root / "phase3_report.md").write_text(_markdown(go_no_go) + "\n")
    print(json.dumps(go_no_go, indent=2, sort_keys=True))


def _compact_dataset(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_samples": data.get("selected_samples"),
        "raw_samples": data.get("raw_samples"),
        "label_complete_rate": data.get("label_complete_rate"),
        "overall_positive_rate": data.get("overall_positive_rate"),
        "per_skill_label_counts": data.get("per_skill_label_counts"),
        "per_skill_positive_rate": data.get("per_skill_positive_rate"),
        "checks": data.get("checks", {}),
        "selection_warnings": data.get("selection_warnings", []),
        "recommendations": data.get("recommendations", []),
    }


def _compact_critic(data: dict[str, Any]) -> dict[str, Any]:
    test = data.get("overall", {}).get("test", {})
    return {
        "test_auroc": test.get("auroc"),
        "test_auroc_available": test.get("auroc_available"),
        "test_brier": test.get("brier"),
        "constant_brier": test.get("constant_brier"),
        "per_skill_test": data.get("per_skill", {}).get("test", {}),
        "checks": data.get("checks", {}),
    }


def _compact_counterfactual(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch_count": data.get("branch_count"),
        "completed_branch_count": data.get("completed_branch_count"),
        "mean_repeat_consistency": data.get("mean_repeat_consistency"),
        "label_counts": data.get("label_counts"),
        "complete_100x4x5": data.get("complete_100x4x5"),
        "repeat_consistency_ge_0p90": data.get("repeat_consistency_ge_0p90"),
    }


def _compact_audit(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {}
    return {
        "overall_status": data.get("overall_status"),
        "blocking_reasons": data.get("blocking_reasons", []),
        "instant_unsafe_rate_among_unsafe": data.get("dataset_audit", {}).get("instant_unsafe_rate_among_unsafe"),
        "top_shortcut_single_feature_auc": data.get("feature_audit", {}).get("top_shortcut_single_feature_auc"),
        "shuffle_label_test_auroc": data.get("critic_audit", {}).get("shuffle_label_test_auroc"),
        "leakage_analysis_passed": data.get("leakage_analysis", {}).get("passed"),
        "attempt_level_hang_rate_available": data.get("rollout_audit", {}).get("attempt_level_hang_rate_available"),
        "attempt_level_hang_rate": data.get("rollout_audit", {}).get("attempt_level_hang_rate"),
        "throughput_samples_per_hour_selected_summaries": data.get("rollout_audit", {}).get(
            "throughput_samples_per_hour_selected_summaries"
        ),
    }


def _markdown(go_no_go: dict[str, Any]) -> str:
    dataset = go_no_go["dataset_summary"]
    critic = go_no_go["mini_critic_summary"]
    counterfactual = go_no_go["counterfactual_summary"]
    audit = go_no_go.get("acceptance_audit_summary", {})
    lines = [
        "# CASA Phase3 Report",
        "",
        f"Overall: **{'GO' if go_no_go['go'] else 'NO-GO / INCOMPLETE'}**",
        "",
        "## Dataset",
        "",
        f"- selected_samples: `{dataset.get('selected_samples')}`",
        f"- raw_samples: `{dataset.get('raw_samples')}`",
        f"- label_complete_rate: `{dataset.get('label_complete_rate')}`",
        f"- overall_positive_rate: `{dataset.get('overall_positive_rate')}`",
        f"- checks: `{json.dumps(dataset.get('checks', {}), sort_keys=True)}`",
        f"- recommendations: `{', '.join(dataset.get('recommendations', []))}`",
        "",
        "| Skill | Safe | Unsafe | Positive Rate |",
        "|---|---:|---:|---:|",
    ]
    counts = dataset.get("per_skill_label_counts") or {}
    rates = dataset.get("per_skill_positive_rate") or {}
    for skill in ["walk", "turn", "gesture", "passive"]:
        skill_counts = counts.get(skill, {})
        lines.append(
            f"| {skill} | {skill_counts.get('safe')} | {skill_counts.get('unsafe')} | {rates.get(skill)} |"
        )
    lines.extend(
        [
            "",
            "## Mini Critic",
            "",
            f"- test_auroc: `{critic.get('test_auroc')}`",
            f"- test_brier: `{critic.get('test_brier')}`",
            f"- constant_brier: `{critic.get('constant_brier')}`",
            f"- checks: `{json.dumps(critic.get('checks', {}), sort_keys=True)}`",
            "",
            "| Skill | Test Count | Pos Rate | AUROC | Brier |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for skill, stats in (critic.get("per_skill_test") or {}).items():
        lines.append(
            f"| {skill} | {stats.get('count')} | {stats.get('positive_rate')} | {stats.get('auroc')} | {stats.get('brier')} |"
        )
    lines.extend(
        [
            "",
            "## Counterfactual",
            "",
            f"- branch_count: `{counterfactual.get('branch_count')}`",
            f"- completed_branch_count: `{counterfactual.get('completed_branch_count')}`",
            f"- mean_repeat_consistency: `{counterfactual.get('mean_repeat_consistency')}`",
            f"- checks: `complete={counterfactual.get('complete_100x4x5')}, "
            f"consistent={counterfactual.get('repeat_consistency_ge_0p90')}`",
            "",
            "## Acceptance Audit",
            "",
            f"- status: `{audit.get('overall_status')}`",
            f"- blocking_reasons: `{json.dumps(audit.get('blocking_reasons', []), ensure_ascii=False)}`",
            f"- leakage_analysis_passed: `{audit.get('leakage_analysis_passed')}`",
            f"- instant_unsafe_rate_among_unsafe: `{audit.get('instant_unsafe_rate_among_unsafe')}`",
            f"- top_shortcut_single_feature_auc: `{audit.get('top_shortcut_single_feature_auc')}`",
            f"- shuffle_label_test_auroc: `{audit.get('shuffle_label_test_auroc')}`",
            f"- attempt_level_hang_rate_available: `{audit.get('attempt_level_hang_rate_available')}`",
            f"- attempt_level_hang_rate: `{audit.get('attempt_level_hang_rate')}`",
            f"- throughput_samples_per_hour_selected_summaries: `{audit.get('throughput_samples_per_hour_selected_summaries')}`",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    main()
