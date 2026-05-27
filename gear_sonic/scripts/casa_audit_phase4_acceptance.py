"""Audit CASA Phase 4 Dataset v1 and Raw Critic acceptance criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--raw-critic-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--go-no-go-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase4_root = args.phase4_root
    dataset_dir = args.dataset_dir or phase4_root / "dataset_v1"
    critic_dir = args.raw_critic_dir or phase4_root / "raw_critic"
    output_json = args.output_json or phase4_root / "phase4_acceptance_audit.json"
    output_md = args.output_md or phase4_root / "phase4_acceptance_audit.md"
    go_no_go_json = args.go_no_go_json or phase4_root / "phase4_go_no_go.json"

    dataset = _read_json(dataset_dir / "dataset_summary.json")
    split_audit = _read_json(dataset_dir / "split_audit.json")
    critic = _read_json(critic_dir / "metrics.json")

    dataset_checks = dataset.get("checks", {})
    critic_checks = critic.get("checks", {})
    blocking_reasons = _blocking_reasons(dataset_checks, critic_checks)
    warning_reasons = _warning_reasons(dataset, critic)
    strict_pass = not blocking_reasons
    audit = {
        "phase": "CASA Phase4 acceptance audit",
        "overall_status": "PASS_STRICT" if strict_pass else "STRICT_NO_GO",
        "strict_pass": strict_pass,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "dataset_summary": _compact_dataset(dataset),
        "raw_critic_summary": _compact_critic(critic),
        "split_audit": split_audit,
        "dataset_checks": dataset_checks,
        "raw_critic_checks": critic_checks,
    }
    go_no_go = _go_no_go(audit)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown(audit) + "\n")
    go_no_go_json.write_text(json.dumps(go_no_go, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def _blocking_reasons(dataset_checks: dict[str, Any], critic_checks: dict[str, Any]) -> list[str]:
    reasons = []
    for name, passed in dataset_checks.items():
        if name == "target_80k_met":
            continue
        if not passed:
            reasons.append(f"dataset_{name}_failed")
    for name, passed in critic_checks.items():
        if not passed:
            reasons.append(f"raw_critic_{name}_failed")
    return reasons


def _warning_reasons(dataset: dict[str, Any], critic: dict[str, Any]) -> list[str]:
    warnings = []
    if not dataset.get("checks", {}).get("target_80k_met", False):
        warnings.append("dataset_target_80k_not_met")
    if dataset.get("status") == "BOOTSTRAP_ONLY":
        warnings.append("dataset_bootstrap_only")
    test_auroc = critic.get("overall", {}).get("test", {}).get("auroc", 0.0)
    if 0.70 <= test_auroc < 0.75:
        warnings.append("raw_critic_overall_auroc_warning_0p70_0p75")
    for skill, stats in critic.get("per_skill", {}).get("test", {}).items():
        auroc = stats.get("auroc", 0.0)
        if stats.get("auroc_available") and 0.60 <= auroc < 0.65:
            warnings.append(f"raw_critic_{skill}_auroc_warning_0p60_0p65")
    return warnings


def _compact_dataset(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": data.get("status"),
        "strict_go": data.get("strict_go"),
        "selected_samples": data.get("selected_samples"),
        "target_samples_min": data.get("target_samples_min"),
        "target_samples_ideal": data.get("target_samples_ideal"),
        "raw_samples": data.get("raw_samples"),
        "usable_samples": data.get("usable_samples"),
        "clean_available_samples": data.get("clean_available_samples"),
        "excluded_samples": data.get("excluded_samples"),
        "runtime_stress_quarantine_samples": data.get("runtime_stress_quarantine_samples"),
        "excluded_reason_counts": data.get("excluded_reason_counts"),
        "positive_rate": data.get("positive_rate"),
        "label_missing_rate": data.get("label_missing_rate"),
        "per_skill": data.get("per_skill"),
        "per_split": data.get("per_split"),
        "deficits": data.get("deficits"),
    }


def _compact_critic(data: dict[str, Any]) -> dict[str, Any]:
    test = data.get("overall", {}).get("test", {})
    pareto = data.get("pareto_summary", {})
    return {
        "go_criteria_passed": data.get("go_criteria_passed"),
        "sample_count": data.get("sample_count"),
        "split_counts": data.get("split_counts"),
        "test_auroc": test.get("auroc"),
        "test_auprc": test.get("auprc"),
        "test_auprc_lift": test.get("auprc_lift"),
        "test_brier": test.get("brier"),
        "test_constant_brier": test.get("constant_brier"),
        "test_ece": test.get("ece"),
        "per_skill_test": data.get("per_skill", {}).get("test"),
        "raw_vs_hard_unsafe_reduction": pareto.get("raw_vs_hard_unsafe_reduction"),
        "raw_reduces_unsafe_ge_15pct_at_10_or_20": pareto.get("raw_reduces_unsafe_ge_15pct_at_10_or_20"),
    }


def _go_no_go(audit: dict[str, Any]) -> dict[str, Any]:
    dataset = audit["dataset_summary"]
    critic = audit["raw_critic_summary"]
    return {
        "phase": "CASA Phase4",
        "go": bool(audit["strict_pass"]),
        "status": audit["overall_status"],
        "blocking_reasons": audit["blocking_reasons"],
        "warning_reasons": audit["warning_reasons"],
        "dataset": {
            "selected_samples": dataset.get("selected_samples"),
            "clean_available_samples": dataset.get("clean_available_samples"),
            "target_samples_min": dataset.get("target_samples_min"),
            "target_samples_ideal": dataset.get("target_samples_ideal"),
            "positive_rate": dataset.get("positive_rate"),
            "deficits": dataset.get("deficits"),
            "excluded_reason_counts": dataset.get("excluded_reason_counts"),
        },
        "raw_critic": {
            "go_criteria_passed": critic.get("go_criteria_passed"),
            "test_auroc": critic.get("test_auroc"),
            "test_auprc": critic.get("test_auprc"),
            "test_brier": critic.get("test_brier"),
            "raw_vs_hard_unsafe_reduction": critic.get("raw_vs_hard_unsafe_reduction"),
        },
    }


def _markdown(audit: dict[str, Any]) -> str:
    dataset = audit["dataset_summary"]
    critic = audit["raw_critic_summary"]
    lines = [
        "# CASA Phase4 Acceptance Audit",
        "",
        f"- overall_status: `{audit['overall_status']}`",
        f"- strict_pass: `{audit['strict_pass']}`",
        f"- selected_samples: `{dataset.get('selected_samples')}` / min `{dataset.get('target_samples_min')}` / ideal `{dataset.get('target_samples_ideal')}`",
        f"- clean_available_samples: `{dataset.get('clean_available_samples')}`",
        f"- excluded_samples: `{dataset.get('excluded_samples')}`",
        f"- runtime_stress_quarantine_samples: `{dataset.get('runtime_stress_quarantine_samples')}`",
        f"- raw_critic_test_auroc: `{critic.get('test_auroc')}`",
        f"- raw_critic_test_auprc: `{critic.get('test_auprc')}`",
        f"- raw_critic_test_brier: `{critic.get('test_brier')}`",
        "",
        "## Blocking Reasons",
        "",
    ]
    if audit["blocking_reasons"]:
        lines.extend(f"- {reason}" for reason in audit["blocking_reasons"])
    else:
        lines.append("- none")
    lines.extend(["", "## Dataset Deficits", "", "```json", json.dumps(dataset.get("deficits"), indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Excluded Artifacts", "", "```json", json.dumps(dataset.get("excluded_reason_counts"), indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Raw Critic Checks", ""])
    for name, passed in audit["raw_critic_checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Dataset Checks", ""])
    for name, passed in audit["dataset_checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON {path}: {exc}") from exc


if __name__ == "__main__":
    main()
