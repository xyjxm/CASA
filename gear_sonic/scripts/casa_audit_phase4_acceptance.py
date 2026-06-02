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
    parser.add_argument(
        "--hc-filtered-calibration-summary",
        type=Path,
        help=(
            "Optional Hard-Contract-filtered calibration build summary or Phase 5 "
            "calibration audit used to prove the deployment-distribution calibration gate."
        ),
    )
    parser.add_argument(
        "--require-hc-filtered-calibration",
        action="store_true",
        help="Fail strict Phase 4 acceptance unless the HC-filtered calibration overlay is usable.",
    )
    parser.add_argument("--min-hc-filtered-unsafe-per-skill", type=int, default=200)
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
    hc_filtered_calibration = _load_hc_filtered_calibration(
        args.hc_filtered_calibration_summary,
        min_unsafe_per_skill=args.min_hc_filtered_unsafe_per_skill,
    )

    dataset_checks = dataset.get("checks", {})
    critic_checks = critic.get("checks", {})
    blocking_reasons = _blocking_reasons(
        dataset_checks,
        critic_checks,
        hc_filtered_calibration,
        require_hc_filtered_calibration=args.require_hc_filtered_calibration,
    )
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
        "require_hc_filtered_calibration": bool(args.require_hc_filtered_calibration),
        "hard_contract_filtered_calibration": hc_filtered_calibration,
    }
    go_no_go = _go_no_go(audit)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown(audit) + "\n")
    go_no_go_json.write_text(json.dumps(go_no_go, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def _blocking_reasons(
    dataset_checks: dict[str, Any],
    critic_checks: dict[str, Any],
    hc_filtered_calibration: dict[str, Any] | None = None,
    *,
    require_hc_filtered_calibration: bool = False,
) -> list[str]:
    reasons = []
    for name, passed in dataset_checks.items():
        if name == "target_80k_met":
            continue
        if not passed:
            reasons.append(f"dataset_{name}_failed")
    for name, passed in critic_checks.items():
        if not passed:
            reasons.append(f"raw_critic_{name}_failed")
    if require_hc_filtered_calibration:
        if not hc_filtered_calibration:
            reasons.append("hc_filtered_calibration_missing")
        else:
            for name, passed in hc_filtered_calibration.get("checks", {}).items():
                if not passed:
                    reasons.append(f"hc_filtered_calibration_{name}_failed")
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
        "hard_contract_filtered_calibration": audit.get("hard_contract_filtered_calibration"),
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
    hc_filtered = audit.get("hard_contract_filtered_calibration")
    if hc_filtered:
        lines.extend(
            [
                "",
                "## Hard-Contract-filtered Calibration",
                "",
                f"- source: `{hc_filtered.get('source_path')}`",
                f"- calibration_distribution: `{hc_filtered.get('calibration_distribution')}`",
                f"- calibration_rows: `{hc_filtered.get('calibration_rows')}`",
                f"- usable_as_main_calibration: `{hc_filtered.get('usable_as_main_calibration')}`",
                "",
            ]
        )
        for name, passed in hc_filtered.get("checks", {}).items():
            lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
        lines.extend(["", "```json", json.dumps(hc_filtered.get("per_skill"), indent=2, sort_keys=True), "```"])
    return "\n".join(lines)


def _load_hc_filtered_calibration(path: Path | None, *, min_unsafe_per_skill: int) -> dict[str, Any] | None:
    if path is None:
        return None
    data = _read_json(path)
    if "hard_contract_filtered_calibration" in data:
        return _hc_from_calibration_audit(data, path, min_unsafe_per_skill=min_unsafe_per_skill)
    return _hc_from_build_summary(data, path, min_unsafe_per_skill=min_unsafe_per_skill)


def _hc_from_build_summary(data: dict[str, Any], path: Path, *, min_unsafe_per_skill: int) -> dict[str, Any]:
    per_skill = data.get("per_skill") or {}
    checks = data.get("checks") or {}
    normalized = {
        skill: {
            "total": int(stats.get("total", 0) or 0),
            "safe": int(stats.get("safe", 0) or 0),
            "unsafe": int(stats.get("unsafe", 0) or 0),
            "dangerous_ge_min": int(stats.get("unsafe", 0) or 0) >= min_unsafe_per_skill,
        }
        for skill, stats in per_skill.items()
    }
    return _hc_summary(
        source_path=path,
        calibration_distribution=data.get("calibration_distribution"),
        calibration_rows=int(data.get("calibration_rows", 0) or 0),
        per_skill=normalized,
        all_rows_hard_contract_accepted=bool(checks.get("all_calibration_rows_hard_contract_accepted")),
        min_unsafe_per_skill=min_unsafe_per_skill,
        extra={
            "phase": data.get("phase"),
            "combined_rows": data.get("combined_rows"),
            "method_filter": data.get("method_filter"),
            "output_predictions_csv": data.get("output_predictions_csv"),
        },
    )


def _hc_from_calibration_audit(data: dict[str, Any], path: Path, *, min_unsafe_per_skill: int) -> dict[str, Any]:
    hc = data.get("hard_contract_filtered_calibration") or {}
    per_skill = hc.get("by_skill") or {}
    normalized = {
        skill: {
            "total": int(stats.get("total", 0) or 0),
            "safe": int(stats.get("safe", 0) or 0),
            "unsafe": int(stats.get("unsafe", 0) or 0),
            "hard_contract_rejected": int(stats.get("hard_contract_rejected", 0) or 0),
            "dangerous_ge_min": int(stats.get("unsafe", 0) or 0) >= min_unsafe_per_skill,
        }
        for skill, stats in per_skill.items()
    }
    return _hc_summary(
        source_path=path,
        calibration_distribution=data.get("calibration_distribution"),
        calibration_rows=int(hc.get("total", 0) or 0),
        per_skill=normalized,
        all_rows_hard_contract_accepted=bool(hc.get("all_calibration_rows_hard_contract_accepted")),
        min_unsafe_per_skill=min_unsafe_per_skill,
        extra={
            "phase": data.get("phase"),
            "usable_as_main_calibration": hc.get("usable_as_main_calibration"),
            "hard_contract_rejected": hc.get("hard_contract_rejected"),
        },
    )


def _hc_summary(
    *,
    source_path: Path,
    calibration_distribution: Any,
    calibration_rows: int,
    per_skill: dict[str, dict[str, Any]],
    all_rows_hard_contract_accepted: bool,
    min_unsafe_per_skill: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    main_skills = {"walk", "turn", "gesture", "passive"}
    each_main_skill_present = main_skills.issubset(per_skill)
    each_skill_unsafe_ge_min = all(
        int(per_skill.get(skill, {}).get("unsafe", 0) or 0) >= min_unsafe_per_skill for skill in main_skills
    )
    calibration_rows_present = calibration_rows > 0
    checks = {
        "calibration_rows_present": calibration_rows_present,
        "all_calibration_rows_hard_contract_accepted": all_rows_hard_contract_accepted,
        "each_main_skill_present": each_main_skill_present,
        "each_skill_unsafe_ge_min": each_skill_unsafe_ge_min,
    }
    usable = all(checks.values())
    return {
        "source_path": str(source_path),
        "calibration_distribution": calibration_distribution,
        "calibration_rows": calibration_rows,
        "min_unsafe_per_skill": min_unsafe_per_skill,
        "per_skill": per_skill,
        "checks": checks,
        "usable_as_main_calibration": usable,
        **{key: value for key, value in extra.items() if value is not None},
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON {path}: {exc}") from exc


if __name__ == "__main__":
    main()
