"""Write the CASA Phase 5 conformal baseline report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import MAIN_SKILLS, METHOD_ORDER, read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_md = args.output_md or args.phase5_root / "phase5_report.md"
    calibration = read_json(args.phase5_root / "calibration_audit.json")
    baseline = read_json(args.phase5_root / "baseline_summary.json")
    audit = read_json(args.phase5_root / "phase5_go_no_go.json")
    output_md.write_text(_markdown(args.phase4_root, args.phase5_root, calibration, baseline, audit) + "\n")
    print(str(output_md))


def _markdown(
    phase4_root: Path,
    phase5_root: Path,
    calibration: dict[str, Any],
    baseline: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    thresholds = calibration.get("thresholds", {})
    methods = baseline.get("methods", {})
    per_skill = baseline.get("per_skill", {})
    lines = [
        "# CASA Phase5 Implementation Report",
        "",
        "## Summary",
        "",
        "Phase 5 calibrates global and per-skill conformal gates on the "
        "untouched Phase 4 calibration split, independent from raw-critic "
        "model selection, then evaluates the five Version A baselines on the "
        f"Phase 4 test split. Current status is `{audit.get('status')}`.",
        "",
        "## Artifacts",
        "",
        f"- phase4_root: `{phase4_root}`",
        f"- phase5_root: `{phase5_root}`",
        f"- thresholds: `{phase5_root / 'conformal_thresholds.json'}`",
        f"- baseline_results: `{phase5_root / 'baseline_results.csv'}`",
        f"- per_skill_fnr: `{phase5_root / 'per_skill_fnr.csv'}`",
        f"- rejection_risk_curve: `{phase5_root / 'rejection_risk_curve.csv'}`",
        f"- go_no_go: `{phase5_root / 'phase5_go_no_go.json'}`",
        "",
        "## Thresholds",
        "",
        f"- alpha: `{calibration.get('alpha')}`",
        f"- global: `{thresholds.get('global')}`",
    ]
    for skill in MAIN_SKILLS:
        lines.append(f"- {skill}: `{thresholds.get('per_skill', {}).get(skill)}`")
    lines.extend(
        [
            "",
            "## Main Baselines",
            "",
            "| method | unsafe_invocations | FNR | reject_rate | safe_acceptance_rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_ORDER:
        row = methods.get(method, {})
        lines.append(
            f"| {row.get('method_display', method)} | {row.get('unsafe_invocation_count')} | "
            f"{_fmt(row.get('fnr'))} | {_fmt(row.get('reject_rate'))} | "
            f"{_fmt(row.get('safe_acceptance_rate', row.get('task_success_proxy')))} |"
        )
    lines.extend(["", "## CASA-A Per-skill FNR", "", "| skill | CASA-A FNR | Global FNR | Raw Critic FNR |", "|---|---:|---:|---:|"])
    for skill in MAIN_SKILLS:
        casa = per_skill.get("casa_a_per_skill", {}).get(skill, {})
        global_conf = per_skill.get("global_conformal", {}).get(skill, {})
        raw = per_skill.get("raw_critic_0p5", {}).get(skill, {})
        lines.append(f"| {skill} | {_fmt(casa.get('fnr'))} | {_fmt(global_conf.get('fnr'))} | {_fmt(raw.get('fnr'))} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- status: `{audit.get('status')}`",
            f"- go: `{audit.get('go')}`",
            f"- blocking_reasons: `{json.dumps(audit.get('blocking_reasons'), ensure_ascii=False)}`",
            f"- warning_reasons: `{json.dumps(audit.get('warning_reasons'), ensure_ascii=False)}`",
            f"- casa_vs_sonic_unsafe_reduction: `{baseline.get('casa_vs_sonic_unsafe_reduction')}`",
            f"- casa_vs_hard_unsafe_reduction: `{baseline.get('casa_vs_hard_unsafe_reduction')}`",
            f"- casa_vs_matched_hard_unsafe_reduction: `{baseline.get('casa_vs_matched_hard_unsafe_reduction')}`",
            f"- fixed_hard_comparison_mode: `{baseline.get('fixed_hard_comparison_mode')}`",
            f"- casa_safe_acceptance_drop_abs_vs_sonic: `{baseline.get('casa_safe_acceptance_drop_abs_vs_sonic')}`",
            f"- casa_safe_acceptance_drop_rel_vs_sonic: `{baseline.get('casa_safe_acceptance_drop_rel_vs_sonic')}`",
            f"- casa_vs_global_fnr_closer_skills: `{baseline.get('casa_vs_global_fnr_closer_skills')}`",
            "",
            "## Calibration Note",
            "",
            "The main conformal calibration uses the untouched Phase 4 calibration split, independent from the raw-critic validation split. The hard-contract-filtered subset is reported only as a diagnostic unless a matched safe-rejection-budget comparison is configured.",
        ]
    )
    smoke_summary = phase5_root / "online_smoke_dry_run" / "method_summary.json"
    if smoke_summary.exists():
        lines.extend(
            [
                "",
                "## Online Runner Smoke",
                "",
                f"- dry_run_smoke_summary: `{smoke_summary}`",
                f"- dry_run_gate_decisions: `{phase5_root / 'online_smoke_dry_run' / 'gate_decisions.csv'}`",
                f"- dry_run_episode_results: `{phase5_root / 'online_smoke_dry_run' / 'online_episode_results.csv'}`",
            ]
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
