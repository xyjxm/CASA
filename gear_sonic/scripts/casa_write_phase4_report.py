"""Write a concise CASA Phase 4 implementation report."""

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
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase4_root = args.phase4_root
    dataset_dir = args.dataset_dir or phase4_root / "dataset_v1"
    critic_dir = args.raw_critic_dir or phase4_root / "raw_critic"
    audit_json = args.audit_json or phase4_root / "phase4_acceptance_audit.json"
    output_md = args.output_md or phase4_root / "phase4_report.md"

    dataset = _read_json(dataset_dir / "dataset_summary.json")
    critic = _read_json(critic_dir / "metrics.json")
    audit = _read_json(audit_json)
    report = _markdown(dataset, critic, audit, phase4_root)
    output_md.write_text(report + "\n")
    print(str(output_md))


def _markdown(dataset: dict[str, Any], critic: dict[str, Any], audit: dict[str, Any], phase4_root: Path) -> str:
    test = critic.get("overall", {}).get("test", {})
    pareto = critic.get("pareto_summary", {})
    hc_filtered = audit.get("hard_contract_filtered_calibration")
    lines = [
        "# CASA Phase4 Implementation Report",
        "",
        "## Summary",
        "",
        f"Phase 4 strict tooling is implemented and run on the currently available Phase 3/Phase 2 sources. The current artifact status is `{audit.get('overall_status')}`: strict Go is granted only when the clean 50k dataset checks and Raw Critic checks both pass.",
        "",
        "## Artifacts",
        "",
        f"- phase4_root: `{phase4_root}`",
        f"- dataset: `{phase4_root / 'dataset_v1'}`",
        f"- raw_critic: `{phase4_root / 'raw_critic/raw_critic.pt'}`",
        f"- predictions: `{phase4_root / 'raw_critic/predictions.csv'}`",
        f"- pareto_curve: `{phase4_root / 'raw_critic/pareto_rejection_risk.csv'}`",
        f"- acceptance_audit: `{phase4_root / 'phase4_acceptance_audit.json'}`",
        f"- go_no_go: `{phase4_root / 'phase4_go_no_go.json'}`",
        "",
        "## Dataset",
        "",
        f"- selected_samples: `{dataset.get('selected_samples')}`",
        f"- clean_available_samples: `{dataset.get('clean_available_samples')}`",
        f"- excluded_samples: `{dataset.get('excluded_samples')}`",
        f"- runtime_stress_quarantine_samples: `{dataset.get('runtime_stress_quarantine_samples')}`",
        f"- raw_samples: `{dataset.get('raw_samples')}`",
        f"- usable_samples: `{dataset.get('usable_samples')}`",
        f"- positive_rate: `{dataset.get('positive_rate')}`",
        f"- status: `{dataset.get('status')}`",
        "",
        "| skill | total | unsafe | positive_rate |",
        "|---|---:|---:|---:|",
    ]
    for skill, stats in (dataset.get("per_skill") or {}).items():
        lines.append(f"| {skill} | {stats.get('total')} | {stats.get('unsafe')} | {stats.get('positive_rate'):.4f} |")
    if hc_filtered:
        lines.extend(
            [
                "",
                "## Hard-Contract-filtered Calibration Overlay",
                "",
                "Phase 4 Dataset v1 keeps the raw train/test/critic-val split intact, and the deployment-distribution calibration overlay supplies the conformal calibration rows required by Plan A.",
                "",
                f"- source: `{hc_filtered.get('source_path')}`",
                f"- calibration_distribution: `{hc_filtered.get('calibration_distribution')}`",
                f"- calibration_rows: `{hc_filtered.get('calibration_rows')}`",
                f"- usable_as_main_calibration: `{hc_filtered.get('usable_as_main_calibration')}`",
                "",
                "| skill | total | unsafe | hard_contract_rejected | dangerous_ge_min |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for skill, stats in (hc_filtered.get("per_skill") or {}).items():
            lines.append(
                f"| {skill} | {stats.get('total')} | {stats.get('unsafe')} | "
                f"{stats.get('hard_contract_rejected', 0)} | {stats.get('dangerous_ge_min')} |"
            )
    lines.extend(
        [
            "",
            "## Raw Critic",
            "",
            f"- test_AUROC: `{test.get('auroc')}`",
            f"- test_AUPRC: `{test.get('auprc')}`",
            f"- test_AUPRC_lift: `{test.get('auprc_lift')}`",
            f"- test_Brier: `{test.get('brier')}`",
            f"- constant_Brier: `{test.get('constant_brier')}`",
            f"- test_ECE: `{test.get('ece')}`",
            f"- go_criteria_passed: `{critic.get('go_criteria_passed')}`",
            "",
            "## Rejection-Risk",
            "",
            f"- raw_vs_hard_unsafe_reduction: `{json.dumps(pareto.get('raw_vs_hard_unsafe_reduction'), sort_keys=True)}`",
            f"- raw_reduces_unsafe_ge_15pct_at_10_or_20: `{pareto.get('raw_reduces_unsafe_ge_15pct_at_10_or_20')}`",
            "",
            "## Acceptance",
            "",
            f"- overall_status: `{audit.get('overall_status')}`",
            f"- blocking_reasons: `{json.dumps(audit.get('blocking_reasons'), ensure_ascii=False)}`",
            f"- excluded_reason_counts: `{json.dumps(dataset.get('excluded_reason_counts'), sort_keys=True)}`",
            "",
            "The important distinction is that the code path is now strict-clean end to end: runtime and injected-latency artifacts are quarantined, and the Phase 4 dataset gate is intentionally not waived when clean data is below 50k or per-skill requirements.",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON {path}: {exc}") from exc


if __name__ == "__main__":
    main()
