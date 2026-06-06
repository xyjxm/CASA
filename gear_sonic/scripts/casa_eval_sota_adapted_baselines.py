"""Offline replay for CASA-adapted external SOTA baseline gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.baselines.sota_adapters import (  # noqa: E402
    SOTA_METHOD_DISPLAY,
    SOTA_METHOD_ORDER,
    SotaDecisionContext,
    build_sota_registry,
)
from gear_sonic.casa.phase5 import (  # noqa: E402
    MAIN_SKILLS,
    metrics_from_reject_flags,
    read_json,
    read_prediction_rows,
    split_role,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--thresholds-json", type=Path)
    parser.add_argument("--methods", default=",".join(SOTA_METHOD_ORDER))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.phase5_root / "sota_adapted_offline_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = args.predictions_csv or args.phase4_root / "raw_critic" / "predictions.csv"
    thresholds_json = args.thresholds_json or args.phase5_root / "conformal_thresholds.json"
    rows = read_prediction_rows(predictions_csv)
    thresholds = read_json(thresholds_json)["thresholds"]
    methods = _parse_methods(args.methods)

    calibration_rows = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
    split_rows = [row for row in rows if split_role(row.get("phase4_split")) == args.split]
    if args.max_rows > 0:
        split_rows = split_rows[: args.max_rows]
    if not split_rows:
        raise SystemExit(f"No rows found for split {args.split!r} in {predictions_csv}")

    registry = build_sota_registry()
    registry.calibrate(calibration_rows)
    decision_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    per_skill_rows: list[dict[str, Any]] = []

    for method in methods:
        method_decisions = [_decide_row(registry, method, row, thresholds) for row in split_rows]
        decision_rows.extend(method_decisions)
        reject_flags = [row["decision"] == "reject" for row in method_decisions]
        metrics = metrics_from_reject_flags(split_rows, reject_flags, method=method, split=args.split)
        metrics["method_display"] = SOTA_METHOD_DISPLAY.get(method, method)
        metrics_rows.append(metrics)
        for skill in MAIN_SKILLS:
            skill_pairs = [
                (source_row, decision_row)
                for source_row, decision_row in zip(split_rows, method_decisions)
                if source_row.get("skill_name") == skill
            ]
            if not skill_pairs:
                continue
            skill_source_rows = [pair[0] for pair in skill_pairs]
            skill_reject_flags = [pair[1]["decision"] == "reject" for pair in skill_pairs]
            skill_metrics = metrics_from_reject_flags(
                skill_source_rows,
                skill_reject_flags,
                method=method,
                split=args.split,
                skill_name=skill,
            )
            skill_metrics["method_display"] = SOTA_METHOD_DISPLAY.get(method, method)
            per_skill_rows.append(skill_metrics)

    split_manifest = _split_manifest(rows, predictions_csv, thresholds_json, args.split, methods)
    feature_schema = _feature_schema()
    summary = {
        "phase": "CASA SOTA adapted baseline offline replay",
        "predictions_csv": str(predictions_csv),
        "thresholds_json": str(thresholds_json),
        "split": args.split,
        "max_rows": args.max_rows,
        "methods": metrics_rows,
        "per_skill": per_skill_rows,
        "adapter_calibration": registry.metadata(),
        "split_manifest": split_manifest,
        "feature_schema": feature_schema,
        "notes": [
            "Offline replay uses CASA Phase4/Phase5 logged candidate decisions and labels.",
            "No online test labels are used for adapter calibration.",
            "Reported metrics are rerun metrics in the CASA score space, not copied paper numbers.",
        ],
    }

    write_csv(output_dir / "offline_replay_results.csv", decision_rows)
    write_csv(output_dir / "offline_replay_method_metrics.csv", metrics_rows)
    write_csv(output_dir / "offline_replay_per_skill_metrics.csv", per_skill_rows)
    write_json(output_dir / "offline_replay_metrics.json", summary)
    write_json(output_dir / "sota_split_manifest.json", split_manifest)
    write_json(output_dir / "sota_feature_schema.json", feature_schema)
    (output_dir / "offline_replay_report.md").write_text(_report_markdown(summary) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _decide_row(
    registry: Any,
    method: str,
    row: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    context = SotaDecisionContext(
        skill_name=str(row.get("skill_name", "")),
        skill_params={},
        raw_critic_risk=float(row["raw_critic_risk_float"]),
        hard_contract_score=float(row["hard_contract_score_float"]),
        hard_contract_fixed_reject=bool(row["hard_contract_fixed_reject_bool"]),
        thresholds=thresholds,
        row=row,
    )
    decision = registry.decide(method, context)
    allow = bool(decision.allow)
    label = int(row["label_int"])
    return {
        "sample_id": row.get("sample_id", ""),
        "phase4_split": row.get("phase4_split", ""),
        "method": method,
        "method_display": SOTA_METHOD_DISPLAY.get(method, method),
        "source_method": decision.source_method,
        "implementation_fidelity": decision.implementation_fidelity,
        "baseline_mode": decision.mode,
        "skill_name": row.get("skill_name", ""),
        "label": label,
        "safe_label": row.get("safe_label", ""),
        "raw_critic_risk": row.get("raw_critic_risk", ""),
        "hard_contract_score": row.get("hard_contract_score", ""),
        "hard_contract_fixed_reject": int(bool(row["hard_contract_fixed_reject_bool"])),
        "decision": "allow" if allow else "reject",
        "allow": int(allow),
        "reject": int(not allow),
        "unsafe_invocation_if_allowed": int(label == 1 and allow),
        "risk_score": "" if decision.risk_score is None else decision.risk_score,
        "threshold": "" if decision.threshold is None else decision.threshold,
        "risk_margin": "" if decision.risk_margin is None else decision.risk_margin,
        "fallback_mode": "" if decision.fallback_mode is None else decision.fallback_mode,
        "reject_reason": decision.reject_reason,
        "runtime_ms": "" if decision.runtime_ms is None else decision.runtime_ms,
        "solver_status": decision.solver_status,
        "diagnostics_json": json.dumps(decision.diagnostics, sort_keys=True),
    }


def _split_manifest(
    rows: list[dict[str, Any]],
    predictions_csv: Path,
    thresholds_json: Path,
    split: str,
    methods: list[str],
) -> dict[str, Any]:
    split_counts: dict[str, Any] = {}
    for role in ("train", "critic_val", "calibration", "test"):
        role_rows = [row for row in rows if split_role(row.get("phase4_split")) == role]
        split_counts[role] = {
            "count": len(role_rows),
            "unsafe": sum(int(row.get("label_int", 0)) for row in role_rows),
            "per_skill": {
                skill: sum(1 for row in role_rows if row.get("skill_name") == skill)
                for skill in MAIN_SKILLS
            },
        }
    return {
        "predictions_csv": str(predictions_csv),
        "thresholds_json": str(thresholds_json),
        "evaluated_split": split,
        "calibration_split": "calibration",
        "methods": methods,
        "split_counts": split_counts,
        "no_test_leakage_rule": "adapter thresholds are calibrated only on phase4_split == calibration",
    }


def _feature_schema() -> dict[str, Any]:
    return {
        "adapter_contract": {
            "input": [
                "state/shared_feature_vector_if_online",
                "candidate skill",
                "candidate skill parameters",
                "raw_critic_risk",
                "hard_contract_score",
                "hard_contract_fixed_reject",
                "optional obstacle/user proxy features",
            ],
            "output": [
                "allow/reject",
                "risk_score",
                "threshold",
                "fallback_mode",
                "runtime_ms",
                "solver_status",
                "diagnostics",
            ],
        },
        "offline_replay_fields": [
            "skill_name",
            "label",
            "raw_critic_risk",
            "hard_contract_score",
            "hard_contract_fixed_reject",
        ],
        "online_extra_fields": [
            "env/min_user_distance/current",
            "env/min_arm_user_distance/current",
            "env/min_obstacle_distance/current",
            "robot/torso_roll/current",
            "robot/torso_pitch/current",
            "candidate skill parameters",
        ],
    }


def _report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CASA Adapted SOTA Offline Replay",
        "",
        f"- predictions_csv: `{summary['predictions_csv']}`",
        f"- thresholds_json: `{summary['thresholds_json']}`",
        f"- split: `{summary['split']}`",
        "",
        "## Method Metrics",
        "",
        "| method | fidelity | count | unsafe_total | reject_rate | fnr | safe_acceptance_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fidelity = {
        method: meta["implementation_fidelity"]
        for method, meta in summary["adapter_calibration"].items()
    }
    for row in summary["methods"]:
        method = row["method"]
        lines.append(
            f"| {row.get('method_display', method)} | {fidelity.get(method, '')} | {row['count']} | "
            f"{row['unsafe_total']} | {float(row['reject_rate']):.4f} | {float(row['fnr']):.4f} | "
            f"{float(row['safe_acceptance_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fidelity Notes",
            "",
            "- These are CASA-adapted gate-only reruns in the Phase5 score space.",
            "- No external paper result numbers are copied into this report.",
            (
                "- `mpc_cbf_humanoid_adapted` is a reduced-order feasibility proxy, "
                "not the full Poisson safety-function MPC solver."
            ),
        ]
    )
    return "\n".join(lines)


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [method for method in methods if method not in SOTA_METHOD_ORDER]
    if unknown:
        raise SystemExit(f"Unknown SOTA adapted methods: {unknown}; valid methods are {SOTA_METHOD_ORDER}")
    if not methods:
        raise SystemExit("--methods must not be empty")
    return methods


if __name__ == "__main__":
    main()
