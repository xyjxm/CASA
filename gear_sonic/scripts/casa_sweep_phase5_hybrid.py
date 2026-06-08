"""Sweep CASA-Hybrid configs on non-final Phase 4 calibration rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5_hybrid import HYBRID_METHOD_STRATEGIES, load_phase5_hybrid_config  # noqa: E402
from gear_sonic.scripts.casa_calibrate_phase5_hybrid import (  # noqa: E402
    _calibrated_registry,
    _replay_hybrid_rows,
    _write_json,
    _write_yaml,
)
from gear_sonic.casa.phase5 import metrics_from_reject_flags, read_json, read_prediction_rows, split_role  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/phase5_hybrid_mpc_safedpa.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gray-margins", default="0.04,0.06,0.08")
    parser.add_argument("--ensemble-thresholds", default="-0.03,0.0,0.03")
    parser.add_argument(
        "--strategies",
        default="gray_refine,budgeted_selector,calibrated_ensemble,consensus,anchor_or",
    )
    parser.add_argument("--emit-run-commands", action="store_true")
    parser.add_argument("--online-root", type=Path)
    parser.add_argument("--seeds", default="3001,3002,3003,3004,3005")
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_prediction_rows(args.phase4_root / "raw_critic" / "predictions.csv")
    calibration_rows = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
    thresholds = read_json(args.phase5_root / "conformal_thresholds.json")["thresholds"]
    registry, registry_summary = _calibrated_registry(rows)
    base_config = load_phase5_hybrid_config(args.base_config)
    candidates = _candidate_configs(args, base_config.to_dict())
    result_rows = []
    for candidate in candidates:
        config_path = args.output_dir / "configs" / f"{candidate['candidate_id']}.yaml"
        _write_yaml(config_path, candidate["config"])
        config = load_phase5_hybrid_config(config_path)
        method = "casa_h_mpc_safedpa_casa_refine"
        decisions = _replay_hybrid_rows(calibration_rows, thresholds, registry, config, method)
        reject_flags = [row["decision"] == "reject" for row in decisions]
        metric = metrics_from_reject_flags(
            calibration_rows,
            reject_flags,
            method=method,
            split="calibration",
        )
        metric.update(
            {
                "candidate_id": candidate["candidate_id"],
                "strategy": candidate["strategy"],
                "gray_margin": candidate.get("gray_margin"),
                "ensemble_threshold": candidate.get("ensemble_threshold"),
                "config_path": str(config_path),
            }
        )
        result_rows.append(metric)
    report = {
        "phase": "CASA-Hybrid Phase5 calibration-only sweep",
        "source_split": "phase4_calibration",
        "phase4_root": str(args.phase4_root),
        "phase5_root": str(args.phase5_root),
        "base_config": str(args.base_config),
        "candidate_count": len(candidates),
        "registry_summary": registry_summary,
        "pareto_frontier": _pareto(result_rows),
        "rows": result_rows,
        "note": "This sweep intentionally does not read final held-out online test labels.",
    }
    _write_json(args.output_dir / "hybrid_sweep_report.json", report)
    (args.output_dir / "hybrid_sweep_report.md").write_text(_markdown(report) + "\n")
    if args.emit_run_commands:
        (args.output_dir / "hybrid_sweep_run_commands.sh").write_text(_run_commands(args, result_rows) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _candidate_configs(args: argparse.Namespace, base: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    gray_margins = _floats(args.gray_margins)
    ensemble_thresholds = _floats(args.ensemble_thresholds)
    output = []
    for strategy in strategies:
        if strategy not in set(HYBRID_METHOD_STRATEGIES.values()):
            raise SystemExit(f"Unknown strategy {strategy!r}")
        if strategy == "gray_refine":
            for gray_margin in gray_margins:
                config = _config_with_strategy(base, strategy)
                config.setdefault("gray_refine", {})["gray_margin"] = gray_margin
                output.append(
                    {
                        "candidate_id": f"candidate_{len(output):04d}",
                        "strategy": strategy,
                        "gray_margin": gray_margin,
                        "config": config,
                    }
                )
        elif strategy == "calibrated_ensemble":
            for threshold in ensemble_thresholds:
                config = _config_with_strategy(base, strategy)
                config.setdefault("calibrated_ensemble", {})["threshold"] = threshold
                output.append(
                    {
                        "candidate_id": f"candidate_{len(output):04d}",
                        "strategy": strategy,
                        "ensemble_threshold": threshold,
                        "config": config,
                    }
                )
        else:
            output.append(
                {
                    "candidate_id": f"candidate_{len(output):04d}",
                    "strategy": strategy,
                    "config": _config_with_strategy(base, strategy),
                }
            )
    return output


def _config_with_strategy(base: dict[str, Any], strategy: str) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["default_strategy"] = strategy
    config.setdefault("method_strategies", {})["casa_h_mpc_safedpa_casa_refine"] = strategy
    return config


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                int(other["unsafe_invocation_count"]) <= int(row["unsafe_invocation_count"])
                and float(other["safe_acceptance_rate"]) >= float(row["safe_acceptance_rate"])
                and float(other["reject_rate"]) <= float(row["reject_rate"])
                and (
                    int(other["unsafe_invocation_count"]) < int(row["unsafe_invocation_count"])
                    or float(other["safe_acceptance_rate"]) > float(row["safe_acceptance_rate"])
                    or float(other["reject_rate"]) < float(row["reject_rate"])
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (int(row["unsafe_invocation_count"]), -float(row["safe_acceptance_rate"])))


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CASA-Hybrid Phase5 Sweep",
        "",
        f"- source_split: `{report['source_split']}`",
        f"- candidate_count: `{report['candidate_count']}`",
        "",
        "## Pareto Frontier",
        "",
        "| candidate | strategy | unsafe | reject_rate | safe_acceptance | config |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["pareto_frontier"]:
        lines.append(
            f"| {row['candidate_id']} | {row['strategy']} | {row['unsafe_invocation_count']} | "
            f"{float(row['reject_rate']):.4f} | {float(row['safe_acceptance_rate']):.4f} | "
            f"`{row['config_path']}` |"
        )
    if not report["pareto_frontier"]:
        lines.append("| none |  |  |  |  |  |")
    return "\n".join(lines)


def _run_commands(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    online_root = args.online_root or args.output_dir / "online_sweep"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in _pareto(rows):
        root = online_root / str(row["candidate_id"])
        lines.extend(
            [
                ".venv_sim/bin/python -u gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py \\",
                f"  --online-root {root} \\",
                "  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill,mpc_cbf_humanoid_adapted,safedpa_adapted,casa_h_mpc_safedpa_casa_refine \\",
                "  --casa-method casa_h_mpc_safedpa_casa_refine \\",
                f"  --hybrid-config {row['config_path']} \\",
                f"  --seeds {args.seeds} \\",
                f"  --episodes-per-seed {args.episodes_per_seed} \\",
                "  --max-parallel 3 \\",
                "  --chunk-size 25 \\",
                "  --max-sweeps 6",
                "",
            ]
        )
    return "\n".join(lines)


def _floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
