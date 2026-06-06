"""Write paper-table scaffold for CASA adapted SOTA baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIRST_SLICE_METHODS = ("pcbf_adapted", "crc_cbf_adapted", "mpc_cbf_humanoid_adapted")
FULL_METHODS = (
    "sonic_only",
    "hard_contract",
    "raw_critic_0p5",
    "global_conformal",
    "casa_a_per_skill",
    *FIRST_SLICE_METHODS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase4-root", required=True)
    parser.add_argument("--phase5-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = args.output_root / "reports"
    manifests_dir = args.output_root / "manifests"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.output_root / "data" / "sota_method_metadata.json").read_text())
    methods = metadata["methods"]
    implemented = [item for item in methods if item["implemented_in_this_run"]]
    online_root = args.output_root / "online" / "full_online_eval"
    full_command = _full_online_command(args.phase4_root, args.phase5_root, online_root)
    scaffold = {
        "method_adaptation_table": str(reports_dir / "method_adaptation_table.md"),
        "evaluation_protocol_table": str(reports_dir / "evaluation_protocol_table.md"),
        "metrics_table_template": str(reports_dir / "metrics_table_template.md"),
        "implementation_fidelity_table": str(reports_dir / "implementation_fidelity_table.md"),
        "full_online_eval_manifest": str(manifests_dir / "full_online_eval_manifest.json"),
    }
    (reports_dir / "method_adaptation_table.md").write_text(_method_adaptation_table(methods) + "\n")
    (reports_dir / "evaluation_protocol_table.md").write_text(_evaluation_protocol_table() + "\n")
    (reports_dir / "metrics_table_template.md").write_text(_metrics_template(FULL_METHODS) + "\n")
    (reports_dir / "implementation_fidelity_table.md").write_text(
        _implementation_fidelity_table(methods) + "\n"
    )
    manifest = {
        "phase": "CASA adapted SOTA full online evaluation manifest",
        "methods": list(FULL_METHODS),
        "first_slice_sota_methods": list(FIRST_SLICE_METHODS),
        "implemented_sota_metadata_methods": [item["method_name"] for item in implemented],
        "seeds": [1234, 1235, 1236, 1237, 1238],
        "episodes_per_seed": 100,
        "total_episodes": len(FULL_METHODS) * 5 * 100,
        "phase4_root": args.phase4_root,
        "phase5_root": args.phase5_root,
        "online_root": str(online_root),
        "full_online_command": full_command,
        "merge_output": str(online_root / "merged"),
        "notes": [
            "Run only when live MuJoCo/deploy lanes are available.",
            "This command reruns methods inside CASA; it must not copy external paper numbers.",
            "Remaining SOTA candidates can be added after their adapters are implemented and smoke-tested.",
        ],
    }
    (manifests_dir / "full_online_eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (manifests_dir / "full_online_eval_command.sh").write_text(full_command + "\n")
    print(json.dumps(scaffold, indent=2, sort_keys=True))


def _method_adaptation_table(methods: list[dict[str, Any]]) -> str:
    lines = [
        "# Method Adaptation Table",
        "",
        "| method | original domain | safety mechanism | CASA adaptation | main-table fidelity | implemented |",
        "|---|---|---|---|---|---:|",
    ]
    for item in methods:
        lines.append(
            f"| `{item['method_name']}` | {item['original_domain']} | "
            f"{item['original_safety_mechanism']} | {item['casa_adaptation_route']} | "
            f"`{item['main_table_fidelity_label']}` | {int(bool(item['implemented_in_this_run']))} |"
        )
    lines.extend(
        [
            "",
            "Main-table wording must state that these are adapted implementations rerun in CASA/SONIC Phase5.",
        ]
    )
    return "\n".join(lines)


def _evaluation_protocol_table() -> str:
    rows = [
        ("Controller", "Frozen SONIC / GEAR-SONIC humanoid controller"),
        ("Simulation", "CASA Phase5 MuJoCo sim2sim/deploy lane"),
        ("Task sequence", "walk, turn, passive stop, gesture, walk, turn, passive wait, walk"),
        ("Skill set", "walk, turn, gesture, passive"),
        ("Scene factors", "randomized obstacles, user proxy, target bucket, runtime perturbation"),
        ("Safety oracle", "collision, near_collision, fall, human_distance_violation, unsafe_gesture, timeout"),
        ("Calibration data", "phase4_split == calibration only"),
        ("Test data", "online episodes or offline phase4_split == test; no threshold tuning on test"),
        ("Fallback", "same passive fallback contract as CASA main gate"),
        ("Reporting rule", "report rerun CASA metrics only, never copied source-paper numbers"),
    ]
    lines = ["# Evaluation Protocol Table", "", "| item | protocol |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def _metrics_template(methods: tuple[str, ...]) -> str:
    headers = [
        "method",
        "implementation fidelity",
        "unsafe rate",
        "unsafe reduction vs SONIC",
        "safe completion",
        "fallback / episode",
        "reject / decision",
        "fall count",
        "collision count",
        "runtime p90 ms",
        "solver infeasible rate",
    ]
    lines = ["# Metrics Table Template", "", "| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for method in methods:
        lines.append(f"| `{method}` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
    lines.append("")
    lines.append("Fill this table only from offline/online CASA reruns under the same seeds and protocol.")
    return "\n".join(lines)


def _implementation_fidelity_table(methods: list[dict[str, Any]]) -> str:
    lines = [
        "# Implementation Fidelity Table",
        "",
        "| method | code status | official code | commit | license | fidelity label | caveat |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in methods:
        code = item.get("official_code_url") or "not verified"
        commit = item.get("official_code_commit") or "n/a"
        license_name = item.get("license") or "n/a"
        lines.append(
            f"| `{item['method_name']}` | {item['code_status']} | {code} | `{commit}` | "
            f"{license_name} | `{item['main_table_fidelity_label']}` | {item['notes']} |"
        )
    return "\n".join(lines)


def _full_online_command(phase4_root: str, phase5_root: str, online_root: Path) -> str:
    return " \\\n  ".join(
        [
            ".venv_sim/bin/python -u gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py",
            f"--phase4-root {phase4_root}",
            f"--phase5-root {phase5_root}",
            f"--online-root {online_root}",
            f"--methods {','.join(FULL_METHODS)}",
            "--casa-method casa_a_per_skill",
            "--seeds 1234,1235,1236,1237,1238",
            "--episodes-per-seed 100",
            "--max-parallel 3",
            "--chunk-size 25",
            "--cuda-devices 0,1",
            "--performance-preset custom",
        ]
    )


if __name__ == "__main__":
    main()
