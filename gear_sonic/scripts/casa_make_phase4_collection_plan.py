"""Create a Phase 4 strict-clean collection recovery plan.

This script does not run simulation. It turns the current Phase 4 deficits into
scene-plan CSVs and lane commands that can be launched in controlled batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_SKILLS = ("walk", "turn", "gesture", "passive")


GROUPS = {
    "gesture_unsafe": {
        "skill": "gesture",
        "target_bucket": "phase4_gesture_user_close",
        "skill_schedule": "gesture_user_close_probe",
        "complexities": ["medium", "hard"],
        "samples_per_rollout": 8.0,
        "unsafe_per_rollout": 6.0,
    },
    "turn_unsafe": {
        "skill": "turn",
        "target_bucket": "phase4_turn_collision",
        "skill_schedule": "turn_collision_probe",
        "complexities": ["medium", "hard"],
        "samples_per_rollout": 8.0,
        "unsafe_per_rollout": 3.0,
    },
    "passive_unsafe": {
        "skill": "passive",
        "target_bucket": "phase4_passive_delayed_fall",
        "skill_schedule": "passive_delayed_fall_probe",
        "complexities": ["medium", "hard"],
        "samples_per_rollout": 5.0,
        "unsafe_per_rollout": 4.0,
    },
    "walk_safe_normal": {
        "skill": "walk",
        "target_bucket": "phase4_safe_balanced",
        "skill_schedule": "safe_balanced_phase4",
        "complexities": ["simple", "medium", "hard", "medium"],
        "samples_per_rollout": 3.0,
        "unsafe_per_rollout": 0.0,
    },
    "safe_balanced": {
        "skill": "",
        "target_bucket": "phase4_safe_balanced",
        "skill_schedule": "safe_balanced_phase4",
        "complexities": ["simple", "medium", "hard", "medium"],
        "samples_per_rollout": 4.0,
        "unsafe_per_rollout": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-multiplier", type=float, default=1.35)
    parser.add_argument("--dangerous-per-skill-target", type=int, default=1500)
    parser.add_argument("--skill-quota", type=int, default=12500)
    parser.add_argument("--num-lanes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--domain-id-base", type=int, default=70)
    parser.add_argument("--zmq-port-base", type=int, default=5656)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.phase4_root / "collection_recovery_plan"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_json(args.phase4_root / "dataset_v1" / "dataset_summary.json")
    plan = _make_plan(summary, args)

    for group_name, group in plan["groups"].items():
        csv_path = output_dir / f"{group_name}_scene_plan.csv"
        _write_scene_plan(csv_path, group["target_bucket"], group["rollout_count"], group["complexities"])
        group["scene_plan_csv"] = str(csv_path)

    commands_md = _commands_markdown(plan, args, output_dir)
    (output_dir / "phase4_collection_recovery_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (output_dir / "phase4_collection_recovery_commands.md").write_text(commands_md + "\n")
    print(json.dumps({"output_dir": str(output_dir), "groups": plan["groups"]}, indent=2, sort_keys=True))


def _make_plan(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    per_skill = summary.get("per_skill", {})
    groups: dict[str, Any] = {}
    total_rollouts = 0
    targeted_sample_capacity = 0

    for group_name in ["gesture_unsafe", "turn_unsafe", "passive_unsafe", "walk_safe_normal"]:
        spec = GROUPS[group_name]
        skill = spec["skill"]
        skill_stats = per_skill.get(skill, {})
        total_deficit = max(0, args.skill_quota - int(skill_stats.get("total", 0)))
        unsafe_deficit = max(0, args.dangerous_per_skill_target - int(skill_stats.get("unsafe", 0)))
        unsafe_rollout_need = 0.0 if spec["unsafe_per_rollout"] <= 0 else unsafe_deficit / float(spec["unsafe_per_rollout"])
        rollout_count = int(
            math.ceil(
                max(
                    total_deficit / float(spec["samples_per_rollout"]),
                    unsafe_rollout_need,
                )
                * args.candidate_multiplier
            )
        )
        groups[group_name] = {
            **spec,
            "current_total": int(skill_stats.get("total", 0)),
            "current_unsafe": int(skill_stats.get("unsafe", 0)),
            "total_deficit_to_skill_quota": total_deficit,
            "unsafe_deficit_to_target": unsafe_deficit,
            "rollout_count": rollout_count,
        }
        total_rollouts += rollout_count
        targeted_sample_capacity += int(rollout_count * spec["samples_per_rollout"])

    remaining_total_deficit = max(0, int(summary.get("target_samples_min", 50_000)) - int(summary.get("selected_samples", 0)))
    safe_needed = max(0, remaining_total_deficit - targeted_sample_capacity)
    safe_spec = GROUPS["safe_balanced"]
    safe_rollouts = int(math.ceil((safe_needed / float(safe_spec["samples_per_rollout"])) * args.candidate_multiplier))
    groups["safe_balanced"] = {
        **safe_spec,
        "rollout_count": safe_rollouts,
        "sample_deficit_after_targeted_capacity": safe_needed,
    }
    total_rollouts += safe_rollouts

    return {
        "phase": "CASA Phase4 strict-clean collection recovery plan",
        "source_phase4_root": str(args.phase4_root),
        "candidate_multiplier": args.candidate_multiplier,
        "num_lanes": args.num_lanes,
        "strict_target_samples": summary.get("target_samples_min", 50_000),
        "current_selected_samples": summary.get("selected_samples", 0),
        "current_clean_available_samples": summary.get("clean_available_samples", 0),
        "groups": groups,
        "estimated_total_rollouts": total_rollouts,
        "notes": [
            "Counts are deliberately over-collected; rebuild the strict dataset after each batch and stop once phase4_go_no_go.json has go=true.",
            "All commands omit --inject-latency-ms and use casa_run_phase2_v2_lane.py, whose sim command includes --drop-on-start.",
        ],
    }


def _write_scene_plan(path: Path, target_bucket: str, rollout_count: int, complexities: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode_index", "target_bucket", "scene_complexity"])
        writer.writeheader()
        for index in range(rollout_count):
            writer.writerow(
                {
                    "episode_index": index,
                    "target_bucket": target_bucket,
                    "scene_complexity": complexities[index % len(complexities)],
                }
            )


def _commands_markdown(plan: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> str:
    lines = [
        "# CASA Phase4 Strict Collection Recovery Commands",
        "",
        "Run groups in batches, then rebuild Phase 4 strict dataset and re-run Raw Critic/audit.",
        "",
    ]
    for group_index, (group_name, group) in enumerate(plan["groups"].items()):
        if int(group.get("rollout_count", 0)) <= 0:
            continue
        lines.extend([f"## {group_name}", ""])
        for lane in range(args.num_lanes):
            domain_id = args.domain_id_base + group_index * args.num_lanes + lane
            zmq_port = args.zmq_port_base + group_index * 20 + lane * 2
            output_path = output_dir / "runs" / group_name / f"lane{lane}"
            cmd = [
                ".venv_sim/bin/python",
                "gear_sonic/scripts/casa_run_phase2_v2_lane.py",
                "--run-id",
                f"phase4_{group_name}_lane{lane}_20260522",
                "--output-dir",
                str(output_path),
                "--scene-plan-csv",
                str(group["scene_plan_csv"]),
                "--lane-index",
                str(lane),
                "--num-lanes",
                str(args.num_lanes),
                "--domain-id",
                str(domain_id),
                "--zmq-port",
                str(zmq_port),
                "--zmq-out-port",
                str(zmq_port + 1),
                "--duration-seconds",
                "18",
                "--min-skill-duration",
                "0.9",
                "--max-skill-duration",
                "1.4",
                "--stabilize-seconds",
                "0.5",
                "--skill-schedule",
                group["skill_schedule"],
                "--seed",
                str(args.seed + group_index * 1000),
            ]
            lines.extend(["```bash", " ".join(cmd), "```", ""])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
