"""Run a CASA Phase 0 long-horizon skill sanity check."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.runner_utils import (
    build_evaluation_config,
    create_executor,
    resolve_output_dir,
    resolve_run_id,
    resolve_sim_log_dir,
    summarize_results,
    write_json,
)
from gear_sonic.casa.skills import GestureSkill, PassiveSkill, TurnSkill, WalkSkill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run id used when deriving the default output dir")
    parser.add_argument("--output-dir", type=Path, help="Output dir for CASA logs")
    parser.add_argument("--sim-log-dir", type=Path, help="MuJoCo sim episode log dir")
    parser.add_argument("--zmq-host", default="*", help="ZMQ bind host for command/planner publisher")
    parser.add_argument("--zmq-port", type=int, default=5556, help="ZMQ port used by deploy --zmq-port")
    parser.add_argument("--publish-fps", type=float, default=10.0, help="Planner publish frequency")
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--min-skill-duration", type=float, default=1.5)
    parser.add_argument("--max-skill-duration", type=float, default=5.0)
    parser.add_argument("--stabilize-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--passive-speed-threshold", type=float, default=0.08)
    parser.add_argument("--passive-min-ratio", type=float, default=0.7)
    parser.add_argument("--passive-tail-fraction", type=float, default=0.5)
    parser.add_argument("--turn-yaw-threshold-rad", type=float, default=0.3)
    parser.add_argument("--walk-speed-threshold", type=float, default=0.1)
    parser.add_argument("--walk-min-ratio", type=float, default=0.5)
    parser.add_argument("--gesture-min-rom", type=float, default=0.05)
    parser.add_argument("--middle-trim-fraction", type=float, default=0.2)
    parser.add_argument("--send-final-stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build payloads and logs without opening ZMQ")
    return parser.parse_args()


def sample_skill(rng: random.Random, args: argparse.Namespace):
    duration = rng.uniform(args.min_skill_duration, args.max_skill_duration)
    name = rng.choice(["walk", "turn", "gesture", "passive"])
    if name == "walk":
        return WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=rng.choice([0.0, 15.0, -15.0]), duration=duration)
    if name == "turn":
        return TurnSkill(face_yaw_deg=rng.choice([-45.0, -30.0, 0.0, 30.0, 45.0]), duration=duration)
    if name == "gesture":
        return GestureSkill(
            amplitude=rng.choice([0.3, 0.5, 0.7]),
            frequency=rng.choice([0.5, 0.75, 1.0]),
            duration=duration,
            side=rng.choice(["left", "right", "both"]),
        )
    return PassiveSkill(duration=duration, mode="wait" if duration > 1.0 else "stop")


def main() -> None:
    args = parse_args()
    run_id = resolve_run_id(args.run_id)
    output_dir = resolve_output_dir(args.output_dir, run_id)
    sim_log_dir = resolve_sim_log_dir(args.sim_log_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    evaluation_config = build_evaluation_config(args)

    executor = create_executor(args, output_dir, run_id, sim_log_dir)
    results = []
    planned_elapsed = 0.0
    start = time.monotonic()
    try:
        executor.start_policy()
        while planned_elapsed < args.duration_seconds:
            skill = sample_skill(rng, args)
            if skill.name in {"walk", "gesture"} and args.stabilize_seconds > 0:
                result = executor.execute_one(PassiveSkill(duration=args.stabilize_seconds, mode="wait"))
                results.append(result)
                planned_elapsed += args.stabilize_seconds
            result = executor.execute_one(skill)
            results.append(result)
            planned_elapsed += skill.estimated_duration
            if not args.dry_run:
                planned_elapsed = time.monotonic() - start
        if args.send_final_stop:
            executor.stop_policy()
    finally:
        sent_message_count = len(executor.publisher.sent_messages)
        executor.close()

    summary = {
        "run_id": run_id,
        "dry_run": args.dry_run,
        "output_dir": str(output_dir),
        "sim_log_dir": str(sim_log_dir),
        "duration_seconds": args.duration_seconds,
        "evaluation_config": asdict(evaluation_config),
        "planned_elapsed_seconds": planned_elapsed,
        "executed_skill_count": len(results),
        "overall": summarize_results(results, dry_run=args.dry_run),
        "go_criteria_passed": (
            not args.dry_run and planned_elapsed >= args.duration_seconds and summarize_results(results)["success_rate"] >= 0.9
        ),
        "sent_message_count": sent_message_count,
        "skill_events_csv": str(output_dir / "skill_events.csv"),
    }
    write_json(output_dir / "sanity_summary.json", summary)
    write_report(output_dir / "sanity_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# CASA Phase 0 Long Sanity Report",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- dry_run: `{summary['dry_run']}`",
        f"- requested_duration_seconds: `{summary['duration_seconds']}`",
        f"- planned_elapsed_seconds: `{summary['planned_elapsed_seconds']:.3f}`",
        f"- executed_skill_count: `{summary['executed_skill_count']}`",
        f"- success_rate: `{summary['overall']['success_rate']:.3f}`",
        f"- go_criteria_passed: `{summary['go_criteria_passed']}`",
        "",
        "ZMQ has no Python-side ack; execution is inferred from sim logs.",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
