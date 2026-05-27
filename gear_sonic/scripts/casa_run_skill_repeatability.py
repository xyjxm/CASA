"""Run CASA Phase 0/1 skill repeatability checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
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
    result_is_success,
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
    parser.add_argument("--trials-per-skill", type=int, default=20)
    parser.add_argument("--stabilize-seconds", type=float, default=1.0)
    parser.add_argument("--walk-duration", type=float, default=3.0)
    parser.add_argument("--turn-duration", type=float, default=3.0)
    parser.add_argument("--turn-yaw-deg", type=float, default=30.0)
    parser.add_argument("--gesture-duration", type=float, default=3.0)
    parser.add_argument("--gesture-amplitude", type=float, default=0.5)
    parser.add_argument("--gesture-frequency", type=float, default=0.75)
    parser.add_argument("--passive-duration", type=float, default=2.0)
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


def build_skill(name: str, args: argparse.Namespace):
    if name == "walk":
        return WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=0.0, duration=args.walk_duration)
    if name == "turn":
        return TurnSkill(face_yaw_deg=args.turn_yaw_deg, duration=args.turn_duration)
    if name == "gesture":
        return GestureSkill(
            amplitude=args.gesture_amplitude,
            frequency=args.gesture_frequency,
            duration=args.gesture_duration,
            side="both",
        )
    if name == "passive":
        return PassiveSkill(duration=args.passive_duration, mode="wait")
    raise ValueError(f"unknown skill: {name}")


def main() -> None:
    args = parse_args()
    if args.trials_per_skill <= 0:
        raise ValueError("--trials-per-skill must be positive")
    run_id = resolve_run_id(args.run_id)
    output_dir = resolve_output_dir(args.output_dir, run_id)
    sim_log_dir = resolve_sim_log_dir(args.sim_log_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_config = build_evaluation_config(args)

    executor = create_executor(args, output_dir, run_id, sim_log_dir)
    main_results = []
    try:
        executor.start_policy()
        skill_names = ["walk", "turn", "gesture", "passive"]
        for skill_name in skill_names:
            for trial in range(args.trials_per_skill):
                if args.stabilize_seconds > 0:
                    executor.execute_one(
                        PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
                        episode_id=f"{skill_name}_{trial}",
                    )
                result = executor.execute_one(build_skill(skill_name, args), episode_id=f"{skill_name}_{trial}")
                main_results.append(result)
        if args.send_final_stop:
            executor.stop_policy()
    finally:
        sent_message_count = len(executor.publisher.sent_messages)
        first_messages = executor.publisher.sent_messages[:5]
        executor.close()

    by_skill = {
        name: summarize_results([result for result in main_results if result.skill_name == name], dry_run=args.dry_run)
        for name in ["walk", "turn", "gesture", "passive"]
    }
    all_summary = summarize_results(main_results, dry_run=args.dry_run)
    go_passed = (
        not args.dry_run
        and all(summary["total"] >= args.trials_per_skill and summary["success_rate"] >= 0.9 for summary in by_skill.values())
    )
    summary = {
        "run_id": run_id,
        "dry_run": args.dry_run,
        "output_dir": str(output_dir),
        "sim_log_dir": str(sim_log_dir),
        "trials_per_skill": args.trials_per_skill,
        "evaluation_config": asdict(evaluation_config),
        "by_skill": by_skill,
        "overall": all_summary,
        "go_criteria_passed": go_passed,
        "sent_message_count": sent_message_count,
        "first_messages": first_messages,
        "skill_events_csv": str(output_dir / "skill_events.csv"),
    }
    write_json(output_dir / "repeatability_summary.json", summary)
    write_report(output_dir / "sanity_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# CASA Phase 0/1 Sanity Report",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- dry_run: `{summary['dry_run']}`",
        f"- go_criteria_passed: `{summary['go_criteria_passed']}`",
        f"- skill_events_csv: `{summary['skill_events_csv']}`",
        "",
        "## Repeatability",
        "",
        "| skill | total | successes | success_rate | statuses |",
        "|---|---:|---:|---:|---|",
    ]
    for skill_name, data in summary["by_skill"].items():
        lines.append(
            f"| {skill_name} | {data['total']} | {data['successes']} | "
            f"{data['success_rate']:.3f} | `{data['statuses']}` |"
        )
    lines.extend(["", "## Notes", "", "- ZMQ has no Python-side ack; execution is inferred from sim logs."])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
