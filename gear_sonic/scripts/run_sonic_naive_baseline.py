"""Run the Stage 2 SONIC-Naive task baseline over ZMQManager.

This is intentionally a thin task sequencer: it does not implement SkillGuard,
closed-loop navigation, or safety gating. It publishes direct planner commands
for a fixed walk -> stop -> gesture -> walk smoke task and records the task
timeline for offline episode summarization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.io.zmq_publisher import LOCOMOTION_IDLE, LOCOMOTION_WALK, ZMQPlannerPublisher
from gear_sonic.casa.skills import facing_from_yaw, gesture_upper_body


@dataclass(frozen=True)
class Phase:
    name: str
    skill: str
    duration_s: float
    mode: int
    movement: tuple[float, float, float]
    facing: tuple[float, float, float]
    speed: float = -1.0
    height: float = -1.0
    gesture: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run id used when deriving the default output dir")
    parser.add_argument("--output-dir", type=Path, help="Task log output dir")
    parser.add_argument("--zmq-host", default="*", help="ZMQ bind host for command/planner publisher")
    parser.add_argument("--zmq-port", type=int, default=5556, help="ZMQ port used by deploy --zmq-port")
    parser.add_argument("--publish-fps", type=float, default=10.0, help="Planner publish frequency")
    parser.add_argument("--walk-a-seconds", type=float, default=6.0)
    parser.add_argument("--stop-seconds", type=float, default=2.0)
    parser.add_argument("--face-seconds", type=float, default=2.0)
    parser.add_argument("--face-yaw-deg", type=float, default=0.0)
    parser.add_argument("--gesture-seconds", type=float, default=5.0)
    parser.add_argument("--walk-b-seconds", type=float, default=6.0)
    parser.add_argument("--gesture-amplitude", type=float, default=0.5)
    parser.add_argument("--gesture-frequency", type=float, default=0.75)
    parser.add_argument("--send-final-stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build payloads and logs without opening ZMQ")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "sonic_stage2" / run_id / "task"


def build_phases(args: argparse.Namespace) -> list[Phase]:
    forward = (1.0, 0.0, 0.0)
    idle = (0.0, 0.0, 0.0)
    face_user = facing_from_yaw(args.face_yaw_deg)
    return [
        Phase("walk_to_A", "walk", args.walk_a_seconds, LOCOMOTION_WALK, forward, forward),
        Phase("stop", "stop", args.stop_seconds, LOCOMOTION_IDLE, idle, forward),
        Phase("face_user", "stop", args.face_seconds, LOCOMOTION_IDLE, idle, face_user),
        Phase("gesture", "gesture", args.gesture_seconds, LOCOMOTION_IDLE, idle, face_user, gesture=True),
        Phase("walk_to_B", "walk", args.walk_b_seconds, LOCOMOTION_WALK, forward, forward),
    ]


def write_event(writer: csv.DictWriter[str], start_time: float, **values: Any) -> None:
    now = time.monotonic()
    row = {
        "task_time_s": f"{now - start_time:.6f}",
        "wall_time_s": f"{time.time():.6f}",
        **values,
    }
    writer.writerow(row)


def run_phase(
    *,
    phase: Phase,
    switch_id: int,
    source_skill: str,
    publisher: ZMQPlannerPublisher,
    writer: csv.DictWriter[str],
    start_time: float,
    publish_dt: float,
    args: argparse.Namespace,
) -> int:
    write_event(
        writer,
        start_time,
        event="phase_start",
        phase=phase.name,
        skill=phase.skill,
        mode=phase.mode,
        movement_x=phase.movement[0],
        movement_y=phase.movement[1],
        movement_z=phase.movement[2],
        facing_x=phase.facing[0],
        facing_y=phase.facing[1],
        facing_z=phase.facing[2],
        speed=phase.speed,
        height=phase.height,
        has_upper_body=int(phase.gesture),
        switch_id=switch_id,
        source_skill=source_skill,
        target_skill=phase.skill,
        switch_requested=1,
        switch_allowed=1,
        switch_rejected_reason="",
    )

    publishes = 0
    phase_start = time.monotonic()
    next_publish = phase_start
    while time.monotonic() - phase_start < phase.duration_s:
        elapsed = time.monotonic() - phase_start
        upper_body_position = None
        upper_body_velocity = None
        if phase.gesture:
            upper_body_position, upper_body_velocity = gesture_upper_body(
                elapsed,
                args.gesture_amplitude,
                args.gesture_frequency,
            )
        publisher.send_planner(
            mode=phase.mode,
            movement=phase.movement,
            facing=phase.facing,
            speed=phase.speed,
            height=phase.height,
            upper_body_position=upper_body_position,
            upper_body_velocity=upper_body_velocity,
        )
        publishes += 1
        write_event(
            writer,
            start_time,
            event="planner_publish",
            phase=phase.name,
            skill=phase.skill,
            mode=phase.mode,
            movement_x=phase.movement[0],
            movement_y=phase.movement[1],
            movement_z=phase.movement[2],
            facing_x=phase.facing[0],
            facing_y=phase.facing[1],
            facing_z=phase.facing[2],
            speed=phase.speed,
            height=phase.height,
            has_upper_body=int(phase.gesture),
        )
        next_publish += publish_dt
        time.sleep(max(0.0, next_publish - time.monotonic()))

    write_event(
        writer,
        start_time,
        event="phase_end",
        phase=phase.name,
        skill=phase.skill,
        mode=phase.mode,
        movement_x=phase.movement[0],
        movement_y=phase.movement[1],
        movement_z=phase.movement[2],
        facing_x=phase.facing[0],
        facing_y=phase.facing[1],
        facing_z=phase.facing[2],
        speed=phase.speed,
        height=phase.height,
        has_upper_body=int(phase.gesture),
    )
    return publishes


def main() -> None:
    args = parse_args()
    if args.publish_fps <= 0:
        raise ValueError("--publish-fps must be positive")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "task_events.csv"
    summary_path = output_dir / "task_summary.json"
    fields = [
        "task_time_s",
        "wall_time_s",
        "event",
        "phase",
        "skill",
        "mode",
        "movement_x",
        "movement_y",
        "movement_z",
        "facing_x",
        "facing_y",
        "facing_z",
        "speed",
        "height",
        "has_upper_body",
        "switch_id",
        "source_skill",
        "target_skill",
        "switch_requested",
        "switch_allowed",
        "switch_rejected_reason",
    ]

    phases = build_phases(args)
    publisher = ZMQPlannerPublisher(args.zmq_host, args.zmq_port, dry_run=args.dry_run)
    publish_dt = 1.0 / args.publish_fps
    start_time = time.monotonic()
    phase_publish_counts: dict[str, int] = {}
    task_completed = False

    try:
        with events_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            write_event(
                writer,
                start_time,
                event="task_start",
                phase="init",
                skill="init",
                mode=LOCOMOTION_IDLE,
                movement_x=0.0,
                movement_y=0.0,
                movement_z=0.0,
                facing_x=1.0,
                facing_y=0.0,
                facing_z=0.0,
                speed=-1.0,
                height=-1.0,
                has_upper_body=0,
            )
            publisher.send_command(start=True, stop=False, planner=True)
            write_event(
                writer,
                start_time,
                event="start_policy",
                phase="init",
                skill="init",
                mode=LOCOMOTION_IDLE,
                movement_x=0.0,
                movement_y=0.0,
                movement_z=0.0,
                facing_x=1.0,
                facing_y=0.0,
                facing_z=0.0,
                speed=-1.0,
                height=-1.0,
                has_upper_body=0,
            )

            previous_skill = "init"
            for switch_id, phase in enumerate(phases, start=1):
                phase_publish_counts[phase.name] = run_phase(
                    phase=phase,
                    switch_id=switch_id,
                    source_skill=previous_skill,
                    publisher=publisher,
                    writer=writer,
                    start_time=start_time,
                    publish_dt=publish_dt,
                    args=args,
                )
                previous_skill = phase.skill

            if args.send_final_stop:
                publisher.send_command(start=False, stop=True, planner=True)
                final_event = "stop_control"
            else:
                final_event = "done"
            write_event(
                writer,
                start_time,
                event=final_event,
                phase="done",
                skill="done",
                mode=LOCOMOTION_IDLE,
                movement_x=0.0,
                movement_y=0.0,
                movement_z=0.0,
                facing_x=1.0,
                facing_y=0.0,
                facing_z=0.0,
                speed=-1.0,
                height=-1.0,
                has_upper_body=0,
            )
            task_completed = True
    finally:
        publisher.close()

    summary = {
        "baseline": "SONIC-Naive",
        "task_completed": task_completed,
        "dry_run": args.dry_run,
        "events_csv": str(events_path),
        "phase_sequence": [phase.name for phase in phases],
        "skill_sequence": [phase.skill for phase in phases],
        "phase_publish_counts": phase_publish_counts,
        "publish_fps": args.publish_fps,
        "zmq_host": args.zmq_host,
        "zmq_port": args.zmq_port,
        "send_final_stop": args.send_final_stop,
        "sent_message_count": len(publisher.sent_messages),
        "first_messages": publisher.sent_messages[:5],
    }
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Wrote {events_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
