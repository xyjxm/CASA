"""Summarize a Stage 2 SONIC-Naive baseline episode.

The Stage 2 task runner records task phase events while the existing Stage 1
sim/deploy loggers record robot state. This script aligns those relative
timelines, prefers task phases for skill labels, and emits one episode CSV plus
a JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_sonic_stage1_episode import (
    compressed,
    compute_tracking_errors,
    contains_subsequence,
    load_deploy_rows,
    prepare_sim_rows,
    read_dict_csv,
    read_input_log,
    read_raw_csv,
    resolve_sim_state_csv,
    to_float,
)


PHASE_TO_SKILL = {
    "walk_to_A": "walk",
    "walk_to_B": "walk",
    "stop": "stop",
    "face_user": "stop",
    "gesture": "gesture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-log-dir", type=Path, help="Directory containing sim_state.csv")
    parser.add_argument("--sim-state-csv", type=Path, help="Path to sim_state.csv")
    parser.add_argument("--deploy-log-dir", type=Path, required=True, help="Deploy StateLogger CSV dir")
    parser.add_argument("--task-log-dir", type=Path, help="Directory containing task_events.csv")
    parser.add_argument("--task-events-csv", type=Path, help="Path to task_events.csv")
    parser.add_argument("--input-log", type=Path, help="Optional deploy input.csv")
    parser.add_argument("--target-motion-log", type=Path, help="Optional target_motion.csv")
    parser.add_argument("--output-dir", type=Path, help="Output dir for episode.csv/json")
    parser.add_argument("--goal-displacement-threshold", type=float, default=0.1)
    return parser.parse_args()


def resolve_task_events_csv(args: argparse.Namespace) -> Path | None:
    if args.task_events_csv:
        return args.task_events_csv
    if args.task_log_dir:
        return args.task_log_dir / "task_events.csv"
    return None


def load_task_events(path: Path | None) -> list[dict[str, Any]]:
    events = []
    for row in read_dict_csv(path):
        events.append(
            {
                **row,
                "task_time_s": to_float(row.get("task_time_s")),
                "mode": int(to_float(row.get("mode"))),
                "movement_x": to_float(row.get("movement_x")),
                "movement_y": to_float(row.get("movement_y")),
                "movement_z": to_float(row.get("movement_z")),
                "facing_x": to_float(row.get("facing_x")),
                "facing_y": to_float(row.get("facing_y")),
                "facing_z": to_float(row.get("facing_z")),
                "speed": to_float(row.get("speed"), -1.0),
                "height": to_float(row.get("height"), -1.0),
                "has_upper_body": int(to_float(row.get("has_upper_body"))),
            }
        )
    return events


def phase_windows(task_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    windows = []
    for event in task_events:
        phase = str(event.get("phase", ""))
        if event.get("event") == "phase_start":
            starts[phase] = event
        elif event.get("event") == "phase_end" and phase in starts:
            start = starts.pop(phase)
            windows.append(
                {
                    "phase": phase,
                    "skill": PHASE_TO_SKILL.get(phase, str(start.get("skill", ""))),
                    "start_s": start["task_time_s"],
                    "end_s": max(start["task_time_s"], event["task_time_s"]),
                }
            )
    return sorted(windows, key=lambda row: row["start_s"])


def task_phase_at_time(windows: list[dict[str, Any]], rel_time_s: float) -> dict[str, Any] | None:
    for window in windows:
        if window["start_s"] <= rel_time_s <= window["end_s"]:
            return window
    if not windows:
        return None
    past = [window for window in windows if window["start_s"] <= rel_time_s]
    return past[-1] if past else None


def nearest_sim_row(sim_rows: list[dict[str, Any]], rel_time_s: float, start_index: int) -> tuple[int, dict[str, Any] | None]:
    if not sim_rows:
        return start_index, None
    index = min(start_index, len(sim_rows) - 1)
    while index + 1 < len(sim_rows) and abs(sim_rows[index + 1]["rel_time_s"] - rel_time_s) <= abs(
        sim_rows[index]["rel_time_s"] - rel_time_s
    ):
        index += 1
    return index, sim_rows[index]


def displacement_for_phase(sim_rows: list[dict[str, Any]], window: dict[str, Any] | None) -> float | None:
    if not sim_rows or window is None:
        return None
    rows = [
        row
        for row in sim_rows
        if window["start_s"] <= row["rel_time_s"] <= window["end_s"]
    ]
    if len(rows) < 2:
        return None
    dx = rows[-1]["base_pos_x"] - rows[0]["base_pos_x"]
    dy = rows[-1]["base_pos_y"] - rows[0]["base_pos_y"]
    return math.sqrt(dx * dx + dy * dy)


def write_episode_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "deploy_time_s",
        "task_phase",
        "current_skill",
        "motion_name",
        "motion_playing",
        "tracking_error_l2",
        "sim_time",
        "base_pos_x",
        "base_pos_y",
        "base_pos_z",
        "torso_pitch",
        "torso_roll",
        "left_foot_contact",
        "right_foot_contact",
        "fall_flag",
        "self_collision",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    sim_state_csv = resolve_sim_state_csv(args)
    task_events_csv = resolve_task_events_csv(args)
    output_dir = args.output_dir or args.deploy_log_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_rows, q_by_index = load_deploy_rows(args.deploy_log_dir)
    sim_rows = prepare_sim_rows(sim_state_csv)
    task_events = load_task_events(task_events_csv)
    windows = phase_windows(task_events)
    read_input_log(args.input_log)  # Keep input-log accepted for interface parity.
    target_rows = read_raw_csv(args.target_motion_log)
    tracking_errors = compute_tracking_errors(deploy_rows, q_by_index, target_rows)

    merged_rows: list[dict[str, Any]] = []
    sim_cursor = 0
    for row, tracking_error in zip(deploy_rows, tracking_errors):
        window = task_phase_at_time(windows, row["deploy_time_s"])
        skill = window["skill"] if window else "unknown"
        phase = window["phase"] if window else ""
        sim_cursor, sim_row = nearest_sim_row(sim_rows, row["deploy_time_s"], sim_cursor)
        merged = {
            **row,
            "task_phase": phase,
            "current_skill": skill,
            "tracking_error_l2": "" if tracking_error is None else tracking_error,
        }
        if sim_row:
            merged.update(sim_row)
        merged_rows.append(merged)

    episode_csv = output_dir / "episode.csv"
    write_episode_csv(episode_csv, merged_rows)

    valid_tracking = [err for err in tracking_errors if err is not None]
    phase_sequence = compressed([window["phase"] for window in windows])
    skill_sequence = compressed([row["current_skill"] for row in merged_rows if row["current_skill"] != "unknown"])
    skill_counts = Counter(row["current_skill"] for row in merged_rows)
    task_completed = bool(task_events and task_events[-1].get("event") in {"done", "stop_control"})

    sim_duration = sim_rows[-1]["rel_time_s"] if sim_rows else 0.0
    deploy_duration = merged_rows[-1]["deploy_time_s"] if merged_rows else 0.0
    task_duration = task_events[-1]["task_time_s"] if task_events else 0.0
    overlap_seconds = max(0.0, min(sim_duration, deploy_duration, task_duration))
    goal_a_window = next((window for window in windows if window["phase"] == "walk_to_A"), None)
    goal_b_window = next((window for window in windows if window["phase"] == "walk_to_B"), None)
    goal_a_displacement = displacement_for_phase(sim_rows, goal_a_window)
    goal_b_displacement = displacement_for_phase(sim_rows, goal_b_window)

    summary = {
        "baseline": "SONIC-Naive",
        "episode_csv": str(episode_csv),
        "sim_state_csv": str(sim_state_csv) if sim_state_csv else None,
        "deploy_log_dir": str(args.deploy_log_dir),
        "task_events_csv": str(task_events_csv) if task_events_csv else None,
        "input_log": str(args.input_log) if args.input_log else None,
        "target_motion_log": str(args.target_motion_log) if args.target_motion_log else None,
        "row_count": len(merged_rows),
        "phase_sequence": phase_sequence,
        "skill_sequence": skill_sequence,
        "skill_counts": dict(skill_counts),
        "task_completed": task_completed,
        "has_walk_stop_gesture_walk": contains_subsequence(
            skill_sequence,
            ["walk", "stop", "gesture", "walk"],
        ),
        "fall_count": sum(row["fall_flag"] for row in sim_rows),
        "self_collision_count": sum(row["self_collision"] for row in sim_rows),
        "sim_deploy_task_overlap_seconds": overlap_seconds,
        "sim_deploy_time_ranges_overlap": overlap_seconds > 0.0,
        "goal_a_displacement_xy": goal_a_displacement,
        "goal_b_displacement_xy": goal_b_displacement,
        "goal_displacement_threshold": args.goal_displacement_threshold,
        "goal_a_displacement_ok": (
            goal_a_displacement is not None and goal_a_displacement >= args.goal_displacement_threshold
        ),
        "goal_b_displacement_ok": (
            goal_b_displacement is not None and goal_b_displacement >= args.goal_displacement_threshold
        ),
        "tracking_error_available": bool(valid_tracking),
        "tracking_error_min": min(valid_tracking) if valid_tracking else None,
        "tracking_error_mean": (
            sum(valid_tracking) / len(valid_tracking) if valid_tracking else None
        ),
        "tracking_error_max": max(valid_tracking) if valid_tracking else None,
    }

    summary_path = output_dir / "episode_summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"Wrote {episode_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
