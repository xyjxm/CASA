"""Summarize a Stage 1 SONIC smoke-test episode.

This script merges the MuJoCo sim episode log with deployment StateLogger CSVs
and produces a compact episode timeline plus a JSON summary for the
walk -> stop -> gesture -> walk validation pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


COMMON_DEPLOY_COLUMNS = {
    "index",
    "time_ms",
    "time_realtime_ms",
    "time_monotonic_ms",
    "ros_timestamp",
}


INPUT_LOG_FIELDS = [
    "motion_index",
    "current_frame",
    "play",
    "start",
    "stop",
    "planner_enabled",
    "planner_initialized",
    "locomotion_mode",
    "movement_x",
    "movement_y",
    "movement_z",
    "facing_x",
    "facing_y",
    "facing_z",
    "movement_speed",
    "height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-log-dir", type=Path, help="Directory containing sim_state.csv")
    parser.add_argument("--sim-state-csv", type=Path, help="Path to sim_state.csv")
    parser.add_argument("--deploy-log-dir", type=Path, required=True, help="Deploy StateLogger CSV dir")
    parser.add_argument("--input-log", type=Path, help="Optional deploy input.csv")
    parser.add_argument("--target-motion-log", type=Path, help="Optional target_motion.csv")
    parser.add_argument("--output-dir", type=Path, help="Output dir for episode.csv/json")
    return parser.parse_args()


def read_dict_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_raw_csv(path: Path | None) -> list[list[float]]:
    if path is None or not path.exists():
        return []
    rows: list[list[float]] = []
    with path.open(newline="") as file:
        for row in csv.reader(file):
            if not row:
                continue
            try:
                rows.append([float(value) for value in row if value != ""])
            except ValueError:
                continue
    return rows


def read_input_log(path: Path | None) -> list[dict[str, float]]:
    raw_rows = read_raw_csv(path)
    input_rows = []
    for row in raw_rows:
        if len(row) < len(INPUT_LOG_FIELDS):
            continue
        input_rows.append({field: row[i] for i, field in enumerate(INPUT_LOG_FIELDS)})
    return input_rows


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    return int(round(to_float(value, float(default))))


def vector_from_row(row: dict[str, str], prefix: str) -> list[float]:
    values = []
    index = 0
    while f"{prefix}_{index}" in row:
        values.append(to_float(row[f"{prefix}_{index}"]))
        index += 1
    return values


def first_signal_value(row: dict[str, str], default: float = 0.0) -> float:
    for key, value in row.items():
        if key not in COMMON_DEPLOY_COLUMNS:
            return to_float(value, default)
    return default


def resolve_sim_state_csv(args: argparse.Namespace) -> Path | None:
    if args.sim_state_csv:
        return args.sim_state_csv
    if args.sim_log_dir:
        return args.sim_log_dir / "sim_state.csv"
    return None


def load_deploy_rows(deploy_log_dir: Path) -> tuple[list[dict[str, Any]], dict[int, list[float]]]:
    motion_rows = read_dict_csv(deploy_log_dir / "motion_name.csv")
    playing_rows = read_dict_csv(deploy_log_dir / "motion_playing.csv")
    q_rows = read_dict_csv(deploy_log_dir / "q.csv")

    playing_by_index = {to_int(row.get("index")): row for row in playing_rows}
    q_by_index = {to_int(row.get("index")): vector_from_row(row, "q") for row in q_rows}

    source_rows = motion_rows if motion_rows else playing_rows
    deploy_rows: list[dict[str, Any]] = []
    for position, row in enumerate(source_rows):
        index = to_int(row.get("index"), position)
        playing_row = playing_by_index.get(index, {})
        motion_name = row.get("motion_name", "") if motion_rows else ""
        deploy_rows.append(
            {
                "index": index,
                "position": position,
                "deploy_time_s": to_float(row.get("time_ms")) / 1000.0,
                "motion_name": motion_name.strip().strip('"'),
                "motion_playing": int(first_signal_value(playing_row, 0.0)),
            }
        )
    return deploy_rows, q_by_index


def planner_skill_for_row(input_rows: list[dict[str, float]], deploy_position: int) -> str:
    if not input_rows:
        return "motion"
    input_index = min(len(input_rows) - 1, deploy_position * 2)
    row = input_rows[input_index]
    movement_norm = math.sqrt(
        row["movement_x"] ** 2 + row["movement_y"] ** 2 + row["movement_z"] ** 2
    )
    facing_y = abs(row["facing_y"])
    speed = row["movement_speed"]
    mode = int(row["locomotion_mode"])
    if mode == 0 or speed == 0.0 or movement_norm < 0.05:
        return "turn" if facing_y > 0.2 else "stop"
    return "walk"


def classify_skill(
    motion_name: str,
    motion_playing: int,
    deploy_position: int,
    input_rows: list[dict[str, float]],
) -> str:
    if motion_playing == 0:
        return "stop"

    name = motion_name.lower()
    if name == "planner_motion":
        return planner_skill_for_row(input_rows, deploy_position)
    if name.startswith("walking") or "walk" in name or "quip" in name:
        return "walk"
    if (
        name.startswith("macarena")
        or name.startswith("dance")
        or "gesture" in name
        or "point" in name
    ):
        return "gesture"
    return "motion"


def compute_tracking_errors(
    deploy_rows: list[dict[str, Any]],
    q_by_index: dict[int, list[float]],
    target_motion_rows: list[list[float]],
) -> list[float | None]:
    errors: list[float | None] = []
    for position, row in enumerate(deploy_rows):
        q = q_by_index.get(row["index"])
        if not q or position >= len(target_motion_rows):
            errors.append(None)
            continue
        target = target_motion_rows[position][-29:]
        if len(q) < 29 or len(target) < 29:
            errors.append(None)
            continue
        errors.append(math.sqrt(sum((q[i] - target[i]) ** 2 for i in range(29))))
    return errors


def prepare_sim_rows(sim_state_csv: Path | None) -> list[dict[str, Any]]:
    rows = read_dict_csv(sim_state_csv)
    if not rows:
        return []
    first_time = to_float(rows[0].get("sim_time"))
    prepared = []
    for row in rows:
        prepared.append(
            {
                "rel_time_s": to_float(row.get("sim_time")) - first_time,
                "sim_time": to_float(row.get("sim_time")),
                "base_pos_x": to_float(row.get("base_pos_x")),
                "base_pos_y": to_float(row.get("base_pos_y")),
                "base_pos_z": to_float(row.get("base_pos_z")),
                "torso_pitch": to_float(row.get("torso_pitch")),
                "torso_roll": to_float(row.get("torso_roll")),
                "left_foot_contact": to_int(row.get("left_foot_contact")),
                "right_foot_contact": to_int(row.get("right_foot_contact")),
                "fall_flag": to_int(row.get("fall_flag")),
                "self_collision": to_int(row.get("self_collision")),
            }
        )
    return prepared


def nearest_sim_row(sim_rows: list[dict[str, Any]], deploy_time_s: float, start_index: int) -> tuple[int, dict[str, Any] | None]:
    if not sim_rows:
        return start_index, None
    index = min(start_index, len(sim_rows) - 1)
    while index + 1 < len(sim_rows) and abs(sim_rows[index + 1]["rel_time_s"] - deploy_time_s) <= abs(
        sim_rows[index]["rel_time_s"] - deploy_time_s
    ):
        index += 1
    return index, sim_rows[index]


def compressed(values: list[str]) -> list[str]:
    out = []
    for value in values:
        if value and (not out or out[-1] != value):
            out.append(value)
    return out


def contains_subsequence(values: list[str], expected: list[str]) -> bool:
    cursor = 0
    for value in values:
        if cursor < len(expected) and value == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def write_episode_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "deploy_time_s",
        "motion_name",
        "motion_playing",
        "current_skill",
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
    output_dir = args.output_dir or args.deploy_log_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    deploy_rows, q_by_index = load_deploy_rows(args.deploy_log_dir)
    sim_rows = prepare_sim_rows(sim_state_csv)
    input_rows = read_input_log(args.input_log)
    target_rows = read_raw_csv(args.target_motion_log)
    tracking_errors = compute_tracking_errors(deploy_rows, q_by_index, target_rows)

    merged_rows: list[dict[str, Any]] = []
    sim_cursor = 0
    for row, tracking_error in zip(deploy_rows, tracking_errors):
        skill = classify_skill(
            row["motion_name"],
            row["motion_playing"],
            row["position"],
            input_rows,
        )
        sim_cursor, sim_row = nearest_sim_row(sim_rows, row["deploy_time_s"], sim_cursor)
        merged = {
            **row,
            "current_skill": skill,
            "tracking_error_l2": "" if tracking_error is None else tracking_error,
        }
        if sim_row:
            merged.update(sim_row)
        merged_rows.append(merged)

    episode_csv = output_dir / "episode.csv"
    write_episode_csv(episode_csv, merged_rows)

    valid_tracking = [err for err in tracking_errors if err is not None]
    skill_sequence = compressed([row["current_skill"] for row in merged_rows])
    skill_counts = Counter(row["current_skill"] for row in merged_rows)
    unique_motion_names = sorted({row["motion_name"] for row in merged_rows if row["motion_name"]})

    sim_duration = sim_rows[-1]["rel_time_s"] if sim_rows else 0.0
    deploy_duration = merged_rows[-1]["deploy_time_s"] if merged_rows else 0.0
    overlap_seconds = max(0.0, min(sim_duration, deploy_duration))

    summary = {
        "episode_csv": str(episode_csv),
        "sim_state_csv": str(sim_state_csv) if sim_state_csv else None,
        "deploy_log_dir": str(args.deploy_log_dir),
        "input_log": str(args.input_log) if args.input_log else None,
        "target_motion_log": str(args.target_motion_log) if args.target_motion_log else None,
        "row_count": len(merged_rows),
        "unique_motion_count": len(unique_motion_names),
        "unique_motion_names": unique_motion_names,
        "skill_sequence": skill_sequence,
        "skill_counts": dict(skill_counts),
        "has_walk_stop_gesture_walk": contains_subsequence(
            skill_sequence,
            ["walk", "stop", "gesture", "walk"],
        ),
        "fall_count": sum(row["fall_flag"] for row in sim_rows),
        "self_collision_count": sum(row["self_collision"] for row in sim_rows),
        "sim_deploy_overlap_seconds": overlap_seconds,
        "sim_deploy_time_ranges_overlap": overlap_seconds > 0.0,
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
