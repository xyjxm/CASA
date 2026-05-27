"""Compute Stage 3 switch-safety metrics for SONIC-Naive episodes.

Stage 3 is an offline judge. It aligns task, deploy, and sim logs by wall time,
marks unsafe states and switch outcomes, and deliberately separates pre-task
and post-task falls from task-window or switch-induced failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from summarize_sonic_stage1_episode import (
    compressed,
    compute_tracking_errors,
    contains_subsequence,
    first_signal_value,
    read_dict_csv,
    read_raw_csv,
    to_float,
    to_int,
    vector_from_row,
)


PHASE_TO_SKILL = {
    "walk_to_A": "walk",
    "walk_to_B": "walk",
    "stop": "stop",
    "face_user": "stop",
    "gesture": "gesture",
}


@dataclass(frozen=True)
class MetricsConfig:
    near_fall_height: float = 0.55
    torso_unstable_rad: float = 0.35
    near_fall_torso_rad: float = 0.6
    gesture_max_speed: float = 0.25
    tracking_precondition_threshold: float = 1.5
    tracking_failure_threshold: float = 2.0
    pre_switch_window: float = 1.0
    post_switch_window: float = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-log-dir", type=Path, help="Directory containing sim_state.csv")
    parser.add_argument("--sim-state-csv", type=Path, help="Path to sim_state.csv")
    parser.add_argument("--deploy-log-dir", type=Path, required=True, help="Deploy StateLogger CSV dir")
    parser.add_argument("--task-log-dir", type=Path, help="Directory containing task_events.csv")
    parser.add_argument("--task-events-csv", type=Path, help="Path to task_events.csv")
    parser.add_argument("--target-motion-log", type=Path, help="Optional target_motion.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--near-fall-height", type=float, default=MetricsConfig.near_fall_height)
    parser.add_argument("--torso-unstable-rad", type=float, default=MetricsConfig.torso_unstable_rad)
    parser.add_argument("--near-fall-torso-rad", type=float, default=MetricsConfig.near_fall_torso_rad)
    parser.add_argument("--gesture-max-speed", type=float, default=MetricsConfig.gesture_max_speed)
    parser.add_argument(
        "--tracking-precondition-threshold",
        type=float,
        default=MetricsConfig.tracking_precondition_threshold,
    )
    parser.add_argument(
        "--tracking-failure-threshold",
        type=float,
        default=MetricsConfig.tracking_failure_threshold,
    )
    parser.add_argument("--pre-switch-window", type=float, default=MetricsConfig.pre_switch_window)
    parser.add_argument("--post-switch-window", type=float, default=MetricsConfig.post_switch_window)
    return parser.parse_args()


def resolve_sim_state_csv(args: argparse.Namespace) -> Path | None:
    if args.sim_state_csv:
        return args.sim_state_csv
    if args.sim_log_dir:
        return args.sim_log_dir / "sim_state.csv"
    return None


def resolve_task_events_csv(args: argparse.Namespace) -> Path | None:
    if args.task_events_csv:
        return args.task_events_csv
    if args.task_log_dir:
        return args.task_log_dir / "task_events.csv"
    return None


def wall_seconds_from_realtime_ms(row: dict[str, str]) -> float:
    value = to_float(row.get("time_realtime_ms"))
    return value / 1000.0 if value else 0.0


def read_sim_rows(path: Path | None) -> list[dict[str, Any]]:
    rows = []
    for row in read_dict_csv(path):
        sim_row = {
            "sim_wall_time_s": to_float(row.get("wall_time")),
            "sim_time": to_float(row.get("sim_time")),
            "reset_count": to_int(row.get("reset_count")),
            "fall_flag": to_int(row.get("fall_flag")),
            "fall_count": to_int(row.get("fall_count")),
            "base_pos_x": to_float(row.get("base_pos_x")),
            "base_pos_y": to_float(row.get("base_pos_y")),
            "base_pos_z": to_float(row.get("base_pos_z")),
            "base_lin_vel_x": to_float(row.get("base_lin_vel_x")),
            "base_lin_vel_y": to_float(row.get("base_lin_vel_y")),
            "base_lin_vel_z": to_float(row.get("base_lin_vel_z")),
            "base_ang_vel_x": to_float(row.get("base_ang_vel_x")),
            "base_ang_vel_y": to_float(row.get("base_ang_vel_y")),
            "base_ang_vel_z": to_float(row.get("base_ang_vel_z")),
            "torso_pitch": to_float(row.get("torso_pitch")),
            "torso_roll": to_float(row.get("torso_roll")),
            "left_foot_contact": to_int(row.get("left_foot_contact")),
            "right_foot_contact": to_int(row.get("right_foot_contact")),
            "self_collision": to_int(row.get("self_collision")),
            "self_collision_pair_count": to_int(row.get("self_collision_pair_count")),
        }
        rows.append(sim_row)
    return sorted(rows, key=lambda item: item["sim_wall_time_s"])


def load_deploy_wall_rows(deploy_log_dir: Path) -> tuple[list[dict[str, Any]], dict[int, list[float]]]:
    motion_rows = read_dict_csv(deploy_log_dir / "motion_name.csv")
    playing_rows = read_dict_csv(deploy_log_dir / "motion_playing.csv")
    q_rows = read_dict_csv(deploy_log_dir / "q.csv")

    playing_by_index = {to_int(row.get("index")): row for row in playing_rows}
    q_by_index = {to_int(row.get("index")): vector_from_row(row, "q") for row in q_rows}

    source_rows = motion_rows if motion_rows else playing_rows
    deploy_rows = []
    for position, row in enumerate(source_rows):
        index = to_int(row.get("index"), position)
        playing_row = playing_by_index.get(index, {})
        deploy_rows.append(
            {
                "index": index,
                "position": position,
                "deploy_time_s": to_float(row.get("time_ms")) / 1000.0,
                "deploy_wall_time_s": wall_seconds_from_realtime_ms(row),
                "motion_name": row.get("motion_name", "").strip().strip('"') if motion_rows else "",
                "motion_playing": int(first_signal_value(playing_row, 0.0)),
            }
        )
    return deploy_rows, q_by_index


def load_task_events(path: Path | None) -> list[dict[str, Any]]:
    events = []
    for row in read_dict_csv(path):
        event = {
            **row,
            "task_time_s": to_float(row.get("task_time_s")),
            "task_wall_time_s": to_float(row.get("wall_time_s")),
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
            "switch_id": to_int(row.get("switch_id"), 0) if row.get("switch_id") else 0,
            "switch_requested": to_int(row.get("switch_requested"), 0)
            if row.get("switch_requested")
            else 0,
            "switch_allowed": to_int(row.get("switch_allowed"), 1) if row.get("switch_allowed") else 1,
            "switch_rejected_reason": row.get("switch_rejected_reason", ""),
        }
        events.append(event)
    return sorted(events, key=lambda item: item["task_wall_time_s"])


def task_bounds(task_events: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not task_events:
        return None, None
    start = next(
        (event["task_wall_time_s"] for event in task_events if event.get("event") == "task_start"),
        task_events[0]["task_wall_time_s"],
    )
    end = next(
        (
            event["task_wall_time_s"]
            for event in reversed(task_events)
            if event.get("event") in {"done", "stop_control"}
        ),
        task_events[-1]["task_wall_time_s"],
    )
    return start, max(start, end)


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
                    "start_wall_time_s": start["task_wall_time_s"],
                    "end_wall_time_s": max(start["task_wall_time_s"], event["task_wall_time_s"]),
                    "start_task_time_s": start["task_time_s"],
                    "end_task_time_s": max(start["task_time_s"], event["task_time_s"]),
                }
            )
    return sorted(windows, key=lambda row: row["start_wall_time_s"])


def derive_switch_events(task_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    switches = []
    previous_skill = "init"
    for event in task_events:
        if event.get("event") != "phase_start":
            continue
        phase = str(event.get("phase", ""))
        target_skill = event.get("target_skill") or PHASE_TO_SKILL.get(phase, str(event.get("skill", "")))
        source_skill = event.get("source_skill") or previous_skill
        switch_id = event["switch_id"] or (len(switches) + 1)
        requested = event["switch_requested"] or 1
        switches.append(
            {
                "switch_id": switch_id,
                "switch_wall_time_s": event["task_wall_time_s"],
                "switch_task_time_s": event["task_time_s"],
                "phase": phase,
                "source_skill": source_skill,
                "target_skill": target_skill,
                "switch_requested": requested,
                "switch_allowed": event["switch_allowed"],
                "switch_rejected_reason": event["switch_rejected_reason"],
            }
        )
        previous_skill = target_skill
    return switches


def nearest_by_wall(
    rows: list[dict[str, Any]],
    wall_time_s: float,
    start_index: int,
    wall_key: str,
) -> tuple[int, dict[str, Any] | None]:
    if not rows:
        return start_index, None
    index = min(start_index, len(rows) - 1)
    while index + 1 < len(rows) and abs(rows[index + 1][wall_key] - wall_time_s) <= abs(
        rows[index][wall_key] - wall_time_s
    ):
        index += 1
    return index, rows[index]


def nearest_sim_for_segment(
    sim_rows: list[dict[str, Any]],
    wall_time_s: float,
    start_index: int,
    segment: str,
    task_start: float | None,
    task_end: float | None,
) -> tuple[int, dict[str, Any] | None]:
    index, candidate = nearest_by_wall(sim_rows, wall_time_s, start_index, "sim_wall_time_s")
    if candidate is None or segment == "unknown":
        return index, candidate
    if segment_for_wall(candidate["sim_wall_time_s"], task_start, task_end) == segment:
        return index, candidate

    best_index = index
    best_row = None
    best_distance = float("inf")
    for direction in (-1, 1):
        cursor = index + direction
        while 0 <= cursor < len(sim_rows):
            row = sim_rows[cursor]
            distance = abs(row["sim_wall_time_s"] - wall_time_s)
            if distance > best_distance:
                break
            if segment_for_wall(row["sim_wall_time_s"], task_start, task_end) == segment:
                best_index = cursor
                best_row = row
                best_distance = distance
                break
            cursor += direction
    return best_index, best_row


def segment_for_wall(wall_time_s: float, task_start: float | None, task_end: float | None) -> str:
    if task_start is None or task_end is None:
        return "unknown"
    if wall_time_s < task_start:
        return "pre_task"
    if wall_time_s > task_end:
        return "post_task"
    return "task"


def phase_at_wall(windows: list[dict[str, Any]], wall_time_s: float) -> dict[str, Any] | None:
    for window in windows:
        if window["start_wall_time_s"] <= wall_time_s <= window["end_wall_time_s"]:
            return window
    past = [window for window in windows if window["start_wall_time_s"] <= wall_time_s]
    return past[-1] if past else None


def metric_flags(
    sim_row: dict[str, Any] | None,
    tracking_error: float | None,
    config: MetricsConfig,
) -> dict[str, Any]:
    if sim_row is None:
        tracking_precondition_high = (
            tracking_error is not None and tracking_error > config.tracking_precondition_threshold
        )
        tracking_diverged = (
            tracking_error is not None and tracking_error > config.tracking_failure_threshold
        )
        return {
            "base_speed": 0.0,
            "near_fall_flag": 0,
            "torso_unstable": 0,
            "foot_contact_unstable": 0,
            "high_speed_for_gesture": 0,
            "tracking_precondition_high": int(tracking_precondition_high),
            "tracking_diverged": int(tracking_diverged),
            "fall_flag": 0,
            "collision_flag": 0,
            "failure_flag": int(tracking_diverged),
        }

    base_speed = math.sqrt(
        to_float(sim_row.get("base_lin_vel_x")) ** 2 + to_float(sim_row.get("base_lin_vel_y")) ** 2
    )
    torso_pitch_abs = abs(to_float(sim_row.get("torso_pitch")))
    torso_roll_abs = abs(to_float(sim_row.get("torso_roll")))
    near_fall = (
        to_float(sim_row.get("base_pos_z")) < config.near_fall_height
        or torso_pitch_abs > config.near_fall_torso_rad
        or torso_roll_abs > config.near_fall_torso_rad
    )
    torso_unstable = (
        torso_pitch_abs > config.torso_unstable_rad
        or torso_roll_abs > config.torso_unstable_rad
    )
    left_contact = to_int(sim_row.get("left_foot_contact"))
    right_contact = to_int(sim_row.get("right_foot_contact"))
    tracking_precondition_high = (
        tracking_error is not None and tracking_error > config.tracking_precondition_threshold
    )
    tracking_diverged = (
        tracking_error is not None and tracking_error > config.tracking_failure_threshold
    )
    fall_flag = to_int(sim_row.get("fall_flag"))
    collision_flag = to_int(sim_row.get("self_collision"))
    return {
        "base_speed": base_speed,
        "near_fall_flag": int(near_fall),
        "torso_unstable": int(torso_unstable),
        "foot_contact_unstable": int(not (left_contact and right_contact)),
        "high_speed_for_gesture": int(base_speed > config.gesture_max_speed),
        "tracking_precondition_high": int(tracking_precondition_high),
        "tracking_diverged": int(tracking_diverged),
        "fall_flag": fall_flag,
        "collision_flag": collision_flag,
        "failure_flag": int(fall_flag or near_fall or collision_flag or tracking_diverged),
    }


def build_timeline(
    deploy_rows: list[dict[str, Any]],
    sim_rows: list[dict[str, Any]],
    task_start: float | None,
    task_end: float | None,
    windows: list[dict[str, Any]],
    tracking_errors: list[float | None],
    config: MetricsConfig,
) -> list[dict[str, Any]]:
    rows = []
    sim_cursor = 0
    for deploy_row, tracking_error in zip(deploy_rows, tracking_errors):
        wall_time_s = deploy_row["deploy_wall_time_s"]
        segment = segment_for_wall(wall_time_s, task_start, task_end)
        sim_cursor, sim_row = nearest_sim_for_segment(
            sim_rows,
            wall_time_s,
            sim_cursor,
            segment,
            task_start,
            task_end,
        )
        phase = phase_at_wall(windows, wall_time_s) if segment == "task" else None
        current_skill = phase["skill"] if phase else segment
        flags = metric_flags(sim_row, tracking_error, config)
        gesture_active = current_skill == "gesture"
        unsafe_gesture = int(
            segment == "task"
            and gesture_active
            and (
                flags["high_speed_for_gesture"]
                or flags["torso_unstable"]
                or flags["near_fall_flag"]
                or flags["collision_flag"]
                or flags["tracking_diverged"]
                or flags["foot_contact_unstable"]
            )
        )
        merged = {
            "wall_time_s": wall_time_s,
            "segment": segment,
            "task_phase": phase["phase"] if phase else "",
            "current_skill": current_skill,
            "deploy_index": deploy_row["index"],
            "deploy_time_s": deploy_row["deploy_time_s"],
            "motion_name": deploy_row["motion_name"],
            "motion_playing": deploy_row["motion_playing"],
            "tracking_error_l2": "" if tracking_error is None else tracking_error,
            "gesture_active": int(gesture_active),
            "unsafe_gesture_flag": unsafe_gesture,
            **flags,
        }
        if sim_row:
            merged.update(
                {
                    "sim_time": sim_row["sim_time"],
                    "sim_wall_time_s": sim_row["sim_wall_time_s"],
                    "base_pos_x": sim_row["base_pos_x"],
                    "base_pos_y": sim_row["base_pos_y"],
                    "base_pos_z": sim_row["base_pos_z"],
                    "base_lin_vel_x": sim_row["base_lin_vel_x"],
                    "base_lin_vel_y": sim_row["base_lin_vel_y"],
                    "base_lin_vel_z": sim_row["base_lin_vel_z"],
                    "torso_pitch": sim_row["torso_pitch"],
                    "torso_roll": sim_row["torso_roll"],
                    "left_foot_contact": sim_row["left_foot_contact"],
                    "right_foot_contact": sim_row["right_foot_contact"],
                    "self_collision_pair_count": sim_row["self_collision_pair_count"],
                }
            )
        rows.append(merged)
    return rows


def failure_reasons(row: dict[str, Any]) -> set[str]:
    reasons = set()
    if to_int(row.get("fall_flag")):
        reasons.add("fall")
    if to_int(row.get("near_fall_flag")):
        reasons.add("near_fall")
    if to_int(row.get("collision_flag")):
        reasons.add("collision")
    if to_int(row.get("tracking_diverged")):
        reasons.add("tracking_divergence")
    return reasons


def rows_in_window(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["segment"] == "task" and start <= row["wall_time_s"] <= end
    ]


def nearest_timeline_row(rows: list[dict[str, Any]], wall_time_s: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(row["wall_time_s"] - wall_time_s))


def is_expressive_target(switch: dict[str, Any]) -> bool:
    return switch["target_skill"] == "gesture" or "gesture" in switch["phase"].lower()


def evaluate_switches(
    switches: list[dict[str, Any]],
    timeline_rows: list[dict[str, Any]],
    config: MetricsConfig,
) -> list[dict[str, Any]]:
    evaluated = []
    for switch in switches:
        switch_time = switch["switch_wall_time_s"]
        pre_rows = rows_in_window(timeline_rows, switch_time - config.pre_switch_window, switch_time)
        post_rows = rows_in_window(timeline_rows, switch_time, switch_time + config.post_switch_window)
        switch_row = nearest_timeline_row(timeline_rows, switch_time)

        invalid_reasons = []
        if is_expressive_target(switch):
            if any(row["high_speed_for_gesture"] for row in pre_rows):
                invalid_reasons.append("high_speed_pre")
            if any(row["torso_unstable"] for row in pre_rows):
                invalid_reasons.append("torso_unstable_pre")
            if any(row["near_fall_flag"] for row in pre_rows):
                invalid_reasons.append("near_fall_pre")
            if any(row["collision_flag"] for row in pre_rows):
                invalid_reasons.append("collision_pre")
            if any(row["tracking_precondition_high"] for row in pre_rows):
                invalid_reasons.append("tracking_high_pre")
            if switch_row and switch_row["foot_contact_unstable"]:
                invalid_reasons.append("foot_contact_unstable_at_switch")

        pre_reasons = set().union(*(failure_reasons(row) for row in pre_rows)) if pre_rows else set()
        induced_reasons: set[str] = set()
        first_failure_time = None
        for row in post_rows:
            new_reasons = failure_reasons(row) - pre_reasons
            if new_reasons:
                induced_reasons = new_reasons
                first_failure_time = row["wall_time_s"]
                break

        evaluated.append(
            {
                **switch,
                "pre_window_rows": len(pre_rows),
                "post_window_rows": len(post_rows),
                "invalid_switch": int(bool(invalid_reasons)),
                "invalid_switch_reasons": ";".join(invalid_reasons),
                "pre_failure_reasons": ";".join(sorted(pre_reasons)),
                "switch_induced_failure": int(bool(induced_reasons)),
                "switch_induced_failure_reasons": ";".join(sorted(induced_reasons)),
                "first_failure_after_switch_wall_time_s": "" if first_failure_time is None else first_failure_time,
            }
        )
    return evaluated


def sim_failure_counts(
    sim_rows: list[dict[str, Any]],
    task_start: float | None,
    task_end: float | None,
    segment: str,
    config: MetricsConfig,
) -> dict[str, int]:
    counts = {"rows": 0, "fall_rows": 0, "near_fall_rows": 0, "collision_rows": 0}
    for row in sim_rows:
        current_segment = segment_for_wall(row["sim_wall_time_s"], task_start, task_end)
        if current_segment != segment:
            continue
        counts["rows"] += 1
        flags = metric_flags(row, None, config)
        counts["fall_rows"] += flags["fall_flag"]
        counts["near_fall_rows"] += flags["near_fall_flag"]
        counts["collision_rows"] += flags["collision_flag"]
    return counts


def count_unsafe_gesture_phases(timeline_rows: list[dict[str, Any]]) -> int:
    unsafe_phases = {
        row["task_phase"]
        for row in timeline_rows
        if row["segment"] == "task" and row["current_skill"] == "gesture" and row["unsafe_gesture_flag"]
    }
    return len(unsafe_phases)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    config = MetricsConfig(
        near_fall_height=args.near_fall_height,
        torso_unstable_rad=args.torso_unstable_rad,
        near_fall_torso_rad=args.near_fall_torso_rad,
        gesture_max_speed=args.gesture_max_speed,
        tracking_precondition_threshold=args.tracking_precondition_threshold,
        tracking_failure_threshold=args.tracking_failure_threshold,
        pre_switch_window=args.pre_switch_window,
        post_switch_window=args.post_switch_window,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim_state_csv = resolve_sim_state_csv(args)
    task_events_csv = resolve_task_events_csv(args)
    deploy_rows, q_by_index = load_deploy_wall_rows(args.deploy_log_dir)
    sim_rows = read_sim_rows(sim_state_csv)
    task_events = load_task_events(task_events_csv)
    windows = phase_windows(task_events)
    switches = derive_switch_events(task_events)
    task_start, task_end = task_bounds(task_events)
    target_rows = read_raw_csv(args.target_motion_log)
    tracking_errors = compute_tracking_errors(deploy_rows, q_by_index, target_rows)
    timeline_rows = build_timeline(
        deploy_rows,
        sim_rows,
        task_start,
        task_end,
        windows,
        tracking_errors,
        config,
    )
    switch_rows = evaluate_switches(switches, timeline_rows, config)

    metrics_timeline_path = args.output_dir / "metrics_timeline.csv"
    switch_events_path = args.output_dir / "switch_events.csv"
    episode_metrics_path = args.output_dir / "episode_metrics.json"
    metrics_config_path = args.output_dir / "metrics_config.json"

    timeline_fields = [
        "wall_time_s",
        "segment",
        "task_phase",
        "current_skill",
        "deploy_index",
        "deploy_time_s",
        "motion_name",
        "motion_playing",
        "tracking_error_l2",
        "sim_time",
        "base_pos_x",
        "base_pos_y",
        "base_pos_z",
        "base_lin_vel_x",
        "base_lin_vel_y",
        "base_lin_vel_z",
        "base_speed",
        "torso_pitch",
        "torso_roll",
        "left_foot_contact",
        "right_foot_contact",
        "foot_contact_unstable",
        "fall_flag",
        "near_fall_flag",
        "torso_unstable",
        "collision_flag",
        "self_collision_pair_count",
        "tracking_precondition_high",
        "tracking_diverged",
        "failure_flag",
        "gesture_active",
        "unsafe_gesture_flag",
    ]
    switch_fields = [
        "switch_id",
        "switch_wall_time_s",
        "switch_task_time_s",
        "phase",
        "source_skill",
        "target_skill",
        "switch_requested",
        "switch_allowed",
        "switch_rejected_reason",
        "pre_window_rows",
        "post_window_rows",
        "invalid_switch",
        "invalid_switch_reasons",
        "pre_failure_reasons",
        "switch_induced_failure",
        "switch_induced_failure_reasons",
        "first_failure_after_switch_wall_time_s",
    ]
    write_csv(metrics_timeline_path, timeline_rows, timeline_fields)
    write_csv(switch_events_path, switch_rows, switch_fields)

    task_timeline_rows = [row for row in timeline_rows if row["segment"] == "task"]
    valid_tracking = [
        to_float(row["tracking_error_l2"])
        for row in timeline_rows
        if row.get("tracking_error_l2") not in {"", None}
    ]
    skill_sequence = compressed(
        [
            row["current_skill"]
            for row in task_timeline_rows
            if row["current_skill"] not in {"unknown", "pre_task", "post_task"}
        ]
    )
    task_completed = bool(task_events and task_events[-1].get("event") in {"done", "stop_control"})
    has_walk_stop_gesture_walk = contains_subsequence(skill_sequence, ["walk", "stop", "gesture", "walk"])
    pre_task_failures = sim_failure_counts(sim_rows, task_start, task_end, "pre_task", config)
    task_failures = sim_failure_counts(sim_rows, task_start, task_end, "task", config)
    post_task_failures = sim_failure_counts(sim_rows, task_start, task_end, "post_task", config)

    falls = task_failures["fall_rows"]
    episode_metrics = {
        "baseline": "SONIC-Naive",
        "task_success": bool(task_completed and has_walk_stop_gesture_walk and falls == 0),
        "task_completed": task_completed,
        "has_walk_stop_gesture_walk": has_walk_stop_gesture_walk,
        "completion_time": None if task_start is None or task_end is None else task_end - task_start,
        "number_of_switches": len(switch_rows),
        "invalid_switch_attempts": sum(row["invalid_switch"] for row in switch_rows),
        "rejected_switches": sum(1 for row in switch_rows if not row["switch_allowed"]),
        "switch_induced_failures": sum(row["switch_induced_failure"] for row in switch_rows),
        "unsafe_gestures": count_unsafe_gesture_phases(timeline_rows),
        "unsafe_gesture_rows": sum(row["unsafe_gesture_flag"] for row in task_timeline_rows),
        "near_falls": task_failures["near_fall_rows"],
        "falls": falls,
        "collisions": task_failures["collision_rows"],
        "recovery_triggered": False,
        "recovery_success": None,
        "distance_to_obstacle_available": False,
        "distance_to_edge_available": False,
        "recovery_available": False,
        "tracking_error_available": bool(valid_tracking),
        "tracking_error_min": min(valid_tracking) if valid_tracking else None,
        "tracking_error_mean": sum(valid_tracking) / len(valid_tracking) if valid_tracking else None,
        "tracking_error_max": max(valid_tracking) if valid_tracking else None,
        "skill_sequence": skill_sequence,
        "phase_sequence": compressed([window["phase"] for window in windows]),
        "pre_task_failures": pre_task_failures,
        "task_failures": task_failures,
        "post_task_failures": post_task_failures,
        "metrics_timeline_csv": str(metrics_timeline_path),
        "switch_events_csv": str(switch_events_path),
        "metrics_config_json": str(metrics_config_path),
        "sim_state_csv": str(sim_state_csv) if sim_state_csv else None,
        "deploy_log_dir": str(args.deploy_log_dir),
        "task_events_csv": str(task_events_csv) if task_events_csv else None,
        "target_motion_log": str(args.target_motion_log) if args.target_motion_log else None,
    }

    with episode_metrics_path.open("w") as file:
        json.dump(episode_metrics, file, indent=2)

    config_payload = {
        "thresholds": asdict(config),
        "input_paths": {
            "sim_state_csv": str(sim_state_csv) if sim_state_csv else None,
            "deploy_log_dir": str(args.deploy_log_dir),
            "task_events_csv": str(task_events_csv) if task_events_csv else None,
            "target_motion_log": str(args.target_motion_log) if args.target_motion_log else None,
        },
        "output_paths": {
            "metrics_timeline_csv": str(metrics_timeline_path),
            "switch_events_csv": str(switch_events_path),
            "episode_metrics_json": str(episode_metrics_path),
        },
    }
    with metrics_config_path.open("w") as file:
        json.dump(config_payload, file, indent=2)

    print(f"Wrote {metrics_timeline_path}")
    print(f"Wrote {switch_events_path}")
    print(f"Wrote {episode_metrics_path}")
    print(f"Wrote {metrics_config_path}")


if __name__ == "__main__":
    main()
