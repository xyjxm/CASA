"""Run CASA oracle on one rollout directory or explicit log files."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.loggers.rollout_logger import RolloutLogger
from gear_sonic.casa.oracle import SafetyOracle
from gear_sonic.casa.oracle.rollout_summary import build_rollout_summary
from gear_sonic.casa.runner_utils import create_executor
from gear_sonic.casa.skills import GestureSkill, PassiveSkill, TurnSkill, WalkSkill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="phase2_oracle")
    parser.add_argument("--rollout-id", default="episode_000000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-log-dir", type=Path, help="Directory containing sim_state.csv")
    parser.add_argument("--sim-state-csv", type=Path, help="Explicit sim_state.csv path")
    parser.add_argument("--skill-events-csv", type=Path)
    parser.add_argument("--thresholds", type=Path, help="Optional thresholds.yaml")
    parser.add_argument("--scene-props-json", type=Path, help="Optional actual prop placement json")
    parser.add_argument("--post-horizon", type=float, default=2.0)
    parser.add_argument("--run-skills", action="store_true", help="Publish a short random skill rollout before oracle")
    parser.add_argument("--zmq-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--min-skill-duration", type=float, default=3.0)
    parser.add_argument("--max-skill-duration", type=float, default=5.0)
    parser.add_argument("--stabilize-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--inject-latency-ms", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_state_csv = args.sim_state_csv or (args.sim_log_dir / "sim_state.csv" if args.sim_log_dir else None)
    if sim_state_csv is None:
        raise ValueError("Provide --sim-state-csv or --sim-log-dir")
    if not sim_state_csv.exists():
        raise FileNotFoundError(sim_state_csv)
    skill_events_csv = args.skill_events_csv
    if args.run_skills:
        _run_skill_rollout(args)
        skill_events_csv = args.output_dir / "skill_events.csv"
    if skill_events_csv is None:
        raise ValueError("Provide --skill-events-csv or use --run-skills")
    if not skill_events_csv.exists():
        raise FileNotFoundError(skill_events_csv)

    oracle = SafetyOracle.from_thresholds_file(args.thresholds)
    violations = oracle.detect(sim_state_csv, skill_events_csv)
    labels = oracle.label_skill_calls(
        violations,
        sim_state_csv,
        skill_events_csv,
        post_horizon=args.post_horizon,
    )
    scene_props = {}
    scene_props_json = args.scene_props_json
    if scene_props_json is None and args.sim_log_dir is not None:
        candidate = args.sim_log_dir / "scene_props.json"
        if candidate.exists():
            scene_props_json = candidate
    if scene_props_json and scene_props_json.exists():
        scene_props = json.loads(scene_props_json.read_text())
    summary = build_rollout_summary(
        run_id=args.run_id,
        rollout_id=args.rollout_id,
        violations=violations,
        skill_labels=labels,
        scene_props=scene_props,
        extra={
            "sim_state_csv": str(sim_state_csv),
            "skill_events_csv": str(skill_events_csv),
            "post_horizon": args.post_horizon,
            "event_start_time_s": _event_start_time_s(violations),
            **_sim_aggregates(sim_state_csv),
        },
    )
    RolloutLogger(args.output_dir).write(violations, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run_skill_rollout(args: argparse.Namespace) -> None:
    if args.sim_log_dir is None:
        raise ValueError("--run-skills requires --sim-log-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    executor = create_executor(args, args.output_dir, args.run_id, args.sim_log_dir)
    elapsed = 0.0
    start = time.monotonic()
    try:
        executor.start_policy()
        while elapsed < args.duration_seconds:
            skill = _sample_skill(rng, args)
            if skill.name in {"walk", "gesture"} and args.stabilize_seconds > 0:
                executor.execute_one(PassiveSkill(duration=args.stabilize_seconds, mode="wait"))
                if args.inject_latency_ms > 0:
                    time.sleep(args.inject_latency_ms / 1000.0)
            executor.execute_one(skill)
            if args.inject_latency_ms > 0:
                time.sleep(args.inject_latency_ms / 1000.0)
            elapsed = time.monotonic() - start if not args.dry_run else elapsed + skill.estimated_duration
    finally:
        executor.close()
    if args.inject_latency_ms > 0:
        _annotate_injected_latency(args.output_dir / "skill_events.csv", args.inject_latency_ms)


def _sample_skill(rng: random.Random, args: argparse.Namespace):
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
    return PassiveSkill(duration=duration, mode="wait")


def _annotate_injected_latency(skill_events_csv: Path, latency_ms: float) -> None:
    if not skill_events_csv.exists():
        return
    rows = list(csv.DictReader(skill_events_csv.open(newline="")))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        evidence = {}
        if row.get("evidence_json"):
            try:
                evidence = json.loads(row["evidence_json"])
            except json.JSONDecodeError:
                evidence = {}
        evidence["injected_latency_ms"] = latency_ms
        row["evidence_json"] = json.dumps(evidence, sort_keys=True)
    with skill_events_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sim_aggregates(sim_state_csv: Path) -> dict:
    rows = list(csv.DictReader(sim_state_csv.open(newline="")))
    first_row = rows[0] if rows else {}
    event_rows = [
        row
        for row in rows
        if _sum_row_flags(row, ["external_collision_user", "external_collision_obstacle", "fall_flag"]) > 0
    ]
    elastic_enabled_rows = _sum_int(rows, "elastic_band_enabled")
    return {
        "min_obstacle_distance": _min_optional(rows, "min_obstacle_distance"),
        "min_user_distance": _min_optional(rows, "min_user_distance"),
        "min_arm_user_distance": _min_optional(rows, "min_arm_user_distance"),
        "initial_min_user_distance": _float(first_row.get("min_user_distance")),
        "initial_min_obstacle_distance": _float(first_row.get("min_obstacle_distance")),
        "initial_external_collision_user": _int(first_row.get("external_collision_user")),
        "initial_external_collision_obstacle": _int(first_row.get("external_collision_obstacle")),
        "first_physical_event_sim_time": _float(event_rows[0].get("sim_time")) if event_rows else None,
        "max_torso_pitch": _max_abs(rows, "torso_pitch"),
        "max_torso_roll": _max_abs(rows, "torso_roll"),
        "min_base_height": _min_optional(rows, "base_pos_z"),
        "left_foot_contact_rows": _sum_int(rows, "left_foot_contact"),
        "right_foot_contact_rows": _sum_int(rows, "right_foot_contact"),
        "control_loop_overrun_rows": _sum_int(rows, "control_loop_overrun"),
        "elastic_band_enabled_any": bool(elastic_enabled_rows),
        "elastic_band_enabled_ratio": elastic_enabled_rows / len(rows) if rows else None,
        "max_elastic_band_force_norm": _max_optional(rows, "elastic_band_force_norm"),
    }


def _min_optional(rows: list[dict], field: str) -> float | None:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _max_abs(rows: list[dict], field: str) -> float | None:
    values = [_float(row.get(field)) for row in rows]
    values = [abs(value) for value in values if value is not None]
    return max(values) if values else None


def _max_optional(rows: list[dict], field: str) -> float | None:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _sum_int(rows: list[dict], field: str) -> int:
    return sum(int(float(row.get(field) or 0)) for row in rows)


def _sum_row_flags(row: dict, fields: list[str]) -> int:
    return sum(_int(row.get(field)) for field in fields)


def _int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _event_start_time_s(violations) -> float | None:
    if not violations:
        return None
    first = min(violations, key=lambda violation: violation.start_sim_time or 0.0)
    if first.start_sim_time is not None:
        return max(0.0, float(first.start_sim_time))
    return None


def _float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
