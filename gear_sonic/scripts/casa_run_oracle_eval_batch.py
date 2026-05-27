"""Run a batch of CASA Phase 2 oracle rollouts against a live sim/deploy pair."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.loggers.rollout_logger import RolloutLogger
from gear_sonic.casa.oracle import SafetyOracle
from gear_sonic.casa.oracle.rollout_summary import build_rollout_summary
from gear_sonic.casa.runner_utils import create_executor
from gear_sonic.casa.scene.phase2_v2 import make_phase2_v2_scene_command
from gear_sonic.casa.skills import GestureSkill, PassiveSkill, TurnSkill, WalkSkill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sim-log-dir", type=Path, required=True)
    parser.add_argument("--rollout-count", type=int, default=300)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--post-horizon", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--zmq-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--min-skill-duration", type=float, default=3.0)
    parser.add_argument("--max-skill-duration", type=float, default=6.0)
    parser.add_argument("--stabilize-seconds", type=float, default=2.0)
    parser.add_argument(
        "--pre-rollout-settle-seconds",
        type=float,
        default=0.0,
        help="After policy start, wait this long before slicing the rollout log.",
    )
    parser.add_argument("--casa-props-command-file", type=Path)
    parser.add_argument("--casa-props-ack-file", type=Path)
    parser.add_argument(
        "--scenario-mode",
        choices=["random", "balanced", "cycle"],
        default="balanced",
        help="balanced cycles through default scenes; cycle uses --scenario-cycle.",
    )
    parser.add_argument(
        "--scenario-cycle",
        default="",
        help="Comma-separated scene cycle for --scenario-mode cycle.",
    )
    parser.add_argument("--phase2-v2", action="store_true", help="Use clean natural Phase 2 v2 scene generator.")
    parser.add_argument(
        "--target-bucket-cycle",
        default="",
        help="Comma-separated v2 target bucket cycle, e.g. clean_safe,visual_fall.",
    )
    parser.add_argument(
        "--scene-complexity-cycle",
        default="",
        help="Comma-separated v2 complexity cycle, e.g. simple,medium,hard.",
    )
    parser.add_argument(
        "--scene-plan-csv",
        type=Path,
        help="Optional CSV with columns episode_index,target_bucket,scene_complexity.",
    )
    parser.add_argument("--close-user-bias-prob", type=float, default=0.5)
    parser.add_argument("--inject-latency-ms", type=float, default=0.0)
    parser.add_argument("--inject-latency-every", type=int, default=0)
    parser.add_argument(
        "--skill-schedule",
        choices=[
            "default",
            "safe_balanced",
            "safe_stationary",
            "gesture_heavy",
            "turn_heavy",
            "walk_probe",
            "fall_probe",
            "gesture_probe",
            "gesture_user_close_probe",
            "turn_collision_probe",
            "walk_collision_probe",
            "passive_delayed_fall_probe",
            "safe_balanced_phase4",
            "safe_active_phase4",
            "latency_balanced",
        ],
        default="default",
        help="Phase3 collection schedule. default preserves Phase2 behavior.",
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assisted-review", action="store_true", default=True)
    parser.add_argument("--reviewer", default="codex_assisted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scene_plan_rows = _read_scene_plan(args.scene_plan_csv)
    run_id = args.run_id or time.strftime("phase2_oracle_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("outputs") / "casa" / "phase2_oracle" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    oracle = SafetyOracle.from_thresholds_file(args.thresholds)
    attempts_path = output_dir / "attempts.csv"
    _append_attempt(
        attempts_path,
        {
            "run_id": run_id,
            "episode_id": "",
            "episode_index": "",
            "status": "run_started",
            "scenario": "",
            "skill_schedule": args.skill_schedule,
            "start_wall_time": time.time(),
            "end_wall_time": "",
            "duration_seconds": "",
            "error": "",
        },
    )
    for episode_index in range(args.rollout_count):
        episode_id = f"episode_{episode_index:06d}"
        episode_dir = output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        attempt_start = time.time()
        scenario = ""
        try:
            ack = _request_props(args, run_id, episode_id, episode_index)
            scenario = str(ack.get("scenario") if ack else "")
            if ack:
                (episode_dir / "scene_props_ack.json").write_text(json.dumps(ack, indent=2, sort_keys=True) + "\n")

            fallback_start_wall = _float(ack.get("wall_time"), time.time()) if ack else time.time()
            skill_events_csv, episode_start_wall = _run_episode(args, rng, run_id, episode_id, episode_dir, episode_index)
            episode_start_wall = episode_start_wall or fallback_start_wall
            episode_end_wall = time.time()

            sim_state_csv = _slice_sim_log(
                source=args.sim_log_dir / "sim_state.csv",
                destination=episode_dir / "sim_log" / "sim_state.csv",
                start_wall_time=episode_start_wall,
                end_wall_time=episode_end_wall + args.post_horizon + 0.25,
            )
            scene_props = {}
            if ack and ack.get("scene_props"):
                scene_props = ack["scene_props"]
                scene_props_path = episode_dir / "sim_log" / "scene_props.json"
                scene_props_path.parent.mkdir(parents=True, exist_ok=True)
                scene_props_path.write_text(json.dumps(scene_props, indent=2, sort_keys=True) + "\n")

            violations = oracle.detect(sim_state_csv, skill_events_csv)
            labels = oracle.label_skill_calls(
                violations,
                sim_state_csv,
                skill_events_csv,
                post_horizon=args.post_horizon,
            )
            summary = build_rollout_summary(
                run_id=run_id,
                rollout_id=episode_id,
                violations=violations,
                skill_labels=labels,
                scene_props=scene_props,
                extra={
                    "sim_state_csv": str(sim_state_csv),
                    "skill_events_csv": str(skill_events_csv),
                    "post_horizon": args.post_horizon,
                    "episode_start_wall_time": episode_start_wall,
                    "episode_end_wall_time": episode_end_wall,
                    "scenario": scenario or None,
                    "skill_schedule": args.skill_schedule,
                    "event_start_time_s": _event_start_time_s(violations, episode_start_wall),
                    **_sim_aggregates(sim_state_csv),
                },
            )
            RolloutLogger(episode_dir).write(violations, summary)
            _append_attempt(
                attempts_path,
                {
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "status": "completed",
                    "scenario": scenario,
                    "skill_schedule": args.skill_schedule,
                    "start_wall_time": attempt_start,
                    "end_wall_time": time.time(),
                    "duration_seconds": time.time() - attempt_start,
                    "error": "",
                },
            )
            print(
                json.dumps(
                    {
                        "episode": episode_id,
                        "violation_types": summary["violation_types"],
                        "skill_label_counts": summary["skill_label_counts"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            _append_attempt(
                attempts_path,
                {
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "status": "failed",
                    "scenario": scenario,
                    "skill_schedule": args.skill_schedule,
                    "start_wall_time": attempt_start,
                    "end_wall_time": time.time(),
                    "duration_seconds": time.time() - attempt_start,
                    "error": repr(exc),
                },
            )
            print(
                json.dumps({"episode": episode_id, "status": "failed", "error": repr(exc)}, sort_keys=True),
                flush=True,
            )
            continue

    review_dir = output_dir / "review"
    collect_cmd = [
        sys.executable,
        str(REPO_ROOT / "gear_sonic/scripts/casa_collect_oracle_eval_set.py"),
        "--run-id",
        run_id,
        "--episodes-root",
        str(output_dir),
        "--output-dir",
        str(review_dir),
    ]
    if args.assisted_review:
        collect_cmd.extend(["--assisted-review", "--reviewer", args.reviewer])
    subprocess.run(collect_cmd, check=True)
    print(json.dumps({"run_id": run_id, "output_dir": str(output_dir), "review_dir": str(review_dir)}, indent=2))


def _request_props(args: argparse.Namespace, run_id: str, episode_id: str, episode_index: int) -> dict:
    if args.casa_props_command_file is None:
        return {}
    command_path = _resolve_path(args.casa_props_command_file)
    ack_path = _resolve_path(args.casa_props_ack_file) if args.casa_props_ack_file else command_path.with_suffix(
        command_path.suffix + ".ack"
    )
    command_id = f"{run_id}:{episode_id}:{time.time_ns()}"
    command = {
        "command_id": command_id,
        "episode_id": episode_id,
        "reset": True,
        "randomize": args.scenario_mode == "random",
        "seed": args.seed + episode_index,
        "close_user_bias_prob": args.close_user_bias_prob,
    }
    if args.phase2_v2:
        command.update(_phase2_v2_scene_command(args, episode_index))
    elif args.scenario_mode == "balanced":
        scenario = _episode_scenario(episode_index)
        command["scenario"] = scenario
        command["placements"] = _scenario_placements(scenario)
        command["force_base_height"] = _force_base_height_for_scenario(scenario)
    elif args.scenario_mode == "cycle":
        scenario = _cycle_scenario(args.scenario_cycle, episode_index)
        command["scenario"] = scenario
        command["placements"] = _scenario_placements(scenario)
        command["force_base_height"] = _force_base_height_for_scenario(scenario)
    else:
        scenario = "random"
        command["scenario"] = scenario
    command_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = command_path.with_suffix(command_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(command, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(command_path)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if ack_path.exists():
            try:
                ack = json.loads(ack_path.read_text())
            except json.JSONDecodeError:
                time.sleep(0.05)
                continue
            if ack.get("command_id") == command_id:
                return ack
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for CASA props ack: {ack_path}")


def _run_episode(
    args: argparse.Namespace,
    rng: random.Random,
    run_id: str,
    episode_id: str,
    episode_dir: Path,
    episode_index: int,
) -> tuple[Path, float | None]:
    executor_args = SimpleNamespace(
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        publish_fps=args.publish_fps,
        dry_run=args.dry_run,
    )
    executor = create_executor(executor_args, episode_dir, run_id, args.sim_log_dir)
    latency_ms = args.inject_latency_ms if _inject_latency_this_episode(args, episode_index) else 0.0
    skill_idx = 0
    start = time.monotonic()
    rollout_start_wall = None
    try:
        executor.start_policy()
        if args.pre_rollout_settle_seconds > 0:
            time.sleep(args.pre_rollout_settle_seconds)
        rollout_start_wall = time.time()
        for skill in _episode_skills(args, rng):
            if time.monotonic() - start >= args.duration_seconds:
                break
            skill_idx += 1
            executor.execute_one(skill, episode_id=episode_id, skill_idx=skill_idx)
            if latency_ms > 0:
                time.sleep(latency_ms / 1000.0)
    finally:
        executor.close()
    if latency_ms > 0:
        _annotate_injected_latency(episode_dir / "skill_events.csv", latency_ms)
    return episode_dir / "skill_events.csv", rollout_start_wall


def _episode_skills(args: argparse.Namespace, rng: random.Random):
    elapsed = 0.0
    while elapsed < args.duration_seconds:
        sequence = _schedule_sequence(args, rng)
        for skill in sequence:
            elapsed += skill.estimated_duration
            yield skill
            if elapsed >= args.duration_seconds:
                break


def _schedule_sequence(args: argparse.Namespace, rng: random.Random) -> list:
    if args.skill_schedule == "walk_probe":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=0.0, duration=_duration(args, rng), speed=-1.0),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
        ]
    if args.skill_schedule == "fall_probe":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=0.0, duration=_duration(args, rng), speed=-1.0),
            PassiveSkill(duration=max(0.4, args.stabilize_seconds * 0.5), mode="wait"),
            WalkSkill(
                vx=0.9,
                vy=rng.choice([-0.35, 0.35]),
                facing_yaw_deg=rng.choice([-18.0, 18.0]),
                duration=_duration(args, rng),
                speed=-1.0,
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
        ]
    if args.skill_schedule == "gesture_probe":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.65, 0.8, 0.95]),
                frequency=rng.choice([0.75, 1.0, 1.25]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
        ]
    if args.skill_schedule == "gesture_user_close_probe":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.8, 0.95, 1.1]),
                frequency=rng.choice([0.75, 1.0, 1.25]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=max(0.5, args.stabilize_seconds * 0.5), mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.9, 1.05, 1.15]),
                frequency=rng.choice([1.0, 1.25]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
        ]
    if args.skill_schedule == "turn_collision_probe":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-110.0, -90.0, -75.0, 75.0, 90.0, 110.0]), duration=_duration(args, rng)),
            PassiveSkill(duration=max(0.5, args.stabilize_seconds * 0.5), mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-120.0, -100.0, 100.0, 120.0]), duration=_duration(args, rng)),
        ]
    if args.skill_schedule == "walk_collision_probe":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(
                vx=1.0,
                vy=rng.choice([-0.08, 0.0, 0.08]),
                facing_yaw_deg=rng.choice([-8.0, 0.0, 8.0]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=max(0.5, args.stabilize_seconds * 0.5), mode="wait"),
            WalkSkill(
                vx=1.0,
                vy=rng.choice([-0.15, 0.0, 0.15]),
                facing_yaw_deg=rng.choice([-15.0, 0.0, 15.0]),
                duration=_duration(args, rng),
            ),
        ]
    if args.skill_schedule == "passive_delayed_fall_probe":
        return [
            PassiveSkill(duration=max(args.stabilize_seconds, 4.0), mode="wait"),
            PassiveSkill(duration=max(args.stabilize_seconds, 3.0), mode="wait"),
        ]
    if args.skill_schedule == "latency_balanced":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(vx=0.8, vy=0.0, facing_yaw_deg=0.0, duration=_duration(args, rng), speed=0.4),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-45.0, 45.0]), duration=_duration(args, rng)),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(amplitude=0.4, frequency=0.75, side=rng.choice(["left", "right", "both"]), duration=_duration(args, rng)),
        ]
    if args.skill_schedule == "safe_stationary":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-15.0, 0.0, 15.0]), duration=_duration(args, rng)),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.15, 0.2, 0.25]),
                frequency=rng.choice([0.5, 0.75]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-20.0, 0.0, 20.0]), duration=_duration(args, rng)),
        ]
    if args.skill_schedule == "safe_balanced":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(
                vx=0.8,
                vy=0.0,
                facing_yaw_deg=rng.choice([-10.0, 0.0, 10.0]),
                duration=_duration(args, rng),
                speed=0.4,
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-30.0, 0.0, 30.0]), duration=_duration(args, rng)),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.2, 0.3, 0.4]),
                frequency=rng.choice([0.5, 0.75]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(
                vx=0.8,
                vy=0.0,
                facing_yaw_deg=rng.choice([-10.0, 0.0, 10.0]),
                duration=_duration(args, rng),
                speed=0.4,
            ),
        ]
    if args.skill_schedule == "safe_balanced_phase4":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(
                vx=rng.choice([0.5, 0.65, 0.8]),
                vy=rng.choice([-0.10, 0.0, 0.10]),
                facing_yaw_deg=rng.choice([-10.0, 0.0, 10.0]),
                duration=_duration(args, rng),
                speed=0.4,
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-30.0, -15.0, 0.0, 15.0, 30.0]), duration=_duration(args, rng)),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.15, 0.25, 0.35]),
                frequency=rng.choice([0.5, 0.75]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
        ]
    if args.skill_schedule == "safe_active_phase4":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(
                vx=rng.choice([0.5, 0.65, 0.8]),
                vy=rng.choice([-0.08, 0.0, 0.08]),
                facing_yaw_deg=rng.choice([-8.0, 0.0, 8.0]),
                duration=_duration(args, rng),
                speed=0.4,
            ),
            TurnSkill(face_yaw_deg=rng.choice([-30.0, -15.0, 0.0, 15.0, 30.0]), duration=_duration(args, rng)),
            GestureSkill(
                amplitude=rng.choice([0.15, 0.25, 0.35]),
                frequency=rng.choice([0.5, 0.75]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            WalkSkill(
                vx=rng.choice([0.5, 0.65, 0.8]),
                vy=rng.choice([-0.08, 0.0, 0.08]),
                facing_yaw_deg=rng.choice([-8.0, 0.0, 8.0]),
                duration=_duration(args, rng),
                speed=0.4,
            ),
            TurnSkill(face_yaw_deg=rng.choice([-30.0, -15.0, 0.0, 15.0, 30.0]), duration=_duration(args, rng)),
            GestureSkill(
                amplitude=rng.choice([0.15, 0.25, 0.35]),
                frequency=rng.choice([0.5, 0.75]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
        ]
    if args.skill_schedule == "gesture_heavy":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.4, 0.6, 0.8]),
                frequency=rng.choice([0.75, 1.0, 1.25]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            GestureSkill(
                amplitude=rng.choice([0.5, 0.7, 0.9]),
                frequency=rng.choice([0.75, 1.0]),
                side=rng.choice(["left", "right", "both"]),
                duration=_duration(args, rng),
            ),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(vx=0.8, vy=0.0, facing_yaw_deg=0.0, duration=_duration(args, rng), speed=0.4),
        ]
    if args.skill_schedule == "turn_heavy":
        return [
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-75.0, -45.0, -30.0, 30.0, 45.0, 75.0]), duration=_duration(args, rng)),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            TurnSkill(face_yaw_deg=rng.choice([-90.0, -60.0, 60.0, 90.0]), duration=_duration(args, rng)),
            PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
            WalkSkill(vx=0.8, vy=0.0, facing_yaw_deg=rng.choice([-15.0, 0.0, 15.0]), duration=_duration(args, rng), speed=0.4),
        ]
    return [
        PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
        WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=rng.choice([-15.0, 0.0, 15.0]), duration=_duration(args, rng)),
        PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
        TurnSkill(face_yaw_deg=rng.choice([-45.0, -30.0, 0.0, 30.0, 45.0]), duration=_duration(args, rng)),
        PassiveSkill(duration=args.stabilize_seconds, mode="wait"),
        GestureSkill(
            amplitude=rng.choice([0.3, 0.5, 0.7]),
            frequency=rng.choice([0.5, 0.75, 1.0]),
            side=rng.choice(["left", "right", "both"]),
            duration=_duration(args, rng),
        ),
        WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=rng.choice([-15.0, 0.0, 15.0]), duration=_duration(args, rng)),
    ]


def _episode_scenario(episode_index: int) -> str:
    cycle = [
        "safe",
        "human_close",
        "user_collision",
        "near_obstacle",
        "obstacle_collision",
        "fall",
    ]
    return cycle[episode_index % len(cycle)]


def _cycle_scenario(scenario_cycle: str, episode_index: int) -> str:
    cycle = [item.strip() for item in scenario_cycle.split(",") if item.strip()]
    if not cycle:
        raise ValueError("--scenario-cycle is required when --scenario-mode cycle")
    return cycle[episode_index % len(cycle)]


def _phase2_v2_scene_command(args: argparse.Namespace, episode_index: int) -> dict:
    plan_row = args.scene_plan_rows.get(episode_index, {})
    target_bucket = plan_row.get("target_bucket") or _cycle_value(
        args.target_bucket_cycle,
        episode_index,
        default="clean_safe",
    )
    scene_complexity = plan_row.get("scene_complexity") or _cycle_value(
        args.scene_complexity_cycle,
        episode_index,
        default="medium",
    )
    return make_phase2_v2_scene_command(
        target_bucket=target_bucket,
        scene_complexity=scene_complexity,
        seed=args.seed,
        episode_index=episode_index,
    )


def _cycle_value(raw_cycle: str, episode_index: int, *, default: str) -> str:
    cycle = [item.strip() for item in raw_cycle.split(",") if item.strip()]
    if not cycle:
        return default
    return cycle[episode_index % len(cycle)]


def _read_scene_plan(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    rows: dict[int, dict[str, str]] = {}
    with _resolve_path(path).open(newline="") as file:
        for row in csv.DictReader(file):
            rows[int(row["episode_index"])] = row
    return rows


def _scenario_placements(scenario: str) -> dict[str, dict]:
    placements = {
        "user_proxy_0": {"enabled": True, "position": [2.6, 1.4, 0.0], "yaw_deg": 0.0},
    }
    for index in range(8):
        placements[f"obstacle_{index}"] = {"enabled": False, "position": [0.0, 0.0, -10.0], "yaw_deg": 0.0}

    if scenario == "safe_empty":
        placements["user_proxy_0"] = {"enabled": False, "position": [0.0, 0.0, -10.0], "yaw_deg": 0.0}
    elif scenario == "safe_near_obstacle":
        placements["user_proxy_0"] = {"enabled": False, "position": [0.0, 0.0, -10.0], "yaw_deg": 0.0}
        placements["obstacle_0"] = {"enabled": True, "position": [1.05, 0.85, 0.0], "yaw_deg": 0.0}
    elif scenario == "safe_near_user":
        placements["user_proxy_0"] = {"enabled": True, "position": [1.15, 0.85, 0.0], "yaw_deg": 0.0}
    elif scenario == "future_obstacle_center":
        placements["user_proxy_0"] = {"enabled": False, "position": [0.0, 0.0, -10.0], "yaw_deg": 0.0}
        placements["obstacle_0"] = {"enabled": True, "position": [1.05, 0.0, 0.0], "yaw_deg": 0.0}
    elif scenario == "future_obstacle_offset":
        placements["user_proxy_0"] = {"enabled": False, "position": [0.0, 0.0, -10.0], "yaw_deg": 0.0}
        placements["obstacle_0"] = {"enabled": True, "position": [1.05, 0.18, 0.0], "yaw_deg": 0.0}
    elif scenario == "future_user_center":
        placements["user_proxy_0"] = {"enabled": True, "position": [1.25, 0.0, 0.0], "yaw_deg": 0.0}
    elif scenario == "future_gesture_user":
        placements["user_proxy_0"] = {"enabled": True, "position": [0.75, 0.36, 0.0], "yaw_deg": 0.0}
    elif scenario == "safe_gesture_user":
        placements["user_proxy_0"] = {"enabled": True, "position": [0.85, 0.85, 0.0], "yaw_deg": 0.0}
    elif scenario == "human_close":
        placements["user_proxy_0"] = {"enabled": True, "position": [0.45, 0.0, 0.0], "yaw_deg": 0.0}
    elif scenario == "user_collision":
        placements["user_proxy_0"] = {"enabled": True, "position": [0.45, 0.0, 0.0], "yaw_deg": 0.0}
    elif scenario == "near_obstacle":
        placements["obstacle_0"] = {"enabled": True, "position": [0.55, 0.35, 0.0], "yaw_deg": 0.0}
    elif scenario == "obstacle_collision":
        placements["obstacle_0"] = {"enabled": True, "position": [0.05, 0.0, 0.0], "yaw_deg": 0.0}
    elif scenario == "obstacle_collision_fall":
        placements["user_proxy_0"] = {"enabled": False, "position": [0.0, 0.0, -10.0], "yaw_deg": 0.0}
        placements["obstacle_0"] = {"enabled": True, "position": [0.28, 0.0, 0.0], "yaw_deg": 0.0}
    elif scenario == "fall_forward_obstacle":
        placements["user_proxy_0"] = {"enabled": False, "position": [0.0, 0.0, -10.0], "yaw_deg": 0.0}
        placements["obstacle_0"] = {"enabled": True, "position": [0.42, 0.0, 0.0], "yaw_deg": 0.0}
    elif scenario == "fall":
        pass
    elif scenario != "safe":
        raise ValueError(f"Unknown balanced Phase2 scenario: {scenario}")
    return placements


def _force_base_height_for_scenario(scenario: str) -> float | None:
    if scenario in {"user_collision", "fall", "obstacle_collision_fall", "fall_forward_obstacle"}:
        return 0.25
    return None


def _append_attempt(path: Path, row: dict) -> None:
    fieldnames = [
        "run_id",
        "episode_id",
        "episode_index",
        "status",
        "scenario",
        "skill_schedule",
        "start_wall_time",
        "end_wall_time",
        "duration_seconds",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _duration(args: argparse.Namespace, rng: random.Random) -> float:
    return rng.uniform(args.min_skill_duration, args.max_skill_duration)


def _inject_latency_this_episode(args: argparse.Namespace, episode_index: int) -> bool:
    return args.inject_latency_ms > 0 and args.inject_latency_every > 0 and episode_index % args.inject_latency_every == 0


def _slice_sim_log(source: Path, destination: Path, start_wall_time: float, end_wall_time: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if start_wall_time <= _float(row.get("wall_time"), -float("inf")) <= end_wall_time
        ]
    with destination.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _annotate_injected_latency(skill_events_csv: Path, latency_ms: float) -> None:
    rows = list(csv.DictReader(skill_events_csv.open(newline="")))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        evidence = _json_dict(row.get("evidence_json"))
        evidence["injected_latency_ms"] = latency_ms
        row["evidence_json"] = json.dumps(evidence, sort_keys=True)
    with skill_events_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sim_aggregates(sim_state_csv: Path) -> dict:
    rows = list(csv.DictReader(sim_state_csv.open(newline="")))
    event_rows = [
        row
        for row in rows
        if _sum_row_flags(row, ["external_collision_user", "external_collision_obstacle", "fall_flag"]) > 0
    ]
    first_row = rows[0] if rows else {}
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


def _event_start_time_s(violations, episode_start_wall: float) -> float | None:
    if not violations:
        return None
    first = min(violations, key=lambda violation: violation.start_wall_time)
    return max(0.0, float(first.start_wall_time) - float(episode_start_wall))


def _json_dict(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _float(value, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
