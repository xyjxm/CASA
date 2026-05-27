"""Run the CASA Phase 3 counterfactual skill subset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.io.episode_log_reader import SkillEvaluationConfig
from gear_sonic.casa.io.zmq_publisher import ZMQPlannerPublisher
from gear_sonic.casa.loggers.rollout_logger import RolloutLogger
from gear_sonic.casa.oracle import SafetyOracle
from gear_sonic.casa.oracle.rollout_summary import build_rollout_summary
from gear_sonic.casa.runner_utils import build_evaluation_config
from gear_sonic.casa.skills import GestureSkill, SkillExecutor, WalkSkill
from gear_sonic.casa.loggers.skill_logger import SkillExecutionLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-log-dir", type=Path, default=None)
    parser.add_argument("--casa-props-command-file", type=Path, default=None)
    parser.add_argument("--casa-props-ack-file", type=Path, default=None)
    parser.add_argument("--zmq-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--num-states", type=int, default=100)
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-stride", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--candidate-set", choices=["version_a"], default="version_a")
    parser.add_argument("--post-horizon", type=float, default=2.0)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = _select_states(
        args.dataset_dir,
        args.num_states,
        args.seed,
        state_offset=args.state_offset,
        state_stride=args.state_stride,
    )
    candidates = _candidate_skills(args.candidate_set)
    branch_rows: list[dict[str, Any]] = []

    if args.dry_run:
        branch_rows = _planned_rows(states, candidates, args.repeats)
    else:
        _require_live_args(args)
        branch_rows = _run_live_counterfactual(args, states, candidates)

    _write_csv(args.output_dir / "counterfactual_branches.csv", branch_rows)
    summary = _counterfactual_summary(branch_rows, requested_states=args.num_states, repeats=args.repeats)
    (args.output_dir / "counterfactual_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _select_states(
    dataset_dir: Path,
    num_states: int,
    seed: int,
    *,
    state_offset: int = 0,
    state_stride: int = 1,
) -> list[dict[str, Any]]:
    if state_offset < 0:
        raise ValueError("--state-offset must be non-negative")
    if state_stride <= 0:
        raise ValueError("--state-stride must be positive")
    rows = list(csv.DictReader((dataset_dir / "invocations.csv").open(newline="")))
    preferred = [row for row in rows if row.get("safe_label") == "safe"]
    pool = preferred or rows
    rng = random.Random(seed)
    rng.shuffle(pool)
    candidates = []
    seen_keys = set()
    sim_cache: dict[str, list[dict[str, Any]]] = {}
    needed = state_offset + num_states * state_stride
    for row in pool:
        key = (row.get("sim_state_csv"), row.get("pre_state_wall_time"))
        if key in seen_keys:
            continue
        snapshot = _snapshot_from_row(row, sim_cache)
        if snapshot is None:
            continue
        sim_row = snapshot["sim_row"]
        if _float(sim_row.get("fall_flag"), 0.0) > 0.5:
            continue
        if _float(sim_row.get("base_pos_z"), 1.0) < 0.45:
            continue
        candidate = dict(row)
        candidate["state_id"] = f"source_state_{len(candidates):04d}"
        candidate["snapshot"] = snapshot
        candidates.append(candidate)
        seen_keys.add(key)
        if len(candidates) >= needed:
            break
    selected = candidates[state_offset::state_stride][:num_states]
    for index, row in enumerate(selected):
        row["state_id"] = f"state_{index:04d}_o{state_offset}_s{state_stride}"
    return selected


def _snapshot_from_row(row: dict[str, Any], sim_cache: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    sim_path = Path(row.get("sim_state_csv", ""))
    if not sim_path.exists():
        return None
    if str(sim_path) not in sim_cache:
        sim_cache[str(sim_path)] = list(csv.DictReader(sim_path.open(newline="")))
    sim_rows = sim_cache[str(sim_path)]
    target_wall = _float(row.get("pre_state_wall_time"), None)
    if target_wall is None or not sim_rows:
        return None
    sim_row = min(sim_rows, key=lambda item: abs(_float(item.get("wall_time"), 0.0) - target_wall))
    qpos = _state_vector(sim_row, "qpos")
    qvel = _state_vector(sim_row, "qvel")
    if qpos is None or qvel is None:
        return None
    summary = _read_json(Path(row.get("summary_path", "")))
    scene_props = summary.get("scene_props", {}) if isinstance(summary, dict) else {}
    return {
        "qpos": qpos,
        "qvel": qvel,
        "sim_time": _float(sim_row.get("sim_time"), 0.0),
        "wall_time": _float(sim_row.get("wall_time"), 0.0),
        "sim_row": sim_row,
        "scene_props": scene_props,
    }


def _state_vector(sim_row: dict[str, Any], vector_name: str) -> list[float] | None:
    if vector_name == "qpos":
        base_fields = ["base_pos_x", "base_pos_y", "base_pos_z", "base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z"]
        joint_prefix = "body_q_"
    else:
        base_fields = ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z", "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
        joint_prefix = "body_dq_"
    if any(field not in sim_row for field in base_fields):
        return None
    joints = sorted([field for field in sim_row if field.startswith(joint_prefix)], key=_joint_sort_key)
    if not joints:
        return None
    return [_float(sim_row.get(field), 0.0) for field in base_fields + joints]


def _candidate_skills(candidate_set: str) -> dict[str, Any]:
    if candidate_set != "version_a":
        raise ValueError(f"Unknown candidate set: {candidate_set}")
    return {
        "walk_slow": WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=0.0, duration=3.0, speed=0.4),
        "walk_fast": WalkSkill(vx=1.0, vy=0.0, facing_yaw_deg=0.0, duration=3.0, speed=1.0),
        "gesture_small": GestureSkill(amplitude=0.3, frequency=0.75, side="both", duration=4.0),
        "gesture_large": GestureSkill(amplitude=0.7, frequency=0.75, side="both", duration=4.0),
    }


def _planned_rows(states: list[dict[str, Any]], candidates: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    rows = []
    for state in states:
        for candidate_name, skill in candidates.items():
            for repeat in range(repeats):
                rows.append(
                    {
                        "branch_id": _branch_id(state["state_id"], candidate_name, repeat),
                        "state_id": state["state_id"],
                        "candidate": candidate_name,
                        "repeat": repeat,
                        "skill_name": skill.name,
                        "params_json": json.dumps(skill.params(), sort_keys=True),
                        "status": "planned",
                        "safe_label": "",
                        "triggered_violation_types": "[]",
                    }
                )
    return rows


def _run_live_counterfactual(args: argparse.Namespace, states: list[dict[str, Any]], candidates: dict[str, Any]) -> list[dict[str, Any]]:
    oracle = SafetyOracle.from_thresholds_file(args.thresholds)
    publisher = ZMQPlannerPublisher(args.zmq_host, args.zmq_port, dry_run=False)
    rows = []
    try:
        for state in states:
            snapshot = state["snapshot"]
            for candidate_name, skill in candidates.items():
                for repeat in range(args.repeats):
                    branch_id = _branch_id(state["state_id"], candidate_name, repeat)
                    branch_dir = args.output_dir / "branches" / branch_id
                    branch_dir.mkdir(parents=True, exist_ok=True)
                    sim_source = args.sim_log_dir / "sim_state.csv"
                    start_offset = sim_source.stat().st_size if sim_source.exists() else 0
                    ack = _restore_state(args, branch_id, snapshot)
                    episode_start_wall = _float(ack.get("wall_time"), time.time())
                    logger = SkillExecutionLogger(branch_dir)
                    executor = SkillExecutor(
                        publisher=publisher,
                        logger=logger,
                        run_id="phase3_counterfactual",
                        publish_fps=args.publish_fps,
                        sim_log_dir=None,
                        evaluation_config=build_evaluation_config(SimpleNamespace()),
                        dry_run=False,
                    )
                    try:
                        executor.start_policy()
                        result = executor.execute_one(skill, episode_id=branch_id, skill_idx=1)
                    finally:
                        logger.close()
                    sim_state_csv = _slice_sim_log(
                        source=sim_source,
                        destination=branch_dir / "sim_log" / "sim_state.csv",
                        start_wall_time=episode_start_wall,
                        end_wall_time=result.end_wall_time + args.post_horizon + 0.25,
                        start_offset=start_offset,
                    )
                    scene_props = snapshot.get("scene_props", {})
                    if scene_props:
                        scene_props_path = branch_dir / "sim_log" / "scene_props.json"
                        scene_props_path.parent.mkdir(parents=True, exist_ok=True)
                        scene_props_path.write_text(json.dumps(scene_props, indent=2, sort_keys=True) + "\n")
                    violations = oracle.detect(sim_state_csv, branch_dir / "skill_events.csv")
                    labels = oracle.label_skill_calls(
                        violations,
                        sim_state_csv,
                        branch_dir / "skill_events.csv",
                        post_horizon=args.post_horizon,
                    )
                    summary = build_rollout_summary(
                        run_id="phase3_counterfactual",
                        rollout_id=branch_id,
                        violations=violations,
                        skill_labels=labels,
                        scene_props=scene_props,
                        extra={
                            "counterfactual_state_id": state["state_id"],
                            "candidate": candidate_name,
                            "repeat": repeat,
                            "restore_ack": ack,
                            "sim_state_csv": str(sim_state_csv),
                            "skill_events_csv": str(branch_dir / "skill_events.csv"),
                        },
                    )
                    RolloutLogger(branch_dir).write(violations, summary)
                    label = labels[0] if labels else {}
                    rows.append(
                        {
                            "branch_id": branch_id,
                            "state_id": state["state_id"],
                            "candidate": candidate_name,
                            "repeat": repeat,
                            "skill_name": skill.name,
                            "params_json": json.dumps(skill.params(), sort_keys=True),
                            "status": result.status,
                            "safe_label": label.get("safe_label", ""),
                            "triggered_violation_types": json.dumps(label.get("triggered_violation_types", []), sort_keys=True),
                            "rollout_summary": str(branch_dir / "rollout_summary.json"),
                        }
                    )
    finally:
        publisher.close()
    return rows


def _restore_state(args: argparse.Namespace, branch_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    command_path = _resolve(args.casa_props_command_file)
    ack_path = _resolve(args.casa_props_ack_file) if args.casa_props_ack_file else command_path.with_suffix(command_path.suffix + ".ack")
    command_id = f"phase3_counterfactual:{branch_id}:{time.time_ns()}"
    scene_props = snapshot.get("scene_props", {})
    command = {
        "command_id": command_id,
        "episode_id": branch_id,
        "reset": True,
        "randomize": False,
        "placements": scene_props.get("placements", {}),
        "restore_state": {
            "qpos": snapshot["qpos"],
            "qvel": snapshot["qvel"],
            "sim_time": snapshot.get("sim_time", 0.0),
        },
    }
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
    raise TimeoutError(f"Timed out waiting for restore ack: {ack_path}")


def _slice_sim_log(
    source: Path,
    destination: Path,
    start_wall_time: float,
    end_wall_time: float,
    start_offset: int | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        if start_offset is not None and start_offset > 0:
            src.seek(start_offset)
            reader = csv.DictReader(src, fieldnames=fieldnames)
        rows = [row for row in reader if start_wall_time <= _float(row.get("wall_time"), -float("inf")) <= end_wall_time]
    with destination.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _counterfactual_summary(rows: list[dict[str, Any]], *, requested_states: int, repeats: int) -> dict[str, Any]:
    completed_rows = [row for row in rows if row.get("safe_label") in {"safe", "unsafe"}]
    grouped = defaultdict(list)
    for row in completed_rows:
        grouped[(row.get("state_id"), row.get("candidate"))].append(row.get("safe_label"))
    consistencies = []
    for labels in grouped.values():
        counts = Counter(labels)
        consistencies.append(max(counts.values()) / len(labels))
    return {
        "requested_states": requested_states,
        "observed_states": len({row.get("state_id") for row in rows}),
        "requested_repeats": repeats,
        "branch_count": len(rows),
        "completed_branch_count": len(completed_rows),
        "label_counts": dict(Counter(row.get("safe_label", "") for row in rows)),
        "candidate_counts": dict(Counter(row.get("candidate", "") for row in rows)),
        "mean_repeat_consistency": sum(consistencies) / len(consistencies) if consistencies else 0.0,
        "min_repeat_consistency": min(consistencies) if consistencies else 0.0,
        "repeat_consistency_groups_below_0p90": sum(1 for value in consistencies if value < 0.9),
        "repeat_consistency_ge_0p90": (sum(consistencies) / len(consistencies) >= 0.9) if consistencies else False,
        "complete_100x4x5": len(completed_rows) == requested_states * 4 * repeats,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_live_args(args: argparse.Namespace) -> None:
    missing = []
    if args.sim_log_dir is None:
        missing.append("--sim-log-dir")
    if args.casa_props_command_file is None:
        missing.append("--casa-props-command-file")
    if missing:
        raise SystemExit(f"Live counterfactual requires {' and '.join(missing)}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _branch_id(state_id: str, candidate: str, repeat: int) -> str:
    return f"{state_id}__{candidate}__r{repeat:02d}"


def _resolve(path: Path | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    return path if path.is_absolute() else REPO_ROOT / path


def _joint_sort_key(field: str) -> tuple[str, int]:
    prefix, _, value = field.rpartition("_")
    try:
        return prefix, int(value)
    except ValueError:
        return prefix, -1


def _float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
