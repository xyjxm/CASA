"""Generate Phase 2 v2 scenes without policy/video and validate initial placement."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.scene import ScenePropsManager, load_scene_props_config
from gear_sonic.casa.scene.phase2_v2 import (
    OBSTACLE_PREFIX,
    USER_PREFIX,
    count_enabled,
    make_phase2_v2_scene_command,
)
from gear_sonic.utils.mujoco_sim.sim_utils import get_body_geom_ids, get_subtree_body_names


DEFAULT_MODEL_XML = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_casa_v1.xml"
DEFAULT_PROPS_CONFIG = REPO_ROOT / "gear_sonic/casa/scene/props.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/casa/phase2_clean_visual_1000_v2_20260519/placement_test"

BUCKET_CYCLE = [
    "clean_safe",
    "visual_collision_or_close",
    "visual_near_boundary",
    "visual_fall",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL_XML)
    parser.add_argument("--props-config", type=Path, default=DEFAULT_PROPS_CONFIG)
    parser.add_argument("--per-complexity", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--user-clearance", type=float, default=0.35)
    parser.add_argument("--obstacle-clearance", type=float, default=0.25)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.model_xml.resolve()))
    data = mujoco.MjData(model)
    props_config = load_scene_props_config(args.props_config.resolve())
    props_manager = ScenePropsManager(model, data, props_config)
    geom_sets = geom_sets_for_model(model)

    rows: list[dict[str, Any]] = []
    for complexity in ["simple", "medium", "hard"]:
        for index in range(args.per_complexity):
            bucket = BUCKET_CYCLE[index % len(BUCKET_CYCLE)]
            episode_index = len(rows)
            command = make_phase2_v2_scene_command(
                target_bucket=bucket,
                scene_complexity=complexity,
                seed=args.seed,
                episode_index=episode_index,
            )
            mujoco.mj_resetData(model, data)
            props_manager.apply_placements(command["placements"])
            metrics = placement_metrics(model, data, geom_sets)
            num_users = count_enabled(command["placements"], USER_PREFIX)
            num_obstacles = count_enabled(command["placements"], OBSTACLE_PREFIX)
            errors = []
            if (metrics["initial_min_user_distance"] or 99.0) < args.user_clearance:
                errors.append("user_clearance")
            if (metrics["initial_min_obstacle_distance"] or 99.0) < args.obstacle_clearance:
                errors.append("obstacle_clearance")
            if metrics["initial_external_collision_user"]:
                errors.append("user_collision")
            if metrics["initial_external_collision_obstacle"]:
                errors.append("obstacle_collision")
            if complexity == "hard" and not (2 <= num_users <= 3):
                errors.append("hard_user_count")
            if complexity == "hard" and not (4 <= num_obstacles <= 8):
                errors.append("hard_obstacle_count")
            rows.append(
                {
                    "episode_index": episode_index,
                    "target_bucket": bucket,
                    "scene_complexity": complexity,
                    "scene_family": command["scene_family"],
                    "num_users": num_users,
                    "num_obstacles": num_obstacles,
                    **metrics,
                    "ok": int(not errors),
                    "errors": "|".join(errors),
                }
            )

    csv_path = args.output_root / "placement_test.csv"
    write_csv(csv_path, rows)
    summary = {
        "total": len(rows),
        "ok": sum(int(row["ok"]) for row in rows),
        "error_counts": dict(Counter(error for row in rows for error in str(row["errors"]).split("|") if error)),
        "complexity_counts": dict(Counter(row["scene_complexity"] for row in rows)),
        "hard_user_count_values": dict(
            Counter(str(row["num_users"]) for row in rows if row["scene_complexity"] == "hard")
        ),
        "hard_obstacle_count_values": dict(
            Counter(str(row["num_obstacles"]) for row in rows if row["scene_complexity"] == "hard")
        ),
        "csv": str(csv_path),
    }
    summary_path = args.output_root / "placement_test.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["ok"] != summary["total"]:
        raise SystemExit("placement test failed")


def geom_sets_for_model(model: mujoco.MjModel) -> dict[str, set[int]]:
    root_body_id = model.body("pelvis").id
    robot_bodies = get_subtree_body_names(model, root_body_id)
    robot_geom_ids: set[int] = set()
    for body_name in robot_bodies:
        robot_geom_ids.update(get_body_geom_ids(model, model.body(body_name).id))
    user_geom_ids = geom_ids_by_prefix(model, USER_PREFIX)
    obstacle_geom_ids = geom_ids_by_prefix(model, OBSTACLE_PREFIX)
    return {
        "robot": robot_geom_ids - user_geom_ids - obstacle_geom_ids,
        "user": user_geom_ids,
        "obstacle": obstacle_geom_ids,
    }


def geom_ids_by_prefix(model: mujoco.MjModel, prefix: str) -> set[int]:
    geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name
        body_name = model.body(model.geom(geom_id).bodyid).name
        if geom_name.startswith(prefix) or body_name.startswith(prefix):
            geom_ids.add(geom_id)
    return geom_ids


def placement_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_sets: dict[str, set[int]],
) -> dict[str, Any]:
    mujoco.mj_forward(model, data)
    return {
        "initial_min_user_distance": min_geom_distance(model, data, geom_sets["robot"], geom_sets["user"]),
        "initial_min_obstacle_distance": min_geom_distance(model, data, geom_sets["robot"], geom_sets["obstacle"]),
        "initial_external_collision_user": int(external_collision(model, data, geom_sets["robot"], geom_sets["user"])),
        "initial_external_collision_obstacle": int(
            external_collision(model, data, geom_sets["robot"], geom_sets["obstacle"])
        ),
    }


def min_geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_geom_ids: set[int],
    target_geom_ids: set[int],
) -> float | None:
    if not source_geom_ids or not target_geom_ids:
        return None
    min_distance = float("inf")
    for source_id in source_geom_ids:
        source_pos = data.geom_xpos[source_id]
        source_radius = float(model.geom_rbound[source_id])
        for target_id in target_geom_ids:
            target_pos = data.geom_xpos[target_id]
            target_radius = float(model.geom_rbound[target_id])
            center_distance = float(np.linalg.norm(source_pos - target_pos))
            min_distance = min(min_distance, max(0.0, center_distance - source_radius - target_radius))
    return None if min_distance == float("inf") else min_distance


def external_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: set[int],
    external_geom_ids: set[int],
) -> bool:
    del model
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = contact.geom1
        geom2 = contact.geom2
        if (geom1 in robot_geom_ids and geom2 in external_geom_ids) or (
            geom2 in robot_geom_ids and geom1 in external_geom_ids
        ):
            return True
    return False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

