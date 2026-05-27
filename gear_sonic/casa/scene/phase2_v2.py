"""Phase 2 v2 scene generation for clean visual rollout collection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


HIDDEN_POSITION = [0.0, 0.0, -10.0]
USER_PREFIX = "user_proxy_"
OBSTACLE_PREFIX = "obstacle_"
USER_SLOTS = 3
OBSTACLE_SLOTS = 8


@dataclass(frozen=True)
class ComplexitySpec:
    users: tuple[int, int]
    obstacles: tuple[int, int]


COMPLEXITY_SPECS = {
    "simple": ComplexitySpec(users=(0, 1), obstacles=(0, 1)),
    "medium": ComplexitySpec(users=(1, 1), obstacles=(2, 4)),
    "hard": ComplexitySpec(users=(2, 3), obstacles=(4, 8)),
}

BUCKET_FAMILIES = {
    "clean_safe": ["safe_empty", "safe_near_user", "safe_near_obstacle", "safe_gesture_user"],
    "visual_collision_or_close": [
        "late_user_close",
        "late_obstacle_collision",
        "narrow_gap_collision",
    ],
    "visual_near_boundary": ["near_boundary_user", "near_boundary_obstacle", "narrow_gap_near"],
    "visual_fall": ["post_collision_fall", "narrow_gap_trip", "cluttered_fall"],
    "phase4_gesture_user_close": ["gesture_user_close"],
    "phase4_turn_collision": ["turn_side_user_close", "turn_side_obstacle_close", "turn_narrow_obstacle"],
    "phase4_passive_delayed_fall": ["passive_delayed_fall"],
    "phase4_safe_balanced": ["safe_empty", "safe_near_user", "safe_near_obstacle", "safe_gesture_user"],
    "phase4_safe_empty": ["safe_empty"],
    "phase4_walk_collision": ["late_obstacle_collision", "narrow_gap_collision", "post_collision_fall"],
}


def make_phase2_v2_scene_command(
    *,
    target_bucket: str,
    scene_complexity: str,
    seed: int,
    episode_index: int = 0,
) -> dict[str, Any]:
    """Return command fields for ``run_sim_loop.py`` CASA prop placement."""
    if target_bucket not in BUCKET_FAMILIES:
        raise ValueError(f"unknown target bucket: {target_bucket}")
    if scene_complexity not in COMPLEXITY_SPECS:
        raise ValueError(f"unknown scene complexity: {scene_complexity}")

    rng = random.Random(seed * 1000003 + episode_index)
    families = BUCKET_FAMILIES[target_bucket]
    scene_family = families[episode_index % len(families)]
    spec = COMPLEXITY_SPECS[scene_complexity]
    num_users = rng.randint(*spec.users)
    num_obstacles = rng.randint(*spec.obstacles)

    if scene_family in {
        "safe_near_user",
        "safe_gesture_user",
        "late_user_close",
        "near_boundary_user",
        "gesture_user_close",
        "turn_side_user_close",
    }:
        num_users = max(1, num_users)
    if scene_family in {
        "safe_near_obstacle",
        "late_obstacle_collision",
        "narrow_gap_collision",
        "near_boundary_obstacle",
        "narrow_gap_near",
        "post_collision_fall",
        "narrow_gap_trip",
        "cluttered_fall",
        "turn_side_obstacle_close",
        "turn_narrow_obstacle",
        "passive_delayed_fall",
    }:
        num_obstacles = max(1, num_obstacles)

    placements = hidden_placements()
    users: list[tuple[float, float]] = []
    obstacles: list[tuple[float, float]] = []

    def add_user(x: float, y: float) -> None:
        if len(users) < num_users:
            users.append((x, y))

    def add_obstacle(x: float, y: float) -> None:
        if len(obstacles) < num_obstacles:
            obstacles.append((x, y))

    if scene_family == "safe_empty":
        if scene_complexity == "simple" or target_bucket == "phase4_safe_empty":
            num_users = 0
            num_obstacles = 0
    elif scene_family == "safe_near_user":
        add_user(rng.uniform(1.50, 2.00), rng.choice([-1.0, 1.0]) * rng.uniform(1.25, 1.55))
    elif scene_family == "safe_near_obstacle":
        add_obstacle(rng.uniform(1.25, 1.75), rng.choice([-1.0, 1.0]) * rng.uniform(0.95, 1.20))
    elif scene_family == "safe_gesture_user":
        add_user(rng.uniform(1.45, 1.90), rng.choice([-1.0, 1.0]) * rng.uniform(1.25, 1.55))
    elif scene_family == "late_user_close":
        add_user(rng.uniform(1.45, 1.95), rng.choice([-1.0, 1.0]) * rng.uniform(0.82, 1.02))
    elif scene_family == "late_obstacle_collision":
        add_obstacle(rng.uniform(1.45, 1.85), rng.uniform(-0.08, 0.08))
    elif scene_family == "narrow_gap_collision":
        x = rng.uniform(1.45, 1.85)
        add_obstacle(x, rng.uniform(0.18, 0.28))
        add_obstacle(x + rng.uniform(0.12, 0.38), -rng.uniform(0.18, 0.28))
    elif scene_family == "near_boundary_user":
        add_user(rng.uniform(1.45, 1.90), rng.choice([-1.0, 1.0]) * rng.uniform(0.96, 1.16))
    elif scene_family == "near_boundary_obstacle":
        add_obstacle(rng.uniform(1.40, 1.80), rng.choice([-1.0, 1.0]) * rng.uniform(0.38, 0.52))
    elif scene_family == "narrow_gap_near":
        x = rng.uniform(1.45, 1.85)
        add_obstacle(x, rng.uniform(0.35, 0.46))
        add_obstacle(x + rng.uniform(0.15, 0.40), -rng.uniform(0.35, 0.46))
    elif scene_family == "post_collision_fall":
        add_obstacle(rng.uniform(1.05, 1.22), rng.uniform(-0.04, 0.04))
        add_obstacle(rng.uniform(1.26, 1.48), rng.choice([-1.0, 1.0]) * rng.uniform(0.12, 0.24))
    elif scene_family == "narrow_gap_trip":
        x = rng.uniform(1.05, 1.22)
        add_obstacle(x, rng.uniform(0.10, 0.18))
        add_obstacle(x + rng.uniform(0.12, 0.24), -rng.uniform(0.10, 0.18))
        add_obstacle(x + rng.uniform(0.28, 0.48), rng.uniform(-0.05, 0.05))
    elif scene_family == "cluttered_fall":
        side = rng.choice([-1.0, 1.0])
        add_obstacle(rng.uniform(1.05, 1.22), rng.uniform(-0.06, 0.06))
        add_obstacle(rng.uniform(1.22, 1.48), side * rng.uniform(0.10, 0.22))
        add_obstacle(rng.uniform(1.36, 1.68), -side * rng.uniform(0.10, 0.24))
    elif scene_family == "gesture_user_close":
        side = rng.choice([-1.0, 1.0])
        add_user(rng.uniform(0.72, 0.92), side * rng.uniform(0.52, 0.70))
    elif scene_family == "turn_side_user_close":
        side = rng.choice([-1.0, 1.0])
        add_user(rng.uniform(0.82, 1.05), side * rng.uniform(0.62, 0.82))
    elif scene_family == "turn_side_obstacle_close":
        side = rng.choice([-1.0, 1.0])
        add_obstacle(rng.uniform(0.74, 0.98), side * rng.uniform(0.34, 0.48))
    elif scene_family == "turn_narrow_obstacle":
        x = rng.uniform(0.88, 1.16)
        add_obstacle(x, rng.uniform(0.28, 0.42))
        add_obstacle(x + rng.uniform(0.12, 0.28), -rng.uniform(0.28, 0.42))
    elif scene_family == "passive_delayed_fall":
        add_obstacle(rng.uniform(1.10, 1.40), rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 0.70))
    else:
        raise ValueError(f"unknown scene family: {scene_family}")

    avoid_safe_path = target_bucket in {"clean_safe", "phase4_safe_balanced", "phase4_safe_empty"}
    fill_users(users, num_users, rng, scene_complexity, avoid_walk_path=avoid_safe_path)
    fill_obstacles(
        obstacles,
        num_obstacles,
        rng,
        scene_complexity,
        users,
        avoid_walk_path=avoid_safe_path,
    )

    for index, (x, y) in enumerate(users[:USER_SLOTS]):
        placements[f"{USER_PREFIX}{index}"] = placement(x, y, rng)
    for index, (x, y) in enumerate(obstacles[:OBSTACLE_SLOTS]):
        placements[f"{OBSTACLE_PREFIX}{index}"] = placement(x, y, rng)

    command = {
        "randomize": False,
        "placements": placements,
        "scenario": scene_family,
        "target_bucket": target_bucket,
        "scene_complexity": scene_complexity,
        "scene_family": scene_family,
        "num_users": min(num_users, USER_SLOTS),
        "num_obstacles": min(num_obstacles, OBSTACLE_SLOTS),
        "generator_version": "phase2_v2_clean_natural_20260519",
    }
    if target_bucket in {"visual_fall", "phase4_passive_delayed_fall"}:
        command["fall_trigger"] = (
            "passive_delayed_velocity_perturbation"
            if target_bucket == "phase4_passive_delayed_fall"
            else "obstacle_collision_plus_rollout_velocity_perturbation"
        )
        command["velocity_perturbations"] = []
        perturb_times = (
            [rng.uniform(3.2, 3.8), rng.uniform(4.4, 5.0)]
            if target_bucket == "phase4_passive_delayed_fall"
            else [rng.uniform(4.0, 4.8), rng.uniform(5.4, 6.2)]
        )
        for time_s in perturb_times:
            command["velocity_perturbations"].append(
                {
                    "time_s": round(time_s, 3),
                    "velocity_body": [
                        round(rng.uniform(2.6, 3.8), 3),
                        round(rng.choice([-1.0, 1.0]) * rng.uniform(1.2, 2.0), 3),
                        0.0,
                    ],
                    "angular_velocity_body": [
                        round(rng.choice([-1.0, 1.0]) * rng.uniform(2.6, 4.2), 3),
                        round(rng.choice([-1.0, 1.0]) * rng.uniform(1.8, 3.2), 3),
                        0.0,
                    ],
                }
            )
    return command


def hidden_placements() -> dict[str, dict[str, Any]]:
    placements: dict[str, dict[str, Any]] = {}
    for index in range(USER_SLOTS):
        placements[f"{USER_PREFIX}{index}"] = hidden_placement()
    for index in range(OBSTACLE_SLOTS):
        placements[f"{OBSTACLE_PREFIX}{index}"] = hidden_placement()
    return placements


def hidden_placement() -> dict[str, Any]:
    return {"enabled": False, "position": list(HIDDEN_POSITION), "yaw_deg": 0.0}


def placement(x: float, y: float, rng: random.Random) -> dict[str, Any]:
    return {
        "enabled": True,
        "position": [round(float(x), 4), round(float(y), 4), 0.0],
        "yaw_deg": round(rng.uniform(-180.0, 180.0), 3),
    }


def fill_users(
    users: list[tuple[float, float]],
    target_count: int,
    rng: random.Random,
    scene_complexity: str,
    *,
    avoid_walk_path: bool = False,
) -> None:
    while len(users) < min(target_count, USER_SLOTS):
        if avoid_walk_path:
            candidate = (rng.uniform(1.45, 2.9), rng.choice([-1.0, 1.0]) * rng.uniform(1.30, 1.75))
        elif scene_complexity == "hard":
            candidate = (rng.uniform(1.35, 2.8), rng.choice([-1.0, 1.0]) * rng.uniform(1.05, 1.60))
        else:
            candidate = (rng.uniform(1.45, 2.6), rng.choice([-1.0, 1.0]) * rng.uniform(1.00, 1.45))
        if far_from_existing(candidate, users, 0.70) and initial_clearance_ok(candidate, "user"):
            users.append(candidate)


def fill_obstacles(
    obstacles: list[tuple[float, float]],
    target_count: int,
    rng: random.Random,
    scene_complexity: str,
    users: list[tuple[float, float]],
    *,
    avoid_walk_path: bool = False,
) -> None:
    attempts = 0
    while len(obstacles) < min(target_count, OBSTACLE_SLOTS) and attempts < 500:
        attempts += 1
        if avoid_walk_path:
            x = rng.uniform(1.25, 2.8)
            y = rng.choice([-1.0, 1.0]) * rng.uniform(0.85, 1.55)
        elif scene_complexity == "hard":
            x = rng.uniform(1.20, 2.8)
            y = rng.uniform(-1.35, 1.35)
        elif scene_complexity == "medium":
            x = rng.uniform(1.25, 2.5)
            y = rng.uniform(-1.15, 1.15)
        else:
            x = rng.uniform(1.25, 2.2)
            y = rng.uniform(-0.95, 0.95)
        candidate = (x, y)
        if not initial_clearance_ok(candidate, "obstacle"):
            continue
        if not far_from_existing(candidate, obstacles, 0.48):
            continue
        if not far_from_existing(candidate, users, 0.62):
            continue
        obstacles.append(candidate)


def initial_clearance_ok(candidate: tuple[float, float], prop_type: str) -> bool:
    distance = math.hypot(candidate[0], candidate[1])
    if prop_type == "user":
        return distance >= 1.55 and abs(candidate[1]) >= 0.80
    return distance >= 0.95


def far_from_existing(
    candidate: tuple[float, float],
    existing: list[tuple[float, float]],
    min_distance: float,
) -> bool:
    return all(math.hypot(candidate[0] - x, candidate[1] - y) >= min_distance for x, y in existing)


def count_enabled(placements: dict[str, dict[str, Any]], prefix: str) -> int:
    return sum(1 for name, value in placements.items() if name.startswith(prefix) and value.get("enabled", True))
