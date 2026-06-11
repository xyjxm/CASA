"""Training-time maze utilities for Gymnasium-Robotics scale2 maps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from .actions import VLNAction


Cell = tuple[int, int]


def _cell_to_xy(cell: Cell, *, rows: int, cols: int, scale: float) -> tuple[float, float]:
    row, col = cell
    x = (col - (cols / 2.0 - 0.5)) * scale
    y = ((rows / 2.0 - 0.5) - row) * scale
    return (float(x), float(y))


def _parse_furniture_obstacle(
    item: dict,
    *,
    idx: int,
    rows: int,
    cols: int,
    scale: float,
) -> "FurnitureObstacle":
    raw_cell = item["cell"]
    cell = (int(raw_cell[0]), int(raw_cell[1]))
    raw_xy = item.get("xy")
    xy = (float(raw_xy[0]), float(raw_xy[1])) if raw_xy is not None else _cell_to_xy(cell, rows=rows, cols=cols, scale=scale)
    raw_size = item.get("size", [0.55, 0.55, 0.45])
    raw_rgba = item.get("rgba", [0.2, 0.35, 0.85, 1.0])
    return FurnitureObstacle(
        name=str(item.get("name", f"furniture_{idx}")),
        kind=str(item.get("kind", "obstacle")),
        cell=cell,
        xy=xy,
        size=(float(raw_size[0]), float(raw_size[1]), float(raw_size[2])),
        rgba=(float(raw_rgba[0]), float(raw_rgba[1]), float(raw_rgba[2]), float(raw_rgba[3])),
    )


@dataclass(frozen=True)
class MazeRoute:
    start_cell: Cell
    goal_cell: Cell
    path_cells: list[Cell]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    stop_xy: tuple[float, float]
    start_yaw_deg: float
    route_actions: list[VLNAction]
    instruction: str
    visual_cues: list[str]


@dataclass(frozen=True)
class FurnitureObstacle:
    name: str
    kind: str
    cell: Cell
    xy: tuple[float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class MazeMap:
    name: str
    occupancy: np.ndarray
    scale: float
    metadata: dict
    furniture_obstacles: tuple[FurnitureObstacle, ...] = ()

    @classmethod
    def from_metadata(cls, path: Path) -> "MazeMap":
        data = json.loads(Path(path).read_text())
        maze_map = data["maze_map"]
        occupancy = np.array([[1 if value == 1 else 0 for value in row] for row in maze_map], dtype=np.uint8)
        rows, cols = occupancy.shape
        scale = float(data["maze_size_scaling"])
        furniture_obstacles = tuple(
            _parse_furniture_obstacle(item, idx=idx, rows=rows, cols=cols, scale=scale)
            for idx, item in enumerate(data.get("furniture_obstacles", []))
        )
        return cls(
            name=data["name"],
            occupancy=occupancy,
            scale=scale,
            metadata=data,
            furniture_obstacles=furniture_obstacles,
        )

    @property
    def rows(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def cols(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def free_cells(self) -> list[Cell]:
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.is_free((row, col))
        ]

    def cell_to_xy(self, cell: Cell) -> tuple[float, float]:
        return _cell_to_xy(cell, rows=self.rows, cols=self.cols, scale=self.scale)

    def xy_to_cell(self, x: float, y: float) -> Cell:
        col = int(round(x / self.scale + (self.cols / 2.0 - 0.5)))
        row = int(round((self.rows / 2.0 - 0.5) - y / self.scale))
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))
        return (row, col)

    def is_free(self, cell: Cell) -> bool:
        row, col = cell
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and int(self.occupancy[row, col]) == 0
            and cell not in self.furniture_cells()
        )

    def is_official_free(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols and int(self.occupancy[row, col]) == 0

    def neighbors(self, cell: Cell) -> list[Cell]:
        row, col = cell
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        return [candidate for candidate in candidates if self.is_free(candidate)]

    def official_neighbors(self, cell: Cell) -> list[Cell]:
        row, col = cell
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        return [candidate for candidate in candidates if self.is_official_free(candidate)]

    def wall_cells(self) -> list[Cell]:
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if int(self.occupancy[row, col]) == 1
        ]

    def furniture_cells(self) -> set[Cell]:
        return {obstacle.cell for obstacle in self.furniture_obstacles}

    def blocking_cells(self) -> list[Cell]:
        return [*self.wall_cells(), *sorted(self.furniture_cells())]

    def is_xy_safe(self, x: float, y: float, *, robot_radius: float = 0.28) -> bool:
        half = self.scale / 2.0
        outer_x = self.cols * self.scale / 2.0
        outer_y = self.rows * self.scale / 2.0
        if not (-outer_x + robot_radius <= x <= outer_x - robot_radius):
            return False
        if not (-outer_y + robot_radius <= y <= outer_y - robot_radius):
            return False
        for blocking_cell in self.blocking_cells():
            wx, wy = self.cell_to_xy(blocking_cell)
            dx = max(abs(x - wx) - half, 0.0)
            dy = max(abs(y - wy) - half, 0.0)
            if math.hypot(dx, dy) < robot_radius:
                return False
            if abs(x - wx) <= half + robot_radius and abs(y - wy) <= half + robot_radius:
                # Expanded AABB check catches side contacts against axis-aligned walls.
                return False
        return self.is_free(self.xy_to_cell(x, y))

    def astar(self, start: Cell, goal: Cell) -> list[Cell]:
        if not self.is_free(start) or not self.is_free(goal):
            raise ValueError(f"A* endpoints must be free cells: start={start}, goal={goal}")
        return self._astar_with_neighbors(start, goal, neighbor_fn=self.neighbors)

    def official_astar_without_furniture(self, start: Cell, goal: Cell) -> list[Cell]:
        if not self.is_official_free(start) or not self.is_official_free(goal):
            raise ValueError(f"official A* endpoints must be free cells: start={start}, goal={goal}")
        return self._astar_with_neighbors(start, goal, neighbor_fn=self.official_neighbors)

    def route_requires_furniture_detour(self, start: Cell, goal: Cell, furnished_path: list[Cell] | None = None) -> bool:
        if not self.furniture_obstacles:
            return False
        official_path = self.official_astar_without_furniture(start, goal)
        blocked_by_furniture = bool(set(official_path).intersection(self.furniture_cells()))
        if blocked_by_furniture:
            return True
        furnished_path = furnished_path or self.astar(start, goal)
        return len(furnished_path) > len(official_path)

    def _astar_with_neighbors(self, start: Cell, goal: Cell, *, neighbor_fn) -> list[Cell]:
        frontier: list[tuple[float, Cell]] = [(0.0, start)]
        came_from: dict[Cell, Cell | None] = {start: None}
        cost_so_far: dict[Cell, float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for neighbor in neighbor_fn(current):
                new_cost = cost_so_far[current] + 1.0
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            raise ValueError(f"no path from {start} to {goal}")
        path = []
        current: Cell | None = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def shortest_path_distance_xy(self, xy: tuple[float, float], goal_xy: tuple[float, float]) -> float:
        start_cell = self.xy_to_cell(*xy)
        goal_cell = self.xy_to_cell(*goal_xy)
        if not self.is_free(start_cell):
            return float("inf")
        if start_cell == goal_cell:
            return math.hypot(goal_xy[0] - xy[0], goal_xy[1] - xy[1])
        path = self.astar(start_cell, goal_cell)
        start_center = self.cell_to_xy(start_cell)
        goal_center = self.cell_to_xy(goal_cell)
        return (
            math.hypot(start_center[0] - xy[0], start_center[1] - xy[1])
            + (len(path) - 1) * self.scale
            + math.hypot(goal_xy[0] - goal_center[0], goal_xy[1] - goal_center[1])
        )

    def connected_free_pairs(self, *, min_edges: int = 2, max_edges: int | None = None) -> Iterable[tuple[Cell, Cell, list[Cell]]]:
        free = self.free_cells
        for start in free:
            for goal in free:
                if start == goal:
                    continue
                path = self.astar(start, goal)
                edges = len(path) - 1
                if edges >= min_edges and (max_edges is None or edges <= max_edges):
                    yield start, goal, path

    def euclidean_distance_cells(self, start: Cell, goal: Cell) -> float:
        start_xy = self.cell_to_xy(start)
        goal_xy = self.cell_to_xy(goal)
        return math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])


def heading_for_delta(delta: tuple[int, int]) -> float:
    drow, dcol = delta
    if (drow, dcol) == (0, 1):
        return 0.0
    if (drow, dcol) == (-1, 0):
        return 90.0
    if (drow, dcol) == (0, -1):
        return 180.0
    if (drow, dcol) == (1, 0):
        return -90.0
    raise ValueError(f"non-cardinal delta: {delta}")


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def turn_actions(current_yaw: float, target_yaw: float, *, turn_degrees: float = 30.0) -> tuple[list[VLNAction], float]:
    error = wrap_degrees(target_yaw - current_yaw)
    if abs(error) < 1e-6:
        return [], target_yaw
    action = VLNAction.TURN_LEFT if error > 0.0 else VLNAction.TURN_RIGHT
    count = int(round(abs(error) / turn_degrees))
    return [action] * count, target_yaw


def route_actions_from_path(
    path_cells: list[Cell],
    *,
    scale: float,
    forward_step_m: float = 0.50,
    turn_degrees: float = 30.0,
) -> tuple[list[VLNAction], float, tuple[float, float]]:
    if len(path_cells) < 2:
        raise ValueError("route path must contain at least two cells")
    first_delta = (path_cells[1][0] - path_cells[0][0], path_cells[1][1] - path_cells[0][1])
    yaw = heading_for_delta(first_delta)
    actions: list[VLNAction] = []
    steps_per_cell = max(1, int(round(scale / forward_step_m)))

    for edge_idx, (cell_a, cell_b) in enumerate(zip(path_cells[:-1], path_cells[1:])):
        delta = (cell_b[0] - cell_a[0], cell_b[1] - cell_a[1])
        target_yaw = heading_for_delta(delta)
        turns, yaw = turn_actions(yaw, target_yaw, turn_degrees=turn_degrees)
        actions.extend(turns)
        forward_count = steps_per_cell - 1 if edge_idx == len(path_cells) - 2 else steps_per_cell
        actions.extend([VLNAction.FORWARD] * max(1, forward_count))

    goal_delta = (path_cells[-1][0] - path_cells[-2][0], path_cells[-1][1] - path_cells[-2][1])
    approach_yaw = heading_for_delta(goal_delta)
    stop_offset = forward_step_m
    dx = math.cos(math.radians(approach_yaw)) * stop_offset
    dy = math.sin(math.radians(approach_yaw)) * stop_offset
    return actions, yaw, (-dx, -dy)


def _world_side_for_relative_cell(path_cell: Cell, obstacle_cell: Cell) -> str:
    drow = path_cell[0] - obstacle_cell[0]
    dcol = path_cell[1] - obstacle_cell[1]
    if abs(dcol) >= abs(drow):
        return "east/right" if dcol > 0 else "west/left"
    return "south/lower" if drow > 0 else "north/upper"


def _furniture_phrase(maze: MazeMap, path_cells: list[Cell]) -> tuple[list[str], list[str]]:
    phrases: list[str] = []
    visual_cues: list[str] = []
    path_set = set(path_cells)
    color_by_kind = {
        "table": "blue",
        "chair": "green",
    }
    for obstacle in maze.furniture_obstacles:
        color = color_by_kind.get(obstacle.kind, "colored")
        visual_cues.append(f"{color} {obstacle.kind}")
        adjacent = [
            cell
            for cell in path_set
            if abs(cell[0] - obstacle.cell[0]) + abs(cell[1] - obstacle.cell[1]) == 1
        ]
        if adjacent:
            side = _world_side_for_relative_cell(adjacent[0], obstacle.cell)
            phrases.append(f"pass the {color} {obstacle.kind} on its {side} side")
        else:
            phrases.append(f"keep clear of the {color} {obstacle.kind} if it appears")
    return phrases, visual_cues


def route_instruction(maze: MazeMap, path_cells: list[Cell], route_actions: list[VLNAction]) -> tuple[str, list[str]]:
    if not route_actions:
        return "Stop when the gallery wall painting is close and centered.", ["gallery wall painting"]
    segments: list[tuple[VLNAction, int]] = []
    current = route_actions[0]
    count = 0
    for action in route_actions:
        if action == current:
            count += 1
            continue
        segments.append((current, count))
        current = action
        count = 1
    segments.append((current, count))

    maneuver_phrases = []
    visual_cues = ["corridor openings", "wall bends", "gallery wall painting"]
    for action, _count in segments:
        if action is VLNAction.FORWARD:
            if not maneuver_phrases or maneuver_phrases[-1] != "continue through the open corridor":
                maneuver_phrases.append("continue through the open corridor")
        elif action is VLNAction.TURN_LEFT:
            maneuver_phrases.append("turn left when the corridor bends")
        elif action is VLNAction.TURN_RIGHT:
            maneuver_phrases.append("turn right when the corridor bends")
        elif action is VLNAction.BACKOFF:
            maneuver_phrases.append("back away briefly if the wall fills the view")
    # Keep language natural: no explicit step counts or hidden route indices.
    furniture_phrases, furniture_cues = _furniture_phrase(maze, path_cells)
    visual_cues.extend(furniture_cues)
    route_cues = [*furniture_phrases, *maneuver_phrases[:6]]
    route_text = "; then ".join(route_cues)
    if route_text:
        route_text = route_text[0].upper() + route_text[1:]
    instruction = (
        "Navigate using only the first-person camera. "
        f"{route_text}. Move smoothly through visible corridor openings, avoid the furniture, "
        "and stop when the gallery wall painting is close in front of you."
    )
    return instruction, visual_cues


def build_route(
    maze: MazeMap,
    start_cell: Cell,
    goal_cell: Cell,
    *,
    forward_step_m: float = 0.50,
    turn_degrees: float = 30.0,
) -> MazeRoute:
    path = maze.astar(start_cell, goal_cell)
    actions, final_yaw, stop_offset = route_actions_from_path(
        path,
        scale=maze.scale,
        forward_step_m=forward_step_m,
        turn_degrees=turn_degrees,
    )
    first_delta = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    start_yaw = heading_for_delta(first_delta)
    goal_xy = maze.cell_to_xy(goal_cell)
    stop_xy = (goal_xy[0] + stop_offset[0], goal_xy[1] + stop_offset[1])
    del final_yaw
    instruction, visual_cues = route_instruction(maze, path, actions)
    return MazeRoute(
        start_cell=start_cell,
        goal_cell=goal_cell,
        path_cells=path,
        start_xy=maze.cell_to_xy(start_cell),
        goal_xy=goal_xy,
        stop_xy=stop_xy,
        start_yaw_deg=start_yaw,
        route_actions=actions,
        instruction=instruction,
        visual_cues=visual_cues,
    )


def generate_routes(
    maze: MazeMap,
    *,
    count: int,
    seed: int,
    min_edges: int = 2,
    max_edges: int = 7,
    min_euclidean_m: float = 0.0,
    require_furniture_detour: bool = False,
    exclude_pairs: set[tuple[Cell, Cell]] | None = None,
) -> list[MazeRoute]:
    rng = random.Random(seed)
    candidates = list(maze.connected_free_pairs(min_edges=min_edges, max_edges=max_edges))
    if min_euclidean_m > 0.0:
        candidates = [
            item
            for item in candidates
            if maze.euclidean_distance_cells(item[0], item[1]) >= min_euclidean_m
        ]
    if require_furniture_detour:
        candidates = [
            item
            for item in candidates
            if maze.route_requires_furniture_detour(item[0], item[1], furnished_path=item[2])
        ]
    if exclude_pairs:
        candidates = [item for item in candidates if (item[0], item[1]) not in exclude_pairs]
    rng.shuffle(candidates)
    routes = []
    seen: set[tuple[Cell, Cell]] = set()
    for start, goal, _ in candidates:
        if (start, goal) in seen:
            continue
        routes.append(build_route(maze, start, goal))
        seen.add((start, goal))
        if len(routes) >= count:
            break
    if len(routes) < count:
        raise ValueError(f"only generated {len(routes)} routes, requested {count}")
    return routes


def reachable_cells(maze: MazeMap, start: Cell) -> set[Cell]:
    queue: deque[Cell] = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        for neighbor in maze.neighbors(current):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen
