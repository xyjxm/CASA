"""Build a furnished Gymnasium-Robotics MuJoCo maze for SONIC VLN runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.metrics import write_json
from gear_sonic.vln.no_casa_mujoco import draw_topdown_map


BASE_ROOT = Path("/mnt/data/students/lph/recording/gymnasium_robotics_maze_maps_scale2_20260610_233102")
BASE_METADATA = BASE_ROOT / "data/MEDIUM_MAZE_DIVERSE_GR_scale2.json"
BASE_XML = BASE_ROOT / "mjcf/sonic_scene_MEDIUM_MAZE_DIVERSE_GR_scale2_official_gymnasium.xml"
OUTPUT_ROOT = Path("/mnt/data/students/lph/recording")


FURNITURE_PRESETS = {
    "default": [
        {
            "name": "table_center_bend",
            "kind": "table",
            "cell": [3, 3],
            "size": [0.58, 0.58, 0.42],
            "rgba": [0.18, 0.36, 0.78, 1.0],
            "reason": "blocks a central bend used by many official shortest paths; routes must detour around it",
        },
        {
            "name": "chair_east_goal_bend",
            "kind": "chair",
            "cell": [6, 5],
            "size": [0.42, 0.42, 0.38],
            "rgba": [0.10, 0.55, 0.38, 1.0],
            "reason": "adds a second furniture blocker near the east lower bend without disconnecting the official topology",
        },
    ],
    "train_gallery": [
        {
            "name": "table_center_bend_train",
            "kind": "table",
            "cell": [3, 3],
            "size": [0.58, 0.58, 0.42],
            "rgba": [0.18, 0.36, 0.78, 1.0],
            "reason": "training distribution central table detour",
        },
        {
            "name": "chair_east_bend_train",
            "kind": "chair",
            "cell": [6, 5],
            "size": [0.42, 0.42, 0.38],
            "rgba": [0.10, 0.55, 0.38, 1.0],
            "reason": "training distribution lower bend chair",
        },
        {
            "name": "table_north_gallery_train",
            "kind": "table",
            "cell": [1, 5],
            "size": [0.50, 0.50, 0.40],
            "rgba": [0.58, 0.26, 0.70, 1.0],
            "reason": "extra training furniture so held-out is not the same furnished distribution",
        },
    ],
    "test_gallery": [
        {
            "name": "table_north_bend_test",
            "kind": "table",
            "cell": [1, 2],
            "size": [0.54, 0.54, 0.42],
            "rgba": [0.20, 0.38, 0.80, 1.0],
            "reason": "held-out distribution shifts the central obstacle north-west",
        },
        {
            "name": "chair_south_gallery_test",
            "kind": "chair",
            "cell": [1, 6],
            "size": [0.42, 0.42, 0.38],
            "rgba": [0.10, 0.55, 0.38, 1.0],
            "reason": "held-out distribution places the chair on a different lower corridor",
        },
        {
            "name": "table_lower_west_test",
            "kind": "table",
            "cell": [4, 2],
            "size": [0.50, 0.50, 0.40],
            "rgba": [0.62, 0.30, 0.72, 1.0],
            "reason": "extra held-out furniture not seen in the training preset",
        },
    ],
}

DEFAULT_FURNITURE = FURNITURE_PRESETS["default"]

WALL_PAINTING_GEOMS = [
    {
        "name": "gallery_wall_painting_panel_x",
        "pos": (0.0, 0.0, 0.94),
        "size": (0.035, 0.50, 0.34),
        "rgba": (0.05, 0.82, 0.95, 1.0),
    },
    {
        "name": "gallery_wall_painting_panel_y",
        "pos": (0.0, 0.0, 0.94),
        "size": (0.50, 0.035, 0.34),
        "rgba": (0.10, 0.65, 0.22, 1.0),
    },
]


def _default_run_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT / f"vln_furnished_maze_obstacle_nav_{stamp}"


def _add_geom(
    parent: ET.Element,
    *,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, ...],
    geom_type: str,
    rgba: tuple[float, float, float, float],
    contype: int = 1,
    conaffinity: int = 1,
) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "pos": " ".join(f"{value:.4f}" for value in pos),
            "size": " ".join(f"{value:.4f}" for value in size),
            "type": geom_type,
            "contype": str(contype),
            "conaffinity": str(conaffinity),
            "rgba": " ".join(f"{value:.4f}" for value in rgba),
        },
    )


def _add_table(worldbody: ET.Element, obstacle: dict) -> None:
    x, y = obstacle["xy"]
    sx, sy, sz = obstacle["size"]
    rgba = tuple(obstacle["rgba"])
    _add_geom(
        worldbody,
        name=f"{obstacle['name']}_top",
        pos=(x, y, 0.58),
        size=(sx, sy, 0.06),
        geom_type="box",
        rgba=rgba,
    )
    leg_rgba = (0.12, 0.12, 0.12, 1.0)
    for leg_idx, (dx, dy) in enumerate([(-sx * 0.72, -sy * 0.72), (-sx * 0.72, sy * 0.72), (sx * 0.72, -sy * 0.72), (sx * 0.72, sy * 0.72)]):
        _add_geom(
            worldbody,
            name=f"{obstacle['name']}_leg_{leg_idx}",
            pos=(x + dx, y + dy, 0.30),
            size=(0.045, 0.28),
            geom_type="cylinder",
            rgba=leg_rgba,
        )


def _add_chair(worldbody: ET.Element, obstacle: dict) -> None:
    x, y = obstacle["xy"]
    sx, sy, _ = obstacle["size"]
    rgba = tuple(obstacle["rgba"])
    _add_geom(
        worldbody,
        name=f"{obstacle['name']}_seat",
        pos=(x, y, 0.42),
        size=(sx, sy, 0.05),
        geom_type="box",
        rgba=rgba,
    )
    _add_geom(
        worldbody,
        name=f"{obstacle['name']}_back",
        pos=(x, y - sy * 0.85, 0.72),
        size=(sx, 0.045, 0.32),
        geom_type="box",
        rgba=rgba,
    )
    leg_rgba = (0.10, 0.10, 0.10, 1.0)
    for leg_idx, (dx, dy) in enumerate([(-sx * 0.70, -sy * 0.70), (-sx * 0.70, sy * 0.70), (sx * 0.70, -sy * 0.70), (sx * 0.70, sy * 0.70)]):
        _add_geom(
            worldbody,
            name=f"{obstacle['name']}_leg_{leg_idx}",
            pos=(x + dx, y + dy, 0.22),
            size=(0.035, 0.22),
            geom_type="cylinder",
            rgba=leg_rgba,
        )


def _add_wall_painting_target(worldbody: ET.Element, insert_index: int) -> None:
    for offset, item in enumerate(WALL_PAINTING_GEOMS):
        _add_geom(
            worldbody,
            name=item["name"],
            pos=item["pos"],
            size=item["size"],
            geom_type="box",
            rgba=item["rgba"],
            contype=0,
            conaffinity=0,
        )
        geom = worldbody[-1]
        worldbody.remove(geom)
        worldbody.insert(insert_index + offset, geom)


def build_furnished_maze(
    output_dir: Path,
    *,
    base_metadata: Path,
    base_xml: Path,
    furniture_preset: str = "default",
    overwrite: bool = False,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")
    data_dir = output_dir / "data"
    mjcf_dir = output_dir / "mjcf"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(base_metadata.read_text())
    maze = MazeMap.from_metadata(base_metadata)
    if furniture_preset not in FURNITURE_PRESETS:
        raise ValueError(f"unknown furniture preset {furniture_preset!r}; choices={sorted(FURNITURE_PRESETS)}")

    furniture = []
    for item in FURNITURE_PRESETS[furniture_preset]:
        cell = tuple(item["cell"])
        if int(maze.occupancy[cell[0], cell[1]]) != 0:
            raise ValueError(f"furniture cell must be an official free cell: {item['name']} cell={cell}")
        furniture.append({**item, "xy": list(maze.cell_to_xy(cell))})

    furnished_metadata = {
        **base,
        "name": f"{base['name']}_FURNISHED_{furniture_preset.upper()}_WALL_PAINTING",
        "furniture_obstacles": furniture,
        "furniture_obstacle_count": len(furniture),
        "furniture_preset": furniture_preset,
        "target_visual": "gallery_wall_painting",
        "target_description": "A non-colliding cyan-and-green cross-panel gallery wall painting is moved to the goal at each reset.",
        "red_ball_target_visible": False,
        "effective_blocking_cell_count": int(sum(sum(1 if value == 1 else 0 for value in row) for row in base["maze_map"])) + len(furniture),
        "obstacle_navigation_requirement": (
            "The listed furniture cells are treated as additional blocking cells by the teacher, "
            "online collision checker, topdown renderer, and MuJoCo scene geoms."
        ),
        "base_map_metadata": str(base_metadata),
        "base_scene_xml": str(base_xml),
    }
    metadata_path = data_dir / f"MEDIUM_MAZE_DIVERSE_GR_scale2_furnished_{furniture_preset}_wall_painting.json"
    metadata_path.write_text(json.dumps(furnished_metadata, indent=2, sort_keys=True) + "\n")

    tree = ET.parse(base_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("base XML does not contain a worldbody")
    target_site = worldbody.find("./site[@name='target']")
    com_marker_site = worldbody.find("./site[@name='com_marker']")
    insert_index = list(worldbody).index(target_site) if target_site is not None else len(list(worldbody))
    if com_marker_site is not None:
        com_marker_site.set("size", "0.001")
        com_marker_site.set("rgba", "0 0 0 0")
    if target_site is not None:
        target_site.set("size", "0.001")
        target_site.set("rgba", "0 0 0 0")
    _add_wall_painting_target(worldbody, insert_index)
    insert_index += len(WALL_PAINTING_GEOMS)
    furniture_container = ET.Element("body", {"name": "furnished_navigation_obstacles", "pos": "0 0 0"})
    for obstacle in furniture:
        if obstacle["kind"] == "table":
            _add_table(furniture_container, obstacle)
        elif obstacle["kind"] == "chair":
            _add_chair(furniture_container, obstacle)
        else:
            _add_geom(
                furniture_container,
                name=obstacle["name"],
                pos=(obstacle["xy"][0], obstacle["xy"][1], obstacle["size"][2]),
                size=tuple(obstacle["size"]),
                geom_type="box",
                rgba=tuple(obstacle["rgba"]),
            )
    worldbody.insert(insert_index, furniture_container)

    xml_path = mjcf_dir / f"sonic_scene_MEDIUM_MAZE_DIVERSE_GR_scale2_furnished_{furniture_preset}_wall_painting.xml"
    tree.write(xml_path, encoding="unicode")

    furnished_maze = MazeMap.from_metadata(metadata_path)
    draw_topdown_map(furnished_maze, figures_dir / "furnished_maze_topdown.png")
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "scene_xml": str(xml_path),
        "base_metadata": str(base_metadata),
        "base_scene_xml": str(base_xml),
        "furniture_preset": furniture_preset,
        "furniture_obstacles": furniture,
        "target_visual": "gallery_wall_painting",
        "red_ball_target_visible": False,
        "free_cells_after_furniture": len(furnished_maze.free_cells),
        "connected_route_generation_ready": True,
        "topdown_path": str(figures_dir / "furnished_maze_topdown.png"),
    }
    write_json(output_dir / "furnished_maze_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "# Furnished SONIC VLN Maze\n\n"
        f"- metadata: `{metadata_path}`\n"
        f"- scene_xml: `{xml_path}`\n"
        f"- furniture_preset: `{furniture_preset}`\n"
        f"- furniture_obstacle_count: `{len(furniture)}`\n"
        "- visible target: `gallery_wall_painting` (the legacy red target site is hidden and used only by the evaluator)\n"
        "- The furniture cells are additional blocking cells for training-time A*, online collision checking, and MuJoCo rendering.\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_run_dir())
    parser.add_argument("--base-metadata", type=Path, default=BASE_METADATA)
    parser.add_argument("--base-xml", type=Path, default=BASE_XML)
    parser.add_argument("--furniture-preset", choices=sorted(FURNITURE_PRESETS), default="default")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_furnished_maze(
        args.output_dir,
        base_metadata=args.base_metadata,
        base_xml=args.base_xml,
        furniture_preset=args.furniture_preset,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
