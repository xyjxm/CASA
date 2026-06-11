"""MuJoCo scene executor for no-CASA SONIC maze VLN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .maze import MazeMap
from .oracle import RobotPose2D
from .skill_mapping import PassiveSkill, SonicSkill, TurnSkill, WalkSkill, wrap_degrees


@dataclass(frozen=True)
class SkillExecutionResult:
    status: str
    wall_collision: bool
    fall: bool
    pose: RobotPose2D
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def yaw_to_quat_wxyz(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def quat_wxyz_to_yaw_deg(quat: np.ndarray) -> float:
    w, _, _, z = quat
    return math.degrees(math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z))


class MujocoMazeSkillEnv:
    """Executes high-level skills in the official MuJoCo scene without CASA."""

    def __init__(
        self,
        *,
        scene_xml: Path,
        maze: MazeMap,
        output_dir: Path,
        width: int = 320,
        height: int = 240,
        microsteps_per_skill: int = 8,
        robot_radius: float = 0.28,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.scene_xml = Path(scene_xml)
        self.maze = maze
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.microsteps_per_skill = microsteps_per_skill
        self.robot_radius = robot_radius
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.wall_painting_geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in [
                "gallery_wall_painting_panel_x",
                "gallery_wall_painting_panel_y",
            ]
        }
        self.wall_painting_geom_ids = {
            name: geom_id for name, geom_id in self.wall_painting_geom_ids.items() if geom_id >= 0
        }
        if self.camera_id < 0:
            raise ValueError("scene does not contain camera named head_camera")
        if self.target_site_id < 0:
            raise ValueError("scene does not contain site named target")
        self.base_z = float(self.model.qpos0[2]) if self.model.nq >= 3 else 0.793
        self.trajectory: list[dict[str, Any]] = []
        self.current_goal_xy = (0.0, 0.0)
        self.current_episode_id = ""
        self.reset((0.0, 0.0), 0.0, (0.0, 0.0), episode_id="init")

    def close(self) -> None:
        self.renderer.close()

    def reset(
        self,
        start_xy: tuple[float, float],
        start_yaw_deg: float,
        goal_xy: tuple[float, float],
        *,
        episode_id: str,
    ) -> RobotPose2D:
        mujoco = self.mujoco
        self.current_episode_id = episode_id
        self.current_goal_xy = goal_xy
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qpos[0] = start_xy[0]
        self.data.qpos[1] = start_xy[1]
        self.data.qpos[2] = self.base_z
        self.data.qpos[3:7] = yaw_to_quat_wxyz(start_yaw_deg)
        self._set_goal_visual(goal_xy)
        mujoco.mj_forward(self.model, self.data)
        self.trajectory = []
        pose = self.pose()
        self._append_trajectory("reset", pose, wall_collision=False, fall=False)
        return pose

    def _set_goal_visual(self, goal_xy: tuple[float, float]) -> None:
        self.model.site_pos[self.target_site_id] = np.array([goal_xy[0], goal_xy[1], 0.5], dtype=np.float64)
        if not self.wall_painting_geom_ids:
            return
        x, y = goal_xy
        placements = {
            "gallery_wall_painting_panel_x": (x, y, 0.94),
            "gallery_wall_painting_panel_y": (x, y, 0.94),
        }
        for name, geom_id in self.wall_painting_geom_ids.items():
            self.model.geom_pos[geom_id] = np.asarray(placements[name], dtype=np.float64)

    def pose(self) -> RobotPose2D:
        return RobotPose2D(
            x=float(self.data.qpos[0]),
            y=float(self.data.qpos[1]),
            yaw_deg=wrap_degrees(quat_wxyz_to_yaw_deg(self.data.qpos[3:7])),
        )

    def _set_pose(self, pose: RobotPose2D) -> None:
        self.data.qpos[0] = pose.x
        self.data.qpos[1] = pose.y
        self.data.qpos[2] = self.base_z
        self.data.qpos[3:7] = yaw_to_quat_wxyz(pose.yaw_deg)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def _fall_flag(self) -> bool:
        return bool(self.data.qpos[2] < 0.5)

    def _append_trajectory(self, event: str, pose: RobotPose2D, *, wall_collision: bool, fall: bool) -> None:
        self.trajectory.append(
            {
                "episode_id": self.current_episode_id,
                "event": event,
                "pose_x": pose.x,
                "pose_y": pose.y,
                "pose_yaw_deg": pose.yaw_deg,
                "goal_x": self.current_goal_xy[0],
                "goal_y": self.current_goal_xy[1],
                "wall_collision": int(wall_collision),
                "fall": int(fall),
            }
        )

    def render_head(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        try:
            self.renderer.scene.flags[self.mujoco.mjtVisFlag.mjVIS_SITE] = 0 if self.wall_painting_geom_ids else 1
        except Exception:
            pass
        rgb = self.renderer.render()
        Image.fromarray(rgb).save(path)
        return path

    def execute_skill(self, skill: SonicSkill) -> SkillExecutionResult:
        pose = self.pose()
        wall_collision = False
        status = "ok"

        if isinstance(skill, PassiveSkill):
            self._append_trajectory(skill.name, pose, wall_collision=False, fall=self._fall_flag())
            return SkillExecutionResult(status="stopped", wall_collision=False, fall=self._fall_flag(), pose=pose, metadata=skill.params())

        if isinstance(skill, TurnSkill):
            delta = skill.delta_yaw_deg / float(self.microsteps_per_skill)
            for _ in range(self.microsteps_per_skill):
                pose = RobotPose2D(pose.x, pose.y, wrap_degrees(pose.yaw_deg + delta))
                self._set_pose(pose)
                self._append_trajectory(skill.name, pose, wall_collision=False, fall=self._fall_flag())
            return SkillExecutionResult(status=status, wall_collision=False, fall=self._fall_flag(), pose=pose, metadata=skill.params())

        if isinstance(skill, WalkSkill):
            step = skill.step_target_m / float(self.microsteps_per_skill)
            for _ in range(self.microsteps_per_skill):
                next_x = pose.x + skill.vx * step
                next_y = pose.y + skill.vy * step
                if not self.maze.is_xy_safe(next_x, next_y, robot_radius=self.robot_radius):
                    wall_collision = True
                    status = "blocked"
                    self._append_trajectory(skill.name, pose, wall_collision=True, fall=self._fall_flag())
                    break
                pose = RobotPose2D(next_x, next_y, skill.facing_yaw_deg)
                self._set_pose(pose)
                self._append_trajectory(skill.name, pose, wall_collision=False, fall=self._fall_flag())
            return SkillExecutionResult(status=status, wall_collision=wall_collision, fall=self._fall_flag(), pose=pose, metadata=skill.params())

        raise ValueError(f"unsupported skill type: {type(skill)!r}")


def write_video(frame_paths: list[Path], video_path: Path, *, fps: int = 8) -> Path | None:
    if not frame_paths:
        return None
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is None:
        return None
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return None
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    return video_path if video_path.exists() else None


def draw_topdown_map(
    maze: MazeMap,
    path: Path,
    *,
    trajectory_xy: list[tuple[float, float]] | None = None,
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    size_px: int = 640,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size_px, size_px), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    cell = size_px / maze.rows

    def xy_to_px(x: float, y: float) -> tuple[float, float]:
        col = x / maze.scale + (maze.cols / 2.0 - 0.5)
        row = (maze.rows / 2.0 - 0.5) - y / maze.scale
        return ((col + 0.5) * cell, (row + 0.5) * cell)

    for row in range(maze.rows):
        for col in range(maze.cols):
            x0 = col * cell
            y0 = row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            fill = (128, 96, 68) if int(maze.occupancy[row, col]) == 1 else (235, 235, 228)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(210, 210, 210))

    for obstacle in maze.furniture_obstacles:
        row, col = obstacle.cell
        x0 = col * cell + cell * 0.18
        y0 = row * cell + cell * 0.18
        x1 = (col + 1) * cell - cell * 0.18
        y1 = (row + 1) * cell - cell * 0.18
        color = tuple(int(max(0.0, min(1.0, value)) * 255) for value in obstacle.rgba[:3])
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(20, 20, 20), width=3)
        draw.text((x0 + 4, y0 + 4), obstacle.kind[:1].upper(), fill=(255, 255, 255))

    if trajectory_xy and len(trajectory_xy) >= 2:
        points = [xy_to_px(x, y) for x, y in trajectory_xy]
        draw.line(points, fill=(25, 105, 210), width=4)
    if start_xy:
        sx, sy = xy_to_px(*start_xy)
        draw.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], fill=(30, 120, 255))
    if goal_xy:
        gx, gy = xy_to_px(*goal_xy)
        draw.rectangle([gx - 10, gy - 10, gx + 10, gy + 10], fill=(170, 55, 210), outline=(230, 180, 40), width=3)
        draw.text((gx - 4, gy - 8), "P", fill=(255, 255, 255))

    image.save(path)
    return path
