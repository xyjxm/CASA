"""Runtime randomization for CASA MuJoCo mocap props."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from .props_config import PropConfig, Range3, ScenePropsConfig


class ScenePropsManager:
    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, config: ScenePropsConfig) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.config = config
        self.hidden_position = np.array(config.hidden_position, dtype=float)
        self.placements: dict[str, dict[str, Any]] = {}
        self._slots = self._discover_slots()
        self._validate_slots()
        self.disable_all()

    def randomize_episode(
        self,
        rng: random.Random | None = None,
        *,
        close_user_bias_prob: float | None = None,
    ) -> dict[str, Any]:
        rng = rng or random.Random()
        self.placements = {}
        self.disable_all()

        user_count = self._sample_enabled_count(self.config.user, rng)
        bias_prob = self.config.close_user_bias_prob if close_user_bias_prob is None else close_user_bias_prob
        use_close_user = rng.random() < bias_prob
        for index in range(user_count):
            name = f"{self.config.user.prefix}{index}"
            pos_range = (
                self.config.user.close_position
                if use_close_user and self.config.user.close_position is not None
                else self.config.user.position
            )
            self._place(name, self.config.user, pos_range, rng, enabled=True)

        obstacle_count = self._sample_enabled_count(self.config.obstacles, rng)
        for index in range(obstacle_count):
            name = f"{self.config.obstacles.prefix}{index}"
            self._place(name, self.config.obstacles, self.config.obstacles.position, rng, enabled=True)

        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.current_placements()

    def disable_all(self) -> None:
        for name in self._slots:
            self.disable(name)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def apply_current_placements(self) -> None:
        for name, placement in self.placements.items():
            mocap_id = self._slots[name]
            self.mj_data.mocap_pos[mocap_id] = np.array(placement["position"], dtype=float)
            self.mj_data.mocap_quat[mocap_id] = np.array(placement["quat_wxyz"], dtype=float)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def apply_placements(self, placements: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.placements = {}
        self.disable_all()
        for name, placement in placements.items():
            if name not in self._slots:
                raise ValueError(f"Unknown CASA mocap prop slot: {name}")
            if not placement.get("enabled", True):
                self.disable(name)
                continue
            self._set_placement(
                name,
                position=placement.get("position", self.hidden_position.tolist()),
                quat_wxyz=placement.get("quat_wxyz"),
                yaw_deg=placement.get("yaw_deg"),
                enabled=True,
            )
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.current_placements()

    def disable(self, name: str) -> None:
        self._set_placement(
            name,
            position=self.hidden_position.tolist(),
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            enabled=False,
        )

    def step_episode(self, t: float) -> None:
        del t

    def current_placements(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "placements": self.placements,
        }

    def _discover_slots(self) -> dict[str, int]:
        slots: dict[str, int] = {}
        for body_id in range(self.mj_model.nbody):
            name = self.mj_model.body(body_id).name
            if not name.startswith((self.config.user.prefix, self.config.obstacles.prefix)):
                continue
            mocap_id = int(self.mj_model.body_mocapid[body_id])
            if mocap_id >= 0:
                slots[name] = mocap_id
        return slots

    def _validate_slots(self) -> None:
        missing = []
        for config in [self.config.user, self.config.obstacles]:
            for index in range(config.count):
                name = f"{config.prefix}{index}"
                if name not in self._slots:
                    missing.append(name)
        if missing:
            raise ValueError(f"Missing CASA mocap prop slots in MuJoCo model: {missing}")

    def _place(
        self,
        name: str,
        config: PropConfig,
        position_range: Range3,
        rng: random.Random,
        *,
        enabled: bool,
    ) -> None:
        mocap_id = self._slots[name]
        yaw = math.radians(rng.uniform(*config.yaw_deg))
        quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        position = np.array(
            [
                rng.uniform(*position_range.x),
                rng.uniform(*position_range.y),
                rng.uniform(*position_range.z),
            ],
            dtype=float,
        )
        self.mj_data.mocap_pos[mocap_id] = position
        self.mj_data.mocap_quat[mocap_id] = quat
        self.placements[name] = {
            "enabled": enabled,
            "position": position.tolist(),
            "quat_wxyz": quat.tolist(),
            "yaw_deg": math.degrees(yaw),
        }

    def _sample_enabled_count(self, config: PropConfig, rng: random.Random) -> int:
        enabled_count = config.enabled_count
        if isinstance(enabled_count, tuple):
            count = rng.randint(enabled_count[0], enabled_count[1])
        else:
            count = enabled_count
        return max(0, min(config.count, int(count)))

    def _set_placement(
        self,
        name: str,
        *,
        position: list[float] | tuple[float, float, float],
        quat_wxyz: list[float] | tuple[float, float, float, float] | None,
        enabled: bool,
        yaw_deg: float | None = None,
    ) -> None:
        mocap_id = self._slots[name]
        if quat_wxyz is None:
            yaw = math.radians(float(yaw_deg or 0.0))
            quat = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        else:
            quat = np.array(quat_wxyz, dtype=float)
        position_array = np.array(position, dtype=float)
        self.mj_data.mocap_pos[mocap_id] = position_array
        self.mj_data.mocap_quat[mocap_id] = quat
        record = {
            "enabled": enabled,
            "position": position_array.tolist(),
            "quat_wxyz": quat.tolist(),
        }
        if yaw_deg is not None:
            record["yaw_deg"] = float(yaw_deg)
        self.placements[name] = record
