"""Configuration loading for CASA MuJoCo scene props."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Range3:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Range3":
        return cls(
            x=_range_tuple(data.get("x", [0.0, 0.0])),
            y=_range_tuple(data.get("y", [0.0, 0.0])),
            z=_range_tuple(data.get("z", [0.0, 0.0])),
        )


@dataclass(frozen=True)
class PropConfig:
    prefix: str
    count: int
    enabled_count: int | tuple[int, int]
    position: Range3
    yaw_deg: tuple[float, float]
    close_position: Range3 | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PropConfig":
        enabled_raw = data.get("enabled_count", data.get("count", 0))
        if isinstance(enabled_raw, list):
            enabled_count: int | tuple[int, int] = (int(enabled_raw[0]), int(enabled_raw[1]))
        else:
            enabled_count = int(enabled_raw)
        close_position = data.get("close_position")
        return cls(
            prefix=str(data["prefix"]),
            count=int(data["count"]),
            enabled_count=enabled_count,
            position=Range3.from_dict(data.get("position", {})),
            close_position=Range3.from_dict(close_position) if close_position else None,
            yaw_deg=_range_tuple(data.get("yaw_deg", [0.0, 0.0])),
        )


@dataclass(frozen=True)
class ScenePropsConfig:
    hidden_position: tuple[float, float, float]
    close_user_bias_prob: float
    user: PropConfig
    obstacles: PropConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenePropsConfig":
        return cls(
            hidden_position=_vector3(data.get("hidden_position", [0.0, 0.0, -10.0])),
            close_user_bias_prob=float(data.get("close_user_bias_prob", 0.0)),
            user=PropConfig.from_dict(data["user"]),
            obstacles=PropConfig.from_dict(data["obstacles"]),
        )


def load_scene_props_config(path: str | Path) -> ScenePropsConfig:
    with Path(path).open() as file:
        data = yaml.safe_load(file) or {}
    return ScenePropsConfig.from_dict(data)


def _range_tuple(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected two-value range, got {value!r}")
    return (float(value[0]), float(value[1]))


def _vector3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"expected 3-vector, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))
