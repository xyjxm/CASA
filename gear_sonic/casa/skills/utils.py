"""Small math helpers for CASA skill commands."""

from __future__ import annotations

import math

from .base import Vector3


def facing_from_yaw(degrees: float) -> Vector3:
    radians = math.radians(degrees)
    return (math.cos(radians), math.sin(radians), 0.0)


def normalize_xy(x: float, y: float) -> Vector3:
    norm = math.hypot(x, y)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (x / norm, y / norm, 0.0)


def coerce_facing(value: object) -> Vector3:
    if isinstance(value, tuple) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    if isinstance(value, list) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return (1.0, 0.0, 0.0)

