"""CASA MuJoCo scene prop utilities."""

from .props_config import PropConfig, ScenePropsConfig, load_scene_props_config
from .phase2_v2 import make_phase2_v2_scene_command

try:
    from .props_manager import ScenePropsManager
except ModuleNotFoundError as exc:  # Allows pure scene-plan utilities without MuJoCo installed.
    if exc.name != "mujoco":
        raise
    ScenePropsManager = None  # type: ignore[assignment]

__all__ = [
    "PropConfig",
    "ScenePropsConfig",
    "ScenePropsManager",
    "load_scene_props_config",
    "make_phase2_v2_scene_command",
]
