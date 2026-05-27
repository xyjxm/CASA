"""Registry for CASA skill constructors."""

from __future__ import annotations

from typing import Any, Callable

from .gesture import GestureSkill
from .passive import PassiveSkill
from .turn import TurnSkill
from .walk import WalkSkill


class SkillRegistry:
    def __init__(self) -> None:
        self._constructors: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        self._constructors[name] = constructor

    def create(self, name: str, **params: Any) -> Any:
        if name not in self._constructors:
            raise KeyError(f"Unknown skill: {name}")
        return self._constructors[name](**params)

    def names(self) -> list[str]:
        return sorted(self._constructors)


def default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register("walk", WalkSkill)
    registry.register("turn", TurnSkill)
    registry.register("gesture", GestureSkill)
    registry.register("passive", PassiveSkill)
    return registry

