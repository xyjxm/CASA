"""No-CASA end-to-end policy surface for MuJoCo maze VLN."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .actions import ActionDecision, VLNAction


@dataclass(frozen=True)
class InferenceInput:
    episode_id: str
    step_idx: int
    instruction: str
    image_path: str
    history_image_paths: list[str] = field(default_factory=list)
    previous_actions: list[str] = field(default_factory=list)
    previous_skill_status: list[str] = field(default_factory=list)

    def to_schema_dict(self) -> dict[str, Any]:
        return {
            "episode_id": "str",
            "step_idx": "int",
            "instruction": "str",
            "image_path": "head_camera RGB image path",
            "history_image_paths": "list[str] of prior head_camera RGB image paths",
            "previous_actions": "list[str] from forward, turn_left, turn_right, backoff, stop",
            "previous_skill_status": "list[str] non-privileged runtime statuses",
        }


@dataclass(frozen=True)
class VisualTargetEvidence:
    red_ratio: float
    bbox_area_ratio: float
    center_x: float | None
    center_y: float | None
    visible: bool
    target_kind: str = "red_target"
    target_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["target_ratio"] is None:
            data["target_ratio"] = self.red_ratio
        return data


def parse_route_actions_from_instruction(instruction: str) -> list[VLNAction]:
    text = instruction.lower()
    actions: list[VLNAction] = []
    patterns = [
        (r"(?:move\s+)?forward\s*x\s*(\d+)", VLNAction.FORWARD),
        (r"turn\s+left\s*x\s*(\d+)", VLNAction.TURN_LEFT),
        (r"turn\s+right\s*x\s*(\d+)", VLNAction.TURN_RIGHT),
        (r"back\s*off\s*x\s*(\d+)", VLNAction.BACKOFF),
    ]
    matches: list[tuple[int, VLNAction, int]] = []
    for pattern, action in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), action, int(match.group(1))))
    for _, action, count in sorted(matches, key=lambda item: item[0]):
        actions.extend([action] * max(0, count))
    return actions


def detect_red_target(image_path: str | Path) -> VisualTargetEvidence:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    mask = (red > 130.0) & (red > green * 1.45 + 20.0) & (red > blue * 1.45 + 20.0)
    red_ratio = float(mask.mean())
    if not mask.any():
        return VisualTargetEvidence(
            red_ratio=red_ratio,
            bbox_area_ratio=0.0,
            center_x=None,
            center_y=None,
            visible=False,
        )
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(width * height)
    center_x = float(xs.mean() / max(1, width - 1))
    center_y = float(ys.mean() / max(1, height - 1))
    visible = red_ratio >= 0.0005 and bbox_area >= 0.001
    return VisualTargetEvidence(
        red_ratio=red_ratio,
        bbox_area_ratio=bbox_area,
        center_x=center_x,
        center_y=center_y,
        visible=visible,
        target_kind="red_target",
        target_ratio=red_ratio,
    )


def detect_gallery_wall_painting(image_path: str | Path) -> VisualTargetEvidence:
    """Detect the cyan/green gallery painting using only the policy-facing RGB frame."""

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    cyan_mask = (green > 0.48) & (blue > 0.55) & (red < 0.38)
    green_panel_mask = (green > 0.42) & (red < 0.34) & (blue < 0.45)
    mask = cyan_mask | green_panel_mask
    target_ratio = float(mask.mean())
    if not mask.any():
        return VisualTargetEvidence(
            red_ratio=0.0,
            bbox_area_ratio=0.0,
            center_x=None,
            center_y=None,
            visible=False,
            target_kind="gallery_wall_painting",
            target_ratio=0.0,
        )
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(width * height)
    center_x = float(xs.mean() / max(1, width - 1))
    center_y = float(ys.mean() / max(1, height - 1))
    visible = target_ratio >= 0.02 and bbox_area >= 0.05
    return VisualTargetEvidence(
        red_ratio=target_ratio,
        bbox_area_ratio=bbox_area,
        center_x=center_x,
        center_y=center_y,
        visible=visible,
        target_kind="gallery_wall_painting",
        target_ratio=target_ratio,
    )


class RouteInstructionPolicy:
    """Policy with only language, camera history, action history, and skill status inputs."""

    name = "route_instruction_visual_stop_adapter"

    def __init__(
        self,
        *,
        stop_red_ratio_threshold: float = 0.001,
        stop_bbox_area_threshold: float = 0.002,
        max_consecutive_stop: int = 1,
    ) -> None:
        self.stop_red_ratio_threshold = stop_red_ratio_threshold
        self.stop_bbox_area_threshold = stop_bbox_area_threshold
        self.max_consecutive_stop = max_consecutive_stop
        self._routes: dict[str, list[VLNAction]] = {}

    def reset(self, episode_id: str) -> None:
        self._routes.pop(episode_id, None)

    def _route_for(self, obs: InferenceInput) -> list[VLNAction]:
        if obs.episode_id not in self._routes:
            self._routes[obs.episode_id] = parse_route_actions_from_instruction(obs.instruction)
        return self._routes[obs.episode_id]

    def next_action(self, obs: InferenceInput) -> ActionDecision:
        route = self._route_for(obs)
        evidence = detect_red_target(obs.image_path)
        route_index = self._route_progress(route, obs.previous_actions, obs.previous_skill_status)
        route_complete = route_index >= len(route)
        recent_stop_count = 0
        for action in reversed(obs.previous_actions):
            if action != VLNAction.STOP.value:
                break
            recent_stop_count += 1

        blocked = bool(obs.previous_skill_status and obs.previous_skill_status[-1] in {"blocked", "collision"})
        if blocked:
            action = VLNAction.BACKOFF
            reason = "runtime_skill_blocked"
            p_stop = 0.0
        elif route_complete:
            action = VLNAction.STOP
            reason = "route_complete"
            p_stop = 0.98 if evidence.visible else 0.82
        else:
            action = route[route_index] if route else VLNAction.FORWARD
            reason = "follow_instruction_route"
            p_stop = 0.02

        if action is VLNAction.STOP and recent_stop_count >= self.max_consecutive_stop:
            action = VLNAction.BACKOFF
            reason = "suppress_infinite_stop"
            p_stop = 0.0

        return ActionDecision(
            action=action,
            raw_output=action.value,
            source=self.name,
            confidence=0.95 if action is not VLNAction.BACKOFF else 0.6,
            metadata={
                "p_stop": p_stop,
                "stop_reason": reason,
                "route_index": route_index,
                "route_length": len(route),
                "route_complete": route_complete,
                "visual_target": evidence.to_dict(),
                "input_contract": "instruction+head_camera_history+previous_actions+runtime_skill_status",
                "privileged_policy_usage": False,
            },
        )

    @staticmethod
    def _route_progress(route: list[VLNAction], previous_actions: list[str], previous_status: list[str]) -> int:
        progress = 0
        for idx, action_value in enumerate(previous_actions):
            if action_value == VLNAction.STOP.value:
                break
            if action_value == VLNAction.BACKOFF.value:
                continue
            if progress < len(route) and action_value == route[progress].value:
                progress += 1
        return progress
