"""JSONL schema helpers for auto-generated VLN adapter data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .actions import VLNAction


REQUIRED_DATASET_KEYS = {
    "episode_id",
    "step_idx",
    "instruction",
    "image_path",
    "history_image_paths",
    "teacher_action",
    "sonic_skill",
    "should_stop",
    "stop_label_reason",
    "teacher_metadata",
}


@dataclass(frozen=True)
class AutoVLNDemoRecord:
    episode_id: str
    step_idx: int
    instruction: str
    image_path: str | None
    history_image_paths: list[str]
    teacher_action: str
    sonic_skill: dict[str, Any]
    should_stop: bool
    stop_label_reason: str
    teacher_metadata: dict[str, Any]
    previous_actions: list[str] = field(default_factory=list)
    previous_skill_status: list[str] = field(default_factory=list)
    goal_xy: list[float] | None = None
    pose: dict[str, float] | None = None
    navid_raw_action: str | None = None
    source: str = "oracle_teacher"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def validate_demo_record(data: dict[str, Any]) -> None:
    missing = REQUIRED_DATASET_KEYS - set(data)
    if missing:
        raise ValueError(f"auto VLN record missing keys: {sorted(missing)}")
    if data["teacher_action"] not in {action.value for action in VLNAction}:
        raise ValueError(f"invalid teacher_action: {data['teacher_action']}")
    if data.get("goal_xy") is not None and (not isinstance(data["goal_xy"], list) or len(data["goal_xy"]) != 2):
        raise ValueError("goal_xy must be a two-value list when present")
    pose = data.get("pose")
    if pose is not None and (not isinstance(pose, dict) or not {"x", "y", "yaw_deg"}.issubset(pose)):
        raise ValueError("pose must contain x, y, yaw_deg when present")
    skill = data["sonic_skill"]
    if not isinstance(skill, dict) or "name" not in skill or "params" not in skill:
        raise ValueError("sonic_skill must contain name and params")
    if not isinstance(data["history_image_paths"], list):
        raise ValueError("history_image_paths must be a list")
    if not isinstance(data.get("previous_actions", []), list):
        raise ValueError("previous_actions must be a list when present")
    if not isinstance(data.get("previous_skill_status", []), list):
        raise ValueError("previous_skill_status must be a list when present")
    if not isinstance(data["should_stop"], bool):
        raise ValueError("should_stop must be bool")
    if not isinstance(data["teacher_metadata"], dict):
        raise ValueError("teacher_metadata must be a dict")


def write_jsonl(path: Path, records: list[AutoVLNDemoRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for record in records:
            data = asdict(record)
            validate_demo_record(data)
            file.write(json.dumps(data, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            validate_demo_record(data)
            rows.append(data)
    return rows
