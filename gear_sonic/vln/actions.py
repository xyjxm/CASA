"""Normalize VLN/VLA next-step outputs into the no-CASA SONIC action space."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any


class VLNAction(str, Enum):
    FORWARD = "forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    BACKOFF = "backoff"
    STOP = "stop"


ALLOWED_ACTION_VALUES = tuple(action.value for action in VLNAction)

HABITAT_ACTIONS = {
    0: VLNAction.STOP,
    1: VLNAction.FORWARD,
    2: VLNAction.TURN_LEFT,
    3: VLNAction.TURN_RIGHT,
}

ACTION_ALIASES = {
    "forward": VLNAction.FORWARD,
    "move_forward": VLNAction.FORWARD,
    "go_forward": VLNAction.FORWARD,
    "straight": VLNAction.FORWARD,
    "ahead": VLNAction.FORWARD,
    "left": VLNAction.TURN_LEFT,
    "turn_left": VLNAction.TURN_LEFT,
    "rotate_left": VLNAction.TURN_LEFT,
    "right": VLNAction.TURN_RIGHT,
    "turn_right": VLNAction.TURN_RIGHT,
    "rotate_right": VLNAction.TURN_RIGHT,
    "back": VLNAction.BACKOFF,
    "backoff": VLNAction.BACKOFF,
    "back_off": VLNAction.BACKOFF,
    "move_back": VLNAction.BACKOFF,
    "reverse": VLNAction.BACKOFF,
    "stop": VLNAction.STOP,
    "done": VLNAction.STOP,
    "halt": VLNAction.STOP,
}


@dataclass(frozen=True)
class ActionDecision:
    action: VLNAction
    raw_output: Any
    source: str
    confidence: float = 1.0
    magnitude: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _clean_token(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def parse_vln_action(raw_output: Any, *, source: str = "vln") -> ActionDecision:
    """Parse model, route-policy, or Habitat-style output into five actions only."""

    if isinstance(raw_output, ActionDecision):
        return raw_output

    if isinstance(raw_output, VLNAction):
        return ActionDecision(action=raw_output, raw_output=raw_output.value, source=source)

    if isinstance(raw_output, int):
        action = HABITAT_ACTIONS.get(raw_output, VLNAction.BACKOFF)
        return ActionDecision(
            action=action,
            raw_output=raw_output,
            source=source,
            confidence=1.0 if raw_output in HABITAT_ACTIONS else 0.0,
            metadata={"habitat_action_id": raw_output},
        )

    if isinstance(raw_output, dict):
        value = raw_output.get("action", raw_output.get("next_action", raw_output.get("text")))
        decision = parse_vln_action("" if value is None else value, source=source)
        metadata = dict(raw_output)
        metadata.update(decision.metadata)
        return ActionDecision(
            action=decision.action,
            raw_output=raw_output,
            source=source,
            confidence=decision.confidence,
            magnitude=decision.magnitude,
            metadata=metadata,
        )

    text = str(raw_output or "").strip()
    if not text:
        return ActionDecision(
            action=VLNAction.BACKOFF,
            raw_output=raw_output,
            source=source,
            confidence=0.0,
            metadata={"parse_reason": "empty_fallback_to_backoff"},
        )

    lower = text.lower()
    magnitude = _first_number(lower)

    # First version forbids strafe labels. Treat sidestep language as a turn cue.
    if re.search(r"\b(strafe|sidestep|move)\s+left\b", lower):
        return ActionDecision(
            VLNAction.TURN_LEFT,
            raw_output,
            source,
            magnitude=magnitude,
            metadata={"normalized_from_forbidden_label": "strafe_left"},
        )
    if re.search(r"\b(strafe|sidestep|move)\s+right\b", lower):
        return ActionDecision(
            VLNAction.TURN_RIGHT,
            raw_output,
            source,
            magnitude=magnitude,
            metadata={"normalized_from_forbidden_label": "strafe_right"},
        )

    tokens = [_clean_token(token) for token in re.split(r"[\s,;:/|]+", lower)]
    joined = _clean_token(lower)
    candidates = [joined, *tokens]

    for token in candidates:
        if token in ACTION_ALIASES:
            return ActionDecision(
                action=ACTION_ALIASES[token],
                raw_output=raw_output,
                source=source,
                magnitude=magnitude,
            )

    if "stop" in lower:
        action = VLNAction.STOP
    elif "back" in lower or "reverse" in lower:
        action = VLNAction.BACKOFF
    elif "forward" in lower or "ahead" in lower or "straight" in lower:
        action = VLNAction.FORWARD
    elif "left" in lower:
        action = VLNAction.TURN_LEFT
    elif "right" in lower:
        action = VLNAction.TURN_RIGHT
    else:
        action = VLNAction.BACKOFF

    return ActionDecision(
        action=action,
        raw_output=raw_output,
        source=source,
        confidence=0.75 if action is not VLNAction.BACKOFF else 0.0,
        magnitude=magnitude,
        metadata={"parse_reason": "keyword" if action is not VLNAction.BACKOFF else "unrecognized_backoff"},
    )
