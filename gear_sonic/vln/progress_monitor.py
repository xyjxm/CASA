"""Non-privileged progress monitor for CASA-gated VLN recovery."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class ProgressMonitorState:
    recent_actions: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_casa_rejects: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_selected_recoveries: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_blocked_flags: deque[bool] = field(default_factory=lambda: deque(maxlen=10))
    recent_stop_sources: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_visual_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    turn_loop_counter: int = 0
    repeated_action_counter: int = 0
    repeated_reject_counter: int = 0
    backoff_overuse_counter: int = 0
    stop_overuse_counter: int = 0
    recovery_stuck_counter: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


class ProgressMonitor:
    """Track short-term non-privileged recovery state.

    The monitor consumes only action history, reject flags, skill statuses,
    stop sources, selected recovery actions, and image hashes. It deliberately
    does not accept goal coordinates, distance-to-goal, shortest path, A*, or
    oracle waypoint inputs.
    """

    def __init__(self, *, episode_id: str, window: int = 10) -> None:
        self.episode_id = episode_id
        self.window = int(window)
        self.state = ProgressMonitorState(
            recent_actions=deque(maxlen=window),
            recent_casa_rejects=deque(maxlen=window),
            recent_selected_recoveries=deque(maxlen=window),
            recent_blocked_flags=deque(maxlen=window),
            recent_stop_sources=deque(maxlen=window),
            recent_visual_hashes=deque(maxlen=max(4, window // 2)),
        )

    def pre_step(self, *, step_idx: int, image_path: str | Path | None) -> dict[str, Any]:
        visual_hash = visual_state_hash(image_path)
        if visual_hash:
            self.state.recent_visual_hashes.append(visual_hash)
        return self.snapshot(step_idx=step_idx, visual_state_hash=visual_hash)

    def record_reject(self, *, step_idx: int, nominal_action: str, reject_reason: str) -> None:
        self.state.recent_casa_rejects.append(nominal_action)
        self.state.repeated_reject_counter = _tail_count(list(self.state.recent_casa_rejects), nominal_action)
        self._record_event(
            step_idx=step_idx,
            event_type="casa_reject",
            action=nominal_action,
            reason=reject_reason,
        )

    def record_step(
        self,
        *,
        step_idx: int,
        executed_action: str,
        skill_status: str,
        stop_source: str,
        selected_recovery_action: str | None = None,
    ) -> None:
        self.state.recent_actions.append(executed_action)
        self.state.recent_blocked_flags.append(skill_status in {"blocked", "collision"})
        self.state.recent_stop_sources.append(stop_source or "")
        if selected_recovery_action:
            self.state.recent_selected_recoveries.append(selected_recovery_action)
        self._update_loop_counters()
        for event_type in self.detect_failure_modes():
            self._record_event(step_idx=step_idx, event_type=event_type, action=executed_action, reason="detected")

    def penalties_for_candidate(self, *, nominal_action: str, candidate_sequence: list[str]) -> dict[str, float]:
        first = candidate_sequence[0] if candidate_sequence else "stop_as_last_resort"
        first_family = _action_family(first)
        recent_actions = list(self.state.recent_actions)
        recent_families = [_action_family(action) for action in recent_actions]
        recent_rejects = list(self.state.recent_casa_rejects)
        recovery_loop_penalty = 0.0
        repeated_reject_penalty = 0.0
        unnecessary_stop_penalty = 0.0
        action_switching_penalty = 0.0

        if first_family in recent_families[-3:]:
            recovery_loop_penalty += 0.15 * recent_families[-3:].count(first_family)
        if first == "backoff" and self.state.backoff_overuse_counter:
            recovery_loop_penalty += min(0.45, 0.12 * self.state.backoff_overuse_counter)
        if first_family in {"turn_left", "turn_right"} and self.state.turn_loop_counter:
            recovery_loop_penalty += min(1.20, 0.22 * self.state.turn_loop_counter)
        if first_family in {"turn_left", "turn_right"} and self.state.recovery_stuck_counter:
            recovery_loop_penalty += min(0.60, 0.15 * self.state.recovery_stuck_counter)
        if nominal_action in recent_rejects[-4:]:
            repeated_reject_penalty += min(0.55, 0.14 * recent_rejects[-4:].count(nominal_action))
        if first_family == _action_family(nominal_action) and nominal_action in recent_rejects[-3:]:
            repeated_reject_penalty += 0.25
        if first == "stop_as_last_resort":
            unnecessary_stop_penalty = 1.0 + 0.15 * self.state.stop_overuse_counter
        if recent_actions and first_family != _action_family(recent_actions[-1]):
            action_switching_penalty = 0.04
        if self._is_turn_loop_candidate(first):
            recovery_loop_penalty += 0.30
        if self._is_backoff_forward_loop_candidate(first):
            recovery_loop_penalty += 0.30

        return {
            "recovery_loop_penalty": recovery_loop_penalty,
            "repeated_reject_penalty": repeated_reject_penalty,
            "unnecessary_stop_penalty": unnecessary_stop_penalty,
            "action_switching_penalty": action_switching_penalty,
            "turn_loop_counter": float(self.state.turn_loop_counter),
            "repeated_reject_counter": float(self.state.repeated_reject_counter),
            "backoff_overuse_counter": float(self.state.backoff_overuse_counter),
            "stop_overuse_counter": float(self.state.stop_overuse_counter),
            "recovery_stuck_counter": float(self.state.recovery_stuck_counter),
        }

    def detect_failure_modes(self) -> list[str]:
        modes: list[str] = []
        actions = list(self.state.recent_actions)
        action_families = [_action_family(action) for action in actions]
        rejects = list(self.state.recent_casa_rejects)
        blocked = list(self.state.recent_blocked_flags)
        stops = list(self.state.recent_stop_sources)
        visual_hashes = list(self.state.recent_visual_hashes)

        if len(rejects) >= 2 and rejects[-1] == rejects[-2] == "forward":
            modes.append("repeated_forward_reject")
        if len(action_families) >= 4 and action_families[-4:] in (["turn_left", "turn_right", "turn_left", "turn_right"], ["turn_right", "turn_left", "turn_right", "turn_left"]):
            modes.append("turn_left_right_loop")
        if len(actions) >= 4 and actions[-4:] in (["backoff", "forward", "backoff", "forward"], ["forward", "backoff", "forward", "backoff"]):
            modes.append("backoff_forward_loop")
        if sum(1 for source in stops[-5:] if source and source not in {"vln_policy", "policy_internal_guard"}) >= 2:
            modes.append("repeated_stop_without_visual_goal")
        if len(blocked) >= 4 and sum(bool(item) for item in blocked[-4:]) >= 3:
            modes.append("recovery_stuck")
        if len(rejects) >= 4 and len(set(visual_hashes[-3:])) <= 1 and visual_hashes:
            modes.append("high_reject_low_progress_proxy")
        return modes

    def snapshot(self, *, step_idx: int, visual_state_hash: str | None = None) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_idx": step_idx,
            "recent_actions": list(self.state.recent_actions),
            "recent_casa_rejects": list(self.state.recent_casa_rejects),
            "recent_selected_recoveries": list(self.state.recent_selected_recoveries),
            "recent_blocked_flags": list(self.state.recent_blocked_flags),
            "recent_stop_sources": list(self.state.recent_stop_sources),
            "visual_state_hash": visual_state_hash or (self.state.recent_visual_hashes[-1] if self.state.recent_visual_hashes else ""),
            "turn_loop_counter": self.state.turn_loop_counter,
            "repeated_action_counter": self.state.repeated_action_counter,
            "repeated_reject_counter": self.state.repeated_reject_counter,
            "backoff_overuse_counter": self.state.backoff_overuse_counter,
            "stop_overuse_counter": self.state.stop_overuse_counter,
            "recovery_stuck_counter": self.state.recovery_stuck_counter,
            "failure_modes": self.detect_failure_modes(),
        }

    def event_rows(self) -> list[dict[str, Any]]:
        return list(self.state.events)

    def audit_dict(self) -> dict[str, Any]:
        counts = Counter(row["event_type"] for row in self.state.events)
        return {
            "episode_id": self.episode_id,
            "event_counts": dict(counts),
            "final_snapshot": self.snapshot(step_idx=-1),
            "forbidden_online_inputs": ["goal_xy", "goal_distance", "shortest_path", "astar_path", "oracle_waypoint"],
            "uses_forbidden_online_inputs": False,
        }

    def _update_loop_counters(self) -> None:
        actions = list(self.state.recent_actions)
        action_families = [_action_family(action) for action in actions]
        if actions:
            self.state.repeated_action_counter = _tail_count(actions, actions[-1])
            self.state.backoff_overuse_counter = actions[-6:].count("backoff")
            self.state.stop_overuse_counter = sum(1 for action in actions[-6:] if action == "stop")
        if len(action_families) >= 4 and action_families[-4:] in (["turn_left", "turn_right", "turn_left", "turn_right"], ["turn_right", "turn_left", "turn_right", "turn_left"]):
            self.state.turn_loop_counter += 1
        if len(self.state.recent_blocked_flags) >= 4 and sum(bool(item) for item in list(self.state.recent_blocked_flags)[-4:]) >= 3:
            self.state.recovery_stuck_counter += 1

    def _is_turn_loop_candidate(self, action: str) -> bool:
        recent = list(self.state.recent_actions)
        action_family = _action_family(action)
        if len(recent) < 3 or action_family not in {"turn_left", "turn_right"}:
            return False
        hypothetical = [_action_family(item) for item in recent[-3:]] + [action_family]
        return hypothetical in (
            ["turn_left", "turn_right", "turn_left", "turn_right"],
            ["turn_right", "turn_left", "turn_right", "turn_left"],
        )

    def _is_backoff_forward_loop_candidate(self, action: str) -> bool:
        recent = list(self.state.recent_actions)
        if len(recent) < 3 or action not in {"backoff", "forward", "short_forward"}:
            return False
        normalized = ["forward" if item in {"short_forward", "short_forward_segment"} else item for item in recent[-3:] + [action]]
        return normalized in (
            ["backoff", "forward", "backoff", "forward"],
            ["forward", "backoff", "forward", "backoff"],
        )

    def _record_event(self, *, step_idx: int, event_type: str, action: str, reason: str) -> None:
        self.state.events.append(
            {
                "episode_id": self.episode_id,
                "step_idx": step_idx,
                "event_type": event_type,
                "action": action,
                "reason": reason,
                "snapshot": self.snapshot(step_idx=step_idx),
            }
        )


def visual_state_hash(image_path: str | Path | None) -> str:
    if image_path is None:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    try:
        image = Image.open(path).convert("L").resize((16, 12))
    except Exception:
        return ""
    return hashlib.sha1(image.tobytes()).hexdigest()[:16]


def _tail_count(values: list[str], item: str) -> int:
    count = 0
    for value in reversed(values):
        if value != item:
            break
        count += 1
    return count


def _action_family(action: str) -> str:
    if action in {"small_turn_left", "turn_left", "wide_turn_left", "target_reacquire_turn_left"}:
        return "turn_left"
    if action in {"small_turn_right", "turn_right", "wide_turn_right", "target_reacquire_turn_right"}:
        return "turn_right"
    if action in {
        "short_forward",
        "short_forward_segment",
        "open_space_seek",
        "wide_arc_left",
        "wide_arc_right",
        "wall_follow_left",
        "wall_follow_right",
    }:
        return "forward"
    return action
