from __future__ import annotations

import math

from gear_sonic.vln.actions import ALLOWED_ACTION_VALUES, VLNAction, parse_vln_action
from gear_sonic.vln.skill_mapping import ActionSkillMapper, PassiveSkill, SkillMappingConfig, TurnSkill, WalkSkill


def test_action_space_is_success30_first_version() -> None:
    assert set(ALLOWED_ACTION_VALUES) == {"forward", "turn_left", "turn_right", "backoff", "stop"}


def test_parse_outputs_without_forbidden_actions() -> None:
    assert parse_vln_action(0).action is VLNAction.STOP
    assert parse_vln_action(1).action is VLNAction.FORWARD
    assert parse_vln_action(2).action is VLNAction.TURN_LEFT
    assert parse_vln_action(3).action is VLNAction.TURN_RIGHT
    assert parse_vln_action("move forward 25 cm").action is VLNAction.FORWARD
    assert parse_vln_action("turn left 30 degrees").action is VLNAction.TURN_LEFT
    assert parse_vln_action("please strafe left").action is VLNAction.TURN_LEFT
    assert parse_vln_action("wait").action is VLNAction.BACKOFF


def test_no_casa_skill_mapping_forward_turn_backoff_stop() -> None:
    mapper = ActionSkillMapper(SkillMappingConfig(turn_degrees=30.0, forward_step_m=0.5))

    forward = mapper.action_to_skill(VLNAction.FORWARD)
    assert isinstance(forward, WalkSkill)
    assert math.isclose(forward.vx, 1.0, abs_tol=1e-6)
    assert math.isclose(forward.vy, 0.0, abs_tol=1e-6)
    assert math.isclose(forward.step_target_m, 0.5, abs_tol=1e-6)

    turn = mapper.action_to_skill(VLNAction.TURN_LEFT)
    assert isinstance(turn, TurnSkill)
    assert math.isclose(turn.face_yaw_deg, 30.0, abs_tol=1e-6)

    backoff = mapper.action_to_skill(VLNAction.BACKOFF)
    assert isinstance(backoff, WalkSkill)
    assert backoff.name == "backoff_walk"

    assert isinstance(mapper.action_to_skill(VLNAction.STOP), PassiveSkill)
