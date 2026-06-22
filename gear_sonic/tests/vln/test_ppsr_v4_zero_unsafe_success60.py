from __future__ import annotations

from pathlib import Path

from PIL import Image

from gear_sonic.vln.actions import ActionDecision, VLNAction
from gear_sonic.vln.casa_bridge import is_policy_stop_source
from gear_sonic.vln.casa_runner import (
    METHODS,
    CasaVlnRunnerConfig,
    audit_final_status,
    method_list,
    write_v4_attempt_history,
)
from gear_sonic.vln.dmps_mpc_cbf_replan import (
    CandidateScore,
    DMPS_METHOD_NAME,
    PPSR_V2_METHOD_NAME,
    PPSR_V3_METHOD_NAME,
    PPSR_V4_ABLATE_NO_LATE_STOP_RECOVERY_METHOD_NAME,
    PPSR_V4_ABLATE_NO_STOP_VERIFIER_METHOD_NAME,
    PPSR_V4_ABLATE_NO_TASK_RETURN_REPLAN_METHOD_NAME,
    PPSR_V4_ABLATE_NO_VISUAL_GOAL_TRACKER_METHOD_NAME,
    PPSR_V4_ABLATION_METHODS,
    PPSR_V4_LAST_RESORT_STOP_SOURCE,
    PPSR_V4_METHOD_NAME,
    add_ppsr_v4_recovery_sequences,
    apply_ppsr_v4_late_sweep_bonus,
    apply_ppsr_v4_recovery_hard_mask,
    build_ppsr_v3_candidate_sequences,
    is_ppsr_v2_method,
    is_ppsr_v3_method,
    is_ppsr_v4_method,
    ppsr_v4_task_return_replan_enabled,
)
from gear_sonic.vln.no_casa_policy import InferenceInput
from gear_sonic.vln.oracle import RobotPose2D
from gear_sonic.vln.ppsr_v4_stop import (
    PpsrV4StopVerifierConfig,
    apply_ppsr_v4_stop_verifier,
    build_v4_stop_audit,
    detect_privileged_online_leakage,
    ppsr_v4_stop_verifier_config_for_method,
)


def test_ppsr_v4_method_registration() -> None:
    assert PPSR_V4_METHOD_NAME in METHODS
    assert method_list(PPSR_V4_METHOD_NAME) == (PPSR_V4_METHOD_NAME,)
    assert method_list("vln_ppsr_v4") == (PPSR_V4_METHOD_NAME,)
    assert is_ppsr_v4_method(PPSR_V4_METHOD_NAME)
    for method in PPSR_V4_ABLATION_METHODS:
        assert method in METHODS
        assert method_list(method) == (method,)
        assert is_ppsr_v4_method(method)


def test_ppsr_v4_ablation_config_mapping() -> None:
    no_stop = ppsr_v4_stop_verifier_config_for_method(PPSR_V4_ABLATE_NO_STOP_VERIFIER_METHOD_NAME)
    assert no_stop.ablation_variant == "no_stop_verifier"
    assert no_stop.stop_verifier_enabled is False
    assert no_stop.late_stop_recovery_enabled is False
    assert no_stop.visual_goal_tracker_enabled is False

    no_late = ppsr_v4_stop_verifier_config_for_method(PPSR_V4_ABLATE_NO_LATE_STOP_RECOVERY_METHOD_NAME)
    assert no_late.stop_verifier_enabled is True
    assert no_late.late_stop_recovery_enabled is False
    assert no_late.visual_goal_tracker_enabled is True

    no_visual = ppsr_v4_stop_verifier_config_for_method(PPSR_V4_ABLATE_NO_VISUAL_GOAL_TRACKER_METHOD_NAME)
    assert no_visual.stop_verifier_enabled is True
    assert no_visual.late_stop_recovery_enabled is True
    assert no_visual.visual_goal_tracker_enabled is False

    no_task = ppsr_v4_stop_verifier_config_for_method(PPSR_V4_ABLATE_NO_TASK_RETURN_REPLAN_METHOD_NAME)
    assert no_task.stop_verifier_enabled is True
    assert no_task.late_stop_recovery_enabled is True
    assert no_task.visual_goal_tracker_enabled is True
    assert ppsr_v4_task_return_replan_enabled(PPSR_V4_METHOD_NAME) is True
    assert ppsr_v4_task_return_replan_enabled(PPSR_V4_ABLATE_NO_TASK_RETURN_REPLAN_METHOD_NAME) is False


def test_ppsr_v4_does_not_change_v1_v2_v3_registration() -> None:
    assert method_list("vln_ppsr") == (DMPS_METHOD_NAME,)
    assert method_list("vln_escape_macro_progress") == (PPSR_V2_METHOD_NAME,)
    assert method_list("vln_task_return_replan") == (PPSR_V3_METHOD_NAME,)
    assert is_ppsr_v2_method(PPSR_V2_METHOD_NAME)
    assert is_ppsr_v3_method(PPSR_V3_METHOD_NAME)
    assert not is_ppsr_v4_method(PPSR_V3_METHOD_NAME)


def test_v4_recovery_stop_cannot_count_as_policy_stop() -> None:
    assert not is_policy_stop_source(PPSR_V4_LAST_RESORT_STOP_SOURCE)
    status = audit_final_status(
        [
            _summary(DMPS_METHOD_NAME, safe_success_rate=0.20),
            _summary(PPSR_V3_METHOD_NAME, safe_success_rate=0.30),
            _summary(
                PPSR_V4_METHOD_NAME,
                safe_success_rate=0.70,
                recovery_stop_success_leakage_count=1,
            ),
        ],
        _minimal_audits(v4_recovery_leak=1),
    )
    assert status == "FAILED_STOP_LEAKAGE"


def test_v4_privileged_online_leakage_detector_catches_forbidden_fields() -> None:
    audit = detect_privileged_online_leakage(
        [
            {
                "policy_metadata": {
                    "goal_distance": 0.1,
                    "test_time_astar_used": True,
                }
            }
        ]
    )
    assert audit["privileged_online_leakage"] is True
    assert audit["used_goal_distance_for_online_replan"] is True
    assert audit["used_a_star_for_online_replan"] is True


def test_v4_stop_verifier_improves_recall_without_accepting_premature_stop(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    blank = tmp_path / "blank.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(target)
    Image.new("RGB", (80, 60), (30, 30, 30)).save(blank)
    cfg = PpsrV4StopVerifierConfig(
        min_route_motion_actions=2,
        strong_visual_ratio=0.05,
        route_complete_visual_ratio=0.02,
    )

    stop_obs = InferenceInput(
        episode_id="ep",
        step_idx=3,
        instruction="move forward x2; stop near the gallery wall painting",
        image_path=str(target),
        previous_actions=["forward", "forward"],
        previous_skill_status=["ok", "ok"],
    )
    recalled = apply_ppsr_v4_stop_verifier(
        obs=stop_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert recalled.action is VLNAction.STOP
    assert recalled.metadata["ppsr_v4_stop_verifier_applied"] is True

    premature_obs = InferenceInput(
        episode_id="ep",
        step_idx=0,
        instruction="move forward x2; stop near the gallery wall painting",
        image_path=str(blank),
        previous_actions=[],
        previous_skill_status=[],
    )
    filtered = apply_ppsr_v4_stop_verifier(
        obs=premature_obs,
        decision=ActionDecision(VLNAction.STOP, "stop", "vln_policy"),
        config=cfg,
    )
    assert filtered.action is not VLNAction.STOP
    assert filtered.metadata["ppsr_v4_premature_stop_suppressed"] is True


def test_v4_no_stop_verifier_ablation_bypasses_all_stop_rewrites(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    blank = tmp_path / "blank.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(target)
    Image.new("RGB", (80, 60), (20, 20, 20)).save(blank)
    cfg = ppsr_v4_stop_verifier_config_for_method(
        PPSR_V4_ABLATE_NO_STOP_VERIFIER_METHOD_NAME,
        min_route_motion_actions=2,
        route_complete_visual_ratio=0.02,
    )
    stop_obs = InferenceInput(
        episode_id="ep",
        step_idx=3,
        instruction="move forward x2; stop near the gallery wall painting",
        image_path=str(target),
        previous_actions=["forward", "forward"],
        previous_skill_status=["ok", "ok"],
    )
    recalled = apply_ppsr_v4_stop_verifier(
        obs=stop_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert recalled.action is VLNAction.FORWARD
    assert recalled.metadata["ppsr_v4_stop_verifier_bypassed"] is True
    assert recalled.metadata["ppsr_v4_stop_verifier_applied"] is False

    premature_obs = InferenceInput(
        episode_id="ep",
        step_idx=0,
        instruction="move forward x2; stop near the gallery wall painting",
        image_path=str(blank),
        previous_actions=[],
        previous_skill_status=[],
    )
    filtered = apply_ppsr_v4_stop_verifier(
        obs=premature_obs,
        decision=ActionDecision(VLNAction.STOP, "stop", "vln_policy"),
        config=cfg,
    )
    assert filtered.action is VLNAction.STOP
    assert filtered.metadata["ppsr_v4_premature_stop_suppressed"] is False


def test_v4_stop_verifier_does_not_treat_metadata_keys_as_stop_hint(tmp_path: Path) -> None:
    image_path = tmp_path / "far_wall.png"
    image = Image.new("RGB", (80, 60), (30, 30, 30))
    for y in range(20, 23):
        for x in range(80):
            image.putpixel((x, y), (20, 180, 200))
    image.save(image_path)
    obs = InferenceInput(
        episode_id="ep",
        step_idx=4,
        instruction="continue through the corridor and stop when the gallery wall painting is close",
        image_path=str(image_path),
        previous_actions=["forward", "turn_left", "forward", "turn_right"],
        previous_skill_status=["ok", "ok", "ok", "ok"],
    )
    decision = ActionDecision(
        VLNAction.TURN_LEFT,
        {"near_wall_painting_stop_guard_applied": False, "raw_text": "The next action is to turn left."},
        "vln_policy",
    )
    checked = apply_ppsr_v4_stop_verifier(obs=obs, decision=decision)
    assert checked.action is VLNAction.TURN_LEFT
    assert checked.metadata["ppsr_v4_stop_verifier_accept"] is False


def test_v4_stop_verifier_uses_natural_turn_route_before_visual_takeover(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(target)
    cfg = PpsrV4StopVerifierConfig(
        min_route_motion_actions=3,
        route_complete_visual_ratio=0.05,
        strong_visual_ratio=0.90,
    )

    early_obs = InferenceInput(
        episode_id="ep",
        step_idx=4,
        instruction="continue; then turn left when the corridor bends; then continue; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(target),
        previous_actions=["forward", "turn_left", "forward"],
        previous_skill_status=["ok", "ok", "ok"],
    )
    early = apply_ppsr_v4_stop_verifier(
        obs=early_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert early.action is VLNAction.FORWARD
    assert early.metadata["ppsr_v4_route_mode"] == "natural_turns"
    assert early.metadata["ppsr_v4_route_complete"] is False

    complete_obs = InferenceInput(
        episode_id="ep",
        step_idx=7,
        instruction=early_obs.instruction,
        image_path=str(target),
        previous_actions=["forward", "turn_left", "forward", "turn_right", "forward"],
        previous_skill_status=["ok", "ok", "ok", "ok", "ok"],
    )
    complete = apply_ppsr_v4_stop_verifier(
        obs=complete_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert complete.action is VLNAction.STOP
    assert complete.metadata["ppsr_v4_stop_verifier_applied"] is True


def test_v4_calibrated_route_visual_stop_fills_late_stop_gap(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(target)
    cfg = PpsrV4StopVerifierConfig(
        min_calibrated_route_motion_actions=2,
        strong_visual_ratio=0.95,
        route_complete_visual_ratio=0.95,
        calibrated_route_visual_ratio=0.30,
        calibrated_route_bbox_area=0.60,
        calibrated_route_visual_growth_ratio=0.10,
    )
    obs = InferenceInput(
        episode_id="ep",
        step_idx=6,
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(target),
        previous_actions=["turn_left", "forward", "turn_right", "forward"],
        previous_skill_status=["ok", "ok", "ok", "ok"],
    )
    checked = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert checked.action is VLNAction.STOP
    assert checked.metadata["ppsr_v4_calibrated_route_visual_stop"] is True


def test_v4_visual_goal_tracker_steers_to_visible_route_goal(tmp_path: Path) -> None:
    image_path = tmp_path / "right_target.png"
    image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(60):
        for x in range(56, 80):
            image.putpixel((x, y), (20, 180, 200))
    image.save(image_path)
    cfg = PpsrV4StopVerifierConfig(
        min_visual_goal_tracking_motion_actions=2,
        strong_visual_ratio=0.95,
        route_complete_visual_ratio=0.95,
        calibrated_route_visual_ratio=0.95,
        visual_goal_tracking_ratio=0.05,
        max_visual_goal_tracking_ratio=0.50,
        max_visual_goal_tracking_growth_ratio=0.50,
        min_visual_goal_tracking_bbox_area=0.05,
    )
    obs = InferenceInput(
        episode_id="ep",
        step_idx=6,
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(image_path),
        previous_actions=["turn_left", "forward", "turn_right", "forward"],
        previous_skill_status=["ok", "ok", "ok", "ok"],
    )
    tracked = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.TURN_LEFT, "turn_left", "vln_policy"),
        config=cfg,
    )
    assert tracked.action is VLNAction.TURN_RIGHT
    assert tracked.metadata["ppsr_v4_visual_goal_tracker_applied"] is True


def test_v4_no_visual_goal_tracker_ablation_records_disabled_candidate(tmp_path: Path) -> None:
    image_path = tmp_path / "right_target.png"
    image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(60):
        for x in range(56, 80):
            image.putpixel((x, y), (20, 180, 200))
    image.save(image_path)
    cfg = ppsr_v4_stop_verifier_config_for_method(
        PPSR_V4_ABLATE_NO_VISUAL_GOAL_TRACKER_METHOD_NAME,
        min_visual_goal_tracking_motion_actions=2,
        strong_visual_ratio=0.95,
        route_complete_visual_ratio=0.95,
        calibrated_route_visual_ratio=0.95,
        visual_goal_tracking_ratio=0.05,
        max_visual_goal_tracking_ratio=0.50,
        max_visual_goal_tracking_growth_ratio=0.50,
        min_visual_goal_tracking_bbox_area=0.05,
    )
    obs = InferenceInput(
        episode_id="ep",
        step_idx=6,
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(image_path),
        previous_actions=["turn_left", "forward", "turn_right", "forward"],
        previous_skill_status=["ok", "ok", "ok", "ok"],
    )
    tracked = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.TURN_LEFT, "turn_left", "vln_policy"),
        config=cfg,
    )
    assert tracked.action is VLNAction.TURN_LEFT
    assert tracked.metadata["ppsr_v4_visual_goal_tracker_candidate_action"] == VLNAction.TURN_RIGHT.value
    assert tracked.metadata["ppsr_v4_visual_goal_tracker_disabled"] is True
    assert tracked.metadata["ppsr_v4_visual_goal_tracker_applied"] is False


def test_v4_default_low_ratio_route_goal_tracks_instead_of_stopping(tmp_path: Path) -> None:
    image_path = tmp_path / "sparse_right_goal.png"
    image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(60):
        image.putpixel((0, y), (20, 180, 200))
        for x in range(70, 73):
            image.putpixel((x, y), (20, 180, 200))
    image.save(image_path)
    obs = InferenceInput(
        episode_id="ep",
        step_idx=181,
        instruction="turn left when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(image_path),
        previous_actions=[VLNAction.TURN_LEFT.value, *([VLNAction.FORWARD.value] * 180)],
        previous_skill_status=["ok"] * 181,
    )

    tracked = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
    )

    assert tracked.action is VLNAction.TURN_RIGHT
    assert tracked.metadata["ppsr_v4_visual_goal_tracker_applied"] is True
    assert tracked.metadata["ppsr_v4_ultra_late_forward_streak_visual_stop"] is False


def test_v4_late_route_forward_stabilizer_limits_turn_drift(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("RGB", (80, 60), (25, 25, 25)).save(blank)
    cfg = PpsrV4StopVerifierConfig(
        min_late_route_forward_motion_actions=6,
        min_late_route_forward_recent_turns=3,
        late_route_forward_visual_ratio=0.12,
        max_late_route_forward_streak=2,
    )
    obs = InferenceInput(
        episode_id="ep",
        step_idx=12,
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(blank),
        previous_actions=[
            "turn_left",
            "forward",
            "turn_right",
            "forward",
            "turn_right",
            "turn_right",
            "forward",
        ],
        previous_skill_status=["ok"] * 7,
    )
    stabilized = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.TURN_RIGHT, "turn_right", "vln_policy"),
        config=cfg,
    )
    assert stabilized.action is VLNAction.FORWARD
    assert stabilized.metadata["ppsr_v4_late_route_forward_stabilizer_applied"] is True


def test_v4_recent_visual_memory_stop_recalls_after_close_occlusion(tmp_path: Path) -> None:
    strong = tmp_path / "strong_target.png"
    current = tmp_path / "near_occluded_target.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(strong)
    image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(20, 40):
        for x in range(39, 41):
            image.putpixel((x, y), (20, 180, 200))
    image.save(current)
    previous_actions = [
        VLNAction.TURN_LEFT.value,
        VLNAction.FORWARD.value,
        VLNAction.TURN_RIGHT.value,
        *([VLNAction.FORWARD.value] * 112),
        VLNAction.TURN_LEFT.value,
        VLNAction.TURN_RIGHT.value,
        VLNAction.TURN_LEFT.value,
        VLNAction.TURN_RIGHT.value,
        VLNAction.FORWARD.value,
    ]
    obs = InferenceInput(
        episode_id="ep",
        step_idx=len(previous_actions),
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(current),
        history_image_paths=[str(strong), str(strong)],
        previous_actions=previous_actions,
        previous_skill_status=["ok"] * len(previous_actions),
    )

    recalled = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
    )

    assert recalled.action is VLNAction.STOP
    assert recalled.metadata["ppsr_v4_recent_visual_memory_stop"] is True
    assert recalled.metadata["ppsr_v4_stop_verifier_applied"] is True


def test_v4_no_late_stop_recovery_ablation_disables_late_memory_stop(tmp_path: Path) -> None:
    strong = tmp_path / "strong_target.png"
    current = tmp_path / "near_occluded_target.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(strong)
    image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(20, 40):
        for x in range(39, 41):
            image.putpixel((x, y), (20, 180, 200))
    image.save(current)
    previous_actions = [
        VLNAction.TURN_LEFT.value,
        VLNAction.FORWARD.value,
        VLNAction.TURN_RIGHT.value,
        *([VLNAction.FORWARD.value] * 112),
        VLNAction.TURN_LEFT.value,
        VLNAction.TURN_RIGHT.value,
        VLNAction.TURN_LEFT.value,
        VLNAction.TURN_RIGHT.value,
        VLNAction.FORWARD.value,
    ]
    obs = InferenceInput(
        episode_id="ep",
        step_idx=len(previous_actions),
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(current),
        history_image_paths=[str(strong), str(strong)],
        previous_actions=previous_actions,
        previous_skill_status=["ok"] * len(previous_actions),
    )

    checked = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=ppsr_v4_stop_verifier_config_for_method(PPSR_V4_ABLATE_NO_LATE_STOP_RECOVERY_METHOD_NAME),
    )

    assert checked.action is VLNAction.FORWARD
    assert checked.metadata["ppsr_v4_recent_visual_memory_stop_candidate"] is True
    assert checked.metadata["ppsr_v4_recent_visual_memory_stop"] is False
    assert checked.metadata["ppsr_v4_late_stop_recovery_candidate"] is True
    assert checked.metadata["ppsr_v4_late_stop_recovery_disabled"] is True
    assert checked.metadata["ppsr_v4_late_stop_recovery_applied"] is False
    assert "recent_visual_memory_stop" in checked.metadata["ppsr_v4_late_stop_recovery_disabled_sources"]


def test_v4_late_forward_visual_memory_stop_recalls_after_long_approach(tmp_path: Path) -> None:
    moderate = tmp_path / "moderate_full_bbox_target.png"
    current = tmp_path / "low_full_bbox_target.png"
    moderate_image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(0, 60, 3):
        for x in range(0, 80, 4):
            moderate_image.putpixel((x, y), (20, 180, 200))
    moderate_image.putpixel((79, 59), (20, 180, 200))
    moderate_image.save(moderate)
    current_image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(0, 60, 10):
        for x in range(0, 80, 10):
            current_image.putpixel((x, y), (20, 180, 200))
    current_image.putpixel((79, 59), (20, 180, 200))
    current_image.save(current)
    previous_actions = [
        VLNAction.TURN_LEFT.value,
        VLNAction.TURN_RIGHT.value,
        *([VLNAction.FORWARD.value] * 130),
    ]
    obs = InferenceInput(
        episode_id="ep",
        step_idx=len(previous_actions),
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(current),
        history_image_paths=[str(moderate), str(moderate)],
        previous_actions=previous_actions,
        previous_skill_status=["ok"] * len(previous_actions),
    )

    recalled = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
    )

    assert recalled.action is VLNAction.STOP
    assert recalled.metadata["ppsr_v4_late_forward_visual_memory_stop"] is True
    assert recalled.metadata["ppsr_v4_stop_verifier_applied"] is True


def test_v4_low_ratio_full_bbox_memory_stop_uses_soft_stop_probability(tmp_path: Path) -> None:
    moderate = tmp_path / "moderate_full_bbox_target.png"
    current = tmp_path / "low_ratio_full_bbox_target.png"
    moderate_image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(0, 60, 3):
        for x in range(0, 80, 4):
            moderate_image.putpixel((x, y), (20, 180, 200))
    moderate_image.putpixel((79, 59), (20, 180, 200))
    moderate_image.save(moderate)
    current_image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(0, 60, 12):
        for x in range(0, 80, 8):
            current_image.putpixel((x, y), (20, 180, 200))
    current_image.putpixel((79, 59), (20, 180, 200))
    current_image.save(current)
    previous_actions = [
        VLNAction.TURN_LEFT.value,
        VLNAction.TURN_RIGHT.value,
        VLNAction.TURN_LEFT.value,
        *([VLNAction.FORWARD.value] * 4),
    ]
    obs = InferenceInput(
        episode_id="ep",
        step_idx=len(previous_actions),
        instruction="turn left when the corridor bends; then turn right when the corridor bends; then turn left when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(current),
        history_image_paths=[str(moderate), str(moderate)],
        previous_actions=previous_actions,
        previous_skill_status=["ok"] * len(previous_actions),
    )

    recalled = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(
            VLNAction.FORWARD,
            "forward",
            "vln_policy",
            metadata={"probabilities": {"stop": 0.98, "forward": 0.0, "turn_left": 0.0, "turn_right": 0.0}},
        ),
        config=PpsrV4StopVerifierConfig(
            min_low_ratio_full_bbox_memory_motion_actions=6,
            min_low_ratio_full_bbox_memory_recent_turns=3,
        ),
    )

    assert recalled.action is VLNAction.STOP
    assert recalled.metadata["ppsr_v4_low_ratio_full_bbox_memory_stop"] is True
    assert recalled.metadata["ppsr_v4_stop_verifier_applied"] is True


def test_v4_late_forward_momentum_lock_suppresses_turn_after_forward_streak(tmp_path: Path) -> None:
    img = tmp_path / "weak_target.png"
    Image.new("RGB", (80, 60), (120, 140, 140)).save(img)
    obs = InferenceInput(
        episode_id="ep",
        step_idx=116,
        instruction="turn left when the corridor bends; turn right when the corridor bends; turn left when the corridor bends",
        image_path=str(img),
        previous_actions=[
            *([VLNAction.TURN_LEFT.value, VLNAction.TURN_RIGHT.value] * 3),
            *([VLNAction.FORWARD.value] * 4),
        ],
        previous_skill_status=["ok"] * 10,
    )
    decision = ActionDecision(
        action=VLNAction.TURN_LEFT,
        raw_output="turn left",
        source="base",
        confidence=0.60,
        metadata={},
    )
    locked = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=decision,
        config=PpsrV4StopVerifierConfig(
            min_late_forward_momentum_motion_actions=6,
            min_late_forward_momentum_streak=3,
            late_forward_momentum_visual_ratio=0.20,
            min_late_route_forward_motion_actions=999,
        ),
    )

    assert locked.action is VLNAction.FORWARD
    assert locked.metadata["ppsr_v4_late_forward_momentum_lock_applied"] is True


def test_v4_ultra_late_forward_streak_stop_requires_stable_approach(tmp_path: Path) -> None:
    img = tmp_path / "target.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(img)
    cfg = PpsrV4StopVerifierConfig(
        strong_visual_ratio=1.1,
        policy_stop_visual_ratio=1.1,
        route_complete_visual_ratio=1.1,
        calibrated_route_visual_ratio=1.1,
        late_calibrated_route_visual_ratio=1.1,
        late_recovery_visual_ratio=1.1,
        ultra_late_route_visual_ratio=1.1,
        min_ultra_late_forward_streak_stop_motion_actions=8,
        min_ultra_late_forward_streak_stop_streak=8,
        max_ultra_late_forward_streak_stop_recent_turns=1,
        ultra_late_forward_streak_stop_visual_ratio=0.9,
        ultra_late_forward_streak_stop_bbox_area=0.9,
    )
    mature_obs = InferenceInput(
        episode_id="ep",
        step_idx=158,
        instruction="turn left when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(img),
        previous_actions=[VLNAction.TURN_LEFT.value, *([VLNAction.FORWARD.value] * 9)],
        previous_skill_status=["ok"] * 10,
    )
    recalled = apply_ppsr_v4_stop_verifier(
        obs=mature_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert recalled.action is VLNAction.STOP
    assert recalled.metadata["ppsr_v4_ultra_late_forward_streak_visual_stop"] is True

    early_obs = InferenceInput(
        episode_id="ep",
        step_idx=80,
        instruction=mature_obs.instruction,
        image_path=str(img),
        previous_actions=[VLNAction.TURN_LEFT.value, *([VLNAction.FORWARD.value] * 3)],
        previous_skill_status=["ok"] * 4,
    )
    filtered = apply_ppsr_v4_stop_verifier(
        obs=early_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert filtered.action is VLNAction.FORWARD
    assert filtered.metadata["ppsr_v4_ultra_late_forward_streak_visual_stop"] is False


def test_v4_route_cadence_turn_filter_breaks_midroute_turn_fixation(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("RGB", (80, 60), (25, 25, 25)).save(blank)
    instruction = (
        "turn left when the corridor bends; then turn right when the corridor bends; "
        "then turn left when the corridor bends; stop when the gallery wall painting is close"
    )
    obs = InferenceInput(
        episode_id="ep",
        step_idx=9,
        instruction=instruction,
        image_path=str(blank),
        previous_actions=[
            "turn_left",
            "turn_left",
            "turn_left",
            "forward",
            "turn_right",
            "turn_right",
            "forward",
            "forward",
            "forward",
        ],
        previous_skill_status=["ok"] * 9,
    )
    filtered = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(
            VLNAction.TURN_LEFT,
            "turn_left",
            "vln_policy",
            metadata={"probabilities": {"forward": 0.0, "turn_left": 1.0, "turn_right": 0.0}},
        ),
        config=PpsrV4StopVerifierConfig(min_route_cadence_motion_actions=8),
    )
    assert filtered.action is VLNAction.FORWARD
    assert filtered.metadata["ppsr_v4_route_cadence_turn_filter_applied"] is True


def test_v4_low_conf_corridor_turn_filter_keeps_forward_progress(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("RGB", (80, 60), (25, 25, 25)).save(blank)
    instruction = (
        "turn left when the corridor bends; then turn right when the corridor bends; "
        "then turn left when the corridor bends; stop when the gallery wall painting is close"
    )
    obs = InferenceInput(
        episode_id="ep",
        step_idx=11,
        instruction=instruction,
        image_path=str(blank),
        previous_actions=[
            "forward",
            "forward",
            "forward",
            "turn_left",
            "turn_left",
            "turn_left",
            "forward",
            "forward",
            "forward",
        ],
        previous_skill_status=["ok"] * 9,
    )
    filtered = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(
            VLNAction.TURN_RIGHT,
            "turn_right",
            "vln_policy",
            metadata={"probabilities": {"forward": 0.37, "turn_left": 0.0, "turn_right": 0.63}},
        ),
    )
    assert filtered.action is VLNAction.FORWARD
    assert filtered.metadata["ppsr_v4_low_conf_corridor_turn_filter_applied"] is True

    confident = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(
            VLNAction.TURN_RIGHT,
            "turn_right",
            "vln_policy",
            metadata={"probabilities": {"forward": 0.0, "turn_left": 0.0, "turn_right": 1.0}},
        ),
    )
    assert confident.action is VLNAction.TURN_RIGHT
    assert confident.metadata["ppsr_v4_low_conf_corridor_turn_filter_applied"] is False

    route_complete_obs = InferenceInput(
        episode_id="ep",
        step_idx=29,
        instruction=instruction,
        image_path=str(blank),
        previous_actions=[
            "turn_left",
            "forward",
            "turn_right",
            "forward",
            "turn_left",
            "forward",
            "forward",
            "forward",
        ],
        previous_skill_status=["ok"] * 8,
    )
    complete_filtered = apply_ppsr_v4_stop_verifier(
        obs=route_complete_obs,
        decision=ActionDecision(
            VLNAction.TURN_RIGHT,
            "turn_right",
            "vln_policy",
            metadata={"probabilities": {"forward": 0.29, "turn_left": 0.0, "turn_right": 0.71}},
        ),
        config=PpsrV4StopVerifierConfig(min_low_conf_route_complete_motion_actions=6),
    )
    assert complete_filtered.action is VLNAction.FORWARD
    assert complete_filtered.metadata["ppsr_v4_low_conf_corridor_turn_filter_applied"] is True


def test_v4_late_calibrated_route_stop_requires_high_motion_count(tmp_path: Path) -> None:
    image_path = tmp_path / "target.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(image_path)
    cfg = PpsrV4StopVerifierConfig(
        min_calibrated_route_motion_actions=20,
        calibrated_route_visual_ratio=0.95,
        min_late_calibrated_route_motion_actions=6,
        late_calibrated_route_visual_ratio=0.30,
        late_calibrated_route_bbox_area=0.60,
        late_calibrated_route_visual_growth_ratio=0.10,
    )
    early_obs = InferenceInput(
        episode_id="ep",
        step_idx=4,
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(image_path),
        previous_actions=["turn_left", "forward", "turn_right", "forward"],
        previous_skill_status=["ok"] * 4,
    )
    early = apply_ppsr_v4_stop_verifier(
        obs=early_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert early.action is VLNAction.FORWARD

    late_obs = InferenceInput(
        episode_id="ep",
        step_idx=8,
        instruction=early_obs.instruction,
        image_path=str(image_path),
        previous_actions=["turn_left", "forward", "turn_right", "forward", "forward", "forward"],
        previous_skill_status=["ok"] * 6,
    )
    late = apply_ppsr_v4_stop_verifier(
        obs=late_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert late.action is VLNAction.STOP
    assert late.metadata["ppsr_v4_late_calibrated_route_visual_stop"] is True


def test_v4_borderline_visual_stop_holdoff_delays_marginal_calibrated_stop(tmp_path: Path) -> None:
    sparse_target = tmp_path / "sparse_target.png"
    image = Image.new("RGB", (80, 60), (20, 20, 20))
    for y in range(60):
        for x in range(80):
            if (x + y) % 3 == 0:
                image.putpixel((x, y), (20, 180, 200))
    image.putpixel((0, 0), (20, 180, 200))
    image.putpixel((79, 59), (20, 180, 200))
    image.save(sparse_target)
    cfg = PpsrV4StopVerifierConfig(
        strong_visual_ratio=0.95,
        policy_stop_visual_ratio=0.95,
        route_complete_visual_ratio=0.95,
        min_late_calibrated_route_motion_actions=6,
        late_calibrated_route_visual_ratio=0.30,
        late_calibrated_route_bbox_area=0.60,
        late_calibrated_route_visual_growth_ratio=0.10,
        min_borderline_visual_stop_holdoff_motion_actions=6,
        borderline_visual_stop_holdoff_ratio=0.40,
    )
    obs = InferenceInput(
        episode_id="ep",
        step_idx=9,
        instruction="turn left when the corridor bends; then turn right when the corridor bends; stop when the gallery wall painting is close",
        image_path=str(sparse_target),
        previous_actions=["turn_left", "forward", "turn_right", "forward", "turn_left", "forward"],
        previous_skill_status=["ok"] * 6,
    )
    held = apply_ppsr_v4_stop_verifier(
        obs=obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )

    assert held.action is VLNAction.FORWARD
    assert held.metadata["ppsr_v4_late_calibrated_route_visual_stop"] is True
    assert held.metadata["ppsr_v4_borderline_visual_stop_holdoff"] is True
    assert held.metadata["ppsr_v4_stop_verifier_applied"] is False


def test_v4_ultra_late_route_visual_stop_requires_late_motion_and_turns(tmp_path: Path) -> None:
    image_path = tmp_path / "target.png"
    Image.new("RGB", (80, 60), (20, 180, 200)).save(image_path)
    cfg = PpsrV4StopVerifierConfig(
        strong_visual_ratio=1.01,
        policy_stop_visual_ratio=1.01,
        route_complete_visual_ratio=1.01,
        calibrated_route_visual_ratio=1.01,
        late_calibrated_route_visual_ratio=1.01,
        min_late_recovery_motion_actions=999,
        min_ultra_late_route_motion_actions=6,
        min_ultra_late_route_recent_turns=3,
        ultra_late_route_visual_ratio=0.30,
        ultra_late_route_bbox_area=0.60,
    )
    instruction = (
        "turn left when the corridor bends; then turn right when the corridor bends; "
        "stop when the gallery wall painting is close"
    )
    early_obs = InferenceInput(
        episode_id="ep",
        step_idx=5,
        instruction=instruction,
        image_path=str(image_path),
        previous_actions=["turn_left", "forward", "turn_right", "forward"],
        previous_skill_status=["ok"] * 4,
    )
    early = apply_ppsr_v4_stop_verifier(
        obs=early_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert early.action is VLNAction.FORWARD
    assert early.metadata["ppsr_v4_ultra_late_route_visual_stop"] is False

    late_obs = InferenceInput(
        episode_id="ep",
        step_idx=12,
        instruction=instruction,
        image_path=str(image_path),
        previous_actions=[
            "turn_left",
            "forward",
            "turn_right",
            "forward",
            "turn_left",
            "forward",
            "turn_right",
            "forward",
        ],
        previous_skill_status=["ok"] * 8,
    )
    late = apply_ppsr_v4_stop_verifier(
        obs=late_obs,
        decision=ActionDecision(VLNAction.FORWARD, "forward", "vln_policy"),
        config=cfg,
    )
    assert late.action is VLNAction.STOP
    assert late.metadata["ppsr_v4_ultra_late_route_visual_stop"] is True


def test_v4_recovery_candidates_extend_v4_without_changing_v3_library() -> None:
    class _State:
        recent_selected_recoveries = []
        recent_actions = []
        turn_loop_counter = 0
        repeated_reject_counter = 0
        backoff_overuse_counter = 0
        recovery_stuck_counter = 0

    class _Monitor:
        state = _State()

        def detect_failure_modes(self):
            return set()

    pose = RobotPose2D(0.0, 0.0, 0.0)
    v3_candidates, _ = build_ppsr_v3_candidate_sequences(
        pose=pose,
        nominal_action=VLNAction.FORWARD,
        progress_monitor=_Monitor(),
        horizon=5,
    )
    v3_sequences = {candidate.actions for candidate in v3_candidates}
    v4_candidates = add_ppsr_v4_recovery_sequences(v3_candidates, pose=pose, horizon=5)
    v4_sequences = {candidate.actions for candidate in v4_candidates}
    assert v3_sequences.issubset(v4_sequences)
    assert ("target_reacquire_turn_left", "backoff", "turn_left", "short_forward") not in v3_sequences
    assert ("target_reacquire_turn_left", "backoff", "turn_left", "short_forward") in v4_sequences


def test_v4_late_sweep_ranker_is_v4_only_and_late_triggered() -> None:
    wall_follow = _candidate_score(
        ("wall_follow_left", "short_forward"),
        total_score=5.00,
        terminal_front_clearance=0.10,
        terminal_side_clearance=0.18,
        min_clearance=0.16,
    )
    wide_arc = _candidate_score(
        ("wide_arc_right", "short_forward"),
        total_score=4.70,
        terminal_front_clearance=0.08,
        terminal_side_clearance=0.16,
        min_clearance=0.14,
    )
    early_signal = {"step_idx": 50, "repeated_reject_counter": 4, "enable_late_sweep_bonus": True}
    late_signal = {"step_idx": 88, "repeated_reject_counter": 4, "enable_late_sweep_bonus": True}

    assert apply_ppsr_v4_late_sweep_bonus(score=wide_arc, signal=early_signal, total=wide_arc.total_score) == 4.70
    late_wall = apply_ppsr_v4_late_sweep_bonus(score=wall_follow, signal=late_signal, total=wall_follow.total_score)
    late_arc = apply_ppsr_v4_late_sweep_bonus(score=wide_arc, signal=late_signal, total=wide_arc.total_score)

    assert late_arc > wide_arc.total_score
    assert late_wall < wall_follow.total_score
    assert late_arc > late_wall


def test_v4_late_corridor_mask_prefers_corridor_over_reacquire_loop() -> None:
    class _State:
        backoff_overuse_counter = 0

    class _Monitor:
        state = _State()

    mixed_reacquire = _candidate_score(
        ("target_reacquire_turn_left", "backoff", "wide_turn_left", "short_forward"),
        total_score=8.0,
        terminal_front_clearance=0.35,
        terminal_side_clearance=0.35,
        min_clearance=0.20,
    )
    direct_reacquire = _candidate_score(
        ("target_reacquire_turn_left", "short_forward"),
        total_score=7.0,
        terminal_front_clearance=0.20,
        terminal_side_clearance=0.22,
        min_clearance=0.18,
    )
    corridor = _candidate_score(
        ("wall_follow_left", "short_forward"),
        total_score=5.0,
        terminal_front_clearance=0.16,
        terminal_side_clearance=0.24,
        min_clearance=0.18,
    )

    early = apply_ppsr_v4_recovery_hard_mask(
        [mixed_reacquire, direct_reacquire, corridor],
        progress_monitor=_Monitor(),
        signal={"step_idx": 70, "repeated_reject_counter": 3, "target_cue_score": 0.03},
    )
    assert not any(score.hard_mask_applied for score in early)

    late = apply_ppsr_v4_recovery_hard_mask(
        [mixed_reacquire, direct_reacquire, corridor],
        progress_monitor=_Monitor(),
        signal={"step_idx": 96, "repeated_reject_counter": 3, "target_cue_score": 0.03},
    )
    by_sequence = {score.candidate_sequence: score for score in late}
    assert by_sequence[corridor.candidate_sequence].hard_mask_applied is False
    assert "v4_late_corridor_candidate_preferred_over_reacquire_loop" in by_sequence[
        mixed_reacquire.candidate_sequence
    ].hard_mask_reasons
    assert "v4_late_corridor_candidate_preferred_over_reacquire_loop" in by_sequence[
        direct_reacquire.candidate_sequence
    ].hard_mask_reasons


def test_v4_stop_audit_records_precision_recall_premature_late() -> None:
    audit = build_v4_stop_audit(
        method_summary=[
            _summary(
                PPSR_V4_METHOD_NAME,
                safe_success_rate=0.65,
                stop_precision=0.9,
                stop_recall=0.85,
                premature_stop_rate=0.05,
                late_stop_rate=0.10,
            )
        ],
        episode_rows=[
            {"method": PPSR_V4_METHOD_NAME, "stop_failure_type": "premature_stop"},
            {"method": PPSR_V4_METHOD_NAME, "stop_failure_type": "late_stop", "failure_reason": "missing_policy_stop"},
        ],
        decision_rows=[
            {
                "method": PPSR_V4_METHOD_NAME,
                "policy_metadata": {
                    "ppsr_v4_stop_verifier_checked": True,
                    "ppsr_v4_stop_verifier_applied": True,
                    "ppsr_v4_late_stop_recovery_applied": True,
                    "ppsr_v4_late_calibrated_route_visual_stop": True,
                    "ppsr_v4_ultra_late_route_visual_stop": True,
                    "ppsr_v4_ultra_late_forward_streak_visual_stop": True,
                    "ppsr_v4_recent_visual_memory_stop": True,
                    "ppsr_v4_late_forward_visual_memory_stop": True,
                    "ppsr_v4_late_route_forward_stabilizer_applied": True,
                    "ppsr_v4_late_forward_momentum_lock_applied": True,
                },
            }
        ],
        leakage_audit={"privileged_online_leakage": False},
    )
    assert audit["stop_precision"] == 0.9
    assert audit["stop_recall"] == 0.85
    assert audit["premature_stop_count"] == 1
    assert audit["late_stop_count"] == 1
    assert audit["stop_verifier_applied_count"] == 1
    assert audit["late_calibrated_route_visual_stop_count"] == 1
    assert audit["ultra_late_route_visual_stop_count"] == 1
    assert audit["ultra_late_forward_streak_visual_stop_count"] == 1
    assert audit["recent_visual_memory_stop_count"] == 1
    assert audit["late_forward_visual_memory_stop_count"] == 1
    assert audit["late_route_forward_stabilizer_count"] == 1
    assert audit["late_forward_momentum_lock_count"] == 1


def test_v4_unsafe_forces_failed_unsafe_not_zero() -> None:
    status = audit_final_status(
        [
            _summary(DMPS_METHOD_NAME, safe_success_rate=0.20),
            _summary(PPSR_V3_METHOD_NAME, safe_success_rate=0.30),
            _summary(PPSR_V4_METHOD_NAME, safe_success_rate=0.70, unsafe_violation_count=1, unsafe_violation_rate=0.02),
        ],
        _minimal_audits(),
    )
    assert status == "FAILED_UNSAFE_NOT_ZERO"


def test_v4_safe_success_below_60_cannot_pass() -> None:
    status = audit_final_status(
        [
            _summary(DMPS_METHOD_NAME, safe_success_rate=0.20),
            _summary(PPSR_V3_METHOD_NAME, safe_success_rate=0.30),
            _summary(PPSR_V4_METHOD_NAME, safe_success_rate=0.55),
        ],
        _minimal_audits(),
    )
    assert status == "FAILED_SUCCESS_BELOW_60"


def test_v4_locked_attempts_are_not_hidden(tmp_path: Path) -> None:
    config = _config(tmp_path, eval_stage="locked")
    summary = [
        _summary(DMPS_METHOD_NAME, safe_success_rate=0.20),
        _summary(PPSR_V3_METHOD_NAME, safe_success_rate=0.30),
        _summary(PPSR_V4_METHOD_NAME, safe_success_rate=0.65),
    ]
    audits = _minimal_audits()
    write_v4_attempt_history(
        data_dir=tmp_path,
        config=config,
        method_summary=summary,
        audits=audits,
        final_status="FAILED_SUCCESS_BELOW_60",
    )
    write_v4_attempt_history(
        data_dir=tmp_path,
        config=config,
        method_summary=summary,
        audits=audits,
        final_status="PASS_ZERO_UNSAFE_SUCCESS60",
    )
    assert (tmp_path / "v4_locked_attempts.json").read_text().count("final_status") == 2


def test_v4_mock_backend_cannot_count_as_final_success() -> None:
    audits = _minimal_audits(real_backend=False)
    status = audit_final_status(
        [
            _summary(DMPS_METHOD_NAME, safe_success_rate=0.20),
            _summary(PPSR_V3_METHOD_NAME, safe_success_rate=0.30),
            _summary(PPSR_V4_METHOD_NAME, safe_success_rate=0.70),
        ],
        audits,
    )
    assert status != "PASS_ZERO_UNSAFE_SUCCESS60"
    assert status == "PARTIAL_BLOCKED_ENGINEERING"


def _summary(method: str, **overrides):
    base = {
        "method": method,
        "total_episodes": 50,
        "safe_success_rate": 0.65 if method == PPSR_V4_METHOD_NAME else 0.30,
        "success_rate": 0.65 if method == PPSR_V4_METHOD_NAME else 0.30,
        "unsafe_violation_count": 0,
        "unsafe_violation_rate": 0.0,
        "safety_stop_success_leakage_count": 0,
        "recovery_stop_success_leakage_count": 0,
        "stop_precision": 0.90,
        "stop_recall": 0.85,
        "premature_stop_rate": 0.05,
        "late_stop_rate": 0.05,
    }
    base.update(overrides)
    return base


def _candidate_score(
    candidate_sequence: tuple[str, ...],
    *,
    total_score: float,
    terminal_front_clearance: float,
    terminal_side_clearance: float,
    min_clearance: float,
) -> CandidateScore:
    return CandidateScore(
        sequence_candidate_id="_".join(candidate_sequence),
        candidate_sequence=candidate_sequence,
        sequence_length=len(candidate_sequence),
        min_clearance=min_clearance,
        min_barrier_h=min_clearance,
        cbf_violation=False,
        would_block=False,
        predicted_wall_contact_steps=0,
        safety_feasible=True,
        safety_rejection_reason="safe",
        intent_consistency_score=0.5,
        visual_free_space_score=0.5,
        target_or_stop_cue_score=0.0,
        recovery_loop_penalty=0.0,
        repeated_reject_penalty=0.0,
        unnecessary_stop_penalty=0.0,
        action_switching_penalty=0.0,
        candidate_sequence_length_penalty=0.0,
        normalized_clearance=0.5,
        start_clearance=0.2,
        end_clearance=0.3,
        clearance_gain=0.1,
        end_front_clearance=terminal_front_clearance,
        terminal_can_short_forward=True,
        terminal_short_forward_min_clearance=terminal_front_clearance,
        terminal_short_forward_would_block=False,
        terminal_visual_novelty_score=0.5,
        terminal_heading_change_abs=30.0,
        sequence_translation_distance=0.3,
        turn_only_sequence=False,
        contains_translation=True,
        escape_macro_candidate=True,
        escape_macro_completed_candidate=True,
        hard_mask_applied=False,
        hard_mask_reasons=(),
        masked_candidates=(),
        translation_required=False,
        turn_only_candidate_blocked=False,
        repeated_turn_loop_blocked=False,
        small_turn_loop_risk=0.0,
        total_score=total_score,
        terminal_front_clearance=terminal_front_clearance,
        terminal_side_clearance=terminal_side_clearance,
        next_vln_action="forward",
        next_vln_action_allowed=True,
        policy_ready_state=True,
    )


def _minimal_audits(*, v4_safety_leak: int = 0, v4_recovery_leak: int = 0, real_backend: bool = True):
    return {
        "casa_vln_bridge_audit": {"loaded_real_casa_artifacts": True},
        "privileged_leakage_audit": {
            "used_goal_distance_for_online_replan": False,
            "used_shortest_path_for_online_replan": False,
            "used_a_star_for_online_replan": False,
            "used_oracle_waypoint_for_online_replan": False,
            "used_evaluator_success_for_online_replan": False,
            "used_map_cell_progress_for_online_replan": False,
            "privileged_online_leakage": False,
        },
        "stop_source_audit": {"casa_stop_rows_counted_as_policy_stop": 0},
        "v4_stop_audit": {
            "safety_stop_success_leakage_count": v4_safety_leak,
            "recovery_stop_success_leakage_count": v4_recovery_leak,
        },
        "v4_real_backend_audit": {"real_navid_or_uninavid_used": real_backend},
    }


def _config(tmp_path: Path, *, eval_stage: str) -> CasaVlnRunnerConfig:
    return CasaVlnRunnerConfig(
        output_dir=tmp_path,
        scene_xml=tmp_path / "scene.xml",
        map_metadata=tmp_path / "map.json",
        train_scene_xml=tmp_path / "train_scene.xml",
        train_map_metadata=tmp_path / "train_map.json",
        methods=(PPSR_V4_METHOD_NAME,),
        eval_stage=eval_stage,
        locked_attempt_id="attempt",
        navid_cuda_visible_devices="GPU-test",
    )
