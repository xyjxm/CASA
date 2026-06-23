from __future__ import annotations

from gear_sonic.vln.actions import VLNAction
from gear_sonic.vln.casa_runner import (
    METHODS,
    _audit_value_present,
    detect_excessive_intervention_win,
    heldout_result_claimable,
    method_list,
)
from gear_sonic.vln.dmps_mpc_cbf_replan import (
    EXTERNAL_ADAPTED_METHODS,
    EXTERNAL_BASELINE_DEBUG_FIELDS,
    EXTERNAL_BASELINE_METHOD_METADATA,
    MPC_CBF_HUMANOID_ADAPTED_METHOD_NAME,
    SAFEDPA_ADAPTED_METHOD_NAME,
    SPARK_STYLE_FILTER_ADAPTED_METHOD_NAME,
    is_external_adapted_method,
    is_ppsr_v4_method,
    select_external_adapted_baseline,
)
from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.no_casa_runner import MAP_METADATA
from gear_sonic.vln.oracle import RobotPose2D
from gear_sonic.vln.progress_monitor import ProgressMonitor
from gear_sonic.vln.skill_mapping import WalkSkill


def test_external_baseline_method_registration_and_no_aliasing() -> None:
    for method in EXTERNAL_ADAPTED_METHODS:
        assert method in METHODS
        assert method_list(method) == (method,)
        assert is_external_adapted_method(method)
        assert not is_ppsr_v4_method(method)
        assert method not in {"vln_casa_replan", "vln_casa_reject_only", "vln_ppsr_v4_zero_unsafe_success60"}


def test_external_baselines_emit_required_method_specific_debug_fields() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose, unsafe_skill = _find_forward_into_wall_case(maze)
    for method in EXTERNAL_ADAPTED_METHODS:
        selection = select_external_adapted_baseline(
            method=method,
            pose=pose,
            maze=maze,
            robot_radius=0.28,
            nominal_action=VLNAction.FORWARD,
            nominal_skill=unsafe_skill,
            image_path=None,
            progress_monitor=ProgressMonitor(episode_id=f"{method}_ep"),
            step_idx=0,
            safety_margin=0.05,
            horizon=3,
        )
        assert selection.method == method
        assert selection.intervention_applied is True
        assert selection.selected_sequence.sequence_id != "vln_ppsr_v4_zero_unsafe_success60"
        for field in EXTERNAL_BASELINE_DEBUG_FIELDS[method]:
            assert field in selection.debug


def test_external_debug_fields_are_not_casa_wrapper_fields() -> None:
    mpc_fields = set(EXTERNAL_BASELINE_DEBUG_FIELDS[MPC_CBF_HUMANOID_ADAPTED_METHOD_NAME])
    safedpa_fields = set(EXTERNAL_BASELINE_DEBUG_FIELDS[SAFEDPA_ADAPTED_METHOD_NAME])
    spark_fields = set(EXTERNAL_BASELINE_DEBUG_FIELDS[SPARK_STYLE_FILTER_ADAPTED_METHOD_NAME])
    assert "cbf_margin" in mpc_fields
    assert "adaptation_margin" in safedpa_fields
    assert "spark_constraint_margin" in spark_fields
    assert mpc_fields != safedpa_fields
    assert safedpa_fields != spark_fields
    assert mpc_fields != spark_fields


def test_unsafe_oracle_contract_is_unchanged_by_external_baselines() -> None:
    maze = MazeMap.from_metadata(MAP_METADATA)
    pose, unsafe_skill = _find_forward_into_wall_case(maze)
    assert not maze.is_xy_safe(
        pose.x + unsafe_skill.vx * unsafe_skill.step_target_m,
        pose.y + unsafe_skill.vy * unsafe_skill.step_target_m,
        robot_radius=0.28,
    )
    selection = select_external_adapted_baseline(
        method=SPARK_STYLE_FILTER_ADAPTED_METHOD_NAME,
        pose=pose,
        maze=maze,
        robot_radius=0.28,
        nominal_action=VLNAction.FORWARD,
        nominal_skill=unsafe_skill,
        image_path=None,
        progress_monitor=ProgressMonitor(episode_id="spark_ep"),
        step_idx=0,
        safety_margin=0.05,
    )
    assert selection.nominal_rollout.would_block is True
    assert selection.reject_applied is True
    assert selection.debug["spark_official_code_used"] is False


def test_heldout_result_claim_requires_50_episodes() -> None:
    assert heldout_result_claimable({"method": "m", "total_episodes": 50})
    assert not heldout_result_claimable({"method": "m", "total_episodes": 49})


def test_excessive_intervention_win_is_flagged() -> None:
    casa = _summary("vln_ppsr_v4_zero_unsafe_success60", safe_success_rate=0.60, interventions=50, steps=5000)
    high_budget_win = _summary("spark_style_filter_adapted", safe_success_rate=0.70, interventions=3000, steps=5000)
    efficient_win = _summary("mpc_cbf_humanoid_adapted", safe_success_rate=0.70, interventions=70, steps=5000)
    assert detect_excessive_intervention_win(casa, high_budget_win)
    assert not detect_excessive_intervention_win(casa, efficient_win)


def test_spark_style_method_is_marked_style_adaptation_unless_official_code_used() -> None:
    metadata = EXTERNAL_BASELINE_METHOD_METADATA[SPARK_STYLE_FILTER_ADAPTED_METHOD_NAME]
    assert metadata["claim"] == "SPARK-style adaptation"
    assert metadata["official_reproduction"] is False
    assert metadata["official_code_used"] is False
    assert "not an official SPARK reproduction" in metadata["style_adaptation_note"]


def test_external_audit_value_presence_handles_list_debug_fields() -> None:
    assert not _audit_value_present(None)
    assert not _audit_value_present("")
    assert not _audit_value_present([])
    assert _audit_value_present(["blocked_forward", "low_clearance"])
    assert _audit_value_present({"reason": "blocked_forward"})


def _summary(method: str, *, safe_success_rate: float, interventions: int, steps: int) -> dict[str, float | int | str]:
    return {
        "method": method,
        "total_episodes": 50,
        "safe_success_rate": safe_success_rate,
        "unsafe_violation_rate": 0.0,
        "reject_rate_per_step": interventions / steps,
        "replan_count": interventions,
        "safety_filter_intervention_count": interventions,
        "average_steps_per_episode": steps / 50,
        "fallback_count": 0,
    }


def _find_forward_into_wall_case(maze: MazeMap) -> tuple[RobotPose2D, WalkSkill]:
    directions = [
        ((0, 1), 0.0),
        ((-1, 0), 90.0),
        ((0, -1), 180.0),
        ((1, 0), -90.0),
    ]
    for cell in maze.free_cells:
        x, y = maze.cell_to_xy(cell)
        for (dr, dc), yaw in directions:
            if (cell[0] + dr, cell[1] + dc) in maze.blocking_cells():
                vx, vy = _unit_from_yaw(yaw)
                return (
                    RobotPose2D(x, y, yaw),
                    WalkSkill(vx=vx, vy=vy, facing_yaw_deg=yaw, duration=0.7, step_target_m=1.5),
                )
    raise AssertionError("could not find forward-into-wall case")


def _unit_from_yaw(degrees: float) -> tuple[float, float]:
    import math

    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)
