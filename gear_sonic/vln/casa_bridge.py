"""Bridge CASA Phase 5 safety gates into SONIC VLN action execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.casa.phase5 import hard_contract_scores, read_json
from gear_sonic.casa.phase5_policy import evaluate_online_method, scale_thresholds
from gear_sonic.vln.actions import VLNAction
from gear_sonic.vln.maze import MazeMap
from gear_sonic.vln.oracle import RobotPose2D
from gear_sonic.vln.skill_mapping import PassiveSkill, SonicSkill, TurnSkill, WalkSkill, wrap_degrees


CASA_SKILL_BY_VLN_ACTION = {
    VLNAction.FORWARD: "walk",
    VLNAction.BACKOFF: "walk",
    VLNAction.TURN_LEFT: "turn",
    VLNAction.TURN_RIGHT: "turn",
    VLNAction.STOP: "passive",
}

POLICY_STOP_SOURCES = {"vln_policy", "policy_internal_guard"}
NON_SUCCESS_STOP_SOURCES = {
    "casa_reject_only_stop",
    "casa_replan_last_resort_stop",
    "dmps_mpc_cbf_last_resort_stop",
    "runtime_wait",
    "episode_boundary_idle",
    "safety_reset_idle",
}


@dataclass(frozen=True)
class WallRiskFeatures:
    candidate_min_clearance: float
    candidate_would_block: bool
    current_min_obstacle_distance: float
    current_external_collision_obstacle: bool
    nearest_obstacle_rel_x: float
    nearest_obstacle_rel_y: float
    sampled_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "vln/candidate_min_clearance": self.candidate_min_clearance,
            "vln/candidate_would_block": int(self.candidate_would_block),
            "env/min_obstacle_distance/current": self.current_min_obstacle_distance,
            "env/external_collision_obstacle/current": int(self.current_external_collision_obstacle),
            "env/nearest_obstacle/rel_x": self.nearest_obstacle_rel_x,
            "env/nearest_obstacle/rel_y": self.nearest_obstacle_rel_y,
            "sampled_points": self.sampled_points,
        }


@dataclass(frozen=True)
class CasaGateDecision:
    raw_risk: float
    threshold: float | None
    risk_margin: float | None
    hard_contract_score: float
    hard_contract_fixed_reject: bool
    casa_reject: bool
    reject_reason: str
    candidate_casa_skill_name: str
    wall_features: WallRiskFeatures
    risk_source: str

    def to_log_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hard_contract_fixed_reject"] = int(self.hard_contract_fixed_reject)
        data["casa_reject"] = int(self.casa_reject)
        data["wall_features"] = self.wall_features.to_dict()
        return data


@dataclass(frozen=True)
class RecoveryCandidate:
    action: str
    skill: SonicSkill
    casa_skill_name: str
    prior_rank: int
    stop_as_last_resort: bool = False


class CasaVlnBridge:
    """CASA Phase 5 gate wrapper for VLN candidate skills.

    The bridge converts VLN actions to the original CASA skill names
    (`walk`, `turn`, `passive`) and computes simulator-derived safety-monitor
    features. It does not expose map, goal, or path information to the VLN
    policy.
    """

    def __init__(
        self,
        *,
        phase4_root: Path,
        phase5_root: Path,
        gate_method: str = "casa_a_hard_or_per_skill",
        threshold_scale_global: float = 1.0,
    ) -> None:
        self.phase4_root = Path(phase4_root)
        self.phase5_root = Path(phase5_root)
        self.gate_method = gate_method
        thresholds_json = read_json(self.phase5_root / "conformal_thresholds.json")
        self.thresholds = scale_thresholds(
            thresholds_json["thresholds"],
            global_scale=threshold_scale_global,
            by_skill={},
        )
        self.thresholds_path = self.phase5_root / "conformal_thresholds.json"
        self.critic_artifact_path = self.phase4_root / "raw_critic" / "raw_critic.pt"
        self.torch = None
        self.model = None
        self.feature_mean: np.ndarray
        self.feature_std: np.ndarray
        self.feature_names: list[str]
        self.skill_type_map: dict[str, int] = {}
        self._load_critic()

    @property
    def critic_loaded(self) -> bool:
        return self.model is not None and self.torch is not None

    @property
    def thresholds_loaded(self) -> bool:
        return bool(self.thresholds.get("per_skill")) and "global" in self.thresholds

    def vln_action_to_casa_skill_name(self, action: VLNAction | str) -> str:
        action_enum = action if isinstance(action, VLNAction) else VLNAction(str(action))
        return CASA_SKILL_BY_VLN_ACTION[action_enum]

    def executable_casa_skill_name(self, skill: SonicSkill, *, action: VLNAction | str | None = None) -> str:
        if action is not None:
            return self.vln_action_to_casa_skill_name(action)
        if isinstance(skill, WalkSkill):
            return "walk"
        if isinstance(skill, TurnSkill):
            return "turn"
        if isinstance(skill, PassiveSkill):
            return "passive"
        raise ValueError(f"unsupported skill for CASA mapping: {skill!r}")

    def decide(
        self,
        *,
        action: VLNAction | str,
        skill: SonicSkill,
        pose: RobotPose2D,
        maze: MazeMap,
        robot_radius: float,
    ) -> CasaGateDecision:
        casa_skill_name = self.executable_casa_skill_name(skill, action=action)
        wall_features = compute_wall_risk_features(skill, pose=pose, maze=maze, robot_radius=robot_radius)
        vector = self._feature_vector(
            skill=skill,
            casa_skill_name=casa_skill_name,
            pose=pose,
            wall_features=wall_features,
        )
        hard_score, hard_fixed = hard_contract_scores(vector[None, :], self.feature_names)
        raw_risk = self._risk(vector)
        decision = evaluate_online_method(
            method=self.gate_method,
            skill_name=casa_skill_name,
            raw_risk=raw_risk,
            hard_contract_fixed_reject=bool(hard_fixed[0]),
            thresholds=self.thresholds,
        )
        return CasaGateDecision(
            raw_risk=raw_risk,
            threshold=decision.threshold,
            risk_margin=decision.risk_margin,
            hard_contract_score=float(hard_score[0]),
            hard_contract_fixed_reject=bool(hard_fixed[0]),
            casa_reject=bool(decision.reject),
            reject_reason=decision.reject_reason,
            candidate_casa_skill_name=casa_skill_name,
            wall_features=wall_features,
            risk_source="real_casa_critic",
        )

    def recovery_candidates(self, *, pose: RobotPose2D) -> list[RecoveryCandidate]:
        heading = float(pose.yaw_deg)
        back_vx, back_vy = _unit_from_yaw(heading + 180.0)
        fwd_vx, fwd_vy = _unit_from_yaw(heading)
        return [
            RecoveryCandidate(
                action="backoff",
                skill=WalkSkill(
                    vx=back_vx,
                    vy=back_vy,
                    facing_yaw_deg=heading,
                    duration=0.25,
                    step_target_m=0.14,
                    name="backoff_walk",
                ),
                casa_skill_name="walk",
                prior_rank=0,
            ),
            RecoveryCandidate(
                action="turn_left",
                skill=TurnSkill(delta_yaw_deg=30.0, face_yaw_deg=wrap_degrees(heading + 30.0), duration=0.30),
                casa_skill_name="turn",
                prior_rank=1,
            ),
            RecoveryCandidate(
                action="turn_right",
                skill=TurnSkill(delta_yaw_deg=-30.0, face_yaw_deg=wrap_degrees(heading - 30.0), duration=0.30),
                casa_skill_name="turn",
                prior_rank=2,
            ),
            RecoveryCandidate(
                action="short_forward_segment",
                skill=WalkSkill(vx=fwd_vx, vy=fwd_vy, facing_yaw_deg=heading, duration=0.25, step_target_m=0.12),
                casa_skill_name="walk",
                prior_rank=3,
            ),
            RecoveryCandidate(
                action="stop_as_last_resort",
                skill=PassiveSkill(duration=0.20, mode="stop"),
                casa_skill_name="passive",
                prior_rank=4,
                stop_as_last_resort=True,
            ),
        ]

    def split_forward_skill(self, skill: SonicSkill, *, max_segment_step_m: float = 0.25) -> list[SonicSkill]:
        if not isinstance(skill, WalkSkill) or skill.name != "walk" or skill.step_target_m <= max_segment_step_m:
            return [skill]
        segment_count = max(1, int(math.ceil(skill.step_target_m / max_segment_step_m)))
        segment_step = skill.step_target_m / segment_count
        segment_duration = skill.duration / segment_count
        return [
            WalkSkill(
                vx=skill.vx,
                vy=skill.vy,
                facing_yaw_deg=skill.facing_yaw_deg,
                duration=segment_duration,
                step_target_m=segment_step,
                name=skill.name,
            )
            for _ in range(segment_count)
        ]

    def _load_critic(self) -> None:
        if not self.critic_artifact_path.exists():
            raise FileNotFoundError(f"missing CASA raw critic artifact: {self.critic_artifact_path}")
        import torch  # noqa: PLC0415

        from gear_sonic.scripts.casa_train_raw_critic import RawRiskCritic  # noqa: PLC0415

        checkpoint = torch.load(self.critic_artifact_path, map_location="cpu", weights_only=False)
        schema = checkpoint["feature_schema"]
        self.feature_names = list(schema["feature_names"])
        self.skill_type_map = {str(key): int(value) for key, value in schema.get("skill_type_map", {}).items()}
        groups = {name: [int(index) for index in values] for name, values in schema.get("groups", {}).items()}
        args = checkpoint.get("args", {})
        model = RawRiskCritic(
            len(self.feature_names),
            groups,
            hidden_dim=int(args.get("hidden_dim", 128)),
            dropout=float(args.get("dropout", 0.05)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.torch = torch
        self.model = model
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    def _feature_vector(
        self,
        *,
        skill: SonicSkill,
        casa_skill_name: str,
        pose: RobotPose2D,
        wall_features: WallRiskFeatures,
    ) -> np.ndarray:
        values: dict[str, float] = {}

        def set_stats(prefix: str, value: float) -> None:
            for suffix in ("current", "mean", "max_abs", "min"):
                values[f"{prefix}/{suffix}"] = float(value)
            values[f"{prefix}/std"] = 0.0

        yaw = math.radians(pose.yaw_deg)
        quat_w = math.cos(yaw / 2.0)
        quat_z = math.sin(yaw / 2.0)
        set_stats("robot/base_pos_x", pose.x)
        set_stats("robot/base_pos_y", pose.y)
        set_stats("robot/base_pos_z", 0.793)
        set_stats("robot/base_quat_w", quat_w)
        set_stats("robot/base_quat_x", 0.0)
        set_stats("robot/base_quat_y", 0.0)
        set_stats("robot/base_quat_z", quat_z)
        set_stats("robot/torso_roll", 0.0)
        set_stats("robot/torso_pitch", 0.0)
        set_stats("robot/torso_yaw", yaw)
        set_stats("env/min_user_distance", 10.0)
        set_stats("env/min_arm_user_distance", 10.0)
        set_stats("env/min_obstacle_distance", wall_features.candidate_min_clearance)
        values["env/external_collision_user/current"] = 0.0
        values["env/external_collision_user/any"] = 0.0
        values["env/external_collision_user/mean"] = 0.0
        values["env/external_collision_obstacle/current"] = float(wall_features.candidate_would_block)
        values["env/external_collision_obstacle/any"] = float(wall_features.candidate_would_block)
        values["env/external_collision_obstacle/mean"] = float(wall_features.candidate_would_block)
        values["env/nearest_obstacle/available"] = 1.0
        values["env/nearest_obstacle/rel_x"] = wall_features.nearest_obstacle_rel_x
        values["env/nearest_obstacle/rel_y"] = wall_features.nearest_obstacle_rel_y
        values["env/nearest_obstacle/rel_z"] = 0.0
        values["env/nearest_obstacle/dist_xy"] = wall_features.current_min_obstacle_distance
        values["runtime/control_overrun_ratio"] = 0.0

        values["skill/type_id"] = float(self.skill_type_map.get(casa_skill_name, 0))
        values["skill/is_walk"] = float(casa_skill_name == "walk")
        values["skill/is_turn"] = float(casa_skill_name == "turn")
        values["skill/is_gesture"] = 0.0
        values["skill/is_passive"] = float(casa_skill_name == "passive")
        values["skill/duration"] = float(getattr(skill, "duration", 0.0))
        values["skill/walk_vx"] = float(getattr(skill, "vx", 0.0))
        values["skill/walk_vy"] = float(getattr(skill, "vy", 0.0))
        values["skill/walk_speed"] = math.hypot(values["skill/walk_vx"], values["skill/walk_vy"])
        values["skill/facing_yaw_deg"] = float(getattr(skill, "facing_yaw_deg", pose.yaw_deg))
        values["skill/turn_face_yaw_deg"] = float(getattr(skill, "face_yaw_deg", pose.yaw_deg))
        values["skill/passive_mode_stop"] = float(isinstance(skill, PassiveSkill) and skill.mode == "stop")
        values["skill/passive_mode_wait"] = float(isinstance(skill, PassiveSkill) and skill.mode == "wait")

        vector = np.asarray([float(values.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def _risk(self, feature_vector: np.ndarray) -> float:
        if self.model is None or self.torch is None:
            raise RuntimeError("CASA raw critic is not loaded")
        normalized = (feature_vector.astype(np.float32) - self.feature_mean) / self.feature_std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(normalized[None, :]).float()
            return float(self.torch.sigmoid(self.model(tensor)).item())


def is_policy_stop_source(stop_source: str | None) -> bool:
    return str(stop_source or "") in POLICY_STOP_SOURCES


def stop_source_for_policy_decision(decision_source: str, metadata: dict[str, Any]) -> str:
    if metadata.get("near_wall_painting_stop_guard_applied"):
        return "policy_internal_guard"
    if "guard" in decision_source:
        return "policy_internal_guard"
    return "vln_policy"


def policy_internal_guard_summary(metadata: dict[str, Any]) -> tuple[bool, str]:
    guard_keys = [
        "near_wall_painting_stop_guard_applied",
        "premature_visual_stop_suppressed",
        "blocked_forward_backoff_guard_applied",
        "turn_streak_forward_guard_applied",
        "turn_cycle_forward_burst_guard_applied",
        "turn_cycle_opposite_turn_guard_applied",
    ]
    applied = [key for key in guard_keys if metadata.get(key)]
    return bool(applied), ",".join(applied)


def compute_wall_risk_features(
    skill: SonicSkill,
    *,
    pose: RobotPose2D,
    maze: MazeMap,
    robot_radius: float,
) -> WallRiskFeatures:
    points = candidate_points(skill, pose=pose)
    clearances = [clearance_to_blocking_geometry(x, y, maze=maze, robot_radius=robot_radius)[0] for x, y in points]
    candidate_min_clearance = min(clearances) if clearances else 0.0
    candidate_would_block = any(not maze.is_xy_safe(x, y, robot_radius=robot_radius) for x, y in points)
    current_clearance, rel = clearance_to_blocking_geometry(pose.x, pose.y, maze=maze, robot_radius=robot_radius)
    return WallRiskFeatures(
        candidate_min_clearance=float(candidate_min_clearance),
        candidate_would_block=bool(candidate_would_block),
        current_min_obstacle_distance=float(current_clearance),
        current_external_collision_obstacle=not maze.is_xy_safe(pose.x, pose.y, robot_radius=robot_radius),
        nearest_obstacle_rel_x=float(rel[0]),
        nearest_obstacle_rel_y=float(rel[1]),
        sampled_points=len(points),
    )


def candidate_points(skill: SonicSkill, *, pose: RobotPose2D, microsteps: int = 8) -> list[tuple[float, float]]:
    if isinstance(skill, WalkSkill):
        step = skill.step_target_m / float(max(1, microsteps))
        return [(pose.x + skill.vx * step * idx, pose.y + skill.vy * step * idx) for idx in range(1, microsteps + 1)]
    return [(pose.x, pose.y)]


def clearance_to_blocking_geometry(
    x: float,
    y: float,
    *,
    maze: MazeMap,
    robot_radius: float,
) -> tuple[float, tuple[float, float]]:
    half = maze.scale / 2.0
    outer_x = maze.cols * maze.scale / 2.0
    outer_y = maze.rows * maze.scale / 2.0
    boundary_clearance = min(outer_x - abs(x), outer_y - abs(y)) - robot_radius
    best_clearance = boundary_clearance
    best_rel = (0.0, 0.0)
    for cell in maze.blocking_cells():
        ox, oy = maze.cell_to_xy(cell)
        dx = max(abs(x - ox) - half, 0.0)
        dy = max(abs(y - oy) - half, 0.0)
        unsigned = math.hypot(dx, dy)
        if unsigned == 0.0:
            signed = -robot_radius
        else:
            signed = unsigned - robot_radius
        if signed < best_clearance:
            best_clearance = signed
            best_rel = (ox - x, oy - y)
    return float(best_clearance), best_rel


def _unit_from_yaw(degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return math.cos(radians), math.sin(radians)


def bridge_audit_dict(bridge: CasaVlnBridge) -> dict[str, Any]:
    return {
        "vln_to_casa_skill_name": {action.value: skill for action, skill in CASA_SKILL_BY_VLN_ACTION.items()},
        "critic_loaded": bridge.critic_loaded,
        "critic_artifact_path": str(bridge.critic_artifact_path),
        "thresholds_loaded": bridge.thresholds_loaded,
        "thresholds_path": str(bridge.thresholds_path),
        "gate_method": bridge.gate_method,
        "risk_source": "real_casa_critic",
        "note": "VLN backoff remains an action label and executable backoff_walk skill, but CASA threshold lookup uses walk.",
    }


def dumps_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True)
