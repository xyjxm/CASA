"""Run the CASA Phase 5 online five-baseline episode experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.baselines.sota_adapters import (  # noqa: E402
    SotaDecisionContext,
    build_sota_registry,
    is_sota_method,
)
from gear_sonic.casa.dataset.invocation_dataset import _extract_features  # noqa: E402
from gear_sonic.casa.io.zmq_publisher import LOCOMOTION_IDLE  # noqa: E402
from gear_sonic.casa.loggers.rollout_logger import RolloutLogger  # noqa: E402
from gear_sonic.casa.oracle import SafetyOracle  # noqa: E402
from gear_sonic.casa.oracle.rollout_summary import build_rollout_summary  # noqa: E402
from gear_sonic.casa.phase5 import (  # noqa: E402
    METHOD_ORDER,
    hard_contract_scores,
    read_json,
    read_prediction_rows,
    split_role,
    write_csv,
    write_json,
)
from gear_sonic.casa.phase5_policy import (  # noqa: E402
    ONLINE_METHOD_ORDER,
    evaluate_online_method,
    method_behavior,
    method_display,
    parse_threshold_scale_by_skill,
    scale_thresholds,
)
from gear_sonic.casa.phase5_recovery import (  # noqa: E402
    choose_fallback_policy,
    plan_recovery,
    rewrite_skill_for_retry,
    split_skill,
)
from gear_sonic.casa.runner_utils import create_executor  # noqa: E402
from gear_sonic.casa.scene.phase2_v2 import make_phase2_v2_scene_command  # noqa: E402
from gear_sonic.casa.skills import GestureSkill, PassiveSkill, TurnSkill, WalkSkill  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Run 1 seed x 2 episodes per method.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sim-log-dir", type=Path)
    parser.add_argument("--zmq-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--pre-window-seconds", type=float, default=1.0)
    parser.add_argument("--post-horizon", type=float, default=2.0)
    parser.add_argument("--pre-episode-settle-seconds", type=float, default=0.0)
    parser.add_argument("--episode-boundary-stop-seconds", type=float, default=0.75)
    parser.add_argument("--post-reset-idle-seconds", type=float, default=1.5)
    parser.add_argument("--episode-upright-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--episode-upright-min-z", type=float, default=0.65)
    parser.add_argument("--episode-upright-retries", type=int, default=2)
    parser.add_argument("--initial-warmup-resets", type=int, default=1)
    parser.add_argument("--initial-warmup-idle-seconds", type=float, default=4.0)
    parser.add_argument("--episode-start-command-seconds", type=float, default=0.75)
    parser.add_argument("--velocity-perturbation-clean-start-margin-seconds", type=float, default=0.5)
    parser.add_argument(
        "--continue-after-initial-upright-failure",
        action="store_true",
        help="Debug only: keep using the same live sim/deploy lane after a clean-start failure.",
    )
    parser.add_argument(
        "--allow-dirty-initial-upright",
        action="store_true",
        help="Debug only: run an episode even if the post-reset upright gate fails.",
    )
    parser.add_argument("--fallback-duration", type=float, default=1.0)
    parser.add_argument("--casa-props-command-file", type=Path)
    parser.add_argument("--casa-props-ack-file", type=Path)
    parser.add_argument(
        "--target-bucket-cycle",
        default="clean_safe,visual_collision_or_close,visual_near_boundary,visual_fall",
    )
    parser.add_argument("--scene-complexity-cycle", default="medium,hard,medium,simple")
    parser.add_argument(
        "--fallback-policy",
        choices=["auto", "stop", "adaptive", "adaptive_retry"],
        default="auto",
    )
    parser.add_argument("--adaptive-retry-count", type=int, default=1)
    parser.add_argument("--segment-long-skills", action="store_true")
    parser.add_argument("--max-segment-duration", type=float, default=0.5)
    parser.add_argument("--recheck-before-segment", action="store_true")
    parser.add_argument("--threshold-scale-global", type=float, default=1.0)
    parser.add_argument("--threshold-scale-by-skill", default="")
    parser.add_argument("--randomize-method-order", action="store_true")
    parser.add_argument("--method-order-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.phase5_root / "online_main_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.methods)
    seeds = _parse_seeds(args.seeds)
    episodes_per_seed = args.episodes_per_seed
    if args.smoke:
        seeds = seeds[:1]
        args.episode_start = 0
        episodes_per_seed = min(episodes_per_seed, 2)
    if not args.dry_run and args.sim_log_dir is None:
        raise SystemExit("--sim-log-dir is required unless --dry-run is set")

    thresholds = scale_thresholds(
        read_json(args.phase5_root / "conformal_thresholds.json")["thresholds"],
        global_scale=args.threshold_scale_global,
        by_skill=parse_threshold_scale_by_skill(args.threshold_scale_by_skill),
    )
    gate = Phase5Gate(
        args.phase4_root,
        thresholds,
        pre_window_seconds=args.pre_window_seconds,
        require_model=not args.dry_run,
    )
    write_json(output_dir / "sota_adapter_calibration.json", gate.sota_calibration_summary)
    oracle = SafetyOracle()
    episode_rows = []
    decision_rows = []
    _run_initial_warmup(args, output_dir)

    for seed in seeds:
        for episode_index in range(args.episode_start, args.episode_start + episodes_per_seed):
            scene_command = _scene_command(args, seed, episode_index)
            skill_sequence = _episode_skills(seed, episode_index)
            for method in _methods_for_episode(methods, args, seed, episode_index):
                row, decisions = _run_one_episode(
                    args=args,
                    output_dir=output_dir,
                    gate=gate,
                    oracle=oracle,
                    method=method,
                    seed=seed,
                    episode_index=episode_index,
                    scene_command=scene_command,
                    skill_sequence=skill_sequence,
                )
                episode_rows.append(row)
                decision_rows.extend(decisions)
                write_csv(output_dir / "online_episode_results.csv", episode_rows)
                write_csv(output_dir / "gate_decisions.csv", decision_rows)
                write_json(output_dir / "method_summary.json", _method_summary(episode_rows))
                abort_after_upright_failure = (
                    row.get("status") == "initial_upright_failed"
                    and not args.continue_after_initial_upright_failure
                )
                if abort_after_upright_failure:
                    write_csv(output_dir / "method_summary.csv", _method_summary_rows(episode_rows))
                    write_json(
                        output_dir / "online_experiment_manifest.json",
                        _manifest(args, output_dir, methods, seeds, episodes_per_seed)
                        | {"aborted_after_initial_upright_failure": True},
                    )
                    print(json.dumps(_method_summary(episode_rows), indent=2, sort_keys=True))
                    return
    write_csv(output_dir / "method_summary.csv", _method_summary_rows(episode_rows))
    write_json(
        output_dir / "online_experiment_manifest.json",
        _manifest(args, output_dir, methods, seeds, episodes_per_seed)
        | {"aborted_after_initial_upright_failure": False},
    )
    print(json.dumps(_method_summary(episode_rows), indent=2, sort_keys=True))


def _manifest(
    args: argparse.Namespace,
    output_dir: Path,
    methods: list[str],
    seeds: list[int],
    episodes_per_seed: int,
) -> dict[str, Any]:
    return {
        "phase": "CASA Phase5 online experiment",
        "phase4_root": str(args.phase4_root),
        "phase5_root": str(args.phase5_root),
        "output_dir": str(output_dir),
        "methods": methods,
        "seeds": seeds,
        "episode_start": args.episode_start,
        "episode_end_exclusive": args.episode_start + episodes_per_seed,
        "episodes_per_seed": episodes_per_seed,
        "dry_run": bool(args.dry_run),
        "smoke": bool(args.smoke),
        "gate_inputs": {
            "thresholds_json": str(args.phase5_root / "conformal_thresholds.json"),
            "raw_critic_checkpoint": str(args.phase4_root / "raw_critic" / "raw_critic.pt"),
            "feature_schema": str(args.phase4_root / "dataset_v1" / "feature_schema.json"),
        },
        "runtime_inputs": {
            "sim_log_dir": "" if args.sim_log_dir is None else str(args.sim_log_dir),
            "zmq_host": args.zmq_host,
            "zmq_port": args.zmq_port,
            "publish_fps": args.publish_fps,
            "pre_window_seconds": args.pre_window_seconds,
            "post_horizon": args.post_horizon,
            "fallback_policy": args.fallback_policy,
            "adaptive_retry_count": args.adaptive_retry_count,
            "segment_long_skills": bool(args.segment_long_skills),
            "max_segment_duration": args.max_segment_duration,
            "recheck_before_segment": bool(args.recheck_before_segment),
            "randomize_method_order": bool(args.randomize_method_order),
            "method_order_seed": args.method_order_seed,
        },
        "threshold_scaling": {
            "global": args.threshold_scale_global,
            "by_skill": parse_threshold_scale_by_skill(args.threshold_scale_by_skill),
        },
        "expected_artifacts": [
            "online_episode_results.csv",
            "gate_decisions.csv",
            "method_summary.csv",
            "method_summary.json",
            "online_experiment_manifest.json",
        ],
        "notes": [
            "All methods receive the same scene command and deterministic skill sequence for each seed/episode.",
            (
                "Rejected candidate skills execute stop/adaptive/adaptive_retry fallback "
                "according to method and flags."
            ),
            "Segmented methods log one gate decision per executed segment.",
        ],
    }


class Phase5Gate:
    def __init__(
        self,
        phase4_root: Path,
        thresholds: dict[str, Any],
        *,
        pre_window_seconds: float,
        require_model: bool,
    ) -> None:
        self.torch = None
        self.model = None
        checkpoint = None
        try:
            import torch  # noqa: PLC0415

            from gear_sonic.scripts.casa_train_raw_critic import RawRiskCritic  # noqa: PLC0415

            checkpoint = torch.load(
                phase4_root / "raw_critic" / "raw_critic.pt",
                map_location="cpu",
                weights_only=False,
            )
            schema = checkpoint["feature_schema"]
            feature_names = list(schema["feature_names"])
            groups = {name: [int(index) for index in values] for name, values in schema.get("groups", {}).items()}
            args = checkpoint.get("args", {})
            model = RawRiskCritic(
                len(feature_names),
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
        except ModuleNotFoundError as exc:
            if require_model:
                raise SystemExit("Torch is required for non-dry-run Phase 5 online gating.") from exc
            schema = read_json(phase4_root / "dataset_v1" / "feature_schema.json")
            feature_names = list(schema["feature_names"])
            self.feature_mean = np.zeros(len(feature_names), dtype=np.float32)
            self.feature_std = np.ones(len(feature_names), dtype=np.float32)
        if checkpoint is not None:
            schema = checkpoint["feature_schema"]
            feature_names = list(schema["feature_names"])
        else:
            schema = read_json(phase4_root / "dataset_v1" / "feature_schema.json")
            feature_names = list(schema["feature_names"])
        self.feature_names = feature_names
        self.thresholds = thresholds
        self.pre_window_seconds = pre_window_seconds
        self.previous_skill_event: dict[str, Any] | None = None
        self.sota_registry = build_sota_registry()
        self.sota_calibration_summary = self._calibrate_sota_adapters(phase4_root)

    def decide(
        self,
        *,
        method: str,
        skill: Any,
        sim_log_dir: Path | None,
        scene_props: dict[str, Any],
    ) -> dict[str, Any]:
        feature_vector, hard_score, hard_fixed = self._features(skill, sim_log_dir, scene_props)
        raw_risk = self._risk(feature_vector)
        if is_sota_method(method):
            return self._decide_sota(
                method=method,
                skill=skill,
                scene_props=scene_props,
                feature_vector=feature_vector,
                raw_risk=raw_risk,
                hard_score=hard_score,
                hard_fixed=hard_fixed,
            )
        policy_decision = evaluate_online_method(
            method=method,
            skill_name=skill.name,
            raw_risk=raw_risk,
            hard_contract_fixed_reject=bool(hard_fixed),
            thresholds=self.thresholds,
        )
        return {
            "method": method,
            "candidate_skill": skill.name,
            "candidate_params_json": json.dumps(skill.params(), sort_keys=True),
            "raw_critic_risk": raw_risk,
            "hard_contract_score": hard_score,
            "hard_contract_fixed_reject": int(bool(hard_fixed)),
            "threshold": "" if policy_decision.threshold is None else policy_decision.threshold,
            "risk_margin": "" if policy_decision.risk_margin is None else policy_decision.risk_margin,
            "decision": "reject" if policy_decision.reject else "allow",
            "reject_reason": policy_decision.reject_reason,
        }

    def _decide_sota(
        self,
        *,
        method: str,
        skill: Any,
        scene_props: dict[str, Any],
        feature_vector: np.ndarray,
        raw_risk: float,
        hard_score: float,
        hard_fixed: bool,
    ) -> dict[str, Any]:
        context = SotaDecisionContext(
            skill_name=skill.name,
            skill_params=skill.params(),
            raw_critic_risk=raw_risk,
            hard_contract_score=hard_score,
            hard_contract_fixed_reject=bool(hard_fixed),
            thresholds=self.thresholds,
            feature_vector=feature_vector,
            feature_names=self.feature_names,
            scene_props=scene_props,
        )
        decision = self.sota_registry.decide(method, context)
        return {
            "method": method,
            "candidate_skill": skill.name,
            "candidate_params_json": json.dumps(skill.params(), sort_keys=True),
            "raw_critic_risk": raw_risk,
            "hard_contract_score": hard_score,
            "hard_contract_fixed_reject": int(bool(hard_fixed)),
            "threshold": "" if decision.threshold is None else decision.threshold,
            "risk_margin": "" if decision.risk_margin is None else decision.risk_margin,
            "decision": "allow" if decision.allow else "reject",
            "reject_reason": decision.reject_reason,
            "source_method": decision.source_method,
            "implementation_fidelity": decision.implementation_fidelity,
            "baseline_mode": decision.mode,
            "risk_score": "" if decision.risk_score is None else decision.risk_score,
            "fallback_mode": "" if decision.fallback_mode is None else decision.fallback_mode,
            "corrected_skill_json": "" if decision.corrected_skill is None else json.dumps(
                decision.corrected_skill,
                sort_keys=True,
            ),
            "runtime_ms": "" if decision.runtime_ms is None else decision.runtime_ms,
            "solver_status": decision.solver_status,
            "diagnostics_json": json.dumps(decision.diagnostics, sort_keys=True),
        }

    def _calibrate_sota_adapters(self, phase4_root: Path) -> dict[str, Any]:
        predictions_csv = phase4_root / "raw_critic" / "predictions.csv"
        if not predictions_csv.exists():
            return {
                "status": "missing_predictions_csv",
                "predictions_csv": str(predictions_csv),
                "adapters": self.sota_registry.metadata(),
            }
        try:
            rows = read_prediction_rows(predictions_csv)
        except Exception as exc:  # noqa: BLE001 - online Plan A methods must remain usable.
            return {
                "status": "calibration_failed",
                "predictions_csv": str(predictions_csv),
                "error": repr(exc),
                "adapters": self.sota_registry.metadata(),
            }
        calibration_rows = [row for row in rows if split_role(row.get("phase4_split")) == "calibration"]
        self.sota_registry.calibrate(calibration_rows)
        return {
            "status": "calibrated",
            "predictions_csv": str(predictions_csv),
            "calibration_rows": len(calibration_rows),
            "adapters": self.sota_registry.metadata(),
        }

    def remember_result(self, result: Any) -> None:
        self.previous_skill_event = {
            "status": getattr(result, "status", ""),
            "params_json": json.dumps(getattr(result, "params", {}), sort_keys=True),
            "skill_name": getattr(result, "skill_name", ""),
        }

    def _features(
        self,
        skill: Any,
        sim_log_dir: Path | None,
        scene_props: dict[str, Any],
    ) -> tuple[np.ndarray, float, bool]:
        rows = []
        if sim_log_dir is not None and (Path(sim_log_dir) / "sim_state.csv").exists():
            rows = _read_recent_sim_rows(Path(sim_log_dir) / "sim_state.csv")
        now = time.time()
        pre_rows = [
            row for row in rows if now - self.pre_window_seconds <= _float(row.get("wall_time")) <= now
        ]
        if not pre_rows:
            before = [row for row in rows if _float(row.get("wall_time"), -1.0) <= now]
            pre_rows = before[-1:] if before else [_empty_sim_row(now)]
        current_row = pre_rows[-1]
        skill_event = {
            "params_json": json.dumps(skill.params(), sort_keys=True),
            "status": "",
            "termination_reason": "",
        }
        record = _extract_features(
            pre_rows=pre_rows,
            current_row=current_row,
            params=skill.params(),
            evidence={},
            skill_name=skill.name,
            skill_event=skill_event,
            previous_skill_event=self.previous_skill_event,
            scene_props=scene_props,
        )
        vector = np.asarray([[float(record.get(name, 0.0)) for name in self.feature_names]], dtype=np.float32)
        hard_score, hard_fixed = hard_contract_scores(vector, self.feature_names)
        return vector[0], float(hard_score[0]), bool(hard_fixed[0])

    def _risk(self, feature_vector: np.ndarray) -> float:
        if self.model is None or self.torch is None:
            return 0.0
        normalized = (feature_vector.astype(np.float32) - self.feature_mean) / self.feature_std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(normalized[None, :]).float()
            return float(self.torch.sigmoid(self.model(tensor)).item())


@dataclass(frozen=True)
class SegmentPolicyResult:
    fallback_count_delta: int
    next_skill_idx: int


def _execute_segment_with_policy(
    *,
    executor: Any,
    gate: Any,
    method: str,
    segment: Any,
    original_skill_idx: int,
    segment_idx: int,
    segment_count: int,
    initial_decision: dict[str, Any] | None,
    episode_id: str,
    episode_index: int,
    seed: int,
    sim_log_dir: Path | None,
    scene_props: dict[str, Any],
    fallback_policy: str,
    fallback_duration: float,
    adaptive_retry_count: int,
    next_skill_idx: int,
    decisions: list[dict[str, Any]],
) -> tuple[int, int]:
    decision = initial_decision or gate.decide(
        method=method,
        skill=segment,
        sim_log_dir=sim_log_dir,
        scene_props=scene_props,
    )
    parent_skill_idx = next_skill_idx
    if decision["decision"] != "reject":
        skill_idx = next_skill_idx
        result = executor.execute_one(segment, episode_id=episode_id, skill_idx=skill_idx)
        gate.remember_result(result)
        _append_decision_row(
            decisions,
            decision,
            method=method,
            episode_id=episode_id,
            episode_index=episode_index,
            seed=seed,
            skill_idx=skill_idx,
            parent_skill_idx=parent_skill_idx,
            original_skill_idx=original_skill_idx,
            segment_idx=segment_idx,
            segment_count=segment_count,
            original_candidate_skill=segment,
            executed_skill=segment,
            result=result,
            fallback_policy=fallback_policy,
            fallback_executed=False,
            attempt_type="candidate_allow",
            attempt_index=0,
            recovery_retry_count=0,
            recovery_attempt_count=0,
            retry_decision="",
            retry_reject_reason="",
            retry_executed=False,
            task_progress_executed=True,
            final_segment_outcome="candidate_allow",
        )
        return 0, next_skill_idx + 1

    if fallback_policy == "adaptive_retry":
        result = _execute_recovery_retry_loop(
            executor=executor,
            gate=gate,
            method=method,
            segment=segment,
            initial_decision=decision,
            episode_id=episode_id,
            episode_index=episode_index,
            seed=seed,
            sim_log_dir=sim_log_dir,
            scene_props=scene_props,
            fallback_duration=fallback_duration,
            adaptive_retry_count=adaptive_retry_count,
            next_skill_idx=next_skill_idx,
            parent_skill_idx=parent_skill_idx,
            original_skill_idx=original_skill_idx,
            segment_idx=segment_idx,
            segment_count=segment_count,
            decisions=decisions,
        )
        return result.fallback_count_delta, result.next_skill_idx

    recovery = plan_recovery(
        segment,
        fallback_policy=fallback_policy,
        fallback_duration=fallback_duration,
        max_retry_count=adaptive_retry_count,
        reject_reason=str(decision.get("reject_reason", "")),
        raw_risk=_float(decision.get("raw_critic_risk")),
        hard_contract_fixed_reject=_decision_hard_fixed(decision),
    )
    skill_idx = next_skill_idx
    result = executor.execute_one(recovery.skill, episode_id=episode_id, skill_idx=skill_idx)
    gate.remember_result(result)
    task_progress = _task_progress_executed(
        candidate_skill=segment,
        executed_skill=recovery.skill,
        attempt_type="recovery_only",
    )
    _append_decision_row(
        decisions,
        decision,
        method=method,
        episode_id=episode_id,
        episode_index=episode_index,
        seed=seed,
        skill_idx=skill_idx,
        parent_skill_idx=parent_skill_idx,
        original_skill_idx=original_skill_idx,
        segment_idx=segment_idx,
        segment_count=segment_count,
        original_candidate_skill=segment,
        executed_skill=recovery.skill,
        result=result,
        fallback_policy=fallback_policy,
        fallback_executed=True,
        recovery_action=recovery.action,
        recovery_reason=recovery.reason,
        attempt_type="recovery_only",
        attempt_index=0,
        recovery_retry_count=recovery.retry_count,
        recovery_attempt_count=1,
        retry_decision="",
        retry_reject_reason="",
        retry_executed=False,
        task_progress_executed=task_progress,
        final_segment_outcome="recovery_only",
    )
    return 1, next_skill_idx + 1


def _execute_recovery_retry_loop(
    *,
    executor: Any,
    gate: Any,
    method: str,
    segment: Any,
    initial_decision: dict[str, Any],
    episode_id: str,
    episode_index: int,
    seed: int,
    sim_log_dir: Path | None,
    scene_props: dict[str, Any],
    fallback_duration: float,
    adaptive_retry_count: int,
    next_skill_idx: int,
    parent_skill_idx: int,
    original_skill_idx: int,
    segment_idx: int,
    segment_count: int,
    decisions: list[dict[str, Any]],
) -> SegmentPolicyResult:
    max_retry_count = max(0, int(adaptive_retry_count))
    fallback_count = 0
    current_skill_idx = next_skill_idx
    last_recovery_skill: Any = PassiveSkill(duration=max(0.05, float(fallback_duration)), mode="stop")
    last_recovery_action = ""
    last_recovery_reason = ""
    last_retry_decision: dict[str, Any] | None = None

    for attempt_index in range(max_retry_count + 1):
        recovery = plan_recovery(
            segment,
            fallback_policy="adaptive_retry",
            fallback_duration=fallback_duration,
            max_retry_count=max_retry_count,
            reject_reason=str(initial_decision.get("reject_reason", "")),
            raw_risk=_float(initial_decision.get("raw_critic_risk")),
            hard_contract_fixed_reject=_decision_hard_fixed(initial_decision),
        )
        last_recovery_skill = recovery.skill
        last_recovery_action = recovery.action
        last_recovery_reason = recovery.reason
        recovery_result = executor.execute_one(recovery.skill, episode_id=episode_id, skill_idx=current_skill_idx)
        fallback_count += 1
        gate.remember_result(recovery_result)
        _append_decision_row(
            decisions,
            initial_decision,
            method=method,
            episode_id=episode_id,
            episode_index=episode_index,
            seed=seed,
            skill_idx=current_skill_idx,
            parent_skill_idx=parent_skill_idx,
            original_skill_idx=original_skill_idx,
            segment_idx=segment_idx,
            segment_count=segment_count,
            original_candidate_skill=segment,
            executed_skill=recovery.skill,
            result=recovery_result,
            fallback_policy="adaptive_retry",
            fallback_executed=True,
            recovery_action=recovery.action,
            recovery_reason=recovery.reason,
            attempt_type="recovery_attempt",
            attempt_index=attempt_index,
            recovery_retry_count=recovery.retry_count,
            recovery_attempt_count=attempt_index + 1,
            retry_decision="",
            retry_reject_reason="",
            retry_executed=False,
            task_progress_executed=_task_progress_executed(
                candidate_skill=segment,
                executed_skill=recovery.skill,
                attempt_type="recovery_attempt",
            ),
            final_segment_outcome="recovery_attempt",
        )
        current_skill_idx += 1

        retry_skill = rewrite_skill_for_retry(
            segment,
            attempt_index=attempt_index,
            max_retry_count=max_retry_count,
            fallback_duration=fallback_duration,
            scene_props=scene_props,
        )
        retry_decision = gate.decide(
            method=method,
            skill=retry_skill,
            sim_log_dir=sim_log_dir,
            scene_props=scene_props,
        )
        last_retry_decision = retry_decision
        retry_allowed = retry_decision["decision"] != "reject"
        _append_decision_row(
            decisions,
            retry_decision,
            method=method,
            episode_id=episode_id,
            episode_index=episode_index,
            seed=seed,
            skill_idx=current_skill_idx,
            parent_skill_idx=parent_skill_idx,
            original_skill_idx=original_skill_idx,
            segment_idx=segment_idx,
            segment_count=segment_count,
            original_candidate_skill=segment,
            executed_skill=retry_skill,
            result=None,
            fallback_policy="adaptive_retry",
            fallback_executed=False,
            attempt_type="retry_decision",
            attempt_index=attempt_index,
            recovery_retry_count=recovery.retry_count,
            recovery_attempt_count=attempt_index + 1,
            retry_decision=retry_decision["decision"],
            retry_reject_reason="" if retry_allowed else str(retry_decision.get("reject_reason", "")),
            retry_executed=False,
            task_progress_executed=False,
            final_segment_outcome="retry_allowed" if retry_allowed else "retry_rejected",
        )
        current_skill_idx += 1
        if not retry_allowed:
            continue

        retry_result = executor.execute_one(retry_skill, episode_id=episode_id, skill_idx=current_skill_idx)
        gate.remember_result(retry_result)
        _append_decision_row(
            decisions,
            retry_decision,
            method=method,
            episode_id=episode_id,
            episode_index=episode_index,
            seed=seed,
            skill_idx=current_skill_idx,
            parent_skill_idx=parent_skill_idx,
            original_skill_idx=original_skill_idx,
            segment_idx=segment_idx,
            segment_count=segment_count,
            original_candidate_skill=segment,
            executed_skill=retry_skill,
            result=retry_result,
            fallback_policy="adaptive_retry",
            fallback_executed=False,
            attempt_type="retry_executed",
            attempt_index=attempt_index,
            recovery_retry_count=recovery.retry_count,
            recovery_attempt_count=attempt_index + 1,
            retry_decision="allow",
            retry_reject_reason="",
            retry_executed=True,
            task_progress_executed=True,
            final_segment_outcome="recovery_then_retry",
        )
        return SegmentPolicyResult(fallback_count, current_skill_idx + 1)

    final_decision = last_retry_decision or initial_decision
    _append_decision_row(
        decisions,
        final_decision,
        method=method,
        episode_id=episode_id,
        episode_index=episode_index,
        seed=seed,
        skill_idx=current_skill_idx,
        parent_skill_idx=parent_skill_idx,
        original_skill_idx=original_skill_idx,
        segment_idx=segment_idx,
        segment_count=segment_count,
        original_candidate_skill=segment,
        executed_skill=last_recovery_skill,
        result=None,
        fallback_policy="adaptive_retry",
        fallback_executed=False,
        recovery_action=last_recovery_action,
        recovery_reason=last_recovery_reason,
        attempt_type="final_reject",
        attempt_index=max_retry_count,
        recovery_retry_count=max_retry_count,
        recovery_attempt_count=max_retry_count + 1,
        retry_decision="reject",
        retry_reject_reason=str(final_decision.get("reject_reason", "")),
        retry_executed=False,
        task_progress_executed=False,
        final_segment_outcome="recovery_only_final_reject",
    )
    return SegmentPolicyResult(fallback_count, current_skill_idx + 1)


def _append_decision_row(
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    *,
    method: str,
    episode_id: str,
    episode_index: int,
    seed: int,
    skill_idx: int,
    parent_skill_idx: int,
    original_skill_idx: int,
    segment_idx: int,
    segment_count: int,
    original_candidate_skill: Any,
    executed_skill: Any,
    result: Any | None,
    fallback_policy: str,
    fallback_executed: bool,
    attempt_type: str,
    attempt_index: int,
    recovery_retry_count: int,
    recovery_attempt_count: int,
    retry_decision: str,
    retry_reject_reason: str,
    retry_executed: bool,
    task_progress_executed: bool,
    final_segment_outcome: str,
    recovery_action: str = "",
    recovery_reason: str = "",
) -> None:
    row = dict(decision)
    row["method"] = method
    row.setdefault("candidate_skill", getattr(executed_skill, "name", ""))
    row.setdefault("candidate_params_json", _skill_params_json(executed_skill))
    row.update(
        {
            "episode_id": episode_id,
            "episode_index": episode_index,
            "seed": seed,
            "skill_idx": skill_idx,
            "attempt_type": attempt_type,
            "attempt_index": attempt_index,
            "parent_skill_idx": parent_skill_idx,
            "original_skill_idx": original_skill_idx,
            "segment_idx": segment_idx,
            "segment_count": segment_count,
            "original_candidate_skill": original_candidate_skill.name,
            "original_candidate_params_json": _skill_params_json(original_candidate_skill),
            "executed_skill": executed_skill.name,
            "executed_params_json": _skill_params_json(executed_skill),
            "fallback_policy": fallback_policy,
            "fallback_executed": int(bool(fallback_executed)),
            "recovery_action": recovery_action,
            "recovery_reason": recovery_reason,
            "recovery_attempt_count": recovery_attempt_count,
            "retry_decision": retry_decision,
            "retry_reject_reason": retry_reject_reason,
            "retry_executed": int(bool(retry_executed)),
            "task_progress_executed": int(bool(task_progress_executed)),
            "final_segment_outcome": final_segment_outcome,
            "recovery_retry_count": recovery_retry_count,
            "result_status": "unverified" if result is None else getattr(result, "status", ""),
            "result_termination_reason": "" if result is None else getattr(result, "termination_reason", ""),
        }
    )
    rows.append(row)


def _skill_params_json(skill: Any) -> str:
    return json.dumps(skill.params(), sort_keys=True)


def _task_progress_executed(*, candidate_skill: Any, executed_skill: Any, attempt_type: str) -> bool:
    if attempt_type in {"candidate_allow", "retry_executed"}:
        return True
    if executed_skill.name == candidate_skill.name and candidate_skill.name != "passive":
        return True
    return False


def _decision_hard_fixed(decision: dict[str, Any]) -> bool:
    text = str(decision.get("hard_contract_fixed_reject", 0)).strip().lower()
    return text in {"1", "1.0", "true", "t", "yes", "y"}


def _run_one_episode(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    gate: Phase5Gate,
    oracle: SafetyOracle,
    method: str,
    seed: int,
    episode_index: int,
    scene_command: dict[str, Any],
    skill_sequence: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_id = f"{method}__seed_{seed}__episode_{episode_index:04d}"
    episode_dir = output_dir / method / f"seed_{seed}" / f"episode_{episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    gate.previous_skill_event = None
    (episode_dir / "scene_command.json").write_text(json.dumps(scene_command, indent=2, sort_keys=True) + "\n")
    (episode_dir / "skill_sequence.json").write_text(
        json.dumps(
            [{"name": skill.name, "params": skill.params()} for skill in skill_sequence],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    executor_args = SimpleNamespace(
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        publish_fps=args.publish_fps,
        dry_run=args.dry_run,
    )
    sim_log_dir = args.sim_log_dir if args.sim_log_dir is not None else episode_dir / "sim_log"
    executor = create_executor(executor_args, episode_dir, f"phase5_{method}_{seed}", sim_log_dir)
    _publish_boundary_idle(executor, args)
    scene_ack = {}
    upright_ok = True
    reset_attempts = 0
    for reset_attempts in range(max(0, int(args.episode_upright_retries)) + 1):
        scene_ack = _request_props(args, method, seed, episode_index, scene_command)
        _publish_idle_for(executor, args, float(args.post_reset_idle_seconds))
        min_wall_time = _float(scene_ack.get("wall_time"), 0.0) if isinstance(scene_ack, dict) else 0.0
        upright_ok = _wait_until_upright(executor, args, sim_log_dir, min_wall_time=min_wall_time)
        if upright_ok:
            break
    if scene_ack:
        (episode_dir / "scene_props_ack.json").write_text(json.dumps(scene_ack, indent=2, sort_keys=True) + "\n")
    scene_props = scene_ack.get("scene_props", scene_ack) if isinstance(scene_ack, dict) else {}
    if not upright_ok and not args.allow_dirty_initial_upright:
        now = time.time()
        error = "initial_upright_failed"
        try:
            _publish_boundary_idle(executor, args)
        finally:
            executor.close()
        row = {
            "method": method,
            "method_display": method_display(method),
            "seed": seed,
            "episode_index": episode_index,
            "episode_id": episode_id,
            "status": "initial_upright_failed",
            "error": error,
            "task_success": 0,
            "fallback_count": 0,
            "unsafe_invocation_count": 0,
            "violation_count": 0,
            "violation_types": json.dumps([], sort_keys=True),
            "completion_time_s": 0.0,
            "episode_dir": str(episode_dir),
            "initial_upright_ok": 0,
            "reset_attempts": reset_attempts + 1,
        }
        (episode_dir / "rollout_summary.json").write_text(
            json.dumps(
                {
                    "run_id": f"phase5_{method}_{seed}",
                    "rollout_id": episode_id,
                    "method": method,
                    "seed": seed,
                    "episode_index": episode_index,
                    "status": "initial_upright_failed",
                    "error": error,
                    "episode_start_wall_time": now,
                    "episode_end_wall_time": now,
                    "initial_upright_ok": False,
                    "reset_attempts": reset_attempts + 1,
                    "scene_props": scene_props,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return row, []
    decisions = []
    start_wall = time.time()
    fallback_count = 0
    unsafe_invocation_count = 0
    status = "completed"
    error = ""
    try:
        _publish_policy_start(executor, args)
        if args.pre_episode_settle_seconds > 0:
            time.sleep(args.pre_episode_settle_seconds)
        post_start_upright_ok = _wait_until_upright(
            executor,
            args,
            sim_log_dir,
            min_wall_time=time.time() - 0.25,
        )
        if not post_start_upright_ok and not args.allow_dirty_initial_upright:
            upright_ok = False
            status = "initial_upright_failed"
            error = "initial_upright_failed_after_policy_start"
        else:
            start_wall = time.time()
            next_skill_idx = 1
            behavior = method_behavior(method)
            fallback_policy = choose_fallback_policy(behavior.fallback_policy, args.fallback_policy)
            use_segmentation = bool(args.segment_long_skills or behavior.segment_long_skills)
            recheck_segments = bool(args.recheck_before_segment or behavior.segment_long_skills)
            for original_skill_idx, skill in enumerate(skill_sequence, start=1):
                if use_segmentation:
                    segments = split_skill(skill, max_segment_duration=args.max_segment_duration)
                else:
                    segments = [skill]
                shared_decision: dict[str, Any] | None = None
                for segment_idx, segment in enumerate(segments, start=1):
                    if shared_decision is None or recheck_segments:
                        decision = gate.decide(
                            method=method,
                            skill=segment,
                            sim_log_dir=sim_log_dir,
                            scene_props=scene_props,
                        )
                        if not recheck_segments:
                            shared_decision = dict(decision)
                    else:
                        decision = dict(shared_decision)
                        decision["candidate_skill"] = segment.name
                        decision["candidate_params_json"] = json.dumps(segment.params(), sort_keys=True)
                    fallback_delta, next_skill_idx = _execute_segment_with_policy(
                        executor=executor,
                        gate=gate,
                        method=method,
                        segment=segment,
                        original_skill_idx=original_skill_idx,
                        segment_idx=segment_idx,
                        segment_count=len(segments),
                        initial_decision=decision,
                        episode_id=episode_id,
                        episode_index=episode_index,
                        seed=seed,
                        sim_log_dir=sim_log_dir,
                        scene_props=scene_props,
                        fallback_policy=fallback_policy,
                        fallback_duration=args.fallback_duration,
                        adaptive_retry_count=args.adaptive_retry_count,
                        next_skill_idx=next_skill_idx,
                        decisions=decisions,
                    )
                    fallback_count += fallback_delta
    except Exception as exc:  # noqa: BLE001 - keep the experiment moving and logged.
        status = "failed"
        error = repr(exc)
    finally:
        _publish_boundary_idle(executor, args)
        executor.close()
    end_wall = time.time()

    sim_state_csv = episode_dir / "sim_log" / "sim_state.csv"
    skill_events_csv = episode_dir / "skill_events.csv"
    task_success = False
    violation_count = 0
    violation_types: list[str] = []
    if status == "initial_upright_failed":
        summary = {
            "run_id": f"phase5_{method}_{seed}",
            "rollout_id": episode_id,
            "method": method,
            "method_display": method_display(method),
            "seed": seed,
            "episode_index": episode_index,
            "status": status,
            "error": error,
            "violation_yes_no": False,
            "violation_count": 0,
            "skill_labels": [],
            "fallback_count": fallback_count,
            "task_success": False,
            "episode_start_wall_time": start_wall,
            "episode_end_wall_time": end_wall,
            "initial_upright_ok": False,
            "reset_attempts": reset_attempts + 1,
            "scene_props": scene_props,
        }
        (episode_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    elif args.dry_run:
        skill_labels = []
        summary = {
            "run_id": f"phase5_{method}_{seed}",
            "rollout_id": episode_id,
            "dry_run": True,
            "violation_yes_no": False,
            "violation_count": 0,
            "skill_labels": [],
            "fallback_count": fallback_count,
        }
        RolloutLogger(episode_dir).write([], summary)
        task_success = status == "completed" and fallback_count == 0
    elif (
        args.sim_log_dir is not None
        and (args.sim_log_dir / "sim_state.csv").exists()
        and skill_events_csv.exists()
    ):
        time.sleep(max(0.0, args.post_horizon + 0.25))
        sliced = _slice_sim_log(
            args.sim_log_dir / "sim_state.csv",
            sim_state_csv,
            start_wall,
            end_wall + args.post_horizon + 0.25,
        )
        violations = oracle.detect(sliced, skill_events_csv)
        skill_labels = oracle.label_skill_calls(
            violations,
            sliced,
            skill_events_csv,
            post_horizon=args.post_horizon,
        )
        unsafe_invocation_count = sum(1 for label in skill_labels if label.get("safe_label") == "unsafe")
        violation_count = len(violations)
        violation_types = sorted({violation.violation_type.value for violation in violations})
        # Fallbacks are reported separately. An online episode succeeds when the
        # sequence completes without unsafe invocations; a safety fallback is not
        # itself a task failure.
        task_success = status == "completed" and unsafe_invocation_count == 0
        summary = build_rollout_summary(
            run_id=f"phase5_{method}_{seed}",
            rollout_id=episode_id,
            violations=violations,
            skill_labels=skill_labels,
            scene_props=scene_props,
            extra={
                "method": method,
                "method_display": method_display(method),
                "seed": seed,
                "episode_index": episode_index,
                "fallback_count": fallback_count,
                "task_success": task_success,
                "episode_start_wall_time": start_wall,
                "episode_end_wall_time": end_wall,
                "initial_upright_ok": upright_ok,
                "reset_attempts": reset_attempts + 1,
            },
        )
        RolloutLogger(episode_dir).write(violations, summary)
    else:
        status = "unverified"
        error = error or "sim_state_csv_or_skill_events_csv_missing"

    row = {
        "method": method,
        "method_display": method_display(method),
        "seed": seed,
        "episode_index": episode_index,
        "episode_id": episode_id,
        "status": status,
        "error": error,
        "task_success": int(bool(task_success)),
        "fallback_count": fallback_count,
        "unsafe_invocation_count": unsafe_invocation_count,
        "violation_count": violation_count,
        "violation_types": json.dumps(violation_types, sort_keys=True),
        "completion_time_s": end_wall - start_wall,
        "episode_dir": str(episode_dir),
        "initial_upright_ok": int(bool(upright_ok)),
        "reset_attempts": reset_attempts + 1,
    }
    return row, decisions


def _episode_skills(seed: int, episode_index: int) -> list[Any]:
    rng = random.Random(seed * 1000003 + episode_index)
    turn_a = rng.choice([-45.0, -30.0, 30.0, 45.0])
    turn_b = rng.choice([-90.0, -60.0, 60.0, 90.0])
    lateral = rng.choice([-0.10, 0.0, 0.10])
    return [
        WalkSkill(vx=0.8, vy=lateral, facing_yaw_deg=rng.choice([-10.0, 0.0, 10.0]), duration=3.0, speed=0.4),
        TurnSkill(face_yaw_deg=turn_a, duration=2.0),
        PassiveSkill(duration=1.5, mode="stop"),
        GestureSkill(
            amplitude=rng.choice([0.25, 0.35, 0.5, 0.7]),
            frequency=rng.choice([0.5, 0.75, 1.0]),
            side=rng.choice(["left", "right", "both"]),
            duration=3.0,
        ),
        WalkSkill(vx=0.7, vy=-lateral, facing_yaw_deg=rng.choice([170.0, 180.0, -170.0]), duration=3.0, speed=0.4),
        TurnSkill(face_yaw_deg=turn_b, duration=2.0),
        PassiveSkill(duration=2.0, mode="wait"),
        WalkSkill(vx=0.6, vy=0.0, facing_yaw_deg=rng.choice([-15.0, 0.0, 15.0]), duration=2.5, speed=0.35),
    ]


def _scene_command(args: argparse.Namespace, seed: int, episode_index: int) -> dict[str, Any]:
    target_bucket = _cycle_value(args.target_bucket_cycle, episode_index, "clean_safe")
    complexity = _cycle_value(args.scene_complexity_cycle, episode_index, "medium")
    return make_phase2_v2_scene_command(
        target_bucket=target_bucket,
        scene_complexity=complexity,
        seed=seed,
        episode_index=episode_index,
    )


def _request_props(
    args: argparse.Namespace,
    method: str,
    seed: int,
    episode_index: int,
    scene_command: dict[str, Any],
) -> dict[str, Any]:
    if args.casa_props_command_file is None or args.casa_props_ack_file is None:
        return scene_command
    command = dict(scene_command)
    command = _offset_velocity_perturbations_for_clean_start(command, args)
    command.update(
        {
            "command_id": f"phase5:{method}:{seed}:{episode_index}:{time.time_ns()}",
            "run_id": f"phase5_{method}_{seed}",
            "episode_id": f"{method}__seed_{seed}__episode_{episode_index:04d}",
            "episode_index": episode_index,
            "seed": seed,
        }
    )
    args.casa_props_command_file.parent.mkdir(parents=True, exist_ok=True)
    args.casa_props_command_file.write_text(json.dumps(command, sort_keys=True) + "\n")
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if args.casa_props_ack_file.exists():
            try:
                ack = json.loads(args.casa_props_ack_file.read_text())
            except json.JSONDecodeError:
                time.sleep(0.05)
                continue
            if ack.get("command_id") == command["command_id"]:
                return ack
        time.sleep(0.05)
    return command


def _offset_velocity_perturbations_for_clean_start(
    command: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    events = command.get("velocity_perturbations")
    if not isinstance(events, list) or not events:
        return command
    offset = (
        max(0.0, float(args.post_reset_idle_seconds))
        + max(0.0, float(args.episode_start_command_seconds))
        + max(0.0, float(args.pre_episode_settle_seconds))
        + max(0.0, float(args.velocity_perturbation_clean_start_margin_seconds))
    )
    shifted_events = []
    for event in events:
        if not isinstance(event, dict):
            continue
        shifted = dict(event)
        shifted["time_s"] = round(_float(event.get("time_s"), 0.0) + offset, 3)
        shifted_events.append(shifted)
    command["velocity_perturbations"] = shifted_events
    command["velocity_perturbation_clean_start_offset_s"] = round(offset, 3)
    return command


def _run_initial_warmup(args: argparse.Namespace, output_dir: Path) -> None:
    if args.dry_run or args.sim_log_dir is None or int(args.initial_warmup_resets) <= 0:
        return
    warmup_dir = output_dir / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    executor_args = SimpleNamespace(
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        publish_fps=args.publish_fps,
        dry_run=args.dry_run,
    )
    executor = create_executor(executor_args, warmup_dir, "phase5_online_warmup", args.sim_log_dir)
    scene_command = make_phase2_v2_scene_command(
        target_bucket="clean_safe",
        scene_complexity="simple",
        seed=0,
        episode_index=0,
    )
    results = []
    try:
        for attempt in range(int(args.initial_warmup_resets)):
            ack = _request_props(args, "warmup", 0, -1 - attempt, scene_command)
            _publish_idle_for(executor, args, float(args.initial_warmup_idle_seconds))
            min_wall_time = _float(ack.get("wall_time"), 0.0) if isinstance(ack, dict) else 0.0
            upright_ok = _wait_until_upright(executor, args, args.sim_log_dir, min_wall_time=min_wall_time)
            results.append(
                {
                    "attempt": attempt + 1,
                    "ack": ack,
                    "upright_ok": upright_ok,
                }
            )
            if upright_ok:
                break
    finally:
        executor.close()
    (warmup_dir / "warmup_summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def _publish_boundary_idle(executor: Any, args: argparse.Namespace) -> None:
    _publish_idle_for(executor, args, float(args.episode_boundary_stop_seconds))


def _publish_idle_for(executor: Any, args: argparse.Namespace, duration: float) -> None:
    if args.dry_run or duration <= 0:
        return
    dt = 1.0 / max(1.0, float(args.publish_fps))
    deadline = time.monotonic() + duration
    first = True
    while first or time.monotonic() < deadline:
        first = False
        executor.publisher.send_command(start=True, stop=False, planner=True)
        executor.publisher.send_planner(
            mode=LOCOMOTION_IDLE,
            movement=(0.0, 0.0, 0.0),
            facing=(1.0, 0.0, 0.0),
        )
        time.sleep(dt)
    executor.publisher.send_command(start=True, stop=False, planner=True)


def _wait_until_upright(
    executor: Any,
    args: argparse.Namespace,
    sim_log_dir: Path | None,
    *,
    min_wall_time: float,
) -> bool:
    if args.dry_run or sim_log_dir is None:
        return True
    sim_state_csv = Path(sim_log_dir) / "sim_state.csv"
    if not sim_state_csv.exists():
        return False
    timeout = max(0.0, float(args.episode_upright_timeout_seconds))
    deadline = time.monotonic() + timeout
    dt = 1.0 / max(1.0, float(args.publish_fps))
    while True:
        rows = _read_recent_sim_rows(sim_state_csv, tail_bytes=2 * 1024 * 1024)
        recent_rows = [
            row
            for row in rows
            if _float(row.get("wall_time"), -float("inf")) >= min_wall_time
        ]
        row = recent_rows[-1] if recent_rows else (rows[-1] if rows else None)
        if row is not None:
            base_z = _float(row.get("base_pos_z"), 0.0)
            fall_flag = _float(row.get("fall_flag"), 0.0)
            torso_roll = abs(_float(row.get("torso_roll"), 0.0))
            torso_pitch = abs(_float(row.get("torso_pitch"), 0.0))
            if (
                base_z >= float(args.episode_upright_min_z)
                and fall_flag < 0.5
                and torso_roll < 0.60
                and torso_pitch < 0.60
            ):
                return True
        if time.monotonic() >= deadline:
            return False
        executor.publisher.send_command(start=True, stop=False, planner=True)
        executor.publisher.send_planner(
            mode=LOCOMOTION_IDLE,
            movement=(0.0, 0.0, 0.0),
            facing=(1.0, 0.0, 0.0),
        )
        time.sleep(dt)


def _publish_policy_start(executor: Any, args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    duration = max(0.0, float(args.episode_start_command_seconds))
    dt = 1.0 / max(1.0, float(args.publish_fps))
    deadline = time.monotonic() + duration
    first = True
    while first or time.monotonic() < deadline:
        first = False
        executor.publisher.send_command(start=True, stop=False, planner=True)
        time.sleep(dt)


def _slice_sim_log(source: Path, destination: Path, start_wall_time: float, end_wall_time: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if start_wall_time <= _float(row.get("wall_time"), -float("inf")) <= end_wall_time
        ]
    with destination.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _read_recent_sim_rows(path: Path, tail_bytes: int = 16 * 1024 * 1024) -> list[dict[str, Any]]:
    """Read only the tail of a live sim log for online gating.

    Live Phase 5 lanes can produce multi-GB sim_state.csv files. The gate only
    needs the most recent pre-window, so reading the full CSV on every skill
    decision can exhaust system memory when many lanes run in parallel.
    """
    size = path.stat().st_size
    if size <= tail_bytes:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))

    with path.open("rb") as file:
        header = file.readline().decode("utf-8", errors="ignore").strip()
        offset = max(0, size - tail_bytes)
        file.seek(offset)
        if offset > 0:
            file.readline()
        text = file.read().decode("utf-8", errors="ignore")

    lines = [line for line in text.splitlines() if line.strip()]
    if not header or not lines:
        return []
    return list(csv.DictReader([header, *lines]))


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"methods": {row["method"]: row for row in _method_summary_rows(rows)}}


def _method_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in _parse_methods(",".join(ONLINE_METHOD_ORDER)):
        method_rows = [row for row in rows if row.get("method") == method]
        if not method_rows:
            continue
        count = len(method_rows)
        output.append(
            {
                "method": method,
                "method_display": method_display(method),
                "episode_count": count,
                "task_success_rate": sum(int(row.get("task_success", 0)) for row in method_rows) / count,
                "unsafe_invocation_count": sum(int(row.get("unsafe_invocation_count", 0)) for row in method_rows),
                "fallback_count": sum(int(row.get("fallback_count", 0)) for row in method_rows),
                "mean_completion_time_s": (
                    sum(float(row.get("completion_time_s", 0.0)) for row in method_rows) / count
                ),
                "failed_or_unverified": sum(
                    1 for row in method_rows if row.get("status") not in {"completed"}
                ),
            }
        )
    return output


def _parse_methods(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [value for value in values if value not in ONLINE_METHOD_ORDER]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}; valid methods are {ONLINE_METHOD_ORDER}")
    return values


def _methods_for_episode(
    methods: list[str],
    args: argparse.Namespace,
    seed: int,
    episode_index: int,
) -> list[str]:
    ordered = list(methods)
    if not args.randomize_method_order:
        return ordered
    rng = random.Random(int(args.method_order_seed) + seed * 1000003 + episode_index)
    rng.shuffle(ordered)
    return ordered


def _parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _cycle_value(raw: str, index: int, default: str) -> str:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values[index % len(values)] if values else default


def _empty_sim_row(now: float) -> dict[str, Any]:
    return {
        "wall_time": now,
        "base_pos_z": 0.75,
        "min_user_distance": 10.0,
        "min_arm_user_distance": 10.0,
        "min_obstacle_distance": 10.0,
    }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
