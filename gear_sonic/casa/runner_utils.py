"""Shared helpers for CASA runner scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gear_sonic.casa.io.episode_log_reader import SkillEvaluationConfig
from gear_sonic.casa.skills.base import ExecutionResult
from gear_sonic.casa.skills.executor import SkillExecutor


def resolve_run_id(run_id: str | None) -> str:
    return run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_output_dir(output_dir: Path | None, run_id: str) -> Path:
    return output_dir or Path("outputs") / "casa" / "phase0_sanity" / run_id


def resolve_sim_log_dir(sim_log_dir: Path | None, output_dir: Path) -> Path:
    return sim_log_dir or output_dir / "sim_log"


def build_evaluation_config(args: Any) -> SkillEvaluationConfig:
    default = SkillEvaluationConfig()
    return SkillEvaluationConfig(
        walk_speed_threshold=getattr(args, "walk_speed_threshold", default.walk_speed_threshold),
        walk_min_ratio=getattr(args, "walk_min_ratio", default.walk_min_ratio),
        passive_speed_threshold=getattr(args, "passive_speed_threshold", default.passive_speed_threshold),
        passive_min_ratio=getattr(args, "passive_min_ratio", default.passive_min_ratio),
        passive_tail_fraction=getattr(args, "passive_tail_fraction", default.passive_tail_fraction),
        turn_yaw_threshold_rad=getattr(args, "turn_yaw_threshold_rad", default.turn_yaw_threshold_rad),
        gesture_min_rom=getattr(args, "gesture_min_rom", default.gesture_min_rom),
        middle_trim_fraction=getattr(args, "middle_trim_fraction", default.middle_trim_fraction),
    )


def create_executor(args: Any, output_dir: Path, run_id: str, sim_log_dir: Path) -> SkillExecutor:
    evaluation_config = build_evaluation_config(args)
    return SkillExecutor.from_paths(
        output_dir=output_dir,
        sim_log_dir=sim_log_dir,
        run_id=run_id,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        publish_fps=args.publish_fps,
        dry_run=args.dry_run,
        evaluation_config=evaluation_config,
    )


def result_is_success(result: ExecutionResult, *, count_dry_run_as_success: bool = False) -> bool:
    if result.status == "success":
        return True
    if count_dry_run_as_success and result.status == "dry_run":
        return True
    return False


def summarize_results(results: list[ExecutionResult], *, dry_run: bool = False) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for result in results if result_is_success(result, count_dry_run_as_success=dry_run))
    return {
        "total": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total if total else 0.0,
        "statuses": _count_by(results, "status"),
        "termination_reasons": _count_by(results, "termination_reason"),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def _count_by(results: list[ExecutionResult], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        value = str(getattr(result, attr) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts
