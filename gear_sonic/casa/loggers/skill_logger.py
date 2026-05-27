"""Skill execution logger for CASA."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from gear_sonic.casa.skills.base import ExecutionResult


class SkillExecutionLogger:
    fieldnames = [
        "run_id",
        "episode_id",
        "skill_idx",
        "skill_name",
        "params_json",
        "start_wall_time",
        "end_wall_time",
        "start_monotonic",
        "end_monotonic",
        "estimated_duration",
        "actual_duration",
        "planner_publishes",
        "publish_ok",
        "downstream_state_observed",
        "status",
        "termination_reason",
        "evidence_json",
    ]

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "skill_events.jsonl"
        self.csv_path = self.output_dir / "skill_events.csv"
        self._jsonl_file = self.jsonl_path.open("w")
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
        self._csv_writer.writeheader()

    def close(self) -> None:
        self._jsonl_file.close()
        self._csv_file.close()

    def record(self, result: ExecutionResult) -> None:
        data = {
            "run_id": result.run_id,
            "episode_id": result.episode_id,
            "skill_idx": result.skill_idx,
            "skill_name": result.skill_name,
            "params": result.params,
            "start_wall_time": result.start_wall_time,
            "end_wall_time": result.end_wall_time,
            "start_monotonic": result.start_monotonic,
            "end_monotonic": result.end_monotonic,
            "estimated_duration": result.estimated_duration,
            "actual_duration": result.actual_duration,
            "planner_publishes": result.planner_publishes,
            "publish_ok": result.publish_ok,
            "downstream_state_observed": result.downstream_state_observed,
            "status": result.status,
            "termination_reason": result.termination_reason,
            "evidence": result.evidence,
        }
        self._jsonl_file.write(json.dumps(data, sort_keys=True) + "\n")
        self._jsonl_file.flush()
        self._csv_writer.writerow(
            {
                "run_id": result.run_id,
                "episode_id": result.episode_id,
                "skill_idx": result.skill_idx,
                "skill_name": result.skill_name,
                "params_json": json.dumps(result.params, sort_keys=True),
                "start_wall_time": f"{result.start_wall_time:.6f}",
                "end_wall_time": f"{result.end_wall_time:.6f}",
                "start_monotonic": f"{result.start_monotonic:.6f}",
                "end_monotonic": f"{result.end_monotonic:.6f}",
                "estimated_duration": f"{result.estimated_duration:.6f}",
                "actual_duration": f"{result.actual_duration:.6f}",
                "planner_publishes": result.planner_publishes,
                "publish_ok": int(result.publish_ok),
                "downstream_state_observed": int(result.downstream_state_observed),
                "status": result.status,
                "termination_reason": result.termination_reason,
                "evidence_json": json.dumps(result.evidence, sort_keys=True),
            }
        )
        self._csv_file.flush()

    def __enter__(self) -> "SkillExecutionLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

