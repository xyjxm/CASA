"""Executor for CASA skill wrappers."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from gear_sonic.casa.io.episode_log_reader import EpisodeLogReader, SkillEvaluationConfig
from gear_sonic.casa.io.zmq_publisher import ZMQPlannerPublisher
from gear_sonic.casa.loggers.skill_logger import SkillExecutionLogger

from .base import ExecutionResult, Skill, monotonic_and_wall_time


class SkillExecutor:
    def __init__(
        self,
        *,
        publisher: ZMQPlannerPublisher,
        logger: SkillExecutionLogger | None,
        run_id: str,
        publish_fps: float = 10.0,
        sim_log_dir: Path | None = None,
        sim_log_reader: EpisodeLogReader | None = None,
        evaluation_config: SkillEvaluationConfig | None = None,
        dry_run: bool = False,
        default_facing: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        if publish_fps <= 0:
            raise ValueError("publish_fps must be positive")
        self.publisher = publisher
        self.logger = logger
        self.run_id = run_id
        self.publish_fps = publish_fps
        self.publish_dt = 1.0 / publish_fps
        self.sim_log_dir = Path(sim_log_dir) if sim_log_dir is not None else None
        self.sim_log_reader = sim_log_reader
        self.evaluation_config = evaluation_config or SkillEvaluationConfig()
        self.dry_run = dry_run or publisher.dry_run
        self.state: dict[str, Any] = {"last_facing": default_facing}
        self._next_skill_idx = 0

    @classmethod
    def from_paths(
        cls,
        *,
        output_dir: Path,
        sim_log_dir: Path | None,
        run_id: str,
        zmq_host: str,
        zmq_port: int,
        publish_fps: float,
        dry_run: bool,
        evaluation_config: SkillEvaluationConfig | None = None,
    ) -> "SkillExecutor":
        publisher = ZMQPlannerPublisher(zmq_host, zmq_port, dry_run=dry_run)
        logger = SkillExecutionLogger(output_dir)
        reader = None
        if sim_log_dir is not None and (Path(sim_log_dir) / "sim_state.csv").exists():
            reader = EpisodeLogReader.from_log_dir(Path(sim_log_dir), config=evaluation_config)
        return cls(
            publisher=publisher,
            logger=logger,
            run_id=run_id,
            publish_fps=publish_fps,
            sim_log_dir=sim_log_dir,
            sim_log_reader=reader,
            evaluation_config=evaluation_config,
            dry_run=dry_run,
        )

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()
        self.publisher.close()

    def start_policy(self) -> None:
        self.publisher.send_command(start=True, stop=False, planner=True)

    def stop_policy(self) -> None:
        self.publisher.send_command(start=False, stop=True, planner=True)

    def reload_sim_log(self, sim_log_dir: Path | None) -> None:
        if sim_log_dir is None:
            self.sim_log_dir = None
            self.sim_log_reader = None
            return
        self.sim_log_dir = Path(sim_log_dir)
        sim_state_csv = Path(sim_log_dir) / "sim_state.csv"
        self.sim_log_reader = EpisodeLogReader(sim_state_csv, config=self.evaluation_config) if sim_state_csv.exists() else None

    def execute_one(
        self,
        skill: Skill,
        *,
        episode_id: str = "0",
        skill_idx: int | None = None,
    ) -> ExecutionResult:
        if skill_idx is None:
            self._next_skill_idx += 1
            skill_idx = self._next_skill_idx
        start_monotonic, start_wall = monotonic_and_wall_time()
        planner_publishes = 0
        publish_ok = True
        termination_reason = ""

        try:
            if self.dry_run:
                planner_publishes = self._execute_dry_run(skill)
            else:
                planner_publishes = self._execute_realtime(skill)
        except Exception as exc:  # noqa: BLE001 - keep logs for debugging.
            publish_ok = False
            termination_reason = f"publish_exception:{exc}"

        end_monotonic, end_wall = monotonic_and_wall_time()
        actual_duration = end_monotonic - start_monotonic
        status = "dry_run" if self.dry_run else ("published" if publish_ok else "failed")
        downstream_state_observed = False
        evidence: dict[str, Any] = {}

        if self.dry_run:
            downstream_state_observed = False
            evidence = {"dry_run": True}
        elif publish_ok and self.sim_log_dir is not None:
            self.reload_sim_log(self.sim_log_dir)
            if self.sim_log_reader is None:
                status = "unverified"
                termination_reason = "sim_log_unavailable"
                evidence = {"sim_log_available": False, "sim_log_dir": str(self.sim_log_dir)}
            else:
                evaluation = self.sim_log_reader.evaluate(skill.name, skill.params(), start_wall, end_wall)
                status = evaluation.status
                termination_reason = termination_reason or evaluation.termination_reason
                downstream_state_observed = evaluation.downstream_state_observed
                evidence = evaluation.details
        elif publish_ok and self.sim_log_reader is not None:
            evaluation = self.sim_log_reader.evaluate(skill.name, skill.params(), start_wall, end_wall)
            status = evaluation.status
            termination_reason = termination_reason or evaluation.termination_reason
            downstream_state_observed = evaluation.downstream_state_observed
            evidence = evaluation.details
        elif publish_ok:
            status = "unverified"
            termination_reason = "sim_log_unavailable"
            evidence = {"sim_log_available": False}

        result = ExecutionResult(
            run_id=self.run_id,
            episode_id=episode_id,
            skill_idx=skill_idx,
            skill_name=skill.name,
            params=skill.params(),
            start_wall_time=start_wall,
            end_wall_time=end_wall,
            start_monotonic=start_monotonic,
            end_monotonic=end_monotonic,
            estimated_duration=skill.estimated_duration,
            actual_duration=actual_duration,
            planner_publishes=planner_publishes,
            publish_ok=publish_ok,
            downstream_state_observed=downstream_state_observed,
            status=status,
            termination_reason=termination_reason,
            evidence=evidence,
        )
        if self.logger is not None:
            self.logger.record(result)
        return result

    def _execute_dry_run(self, skill: Skill) -> int:
        steps = max(1, int(math.ceil(skill.estimated_duration * self.publish_fps)))
        for index in range(steps):
            command = skill.command_at(index * self.publish_dt, self.state)
            self._send_command(command)
        return steps

    def _execute_realtime(self, skill: Skill) -> int:
        publishes = 0
        phase_start = time.monotonic()
        next_publish = phase_start
        while time.monotonic() - phase_start < skill.estimated_duration:
            elapsed_s = time.monotonic() - phase_start
            command = skill.command_at(elapsed_s, self.state)
            self._send_command(command)
            publishes += 1
            next_publish += self.publish_dt
            time.sleep(max(0.0, next_publish - time.monotonic()))
        return publishes

    def _send_command(self, command) -> None:
        self.publisher.send_planner(
            mode=command.mode,
            movement=command.movement,
            facing=command.facing,
            speed=command.speed,
            height=command.height,
            upper_body_position=command.upper_body_position,
            upper_body_velocity=command.upper_body_velocity,
        )
        self.state["last_facing"] = command.facing
