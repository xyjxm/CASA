from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CASA Phase 2 v2 collection lane.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-plan-csv", type=Path, required=True)
    parser.add_argument("--lane-index", type=int, default=0)
    parser.add_argument("--num-lanes", type=int, default=1)
    parser.add_argument("--domain-id", type=int, default=60)
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-out-port", type=int, default=5557)
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--duration-seconds", type=float, default=18.0)
    parser.add_argument("--min-skill-duration", type=float, default=2.5)
    parser.add_argument("--max-skill-duration", type=float, default=4.5)
    parser.add_argument("--stabilize-seconds", type=float, default=1.25)
    parser.add_argument("--pre-rollout-settle-seconds", type=float, default=2.0)
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--startup-seconds", type=float, default=25.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--skill-schedule", default="safe_balanced")
    parser.add_argument("--sim-cpus", default="")
    parser.add_argument("--deploy-cpus", default="")
    parser.add_argument("--cuda-visible-devices", default="")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lane_plan = write_lane_plan(args.scene_plan_csv.resolve(), output_dir / "lane_scene_plan.csv", args.lane_index, args.num_lanes)
    if not lane_plan:
        raise SystemExit("lane plan is empty")

    live_sim_log = output_dir / "live_sim_log"
    command_file = output_dir / "props_command.json"
    ack_file = output_dir / "props_ack.json"
    logs_dir = output_dir / "process_logs"
    deploy_log_dir = output_dir / "deploy"
    logs_dir.mkdir(exist_ok=True)
    deploy_log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    sim_log = (logs_dir / "sim.log").open("w")
    deploy_log = (logs_dir / "deploy.log").open("w")
    sim_proc = None
    deploy_proc = None
    try:
        sim_cmd = maybe_taskset(
            args.sim_cpus,
            [
                str(REPO_ROOT / ".venv_sim/bin/python"),
                "-u",
                str(REPO_ROOT / "gear_sonic/scripts/run_sim_loop.py"),
                "--no-enable-onscreen",
                "--drop-on-start",
                "--disable-reset-on-fall",
                "--sim-frequency",
                "200",
                "--domain-id",
                str(args.domain_id),
                "--fall-log-interval-seconds",
                "30",
                "--fall-stats",
                "--sim-timing-log-interval-seconds",
                "0",
                "--episode-log-dir",
                str(live_sim_log),
                "--episode-log-fps",
                "50",
                "--episode-name",
                args.run_id,
                "--robot-scene",
                "gear_sonic/data/robot_model/model_data/g1/scene_casa_v1.xml",
                "--casa-props-config",
                "gear_sonic/casa/scene/props.yaml",
                "--casa-props-command-file",
                str(command_file),
                "--casa-props-ack-file",
                str(ack_file),
            ],
        )
        sim_proc = subprocess.Popen(sim_cmd, cwd=REPO_ROOT, env=env, stdout=sim_log, stderr=subprocess.STDOUT)
        time.sleep(3.0)
        if sim_proc.poll() is not None:
            raise RuntimeError(f"sim exited early with code {sim_proc.returncode}; see {logs_dir / 'sim.log'}")

        deploy_cmd = maybe_taskset(
            args.deploy_cpus,
            [
                "bash",
                "-lc",
                "source ../scripts/setup_no_root_env.sh >/dev/null 2>&1; "
                "./target/release/g1_deploy_onnx_ref "
                "lo policy/release/model_decoder.onnx reference/example/ "
                "--obs-config policy/release/observation_config.yaml "
                "--encoder-file policy/release/model_encoder.onnx "
                "--planner-file planner/target_vel/V2/planner_sonic.onnx "
                f"--input-type zmq_manager --output-type all --zmq-host localhost --zmq-port {args.zmq_port} "
                f"--domain-id {args.domain_id} --zmq-out-port {args.zmq_out_port} "
                "--disable-crc-check --quiet --timing-log-interval-seconds 10 "
                "--command-publish-frequency 100 --enable-csv-logs "
                f"--logs-dir {deploy_log_dir}",
            ],
        )
        deploy_proc = subprocess.Popen(deploy_cmd, cwd=DEPLOY_ROOT, env=env, stdout=deploy_log, stderr=subprocess.STDOUT)
        time.sleep(args.startup_seconds)
        if deploy_proc.poll() is not None:
            raise RuntimeError(f"deploy exited early with code {deploy_proc.returncode}; see {logs_dir / 'deploy.log'}")
        wait_for_deploy_ready(
            deploy_log_path=logs_dir / "deploy.log",
            proc=deploy_proc,
            timeout_s=args.startup_timeout_seconds,
        )

        batch_cmd = [
            str(REPO_ROOT / ".venv_sim/bin/python"),
            "-u",
            str(REPO_ROOT / "gear_sonic/scripts/casa_run_oracle_eval_batch.py"),
            "--run-id",
            args.run_id,
            "--output-dir",
            str(output_dir),
            "--sim-log-dir",
            str(live_sim_log),
            "--rollout-count",
            str(len(lane_plan)),
            "--duration-seconds",
            str(args.duration_seconds),
            "--min-skill-duration",
            str(args.min_skill_duration),
            "--max-skill-duration",
            str(args.max_skill_duration),
            "--stabilize-seconds",
            str(args.stabilize_seconds),
            "--pre-rollout-settle-seconds",
            str(args.pre_rollout_settle_seconds),
            "--publish-fps",
            str(args.publish_fps),
            "--zmq-port",
            str(args.zmq_port),
            "--seed",
            str(args.seed + args.lane_index * 1000),
            "--skill-schedule",
            args.skill_schedule,
            "--phase2-v2",
            "--scene-plan-csv",
            str(output_dir / "lane_scene_plan.csv"),
            "--casa-props-command-file",
            str(command_file),
            "--casa-props-ack-file",
            str(ack_file),
        ]
        batch_log_path = logs_dir / "batch.log"
        with batch_log_path.open("w") as batch_log:
            result = subprocess.run(batch_cmd, cwd=REPO_ROOT, env=env, stdout=batch_log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"batch failed with code {result.returncode}; see {batch_log_path}")
    finally:
        terminate_process(deploy_proc)
        terminate_process(sim_proc)
        close_log(deploy_log)
        close_log(sim_log)

    payload = {
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "lane_plan_rows": len(lane_plan),
        "domain_id": args.domain_id,
        "zmq_port": args.zmq_port,
        "review_csv": str(output_dir / "review/rollouts.csv"),
    }
    (output_dir / "lane_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_lane_plan(source_csv: Path, output_csv: Path, lane_index: int, num_lanes: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(source_csv.open(newline="")))
    selected = [row for index, row in enumerate(rows) if index % num_lanes == lane_index]
    for index, row in enumerate(selected):
        row["episode_index"] = str(index)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode_index", "target_bucket", "scene_complexity"])
        writer.writeheader()
        writer.writerows(selected)
    return selected


def maybe_taskset(cpus: str, cmd: list[str]) -> list[str]:
    if not cpus:
        return cmd
    return ["taskset", "-c", cpus, *cmd]


def terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def wait_for_deploy_ready(deploy_log_path: Path, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    ready_needles = ["Init Done", "Planner initialized successfully"]
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"deploy exited before ready with code {proc.returncode}; see {deploy_log_path}")
        text = ""
        try:
            text = deploy_log_path.read_text(errors="ignore")
        except OSError:
            pass
        if any(needle in text for needle in ready_needles):
            # Give the control stack one more breath after initialization logs flush.
            time.sleep(2.0)
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for deploy readiness in {deploy_log_path}")


def close_log(file: TextIO) -> None:
    try:
        file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
