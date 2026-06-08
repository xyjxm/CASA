"""Launch one live sim/deploy lane for the CASA Phase 5 online experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--phase4-root",
        type=Path,
        default=Path("outputs/casa/phase4_dataset_v1_strict_50k_20260522"),
    )
    parser.add_argument(
        "--phase5-root",
        type=Path,
        default=Path("outputs/casa/phase5_conformal_baselines_20260522"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--seeds", default="1234,1235,1236,1237,1238")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--domain-id", type=int, default=150)
    parser.add_argument("--zmq-port", type=int, default=6200)
    parser.add_argument("--zmq-out-port", type=int, default=6201)
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--startup-seconds", type=float, default=25.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--pre-episode-settle-seconds", type=float, default=1.0)
    parser.add_argument("--episode-boundary-stop-seconds", type=float, default=0.75)
    parser.add_argument("--post-reset-idle-seconds", type=float, default=1.5)
    parser.add_argument("--episode-upright-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--episode-upright-min-z", type=float, default=0.65)
    parser.add_argument("--episode-upright-retries", type=int, default=2)
    parser.add_argument("--initial-warmup-resets", type=int, default=1)
    parser.add_argument("--initial-warmup-idle-seconds", type=float, default=4.0)
    parser.add_argument("--episode-start-command-seconds", type=float, default=0.75)
    parser.add_argument("--fallback-duration", type=float, default=1.0)
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
    parser.add_argument("--hybrid-config", type=Path)
    parser.add_argument("--randomize-method-order", action="store_true")
    parser.add_argument("--method-order-seed", type=int, default=0)
    parser.add_argument("--sim-cpus", default="")
    parser.add_argument("--deploy-cpus", default="")
    parser.add_argument("--cuda-visible-devices", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    live_sim_log = output_dir / "live_sim_log"
    command_file = output_dir / "props_command.json"
    ack_file = output_dir / "props_ack.json"
    logs_dir = output_dir / "process_logs"
    deploy_log_dir = output_dir / "deploy"
    logs_dir.mkdir(exist_ok=True)
    deploy_log_dir.mkdir(exist_ok=True)

    env = _env(args)
    sim_log = (logs_dir / "sim.log").open("w")
    deploy_log = (logs_dir / "deploy.log").open("w")
    sim_proc = None
    deploy_proc = None
    try:
        sim_proc = subprocess.Popen(
            _sim_cmd(args, live_sim_log, command_file, ack_file),
            cwd=REPO_ROOT,
            env=env,
            stdout=sim_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(3.0)
        if sim_proc.poll() is not None:
            raise RuntimeError(f"sim exited early with code {sim_proc.returncode}; see {logs_dir / 'sim.log'}")

        deploy_proc = subprocess.Popen(
            _deploy_cmd(args, deploy_log_dir),
            cwd=DEPLOY_ROOT,
            env=env,
            stdout=deploy_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(args.startup_seconds)
        if deploy_proc.poll() is not None:
            raise RuntimeError(
                f"deploy exited early with code {deploy_proc.returncode}; see {logs_dir / 'deploy.log'}"
            )
        wait_for_deploy_ready(logs_dir / "deploy.log", deploy_proc, args.startup_timeout_seconds)

        online_cmd = [
            str(REPO_ROOT / ".venv_sim/bin/python"),
            "-u",
            str(REPO_ROOT / "gear_sonic/scripts/casa_run_phase5_online_experiment.py"),
            "--phase4-root",
            str(args.phase4_root),
            "--phase5-root",
            str(args.phase5_root),
            "--output-dir",
            str(output_dir / "online"),
            "--methods",
            args.methods,
            "--seeds",
            args.seeds,
            "--episode-start",
            str(args.episode_start),
            "--episodes-per-seed",
            str(args.episodes_per_seed),
            "--sim-log-dir",
            str(live_sim_log),
            "--zmq-port",
            str(args.zmq_port),
            "--publish-fps",
            str(args.publish_fps),
            "--pre-episode-settle-seconds",
            str(args.pre_episode_settle_seconds),
            "--episode-boundary-stop-seconds",
            str(args.episode_boundary_stop_seconds),
            "--post-reset-idle-seconds",
            str(args.post_reset_idle_seconds),
            "--episode-upright-timeout-seconds",
            str(args.episode_upright_timeout_seconds),
            "--episode-upright-min-z",
            str(args.episode_upright_min_z),
            "--episode-upright-retries",
            str(args.episode_upright_retries),
            "--initial-warmup-resets",
            str(args.initial_warmup_resets),
            "--initial-warmup-idle-seconds",
            str(args.initial_warmup_idle_seconds),
            "--episode-start-command-seconds",
            str(args.episode_start_command_seconds),
            "--continue-after-initial-upright-failure",
            "--fallback-duration",
            str(args.fallback_duration),
            "--fallback-policy",
            args.fallback_policy,
            "--adaptive-retry-count",
            str(args.adaptive_retry_count),
            "--max-segment-duration",
            str(args.max_segment_duration),
            "--threshold-scale-global",
            str(args.threshold_scale_global),
            "--threshold-scale-by-skill",
            args.threshold_scale_by_skill,
            "--method-order-seed",
            str(args.method_order_seed),
            "--casa-props-command-file",
            str(command_file),
            "--casa-props-ack-file",
            str(ack_file),
        ]
        if args.hybrid_config is not None:
            online_cmd.extend(["--hybrid-config", str(args.hybrid_config)])
        if args.segment_long_skills:
            online_cmd.append("--segment-long-skills")
        if args.recheck_before_segment:
            online_cmd.append("--recheck-before-segment")
        if args.randomize_method_order:
            online_cmd.append("--randomize-method-order")
        online_log_path = logs_dir / "online.log"
        with online_log_path.open("w") as online_log:
            result = subprocess.run(
                online_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=online_log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise RuntimeError(f"online experiment failed with code {result.returncode}; see {online_log_path}")
    finally:
        terminate_process(deploy_proc)
        terminate_process(sim_proc)
        close_log(deploy_log)
        close_log(sim_log)

    summary = {
        "run_id": args.run_id,
        "phase4_root": str(args.phase4_root),
        "phase5_root": str(args.phase5_root),
        "output_dir": str(output_dir),
        "online_dir": str(output_dir / "online"),
        "methods": [item.strip() for item in args.methods.split(",") if item.strip()],
        "seeds": [int(item.strip()) for item in args.seeds.split(",") if item.strip()],
        "episode_start": args.episode_start,
        "episode_end_exclusive": args.episode_start + args.episodes_per_seed,
        "episodes_per_seed": args.episodes_per_seed,
        "domain_id": args.domain_id,
        "zmq_port": args.zmq_port,
        "zmq_out_port": args.zmq_out_port,
        "policy_checkpoint_dir": str(DEPLOY_ROOT / "policy/release"),
        "sim_scene": "gear_sonic/data/robot_model/model_data/g1/scene_casa_v1.xml",
        "online_policy": {
            "fallback_policy": args.fallback_policy,
            "adaptive_retry_count": args.adaptive_retry_count,
            "segment_long_skills": bool(args.segment_long_skills),
            "max_segment_duration": args.max_segment_duration,
            "recheck_before_segment": bool(args.recheck_before_segment),
            "threshold_scale_global": args.threshold_scale_global,
            "threshold_scale_by_skill": args.threshold_scale_by_skill,
            "hybrid_config": "" if args.hybrid_config is None else str(args.hybrid_config),
            "randomize_method_order": bool(args.randomize_method_order),
            "method_order_seed": args.method_order_seed,
        },
        "expected_artifacts": [
            "online/online_episode_results.csv",
            "online/gate_decisions.csv",
            "online/method_summary.json",
            "lane_summary.json",
        ],
    }
    (output_dir / "lane_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _env(args: argparse.Namespace) -> dict[str, str]:
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
    return env


def _sim_cmd(args: argparse.Namespace, live_sim_log: Path, command_file: Path, ack_file: Path) -> list[str]:
    cmd = [
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
    ]
    return maybe_taskset(args.sim_cpus, cmd)


def _deploy_cmd(args: argparse.Namespace, deploy_log_dir: Path) -> list[str]:
    cmd = [
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
    ]
    return maybe_taskset(args.deploy_cpus, cmd)


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
        try:
            text = deploy_log_path.read_text(errors="ignore")
        except OSError:
            text = ""
        if any(needle in text for needle in ready_needles):
            time.sleep(2.0)
            return
        time.sleep(1.0)
    # The deploy binary writes to a regular file here, so its stdout can be
    # block-buffered under load.  If it is still alive and has initialized the
    # state logger, avoid killing an otherwise-ready lane only because the
    # readiness line has not flushed yet.
    metadata_path = deploy_log_path.parent.parent / "deploy" / "metadata.json"
    if proc.poll() is None and metadata_path.exists() and metadata_path.stat().st_size > 0:
        time.sleep(2.0)
        return
    raise TimeoutError(f"Timed out waiting for deploy readiness in {deploy_log_path}")


def close_log(file: TextIO) -> None:
    try:
        file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
