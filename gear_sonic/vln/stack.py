"""Process launcher for online SONIC MuJoCo VLN runs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SonicStack:
    run_dir: Path
    sim_process: subprocess.Popen | None
    deploy_process: subprocess.Popen | None
    sim_log_dir: Path
    deploy_log_dir: Path
    camera_port: int
    zmq_port: int

    def close(self) -> None:
        for process in [self.deploy_process, self.sim_process]:
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + 8.0
        for process in [self.deploy_process, self.sim_process]:
            if process is None:
                continue
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w")


def launch_sonic_stack(
    *,
    run_dir: Path,
    robot_scene: Path | str | None = None,
    camera_port: int = 5555,
    zmq_port: int = 5556,
    domain_id: int = 0,
    episode_name: str = "vln_sonic_navid",
    sim_frequency: int = 200,
    image_publish_fps: int = 10,
    episode_log_fps: int = 50,
    offscreen_camera: str = "head_camera",
    offscreen_camera_width: int = 480,
    offscreen_camera_height: int = 360,
    drop_on_start: bool = True,
    drop_after_seconds: float | None = None,
    disable_reset_on_fall: bool = False,
) -> SonicStack:
    run_dir = Path(run_dir)
    sim_log_dir = run_dir / "sim"
    deploy_log_dir = run_dir / "deploy"
    sim_log_dir.mkdir(parents=True, exist_ok=True)
    deploy_log_dir.mkdir(parents=True, exist_ok=True)

    sim_env = os.environ.copy()
    sim_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    sim_cmd = [
        str(REPO_ROOT / ".venv_sim/bin/python"),
        "-u",
        str(REPO_ROOT / "gear_sonic/scripts/run_sim_loop.py"),
        "--no-enable-onscreen",
        "--enable-offscreen",
        "--enable-image-publish",
        "--camera-port",
        str(camera_port),
        "--image-publish-fps",
        str(image_publish_fps),
        "--offscreen-camera",
        offscreen_camera,
        "--offscreen-camera-width",
        str(offscreen_camera_width),
        "--offscreen-camera-height",
        str(offscreen_camera_height),
        "--sim-frequency",
        str(sim_frequency),
        "--fall-log-interval-seconds",
        "30",
        "--fall-stats",
        "--sim-timing-log-interval-seconds",
        "10",
        "--episode-log-dir",
        str(sim_log_dir),
        "--episode-log-fps",
        str(episode_log_fps),
        "--episode-name",
        episode_name,
        "--domain-id",
        str(domain_id),
    ]
    if drop_on_start:
        sim_cmd.append("--drop-on-start")
    if drop_after_seconds is not None:
        sim_cmd.extend(["--drop-after-seconds", str(drop_after_seconds)])
    if disable_reset_on_fall:
        sim_cmd.append("--disable-reset-on-fall")
    if robot_scene is not None:
        sim_cmd.extend(["--robot-scene", str(robot_scene)])
    sim_out = _open_log(run_dir / "logs/sim_stdout.log")
    sim_process = subprocess.Popen(
        sim_cmd,
        cwd=REPO_ROOT,
        env=sim_env,
        stdout=sim_out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deploy_cmd = (
        "source ../scripts/setup_no_root_env.sh && "
        "printf '\\n' | bash deploy.sh "
        f"--input-type zmq_manager --output-type all --zmq-host localhost --zmq-port {zmq_port} "
        f"--domain-id {domain_id} sim --quiet --timing-log-interval-seconds 10 "
        "--command-publish-frequency 100 --enable-csv-logs "
        f"--logs-dir '{deploy_log_dir}' "
        f"--record-input-file '{run_dir / 'input.csv'}' "
        f"--target-motion-logfile '{run_dir / 'target_motion.csv'}' "
        f"--policy-input-logfile '{run_dir / 'policy_input.csv'}'"
    )
    deploy_env = os.environ.copy()
    deploy_env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    deploy_out = _open_log(run_dir / "logs/deploy_stdout.log")
    deploy_process = subprocess.Popen(
        ["bash", "-lc", deploy_cmd],
        cwd=REPO_ROOT / "gear_sonic_deploy",
        env=deploy_env,
        stdout=deploy_out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    return SonicStack(
        run_dir=run_dir,
        sim_process=sim_process,
        deploy_process=deploy_process,
        sim_log_dir=sim_log_dir,
        deploy_log_dir=deploy_log_dir,
        camera_port=camera_port,
        zmq_port=zmq_port,
    )


def wait_for_file(path: Path, *, timeout_s: float, min_size: int = 1) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size >= min_size:
            return True
        time.sleep(0.25)
    return False


def wait_for_log_contains(path: Path, needles: list[str], *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text(errors="ignore")
            if any(needle in text for needle in needles):
                return True
        time.sleep(0.5)
    return False
