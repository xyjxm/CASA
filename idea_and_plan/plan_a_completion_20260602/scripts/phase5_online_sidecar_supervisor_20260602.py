#!/usr/bin/env python3
"""Conservative sidecar launcher for the CASA Phase 5 online run.

The main low-memory runner builds its job list once per sweep.  To avoid
duplicating future main-lane work, this supervisor only retries chunks that the
current sweep has already started and then finished or failed.
"""

from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import time

WORKTREE = Path("/tmp/casa_upload_worktree")
ROOT = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602"
)
PHASE4_ROOT = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase4_dataset_v1_strict_50k_20260522"
)
PHASE5_ROOT = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/hc_filtered_online_calibration_20260601"
)
LOWMEM_LOG = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602.lowmem.log"
)
LOG_PATH = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602.sidecar_supervisor.log"
)
PID_PATH = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602.sidecar_supervisor.pid"
)

METHODS = ["sonic_only", "hard_contract", "raw_critic_0p5", "global_conformal", "casa_a_per_skill"]
SEEDS = [1234, 1235, 1236, 1237, 1238]
EXPECTED = 2500
MAX_EPISODES_PER_SIDECAR = 15
MAX_ACTIVE_SIDECARS = 3
DOMAINS = list(range(226, 233))
PORTS = list(range(9312, 9326, 2))
POLL_SECONDS = 90


def stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as file:
        file.write(f"[{stamp()}] {message}\n")


def proc_cmdlines() -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except OSError:
            continue
        if cmdline:
            items.append((int(entry.name), cmdline))
    return items


def main_running(cmdlines: list[tuple[int, str]]) -> bool:
    script = "gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py"
    root = str(ROOT)
    return any(script in cmdline and root in cmdline for _, cmdline in cmdlines)


def active_sidecars(cmdlines: list[tuple[int, str]]) -> list[int]:
    root = str(ROOT)
    return [
        pid
        for pid, cmdline in cmdlines
        if "gear_sonic/scripts/casa_run_phase5_online_lane.py" in cmdline
        and root in cmdline
        and "/lane_sidecar_" in cmdline
    ]


def active_main_lanes(cmdlines: list[tuple[int, str]]) -> set[str]:
    root = str(ROOT)
    active: set[str] = set()
    for _, cmdline in cmdlines:
        if "gear_sonic/scripts/casa_run_phase5_online_lane.py" not in cmdline or root not in cmdline:
            continue
        match = re.search(r"/(lane_v2_[^/ ]+)", cmdline)
        if match:
            active.add(match.group(1))
    return active


def active_episode_keys(cmdlines: list[tuple[int, str]]) -> set[tuple[str, int, int]]:
    root = str(ROOT)
    active: set[tuple[str, int, int]] = set()
    for _, cmdline in cmdlines:
        if "gear_sonic/scripts/casa_run_phase5_online_lane.py" not in cmdline or root not in cmdline:
            continue
        method = re.search(r"--methods\s+(\S+)", cmdline)
        seed = re.search(r"--seeds\s+(\d+)", cmdline)
        start = re.search(r"--episode-start\s+(\d+)", cmdline)
        count = re.search(r"--episodes-per-seed\s+(\d+)", cmdline)
        if not (method and seed and start and count):
            continue
        method_name = method.group(1)
        seed_value = int(seed.group(1))
        start_value = int(start.group(1))
        count_value = int(count.group(1))
        for episode in range(start_value, start_value + count_value):
            active.add((method_name, seed_value, episode))
    return active


def used_domains_ports(cmdlines: list[tuple[int, str]]) -> tuple[set[int], set[int]]:
    domains: set[int] = set()
    ports: set[int] = set()
    for _, cmdline in cmdlines:
        if str(ROOT) not in cmdline:
            continue
        domain = re.search(r"--domain-id\s+(\d+)", cmdline)
        port = re.search(r"--zmq-port\s+(\d+)", cmdline)
        out_port = re.search(r"--zmq-out-port\s+(\d+)", cmdline)
        if domain:
            domains.add(int(domain.group(1)))
        if port:
            ports.add(int(port.group(1)))
        if out_port:
            ports.add(int(out_port.group(1)))
    return domains, ports


def completed_map() -> dict[tuple[str, int], set[int]]:
    done = {(method, seed): set() for method in METHODS for seed in SEEDS}
    for csv_path in ROOT.glob("lane_*/online/online_episode_results.csv"):
        try:
            with csv_path.open(newline="") as file:
                for row in csv.DictReader(file):
                    if row.get("status") != "completed":
                        continue
                    if row.get("initial_upright_ok") not in {"1", "1.0", "True", "true"}:
                        continue
                    key = (row.get("method", ""), int(row.get("seed", "-1")))
                    if key in done:
                        done[key].add(int(row.get("episode_index", "-1")))
        except Exception as exc:
            log(f"skip unreadable csv {csv_path}: {exc}")
    return done


def completed_count(done: dict[tuple[str, int], set[int]]) -> int:
    return sum(len(values) for values in done.values())


def latest_sweep_jobs() -> dict[str, dict[str, object]]:
    if not LOWMEM_LOG.exists():
        return {}
    lines = LOWMEM_LOG.read_text(errors="ignore").splitlines()
    start = 0
    for index, line in enumerate(lines):
        if "[lowmem] sweep=" in line:
            start = index
    pattern = re.compile(
        r"\[lowmem\] (started|finished|FAILED) "
        r"(lane_v2_(?P<method>.+?)__seed_(?P<seed>\d+)__ep_(?P<start>\d+)_(?P<end>\d+))"
    )
    jobs: dict[str, dict[str, object]] = {}
    for line in lines[start:]:
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(2)
        status = match.group(1)
        jobs.setdefault(
            name,
            {
                "name": name,
                "method": match.group("method"),
                "seed": int(match.group("seed")),
                "start": int(match.group("start")),
                "end": int(match.group("end")),
                "status": "started",
            },
        )
        jobs[name]["status"] = status
    return jobs


def contiguous_ranges(values: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if not values:
        return ranges
    start = values[0]
    last = values[0]
    for value in values[1:]:
        if value == last + 1:
            last = value
            continue
        ranges.append((start, last + 1))
        start = last = value
    ranges.append((start, last + 1))
    return ranges


def choose_candidate(
    done: dict[tuple[str, int], set[int]],
    active_lanes: set[str],
    active_episodes: set[tuple[str, int, int]],
) -> dict[str, object] | None:
    candidates: list[tuple[int, int, str, dict[str, object]]] = []
    for name, job in latest_sweep_jobs().items():
        if job.get("status") not in {"finished", "FAILED"} or name in active_lanes:
            continue
        method = str(job["method"])
        seed = int(job["seed"])
        start = int(job["start"])
        end = int(job["end"])
        missing = [
            episode
            for episode in range(start, end)
            if episode not in done[(method, seed)] and (method, seed, episode) not in active_episodes
        ]
        for range_start, range_end in contiguous_ranges(missing):
            capped_end = min(range_end, range_start + MAX_EPISODES_PER_SIDECAR)
            length = capped_end - range_start
            if length <= 0:
                continue
            item = {
                "source_name": name,
                "method": method,
                "seed": seed,
                "start": range_start,
                "end": capped_end,
                "missing_in_source": len(missing),
            }
            candidates.append((length, int(job["start"]), name, item))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][3]


def choose_domain_port(
    cmdlines: list[tuple[int, str]],
    reserved_domains: set[int] | None = None,
    reserved_ports: set[int] | None = None,
) -> tuple[int, int] | None:
    used_domains, used_ports = used_domains_ports(cmdlines)
    if reserved_domains:
        used_domains.update(reserved_domains)
    if reserved_ports:
        used_ports.update(reserved_ports)
    for domain in DOMAINS:
        if domain in used_domains:
            continue
        for port in PORTS:
            if port not in used_ports and port + 1 not in used_ports:
                return domain, port
    return None


def launch(candidate: dict[str, object], domain: int, port: int) -> subprocess.Popen[bytes]:
    method = str(candidate["method"])
    seed = int(candidate["seed"])
    start = int(candidate["start"])
    end = int(candidate["end"])
    run_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    lane_name = f"lane_sidecar_{method}__seed_{seed}__ep_{start:03d}_{end:03d}_{run_stamp}"
    output_dir = ROOT / lane_name
    stdout_path = ROOT / f"{lane_name}.stdout.log"
    cmd = [
        ".venv_sim/bin/python",
        "-u",
        "gear_sonic/scripts/casa_run_phase5_online_lane.py",
        "--run-id",
        f"phase5_online_sidecar_{method}_{seed}_{start:03d}_{end:03d}_{run_stamp}",
        "--phase4-root",
        str(PHASE4_ROOT),
        "--phase5-root",
        str(PHASE5_ROOT),
        "--output-dir",
        str(output_dir),
        "--methods",
        method,
        "--seeds",
        str(seed),
        "--episode-start",
        str(start),
        "--episodes-per-seed",
        str(end - start),
        "--domain-id",
        str(domain),
        "--zmq-port",
        str(port),
        "--zmq-out-port",
        str(port + 1),
        "--startup-seconds",
        "20.0",
        "--startup-timeout-seconds",
        "180.0",
        "--pre-episode-settle-seconds",
        "1.0",
        "--episode-boundary-stop-seconds",
        "0.75",
        "--post-reset-idle-seconds",
        "1.5",
        "--episode-upright-timeout-seconds",
        "3.0",
        "--episode-upright-min-z",
        "0.65",
        "--episode-upright-retries",
        "2",
        "--initial-warmup-resets",
        "1",
        "--initial-warmup-idle-seconds",
        "4.0",
        "--episode-start-command-seconds",
        "0.75",
        "--fallback-policy",
        "auto",
        "--adaptive-retry-count",
        "1",
        "--max-segment-duration",
        "0.5",
        "--threshold-scale-global",
        "1.0",
        "--threshold-scale-by-skill",
        "",
        "--method-order-seed",
        "0",
        "--cuda-visible-devices",
        "2",
    ]
    log(
        "launch "
        f"name={lane_name} method={method} seed={seed} ep={start:03d}-{end:03d} "
        f"domain={domain} port={port} source={candidate['source_name']}"
    )
    with stdout_path.open("w") as stdout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKTREE),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"launched pid={proc.pid} name={lane_name}")
    return proc


def main() -> None:
    PID_PATH.write_text(f"{os.getpid()}\n")
    log("sidecar supervisor started")
    children: dict[int, subprocess.Popen[bytes]] = {}
    while True:
        for pid, proc in list(children.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log(f"lane exited rc={rc} pid={pid}")
            children.pop(pid, None)
        done = completed_map()
        total = completed_count(done)
        if total >= EXPECTED:
            log(f"completed={total}/{EXPECTED}; stopping")
            break
        cmdlines = proc_cmdlines()
        if not main_running(cmdlines):
            log(f"main not running; completed={total}/{EXPECTED}; waiting")
            time.sleep(POLL_SECONDS)
            continue
        sidecars = active_sidecars(cmdlines)
        if len(sidecars) >= MAX_ACTIVE_SIDECARS:
            log(
                f"active sidecar pids={sidecars}; "
                f"completed={total}/{EXPECTED}; max_active={MAX_ACTIVE_SIDECARS}; waiting"
            )
            time.sleep(POLL_SECONDS)
            continue
        active_episodes = active_episode_keys(cmdlines)
        reserved_domains: set[int] = set()
        reserved_ports: set[int] = set()
        launched = 0
        while len(sidecars) + launched < MAX_ACTIVE_SIDECARS:
            candidate = choose_candidate(done, active_main_lanes(cmdlines), active_episodes)
            if candidate is None:
                break
            domain_port = choose_domain_port(cmdlines, reserved_domains, reserved_ports)
            if domain_port is None:
                log("no free sidecar domain/port pair; waiting")
                break
            domain, port = domain_port
            proc = launch(candidate, *domain_port)
            children[proc.pid] = proc
            reserved_domains.add(domain)
            reserved_ports.update({port, port + 1})
            method = str(candidate["method"])
            seed = int(candidate["seed"])
            start = int(candidate["start"])
            end = int(candidate["end"])
            for episode in range(start, end):
                active_episodes.add((method, seed, episode))
            cmdlines = proc_cmdlines()
            sidecars = active_sidecars(cmdlines)
            launched += 1
        if launched == 0:
            log(f"no eligible passed-chunk candidate; completed={total}/{EXPECTED}; waiting")
            time.sleep(POLL_SECONDS)
            continue
        time.sleep(30)
    log("sidecar supervisor stopped")


if __name__ == "__main__":
    main()
