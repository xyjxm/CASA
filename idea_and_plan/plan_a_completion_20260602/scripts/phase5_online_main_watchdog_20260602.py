#!/usr/bin/env python3
"""Watch and resume the CASA Phase 5 online main run for Plan A."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

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
MAIN_PID_FILE = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602.pid"
)
LOWMEM_LOG = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602.lowmem.log"
)
AUDIT = ROOT / "merged" / "online_acceptance_audit.json"
STRICT_AUDIT_SCRIPT = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "plan_a_strict_completion_audit_20260602.py"
)
METHODS = ["sonic_only", "hard_contract", "raw_critic_0p5", "global_conformal", "casa_a_per_skill"]
SEEDS = [1234, 1235, 1236, 1237, 1238]
EXPECTED = 2500
MAX_RESTARTS = 20


def stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def main_pids() -> list[int]:
    pids: list[int] = []
    me = os.getpid()
    script = "gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py"
    root = str(ROOT)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except OSError:
            continue
        if script in cmdline and root in cmdline:
            pids.append(pid)
    return sorted(pids)


def completed_count() -> tuple[int, dict[str, dict[int, int]], dict[str, int]]:
    completed: set[tuple[str, int, int]] = set()
    by = {method: {seed: 0 for seed in SEEDS} for method in METHODS}
    statuses: dict[str, int] = {}
    for csv_path in ROOT.glob("lane_*/online/online_episode_results.csv"):
        try:
            with csv_path.open(newline="") as file:
                for row in csv.DictReader(file):
                    status = row.get("status", "")
                    statuses[status] = statuses.get(status, 0) + 1
                    if status != "completed":
                        continue
                    if row.get("initial_upright_ok") not in {"1", "1.0", "True", "true"}:
                        continue
                    method = row.get("method", "")
                    seed = int(row.get("seed", "-1"))
                    episode = int(row.get("episode_index", "-1"))
                    key = (method, seed, episode)
                    if method in by and seed in by[method] and key not in completed:
                        completed.add(key)
                        by[method][seed] += 1
        except Exception as exc:
            log(f"skip unreadable {csv_path}: {exc}")
    return len(completed), by, statuses


def audit_passes() -> bool:
    if not AUDIT.exists():
        return False
    try:
        data = json.loads(AUDIT.read_text())
    except Exception as exc:
        log(f"audit unreadable: {exc}")
        return False
    status = data.get("status") or data.get("overall_status")
    blockers = data.get("blocking_reasons") or data.get("blockers") or []
    return bool(data.get("go")) and status == "PASS_STRICT_ONLINE" and not blockers


def lowmem_cmd() -> list[str]:
    return [
        ".venv_sim/bin/python",
        "-u",
        "gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py",
        "--phase4-root",
        str(PHASE4_ROOT),
        "--phase5-root",
        str(PHASE5_ROOT),
        "--online-root",
        str(ROOT),
        "--max-parallel",
        "4",
        "--chunk-size",
        "25",
        "--episodes-per-seed",
        "100",
        "--seeds",
        "1234,1235,1236,1237,1238",
        "--cuda-devices",
        "2,2,2,2",
        "--base-domain-id",
        "130",
        "--domain-pool-size",
        "80",
        "--base-zmq-port",
        "8500",
        "--startup-seconds",
        "20",
        "--startup-timeout-seconds",
        "180",
        "--max-sweeps",
        "8",
    ]


def merge_cmd() -> list[str]:
    return [
        ".venv_sim/bin/python",
        "gear_sonic/scripts/casa_merge_phase5_online_results.py",
        "--online-root",
        str(ROOT),
        "--output-dir",
        str(ROOT / "merged"),
        "--expected-episodes",
        str(EXPECTED),
        "--expected-methods",
        ",".join(METHODS),
        "--expected-seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--episodes-per-seed",
        "100",
        "--skills-per-episode",
        "8",
        "--casa-method",
        "casa_a_per_skill",
    ]


def run_merge() -> int:
    log("running merge/audit command")
    return subprocess.call(merge_cmd(), cwd=str(WORKTREE))


def run_strict_audit() -> int:
    if not STRICT_AUDIT_SCRIPT.exists():
        log(f"strict audit script missing: {STRICT_AUDIT_SCRIPT}")
        return 2
    log(f"running strict Plan A audit: {STRICT_AUDIT_SCRIPT}")
    return subprocess.call(["python", str(STRICT_AUDIT_SCRIPT)], cwd=str(WORKTREE))


def run_lowmem(attempt: int) -> int:
    cmd = lowmem_cmd()
    log(f"starting lowmem resume attempt={attempt}: {' '.join(cmd)}")
    LOWMEM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOWMEM_LOG.open("a") as log_file:
        log_file.write(f"\n[watchdog] {stamp()} starting resume attempt={attempt}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKTREE),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        MAIN_PID_FILE.write_text(f"{proc.pid}\n")
        rc = proc.wait()
        log_file.write(f"\n[watchdog] {stamp()} resume attempt={attempt} exited rc={rc}\n")
        log_file.flush()
    return int(rc)


def main() -> None:
    log("watchdog started")
    restarts = 0
    last_detail = 0.0
    while True:
        if audit_passes():
            count, _, statuses = completed_count()
            log(f"PASS audit detected; completed={count}/{EXPECTED}; statuses={statuses}")
            rc = run_strict_audit()
            log(f"strict Plan A audit exited rc={rc}")
            break
        pids = main_pids()
        if pids:
            now = time.time()
            if now - last_detail >= 600:
                count, by, statuses = completed_count()
                log(f"main running pids={pids}; completed={count}/{EXPECTED}; statuses={statuses}; by={by}")
                last_detail = now
            time.sleep(120)
            continue
        count, by, statuses = completed_count()
        log(f"no main process; completed={count}/{EXPECTED}; statuses={statuses}; by={by}")
        if count >= EXPECTED:
            rc = run_merge()
            log(f"merge exited rc={rc}")
            if audit_passes():
                log("PASS audit detected after merge")
                rc = run_strict_audit()
                log(f"strict Plan A audit exited rc={rc}")
                break
        if restarts >= MAX_RESTARTS:
            log(f"max restarts exhausted ({MAX_RESTARTS}); leaving for manual inspection")
            break
        restarts += 1
        rc = run_lowmem(restarts)
        log(f"lowmem resume attempt={restarts} exited rc={rc}")
        time.sleep(30)
    log("watchdog stopped")


if __name__ == "__main__":
    main()
