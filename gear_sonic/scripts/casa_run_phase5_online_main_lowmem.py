"""Run/resume the CASA Phase 5 online main experiment with low concurrency."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

METHODS = ["sonic_only", "hard_contract", "raw_critic_0p5", "global_conformal", "casa_a_per_skill"]
SEEDS = [1234, 1235, 1236, 1237, 1238]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", default="outputs/casa/phase4_dataset_v1_strict_50k_20260522")
    parser.add_argument("--phase5-root", default="outputs/casa/phase5_conformal_baselines_20260522")
    parser.add_argument(
        "--online-root",
        type=Path,
        default=Path("outputs/casa/phase5_conformal_baselines_20260522/online_real_main_2500_v2_20260523"),
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--base-domain-id", type=int, default=180)
    parser.add_argument("--domain-pool-size", type=int, default=40)
    parser.add_argument("--base-zmq-port", type=int, default=7800)
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--startup-seconds", type=float, default=20.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--pre-episode-settle-seconds", type=float, default=1.0)
    parser.add_argument("--episode-boundary-stop-seconds", type=float, default=0.75)
    parser.add_argument("--post-reset-idle-seconds", type=float, default=1.5)
    parser.add_argument("--episode-upright-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--episode-upright-min-z", type=float, default=0.65)
    parser.add_argument("--episode-upright-retries", type=int, default=2)
    parser.add_argument("--initial-warmup-resets", type=int, default=1)
    parser.add_argument("--initial-warmup-idle-seconds", type=float, default=4.0)
    parser.add_argument("--episode-start-command-seconds", type=float, default=0.75)
    parser.add_argument("--max-sweeps", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    global METHODS, SEEDS
    args = parse_args()
    METHODS = [item.strip() for item in args.methods.split(",") if item.strip()]
    SEEDS = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    args.online_root.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.cuda_devices.split(",") if item.strip()]
    if not devices:
        raise SystemExit("--cuda-devices must not be empty")
    expected_total = len(METHODS) * len(SEEDS) * args.episodes_per_seed
    for sweep in range(1, max(1, int(args.max_sweeps)) + 1):
        done = completed_map(args.online_root)
        completed_before = sum(len(values) for values in done.values())
        jobs = build_jobs(args, done)
        missing = sum(job["end"] - job["start"] for job in jobs)
        print(
            f"[lowmem] sweep={sweep}/{args.max_sweeps} jobs={len(jobs)} "
            f"missing_episodes={missing} completed={completed_before}/{expected_total} "
            f"max_parallel={args.max_parallel} chunk_size={args.chunk_size}",
            flush=True,
        )
        if not jobs:
            break
        run_jobs(args, jobs, devices, expected_total)
        completed_after = sum(len(values) for values in completed_map(args.online_root).values())
        if completed_after >= expected_total:
            break
        if completed_after <= completed_before:
            print(
                f"[lowmem] sweep={sweep} made no completed-episode progress; "
                "remaining episodes will be retried until --max-sweeps is exhausted",
                flush=True,
            )
    total = sum(len(values) for values in completed_map(args.online_root).values())
    print(f"[lowmem] completed unique episodes={total}/{expected_total}", flush=True)
    merge_cmd = [
        ".venv_sim/bin/python",
        "gear_sonic/scripts/casa_merge_phase5_online_results.py",
        "--online-root",
        str(args.online_root),
        "--output-dir",
        str(args.online_root / "merged"),
        "--expected-episodes",
        str(expected_total),
        "--expected-methods",
        args.methods,
        "--expected-seeds",
        args.seeds,
        "--episodes-per-seed",
        str(args.episodes_per_seed),
        "--skills-per-episode",
        "8",
    ]
    print("[lowmem] merging:", " ".join(merge_cmd), flush=True)
    raise SystemExit(subprocess.call(merge_cmd, cwd="."))


def build_jobs(args: argparse.Namespace, done: dict[tuple[str, int], set[int]]) -> list[dict[str, Any]]:
    job_specs: list[dict[str, Any]] = []
    method_rank = {method: index for index, method in enumerate(METHODS)}
    for method in METHODS:
        for seed in SEEDS:
            for start, end in missing_ranges(done[(method, seed)], args.episodes_per_seed):
                chunk_start = start
                while chunk_start < end:
                    chunk_end = min(end, chunk_start + args.chunk_size)
                    lane_name = f"lane_v2_{method}__seed_{seed}__ep_{chunk_start:03d}_{chunk_end:03d}"
                    job_specs.append(
                        {
                            "name": lane_name,
                            "method": method,
                            "seed": seed,
                            "start": chunk_start,
                            "end": chunk_end,
                            "stdout": args.online_root / f"{lane_name}.stdout.log",
                            "output_dir": args.online_root / lane_name,
                            "_method_rank": method_rank[method],
                        }
                    )
                    chunk_start = chunk_end
    job_specs.sort(key=lambda job: (job["start"], job["seed"], job["_method_rank"]))
    jobs: list[dict[str, Any]] = []
    for job_index, spec in enumerate(job_specs):
        port = args.base_zmq_port + job_index * 2
        domain = args.base_domain_id + (job_index % args.domain_pool_size)
        spec = dict(spec)
        spec.pop("_method_rank", None)
        spec.update({"index": job_index, "domain": domain, "port": port})
        jobs.append(spec)
    return jobs


def run_jobs(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    devices: list[str],
    expected_total: int,
) -> None:
    running: list[dict[str, Any]] = []
    next_index = 0
    last_report = 0.0
    available_devices = list(devices)
    while next_index < len(jobs) or running:
        failed: list[tuple[dict[str, Any], int]] = []
        still_running: list[dict[str, Any]] = []
        for job in running:
            returncode = job["proc"].poll()
            if returncode is None:
                still_running.append(job)
                continue
            job["log_handle"].close()
            available_devices.append(job["device"])
            elapsed = time.time() - job["started_at"]
            if returncode:
                failed.append((job, int(returncode)))
                print(
                    f"[lowmem] FAILED {job['name']} rc={returncode} "
                    f"elapsed={elapsed:.1f}s log={job['stdout']}",
                    flush=True,
                )
            else:
                print(f"[lowmem] finished {job['name']} elapsed={elapsed:.1f}s", flush=True)
        running = still_running

        while next_index < len(jobs) and len(running) < args.max_parallel and available_devices:
            job = jobs[next_index]
            device = available_devices.pop(0)
            cmd = lane_cmd(args, job, device)
            log = job["stdout"].open("w")
            proc = subprocess.Popen(cmd, cwd=".", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update({"proc": proc, "log_handle": log, "started_at": time.time(), "device": device})
            running.append(job)
            print(
                f"[lowmem] started {job['name']} pid={proc.pid} gpu={device} "
                f"domain={job['domain']} port={job['port']}",
                flush=True,
            )
            next_index += 1
            time.sleep(2.0)

        now = time.time()
        if now - last_report >= 60 or failed:
            print_progress(args.online_root, expected_total)
            last_report = now

        if failed:
            for job, returncode in failed:
                print(
                    f"[lowmem] will retry missing episodes after transient lane failure: "
                    f"{job['name']} rc={returncode}",
                    flush=True,
                )

        if next_index < len(jobs) or running:
            time.sleep(5.0)


def lane_cmd(args: argparse.Namespace, job: dict[str, Any], device: str) -> list[str]:
    return [
        ".venv_sim/bin/python",
        "-u",
        "gear_sonic/scripts/casa_run_phase5_online_lane.py",
        "--run-id",
        f"phase5_online_v2_{job['method']}_{job['seed']}_{job['start']:03d}_{job['end']:03d}_20260523",
        "--phase4-root",
        args.phase4_root,
        "--phase5-root",
        args.phase5_root,
        "--output-dir",
        str(job["output_dir"]),
        "--methods",
        str(job["method"]),
        "--seeds",
        str(job["seed"]),
        "--episode-start",
        str(job["start"]),
        "--episodes-per-seed",
        str(job["end"] - job["start"]),
        "--domain-id",
        str(job["domain"]),
        "--zmq-port",
        str(job["port"]),
        "--zmq-out-port",
        str(job["port"] + 1),
        "--startup-seconds",
        str(args.startup_seconds),
        "--startup-timeout-seconds",
        str(args.startup_timeout_seconds),
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
        "--cuda-visible-devices",
        device,
    ]


def completed_map(root: Path) -> dict[tuple[str, int], set[int]]:
    done = {(method, seed): set() for method in METHODS for seed in SEEDS}
    for csv_path in root.glob("lane_*/online/online_episode_results.csv"):
        try:
            with csv_path.open(newline="") as file:
                for row in csv.DictReader(file):
                    if row.get("status") != "completed":
                        continue
                    if row.get("initial_upright_ok") not in {"1", "1.0", "True", "true"}:
                        continue
                    method = row.get("method", "")
                    seed = int(row.get("seed", "-1"))
                    episode = int(row.get("episode_index", "-1"))
                    if (method, seed) in done:
                        done[(method, seed)].add(episode)
        except Exception as exc:
            print(f"[lowmem] skip unreadable {csv_path}: {exc}", flush=True)
    return done


def missing_ranges(done: set[int], episode_count: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = None
    last = None
    for episode in range(episode_count):
        if episode in done:
            if start is not None and last is not None:
                ranges.append((start, last + 1))
                start = None
                last = None
        else:
            if start is None:
                start = episode
            last = episode
    if start is not None and last is not None:
        ranges.append((start, last + 1))
    return ranges


def terminate_job(job: dict[str, Any]) -> None:
    proc = job.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
    try:
        job["log_handle"].close()
    except Exception:
        pass


def print_progress(root: Path, expected_total: int) -> None:
    done = completed_map(root)
    total = sum(len(values) for values in done.values())
    print(
        f"\n[lowmem-progress] {datetime.now().isoformat(timespec='seconds')} "
        f"{total}/{expected_total} completed",
        flush=True,
    )
    for method in METHODS:
        counts = {seed: len(done[(method, seed)]) for seed in SEEDS}
        print(f"[lowmem-progress] {method} {sum(counts.values())} {counts}", flush=True)


if __name__ == "__main__":
    main()
