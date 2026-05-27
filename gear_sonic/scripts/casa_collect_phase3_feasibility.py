"""Plan or run Phase 3 feasibility data collection and dataset rebuilds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SEED_POOL = (
    REPO_ROOT
    / "outputs/casa/phase2_clean_visual_1000_v2_20260519"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed-pool", action="append", type=Path, default=[])
    parser.add_argument(
        "--source-manifest",
        action="append",
        type=Path,
        default=[],
        help="JSON file with a sources list from a previous Phase 3 run.",
    )
    parser.add_argument("--sim-log-dir", type=Path, default=None)
    parser.add_argument("--casa-props-command-file", type=Path, default=None)
    parser.add_argument("--casa-props-ack-file", type=Path, default=None)
    parser.add_argument("--zmq-host", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--rollout-count", type=int, default=350)
    parser.add_argument("--duration-seconds", type=float, default=45.0)
    parser.add_argument("--scenario-cycle", default="safe_empty")
    parser.add_argument(
        "--skill-schedule",
        choices=["safe_balanced", "gesture_heavy", "turn_heavy", "default"],
        default="safe_balanced",
    )
    parser.add_argument("--run-collection", action="store_true")
    parser.add_argument("--target-samples", type=int, default=5000)
    parser.add_argument("--min-per-skill", type=int, default=500)
    parser.add_argument("--positive-rate-min", type=float, default=0.10)
    parser.add_argument("--positive-rate-max", type=float, default=0.50)
    parser.add_argument("--pre-window-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("phase3_feasibility_%Y%m%d_%H%M%S")
    output_root = args.output_root or Path("outputs") / "casa" / "phase3_feasibility" / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    collection_dir = output_root / "collection"
    dataset_dir = output_root / "dataset"

    seed_pools = list(args.seed_pool)
    for manifest_path in args.source_manifest:
        seed_pools.extend(_sources_from_manifest(manifest_path))
    if not seed_pools:
        seed_pools = [DEFAULT_SEED_POOL]
    episodes_roots = list(seed_pools)

    collection_cmd = _collection_command(args, run_id, collection_dir)
    plan = {
        "run_id": run_id,
        "output_root": str(output_root),
        "seed_pools": [str(path) for path in seed_pools],
        "collection_dir": str(collection_dir),
        "collection_command": collection_cmd,
        "run_collection": args.run_collection,
    }
    (output_root / "collection_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (output_root / "collection_plan.md").write_text(_plan_markdown(plan) + "\n")

    if args.run_collection:
        _require_live_collection_args(args)
        subprocess.run(collection_cmd, cwd=REPO_ROOT, check=True)
        episodes_roots.append(collection_dir)

    source_manifest = {
        "run_id": run_id,
        "sources": [str(path) for path in episodes_roots],
        "seed_pools": [str(path) for path in seed_pools],
        "collection_dir": str(collection_dir) if args.run_collection else "",
        "run_collection": bool(args.run_collection),
    }
    (output_root / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    )

    build_cmd = [
        sys.executable,
        str(REPO_ROOT / "gear_sonic/scripts/casa_build_invocation_dataset.py"),
        "--output-dir",
        str(dataset_dir),
        "--target-samples",
        str(args.target_samples),
        "--pre-window-seconds",
        str(args.pre_window_seconds),
        "--min-per-skill",
        str(args.min_per_skill),
        "--positive-rate-min",
        str(args.positive_rate_min),
        "--positive-rate-max",
        str(args.positive_rate_max),
        "--seed",
        str(args.seed),
    ]
    for root in episodes_roots:
        build_cmd.extend(["--episodes-root", str(root)])
    subprocess.run(build_cmd, cwd=REPO_ROOT, check=True)
    print(json.dumps({"run_id": run_id, "output_root": str(output_root), "dataset_dir": str(dataset_dir)}, indent=2))


def _collection_command(args: argparse.Namespace, run_id: str, collection_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "gear_sonic/scripts/casa_run_oracle_eval_batch.py"),
        "--run-id",
        f"{run_id}_collect",
        "--output-dir",
        str(collection_dir),
        "--rollout-count",
        str(args.rollout_count),
        "--duration-seconds",
        str(args.duration_seconds),
        "--scenario-mode",
        "cycle",
        "--scenario-cycle",
        args.scenario_cycle,
        "--skill-schedule",
        args.skill_schedule,
        "--zmq-host",
        args.zmq_host,
        "--zmq-port",
        str(args.zmq_port),
        "--publish-fps",
        str(args.publish_fps),
        "--seed",
        str(args.seed),
    ]
    if args.sim_log_dir is not None:
        cmd.extend(["--sim-log-dir", str(args.sim_log_dir)])
    if args.casa_props_command_file is not None:
        cmd.extend(["--casa-props-command-file", str(args.casa_props_command_file)])
    if args.casa_props_ack_file is not None:
        cmd.extend(["--casa-props-ack-file", str(args.casa_props_ack_file)])
    return cmd


def _sources_from_manifest(path: Path) -> list[Path]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read source manifest {path}: {exc}") from exc
    sources = manifest.get("sources", [])
    if not isinstance(sources, list):
        raise SystemExit(f"source manifest {path} must contain a list field named 'sources'")
    return [Path(str(source)) for source in sources]


def _require_live_collection_args(args: argparse.Namespace) -> None:
    missing = []
    if args.sim_log_dir is None:
        missing.append("--sim-log-dir")
    if args.casa_props_command_file is None:
        missing.append("--casa-props-command-file")
    if missing:
        raise SystemExit(f"--run-collection requires {' and '.join(missing)}")


def _plan_markdown(plan: dict) -> str:
    command = " \\\n  ".join(plan["collection_command"])
    return (
        "# CASA Phase3 Collection Plan\n\n"
        f"- run_id: `{plan['run_id']}`\n"
        f"- output_root: `{plan['output_root']}`\n"
        f"- run_collection: `{plan['run_collection']}`\n"
        f"- seed_pools: `{', '.join(plan['seed_pools'])}`\n\n"
        "Collection command:\n\n"
        "```bash\n"
        f"{command}\n"
        "```\n"
    )


if __name__ == "__main__":
    main()
