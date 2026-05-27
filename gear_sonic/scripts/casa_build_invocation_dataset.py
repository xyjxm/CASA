"""Build CASA Phase 3 state-skill-label invocation datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.dataset import build_invocation_dataset, write_dataset_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-root",
        action="append",
        type=Path,
        required=True,
        help="Phase2 root containing episode_*/rollout_summary.json. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-samples", type=int, default=5000)
    parser.add_argument("--pre-window-seconds", type=float, default=1.0)
    parser.add_argument("--exclude-unverified", action="store_true", default=True)
    parser.add_argument(
        "--include-unverified",
        action="store_false",
        dest="exclude_unverified",
        help="Keep unverified rows in the curated dataset. invocations_all.csv always keeps them.",
    )
    parser.add_argument("--min-per-skill", type=int, default=500)
    parser.add_argument("--positive-rate-min", type=float, default=0.10)
    parser.add_argument("--positive-rate-max", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--current-violation-grace-seconds", type=float, default=0.05)
    parser.add_argument(
        "--exclude-current-violations",
        action="store_true",
        default=True,
        help="Drop unsafe samples whose first violation is already present at skill start.",
    )
    parser.add_argument(
        "--include-current-violations",
        action="store_false",
        dest="exclude_current_violations",
        help="Keep instantaneous/current violation labels for legacy audits.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when go criteria are not met.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_invocation_dataset(
        episodes_roots=args.episodes_root,
        target_samples=args.target_samples,
        pre_window_seconds=args.pre_window_seconds,
        exclude_unverified=args.exclude_unverified,
        min_per_skill=args.min_per_skill,
        positive_rate_min=args.positive_rate_min,
        positive_rate_max=args.positive_rate_max,
        seed=args.seed,
        exclude_current_violations=args.exclude_current_violations,
        current_violation_grace_seconds=args.current_violation_grace_seconds,
    )
    write_dataset_outputs(result, args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    if args.strict and not result.summary.get("go_criteria_passed", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
