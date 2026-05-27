"""Merge parallel CASA Phase 2 rollout episode directories into one eval root."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="phase2_oracle_merged")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--max-rollouts", type=int, default=300)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy episode directories instead of creating symlinks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for source in args.source_run:
        source = source.resolve()
        for summary_path in sorted(source.glob("episode_*/rollout_summary.json")):
            episodes.append((source, summary_path.parent.resolve()))
    if args.max_rollouts > 0:
        episodes = episodes[: args.max_rollouts]

    manifest_rows = []
    for index, (source_root, episode_dir) in enumerate(episodes):
        target = args.output_dir / f"episode_{index:06d}"
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise FileExistsError(f"Refusing to overwrite existing episode directory: {target}")
        if args.copy:
            shutil.copytree(episode_dir, target)
        else:
            target.symlink_to(episode_dir, target_is_directory=True)
        manifest_rows.append(
            {
                "merged_run_id": args.run_id,
                "merged_rollout_id": target.name,
                "source_run": source_root.name,
                "source_episode": episode_dir.name,
                "source_path": str(episode_dir),
                "merged_path": str(target),
            }
        )

    manifest_csv = args.output_dir / "merge_manifest.csv"
    if manifest_rows:
        with manifest_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    else:
        manifest_csv.write_text("")

    manifest_json = args.output_dir / "merge_manifest.json"
    with manifest_json.open("w") as file:
        json.dump(
            {
                "run_id": args.run_id,
                "output_dir": str(args.output_dir),
                "rollouts": len(manifest_rows),
                "source_runs": [str(path.resolve()) for path in args.source_run],
                "copy": args.copy,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": str(args.output_dir),
                "rollouts": len(manifest_rows),
                "merge_manifest_csv": str(manifest_csv),
                "merge_manifest_json": str(manifest_json),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
