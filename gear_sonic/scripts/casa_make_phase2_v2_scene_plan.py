from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


CROSS_QUOTAS = {
    ("clean_safe", "simple"): 80,
    ("clean_safe", "medium"): 220,
    ("clean_safe", "hard"): 100,
    ("visual_collision_or_close", "simple"): 30,
    ("visual_collision_or_close", "medium"): 150,
    ("visual_collision_or_close", "hard"): 120,
    ("visual_near_boundary", "simple"): 25,
    ("visual_near_boundary", "medium"): 70,
    ("visual_near_boundary", "hard"): 55,
    ("visual_fall", "simple"): 15,
    ("visual_fall", "medium"): 60,
    ("visual_fall", "hard"): 75,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CASA Phase 2 v2 scene plan CSV.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--mode", choices=["pilot", "formal"], default="formal")
    parser.add_argument("--pilot-per-cell", type=int, default=10)
    parser.add_argument("--formal-multiplier", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    rows = []
    if args.mode == "pilot":
        for bucket, complexity in CROSS_QUOTAS:
            for _ in range(args.pilot_per_cell):
                rows.append({"target_bucket": bucket, "scene_complexity": complexity})
    else:
        for (bucket, complexity), quota in CROSS_QUOTAS.items():
            count = int(round(quota * args.formal_multiplier + 0.499999))
            for _ in range(count):
                rows.append({"target_bucket": bucket, "scene_complexity": complexity})

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["episode_index"] = index

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode_index", "target_bucket", "scene_complexity"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_csv": str(args.output_csv),
        "mode": args.mode,
        "rows": len(rows),
        "seed": args.seed,
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
