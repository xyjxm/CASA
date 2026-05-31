"""Diagnose CASA Phase 5 online no-go artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5_online import read_csv_rows  # noqa: E402
from gear_sonic.casa.phase5_online_diagnostics import (  # noqa: E402
    build_failure_breakdown,
    failure_breakdown_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sonic-method", default="sonic_only")
    parser.add_argument("--casa-method", default="casa_a_per_skill")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.online_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = read_csv_rows(args.online_dir / "online_episode_results.csv")
    decision_rows = read_csv_rows(args.online_dir / "gate_decisions.csv")
    breakdown = build_failure_breakdown(
        episode_rows,
        decision_rows,
        sonic_method=args.sonic_method,
        casa_method=args.casa_method,
    )
    (output_dir / "phase5_online_failure_breakdown.json").write_text(
        json.dumps(breakdown, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "phase5_online_failure_breakdown.md").write_text(
        failure_breakdown_markdown(breakdown) + "\n"
    )
    print(json.dumps(breakdown, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
