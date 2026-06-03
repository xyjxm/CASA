"""Audit CASA Phase 5 online artifacts with strict validation and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import METHOD_ORDER  # noqa: E402
from gear_sonic.casa.phase5_online import (  # noqa: E402
    build_online_audit,
    method_summary_rows,
    online_go_no_go,
    online_report_markdown,
    read_csv_rows,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=2500)
    parser.add_argument("--expected-methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--episodes-per-seed", type=int)
    parser.add_argument("--skills-per-episode", type=int, default=8)
    parser.add_argument("--casa-method", default="casa_a_per_skill")
    parser.add_argument("--strict-plan-a-claim", action="store_true")
    parser.add_argument("--min-global-unsafe-reduction", type=float, default=0.10)
    parser.add_argument("--min-global-task-progress-advantage", type=float, default=0.10)
    parser.add_argument("--min-raw-unsafe-reduction", type=float, default=0.10)
    parser.add_argument("--max-fallback-rate-per-episode", type=float, default=2.0)
    parser.add_argument("--max-reject-rate", type=float, default=0.50)
    parser.add_argument("--max-walk-reject-rate", type=float, default=0.75)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.online_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = read_csv_rows(args.online_dir / "online_episode_results.csv")
    decision_rows = read_csv_rows(args.online_dir / "gate_decisions.csv")
    expected_methods = _parse_methods(args.expected_methods)
    expected_seeds = _parse_seeds(args.expected_seeds) if args.expected_seeds else None
    audit = build_online_audit(
        episode_rows,
        decision_rows,
        expected_episodes=args.expected_episodes,
        expected_methods=expected_methods,
        expected_seeds=expected_seeds,
        episodes_per_seed=args.episodes_per_seed,
        skills_per_episode=args.skills_per_episode,
        casa_method=args.casa_method,
        strict_plan_a_claim=args.strict_plan_a_claim,
        min_global_unsafe_reduction=args.min_global_unsafe_reduction,
        min_global_task_progress_advantage=args.min_global_task_progress_advantage,
        min_raw_unsafe_reduction=args.min_raw_unsafe_reduction,
        max_fallback_rate_per_episode=args.max_fallback_rate_per_episode,
        max_reject_rate=args.max_reject_rate,
        max_walk_reject_rate=args.max_walk_reject_rate,
    )
    method_summary = method_summary_rows(episode_rows, decision_rows)
    write_csv_rows(output_dir / "method_summary.csv", method_summary)
    (output_dir / "method_summary.json").write_text(
        json.dumps({"methods": {row["method"]: row for row in method_summary}}, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "online_acceptance_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (output_dir / "online_go_no_go.json").write_text(
        json.dumps(online_go_no_go(audit), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "online_report.md").write_text(online_report_markdown(audit) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.strict and not audit["go"]:
        raise SystemExit(2)


def _parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise SystemExit("--expected-methods must not be empty")
    return methods


def _parse_seeds(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --expected-seeds: {value}") from exc


if __name__ == "__main__":
    main()
