"""Create or summarize Phase 5 online policy pilot sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5_online import write_csv_rows  # noqa: E402
from gear_sonic.casa.phase5_online_sweep import (  # noqa: E402
    build_sweep_grid,
    evaluate_sweep_candidates,
    sweep_report,
    sweep_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Evaluated candidate in name:path form; path contains online_episode_results.csv or merged/.",
    )
    parser.add_argument("--sonic-method", default="sonic_only")
    parser.add_argument("--fallback-policies", default="stop,adaptive,adaptive_retry")
    parser.add_argument("--hard-or-casa", default="false,true")
    parser.add_argument("--segment-long-skills", default="false,true")
    parser.add_argument("--max-segment-durations", default="0.5,0.75,1.0")
    parser.add_argument("--threshold-scale-global", default="0.7,0.8,0.9,1.0")
    parser.add_argument(
        "--threshold-scale-by-skill",
        default="",
        help="Semicolon-separated per-skill scale specs; empty item means global scale only.",
    )
    parser.add_argument("--recovery-retry-counts", default="0,1,2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.candidate:
        rows.extend(evaluate_sweep_candidates(_parse_candidates(args.candidate), sonic_method=args.sonic_method))
    else:
        rows.extend(
            build_sweep_grid(
                fallback_policies=_parse_list(args.fallback_policies),
                hard_or_values=_parse_bool_list(args.hard_or_casa),
                segmentation_values=_parse_bool_list(args.segment_long_skills),
                max_segment_durations=[float(item) for item in _parse_list(args.max_segment_durations)],
                global_threshold_scales=[float(item) for item in _parse_list(args.threshold_scale_global)],
                per_skill_threshold_scales=_parse_semicolon_list(args.threshold_scale_by_skill),
                recovery_retry_counts=[int(item) for item in _parse_list(args.recovery_retry_counts)],
            )
        )
    report = sweep_report(rows)
    write_csv_rows(args.output_dir / "phase5_online_policy_sweep.csv", rows)
    (args.output_dir / "phase5_online_policy_sweep.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "phase5_online_policy_sweep.md").write_text(sweep_report_markdown(report) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _parse_candidates(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if ":" not in value:
            raise SystemExit(f"Invalid --candidate {value!r}; expected name:path")
        name, path = value.split(":", 1)
        output.append((name, Path(path)))
    return output


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_semicolon_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(";")]
    return values if values and any(values) else [""]


def _parse_bool_list(raw: str) -> list[bool]:
    output = []
    for item in _parse_list(raw):
        text = item.lower()
        if text in {"1", "true", "yes", "y"}:
            output.append(True)
        elif text in {"0", "false", "no", "n"}:
            output.append(False)
        else:
            raise SystemExit(f"Invalid boolean value: {item!r}")
    return output


if __name__ == "__main__":
    main()
