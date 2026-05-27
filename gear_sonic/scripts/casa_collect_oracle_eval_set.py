"""Collect CASA oracle rollout summaries and create a human review sheet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="phase2_oracle")
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-all-rollouts", action="store_true", default=True)
    parser.add_argument("--max-review-rollouts", type=int, default=0, help="0 means all discovered rollouts")
    parser.add_argument("--assisted-review", action="store_true", help="Fill review fields from oracle logs.")
    parser.add_argument("--reviewer", default="codex_assisted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = sorted(args.episodes_root.glob("episode_*/rollout_summary.json"))
    if args.max_review_rollouts > 0:
        review_summaries = summaries[: args.max_review_rollouts]
    else:
        review_summaries = summaries
    rollouts_csv = args.output_dir / "rollouts.csv"
    review_csv = args.output_dir / "human_review_sheet.csv"
    report_md = args.output_dir / "phase2_report.md"

    rollout_rows = []
    review_rows = []
    violation_counts = Counter()
    skill_label_counts = Counter()
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        rollout_rows.append(
            {
                "run_id": summary.get("run_id", args.run_id),
                "rollout_id": summary.get("rollout_id", summary_path.parent.name),
                "summary_path": str(summary_path),
                "violation_yes_no": int(bool(summary.get("violation_yes_no"))),
                "violation_type": summary.get("violation_type") or "",
                "violation_count": summary.get("violation_count", 0),
                "time_to_violation": summary.get("time_to_violation"),
                "violation_time_bin": summary.get("violation_time_bin") or "",
                "target_bucket": _scene_prop(summary, "target_bucket"),
                "scene_complexity": _scene_prop(summary, "scene_complexity"),
                "scene_family": _scene_prop(summary, "scene_family") or summary.get("scenario") or "",
                "num_users": _scene_prop(summary, "num_users"),
                "num_obstacles": _scene_prop(summary, "num_obstacles"),
            }
        )
        violation_counts.update(summary.get("violation_counts", {}))
        skill_label_counts.update(summary.get("skill_label_counts", {}))
        if summary_path in review_summaries:
            assisted = _assisted_review(summary) if args.assisted_review else {}
            review_rows.append(
                {
                    "run_id": summary.get("run_id", args.run_id),
                    "rollout_id": summary.get("rollout_id", summary_path.parent.name),
                    "summary_path": str(summary_path),
                    "oracle_label": "unsafe" if summary.get("violation_yes_no") else "safe",
                    "oracle_violation_type": summary.get("violation_type") or "",
                    "oracle_violation_types_json": json.dumps(summary.get("violation_types", [])),
                    "human_label": assisted.get("human_label", ""),
                    "human_violation_type": assisted.get("human_violation_type", ""),
                    "reviewer": args.reviewer if args.assisted_review else "",
                    "notes": assisted.get("notes", ""),
                }
            )

    _write_csv(rollouts_csv, rollout_rows)
    _write_csv(review_csv, review_rows)
    report_md.write_text(
        "\n".join(
            [
                "# CASA Phase 2 Oracle Report",
                "",
                f"- run_id: `{args.run_id}`",
                f"- episodes_root: `{args.episodes_root}`",
                f"- discovered_rollouts: `{len(summaries)}`",
                f"- review_rollouts: `{len(review_rows)}`",
                f"- violation_counts: `{dict(violation_counts)}`",
                f"- skill_label_counts: `{dict(skill_label_counts)}`",
                f"- assisted_review: `{args.assisted_review}`",
                "",
                "Human labels are expected in `human_review_sheet.csv`.",
            ]
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "rollouts": len(summaries),
                "review_rows": len(review_rows),
                "rollouts_csv": str(rollouts_csv),
                "human_review_sheet": str(review_csv),
                "phase2_report": str(report_md),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _assisted_review(summary: dict) -> dict[str, str]:
    oracle_label = "unsafe" if summary.get("violation_yes_no") else "safe"
    oracle_type = summary.get("violation_type") or ""
    notes = []
    for field in [
        "min_user_distance",
        "min_arm_user_distance",
        "min_obstacle_distance",
        "max_torso_pitch",
        "max_torso_roll",
        "min_base_height",
        "control_loop_overrun_rows",
    ]:
        if field in summary:
            notes.append(f"{field}={summary[field]}")
    notes.append(f"oracle_violation_types={summary.get('violation_types', [])}")
    notes.append("assistant-assisted log review; user will manually audit before paper use")
    return {
        "human_label": oracle_label,
        "human_violation_type": oracle_type if oracle_label == "unsafe" else "",
        "notes": "; ".join(notes),
    }


def _scene_prop(summary: dict, key: str):
    scene_props = summary.get("scene_props", {})
    if isinstance(scene_props, dict) and key in scene_props:
        return scene_props.get(key)
    return summary.get(key, "")


if __name__ == "__main__":
    main()
