"""Merge CASA Phase 5 online lane outputs into one experiment summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import METHOD_ORDER  # noqa: E402
from gear_sonic.casa.phase5_online import (  # noqa: E402
    build_online_audit,
    method_summary_rows,
    online_go_no_go,
    online_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=2500)
    parser.add_argument("--expected-methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--episodes-per-seed", type=int)
    parser.add_argument("--skills-per-episode", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = []
    decision_rows = []
    lane_summaries = []
    for lane_dir in sorted(path for path in args.online_root.glob("lane_*") if path.is_dir()):
        summary_path = lane_dir / "lane_summary.json"
        if summary_path.exists():
            lane_summaries.append(json.loads(summary_path.read_text()))
        episode_rows.extend(_read_csv(lane_dir / "online" / "online_episode_results.csv"))
        decision_rows.extend(_read_csv(lane_dir / "online" / "gate_decisions.csv"))
    raw_episode_row_count = len(episode_rows)
    raw_decision_row_count = len(decision_rows)
    episode_rows = _dedupe_episode_rows(episode_rows)
    decision_rows = _dedupe_decision_rows(decision_rows, episode_rows)
    expected_methods = _parse_methods(args.expected_methods)
    expected_seeds = _parse_seeds(args.expected_seeds) if args.expected_seeds else None
    method_summary = method_summary_rows(episode_rows)
    audit = build_online_audit(
        episode_rows,
        decision_rows,
        expected_episodes=args.expected_episodes,
        expected_methods=expected_methods,
        expected_seeds=expected_seeds,
        episodes_per_seed=args.episodes_per_seed,
        skills_per_episode=args.skills_per_episode,
        lane_summaries=lane_summaries,
        raw_episode_row_count=raw_episode_row_count,
        raw_decision_row_count=raw_decision_row_count,
    )
    audit["raw_episode_row_count"] = raw_episode_row_count
    audit["raw_decision_row_count"] = raw_decision_row_count
    audit["deduped_episode_row_count"] = len(episode_rows)
    audit["deduped_decision_row_count"] = len(decision_rows)
    _write_csv(args.output_dir / "online_episode_results.csv", episode_rows)
    _write_csv(args.output_dir / "gate_decisions.csv", decision_rows)
    _write_csv(args.output_dir / "method_summary.csv", method_summary)
    (args.output_dir / "method_summary.json").write_text(
        json.dumps({"methods": {row["method"]: row for row in method_summary}}, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "online_acceptance_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "online_go_no_go.json").write_text(
        json.dumps(online_go_no_go(audit), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "online_report.md").write_text(online_report_markdown(audit) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dedupe_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int, int] | tuple[str, int], tuple[tuple[int, int, int], int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = _episode_key(row)
        if key is None:
            key = ("unkeyed", index)
        rank = (_status_rank(row), index)
        current = selected.get(key)
        if current is None or rank > (current[0], current[1]):
            selected[key] = (rank[0], rank[1], row)
    return sorted((item[2] for item in selected.values()), key=_episode_sort_key)


def _dedupe_decision_rows(
    rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_episodes = set()
    for row in episode_rows:
        key = _episode_key(row)
        if key is not None and row_status_is_completed(row):
            selected_episodes.add(key)
    selected: dict[tuple[str, int, int, int], tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        episode_key = _episode_key(row)
        if episode_key not in selected_episodes:
            continue
        try:
            key = (
                row.get("method", ""),
                int(row.get("seed", "-1")),
                int(row.get("episode_index", "-1")),
                int(row.get("skill_idx", "-1")),
            )
        except ValueError:
            continue
        selected[key] = (index, row)
    return [item[1] for key, item in sorted(selected.items(), key=lambda pair: _decision_sort_key(pair[0]))]


def _episode_key(row: dict[str, Any]) -> tuple[str, int, int] | None:
    try:
        method = row.get("method", "")
        seed = int(row.get("seed", "-1"))
        episode = int(row.get("episode_index", "-1"))
    except ValueError:
        return None
    if not method or seed < 0 or episode < 0:
        return None
    return (method, seed, episode)


def _status_rank(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row_status_is_completed(row)),
        int(_int(row.get("initial_upright_ok")) == 1),
        int(not row.get("error")),
    )


def row_status_is_completed(row: dict[str, Any]) -> bool:
    return row.get("status") == "completed"


def _episode_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    key = _episode_key(row)
    if key is None:
        return (len(METHOD_ORDER), 10**9, 10**9)
    method, seed, episode = key
    method_rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (method_rank, seed, episode)


def _decision_sort_key(key: tuple[str, int, int, int]) -> tuple[int, int, int, int]:
    method, seed, episode, skill_idx = key
    method_rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (method_rank, seed, episode, skill_idx)


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
