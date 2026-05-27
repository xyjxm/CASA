"""Merge CASA Phase 5 online lane outputs into one experiment summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHOD_ORDER = ["sonic_only", "hard_contract", "raw_critic_0p5", "global_conformal", "casa_a_per_skill"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=2500)
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
    method_summary = _method_summary_rows(episode_rows)
    audit = _audit(episode_rows, decision_rows, lane_summaries, args.expected_episodes)
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
    (args.output_dir / "online_acceptance_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "online_report.md").write_text(_markdown(audit, method_summary) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


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


def _method_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({row.get("method", "") for row in rows if row.get("method")})
    output = []
    for method in methods:
        method_rows = [row for row in rows if row.get("method") == method]
        count = len(method_rows)
        if count == 0:
            continue
        unsafe = sum(_int(row.get("unsafe_invocation_count")) for row in method_rows)
        fallback = sum(_int(row.get("fallback_count")) for row in method_rows)
        completed = sum(1 for row in method_rows if row.get("status") == "completed")
        task_success = sum(_int(row.get("task_success")) for row in method_rows)
        output.append(
            {
                "method": method,
                "method_display": method_rows[0].get("method_display", method),
                "episode_count": count,
                "completed_count": completed,
                "failed_or_unverified": count - completed,
                "task_success_count": task_success,
                "task_success_rate": task_success / count,
                "unsafe_invocation_count": unsafe,
                "unsafe_invocation_rate_per_episode": unsafe / count,
                "fallback_count": fallback,
                "fallback_rate_per_episode": fallback / count,
                "mean_completion_time_s": sum(_float(row.get("completion_time_s")) for row in method_rows) / count,
            }
        )
    return output


def _audit(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    lane_summaries: list[dict[str, Any]],
    expected_episodes: int,
) -> dict[str, Any]:
    methods = sorted({row.get("method", "") for row in episode_rows if row.get("method")})
    method_summary = {row["method"]: row for row in _method_summary_rows(episode_rows)}
    completed = sum(1 for row in episode_rows if row.get("status") == "completed")
    same_count = len(episode_rows) == expected_episodes
    all_completed = completed == len(episode_rows) and bool(episode_rows)
    gate_decision_count = len(decision_rows)
    expected_decisions_min = len(episode_rows) * 8
    checks = {
        "expected_episode_count_met": same_count,
        "all_episodes_completed": all_completed,
        "all_initial_upright_ok": all(_int(row.get("initial_upright_ok")) == 1 for row in episode_rows),
        "all_five_methods_present": len(methods) == 5,
        "gate_decisions_present": gate_decision_count >= expected_decisions_min,
    }
    if {"sonic_only", "hard_contract", "casa_a_per_skill"}.issubset(method_summary):
        sonic = method_summary["sonic_only"]
        hard = method_summary["hard_contract"]
        casa = method_summary["casa_a_per_skill"]
        checks.update(
            {
                "casa_vs_sonic_unsafe_reduction_ge_40pct": _reduction(
                    sonic["unsafe_invocation_count"], casa["unsafe_invocation_count"]
                )
                >= 0.40,
                "casa_vs_hard_unsafe_reduction_ge_20pct": _reduction(
                    hard["unsafe_invocation_count"], casa["unsafe_invocation_count"]
                )
                >= 0.20,
                "casa_task_success_drop_abs_le_10pp": (
                    sonic["task_success_rate"] - casa["task_success_rate"]
                )
                <= 0.10,
                "casa_task_success_drop_rel_le_30pct": (
                    (sonic["task_success_rate"] - casa["task_success_rate"]) / sonic["task_success_rate"]
                    if sonic["task_success_rate"]
                    else 0.0
                )
                <= 0.30,
            }
        )
    blocking = [name for name, passed in checks.items() if not passed]
    return {
        "phase": "CASA Phase5 online main experiment",
        "expected_episodes": expected_episodes,
        "episode_count": len(episode_rows),
        "completed_count": completed,
        "gate_decision_count": gate_decision_count,
        "methods": methods,
        "lane_summaries": lane_summaries,
        "method_summary": method_summary,
        "checks": checks,
        "blocking_reasons": blocking,
        "go": not blocking,
        "status": "PASS_STRICT_ONLINE" if not blocking else "ONLINE_NO_GO",
    }


def _markdown(audit: dict[str, Any], method_summary: list[dict[str, Any]]) -> str:
    lines = [
        "# CASA Phase5 Online Main Experiment Report",
        "",
        f"- status: `{audit['status']}`",
        f"- go: `{audit['go']}`",
        f"- episodes: `{audit['episode_count']}` / expected `{audit['expected_episodes']}`",
        f"- completed: `{audit['completed_count']}`",
        f"- gate_decisions: `{audit['gate_decision_count']}`",
        "",
        "## Method Summary",
        "",
        "| method | episodes | unsafe | fallback | task_success_rate | mean_time_s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method_display']} | {row['episode_count']} | {row['unsafe_invocation_count']} | "
            f"{row['fallback_count']} | {float(row['task_success_rate']):.4f} | "
            f"{float(row['mean_completion_time_s']):.2f} |"
        )
    lines.extend(["", "## Checks", ""])
    for name, passed in audit["checks"].items():
        lines.append(f"- {name}: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(["", "## Blocking Reasons", ""])
    if audit["blocking_reasons"]:
        lines.extend(f"- {reason}" for reason in audit["blocking_reasons"])
    else:
        lines.append("- none")
    return "\n".join(lines)


def _reduction(baseline_unsafe: int | float, method_unsafe: int | float) -> float:
    baseline = float(baseline_unsafe)
    if baseline <= 0:
        return 0.0
    return (baseline - float(method_unsafe)) / baseline


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
