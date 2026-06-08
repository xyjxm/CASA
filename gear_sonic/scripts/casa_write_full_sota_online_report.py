"""Write full CASA adapted-SOTA online comparison report artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.baselines.sota_adapters import SOTA_METHOD_DISPLAY, SOTA_METHOD_ORDER  # noqa: E402
from gear_sonic.casa.phase5 import METHOD_DISPLAY, METHOD_ORDER  # noqa: E402


DEFAULT_METHODS = (
    "sonic_only",
    "hard_contract",
    "raw_critic_0p5",
    "global_conformal",
    "casa_a_per_skill",
    *SOTA_METHOD_ORDER,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--completed-online-dir", type=Path, action="append", default=[])
    parser.add_argument("--offline-dir", type=Path)
    parser.add_argument("--dry-run-dir", type=Path)
    parser.add_argument("--large-online-root", type=Path)
    parser.add_argument("--large-command-file", type=Path)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--smoke-command", default="")
    parser.add_argument("--dry-run-command", default="")
    parser.add_argument("--offline-command", default="")
    parser.add_argument("--blocker", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    completed = [_load_online_dir(path, methods) for path in args.completed_online_dir]
    primary = completed[-1] if completed else _empty_online_summary(methods)
    offline = _load_optional_file(args.offline_dir / "offline_replay_metrics.json" if args.offline_dir else None)
    dry_run = _load_online_dir(args.dry_run_dir, methods) if args.dry_run_dir else None
    large_status = _large_status(args.large_online_root, methods)
    metadata = _load_metadata(args.output_root)
    method_table = _method_table(primary, methods, metadata)
    strict = _strict_conclusion(primary, methods)
    commands = {
        "offline": args.offline_command,
        "dry_run": args.dry_run_command,
        "smoke": args.smoke_command,
        "large": _read_text(args.large_command_file),
    }

    metrics = {
        "generated_at": _now(),
        "phase4_root": str(args.phase4_root),
        "phase5_root": str(args.phase5_root),
        "methods": methods,
        "completed_online": completed,
        "primary_completed_online_dir": primary.get("online_dir", ""),
        "method_table": method_table,
        "strict_conclusion": strict,
        "offline_summary": offline,
        "dry_run_summary": dry_run,
        "large_online_status": large_status,
        "blockers": args.blocker,
        "commands": commands,
    }
    runtime_stats = {
        "generated_at": metrics["generated_at"],
        "primary_completed_online_dir": primary.get("online_dir", ""),
        "runtime_by_method_ms": {
            row["method"]: {
                "p50": row.get("runtime_ms_p50"),
                "p90": row.get("runtime_ms_p90"),
                "p99": row.get("runtime_ms_p99"),
                "decision_count": row.get("decision_count"),
            }
            for row in method_table
        },
    }

    reports_dir = args.output_root / "reports"
    manifests_dir = args.output_root / "manifests"
    metrics_dir = args.output_root / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "full_sota_online_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (metrics_dir / "runtime_stats.json").write_text(json.dumps(runtime_stats, indent=2, sort_keys=True) + "\n")
    manifest = _manifest(args, methods, completed, large_status, commands)
    (manifests_dir / "full_sota_online_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    report = _report_markdown(metrics)
    (reports_dir / "full_sota_online_comparison_report.md").write_text(report + "\n")
    (args.output_root / "README.md").write_text(_readme_markdown(metrics) + "\n")
    print(json.dumps({"report": str(reports_dir / "full_sota_online_comparison_report.md")}, indent=2))


def _load_online_dir(path: Path, methods: list[str]) -> dict[str, Any]:
    episode_rows = _read_csv(path / "online_episode_results.csv")
    decision_rows = _read_csv(path / "gate_decisions.csv")
    return {
        "online_dir": str(path),
        "episode_count": len(episode_rows),
        "decision_count": len(decision_rows),
        "method_summary": _summarize_online(episode_rows, decision_rows, methods),
        "manifest": _load_optional_file(path / "online_experiment_manifest.json"),
    }


def _empty_online_summary(methods: list[str]) -> dict[str, Any]:
    return {
        "online_dir": "",
        "episode_count": 0,
        "decision_count": 0,
        "method_summary": {method: _empty_method_row(method) for method in methods},
    }


def _summarize_online(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        episodes = [row for row in episode_rows if row.get("method") == method]
        decisions = [row for row in decision_rows if row.get("method") == method]
        if not episodes:
            summary[method] = _empty_method_row(method)
            continue
        episode_count = len(episodes)
        unsafe_count = sum(_int(row.get("unsafe_invocation_count")) for row in episodes)
        unsafe_episode_count = sum(_int(row.get("unsafe_invocation_count")) > 0 for row in episodes)
        fallback_count = sum(_int(row.get("fallback_count")) for row in episodes)
        task_success_count = sum(_int(row.get("task_success")) for row in episodes)
        violation_count = sum(_int(row.get("violation_count")) for row in episodes)
        violation_types: dict[str, int] = {}
        for row in episodes:
            for violation_type in _violation_types(row.get("violation_types")):
                violation_types[violation_type] = violation_types.get(violation_type, 0) + 1
        reject_count = sum(1 for row in decisions if row.get("decision") == "reject")
        runtimes = [
            value
            for value in (_float_or_none(row.get("runtime_ms")) for row in decisions)
            if value is not None
        ]
        summary[method] = {
            "method": method,
            "method_display": _display(method),
            "episode_count": episode_count,
            "completed_count": sum(1 for row in episodes if row.get("status") == "completed"),
            "failed_or_unverified": sum(1 for row in episodes if row.get("status") != "completed"),
            "task_success_count": task_success_count,
            "task_success_rate": task_success_count / episode_count,
            "safe_completion_rate": task_success_count / episode_count,
            "unsafe_invocation_count": unsafe_count,
            "unsafe_invocation_rate_per_episode": unsafe_count / episode_count,
            "unsafe_episode_count": unsafe_episode_count,
            "unsafe_episode_rate": unsafe_episode_count / episode_count,
            "fallback_count": fallback_count,
            "fallback_rate_per_episode": fallback_count / episode_count,
            "decision_count": len(decisions),
            "reject_count": reject_count,
            "reject_rate_per_decision": reject_count / len(decisions) if decisions else None,
            "violation_count": violation_count,
            "fall_count": violation_types.get("fall", 0),
            "collision_count": violation_types.get("collision", 0),
            "near_collision_count": violation_types.get("near_collision", 0),
            "human_distance_violation_count": violation_types.get("human_distance_violation", 0),
            "runtime_ms_p50": _percentile(runtimes, 0.50),
            "runtime_ms_p90": _percentile(runtimes, 0.90),
            "runtime_ms_p99": _percentile(runtimes, 0.99),
        }
    sonic = summary.get("sonic_only", {})
    sonic_rate = _float_or_none(sonic.get("unsafe_invocation_rate_per_episode"))
    for row in summary.values():
        rate = _float_or_none(row.get("unsafe_invocation_rate_per_episode"))
        row["unsafe_reduction_vs_sonic_only"] = (
            None if sonic_rate in (None, 0.0) or rate is None else (sonic_rate - rate) / sonic_rate
        )
    return summary


def _empty_method_row(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "method_display": _display(method),
        "episode_count": 0,
        "completed_count": 0,
        "failed_or_unverified": 0,
        "task_success_rate": None,
        "safe_completion_rate": None,
        "unsafe_invocation_count": 0,
        "unsafe_invocation_rate_per_episode": None,
        "unsafe_reduction_vs_sonic_only": None,
        "fallback_rate_per_episode": None,
        "reject_rate_per_decision": None,
        "fall_count": 0,
        "collision_count": 0,
        "near_collision_count": 0,
        "human_distance_violation_count": 0,
        "runtime_ms_p50": None,
        "runtime_ms_p90": None,
        "runtime_ms_p99": None,
    }


def _method_table(
    primary: dict[str, Any],
    methods: list[str],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    method_summary = primary.get("method_summary", {})
    fidelity = _fidelity_by_method(metadata)
    rows = []
    for method in methods:
        row = dict(method_summary.get(method) or _empty_method_row(method))
        row["implementation_fidelity"] = fidelity.get(method, "exact_code")
        rows.append(row)
    return rows


def _strict_conclusion(primary: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    summary = primary.get("method_summary", {})
    casa = summary.get("casa_a_per_skill", {})
    casa_unsafe = _float_or_none(casa.get("unsafe_invocation_rate_per_episode"))
    casa_success = _float_or_none(casa.get("task_success_rate"))
    outperform = []
    underperform = []
    unsupported = []
    for method in methods:
        if method in set(METHOD_ORDER):
            continue
        row = summary.get(method, {})
        unsafe = _float_or_none(row.get("unsafe_invocation_rate_per_episode"))
        success = _float_or_none(row.get("task_success_rate"))
        if not row.get("episode_count") or casa_unsafe is None or casa_success is None or unsafe is None or success is None:
            unsupported.append(method)
        elif unsafe < casa_unsafe and success >= casa_success:
            outperform.append(method)
        elif unsafe > casa_unsafe or success < casa_success:
            underperform.append(method)
        else:
            unsupported.append(method)
    return {
        "outperform_casa_on_completed_online_evidence": outperform,
        "underperform_casa_on_completed_online_evidence": underperform,
        "not_yet_supported_or_tied": unsupported,
        "rule": (
            "outperform requires lower unsafe_invocation_rate_per_episode than "
            "casa_a_per_skill and task_success_rate no lower than CASA on completed online rows"
        ),
    }


def _large_status(path: Path | None, methods: list[str]) -> dict[str, Any]:
    if path is None:
        return {"status": "not_launched"}
    lane_csvs = sorted(path.glob("lane_*/online/online_episode_results.csv"))
    episodes = []
    for csv_path in lane_csvs:
        episodes.extend(_read_csv(csv_path))
    by_method = {
        method: sum(1 for row in episodes if row.get("method") == method and row.get("status") == "completed")
        for method in methods
    }
    return {
        "status": "has_outputs" if lane_csvs else "launched_or_command_prepared_no_lane_outputs_yet",
        "large_online_root": str(path),
        "lane_episode_csv_count": len(lane_csvs),
        "completed_episode_rows_by_method": by_method,
        "completed_episode_rows_total": sum(by_method.values()),
    }


def _manifest(
    args: argparse.Namespace,
    methods: list[str],
    completed: list[dict[str, Any]],
    large_status: dict[str, Any],
    commands: dict[str, str],
) -> dict[str, Any]:
    return {
        "phase": "CASA/SONIC Phase5 full adapted SOTA online comparison",
        "generated_at": _now(),
        "output_root": str(args.output_root),
        "phase4_root": str(args.phase4_root),
        "phase5_root": str(args.phase5_root),
        "methods": methods,
        "sota_methods": list(SOTA_METHOD_ORDER),
        "completed_online_dirs": [
            {
                "path": item.get("online_dir", ""),
                "episode_count": item.get("episode_count", 0),
                "decision_count": item.get("decision_count", 0),
            }
            for item in completed
        ],
        "offline_dir": "" if args.offline_dir is None else str(args.offline_dir),
        "dry_run_dir": "" if args.dry_run_dir is None else str(args.dry_run_dir),
        "large_online_status": large_status,
        "commands": commands,
        "expected_artifacts": [
            "online_episode_results.csv",
            "gate_decisions.csv",
            "method_summary.csv/json",
            "metrics/full_sota_online_metrics.json",
            "metrics/runtime_stats.json",
            "reports/full_sota_online_comparison_report.md",
        ],
    }


def _report_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Full Adapted SOTA Online Comparison",
        "",
        f"- generated_at: `{metrics['generated_at']}`",
        f"- primary completed online dir: `{metrics['primary_completed_online_dir'] or 'none'}`",
        f"- phase4_root: `{metrics['phase4_root']}`",
        f"- phase5_root: `{metrics['phase5_root']}`",
        "",
        "## Completed Online Results",
        "",
        _markdown_table(metrics["method_table"]),
        "",
        "## Strict Conclusion",
        "",
    ]
    strict = metrics["strict_conclusion"]
    lines.extend(
        [
            "- Outperform CASA on completed online evidence: "
            + _fmt_list(strict["outperform_casa_on_completed_online_evidence"]),
            "- Underperform CASA on completed online evidence: "
            + _fmt_list(strict["underperform_casa_on_completed_online_evidence"]),
            "- Not yet supported or tied: " + _fmt_list(strict["not_yet_supported_or_tied"]),
            f"- Decision rule: {strict['rule']}",
            "",
            "## Still-Running Or Continuation",
            "",
            f"- large online status: `{metrics['large_online_status'].get('status')}`",
            f"- large online root: `{metrics['large_online_status'].get('large_online_root', '')}`",
            f"- completed large rows total: `{metrics['large_online_status'].get('completed_episode_rows_total', 0)}`",
            "",
            "## Offline And Dry-Run Sanity Checks",
            "",
            f"- offline replay metrics loaded: `{bool(metrics.get('offline_summary'))}`",
            f"- dry-run online dir: `{(metrics.get('dry_run_summary') or {}).get('online_dir', '')}`",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = metrics.get("blockers") or []
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None recorded by report generator.")
    lines.extend(
        [
            "",
            "## Reproducibility Commands",
            "",
        ]
    )
    for name, command in metrics["commands"].items():
        if command:
            lines.extend([f"### {name}", "", "```bash", command.strip(), "```", ""])
    return "\n".join(lines)


def _readme_markdown(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CASA Full Adapted SOTA Online Comparison",
            "",
            f"- Report: `reports/full_sota_online_comparison_report.md`",
            f"- Manifest: `manifests/full_sota_online_comparison_manifest.json`",
            f"- Metrics: `metrics/full_sota_online_metrics.json`",
            f"- Runtime stats: `metrics/runtime_stats.json`",
            f"- Primary completed online dir: `{metrics['primary_completed_online_dir'] or 'none'}`",
            "",
            "The report separates completed online evidence, offline/dry-run checks, and any larger evaluation status.",
        ]
    )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "method",
        "fidelity",
        "episodes",
        "unsafe rate/count",
        "unsafe reduction vs SONIC",
        "safe completion",
        "fallback/ep",
        "reject/decision",
        "fall",
        "collision",
        "near",
        "human-dist",
        "runtime p50/p90/p99 ms",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        runtime = "/".join(
            _fmt(row.get(key)) for key in ("runtime_ms_p50", "runtime_ms_p90", "runtime_ms_p99")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['method']}`",
                    f"`{row.get('implementation_fidelity', '')}`",
                    str(row.get("episode_count", 0)),
                    f"{_fmt(row.get('unsafe_invocation_rate_per_episode'))}/{row.get('unsafe_invocation_count', 0)}",
                    _fmt(row.get("unsafe_reduction_vs_sonic_only")),
                    _fmt(row.get("safe_completion_rate")),
                    _fmt(row.get("fallback_rate_per_episode")),
                    _fmt(row.get("reject_rate_per_decision")),
                    str(row.get("fall_count", 0)),
                    str(row.get("collision_count", 0)),
                    str(row.get("near_collision_count", 0)),
                    str(row.get("human_distance_violation_count", 0)),
                    runtime,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _load_metadata(output_root: Path) -> dict[str, Any]:
    path = output_root / "data" / "sota_method_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _fidelity_by_method(metadata: dict[str, Any]) -> dict[str, str]:
    output = {method: "exact_code" for method in METHOD_ORDER}
    for item in metadata.get("methods", []):
        output[str(item.get("method_name"))] = str(item.get("main_table_fidelity_label", ""))
    for method in SOTA_METHOD_ORDER:
        output.setdefault(method, "paper_faithful_proxy")
    return output


def _display(method: str) -> str:
    return {**METHOD_DISPLAY, **SOTA_METHOD_DISPLAY}.get(method, method)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _load_optional_file(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return path.read_text(errors="ignore")


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(errors="ignore").strip()


def _violation_types(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded if str(item)]


def _percentile(values: list[float], q: float) -> float | None:
    finite = sorted(value for value in values if value == value)
    if not finite:
        return None
    index = min(len(finite) - 1, int(round(q * (len(finite) - 1))))
    return float(finite[index])


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def _fmt_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
