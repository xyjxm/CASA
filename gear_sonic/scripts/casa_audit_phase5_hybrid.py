"""Audit CASA-Hybrid held-out online results against its anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5_online import method_summary_rows, read_csv_rows  # noqa: E402


DEFAULT_ANCHORS = "mpc_cbf_humanoid_adapted,safedpa_adapted,casa_a_per_skill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hybrid-method", default="casa_h_mpc_safedpa_casa_refine")
    parser.add_argument("--anchor-methods", default=DEFAULT_ANCHORS)
    parser.add_argument("--intervention-tolerance-rel", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.online_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = read_csv_rows(args.online_dir / "online_episode_results.csv")
    decision_rows = read_csv_rows(args.online_dir / "gate_decisions.csv")
    summary_rows = method_summary_rows(episode_rows, decision_rows)
    anchors = [item.strip() for item in args.anchor_methods.split(",") if item.strip()]
    report = build_hybrid_audit(
        summary_rows,
        hybrid_method=args.hybrid_method,
        anchor_methods=anchors,
        intervention_tolerance_rel=args.intervention_tolerance_rel,
        online_dir=args.online_dir,
    )
    (output_dir / "hybrid_anchor_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "hybrid_anchor_audit.md").write_text(_markdown(report) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["matched_intervention_budget_passes_all_anchors"]:
        raise SystemExit(2)


def build_hybrid_audit(
    summary_rows: list[dict[str, Any]],
    *,
    hybrid_method: str,
    anchor_methods: list[str],
    intervention_tolerance_rel: float,
    online_dir: Path | None = None,
) -> dict[str, Any]:
    summary = {row["method"]: row for row in summary_rows}
    hybrid = summary.get(hybrid_method)
    if not hybrid:
        raise ValueError(f"Missing hybrid method {hybrid_method!r} in method_summary")
    comparisons = {}
    for anchor in anchor_methods:
        comparisons[anchor] = _compare(hybrid, summary.get(anchor, {}), anchor, intervention_tolerance_rel)
    pareto = _pareto(summary_rows)
    beats = {anchor: comparisons[anchor]["hybrid_strictly_exceeds_anchor"] for anchor in anchor_methods}
    matched = {
        anchor: comparisons[anchor]["matched_intervention_budget_pass"]
        for anchor in anchor_methods
    }
    only_more = {
        anchor: comparisons[anchor]["hybrid_win_depends_on_more_intervention"]
        for anchor in anchor_methods
    }
    conclusion = _conclusion(hybrid_method, comparisons)
    return {
        "phase": "CASA-Hybrid Phase5 held-out online anchor audit",
        "online_dir": "" if online_dir is None else str(online_dir),
        "hybrid_method": hybrid_method,
        "anchor_methods": anchor_methods,
        "definition_of_exceeds": (
            "lower unsafe_invocation_rate_per_episode and no lower task_progress_success_rate"
        ),
        "hybrid_beats_mpc_cbf_humanoid_adapted": beats.get("mpc_cbf_humanoid_adapted", False),
        "hybrid_beats_safedpa_adapted": beats.get("safedpa_adapted", False),
        "hybrid_beats_casa_a_per_skill": beats.get("casa_a_per_skill", False),
        "matched_intervention_budget_passes_all_anchors": all(matched.values()) if matched else False,
        "hybrid_win_depends_on_more_fallback_or_reject": any(only_more.values()) if only_more else False,
        "comparisons": comparisons,
        "pareto_frontier": pareto,
        "final_audit_conclusion": conclusion,
    }


def _compare(
    hybrid: dict[str, Any],
    anchor: dict[str, Any],
    anchor_method: str,
    tolerance_rel: float,
) -> dict[str, Any]:
    if not anchor:
        return {
            "anchor_method": anchor_method,
            "status": "missing_anchor",
            "hybrid_strictly_exceeds_anchor": False,
            "matched_intervention_budget_pass": False,
            "hybrid_win_depends_on_more_intervention": False,
        }
    h_unsafe = _float(hybrid.get("unsafe_invocation_rate_per_episode"))
    a_unsafe = _float(anchor.get("unsafe_invocation_rate_per_episode"))
    h_progress = _float_or_none(hybrid.get("task_progress_success_rate"))
    a_progress = _float_or_none(anchor.get("task_progress_success_rate"))
    if h_progress is None:
        h_progress = _float_or_none(hybrid.get("task_success_rate"))
    if a_progress is None:
        a_progress = _float_or_none(anchor.get("task_success_rate"))
    h_reject = _float_or_none(hybrid.get("reject_rate_per_decision"))
    a_reject = _float_or_none(anchor.get("reject_rate_per_decision"))
    h_fallback = _float(hybrid.get("fallback_rate_per_episode"))
    a_fallback = _float(anchor.get("fallback_rate_per_episode"))
    reject_matched = _matched(h_reject, a_reject, tolerance_rel)
    fallback_matched = h_fallback <= a_fallback + tolerance_rel * max(1.0, abs(a_fallback))
    strict_exceeds = h_unsafe < a_unsafe and h_progress is not None and a_progress is not None and h_progress >= a_progress
    more_intervention = (not reject_matched) or (not fallback_matched)
    return {
        "anchor_method": anchor_method,
        "hybrid_unsafe_invocation_rate_per_episode": h_unsafe,
        "anchor_unsafe_invocation_rate_per_episode": a_unsafe,
        "hybrid_task_progress_success_rate": h_progress,
        "anchor_task_progress_success_rate": a_progress,
        "hybrid_fallback_rate_per_episode": h_fallback,
        "anchor_fallback_rate_per_episode": a_fallback,
        "hybrid_reject_rate_per_decision": h_reject,
        "anchor_reject_rate_per_decision": a_reject,
        "reject_budget_matched": reject_matched,
        "fallback_budget_matched": fallback_matched,
        "hybrid_strictly_exceeds_anchor": strict_exceeds,
        "matched_intervention_budget_pass": strict_exceeds and reject_matched and fallback_matched,
        "hybrid_win_depends_on_more_intervention": strict_exceeds and more_intervention,
    }


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "method": row.get("method", ""),
                "unsafe_invocation_rate_per_episode": _float(row.get("unsafe_invocation_rate_per_episode")),
                "task_progress_success_rate": _float_or_none(row.get("task_progress_success_rate")),
                "fallback_rate_per_episode": _float(row.get("fallback_rate_per_episode")),
                "reject_rate_per_decision": _float_or_none(row.get("reject_rate_per_decision")),
            }
        )
    frontier = []
    for row in compact:
        dominated = False
        for other in compact:
            if other is row:
                continue
            if _dominates(other, row):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(
        frontier,
        key=lambda row: (
            row["unsafe_invocation_rate_per_episode"],
            -_float(row.get("task_progress_success_rate")),
            row["fallback_rate_per_episode"],
        ),
    )


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_progress = _float(left.get("task_progress_success_rate"))
    right_progress = _float(right.get("task_progress_success_rate"))
    left_reject = _float(left.get("reject_rate_per_decision"))
    right_reject = _float(right.get("reject_rate_per_decision"))
    not_worse = (
        left["unsafe_invocation_rate_per_episode"] <= right["unsafe_invocation_rate_per_episode"]
        and left_progress >= right_progress
        and left["fallback_rate_per_episode"] <= right["fallback_rate_per_episode"]
        and left_reject <= right_reject
    )
    strictly_better = (
        left["unsafe_invocation_rate_per_episode"] < right["unsafe_invocation_rate_per_episode"]
        or left_progress > right_progress
        or left["fallback_rate_per_episode"] < right["fallback_rate_per_episode"]
        or left_reject < right_reject
    )
    return not_worse and strictly_better


def _conclusion(hybrid_method: str, comparisons: dict[str, Any]) -> str:
    missing = [name for name, row in comparisons.items() if row.get("status") == "missing_anchor"]
    if missing:
        return f"NO_CLAIM: missing anchor methods {missing}."
    beats = [name for name, row in comparisons.items() if row.get("hybrid_strictly_exceeds_anchor")]
    matched = [name for name, row in comparisons.items() if row.get("matched_intervention_budget_pass")]
    if len(beats) == len(comparisons) and len(matched) == len(comparisons):
        return f"PASS: {hybrid_method} exceeds all anchors at matched intervention budget."
    if beats:
        return (
            f"PARTIAL: {hybrid_method} exceeds {beats}, but matched-budget or remaining-anchor checks do not all pass."
        )
    return f"NO_SUCCESS_CLAIM: {hybrid_method} does not strictly exceed the anchors; use the Pareto frontier."


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CASA-Hybrid Anchor Audit",
        "",
        f"- hybrid_method: `{report['hybrid_method']}`",
        f"- conclusion: {report['final_audit_conclusion']}",
        f"- beats MPC-CBF-Humanoid: `{report['hybrid_beats_mpc_cbf_humanoid_adapted']}`",
        f"- beats SafeDPA: `{report['hybrid_beats_safedpa_adapted']}`",
        f"- beats CASA-A: `{report['hybrid_beats_casa_a_per_skill']}`",
        f"- matched intervention budget passes all anchors: `{report['matched_intervention_budget_passes_all_anchors']}`",
        f"- win depends on more fallback/reject: `{report['hybrid_win_depends_on_more_fallback_or_reject']}`",
        "",
        "## Anchor Comparisons",
        "",
        "| anchor | exceeds | matched budget | more intervention win | unsafe hybrid/anchor | progress hybrid/anchor | fallback hybrid/anchor | reject hybrid/anchor |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for anchor, row in report["comparisons"].items():
        lines.append(
            f"| `{anchor}` | {row.get('hybrid_strictly_exceeds_anchor')} | "
            f"{row.get('matched_intervention_budget_pass')} | "
            f"{row.get('hybrid_win_depends_on_more_intervention')} | "
            f"{_fmt(row.get('hybrid_unsafe_invocation_rate_per_episode'))}/"
            f"{_fmt(row.get('anchor_unsafe_invocation_rate_per_episode'))} | "
            f"{_fmt(row.get('hybrid_task_progress_success_rate'))}/"
            f"{_fmt(row.get('anchor_task_progress_success_rate'))} | "
            f"{_fmt(row.get('hybrid_fallback_rate_per_episode'))}/"
            f"{_fmt(row.get('anchor_fallback_rate_per_episode'))} | "
            f"{_fmt(row.get('hybrid_reject_rate_per_decision'))}/"
            f"{_fmt(row.get('anchor_reject_rate_per_decision'))} |"
        )
    lines.extend(["", "## Pareto Frontier", ""])
    lines.append("| method | unsafe/ep | progress | fallback/ep | reject/decision |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report["pareto_frontier"]:
        lines.append(
            f"| `{row['method']}` | {_fmt(row.get('unsafe_invocation_rate_per_episode'))} | "
            f"{_fmt(row.get('task_progress_success_rate'))} | "
            f"{_fmt(row.get('fallback_rate_per_episode'))} | "
            f"{_fmt(row.get('reject_rate_per_decision'))} |"
        )
    return "\n".join(lines)


def _matched(left: float | None, right: float | None, tolerance_rel: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left <= right + tolerance_rel * max(1.0, abs(right))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    number = _float_or_none(value)
    return 0.0 if number is None else number


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    return "n/a" if number is None else f"{number:.4f}"


if __name__ == "__main__":
    main()
