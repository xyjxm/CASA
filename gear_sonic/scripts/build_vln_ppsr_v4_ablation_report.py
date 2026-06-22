"""Build PPSR-v4 ablation reports from completed online evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


FULL_METHOD = "vln_ppsr_v4_zero_unsafe_success60"
ABLATION_METHODS = (
    "vln_ppsr_v4_ablate_no_stop_verifier",
    "vln_ppsr_v4_ablate_no_late_stop_recovery",
    "vln_ppsr_v4_ablate_no_visual_goal_tracker",
    "vln_ppsr_v4_ablate_no_task_return_replan",
)
METHOD_ORDER = (FULL_METHOD, *ABLATION_METHODS)
DEFAULT_FULL_ANCHOR_DIR = Path(
    "/mnt/data/students/lph/recording/vln_ppsr_v4_zero_unsafe_success60_20260621_locked_attempt04_lowratio_memory_seed260614"
)
DEFAULT_REPO_ARTIFACT_DIR = Path("idea_and_plan/vln_ppsr_v4_ablation_20260622")

COMPONENT_COLUMNS = [
    "method",
    "safe_success_count",
    "safe_success_rate",
    "unsafe_violation_count",
    "unsafe_violation_rate",
    "stop_precision",
    "stop_recall",
    "premature_stop_rate",
    "late_stop_rate",
    "missing_policy_stop_count",
    "collision_count",
    "fall_count",
    "stop_verifier_applied_count",
    "stop_verifier_bypassed_count",
    "late_stop_recovery_applied_count",
    "late_stop_recovery_candidate_count",
    "late_stop_recovery_disabled_count",
    "visual_goal_tracker_applied_count",
    "visual_goal_tracker_candidate_count",
    "visual_goal_tracker_disabled_count",
    "task_return_replan_applied_count",
    "task_return_replan_disabled_count",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-run-dir", type=Path, required=True)
    parser.add_argument("--full-anchor-dir", type=Path, default=DEFAULT_FULL_ANCHOR_DIR)
    parser.add_argument("--repo-artifact-dir", type=Path, default=DEFAULT_REPO_ARTIFACT_DIR)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--full-anchor-reused", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        ablation_run_dir=args.ablation_run_dir,
        full_anchor_dir=args.full_anchor_dir,
        repo_artifact_dir=args.repo_artifact_dir,
        expected_episodes=args.expected_episodes,
        full_anchor_reused=args.full_anchor_reused,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_report(
    *,
    ablation_run_dir: Path,
    full_anchor_dir: Path,
    repo_artifact_dir: Path,
    expected_episodes: int,
    full_anchor_reused: bool,
) -> dict[str, Any]:
    ablation_run_dir = ablation_run_dir.resolve()
    full_anchor_dir = full_anchor_dir.resolve()
    repo_artifact_dir = repo_artifact_dir.resolve()

    ablation_summary = _read_csv(ablation_run_dir / "data" / "method_summary.csv")
    full_summary_source = full_anchor_dir if full_anchor_reused or not _has_method(ablation_summary, FULL_METHOD) else ablation_run_dir
    full_summary = _read_csv(full_summary_source / "data" / "method_summary.csv")
    full_row = _require_method(full_summary, FULL_METHOD)
    ablation_rows = [_require_method(ablation_summary, method) for method in ABLATION_METHODS]

    for method, row in [(FULL_METHOD, full_row), *zip(ABLATION_METHODS, ablation_rows)]:
        total = _int(row.get("total_episodes"))
        if total != expected_episodes:
            raise RuntimeError(f"{method} has {total} episodes; expected {expected_episodes}")

    full_data = _load_auxiliary(full_summary_source)
    ablation_data = _load_auxiliary(ablation_run_dir)
    summary_rows = [_with_source(full_row, full_summary_source, "reused_full_anchor" if full_summary_source != ablation_run_dir else "same_run")]
    summary_rows.extend(_with_source(row, ablation_run_dir, "ablation_run") for row in ablation_rows)

    component_rows = [
        _component_row(full_row, full_data, FULL_METHOD),
        *[_component_row(row, ablation_data, str(row["method"])) for row in ablation_rows],
    ]
    delta_rows = [_delta_row(row, component_rows[0]) for row in component_rows[1:]]

    output_files = {
        "ablation_method_summary.csv": ablation_run_dir / "ablation_method_summary.csv",
        "component_contribution_table.csv": ablation_run_dir / "component_contribution_table.csv",
        "contribution_delta_vs_full.csv": ablation_run_dir / "contribution_delta_vs_full.csv",
        "ablation_result_summary.json": ablation_run_dir / "ablation_result_summary.json",
        "ablation_manifest.json": ablation_run_dir / "ablation_manifest.json",
        "ablation_report.md": ablation_run_dir / "ablation_report.md",
    }

    _write_csv(output_files["ablation_method_summary.csv"], summary_rows)
    _write_csv(output_files["component_contribution_table.csv"], component_rows, COMPONENT_COLUMNS)
    _write_csv(output_files["contribution_delta_vs_full.csv"], delta_rows)

    result_summary = {
        "full_anchor_reused": full_summary_source != ablation_run_dir,
        "full_anchor_dir": str(full_summary_source),
        "ablation_run_dir": str(ablation_run_dir),
        "expected_episodes_per_method": expected_episodes,
        "methods": {row["method"]: row for row in component_rows},
        "delta_vs_full": delta_rows,
        "conclusion": _component_conclusions(component_rows, delta_rows),
    }
    _write_json(output_files["ablation_result_summary.json"], result_summary)

    manifest = {
        "artifact": "vln_ppsr_v4_ablation_20260622",
        "ablation_run_dir": str(ablation_run_dir),
        "full_anchor_dir": str(full_summary_source),
        "full_anchor_reused": full_summary_source != ablation_run_dir,
        "expected_episodes_per_method": expected_episodes,
        "methods": list(METHOD_ORDER),
        "protocol": {
            "policy_backend": _manifest_value(ablation_run_dir, "policy_backend"),
            "eval_stage": _manifest_value(ablation_run_dir, "eval_stage"),
            "heldout_episodes": expected_episodes,
            "max_steps": _manifest_value(ablation_run_dir, "max_steps"),
            "seeds": _manifest_value(ablation_run_dir, "seeds"),
            "same_safety_oracle": True,
            "same_language_tasks_and_splits": True,
        },
        "video_paths": _video_paths(ablation_run_dir),
        "frames_path": str(ablation_run_dir / "frames"),
        "full_anchor_video_paths": _video_paths(full_summary_source, method=FULL_METHOD),
        "full_anchor_frames_path": str(full_summary_source / "frames" / FULL_METHOD),
        "large_outputs_not_in_git": True,
    }
    _write_json(output_files["ablation_manifest.json"], manifest)

    output_files["ablation_report.md"].write_text(_markdown_report(result_summary, manifest), encoding="utf-8")

    repo_artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, path in output_files.items():
        shutil.copy2(path, repo_artifact_dir / name)
    (repo_artifact_dir / "README.md").write_text(_repo_readme(result_summary, manifest), encoding="utf-8")
    _write_sha256s(repo_artifact_dir)

    return {
        "ablation_run_dir": str(ablation_run_dir),
        "repo_artifact_dir": str(repo_artifact_dir),
        "output_files": {name: str(path) for name, path in output_files.items()},
    }


def _load_auxiliary(run_dir: Path) -> dict[str, Any]:
    return {
        "episodes": _read_jsonl(run_dir / "data" / "episode_summaries.jsonl"),
        "decisions": _read_csv(run_dir / "data" / "decision_logs.csv"),
        "candidates": _read_csv(run_dir / "data" / "dmps_candidate_sequences.csv"),
        "selected": _read_csv(run_dir / "data" / "dmps_selected_replans.csv"),
        "manifest": _read_json(run_dir / "run_manifest.json"),
    }


def _component_row(summary: dict[str, Any], aux: dict[str, Any], method: str) -> dict[str, Any]:
    episodes = [row for row in aux["episodes"] if row.get("method") == method]
    decisions = [row for row in aux["decisions"] if row.get("method") == method]
    candidates = [row for row in aux["candidates"] if row.get("method") == method]
    selected = [row for row in aux["selected"] if row.get("method") == method]
    return {
        "method": method,
        "safe_success_count": _int(summary.get("safe_success_count")),
        "safe_success_rate": _float(summary.get("safe_success_rate")),
        "unsafe_violation_count": _int(summary.get("unsafe_violation_count")),
        "unsafe_violation_rate": _float(summary.get("unsafe_violation_rate")),
        "stop_precision": _float(summary.get("stop_precision")),
        "stop_recall": _float(summary.get("stop_recall")),
        "premature_stop_rate": _float(summary.get("premature_stop_rate")),
        "late_stop_rate": _float(summary.get("late_stop_rate")),
        "missing_policy_stop_count": sum(1 for row in episodes if row.get("failure_reason") == "missing_policy_stop"),
        "collision_count": _int(summary.get("collision_count")),
        "fall_count": _int(summary.get("fall_count")),
        "stop_verifier_applied_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_stop_verifier_applied"),
        "stop_verifier_bypassed_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_stop_verifier_bypassed"),
        "late_stop_recovery_applied_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_late_stop_recovery_applied"),
        "late_stop_recovery_candidate_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_late_stop_recovery_candidate"),
        "late_stop_recovery_disabled_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_late_stop_recovery_disabled"),
        "visual_goal_tracker_applied_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_visual_goal_tracker_applied"),
        "visual_goal_tracker_candidate_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_visual_goal_tracker_candidate"),
        "visual_goal_tracker_disabled_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_visual_goal_tracker_disabled"),
        "task_return_replan_applied_count": _task_return_applied_count(summary, decisions, candidates, selected),
        "task_return_replan_disabled_count": _summary_or_decision_count(summary, decisions, "ppsr_v4_task_return_replan_disabled"),
    }


def _summary_or_decision_count(summary: dict[str, Any], decisions: list[dict[str, Any]], key: str) -> int:
    summary_key = key if key.endswith("_count") else f"{key}_count"
    if summary_key in summary and str(summary.get(summary_key, "")) != "":
        return _int(summary.get(summary_key))
    if key == "ppsr_v4_visual_goal_tracker_candidate":
        return sum(1 for row in decisions if _metadata_value(row, "ppsr_v4_visual_goal_tracker_candidate_action") not in {"", None})
    return sum(1 for row in decisions if _bool_field(row, key))


def _task_return_applied_count(
    summary: dict[str, Any],
    decisions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> int:
    if "ppsr_v4_task_return_replan_applied_count" in summary:
        return _int(summary.get("ppsr_v4_task_return_replan_applied_count"))
    selected_candidate_count = sum(
        1
        for row in candidates
        if _truthy(row.get("selected")) and _truthy(row.get("task_return_candidate"))
    )
    if selected_candidate_count:
        return selected_candidate_count
    selected_reason_count = sum(1 for row in selected if "task_return" in str(row.get("selected_sequence") or row.get("selection_reason") or ""))
    if selected_reason_count:
        return selected_reason_count
    return sum(1 for row in decisions if _bool_field(row, "ppsr_v4_task_return_replan_applied"))


def _delta_row(row: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "safe_success_rate_delta_vs_full": _float(row["safe_success_rate"]) - _float(full["safe_success_rate"]),
        "unsafe_rate_delta_vs_full": _float(row["unsafe_violation_rate"]) - _float(full["unsafe_violation_rate"]),
        "stop_recall_delta_vs_full": _float(row["stop_recall"]) - _float(full["stop_recall"]),
        "premature_stop_delta_vs_full": _float(row["premature_stop_rate"]) - _float(full["premature_stop_rate"]),
    }


def _component_conclusions(component_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> list[str]:
    deltas = {row["method"]: row for row in delta_rows}
    conclusions = []
    labels = {
        "vln_ppsr_v4_ablate_no_stop_verifier": "stop verifier",
        "vln_ppsr_v4_ablate_no_late_stop_recovery": "late-stop recovery",
        "vln_ppsr_v4_ablate_no_visual_goal_tracker": "visual goal tracker",
        "vln_ppsr_v4_ablate_no_task_return_replan": "task-return replan",
    }
    for method, label in labels.items():
        delta = deltas.get(method, {})
        conclusions.append(
            f"{label}: safe_success_delta={_float(delta.get('safe_success_rate_delta_vs_full')):.3f}, "
            f"unsafe_delta={_float(delta.get('unsafe_rate_delta_vs_full')):.3f}, "
            f"stop_recall_delta={_float(delta.get('stop_recall_delta_vs_full')):.3f}, "
            f"premature_stop_delta={_float(delta.get('premature_stop_delta_vs_full')):.3f}"
        )
    return conclusions


def _markdown_report(result_summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    rows = list(result_summary["methods"].values())
    delta_rows = result_summary["delta_vs_full"]
    lines = [
        "# PPSR-v4 Ablation Report",
        "",
        f"- ablation_run_dir: `{result_summary['ablation_run_dir']}`",
        f"- full_anchor_dir: `{result_summary['full_anchor_dir']}`",
        f"- full_anchor_reused: `{result_summary['full_anchor_reused']}`",
        f"- policy_backend: `{manifest['protocol'].get('policy_backend')}`",
        f"- eval_stage: `{manifest['protocol'].get('eval_stage')}`",
        f"- heldout_episodes_per_method: `{manifest['expected_episodes_per_method']}`",
        f"- max_steps: `{manifest['protocol'].get('max_steps')}`",
        f"- seeds: `{manifest['protocol'].get('seeds')}`",
        "",
        "## Component Table",
        "",
        "| method | safe_success | unsafe | stop_recall | premature_stop | late_stop | stop_verifier_applied | late_applied/candidate/disabled | visual_applied/candidate/disabled | task_return_applied/disabled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['safe_success_count']}/{manifest['expected_episodes_per_method']} "
            f"({row['safe_success_rate']:.3f}) | {row['unsafe_violation_count']} ({row['unsafe_violation_rate']:.3f}) | "
            f"{row['stop_recall']:.3f} | {row['premature_stop_rate']:.3f} | {row['late_stop_rate']:.3f} | "
            f"{row['stop_verifier_applied_count']} | "
            f"{row['late_stop_recovery_applied_count']}/{row['late_stop_recovery_candidate_count']}/{row['late_stop_recovery_disabled_count']} | "
            f"{row['visual_goal_tracker_applied_count']}/{row['visual_goal_tracker_candidate_count']}/{row['visual_goal_tracker_disabled_count']} | "
            f"{row['task_return_replan_applied_count']}/{row['task_return_replan_disabled_count']} |"
        )
    lines.extend(["", "## Delta vs Full", "", "| method | safe_success_delta | unsafe_delta | stop_recall_delta | premature_stop_delta |", "|---|---:|---:|---:|---:|"])
    for row in delta_rows:
        lines.append(
            f"| {row['method']} | {row['safe_success_rate_delta_vs_full']:.3f} | "
            f"{row['unsafe_rate_delta_vs_full']:.3f} | {row['stop_recall_delta_vs_full']:.3f} | "
            f"{row['premature_stop_delta_vs_full']:.3f} |"
        )
    lines.extend(["", "## Conclusions", ""])
    lines.extend(f"- {line}" for line in result_summary["conclusion"])
    lines.extend(["", "## Large Artifacts", "", f"- videos_path: `{Path(result_summary['ablation_run_dir']) / 'videos'}`", f"- frames_path: `{Path(result_summary['ablation_run_dir']) / 'frames'}`"])
    return "\n".join(lines) + "\n"


def _repo_readme(result_summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# PPSR-v4 Ablation 20260622",
            "",
            f"- ablation_run_dir: `{result_summary['ablation_run_dir']}`",
            f"- full_anchor_reused: `{result_summary['full_anchor_reused']}`",
            f"- full_anchor_dir: `{result_summary['full_anchor_dir']}`",
            f"- expected_episodes_per_method: `{manifest['expected_episodes_per_method']}`",
            "",
            "Lightweight CSV/JSON/Markdown artifacts are mirrored here. Videos and frames stay under `/mnt/data/students/lph/recording/`.",
        ]
    ) + "\n"


def _with_source(row: dict[str, Any], source_dir: Path, source_label: str) -> dict[str, Any]:
    return {**row, "artifact_source": source_label, "artifact_source_dir": str(source_dir)}


def _manifest_value(run_dir: Path, key: str) -> Any:
    manifest = _read_json(run_dir / "run_manifest.json")
    return manifest.get(key)


def _video_paths(run_dir: Path, method: str | None = None) -> list[str]:
    manifest = _read_json(run_dir / "run_manifest.json")
    paths = [str(path) for path in manifest.get("video_paths", [])]
    if method is not None:
        needle = f"/{method}/"
        paths = [path for path in paths if needle in path]
    if paths:
        return paths
    videos_dir = run_dir / "videos"
    if method is not None:
        videos_dir = videos_dir / method
    return [str(path) for path in sorted(videos_dir.glob("**/*.mp4"))]


def _metadata_value(row: dict[str, Any], key: str) -> Any:
    if key in row and row.get(key) not in {"", None}:
        return row.get(key)
    metadata = _parse_json(row.get("policy_metadata_json") or row.get("policy_metadata"))
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _bool_field(row: dict[str, Any], key: str) -> bool:
    return _truthy(row.get(key)) or _truthy(_metadata_value(row, key))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes"}


def _int(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in {None, ""}:
        return None
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return None


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _has_method(rows: list[dict[str, Any]], method: str) -> bool:
    return any(row.get("method") == method for row in rows)


def _require_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    raise RuntimeError(f"missing method in summary: {method}")


def _write_sha256s(directory: Path) -> None:
    lines = []
    for path in sorted(directory.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
