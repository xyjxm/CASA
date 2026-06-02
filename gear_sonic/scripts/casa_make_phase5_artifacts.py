"""Generate CASA Phase 5 visual artifacts from audited CSV/JSON outputs.

The script intentionally avoids plotting dependencies. It writes SVG charts with
labels and uses ffmpeg for PNG snapshots and MP4 trajectory rendering.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_COUNTERFACTUAL = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase3_feasibility/phase3_hybrid_phase2v2_manifest_20260521/"
    "counterfactual/counterfactual_summary.json"
)
DEFAULT_FAILURE_BREAKDOWN = Path(
    "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/"
    "phase5_conformal_baselines_20260522/"
    "issue7_heldout_hard_or_2500_20260531_verification/"
    "diagnostics/phase5_online_failure_breakdown.json"
)

SKILL_ORDER = ["walk", "turn", "gesture", "passive"]
METHOD_COLORS = {
    "sonic_only": "#4b5563",
    "hard_contract": "#0f766e",
    "raw_critic_0p5": "#7c3aed",
    "global_conformal": "#dc2626",
    "casa_a_per_skill": "#2563eb",
    "raw_critic_budgeted": "#7c3aed",
    "hard_contract_budgeted": "#0f766e",
}
METHOD_LABELS = {
    "sonic_only": "SONIC",
    "hard_contract": "Hard Contract",
    "raw_critic_0p5": "Raw Critic",
    "global_conformal": "Global Conformal",
    "casa_a_per_skill": "CASA-A",
    "raw_critic_budgeted": "Raw Critic",
    "hard_contract_budgeted": "Hard Contract",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--counterfactual-summary", type=Path, default=DEFAULT_COUNTERFACTUAL)
    parser.add_argument("--failure-breakdown", type=Path, default=DEFAULT_FAILURE_BREAKDOWN)
    parser.add_argument("--demo-episode-dir", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-video-frames", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase5_root = args.phase5_root
    output_dir = args.output_dir or phase5_root / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which(args.ffmpeg) or args.ffmpeg
    artifacts: list[dict[str, Any]] = []

    predictions_csv = args.predictions_csv or _default_predictions_csv(phase5_root)
    if predictions_csv.exists():
        artifacts.extend(_make_reliability_artifacts(predictions_csv, output_dir, ffmpeg))

    curve_csv = phase5_root / "rejection_risk_curve.csv"
    if curve_csv.exists():
        artifacts.extend(_make_rejection_curve_artifacts(curve_csv, output_dir, ffmpeg))

    fnr_csv = phase5_root / "per_skill_fnr.csv"
    if fnr_csv.exists():
        artifacts.extend(_make_per_skill_fnr_artifacts(fnr_csv, output_dir, ffmpeg))

    if args.counterfactual_summary and args.counterfactual_summary.exists():
        artifacts.extend(_make_counterfactual_artifacts(args.counterfactual_summary, output_dir, ffmpeg))

    if args.failure_breakdown and args.failure_breakdown.exists():
        artifacts.extend(_make_failure_artifacts(args.failure_breakdown, output_dir, ffmpeg))

    episode_dir = args.demo_episode_dir or _find_demo_episode(phase5_root)
    if episode_dir:
        artifacts.extend(
            _make_demo_video_artifacts(
                episode_dir=episode_dir,
                output_dir=output_dir,
                ffmpeg=ffmpeg,
                fps=args.fps,
                max_frames=args.max_video_frames,
            )
        )

    manifest = {
        "phase": "CASA Phase5 visual artifacts",
        "phase5_root": str(phase5_root),
        "output_dir": str(output_dir),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "phase5_artifact_manifest.json", manifest)
    (output_dir / "phase5_artifacts.md").write_text(_artifact_markdown(manifest) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _default_predictions_csv(phase5_root: Path) -> Path:
    hc = phase5_root / "hc_filtered_online_calibration_predictions.csv"
    if hc.exists():
        return hc
    return phase5_root.parent / "phase4_dataset_v1_strict_50k_20260522" / "raw_critic" / "predictions.csv"


def _make_reliability_artifacts(predictions_csv: Path, output_dir: Path, ffmpeg: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _read_csv(predictions_csv)
        if row.get("phase4_split") == "test" and _float(row.get("raw_critic_risk")) is not None
    ]
    bins = [{"lo": i / 10.0, "hi": (i + 1) / 10.0, "count": 0, "unsafe": 0, "risk_sum": 0.0} for i in range(10)]
    for row in rows:
        risk = _float(row.get("raw_critic_risk"))
        if risk is None:
            continue
        idx = min(9, max(0, int(risk * 10)))
        bins[idx]["count"] += 1
        bins[idx]["unsafe"] += int(_label(row) == 1)
        bins[idx]["risk_sum"] += risk
    series = []
    for item in bins:
        count = item["count"]
        empirical = item["unsafe"] / count if count else 0.0
        mean_risk = item["risk_sum"] / count if count else (item["lo"] + item["hi"]) / 2.0
        series.append(
            {
                "label": f"{item['lo']:.1f}-{item['hi']:.1f}",
                "x": mean_risk,
                "y": empirical,
                "count": count,
            }
        )
    svg_path = output_dir / "reliability_diagram.svg"
    png_path = output_dir / "reliability_diagram.png"
    _write_svg(
        svg_path,
        _line_chart_svg(
            title="Reliability Diagram: Raw Critic Risk vs Empirical Unsafe Rate",
            subtitle=f"Source: {predictions_csv.name}; test rows={len(rows)}",
            series=[
                {"label": "ideal", "color": "#9ca3af", "points": [(0.0, 0.0), (1.0, 1.0)], "dash": "6 6"},
                {
                    "label": "raw critic bins",
                    "color": "#2563eb",
                    "points": [(item["x"], item["y"]) for item in series],
                    "labels": [f"n={item['count']}" for item in series],
                },
            ],
            x_label="mean predicted risk",
            y_label="empirical unsafe rate",
            x_max=1.0,
            y_max=1.0,
        ),
    )
    _svg_to_png(svg_path, png_path, ffmpeg)
    return [
        _artifact("reliability_diagram", svg_path, predictions_csv, "Raw-critic calibration reliability chart"),
        _artifact("reliability_diagram_png", png_path, predictions_csv, "PNG render of reliability diagram"),
    ]


def _make_rejection_curve_artifacts(curve_csv: Path, output_dir: Path, ffmpeg: str) -> list[dict[str, Any]]:
    rows = _read_csv(curve_csv)
    wanted = ["raw_critic_budgeted", "hard_contract_budgeted"]
    series = []
    for method in wanted:
        points = []
        for row in rows:
            if row.get("method") != method:
                continue
            budget = _float(row.get("reject_budget"))
            unsafe_rate = _float(row.get("accepted_unsafe_rate"))
            if budget is not None and unsafe_rate is not None:
                points.append((budget, unsafe_rate))
        points.sort()
        if points:
            series.append({"label": METHOD_LABELS.get(method, method), "color": METHOD_COLORS[method], "points": points})
    max_y = _nice_max([point[1] for item in series for point in item["points"]], floor=0.01)
    svg_path = output_dir / "rejection_risk_curve.svg"
    png_path = output_dir / "rejection_risk_curve.png"
    _write_svg(
        svg_path,
        _line_chart_svg(
            title="Rejection Budget vs Accepted Unsafe Rate",
            subtitle=f"Source: {curve_csv.name}",
            series=series,
            x_label="reject budget",
            y_label="accepted unsafe rate",
            x_max=0.5,
            y_max=max_y,
        ),
    )
    _svg_to_png(svg_path, png_path, ffmpeg)
    return [
        _artifact("rejection_risk_curve", svg_path, curve_csv, "Budgeted rejection risk curve"),
        _artifact("rejection_risk_curve_png", png_path, curve_csv, "PNG render of rejection risk curve"),
    ]


def _make_per_skill_fnr_artifacts(fnr_csv: Path, output_dir: Path, ffmpeg: str) -> list[dict[str, Any]]:
    rows = [row for row in _read_csv(fnr_csv) if row.get("split") == "test"]
    methods = ["hard_contract", "raw_critic_0p5", "global_conformal", "casa_a_per_skill"]
    bars = []
    for skill in SKILL_ORDER:
        for method in methods:
            match = next(
                (row for row in rows if row.get("skill_name") == skill and row.get("method") == method),
                None,
            )
            if match:
                bars.append(
                    {
                        "group": skill,
                        "label": METHOD_LABELS.get(method, method),
                        "method": method,
                        "value": _float(match.get("fnr")) or 0.0,
                    }
                )
    svg_path = output_dir / "per_skill_fnr.svg"
    png_path = output_dir / "per_skill_fnr.png"
    _write_svg(
        svg_path,
        _grouped_bar_svg(
            title="Per-skill False Negative Rate",
            subtitle=f"Source: {fnr_csv.name}",
            bars=bars,
            groups=SKILL_ORDER,
            methods=methods,
            y_label="FNR",
            y_max=1.0,
        ),
    )
    _svg_to_png(svg_path, png_path, ffmpeg)
    return [
        _artifact("per_skill_fnr", svg_path, fnr_csv, "Per-skill FNR grouped bar chart"),
        _artifact("per_skill_fnr_png", png_path, fnr_csv, "PNG render of per-skill FNR chart"),
    ]


def _make_counterfactual_artifacts(summary_json: Path, output_dir: Path, ffmpeg: str) -> list[dict[str, Any]]:
    data = _read_json(summary_json)
    risk_by_candidate = data.get("risk_by_candidate", {})
    cells = []
    for candidate, stats in sorted(risk_by_candidate.items()):
        cells.append(
            {
                "label": candidate,
                "value": float(stats.get("unsafe_rate", 0.0)),
                "unsafe": int(stats.get("unsafe", 0)),
                "branches": int(stats.get("branches", 0)),
            }
        )
    svg_path = output_dir / "counterfactual_risk_heatmap.svg"
    png_path = output_dir / "counterfactual_risk_heatmap.png"
    _write_svg(
        svg_path,
        _heatmap_svg(
            title="Counterfactual Candidate Unsafe Rate",
            subtitle=(
                f"Source: {summary_json.name}; states={data.get('observed_states')}; "
                f"branches={data.get('branch_count')}"
            ),
            cells=cells,
        ),
    )
    _svg_to_png(svg_path, png_path, ffmpeg)
    return [
        _artifact("counterfactual_risk_heatmap", svg_path, summary_json, "Counterfactual candidate risk heatmap"),
        _artifact("counterfactual_risk_heatmap_png", png_path, summary_json, "PNG render of counterfactual heatmap"),
    ]


def _make_failure_artifacts(breakdown_json: Path, output_dir: Path, ffmpeg: str) -> list[dict[str, Any]]:
    data = _read_json(breakdown_json)
    per_method = data.get("per_method", {})
    bars = []
    metrics = [
        ("unsafe_after_allow", "allow+unsafe"),
        ("unsafe_after_reject", "reject+unsafe"),
        ("fallback_count", "fallback"),
    ]
    for method, stats in per_method.items():
        for metric, label in metrics:
            bars.append(
                {
                    "group": method,
                    "label": label,
                    "method": metric,
                    "value": float(stats.get(metric, 0)),
                }
            )
    groups = sorted(per_method)
    svg_path = output_dir / "failure_case_summary.svg"
    png_path = output_dir / "failure_case_summary.png"
    _write_svg(
        svg_path,
        _grouped_bar_svg(
            title="Online Failure Case Summary",
            subtitle=f"Source: {breakdown_json.name}; episodes={data.get('episode_count')}",
            bars=bars,
            groups=groups,
            methods=[metric for metric, _ in metrics],
            method_labels={metric: label for metric, label in metrics},
            method_colors={
                "unsafe_after_allow": "#dc2626",
                "unsafe_after_reject": "#ea580c",
                "fallback_count": "#2563eb",
            },
            y_label="count",
            y_max=_nice_max([item["value"] for item in bars], floor=1.0),
        ),
    )
    _svg_to_png(svg_path, png_path, ffmpeg)
    return [
        _artifact("failure_case_summary", svg_path, breakdown_json, "Online failure case grouped bar chart"),
        _artifact("failure_case_summary_png", png_path, breakdown_json, "PNG render of failure case summary"),
    ]


def _make_demo_video_artifacts(
    *,
    episode_dir: Path,
    output_dir: Path,
    ffmpeg: str,
    fps: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    sim_state_csv = episode_dir / "sim_log" / "sim_state.csv"
    skill_events_csv = episode_dir / "skill_events.csv"
    summary_json = episode_dir / "rollout_summary.json"
    if not sim_state_csv.exists():
        return []

    sim_rows = _sample_rows(_read_csv(sim_state_csv), max_frames)
    if not sim_rows:
        return []
    skills = _read_csv(skill_events_csv) if skill_events_csv.exists() else []
    summary = _read_json(summary_json) if summary_json.exists() else {}
    placements = summary.get("scene_props", {}).get("placements", {})
    decisions = _episode_gate_decisions(episode_dir, summary)

    video_path = output_dir / "demo_trajectory.mp4"
    snapshot_svg = output_dir / "demo_trajectory_snapshot.svg"
    snapshot_png = output_dir / "demo_trajectory_snapshot.png"
    with tempfile.TemporaryDirectory(prefix="casa_phase5_frames_") as tmp:
        frame_dir = Path(tmp)
        for idx, row in enumerate(sim_rows):
            _write_svg(
                frame_dir / f"frame_{idx:04d}.svg",
                _trajectory_frame_svg(
                    sim_rows=sim_rows,
                    frame_idx=idx,
                    skills=skills,
                    placements=placements,
                    decisions=decisions,
                    summary=summary,
                ),
            )
        shutil.copyfile(frame_dir / f"frame_{len(sim_rows) - 1:04d}.svg", snapshot_svg)
        _svg_to_png(snapshot_svg, snapshot_png, ffmpeg)
        _run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%04d.svg"),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ]
        )
    return [
        _artifact("demo_trajectory_video", video_path, sim_state_csv, "MP4 top-down trajectory video from real Phase5 sim_state"),
        _artifact("demo_trajectory_snapshot", snapshot_svg, sim_state_csv, "Final frame SVG from the trajectory video"),
        _artifact("demo_trajectory_snapshot_png", snapshot_png, sim_state_csv, "Final frame PNG from the trajectory video"),
    ]


def _find_demo_episode(phase5_root: Path) -> Path | None:
    preferred = (
        phase5_root.parent
        / "issue7_pilot_hard_or_50ep_20260531"
        / "lane_v2_casa_a_hard_or_recovery__seed_9003__ep_000_005"
        / "online"
        / "casa_a_hard_or_recovery"
        / "seed_9003"
        / "episode_0000"
    )
    if (preferred / "sim_log" / "sim_state.csv").exists():
        return preferred
    roots = [phase5_root, phase5_root.parent]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for csv_path in root.rglob("sim_log/sim_state.csv"):
            candidates.append(csv_path.parents[1])
            if len(candidates) > 200:
                break
        if candidates:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("casa" not in str(path), "issue7" not in str(path), str(path)))
    return candidates[0]


def _episode_gate_decisions(episode_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    online_dir = _find_parent_named(episode_dir, "online")
    if online_dir is None:
        return []
    gate_csv = online_dir / "gate_decisions.csv"
    if not gate_csv.exists():
        return []
    episode_id = summary.get("rollout_id") or episode_dir.name
    rows = []
    for row in _read_csv(gate_csv):
        if row.get("episode_id") == episode_id:
            rows.append(row)
    return rows


def _find_parent_named(path: Path, name: str) -> Path | None:
    for parent in [path, *path.parents]:
        if parent.name == name:
            return parent
    return None


def _line_chart_svg(
    *,
    title: str,
    subtitle: str,
    series: list[dict[str, Any]],
    x_label: str,
    y_label: str,
    x_max: float,
    y_max: float,
) -> str:
    width, height = 1100, 720
    left, right, top, bottom = 110, 70, 105, 95
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (max(0.0, min(x_max, x)) / x_max) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (max(0.0, min(y_max, y)) / y_max) * plot_h

    parts = _svg_header(width, height, title, subtitle)
    parts.append(_chart_axes(left, top, plot_w, plot_h, x_label, y_label, x_max, y_max))
    legend_x, legend_y = left + plot_w - 260, top + 5
    for idx, item in enumerate(series):
        color = item.get("color", "#2563eb")
        label = item.get("label", f"series {idx + 1}")
        dash = f" stroke-dasharray=\"{_esc(item.get('dash', ''))}\"" if item.get("dash") else ""
        points = item.get("points", [])
        if points:
            path_d = " ".join(f"{'M' if i == 0 else 'L'} {sx(x):.2f} {sy(y):.2f}" for i, (x, y) in enumerate(points))
            parts.append(f"<path d=\"{path_d}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"4\"{dash}/>")
            labels = item.get("labels") or []
            for point_idx, (x, y) in enumerate(points):
                px, py = sx(x), sy(y)
                parts.append(f"<circle cx=\"{px:.2f}\" cy=\"{py:.2f}\" r=\"5\" fill=\"{color}\"/>")
                if point_idx < len(labels) and labels[point_idx]:
                    parts.append(
                        f"<text x=\"{px + 7:.2f}\" y=\"{py - 7:.2f}\" "
                        f"class=\"small muted\">{_esc(labels[point_idx])}</text>"
                    )
        parts.append(
            f"<rect x=\"{legend_x}\" y=\"{legend_y + idx * 26}\" width=\"18\" height=\"4\" fill=\"{color}\"/>"
            f"<text x=\"{legend_x + 28}\" y=\"{legend_y + 8 + idx * 26}\" class=\"small\">{_esc(label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _grouped_bar_svg(
    *,
    title: str,
    subtitle: str,
    bars: list[dict[str, Any]],
    groups: list[str],
    methods: list[str],
    y_label: str,
    y_max: float,
    method_labels: dict[str, str] | None = None,
    method_colors: dict[str, str] | None = None,
) -> str:
    width, height = 1180, 720
    left, right, top, bottom = 115, 70, 105, 115
    plot_w = width - left - right
    plot_h = height - top - bottom
    method_labels = method_labels or METHOD_LABELS
    method_colors = method_colors or METHOD_COLORS
    by_key = {(bar["group"], bar["method"]): bar for bar in bars}
    group_w = plot_w / max(1, len(groups))
    bar_w = min(30.0, group_w / max(1, len(methods)) * 0.68)

    def sy(y: float) -> float:
        return top + plot_h - (max(0.0, min(y_max, y)) / y_max) * plot_h

    parts = _svg_header(width, height, title, subtitle)
    parts.append(_chart_axes(left, top, plot_w, plot_h, "skill/method", y_label, len(groups), y_max, x_ticks=False))
    for group_idx, group in enumerate(groups):
        center = left + group_idx * group_w + group_w / 2.0
        parts.append(f"<text x=\"{center:.2f}\" y=\"{top + plot_h + 36}\" text-anchor=\"middle\" class=\"axis\">{_esc(group)}</text>")
        for method_idx, method in enumerate(methods):
            bar = by_key.get((group, method))
            if not bar:
                continue
            x = center - (len(methods) * bar_w) / 2.0 + method_idx * bar_w + 2
            y = sy(float(bar["value"]))
            h = top + plot_h - y
            color = method_colors.get(method, "#2563eb")
            parts.append(f"<rect x=\"{x:.2f}\" y=\"{y:.2f}\" width=\"{bar_w - 4:.2f}\" height=\"{h:.2f}\" fill=\"{color}\"/>")
            parts.append(
                f"<text x=\"{x + (bar_w - 4) / 2:.2f}\" y=\"{max(top + 14, y - 6):.2f}\" "
                f"text-anchor=\"middle\" class=\"tiny\">{_fmt_short(float(bar['value']))}</text>"
            )
    legend_x = left
    legend_y = height - 38
    for idx, method in enumerate(methods):
        color = method_colors.get(method, "#2563eb")
        label = method_labels.get(method, method)
        x = legend_x + idx * 210
        parts.append(
            f"<rect x=\"{x}\" y=\"{legend_y - 11}\" width=\"14\" height=\"14\" fill=\"{color}\"/>"
            f"<text x=\"{x + 21}\" y=\"{legend_y}\" class=\"small\">{_esc(label)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _heatmap_svg(*, title: str, subtitle: str, cells: list[dict[str, Any]]) -> str:
    width, height = 1080, 520
    parts = _svg_header(width, height, title, subtitle)
    left, top = 90, 150
    cell_w = 210
    cell_h = 165
    max_value = max([cell["value"] for cell in cells] or [1.0])
    for idx, cell in enumerate(cells):
        x = left + idx * (cell_w + 30)
        value = cell["value"]
        intensity = 0.15 + 0.85 * (value / max_value if max_value else 0.0)
        color = _blend("#dbeafe", "#dc2626", intensity)
        parts.append(f"<rect x=\"{x}\" y=\"{top}\" width=\"{cell_w}\" height=\"{cell_h}\" rx=\"8\" fill=\"{color}\"/>")
        parts.append(f"<text x=\"{x + cell_w / 2}\" y=\"{top + 42}\" text-anchor=\"middle\" class=\"label\">{_esc(cell['label'])}</text>")
        parts.append(f"<text x=\"{x + cell_w / 2}\" y=\"{top + 93}\" text-anchor=\"middle\" class=\"metric\">{value:.3f}</text>")
        parts.append(
            f"<text x=\"{x + cell_w / 2}\" y=\"{top + 132}\" text-anchor=\"middle\" class=\"small\">"
            f"{cell['unsafe']} unsafe / {cell['branches']} branches</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _trajectory_frame_svg(
    *,
    sim_rows: list[dict[str, str]],
    frame_idx: int,
    skills: list[dict[str, str]],
    placements: dict[str, Any],
    decisions: list[dict[str, str]],
    summary: dict[str, Any],
) -> str:
    width, height = 1280, 720
    left, top, plot_w, plot_h = 70, 85, 760, 570
    current = sim_rows[frame_idx]
    trail = sim_rows[: frame_idx + 1]
    points = [(_float(row.get("base_pos_x")), _float(row.get("base_pos_y"))) for row in sim_rows]
    scene_points = []
    for placement in placements.values():
        if isinstance(placement, dict) and placement.get("enabled"):
            pos = placement.get("position") or []
            if len(pos) >= 2:
                scene_points.append((_float(pos[0]), _float(pos[1])))
    xs = [x for x, _ in points + scene_points if x is not None]
    ys = [y for _, y in points + scene_points if y is not None]
    x_min, x_max = _bounds(xs, pad=0.45)
    y_min, y_max = _bounds(ys, pad=0.45)
    if (x_max - x_min) < 1.0:
        x_min -= 0.5
        x_max += 0.5
    if (y_max - y_min) < 1.0:
        y_min -= 0.5
        y_max += 0.5

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = _svg_header(
        width,
        height,
        "CASA Phase5 Demo Trajectory",
        f"{summary.get('method_display', summary.get('method', 'online'))}; rollout={summary.get('rollout_id', '')}",
    )
    parts.append(f"<rect x=\"{left}\" y=\"{top}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>")
    for i in range(6):
        x = left + i * plot_w / 5
        y = top + i * plot_h / 5
        parts.append(f"<line x1=\"{x:.2f}\" y1=\"{top}\" x2=\"{x:.2f}\" y2=\"{top + plot_h}\" stroke=\"#e2e8f0\"/>")
        parts.append(f"<line x1=\"{left}\" y1=\"{y:.2f}\" x2=\"{left + plot_w}\" y2=\"{y:.2f}\" stroke=\"#e2e8f0\"/>")
    for name, placement in sorted(placements.items()):
        if not isinstance(placement, dict) or not placement.get("enabled"):
            continue
        pos = placement.get("position") or []
        if len(pos) < 2:
            continue
        px = _float(pos[0])
        py = _float(pos[1])
        if px is None or py is None:
            continue
        color = "#f97316" if name.startswith("obstacle") else "#dc2626"
        radius = 13 if name.startswith("obstacle") else 16
        parts.append(f"<circle cx=\"{sx(px):.2f}\" cy=\"{sy(py):.2f}\" r=\"{radius}\" fill=\"{color}\" opacity=\"0.88\"/>")
        parts.append(f"<text x=\"{sx(px) + 17:.2f}\" y=\"{sy(py) + 4:.2f}\" class=\"tiny\">{_esc(name)}</text>")
    trail_points = []
    for row in trail:
        x = _float(row.get("base_pos_x"))
        y = _float(row.get("base_pos_y"))
        if x is not None and y is not None:
            trail_points.append(f"{sx(x):.2f},{sy(y):.2f}")
    if len(trail_points) >= 2:
        parts.append(f"<polyline points=\"{' '.join(trail_points)}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"5\"/>")
    x = _float(current.get("base_pos_x")) or 0.0
    y = _float(current.get("base_pos_y")) or 0.0
    yaw = _float(current.get("torso_yaw")) or 0.0
    robot_x, robot_y = sx(x), sy(y)
    heading_x = robot_x + 28 * math.cos(yaw)
    heading_y = robot_y - 28 * math.sin(yaw)
    parts.append(f"<circle cx=\"{robot_x:.2f}\" cy=\"{robot_y:.2f}\" r=\"20\" fill=\"#1d4ed8\"/>")
    parts.append(f"<line x1=\"{robot_x:.2f}\" y1=\"{robot_y:.2f}\" x2=\"{heading_x:.2f}\" y2=\"{heading_y:.2f}\" stroke=\"#0f172a\" stroke-width=\"5\"/>")

    side_x = 875
    wall_time = _float(current.get("wall_time")) or 0.0
    sim_time = _float(current.get("sim_time")) or 0.0
    min_user = _float(current.get("min_user_distance"))
    min_obs = _float(current.get("min_obstacle_distance"))
    min_arm = _float(current.get("min_arm_user_distance"))
    skill = _active_skill(skills, wall_time)
    decision = _decision_for_skill(decisions, skill.get("skill_idx") if skill else None)
    parts.append(f"<rect x=\"{side_x}\" y=\"{top}\" width=\"345\" height=\"{plot_h}\" rx=\"8\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>")
    info = [
        ("sim_time", f"{sim_time:.2f}s"),
        ("wall_time", f"{wall_time:.2f}"),
        ("skill", skill.get("skill_name", "none") if skill else "none"),
        ("skill_status", skill.get("status", "") if skill else ""),
        ("gate_decision", decision.get("decision", "") if decision else ""),
        ("reject_reason", decision.get("reject_reason", "") if decision else ""),
        ("risk", _fmt_optional(decision.get("raw_critic_risk") if decision else None)),
        ("threshold", _fmt_optional(decision.get("threshold") if decision else None)),
        ("fallback", decision.get("fallback_executed", "") if decision else ""),
        ("min_user_m", _fmt_optional(min_user)),
        ("min_arm_user_m", _fmt_optional(min_arm)),
        ("min_obstacle_m", _fmt_optional(min_obs)),
        ("fall_flag", current.get("fall_flag", "")),
    ]
    y_text = top + 40
    for key, value in info:
        parts.append(f"<text x=\"{side_x + 22}\" y=\"{y_text}\" class=\"small muted\">{_esc(key)}</text>")
        parts.append(f"<text x=\"{side_x + 150}\" y=\"{y_text}\" class=\"small value\">{_esc(value)}</text>")
        y_text += 31
    counts = summary.get("per_skill_label_counts", {})
    parts.append(f"<text x=\"{side_x + 22}\" y=\"{y_text + 18}\" class=\"label\">label counts</text>")
    y_text += 50
    for skill_name in SKILL_ORDER:
        parts.append(
            f"<text x=\"{side_x + 22}\" y=\"{y_text}\" class=\"small\">{_esc(skill_name)}: "
            f"{_esc(json.dumps(counts.get(skill_name, {}), sort_keys=True))}</text>"
        )
        y_text += 28
    parts.append("</svg>")
    return "\n".join(parts)


def _chart_axes(
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    x_label: str,
    y_label: str,
    x_max: float,
    y_max: float,
    *,
    x_ticks: bool = True,
) -> str:
    parts = [
        f"<rect x=\"{left}\" y=\"{top}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>"
    ]
    for i in range(6):
        x = left + i * plot_w / 5
        y = top + i * plot_h / 5
        parts.append(f"<line x1=\"{left}\" y1=\"{y:.2f}\" x2=\"{left + plot_w}\" y2=\"{y:.2f}\" stroke=\"#e5e7eb\"/>")
        parts.append(f"<text x=\"{left - 16}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" class=\"axis\">{_fmt_short(y_max * (1 - i / 5))}</text>")
        if x_ticks:
            parts.append(f"<line x1=\"{x:.2f}\" y1=\"{top}\" x2=\"{x:.2f}\" y2=\"{top + plot_h}\" stroke=\"#f1f5f9\"/>")
            parts.append(f"<text x=\"{x:.2f}\" y=\"{top + plot_h + 28}\" text-anchor=\"middle\" class=\"axis\">{_fmt_short(x_max * i / 5)}</text>")
    parts.append(f"<text x=\"{left + plot_w / 2}\" y=\"{top + plot_h + 65}\" text-anchor=\"middle\" class=\"axis label\">{_esc(x_label)}</text>")
    parts.append(
        f"<text x=\"28\" y=\"{top + plot_h / 2}\" transform=\"rotate(-90 28 {top + plot_h / 2})\" "
        f"text-anchor=\"middle\" class=\"axis label\">{_esc(y_label)}</text>"
    )
    return "\n".join(parts)


def _svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#0f172a}.title{font-size:30px;font-weight:700}",
        ".subtitle{font-size:15px;fill:#475569}.axis{font-size:13px;fill:#475569}.label{font-size:16px;font-weight:700}",
        ".small{font-size:13px}.tiny{font-size:10px}.metric{font-size:42px;font-weight:800}.muted{fill:#64748b}.value{font-weight:700}",
        "</style>",
        "<rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>",
        f"<text x=\"42\" y=\"48\" class=\"title\">{_esc(title)}</text>",
        f"<text x=\"42\" y=\"76\" class=\"subtitle\">{_esc(subtitle)}</text>",
    ]


def _artifact(name: str, path: Path, source: Path, description: str) -> dict[str, str]:
    return {
        "name": name,
        "path": str(path),
        "source": str(source),
        "description": description,
    }


def _artifact_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# CASA Phase5 Visual Artifacts",
        "",
        f"- phase5_root: `{manifest.get('phase5_root')}`",
        f"- output_dir: `{manifest.get('output_dir')}`",
        f"- artifact_count: `{manifest.get('artifact_count')}`",
        "",
        "| artifact | path | source |",
        "|---|---|---|",
    ]
    for item in manifest.get("artifacts", []):
        lines.append(f"| {item['name']} | `{item['path']}` | `{item['source']}` |")
    return "\n".join(lines)


def _write_svg(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def _svg_to_png(svg_path: Path, png_path: Path, ffmpeg: str) -> None:
    _run([ffmpeg, "-v", "error", "-y", "-i", str(svg_path), str(png_path)])


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing executable: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed ({exc.returncode}): {' '.join(cmd)}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sample_rows(rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    if len(rows) <= max_rows:
        return rows
    stride = max(1, math.ceil(len(rows) / max_rows))
    sampled = rows[::stride]
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled[:max_rows]


def _active_skill(skills: list[dict[str, str]], wall_time: float) -> dict[str, str]:
    for skill in skills:
        start = _float(skill.get("start_wall_time"))
        end = _float(skill.get("end_wall_time"))
        if start is not None and end is not None and start <= wall_time <= end:
            return skill
    past = [
        skill
        for skill in skills
        if (_float(skill.get("start_wall_time")) is not None and _float(skill.get("start_wall_time")) <= wall_time)
    ]
    return past[-1] if past else {}


def _decision_for_skill(decisions: list[dict[str, str]], skill_idx: str | None) -> dict[str, str]:
    if not skill_idx:
        return {}
    for decision in decisions:
        if decision.get("skill_idx") == str(skill_idx):
            return decision
    return {}


def _label(row: dict[str, str]) -> int:
    if row.get("label") not in (None, ""):
        return int(float(row["label"]))
    return 1 if row.get("safe_label") == "unsafe" else 0


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounds(values: list[float], pad: float) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    return min(values) - pad, max(values) + pad


def _nice_max(values: list[float], floor: float) -> float:
    if not values:
        return floor
    max_value = max(max(values), floor)
    if max_value <= 1.0:
        return min(1.0, math.ceil(max_value * 10) / 10)
    magnitude = 10 ** math.floor(math.log10(max_value))
    return math.ceil(max_value / magnitude) * magnitude


def _fmt_short(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _fmt_optional(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return "" if value is None else str(value)
    return _fmt_short(parsed)


def _blend(start: str, end: str, alpha: float) -> str:
    alpha = max(0.0, min(1.0, alpha))
    s = _hex_to_rgb(start)
    e = _hex_to_rgb(end)
    rgb = tuple(round(sv + (ev - sv) * alpha) for sv, ev in zip(s, e))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
