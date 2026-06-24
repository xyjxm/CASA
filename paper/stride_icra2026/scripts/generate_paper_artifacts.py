#!/usr/bin/env python3
"""Generate STRIDE paper tables, figures, and table consistency checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "source_manifest.json"
TABLE_DIR = ROOT / "tables"
FIGURE_DIR = ROOT / "figures"
CHECK_PATH = ROOT / "table_consistency_check.json"

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#6F6F6F",
    "black": "#222222",
}


def read_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return int(float(value))


def rate(row: dict[str, str], primary: str, fallback: str = "") -> float:
    if row.get(primary, "") not in ("", None):
        return as_float(row, primary)
    if fallback:
        return as_float(row, fallback)
    return 0.0


def source_path(manifest: dict, dataset: str, kind: str) -> Path:
    return (ROOT / manifest["datasets"][dataset][f"{kind}_path"]).resolve()


def load_sources(manifest: dict) -> dict[str, dict[str, dict[str, str]]]:
    sources: dict[str, dict[str, dict[str, str]]] = {}
    for name in manifest["datasets"]:
        rows = read_csv_rows(source_path(manifest, name, "csv"))
        sources[name] = {row["method"]: row for row in rows}
    return sources


def selected_row(
    manifest: dict, sources: dict[str, dict[str, dict[str, str]]], item: dict
) -> dict[str, str]:
    dataset = item["dataset"]
    method = item["source_method_id"]
    try:
        return sources[dataset][method]
    except KeyError as exc:
        raise KeyError(f"Missing source row for display row {item['display']}") from exc


def success_text(row: dict[str, str]) -> str:
    count = as_int(row, "safe_success_count")
    total = as_int(row, "total_episodes")
    value = rate(row, "safe_success_rate")
    return f"{count}/{total} ({value:.2f})"


def count_rate_text(row: dict[str, str], count_key: str, rate_key: str) -> str:
    return f"{as_int(row, count_key)} ({as_float(row, rate_key):.2f})"


def fmt(value: float, digits: int = 2) -> str:
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return "--"


def intervention_rate(row: dict[str, str]) -> float:
    if row.get("safety_filter_intervention_rate", "") not in ("", None):
        return as_float(row, "safety_filter_intervention_rate")
    return as_float(row, "reject_rate_per_step")


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def table_label(text: str) -> str:
    linebreaks = {
        "MPC-CBF humanoid adapted": r"MPC-CBF humanoid\\adapted",
        "SafeDPA adapted": r"SafeDPA\\adapted",
        "SPARK-style adapted": r"SPARK-style\\adapted",
        "w/o stop verifier": r"w/o stop\\verifier",
        "w/o late-stop recovery": r"w/o late-stop\\recovery",
        "w/o visual goal tracker": r"w/o visual goal\\tracker",
        "w/o task-return replan": r"w/o task-return\\replan",
    }
    label = linebreaks.get(text, tex_escape(text))
    return rf"\shortstack[l]{{{label}}}"


def write_table(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def generate_main_table(manifest: dict, sources: dict) -> list[dict]:
    rows = []
    for item in manifest["table_row_mapping"]["main_comparison"]:
        row = selected_row(manifest, sources, item)
        rows.append(
            {
                "display": item["display"],
                "type": item["note"],
                "safe_success": success_text(row),
                "unsafe": count_rate_text(row, "unsafe_violation_count", "unsafe_violation_rate"),
                "mean_final_distance": fmt(as_float(row, "mean_final_distance")),
                "raw": row,
            }
        )

    body = "\n".join(
        rf"{table_label(r['display'])} & {tex_escape(r['type'])} & {r['safe_success']} & {r['unsafe']} & {r['mean_final_distance']} \\"
        for r in rows
    )
    write_table(
        TABLE_DIR / "main_comparison.tex",
        rf"""
\begin{{table}}[t]
\caption{{Main locked comparison. External methods are adapted baselines, not official reproductions.}}
\label{{tab:main_comparison}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.7pt}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\hline
Method & Type & Safe & Unsafe & Dist. \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
""",
    )
    return rows


def generate_compact_table(manifest: dict, sources: dict) -> list[dict]:
    rows = []
    for item in manifest["table_row_mapping"]["main_comparison"]:
        row = selected_row(manifest, sources, item)
        rows.append(
            {
                "display": item["display"],
                "safe_success": success_text(row),
                "unsafe": count_rate_text(row, "unsafe_violation_count", "unsafe_violation_rate"),
                "timeout": count_rate_text(row, "timeout_count", "timeout_rate"),
                "intervention_rate": fmt(intervention_rate(row), 3),
                "mean_final_distance": fmt(as_float(row, "mean_final_distance")),
                "raw": row,
            }
        )

    body = "\n".join(
        rf"{table_label(r['display'])} & {r['safe_success']} & {r['unsafe']} & {r['timeout']} & {r['intervention_rate']} & {r['mean_final_distance']} \\"
        for r in rows
    )
    write_table(
        TABLE_DIR / "compact_metrics.tex",
        rf"""
\begin{{table}}[t]
\caption{{Compact metrics table. Intervention rate is per executed step.}}
\label{{tab:compact_metrics}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.45pt}}
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\hline
Method & Safe & Unsafe & Timeout & Interv. & Dist. \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
""",
    )
    return rows


def generate_ablation_table(manifest: dict, sources: dict) -> list[dict]:
    rows = []
    for item in manifest["table_row_mapping"]["ablation"]:
        row = selected_row(manifest, sources, item)
        rows.append(
            {
                "display": item["display"],
                "safe_success": success_text(row),
                "unsafe": count_rate_text(row, "unsafe_violation_count", "unsafe_violation_rate"),
                "stop_recall": fmt(as_float(row, "stop_recall")),
                "late_stop": fmt(as_float(row, "late_stop_rate")),
                "mean_final_distance": fmt(as_float(row, "mean_final_distance")),
                "raw": row,
            }
        )

    body = "\n".join(
        rf"{table_label(r['display'])} & {r['safe_success']} & {r['unsafe']} & {r['stop_recall']} & {r['late_stop']} & {r['mean_final_distance']} \\"
        for r in rows
    )
    write_table(
        TABLE_DIR / "ablation.tex",
        rf"""
\begin{{table}}[t]
\caption{{Ablation on STRIDE components.}}
\label{{tab:ablation}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{1.45pt}}
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\hline
Variant & Safe & Unsafe & Stop & Late & Dist. \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
""",
    )
    return rows


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def save_results_plot(rows: list[dict]) -> None:
    labels = [r["display"] for r in rows]
    safe = [as_float(r["raw"], "safe_success_rate") for r in rows]
    unsafe = [as_float(r["raw"], "unsafe_violation_rate") for r in rows]
    counts = [as_int(r["raw"], "safe_success_count") for r in rows]
    totals = [as_int(r["raw"], "total_episodes") for r in rows]
    x = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.set_facecolor("white")
    ax.bar([i - width / 2 for i in x], safe, width, label="Safe success", color=PALETTE["blue"])
    ax.bar([i + width / 2 for i in x], unsafe, width, label="Unsafe", color=PALETTE["vermillion"])
    for i, value in enumerate(safe):
        ax.text(i - width / 2, value + 0.025, f"{counts[i]}/{totals[i]}", ha="center", va="bottom", fontsize=7)
    for i, value in enumerate(unsafe):
        ax.text(i + width / 2, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(["STRIDE", "MPC-CBF\nadapted", "SafeDPA\nadapted", "SPARK-style\nadapted"])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_DIR / "results_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def save_ablation_plot(rows: list[dict]) -> None:
    labels = [r["display"] for r in rows]
    safe = [as_float(r["raw"], "safe_success_rate") for r in rows]
    counts = [as_int(r["raw"], "safe_success_count") for r in rows]
    totals = [as_int(r["raw"], "total_episodes") for r in rows]
    colors = [
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["green"],
        PALETTE["purple"],
        PALETTE["vermillion"],
    ]

    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.set_facecolor("white")
    bars = ax.bar(range(len(labels)), safe, color=colors, width=0.62)
    for bar, count, total in zip(bars, counts, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{count}/{total}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Safe success rate")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["Full", "No stop\nverifier", "No late\nrecovery", "No visual\ntracker", "No task\nreturn"])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_DIR / "ablation_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def add_box(ax, xy, width, height, text, fc="#FFFFFF", ec=PALETTE["black"], lw=0.9):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8)
    return box


def add_arrow(ax, start, end, color=PALETTE["black"], rad=0.0, lw=0.9):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def save_overview_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    boxes = [
        ((0.03, 0.58), 0.16, 0.22, "Instruction\n+ RGB history", "#EAF4FB"),
        ((0.25, 0.58), 0.16, 0.22, "Intent\nstate", "#FDF2D0"),
        ((0.47, 0.58), 0.16, 0.22, "Candidate\nmotion skill", "#EAF4FB"),
        ((0.69, 0.58), 0.16, 0.22, "Safety\nand stop gate", "#F7E7EE"),
        ((0.81, 0.18), 0.15, 0.18, "Task-return\nreplan", "#E7F3EA"),
        ((0.69, 0.18), 0.10, 0.18, "Execute", "#FFFFFF"),
    ]
    for xy, w, h, text, fc in boxes:
        add_box(ax, xy, w, h, text, fc=fc)

    add_arrow(ax, (0.19, 0.69), (0.25, 0.69))
    add_arrow(ax, (0.41, 0.69), (0.47, 0.69))
    add_arrow(ax, (0.63, 0.69), (0.69, 0.69))
    add_arrow(ax, (0.74, 0.58), (0.74, 0.36), color=PALETTE["blue"])
    add_arrow(ax, (0.85, 0.58), (0.88, 0.36), color=PALETTE["orange"])
    add_arrow(ax, (0.81, 0.27), (0.79, 0.27), color=PALETTE["green"])
    add_arrow(ax, (0.74, 0.36), (0.74, 0.58), color=PALETTE["blue"], rad=-0.18)
    add_arrow(ax, (0.88, 0.36), (0.77, 0.58), color=PALETTE["orange"], rad=-0.2)

    ax.text(0.74, 0.48, "valid stop", ha="center", va="center", fontsize=7, color=PALETTE["blue"])
    ax.text(0.90, 0.49, "unsafe or\nlate stop", ha="center", va="center", fontsize=7, color=PALETTE["orange"])
    ax.text(0.50, 0.12, "STRIDE runs as an execution wrapper around a vision-language action policy.", ha="center", fontsize=8)
    fig.tight_layout(pad=0.15)
    fig.savefig(FIGURE_DIR / "stride_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def save_recovery_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    corridor = Polygon([[0.4, 0.55], [9.6, 0.55], [9.6, 2.75], [0.4, 2.75]], closed=True, fill=False, edgecolor="#AAAAAA", linewidth=0.8)
    ax.add_patch(corridor)
    hazard = FancyBboxPatch((4.25, 1.18), 1.25, 0.9, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="#F8D7C4", edgecolor=PALETTE["vermillion"], linewidth=0.9)
    ax.add_patch(hazard)
    ax.text(4.88, 1.63, "blocked\nmotion", ha="center", va="center", fontsize=8, color=PALETTE["vermillion"])

    points = [(1.0, 1.65), (2.4, 1.65), (3.6, 1.65), (4.2, 1.65)]
    for start, end in zip(points[:-1], points[1:]):
        add_arrow(ax, start, end, color=PALETTE["gray"])
    ax.plot([4.05, 4.32], [1.35, 1.95], color=PALETTE["vermillion"], linewidth=1.3)
    ax.plot([4.05, 4.32], [1.95, 1.35], color=PALETTE["vermillion"], linewidth=1.3)

    add_arrow(ax, (4.05, 1.15), (3.15, 0.9), color=PALETTE["blue"], rad=0.15)
    add_arrow(ax, (3.15, 0.9), (3.15, 2.35), color=PALETTE["blue"], rad=-0.2)
    add_arrow(ax, (3.15, 2.35), (5.95, 2.35), color=PALETTE["green"])
    add_arrow(ax, (5.95, 2.35), (7.05, 1.65), color=PALETTE["green"], rad=-0.2)
    add_arrow(ax, (7.05, 1.65), (9.1, 1.65), color=PALETTE["gray"])

    ax.add_patch(Circle((1.0, 1.65), 0.12, color=PALETTE["blue"]))
    ax.add_patch(Circle((9.1, 1.65), 0.14, color=PALETTE["green"]))
    ax.text(1.0, 1.2, "start", ha="center", fontsize=8)
    ax.text(9.1, 1.2, "goal intent", ha="center", fontsize=8)
    ax.text(3.15, 0.62, "recover", ha="center", fontsize=8, color=PALETTE["blue"])
    ax.text(5.7, 2.62, "reacquire and return", ha="center", fontsize=8, color=PALETTE["green"])
    ax.text(5.0, 0.14, "Task-return recovery preserves the current intent while taking a safe local detour.", ha="center", fontsize=8)

    fig.tight_layout(pad=0.12)
    fig.savefig(FIGURE_DIR / "task_return_recovery.pdf", bbox_inches="tight")
    plt.close(fig)


def latex_contains(path: Path, token: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return token in text


def check_required_numbers(manifest: dict, table_rows: dict[str, list[dict]]) -> list[dict]:
    checks: list[dict] = []
    known = manifest["required_known_numbers"]
    rows_by_display = {}
    for rows in table_rows.values():
        for row in rows:
            rows_by_display[row["display"]] = row["raw"]

    for display, expected in known.items():
        if display not in rows_by_display:
            checks.append({"display": display, "status": "fail", "reason": "missing generated row"})
            continue
        row = rows_by_display[display]
        row_checks = []
        for key, expected_value in expected.items():
            if key == "total_episodes":
                actual = as_int(row, key)
            elif key.endswith("_count"):
                actual = as_int(row, key)
            else:
                actual = as_float(row, key)
            ok = abs(float(actual) - float(expected_value)) < 1e-9
            row_checks.append({"field": key, "expected": expected_value, "actual": actual, "ok": ok})
        safe_token = success_text(row)
        table_file = TABLE_DIR / ("ablation.tex" if display.startswith("w/o") or display == "Full STRIDE" else "main_comparison.tex")
        token_present = latex_contains(table_file, safe_token)
        checks.append(
            {
                "display": display,
                "safe_success_token": safe_token,
                "safe_success_token_present_in_table": token_present,
                "field_checks": row_checks,
                "status": "pass" if token_present and all(c["ok"] for c in row_checks) else "fail",
            }
        )
    return checks


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    sources = load_sources(manifest)

    main_rows = generate_main_table(manifest, sources)
    compact_rows = generate_compact_table(manifest, sources)
    ablation_rows = generate_ablation_table(manifest, sources)

    configure_plot()
    save_overview_schematic()
    save_recovery_schematic()
    save_results_plot(main_rows)
    save_ablation_plot(ablation_rows)

    checks = check_required_numbers(
        manifest,
        {
            "main": main_rows,
            "compact": compact_rows,
            "ablation": ablation_rows,
        },
    )
    source_hashes = {
        name: {
            "csv_sha256": sha256(source_path(manifest, name, "csv")),
            "json_sha256": sha256(source_path(manifest, name, "json")),
        }
        for name in manifest["datasets"]
    }
    output_files = sorted(
        [str(p.relative_to(ROOT)) for p in TABLE_DIR.glob("*.tex")]
        + [str(p.relative_to(ROOT)) for p in FIGURE_DIR.glob("*.pdf")]
    )
    result = {
        "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail",
        "source_manifest": "source_manifest.json",
        "source_hashes": source_hashes,
        "checks": checks,
        "output_files": output_files,
        "legacy_name_policy": "Generated tables and figures use display names only; source row ids remain in source_manifest.json.",
    }
    CHECK_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "pass":
        raise SystemExit("table consistency check failed")


if __name__ == "__main__":
    main()
