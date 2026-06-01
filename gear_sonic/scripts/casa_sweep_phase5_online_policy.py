"""Create or summarize Phase 5 online policy pilot sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
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
    parser.add_argument("--emit-run-commands", action="store_true")
    parser.add_argument("--phase4-root", default="outputs/casa/phase4_dataset_v1_strict_50k_20260522")
    parser.add_argument("--phase5-root", default="outputs/casa/phase5_conformal_baselines_20260522")
    parser.add_argument(
        "--online-root",
        type=Path,
        default=Path("outputs/casa/phase5_online_policy_pilots"),
    )
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--seeds", default="2001,2002")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
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
    if args.emit_run_commands:
        script_path = args.output_dir / "phase5_online_policy_sweep_commands.sh"
        script_path.write_text(_run_command_script(args, rows) + "\n")
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


def _run_command_script(args: argparse.Namespace, rows: list[dict[str, object]]) -> str:
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Pilot promotion rule: unsafe_reduction>=0.45, task_success_drop_rel<=0.20,",
        "# no catastrophic per-skill failure, and no unexplained initial_upright/env failure.",
        "",
        "# Named preset for the Issue #7 / PR #8 online candidate.",
        _lowmem_command(
            args,
            candidate_id="hard_or_receding_adaptive_preset",
            online_root=args.online_root / "hard_or_receding_adaptive_preset",
            extra=[
                "--performance-preset",
                "hard_or_receding_adaptive",
            ],
        ),
        _audit_command(
            args,
            online_root=args.online_root / "hard_or_receding_adaptive_preset",
            methods="sonic_only,hard_contract,casa_a_hard_or_receding_recovery",
            casa_method="casa_a_hard_or_receding_recovery",
        ),
        "",
    ]
    for row in rows:
        if row.get("status") != "planned":
            continue
        method = _candidate_method(row)
        candidate_id = str(row.get("candidate_id", method))
        online_root = args.online_root / candidate_id
        extra = [
            "--methods",
            f"sonic_only,hard_contract,{method}",
            "--casa-method",
            method,
            "--fallback-policy",
            str(row.get("fallback_policy", "auto")),
            "--adaptive-retry-count",
            str(row.get("recovery_retry_count", 0)),
            "--max-segment-duration",
            str(row.get("max_segment_duration", 0.5)),
            "--threshold-scale-global",
            str(row.get("threshold_scale_global", 1.0)),
            "--threshold-scale-by-skill",
            str(row.get("threshold_scale_by_skill", "")),
            "--randomize-method-order",
            "--method-order-seed",
            "20260531",
        ]
        if _bool(row.get("segment_long_skills")):
            extra.extend(["--segment-long-skills", "--recheck-before-segment"])
        commands.extend(
            [
                f"# Candidate {candidate_id}: {method}",
                _lowmem_command(args, candidate_id=candidate_id, online_root=online_root, extra=extra),
                _audit_command(
                    args,
                    online_root=online_root,
                    methods=f"sonic_only,hard_contract,{method}",
                    casa_method=method,
                ),
                "",
            ]
        )
    return "\n".join(commands).rstrip()


def _lowmem_command(
    args: argparse.Namespace,
    *,
    candidate_id: str,
    online_root: Path,
    extra: list[str],
) -> str:
    cmd = [
        ".venv_sim/bin/python",
        "gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py",
        "--phase4-root",
        str(args.phase4_root),
        "--phase5-root",
        str(args.phase5_root),
        "--online-root",
        str(online_root),
        "--episodes-per-seed",
        str(args.episodes_per_seed),
        "--seeds",
        str(args.seeds),
        "--max-parallel",
        str(args.max_parallel),
        "--chunk-size",
        str(args.chunk_size),
        *extra,
    ]
    return f"echo '[sweep] running {candidate_id}'\n{shlex.join(cmd)}"


def _audit_command(
    args: argparse.Namespace,
    *,
    online_root: Path,
    methods: str,
    casa_method: str,
) -> str:
    expected_episodes = len(_parse_list(args.seeds)) * int(args.episodes_per_seed) * len(_parse_list(methods))
    cmd = [
        ".venv_sim/bin/python",
        "gear_sonic/scripts/casa_audit_phase5_online.py",
        "--online-dir",
        str(online_root / "merged"),
        "--expected-episodes",
        str(expected_episodes),
        "--expected-methods",
        methods,
        "--expected-seeds",
        str(args.seeds),
        "--episodes-per-seed",
        str(args.episodes_per_seed),
        "--casa-method",
        casa_method,
    ]
    return shlex.join(cmd)


def _candidate_method(row: dict[str, object]) -> str:
    fallback_policy = str(row.get("fallback_policy", "stop"))
    hard_or = _bool(row.get("hard_or_casa"))
    segmented = _bool(row.get("segment_long_skills"))
    if fallback_policy == "adaptive_retry" and hard_or and segmented:
        return "casa_a_hard_or_receding_recovery"
    if fallback_policy == "adaptive_retry":
        return "casa_a_receding_recovery"
    if fallback_policy == "adaptive" and hard_or:
        return "casa_a_hard_or_recovery"
    if fallback_policy == "adaptive":
        return "casa_a_recovery_per_skill"
    if hard_or:
        return "casa_a_hard_or_per_skill"
    return "casa_a_per_skill"


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
