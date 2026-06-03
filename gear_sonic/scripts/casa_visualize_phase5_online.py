"""Generate Phase 5 online five-baseline visualization evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.phase5 import METHOD_ORDER  # noqa: E402
from gear_sonic.casa.phase5_visualization import (  # noqa: E402
    MetricConsistencyError,
    parse_methods,
    run_visualization_package,
)

DEFAULT_OUTPUT_ROOT = Path("/mnt/data/students/lph/recording")
DEFAULT_RUN_PREFIX = "phase5_online_five_baseline_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit output directory. If omitted, a run directory is created under --output-root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", help="Run directory name used under --output-root when --output-dir is omitted.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output run directory.")
    parser.add_argument(
        "--repo-manifest",
        type=Path,
        help="Small JSON manifest path to write inside the repo. Defaults only when --output-dir is omitted.",
    )
    parser.add_argument("--no-repo-manifest", action="store_true")
    parser.add_argument("--methods", default=",".join(METHOD_ORDER))
    parser.add_argument("--casa-method", default="casa_a_per_skill")
    parser.add_argument("--max-example-episodes", type=int, default=5)
    parser.add_argument("--video-mode", choices=["auto", "real", "timeline-only"], default="auto")
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--videos-dir", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--write-html", action="store_true")
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--fail-on-metric-mismatch", action="store_true")
    parser.add_argument("--strict-five-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    output_dir, run_name = resolve_output_dir(args)
    try:
        result = run_visualization_package(
            artifact_dir=args.artifact_dir,
            audit_dir=args.audit_dir,
            output_dir=output_dir,
            methods=methods,
            casa_method=args.casa_method,
            max_example_episodes=args.max_example_episodes,
            video_mode=args.video_mode,
            frames_dir=args.frames_dir,
            videos_dir=args.videos_dir,
            fps=args.fps,
            write_html=args.write_html,
            write_markdown=args.write_markdown,
            fail_on_metric_mismatch=args.fail_on_metric_mismatch,
            strict_five_baseline=args.strict_five_baseline,
        )
    except MetricConsistencyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    repo_manifest = resolve_repo_manifest(args, run_name)
    if repo_manifest is not None:
        write_repo_manifest(repo_manifest, args=args, result=result, run_name=run_name)
    print("Phase 5 online visualization package written to:")
    print(f"{result.output_dir}/")
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "metric_consistency_status": result.metric_consistency["status"],
                "repo_manifest": str(repo_manifest) if repo_manifest is not None else None,
                "selected_episode_count": len([row for row in result.selected_episodes if row.get("seed") != ""]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def resolve_output_dir(args: argparse.Namespace) -> tuple[Path, str]:
    output_root = args.output_root.resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        if output_dir == output_root:
            raise SystemExit("--output-dir must not point directly at --output-root; use a run-specific subfolder.")
        run_name = output_dir.name
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        run_name = args.run_name or default_run_name(args.artifact_dir)
        output_dir = (output_root / run_name).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_name


def default_run_name(artifact_dir: Path) -> str:
    match = re.search(r"(20\d{6})", artifact_dir.name)
    if match:
        suffix = match.group(1)
    else:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{DEFAULT_RUN_PREFIX}_{suffix}"


def resolve_repo_manifest(args: argparse.Namespace, run_name: str) -> Path | None:
    if args.no_repo_manifest:
        return None
    if args.repo_manifest is not None:
        return args.repo_manifest.resolve()
    if args.output_dir is None:
        return REPO_ROOT / "idea_and_plan" / f"{run_name}_manifest.json"
    return None


def write_repo_manifest(path: Path, *, args: argparse.Namespace, result: object, run_name: str) -> None:
    output_dir = result.output_dir
    generated_files = []
    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        generated_files.append(
            {
                "path": str(file_path.relative_to(output_dir)),
                "bytes": file_path.stat().st_size,
                "sha256": sha256(file_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact": "phase5_online_five_baseline_demo",
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(output_dir),
        "source_artifact_dir": str(args.artifact_dir.resolve()),
        "source_audit_dir": str(args.audit_dir.resolve()),
        "command": [sys.executable, *sys.argv],
        "command_string": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        "large_artifact_policy": "Generated recording outputs are written outside Git and are not committed.",
        "generated_files": generated_files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
