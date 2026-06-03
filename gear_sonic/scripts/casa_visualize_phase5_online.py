"""Generate Phase 5 online five-baseline visualization evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", required=True, default=",".join(METHOD_ORDER))
    parser.add_argument("--casa-method", default="casa_a_per_skill")
    parser.add_argument("--max-example-episodes", type=int, default=12)
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
    try:
        result = run_visualization_package(
            artifact_dir=args.artifact_dir,
            audit_dir=args.audit_dir,
            output_dir=args.output_dir,
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
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "metric_consistency_status": result.metric_consistency["status"],
                "selected_episode_count": len([row for row in result.selected_episodes if row.get("seed") != ""]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
