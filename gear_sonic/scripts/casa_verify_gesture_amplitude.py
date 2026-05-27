"""Verify CASA GestureSkill amplitude response from MuJoCo episode logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.casa.runner_utils import create_executor, resolve_output_dir, resolve_run_id, resolve_sim_log_dir, write_json
from gear_sonic.casa.skills import GestureSkill, PassiveSkill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run id used when deriving the default output dir")
    parser.add_argument("--output-dir", type=Path, help="Output dir for CASA logs")
    parser.add_argument("--sim-log-dir", type=Path, help="MuJoCo sim episode log dir")
    parser.add_argument("--zmq-host", default="*", help="ZMQ bind host for command/planner publisher")
    parser.add_argument("--zmq-port", type=int, default=5556, help="ZMQ port used by deploy --zmq-port")
    parser.add_argument("--publish-fps", type=float, default=10.0, help="Planner publish frequency")
    parser.add_argument("--amplitudes", default="0.1,0.3,0.5,0.7,0.9,1.1")
    parser.add_argument("--trials-per-amplitude", type=int, default=5)
    parser.add_argument("--gesture-duration", type=float, default=3.0)
    parser.add_argument("--gesture-frequency", type=float, default=0.75)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--stabilize-seconds", type=float, default=1.0)
    parser.add_argument("--send-final-stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build payloads and logs without opening ZMQ")
    return parser.parse_args()


def parse_amplitudes(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    run_id = resolve_run_id(args.run_id)
    output_dir = resolve_output_dir(args.output_dir, run_id)
    sim_log_dir = resolve_sim_log_dir(args.sim_log_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    amplitudes = parse_amplitudes(args.amplitudes)

    executor = create_executor(args, output_dir, run_id, sim_log_dir)
    rows: list[dict] = []
    try:
        executor.start_policy()
        for amplitude in amplitudes:
            for trial in range(args.trials_per_amplitude):
                if args.stabilize_seconds > 0:
                    executor.execute_one(PassiveSkill(duration=args.stabilize_seconds, mode="wait"))
                result = executor.execute_one(
                    GestureSkill(
                        amplitude=amplitude,
                        frequency=args.gesture_frequency,
                        duration=args.gesture_duration,
                        side=args.side,
                    ),
                    episode_id=f"amp_{amplitude:g}_{trial}",
                )
                rows.append(
                    {
                        "amplitude": amplitude,
                        "frequency": args.gesture_frequency,
                        "side": args.side,
                        "trial": trial,
                        "status": result.status,
                        "success": int(result.status == "success" or (args.dry_run and result.status == "dry_run")),
                        "measured_rom": result.evidence.get("measured_rom"),
                        "left_shoulder_pitch_rom": result.evidence.get("left_shoulder_pitch_rom"),
                        "right_shoulder_pitch_rom": result.evidence.get("right_shoulder_pitch_rom"),
                        "termination_reason": result.termination_reason,
                    }
                )
        if args.send_final_stop:
            executor.stop_policy()
    finally:
        sent_message_count = len(executor.publisher.sent_messages)
        executor.close()

    sweep_csv = output_dir / "gesture_amplitude_sweep.csv"
    with sweep_csv.open("w", newline="") as file:
        fieldnames = [
            "amplitude",
            "frequency",
            "side",
            "trial",
            "status",
            "success",
            "measured_rom",
            "left_shoulder_pitch_rom",
            "right_shoulder_pitch_rom",
            "termination_reason",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_id": run_id,
        "dry_run": args.dry_run,
        "output_dir": str(output_dir),
        "sim_log_dir": str(sim_log_dir),
        "amplitudes": amplitudes,
        "trials_per_amplitude": args.trials_per_amplitude,
        "sweep_csv": str(sweep_csv),
        "sent_message_count": sent_message_count,
        "distinct_successful_amplitudes": sorted({row["amplitude"] for row in rows if row["success"]}),
        "go_criteria_passed": (
            not args.dry_run and len({row["amplitude"] for row in rows if row["success"]}) >= 3
        ),
    }
    maybe_write_plot(output_dir / "gesture_amplitude_sweep.png", rows)
    write_json(output_dir / "gesture_amplitude_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def maybe_write_plot(path: Path, rows: list[dict]) -> None:
    valid = [row for row in rows if row.get("measured_rom") not in {None, ""}]
    if not valid:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    grouped: dict[float, list[float]] = {}
    for row in valid:
        grouped.setdefault(float(row["amplitude"]), []).append(float(row["measured_rom"]))
    xs = sorted(grouped)
    ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("commanded amplitude")
    plt.ylabel("measured shoulder pitch ROM (rad)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
