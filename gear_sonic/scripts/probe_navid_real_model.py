"""Run a strict offline probe against the real NaVid/Uni-NaVid model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.vln.backends import NaVidBackend, VLNObservation


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _make_probe_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    image[:, :] = (235, 235, 228)
    cv2.rectangle(image, (170, 70), (310, 240), (80, 120, 180), -1)
    cv2.rectangle(image, (215, 150), (265, 240), (40, 60, 90), -1)
    cv2.line(image, (240, 240), (240, 350), (70, 70, 70), 6)
    cv2.arrowedLine(image, (90, 310), (390, 310), (30, 140, 70), 8, tipLength=0.12)
    cv2.putText(image, "NAVID PROBE", (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.imwrite(str(path), image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vision-tower-path", type=Path, required=True)
    parser.add_argument("--variant", choices=["navid", "uni-navid"], default="navid")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.run_dir / "offline_probe"
    image_path = output_dir / "probe_image.jpg"
    raw_path = output_dir / "raw_output.json"
    result_path = output_dir / "offline_probe_result.json"
    _make_probe_image(image_path)

    backend = NaVidBackend(
        repo_path=args.repo_path,
        result_dir=args.run_dir / "navid_backend",
        model_path=args.model_path,
        variant=args.variant,
        strict_model=True,
        python_executable=args.python,
        vision_tower_path=args.vision_tower_path,
        worker_timeout_s=args.timeout_s,
    )
    try:
        if not backend.availability.get("model_loaded"):
            result = {
                "ok": False,
                "stage": "model_load",
                "backend_probe": backend.availability,
                "error": backend.availability.get("model_unavailable_reason"),
            }
            _write_json(result_path, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

        backend.reset("offline_probe")
        observation = VLNObservation(
            episode_id="offline_probe",
            step_idx=0,
            instruction="Walk straight through the doorway and stop.",
            image_path=str(image_path),
        )
        started_at = time.time()
        response = backend.next_action(observation)
        elapsed_s = time.time() - started_at
        raw_payload = {
            "raw_output": response.raw_output,
            "backend_name": response.backend_name,
            "available": response.available,
            "decision": asdict(response.decision),
            "metadata": response.metadata,
            "elapsed_s": elapsed_s,
            "image_path": str(image_path),
        }
        _write_json(raw_path, raw_payload)
        result = {
            "ok": bool(response.available and backend.availability.get("model_loaded")),
            "stage": "offline_inference",
            "raw_output_path": str(raw_path),
            "backend_probe": backend.availability,
            "backend_name": response.backend_name,
            "decision_action": response.decision.action.value,
            "elapsed_s": elapsed_s,
        }
        _write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 3
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
