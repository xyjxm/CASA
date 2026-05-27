"""Rollout-level oracle logger for CASA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gear_sonic.casa.oracle.violations import Violation


class RolloutLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.violations_path = self.output_dir / "violations.jsonl"
        self.summary_path = self.output_dir / "rollout_summary.json"

    def write(self, violations: list[Violation], summary: dict[str, Any]) -> None:
        with self.violations_path.open("w") as file:
            for violation in violations:
                file.write(json.dumps(violation.to_dict(), sort_keys=True) + "\n")
        with self.summary_path.open("w") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
