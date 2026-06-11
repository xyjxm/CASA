"""CLI entry point for CASA-gated SONIC VLN online evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.vln.casa_runner import main


if __name__ == "__main__":
    main()
