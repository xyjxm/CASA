"""Compatibility entry point for the no-CASA MuJoCo VLN runner."""

from __future__ import annotations

from .no_casa_runner import RunnerConfig, main, parse_args, run


__all__ = ["RunnerConfig", "main", "parse_args", "run"]
