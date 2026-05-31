# CASA Package Structure

`gear_sonic/casa/` contains the reusable Python package for CASA experiments.
Command-line entry points live in `gear_sonic/scripts/casa_*.py`.

## Modules

| Module | Contents |
|---|---|
| `skills/` | Skill base types, registry, executor, and walk/turn/gesture/passive skills |
| `oracle/` | Safety thresholds, violation records, oracle evaluation, rollout summaries |
| `loggers/` | Rollout and skill log writers |
| `dataset/` | Invocation dataset helpers |
| `scene/` | MuJoCo prop configuration, placement, and Phase 2 v2 scene generation |
| `io/` | Episode log readers and ZMQ publish helpers |
| `phase5.py` | Shared constants and helpers for conformal/baseline evaluation |
| `phase5_online.py` | Shared validation, audit, and report helpers for Phase 5 online artifacts |
| `runner_utils.py` | Utility functions shared by CASA runner scripts |

## Design Rules

- Keep reusable logic in this package.
- Keep experiment orchestration and CLI parsing in `gear_sonic/scripts/`.
- Write generated files to `outputs/`, not into the package tree.
- Treat runtime and latency artifacts as explicit labels or quarantine reasons,
  not as clean training examples.
- Keep phase-specific acceptance rules in scripts and reports so they remain
  auditable.

## Typical Flow

1. A runner script constructs a CASA skill schedule.
2. Skill wrappers publish commands to the SONIC deploy interface.
3. Loggers write skill, command, and rollout evidence.
4. The oracle reads sim/deploy logs and writes structured violations.
5. Dataset builders convert accepted evidence into clean train/critic_val/
   calibration/test artifacts, keeping raw-critic validation independent from
   conformal calibration.
