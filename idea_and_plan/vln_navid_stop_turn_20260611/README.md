# VLN NaVid Stop-Turn Optimization Evidence

Date: 2026-06-11

This artifact records the lightweight Git-tracked evidence for the SONIC VLN/NaVid stop-turn optimization. Large online outputs remain outside git under `/mnt/data/students/lph/recording/`.

## Result

- Run directory: `/mnt/data/students/lph/recording/vln_wall_painting_split_real_navid_stop_turn_opt2_20260611_152951`
- Policy backend: `real_navid_visual_adapter`
- Held-out episodes: `20`
- Success: `8/20 = 0.40`
- Strict audit: `pass = true`
- Stop precision: `1.0`
- Stop recall: `1.0`
- Premature stop rate: `0.0`
- Late stop rate: `0.0`
- Mean stop distance: `0.5`
- Fall count: `0`

## What Changed

The stop-turn optimization keeps the no-CASA VLN setup intact and improves two failure surfaces:

- STOP is only allowed when the first-person RGB frame strongly sees the cyan/green gallery wall painting nearby.
- Premature visual STOP predictions are suppressed into forward/backoff until the visual target is close.
- Repeated turn cycles are interrupted with forward bursts and occasional opposite-turn recovery.

This result should not be interpreted as CASA solving collision recovery. The remaining failures are still dominated by collision/blocking and missing policy STOP after timeout.

## External Outputs

Large files are intentionally not committed:

- Videos: `/mnt/data/students/lph/recording/vln_wall_painting_split_real_navid_stop_turn_opt2_20260611_152951/videos/`
- Frames: `/mnt/data/students/lph/recording/vln_wall_painting_split_real_navid_stop_turn_opt2_20260611_152951/frames/`
- Decision logs: `/mnt/data/students/lph/recording/vln_wall_painting_split_real_navid_stop_turn_opt2_20260611_152951/data/decision_logs.csv`
- Trajectory logs: `/mnt/data/students/lph/recording/vln_wall_painting_split_real_navid_stop_turn_opt2_20260611_152951/data/trajectory_logs.csv`

See `run_manifest.json`, `result_summary.json`, and `SHA256SUMS` for exact paths and hashes.

## Verification

The following checks were run before upload:

```bash
.venv_sim/bin/python -m py_compile \
  gear_sonic/vln/no_casa_runner.py \
  gear_sonic/vln/no_casa_policy.py \
  gear_sonic/vln/visual_adapter.py \
  gear_sonic/tests/vln/test_oracle_dataset_runner.py

.venv_sim/bin/python -m pytest -q gear_sonic/tests/vln
git diff --check --cached
```
