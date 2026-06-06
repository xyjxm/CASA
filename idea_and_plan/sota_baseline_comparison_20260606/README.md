# CASA SOTA Baseline Comparison Run

Run root:

`/mnt/data/students/lph/recording/codex_goal_sota_baseline_comparison_20260606_144918`

## Summary

Implemented first-slice CASA-adapted SOTA baseline infrastructure and three gate-only adapted baselines:

- `pcbf_adapted`: paper-faithful probabilistic CBF proxy gate.
- `crc_cbf_adapted`: paper-faithful conformal-risk CBF proxy gate.
- `mpc_cbf_humanoid_adapted`: lightweight reduced-order MPC-CBF feasibility proxy gate.

The implementation keeps existing Plan A methods unchanged and routes only registered SOTA method names through the new adapter registry.

## Files Changed

Modified:

- `gear_sonic/casa/phase5_policy.py`
- `gear_sonic/scripts/casa_run_phase5_online_experiment.py`

Added:

- `gear_sonic/casa/baselines/__init__.py`
- `gear_sonic/casa/baselines/sota_adapters/__init__.py`
- `gear_sonic/casa/baselines/sota_adapters/api.py`
- `gear_sonic/casa/baselines/sota_adapters/methods.py`
- `gear_sonic/casa/baselines/sota_adapters/registry.py`
- `gear_sonic/scripts/casa_eval_sota_adapted_baselines.py`
- `gear_sonic/scripts/casa_write_sota_method_metadata.py`
- `gear_sonic/scripts/casa_write_sota_paper_scaffold.py`
- `gear_sonic/tests/casa/test_sota_adapters.py`

Pre-existing untracked directories such as `.venv_sim`, `gear_sonic/data`, and `gear_sonic_deploy/*` were left untouched.

## Commands Run

Metadata:

```bash
PYTHONPATH=. python gear_sonic/scripts/casa_write_sota_method_metadata.py \
  --output-root /mnt/data/students/lph/recording/codex_goal_sota_baseline_comparison_20260606_144918
```

Checks:

```bash
PYTHONPATH=. python -m compileall gear_sonic/casa/baselines \
  gear_sonic/scripts/casa_eval_sota_adapted_baselines.py \
  gear_sonic/scripts/casa_write_sota_method_metadata.py \
  gear_sonic/scripts/casa_write_sota_paper_scaffold.py \
  gear_sonic/scripts/casa_run_phase5_online_experiment.py

PYTHONPATH=. pytest -q gear_sonic/tests/casa/test_sota_adapters.py \
  gear_sonic/tests/casa/test_phase5_online_performance.py::test_hard_or_receding_method_rejects_on_hard_contract \
  gear_sonic/tests/casa/test_phase5_online_performance.py::test_hard_or_receding_method_rejects_on_casa_threshold

PYTHONPATH=. ruff check gear_sonic/casa/baselines \
  gear_sonic/scripts/casa_eval_sota_adapted_baselines.py \
  gear_sonic/scripts/casa_write_sota_method_metadata.py \
  gear_sonic/scripts/casa_write_sota_paper_scaffold.py \
  gear_sonic/scripts/casa_run_phase5_online_experiment.py \
  gear_sonic/casa/phase5_policy.py \
  gear_sonic/tests/casa/test_sota_adapters.py
```

Offline replay smoke:

```bash
PYTHONPATH=. python gear_sonic/scripts/casa_eval_sota_adapted_baselines.py \
  --phase4-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase4_dataset_v1_strict_50k_20260522 \
  --phase5-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522 \
  --output-dir /mnt/data/students/lph/recording/codex_goal_sota_baseline_comparison_20260606_144918/data/offline_replay \
  --split test \
  --max-rows 500
```

Online dry-run smoke:

```bash
PYTHONPATH=. python gear_sonic/scripts/casa_run_phase5_online_experiment.py \
  --phase4-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase4_dataset_v1_strict_50k_20260522 \
  --phase5-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522 \
  --output-dir /mnt/data/students/lph/recording/codex_goal_sota_baseline_comparison_20260606_144918/online/dry_run_sota_smoke \
  --methods pcbf_adapted,crc_cbf_adapted,mpc_cbf_humanoid_adapted \
  --smoke \
  --dry-run
```

Paper scaffold:

```bash
PYTHONPATH=. python gear_sonic/scripts/casa_write_sota_paper_scaffold.py \
  --output-root /mnt/data/students/lph/recording/codex_goal_sota_baseline_comparison_20260606_144918 \
  --phase4-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase4_dataset_v1_strict_50k_20260522 \
  --phase5-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522
```

## Tests And Smoke Results

- `pytest_sota_adapters.log`: 5 tests passed.
- `ruff_sota_changes.log`: all checks passed.
- `compileall.log`: changed modules compiled.
- Offline replay smoke: 500 test rows x 3 methods = 1500 decision rows.
- Online dry-run smoke: 2 dry-run episodes x 3 methods x 8 skills = 48 gate decisions.

## Output Artifacts

Required metadata:

- `data/sota_method_metadata.json`
- `reports/sota_method_metadata.md`

Adapter replay and dry-run:

- `data/offline_replay/offline_replay_results.csv`
- `data/offline_replay/offline_replay_metrics.json`
- `data/offline_replay/offline_replay_method_metrics.csv`
- `data/offline_replay/offline_replay_per_skill_metrics.csv`
- `data/offline_replay/sota_split_manifest.json`
- `data/offline_replay/sota_feature_schema.json`
- `data/offline_replay/offline_replay_report.md`
- `online/dry_run_sota_smoke/gate_decisions.csv`
- `online/dry_run_sota_smoke/online_episode_results.csv`
- `online/dry_run_sota_smoke/method_summary.json`
- `online/dry_run_sota_smoke/sota_adapter_calibration.json`

Paper scaffold:

- `reports/method_adaptation_table.md`
- `reports/evaluation_protocol_table.md`
- `reports/metrics_table_template.md`
- `reports/implementation_fidelity_table.md`
- `manifests/full_online_eval_manifest.json`
- `manifests/full_online_eval_command.sh`

Logs:

- `logs/compileall.log`
- `logs/pytest_sota_adapters.log`
- `logs/ruff_sota_changes.log`
- `logs/offline_replay_smoke.log`
- `logs/online_dry_run_sota_smoke.log`
- `logs/write_sota_method_metadata.log`
- `logs/write_sota_paper_scaffold.log`

## Remaining Blockers

- Full online MuJoCo/deploy evaluation was not run in this turn because it is the long live evaluation.
- `clbf_lbac_adapted`, `safedpa_adapted`, and `safer_splat_cbf_adapted` are verified in metadata but not implemented in this first slice.
- `mpc_cbf_humanoid_adapted` is explicitly a `lightweight_proxy`, not the full Poisson safety-function MPC solver.
- CRC-CBF project/paper metadata is verified, but no repository URL was verified, so it remains `paper_faithful_proxy`.
- Offline replay smoke metrics are sanity checks only and must not be used as the final paper table.

## Exact Next Full Online Command

```bash
.venv_sim/bin/python -u gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py \
  --phase4-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase4_dataset_v1_strict_50k_20260522 \
  --phase5-root /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522 \
  --online-root /mnt/data/students/lph/recording/codex_goal_sota_baseline_comparison_20260606_144918/online/full_online_eval \
  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill,pcbf_adapted,crc_cbf_adapted,mpc_cbf_humanoid_adapted \
  --casa-method casa_a_per_skill \
  --seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100 \
  --max-parallel 3 \
  --chunk-size 25 \
  --cuda-devices 0,1 \
  --performance-preset custom
```

Before launching, confirm the selected GPUs, CycloneDDS domains, and ZMQ ports are not already occupied by existing CASA video-search or online jobs.
