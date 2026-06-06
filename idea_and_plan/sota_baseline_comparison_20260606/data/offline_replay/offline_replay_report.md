# CASA Adapted SOTA Offline Replay

- predictions_csv: `/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase4_dataset_v1_strict_50k_20260522/raw_critic/predictions.csv`
- thresholds_json: `/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522/conformal_thresholds.json`
- split: `test`

## Method Metrics

| method | fidelity | count | unsafe_total | reject_rate | fnr | safe_acceptance_rate |
|---|---:|---:|---:|---:|---:|---:|
| PCBF-adapted | paper_faithful_proxy | 500 | 20 | 0.0200 | 0.6000 | 0.9958 |
| CRC-CBF-adapted | paper_faithful_proxy | 500 | 20 | 0.0080 | 0.8000 | 1.0000 |
| MPC-CBF-Humanoid-adapted | lightweight_proxy | 500 | 20 | 0.0160 | 0.7000 | 0.9958 |

## Fidelity Notes

- These are CASA-adapted gate-only reruns in the Phase5 score space.
- No external paper result numbers are copied into this report.
- `mpc_cbf_humanoid_adapted` is a reduced-order feasibility proxy, not the full Poisson safety-function MPC solver.
