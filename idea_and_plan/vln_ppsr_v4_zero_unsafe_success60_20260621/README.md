# VLN PPSR-v4 Zero-Unsafe Success60 Locked Result

Final locked run dir:
`/mnt/data/students/lph/recording/vln_ppsr_v4_zero_unsafe_success60_20260621_locked_attempt04_lowratio_memory_seed260614`

Locked attempt: `v4_locked_attempt04_20260621_lowratio_memory_seed260614`
Eval stage: `locked`
Final status: `PASS_ZERO_UNSAFE_SUCCESS60`

## v4 Result

| metric | value |
|---|---:|
| unsafe_violation_count | 0 |
| unsafe_violation_rate | 0 |
| safe_success_count | 30/50 |
| safe_success_rate | 0.6 |
| stop_precision | 0.967741935483871 |
| stop_recall | 1 |
| premature_stop_rate | 0.02 |
| safety_stop_success_leakage_count | 0 |
| recovery_stop_success_leakage_count | 0 |

## Baseline Comparison

| method | safe_success_count | safe_success_rate | unsafe_violation_count | unsafe_violation_rate | stop_precision | stop_recall | premature_stop_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| vln_only | 20/50 | 0.4 | 30 | 0.6 | 1 | 1 | 0 |
| vln_dmps_mpc_cbf_progress | 18/50 | 0.36 | 0 | 0 | 1 | 0.9 | 0 |
| vln_ppsr_v3_task_return_replan | 11/50 | 0.22 | 0 | 0 | 1 | 0.6470588235294118 | 0 |
| vln_ppsr_v4_zero_unsafe_success60 | 30/50 | 0.6 | 0 | 0 | 0.967741935483871 | 1 | 0.02 |

## Integrity Notes

- Complete locked evaluation: 4 methods x 50 held-out episodes = 200 decision logs.
- `method_summary.csv`, `run_manifest.json`, per-episode logs, reports, frames, and videos are in the recording run dir.
- Large frames/videos/logs remain under `/mnt/data/students/lph/recording/` and are not stored in git.
- Online replan privileged leakage flags in manifest are false for evaluator success, goal distance, oracle waypoint, and shortest path use.

## Files

- `run_manifest.json`: copied from the locked run.
- `method_summary.csv`: copied from the locked run.
- `result_summary.json`: distilled metrics and pass criteria.
- `v4_stop_audit.json`: stop audit metrics.
- `failure_reason_breakdown.json`: aggregate failure breakdown.
- `SHA256SUMS`: checksums for this artifact directory.
