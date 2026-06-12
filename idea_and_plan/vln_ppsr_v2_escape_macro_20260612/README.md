# PPSR-v2 Escape-Macro VLN Safety Artifact

This lightweight artifact records the PPSR-v2 implementation and locked online
evaluation output. Large videos, figures, frames, logs, and run data are kept
outside git under `/mnt/data/students/lph/recording/`.

## Method

- Canonical method: `vln_ppsr_v2_escape_macro`
- Aliases: `vln_escape_macro_progress`, `vln_dmps_escape_macro_progress`
- Baseline PPSR-v1 method: `vln_dmps_mpc_cbf_progress`
- Policy backend: `real_navid_visual_adapter`
- Real NaVid/Uni-NaVid used: `true`
- Online privileged leakage flags:
  - `used_evaluator_success_for_online_replan: false`
  - `used_map_cell_progress_for_online_replan: false`

## Locked Output

```text
/mnt/data/students/lph/recording/vln_ppsr_v2_escape_macro_locked_20260612_185743/locked_20_real_navid_gpu0_uuid
```

The run was pinned to physical GPU0 by UUID:

```text
GPU-8b1188b9-c3d8-a887-a7f9-073ccd6e33fb
```

This avoided sharing a GPU with the concurrent IsaacLab/SMR processes.

## Audit Conclusion

```text
PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT
```

Interpretation: PPSR-v2 is implemented and auditable. It removed the PPSR-v1
small-turn loop failure mode and used escape macros, but it did not improve
locked-run safe success over PPSR-v1.

## Locked Metrics

| method | safe_success | unsafe | rejects | small_turn_recovery | escape_macros |
|---|---:|---:|---:|---:|---:|
| `vln_only` | 6/20 | 9 | 0 | 0 | 0 |
| `vln_casa_replan` | 3/20 | 0 | 390 | 0 | 0 |
| `vln_dmps_mpc_cbf_progress` | 6/20 | 0 | 368 | 352 | 0 |
| `vln_ppsr_v2_escape_macro` | 3/20 | 0 | 148 | 0 | 108 |

## Key Files Outside Git

- `README.md`
- `ppsr_v2_escape_macro_report.md`
- `run_manifest.json`
- `data/method_summary.csv`
- `data/ppsr_v2_method_summary.csv`
- `data/anti_turn_loop_mask_audit.json`
- `data/readiness_progress_metrics.json`
- `data/ppsr_v2_failure_reason_breakdown.json`
- `figures/safe_success_comparison.png`
- `figures/unsafe_comparison.png`
- `figures/small_turn_recovery_reduction.png`
- `figures/recovery_stuck_comparison.png`
- `figures/readiness_progress_comparison.png`
- `figures/escape_macro_usage.png`

## Validation

```text
py_compile passed
pytest gear_sonic/tests/vln: 47 passed
git diff --check passed
```
