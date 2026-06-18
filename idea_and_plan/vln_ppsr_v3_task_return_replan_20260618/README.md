# PPSR-v3 Task-Return VLN Replan Artifact

- final_status: `DEV_IMPROVED_BUT_LOCKED_NO_IMPROVEMENT`
- output_dir: `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206`
- dev_eval_reference_dir: `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_203208_dev_all_max52`
- GPU UUID: `GPU-8b1188b9-c3d8-a887-a7f9-073ccd6e33fb`
- real NaVid/Uni-NaVid used: `True`
- learned ranker used: `False`
- status note: Dev all-method max52 improved over PPSR-v2 and matched DMPS; locked held-out improved over PPSR-v2 but did not meet DMPS safe_success_rate, so no PASS is claimed.

## Locked Method Summary

| method | safe_success | unsafe | reject_to_ready | next_allowed | loop_rate | stop_leak | recovery_leak |
|---|---:|---:|---:|---:|---:|---:|---:|
| vln_only | 0.350 | 0.500 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| vln_casa_replan | 0.150 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| vln_dmps_mpc_cbf_progress | 0.350 | 0.000 | 0.000 | 0.000 | 1.958 | 0 | 0 |
| vln_ppsr_v2_escape_macro | 0.250 | 0.000 | 0.000 | 0.000 | 0.656 | 0 | 0 |
| vln_ppsr_v3_task_return_replan | 0.300 | 0.000 | 1.000 | 1.000 | 0.000 | 0 | 0 |

## Files

- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/run_manifest.json` sha256 `9e81298daf95f8ed7dba7d656d073d3b50b62ad1cdb12224b10b069532542eda`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/ppsr_v3_task_return_report.md` sha256 `2ea339ee4af0695e1cae96420a690c4d776c453545508da062e912e5dc921532`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/method_summary.csv` sha256 `06614751e103e30289f32ef28db16ac66474ed5840d026b42b59b66c5f2a9145`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_method_summary.csv` sha256 `d227f3f03079bb663d51bf3bfe560ba0aedcdc45564e55b558caffa4d0ccc441`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_candidate_sequences.csv` sha256 `c38384480c619f63875cef527df3fa037698be92db11a80da8343bba7fb84a72`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_selected_replans.csv` sha256 `0051f73d820372c178f7f46a2c07cc9aa204fd98ef25d07e5340b5f29d98d15f`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_policy_ready_audit.csv` sha256 `c38384480c619f63875cef527df3fa037698be92db11a80da8343bba7fb84a72`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_language_progress_audit.csv` sha256 `c38384480c619f63875cef527df3fa037698be92db11a80da8343bba7fb84a72`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_loop_avoidance_audit.json` sha256 `1037b97e644b13f3904302de26faf46f657fdf739d09ef677da24bdf1c97ad2c`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_stop_source_audit.json` sha256 `7226762be80250474a4442c29b04c17c57843b87ef52f79ded51aea2203de8e5`
- `/mnt/data/students/lph/recording/vln_ppsr_v3_task_return_replan_20260618_211206/data/v3_failure_reason_breakdown.json` sha256 `5d44265c7f16304474c326647b4640709ef4c17eb1b3a4e087214c1b492bdb35`
