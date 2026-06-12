# DMPS MPC-CBF Progress-Preserving VLN Safety Report

- final_status: `PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT`
- output_dir: `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246`
- policy_backend: `real_navid_visual_adapter`
- methods: `vln_only,vln_casa_reject_only,vln_casa_replan,vln_dmps_mpc_cbf_progress`
- heldout_episodes_per_method: `20`
- gate_method: `casa_a_hard_or_per_skill`
- dmps_horizon: `2`
- safety_margin: `0.05`
- method_alias: `vln_ppsr` == `vln_dmps_mpc_cbf_progress`
- phase4_root: `/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase4_dataset_v1_strict_50k_20260522`
- phase5_root: `/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522`

## Method Summary

| method | safe_success_rate | success_rate | unsafe_violation_rate | wall_contact_steps | reject_count | replan_count | mean_post_reject_progress | fallback_to_stop_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vln_only | 0.300 | 0.300 | 0.450 | 181 | 0 | 0 | 0.000 | 0.000 |
| vln_casa_reject_only | 0.150 | 0.150 | 0.000 | 0 | 16 | 0 | 0.000 | 0.000 |
| vln_casa_replan | 0.150 | 0.150 | 0.000 | 0 | 390 | 390 | -0.008 | 0.000 |
| vln_dmps_mpc_cbf_progress | 0.300 | 0.300 | 0.000 | 0 | 368 | 368 | 0.000 | 0.000 |

## Required Questions

1. Old `vln_casa_replan` lowers safe success because it prioritizes low-risk single-step recovery without visual progress, intent preservation, or loop avoidance.
2. The new method performs dynamic candidate-sequence search with safety filtering and progress/intent scoring; the old replan uses a fixed single-step risk/priority choice.
3. DMPS-style shielding is implemented: VLN remains nominal policy, the shield intervenes only on unsafe actions, executes only the first recovery action, and returns control to VLN.
4. Skill-level MPC-CBF-style safety is implemented as discrete rollout over skill microsteps with clearance barrier `h(x)=clearance-safety_margin`; this is not full torque-level humanoid MPC-CBF.
5. A non-privileged progress monitor tracks recent actions, rejects, blocked flags, stop sources, visual hashes, and loop counters.
6. Online replan does not use goal distance, shortest path, A*, or oracle waypoints; those are evaluator-only.
7. Fallback-to-stop is reported in `method_summary.csv` and `fallback_to_stop_comparison.png`.
8. Post-reject progress is evaluator-only and reported for comparison, not used for online action selection.
9. Unsafe/wall collision is kept under the same `maze.is_xy_safe` microstep criterion.
10. Safe-success improvement is determined by the final status rule, not assumed.
11. Recovery sequence effectiveness is visible in `dmps_selected_replans.csv` and `dmps_replan_action_distribution.png`.
12. Remaining failures are reported in `failure_reason_breakdown.json`.
13. Visual free-space uses a first-person left/center/right brightness-edge proxy; fallback is logged if image read fails.
14. The paper claim is supported only if final_status is `PASS_DMPS_PROGRESS_REPLAN_IMPROVES`.

## Stop Source Rule

CASA/DMPS safety stops are not counted as policy-issued task success. Only `vln_policy` or `policy_internal_guard` can satisfy the stop condition.

The bridge maps `backoff` to CASA `walk` for threshold lookup while preserving the executable `backoff_walk` skill name in logs.

## Videos
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_000.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_001.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_002.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_003.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_004.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_005.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_006.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_007.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_008.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_009.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_010.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_011.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_012.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_013.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_014.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_015.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_016.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_017.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_018.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_only/heldout_019.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_000.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_001.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_002.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_003.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_004.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_005.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_006.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_007.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_008.mp4`
- `/mnt/data/students/lph/recording/vln_dmps_mpc_cbf_progress_20260611_212246/videos/vln_casa_reject_only/heldout_009.mp4`
