# PPSR-v4 Ablation Report

- ablation_run_dir: `/mnt/data/students/lph/recording/vln_ppsr_v4_ablation_20260622_212354`
- full_anchor_dir: `/mnt/data/students/lph/recording/vln_ppsr_v4_zero_unsafe_success60_20260621_locked_attempt04_lowratio_memory_seed260614`
- full_anchor_reused: `True`
- policy_backend: `real_navid_visual_adapter`
- eval_stage: `locked`
- heldout_episodes_per_method: `50`
- max_steps: `180`
- seeds: `[260614]`

## Component Table

| method | safe_success | unsafe | stop_recall | premature_stop | late_stop | stop_verifier_applied | late_applied/candidate/disabled | visual_applied/candidate/disabled | task_return_applied/disabled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vln_ppsr_v4_zero_unsafe_success60 | 30/50 (0.600) | 0 (0.000) | 1.000 | 0.020 | 0.000 | 17 | 12/0/0 | 170/0/0 | 1561/0 |
| vln_ppsr_v4_ablate_no_stop_verifier | 23/50 (0.460) | 0 (0.000) | 0.852 | 0.000 | 0.080 | 0 | 0/0/0 | 0/0/0 | 1596/0 |
| vln_ppsr_v4_ablate_no_late_stop_recovery | 30/50 (0.600) | 0 (0.000) | 0.968 | 0.000 | 0.020 | 15 | 0/23/23 | 171/198/0 | 1593/0 |
| vln_ppsr_v4_ablate_no_visual_goal_tracker | 30/50 (0.600) | 0 (0.000) | 1.000 | 0.020 | 0.000 | 18 | 21/21/0 | 0/103/103 | 1647/0 |
| vln_ppsr_v4_ablate_no_task_return_replan | 18/50 (0.360) | 0 (0.000) | 1.000 | 0.000 | 0.000 | 2 | 10/10/0 | 0/0/0 | 0/5239 |

## Delta vs Full

| method | safe_success_delta | unsafe_delta | stop_recall_delta | premature_stop_delta |
|---|---:|---:|---:|---:|
| vln_ppsr_v4_ablate_no_stop_verifier | -0.140 | 0.000 | -0.148 | -0.020 |
| vln_ppsr_v4_ablate_no_late_stop_recovery | 0.000 | 0.000 | -0.032 | -0.020 |
| vln_ppsr_v4_ablate_no_visual_goal_tracker | 0.000 | 0.000 | 0.000 | 0.000 |
| vln_ppsr_v4_ablate_no_task_return_replan | -0.240 | 0.000 | 0.000 | -0.020 |

## Conclusions

- stop verifier: safe_success_delta=-0.140, unsafe_delta=0.000, stop_recall_delta=-0.148, premature_stop_delta=-0.020
- late-stop recovery: safe_success_delta=0.000, unsafe_delta=0.000, stop_recall_delta=-0.032, premature_stop_delta=-0.020
- visual goal tracker: safe_success_delta=0.000, unsafe_delta=0.000, stop_recall_delta=0.000, premature_stop_delta=0.000
- task-return replan: safe_success_delta=-0.240, unsafe_delta=0.000, stop_recall_delta=0.000, premature_stop_delta=-0.020

## Large Artifacts

- videos_path: `/mnt/data/students/lph/recording/vln_ppsr_v4_ablation_20260622_212354/videos`
- frames_path: `/mnt/data/students/lph/recording/vln_ppsr_v4_ablation_20260622_212354/frames`
