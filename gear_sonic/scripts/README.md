# CASA Script Index

This directory contains both upstream SONIC scripts and CASA experiment scripts.
CASA scripts use the `casa_*.py` prefix.

## Phase 0-1: Skill Checks

| Script | Purpose |
|---|---|
| `casa_run_sanity_check.py` | Run a CASA Phase 0 long-horizon skill sanity check |
| `casa_run_skill_repeatability.py` | Run Phase 0/1 repeatability checks for CASA skills |
| `casa_verify_gesture_amplitude.py` | Verify gesture command amplitude from MuJoCo logs |
| `casa_run_rollout_with_oracle.py` | Run the CASA oracle on one rollout or explicit log files |

## Phase 2: Visual Dataset

| Script | Purpose |
|---|---|
| `casa_collect_oracle_eval_set.py` | Collect oracle rollout summaries and create a human review sheet |
| `casa_merge_oracle_eval_runs.py` | Merge parallel Phase 2 rollout directories |
| `casa_run_oracle_eval_batch.py` | Run a batch of Phase 2 oracle rollouts |
| `casa_build_clean_visual_dataset.py` | Build the first clean visual-priority rollout dataset |
| `casa_make_phase2_v2_scene_plan.py` | Create the Phase 2 v2 scene plan CSV |
| `casa_phase2_v2_scene_placement_test.py` | Generate scenes without policy/video and validate initial placement |
| `casa_run_phase2_v2_lane.py` | Run one Phase 2 v2 collection lane |
| `casa_render_rollout_videos.py` | Render rollout logs into review videos without rerunning policy |
| `casa_build_clean_visual_dataset_v2.py` | Build the Phase 2 v2 clean natural visual dataset |
| `casa_package_clean_visual_dataset.py` | Copy accepted videos/labels into a review package |

## Phase 3: Invocation Feasibility

| Script | Purpose |
|---|---|
| `casa_build_invocation_dataset.py` | Build state-skill-label invocation datasets |
| `casa_collect_phase3_feasibility.py` | Plan or run Phase 3 feasibility collection |
| `casa_run_counterfactual_subset.py` | Run the Phase 3 counterfactual skill subset |
| `casa_train_mini_critic.py` | Train a small Phase 3 safety critic |
| `casa_audit_phase3_acceptance.py` | Audit Phase 3 acceptance artifacts |
| `casa_write_phase3_report.py` | Write the Phase 3 go/no-go report |

## Phase 4: Strict 50k Dataset And Raw Critic

| Script | Purpose |
|---|---|
| `casa_make_phase4_collection_plan.py` | Create Phase 4 strict-clean collection recovery commands |
| `casa_build_phase4_dataset.py` | Build Phase 4 train/critic_val/calibration/test split roles |
| `casa_train_raw_critic.py` | Train the Phase 4 raw critic using critic_val for model selection |
| `casa_audit_phase4_acceptance.py` | Audit Phase 4 dataset and critic acceptance |
| `casa_write_phase4_report.py` | Write the Phase 4 implementation report |

## Phase 5: Conformal And Online Evaluation

| Script | Purpose |
|---|---|
| `casa_calibrate_phase5_conformal.py` | Calibrate global and per-skill conformal thresholds on the untouched calibration split |
| `casa_build_hc_filtered_online_calibration.py` | Build Hard-Contract-filtered online calibration rows for strict deployment-distribution conformal calibration |
| `casa_eval_phase5_baselines.py` | Evaluate the five baseline gates on the Phase 4 test split |
| `casa_make_phase5_artifacts.py` | Generate Phase 5 reliability, rejection-risk, FNR, counterfactual, failure, and trajectory video artifacts |
| `casa_run_phase5_online_experiment.py` | Run Phase 5 online baseline or candidate policy episodes |
| `casa_run_phase5_online_lane.py` | Launch one online experiment lane |
| `casa_run_phase5_online_main_lowmem.py` | Run/resume the low-concurrency online main experiment |
| `casa_merge_phase5_online_results.py` | Merge Phase 5 online lane outputs and write online GO/NO-GO audit artifacts |
| `casa_audit_phase5_online.py` | Audit an existing merged Phase 5 online directory without rerunning episodes |
| `casa_visualize_phase5_online.py` | Generate five-baseline charts, heatmaps, selected episode timelines, metric timeline animations, and visualization reports from online audit artifacts |
| `casa_diagnose_phase5_online_failures.py` | Decompose an online no-go artifact into gate, recovery, skill, and hygiene failures |
| `casa_sweep_phase5_online_policy.py` | Plan or summarize Phase 5 online policy pilot sweeps |
| `casa_calibrate_phase5_hybrid.py` | Replay CASA-Hybrid methods on the Phase 4 calibration split and write calibration-only metrics/config suggestions |
| `casa_sweep_phase5_hybrid.py` | Sweep CASA-Hybrid configs on calibration rows and optionally emit online run commands for non-final candidates |
| `casa_audit_phase5_hybrid.py` | Audit held-out online CASA-Hybrid results against MPC-CBF-Humanoid, SafeDPA, and CASA-A anchors |
| `casa_audit_phase5_acceptance.py` | Audit Phase 5 acceptance criteria |
| `casa_write_phase5_report.py` | Write the Phase 5 conformal baseline report |

Phase 5 online performance iteration notes:

- `casa_audit_phase5_online.py --strict-plan-a-claim` and
  `casa_merge_phase5_online_results.py --strict-plan-a-claim` run the stronger
  reviewer-facing original Plan A claim audit. This mode requires the original
  five methods and `casa_a_per_skill`, checks CASA-A against global conformal,
  raw critic, and hard contract baselines, audits fallback/reject budgets, and
  reports task-progress success separately from safe completion.
- Strict claim thresholds can be adjusted with
  `--min-global-unsafe-reduction`, `--min-global-task-progress-advantage`,
  `--min-raw-unsafe-reduction`, `--max-fallback-rate-per-episode`,
  `--max-reject-rate`, and `--max-walk-reject-rate`. The defaults are intended
  for claim validation, not for weakening the default online audit.
- `casa_visualize_phase5_online.py --strict-five-baseline --video-mode
  timeline-only` builds the Phase 5 online visualization evidence package from
  committed audit artifacts. It writes a metric consistency check before
  plotting and labels generated animations as metric timelines, not simulator
  recordings. Static plots require `matplotlib`; MP4 animations use `ffmpeg`
  when available and fall back to PNG frame sequences otherwise.
- `casa_run_phase5_online_main_lowmem.py --performance-preset hard_or_receding_adaptive`
  runs the PR #8 pilot candidate with `sonic_only`, `hard_contract`, and
  `casa_a_hard_or_receding_recovery`.
- CASA-Hybrid methods are Phase 5 extensions, not replacements for original
  CASA-A. The final method name is `casa_h_mpc_safedpa_casa_refine`; the default
  config `configs/phase5_hybrid_mpc_safedpa.yaml` maps it to `gray_refine` and
  uses the existing `mpc_cbf_humanoid_adapted`, `safedpa_adapted`, and
  `casa_a_per_skill` anchors. Pass it through online runs with
  `--hybrid-config configs/phase5_hybrid_mpc_safedpa.yaml`.
- `casa_audit_phase5_hybrid.py` reports whether the hybrid exceeds each anchor,
  whether any win depends on more reject/fallback intervention, and whether the
  matched intervention-budget check passes. If the hybrid does not beat the
  anchors, the report should be used as a Pareto frontier, not as a success
  claim.
- `casa_sweep_phase5_online_policy.py --emit-run-commands` writes
  `phase5_online_policy_sweep_commands.sh` with pilot low-memory run commands
  and follow-up audit commands.
- `casa_diagnose_phase5_online_failures.py --casa-method
  casa_a_hard_or_receding_recovery` reports retry-aware failure categories when
  the newer gate decision fields are present.
- Final claims still require a fresh held-out strict run; old 2500-episode
  no-go artifacts are tuning evidence only.

## Script Hygiene

- Prefer adding new reusable behavior under `gear_sonic/casa/`.
- Keep CLI scripts thin and phase-specific.
- Write large artifacts to `outputs/`.
- Add new CASA scripts to this index in the same change.
