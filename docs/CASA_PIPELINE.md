# CASA Pipeline

This document maps the CASA Phase 0-5 workflow to the public source tree. Paths
below are relative to the repository root.

## Common Setup

```bash
git clone https://github.com/xyjxm/CASA.git
cd CASA
git lfs install
git lfs pull
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
python check_environment.py
```

Full rollouts require a MuJoCo sim process, a deploy process, and enough local
storage for `outputs/`. The public repository tracks source code and small
assets; it does not track generated datasets or trained critic checkpoints.

## Phase 0: Skill Sanity

Goal: confirm that the CASA skill wrappers can drive the base controller and
produce complete logs.

Representative scripts:

- `gear_sonic/scripts/casa_run_sanity_check.py`
- `gear_sonic/scripts/casa_run_skill_repeatability.py`
- `gear_sonic/scripts/casa_verify_gesture_amplitude.py`

Expected outputs live under `outputs/casa/phase0_*` and include skill timelines,
rollout summaries, and sanity reports.

## Phase 1: Repeatability And Skill Evidence

Goal: run repeated skill invocations and check that skill boundaries, command
logs, and oracle evidence align.

Representative scripts:

- `gear_sonic/scripts/casa_run_skill_repeatability.py`
- `gear_sonic/scripts/casa_run_rollout_with_oracle.py`

Acceptance focuses on complete logs, stable skill start/end timestamps, and
interpretable oracle summaries.

## Phase 2: Clean Visual Dataset

Goal: collect visual-priority rollout videos with clean scene placement and
structured artifact evidence.

Representative scripts:

- `gear_sonic/scripts/casa_make_phase2_v2_scene_plan.py`
- `gear_sonic/scripts/casa_phase2_v2_scene_placement_test.py`
- `gear_sonic/scripts/casa_run_phase2_v2_lane.py`
- `gear_sonic/scripts/casa_render_rollout_videos.py`
- `gear_sonic/scripts/casa_build_clean_visual_dataset_v2.py`
- `gear_sonic/scripts/casa_package_clean_visual_dataset.py`

Strict Phase 2 v2 accepts only natural rollouts with released elastic-band state,
initial clearance, no startup overlaps, no runtime/latency artifacts, and
visually explainable unsafe events.

## Phase 3: Invocation Dataset Feasibility

Goal: convert rollout logs into state-skill-label invocations and validate that
the feature/label path can train a small safety critic.

Representative scripts:

- `gear_sonic/scripts/casa_build_invocation_dataset.py`
- `gear_sonic/scripts/casa_collect_phase3_feasibility.py`
- `gear_sonic/scripts/casa_run_counterfactual_subset.py`
- `gear_sonic/scripts/casa_train_mini_critic.py`
- `gear_sonic/scripts/casa_audit_phase3_acceptance.py`
- `gear_sonic/scripts/casa_write_phase3_report.py`

Acceptance focuses on leakage-free splits, complete labels, feature integrity,
and evidence that the critic pipeline is trainable.

## Phase 4: Strict Clean 50k Dataset

Goal: build a strict 50k clean invocation dataset and train the raw critic.

Representative scripts:

- `gear_sonic/scripts/casa_make_phase4_collection_plan.py`
- `gear_sonic/scripts/casa_build_phase4_dataset.py`
- `gear_sonic/scripts/casa_train_raw_critic.py`
- `gear_sonic/scripts/casa_audit_phase4_acceptance.py`
- `gear_sonic/scripts/casa_write_phase4_report.py`

Strict Phase 4 excludes runtime timeouts, injected latency, artifact sources,
missing critical fields, and split leakage. The main acceptance file is
`phase4_go_no_go.json`.

## Phase 5: Conformal Gates And Baselines

Goal: calibrate global and per-skill conformal thresholds and evaluate five
methods on the same Phase 4 strict clean test split.

Representative scripts:

- `gear_sonic/scripts/casa_calibrate_phase5_conformal.py`
- `gear_sonic/scripts/casa_eval_phase5_baselines.py`
- `gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py`
- `gear_sonic/scripts/casa_run_phase5_online_lane.py`
- `gear_sonic/scripts/casa_run_phase5_online_experiment.py`
- `gear_sonic/scripts/casa_merge_phase5_online_results.py`
- `gear_sonic/scripts/casa_audit_phase5_acceptance.py`
- `gear_sonic/scripts/casa_write_phase5_report.py`

The five methods are SONIC-only, SONIC plus hard contract, SONIC plus raw critic,
SONIC plus global conformal, and SONIC plus CASA-A per-skill conformal.

## Outputs And Reproducibility

Generated outputs should stay under `outputs/` and remain untracked. Reports that
are small enough for review can be copied into `idea_and_plan/` or `docs/` after
removing local paths, secrets, and large generated data.

Before opening a pull request, run the public repository scan checklist in
`docs/PUBLIC_REPO_SECURITY_SCAN.md`.
