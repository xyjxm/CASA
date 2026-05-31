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

Phase 4 preserves independent roles for Phase 5:

- `train`: raw critic fitting.
- `critic_val` / `val`: raw critic model selection and early stopping.
- `calibration` / `conformal_calibration`: untouched conformal threshold calibration.
- `test`: final held-out evaluation.

The raw critic must not use the conformal calibration role for best-epoch
selection. If an older dataset has no explicit critic validation role,
`casa_train_raw_critic.py` deterministically carves one from `train` and leaves
`calibration` untouched.

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
- `gear_sonic/scripts/casa_audit_phase5_online.py`
- `gear_sonic/scripts/casa_audit_phase5_acceptance.py`
- `gear_sonic/scripts/casa_write_phase5_report.py`

The five methods are SONIC-only, SONIC plus hard contract, SONIC plus raw critic,
SONIC plus global conformal, and SONIC plus CASA-A per-skill conformal.

Phase 5 validates `raw_critic/predictions.csv` before threshold selection. The
CSV must contain `sample_id`, `phase4_split`, `skill_name`, `label`,
`raw_critic_risk`, `hard_contract_score`, and `hard_contract_fixed_reject`;
labels must be exactly `0` or `1`; risk and hard-contract scores must be finite;
all main skills must appear in calibration and test; and calibration must have
enough unsafe samples per skill.

Threshold selection modes:

- `max_fnr`: largest threshold satisfying calibration FNR <= alpha.
- `conservative`: diagnostic lower-FNR thresholds.
- `acceptance_search`: searches thresholds subject to calibration FNR and
  safe-rejection budget constraints, then minimizes accepted unsafe samples.

Offline reports use `safe_acceptance_rate` for safe-sample acceptance /
false-positive control. This is not real task completion; online reports use
`task_success_rate` when episode completion labels exist.

Hard-contract comparisons distinguish blocking checks from diagnostics. If the
fixed hard contract accepts zero unsafe examples, relative unsafe reduction is
undefined and is reported as `null` rather than failing as `0.0`. If the fixed
hard contract over-rejects safe samples, the fixed comparison is diagnostic and
the matched safe-rejection-budget hard comparison is used for the fair blocking
check. CASA-vs-global FNR closeness is diagnostic; the hard safety requirement
is one-sided per-skill FNR control.

### Phase 5 Online Audit

The online experiment evaluates the same five methods with real episode
execution. `casa_run_phase5_online_main_lowmem.py` launches/resumes lane jobs,
then calls `casa_merge_phase5_online_results.py` with the expected method, seed,
episode, and skill-count constraints. A representative 2500-episode main run is:

```bash
python gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py \
  --phase5-root outputs/casa/phase5_conformal_baselines_YYYYMMDD \
  --output-dir outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_2500 \
  --seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100 \
  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill
```

For an already merged online directory, rerun only the audit:

```bash
python gear_sonic/scripts/casa_audit_phase5_online.py \
  --online-dir outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_2500/merged \
  --expected-episodes 2500 \
  --expected-methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill \
  --expected-seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100
```

Use `--strict` in CI or release checks when a non-GO audit should return a
non-zero exit code. Without `--strict`, the script still writes the full audit
artifacts so a failed online run remains inspectable.

The online audit writes:

- `online_episode_results.csv`: one row per method/seed/episode.
- `gate_decisions.csv`: one row per gate decision inside completed episodes.
- `method_summary.csv` and `method_summary.json`: per-method task success,
  fallback, violation, and unsafe-invocation totals/rates.
- `online_acceptance_audit.json`: full schema validation, checks, warnings,
  confidence intervals, baseline comparisons, and per-skill diagnostics.
- `online_go_no_go.json`: compact status, blocking reasons, warnings, and
  actionable next steps.
- `online_report.md`: human-readable online report.

Online acceptance is intentionally stricter than report generation. Missing
fields, invalid numeric values, duplicate method/seed/episode rows, incomplete
expected grids, missing gate decisions, non-upright starts, failed episodes, or
missing per-skill gate coverage block strict acceptance. Warnings and diagnostics
are kept separate from blockers. For example, low baseline task-success rates or
hard-contract over-intervention are reported as warnings because they affect
interpretation, but they do not by themselves force `ONLINE_NO_GO`.

The audit must not force a `PASS_STRICT_ONLINE` result. If the real data still
fails safety or task-success criteria, status remains `ONLINE_NO_GO` and the
report lists actionable blockers such as insufficient CASA-vs-SONIC unsafe
reduction or excessive task-success drop.

## Outputs And Reproducibility

Generated outputs should stay under `outputs/` and remain untracked. Reports that
are small enough for review can be copied into `idea_and_plan/` or `docs/` after
removing local paths, secrets, and large generated data.

Before opening a pull request, run the public repository scan checklist in
`docs/PUBLIC_REPO_SECURITY_SCAN.md`.
