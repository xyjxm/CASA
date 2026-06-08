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
- `gear_sonic/scripts/casa_diagnose_phase5_online_failures.py`
- `gear_sonic/scripts/casa_sweep_phase5_online_policy.py`
- `gear_sonic/scripts/casa_calibrate_phase5_hybrid.py`
- `gear_sonic/scripts/casa_sweep_phase5_hybrid.py`
- `gear_sonic/scripts/casa_audit_phase5_hybrid.py`
- `gear_sonic/scripts/casa_audit_phase5_acceptance.py`
- `gear_sonic/scripts/casa_write_phase5_report.py`

The five methods are SONIC-only, SONIC plus hard contract, SONIC plus raw critic,
SONIC plus global conformal, and SONIC plus CASA-A per-skill conformal.

CASA-Hybrid is a Phase 5 extension, not the original CASA-A result. The final
method name is `casa_h_mpc_safedpa_casa_refine`. The default config
`configs/phase5_hybrid_mpc_safedpa.yaml` maps that method to `gray_refine`,
keeps original `casa_a_per_skill` unchanged, and uses the existing
`mpc_cbf_humanoid_adapted` and `safedpa_adapted` SOTA adapters as anchors.
Hybrid calibration and sweep scripts operate on Phase 4 calibration rows only;
the final held-out online test must not be used for threshold or strategy
tuning.

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
The online audit now also reports `safe_completion_rate` and
`task_progress_success_rate`. The legacy `task_success_rate` field remains for
backward compatibility, while strict claim review should use the task-progress
fields to avoid treating a stopped-but-safe episode as semantic task success.

Hard-contract comparisons distinguish blocking checks from diagnostics. If the
fixed hard contract accepts zero unsafe examples, relative unsafe reduction is
undefined and is reported as `null` rather than failing as `0.0`. If the fixed
hard contract over-rejects safe samples, the fixed comparison is diagnostic and
the matched safe-rejection-budget hard comparison is used for the fair blocking
check. CASA-vs-global FNR closeness is diagnostic; the hard safety requirement
is one-sided per-skill FNR control.

### Phase 5 Online Audit

The online experiment evaluates the original five methods, plus optional
candidate methods, with real episode execution.
`casa_run_phase5_online_main_lowmem.py` launches/resumes lane jobs, then calls
`casa_merge_phase5_online_results.py` with the expected method, seed, episode,
and skill-count constraints. A representative 2500-episode baseline main run is:

```bash
python gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py \
  --phase5-root outputs/casa/phase5_conformal_baselines_YYYYMMDD \
  --online-root outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_2500 \
  --seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100 \
  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill
```

For the CASA-Hybrid held-out comparison, include the two adapted SOTA anchors
and pass the hybrid config through the low-memory runner:

```bash
python gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py \
  --online-root outputs/casa/phase5_hybrid_mpc_safedpa_full/online_full_5seed100 \
  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill,mpc_cbf_humanoid_adapted,safedpa_adapted,casa_h_mpc_safedpa_casa_refine \
  --casa-method casa_h_mpc_safedpa_casa_refine \
  --hybrid-config configs/phase5_hybrid_mpc_safedpa.yaml \
  --seeds 3001,3002,3003,3004,3005 \
  --episodes-per-seed 100 \
  --max-parallel 3 \
  --chunk-size 25 \
  --max-sweeps 6
```

After the run is merged, audit the hybrid claim separately:

```bash
python gear_sonic/scripts/casa_audit_phase5_hybrid.py \
  --online-dir outputs/casa/phase5_hybrid_mpc_safedpa_full/online_full_5seed100/merged \
  --hybrid-method casa_h_mpc_safedpa_casa_refine
```

The hybrid audit explicitly reports whether the hybrid exceeds
`mpc_cbf_humanoid_adapted`, `safedpa_adapted`, and `casa_a_per_skill`; whether a
win is only due to additional fallback/reject intervention; and whether matched
intervention-budget checks pass. If the hybrid does not exceed the anchors, the
correct output is the Pareto frontier and next-step recommendations, not a
success claim.

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

Use `--strict-plan-a-claim` for reviewer-facing checks of the strongest original
Plan A paper claim. This mode keeps the normal online audit available, but it
adds blockers for claim-validity risks that the default engineering GO/NO-GO
does not require:

- all five original methods must be present:
  `sonic_only`, `hard_contract`, `raw_critic_0p5`, `global_conformal`, and
  `casa_a_per_skill`;
- the CASA method must be original `casa_a_per_skill`; recovery, hard-OR, and
  receding methods are Plan A+ variants;
- CASA-A must show a defensible advantage over `global_conformal`;
- CASA-A must also clear raw-critic and hard-contract unsafe-reduction checks;
- fallback/reject budgets must stay within the strict reviewer thresholds; and
- safe completion must be separated from task-progress or semantic success.

For example:

```bash
python gear_sonic/scripts/casa_audit_phase5_online.py \
  --online-dir outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_2500/merged \
  --output-dir outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_2500/strict_claim_audit \
  --expected-episodes 2500 \
  --expected-methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill \
  --expected-seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100 \
  --strict-plan-a-claim
```

The online audit writes:

- `online_episode_results.csv`: one row per method/seed/episode.
- `gate_decisions.csv`: one row per gate decision inside completed episodes.
- `method_summary.csv` and `method_summary.json`: per-method legacy task
  success, safe completion, task-progress, fallback, violation, and
  unsafe-invocation totals/rates.
- `online_acceptance_audit.json`: full schema validation, checks, warnings,
  confidence intervals, baseline comparisons, anti-gaming diagnostics, and
  per-skill diagnostics.
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

### Plan A Completion Evidence

The default tracked evidence for the 2026-06-02 Plan A audit lives in
`idea_and_plan/plan_a_completion_20260602/`. It includes merged online CSVs,
audit reports, the strict phase audit script, `artifact_manifest.json`,
`run_command.sh`, expected method/seed files, source artifact hashes, and
per-CSV sha256 files. The same evidence should also be reachable from a clearly
named tag or release when it is promoted for review.

The archived default online audit reports `PASS_STRICT_ONLINE` for the
engineering acceptance checks. The nested strict Plan A claim diagnostic reports
`STRICT_PLAN_A_NO_GO`: the current original `casa_a_per_skill` evidence does not
outperform `global_conformal`, and it exceeds the strict fallback/reject/walk
reject budgets. That distinction is intentional. Do not describe the archived
evidence as supporting the strongest original Plan A claim unless the
`--strict-plan-a-claim` audit becomes GO on fresh or promoted evidence.

### Phase 5 Visualization Evidence

Use `casa_visualize_phase5_online.py` to turn the five-baseline online audit
artifacts into reviewer-readable figures, heatmaps, representative episode
timelines, and metric timeline animations. The visualization package recomputes
the plotted metrics from `online_episode_results.csv` and `gate_decisions.csv`,
checks them against `method_summary.json` and `online_acceptance_audit.json`,
and writes `data/metric_consistency_check.json` before plotting. It does not
change audit metrics or strict Plan A claim status. Static figures require
`matplotlib`; MP4 metric timeline animations use `ffmpeg` when available and
fall back to PNG frame sequences otherwise.

Generate static figures plus metric timeline animations from the committed Plan
A completion evidence:

```bash
python gear_sonic/scripts/casa_visualize_phase5_online.py \
  --artifact-dir idea_and_plan/plan_a_completion_20260602 \
  --audit-dir idea_and_plan/plan_a_completion_20260602/audit \
  --output-dir outputs/casa/phase5_online_visualizations \
  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill \
  --casa-method casa_a_per_skill \
  --strict-five-baseline \
  --video-mode timeline-only \
  --write-markdown \
  --write-html \
  --fail-on-metric-mismatch
```

If real simulator recordings already exist, point the visualizer at them. When
matching videos are unavailable, it writes only metric timeline animations and
labels them as not simulator recordings:

```bash
python gear_sonic/scripts/casa_visualize_phase5_online.py \
  --artifact-dir idea_and_plan/plan_a_completion_20260602 \
  --audit-dir idea_and_plan/plan_a_completion_20260602/audit \
  --output-dir outputs/casa/phase5_online_visualizations_real_video \
  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill \
  --casa-method casa_a_per_skill \
  --strict-five-baseline \
  --video-mode real \
  --videos-dir outputs/casa/phase5_online_recordings \
  --write-markdown \
  --write-html \
  --fail-on-metric-mismatch
```

The visualizer writes `phase5_online_visualization_report.md`, optional static
HTML, `figures/`, `videos/`, and `data/` outputs under the requested
`--output-dir`. Raw simulator videos are never fabricated; without real video
inputs, the video evidence is a metric timeline animation derived from audit
CSV/JSON rows.

### Phase 5 Online Performance Iteration

Issue #5 made the online audit reproducible and explainable. Issue #7 targets
the actual online performance gap: reaching strict online `GO` on fresh held-out
2500-episode data without weakening the audit.

PR #8 implements the Issue #7 performance iteration candidate:
`casa_a_hard_or_receding_recovery`. It combines hard-contract OR CASA rejection,
long-skill segmentation, per-segment rechecks, adaptive recovery, and a bounded
recovery-then-retry loop. Retry attempts log recovery and retry decision rows
with `attempt_type`, `parent_skill_idx`, original candidate parameters,
recovery counts, retry outcome, task-progress flags, and final segment outcome.

First decompose the current no-go artifact as dev/tuning evidence:

```bash
python gear_sonic/scripts/casa_diagnose_phase5_online_failures.py \
  --online-dir outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_2500/merged \
  --output-dir outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_2500/diagnostics \
  --casa-method casa_a_hard_or_receding_recovery
```

The diagnostic report writes `phase5_online_failure_breakdown.json` and
`phase5_online_failure_breakdown.md`. It separates gate recall
(`allow_then_unsafe`), failed recovery (`reject_but_still_unsafe`),
task-success over-rejection, low-risk unsafe events that suggest online
distribution shift, reset/upright hygiene, per-skill risk/threshold margins,
target buckets, scene complexity, top failure clusters, and recommended next
interventions. Newer retry-aware artifacts also separate
`allow_then_unsafe_segment`, `reject_then_recovery_only`,
`reject_then_recovery_then_retry`, `reject_then_retry_allowed_but_unsafe`,
`reject_then_all_retries_rejected`, `no_task_progress_after_reject`,
`unsafe_after_recovery`, `hard_contract_caught_casa_missed`, and
`over_rejection_safe_segments`; older artifacts report
`diagnostics.missing_retry_fields` instead of fabricated segment metrics.

Candidate online methods are available in addition to the original five
baselines:

- `casa_a_hard_or_per_skill`: reject when hard contract fires OR per-skill CASA
  threshold rejects.
- `casa_a_recovery_per_skill`: per-skill CASA rejection with adaptive recovery.
- `casa_a_hard_or_recovery`: hard-OR-CASA rejection with adaptive recovery.
- `casa_a_receding_recovery`: segmented long skills with per-segment rechecks
  and adaptive retry recovery.
- `casa_a_hard_or_receding_recovery`: hard-OR-CASA rejection with segmented
  receding rechecks and bounded recovery-then-retry.

Online policy flags include:

- `--fallback-policy stop|adaptive|adaptive_retry|auto`
- `--segment-long-skills`
- `--max-segment-duration 0.5`
- `--recheck-before-segment`
- `--threshold-scale-global 0.8`
- `--threshold-scale-by-skill walk=0.8,turn=0.9,gesture=0.7,passive=1.0`
- `--randomize-method-order`
- `--method-order-seed 20260531`
- `--performance-preset hard_or_receding_adaptive`

Run a small pilot of the PR #8 candidate with the low-memory preset:

```bash
python gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py \
  --phase4-root outputs/casa/phase4_dataset_v1_strict_50k_YYYYMMDD \
  --phase5-root outputs/casa/phase5_conformal_baselines_YYYYMMDD \
  --online-root outputs/casa/pilots/pr8_hard_or_receding_adaptive \
  --performance-preset hard_or_receding_adaptive \
  --seeds 2001,2002 \
  --episodes-per-seed 20 \
  --max-parallel 2 \
  --chunk-size 10
```

Plan pilot sweeps before full runs:

```bash
python gear_sonic/scripts/casa_sweep_phase5_online_policy.py \
  --output-dir outputs/casa/phase5_online_policy_sweep_YYYYMMDD \
  --emit-run-commands \
  --phase4-root outputs/casa/phase4_dataset_v1_strict_50k_YYYYMMDD \
  --phase5-root outputs/casa/phase5_conformal_baselines_YYYYMMDD \
  --online-root outputs/casa/pilots/phase5_online_policy_sweep_YYYYMMDD \
  --seeds 2001,2002 \
  --episodes-per-seed 20 \
  --fallback-policies adaptive,adaptive_retry \
  --hard-or-casa true \
  --segment-long-skills true \
  --max-segment-durations 0.5,0.75 \
  --threshold-scale-global 0.7,0.8,0.9 \
  --recovery-retry-counts 1,2
```

The emitted `phase5_online_policy_sweep_commands.sh` contains one low-memory
run command per planned candidate, the named PR #8 performance preset command,
and an audit command for each candidate output.

After pilot runs finish, summarize completed candidate directories:

```bash
python gear_sonic/scripts/casa_sweep_phase5_online_policy.py \
  --output-dir outputs/casa/phase5_online_policy_sweep_YYYYMMDD/summary \
  --candidate hard_or_recovery:outputs/casa/pilots/hard_or_recovery/merged \
  --candidate receding_recovery:outputs/casa/pilots/receding_recovery/merged
```

Only pilot candidates with unsafe reduction versus SONIC of at least `0.45`,
relative task-success drop no more than `0.20`, no catastrophic per-skill
failure, and clean reset/rollout hygiene should be promoted to a fresh held-out
2500-episode strict run. A final held-out command for the promoted PR #8
candidate should use fresh seeds/data and strict audit thresholds:

```bash
python gear_sonic/scripts/casa_run_phase5_online_main_lowmem.py \
  --phase4-root outputs/casa/phase4_dataset_v1_strict_50k_YYYYMMDD \
  --phase5-root outputs/casa/phase5_conformal_baselines_YYYYMMDD \
  --online-root outputs/casa/phase5_conformal_baselines_YYYYMMDD/online_real_main_pr8_2500 \
  --seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100 \
  --methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_hard_or_receding_recovery \
  --casa-method casa_a_hard_or_receding_recovery \
  --fallback-policy auto \
  --adaptive-retry-count 2 \
  --segment-long-skills \
  --recheck-before-segment \
  --max-segment-duration 0.5 \
  --threshold-scale-by-skill walk=0.8,turn=0.9,gesture=0.7,passive=1.0 \
  --randomize-method-order \
  --method-order-seed 20260531
```

The current no-go 2500 artifact is dev/tuning evidence only and must not be
reused as final success evidence after policy tuning.

## Outputs And Reproducibility

Generated outputs should stay under `outputs/` and remain untracked. Reports that
are small enough for review can be copied into `idea_and_plan/` or `docs/` after
removing local paths, secrets, and large generated data.

Before opening a pull request, run the public repository scan checklist in
`docs/PUBLIC_REPO_SECURITY_SCAN.md`.
