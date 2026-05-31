# CASA Plan A Progress Report

Date: 2026-05-31

This report audits the current implementation status of Plan A against
`/mnt/data/students/lph/idea_and_plan/plan_a.md` using the local
`GR00T-WholeBodyControl` workspace.

## Executive Summary

Plan A has reached an end-to-end offline implementation through Phase 5, but it
is not yet fully complete under the strict Plan A claim. Phases 0-4 are largely
implemented and have acceptance artifacts. Phase 5 has offline conformal
calibration and five-baseline evaluation artifacts, but the main calibration
distribution deviates from the plan, and the online main experiment is still
NO-GO.

Overall status:

| Phase | Status | Notes |
|---|---|---|
| Phase 0 | Passed | 30-minute SONIC/CASA sanity run completed with 423 skill executions and 0.983 success rate. |
| Phase 1 | Passed | Four CASA skill wrappers, registry, executor, and skill logging are implemented. |
| Phase 2 | Engineering pass; formal review incomplete | Oracle/logger pipeline is implemented, but the acceptance review is assisted rather than independent blind review. |
| Phase 3 | Passed in later run | 5k feasibility dataset, mini critic, and counterfactual subset pass in `phase3_hybrid_phase2v2_manifest_20260521`. |
| Phase 4 | Passed strict 50k target | 50k clean dataset and Raw Critic pass strict checks; 80k ideal target is not met. |
| Phase 5 | Offline pass; strict claim incomplete | Offline conformal/baselines pass, but calibration does not use the Hard-Contract-filtered deployment subdistribution; online main experiment is NO-GO. |

## Phase Evidence

### Phase 0: SONIC Sanity

Evidence:

- `outputs/casa/phase0_sanity/casa_phase0_full30_video_20260516_110120/full30/sanity_report.md`
- `outputs/casa/phase0_sanity/casa_phase0_full30_video_20260516_110120/full30/sanity_summary.json`

Key result:

- Duration: 1800 seconds
- Executed skills: 423
- Success rate: 0.983
- Go criteria: true

### Phase 1: Skill API

Implemented components:

- `gear_sonic/casa/skills/base.py`
- `gear_sonic/casa/skills/walk.py`
- `gear_sonic/casa/skills/passive.py`
- `gear_sonic/casa/skills/turn.py`
- `gear_sonic/casa/skills/gesture.py`
- `gear_sonic/casa/skills/registry.py`
- `gear_sonic/casa/skills/executor.py`
- `gear_sonic/casa/loggers/skill_logger.py`

The implementation covers:

- `WalkSkill(vx, vy, facing_yaw_deg, duration)`
- `PassiveSkill(duration, mode=stop/wait)`
- `TurnSkill(face_yaw_deg, duration)`
- `GestureSkill(amplitude, frequency, side, duration)`
- `ExecutionResult` with parameters, timing, status, termination reason, and evidence.

### Phase 2: Safety Oracle and Logger

Implemented components:

- `gear_sonic/casa/oracle/oracle.py`
- `gear_sonic/casa/oracle/violations.py`
- `gear_sonic/casa/oracle/thresholds.py`
- `gear_sonic/casa/loggers/rollout_logger.py`

The oracle covers the six Plan A violation types:

- collision
- near_collision
- fall
- human_distance_violation
- unsafe_gesture
- runtime_timeout

Evidence:

- `outputs/casa/phase2_oracle/phase2_oracle_acceptance_plus_targeted_20260516_165220/review/phase2_report.md`

Key result:

- Discovered rollouts: 360
- Review rollouts: 360
- Skill labels: 2431 unsafe, 409 safe, 17 unverified
- Violation counts include collision, near_collision, fall, human distance, unsafe gesture, and runtime timeout.

Remaining issue:

- The Phase 2 report records `assisted_review: True`. This validates the
  pipeline, but it is not the independent blind human review required for a
  stronger formal acceptance claim.

### Phase 3: Hybrid Feasibility and Mini Critic

Latest passing evidence:

- `outputs/casa/phase3_feasibility/phase3_hybrid_phase2v2_manifest_20260521/phase3_report.md`
- `outputs/casa/phase3_feasibility/phase3_hybrid_phase2v2_manifest_20260521/go_no_go.json`

Key result:

- Overall: GO
- Selected samples: 5000
- Raw samples: 14708
- Label complete rate: 0.9825
- Overall positive rate: 0.213
- Per-skill positive rates:
  - walk: 0.4358
  - turn: 0.1104
  - gesture: 0.0506
  - passive: 0.1504
- Mini critic test AUROC: 0.9855
- Mini critic test Brier: 0.0436
- Counterfactual branches: 2000
- Mean repeat consistency: 0.996
- Acceptance audit: PASS

### Phase 4: ID Dataset v1 and Raw Critic

Evidence:

- `outputs/casa/phase4_dataset_v1_strict_50k_20260522/phase4_report.md`
- `outputs/casa/phase4_dataset_v1_strict_50k_20260522/phase4_acceptance_audit.md`
- `outputs/casa/phase4_dataset_v1_strict_50k_20260522/dataset_v1`
- `outputs/casa/phase4_dataset_v1_strict_50k_20260522/raw_critic`

Key result:

- Status: PASS_STRICT
- Selected samples: 50000
- Clean available samples: 57372
- Raw samples: 60189
- Usable samples: 59512
- Positive rate: 0.18998
- Runtime stress quarantine samples: 2140
- Per-skill totals:
  - gesture: 10702 total, 3175 unsafe
  - passive: 16364 total, 1500 unsafe
  - turn: 11422 total, 1967 unsafe
  - walk: 11512 total, 2857 unsafe
- Raw Critic test AUROC: 0.9992
- Raw Critic test AUPRC: 0.9983
- Raw Critic test Brier: 0.0045
- Raw Critic go criteria: true

Remaining issue:

- The strict 50k target is met, but the Plan A ideal target of 80k is not met.

### Phase 5: Per-skill Conformal and Five Baselines

Offline evidence:

- `outputs/casa/phase5_conformal_baselines_20260522/phase5_report.md`
- `outputs/casa/phase5_conformal_baselines_20260522/phase5_acceptance_audit.md`
- `outputs/casa/phase5_conformal_baselines_20260522/conformal_thresholds.json`
- `outputs/casa/phase5_conformal_baselines_20260522/baseline_results.csv`
- `outputs/casa/phase5_conformal_baselines_20260522/per_skill_fnr.csv`
- `outputs/casa/phase5_conformal_baselines_20260522/rejection_risk_curve.csv`

Offline key result:

| Method | Unsafe invocations | FNR | Reject rate | Task success proxy |
|---|---:|---:|---:|---:|
| SONIC-only | 1439 | 1.0000 | 0.0000 | 1.0000 |
| SONIC + Hard Contract | 199 | 0.1383 | 0.1669 | 0.9993 |
| SONIC + Raw Critic (0.5) | 17 | 0.0118 | 0.1937 | 0.9963 |
| SONIC + Global Conformal | 121 | 0.0841 | 0.1770 | 0.9998 |
| SONIC + CASA-A | 117 | 0.0813 | 0.1778 | 0.9995 |

Per-skill FNR:

| Skill | CASA-A FNR | Global FNR | Raw Critic FNR |
|---|---:|---:|---:|
| walk | 0.0925 | 0.1366 | 0.0044 |
| turn | 0.0473 | 0.1236 | 0.0255 |
| gesture | 0.0794 | 0.0060 | 0.0020 |
| passive | 0.1068 | 0.1068 | 0.0340 |

Main blocker:

- Plan A requires conformal calibration on the Hard-Contract-filtered
  deployment subdistribution.
- The current Phase 5 implementation uses the full Phase 4 strict clean
  calibration split.
- The hard-contract-filtered subset is only diagnostic because it has too few
  dangerous samples per skill:
  - gesture unsafe: 2
  - passive unsafe: 7
  - turn unsafe: 23
  - walk unsafe: 174

This is the largest mismatch between the implementation and the written Plan A.

Online evidence:

- `outputs/casa/phase5_conformal_baselines_20260522/online_real_main_2500_v5_fixed_20260524/merged/online_report.md`
- `outputs/casa/phase5_conformal_baselines_20260522/online_real_main_2500_v5_fixed_20260524/merged/online_acceptance_audit.json`

Online key result:

- Status: ONLINE_NO_GO
- Episodes: 2500 / 2500
- Completed: 2500
- Gate decisions: 20000
- CASA-A unsafe invocations: 2737
- SONIC-only unsafe invocations: 3616
- CASA-A task success rate: 0.002
- SONIC-only task success rate: 0.020

Failed checks:

- `casa_vs_sonic_unsafe_reduction_ge_40pct`
- `casa_task_success_drop_rel_le_30pct`

## Main Risks

1. Calibration distribution mismatch

   The plan requires Hard-Contract-filtered calibration. The current offline
   Phase 5 pass uses full calibration because the filtered subset is too small.
   This weakens the exchangeability/deployment-distribution argument.

2. Raw Critic is stronger than CASA-A in the current offline table

   Raw Critic (0.5) has fewer unsafe invocations and lower FNR than CASA-A in
   the offline test table. The report supports per-skill versus global
   conformal more than it supports CASA-A versus Raw Critic.

3. Online main experiment is not passing

   The online 2500-episode experiment is complete, but it is NO-GO under the
   current acceptance checks.

4. Phase 2 independent review is incomplete

   The oracle pipeline is implemented, but the strongest formal claim still
   needs independent human blind review or an equivalent documented review
   process.

5. Repository state is not clean

   The local workspace contains many uncommitted CASA implementation files and
   generated reports. A careful commit strategy is needed to avoid mixing
   source changes, generated artifacts, and review reports in one large commit.

## Recommended Next Steps

1. Collect or resample a Hard-Contract-filtered calibration subset with at least
   200 dangerous samples per skill.
2. Re-run Phase 5 conformal calibration using that filtered calibration split.
3. Re-evaluate the five offline baselines and make the comparison against Raw
   Critic explicit and fair.
4. Investigate the online NO-GO, especially the high unsafe count and low task
   success for CASA-A.
5. Finish Phase 2 independent blind review and attach the agreement report.
6. Split GitHub uploads into small commits:
   - report-only commit
   - CASA source/tooling commit
   - docs/tutorial commit
   - optional generated artifact manifest commit

