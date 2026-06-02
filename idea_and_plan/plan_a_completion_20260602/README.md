# Plan A Completion Evidence (2026-06-02)

This directory archives the strict Plan A completion evidence, excluding only
the explicitly allowed Phase2 independent manual human review.

## Overall Result

- Plan A strict completion audit: `PASS`
- Blocking reasons: none
- Phase0: `PASS`
- Phase1: `PASS`
- Phase2: `PASS_WITH_ALLOWED_EXCEPTION`
- Phase3: `PASS`
- Phase4: `PASS`
- Phase5 offline: `PASS`
- Phase5 online: `PASS`

The allowed exception is limited to independent manual human review for Phase2.
All automated and artifact-backed requirements are included in the strict audit.

## Phase5 Online Main Experiment

- Online audit status: `PASS_STRICT_ONLINE`
- GO status: `true`
- Completed online episodes: `2500 / 2500`
- Deduplicated episode rows: `2500`
- Deduplicated gate-decision rows: `20000`
- Expected method/seed/episode grid: complete
- Duplicate completed rows in the source scan before merge: `0`

Key acceptance metrics from `audit/online_acceptance_audit.json`:

- CASA vs SONIC unsafe reduction: `0.7375667322281539`
- CASA vs hard-contract unsafe reduction: `0.7436178973373593`
- CASA vs SONIC task-success drop rel: `-22.46153846153846`
- CASA vs global-conformal task-success drop rel: `0.07012195121951226`
- CASA task-success drop checks: pass

## Archived Artifacts

- `plan_a_strict_completion_audit_20260602.json`: strict Plan A phase audit.
- `audit/online_acceptance_audit.json`: full Phase5 online acceptance audit.
- `audit/online_go_no_go.json`: compact GO/NO-GO summary.
- `audit/online_report.md`: human-readable online report.
- `audit/method_summary.json` and `audit/method_summary.csv`: method metrics.
- `audit/online_episode_results.csv`: merged online episode table.
- `audit/gate_decisions.csv`: merged per-skill gate decisions.
- `scripts/plan_a_strict_completion_audit_20260602.py`: strict audit script.
- `scripts/phase5_online_main_watchdog_20260602.py`: watchdog used for final online run stability.
- `scripts/phase5_online_sidecar_supervisor_20260602.py`: sidecar supervisor used for tail completion.

Source output root:

`/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602`
