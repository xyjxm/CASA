# Plan A Completion Evidence (2026-06-02)

This directory archives the 2026-06-02 Plan A evidence package, excluding only
the explicitly allowed Phase2 independent manual human review. It now separates
two claims that must not be mixed:

- Automated phase/default online audit: `PASS`
- Strongest original Plan A claim audit: `STRICT_PLAN_A_NO_GO`

## Automated Phase Result

- Plan A strict completion audit: `PASS`
- Blocking reasons: none
- Phase0: `PASS`
- Phase1: `PASS`
- Phase2: `PASS_WITH_ALLOWED_EXCEPTION`
- Phase3: `PASS`
- Phase4: `PASS`
- Phase5 offline: `PASS`
- Phase5 default online audit: `PASS_STRICT_ONLINE`

The allowed exception is limited to independent manual human review for Phase2.
All other automated and artifact-backed phase requirements are included in the
strict phase audit.

## Strict Original Plan A Claim Result

The reviewer-facing strict claim diagnostic in
`audit/online_acceptance_audit.json` is `STRICT_PLAN_A_NO_GO`.

Strict claim blockers:

- `casa_a_per_skill_does_not_outperform_global_conformal`
- `casa_fallback_rate_exceeds_strict_plan_a_budget`
- `casa_reject_rate_exceeds_strict_plan_a_budget`
- `casa_walk_reject_rate_exceeds_strict_plan_a_budget`

Key anti-gaming metrics:

- CASA-vs-global unsafe reduction: `-0.10271546635182999`
- CASA task-progress success rate: `0.3485`
- Global-conformal task-progress success rate: `0.462`
- CASA fallback rate per episode: `5.212`
- CASA reject rate per decision: `0.6515`
- CASA walk reject rate: `0.9246666666666666`
- Matched observed-budget diagnostic winner: `global_conformal`

This evidence is therefore valid for the automated engineering audit, but it
must not be cited as supporting the strongest original Plan A paper claim.

## Phase5 Online Main Experiment

- Online audit status: `PASS_STRICT_ONLINE`
- GO status: `true`
- Completed online episodes: `2500 / 2500`
- Raw episode rows before dedupe/provenance merge: `3173`
- Deduplicated episode rows: `2500`
- Deduplicated gate-decision rows: `20000`
- Expected method/seed/episode grid: complete

Key default acceptance metrics from `audit/online_acceptance_audit.json`:

- CASA vs SONIC unsafe reduction: `0.7375667322281539`
- CASA vs hard-contract unsafe reduction: `0.7436178973373593`
- CASA vs SONIC task-success drop rel: `-22.46153846153846`
- CASA vs global-conformal task-success drop rel: `0.07012195121951226`
- CASA task-success drop checks: pass

## Reproducibility Manifest

- `artifact_manifest.json`: manifest for expected methods, seeds, artifact
  roots, committed artifact hashes, threshold hash, and raw critic checkpoint
  hash.
- `run_command.sh`: reruns the default audit and the expected-failing strict
  Plan A claim audit.
- `git_commit.txt`: evidence source commit.
- `expected_methods.txt`: original five expected methods.
- `expected_seeds.txt`: expected seed list.
- `sha256sums.txt`: committed evidence artifact hashes.
- `thresholds_sha256.txt`: conformal threshold sha256.
- `raw_critic_checkpoint_sha256.txt`: raw critic checkpoint sha256.
- `audit/online_episode_results.csv.sha256`: merged episode CSV sha256.
- `audit/gate_decisions.csv.sha256`: merged gate-decision CSV sha256.

## Archived Artifacts

- `plan_a_strict_completion_audit_20260602.json`: strict Plan A phase audit.
- `audit/online_acceptance_audit.json`: full Phase5 online acceptance audit
  with anti-gaming and strict-claim diagnostics.
- `audit/online_go_no_go.json`: compact GO/NO-GO summary.
- `audit/online_report.md`: human-readable online report.
- `audit/method_summary.json` and `audit/method_summary.csv`: method metrics,
  including safe-completion and task-progress fields.
- `audit/online_episode_results.csv`: merged online episode table.
- `audit/gate_decisions.csv`: merged per-skill gate decisions.
- `scripts/plan_a_strict_completion_audit_20260602.py`: strict phase audit
  script.
- `scripts/phase5_online_main_watchdog_20260602.py`: watchdog used for final
  online run stability.
- `scripts/phase5_online_sidecar_supervisor_20260602.py`: sidecar supervisor
  used for tail completion.

Source output root:

`/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602`
