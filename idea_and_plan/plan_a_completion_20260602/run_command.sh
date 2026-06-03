#!/usr/bin/env bash
set -euo pipefail

# Re-run the default online acceptance audit on committed merged artifacts.
python gear_sonic/scripts/casa_audit_phase5_online.py \
  --online-dir idea_and_plan/plan_a_completion_20260602/audit \
  --output-dir /tmp/casa_plan_a_completion_default_audit \
  --expected-episodes 2500 \
  --expected-methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill \
  --expected-seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100 \
  --strict

# Re-run the reviewer-facing strict Plan A claim audit. This is expected to return
# STRICT_PLAN_A_NO_GO for the current evidence because global conformal and fallback
# budget checks do not support the strongest original paper claim.
python gear_sonic/scripts/casa_audit_phase5_online.py \
  --online-dir idea_and_plan/plan_a_completion_20260602/audit \
  --output-dir /tmp/casa_plan_a_completion_strict_claim_audit \
  --expected-episodes 2500 \
  --expected-methods sonic_only,hard_contract,raw_critic_0p5,global_conformal,casa_a_per_skill \
  --expected-seeds 1234,1235,1236,1237,1238 \
  --episodes-per-seed 100 \
  --strict-plan-a-claim || true
