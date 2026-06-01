# CASA Phase5 Online Main Experiment Report

- status: `PASS_STRICT_ONLINE`
- go: `True`
- episodes: `2500` / expected `2500`
- completed: `2500`
- gate_decisions: `20000`

## Blockers

- none

## Warnings

- sonic_task_success_rate_low_relative_drop_check_is_unstable
- hard_contract_relative_reduction_undefined

## Method Summary

| method | episodes | unsafe | unsafe/episode | fallback/episode | task_success_rate | mean_time_s |
|---|---:|---:|---:|---:|---:|---:|
| SONIC-only | 1250 | 8966 | 7.1728 | 0.0000 | 0.0216 | 28.4461 |
| SONIC + CASA-A hard-OR adaptive recovery | 1250 | 3738 | 2.9904 | 3.8808 | 0.3304 | 22.1936 |

## Checks

- online_artifacts_schema_valid: `PASS`
- expected_episode_count_met: `PASS`
- expected_method_seed_episode_grid_complete: `PASS`
- all_expected_methods_present: `PASS`
- all_episodes_completed: `PASS`
- all_initial_upright_ok: `PASS`
- gate_decisions_present: `PASS`
- each_completed_episode_has_expected_decisions: `PASS`
- per_skill_decision_coverage_present: `PASS`
- casa_vs_sonic_unsafe_reduction_ge_40pct: `PASS`
- casa_vs_hard_unsafe_reduction_ge_20pct_or_undefined: `PASS`
- casa_task_success_drop_abs_le_10pp: `PASS`
- casa_task_success_drop_rel_le_30pct: `PASS`

## Baseline Comparisons

```json
{
  "casa_vs_sonic": {
    "baseline_fallback_rate_per_episode": 0.0,
    "baseline_method": "sonic_only",
    "baseline_task_success_rate": 0.0216,
    "baseline_unsafe_invocation_count": 8966,
    "method": "casa_a_hard_or_recovery",
    "method_fallback_rate_per_episode": 3.8808,
    "method_task_success_rate": 0.3304,
    "method_unsafe_invocation_count": 3738,
    "task_success_drop_abs": -0.3088,
    "task_success_drop_rel": -14.296296296296296,
    "unsafe_reduction": 0.5830916796787865
  }
}
```

## Artifact Validation

```json
{
  "completed_episode_count": 2500,
  "decision_count": 20000,
  "decision_rows_valid": true,
  "episode_count": 2500,
  "episode_rows_valid": true,
  "expected_grid_available": true,
  "expected_grid_count": 2500,
  "extra_expected_episode_count": 0,
  "extra_expected_episode_examples": [],
  "missing_decision_episode_count": 0,
  "missing_decision_episode_examples": [],
  "missing_expected_episode_count": 0,
  "missing_expected_episode_examples": [],
  "missing_per_skill_coverage": [],
  "per_skill_decision_counts": {
    "casa_a_hard_or_recovery": {
      "gesture": 1250,
      "passive": 2500,
      "turn": 2500,
      "walk": 3750
    },
    "sonic_only": {
      "gesture": 1250,
      "passive": 2500,
      "turn": 2500,
      "walk": 3750
    }
  }
}
```

## Per-skill Diagnostics

```json
{
  "casa_a_hard_or_recovery": {
    "gesture": {
      "decision_count": 1250,
      "fallback_count": 128,
      "fallback_rate": 0.1024,
      "hard_contract_fixed_reject_count": 42,
      "raw_critic_risk_max": 0.9999990463256836,
      "raw_critic_risk_mean": 0.11002804547561673,
      "raw_critic_risk_min": 1.8883166603700374e-06,
      "reject_count": 128,
      "reject_rate": 0.1024,
      "result_status_counts": {
        "failed": 1,
        "success": 1249
      }
    },
    "passive": {
      "decision_count": 2500,
      "fallback_count": 733,
      "fallback_rate": 0.2932,
      "hard_contract_fixed_reject_count": 729,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.3154338489675794,
      "raw_critic_risk_min": 2.1223804651526734e-06,
      "reject_count": 733,
      "reject_rate": 0.2932,
      "result_status_counts": {
        "failed": 257,
        "success": 2231,
        "unverified": 12
      }
    },
    "turn": {
      "decision_count": 2500,
      "fallback_count": 815,
      "fallback_rate": 0.326,
      "hard_contract_fixed_reject_count": 623,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.3514042270750977,
      "raw_critic_risk_min": 3.82250482289237e-06,
      "reject_count": 815,
      "reject_rate": 0.326,
      "result_status_counts": {
        "failed": 759,
        "success": 1730,
        "unverified": 11
      }
    },
    "walk": {
      "decision_count": 3750,
      "fallback_count": 3175,
      "fallback_rate": 0.8466666666666667,
      "hard_contract_fixed_reject_count": 1039,
      "raw_critic_risk_max": 0.9999997615814209,
      "raw_critic_risk_mean": 0.8640145462347214,
      "raw_critic_risk_min": 2.140933065675199e-05,
      "reject_count": 3175,
      "reject_rate": 0.8466666666666667,
      "result_status_counts": {
        "failed": 1559,
        "success": 2174,
        "unverified": 17
      }
    }
  },
  "sonic_only": {
    "gesture": {
      "decision_count": 1250,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 1038,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.9508099442139892,
      "raw_critic_risk_min": 1.922842557178228e-06,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 3,
        "success": 1247
      }
    },
    "passive": {
      "decision_count": 2500,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 2145,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.9522393820383576,
      "raw_critic_risk_min": 2.1414857656054664e-06,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 369,
        "success": 2129,
        "unverified": 2
      }
    },
    "turn": {
      "decision_count": 2500,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 2139,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.9460308941424198,
      "raw_critic_risk_min": 3.0507910651067505e-06,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 469,
        "success": 2027,
        "unverified": 4
      }
    },
    "walk": {
      "decision_count": 3750,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 2145,
      "raw_critic_risk_max": 0.9999997615814209,
      "raw_critic_risk_mean": 0.9916109228031018,
      "raw_critic_risk_min": 5.453859102999559e-06,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 41,
        "success": 3703,
        "unverified": 6
      }
    }
  }
}
```
