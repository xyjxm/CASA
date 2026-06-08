# CASA Phase5 Online Main Experiment Report

- status: `PASS_STRICT_ONLINE`
- go: `True`
- episodes: `200` / expected `200`
- completed: `200`
- gate_decisions: `1600`

## Blockers

- none

## Warnings

- hard_contract_intervenes_more_than_casa_diagnostic

## Method Summary

| method | episodes | unsafe | unsafe/episode | fallback/episode | safe_completion_rate | task_progress_success_rate | mean_time_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| SONIC-only | 20 | 140 | 7.0000 | 0.0000 | 0.0500 | 1.0000 | 26.8689 |
| SONIC + Hard Contract | 20 | 143 | 7.1500 | 5.9000 | 0.0000 | 0.2625 | 18.6770 |
| SONIC + Raw Critic (0.5) | 20 | 58 | 2.9000 | 4.0000 | 0.4000 | 0.5000 | 20.4404 |
| SONIC + Global Conformal | 20 | 88 | 4.4000 | 3.8000 | 0.2000 | 0.5250 | 21.0250 |
| SONIC + CASA-A (per-skill conformal) | 20 | 73 | 3.6500 | 3.9000 | 0.2000 | 0.5125 | 20.6111 |
| SafeDPA-adapted | 20 | 61 | 3.0500 | 3.3000 | 0.2500 | 0.5875 | 21.5907 |
| PCBF-adapted | 20 | 87 | 4.3500 | 3.9500 | 0.1500 | 0.5062 | 20.6424 |
| SAFER-Splat-CBF-adapted | 20 | 132 | 6.6000 | 5.3500 | 0.0500 | 0.3312 | 19.0422 |
| MPC-CBF-Humanoid-adapted | 20 | 33 | 1.6500 | 3.6000 | 0.7000 | 0.5500 | 20.3061 |
| CLBF-LBAC-adapted | 20 | 122 | 6.1000 | 5.3000 | 0.1000 | 0.3375 | 19.3055 |

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
  "casa_vs_global": {
    "baseline_fallback_rate_per_episode": 3.8,
    "baseline_method": "global_conformal",
    "baseline_task_success_rate": 0.2,
    "baseline_unsafe_invocation_count": 88,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 3.9,
    "method_task_success_rate": 0.2,
    "method_unsafe_invocation_count": 73,
    "task_success_drop_abs": 0.0,
    "task_success_drop_rel": 0.0,
    "unsafe_reduction": 0.17045454545454544
  },
  "casa_vs_hard": {
    "baseline_fallback_rate_per_episode": 5.9,
    "baseline_method": "hard_contract",
    "baseline_task_success_rate": 0.0,
    "baseline_unsafe_invocation_count": 143,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 3.9,
    "method_task_success_rate": 0.2,
    "method_unsafe_invocation_count": 73,
    "task_success_drop_abs": -0.2,
    "task_success_drop_rel": 0.0,
    "unsafe_reduction": 0.48951048951048953
  },
  "casa_vs_raw_critic": {
    "baseline_fallback_rate_per_episode": 4.0,
    "baseline_method": "raw_critic_0p5",
    "baseline_task_success_rate": 0.4,
    "baseline_unsafe_invocation_count": 58,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 3.9,
    "method_task_success_rate": 0.2,
    "method_unsafe_invocation_count": 73,
    "task_success_drop_abs": 0.2,
    "task_success_drop_rel": 0.5,
    "unsafe_reduction": -0.25862068965517243
  },
  "casa_vs_sonic": {
    "baseline_fallback_rate_per_episode": 0.0,
    "baseline_method": "sonic_only",
    "baseline_task_success_rate": 0.05,
    "baseline_unsafe_invocation_count": 140,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 3.9,
    "method_task_success_rate": 0.2,
    "method_unsafe_invocation_count": 73,
    "task_success_drop_abs": -0.15000000000000002,
    "task_success_drop_rel": -3.0000000000000004,
    "unsafe_reduction": 0.4785714285714286
  }
}
```

## Anti-gaming / Claim-validity Checks

- strict_plan_a_claim_status: `STRICT_PLAN_A_NO_GO`
- strict_plan_a_claim_go: `False`
- method_kind: `original_plan_a`
- global_conformal_advantage_check: `PASS`
- fallback_rate_per_episode: `3.9000`
- reject_rate_per_decision: `0.4875`
- walk_reject_rate: `0.7167`
- matched_budget_winner: `casa_a_per_skill`
- strict_plan_a_claim_blockers:
  - `casa_vs_raw_critic_unsafe_reduction_below_threshold`
  - `casa_fallback_rate_exceeds_strict_plan_a_budget`

```json
{
  "blocking_reasons": [
    "casa_vs_raw_critic_unsafe_reduction_below_threshold",
    "casa_fallback_rate_exceeds_strict_plan_a_budget"
  ],
  "casa_method": "casa_a_per_skill",
  "checks": {
    "all_five_original_methods_present": true,
    "casa_vs_global_advantage": true,
    "casa_vs_hard_unsafe_reduction": true,
    "casa_vs_raw_critic_unsafe_reduction": false,
    "fallback_rate_budget": false,
    "matched_budget_comparison_present": true,
    "original_casa_method": true,
    "reject_rate_budget": true,
    "task_success_split_fields_present": true,
    "walk_reject_rate_budget": true
  },
  "expected_methods": [
    "sonic_only",
    "hard_contract",
    "raw_critic_0p5",
    "global_conformal",
    "casa_a_per_skill",
    "safedpa_adapted",
    "pcbf_adapted",
    "safer_splat_cbf_adapted",
    "mpc_cbf_humanoid_adapted",
    "clbf_lbac_adapted"
  ],
  "fallback_reject_budget": {
    "checks": {
      "fallback_rate_per_episode_within_budget": false,
      "reject_rate_within_budget": true,
      "walk_reject_rate_within_budget": true
    },
    "decision_count": 160,
    "fallback_count": 78,
    "fallback_rate_per_episode": 3.9,
    "intervention_rate": 0.4875,
    "labeled_reject_count": 78,
    "method": "casa_a_per_skill",
    "per_skill_fallback_rate": {
      "gesture": 0.0,
      "passive": 0.375,
      "turn": 0.5,
      "walk": 0.7166666666666667
    },
    "per_skill_reject_rate": {
      "gesture": 0.0,
      "passive": 0.375,
      "turn": 0.5,
      "walk": 0.7166666666666667
    },
    "recovery_only_rate": 1.0,
    "reject_count": 78,
    "reject_rate_per_decision": 0.4875,
    "safe_rejection_count": 35,
    "safe_rejection_rate": 0.44871794871794873,
    "thresholds": {
      "max_fallback_rate_per_episode": 2.0,
      "max_reject_rate": 0.5,
      "max_walk_reject_rate": 0.75
    },
    "walk_reject_rate": 0.7166666666666667
  },
  "global_advantage_evidence": {
    "casa_task_progress_success_rate": 0.5125,
    "global_task_progress_success_rate": 0.525,
    "task_progress_advantage": -0.012500000000000067,
    "unsafe_reduction": 0.17045454545454544
  },
  "go": false,
  "matched_budget_comparison": {
    "baseline_budget_value": 3.8,
    "baseline_method": "global_conformal",
    "baseline_task_progress_success": 0.525,
    "baseline_unsafe": 88,
    "budget_gap": 0.10000000000000009,
    "budget_type": "fallback_rate_per_episode",
    "budget_value": 3.9,
    "casa_task_progress_success": 0.5125,
    "casa_unsafe": 73,
    "global_task_progress_success": 0.525,
    "global_unsafe": 88,
    "matched": true,
    "method": "casa_a_per_skill",
    "note": "Observed-budget diagnostic only; run a threshold sweep to construct a true Pareto matched-intervention frontier.",
    "relative_tolerance": 0.1,
    "winner": "casa_a_per_skill"
  },
  "method_kind": "original_plan_a",
  "original_plan_a_methods": [
    "sonic_only",
    "hard_contract",
    "raw_critic_0p5",
    "global_conformal",
    "casa_a_per_skill"
  ],
  "present_methods": [
    "casa_a_per_skill",
    "clbf_lbac_adapted",
    "global_conformal",
    "hard_contract",
    "mpc_cbf_humanoid_adapted",
    "pcbf_adapted",
    "raw_critic_0p5",
    "safedpa_adapted",
    "safer_splat_cbf_adapted",
    "sonic_only"
  ],
  "status": "STRICT_PLAN_A_NO_GO",
  "thresholds": {
    "min_global_task_progress_advantage": 0.1,
    "min_global_unsafe_reduction": 0.1,
    "min_hard_unsafe_reduction": 0.2,
    "min_raw_unsafe_reduction": 0.1
  },
  "variant_methods_present": []
}
```

## Artifact Validation

```json
{
  "completed_episode_count": 200,
  "decision_count": 1600,
  "decision_rows_valid": true,
  "episode_count": 200,
  "episode_rows_valid": true,
  "expected_grid_available": true,
  "expected_grid_count": 200,
  "extra_expected_episode_count": 0,
  "extra_expected_episode_examples": [],
  "missing_decision_episode_count": 0,
  "missing_decision_episode_examples": [],
  "missing_expected_episode_count": 0,
  "missing_expected_episode_examples": [],
  "missing_per_skill_coverage": [],
  "per_skill_decision_counts": {
    "casa_a_per_skill": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "clbf_lbac_adapted": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "global_conformal": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "hard_contract": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "mpc_cbf_humanoid_adapted": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "pcbf_adapted": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "raw_critic_0p5": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "safedpa_adapted": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "safer_splat_cbf_adapted": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    },
    "sonic_only": {
      "gesture": 20,
      "passive": 40,
      "turn": 40,
      "walk": 60
    }
  }
}
```

## Per-skill Diagnostics

```json
{
  "casa_a_per_skill": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 4,
      "raw_critic_risk_max": 0.9999988079071045,
      "raw_critic_risk_mean": 0.4488833010150756,
      "raw_critic_risk_min": 1.0232512977381703e-05,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "success": 20
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 15,
      "fallback_rate": 0.375,
      "hard_contract_fixed_reject_count": 14,
      "raw_critic_risk_max": 0.9999978542327881,
      "raw_critic_risk_mean": 0.581449479787284,
      "raw_critic_risk_min": 1.3979800314700697e-05,
      "reject_count": 15,
      "reject_rate": 0.375,
      "result_status_counts": {
        "failed": 3,
        "success": 37
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 20,
      "fallback_rate": 0.5,
      "hard_contract_fixed_reject_count": 12,
      "raw_critic_risk_max": 0.9999980926513672,
      "raw_critic_risk_mean": 0.5765601483611136,
      "raw_critic_risk_min": 3.1024599593365565e-05,
      "reject_count": 20,
      "reject_rate": 0.5,
      "result_status_counts": {
        "failed": 11,
        "success": 29
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 43,
      "fallback_rate": 0.7166666666666667,
      "hard_contract_fixed_reject_count": 18,
      "raw_critic_risk_max": 0.9999986886978149,
      "raw_critic_risk_mean": 0.8768302738375496,
      "raw_critic_risk_min": 0.001793990028090775,
      "reject_count": 43,
      "reject_rate": 0.7166666666666667,
      "result_status_counts": {
        "failed": 1,
        "success": 59
      }
    }
  },
  "clbf_lbac_adapted": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 14,
      "fallback_rate": 0.7,
      "hard_contract_fixed_reject_count": 14,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.8998415867715266,
      "raw_critic_risk_min": 2.7223517463426106e-05,
      "reject_count": 14,
      "reject_rate": 0.7,
      "result_status_counts": {
        "failed": 4,
        "success": 16
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 27,
      "fallback_rate": 0.675,
      "hard_contract_fixed_reject_count": 27,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.8554878913524817,
      "raw_critic_risk_min": 9.770848009793554e-06,
      "reject_count": 27,
      "reject_rate": 0.675,
      "result_status_counts": {
        "failed": 5,
        "success": 35
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 27,
      "fallback_rate": 0.675,
      "hard_contract_fixed_reject_count": 27,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.8924781466172135,
      "raw_critic_risk_min": 0.00012870068894699216,
      "reject_count": 27,
      "reject_rate": 0.675,
      "result_status_counts": {
        "failed": 14,
        "success": 26
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 38,
      "fallback_rate": 0.6333333333333333,
      "hard_contract_fixed_reject_count": 26,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.993132151166598,
      "raw_critic_risk_min": 0.7653676271438599,
      "reject_count": 38,
      "reject_rate": 0.6333333333333333,
      "result_status_counts": {
        "failed": 4,
        "success": 56
      }
    }
  },
  "global_conformal": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 8,
      "fallback_rate": 0.4,
      "hard_contract_fixed_reject_count": 9,
      "raw_critic_risk_max": 0.9999992847442627,
      "raw_critic_risk_mean": 0.5997225446624725,
      "raw_critic_risk_min": 1.0454887160449289e-05,
      "reject_count": 8,
      "reject_rate": 0.4,
      "result_status_counts": {
        "failed": 3,
        "success": 17
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 15,
      "fallback_rate": 0.375,
      "hard_contract_fixed_reject_count": 19,
      "raw_critic_risk_max": 0.999998927116394,
      "raw_critic_risk_mean": 0.684485278691227,
      "raw_critic_risk_min": 1.3635843060910702e-05,
      "reject_count": 15,
      "reject_rate": 0.375,
      "result_status_counts": {
        "failed": 3,
        "success": 37
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 16,
      "fallback_rate": 0.4,
      "hard_contract_fixed_reject_count": 16,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.6965306724943275,
      "raw_critic_risk_min": 2.9590741178253666e-05,
      "reject_count": 16,
      "reject_rate": 0.4,
      "result_status_counts": {
        "failed": 8,
        "success": 32
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 37,
      "fallback_rate": 0.6166666666666667,
      "hard_contract_fixed_reject_count": 21,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.9188651159619137,
      "raw_critic_risk_min": 0.001466707675717771,
      "reject_count": 37,
      "reject_rate": 0.6166666666666667,
      "result_status_counts": {
        "failed": 2,
        "success": 58
      }
    }
  },
  "hard_contract": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 17,
      "fallback_rate": 0.85,
      "hard_contract_fixed_reject_count": 17,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.9947885543107986,
      "raw_critic_risk_min": 0.8970034718513489,
      "reject_count": 17,
      "reject_rate": 0.85,
      "result_status_counts": {
        "failed": 3,
        "success": 17
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 34,
      "fallback_rate": 0.85,
      "hard_contract_fixed_reject_count": 34,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.9703081382009714,
      "raw_critic_risk_min": 1.5428437109221704e-05,
      "reject_count": 34,
      "reject_rate": 0.85,
      "result_status_counts": {
        "failed": 3,
        "success": 37
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 33,
      "fallback_rate": 0.825,
      "hard_contract_fixed_reject_count": 33,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.9413614283316292,
      "raw_critic_risk_min": 2.8632583052967675e-05,
      "reject_count": 33,
      "reject_rate": 0.825,
      "result_status_counts": {
        "failed": 8,
        "success": 32
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 34,
      "fallback_rate": 0.5666666666666667,
      "hard_contract_fixed_reject_count": 34,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.9981302271286646,
      "raw_critic_risk_min": 0.9103139638900757,
      "reject_count": 34,
      "reject_rate": 0.5666666666666667,
      "result_status_counts": {
        "failed": 1,
        "success": 59
      }
    }
  },
  "mpc_cbf_humanoid_adapted": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 6,
      "fallback_rate": 0.3,
      "hard_contract_fixed_reject_count": 4,
      "raw_critic_risk_max": 0.9999988079071045,
      "raw_critic_risk_mean": 0.1989286350558359,
      "raw_critic_risk_min": 5.133665126777487e-06,
      "reject_count": 6,
      "reject_rate": 0.3,
      "result_status_counts": {
        "failed": 4,
        "success": 16
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 6,
      "fallback_rate": 0.15,
      "hard_contract_fixed_reject_count": 6,
      "raw_critic_risk_max": 0.9999954700469971,
      "raw_critic_risk_mean": 0.17669204512637862,
      "raw_critic_risk_min": 8.135943971865345e-06,
      "reject_count": 6,
      "reject_rate": 0.15,
      "result_status_counts": {
        "failed": 2,
        "success": 38
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 6,
      "fallback_rate": 0.15,
      "hard_contract_fixed_reject_count": 6,
      "raw_critic_risk_max": 0.9999961853027344,
      "raw_critic_risk_mean": 0.23220750378593494,
      "raw_critic_risk_min": 1.27210269056377e-05,
      "reject_count": 6,
      "reject_rate": 0.15,
      "result_status_counts": {
        "failed": 5,
        "success": 35
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 54,
      "fallback_rate": 0.9,
      "hard_contract_fixed_reject_count": 10,
      "raw_critic_risk_max": 0.9999986886978149,
      "raw_critic_risk_mean": 0.8062190785659671,
      "raw_critic_risk_min": 0.0002980977005790919,
      "reject_count": 54,
      "reject_rate": 0.9,
      "result_status_counts": {
        "failed": 3,
        "success": 57
      }
    }
  },
  "pcbf_adapted": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 6,
      "fallback_rate": 0.3,
      "hard_contract_fixed_reject_count": 6,
      "raw_critic_risk_max": 0.9999988079071045,
      "raw_critic_risk_mean": 0.4498939635485385,
      "raw_critic_risk_min": 1.0343191206629854e-05,
      "reject_count": 6,
      "reject_rate": 0.3,
      "result_status_counts": {
        "failed": 3,
        "success": 17
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 17,
      "fallback_rate": 0.425,
      "hard_contract_fixed_reject_count": 17,
      "raw_critic_risk_max": 0.9999957084655762,
      "raw_critic_risk_mean": 0.6037061239361492,
      "raw_critic_risk_min": 1.3555495570471976e-05,
      "reject_count": 17,
      "reject_rate": 0.425,
      "result_status_counts": {
        "failed": 4,
        "success": 36
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 13,
      "fallback_rate": 0.325,
      "hard_contract_fixed_reject_count": 13,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.591064919515793,
      "raw_critic_risk_min": 2.3283912014449015e-05,
      "reject_count": 13,
      "reject_rate": 0.325,
      "result_status_counts": {
        "failed": 6,
        "success": 34
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 43,
      "fallback_rate": 0.7166666666666667,
      "hard_contract_fixed_reject_count": 20,
      "raw_critic_risk_max": 0.9999986886978149,
      "raw_critic_risk_mean": 0.8642283590006021,
      "raw_critic_risk_min": 0.0023343085777014494,
      "reject_count": 43,
      "reject_rate": 0.7166666666666667,
      "result_status_counts": {
        "failed": 1,
        "success": 59
      }
    }
  },
  "raw_critic_0p5": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 4,
      "fallback_rate": 0.2,
      "hard_contract_fixed_reject_count": 4,
      "raw_critic_risk_max": 0.9999974966049194,
      "raw_critic_risk_mean": 0.2000720116272646,
      "raw_critic_risk_min": 3.5489481433614856e-06,
      "reject_count": 4,
      "reject_rate": 0.2,
      "result_status_counts": {
        "failed": 4,
        "success": 16
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 13,
      "fallback_rate": 0.325,
      "hard_contract_fixed_reject_count": 12,
      "raw_critic_risk_max": 0.9999988079071045,
      "raw_critic_risk_mean": 0.34283542986826204,
      "raw_critic_risk_min": 5.331724878487876e-06,
      "reject_count": 13,
      "reject_rate": 0.325,
      "result_status_counts": {
        "failed": 2,
        "success": 38
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 15,
      "fallback_rate": 0.375,
      "hard_contract_fixed_reject_count": 10,
      "raw_critic_risk_max": 0.9999991655349731,
      "raw_critic_risk_mean": 0.38075494020249606,
      "raw_critic_risk_min": 1.3735032553086057e-05,
      "reject_count": 15,
      "reject_rate": 0.375,
      "result_status_counts": {
        "failed": 7,
        "success": 33
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 48,
      "fallback_rate": 0.8,
      "hard_contract_fixed_reject_count": 16,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.8057288659319359,
      "raw_critic_risk_min": 0.0002918190439231694,
      "reject_count": 48,
      "reject_rate": 0.8,
      "result_status_counts": {
        "failed": 3,
        "success": 57
      }
    }
  },
  "safedpa_adapted": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 4,
      "fallback_rate": 0.2,
      "hard_contract_fixed_reject_count": 4,
      "raw_critic_risk_max": 0.9999980926513672,
      "raw_critic_risk_mean": 0.24997655685895098,
      "raw_critic_risk_min": 7.037952400423819e-06,
      "reject_count": 4,
      "reject_rate": 0.2,
      "result_status_counts": {
        "failed": 3,
        "success": 17
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 11,
      "fallback_rate": 0.275,
      "hard_contract_fixed_reject_count": 11,
      "raw_critic_risk_max": 0.9999980926513672,
      "raw_critic_risk_mean": 0.355257576521808,
      "raw_critic_risk_min": 8.025652277865447e-06,
      "reject_count": 11,
      "reject_rate": 0.275,
      "result_status_counts": {
        "failed": 4,
        "success": 36
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 11,
      "fallback_rate": 0.275,
      "hard_contract_fixed_reject_count": 11,
      "raw_critic_risk_max": 0.9999977350234985,
      "raw_critic_risk_mean": 0.42026800174594425,
      "raw_critic_risk_min": 1.4569895938620903e-05,
      "reject_count": 11,
      "reject_rate": 0.275,
      "result_status_counts": {
        "failed": 9,
        "success": 31
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 40,
      "fallback_rate": 0.6666666666666666,
      "hard_contract_fixed_reject_count": 15,
      "raw_critic_risk_max": 0.9999990463256836,
      "raw_critic_risk_mean": 0.8277673084487712,
      "raw_critic_risk_min": 0.00020905284327454865,
      "reject_count": 40,
      "reject_rate": 0.6666666666666666,
      "result_status_counts": {
        "failed": 1,
        "success": 59
      }
    }
  },
  "safer_splat_cbf_adapted": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 13,
      "fallback_rate": 0.65,
      "hard_contract_fixed_reject_count": 13,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.8596695298705527,
      "raw_critic_risk_min": 3.386390380910598e-05,
      "reject_count": 13,
      "reject_rate": 0.65,
      "result_status_counts": {
        "failed": 2,
        "success": 18
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 29,
      "fallback_rate": 0.725,
      "hard_contract_fixed_reject_count": 29,
      "raw_critic_risk_max": 0.9999992847442627,
      "raw_critic_risk_mean": 0.8675044949568473,
      "raw_critic_risk_min": 0.00011132065992569551,
      "reject_count": 29,
      "reject_rate": 0.725,
      "result_status_counts": {
        "failed": 4,
        "success": 36
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 28,
      "fallback_rate": 0.7,
      "hard_contract_fixed_reject_count": 28,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.8723316714887914,
      "raw_critic_risk_min": 0.0002814648614730686,
      "reject_count": 28,
      "reject_rate": 0.7,
      "result_status_counts": {
        "failed": 8,
        "success": 32
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 37,
      "fallback_rate": 0.6166666666666667,
      "hard_contract_fixed_reject_count": 32,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.9981677393118541,
      "raw_critic_risk_min": 0.9128265976905823,
      "reject_count": 37,
      "reject_rate": 0.6166666666666667,
      "result_status_counts": {
        "failed": 1,
        "success": 59
      }
    }
  },
  "sonic_only": {
    "gesture": {
      "decision_count": 20,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 16,
      "raw_critic_risk_max": 0.9999991655349731,
      "raw_critic_risk_mean": 0.9998388946056366,
      "raw_critic_risk_min": 0.9980643391609192,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 1,
        "success": 19
      }
    },
    "passive": {
      "decision_count": 40,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 33,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.9982825443148613,
      "raw_critic_risk_min": 0.9577867984771729,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 6,
        "success": 34
      }
    },
    "turn": {
      "decision_count": 40,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 32,
      "raw_critic_risk_max": 0.9999992847442627,
      "raw_critic_risk_mean": 0.9792659804224968,
      "raw_critic_risk_min": 0.5868963003158569,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 5,
        "success": 35
      }
    },
    "walk": {
      "decision_count": 60,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 31,
      "raw_critic_risk_max": 0.999998927116394,
      "raw_critic_risk_mean": 0.9982728699843089,
      "raw_critic_risk_min": 0.915141224861145,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 3,
        "success": 57
      }
    }
  }
}
```
