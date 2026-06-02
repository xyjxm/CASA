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
- hard_contract_intervenes_more_than_casa_diagnostic
- raw_lane_episode_duplicates_were_deduped

## Method Summary

| method | episodes | unsafe | unsafe/episode | fallback/episode | task_success_rate | mean_time_s |
|---|---:|---:|---:|---:|---:|---:|
| SONIC-only | 500 | 3559 | 7.1180 | 0.0000 | 0.0260 | 29.3428 |
| SONIC + Hard Contract | 500 | 3643 | 7.2860 | 6.2220 | 0.0220 | 21.3717 |
| SONIC + Raw Critic (0.5) | 500 | 1223 | 2.4460 | 3.8660 | 0.4580 | 23.3016 |
| SONIC + Global Conformal | 500 | 847 | 1.6940 | 4.3040 | 0.6560 | 21.8186 |
| SONIC + CASA-A (per-skill conformal) | 500 | 934 | 1.8680 | 5.2120 | 0.6100 | 22.0777 |

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
    "baseline_fallback_rate_per_episode": 4.304,
    "baseline_method": "global_conformal",
    "baseline_task_success_rate": 0.656,
    "baseline_unsafe_invocation_count": 847,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 5.212,
    "method_task_success_rate": 0.61,
    "method_unsafe_invocation_count": 934,
    "task_success_drop_abs": 0.04600000000000004,
    "task_success_drop_rel": 0.07012195121951226,
    "unsafe_reduction": -0.10271546635182999
  },
  "casa_vs_hard": {
    "baseline_fallback_rate_per_episode": 6.222,
    "baseline_method": "hard_contract",
    "baseline_task_success_rate": 0.022,
    "baseline_unsafe_invocation_count": 3643,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 5.212,
    "method_task_success_rate": 0.61,
    "method_unsafe_invocation_count": 934,
    "task_success_drop_abs": -0.588,
    "task_success_drop_rel": -26.727272727272727,
    "unsafe_reduction": 0.7436178973373593
  },
  "casa_vs_raw_critic": {
    "baseline_fallback_rate_per_episode": 3.866,
    "baseline_method": "raw_critic_0p5",
    "baseline_task_success_rate": 0.458,
    "baseline_unsafe_invocation_count": 1223,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 5.212,
    "method_task_success_rate": 0.61,
    "method_unsafe_invocation_count": 934,
    "task_success_drop_abs": -0.15199999999999997,
    "task_success_drop_rel": -0.33187772925764186,
    "unsafe_reduction": 0.23630417007358953
  },
  "casa_vs_sonic": {
    "baseline_fallback_rate_per_episode": 0.0,
    "baseline_method": "sonic_only",
    "baseline_task_success_rate": 0.026,
    "baseline_unsafe_invocation_count": 3559,
    "method": "casa_a_per_skill",
    "method_fallback_rate_per_episode": 5.212,
    "method_task_success_rate": 0.61,
    "method_unsafe_invocation_count": 934,
    "task_success_drop_abs": -0.584,
    "task_success_drop_rel": -22.46153846153846,
    "unsafe_reduction": 0.7375667322281539
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
    "casa_a_per_skill": {
      "gesture": 500,
      "passive": 1000,
      "turn": 1000,
      "walk": 1500
    },
    "global_conformal": {
      "gesture": 500,
      "passive": 1000,
      "turn": 1000,
      "walk": 1500
    },
    "hard_contract": {
      "gesture": 500,
      "passive": 1000,
      "turn": 1000,
      "walk": 1500
    },
    "raw_critic_0p5": {
      "gesture": 500,
      "passive": 1000,
      "turn": 1000,
      "walk": 1500
    },
    "sonic_only": {
      "gesture": 500,
      "passive": 1000,
      "turn": 1000,
      "walk": 1500
    }
  }
}
```

## Per-skill Diagnostics

```json
{
  "casa_a_per_skill": {
    "gesture": {
      "decision_count": 500,
      "fallback_count": 246,
      "fallback_rate": 0.492,
      "hard_contract_fixed_reject_count": 54,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.1660009059070478,
      "raw_critic_risk_min": 1.7518372033009655e-06,
      "reject_count": 246,
      "reject_rate": 0.492,
      "result_status_counts": {
        "failed": 63,
        "success": 437
      }
    },
    "passive": {
      "decision_count": 1000,
      "fallback_count": 477,
      "fallback_rate": 0.477,
      "hard_contract_fixed_reject_count": 182,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.2033221522790766,
      "raw_critic_risk_min": 2.4474541078234324e-06,
      "reject_count": 477,
      "reject_rate": 0.477,
      "result_status_counts": {
        "failed": 71,
        "success": 927,
        "unverified": 2
      }
    },
    "turn": {
      "decision_count": 1000,
      "fallback_count": 496,
      "fallback_rate": 0.496,
      "hard_contract_fixed_reject_count": 163,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.22885844932301688,
      "raw_critic_risk_min": 6.204558758327039e-06,
      "reject_count": 496,
      "reject_rate": 0.496,
      "result_status_counts": {
        "failed": 138,
        "success": 860,
        "unverified": 2
      }
    },
    "walk": {
      "decision_count": 1500,
      "fallback_count": 1387,
      "fallback_rate": 0.9246666666666666,
      "hard_contract_fixed_reject_count": 278,
      "raw_critic_risk_max": 0.9999997615814209,
      "raw_critic_risk_mean": 0.8651005820071065,
      "raw_critic_risk_min": 2.646804387040902e-05,
      "reject_count": 1387,
      "reject_rate": 0.9246666666666666,
      "result_status_counts": {
        "failed": 140,
        "success": 1357,
        "unverified": 3
      }
    }
  },
  "global_conformal": {
    "gesture": {
      "decision_count": 500,
      "fallback_count": 134,
      "fallback_rate": 0.268,
      "hard_contract_fixed_reject_count": 47,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.1439606078084762,
      "raw_critic_risk_min": 1.4397999166249065e-06,
      "reject_count": 134,
      "reject_rate": 0.268,
      "result_status_counts": {
        "failed": 60,
        "success": 440
      }
    },
    "passive": {
      "decision_count": 1000,
      "fallback_count": 246,
      "fallback_rate": 0.246,
      "hard_contract_fixed_reject_count": 155,
      "raw_critic_risk_max": 0.9999992847442627,
      "raw_critic_risk_mean": 0.16062384681640787,
      "raw_critic_risk_min": 2.062404973912635e-06,
      "reject_count": 246,
      "reject_rate": 0.246,
      "result_status_counts": {
        "failed": 75,
        "success": 925
      }
    },
    "turn": {
      "decision_count": 1000,
      "fallback_count": 333,
      "fallback_rate": 0.333,
      "hard_contract_fixed_reject_count": 137,
      "raw_critic_risk_max": 0.9999992847442627,
      "raw_critic_risk_mean": 0.18060432930316755,
      "raw_critic_risk_min": 6.982923423493048e-06,
      "reject_count": 333,
      "reject_rate": 0.333,
      "result_status_counts": {
        "failed": 154,
        "success": 846
      }
    },
    "walk": {
      "decision_count": 1500,
      "fallback_count": 1439,
      "fallback_rate": 0.9593333333333334,
      "hard_contract_fixed_reject_count": 255,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.8597126277272522,
      "raw_critic_risk_min": 1.0066296454169787e-05,
      "reject_count": 1439,
      "reject_rate": 0.9593333333333334,
      "result_status_counts": {
        "failed": 129,
        "success": 1371
      }
    }
  },
  "hard_contract": {
    "gesture": {
      "decision_count": 500,
      "fallback_count": 429,
      "fallback_rate": 0.858,
      "hard_contract_fixed_reject_count": 429,
      "raw_critic_risk_max": 0.9999997615814209,
      "raw_critic_risk_mean": 0.9605369160590603,
      "raw_critic_risk_min": 1.1605100098677212e-06,
      "reject_count": 429,
      "reject_rate": 0.858,
      "result_status_counts": {
        "failed": 83,
        "success": 417
      }
    },
    "passive": {
      "decision_count": 1000,
      "fallback_count": 896,
      "fallback_rate": 0.896,
      "hard_contract_fixed_reject_count": 896,
      "raw_critic_risk_max": 0.9999997615814209,
      "raw_critic_risk_mean": 0.964083165043952,
      "raw_critic_risk_min": 9.177161928164423e-07,
      "reject_count": 896,
      "reject_rate": 0.896,
      "result_status_counts": {
        "failed": 110,
        "success": 888,
        "unverified": 2
      }
    },
    "turn": {
      "decision_count": 1000,
      "fallback_count": 884,
      "fallback_rate": 0.884,
      "hard_contract_fixed_reject_count": 884,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.9508276136152358,
      "raw_critic_risk_min": 4.330005594965769e-06,
      "reject_count": 884,
      "reject_rate": 0.884,
      "result_status_counts": {
        "failed": 256,
        "success": 742,
        "unverified": 2
      }
    },
    "walk": {
      "decision_count": 1500,
      "fallback_count": 902,
      "fallback_rate": 0.6013333333333334,
      "hard_contract_fixed_reject_count": 902,
      "raw_critic_risk_max": 0.9999997615814209,
      "raw_critic_risk_mean": 0.9953831930428423,
      "raw_critic_risk_min": 2.2940475901123136e-05,
      "reject_count": 902,
      "reject_rate": 0.6013333333333334,
      "result_status_counts": {
        "failed": 102,
        "success": 1393,
        "unverified": 5
      }
    }
  },
  "raw_critic_0p5": {
    "gesture": {
      "decision_count": 500,
      "fallback_count": 87,
      "fallback_rate": 0.174,
      "hard_contract_fixed_reject_count": 64,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.16995432748398526,
      "raw_critic_risk_min": 1.406717956342618e-06,
      "reject_count": 87,
      "reject_rate": 0.174,
      "result_status_counts": {
        "failed": 58,
        "success": 441,
        "unverified": 1
      }
    },
    "passive": {
      "decision_count": 1000,
      "fallback_count": 266,
      "fallback_rate": 0.266,
      "hard_contract_fixed_reject_count": 248,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.27110108994671056,
      "raw_critic_risk_min": 1.8349414858676028e-06,
      "reject_count": 266,
      "reject_rate": 0.266,
      "result_status_counts": {
        "failed": 85,
        "success": 912,
        "unverified": 3
      }
    },
    "turn": {
      "decision_count": 1000,
      "fallback_count": 286,
      "fallback_rate": 0.286,
      "hard_contract_fixed_reject_count": 219,
      "raw_critic_risk_max": 0.9999994039535522,
      "raw_critic_risk_mean": 0.2890124586895886,
      "raw_critic_risk_min": 2.71770886683953e-06,
      "reject_count": 286,
      "reject_rate": 0.286,
      "result_status_counts": {
        "failed": 186,
        "success": 810,
        "unverified": 4
      }
    },
    "walk": {
      "decision_count": 1500,
      "fallback_count": 1294,
      "fallback_rate": 0.8626666666666667,
      "hard_contract_fixed_reject_count": 354,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.8599809224341237,
      "raw_critic_risk_min": 2.7955034965998493e-05,
      "reject_count": 1294,
      "reject_rate": 0.8626666666666667,
      "result_status_counts": {
        "failed": 128,
        "success": 1367,
        "unverified": 5
      }
    }
  },
  "sonic_only": {
    "gesture": {
      "decision_count": 500,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 419,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.9407205592835118,
      "raw_critic_risk_min": 1.3018703839406953e-06,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 1,
        "success": 499
      }
    },
    "passive": {
      "decision_count": 1000,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 850,
      "raw_critic_risk_max": 0.9999996423721313,
      "raw_critic_risk_mean": 0.9473128460447617,
      "raw_critic_risk_min": 1.9649228306661826e-06,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 138,
        "success": 859,
        "unverified": 3
      }
    },
    "turn": {
      "decision_count": 1000,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 852,
      "raw_critic_risk_max": 0.9999995231628418,
      "raw_critic_risk_mean": 0.937079049848064,
      "raw_critic_risk_min": 2.9524646834033774e-06,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 204,
        "success": 796
      }
    },
    "walk": {
      "decision_count": 1500,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 854,
      "raw_critic_risk_max": 0.9999997615814209,
      "raw_critic_risk_mean": 0.9882988533612048,
      "raw_critic_risk_min": 1.2832871107093524e-05,
      "reject_count": 0,
      "reject_rate": 0.0,
      "result_status_counts": {
        "failed": 23,
        "success": 1473,
        "unverified": 4
      }
    }
  }
}
```
