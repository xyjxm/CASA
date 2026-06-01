# CASA Phase5 Online Failure Decomposition

- episodes: `2500`
- gate decisions: `20000`
- CASA method: `casa_a_hard_or_recovery`

## Summary

```json
{
  "casa_task_success_drop_rel": -14.296296296296296,
  "casa_task_success_rate": 0.3304,
  "casa_vs_sonic_unsafe_reduction": 0.5830916796787865,
  "sonic_task_success_rate": 0.0216
}
```

## Dominant Recommendations

- `gate_recall_or_receding_check`: allow_then_unsafe=9978 Next: Pilot hard-OR-CASA and receding segment checks before long skills.
- `fallback_quality`: reject_but_still_unsafe=2726 Next: Use adaptive recovery and inspect rejected unsafe clusters.
- `task_success_over_rejection`: fallback_count=4851 Next: Sweep threshold scales and adaptive_retry count against task success.
- `online_distribution_shift`: low_risk_unsafe_count=1028 Next: Consider an online adapter using only dev/tuning online data.
- `top_cluster`: 794 unsafe labels for sonic_only/walk in visual_collision_or_close/hard Next: Target this cluster in the next pilot policy sweep.

## Per-method Breakdown

| method | episodes | task_success | unsafe | reject | allow_then_unsafe | reject_but_still_unsafe |
|---|---:|---:|---:|---:|---:|---:|
| SONIC + CASA-A hard-OR adaptive recovery | 1250 | 0.3304 | 3738 | 4851 | 1012 | 2726 |
| SONIC-only | 1250 | 0.0216 | 8966 | 0 | 8966 | 0 |

## Per-skill Breakdown

```json
{
  "casa_a_hard_or_recovery": {
    "gesture": {
      "allow_then_unsafe": 267,
      "decision_count": 1250,
      "fallback_count": 128,
      "fallback_rate": 0.1024,
      "hard_contract_fixed_reject_count": 42,
      "low_risk_unsafe_count": 258,
      "method": "casa_a_hard_or_recovery",
      "raw_risk_safe": {
        "count": 869,
        "max": 0.9992088675498962,
        "mean": 0.019447623657112036,
        "median": 5.5641427024966106e-05,
        "min": 1.8883166603700374e-06,
        "p25": 1.8457054466125555e-05,
        "p75": 0.00025084096705541015,
        "p90": 0.0019141942728310827
      },
      "raw_risk_unsafe": {
        "count": 381,
        "max": 0.9999990463256836,
        "mean": 0.3166274852663794,
        "median": 0.013891302980482578,
        "min": 8.318744221469387e-06,
        "p25": 0.0004818191519007087,
        "p75": 0.8630152344703674,
        "p90": 0.9990992546081543
      },
      "reject_but_still_unsafe": 114,
      "reject_count": 128,
      "reject_rate": 0.1024,
      "risk_margin": {
        "count": 1250,
        "max": 0.2999995470046998,
        "mean": -0.589971453845353,
        "median": -0.6998707651990116,
        "min": -0.6999976110043235,
        "p25": -0.6999716326206907,
        "p75": -0.6973792343167587,
        "p90": -0.06266428232192964
      },
      "skill": "gesture",
      "threshold": {
        "count": 1250,
        "max": 0.6999994993209838,
        "mean": 0.6999994993209697,
        "median": 0.6999994993209838,
        "min": 0.6999994993209838,
        "p25": 0.6999994993209838,
        "p75": 0.6999994993209838,
        "p90": 0.6999994993209839
      },
      "unsafe_after_allow": 267,
      "unsafe_after_reject": 114
    },
    "passive": {
      "allow_then_unsafe": 298,
      "decision_count": 2500,
      "fallback_count": 733,
      "fallback_rate": 0.2932,
      "hard_contract_fixed_reject_count": 729,
      "low_risk_unsafe_count": 312,
      "method": "casa_a_hard_or_recovery",
      "raw_risk_safe": {
        "count": 1483,
        "max": 0.9999966621398926,
        "mean": 0.053143859179729004,
        "median": 5.979502748232335e-05,
        "min": 2.1223804651526734e-06,
        "p25": 2.1013428522564936e-05,
        "p75": 0.00029036060732323676,
        "p90": 0.023073685914278025
      },
      "raw_risk_unsafe": {
        "count": 1017,
        "max": 0.9999996423721313,
        "mean": 0.6979078458755266,
        "median": 0.999731719493866,
        "min": 1.7640519217820838e-05,
        "p25": 0.07383988797664642,
        "p75": 0.9999909400939941,
        "p90": 0.9999972581863403
      },
      "reject_but_still_unsafe": 719,
      "reject_count": 733,
      "reject_rate": 0.2932,
      "risk_margin": {
        "count": 2500,
        "max": 0.0001323223114013672,
        "mean": -0.6844334710931506,
        "median": -0.9994955767760985,
        "min": -0.9998651976802648,
        "p25": -0.9998260423008105,
        "p75": -0.005699530243873596,
        "p90": 0.00012409687042236328
      },
      "skill": "passive",
      "threshold": {
        "count": 2500,
        "max": 0.99986732006073,
        "mean": 0.99986732006073,
        "median": 0.99986732006073,
        "min": 0.99986732006073,
        "p25": 0.99986732006073,
        "p75": 0.99986732006073,
        "p90": 0.99986732006073
      },
      "unsafe_after_allow": 298,
      "unsafe_after_reject": 719
    },
    "turn": {
      "allow_then_unsafe": 77,
      "decision_count": 2500,
      "fallback_count": 815,
      "fallback_rate": 0.326,
      "hard_contract_fixed_reject_count": 623,
      "low_risk_unsafe_count": 78,
      "method": "casa_a_hard_or_recovery",
      "raw_risk_safe": {
        "count": 1732,
        "max": 0.9999980926513672,
        "mean": 0.11120342829135274,
        "median": 0.000300101179163903,
        "min": 3.82250482289237e-06,
        "p25": 6.174763257149607e-05,
        "p75": 0.0122367178555578,
        "p90": 0.5699426770210282
      },
      "raw_risk_unsafe": {
        "count": 768,
        "max": 0.9999994039535522,
        "mean": 0.8931070701655225,
        "median": 0.9998466372489929,
        "min": 3.3848533348646015e-05,
        "p25": 0.9944743514060974,
        "p75": 0.9999825060367584,
        "p90": 0.9999947547912598
      },
      "reject_but_still_unsafe": 691,
      "reject_count": 815,
      "reject_rate": 0.326,
      "risk_margin": {
        "count": 2500,
        "max": 0.1230745017528534,
        "mean": -0.5255206751256012,
        "median": -0.8719889700878412,
        "min": -0.876921079695876,
        "p25": -0.876818172439016,
        "p75": 0.11761879920959473,
        "p90": 0.12304816842079164
      },
      "skill": "turn",
      "threshold": {
        "count": 2500,
        "max": 0.8769249022006989,
        "mean": 0.8769249022006989,
        "median": 0.8769249022006989,
        "min": 0.8769249022006989,
        "p25": 0.8769249022006989,
        "p75": 0.8769249022006989,
        "p90": 0.8769249022006989
      },
      "unsafe_after_allow": 77,
      "unsafe_after_reject": 691
    },
    "walk": {
      "allow_then_unsafe": 370,
      "decision_count": 3750,
      "fallback_count": 3175,
      "fallback_rate": 0.8466666666666667,
      "hard_contract_fixed_reject_count": 1039,
      "low_risk_unsafe_count": 315,
      "method": "casa_a_hard_or_recovery",
      "raw_risk_safe": {
        "count": 2178,
        "max": 0.9999996423721313,
        "mean": 0.9081213196460148,
        "median": 0.9994677901268005,
        "min": 2.1901358195464127e-05,
        "p25": 0.9918106347322464,
        "p75": 0.9999200105667114,
        "p90": 0.9999745249748231
      },
      "raw_risk_unsafe": {
        "count": 1572,
        "max": 0.9999997615814209,
        "mean": 0.8029047800198378,
        "median": 0.9999755024909973,
        "min": 2.140933065675199e-05,
        "p25": 0.9905585050582886,
        "p75": 0.999994158744812,
        "p90": 0.999997615814209
      },
      "reject_but_still_unsafe": 1202,
      "reject_count": 3175,
      "reject_rate": 0.8466666666666667,
      "risk_margin": {
        "count": 3750,
        "max": 0.20021820068359375,
        "mean": 0.06423298533689424,
        "median": 0.20000898838043213,
        "min": -0.7997601515671704,
        "p25": 0.1920231580734253,
        "p75": 0.20019805431365967,
        "p90": 0.20021355152130127
      },
      "skill": "walk",
      "threshold": {
        "count": 3750,
        "max": 0.7997815608978271,
        "mean": 0.7997815608978271,
        "median": 0.7997815608978271,
        "min": 0.7997815608978271,
        "p25": 0.7997815608978271,
        "p75": 0.7997815608978271,
        "p90": 0.7997815608978271
      },
      "unsafe_after_allow": 370,
      "unsafe_after_reject": 1202
    }
  },
  "sonic_only": {
    "gesture": {
      "allow_then_unsafe": 1102,
      "decision_count": 1250,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 1038,
      "low_risk_unsafe_count": 6,
      "method": "sonic_only",
      "raw_risk_safe": {
        "count": 148,
        "max": 0.9999953508377075,
        "mean": 0.6408380357824451,
        "median": 0.9475137591362,
        "min": 1.922842557178228e-06,
        "p25": 0.09791344590485096,
        "p75": 0.9995620548725128,
        "p90": 0.9999141693115234
      },
      "raw_risk_unsafe": {
        "count": 1102,
        "max": 0.9999996423721313,
        "mean": 0.9924395653100586,
        "median": 0.9999910593032837,
        "min": 0.00042254291474819183,
        "p25": 0.999957799911499,
        "p75": 0.9999970197677612,
        "p90": 0.9999988079071045
      },
      "reject_but_still_unsafe": 0,
      "reject_count": 0,
      "reject_rate": 0.0,
      "risk_margin": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "skill": "gesture",
      "threshold": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "unsafe_after_allow": 1102,
      "unsafe_after_reject": 0
    },
    "passive": {
      "allow_then_unsafe": 2236,
      "decision_count": 2500,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 2145,
      "low_risk_unsafe_count": 21,
      "method": "sonic_only",
      "raw_risk_safe": {
        "count": 264,
        "max": 0.9999966621398926,
        "mean": 0.6511436479413576,
        "median": 0.978923112154007,
        "min": 2.1414857656054664e-06,
        "p25": 0.025431559421122074,
        "p75": 0.9992149472236633,
        "p90": 0.9999300360679626
      },
      "raw_risk_unsafe": {
        "count": 2236,
        "max": 0.9999996423721313,
        "mean": 0.9877891467081287,
        "median": 0.9999867677688599,
        "min": 4.4550452003022656e-05,
        "p25": 0.9997950196266174,
        "p75": 0.9999970197677612,
        "p90": 0.9999985694885254
      },
      "reject_but_still_unsafe": 0,
      "reject_count": 0,
      "reject_rate": 0.0,
      "risk_margin": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "skill": "passive",
      "threshold": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "unsafe_after_allow": 2236,
      "unsafe_after_reject": 0
    },
    "turn": {
      "allow_then_unsafe": 2209,
      "decision_count": 2500,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 2139,
      "low_risk_unsafe_count": 33,
      "method": "sonic_only",
      "raw_risk_safe": {
        "count": 291,
        "max": 0.9999966621398926,
        "mean": 0.6803052015972505,
        "median": 0.9864403605461121,
        "min": 3.0507910651067505e-06,
        "p25": 0.17288993299007416,
        "p75": 0.9991188645362854,
        "p90": 0.9998881816864014
      },
      "raw_risk_unsafe": {
        "count": 2209,
        "max": 0.9999994039535522,
        "mean": 0.981035953685491,
        "median": 0.999984622001648,
        "min": 0.00011215943231945857,
        "p25": 0.9998884201049805,
        "p75": 0.9999958276748657,
        "p90": 0.9999979734420776
      },
      "reject_but_still_unsafe": 0,
      "reject_count": 0,
      "reject_rate": 0.0,
      "risk_margin": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "skill": "turn",
      "threshold": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "unsafe_after_allow": 2209,
      "unsafe_after_reject": 0
    },
    "walk": {
      "allow_then_unsafe": 3419,
      "decision_count": 3750,
      "fallback_count": 0,
      "fallback_rate": 0.0,
      "hard_contract_fixed_reject_count": 2145,
      "low_risk_unsafe_count": 5,
      "method": "sonic_only",
      "raw_risk_safe": {
        "count": 331,
        "max": 0.9999995231628418,
        "mean": 0.9324300099828882,
        "median": 0.9983731508255005,
        "min": 5.453859102999559e-06,
        "p25": 0.9877123534679413,
        "p75": 0.9997500777244568,
        "p90": 0.9999514818191528
      },
      "raw_risk_unsafe": {
        "count": 3419,
        "max": 0.9999997615814209,
        "mean": 0.9973403413885042,
        "median": 0.9999638795852661,
        "min": 0.0003164679801557213,
        "p25": 0.9998258352279663,
        "p75": 0.9999914169311523,
        "p90": 0.9999967813491821
      },
      "reject_but_still_unsafe": 0,
      "reject_count": 0,
      "reject_rate": 0.0,
      "risk_margin": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "skill": "walk",
      "threshold": {
        "count": 0,
        "max": null,
        "mean": null,
        "median": null,
        "min": null,
        "p25": null,
        "p75": null,
        "p90": null
      },
      "unsafe_after_allow": 3419,
      "unsafe_after_reject": 0
    }
  }
}
```

## Top Failure Clusters

```json
[
  {
    "count": 794,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0001",
      "raw_critic_risk": "0.9999533891677856",
      "skill_idx": "1",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "hard",
    "skill": "walk",
    "target_bucket": "visual_collision_or_close",
    "violation_types": "collision,human_distance_violation,near_collision"
  },
  {
    "count": 604,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0002",
      "raw_critic_risk": "0.9999865293502808",
      "skill_idx": "1",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "medium",
    "skill": "walk",
    "target_bucket": "visual_near_boundary",
    "violation_types": "collision,human_distance_violation,near_collision"
  },
  {
    "count": 485,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0001",
      "raw_critic_risk": "0.999997615814209",
      "skill_idx": "2",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "hard",
    "skill": "turn",
    "target_bucket": "visual_collision_or_close",
    "violation_types": "collision,human_distance_violation,near_collision"
  },
  {
    "count": 429,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0000",
      "raw_critic_risk": "0.9993335604667664",
      "skill_idx": "1",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "medium",
    "skill": "walk",
    "target_bucket": "clean_safe",
    "violation_types": "human_distance_violation"
  },
  {
    "count": 330,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0002",
      "raw_critic_risk": "0.9999891519546509",
      "skill_idx": "2",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "medium",
    "skill": "turn",
    "target_bucket": "visual_near_boundary",
    "violation_types": "collision,human_distance_violation,near_collision"
  },
  {
    "count": 299,
    "decision": "reject",
    "example": {
      "episode_id": "casa_a_hard_or_recovery__seed_9101__episode_0015",
      "raw_critic_risk": "0.9999864101409912",
      "skill_idx": "5",
      "threshold": "0.7997815608978271"
    },
    "method": "casa_a_hard_or_recovery",
    "scene_complexity": "simple",
    "skill": "walk",
    "target_bucket": "visual_fall",
    "violation_types": "fall"
  },
  {
    "count": 278,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0000",
      "raw_critic_risk": "0.9999881982803345",
      "skill_idx": "2",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "medium",
    "skill": "turn",
    "target_bucket": "clean_safe",
    "violation_types": "human_distance_violation"
  },
  {
    "count": 264,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0001",
      "raw_critic_risk": "0.9999985694885254",
      "skill_idx": "7",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "hard",
    "skill": "passive",
    "target_bucket": "visual_collision_or_close",
    "violation_types": "collision,human_distance_violation,near_collision"
  },
  {
    "count": 260,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0000",
      "raw_critic_risk": "0.9998708963394165",
      "skill_idx": "3",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "medium",
    "skill": "passive",
    "target_bucket": "clean_safe",
    "violation_types": "human_distance_violation"
  },
  {
    "count": 246,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0003",
      "raw_critic_risk": "0.9998441934585571",
      "skill_idx": "1",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "simple",
    "skill": "walk",
    "target_bucket": "visual_fall",
    "violation_types": "collision,near_collision"
  },
  {
    "count": 240,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9102__episode_0003",
      "raw_critic_risk": "0.9999921321868896",
      "skill_idx": "5",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "simple",
    "skill": "walk",
    "target_bucket": "visual_fall",
    "violation_types": "fall"
  },
  {
    "count": 213,
    "decision": "allow",
    "example": {
      "episode_id": "sonic_only__seed_9101__episode_0002",
      "raw_critic_risk": "0.9999977350234985",
      "skill_idx": "7",
      "threshold": ""
    },
    "method": "sonic_only",
    "scene_complexity": "medium",
    "skill": "passive",
    "target_bucket": "visual_near_boundary",
    "violation_types": "collision,human_distance_violation,near_collision"
  }
]
```

## Target Buckets

```json
{
  "casa_a_hard_or_recovery": {
    "clean_safe": {
      "episodes": 350,
      "task_success": 240,
      "unsafe": 411
    },
    "visual_collision_or_close": {
      "episodes": 300,
      "task_success": 82,
      "unsafe": 893
    },
    "visual_fall": {
      "episodes": 300,
      "task_success": 1,
      "unsafe": 1627
    },
    "visual_near_boundary": {
      "episodes": 300,
      "task_success": 90,
      "unsafe": 807
    }
  },
  "sonic_only": {
    "clean_safe": {
      "episodes": 350,
      "task_success": 27,
      "unsafe": 1820
    },
    "visual_collision_or_close": {
      "episodes": 300,
      "task_success": 0,
      "unsafe": 2399
    },
    "visual_fall": {
      "episodes": 300,
      "task_success": 0,
      "unsafe": 2357
    },
    "visual_near_boundary": {
      "episodes": 300,
      "task_success": 0,
      "unsafe": 2390
    }
  }
}
```

## Scene Complexity

```json
{
  "casa_a_hard_or_recovery": {
    "hard": {
      "episodes": 300,
      "task_success": 82,
      "unsafe": 893
    },
    "medium": {
      "episodes": 650,
      "task_success": 330,
      "unsafe": 1218
    },
    "simple": {
      "episodes": 300,
      "task_success": 1,
      "unsafe": 1627
    }
  },
  "sonic_only": {
    "hard": {
      "episodes": 300,
      "task_success": 0,
      "unsafe": 2399
    },
    "medium": {
      "episodes": 650,
      "task_success": 27,
      "unsafe": 4210
    },
    "simple": {
      "episodes": 300,
      "task_success": 0,
      "unsafe": 2357
    }
  }
}
```
