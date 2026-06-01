# CASA Phase5 Online Policy Sweep

- candidates: `1`
- evaluated: `1`
- planned: `0`
- pilot pass: `1`

## Pareto Candidates

| candidate | method | unsafe_reduction | task_success_drop_rel | fallback/episode | pilot_pass |
|---|---|---:|---:|---:|---|
| issue7_heldout_2500 | SONIC + CASA-A hard-OR adaptive recovery | 0.5831 | -14.2963 | 3.8808 | True |

## Full Report

```json
{
  "candidate_count": 1,
  "evaluated_count": 1,
  "pareto_candidates": [
    {
      "candidate_id": "issue7_heldout_2500",
      "candidate_path": "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522/issue7_heldout_hard_or_2500_20260531/merged",
      "episodes": 1250,
      "fallback_rate_per_episode": 3.8808,
      "method": "casa_a_hard_or_recovery",
      "method_display": "SONIC + CASA-A hard-OR adaptive recovery",
      "pilot_pass": true,
      "status": "evaluated",
      "task_success_drop_rel_vs_sonic": -14.296296296296296,
      "task_success_rate": 0.3304,
      "unsafe_invocation_count": 3738,
      "unsafe_reduction_vs_sonic": 0.5830916796787865
    }
  ],
  "phase": "CASA Phase5 online policy pilot sweep",
  "pilot_pass_count": 1,
  "planned_count": 0,
  "promotion_rule": {
    "final_claim": "fresh held-out 2500-episode strict run only",
    "task_success_drop_rel_vs_sonic": "<= 0.20",
    "unsafe_reduction_vs_sonic": ">= 0.45"
  },
  "rows": [
    {
      "candidate_id": "issue7_heldout_2500",
      "candidate_path": "/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522/issue7_heldout_hard_or_2500_20260531/merged",
      "episodes": 1250,
      "fallback_rate_per_episode": 3.8808,
      "method": "casa_a_hard_or_recovery",
      "method_display": "SONIC + CASA-A hard-OR adaptive recovery",
      "pilot_pass": true,
      "status": "evaluated",
      "task_success_drop_rel_vs_sonic": -14.296296296296296,
      "task_success_rate": 0.3304,
      "unsafe_invocation_count": 3738,
      "unsafe_reduction_vs_sonic": 0.5830916796787865
    }
  ]
}
```
