# Summary

`casa_h_mpc_safedpa_casa_refine` was implemented as a CASA-Hybrid extension and run on the full Phase5 held-out online grid:

- 8 methods
- 5 seeds: `3001,3002,3003,3004,3005`
- 100 episodes per seed
- 4000/4000 completed unique episodes after lowmem retries

The final hybrid audit is partial, not a full anchor win:

| check | result |
|---|---|
| Beats MPC-CBF-Humanoid | `false` |
| Beats SafeDPA | `false` |
| Beats CASA-A | `true` |
| Matched intervention budget passes all anchors | `false` |
| Win depends on more fallback/reject | `false` |

Key metrics:

| method | task success | progress | unsafe/ep | fallback/ep | reject/decision |
|---|---:|---:|---:|---:|---:|
| `casa_h_mpc_safedpa_casa_refine` | 0.210 | 0.53675 | 3.192 | 3.706 | 0.46325 |
| `mpc_cbf_humanoid_adapted` | 0.532 | 0.53375 | 2.268 | 3.730 | 0.46625 |
| `safedpa_adapted` | 0.138 | 0.56650 | 3.198 | 3.468 | 0.43350 |
| `casa_a_per_skill` | 0.096 | 0.48600 | 5.186 | 4.112 | 0.51400 |

Conclusion: the hybrid is a strict improvement over CASA-A under the audit definition, with lower unsafe rate and higher progress at lower fallback/reject. It does not exceed MPC-CBF-Humanoid because its unsafe rate is higher, and it does not exceed SafeDPA because its progress is lower and intervention budget is higher. The Pareto frontier should be reported instead of claiming full success.

Suggested next step: use calibration-only sweeps to target the tradeoff between SafeDPA progress and MPC-CBF unsafe rate, then reserve a new held-out online run for final validation.
