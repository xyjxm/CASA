# CASA-Hybrid MPC/SafeDPA Phase5 Held-Out Run

This directory records the lightweight manifest for the full held-out online run of
`casa_h_mpc_safedpa_casa_refine`.

Full output path:

`/mnt/data/students/lph/recording/phase5_hybrid_mpc_safedpa_full_20260608_121746/online_full_5seed100`

The large CSV/log artifacts remain under `/mnt/data` and are not committed.

Final audit conclusion:

`PARTIAL: casa_h_mpc_safedpa_casa_refine exceeds ['casa_a_per_skill'], but matched-budget or remaining-anchor checks do not all pass.`

Anchor outcome:

- Beats MPC-CBF-Humanoid: `false`
- Beats SafeDPA: `false`
- Beats CASA-A: `true`
- Matched intervention budget passes all anchors: `false`
- Win depends on more fallback/reject: `false`

The final held-out result was not used for tuning. It is reported as an audit result,
including the Pareto frontier and failed anchor checks.
