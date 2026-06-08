# Full SOTA Online Stage1 Results

This directory records the completed 2-seed x 10-episode online MuJoCo stage1 comparison for CASA and adapted SOTA baselines.

- generated_at: 2026-06-07
- source artifact directory: `/mnt/data/students/lph/recording/codex_goal_full_sota_online_comparison_20260606_235526/online/large_stage1_2seed10ep_all_methods/merged`
- run root: `/mnt/data/students/lph/recording/codex_goal_full_sota_online_comparison_20260606_235526`
- methods: 10
- episodes: 200
- gate decisions: 1600
- status: `PASS_STRICT_ONLINE`
- go: `True`

## Main Conclusion

MPC-CBF-Humanoid-adapted is the strongest method in this stage1 online slice: it has the lowest unsafe count and the highest task-success rate. CASA-A remains clearly better than SONIC-only and beats Global Conformal on unsafe count in this slice, but CASA-A is not the overall winner against Raw Critic and MPC-CBF-Humanoid-adapted.

Strict original Plan A claim remains `STRICT_PLAN_A_NO_GO` because CASA-A underperforms Raw Critic on unsafe reduction and exceeds the strict fallback budget.

## Key Results

| method | unsafe | unsafe reduction vs SONIC | task success | fallback/episode | reject/decision |
|---|---:|---:|---:|---:|---:|
| `mpc_cbf_humanoid_adapted` | 33 | 0.7643 | 0.7000 | 3.6000 | 0.4500 |
| `raw_critic_0p5` | 58 | 0.5857 | 0.4000 | 4.0000 | 0.5000 |
| `safedpa_adapted` | 61 | 0.5643 | 0.2500 | 3.3000 | 0.4125 |
| `casa_a_per_skill` | 73 | 0.4786 | 0.2000 | 3.9000 | 0.4875 |
| `global_conformal` | 88 | 0.3714 | 0.2000 | 3.8000 | 0.4750 |
| `pcbf_adapted` | 87 | 0.3786 | 0.1500 | 3.9500 | 0.4938 |
| `clbf_lbac_adapted` | 122 | 0.1286 | 0.1000 | 5.3000 | 0.6625 |
| `safer_splat_cbf_adapted` | 132 | 0.0571 | 0.0500 | 5.3500 | 0.6687 |
| `sonic_only` | 140 | 0.0000 | 0.0500 | 0.0000 | 0.0000 |
| `hard_contract` | 143 | -0.0214 | 0.0000 | 5.9000 | 0.7375 |

## Files

- `reports/online_report.md`: final online report with audit checks and anti-gaming claim audit.
- `data/method_summary.csv`: compact table for paper/table use.
- `data/method_summary.json`: method summary in JSON.
- `data/online_episode_results.csv`: per-episode online results.
- `data/gate_decisions.csv`: per-skill gate decisions.
- `data/online_acceptance_audit.json`: automated acceptance audit.
- `data/online_go_no_go.json`: GO/NO-GO audit.
- `manifests/large_stage1_2seed10ep_all_methods_command.sh`: command used for this stage1 run.
- `manifests/full_large_5seed100_all_methods_continuation_command.sh`: larger continuation command.
- `logs/large_stage1_2seed10ep_all_methods.tmux.log`: run progress log.
- `SHA256SUMS`: artifact hashes.
