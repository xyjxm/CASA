# CASA-VLN Safety Gate Online Evidence

This lightweight artifact records the locally generated CASA-gated SONIC VLN online evaluation.
Large videos, frames, CSV logs, and model outputs are intentionally kept outside git.

## Source

- Repository branch: `vln-navid-stop-turn-opt-20260611`
- Base commit before this upload: `6bcc12dc783e05cfef3f55c28fb75b705fe01969`
- Full output path: `/mnt/data/students/lph/recording/casa_vln_safety_locked_real_navid_20260611_190121`
- Policy backend: `real_navid_visual_adapter`
- Real NaVid/Uni-NaVid checkpoint loaded: `true`
- CASA risk source: `real_casa_critic`
- CASA gate method: `casa_a_hard_or_per_skill`

## Locked Online Result

Evaluation used 20 held-out language navigation episodes per method in the furnished MuJoCo maze.

| method | safe_success_rate | success_rate | unsafe_violation_rate | wall_contact_steps | CASA rejects | CASA replans |
|---|---:|---:|---:|---:|---:|---:|
| `vln_only` | 0.30 | 0.30 | 0.45 | 181 | 0 | 0 |
| `vln_casa_reject_only` | 0.15 | 0.15 | 0.00 | 0 | 16 | 0 |
| `vln_casa_replan` | 0.15 | 0.15 | 0.00 | 0 | 390 | 390 |

Final audit status:

```text
PASS_IMPLEMENTED_BUT_NO_IMPROVEMENT
```

Interpretation: CASA eliminated observed wall-contact unsafe violations in this locked run, but it reduced safe task success from 30% to 15%. This supports a safety-gate implementation claim, not a safe-success improvement claim.

## Key Files Outside Git

- `run_manifest.json`
- `casa_vln_safety_report.md`
- `data/method_summary.csv`
- `data/gate_decisions.csv`
- `data/replan_logs.csv`
- `data/decision_logs.csv`
- `data/trajectory_logs.csv`
- `videos/vln_only/*.mp4`
- `videos/vln_casa_reject_only/*.mp4`
- `videos/vln_casa_replan/*.mp4`
- `frames/`
