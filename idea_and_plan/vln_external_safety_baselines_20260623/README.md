# VLN External Safety Baselines 2026-06-23

This directory contains lightweight, git-trackable artifacts for the locked online
SONIC VLN comparison across three adapted external safety baselines.

Full run directory:

`/mnt/data/students/lph/recording/vln_external_safety_baselines_20260623_134846`

Protocol:

- eval stage: `locked`
- held-out episodes per method: `50`
- seed: `260614`
- max steps: `180`
- policy backend: `real_navid_visual_adapter`
- GPU: GPU0, UUID `GPU-8b1188b9-c3d8-a887-a7f9-073ccd6e33fb`
- same held-out split, language tasks, unsafe oracle, and success evaluator for all methods

Methods:

- `mpc_cbf_humanoid_adapted`
- `safedpa_adapted`
- `spark_style_filter_adapted`

Result summary:

| method | safe success | unsafe | reject | replan | fallback | intervention rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mpc_cbf_humanoid_adapted` | 18/50 (0.36) | 0 | 4584 | 4584 | 18 | 0.691 |
| `safedpa_adapted` | 18/50 (0.36) | 0 | 5202 | 0 | 18 | 0.784 |
| `spark_style_filter_adapted` | 18/50 (0.36) | 0 | 4712 | 4712 | 159 | 0.710 |

External-only conclusion:

- All three adapted external baselines reached `18/50 = 0.36` safe success.
- All three adapted external baselines had zero unsafe violations in this locked run.
- The adapted methods differ mainly in intervention budget: `safedpa_adapted` intervened most, followed by `spark_style_filter_adapted`, then `mpc_cbf_humanoid_adapted`.

Notes:

- These baselines are adapted method implementations, not official reproductions of the cited external systems.
- `spark_style_filter_adapted` is explicitly a SPARK-style runtime safety filter adaptation; official SPARK code was not used.
- The real online evaluation completed all 50/50 episodes for all three external methods. A first aggregation attempt exited after the online episodes completed due to a list-valued audit-field bug; the bug was fixed and the same completed run was postprocessed to generate the root summaries and this lightweight artifact set.
- Large decision logs, trajectory logs, videos, frames, and stdout logs remain under `/mnt/data/students/lph/recording/` and are not tracked in git.
