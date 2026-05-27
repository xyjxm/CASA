# CASA

CASA is a research codebase for **skill-level safety evaluation and gating for
humanoid whole-body control**. It builds on the GR00T / GEAR-SONIC stack and adds
structured skill wrappers, rollout logging, safety oracles, clean visual dataset
builders, critic training, conformal calibration, and offline/online baseline
evaluation.

The repository is public so collaborators can inspect, reproduce, and extend the
CASA pipeline. Generated datasets, experiment outputs, local caches, model
weights, and machine-specific logs are intentionally excluded from Git.

## What CASA Adds

- **Skill interface** for walk, turn, gesture, and passive/stop invocations.
- **Safety oracle** that labels rollout evidence such as falls, collisions,
  near-boundary behavior, human-distance violations, runtime artifacts, and
  latency artifacts.
- **Clean visual dataset tooling** for Phase 2 video review and VLM-assisted
  triage.
- **Invocation datasets and critics** for Phase 3 and Phase 4 safety learning.
- **Conformal gates** for Phase 5 global and per-skill rejection thresholds.
- **Five-baseline evaluation** comparing SONIC-only, hard contracts, raw critic,
  global conformal, and CASA-A per-skill conformal gating.

## CASA Code Map

| Area | Location | Purpose |
|---|---|---|
| CASA Python package | `gear_sonic/casa/` | Skills, oracle, logging, scene props, dataset helpers, Phase 5 utilities |
| CASA scripts | `gear_sonic/scripts/casa_*.py` | Phase runners, dataset builders, audits, reports, and evaluation tools |
| Research notes | `idea_and_plan/` | Planning and implementation reports from Phase 0-5 |
| CASA overview | `docs/CASA_OVERVIEW.md` | Research story, architecture, labels, and artifact policy |
| CASA pipeline | `docs/CASA_PIPELINE.md` | Reproducibility guide for Phase 0-5 |
| Script index | `gear_sonic/scripts/README.md` | Sorted index of CASA command-line scripts |
| Public safety scan | `docs/PUBLIC_REPO_SECURITY_SCAN.md` | Repository scan checklist and latest results |

## Experiment Phases

CASA is organized as a staged pipeline:

| Phase | Goal | Main outputs |
|---|---|---|
| Phase 0 | Skill sanity and repeatability | Skill logs and long-horizon smoke reports |
| Phase 1 | Skill wrapper validation | Repeatability summaries and per-skill evidence |
| Phase 2 | Clean visual rollout dataset | Videos, contact sheets, VLM JSON, clean rollout CSVs |
| Phase 3 | Invocation feasibility dataset | State-skill-label rows and mini critic checks |
| Phase 4 | Strict clean 50k dataset | Train/calibration/test splits and raw critic artifacts |
| Phase 5 | Conformal and baseline evaluation | Thresholds, baseline results, online episode summaries |

See [`docs/CASA_PIPELINE.md`](docs/CASA_PIPELINE.md) for the detailed workflow
and the scripts used by each phase.

## Quick Start

Clone with Git LFS enabled:

```bash
git clone https://github.com/xyjxm/CASA.git
cd CASA
git lfs install
git lfs pull
```

Create the MuJoCo simulation environment:

```bash
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
python check_environment.py
```

Run a small CASA sanity check:

```bash
python gear_sonic/scripts/casa_run_sanity_check.py \
  --output-dir outputs/casa/phase0_sanity_smoke \
  --duration-seconds 30
```

Most full experiments require a live MuJoCo sim/deploy pair and write large
outputs under `outputs/`, which is ignored by Git. The public repository contains
the source code, documentation, and small reference assets only.

## Upstream Relationship

CASA is built on top of the upstream GR00T Whole-Body Control and GEAR-SONIC
codebase. The upstream stack provides the humanoid controller, MuJoCo/Isaac
integration, deployment tools, teleoperation utilities, and MotionBricks
components. CASA-specific additions are concentrated in `gear_sonic/casa/`,
`gear_sonic/scripts/casa_*.py`, and the CASA documentation under `docs/`.

Useful upstream references:

- GR00T Whole-Body Control docs: <https://nvlabs.github.io/GR00T-WholeBodyControl/>
- GEAR-SONIC project page: <https://nvlabs.github.io/GEAR-SONIC/>
- MotionBricks project page: <https://nvlabs.github.io/motionbricks/>

## Reproducibility Notes

- Large generated artifacts stay outside Git in `outputs/`.
- Model weights and checkpoints are excluded; download upstream checkpoints from
  their official sources when needed.
- CASA dataset builders reject runtime/latency artifacts for strict training
  sets and quarantine samples that are unsafe only by non-visual oracle evidence.
- Phase 5 uses the Phase 4 strict clean calibration/test splits and does not
  change Go/No-Go thresholds during evaluation.

## Collaboration Workflow

Recommended branch flow:

1. Keep `main` stable and protected.
2. Use `dev` as the integration branch.
3. Create `feature/<short-name>` branches from `dev`.
4. Open pull requests for review before merging.
5. Keep generated files, local logs, credentials, and downloaded weights out of
   commits.

Current `main` protection requires pull requests with at least one approving
review for non-admin collaborators and blocks force pushes/deletions. Enable
secret scanning push protection in GitHub settings when available.

## License And Attribution

Source code follows the repository license in [`LICENSE`](LICENSE). Upstream
third-party notices and attribution files are preserved under [`legal/`](legal/).
CASA-specific research code is added as an extension on top of the GR00T /
GEAR-SONIC stack; please cite or acknowledge the upstream projects when using
their controller, deployment, or motion-model components.

If you use CASA-specific safety gating, dataset, or conformal evaluation logic,
please cite this repository and describe the CASA phase outputs used in your
experiment.
