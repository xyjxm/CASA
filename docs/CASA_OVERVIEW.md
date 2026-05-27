# CASA Overview

CASA studies skill-level safety for humanoid whole-body control. The core idea is
to wrap a capable base controller with structured skills, log every invocation,
label safety evidence with an oracle, and learn a gate that can reject unsafe
skill invocations before execution.

## Research Story

The upstream GR00T / GEAR-SONIC stack provides a general humanoid controller and
deployment environment. CASA adds a safety layer around that controller:

1. Convert continuous robot behavior into named skill invocations.
2. Record synchronized sim, deploy, command, scene, and skill logs.
3. Label rollouts with structured safety evidence.
4. Build clean datasets that exclude runtime and latency artifacts.
5. Train a raw critic from invocation features.
6. Calibrate global and per-skill conformal gates.
7. Compare CASA-A against SONIC-only and safety baselines.

## Architecture

| Component | Location | Role |
|---|---|---|
| Skill wrappers | `gear_sonic/casa/skills/` | Walk, turn, gesture, passive stop, registry, executor |
| Oracle | `gear_sonic/casa/oracle/` | Safety thresholds, violation records, rollout summaries |
| Logging | `gear_sonic/casa/loggers/` | Rollout-level and skill-level CSV/JSON logs |
| Dataset helpers | `gear_sonic/casa/dataset/` | Invocation feature extraction and split helpers |
| Scene props | `gear_sonic/casa/scene/` | MuJoCo user/obstacle placement and v2 scene generation |
| I/O helpers | `gear_sonic/casa/io/` | Episode log readers and ZMQ publisher helpers |
| Phase 5 utilities | `gear_sonic/casa/phase5.py` | Shared method names, thresholds, and evaluation helpers |

## Safety Labels

CASA separates semantic safety labels from artifact labels:

- `clean_safe`: oracle-safe and visually safe behavior.
- `visual_collision_or_close`: collision, close human contact, or unsafe gesture
  evidence visible in rollout videos.
- `visual_near_boundary`: visually meaningful near-boundary examples.
- `visual_fall`: visible loss of balance or fall.
- `runtime_artifact`: timeout or control-loop artifact, excluded from strict
  training sets.
- `latency_artifact`: injected latency or latency stress artifact, excluded from
  strict training sets.
- `oracle_only_invisible_quarantine`: oracle-unsafe but not visually
  interpretable, retained for audit rather than main visual training.

## Visual Review Policy

Phase 2 uses video rendering and VLM review as a pre-review filter. VLM output is
not the sole source of truth. Final acceptance combines oracle evidence,
structured artifact filters, contact sheets, and manual review priority queues.

The strongest review mode is oracle-assisted VLM review: the VLM receives both
the video evidence and a short hint such as "oracle suspects
human_distance_violation at 3.0s". This improves triage of subtle close-contact
events while keeping the oracle and human review as the final authority.

## Artifact Policy

Clean CASA datasets exclude:

- runtime timeouts and control-loop overruns,
- injected latency samples,
- elastic-band or startup constraint artifacts,
- initial scene overlaps,
- unsafe events that occur before the natural rollout window,
- generated outputs, local logs, caches, and checkpoints.

Those artifacts can still be archived or quarantined for debugging, but they do
not enter the strict training/calibration/test datasets.
