# SOTA Baseline Comparison Plan for CASA

## 0. Goal

This plan defines how to compare CASA against external SOTA safety methods in the same CASA / SONIC Phase5 environment.

The comparison goal is not to copy numbers from other papers. The comparison goal is:

```text
Adapt each external method into the CASA Phase5 MuJoCo humanoid skill-invocation setting,
run all methods under the same seeds, tasks, controller, safety oracle, and metrics,
then report a reproducible baseline comparison table for the paper.
```

Main paper claim boundary:

```text
Valid:
  "We compare CASA with adapted implementations of representative safe RL,
   CBF, conformal-risk, mapping-based, and predictive safety-filter baselines
   in the same SONIC humanoid skill-invocation benchmark."

Invalid:
  "CASA outperforms the original published systems on their native tasks."
```

The output should support:

```text
Baseline comparison table
Evaluation protocol table
Implementation fidelity / adaptation table
Failure-mode analysis
Runtime overhead table
```

---

## 1. Fixed CASA Evaluation Environment

All adapted SOTA baselines must be evaluated in the same environment as Plan A:

```text
Controller:
  SONIC / GEAR-SONIC humanoid controller, frozen.

Simulation:
  MuJoCo sim2sim / deploy pipeline.

Task:
  Phase5 medium-horizon humanoid interaction sequence.

Episode skill sequence:
  walk
  turn
  passive stop
  gesture
  walk
  turn
  passive wait
  walk

Skill set:
  walk, turn, gesture, passive.

Scene factors:
  randomized obstacles
  user proxy
  target bucket
  runtime perturbation

Safety oracle:
  collision
  near_collision
  fall
  human_distance_violation
  unsafe_gesture
  runtime_timeout
```

Main online root for reference:

```text
/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602
```

All generated comparison artifacts, reports, figures, manifests, and videos must go under a run-specific subfolder:

```text
/mnt/data/students/lph/recording/sota_baseline_comparison_YYYYMMDD_HHMMSS/
```

Do not write directly into `/mnt/data/students/lph/recording/`.

---

## 2. Core Fairness Rule

Use two distinct result categories.

### Category A: Main Apples-to-Apples Table

All methods use the same high-level interaction contract:

```text
input:
  state
  candidate skill
  candidate skill parameters
  obstacle/user proxy features
  optional uncertainty features

output:
  allow / reject
  optional risk score
  optional fallback reason
```

If a method is originally a safety filter that modifies continuous actions, the main table should run its gate-only variant:

```text
if method predicts the candidate skill is safe:
  allow original SONIC skill
else:
  reject and use the same passive fallback policy as CASA
```

This keeps the comparison against CASA-A clean because CASA-A is a skill-invocation gate, not a controller replacement.

### Category B: Supplementary Controller-Modifying Table

Some CBF/MPC methods are naturally action filters. They may be evaluated in an additional table:

```text
input:
  nominal skill command

output:
  corrected skill command or fallback
```

This table must be labeled separately:

```text
Controller-modifying adapted baselines.
Not directly apples-to-apples with CASA-A gate-only results.
```

---

## 3. Candidate External Methods

| Short name for paper | Source method | Venue/status to verify | Core direction | Code status | Main adaptation route |
|---|---|---|---|---|---|
| CLBF-LBAC-adapted | Reinforcement Learning for Safe Robot Control using Control Lyapunov Barrier Functions | ICRA 2023 | safe RL + CLBF | no confirmed official repo | learned CLBF score gate over CASA skill candidates |
| SafeDPA-adapted | Safe Deep Policy Adaptation | ICRA 2024 | RL adaptation + CBF safety filter | code discovery required | SafeDPA-style dynamics adaptation + CBF gate/filter over skill commands |
| PCBF-RL-adapted | Reinforcement Learning with Probabilistically Safe Control Barrier Functions for Ramp Merging / probabilistic CBF family | ICRA 2023 | probabilistic CBF + RL | no confirmed official repo | uncertainty-aware CBF gate using chance constraints |
| SAFER-Splat-adapted | SAFER-Splat: CBF for Safe Navigation with Online Gaussian Splatting Maps | arXiv 2024, repo states ICRA 2025 accepted | CBF + online mapping | public GitHub repo | map obstacles/user proxy to Gaussian/ellipsoid CBF hazards |
| CRC-CBF-adapted | Safe Probabilistic Planning for Human-Robot Interaction using Conformal Risk Control | 2026 preprint/project | conformal risk + CBF planning | project website/code reported | conformal-risk-calibrated CBF gate |
| MPC-CBF-Humanoid-adapted | Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson Safety Functions to CBF Constrained MPC | Humanoids 2025 / arXiv 2025 | predictive MPC + CBF for humanoids | supplement/code discovery required | receding-horizon skill-level predictive safety filter |

Important metadata note:

```text
Before paper submission, verify exact venue, title, authors, code URL, and publication status.
If code is unavailable or materially incompatible, label the result as "paper-faithful adapted proxy".
```

---

## 4. Shared Adapter Interface

Implement every external method through one common interface.

Suggested module:

```text
gear_sonic/casa/baselines/sota_adapters/
```

Suggested interface:

```python
class SotaBaselineAdapter:
    name: str
    mode: Literal["gate", "filter"]

    def fit(self, train_data, val_data, config) -> None:
        ...

    def calibrate(self, calibration_data, config) -> None:
        ...

    def decide(self, state, candidate_skill, context) -> GateDecision:
        ...
```

Suggested decision object:

```python
GateDecision(
    method_name: str,
    allow: bool,
    risk_score: float | None,
    threshold: float | None,
    corrected_skill: Skill | None,
    fallback_mode: str | None,
    diagnostics: dict,
)
```

Every adapted method must log:

```text
method_name
source_method
implementation_fidelity: exact_code | paper_faithful_proxy | lightweight_proxy
allow/reject
risk score or barrier score
threshold / risk budget
fallback reason
runtime_ms
solver_status if applicable
```

---

## 5. Data Splits

Use strict non-overlapping splits.

```text
train:
  train learned models, critics, dynamics residual models, uncertainty models.

calibration / validation:
  tune thresholds, CBF margins, conformal quantiles, risk budgets, solver penalties.

test / online:
  final online evaluation only.
```

Rules:

```text
No method may tune thresholds on test episodes.
No method may use labels from the final online test split.
If a method needs more training data than CASA, report the data budget explicitly.
```

Recommended variants:

```text
data_matched:
  same train/calibration budget as CASA-A.

full_budget:
  method-specific larger training budget, reported separately.
```

Main table should use `data_matched` unless there is a strong reason not to.

---

## 6. Baseline-Specific Implementation Plans

### 6.1 CLBF-LBAC-adapted

Source idea:

```text
Learn a Lyapunov/barrier-style safety certificate or critic from data and use it to guide safe RL/control.
```

CASA adaptation:

```text
Do not retrain SONIC.
Train a learned CLBF-style scorer over (state, skill, skill_params).
Allow a skill only if the learned barrier condition is satisfied with margin.
Otherwise reject to passive fallback.
```

Minimum implementation:

```text
features:
  CASA Phase4/Phase5 state features
  obstacle/user proxy features
  skill one-hot
  skill parameters

labels:
  unsafe label
  fall label
  near-collision/collision label

model:
  shared MLP encoder
  barrier score head B(s, a)
  optional Lyapunov progress head V(s, a)

decision:
  allow if B(s, a) >= threshold and optional progress condition holds
```

Report as:

```text
CLBF-LBAC-adapted gate
implementation_fidelity = paper_faithful_proxy
```

Risks:

```text
Original method is a safe RL controller, not a frozen-controller skill gate.
No official repo is confirmed.
This baseline is useful as a representative CLBF safe-RL adaptation, not an exact reproduction.
```

### 6.2 SafeDPA-adapted

Source idea:

```text
Adapt a deep RL policy to new dynamics/environments while applying a CBF safety filter.
```

CASA adaptation:

```text
Treat SONIC skill command as the nominal policy.
Learn a lightweight environment/dynamics adaptation module from Phase4/Phase5 data.
Use the adapted model to evaluate a CBF-style safety condition for each candidate skill.
```

Minimum gate-only implementation:

```text
learn:
  dynamics residual predictor for next state / safety-relevant features
  environment configuration predictor if useful

decision:
  predict rollout safety margin under candidate skill
  allow if CBF margin stays above threshold
  reject otherwise
```

Optional filter-mode implementation:

```text
project candidate skill params onto a safe set:
  vx, vy, facing_yaw, duration

if projection distance <= max_correction_budget:
  execute corrected skill
else:
  fallback passive
```

Report as:

```text
SafeDPA-adapted gate
SafeDPA-adapted filter, supplementary only
```

Risks:

```text
Original method adapts policies; CASA freezes the humanoid controller.
If official code is unavailable or too environment-specific, use paper-faithful proxy.
```

### 6.3 PCBF-RL-adapted

Source idea:

```text
Use probabilistic CBF constraints to handle model uncertainty and produce safety in probability.
```

CASA adaptation:

```text
Fit an uncertainty model for safety margin prediction.
Use chance-constrained CBF acceptance:
  P(h_next >= 0 | state, skill) >= 1 - epsilon
```

Minimum implementation:

```text
safety margin h:
  min distance to obstacle/user proxy
  fall/uprightness margin
  gesture-human-distance margin

uncertainty:
  ensemble variance, GP-lite residual, quantile model, or conformal residual bound

decision:
  allow if lower confidence bound of predicted h_next is safe
  reject otherwise
```

Recommended risk budgets:

```text
epsilon in {0.01, 0.05, 0.10}
select on calibration only
```

Report as:

```text
PCBF-adapted gate
implementation_fidelity = paper_faithful_proxy
```

Risks:

```text
Original ICRA 2023 example is autonomous driving/ramp merging, not humanoid skill invocation.
Use this as representative probabilistic-CBF adaptation, not direct reproduction.
```

### 6.4 SAFER-Splat-adapted

Source idea:

```text
Use CBF safety filtering against a detailed online Gaussian Splatting map.
```

CASA adaptation:

```text
Use MuJoCo obstacle and user proxy state as the map source.
Represent each obstacle/user proxy as a Gaussian, ellipsoid, or signed-distance primitive.
Evaluate CBF safety for the candidate skill trajectory.
```

Minimum gate-only implementation:

```text
for each candidate skill:
  rollout a short skill-level trajectory approximation
  compute minimum Gaussian/ellipsoid CBF margin
  allow if margin >= threshold
  reject otherwise
```

Optional filter-mode:

```text
solve a small QP to minimally modify candidate velocity/yaw/duration
under Gaussian/ellipsoid CBF constraints
```

Report as:

```text
SAFER-Splat-CBF-adapted gate
SAFER-Splat-CBF-adapted filter, supplementary only
```

Risks:

```text
Original method is perception/mapping-heavy and built around online GSplat/ROS.
Our main adaptation should not give it privileged perception unless explicitly labeled.
If using MuJoCo ground-truth obstacles, label as oracle-map adaptation.
```

### 6.5 CRC-CBF-adapted

Source idea:

```text
Combine conformal risk control with CBF planning to provide probabilistic safety guarantees under prediction error.
```

CASA adaptation:

```text
Train or use a CBF safety-value predictor for each candidate skill.
Calibrate prediction error using conformal risk control on the calibration split.
Use the calibrated risk margin to decide allow/reject.
```

Minimum implementation:

```text
score:
  predicted CBF violation or safety-margin loss

calibration:
  conformal risk control over safety loss

decision:
  allow if calibrated risk <= alpha
  reject otherwise
```

Recommended alpha values:

```text
alpha in {0.01, 0.05, 0.10}
select on calibration only
```

Report as:

```text
CRC-CBF-adapted gate
implementation_fidelity = exact_code if project code is integrated, otherwise paper_faithful_proxy
```

Risks:

```text
Very close in spirit to CASA because both use calibration.
Must distinguish:
  CASA = per-skill conformal gate over learned unsafe critic.
  CRC-CBF = conformal risk margin over CBF safety-value prediction/planning.
```

### 6.6 MPC-CBF-Humanoid-adapted

Source idea:

```text
Use predictive safety filtering with CBF constraints and MPC for humanoid navigation.
```

CASA adaptation:

```text
Build a skill-level receding-horizon model over base pose/yaw and nearby hazards.
For each candidate skill, predict a short horizon and solve an MPC-CBF feasibility problem.
```

Minimum gate-only implementation:

```text
allow if MPC-CBF feasibility problem is feasible with the candidate skill held fixed
reject otherwise
```

Optional filter-mode:

```text
allow corrected skill if MPC finds a minimally changed safe skill command
fallback if infeasible or correction exceeds budget
```

Report as:

```text
MPC-CBF-Humanoid-adapted gate
MPC-CBF-Humanoid-adapted filter, supplementary only
```

Risks:

```text
Solver runtime may be high.
Full geometry-aware Poisson safety functions may be too heavy for a first implementation.
Start with a reduced-order skill-level model and report fidelity clearly.
```

---

## 7. Implementation Phases

### Phase S0: Literature and Metadata Verification

Tasks:

```text
1. Verify exact title, venue, year, authors, code URL, license, and commit hash.
2. Save metadata to JSON.
3. Classify each method:
   exact_code
   paper_faithful_proxy
   lightweight_proxy
4. Decide which methods enter the main paper table.
```

Artifacts:

```text
data/sota_method_metadata.json
reports/sota_method_metadata.md
```

Acceptance:

```text
Each method has verified source metadata or an explicit "unverified / proxy only" label.
No paper table uses unverified venue/code claims.
```

### Phase S1: Common Adapter API

Tasks:

```text
1. Implement SotaBaselineAdapter base class.
2. Implement GateDecision dataclass.
3. Add adapter registry.
4. Add CLI support for new method names.
5. Add logging fields to online result CSV.
```

New method names:

```text
clbf_lbac_adapted
safedpa_adapted
pcbf_adapted
safer_splat_cbf_adapted
crc_cbf_adapted
mpc_cbf_humanoid_adapted
```

Acceptance:

```text
All methods can run in dry-run mode and produce one GateDecision per skill decision.
No online rollout changes are required for CASA-A or existing baselines.
```

### Phase S2: Shared Data and Calibration Pipeline

Tasks:

```text
1. Load Phase4/Phase5 training and calibration data.
2. Build unified feature extraction.
3. Build shared train/calibration/test split manifest.
4. Add no-test-leakage checks.
```

Artifacts:

```text
data/sota_split_manifest.json
data/sota_feature_schema.json
reports/sota_data_audit.md
```

Acceptance:

```text
Feature schema is identical across all adapted methods unless explicitly justified.
Calibration does not touch online test episodes.
```

### Phase S3: Method Adapters

Implement in this order:

```text
1. PCBF-adapted
2. CRC-CBF-adapted
3. MPC-CBF-Humanoid-adapted
4. SAFER-Splat-CBF-adapted
5. CLBF-LBAC-adapted
6. SafeDPA-adapted
```

Rationale:

```text
PCBF / CRC-CBF / MPC-CBF are closest to gate/filter behavior.
SAFER-Splat needs map conversion.
CLBF-LBAC and SafeDPA need the most learning/control adaptation.
```

Acceptance per adapter:

```text
1. Unit tests for feature extraction and decision shape.
2. Offline replay runs on at least 100 episodes.
3. Calibration thresholds are saved.
4. Runtime stats are logged.
5. Adapter produces stable allow/reject decisions without crashing online rollout.
```

### Phase S4: Offline Replay Evaluation

Tasks:

```text
1. Replay candidate skill decisions from logged Phase5 episodes.
2. Compute offline unsafe-rejection precision/recall.
3. Compute reject rate and fallback proxy rate.
4. Rank threshold candidates using calibration only.
```

Artifacts:

```text
data/offline_replay_results.csv
data/offline_replay_metrics.json
figures/offline_roc_pr_curves.png
reports/offline_replay_report.md
```

Acceptance:

```text
Each method has finite metrics.
No method enters online evaluation without offline sanity checks.
```

### Phase S5: Online Phase5 Evaluation

Run all methods in the same online environment.

Recommended first pass:

```text
methods:
  sonic_only
  hard_contract
  raw_critic_0p5
  global_conformal
  casa_a_per_skill
  pcbf_adapted
  crc_cbf_adapted
  mpc_cbf_humanoid_adapted
  safer_splat_cbf_adapted
  clbf_lbac_adapted
  safedpa_adapted

seeds:
  1234, 1235, 1236, 1237, 1238

episodes per method:
  500
```

If full 11-method evaluation is too slow, use staged evaluation:

```text
Stage 1:
  50 episodes per method smoke test.

Stage 2:
  100 episodes per method debug run.

Stage 3:
  500 episodes per method final run.
```

Artifacts:

```text
online/online_episode_results.csv
online/online_gate_decisions.csv
data/online_metrics.json
data/online_method_runtime.json
reports/online_sota_baseline_report.md
```

Acceptance:

```text
All final methods have the same seeds and episode count.
Any failed method is reported, not silently removed.
```

### Phase S6: Statistics, Figures, and Paper Tables

Metrics:

```text
unsafe count
unsafe rate
unsafe reduction vs SONIC-only
safe completion / task success
task-success drop vs SONIC-only
fallback per episode
reject per decision
per-skill unsafe
per-skill reject
fall count
collision count
near-collision count
human-distance violation count
runtime overhead per gate decision
solver infeasibility rate if applicable
```

Statistical analysis:

```text
paired seed/episode comparison where possible
bootstrap confidence intervals
per-seed mean and variance
failure-type breakdown
runtime distribution p50/p90/p99
```

Main table columns:

```text
method
implementation fidelity
uses calibration?
uses dynamics/model?
uses map/perception?
modifies skill command?
unsafe rate
unsafe reduction
safe completion
fallback / episode
reject / decision
runtime p90 ms
```

Acceptance:

```text
No method is reported without implementation fidelity.
No direct cross-paper numeric comparison is placed in the main result table.
```

### Phase S7: Video and Case Study Package

Generate videos only for final selected cases.

Selection rules:

```text
1. CASA clean-safe and at least one SOTA/baseline unsafe.
2. SOTA method clean-safe and CASA unsafe, if such cases exist.
3. all methods unsafe, for limitation analysis.
4. all methods safe, for non-cherry-picked sanity.
```

Artifacts:

```text
videos/
frames/
figures/
data/selected_cases_manifest.json
reports/selected_case_study.md
```

Acceptance:

```text
Every selected video has extracted frames.
Every selected case has a manifest with method outcomes and hashes.
```

---

## 8. Suggested Paper Tables

### Table 1: External Method Adaptation Summary

```text
Method
Original venue
Original domain
Original safety mechanism
Official code used?
CASA adaptation
Main-table fidelity
```

### Table 2: Main Online Results

```text
Method
Unsafe rate
Unsafe reduction
Safe completion
Fallback / episode
Reject / decision
Fall count
Collision count
Runtime p90 ms
```

### Table 3: Gate vs Filter Supplement

```text
Method
Gate-only result
Filter-mode result
Correction rate
Mean correction magnitude
Solver infeasible rate
```

### Table 4: Implementation Cost

```text
Method
Training data
Calibration data
Extra model required
Solver required
Mapping/perception required
Average runtime
Engineering complexity
```

---

## 9. Go / No-Go Criteria

For a method to enter the main table:

```text
1. It runs online in the CASA Phase5 environment.
2. It uses the same test seeds and episode count as the other main methods.
3. It has no test leakage.
4. It logs all required metrics.
5. Its implementation fidelity is stated.
6. Its failure modes are included.
```

For a method to support a strong claim:

```text
unsafe reduction vs SONIC-only >= 0.4000
task-success drop vs SONIC-only <= 0.3000
runtime p90 acceptable for online gate/filter
no hidden privileged test information
```

If a method fails these thresholds:

```text
Keep it in the table as a negative or partial result if it ran correctly.
Do not remove it only because it underperforms.
```

---

## 10. Recommended First Implementation Slice

Start with three baselines:

```text
1. PCBF-adapted
2. CRC-CBF-adapted
3. MPC-CBF-Humanoid-adapted
```

Reason:

```text
They are closest to CASA's runtime gate/filter role and easiest to evaluate fairly.
```

Then add:

```text
4. SAFER-Splat-CBF-adapted
5. CLBF-LBAC-adapted
6. SafeDPA-adapted
```

Reason:

```text
These require more adaptation, mapping, or policy-learning assumptions.
```

---

## 11. Reference Links to Verify

These links are for metadata verification and should be rechecked before paper submission:

```text
CLBF / LBAC:
  https://arxiv.org/abs/2305.09793
  https://discovery.ucl.ac.uk/id/eprint/10185764/

SafeDPA:
  https://arxiv.org/abs/2310.08602

Probabilistic CBF / RL:
  https://www.ri.cmu.edu/publications/reinforcement-learning-with-probabilistically-safe-control-barrier-functions-for-ramp-merging/

SAFER-Splat:
  https://arxiv.org/abs/2409.09868
  https://github.com/chengine/safer-splat

CRC-CBF:
  https://arxiv.org/abs/2603.10392
  https://jakeagonzales.github.io/crc-cbf-website/

MPC-CBF Humanoid:
  https://arxiv.org/abs/2508.11129
```

---

## 12. Final Deliverable Checklist for Codex CLI

Codex CLI should finish with:

```text
1. SOTA method metadata JSON and report.
2. Common adapter API.
3. At least three implemented adapted baselines.
4. Offline replay report.
5. Online smoke test.
6. Full online result table if runtime allows.
7. Figures and case-study videos under /mnt/data/students/lph/recording/<run_name>/.
8. README with source paths, commands, hashes, and timestamps.
9. Paper-ready baseline comparison table.
10. Clear list of methods that are exact-code reproduction vs adapted proxy.
```
