# SONIC Stage 3 Metrics

This runbook records the Stage 3 offline metrics pass. It reuses a Stage 2
SONIC-Naive run and computes switch-safety metrics without changing robot
commands.

Stage 3 is an offline judge. It does not add SkillGuard gating, recovery,
obstacle logic, edge logic, or learned risk estimation.

## 1. Reuse a Stage 2 Run

Use a completed Stage 2 run directory:

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl
export RUN_ID=<stage2_run_id>
```

The expected inputs are:

```text
outputs/sonic_stage2/<RUN_ID>/sim/sim_state.csv
outputs/sonic_stage2/<RUN_ID>/deploy/*.csv
outputs/sonic_stage2/<RUN_ID>/task/task_events.csv
outputs/sonic_stage2/<RUN_ID>/target_motion.csv
```

## 2. Compute Metrics

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl
source .venv_sim/bin/activate

python -u gear_sonic/scripts/summarize_sonic_stage3_metrics.py \
  --sim-log-dir "outputs/sonic_stage2/${RUN_ID}/sim" \
  --deploy-log-dir "outputs/sonic_stage2/${RUN_ID}/deploy" \
  --task-log-dir "outputs/sonic_stage2/${RUN_ID}/task" \
  --target-motion-log "outputs/sonic_stage2/${RUN_ID}/target_motion.csv" \
  --output-dir "outputs/sonic_stage3/${RUN_ID}"
```

Expected outputs:

```text
outputs/sonic_stage3/<RUN_ID>/metrics_timeline.csv
outputs/sonic_stage3/<RUN_ID>/switch_events.csv
outputs/sonic_stage3/<RUN_ID>/episode_metrics.json
outputs/sonic_stage3/<RUN_ID>/metrics_config.json
```

## 3. Metrics Interpretation

`metrics_timeline.csv` is the aligned per-sample timeline. It contains:

```text
segment
current_skill
task_phase
base_speed
near_fall_flag
torso_unstable
collision_flag
tracking_diverged
unsafe_gesture_flag
```

`switch_events.csv` contains one row per phase switch:

```text
source_skill
target_skill
invalid_switch
invalid_switch_reasons
switch_induced_failure
switch_induced_failure_reasons
```

`episode_metrics.json` contains paper-level episode metrics:

```text
task_success
completion_time
number_of_switches
invalid_switch_attempts
switch_induced_failures
unsafe_gestures
near_falls
falls
collisions
pre_task_failures
task_failures
post_task_failures
```

## 4. Why Segments Matter

A video can be trimmed to show only the stable policy task window. Metrics should
still use the full raw run and separate three segments:

```text
pre_task   robot state before task_start / policy start
task       active naive sequence
post_task  after done / final stop_control
```

Falls in `pre_task` or `post_task` are reported, but they are not counted as
`switch_induced_failures`. This prevents natural unpowered falls before policy
startup or after final stop from being misattributed to a skill switch.

## 5. Thresholds

Default thresholds:

```text
near_fall_height: 0.55
torso_unstable_rad: 0.35
near_fall_torso_rad: 0.6
gesture_max_speed: 0.25
tracking_precondition_threshold: 1.5
tracking_failure_threshold: 2.0
pre_switch_window: 1.0
post_switch_window: 2.0
```

Override them from the CLI when running stress tests.
