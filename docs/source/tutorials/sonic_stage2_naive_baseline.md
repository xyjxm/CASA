# SONIC Stage 2 Naive Baseline

This runbook records the Stage 2 SONIC-Naive baseline: a fixed external task
sequencer drives the existing `zmq_manager` interface through
`walk_to_A -> stop -> face_user -> gesture -> walk_to_B`.

Stage 2 does not add SkillGuard, automatic safety gating, Isaac Lab integration,
or closed-loop navigation. The two "goals" are naive fixed-duration walk phases.

## 1. Create a Run ID

Run from the repository root:

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl
export RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/sonic_stage2/${RUN_ID}"
echo "${RUN_ID}"
```

Use the same `RUN_ID` in all terminals.

## 2. Terminal 1: MuJoCo Sim

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl
source .venv_sim/bin/activate

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
taskset -c 0-5 \
python -u gear_sonic/scripts/run_sim_loop.py \
  --no-enable-onscreen \
  --enable-offscreen \
  --enable-image-publish \
  --camera-port 5555 \
  --image-publish-fps 10 \
  --offscreen-camera-width 480 \
  --offscreen-camera-height 360 \
  --drop-on-start \
  --sim-frequency 200 \
  --fall-log-interval-seconds 30 \
  --fall-stats \
  --sim-timing-log-interval-seconds 5 \
  --episode-log-dir "outputs/sonic_stage2/${RUN_ID}/sim" \
  --episode-log-fps 50 \
  --episode-name stage2_naive
```

## 3. Terminal 2: C++ Deployment

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl/gear_sonic_deploy
source ../scripts/setup_no_root_env.sh

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 6-15 \
bash deploy.sh --input-type zmq_manager --output-type all --zmq-host localhost --zmq-port 5556 sim \
  --quiet \
  --timing-log-interval-seconds 10 \
  --command-publish-frequency 100 \
  --enable-csv-logs \
  --logs-dir "../outputs/sonic_stage2/${RUN_ID}/deploy" \
  --record-input-file "../outputs/sonic_stage2/${RUN_ID}/input.csv" \
  --target-motion-logfile "../outputs/sonic_stage2/${RUN_ID}/target_motion.csv" \
  --policy-input-logfile "../outputs/sonic_stage2/${RUN_ID}/policy_input.csv"
```

Confirm the deployment prompt, then wait until initialization is complete.

## 4. Terminal 3: OpenCV Viewer

This terminal is optional, but useful while tuning phase durations.

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl
source .venv_sim/bin/activate

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 16-18 \
python -u gear_sonic/scripts/run_camera_viewer.py \
  --camera-host localhost \
  --camera-port 5555 \
  --fps 10 \
  --max-display-width 480
```

## 5. Terminal 4: Naive Task Runner

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl
source .venv_sim/bin/activate

python -u gear_sonic/scripts/run_sonic_naive_baseline.py \
  --run-id "${RUN_ID}" \
  --output-dir "outputs/sonic_stage2/${RUN_ID}/task" \
  --zmq-host "*" \
  --zmq-port 5556 \
  --publish-fps 10 \
  --walk-a-seconds 6 \
  --stop-seconds 2 \
  --face-seconds 2 \
  --face-yaw-deg 0 \
  --gesture-seconds 5 \
  --walk-b-seconds 6 \
  --send-final-stop
```

Expected task outputs:

```text
outputs/sonic_stage2/<RUN_ID>/task/task_events.csv
outputs/sonic_stage2/<RUN_ID>/task/task_summary.json
```

## 6. Merge Logs

```bash
cd /mnt/data/students/lph/GR00T-WholeBodyControl
source .venv_sim/bin/activate

python -u gear_sonic/scripts/summarize_sonic_stage2_episode.py \
  --sim-log-dir "outputs/sonic_stage2/${RUN_ID}/sim" \
  --deploy-log-dir "outputs/sonic_stage2/${RUN_ID}/deploy" \
  --task-log-dir "outputs/sonic_stage2/${RUN_ID}/task" \
  --input-log "outputs/sonic_stage2/${RUN_ID}/input.csv" \
  --target-motion-log "outputs/sonic_stage2/${RUN_ID}/target_motion.csv" \
  --output-dir "outputs/sonic_stage2/${RUN_ID}"
```

Expected merged outputs:

```text
outputs/sonic_stage2/<RUN_ID>/episode.csv
outputs/sonic_stage2/<RUN_ID>/episode_summary.json
```

## 7. Acceptance Checks

`episode_summary.json` should report:

```text
baseline: SONIC-Naive
task_completed: true
has_walk_stop_gesture_walk: true
fall_count: present
self_collision_count: present
sim_deploy_time_ranges_overlap: true
goal_a_displacement_xy: numeric or null if sim coverage is missing
goal_b_displacement_xy: numeric or null if sim coverage is missing
```

The baseline is allowed to fail physically; Stage 2 records failures instead of
blocking or recovering from them.
