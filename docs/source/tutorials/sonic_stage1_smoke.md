# SONIC Stage 1 Smoke Test

This runbook records the Stage 1 baseline check for CASA work:
run the original SONIC / GEAR-SONIC deployment stack in MuJoCo, manually switch
motions, and produce one merged episode log.

Stage 1 intentionally does not add CASA gating, automatic task scripts, safety
gating, or Isaac Lab integration.

## 1. Create a Run ID

Run from the repository root:

```bash
cd /path/to/CASA
export RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/sonic_stage1/${RUN_ID}"
echo "${RUN_ID}"
```

Use the same `RUN_ID` value in all three terminals. If you open a fresh
terminal, run `export RUN_ID=<the printed value>` before launching commands.

## 2. Terminal 1: MuJoCo Sim

```bash
cd /path/to/CASA
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
  --episode-log-dir "outputs/sonic_stage1/${RUN_ID}/sim" \
  --episode-log-fps 50 \
  --episode-name stage1_smoke
```

Expected sim outputs:

```text
outputs/sonic_stage1/<RUN_ID>/sim/sim_state.csv
outputs/sonic_stage1/<RUN_ID>/sim/episode_summary.json
```

## 3. Terminal 2: C++ Deployment

```bash
cd /path/to/CASA/gear_sonic_deploy
source ../scripts/setup_no_root_env.sh

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 6-15 \
bash deploy.sh --input-type keyboard sim \
  --quiet \
  --timing-log-interval-seconds 10 \
  --command-publish-frequency 100 \
  --enable-csv-logs \
  --logs-dir "../outputs/sonic_stage1/${RUN_ID}/deploy" \
  --record-input-file "../outputs/sonic_stage1/${RUN_ID}/input.csv" \
  --target-motion-logfile "../outputs/sonic_stage1/${RUN_ID}/target_motion.csv" \
  --policy-input-logfile "../outputs/sonic_stage1/${RUN_ID}/policy_input.csv"
```

Expected deploy outputs:

```text
outputs/sonic_stage1/<RUN_ID>/deploy/q.csv
outputs/sonic_stage1/<RUN_ID>/deploy/dq.csv
outputs/sonic_stage1/<RUN_ID>/deploy/action.csv
outputs/sonic_stage1/<RUN_ID>/deploy/base_quat.csv
outputs/sonic_stage1/<RUN_ID>/deploy/torso_quat.csv
outputs/sonic_stage1/<RUN_ID>/deploy/motion_name.csv
outputs/sonic_stage1/<RUN_ID>/deploy/motion_playing.csv
outputs/sonic_stage1/<RUN_ID>/input.csv
outputs/sonic_stage1/<RUN_ID>/target_motion.csv
```

## 4. Terminal 3: OpenCV Viewer

```bash
cd /path/to/CASA
source .venv_sim/bin/activate

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 16-18 \
python -u gear_sonic/scripts/run_camera_viewer.py \
  --camera-host localhost \
  --camera-port 5555 \
  --fps 10 \
  --max-display-width 480
```

## 5. Keyboard Smoke Sequence

After deployment initialization:

```text
]      start policy
N/P    select walking_quip_360_R_002__A428 or another walking motion
T      play walking motion
wait   let the motion pause/settle; this is the Stage 1 stop interval
N/P    select macarena_001__A545 or dance_in_da_party_001__A464
T      play gesture motion
N/P    select a walking motion again
T      play walking motion
O      stop deployment after logs are written
```

The deploy terminal prints the selected and played motion names for `N/P/T/R`,
which is the manual confirmation that at least two motions were switched.

## 6. Merge Logs

Run from the repository root:

```bash
cd /path/to/CASA
source .venv_sim/bin/activate

python -u gear_sonic/scripts/summarize_sonic_stage1_episode.py \
  --sim-log-dir "outputs/sonic_stage1/${RUN_ID}/sim" \
  --deploy-log-dir "outputs/sonic_stage1/${RUN_ID}/deploy" \
  --input-log "outputs/sonic_stage1/${RUN_ID}/input.csv" \
  --target-motion-log "outputs/sonic_stage1/${RUN_ID}/target_motion.csv" \
  --output-dir "outputs/sonic_stage1/${RUN_ID}"
```

Expected merged outputs:

```text
outputs/sonic_stage1/<RUN_ID>/episode.csv
outputs/sonic_stage1/<RUN_ID>/episode_summary.json
```

## 7. Acceptance Checks

`episode_summary.json` should report:

```text
has_walk_stop_gesture_walk: true
unique_motion_count: at least 2
fall_count: present
self_collision_count: present
sim_deploy_time_ranges_overlap: true
tracking_error_available: true when target_motion.csv exists
```

If `tracking_error_available` is false, check that `--target-motion-logfile`
was passed to `deploy.sh` and that the deployment reached CONTROL state.
