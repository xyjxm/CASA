# Evaluation Protocol Table

| item | protocol |
|---|---|
| Controller | Frozen SONIC / GEAR-SONIC humanoid controller |
| Simulation | CASA Phase5 MuJoCo sim2sim/deploy lane |
| Task sequence | walk, turn, passive stop, gesture, walk, turn, passive wait, walk |
| Skill set | walk, turn, gesture, passive |
| Scene factors | randomized obstacles, user proxy, target bucket, runtime perturbation |
| Safety oracle | collision, near_collision, fall, human_distance_violation, unsafe_gesture, timeout |
| Calibration data | phase4_split == calibration only |
| Test data | online episodes or offline phase4_split == test; no threshold tuning on test |
| Fallback | same passive fallback contract as CASA main gate |
| Reporting rule | report rerun CASA metrics only, never copied source-paper numbers |
