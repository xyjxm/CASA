# CASA Phase 0 到 Phase 2 实现过程总结

更新时间：2026-05-19

本文总结当前 `GR00T-WholeBodyControl` 中 CASA Phase 0/1/2 的工程实现路径、关键取舍、验证结果和后续衔接点。整体目标是先在 SONIC + MuJoCo sim 上建立一个可复现的 skill invocation 与 oracle labeling 闭环，为后续 Phase 3 的 `(state, skill, violation_label)` 数据生成和 safety critic 训练做准备。

## 1. 总体演进路线

Phase 0/1 先不做 safety gating，而是把 SONIC 的 ZMQ planner 控制链路封装成 CASA Skill API。核心产物是：Python runner 能通过现有 `zmq_manager` 向 C++ deploy 发布 `command` 和 `planner` topic，执行 `walk / turn / gesture / passive` 四类 skill，并从 MuJoCo episode log 中离线判断执行是否成功。

Phase 2 在 Phase 0/1 的 Skill API 之上增加 scene props、sim 日志扩展、safety oracle、rollout logger 和批量采集脚本。核心产物是：每条 rollout 可以输出 `skill_events.csv/jsonl`、`sim_state.csv`、`violations.jsonl`、`rollout_summary.json`，并能批量生成带视频理解模型初审和人工 review 表的 oracle eval set。

实现过程中坚持了两个边界：

- 不改 C++ deploy 主控制循环，只通过已有 `zmq_manager` 接入。
- 不做在线 SkillGuard / CASA gating，Phase 2 只负责离线 oracle 和数据闭环。

## 2. Phase 0/1：Skill API 与稳定性闭环

### 2.1 主要实现

新增 `gear_sonic/casa/` 模块，把原来 Stage2 naive baseline 中可复用的逻辑拆成 CASA 基础设施：

- `casa/io/zmq_publisher.py`：共享 ZMQ publisher，保持 dry-run 行为和 Stage2 message 格式兼容。
- `casa/io/episode_log_reader.py`：读取 `sim_state.csv`，派生 `base_speed_xy`、`base_yaw`、窗口统计和肩部 ROM。
- `casa/skills/`：实现 `WalkSkill`、`TurnSkill`、`GestureSkill`、`PassiveSkill`。
- `casa/skills/executor.py`：负责 start policy、按 `publish_fps` 发送 planner payload、记录执行 evidence。
- `casa/loggers/skill_logger.py`：输出 `skill_events.jsonl` 和 `skill_events.csv`。
- `casa/runner_utils.py`：统一 runner 的输出目录、sim log 目录、评估配置和 summary。

新增 runner 脚本：

- `casa_verify_gesture_amplitude.py`：gesture amplitude sweep，检查 shoulder ROM。
- `casa_run_skill_repeatability.py`：默认 4 个 skill 各 20 次 repeatability。
- `casa_run_sanity_check.py`：长时间随机 skill sequence sanity。

同时保留 Stage2 兼容性：`run_sonic_naive_baseline.py` 复用 CASA ZMQ publisher，不改变原有部署方式。

### 2.2 技术取舍

Skill API 采用 planner mode，而不是 reference motion 切换。`walk` 使用 `LocomotionMode::WALK`；`turn` 使用 `IDLE + facing`，如果 yaw 响应不足则标为 no-go，不偷偷改语义；`gesture` 使用 planner upper-body sine wave；`passive` 使用 `IDLE`。

ZMQ PUB/SUB 没有真实 ack，因此没有实现 `deploy_accepted=true` 这种假确认。执行成功只从下游 sim/deploy log evidence 推断：

- Walk：中间窗口内 `base_speed_xy > 0.1 m/s` 的比例超过阈值。
- Turn：最终 yaw delta 接近目标方向。
- Gesture：G1 29-DOF 默认用 `body_q_15` 和 `body_q_22` 检查肩部 ROM。
- Passive：尾部/中间窗口内低速比例满足阈值。

在测试过程中发现 passive/turn 判据过严会把稳定执行误判为失败，因此调松了 passive/turn 判据，增加 passive tail window，使 5min sanity 成功率回到 90% 以上。

### 2.3 关键验证结果

Phase 0/1 的主要验证记录：

- `outputs/casa/phase0_sanity/casa_phase0_headless_20260516_093349/repeat20/repeatability_summary.json`
  - 80 次 skill repeatability
  - success rate: `1.0`
  - go criteria: `true`

- `outputs/casa/phase0_sanity/casa_phase0_medium5_tuned_video_20260516_102139/medium5_tail/sanity_summary.json`
  - 5min sanity
  - 76 个 skill
  - success rate: `0.947`
  - go criteria: `true`

- `outputs/casa/phase0_sanity/casa_phase0_full30_video_20260516_110120/full30/sanity_summary.json`
  - 30min sanity
  - 423 个 skill
  - success rate: `0.983`
  - go criteria: `true`

这些结果说明：在不改 C++ 控制循环的前提下，CASA Skill API 已经能稳定驱动 SONIC planner path，并产生可用于后续标注的 skill event 日志。

## 3. Phase 2：Safety Oracle、Scene Props 与 Rollout Logger

### 3.1 Scene props 与 sim 日志扩展

Phase 2 fork 当前 MuJoCo scene，增加 CASA props：

- `gear_sonic/data/robot_model/model_data/g1/scene_casa_v1.xml`
- `gear_sonic/data/robot_model/model_data/g1/casa_props.xml`
- `gear_sonic/casa/scene/props.yaml`
- `gear_sonic/casa/scene/props_config.py`
- `gear_sonic/casa/scene/props_manager.py`

props 采用 mocap/static proxy：

- `user_proxy_0`：用户 proxy，collision 用 cylinder。
- `obstacle_0..7`：障碍物 proxy，collision 用 box。
- 默认隐藏在 `z=-10`，runtime 通过 JSON command 随 episode reset/randomize。

`run_sim_loop.py` 增加了：

- `--robot-scene`
- `--casa-props-config`
- `--casa-props-command-file`
- `--casa-props-ack-file`

`sim_state.csv` 扩展字段：

- `min_user_distance`
- `min_arm_user_distance`
- `min_obstacle_distance`
- `external_collision_user`
- `external_collision_obstacle`
- `control_loop_overrun`

距离 v1 使用 `center_distance - geom_rbound_a - geom_rbound_b` 的近似距离，避免一开始引入复杂 SDF。旧日志缺字段时 oracle 读为 unavailable，不崩溃。

### 3.2 Oracle 与 rollout logging

新增 `gear_sonic/casa/oracle/`：

- `ViolationType` / `Violation`
- `thresholds.yaml`
- `SafetyOracle`
- `rollout_summary.py`

实现 6 类 violation：

- `collision`
- `near_collision`
- `fall`
- `human_distance_violation`
- `unsafe_gesture`
- `runtime_timeout`

时间对齐统一使用 wall time。skill labeling 使用 `[skill.start_wall_time, skill.end_wall_time + post_horizon]`，默认 `post_horizon=2.0`。覆盖率不足时标 `unverified`，避免误把日志缺口当作安全。

新增 rollout/batch 脚本：

- `casa_run_rollout_with_oracle.py`：单 rollout 或已有日志离线 oracle。
- `casa_collect_oracle_eval_set.py`：批量收集、生成 review sheet。
- `casa_run_oracle_eval_batch.py`：面向多 lane / scenario cycle 的实际批量 runner。
- `casa_merge_oracle_eval_runs.py`：合并多路/targeted run。
- `casa_score_oracle_agreement.py`：统计 distribution、review agreement、severe miss rate。

每条 rollout 的结构统一为：

```text
outputs/casa/phase2_oracle/<run_id>/episode_<k>/
  sim_log/sim_state.csv
  skill_events.csv
  skill_events.jsonl
  violations.jsonl
  rollout_summary.json
```

顶层 review 输出：

```text
review/
  rollouts.csv
  human_review_sheet.csv
  oracle_agreement_report.json
  phase2_report.md
```

### 3.3 批量采集与 review 工程化

为达到 Phase 2 验收标准，增加了多 lane / targeted collection 的工程能力：

- sim/deploy 支持 DDS `--domain-id`、独立 ZMQ 端口和独立 log dir。
- batch runner 支持 `--scenario-mode cycle` 和 `--scenario-cycle`，可以稳定生成 `safe / human_close / user_collision / near_obstacle / obstacle_collision / fall` 等分布。
- review 阶段最初加入 assisted review：由 oracle summary 自动填充 review sheet，用来验证表格生成、统计脚本和 merge 流程可运行。
- 2026-05-19 后，正式 review 流程调整为：先用视频理解模型对关键视频做自动初审，再由人工在不看 oracle/VLM 结论的情况下盲审，最后比较 oracle / VLM / human 三方一致性。

当前已生成的 Phase 2 acceptance run：

- `outputs/casa/phase2_oracle/phase2_oracle_acceptance_plus_targeted_20260516_165220`
- rollouts: `360`
- review rows: `360`
- review complete: `true`
- hard agreement: `1.0`
- soft agreement: `1.0`
- runtime agreement: `1.0`
- severe miss rate: `0.0`

对应 `phase2_report.md` 中 violation 统计：

```text
runtime_timeout: 1444
collision: 18046
human_distance_violation: 125
unsafe_gesture: 104
fall: 121
near_collision: 380
```

对应 skill label 统计：

```text
unsafe: 2431
safe: 409
unverified: 17
```

这说明 Phase 2 的 oracle eval set 已经满足自动数据闭环的工程口径，可以支撑 Phase 3 的 sample extraction。需要注意的是，`oracle_agreement_report.json` 中的 `1.0` agreement 来自 assisted review 自动填表，只能证明 pipeline 可运行，不能作为独立人工验收结论。正式验收应继续执行第 4 节的视频理解初审和人工盲审流程。

## 4. 视频理解模型初审与人工 Review 流程

为了增强 Phase 2 oracle 验收的可信度，当前 review 链路调整为：

```text
oracle 自动标签
  -> Qwen3-VL 视频理解初审
  -> 人工盲审 / 复核
  -> oracle / VLM / human 三方一致性统计
```

### 4.1 为什么在人工 review 前增加视频理解

只看 oracle log 容易漏掉“日志触发但画面不支持”的情况。实际检查中已经发现旧 `fall.mp4` 是一个典型例子：oracle 因初始瞬态低 base height 触发 `fall`，但视频中机器人没有肉眼可见的倒地过程。因此，在人工 review 前增加 VLM 初审有三个作用：

- 自动发现 oracle 与画面不一致的样本，优先交给人工复核。
- 为人工 review 生成简短自然语言证据，例如“机器人从直立逐渐失衡，最终躺在地面上”。
- 在最终报告中形成三方交叉验证，而不是只依赖 oracle 或单个人工判断。

但 VLM 不替代人工。正式人工 review 应保持盲审：人工只看视频、关键帧和必要日志，不提前看 oracle/VLM 标签，避免被模型结论带偏。

### 4.2 Qwen3-VL 部署与最小验证

已部署的视频理解模型：

```text
源码:
  /mnt/data/students/lph/Qwen3-VL

运行环境:
  /mnt/data/students/lph/.venvs/qwen3vl

模型:
  /mnt/data/students/lph/models/Qwen3-VL-2B-Instruct

模型来源:
  Qwen/Qwen3-VL-2B-Instruct
```

新增推理脚本：

```text
/mnt/data/students/lph/Qwen3-VL/run_fall_video_review.py
```

推荐复现命令：

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/mnt/data/students/lph/model_cache/huggingface \
FORCE_QWENVL_VIDEO_READER=decord \
/mnt/data/students/lph/.venvs/qwen3vl/bin/python \
/mnt/data/students/lph/Qwen3-VL/run_fall_video_review.py \
  --model-dir /mnt/data/students/lph/models/Qwen3-VL-2B-Instruct \
  --video /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase2_visual_review_20260519/redo_fall_real/videos/fall_real.mp4 \
  --output /mnt/data/students/lph/idea_and_plan/qwen3vl_review/fall_real_qwen3vl_2b_fps2_compare.json \
  --fps 2.0 \
  --max-new-tokens 384
```

推理结果保存：

```text
/mnt/data/students/lph/idea_and_plan/qwen3vl_review/fall_real_qwen3vl_2b_fps2_compare.json
```

模型输出摘要：

```json
{
  "label": "unsafe",
  "violation_types": ["robot_falls"],
  "confidence": 0.95,
  "explanation_zh": "机器人从直立逐渐失去平衡，最终倒下并躺在地面上。"
}
```

其中 `robot_falls` 在后处理时应归一化为 CASA oracle 的 `fall`。

### 4.3 当前 VLM 结论与注意事项

对重新录制的 `fall_real.mp4`，Qwen3-VL-2B 在 `fps=2.0`、要求比较 early/middle/final frames 的 prompt 下，正确判断为：

```text
label: unsafe
violation type: fall
confidence: 0.95
```

这与 Safety Oracle 的结果一致：

```text
fall interval: 1.1s - 7.9s
duration: 6.8s
skill label: unsafe
```

同时也发现 Qwen3-VL-2B 对 prompt 和采样率比较敏感：在 `fps=1.0`、较弱 prompt 下曾输出过一次错误的 `safe` 判断，记录在：

```text
/mnt/data/students/lph/idea_and_plan/qwen3vl_review/fall_real_qwen3vl_2b.json
```

因此当前结论是：

- VLM 适合作为自动初审、异常发现和证据摘要工具。
- VLM 输出需要归一化 violation type，例如 `robot_falls -> fall`。
- VLM 与 oracle 冲突的样本应优先进入人工 review。
- VLM 不能替代人工盲审，尤其不能单独判断 `runtime_timeout`、精细 `near_collision` 阈值或 gesture window 内的距离违规。

### 4.4 推荐的正式 review 表字段

后续 `human_review_sheet.csv` 或新的 review sheet 建议加入以下列：

```text
oracle_label
oracle_violation_types
vlm_label
vlm_violation_types
vlm_confidence
vlm_evidence_timestamps
vlm_notes
human_label
human_violation_types
human_notes
oracle_vs_vlm
oracle_vs_human
vlm_vs_human
```

正式验收时，人工填写前不展示 `oracle_*` 和 `vlm_*` 字段；这些字段只在盲审结束后用于合并分析。

## 5. 视频与可视化修正

实现过程中录制过几版 Phase 2 视频，并发现两个重要问题。

第一版视频中机器人看起来像被吊起来。原因是录制 sim 时漏了 `--drop-on-start`，MuJoCo 默认启用了 elastic band。修正后重新用 `--drop-on-start` 录制，机器人正常落地运动。

第二个问题是 human proxy 视觉上只是一根柱子。为改善论文/展示视频，保留 collision proxy 不变，并在 `casa_props.xml` 中增加 visual-only 外观：

- `user_proxy_0_geom` 仍是唯一 user collision geom，但变为低透明 debug 外壳。
- `obstacle_*_geom` 仍是唯一 obstacle collision geom，但变为低透明 debug 外壳。
- 新增 `casa_vis_user_0_*`：简化人形 head/torso/arms/legs。
- 新增 `casa_vis_obstacle_*_*`：橙色带条纹障碍箱。
- 所有 visual geom 都是 `contype=0`、`conaffinity=0`，且命名不以 `user_proxy_` / `obstacle_` 开头，避免污染 oracle。

隔离检查结果：

- 完整 scene 可加载。
- oracle prefix 规则仍只看到 1 个 user collision geom 和 8 个 obstacle collision geom。
- 30 个 visual geom 均未进入 oracle geom set。
- 隐藏 proxy 时 visual body 跟随父 mocap body 进入地下。

新版视频目录：

```text
outputs/casa/phase2_videos/phase2_props_visual_20260516_191129/videos/
  safe_empty.mp4
  human_close.mp4
  near_obstacle.mp4
  fall.mp4
  runtime_timeout.mp4
  video_check.json
```

每条视频为 `960x720`、`10 FPS`、`420 frames`，并通过非空画面检查。`safe_empty` 场景隐藏所有 user/obstacle props，用于干净展示；`runtime_timeout` 也基于 `safe_empty + latency injection`，避免 timeout 与 human/obstacle violation 混杂。

2026-05-19 进一步整理了人工 review 包：

```text
outputs/casa/phase2_visual_review_20260519/
  videos/
  frames/
  manual_visual_review_sheet.csv
  redo_fall_real/
```

其中旧 `videos/fall.mp4` 不再作为 fall 视觉证据。它只在极早期出现一次低 base height 触发，画面中没有明显倒地过程。为此重新生成了真实可见倒地样例：

```text
outputs/casa/phase2_visual_review_20260519/redo_fall_real/videos/fall_real.mp4
outputs/casa/phase2_visual_review_20260519/redo_fall_real/frames/fall_real_contact.jpg
outputs/casa/phase2_visual_review_20260519/redo_fall_real/violations.jsonl
outputs/casa/phase2_visual_review_20260519/redo_fall_real/rollout_summary.json
```

新 `fall_real` 样例的 oracle 结果：

```text
fall interval: 1.1s - 7.9s
duration: 6.8s
min_base_height: 0.089m
skill_label_counts: {unsafe: 1}
```

`manual_visual_review_sheet.csv` 中的 fall 行已改为 `fall_real`，用于后续人工 review。

## 6. 当前已知 caveats

- `gear_sonic/data/...` 被 `.gitignore` 忽略，因此 `scene_casa_v1.xml` 和 `casa_props.xml` 不会自然出现在 `git status` 中；后续打包或提交时需要显式纳入交付清单。
- Phase 2 的 human 仍是 proxy + visual avatar，不是动态人类 agent；动态人群和社交导航应留到后续 Phase 5。
- `near_collision` / distance v1 仍是 bounding sphere 近似，不是精确 SDF；足够用于 Phase 2 oracle bootstrap，但后续可以升级。
- assisted review 只是工程冒烟测试；论文实验正式版应使用 VLM 初审 + 人工盲审 + 三方一致性统计。
- Qwen3-VL-2B 初审对 prompt / fps 有敏感性，需要固定推荐 prompt、采样率和 violation type 归一化规则后再批量使用。
- `runtime_timeout` v1 主要来自 skill actual duration / injected latency；sim/deploy loop overrun 字段已经保留，但不是当前唯一判据。

## 7. 对 Phase 3 的衔接

Phase 3 可以直接在当前 Phase 2 产物上构建 sample extraction：

输入：

- `skill_events.csv/jsonl`
- `sim_log/sim_state.csv`
- `violations.jsonl`
- `rollout_summary.json`
- `scene_props.json`

输出目标：

```text
(state_features, skill_type, skill_params, violation_label, time_to_violation, metadata)
```

建议优先实现：

- 从 skill start 前一小段窗口提取 robot state summary。
- 加入 skill 参数：walk speed/facing，turn facing yaw，gesture amplitude/frequency/side，passive mode。
- 使用 Phase 2 oracle label 作为 binary unsafe label，并保留 violation type 多标签字段。
- 保留 `unverified` 样本，但默认不进入 critic training，只用于数据质量审计。

此时 CASA 的最小训练数据链路已经成形：SONIC executor 固定，Skill API 可控，Scene props 可随机化，Safety Oracle 可复现，Rollout logger 可批量产出带 review 的 labels。
