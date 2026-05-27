# Phase 2 v2 修改计划：Clean Natural Visual Dataset

日期：2026-05-19

## 背景

当前 Phase 2 v1 已完成 1000 条 clean visual-priority 数据集，但人工 review 发现三个关键质量问题：

1. 部分视频首帧就有人或障碍物贴在机器人身上，属于 `initial_overlap / initial_contact artifact`。
2. 部分视频仍疑似弹性绳未释放，机器人运动被约束。
3. 场景复杂度不足，很多样本只有单个人或单个障碍物，不够接近真实环境。

因此 Phase 2 需要升级为 **Phase 2 v2 clean natural visual dataset**。

## 目标

重新构建一个 1000 条的 Phase 2 v2 主数据集：

- 保留 v1 原始数据和打包结果作为 archive，不删除。
- v2 主集只收自然 rollout 中发生的可解释视觉事件。
- 排除初始重叠、首帧接触、弹性绳未释放、runtime/latency/短视频 artifact。
- 增加场景复杂度，让数据覆盖 simple / medium / hard 三档环境。
- 重新渲染全部视频，重新跑 Qwen3-VL，再生成新的人工 review 队列和 zip 包。

## v2 数据配额

总体仍保持 1000 条，沿用 v1 的 safety bucket 配额，便于和 v1 对比：

| bucket | 数量 |
|---|---:|
| `clean_safe` | 400 |
| `visual_collision_or_close` | 300 |
| `visual_near_boundary` | 150 |
| `visual_fall` | 150 |
| 合计 | 1000 |

新增复杂度配额：

| scene_complexity | 数量 |
|---|---:|
| `simple` | 250 |
| `medium` | 500 |
| `hard` | 250 |

复杂度定义：

- `simple`: 0-1 个 user，0-1 个 obstacle，用于保留清晰因果样本。
- `medium`: 1 个 user，2-4 个 obstacles，形成普通室内复杂度。
- `hard`: 2-3 个 users，4-8 个 obstacles，包含窄通道、遮挡、绕行失败、多目标干扰。

## 必须修复的问题

### 1. 排除首帧重叠和初始接触

v2 不再接受以下样本：

- `external_collision_user == 1` 出现在首帧或 `sim_time <= 0.1s`。
- `external_collision_obstacle == 1` 出现在首帧或 `sim_time <= 0.1s`。
- 首帧 `min_user_distance == 0`。
- 首帧 `min_obstacle_distance == 0`。
- violation 在 `sim_time <= 0.5s` 发生，且原因来自初始化摆放而非机器人运动。

建议 hard filter：

- `event_start_time_s >= 1.0`
- `initial_min_user_distance >= 0.35m`
- `initial_min_obstacle_distance >= 0.25m`
- 首帧外部碰撞必须为 0。

被排除样本写入：

`excluded_rollouts.csv`

排除原因：

`initial_overlap_artifact`

### 2. 确认弹性绳释放

v2 必须在 sim log 中记录弹性绳状态。

代码修改：

- 在 `sim_state.csv` 中增加字段：
  - `elastic_band_enabled`
  - `elastic_band_force_norm`
- 在 `rollout_summary.json` 中增加：
  - `elastic_band_enabled_any`
  - `elastic_band_enabled_first`
  - `elastic_band_enabled_ratio`
  - `max_elastic_band_force_norm`

接收规则：

- clean 主集要求 `elastic_band_enabled_any == false`
- 或者至少要求 `elastic_band_enabled_ratio == 0`
- 若字段缺失，则不进入 v2 主集，放入 `legacy_no_elastic_proof_quarantine`

重要策略：

- v2 默认不使用旧 360 中缺少弹性绳证明的样本。
- 如果旧样本没有 `elastic_band_enabled` 字段，即使视频看起来可用，也只作为 archive，不进 v2 主集。

### 3. 增加场景复杂度

新增/扩展场景族：

safe：

- `safe_empty`
- `safe_single_user`
- `safe_single_obstacle`
- `safe_medium_room`
- `safe_hard_clutter`

near boundary：

- `near_user_passby`
- `near_obstacle_passby`
- `narrow_gap_user_obstacle`
- `multi_obstacle_near_path`
- `crowded_near_boundary`

collision/close：

- `moving_into_user`
- `moving_into_obstacle`
- `late_user_close`
- `late_obstacle_collision`
- `narrow_gap_collision`
- `cluttered_collision`

fall：

- `late_obstacle_trip`
- `narrow_gap_trip`
- `post_collision_fall`
- `cluttered_fall`

核心原则：

- 人和障碍物不能一开始贴在机器人身体上。
- collision/fall 要尽量由机器人 rollout 过程触发。
- fall 可以通过障碍物绊倒、窄通道接触后失衡、轻微初始扰动触发，但不能用纯合成视频进入主集。

## 实现改动

### A. sim/logging 改动

修改：

- `gear_sonic/utils/mujoco_sim/base_sim.py`

新增日志字段：

- `elastic_band_enabled`
- `elastic_band_force_norm`
- `initial_min_user_distance`
- `initial_min_obstacle_distance`
- `initial_external_collision_user`
- `initial_external_collision_obstacle`

### B. 场景生成改动

修改：

- `gear_sonic/scripts/casa_run_oracle_eval_batch.py`
- `gear_sonic/casa/scene/props.yaml`
- 如有必要，扩展 `casa_props.xml` 支持更多 user proxies。

新增能力：

- 支持 `scene_complexity`
- 支持 1-3 users
- 支持 0-8 obstacles
- 支持最小初始 clearance 检查
- 采样失败时自动重采场景布局

建议新增参数：

```bash
--scene-complexity simple|medium|hard
--min-initial-user-distance 0.35
--min-initial-obstacle-distance 0.25
--min-event-start-time 1.0
--max-placement-retries 50
```

### C. clean builder v2

新增或扩展：

- `gear_sonic/scripts/casa_build_clean_visual_dataset_v2.py`

输入：

- 新 v2 rollouts
- render manifest
- Qwen3-VL comparison CSV

输出：

- `rollouts_clean_v2.csv`
- `excluded_rollouts_v2.csv`
- `dataset_summary_v2.json`
- `oracle_vlm_comparison_v2.csv`
- `manual_review_priority_queues_v2.csv`

新增字段：

- `scene_complexity`
- `num_users`
- `num_obstacles`
- `initial_min_user_distance`
- `initial_min_obstacle_distance`
- `event_start_time_s`
- `elastic_band_enabled_any`
- `elastic_band_enabled_ratio`
- `artifact_flags`

新增排除原因：

- `initial_overlap_artifact`
- `early_event_artifact`
- `elastic_band_not_released`
- `legacy_no_elastic_proof_quarantine`
- `runtime_artifact`
- `latency_artifact`
- `short_video_artifact`
- `oracle_only_invisible_quarantine`

## 采集策略

### Step 1：pilot

先采每类少量样本：

| complexity | 每 bucket pilot 数 |
|---|---:|
| simple | 10 |
| medium | 10 |
| hard | 10 |

pilot 验收：

- 首帧无碰撞。
- 弹性绳字段存在且为 false。
- 事件不是一开始就发生。
- 视频中能看清 user/obstacle/robot 关系。

### Step 2：正式补采

通过多 lane 并行补采：

- safe lane
- near lane
- collision/close lane
- fall lane
- hard clutter lane

所有采集命令要求：

- 不传 `--inject-latency-ms`
- sim 固定 `--drop-on-start`
- 使用 v2 scene generator
- 打开 elastic band 状态日志

### Step 3：视频和 VLM

对所有候选：

- 离线渲染 MP4
- 生成 contact sheet
- 使用 Qwen3-VL 重新 review
- `fps=2.0`
- evidence timestamps 最多 3 个

VLM 不作为唯一接收标准，但用于：

- 发现视频不可解释样本
- 排人工 review 优先级
- 标记 oracle/VLM disagreement

## 验收标准

数据完整性：

- `rollouts_clean_v2.csv` 行数 = 1000。
- 每行都有 video、contact sheet、VLM JSON。
- 每行都有 source summary 和 sim state。

清洁性：

- `runtime_timeout == 0`
- `injected_latency_ms > 0` 数量 = 0
- `elastic_band_enabled_ratio > 0` 数量 = 0
- `initial_overlap_artifact` 数量 = 0
- `event_start_time_s < 1.0` 的 unsafe 样本数 = 0，除非明确标记为非接触类早期安全状态。

分布：

- safety bucket 达到 400/300/150/150。
- scene complexity 达到 250/500/250。
- medium/hard 中必须包含多个 user/obstacle。

VLM：

- Qwen3-VL result 数 = 候选视频数。
- JSON 可解析率 > 95%。
- 所有 fall 样本进入人工优先 review。
- oracle/VLM disagreement 进入人工优先 review。

人工抽检：

- 每个 bucket 随机抽 20 条。
- 每个 complexity 随机抽 20 条。
- 所有 `visual_fall` 抽检优先级最高。
- 所有 `oracle_contact_sheet_pending_review` 抽检优先级高。

## 输出目录

建议新建：

`/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase2_clean_visual_1000_v2_20260519/`

最终下载包：

`phase2_clean_visual_1000_v2_videos_labels_20260519.zip`

## 成功定义

Phase 2 v2 完成后，应满足：

- 人工打开随机视频时，不再看到人或障碍物首帧贴在机器人身上。
- 不再看到弹性绳拉住机器人。
- 场景中有足够的 user/obstacle 多样性。
- unsafe 事件主要发生在 rollout 过程中，而不是初始化瞬间。
- 所有样本都有结构化证据证明其不是 runtime、latency、elastic-band、initial-overlap artifact。
