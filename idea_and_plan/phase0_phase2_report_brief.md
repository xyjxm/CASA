# CASA Phase 0-2 汇报版总结

日期：2026-05-19

## 1. 一句话结论

当前 CASA Phase 0/1 已完成并通过稳定性验证；Phase 2 的 Safety Oracle、场景随机化、rollout logger 和批量采集闭环已经实现，可以支撑 Phase 3 的数据抽取与 safety critic 训练。严格验收层面，Phase 2 仍需要补充独立人工 review；在人工 review 前，已新增 Qwen3-VL 视频理解模型作为自动初审层。

当前状态：

```text
Phase 0: 通过
Phase 1: 通过
Phase 2: 工程闭环完成；正式验收需 VLM 初审 + 人工盲审
```

## 2. Phase 0/1 完成情况

Phase 0/1 的目标是把 SONIC + MuJoCo 的原始控制链路稳定封装为 CASA Skill API，同时不改 C++ deploy 主控制循环。

已完成：

- 封装 `walk / turn / gesture / passive` 四类 skill。
- 通过 ZMQ planner topic 驱动现有 SONIC deploy。
- 记录 `skill_events.csv/jsonl`，包括 skill 参数、开始/结束时间、执行状态和 evidence。
- 从 MuJoCo `sim_state.csv` 离线判断 skill 是否成功。

关键验证结果：

```text
repeatability: 80 / 80 success, success rate = 1.0
5min sanity: 76 skills, success rate = 0.947
30min sanity: 423 skills, success rate = 0.983
```

结论：Skill API 和执行日志链路已经稳定，可以作为后续 oracle labeling 的基础。

## 3. Phase 2 完成情况

Phase 2 的目标是在 skill invocation 基础上建立 safety oracle 与可复现数据闭环。

已完成：

- MuJoCo scene 中加入 CASA props：用户 proxy、障碍物 proxy、可视化人形和障碍物外观。
- `sim_state.csv` 扩展安全相关字段：
  - `min_user_distance`
  - `min_arm_user_distance`
  - `min_obstacle_distance`
  - `external_collision_user`
  - `external_collision_obstacle`
  - `control_loop_overrun`
  - `base_pos_z`
  - `torso_pitch / torso_roll`
- 实现 Safety Oracle，覆盖 6 类 violation：
  - `collision`
  - `near_collision`
  - `fall`
  - `human_distance_violation`
  - `unsafe_gesture`
  - `runtime_timeout`
- 统一 rollout 输出结构：

```text
episode_xxxxxx/
  sim_log/sim_state.csv
  skill_events.csv
  skill_events.jsonl
  violations.jsonl
  rollout_summary.json
```

批量采集结果：

```text
rollouts: 360
review rows: 360
unsafe skill labels: 2431
safe skill labels: 409
unverified: 17
```

violation 分布：

```text
runtime_timeout: 1444
collision: 18046
human_distance_violation: 125
unsafe_gesture: 104
fall: 121
near_collision: 380
```

结论：Phase 2 的自动 oracle eval set 已经能稳定生成多类型 violation，并形成 Phase 3 可用的数据基础。

## 4. Review 流程升级

最初的 `oracle_agreement_report.json` 使用 assisted review 自动填表，因此 `1.0 agreement` 只能说明 pipeline 可运行，不能证明 oracle 与真实人工判断一致。

现在的正式 review 流程调整为：

```text
Safety Oracle 自动标注
  -> Qwen3-VL 视频理解初审
  -> 人工盲审 / 复核
  -> oracle / VLM / human 三方一致性统计
```

这样做的原因是：oracle 可能因为日志瞬态触发 violation，但画面并不支持。之前旧 `fall.mp4` 就是例子：oracle 报 `fall`，但视频里机器人没有明显倒地。

## 5. Qwen3-VL 视频理解初审

已部署模型：

```text
Qwen/Qwen3-VL-2B-Instruct
```

本地位置：

```text
源码: /mnt/data/students/lph/Qwen3-VL
环境: /mnt/data/students/lph/.venvs/qwen3vl
模型: /mnt/data/students/lph/models/Qwen3-VL-2B-Instruct
```

测试视频：

```text
outputs/casa/phase2_visual_review_20260519/redo_fall_real/videos/fall_real.mp4
```

Qwen3-VL 输出：

```json
{
  "label": "unsafe",
  "violation_types": ["robot_falls"],
  "confidence": 0.95
}
```

这里的 `robot_falls` 后处理时归一化为 CASA 的 `fall`。

对应 oracle 结果：

```text
fall interval: 1.1s - 7.9s
duration: 6.8s
skill label: unsafe
```

结论：Qwen3-VL 可以作为视频初审和异常发现工具，尤其适合检查 `fall` 这类视觉上明显的事件。但它不能替代人工 review；目前 2B 模型对 prompt 和采样率仍有敏感性。

## 6. 可视化证据修正

已整理 Phase 2 可视化 review 包：

```text
outputs/casa/phase2_visual_review_20260519/
```

包含：

```text
safe_empty
human_close
near_obstacle
fall_real
runtime_timeout
```

重要修正：

- 旧 `videos/fall.mp4` 不再作为 fall 证据。
- 新增真实可见倒地样例 `fall_real.mp4`。
- `manual_visual_review_sheet.csv` 中 fall 行已替换为 `fall_real`。

`fall_real` 画面中机器人从站立变为失衡，随后横躺在地面上；oracle 和 Qwen3-VL 均判断为 unsafe / fall。

## 7. 当前风险与下一步

当前仍需注意：

- assisted review 不能作为正式人工验收结果。
- Qwen3-VL 初审需要固定 prompt、fps 和 violation type 映射规则。
- `near_collision` 仍是 bounding-sphere 近似，不是精确 SDF。
- `runtime_timeout` 更适合 log review，不能只靠视频模型判断。
- `control_loop_overrun` 已记录，但还需要进一步纳入 runtime oracle 统计。

下一步建议：

```text
1. 固定 VLM 初审 prompt 和输出 schema
2. 给 review sheet 增加 vlm_label / vlm_notes / human_label 等字段
3. 做人工盲审，不提前展示 oracle 和 VLM 结论
4. 统计 oracle / VLM / human 三方一致性
5. 进入 Phase 3 数据抽取与 mini critic 训练
```

## 8. 汇报结论

CASA Phase 0-2 已经从“能调用 skill”推进到“能批量生成带 safety oracle 标签的 rollout 数据”。Phase 2 的工程闭环已经可用，且新增 Qwen3-VL 视频理解初审后，人工 review 的证据链更完整。接下来重点不是继续堆更多自动标签，而是完成 VLM 初审 + 人工盲审 + 三方一致性统计，让 Phase 2 的验收更可信。
