# Phase 2 全量视频与 Qwen3-VL 预审记录

日期：2026-05-19

## 做了什么

这次把 Phase 2 验收集合里的 360 条 rollout 全部转成了可人工复查的视频，并在人工 review 之前用 Qwen3-VL-2B-Instruct 做了一轮视频理解预审。

关键点：

- 采用离线重放，不重新运行策略：从每条 rollout 的 `sim_state.csv` 和 `scene_props` 还原机器人、用户代理、障碍物位置。
- 每条 rollout 对应一个 MP4，同时生成一张 contact sheet，方便快速扫帧。
- 视频画面只叠加 rollout id 和时间戳，不叠加 oracle 标签，避免 VLM 被标签文字提示。
- Qwen3-VL 只作为“人工 review 前的筛查器”，不替代 Phase 2 的数值 oracle。

## 输出位置

全量视频包：

`/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase2_full_video_review_20260519/`

主要文件：

- `manifest.csv`：360 条 rollout 到视频/contact sheet 的映射。
- `videos/*.mp4`：360 个视频，5 FPS，640x480。
- `frames/*_contact.jpg`：360 张 contact sheet。
- `qwen3vl_review/vlm_results/*.json`：360 个 Qwen3-VL 原始结果。
- `qwen3vl_review/oracle_vlm_comparison.csv`：oracle 标签和 VLM 结果逐条对比。
- `qwen3vl_review/oracle_vlm_comparison.summary.json`：聚合统计。
- `qwen3vl_review/manual_review_priority_queues.csv`：建议人工 review 优先队列。

脚本：

- `/mnt/data/students/lph/GR00T-WholeBodyControl/gear_sonic/scripts/casa_render_rollout_videos.py`
- `/mnt/data/students/lph/Qwen3-VL/run_casa_video_review_batch.py`

## 数据量

- rollout 数：360
- 视频数：360
- contact sheet 数：360
- VLM JSON 结果数：360
- 视频包大小：约 561 MB
- Qwen review 输出大小：约 2.0 MB

Oracle 标签分布：

- safe：1
- unsafe：359
- collision：187
- near_collision：61
- fall：59
- runtime_timeout：52

## Qwen3-VL 结果

模型：`Qwen/Qwen3-VL-2B-Instruct`

本地模型目录：

`/mnt/data/students/lph/models/Qwen3-VL-2B-Instruct`

运行方式：

- 输入视频采样：1 FPS
- 三张 GPU 分 3 个 shard 并行推理
- 输出 schema：`label`、`violation_types`、`confidence`、`evidence_timestamps`、`explanation_zh`

VLM 标签分布：

- safe：281
- unsafe：79

与 oracle 的粗粒度 unsafe/safe 一致数：

- label_match=True：80
- label_match=False：280

类型交集：

- type_overlap=True：39
- type_overlap=False：321

## 如何理解这个结果

这个结果不说明 oracle 错了，也不说明 VLM 可以替代 oracle。更合理的解释是：

- `runtime_timeout` 很多时候不是纯视觉可判定事件，应该优先看日志。
- `near_collision` 和短暂接触是阈值型事件，1 FPS 视频和固定相机视角容易漏掉。
- 一些 `fall` 标签来自数值状态或局部瞬态，视频里未必是明显“整机倒地”。
- VLM 判成 `unsafe` 的 79 条更适合优先人工看，因为它们通常有可见的碰撞、近距离风险或姿态异常。

## 建议人工 review 顺序

优先看：

1. `manual_review_priority_queues.csv` 中 `1_vlm_visible_issue_type_overlap`：VLM 和 oracle 类型有交集，共 38 条。
2. `2_vlm_visible_issue_type_disagree`：VLM 看到 unsafe，但类型和 oracle 不一致，共 41 条。
3. `3_oracle_fall_vlm_safe_review_video`：oracle 是 fall 但 VLM 看成 safe，共 59 条。
4. `4_oracle_collision_vlm_safe_review_video` 和 `4_oracle_near_collision_vlm_safe_review_video`：阈值型碰撞/近碰撞漏检队列。
5. `5_runtime_timeout_check_logs_first`：优先查日志，再决定是否看视频。

结论：全量视频和 VLM 预审已经完成；后续人工 review 可以从优先队列开始，而不是盲扫 360 条。
