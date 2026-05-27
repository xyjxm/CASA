# Phase 2 Clean Visual 1000 实现报告

日期：2026-05-19

## 结论

已完成 Phase 2 从原始 360 条数据到 1000 条 clean visual-priority 数据集的补采、视频渲染、Qwen3-VL 重跑和最终构建。

最终主数据集位置：

`/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase2_clean_visual_1000_20260519/`

核心产物：

- `rollouts_clean.csv`：最终 1000 条 clean 样本。
- `oracle_vlm_comparison.csv`：最终 1000 条的 oracle/VLM/final bucket 对照。
- `excluded_rollouts.csv`：被排除样本及原因。
- `dataset_summary.json`：最终统计与完整性检查。
- `manual_review_priority_queues.csv`：人工 review 优先队列。

## 最终数据分布

目标配额已全部满足：

| bucket | 数量 |
|---|---:|
| `clean_safe` | 400 |
| `visual_collision_or_close` | 300 |
| `visual_near_boundary` | 150 |
| `visual_fall` | 150 |
| 合计 | 1000 |

最终标签分布：

| final label | 数量 |
|---|---:|
| safe | 400 |
| unsafe | 600 |

最终 violation type：

| type | 数量 |
|---|---:|
| safe | 400 |
| collision | 290 |
| human_distance_violation | 10 |
| near_collision | 140 |
| near_boundary | 10 |
| fall | 150 |

## 补采情况

旧 360 条没有删除，保留为 raw archive。最终 clean 构建使用了三批候选：

| 来源 | 候选数 | 说明 |
|---|---:|---|
| old360 | 360 | 原始 Phase 2 数据，重新用最终 prompt 跑 VLM |
| new candidates | 815 | safe/unsafe/fall/near 多 lane 补采 |
| lane5_collision_only | 56 | 为补足 collision/close 缺口追加补采 |

补采遵守 clean 规则：

- 未使用 `--inject-latency-ms`。
- sim 启动使用 `--drop-on-start`，避免弹性绳未释放 artifact。
- 新增 lane5 固定 `human_close` 场景，得到 56 条有效 collision/human-distance rollout。

## VLM 过程

Qwen3-VL 本地模型：

`/mnt/data/students/lph/models/Qwen3-VL-2B-Instruct`

三批候选全部离线渲染视频并跑 VLM：

| 来源 | VLM 条数 | reviewed | JSON ok |
|---|---:|---:|---:|
| old360 | 360 | 360 | 360 |
| new candidates | 815 | 815 | 815 |
| lane5_collision_only | 56 | 56 | 56 |

总计 VLM review：1231 条，JSON 可解析率 100%。

需要注意：Qwen3-VL-2B 对许多物理 collision/near 视频判为 safe，因此最终 builder 没有把 VLM 当作唯一硬门槛，而是把 VLM 作为“预审 + 人工 review 优先级”信号。VLM 与 oracle 不一致的样本在 `manual_review_priority_queues.csv` 中优先 review。

## 排除规则执行结果

最终排除 198 条：

| exclude_reason | 数量 |
|---|---:|
| runtime_artifact | 182 |
| short_video_artifact | 16 |

独立检查结果：

- `rollouts_clean.csv` 行数：1000。
- 每行 video/contact sheet/VLM JSON/source summary/source sim state 均存在。
- clean 集合中 `runtime_timeout`：0。
- clean 集合中 `injected_latency_ms > 0`：0。
- builder strict 模式通过，`missing_quota = {}`。

## 人工 Review 队列

`manual_review_priority_queues.csv` 共 1000 条，优先级分布：

| review_priority | 数量 |
|---|---:|
| `1_visual_fall` | 150 |
| `2_oracle_vlm_disagreement` | 413 |
| `3_oracle_vlm_type_overlap` | 25 |
| `4_visual_unsafe_type_check` | 12 |
| `5_clean_safe_spot_check` | 400 |

这一步的设计是：先让 VLM 对所有视频做一次机器预审，再把 fall、oracle/VLM 不一致、unsafe 类型校验推给人工快速 review。

## 代码变更

主要新增/修改：

- `gear_sonic/scripts/casa_build_clean_visual_dataset.py`
  - 新增 clean dataset builder。
  - 支持多 manifest、多 VLM comparison 输入。
  - 输出 clean/excluded/summary/manual queue/root-level VLM comparison。
  - 排除 runtime、latency、短视频、VLM runtime stall。
  - 支持配额感知分配，避免 collision+near/fall+collision 样本被单一 bucket 浪费。

- `gear_sonic/scripts/casa_render_rollout_videos.py`
  - 支持从输入 CSV 中读取 `summary_path`，保证新增 run 可以正确离线渲染。

- `gear_sonic/scripts/casa_run_oracle_eval_batch.py`
  - 增加定向 fall/collision 场景。
  - fall 场景可通过障碍物碰撞或初始扰动产生可见失衡。

- `/mnt/data/students/lph/Qwen3-VL/run_casa_video_review_batch.py`
  - 使用最终视频安全 review prompt。
  - evidence timestamps 限制最多 3 个。
  - 支持分片并行、aggregate-only、JSON salvage。

## 验收状态

本次实现已达到计划中的主目标：

- clean 主数据集 1000 条已生成。
- raw 360 保留。
- runtime/latency/短视频 artifact 已排除。
- 所有 clean 样本都有视频、contact sheet 和 VLM JSON。
- Qwen3-VL 已在人工 review 前完整跑过。
- 人工 review 队列已生成，可直接进入抽检。
