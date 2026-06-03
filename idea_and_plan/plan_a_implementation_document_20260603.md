# CASA Plan A 实现文档

Last updated: 2026-06-03

## 1. 摘要

Plan A 是 CASA 的最小可发表版本，目标是在不重新训练 humanoid controller 的前提下，为 SONIC 增加一层 **skill invocation safety gate**。
核心问题不是“机器人怎么执行动作”，而是“当前状态下这个技能该不该被调用”。

实现范围：

- 主执行器：SONIC / GEAR-SONIC
- 主仿真：MuJoCo sim2sim / deploy 链路
- 技能范围：`walk`、`turn`、`gesture`、`passive`
- 主方法：CASA-A per-skill conformal gate
- 主对照：SONIC-only、Hard Contract、Raw Critic、Global Conformal、CASA-A
- 不包含：stair、hazard head、mid-execution interrupt、OOD ensemble、LLM baseline、safe RL baseline、真实机器人部署

当前实现结论：

- Plan A Phase 0-5 工程闭环已完成。
- Phase 2 人工 review 不计入自动完成口径。
- 默认自动化工程 audit：`PASS_STRICT_ONLINE` / `GO`
- 原始最强 claim audit：`STRICT_PLAN_A_NO_GO`，原因是 CASA-A 尚未严格优于 Global Conformal，且 reject/fallback budget 偏高。

## 2. 实验任务 Setting

Plan A 的实际任务是中等长度 humanoid 交互技能序列，不是抓取、搬运或 stair crossing。

实际 Phase5 每个 episode 包含 8 个连续技能调用：

```text
walk
turn
passive stop
gesture
walk
turn
passive wait
walk
```

实验场景包含随机障碍物、用户 proxy、不同 target bucket 和 runtime perturbation。
`task_success` 的实际定义是：episode `completed` 且没有 unsafe skill invocation。fallback 会单独统计，但 fallback 本身不直接等于 task failure。

## 3. 系统架构

```text
Task Phase Manager
  -> Candidate Skill Generator
  -> Hard Contract Filter
  -> Raw CASA Safety Critic
  -> Per-skill Conformal Threshold
  -> allow / reject
  -> SONIC Skill Wrapper or Passive fallback
  -> MuJoCo Simulation
  -> Safety Oracle / Logger
```

主要接口与模块：

- `WalkSkill(vx, vy, facing_yaw_deg, duration)`
- `TurnSkill(face_yaw_deg, duration)`
- `GestureSkill(amplitude, frequency, side, duration)`
- `PassiveSkill(duration, mode=stop|wait)`
- Safety Oracle labels：`collision`、`near_collision`、`fall`、`human_distance_violation`、`unsafe_gesture`、`runtime_timeout`
- Phase5 methods：`sonic_only`、`hard_contract`、`raw_critic_0p5`、`global_conformal`、`casa_a_per_skill`

## 4. Phase 实现状态

| Phase | 目标 | 实现状态 |
|---|---|---|
| Phase 0 | SONIC + MuJoCo sanity check | 完成 |
| Phase 1 | 4 个 SONIC Skill API wrapper | 完成 |
| Phase 2 | Safety Oracle + Logger | 主体完成；人工 review 不计入自动完成 |
| Phase 3 | Hybrid feasibility dataset + mini critic | 完成 |
| Phase 4 | ID Dataset v1 + Raw Critic | 完成 |
| Phase 5 | Per-skill conformal + 5 baseline online experiment | 完成 |

## 5. Phase5 主结果

数据规模：

- methods：5
- seeds：`1234, 1235, 1236, 1237, 1238`
- episodes：每个 method 500，总计 2500
- gate decisions：20000
- completed episodes：2500

五方法结果：

| method | unsafe | unsafe reduction vs SONIC | safe completion | fallback/episode | reject/decision |
|---|---:|---:|---:|---:|---:|
| SONIC-only | 3559 | 0.0000 | 0.0260 | 0.0000 | 0.0000 |
| Hard Contract | 3643 | -0.0236 | 0.0220 | 6.2220 | 0.7778 |
| Raw Critic 0.5 | 1223 | 0.6564 | 0.4580 | 3.8660 | 0.4833 |
| Global Conformal | 847 | 0.7620 | 0.6560 | 4.3040 | 0.5380 |
| CASA-A per-skill | 934 | 0.7376 | 0.6100 | 5.2120 | 0.6515 |

解释：

- CASA-A 相比 SONIC-only 达到大幅 unsafe reduction。
- CASA-A task success/safe completion 明显高于 SONIC-only。
- 但 Global Conformal 在当前在线主结果中 unsafe 更低、safe completion 更高，因此原始强 claim 不能说 CASA-A 严格优于 Global Conformal。

## 6. Audit 与 Claim 状态

默认自动化工程验收：

```text
status = PASS_STRICT_ONLINE
go = True
```

严格原始 Plan A claim audit：

```text
strict Plan A claim = STRICT_PLAN_A_NO_GO
```

主要 blocker：

- `casa_a_per_skill_does_not_outperform_global_conformal`
- `casa_fallback_rate_exceeds_strict_plan_a_budget`
- `casa_reject_rate_exceeds_strict_plan_a_budget`
- `casa_walk_reject_rate_exceeds_strict_plan_a_budget`

因此论文表述应分开：

```text
Plan A automated engineering audit: PASS.
Original strongest Plan A claim: not supported yet.
Strict claim audit correctly reports NO_GO.
```

## 7. 主要产物路径

原始计划：

```text
/mnt/data/students/lph/idea_and_plan/plan_a.md
```

Plan A completion artifact：

```text
/tmp/casa_upload_worktree/idea_and_plan/plan_a_completion_20260602
```

关键 audit 文件：

```text
/tmp/casa_upload_worktree/idea_and_plan/plan_a_completion_20260602/plan_a_strict_completion_audit_20260602.json
/tmp/casa_upload_worktree/idea_and_plan/plan_a_completion_20260602/audit/online_acceptance_audit.json
/tmp/casa_upload_worktree/idea_and_plan/plan_a_completion_20260602/audit/online_go_no_go.json
/tmp/casa_upload_worktree/idea_and_plan/plan_a_completion_20260602/audit/online_report.md
```

Phase5 online root：

```text
/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase5_conformal_baselines_20260522/online_real_main_hc_threshold_2500_20260602
```

视频整理目录：

```text
/mnt/data/students/lph/recording
```

推荐展示视频：

```text
/mnt/data/students/lph/recording/upright_motion
```

## 8. GitHub 与版本

CASA repository：

```text
git@github.com:xyjxm/CASA.git
```

关键提交：

```text
5e74d1b Add Plan A completion evidence
407ad3e Add strict Plan A claim audit
ab9fb06 Add Phase 5 online visualization evidence
```

关键 tag：

```text
plan-a-completion-20260602-strict-claim-audit
```

注意：不要上传到 `NVlabs/GR00T-WholeBodyControl`；应使用 CASA 仓库。

## 9. 验证与测试

已验证内容：

- Phase0-5 strict completion audit
- Phase5 online acceptance audit
- artifact hash consistency
- method/seed/episode count consistency
- metric recomputation consistency
- visualization metric consistency
- simulator video可读性检查
- upright-motion 视频筛选：`base_z >= 0.72`、`fall_flag=0`、torso roll/pitch 小

代表性测试结果：

```text
42 passed
ruff targeted checks passed
git diff --check passed
py_compile passed
```

已知限制：

- 全仓库 `ruff check gear_sonic` 存在大量历史遗留问题，非本次新增文件导致。
- Phase2 人工 review 不属于自动 completion audit。
- 当前 CASA-A 不能宣称严格优于 Global Conformal。
- metric timeline 视频不是仿真视频；真实站立运动视频应看 `recording/upright_motion`。

## 10. 后续建议

Plan A 当前适合支撑的表述：

```text
CASA-A provides an implemented calibrated skill-invocation gate
that substantially reduces unsafe invocation versus SONIC-only,
while preserving a usable safe-completion rate.
```

暂不应强 claim：

```text
CASA-A strictly outperforms Global Conformal under the strongest original Plan A criterion.
```

如果继续增强：

- Plan B：加入 hazard head 和 mid-execution interrupt。
- Plan C：加入 stair、K=20 long-horizon、OOD ensemble、LLM baseline、safe RL baseline。
