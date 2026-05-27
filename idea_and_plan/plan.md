
---

## 0. 已确定决策

本计划基于以下前提：

```text
主执行器：SONIC
主仿真链路：MuJoCo sim2sim / deploy 链路
研究重点：安全技能调用，而不是底层动作控制
第一版任务：SONIC skill invocation safety
数据路线：Hybrid
    主训练数据：随机化初始状态 + 单 branch rollout
    小规模分析数据：strict counterfactual subset
```

第一版不做：

```text
不重新搭 Isaac Lab inference
不重训 SONIC stair policy
不复现完整 Switch
不从零训练 humanoid controller
不做复杂 handover
```

---

# 1. 第一版系统边界

整体结构：

```text
Task Phase Manager
        ↓
Candidate Skill Generator
        ↓
CASA Safety Gate
        ↓
SONIC Skill Wrapper
        ↓
MuJoCo Simulation
        ↓
Safety Oracle / Logger
```

模块定位：

```text
SONIC:
    执行 walk / passive / turn / gesture / stair-placeholder。

MuJoCo:
    负责主仿真、rollout、oracle 标签和 episode evaluation。

CASA:
    在每次 skill invocation 前判断 allow / reject / fallback。

Isaac Lab:
    第一版不进入主路径，后期可作为 cross-sim OOD 加分项。

Switch:
    只作为 related work，不作为第一版实现对象。
```

---

# 2. 第一版技能范围、术语与 calibration 粒度

## 2.1 技能术语统一

为了避免 stop / idle / wait 混用，第一版统一如下：

```text
idle:
    SONIC 底层 LOCOMOTION_IDLE 模式。

stop:
    idle + duration ≤ 1s。
    用于显式中止当前动作或快速稳定。

wait:
    idle + duration > 1s。
    用于主动等待、重新观察或延迟恢复。

passive:
    critic / calibration 中的统一 skill type。
    stop、idle、wait 共享 passive embedding，仅用 duration 参数区分。
```

因此，底层可用的是：

```text
LOCOMOTION_IDLE
```

而 CASA 层看到的是：

```text
PassiveSkill(duration, mode = stop / wait / reobserve)
```

---

## 2.2 直接使用 SONIC 的技能

```text
walk:
    LOCOMOTION_WALK + vx / vy / vyaw

passive:
    LOCOMOTION_IDLE + duration

turn / face:
    idle + face_yaw_deg / face_user phase
```

---

## 2.3 gesture：SONIC 原生能力 + 参数化包装

使用 Stage 2 中已有的：

```text
gesture_upper_body(elapsed_s, amplitude, frequency)
```

包装成：

```text
GestureSkill(
    amplitude,
    frequency,
    side,
    duration
)
```

第一版不重新训练 gesture policy。

---

## 2.4 stair-placeholder：物理真实 box + 高度标定

第一版不重训 stair policy。

做法：

```text
在 MuJoCo 中放置真实 box。
stair skill 实际上调用 walk / approach / step-over 尝试通过。
oracle 根据真实物理结果判断是否成功。
```

关键修正：第一版不固定假设 15cm。

必须先做 stair box 高度 reality check：

```text
候选高度：
    5cm / 8cm / 10cm / 12cm / 15cm

每个高度：
    SONIC-only walk policy 尝试通过 20 次

选择规则：
    选择 violation rate 在 30% - 70% 区间内的最高高度
    作为第一版 stair 默认高度。

如果 15cm 的 violation rate ≈ 100%：
    不使用 15cm。

如果最高可用高度是 8cm：
    第一版 stair-placeholder 就使用 8cm。
```

这样保证 stair skill 同时存在 safe 与 unsafe 样本，不会变成全失败 dead arm。

该 box 不是假 flag，因为：

```text
box 是真实几何体
接触是真的
卡住是真的
摔倒是真的
脚滑是真的
失败是真的
```

---

## 2.5 handover / recovery

第一版不做 handover。

recovery 第一版使用：

```text
fallback = passive stop / wait / reobserve
```

---

## 2.6 Per-skill calibration 粒度

第一版 Mondrian / per-skill calibration 的分组粒度是：

```text
skill type:
    walk
    turn
    gesture
    stair
    passive
```

也就是说：

```text
small gesture 和 large gesture 属于同一个 gesture calibration group。
gesture amplitude / frequency / duration 作为 critic 输入特征，
不作为 calibration group。
```

不采用：

```text
(type, parameter bin)
```

原因：

```text
分组太细会导致每组 dangerous samples 不足，
conformal threshold 不稳定。
```

---

# 3. 数据采集总策略：Hybrid

严格 counterfactual 需要完整系统 state save / restore。

当前 SONIC 链路包含：

```text
Python MuJoCo sim
+
C++ SONIC deploy process
+
ZMQ communication
+
policy internal history
+
motion playback state
+
command buffer
```

因此完整多分支回滚工程风险较高。

最终采用 Hybrid：

```text
主训练数据：
    randomized single-branch skill invocation rollouts

小规模分析数据：
    strict counterfactual subset
```

论文表述：

```text
CASA is trained on automatically generated randomized skill-invocation rollouts.
Additionally, we evaluate counterfactual candidate skills on a subset of synchronized states
to analyze skill-dependent risk.
```

不要把全部训练数据都称为 strict counterfactual。

---

# 4. 阶段顺序总览

```text
Phase 0: SONIC 原始 demo sanity check + stair box reality check
Phase 1: 封装 SONIC Skill API
Phase 2: 建立 Safety Oracle + Logger
Phase 3: 跑可比 SONIC-only baseline
Phase 4: Hybrid 数据采集可行性验证
Phase 4.5: Critic Feasibility Mini-Train
Phase 5: 收集 ID Dataset v1
Phase 6: 训练 Raw Critic
Phase 7: 加入 Hazard Head
Phase 8: Per-skill Conformal + Sequential Calibration
Phase 9: 单独采集 OOD stress-test 数据并加入 OOD Fallback
Phase 10: 完整实验与消融
```

---

# Phase 0：SONIC 原始 demo sanity check + stair box reality check

## 目标

确认当前 SONIC + MuJoCo deploy / sim2sim 链路能稳定运行，同时确认 stair-placeholder 的可用 box 高度范围。

## 主要任务

```text
跑通 Stage 1 / Stage 2 demo
确认 ZMQ 通信稳定
确认 MuJoCo sim loop 稳定
确认 walk / idle / face_user / gesture 相关命令可执行
确认 walk policy 在 0-15cm box 上的可观测响应区间
```

## 命令执行成功定义

```text
command successfully sent
+
SONIC deploy process accepted command
+
MuJoCo 中 robot state 出现对应可观测变化

例如：
    walk 命令后 base velocity > threshold
    idle 命令后 base velocity 下降
    face_user 命令后 yaw error 下降
    gesture 命令后 upper-body joint trajectory 变化
```

## Stair box 高度标定

```text
候选高度：
    5cm / 8cm / 10cm / 12cm / 15cm

每个高度：
    20 次 SONIC-only walk-through trial

记录：
    success rate
    violation rate
    fall rate
    stuck rate
    crossing time

选择：
    violation rate 在 30% - 70% 内的最高高度

如果所有高度 violation rate > 70%：
    暂时降低 stair 任务难度；
    或将 stair 从第一版主 quantitative 实验移出，只做 qualitative case。

如果所有高度 violation rate < 30%：
    增大 approach speed / 添加姿态扰动 / 增加 box friction variation。
```

## Go 判据

```text
连续运行 ≥ 30 min 无进程崩溃
ZMQ 通信无持续性 deadlock
walk / idle / face_user / gesture 命令至少各执行 20 次
命令执行成功率 ≥ 90%

stair box reality check 完成：
    至少找到一个高度，使 violation rate 在 30% - 70%
    或明确决定 stair 只作为 qualitative case
```

## Warning 判据

```text
偶发通信延迟或单次 episode hang
但重启 worker 后可以恢复

stair 只有较低高度可用，例如 5cm / 8cm
但 safe / unsafe 样本都存在
```

## No-go 判据

```text
deploy 进程频繁崩溃
ZMQ 通信不可恢复
基础 locomotion 无法稳定执行
所有 stair box 高度都 100% 失败或 100% 安全，且无法通过扰动调整
```

## 产出

```text
SONIC 原始链路 sanity report
已确认可用技能列表
stair box height calibration report
主要失败模式记录
```

---

# Phase 1：封装 SONIC Skill API

## 目标

把 SONIC 命令统一封装成 CASA 可调用的 skill interface。

## 第一版 Skill API

```text
Skill.name
Skill.params
Skill.estimate_duration
Skill.execute
Skill.status
Skill.termination_reason
```

## 第一版技能

```text
WalkSkill(vx, vy, vyaw, duration)
PassiveSkill(duration, mode = stop / wait / reobserve)
TurnSkill(face_yaw_deg, duration)
GestureSkill(amplitude, frequency, side, duration)
StairPlaceholderSkill(box_height, approach_speed, duration)
```

## Go 判据

```text
至少完成 5 个 skill wrapper：
    walk
    passive
    turn / face
    gesture
    stair-placeholder

每个 skill 都能记录：
    skill_type
    skill_params
    start_time
    estimated_duration
    execution_status
    termination_reason

同一 skill 在同一初始条件下重复 20 次：
    成功执行率 ≥ 90%
    无通信死锁
    无 episode hang
```

gesture 额外 go 判据：

```text
amplitude / frequency / duration 参数实际影响上身轨迹
small / medium / large 三档 amplitude 的 swept volume 有可测差异
```

stair-placeholder 额外 go 判据：

```text
使用 Phase 0 标定出的默认 box height
机器人尝试通过时能产生真实物理结果：
    成功通过
    卡住
    摔倒
    接触异常
```

## No-go 判据

```text
skill wrapper 无法稳定复现
gesture 参数对运动几乎无影响
ZMQ command 经常丢失
episode 结束状态无法可靠记录
stair-placeholder 没有 safe / unsafe 两类样本
```

## 产出

```text
统一 SONIC skill wrapper
skill registry
skill execution log 格式
per-skill calibration group 定义
```

---

# Phase 2：建立 Safety Oracle + Logger

## 目标

建立独立于 CASA 的安全违规判定器和统一日志系统。

## 第一版 Oracle 标签

```text
collision
near_collision
fall
near_fall
human_distance_violation
unsafe_gesture
stair_failure
runtime_timeout
```

## 每次 rollout 记录

```text
min_obstacle_distance
min_user_distance
min_arm_user_distance
max_torso_pitch
max_torso_roll
base_height
foot_contact
stair_crossing_success
control_loop_overrun
time_to_violation
violation_type
```

## Go 判据

```text
8 类 violation 至少实现 6 类
每条 rollout 都输出：
    violation yes/no
    violation_type
    time_to_violation
    violation_time_bin
```

人工抽检采用分层判据：

```text
Hard violation:
    collision
    fall
    stair hard failure

    oracle 与人工判断一致率 ≥ 95%

Soft violation:
    near_collision
    near_fall
    human_distance_violation
    unsafe_gesture

    oracle 与人工判断一致率 ≥ 80%

Runtime violation:
    runtime_timeout
    control_loop_overrun

    oracle 与人工判断一致率 ≥ 90%
```

严重错误率：

```text
明显 collision / fall / stair failure 却标 safe 的比例 ≤ 5%
```

标签分布初步判据：

```text
每个主技能至少有 safe 和 unsafe 样本
每个主 violation type 至少有 100 个正样本
```

## No-go 判据

```text
明显摔倒却标 safe
明显碰撞却标 safe
time_to_violation 无法记录
unsafe_gesture 只能靠 task flag，而不是几何距离 / swept volume
stair_failure 无法和普通 walk failure 区分
```

## 产出

```text
Safety oracle
episode logger
rollout logger
人工抽检报告
```

---

# Phase 3：SONIC-only 可比 baseline

## 目标

在已有 Skill API 和 Oracle 基础上，跑出可比较的 SONIC-only baseline。

## 任务设计

```text
起点
  ↓
walk 穿过动态障碍区域
  ↓
turn / face_user
  ↓
passive stop
  ↓
gesture
  ↓
walk 接近 calibrated box
  ↓
stair-placeholder
  ↓
turn
  ↓
walk 返回
```

## Go 判据

```text
至少跑 100 个完整 episode
episode 无异常中断率 ≥ 95%

每个 episode 记录完整：
    skill sequence
    skill params
    oracle labels
    task success
    violation count
    completion time
```

baseline 有效性判据：

```text
task success rate 在 30% - 90%
violation rate 在 5% - 60%
```

## No-go 判据

```text
SONIC-only 几乎 100% 安全，CASA 没有提升空间
SONIC-only 几乎 100% 失败，底层 executor 太差
episode log 不完整
unsafe invocation count 无法统计
```

## 产出

```text
SONIC-only baseline
failure videos
failure mode distribution
```

---

# Phase 4：Hybrid 数据采集可行性验证

## 目标

验证数据采集速度、标签质量、样本分布和并行 pipeline 是否支持后续训练。

---

## 4.1 主路线：随机化初始状态 + 单 branch rollout

每次采样：

```text
随机化 pre-invocation state
随机采样一个 skill + params
执行 H 秒
oracle 自动标注
保存 state-skill-label
```

---

## 4.2 采集 pipeline 并行化

由于单 worker 可能只有：

```text
90 - 360 samples/hour
```

Phase 4 必须验证多 worker 并行。

目标并行方案：

```text
4 个 worker 起步
每个 worker 包含：
    独立 Python MuJoCo sim process
    独立 C++ SONIC deploy process
    独立 ZMQ port
    独立 config / log directory
```

隔离要求：

```text
ZMQ port 不冲突
MuJoCo process 不共享易冲突状态
C++ deploy 多实例可同时运行
motion library 只读并发安全
每个 worker 独立 random seed
每个 worker 独立 output shard
```

并行 go 判据：

```text
4-worker 运行 ≥ 2 小时无系统性 deadlock
worker 平均成功率 ≥ 95%
总 throughput ≥ 500 samples/hour
```

如果并行失败，降级判据：

```text
单 worker 或少量 worker throughput ≥ 150 samples/hour
先完成 5k feasibility dataset
暂缓 100k 目标
```

---

## 4.3 Hard-case bootstrapping

为了避免 unsafe violation 过稀疏，Phase 4 必须加入 hard-case bootstrapping。

流程：

```text
Step 1:
    先随机采 1k samples

Step 2:
    根据 violation 样本找到高风险 region
    例如：
        user_distance 小
        obstacle_distance 小
        torso_roll 大
        gesture amplitude 大
        box_height 高
        latency 高

Step 3:
    在这些 region 附近加密采样

Step 4:
    将 unsafe positive rate 从 5-10% 拉到 30-50%
```

下表中的区间只是初始预设。  
Phase 4 Step 1 的 1k 随机采样完成后，必须根据实际 violation 分布微调。

初始 hard-case 采样区域示例：

```text
gesture:
    user_distance ∈ [0.4m, 1.0m]
    amplitude ∈ [60°, 100°]

walk:
    obstacle_distance ∈ [0.3m, 1.0m]
    walk_speed ∈ [0.4, 0.8] m/s

turn:
    side_obstacle_distance ∈ [0.3m, 0.8m]
    yaw_angle ∈ [45°, 120°]

stair-placeholder:
    box_height = Phase 0 calibrated height ± perturbation
    approach_speed ∈ [0.2, 0.6] m/s

runtime:
    latency ∈ [100ms, 300ms]
    overrun_rate elevated
```

---

## 4.4 小规模 counterfactual subset

额外选择少量 synchronized states：

```text
同一状态下测试多个 candidate skills
例如：
    walk slow
    walk fast
    turn small
    turn large
    gesture small
    gesture large
    stair-placeholder
    passive wait
```

该 subset 的用途：

```text
1 张 counterfactual risk heatmap / bar plot
3-5 个 case study
用于展示同一状态下不同 skill 参数导致不同风险
不作为主训练数据依赖
```

---

## 4.5 Feasibility Test 目标

```text
先收集 5,000 条 invocation samples
```

## Go 判据

```text
样本保存完整率 ≥ 98%
rollout hang rate ≤ 2%
每个主技能样本数 ≥ 500
总体 unsafe positive rate 在 10% - 50%
每个主技能 positive rate ≥ 5%
每条样本都有：
    pre-state
    skill params
    violation label
    violation type
    time_to_violation
    rollout metadata
```

throughput 判据：

```text
目标：≥ 500 samples/hour
降级可接受：≥ 150 samples/hour，但需要缩小 Dataset v1 目标或增加运行时间
```

counterfactual subset 判据：

```text
至少完成 100 个 synchronized states × 5 candidate skills
重复同一 branch 5 次：
    outcome 一致率 ≥ 90%
```

## No-go 判据

```text
样本缺失率高
rollout 经常 hang
危险样本太少
某些技能几乎没有正样本
throughput < 150 samples/hour
counterfactual subset 完全不可复现
```

## 失败回退方案

如果 strict counterfactual subset 不稳定：

```text
保留主路线 randomized single-branch rollout
counterfactual 改为 qualitative analysis
不把 strict counterfactual 写成核心方法
```

如果 throughput 太低：

```text
减少 horizon H
增加 worker 数
减少非关键日志频率
先收 50k 而不是 100k
```

如果 positive rate 太低：

```text
加强 hard-case bootstrapping
提高高风险 region 采样比例
```

## 产出

```text
5k feasibility dataset
数据质量报告
采集速度报告
并行 worker 报告
hard-case bootstrapping 报告
counterfactual subset 初步报告
```

---

# Phase 4.5：Critic Feasibility Mini-Train

## 目标

在正式扩展到 50k-100k 数据前，验证 5k feasibility data 是否真的能让 critic 学到有效信号。

这是防止大规模数据采集作废的关键阶段。

## 架构要求

Mini critic 必须使用 Phase 6 同款架构的简化版，而不是任意小模型。

具体要求：

```text
共用 Phase 6 的 encoder 设计：
    robot encoder
    human / user encoder
    environment encoder
    runtime encoder
    skill type embedding
    skill parameter encoder

简化方式：
    降低 hidden dimension
    减少 MLP 层数
    去掉 hazard head
    不做 ensemble
    不做 conformal
```

这样 mini critic 的 go/no-go 才能反映数据可学习性，而不是架构差异。

## 方法

使用 5k feasibility dataset 训练：

```text
输入：
    robot state
    user state
    environment state
    runtime state
    task phase
    skill type
    skill parameters

输出：
    violation probability
```

## Go 判据

```text
mini critic overall AUROC ≥ 0.65
至少 3 个主要 skill 的 per-skill AUROC ≥ 0.60
Brier score 优于常数概率 baseline
```

## Warning 判据

```text
overall AUROC 在 0.60 - 0.65
部分 skill 学不到，但整体有趋势
```

## No-go 判据

```text
overall AUROC ≈ 0.55
per-skill AUROC 接近随机
模型只学习到 class prior
```

## 失败诊断

如果 no-go，必须先诊断再进入 Phase 5：

```text
输入特征不足：
    缺 user-relative geometry
    缺 arm swept volume
    缺 foot contact
    缺 torso pitch / roll
    缺 support / stability proxy
    缺 runtime latency

oracle 噪声：
    标签边界不一致
    near-collision / unsafe gesture 规则太模糊
    time_to_violation 不稳定

采样分布问题：
    positive samples 太少
    high-risk / low-risk region 混在一起
    某些 skill 数据不足
```

## Go 后动作

```text
mini critic 通过后，Phase 5 Dataset v1 可以包含 Phase 4 的 5k samples，
前提是这些样本通过质量检查。
```

## 产出

```text
mini critic report
feature sufficiency report
是否允许扩量采集的 go/no-go 决策
```

---

# Phase 5：收集 ID Dataset v1

## 目标

正式收集 CASA 训练、校准和 ID 测试数据。

注意：

```text
OOD 数据不在 Phase 5 采集。
Phase 5 的 train / calibration / test 都在 ID domain randomization range 内。
OOD 数据仅在 Phase 9 OOD stress test 时单独采集。
```

## 数据文件

```text
train_invocations
calibration_invocations
test_invocations
episode_logs
counterfactual_subset
```

Phase 5 Dataset v1 可以包含 Phase 4 的 5k feasibility samples，条件是：

```text
Phase 4 数据质量合格
Phase 4.5 mini-train 通过
这些样本没有 train/test seed leakage
```

如果不满足，则 Phase 4 的 5k 只作为调试数据，不进入 Dataset v1。

## Go 判据

总量：

```text
总样本数 ≥ 50k
目标样本数 100k
```

每技能：

```text
每个主技能 ≥ 8k samples
每个主技能 dangerous samples ≥ 500
calibration set 每个主技能 dangerous samples ≥ 200
```

划分方式：

```text
train / calibration / test 按 scenario seed 划分
同一个 seed 不能同时出现在 train 和 test
```

样本质量：

```text
标签缺失率 ≤ 2%
样本字段缺失率 ≤ 2%
unsafe positive rate 在 10% - 50%
```

## No-go 判据

```text
危险样本太少
某些 skill 没有正样本
train/test 场景泄漏
采集速度太慢
oracle 标签缺失严重
```

## 失败回退方案

```text
危险样本太少：
    增加 hard-case sampling

某个技能正样本不足：
    暂时从主实验中移除该技能
    或只作为 qualitative case

100k 太慢：
    第一版先用 50k，但保证每技能 dangerous samples 足够
```

## 产出

```text
Dataset v1
dataset statistics
train/calibration/test split
```

---

# Phase 6：训练 Raw Critic

## 目标

训练最小 CASA critic：

```text
state × skill parameter → risk
```

## 推荐架构

第一版 critic 使用结构化多编码器架构：

```text
robot encoder:
    MLP(robot_state)

human / user encoder:
    MLP(user_state)

environment encoder:
    MLP(env_state)

runtime encoder:
    MLP(runtime_state)

skill encoder:
    skill type embedding + MLP(skill_params)

fusion:
    concat fusion 或 lightweight attention fusion

heads:
    risk head
    hazard head 在 Phase 7 加入
```

Phase 4.5 mini critic 使用同款架构的降维简化版。

## 输入

```text
robot state
user state
environment state
runtime state
task phase
skill type
skill parameters
```

## 输出

```text
risk score
violation probability
```

## Reject budget 定义

Raw Critic 与 Hard Contract 的比较使用三档拒绝率：

```text
reject rate ≤ 5%
reject rate ≤ 10%
reject rate ≤ 20%
```

也就是：

```text
每 100 次 skill invocation 中，
最多拒绝 5 / 10 / 20 次。
```

主报告使用 Pareto curve：

```text
x-axis:
    rejection rate

y-axis:
    unsafe invocation count
    task success rate
```

## Go 判据

判别能力：

```text
overall AUROC ≥ 0.75
每个主技能 AUROC ≥ 0.65
AUPRC ≥ min(2 × positive prevalence, 0.85)
或 AUPRC lift = AUPRC / prevalence ≥ 1.5
Brier score 优于常数概率 baseline
```

工程价值：

```text
在 reject rate = 10% 或 20% 时，
Raw Critic 相比 Hard Contract 至少减少 15% unsafe invocation

或在相近 unsafe invocation 下，
Raw Critic 提升 task success rate
```

## Warning 判据

```text
overall AUROC 在 0.70 - 0.75
每技能 AUROC 在 0.60 - 0.65
```

## No-go 判据

```text
overall AUROC < 0.70
gesture / stair 等关键技能 AUROC < 0.60
Raw Critic 不优于 Hard Contract
```

## 失败回退方案

```text
检查标签噪声
增加 hard-case 数据
减少输入维度噪声
按技能训练小模型
先移除最不稳定技能
```

## 产出

```text
Raw Critic model
SONIC + Raw Critic baseline
AUROC / AUPRC / Brier report
Pareto rejection-risk curve
```

---

# Phase 7：加入 Hazard Head

## 目标

预测风险发生的时间结构，而不只是是否出事。

## 输出

```text
hazard over time bins
```

例如：

```text
0.0 - 0.5s
0.5 - 1.0s
1.0 - 1.5s
1.5 - 2.0s
2.0 - 3.0s
```

## 评估范围

hazard 时间预测只在 violation subset 上评估。

也就是说：

```text
只对 horizon 内实际发生 violation 的样本，
评估模型是否预测了正确 time bin。
```

不把 no-violation / censored 样本混入 hazard bin macro-F1。

## Go 判据

时间预测：

```text
在 violation 样本上的 hazard bin classification macro-F1 ≥ 0.45
或在 violation 样本上的 top-1 violation bin accuracy ≥ 40%
且明显高于随机猜测
```

time-to-violation：

```text
在 violation 样本上的 MAE ≤ 0.75s
或 ≤ 一个 time bin + 0.25s
```

下游价值：

```text
带 hazard 的 CASA 相比 no-hazard 版本：
    mid-execution violation 减少 ≥ 10%
```

## No-go 判据

```text
hazard 只学到所有风险都在同一个 bin
time-to-violation 误差接近随机
加入 hazard 后没有任何安全指标提升
```

## 失败回退方案

```text
将 hazard 降级为 interpretability analysis
不作为主方法必要模块
只保留 risk critic + conformal 主线
```

## 产出

```text
Hazard head
time-to-violation evaluation on violation subset
hazard visualization cases
```

---

# Phase 8：Per-skill Conformal + Sequential Calibration

## 目标

让 risk score 变成可用于安全调用的校准风险判定，并落实 long-horizon coverage decay 实验。

---

## 8.1 Per-skill conformal calibration

每个技能单独校准：

```text
τ_walk
τ_turn
τ_gesture
τ_stair
τ_passive
```

目标错误放行率：

```text
alpha = 0.10
```

## Go 判据

calibration set：

```text
每个 skill empirical false negative rate ≤ alpha
```

test set：

```text
每个 skill empirical false negative rate ≤ alpha + 0.03
```

也就是：

```text
alpha = 0.10 时，test FNR ≤ 0.13
```

如果样本较少，则用：

```text
Wilson upper bound of FNR ≤ alpha + 0.05
```

相对 Global Conformal：

```text
Per-skill conformal 相比 Global conformal：
    unsafe invocation count 降低 ≥ 10%
    或 task success 在相近 FNR 下提高 ≥ 10%
```

calibration 数据要求：

```text
每个主技能 calibration dangerous samples ≥ 200
```

## No-go 判据

```text
test FNR 明显超过 alpha + 0.05
per-skill 不如 global
某些 skill calibration dangerous samples 不足
```

## 失败回退方案

```text
合并相近技能做 group-wise calibration
增加 calibration 数据
把低样本技能从主 quantitative 结果中移出
改成 qualitative case
```

---

## 8.2 Group Mondrian fallback

如果 per-skill dangerous samples 不足，启用 group-wise calibration：

```text
High-risk group:
    gesture
    stair

Locomotion group:
    walk
    turn

Passive group:
    stop / wait / reobserve
```

Group Mondrian 只作为 fallback 或 ablation，不替代主 per-skill 方案。

---

## 8.3 Sequential calibration variants

为了支撑 long-horizon coverage decay 主图，必须实现三种 calibration allocation：

```text
1. Marginal per-skill conformal
2. Bonferroni allocation:
       alpha_step = alpha / K
3. Online conformal adaptation:
       根据历史错误动态调整阈值 / alpha_t
```

K 的默认设置：

```text
K = 20 decision steps
```

K=20 任务构造方式：

```text
基于 Phase 3 任务脚本扩展。

方法 A：
    将 Phase 3 任务重复 2-2.5 遍，
    去掉中间重复的 setup / return。

方法 B：
    在中段插入额外 gesture / stair / turn 子段，
    将 macro decision steps 扩展到 20。

所有方法仍沿用 Phase 3 的 Skill API，
只扩展 phase sequence 长度。
```

## Sequential coverage Go 判据

```text
在 K = 20 连续技能调用任务上：

online conformal 的 cumulative empirical coverage
相比 marginal conformal 高 ≥ 5 percentage points

且 task success rate 绝对下降 ≤ 10 percentage points
```

同时报告：

```text
marginal
Bonferroni
online conformal
```

三者的：

```text
cumulative empirical coverage
unsafe invocation count
rejection rate
task success rate
```

## No-go 判据

```text
online conformal 没有提升 coverage
或者 coverage 提升来自极端拒绝，导致 task success 大幅下降
Bonferroni 过度保守，几乎无法完成任务
```

## 产出

```text
Per-skill thresholds
Global conformal baseline
Group Mondrian fallback
Marginal / Bonferroni / Online sequential calibration
FNR / coverage / reliability report
long-horizon coverage decay plot
```

---

# Phase 9：OOD Stress Test + OOD Fallback

## 目标

当模型遇到训练中没见过的状态时，不强行 allow。

注意：

```text
OOD 数据在 Phase 9 单独采集。
Phase 5 不采 OOD 数据。
```

## 第一版 OOD 检测主路线

```text
5-head deep ensemble
使用 prediction variance 作为 OOD score
```

具体做法：

```text
训练 5 个共享架构、不同 seed / bootstrap split 的 critic head 或 critic model。
对同一 state-skill 输入输出 5 个 risk prediction。
用 variance / disagreement 作为 OOD score。
```

备选 / ablation：

```text
encoder feature space Mahalanobis distance
k-NN distance in feature space
rule-based runtime OOD
```

## 第一版 OOD 场景

```text
1. 用户速度更快
2. 障碍物更密
3. box height 超出训练范围
4. gesture amplitude 超出训练范围
5. latency 突然增大
6. 机器人初始姿态扰动更强
```

## fallback 动作

```text
passive stop
passive wait
reobserve
```

## Go 判据

OOD 检测：

```text
6 类 OOD 场景平均 recall ≥ 70%
最差一类 recall ≥ 50%
ID false fallback rate ≤ 15%
```

安全提升：

```text
OOD unsafe invocation count 相比 no-OOD fallback 降低 ≥ 20%
```

任务保持：

```text
OOD fallback 后 task success rate 绝对下降 ≤ 10 percentage points
且相对下降 ≤ 30%
```

## No-go 判据

```text
OOD 检不出来
ID 场景误触发 fallback 太多
unsafe 降低了但 task success 崩掉
某一类 OOD recall 极低且无法解释
```

## 失败回退方案

```text
OOD 只保留 rule-based runtime OOD：
    latency out-of-range
    skill param out-of-range
    user distance out-of-range

ensemble variance 降级为 ablation
```

## 产出

```text
OOD detector
OOD stress-test dataset
OOD stress-test results
fallback effectiveness report
```

---

# Phase 10：完整实验与消融

## 10.1 Evaluation Protocol

所有主实验统一使用：

```text
K_seed = 5 random seeds
M = 100 episodes per seed
总计每个 method 500 episodes
```

同一 seed 下所有 method 使用相同环境初始化：

```text
same scenario seed
same user trajectory
same obstacle layout
same task phase sequence
same runtime perturbation seed
```

结果报告：

```text
mean ± std
95% confidence interval where appropriate
Mann-Whitney U test 或 bootstrap significance test
```

多重比较校正：

```text
跨 method × metric 的成对比较使用：
    Holm-Bonferroni
    或 Benjamini-Hochberg, alpha = 0.05

避免多重比较带来的假阳性。
```

对关键主指标报告显著性：

```text
unsafe invocation count
task success rate
ground-truth violation rate
coverage
```

---

## 10.2 主实验 Baselines

在同一个 SONIC executor 上比较：

```text
1. SONIC-only
2. SONIC + Hard Contract
3. SONIC + Raw Critic
4. SONIC + Temperature Scaling
5. SONIC + Global Conformal
6. SONIC + Safety Q-function baseline
7. SONIC + LLM Safety Check
8. SONIC + CASA
```

## Safety Q-function baseline

该 baseline 用于回应 safe RL / recovery RL 类方法。

实现方式：

```text
复用 Phase 5 Dataset v1
训练 Q_safety(s, skill, params)
输出 expected future cost / violation risk
用固定 threshold 决定 allow / reject / fallback
不使用 conformal
不使用 per-skill calibration
不使用 sequential coverage allocation
```

它代表 safe RL 中 cost critic / safety critic 的简化对照。

主要检验：

```text
1. 混合 action space 下，Q-function 是否不如 CASA 稳定
2. 固定 threshold 是否无法控制 false negative rate
3. 长时程连续调用下 coverage decay 是否更严重
```

可选扩展：

```text
Recovery policy:
    当 Q_safety 判危险时，触发 passive stop / wait
```

不要求第一版完整训练 CPO / Lagrangian PPO / SAC-Lagrangian。

---

## 10.3 消融实验

```text
CASA w/o skill parameters
CASA w/o hazard head
CASA w/o per-skill calibration
CASA w/o sequential calibration
CASA w/o OOD fallback
CASA full
```

## 10.4 主指标

```text
unsafe invocation count
ground-truth violation rate
false negative rate
task success rate
fallback count
rejection count
completion time
long-horizon coverage decay
```

## 10.5 Go 判据

相比 SONIC-only：

```text
unsafe invocation count 降低 ≥ 40%
```

相比 Hard Contract：

```text
unsafe invocation count 降低 ≥ 20%
task success rate 绝对下降 ≤ 10 percentage points
且相对下降 ≤ 30%
```

相比 Raw Critic：

```text
test FNR 更接近 alpha
long-horizon coverage decay 更慢
```

相比 Global Conformal：

```text
per-skill FNR 更稳定
至少 3 个主要 skill 中有 2 个表现更好
```

相比 Safety Q-function baseline：

```text
CASA test FNR 更接近目标 alpha
CASA long-horizon coverage 更高
CASA 在相近 rejection rate 下 unsafe invocation 更低
```

效率约束：

```text
CASA task success rate 相比 SONIC-only：
    绝对下降 ≤ 10 percentage points
    且相对下降 ≤ 30%

CASA completion time ≤ SONIC-only × 1.3
```

## 10.6 No-go 判据

```text
CASA 只是通过大量拒绝来减少风险
任务成功率大幅下降
completion time 大幅增加
相比 Global Conformal 没有优势
相比 Safety Q-function 没有优势
per-skill calibration 不稳定
sequential coverage 图无法支撑主结论
```

## 10.7 产出

```text
主结果表
消融表
coverage decay 图
reliability diagram
Pareto rejection-risk curve
Safety Q-function baseline 对比
counterfactual subset visualization
failure case visualization
demo video
```

---

# 5. 最终论文故事

最终论文故事：

```text
SONIC 已经能执行人形全身技能，
但它本身不知道什么时候应该调用这些技能。

真实长时程任务中，错误的技能调用时机会导致：
    碰撞
    近碰撞
    用户距离违规
    不安全手势
    台阶失败
    摔倒
    运行时超时

CASA 在 SONIC 外面加入一个校准过的安全调用层。
它在每次调用技能前，根据机器人状态、用户状态、环境状态、
运行时状态和技能参数预测未来风险。

CASA 使用 randomized skill-invocation rollouts 训练，
并用小规模 counterfactual subset 分析同状态不同技能的风险差异。

与 SONIC-only、规则式 safety gate、raw critic、Safety Q-function、
global conformal、LLM checker 相比，CASA 能减少 unsafe invocation，
并在长时程连续技能调用中保持更可靠的安全覆盖。
```

---

# 6. 最终推荐主路径

```text
1. 跑通 SONIC 原始 MuJoCo deploy / sim2sim 链路
2. 做 stair box 高度 reality check
3. 封装 walk / passive / turn / gesture / stair-placeholder 技能接口
4. 建立 MuJoCo safety oracle 和 logger
5. 跑 SONIC-only 可比 baseline
6. 做 Hybrid 数据采集可行性验证
7. 用 5k 数据 quick-train mini critic
8. mini critic 通过后再收 ID Dataset v1
9. 训练 Raw Critic
10. 加入 Hazard Head
11. 加入 Per-skill Conformal
12. 加入 Sequential Calibration variants
13. 单独采集 OOD stress-test 数据并加入 OOD Fallback
14. 完成主实验、Safety-Q baseline 和消融实验
15. 可选做 Isaac Lab cross-sim 或 HOVER / Switch-style 扩展
```
