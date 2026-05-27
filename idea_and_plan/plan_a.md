
---

## 0. 已确定决策

版本 A 是 CASA 的最小可发表版本，所有 Phase 都围绕一个核心 claim：per-skill conformal calibration 在 humanoid skill invocation 上优于 raw critic 与 global conformal。

```text
主执行器：SONIC
主仿真链路：MuJoCo sim2sim / deploy 链路
研究重点：calibrated skill invocation gate，不接管底层控制
版本目标：跑通端到端 pipeline + 5 个 baseline 严格对比
数据路线：Hybrid
    主训练数据：randomized initial state + single branch rollout
    小规模分析数据：counterfactual subset，仅用于可视化与 case study
calibration：per-skill Mondrian conformal
fallback：固定 passive stop / passive wait
```

版本 A 不做：

```text
不做 hazard head
不做 mid-execution interrupt
不做 OOD ensemble
不做 online conformal
不做 Bonferroni / sequential allocation
不做 LLM safety check baseline
不做 safe RL baseline
不做 stair-placeholder
不做 long-horizon K = 20 coverage decay
不重新训练任何 humanoid controller
不做真实机器人部署
```

---

# 1. 系统边界

整体结构：

```text
Task Phase Manager
        ↓
Candidate Skill Generator
        ↓
Hard Contract Filter
        ↓
CASA Safety Critic (raw)
        ↓
Per-skill Conformal Threshold
        ↓
allow / reject
        ↓
SONIC Skill Wrapper       (allow)
fallback PassiveSkill     (reject)
        ↓
MuJoCo Simulation
        ↓
Safety Oracle / Logger
```

模块定位：

```text
SONIC:
    执行 walk / passive / turn / gesture。

MuJoCo:
    主仿真、rollout、oracle 标签、episode evaluation。

CASA-A:
    输入 state + skill type + skill params，
    输出 raw violation probability，
    经过 per-skill conformal threshold 判断 allow / reject。

Hard Contract Filter:
    版本 A 的"快速规则前置过滤"，
    同时也是版本 A 的强 baseline 之一。
```

版本 A 全程不引入 stair、hazard head、ensemble、online conformal、Bonferroni、LLM、safe RL。

---

# 2. 第一版技能范围与术语

## 2.1 技能术语统一

```text
idle:
    SONIC 底层 LOCOMOTION_IDLE 模式。

stop:
    idle + duration ≤ 1s。
    用于显式中止当前动作或快速稳定。

wait:
    idle + duration > 1s。
    用于主动等待。

passive:
    critic / calibration 中的统一 skill type。
    stop / wait 共享 passive embedding，仅由 duration 区分。

reobserve:
    版本 A 不引入。
```

## 2.2 第一版主技能

```text
walk:
    LOCOMOTION_WALK + vx / vy / facing_yaw_deg + duration

passive:
    LOCOMOTION_IDLE + duration + mode (stop / wait)

turn / face:
    idle + face_yaw_deg / face_user phase + duration

gesture:
    SONIC 原生 gesture_upper_body(elapsed_s, amplitude, frequency)
    + side + duration
```

共 4 个 skill type。不含 stair。不含 handover。不含 recovery policy。

## 2.3 Per-skill calibration 粒度

```text
calibration group:
    walk
    turn
    gesture
    passive

skill 参数（amplitude / frequency / vx / vy / facing_yaw_deg / yaw_deg / duration）
作为 critic 输入特征，不作为 calibration group。
```

不采用 (type, parameter bin) 细分组，原因与原 plan 一致：分组太细 → 每组 dangerous samples 不足 → conformal threshold 不稳定。

---

# 3. 数据采集总策略：Hybrid

```text
主训练数据：
    randomized single-branch skill invocation rollouts

小规模分析数据：
    counterfactual subset on synchronized states，
    用于 1 张 risk heatmap + 3-5 个 case study，
    不作为主训练或主校准数据。
```

论文表述：

```text
CASA-A is trained on automatically generated randomized
skill-invocation rollouts. Additionally, we evaluate counterfactual
candidate skills on a subset of synchronized states to analyze
skill-dependent risk.
```

---

# 4. 阶段顺序总览

```text
Phase 0:  SONIC 原始 demo sanity check（不含 stair box reality check）
Phase 1:  封装 4 个 SONIC Skill API
Phase 2:  建立 Safety Oracle + Logger
Phase 3:  Hybrid 数据采集可行性验证 + 5k mini critic
Phase 4:  收集 ID Dataset v1（≥ 50k） + 训练 Raw Critic
Phase 5:  Per-skill Conformal + 5 baseline 主实验
```

版本 A 一共 6 个 Phase（Phase 0 到 Phase 5），不再外接其它阶段。

---

# Phase 0：SONIC 原始 demo sanity check

## 目标

确认 SONIC + MuJoCo deploy / sim2sim 链路能稳定运行 4 个目标 skill。

## 主要任务

```text
跑通 SONIC Stage 1 / Stage 2 demo
确认 ZMQ 通信稳定
确认 MuJoCo sim loop 稳定
确认 walk / idle / face_user / gesture 命令可执行
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

## Go 判据

```text
连续运行 ≥ 30 min 无进程崩溃
ZMQ 通信无持续性 deadlock
walk / idle / face_user / gesture 至少各执行 20 次
命令执行成功率 ≥ 90%
```

## Warning 判据

```text
偶发通信延迟或单次 episode hang
但重启 worker 后可恢复
```

## No-go 判据

```text
deploy 进程频繁崩溃
ZMQ 通信不可恢复
基础 locomotion 无法稳定执行
```

## 产出

```text
SONIC 原始链路 sanity report
已确认可用技能列表
主要失败模式记录
```

---

# Phase 1：封装 4 个 SONIC Skill API

## 目标

把 SONIC 命令统一封装成 CASA-A 可调用的 skill interface。

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
WalkSkill(vx, vy, facing_yaw_deg, duration)
PassiveSkill(duration, mode = stop / wait)
TurnSkill(face_yaw_deg, duration)
GestureSkill(amplitude, frequency, side, duration)
```

## Go 判据

```text
4 个 skill wrapper 完成
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

## No-go 判据

```text
skill wrapper 无法稳定复现
gesture 参数对运动几乎无影响
ZMQ command 经常丢失
episode 结束状态无法可靠记录
```

## 产出

```text
SONIC skill wrapper（4 个）
skill registry
skill execution log 格式
per-skill calibration group 定义
```

---

# Phase 2：建立 Safety Oracle + Logger

## 目标

建立独立于 CASA 的安全违规判定器和统一日志系统。

## 第一版 Oracle 标签（6 类）

```text
collision
near_collision
fall
human_distance_violation
unsafe_gesture
runtime_timeout
```

版本 A 不采集 stair_failure（无 stair）。near_fall 合入 fall（torso pitch / roll / base height 联合阈值），不单列。

## 每次 rollout 记录

```text
min_obstacle_distance
min_user_distance
min_arm_user_distance
max_torso_pitch
max_torso_roll
base_height
foot_contact
control_loop_overrun
violation_yes_no
violation_type
time_to_violation     # 仍然记录，版本 B 会用
violation_time_bin    # 仍然记录，版本 B 会用
```

`time_to_violation` 和 `violation_time_bin` 在版本 A 不进入 critic 输出，但**必须从一开始就记录**，否则版本 B 要重采。这是免费保险。

## Go 判据

```text
6 类 violation 至少实现 5 类
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

    oracle 与人工判断一致率 ≥ 95%

Soft violation:
    near_collision
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
明显 collision / fall 却标 safe 的比例 ≤ 5%
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
```

## 产出

```text
Safety oracle (6 类)
episode logger
rollout logger
人工抽检报告（至少 300 条 rollout）
```

---

# Phase 3：Hybrid 数据采集可行性验证 + 5k mini critic

## 目标

在投入 50k 大规模采集前，先验证：

```text
1. 采集 pipeline 跑得动
2. oracle 标签质量够
3. 数据真的能让 critic 学到有效信号
```

## 3.1 主路线：randomized single-branch rollout

每次采样：

```text
随机化 pre-invocation state
随机采样一个 skill + params
执行 H 秒
oracle 自动标注
保存 state-skill-label
```

## 3.2 采集 pipeline 并行化

第一版目标 2-4 worker 并行。

隔离要求：

```text
ZMQ port 不冲突
MuJoCo process 不共享易冲突状态
C++ deploy 多实例可同时运行
每个 worker 独立 random seed
每个 worker 独立 output shard
```

并行 go 判据：

```text
2-worker 运行 ≥ 2 小时无系统性 deadlock
worker 平均成功率 ≥ 95%
总 throughput ≥ 300 samples/hour
```

降级判据（如果多 worker 不稳定）：

```text
单 worker throughput ≥ 150 samples/hour
先完成 5k feasibility dataset
延后 50k 目标
```

## 3.3 Hard-case bootstrapping

```text
Step 1: 先随机采 1k samples
Step 2: 找到高风险 region：
    user_distance 小
    obstacle_distance 小
    torso_roll 大
    gesture amplitude 大
    walk_speed 高
    latency 高
Step 3: 在这些 region 附近加密采样
Step 4: 把 unsafe positive rate 从 5-10% 拉到 30-50%
```

## 3.4 5k feasibility dataset 收集

```text
目标样本数：5,000 条 invocation samples
每个主技能 ≥ 500 samples
每个主技能 unsafe positive rate ≥ 5%
```

## 3.5 Counterfactual subset（小规模）

```text
100 个 synchronized states × 4 candidate skills (walk slow / walk fast / gesture small / gesture large)
仅用于：
    1 张 risk heatmap
    3-5 个 case study
不进入主训练
```

## 3.6 Mini critic feasibility train

架构要求：使用 Phase 4 同款 multi-encoder 架构的降维简化版，而不是任意小模型。

```text
共用 Phase 4 的 encoder 设计：
    robot encoder
    user encoder
    environment encoder
    runtime encoder
    skill type embedding
    skill parameter encoder

简化方式：
    降低 hidden dimension
    减少 MLP 层数
    不做 conformal
```

## Go 判据

数据：

```text
样本保存完整率 ≥ 98%
rollout hang rate ≤ 2%
每个主技能样本数 ≥ 500
总体 unsafe positive rate 在 10% - 50%
每个主技能 positive rate ≥ 5%
```

mini critic：

```text
overall AUROC ≥ 0.65
至少 3 个主要 skill 的 per-skill AUROC ≥ 0.60
Brier score 优于常数概率 baseline
```

counterfactual subset：

```text
100 个 synchronized states × 4 candidate skills 全部完成
同一 branch 重复 5 次：outcome 一致率 ≥ 90%
```

## Warning 判据

```text
overall AUROC 在 0.60 - 0.65
部分 skill 学不到，但整体有趋势
throughput 在 150 - 300 samples/hour
```

## No-go 判据

```text
overall AUROC ≈ 0.55
per-skill AUROC 接近随机
模型只学习到 class prior
危险样本太少
某些 skill 几乎没有正样本
counterfactual subset 不可复现
```

## 失败诊断（如果 no-go）

```text
输入特征不足：
    缺 user-relative geometry
    缺 arm swept volume
    缺 foot contact
    缺 torso pitch / roll
    缺 runtime latency

oracle 噪声：
    标签边界不一致
    near-collision / unsafe gesture 规则太模糊

采样分布问题：
    positive samples 太少
    某些 skill 数据不足
```

## 产出

```text
5k feasibility dataset
数据质量报告
采集速度报告
并行 worker 报告
hard-case bootstrapping 报告
counterfactual subset 初步报告
mini critic feasibility report
是否允许扩量采集的 go/no-go 决策
```

---

# Phase 4：收集 ID Dataset v1 + 训练 Raw Critic

## 目标

收集 CASA-A 主训练、calibration 和 ID test 数据，训练核心 Raw Critic。

注意：

```text
版本 A 全部数据都在 ID domain randomization range 内。
OOD 数据不在版本 A 采集（属于版本 C Phase 11）。
```

## 4.1 数据文件

```text
train_invocations
calibration_invocations
test_invocations
episode_logs
counterfactual_subset
```

Phase 4 Dataset v1 可以包含 Phase 3 的 5k feasibility samples，条件是：

```text
Phase 3 数据质量合格
Phase 3 mini-train 通过
这些样本没有 train/test seed leakage
```

## 4.2 Go 判据

总量：

```text
总样本数 ≥ 50k
目标样本数 80k
```

每技能：

```text
每个主技能 ≥ 10k samples
每个主技能 dangerous samples ≥ 600
calibration set 每个主技能 dangerous samples ≥ 200
```

划分方式：

```text
train / calibration / test 按 scenario seed 划分
同一个 seed 不能同时出现在 train 和 test 或 train 和 calibration
calibration set 必须从已经过 Hard Contract Filter 的子分布上采，
保证 calibration distribution 与 deployment distribution 一致
```

样本质量：

```text
标签缺失率 ≤ 2%
样本字段缺失率 ≤ 2%
unsafe positive rate 在 10% - 50%
```

## 4.3 No-go 判据

```text
危险样本太少
某些 skill 没有正样本
train/test 场景泄漏
采集速度太慢
oracle 标签缺失严重
```

## 4.4 失败回退方案

```text
危险样本太少：
    增加 hard-case sampling

某个技能正样本不足：
    暂时从主实验中移除该技能
    或只作为 qualitative case

50k 太慢：
    版本 A 接受 30k 下限，
    但每技能 dangerous samples 仍须 ≥ 600，
    calibration set 每技能 dangerous samples 仍须 ≥ 200
```

## 4.5 Raw Critic 架构

```text
robot encoder:
    MLP(robot_state)

user encoder:
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
    risk head（standalone violation probability）
    hazard head 在版本 B 加入，版本 A 不实现
```

## 4.6 Reject budget 定义

Raw Critic 与 Hard Contract 比较使用三档拒绝率：

```text
reject rate ≤ 5%
reject rate ≤ 10%
reject rate ≤ 20%
```

主报告使用 Pareto curve：

```text
x-axis: rejection rate
y-axis:
    unsafe invocation count
    task success rate
```

## 4.7 Raw Critic Go 判据

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

## 4.8 Warning 判据

```text
overall AUROC 在 0.70 - 0.75
每技能 AUROC 在 0.60 - 0.65
```

## 4.9 No-go 判据

```text
overall AUROC < 0.70
gesture 等关键技能 AUROC < 0.60
Raw Critic 不优于 Hard Contract
```

## 4.10 失败回退方案

```text
检查标签噪声
增加 hard-case 数据
减少输入维度噪声
按技能训练小模型
先移除最不稳定技能
```

## 产出

```text
Dataset v1
dataset statistics
train / calibration / test split
Raw Critic model
SONIC + Raw Critic baseline
AUROC / AUPRC / Brier report
Pareto rejection-risk curve
```

---

# Phase 5：Per-skill Conformal + 5 Baseline 主实验

## 目标

落实版本 A 唯一的核心 claim：per-skill Mondrian conformal calibration 在 humanoid skill invocation 上优于 raw critic 与 global conformal。

## 5.1 Per-skill Conformal Calibration

每个技能单独校准：

```text
τ_walk
τ_turn
τ_gesture
τ_passive
```

目标错误放行率：

```text
alpha = 0.10
```

calibration set 来源：

```text
必须使用已经过 Hard Contract Filter 的子分布
避免 selection bias 破坏 exchangeability
```

## 5.2 Per-skill Conformal Go 判据

calibration set：

```text
每个 skill empirical false negative rate ≤ alpha
```

test set：

```text
每个 skill empirical false negative rate ≤ alpha + 0.03
（alpha = 0.10 时，test FNR ≤ 0.13）
```

如果样本较少：

```text
Wilson upper bound of FNR ≤ alpha + 0.05
```

相对 Global Conformal：

```text
Per-skill conformal 相比 Global conformal：
    unsafe invocation count 降低 ≥ 10%
    或 task success 在相近 FNR 下提高 ≥ 10%
    或至少 3 个 skill 中有 2 个的 per-skill FNR 更接近 alpha
```

calibration 数据要求：

```text
每个主技能 calibration dangerous samples ≥ 200
```

## 5.3 Per-skill Conformal No-go 判据

```text
test FNR 明显超过 alpha + 0.05
per-skill 不如 global
某些 skill calibration dangerous samples 不足
```

## 5.4 失败回退方案

```text
合并相近技能做 group-wise calibration:
    motion group: walk + turn
    interaction group: gesture
    passive group: passive

Group Mondrian 只作为 fallback 或 ablation，
不替代主 per-skill 方案。
```

## 5.5 主实验 Baselines

在同一个 SONIC executor 上比较：

```text
1. SONIC-only
2. SONIC + Hard Contract
3. SONIC + Raw Critic (fixed threshold 0.5)
4. SONIC + Global Conformal
5. SONIC + CASA-A (per-skill conformal)
```

## 5.6 任务设计

中等长度 humanoid 交互任务：

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
walk 返回起点附近
  ↓
turn
  ↓
passive wait
```

约 8-12 个连续技能调用决策点。

不含 stair。
不要求 K = 20 long-horizon coverage decay（属于版本 C）。

## 5.7 Evaluation Protocol

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
Holm-Bonferroni 或 Benjamini-Hochberg, alpha = 0.05
```

## 5.8 主指标

```text
unsafe invocation count   （第一关键指标）
ground-truth violation rate
per-skill empirical false negative rate
task success rate
fallback count
rejection count
completion time
AUROC / AUPRC / Brier
ECE
reliability diagram (marginal + per-skill)
rejection-risk Pareto curve
```

版本 A 不报告 hazard bin macro-F1、不报告 long-horizon coverage decay、不报告 OOD recall。

## 5.9 主 Go 判据

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
per-skill FNR 更稳定
在相近 rejection rate 下 unsafe invocation 更低
```

相比 Global Conformal：

```text
per-skill FNR 更稳定
至少 3 个主要 skill 中有 2 个表现更好
```

效率约束：

```text
CASA-A task success rate 相比 SONIC-only：
    绝对下降 ≤ 10 percentage points
    且相对下降 ≤ 30%

CASA-A completion time ≤ SONIC-only × 1.3
```

## 5.10 主 No-go 判据

```text
CASA-A 只是通过大量 reject 来减少 unsafe invocation
任务成功率大幅下降
completion time 大幅增加
相比 Global Conformal 没有优势
per-skill calibration 不稳定
某个主技能的 dangerous samples 不足以支撑 calibration
```

## 5.11 产出

```text
Per-skill thresholds
Global conformal baseline
Group Mondrian fallback（如启用）
主结果表（5 method × 主指标）
reliability diagram
rejection-risk Pareto curve
per-skill FNR table
counterfactual subset visualization
failure case visualization
demo video
```

---

# 5. 最终论文故事

```text
SONIC 已经能执行 humanoid 全身技能，
但它本身不知道什么时候应该调用这些技能。

真实交互任务中，错误的技能调用时机会导致：
    碰撞
    近碰撞
    用户距离违规
    不安全手势
    摔倒
    运行时超时

CASA-A 在 SONIC 外面加入一层校准过的安全调用 gate。
它在每次调用技能前，根据机器人状态、用户状态、
环境状态、运行时状态和技能参数预测未来风险，
并通过 per-skill Mondrian conformal calibration
给出 distribution-free 的 false negative rate 控制。

CASA-A 使用 randomized skill-invocation rollouts 训练，
并用小规模 counterfactual subset 可视化同状态不同技能的风险差异。

与 SONIC-only、规则式 Hard Contract、Raw Critic、Global Conformal 相比，
CASA-A 在保持任务成功率的前提下显著降低 unsafe invocation。
```

版本 A 的论文 claim 只到这里。
hazard、OOD、sequential、stair、LLM baseline、safe RL baseline 不出现在版本 A 论文里。

---

# 6. 最终推荐主路径

```text
1. 跑通 SONIC 原始 MuJoCo deploy / sim2sim 链路
2. 封装 walk / passive / turn / gesture 4 个技能接口
3. 建立 MuJoCo safety oracle 和 logger（6 类 violation）
4. 做 Hybrid 数据采集可行性验证 + 5k mini critic
5. 收集 ID Dataset v1（≥ 50k） + 训练 Raw Critic
6. Per-skill Conformal + 5 baseline 主实验
7. 写论文
```
