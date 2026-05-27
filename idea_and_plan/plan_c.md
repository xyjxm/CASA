
---

## 0. 已确定决策（相对版本 A + 版本 B 的增量）

版本 C 在版本 A 和版本 B 完整跑通后启动。前提：

```text
版本 A Phase 0 - 5 全部完成（5 baseline、per-skill conformal 主实验已稳定）
版本 B Phase 6 - 7 已完成（hazard head、graceful exit、interrupt ablation 已稳定）
版本 A / B 的 Dataset v1 已沉淀完整字段
```

版本 C 新增范围：

```text
Stair-placeholder skill + box height reality check
K = 20 long-horizon humanoid 任务
Sequential conformal allocation（marginal / Bonferroni / online）
5-head deep ensemble + variance OOD detection
6 类 OOD stress scenario + 单独数据采集
LLM Safety Check baseline
Safe RL baseline（Lagrangian-PPO / SAC-Lagrangian）
完整 8 method × 6 ablation 主实验
```

版本 C 接受的工程成本：

```text
Safe RL baseline 训练：4 - 6 周
LLM baseline prompt 工程与评测：1 - 2 周
OOD ensemble 训练 + stress test 采集：3 - 4 周
K = 20 任务长 horizon 主实验：2 - 3 周
```

版本 C 仍不做：

```text
不做真实机器人部署（仅留作 future work）
不重新训练任何 humanoid controller
不重训 SONIC stair policy
不复现完整 Switch
不做复杂 handover skill
不做 social-norm 类 violation（trust violation / 被冒犯）
```

---

# 1. 系统边界（增量）

整体结构在版本 B 的基础上增加四处：

```text
Task Phase Manager
        ↓
Candidate Skill Generator
        ↓
Hard Contract Filter
        ↓
CASA Critic Ensemble (5 heads)    ← OOD 用 variance
        ├── mean risk head        → per-skill threshold (sequential allocation)
        └── hazard head           → hazard time profile
        ↓
allow / allow-with-interrupt / reject / OOD-fallback
        ↓
SONIC Skill Wrapper
   ├── normal execute
   ├── graceful exit interface
   └── stair-placeholder execute
        ↓
MuJoCo Simulation
        ↓
Safety Oracle / Logger
```

CASA-C 中 ensemble、sequential allocation 与 OOD fallback 都在 CASA-B 的 risk + hazard 框架内并联接入。

---

# 2. 第一版技能范围（增量）

```text
walk:        承自版本 A
turn:        承自版本 A
gesture:     承自版本 A
passive:     承自版本 A
stair:       版本 C 新增
```

stair 作为 5th calibration group：

```text
calibration group:
    walk
    turn
    gesture
    passive
    stair    （新增）
```

---

# 3. 阶段顺序总览（增量）

版本 C 续接版本 B 的 Phase 编号：

```text
Phase 8:   Stair-placeholder + Box height reality check
Phase 9:   K = 20 Long-horizon 任务设计 + 数据扩展
Phase 10:  5-head Ensemble + OOD Stress Test
Phase 11:  Sequential Conformal Allocation（marginal / Bonferroni / online）
Phase 12:  LLM Safety Check Baseline
Phase 13:  Safe RL Baseline（Lagrangian-PPO）
Phase 14:  完整主实验 + 6 项消融 + 论文整合
```

共 7 个 Phase（Phase 8 - 14）。

---

# Phase 8：Stair-placeholder + Box Height Reality Check

## 目标

把 stair 作为高难度 skill 加入主实验，但不重训 stair policy。用 MuJoCo 真实 box + 已有 SONIC walk policy 的 step-over 行为作为 stair-placeholder。

## 8.1 Box height reality check

```text
候选高度：5 / 8 / 10 / 12 / 15 cm
每个高度：SONIC-only walk policy 尝试通过 20 次
记录：
    success rate
    violation rate
    fall rate
    stuck rate
    crossing time
```

选择规则：

```text
选择 violation rate 在 30% - 70% 区间内的最高高度
作为版本 C 默认 stair box height

如果 15cm violation rate ≈ 100%：
    不使用 15cm

如果最高可用高度是 8cm：
    版本 C stair-placeholder 就使用 8cm

如果所有高度 violation rate > 70%：
    降低 approach speed
    或将 stair 从主 quantitative 实验中移出，
    只作为 qualitative case

如果所有高度 violation rate < 30%：
    增加 approach speed
    或加初始姿态扰动
    或增加 box friction variation
```

## 8.2 StairPlaceholderSkill 接口

```text
StairPlaceholderSkill(box_height, approach_speed, duration)
    底层调用 walk + approach + step-over 行为
    通过 SONIC walk policy 执行
    oracle 判定：
        stair_failure
        fall
        stuck
        success
```

## 8.3 数据扩展

```text
在 Dataset v1 基础上补充 stair-only samples ≥ 10k
其中 stair dangerous samples ≥ 600
stair calibration set dangerous samples ≥ 200
```

stair 的 hazard label：

```text
violation_time_bin 覆盖 stair 启动到落地的全过程
特别记录踩 box 中段（approach end → mid-cross）
```

## 8.4 Go 判据

```text
找到一个 box height，violation rate ∈ [30%, 70%]
StairPlaceholderSkill 产生 safe / unsafe / fall / stuck 四类样本
stair 数据采集完成（≥ 10k samples）
stair-only mini critic AUROC ≥ 0.65
（在 stair-only data 上单独跑 mini critic）
```

## 8.5 No-go 判据

```text
所有 box 高度都 100% 失败或 100% 安全，且无法通过扰动调整
stair samples 无法稳定区分 fall 和 stuck
stair-only mini critic AUROC < 0.6
```

## 8.6 失败回退方案

```text
找不到 30%-70% 区间高度：
    stair 从主 quantitative 实验移出
    只作为 qualitative case + 1 张 stair hazard heatmap

stair 学不动：
    stair 不参与 per-skill conformal 主实验
    只展示 stair 在 raw critic 下的失败模式
```

## 8.7 产出

```text
stair box height calibration report
StairPlaceholderSkill wrapper
stair-only dataset
stair-only mini critic report
```

---

# Phase 9：K = 20 Long-horizon 任务设计 + 数据扩展

## 目标

把版本 A / B 的 8-12 步任务扩展到 K = 20 macro decision points，为 long-horizon coverage decay 主图提供任务。

## 9.1 K = 20 任务设计

基于版本 A Phase 5 的任务脚本扩展：

```text
方法 A：
    将版本 A 任务重复 2-2.5 遍，
    去掉中间重复的 setup / return

方法 B：
    在中段插入额外 gesture / stair / turn 子段，
    扩展 macro decision steps 到 20

方法 C（推荐组合）：
    任务包含两个 user
    包含一次 stair crossing
    包含至少 4 次 gesture（不同 amplitude）
    包含至少 6 次 walk（不同速度）
    包含至少 3 次 turn（不同 yaw 范围）
    包含至少 3 次 passive（stop / wait 各几次）
```

所有 skill 仍沿用版本 A / B / C 的 Skill API，只扩展 phase sequence 长度。

## 9.2 数据扩展

```text
针对 long-horizon 任务额外采集：
    long-horizon rollouts ≥ 5k 条完整 K = 20 episodes
    每个 episode 全程记录每步 decision 与 oracle 标签

calibration set 扩展：
    每个 skill calibration dangerous samples ≥ 300（相比版本 A 的 200 更严格）
```

## 9.3 Go 判据

```text
K = 20 任务在 SONIC-only 下 task completion rate ∈ [10%, 70%]
（保证既不是不可完成、也不是必然完成）

K = 20 long-horizon dataset 完整保存率 ≥ 98%

每个 decision step 的 distribution 均有 dangerous samples
```

## 9.4 No-go 判据

```text
K = 20 任务在 SONIC-only 下 0% 完成或 100% 完成
某些 decision step 的 distribution 完全没有 dangerous samples
long-horizon episode 经常 hang
```

## 9.5 产出

```text
K = 20 任务脚本
long-horizon dataset
SONIC-only K = 20 baseline
```

---

# Phase 10：5-head Ensemble + OOD Stress Test

## 目标

训练 5-head deep ensemble，使用 prediction variance 作为 OOD score，并在 6 类 OOD scenario 上量化 OOD detection + fallback 效果。

## 10.1 5-head Ensemble

```text
backbone：复用 CASA-B 的 multi-encoder + risk head + hazard head
ensemble：5 个独立 head（or 5 个独立 critic）
不同 seed
不同 bootstrap split（80% subsampling）
```

实现选项：

```text
选项 A：5 个独立 critic（推荐）
    显存大，训练慢，但 diversity 强
    每个 critic 独立 backbone

选项 B：共享 backbone + 5 个独立 head
    显存小，训练快
    但 diversity 偏弱
    第一版降级方案
```

## 10.2 OOD score

```text
对同一 (s, skill, params) 输入：
    mean_risk = mean(risk_i)
    var_risk = var(risk_i)
    disagreement = max(risk_i) - min(risk_i)

OOD score = var_risk （主）
            或 disagreement （副，作 ablation）
```

τ_OOD 标定：

```text
在 ID validation set 上选 τ_OOD，
满足 ID false fallback rate ≤ 15%
```

## 10.3 OOD stress test 数据

6 类 OOD scenario，每类独立采集：

```text
1. 用户速度更快：
    user speed 超出训练 max × 1.5

2. 障碍物更密：
    obstacle density 超出训练 max × 1.5

3. stair box height 超出训练范围：
    box height 超出 Phase 8 calibrated 范围 + 3cm

4. gesture amplitude 超出训练范围：
    amplitude 超出训练 max × 1.3

5. latency 突然增大：
    latency 进入训练 max × 2

6. 机器人初始姿态扰动更强：
    torso pitch / roll 初始扰动超出训练 max × 1.5
```

每类 OOD scenario：

```text
≥ 500 episodes per scenario
独立 oracle 标签
不进入 ID train / calibration / test
```

## 10.4 OOD fallback 决策

```text
if var_risk > τ_OOD:
    fallback (passive stop / wait / reobserve)
else:
    use CASA-B-Interrupt normal decision
```

reobserve 在版本 C 引入：

```text
PassiveSkill(duration, mode = reobserve)
    停 0.5 - 1.5s
    刷新 perception
    重新触发 CASA decision
```

## 10.5 Go 判据

OOD detection：

```text
6 类 OOD scenario 平均 OOD recall ≥ 70%
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

不破坏 ID：

```text
启用 ensemble + OOD fallback 后，
ID 主实验 unsafe invocation 不上升
ID 主实验 per-skill FNR 不超过 alpha + 0.03
```

## 10.6 No-go 判据

```text
OOD 检不出来（某类 recall < 30%）
ID false fallback rate > 25%
unsafe 降低但 task success 崩盘
某类 OOD recall 极低且无法解释
ensemble 训练显著破坏 ID critic 性能
```

## 10.7 失败回退方案

```text
ensemble variance 不工作：
    降级为 Mahalanobis on encoder features
    或 k-NN distance on features

OOD 整体不工作：
    降级为 rule-based runtime OOD：
        latency out-of-range
        skill param out-of-range
        user distance out-of-range
    ensemble variance 作为 ablation only
```

## 10.8 产出

```text
5-head ensemble
OOD detector + τ_OOD
OOD stress-test dataset (6 类 × 500 episodes)
OOD detection report (per scenario)
OOD fallback effectiveness report
```

---

# Phase 11：Sequential Conformal Allocation

## 目标

在 K = 20 long-horizon 任务上对比三种 conformal allocation：marginal / Bonferroni / online conformal adaptation。

## 11.1 三种 allocation 实现

```text
Marginal per-skill conformal:
    α_step = α 对所有步固定
    每步使用版本 A 的 per-skill τ_skill

Bonferroni allocation:
    α_step = α / K
    重新在 calibration set 上标定每个 skill 的 τ_skill 使其满足 α_step

Online conformal adaptation (ACI-style):
    α_0 = α
    每步执行后：
        e_t = 1 if 本步 allow 且实际 violation else 0
        α_{t+1} = clip(α_t + γ (α - e_t), 0, 1)
    每步根据当前 α_t 重新查 calibration quantile 得 τ_skill
    γ 默认 0.05，可在 [0.02, 0.1] sweep
```

## 11.2 评测协议

K = 20 任务上：

```text
K_seed = 5
M = 100 episodes per seed
500 episodes per allocation × 3 allocations
```

cumulative empirical coverage 定义：

```text
对 episode 中前 k 步：
    coverage(k) = (allow 且实际 no-violation 的次数) / (allow 的次数)
    目标 ≥ 1 - α
```

## 11.3 Sequential coverage Go 判据

```text
在 K = 20 任务上：

online conformal 的 cumulative empirical coverage at K = 20
相比 marginal conformal 高 ≥ 5 percentage points

且 task success rate 绝对下降 ≤ 10 percentage points
```

同时报告：

```text
marginal
Bonferroni
online
```

三者的：

```text
cumulative empirical coverage at k ∈ {5, 10, 15, 20}
unsafe invocation count
rejection rate
task success rate
```

## 11.4 No-go 判据

```text
online conformal 没有提升 coverage
coverage 提升来自极端拒绝，task success 大幅下降
Bonferroni 过度保守，task completion rate < 10%
```

## 11.5 失败回退方案

```text
online 不工作：
    降级 Bonferroni + 适度 K 缩小
    或采用 Faithful CP 作为替代算法

K = 20 太难：
    K 降到 15
    报告中明确说明
```

## 11.6 产出

```text
三种 allocation 实现
long-horizon coverage decay 主图
allocation × metric 表
```

---

# Phase 12：LLM Safety Check Baseline

## 目标

实现 LLM Safety Check baseline，正面回答"为什么不直接用 LLM 当 safety checker"。

## 12.1 模型选择

```text
主结果：GPT-4o
副本：Llama-3-70B 或 Qwen-72B（本地，作成本副本）
不使用 GPT-3.5（语义能力不足）
```

## 12.2 Prompt 结构

```text
system prompt:
    定位为 humanoid safety judge
    输入约束
    输出格式（JSON: {decision, risk_level, reasoning}）

structured state:
    robot:
        velocity, torso_pitch, torso_roll, base_height, foot_contact
    user:
        distance, relative_yaw, visibility
    environment:
        obstacle_density, nearest_obstacle_distance, corridor_width
    runtime:
        latency, control_loop_overrun_rate
    task_phase: <current phase>

candidate skill:
    type
    params

few-shot examples:
    至少 5 个，覆盖 allow / reject / defer
    包含 2 个 hard case：
        1. 看似安全但实际危险（gesture amplitude 大 + user 距离临界）
        2. 看似危险但实际安全（torso pitch 大但快速恢复）
```

## 12.3 决策解析

```text
output:
    {decision: allow / reject / defer,
     risk_level: low / medium / high,
     reasoning: <chain-of-thought>}

defer → mapped to passive wait（与其他 baseline 公平）
```

## 12.4 评测约束

```text
每个 invocation 独立调用 LLM（不复用历史 decision context）
记录决策延迟（API call latency + 模型推理时间）
K_seed × M episodes 与其他方法对等
```

LLM 决策延迟约束：

```text
如果 LLM 平均延迟 > 1s/decision：
    在主表中标注，但仍跑完整评测
    单独报告"LLM real-time feasibility"
```

## 12.5 Go 判据

```text
LLM baseline 完整跑完 500 episodes per seed
每个 decision 输出格式可解析率 ≥ 95%
LLM defer rate 在 [5%, 40%]（不能全 defer 或不 defer）
```

## 12.6 主对比

```text
CASA-C 相比 LLM Safety Check：
    unsafe invocation count 更低
    decision 延迟更低
    long-horizon cumulative coverage 更高
    （在几何边界 hard case 上明显更准）
```

## 12.7 No-go 判据

```text
LLM 输出格式经常解析失败
LLM 全 allow 或全 reject（prompt 失败）
LLM 平均决策延迟 > 5s 导致评测无法完成
```

## 12.8 产出

```text
LLM prompt + few-shot 模板
LLM baseline 评测结果
LLM vs CASA-C 几何边界 hard case 对比
LLM real-time feasibility 报告
```

---

# Phase 13：Safe RL Baseline（Lagrangian-PPO / SAC-Lagrangian）

## 目标

正面跑 safe RL baseline，回应版本 A idea.md 中 "safe RL critic 不够" 的三个具体障碍论证。

## 13.1 算法选择

```text
主选：Lagrangian-PPO
副本：SAC-Lagrangian（off-policy，样本效率高）
可选扩展：Recovery RL 风格的 recovery policy

不主推：CPO（实现复杂，对照同样工程负担）
```

## 13.2 实现细节

```text
state space：与 CASA critic 输入完全相同
action space：
    skill type ∈ {walk, turn, gesture, passive, stair} (categorical)
    skill params ∈ continuous box

reward：task 完成进度（基于 phase manager）
cost：oracle violation flag（与 CASA 训练标签一致）

network：
    使用与 CASA critic 相同的 multi-encoder backbone
    避免 architecture 不公平

训练数据：
    版本 A Dataset v1 作为 offline pretrain（warm start）
    在 MuJoCo 中 fine-tune

baseline 收敛标准：
    cost 在最近 N 个 episode 内的 mean 不再下降
    return 不再上升
    训练 wall-clock 与 CASA 训练时长大致对等
```

## 13.3 部署决策

```text
input: state, skill, params
output: Q_safety(state, skill, params)

固定 threshold：
    if Q_safety > τ_Q:
        reject -> passive
    else:
        allow

τ_Q 标定：
    在 ID validation set 上选 τ_Q，
    使整体 unsafe invocation 与 CASA-C 大致对等（或在多 τ_Q 下报告 Pareto curve）
```

不使用 conformal，不使用 per-skill calibration，不使用 sequential allocation。

## 13.4 可选 Recovery policy

```text
当 Q_safety > τ_Q：
    切到 PassiveSkill(stop / wait)
    或调用 trained recovery policy（如已训练）
```

第一版不强制 Recovery policy。

## 13.5 Go 判据

```text
Safe RL baseline 训练收敛：
    cost 在 100 episode window 内 mean 稳定
    return mean 在 100 episode window 内稳定

部署评测完整：
    K_seed × M episodes 完成
    Pareto curve 至少 5 个 τ_Q 点
```

baseline 公平性约束：

```text
baseline 训练 wall-clock ≥ CASA-C 训练 wall-clock
baseline 总样本量 ≥ CASA-C Dataset v1 大小
（防止 baseline underfit 被审稿人指责）
```

## 13.6 主对比

```text
CASA-C 相比 Safety Q-function：
    test FNR 更接近目标 alpha
    long-horizon cumulative coverage 更高
    在相近 rejection rate 下 unsafe invocation 更低
```

## 13.7 No-go 判据

```text
Safe RL baseline 无法收敛
Safe RL baseline 训练时长远超 CASA-C（不公平比较）
Safe RL baseline 部署完全不可用（task success ≈ 0）
```

## 13.8 失败回退方案

```text
Lagrangian-PPO 不收敛：
    换 SAC-Lagrangian
    或用 offline-only CQL + cost constraint

如果完全跑不起来：
    将 Safe RL baseline 降级到 Related Work 中详述
    论文中明确说明"训练收敛性问题，主对比未纳入"
```

## 13.9 产出

```text
Safe RL baseline model
Safe RL Pareto curve
CASA-C vs Safety Q-function 主对比表
（如启用）Recovery policy 训练报告
```

---

# Phase 14：完整主实验 + 6 项消融 + 论文整合

## 14.1 Evaluation Protocol

```text
K_seed = 5 random seeds
M = 100 episodes per seed
500 episodes per method
任务：K = 20 long-horizon
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
95% confidence interval
Mann-Whitney U test 或 bootstrap significance test
Holm-Bonferroni 或 Benjamini-Hochberg, alpha = 0.05
```

对关键主指标报告显著性：

```text
unsafe invocation count
task success rate
ground-truth violation rate
cumulative empirical coverage at K = 20
OOD recall (worst-case)
```

## 14.2 主实验 Baselines（8 个）

```text
1. SONIC-only
2. SONIC + Hard Contract
3. SONIC + Raw Critic
4. SONIC + Global Conformal
5. SONIC + CASA-A (per-skill conformal)
6. SONIC + CASA-B-Interrupt (per-skill + hazard interrupt)
7. SONIC + LLM Safety Check
8. SONIC + Safety Q-function
9. SONIC + CASA-C (per-skill + hazard + sequential + OOD + stair)
```

## 14.3 消融实验

```text
CASA-C w/o skill parameters
CASA-C w/o hazard interrupt
CASA-C w/o per-skill calibration (use global)
CASA-C w/o sequential allocation (use marginal)
CASA-C w/o OOD fallback
CASA-C w/o stair
CASA-C full
```

## 14.4 主指标

```text
unsafe invocation count                     （第一关键）
ground-truth violation rate
per-skill empirical false negative rate
mid-execution violation count
end-of-skill stability rate
task success rate
fallback count
rejection count
completion time
cumulative empirical coverage at K ∈ {5, 10, 15, 20}
OOD recall (per scenario + worst-case)
OOD fallback effectiveness
stair-only metrics
LLM decision latency
Safety Q-function Pareto curve
```

## 14.5 主 Go 判据

相比 SONIC-only：

```text
unsafe invocation count 降低 ≥ 40%
```

相比 Hard Contract：

```text
unsafe invocation count 降低 ≥ 20%
task success rate 绝对下降 ≤ 10 pp
```

相比 Raw Critic / Global Conformal：

```text
（继承版本 A 判据）
```

相比 CASA-B-Interrupt：

```text
long-horizon cumulative coverage at K = 20 高 ≥ 5 pp
OOD unsafe invocation 降低 ≥ 20%
```

相比 LLM Safety Check：

```text
unsafe invocation count 更低
decision 延迟更低
long-horizon coverage 更高
```

相比 Safety Q-function：

```text
test FNR 更接近 alpha
long-horizon coverage 更高
相近 rejection rate 下 unsafe invocation 更低
```

效率约束：

```text
CASA-C task success rate 相比 SONIC-only：
    绝对下降 ≤ 10 pp，相对下降 ≤ 30%
CASA-C completion time ≤ SONIC-only × 1.3
```

## 14.6 主 No-go 判据

```text
CASA-C 仅通过大量拒绝降低风险
任务成功率大幅下降
completion time 大幅增加
相比 CASA-B-Interrupt 没有 long-horizon 优势
相比 LLM / Safety Q-function 没有显著优势
per-skill calibration 不稳定
sequential coverage 图无法支撑 online conformal 主结论
OOD recall 整体不达标
stair 完全失败
```

## 14.7 产出

```text
主结果表（9 method × 主指标）
消融表（6 ablation + full）
long-horizon coverage decay 主图
reliability diagram (marginal + per-skill)
rejection-risk Pareto curve
OOD stress-test 主图（6 scenario × recall）
OOD fallback effectiveness 图
LLM vs CASA-C 几何边界 hard case 对比
Safety Q-function vs CASA-C Pareto curve
stair case study
hazard heatmap + mid-execution interrupt case
counterfactual subset visualization
failure case visualization
demo video
```

---

# 4. 最终论文故事（完整版本 C）

```text
SONIC 已经能执行 humanoid 全身技能，
但它本身不知道什么时候应该调用这些技能。

CASA-C 在 SONIC 外面加入一层完整的 calibrated safety system：
    skill-parameterized critic
  + per-skill Mondrian conformal calibration
  + discrete-time hazard head
  + hazard-aware mid-execution interrupt
  + sequential conformal allocation (marginal / Bonferroni / online)
  + 5-head ensemble OOD detection + fallback
  + stair-placeholder 作为高风险 skill case

在 K = 20 long-horizon humanoid 任务上，
与 SONIC-only、Hard Contract、Raw Critic、Global Conformal、
LLM Safety Check 和 Safety Q-function 相比：

    CASA-C 显著降低 unsafe invocation
    维持 long-horizon cumulative coverage
    在 OOD 场景下进一步降低 unsafe invocation
    几何边界 hard case 优于 LLM
    长时程 coverage decay 优于 safe RL cost critic
```

---

# 5. 最终推荐主路径（增量）

```text
版本 A 已完成（Phase 0 - 5）
版本 B 已完成（Phase 6 - 7）

15. Stair-placeholder + Box height reality check（Phase 8）
16. K = 20 long-horizon 任务设计 + 数据扩展（Phase 9）
17. 5-head Ensemble + OOD stress test（Phase 10）
18. Sequential conformal allocation 三种变体（Phase 11）
19. LLM Safety Check baseline（Phase 12）
20. Safe RL baseline（Phase 13）
21. 完整 9 method × 6 ablation 主实验（Phase 14）
22. 完整论文写作（Intro / Related Work / Method / Experiments / Limitations）

可选 future work：
    Isaac Lab cross-sim OOD
    HOVER / Switch-style 扩展
    real robot 部署
    social-norm violation
    完整 handover skill
```
