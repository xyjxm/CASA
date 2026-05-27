
---

## 0. 已确定决策（相对版本 A 的增量）

版本 B 在版本 A 完整跑通后启动。前提：

```text
版本 A Phase 0 - Phase 5 全部完成
Per-skill Conformal 主实验已发表或已稳定
ID Dataset v1 已有 time_to_violation 和 violation_time_bin 字段
（Phase 2 oracle 已经记录，版本 B 不需要重采）
```

版本 B 新增范围：

```text
Hazard Head（discrete-time hazard model）
Hazard-aware mid-execution interrupt
SONIC skill graceful exit 接口
hazard 内部评估
CASA-A vs CASA-B 强 ablation
```

版本 B 仍不做：

```text
不做 OOD ensemble
不做 online conformal / sequential allocation
不做 LLM safety check baseline
不做 safe RL baseline
不做 stair-placeholder
不做 K = 20 long-horizon coverage decay
不做真实机器人部署
不引入新 skill type
不引入新任务
```

---

# 1. 系统边界（增量）

整体结构在版本 A 的基础上增加两处：

```text
Task Phase Manager
        ↓
Candidate Skill Generator
        ↓
Hard Contract Filter
        ↓
CASA Safety Critic (raw)
        ├── risk head        → per-skill conformal threshold → allow / reject
        └── hazard head      → hazard time profile → schedule interrupt?
                                                      ↓
SONIC Skill Wrapper            allow / allow-with-interrupt / reject
   ├── normal execute
   └── graceful exit interface （新增）
        ↓
MuJoCo Simulation
        ↓
Safety Oracle / Logger
```

版本 B 的 critic 仍然是 single backbone + two heads；hazard head 与 risk head 共享 encoder。

---

# 2. 阶段顺序总览（增量）

版本 B 续接版本 A 的 Phase 编号：

```text
Phase 6:  Hazard label 准备 + Hazard Head 训练
Phase 7:  SONIC graceful exit 接口 + Hazard-aware Interrupt + ablation
```

共 2 个 Phase（Phase 6 - Phase 7）。版本 B 不重跑 Phase 3 / Phase 4 的主训练数据采集——hazard 标签是已有 rollout 的字段（Phase 2 oracle 在版本 A 阶段就已经记录 `time_to_violation` 和 `violation_time_bin`）。

---

# Phase 6：Hazard Head 数据准备与训练

## 目标

在版本 A 已有 Dataset v1 上增加 hazard 监督信号，训练 hazard head，使 CASA-B 不只输出 horizon-level 违规概率，还输出 horizon 内时间分布。

## 6.1 Hazard label 准备

数据源：

```text
版本 A 已有 train / calibration / test invocations
每条样本已经包含：
    violation_yes_no
    violation_type
    time_to_violation
    violation_time_bin
```

如果版本 A 阶段没记录 `violation_time_bin`：

```text
回看 episode_logs，根据 sim timestep 重算 time_to_violation
如果连 time_to_violation 都没记，必须重采（说明版本 A 没遵守 Phase 2 的记录约定）
```

bin 划分按 skill type 分别确定：

```text
walk:
    K = 5
    bins = [0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-2.5] s

turn:
    K = 5
    bins = [0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-2.5] s

gesture:
    K = 6
    bins = [0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0-2.5, 2.5-3.0] s

passive:
    K = 4
    bins = [0.0-1.0, 1.0-2.0, 2.0-3.0, 3.0-4.0] s
```

bin 选择 go 判据：

```text
每个 skill 的每个 bin 中 violation samples ≥ 50
（如果某个 bin samples 不足，合并相邻 bin）
```

## 6.2 Hazard head 架构

```text
backbone: 复用版本 A 的 multi-encoder + fusion，frozen 可选
hazard head:
    MLP -> K logits per skill
    每个 skill type 独立 head（参数不共享），
    或共享 head + skill type embedding 作为条件输入
```

第一版建议每个 skill type 独立 hazard head（参数不共享），减少 cross-skill 干扰。如果显存或样本量紧张，再考虑共享 head + 条件输入。

## 6.3 损失函数

discrete-time hazard loss：

```text
对一条样本 i，记 violation_yes_no = y_i ∈ {0, 1}，violation_time_bin = b_i ∈ {1, ..., K}

每个 bin 输出 logit ℓ_ik = log h_ik / (1 - h_ik)

如果 y_i = 1：
    在 bin 1 .. b_i - 1 上贡献 BCE(target = 0)
    在 bin b_i 上贡献 BCE(target = 1)
    bin b_i + 1 .. K 不监督（已经发生 violation）

如果 y_i = 0：
    在所有 bin 上贡献 BCE(target = 0)
    （censored at end of horizon）
```

总损失：

```text
L_total = L_risk + λ_hazard * L_hazard
λ_hazard 默认 1.0，可在 0.3 - 3.0 范围内 sweep
```

## 6.4 训练策略

```text
backbone：
    版本 A frozen 起步，可解冻微调（如效果显著）

hazard head：
    从头训练

early stopping：
    based on hazard validation loss
```

## 6.5 Go 判据

时间预测能力（只在 violation subset 上）：

```text
hazard bin classification macro-F1 ≥ 0.45
或 top-1 violation bin accuracy ≥ 40%
明显高于随机猜测（uniform K bins）
```

time-to-violation：

```text
在 violation samples 上的 MAE ≤ 0.75s
或 ≤ 一个 time bin + 0.25s
```

风险总分不被破坏：

```text
hazard 训练后 risk head 的 AUROC 不下降超过 0.01
hazard 训练后 per-skill conformal 的 test FNR 不上升超过 0.02
```

## 6.6 Warning 判据

```text
hazard macro-F1 在 0.35 - 0.45
某些 skill 的 hazard 学到的是 "所有 violation 集中在一个 bin"
risk head AUROC 下降 0.01 - 0.02
```

## 6.7 No-go 判据

```text
hazard 完全学不到时间结构（top-1 ≈ 随机）
所有 skill 的 hazard 都只学到 base rate
hazard 训练显著破坏 risk head 性能（AUROC 下降 > 0.02）
某个主 skill 的 violation samples 不足以训练 hazard
```

## 6.8 失败回退方案

```text
hazard 学不动：
    增大 hazard loss 权重 λ_hazard
    每个 skill 单独训练 hazard head
    增加 hard-case time-localized 样本
    检查 oracle 的 time_to_violation 是否带噪声

某个 skill 数据不足：
    把该 skill 的 K 减半（合并 bin）
    或在该 skill 上不启用 hazard，只保留 risk head

hazard 破坏 risk head：
    冻结 backbone，只训 hazard head
    或在 fusion 后 split 出独立 hazard branch
```

## 6.9 产出

```text
Hazard head (frozen backbone 版本 + fine-tune 版本)
Hazard bin 分布报告
Hazard macro-F1 / top-1 / MAE 报告（only on violation subset）
Hazard heatmap 可视化（每个 skill type × skill 参数 × hazard time）
Hazard 不破坏 risk head 的对照报告
```

---

# Phase 7：SONIC Graceful Exit + Hazard-aware Interrupt + Ablation

## 目标

把 hazard head 输出的时间结构转成下游控制信号：在 hazard 预测后段风险升高时，对正在执行的 skill 触发 graceful interrupt，并量化这种 time-localized fallback 的工程价值。

## 7.1 SONIC Graceful Exit 接口

为每个 skill 实现 graceful exit：

```text
WalkSkill:
    graceful_exit = 切到 LOCOMOTION_IDLE，
    在 ≤ 0.5s 内 base velocity → 0

TurnSkill:
    graceful_exit = yaw rate 衰减到 0，
    保持当前 yaw，切 LOCOMOTION_IDLE

GestureSkill:
    graceful_exit = amplitude 在 ≤ 0.3s 内衰减到 0，
    回到 home pose

PassiveSkill:
    无需 graceful exit（本身就是 idle）
```

## 7.2 Graceful Exit 安全验证（关键前置）

每种 skill 的 graceful exit 本身不能引入新的 violation。

测试方法：

```text
对每个 skill 在 calibration set 的子集上：
    1. 让 skill run-to-completion
    2. 让 skill 在 30% / 50% / 70% / 90% 进度时触发 graceful exit
    3. oracle 判定 exit 后 0.5s 内是否有 violation
```

Go 判据：

```text
每种 skill 在每个 trigger 进度下：
    graceful exit 引入的新 violation rate ≤ 2 percentage points
    （相对 run-to-completion 的 baseline）
```

如果某种 skill 的 graceful exit 危险（典型为 turn 中段切 IDLE 导致侧向失衡）：

```text
该 skill 退化为只支持 allow / reject，
不启用 hazard-aware interrupt
```

## 7.3 Hazard-aware Interrupt 决策逻辑

```text
input:
    risk_score = P(violation in H)
    hazard_profile = [h_1, ..., h_K]

decision:
    if risk_score > τ_skill (per-skill conformal threshold):
        return REJECT
    else:
        late_hazard_max = max(hazard_profile[k_late:])
        if late_hazard_max > τ_interrupt_skill:
            k_first_exceed = first k where hazard_profile[k] > τ_interrupt_skill
            return ALLOW_WITH_INTERRUPT at bin k_first_exceed
        else:
            return ALLOW
```

`k_late` 定义为 horizon 中段（例如 K / 3 之后），避免 interrupt 一启动就触发（等同于 reject）。

`τ_interrupt_skill` 标定方式：

```text
在 calibration set 的 violation subset 上：
    扫描 τ ∈ [0.1, 0.9]
    对每个 τ 计算 "scheduled interrupt 命中真实 violation bin" 的 precision 和 recall
    选择 precision ≥ 0.6 的最低 τ

τ_interrupt_skill 不使用 conformal calibration
（多 bin simultaneous coverage 留给版本 C）
```

## 7.4 主实验设计

复用版本 A Phase 5 的：

```text
任务设计（8-12 个连续技能调用决策点）
K_seed = 5 random seeds
M = 100 episodes per seed
500 episodes per method
```

新增对比：

```text
SONIC + CASA-A                       （版本 A 主方法）
SONIC + CASA-B-noInterrupt           （加 hazard head 但只用于 interpretability）
SONIC + CASA-B-Interrupt             （hazard-aware mid-execution interrupt 启用）
```

版本 B 主对照只在 CASA-A、CASA-B-noInterrupt、CASA-B-Interrupt 三者之间。其他 baseline（SONIC-only、Hard Contract、Raw Critic、Global Conformal）仍展示，但只是参考线。

## 7.5 新增主指标

```text
mid-execution violation count
    （violation 发生在 skill 启动后 30% - 100% 时段的次数）

end-of-skill stability rate
    （skill 完成或 graceful interrupt 后 1s 内机器人姿态稳定的比例）

scheduled interrupt count
    （CASA-B-Interrupt 实际触发 graceful exit 的次数）

interrupt precision
    （触发 graceful exit 的 rollout 中，
     如果不 interrupt 实际会发生 violation 的比例）

interrupt recall
    （所有 mid-execution violation 中，
     CASA-B-Interrupt 提前 graceful exit 拦住的比例）
```

版本 A 已有的主指标（unsafe invocation count、task success rate、per-skill FNR 等）也要继续报告，确认 hazard 不破坏版本 A 的核心保证。

## 7.6 主 Go 判据

强 claim（hazard 作为方法主线）：

```text
CASA-B-Interrupt 相比 CASA-A：
    mid-execution violation count 降低 ≥ 10%
    task success rate 绝对下降 ≤ 5 percentage points
    unsafe invocation count 不上升

CASA-B-Interrupt 相比 CASA-B-noInterrupt：
    mid-execution violation count 降低 ≥ 5%
    （证明 interrupt 本身比仅加 hazard head 更有价值）

interrupt precision ≥ 0.5
    （触发 graceful exit 不是误报为主）
```

弱 claim（hazard 退化为 interpretability）：

```text
hazard head 在 Phase 6 通过 macro-F1 / top-1 / MAE 判据，
但 CASA-B-Interrupt 相比 CASA-A 的 mid-execution violation 下降 < 10%。

此时论文主 claim 改为：
    humanoid skill invocation 风险具有显著时间结构，
    hazard head 提供可解释性。

hazard heatmap 与 case study 仍作为论文亮点。
```

## 7.7 No-go 判据

```text
hazard-aware interrupt 显著降低 task success rate（> 5 pp）
hazard-aware interrupt 不降低 mid-execution violation
某个 skill 的 graceful exit 引入更多 violation
interrupt precision < 0.3（误报为主）
hazard head 破坏 risk head 的 per-skill FNR 保证
```

## 7.8 失败回退方案

```text
某个 skill 的 graceful exit 危险：
    该 skill 不启用 interrupt，
    报告中明确标注该 skill 仅支持 allow / reject

interrupt precision 太低：
    提高 τ_interrupt_skill
    或要求 hazard profile 后段连续多个 bin 超过阈值才触发

task success rate 下降太多：
    降低 interrupt 启用频率
    或限制 interrupt 只用于 gesture / walk，不用于 turn

整体 ablation 没显著差异：
    论文 claim 退化为 interpretability
    hazard 不作为方法主线
    只贡献一组 hazard visualization 与 case study
```

## 7.9 产出

```text
Graceful exit 接口（每个 skill）
Graceful exit 安全验证报告
τ_interrupt_skill 标定报告
CASA-B-Interrupt 模型与决策代码
mid-execution violation 主结果表
interrupt precision / recall 报告
CASA-A vs CASA-B-noInterrupt vs CASA-B-Interrupt 主 ablation 表
hazard heatmap + case study 可视化
demo video（hazard signal 与 mid-execution interrupt 同步显示）
```

---

# 3. 最终推荐主路径（增量）

```text
版本 A 已完成（Phase 0 - 5）

8.  Hazard label 准备 + Hazard Head 训练（Phase 6）
9.  Graceful exit 接口 + safety 验证（Phase 7.1 - 7.2）
10. Hazard-aware interrupt 决策逻辑（Phase 7.3）
11. CASA-A vs CASA-B-noInterrupt vs CASA-B-Interrupt 主 ablation（Phase 7.4 - 7.6）
12. 论文增量章节写作（hazard、interrupt、time-localized fallback）
```
