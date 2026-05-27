1. 概要（版本 B 增量）

版本 B 在版本 A 的基础上回答一个版本 A 无法回答的问题：**风险会在动作的哪个阶段出现？** 版本 A 的 critic 只告诉你"这个 skill 在未来 H 秒内有多大概率出事"，但同一个总风险分数 0.4 可能意味着两种完全不同的情况——一种是动作刚启动就危险（机器人还没站稳就开始挥手），另一种是动作执行到后段才危险（手挥到一半用户突然走近）。版本 A 在这两种情况下做出相同的 reject 决策，但它们其实需要不同的处理：前者根本不该启动，后者可以启动但需要在动作中段提前 graceful exit。版本 B 的目标就是把这种时间结构显式建模出来，并把它转成下游的工程价值。

版本 B 在 critic 之上加一个 hazard head：在不改变版本 A 的输入和总体 risk head 的前提下，额外输出 horizon H 中每一段时间 bin 的瞬时风险率（discrete-time hazard）。版本 B 不重训整个 critic，而是在版本 A 的 multi-encoder 上 attach 一个并联 head，与 risk head 共享 backbone，分别监督。这样的工程代价小，但带来两个新东西：第一是"风险时间结构"作为 critic 输出维度的一部分，让 CASA-B 不只判断 allow / reject，还能判断 "allow with mid-execution interrupt scheduled at bin k"；第二是支持一种新的 fallback 模式——time-localized fallback——当 hazard head 预测后段风险显著高于前段时，CASA-B 可以选择启动 skill 但提前安排一个 graceful interrupt，而不是直接 reject 整个 skill。这把版本 A 的二值 gate 升级为一个带时间结构的 gate。

版本 B 不改变版本 A 的 per-skill conformal 主线：per-skill calibration 仍然只针对 horizon-level violation probability。hazard head 输出的时间结构在版本 B 中**不做 conformal 校准**，只作为附加预测和下游中断信号使用。这是有意的设计选择：horizon-level violation probability 的 conformal 校准已经在版本 A 提供了 distribution-free 的 false negative rate 保证，hazard bin 的额外校准会引入新的多 bin coverage 问题（典型如 simultaneous coverage），这个问题留给版本 C 的 sequential conformal 一起处理。版本 B 只回答："给定我们已经知道某个 skill 在 H 内有违规风险，违规大致发生在哪个时间 bin？"

版本 B 的论文 claim 也只增加一条：**加入 hazard 时间结构后，CASA 能在维持版本 A 同等 false negative rate 的同时，进一步降低 mid-execution violation 数量**。这条 claim 通过 "CASA-A vs CASA-B" 的强 ablation 直接证伪或证实，不依赖任何新的对照方法。版本 B 不引入 LLM / safe RL / OOD / online conformal，这些仍然是版本 C 的事。


---

2. Related Work（增量）

版本 B 在版本 A 的 related work 之外，额外接入一条线：survival analysis 与 discrete-time hazard models。Cox proportional hazards、discrete-time hazard models（Allison 1982、Tutz & Schmid 2016）是统计学中预测"事件什么时候发生"的成熟工具，已经被广泛用于医学生存分析、机械系统失效预测、自动驾驶碰撞时间预测（time-to-collision 估计）和机械臂抓取失败预测。版本 B 把这条线接到 humanoid skill invocation 上：critic 不只回答 "P(violation in H)"，还回答 "given violation, when?"。具体使用 discrete-time hazard formulation：把 horizon H 切成 K 个时间 bin，每个 bin 输出一个 hazard rate h_k（在 bin k 内、给定 bin 1..k-1 未发生 violation 的条件下发生 violation 的概率），与总 violation probability 一起训练。

与"普通预测式 hazard"的差别在于场景：医学和工业 failure prediction 是"被动观测一个固定动作过程会不会失败"，而版本 B 是"主动决定要不要启动一个动作，并在执行中根据 hazard 时间结构选择是否中断"。这把 hazard 模型从"预测工具"变成"控制信号"，并要求底层 skill executor（SONIC）支持 graceful interrupt。版本 B 的 related work 会单独讨论 mid-execution interrupt 在 humanoid skill 上的可行性：哪些 skill 容易 graceful exit（gesture、walk），哪些不容易（turn 到一半中断会带来侧向失衡）。这部分讨论本身就是版本 B 的一个工程贡献。


---

3. 方法与创新点（增量）

版本 B 在版本 A 方法之上增加两个组件：hazard head 和 hazard-aware mid-execution interrupt。

### 3.1 Hazard head

输入仍是版本 A 的 (state, skill type, skill params)，backbone 仍是版本 A 的 multi-encoder + fusion。版本 B 在 fusion 之后并联一个 hazard head，输出 K 个 hazard logits h_1, ..., h_K，对应 horizon H 内的 K 个时间 bin。bin 划分按 skill type 分别确定，例如：

- walk: K = 5，bins = [0–0.5s, 0.5–1.0s, 1.0–1.5s, 1.5–2.0s, 2.0–2.5s]
- turn: K = 5，bins = [0–0.5s, 0.5–1.0s, 1.0–1.5s, 1.5–2.0s, 2.0–2.5s]
- gesture: K = 6，bins = [0–0.5s, ..., 2.5–3.0s]
- passive: K = 4，bins = [0–1.0s, 1.0–2.0s, 2.0–3.0s, 3.0–4.0s]

bin 划分粒度的选择参考 Phase 2 已经记录的 `time_to_violation` 分布，让每个 bin 至少有足够 dangerous samples 支撑训练。

监督方式采用 discrete-time hazard loss：对于一条 rollout，给定其 violation_yes_no 与 violation_time_bin，loss 在每个 bin 上独立监督是否在该 bin 内发生 violation；no-violation 样本视为 censored at end of horizon。总 risk head 仍直接监督 horizon-level violation probability，与 hazard head 共享 backbone 但损失分开加权。在推理时，risk head 提供 horizon-level 概率（接入 per-skill conformal threshold），hazard head 提供 horizon 内时间分布（不接入 conformal，作为下游 interrupt 信号）。

### 3.2 Hazard-aware mid-execution interrupt

版本 A 的决策只有 allow / reject 两种。版本 B 引入第三种：allow-with-interrupt。决策逻辑是：

```text
if risk_score > τ_skill:
    reject -> passive fallback
elif risk_score ≤ τ_skill:
    if max(hazard[k1:]) > τ_interrupt_skill:
        allow, schedule interrupt at bin k_first_exceed
    else:
        allow, run to completion
```

`τ_interrupt_skill` 是一个新的 per-skill 阈值，**与 conformal threshold 解耦**，从 calibration set 上选择一个让 "scheduled interrupt 命中真实 violation bin" 的精度合理（例如 precision ≥ 0.6）的阈值。这里不强行用 conformal，因为 hazard bin 的多 bin coverage 是版本 C 处理的问题。

interrupt 通过 SONIC skill wrapper 实现 graceful exit：

- walk 的 graceful exit 是切到 LOCOMOTION_IDLE
- gesture 的 graceful exit 是衰减回 home pose
- turn 的 graceful exit 是减速到当前 yaw 后切 IDLE
- passive 默认无需 interrupt

版本 B 必须证明：**每种 skill 的 graceful exit 本身不会引入新的 violation**。如果某种 skill 的 graceful exit 比 run-to-completion 还危险，那种 skill 就退化为只支持 allow / reject。

### 3.3 创新点（相对版本 A）

第一，把 humanoid skill invocation gate 从二值 allow / reject 升级为带时间结构的 allow / allow-with-interrupt / reject 三档决策，并通过 discrete-time hazard 显式建模时间结构。这是 formulation 上的扩展，与版本 A 的 skill-parameterized critic 直接兼容，不需要重训 backbone。

第二，**hazard time structure 第一次被作为下游控制信号用于 humanoid skill mid-execution interrupt**，而不是停留在预测和可视化层。版本 B 把这个 claim 量化为 "mid-execution violation 减少 ≥ 10%" 的具体指标，作为相对版本 A 的工程价值证据。如果减少不到 10%，版本 B 的 hazard 就降级为 interpretability tool，不作为方法主线——这一点版本 B 提前给出明确 fallback。


---

4. 实验设计（增量）

版本 B 的实验在版本 A 完整跑通的前提下进行，**任务、environment、seed、episode 数完全复用版本 A**。版本 B 不引入新任务、不引入新 baseline、不增加 long-horizon 长度。版本 B 唯一新增的是两组实验：hazard 内部评估和 hazard-aware ablation。

### 4.1 Hazard 内部评估

只在 violation subset 上评估 hazard 时间预测，不把 no-violation / censored 样本混入 hazard bin macro-F1。具体指标：

- hazard bin classification macro-F1（只在 violation subset 上）
- top-1 violation bin accuracy（只在 violation subset 上）
- time-to-violation MAE（只在 violation subset 上）

go 判据：

- macro-F1 ≥ 0.45 或 top-1 accuracy ≥ 40%
- MAE ≤ 0.75s 或 ≤ 一个 time bin + 0.25s

### 4.2 Hazard-aware ablation（核心新实验）

主要对比：

```text
SONIC + CASA-A             （版本 A 主方法：per-skill conformal，无 hazard）
SONIC + CASA-B-noInterrupt （加 hazard head 但不启用 interrupt，仅用作 interpretability）
SONIC + CASA-B-Interrupt   （加 hazard head 并启用 hazard-aware mid-execution interrupt）
```

主指标在版本 A 全部主指标之外，**额外引入两个时间局部化指标**：

- mid-execution violation count（违规发生在 skill 启动后 30%-100% 时段的次数）
- end-of-skill stability rate（skill 完成或 graceful interrupt 后机器人姿态稳定的比例）

go 判据：

- CASA-B-Interrupt 相比 CASA-A 的 mid-execution violation count 降低 ≥ 10%
- CASA-B-Interrupt 相比 CASA-A 的 task success rate 绝对下降 ≤ 5 percentage points
- CASA-B-Interrupt 相比 CASA-B-noInterrupt 的 mid-execution violation count 降低 ≥ 5%（证明 interrupt 本身比单纯加 hazard head 更有价值）

如果 mid-execution violation 没有显著下降，**hazard head 降级为 interpretability tool**，版本 B 论文 claim 改为 "可视化 humanoid skill invocation 中的风险时间结构" 而不是 "通过时间局部化中断降低中段违规"。

### 4.3 Hazard 可视化

版本 B 引入一组定性图：

- 每个 skill type 给出 hazard heatmap：x 轴时间 bin，y 轴 skill 参数（gesture amplitude / walk speed / turn yaw_deg），颜色表示平均 hazard rate
- 选 3-5 个 case：同一状态下，hazard time profile 在不同 skill 参数下如何变化
- 选 3-5 个 mid-execution interrupt 的真实 rollout，可视化 hazard signal 与实际 violation timing 的对应关系

这组图不是 ablation，是论文的 interpretability 章节核心。即使 hazard 没带来 mid-execution violation 下降，这组图也能独立支撑 "humanoid skill invocation 的风险有显著时间结构" 这一观察。


---

5. 预期结果（增量）

版本 B 期望出现两类结果之一。

类型 1（强 claim 成立）：CASA-B-Interrupt 相比 CASA-A 的 mid-execution violation 显著下降 ≥ 10%，task success rate 维持相近水平。此时论文主 claim 是 "hazard-aware mid-execution interrupt 进一步降低 humanoid skill 调用中段的违规"，hazard head 作为方法主线之一。

类型 2（弱 claim 成立）：CASA-B-Interrupt 的 mid-execution violation 下降不显著，但 hazard heatmap 和 case study 清晰展示 humanoid skill 风险在时间维度上有结构。此时论文主 claim 退化为 "humanoid skill invocation 风险具有显著时间结构，hazard head 提供 interpretability"，hazard head 作为 interpretability tool。

不论哪种结果，版本 B 都是版本 A 的合理增量：版本 A 的论文 claim 不被破坏，版本 B 在版本 A 之上加一个 ablation 即可成文。版本 B 失败时回退到 interpretability，是论文级的安全设计。
