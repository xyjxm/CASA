1. 概要（版本 C 增量）

版本 C 是 CASA 的完整野心版本。它在版本 A 和版本 B 的基础上，把 CASA 从"单次 skill invocation 的校准 gate"扩展为"humanoid long-horizon 任务下的完整 calibrated safety system"。版本 A 解决了"单次调用要不要 allow"，版本 B 解决了"调用过程中要不要中段中断"，但 humanoid 真实任务一次会调用十几到几十个 skill，**前面调用造成的状态变化会改变后面调用看到的分布**，单次校准的保证在长时程下会逐步衰减。同时版本 A 和版本 B 都假设 deployment 状态在训练分布内，对训练分布外的奇怪场景（高人群密度、异常延迟、参数超界）没有显式处理。版本 C 要把这两件事一起解决，并补齐版本 A 和版本 B 故意延后的两类强 baseline（LLM safety checker 和 safe RL critic），把 CASA 放到完整的 long-horizon humanoid safety 场景下与现代方法正面对比。

版本 C 在方法上增加四件事：第一是 sequential conformal allocation，把版本 A 的 marginal per-skill conformal 扩展为 marginal / Bonferroni / online conformal 三种 allocation，在 K = 20 的 long-horizon 任务上对比 cumulative empirical coverage decay；第二是 OOD detection 与 fallback，使用 5-head deep ensemble 的 prediction variance 作为 OOD score，在 6 类 OOD stress scenario 上量化 OOD recall、ID false fallback rate 和 unsafe invocation 降幅；第三是把 stair 作为完整 humanoid skill 加入主实验，提供一个比 walk / gesture 风险结构更复杂的高难度 case；第四是把 LLM Safety Check 和 Safe RL（Lagrangian-PPO 或 Recovery RL）作为完整 baseline 跑出来，正面回应"为什么不直接用 LLM 当 safety gate"和"safe RL 的 cost critic 不够吗"两个审稿人必问的问题。

版本 C 在实验上把任务长度从版本 A 的 8-12 个 decision points 扩展到 K = 20 的 long-horizon humanoid 任务，并把主 baseline 从版本 A 的 5 个、版本 B 的 8 个扩展到 8 个完整方法的对比，加上 6 个 ablation。最终 long-horizon coverage decay 主图、OOD stress test 主图、stair case study、与 LLM / safe RL 的正面对比图组合起来，构成 CASA 完整论文的实验骨架。

版本 C 的论文 claim 由三条独立但互相印证的子 claim 组成：(1) 在长时程连续技能调用下，per-skill conformal + sequential allocation 比 marginal-only 与 Bonferroni 都能保持更好的 cumulative empirical coverage；(2) 在 OOD 场景下，ensemble-variance fallback 能在不显著破坏 ID task success rate 的前提下，把 unsafe invocation 进一步降低 ≥ 20%；(3) 在严格控制 baseline 公平性的条件下，CASA 的 calibrated risk gate 优于 LLM-based safety check 和 safe RL 风格的 cost critic。这三条 claim 不是版本 A 和版本 B 的扩展，而是 long-horizon humanoid safety 这个 setting 下的独立结论。


---

2. Related Work（增量）

版本 C 在版本 A 和版本 B 已有的 related work 之上，增加四条新线。

第一条是 conformal sequential prediction。版本 A 的 per-skill conformal 给的是 single-step marginal coverage，但 long-horizon 连续调用下，K 次 marginal 各自 1 − α 的保证联合起来最多衰减到 1 − Kα 或 1 − (1 − α)^K。要在长时程下保持 cumulative coverage，需要 sequential conformal 工具。版本 C 引入两种工具：第一种是经典 Bonferroni allocation（每步用 α / K），简单但偏保守；第二种是 online conformal adaptation（如 Gibbs & Candès 2021 的 Adaptive Conformal Inference / ACI，或 Bhatnagar 2023 的 Faithful CP），根据历史错误率动态调整每步阈值。版本 C 不主张发明新的 sequential conformal 算法，而是把这两种工具第一次系统应用到 humanoid long-horizon skill invocation 上，并在与 marginal-only 的对比中量化 long-horizon coverage decay。

第二条是 OOD detection for safe robot deployment。Deep ensemble（Lakshminarayanan 2017）、Mahalanobis distance（Lee 2018）、ViM（Wang 2022）、energy-based OOD（Liu 2020）等方法已经在图像分类和分割上被广泛研究。机器人安全场景里 OOD 的应用相对较少，主要集中在自动驾驶感知输入异常检测和机械臂抓取失败预测。版本 C 把 OOD detection 用在 humanoid skill invocation gate 上：当 state-skill 组合落在训练分布外时，CASA 不强行给一个自信的 allow / reject 判断，而是触发 passive stop / wait / reobserve。版本 C 的实现选择 5-head deep ensemble + variance score，并配套 6 类显式构造的 OOD scenario（人群密度、用户速度、box height、gesture amplitude、latency、初始姿态扰动）作为压力测试。

第三条是 LLM-as-judge / LLM safety check。近年用 LLM 直接判断行为是否安全（如 SafeAgentBench、SafeBench、LLM-guided policy filtering）在 robotics 和 autonomy 领域成为热门 baseline 选择。版本 C 不绕过这条线：把当前状态用结构化文本描述（机器人速度、torso tilt、最近人体距离、用户距离、推理延迟、当前任务阶段、候选技能类型和参数）送给 GPT-4o 或本地 Llama-3-70B 等模型，让它用 chain-of-thought 推理输出 allow / reject / defer 加风险等级，并辅以 5-10 个 few-shot 示例。版本 C 想正面回答审稿人一定会问的"既然 LLM 这么强，为什么不直接用 LLM 当 safety checker"——预期 LLM 在语义层够用，但在连续几何细节（具体距离、速度、倾角的边界）和长时程一致性（多次决策的累积错误）上不如校准后的 critic。这个对比必须做得严肃：LLM 不只跑一次，而是 K_seed × M episodes 与其他方法对等评估。

第四条是 safe RL 中的 safety critic / cost critic。版本 A 把这条线放在 related work 里讨论了三个具体障碍（混合 action space、没有 distribution-free 保证、长时程累积失控），但没做主对照。版本 C 必须正面跑出 Safe RL baseline，否则 idea.md 中那段论证会被审稿人指为"自说自话"。具体实现选择 Lagrangian-PPO 或 SAC-Lagrangian 训练一个 (s, skill, params) → Q_safety，并加入一个 Recovery policy 作为可选扩展。版本 C 接受 Safe RL baseline 需要 4-6 周额外工程成本，但这是论文层面不可省的对照。版本 C 不要求 Safe RL 完全超过或不如 CASA，只要求 baseline 训练到收敛、对照公平。


---

3. 方法与创新点（增量）

版本 C 在方法上增加四个组件。

### 3.1 Sequential conformal allocation

在 K = 20 的 long-horizon 任务下，对比三种 allocation：

- Marginal per-skill conformal：版本 A 已有，每步独立使用 per-skill threshold τ_skill，理论上每步 marginal coverage ≥ 1 − α，但 K 步联合 coverage 不受控
- Bonferroni allocation：每步使用 α_step = α / K，理论上联合 coverage ≥ 1 − α，但在多数情况下过度保守，导致 rejection rate 显著上升
- Online conformal adaptation：基于 ACI 风格更新——每步根据历史经验错误率 e_t 调整 α_t = clip(α_{t-1} + γ (α − e_t), 0, 1)，并据此更新阈值

版本 C 不是发明新算法，是把这三种已知 allocation 第一次在 humanoid skill invocation 上系统比较，并报告 cumulative empirical coverage 随 K 的衰减曲线。

### 3.2 OOD ensemble fallback

CASA-C 训练 5 个共享架构、不同 seed / bootstrap split 的 critic head 或 critic model。每次 invocation 时：

```text
mean_risk = mean(risk_i for i in 1..5)
var_risk = var(risk_i for i in 1..5)

if var_risk > τ_OOD:
    return FALLBACK (passive stop / wait / reobserve)
else:
    use mean_risk in normal CASA-B decision logic
```

`τ_OOD` 在 ID validation set 上标定，控制 ID false fallback rate ≤ 15%。版本 C 的备选 / ablation 包括 Mahalanobis on encoder feature space、k-NN distance on feature space、rule-based runtime OOD（latency out-of-range、skill param out-of-range、user distance out-of-range）。

### 3.3 Stair-placeholder skill

版本 A 和版本 B 都不做 stair。版本 C 把 stair 作为高难度 case 加入，但**不重训 stair policy**，而是采用 plan.md 原版的 calibrated box reality check 路线：

```text
在 MuJoCo 中放置真实 box
stair skill 实际上调用 walk / approach / step-over 尝试通过
oracle 根据真实物理结果判断是否成功
box 高度通过 reality check 选择 violation rate 在 30%-70% 的最高高度
```

版本 C 明确承认 stair-placeholder 不是真正的 humanoid stair-climbing skill，但作为风险结构与 walk / gesture 显著不同的高难度 case（涉及双足稳定性、CoM、ZMP），可以独立验证 per-skill conformal 在高 base violation rate 技能上的表现。版本 C 把 stair 处理成"single skill case study + 主实验中作为 5 个 skill 之一参与 calibration"，而不是论文核心卖点。

### 3.4 LLM Safety Check baseline 与 Safe RL baseline

LLM Safety Check：将状态结构化文本送 GPT-4o（或本地 Llama-3-70B 作为成本替代）+ 5-10 个 few-shot 示例覆盖 allow / reject / defer 三类决策，包括"看似安全但实际危险"和"看似危险但实际安全"两类 hard case。输出 allow / reject / defer + 风险等级 + 理由。

Safe RL：使用 Lagrangian-PPO 或 SAC-Lagrangian 训练 Q_safety(s, skill, params)，使用版本 A Dataset v1 作为 offline data 与 environment rollout 混合训练。固定 threshold 决定 allow / reject / fallback，不使用 conformal，不使用 per-skill calibration。可选扩展 Recovery policy。

### 3.5 创新点（相对版本 B）

第一，**把 calibrated skill invocation 从单次 marginal 升级为 long-horizon sequential**，并在 K = 20 humanoid 任务上量化三种 allocation 的 coverage decay。这是 conformal prediction 文献在 humanoid robotics 上的具体应用，配套 cumulative empirical coverage 主图。

第二，**把 OOD detection 引入 humanoid skill invocation gate**，并显式构造 6 类 humanoid 场景下的 OOD 压力测试，建立 OOD recall × ID false fallback rate × unsafe invocation 降幅 的三维评估协议。这一组评估协议本身是版本 C 的 contribution。

第三，**完成 calibrated critic 与 LLM safety check、safe RL critic 的完整对比**。这个对比不是版本 C 发明任何方法，而是为整个 humanoid safety 领域提供一个公平、可复现、长时程的对照基准。

版本 C 的方法贡献是组合性贡献：单独看 sequential conformal / OOD ensemble / LLM baseline / safe RL baseline / stair-placeholder 都不是版本 C 的独创，但**把它们组合在同一个 humanoid skill invocation gate 框架内、在同一个 long-horizon humanoid 任务上完整对照**，是版本 C 才提供的。


---

4. 实验设计（增量）

版本 C 的实验在版本 A 和版本 B 的实验之上扩展任务长度、扩充 skill 集合、引入新对照方法和新评估协议。

### 4.1 任务设计（long-horizon）

版本 C 的主任务从版本 A 的 8-12 个 decision points 扩展到 K = 20：

```text
起点
  ↓
walk 穿过动态人群
  ↓
turn / face_user
  ↓
passive stop
  ↓
gesture A
  ↓
walk 接近 stair box
  ↓
stair-placeholder
  ↓
turn
  ↓
walk 接近 user 2
  ↓
turn / face_user 2
  ↓
gesture B
  ↓
walk 返回
  ↓
turn 进入次任务
  ↓
walk + obstacle dodge
  ↓
gesture C（短）
  ↓
passive wait
  ↓
turn
  ↓
walk 返回起点
  ↓
turn
  ↓
passive stop
```

约 20 个 macro decision points。任务设计同时考虑：长度足够检测 sequential coverage decay；包含 stair / gesture / turn / walk / passive 五种 skill 的多种参数组合；user 数量 ≥ 2 以体现 user 切换造成的分布漂移。

### 4.2 主实验 Baselines（8 个）

```text
1. SONIC-only
2. SONIC + Hard Contract
3. SONIC + Raw Critic
4. SONIC + Global Conformal
5. SONIC + Per-skill Conformal （CASA-A）
6. SONIC + Per-skill + Hazard Interrupt （CASA-B-Interrupt）
7. SONIC + LLM Safety Check
8. SONIC + Safety Q-function （Lagrangian-PPO / SAC-Lagrangian）
9. SONIC + CASA-C （per-skill + hazard + sequential conformal + OOD fallback + stair）
```

主对照里 CASA-C 是版本 C 完整方法。Safety Q-function baseline 与 LLM baseline 必须按 K_seed × M episodes 跑齐，与 CASA-C 严格对等。

### 4.3 消融实验

```text
CASA-C w/o hazard interrupt        （= CASA-A + sequential + OOD + stair）
CASA-C w/o per-skill calibration   （改用 global conformal）
CASA-C w/o sequential allocation   （只用 marginal）
CASA-C w/o OOD fallback            （只用 hard fallback）
CASA-C w/o skill parameters        （只用 skill type embedding）
CASA-C full
```

### 4.4 新增主指标

在版本 A 和版本 B 已有主指标之外：

```text
sequential coverage:
    cumulative empirical coverage at K = 5, 10, 15, 20
    （三种 allocation：marginal / Bonferroni / online）

OOD metrics:
    OOD recall (per scenario)
    OOD recall (worst-case scenario)
    ID false fallback rate
    OOD-conditioned unsafe invocation count

LLM baseline metrics:
    LLM decision agreement rate vs CASA-C
    LLM defer rate
    LLM-induced unsafe invocation count
    LLM 平均决策延迟

Safe RL baseline metrics:
    Q_safety threshold sweep 下的 Pareto curve
    Q_safety vs CASA-C 在相近 rejection rate 下的 unsafe invocation 对比

Stair-only metrics:
    stair success rate
    stair-conditioned violation rate
    stair-conditioned mid-execution interrupt count
```

### 4.5 Long-horizon coverage decay 主图

```text
x-axis: decision step k ∈ {1, 2, ..., 20}
y-axis: cumulative empirical coverage = (allow & no-violation) / allow up to step k

曲线：
    marginal per-skill conformal
    Bonferroni allocation
    online conformal adaptation
    + 三种 allocation 对应的 rejection rate / task success rate 副图
```

预期：marginal 在 k 增大时 coverage 显著下降；Bonferroni 维持 coverage 但 rejection rate 过高、task success rate 大幅下降；online conformal 在 coverage 与 task success 之间取得最好 trade-off。

### 4.6 OOD Stress Test

6 类 OOD scenario：

```text
1. 用户速度更快
2. 障碍物更密
3. stair box height 超出训练范围
4. gesture amplitude 超出训练范围
5. latency 突然增大
6. 机器人初始姿态扰动更强
```

每类 ≥ 100 episodes。Go 判据：

```text
6 类 OOD scenario 平均 OOD recall ≥ 70%
最差一类 recall ≥ 50%
ID false fallback rate ≤ 15%
OOD unsafe invocation count 相比 no-OOD fallback 降低 ≥ 20%
OOD fallback 后 task success rate 绝对下降 ≤ 10 percentage points，相对下降 ≤ 30%
```

### 4.7 LLM baseline 评测协议

```text
模型选择：
    主结果使用 GPT-4o
    成本副本使用 Llama-3-70B（或 Qwen-72B）

prompt 结构：
    system：CASA safety judge 角色描述
    structured state：robot / user / env / runtime 字段
    candidate skill：type + params
    5-10 个 few-shot 示例（覆盖 allow / reject / defer，含 hard case）
    输出格式：{decision, risk_level, reasoning}

每个 invocation 调用一次 LLM，记录决策延迟。
长 horizon 任务下不复用历史 decision context（保证与其他 baseline 公平）。
```

### 4.8 Safe RL baseline 训练协议

```text
算法：Lagrangian-PPO 或 SAC-Lagrangian
state space：与 CASA critic 输入相同
action space：(skill type discrete, skill params continuous)
reward：task 完成进度
cost：oracle violation flag

训练数据：
    版本 A Dataset v1 作为 offline pretrain
    + environment rollout 在 MuJoCo 中 fine-tune

终止条件：cost 收敛到训练 budget 内

baseline 公平性约束：
    使用与 CASA 相同 architecture backbone
    训练时长与 CASA 大致对等（防止 baseline underfit）
```

可选扩展 Recovery policy：Q_safety 判危险时切到 passive stop / wait。

### 4.9 主 Go 判据（增量）

相对版本 A / 版本 B 的所有判据继续保留。版本 C 新增：

```text
Long-horizon coverage：
    online conformal 的 cumulative empirical coverage at K = 20
    相比 marginal conformal 高 ≥ 5 percentage points
    task success rate 绝对下降 ≤ 10 percentage points

OOD：
    平均 OOD recall ≥ 70%，最差一类 ≥ 50%
    ID false fallback rate ≤ 15%
    OOD unsafe invocation 相比 no-OOD fallback 降低 ≥ 20%

LLM 对比：
    CASA-C 相比 LLM Safety Check：
        unsafe invocation count 更低
        decision 延迟更低
        long-horizon coverage 更稳定

Safe RL 对比：
    CASA-C 相比 Safety Q-function：
        test FNR 更接近目标 alpha
        long-horizon cumulative coverage 更高
        在相近 rejection rate 下 unsafe invocation 更低
```

### 4.10 主 No-go 判据（增量）

```text
online conformal 不优于 marginal
Bonferroni 几乎完全无法完成任务
某一类 OOD recall 极低且无法解释
ID false fallback rate 大幅超过 15%
unsafe invocation 降低但 task success 崩盘
CASA-C 相比 LLM 或 Safe RL 没有显著优势
stair 完全无法学习（所有方法在 stair 上都失败）
```


---

5. 预期结果（增量）

版本 C 的预期结果是一个 long-horizon 完整对比：

- Long-horizon coverage：marginal per-skill 在 K 增大时显著衰减；Bonferroni 保持 coverage 但牺牲 task success；online conformal 在 coverage 与 task success 之间取得最好 trade-off
- OOD：ensemble variance fallback 在 ID 上误报可控、在 OOD 上 recall ≥ 70%，配合 fallback 显著降低 OOD 场景的 unsafe invocation
- LLM 对比：LLM 在语义层面给出合理理由，但在连续几何边界（user distance 0.6m vs 0.4m 的差异、torso pitch 10° vs 15° 的差异）和 long-horizon 一致性（多次决策的累积错误）上不如 calibrated critic
- Safe RL 对比：Q_safety 在固定 threshold 下能减少部分 unsafe invocation，但 false negative rate 难以稳定控制在 α 附近，long-horizon coverage 衰减比 CASA-C 更快
- Stair：stair skill 的 base violation rate 显著高于其他 skill，per-skill conformal 在 stair 上的阈值显著低于 walk / passive，证明 per-skill calibration 在高风险 skill 上的必要性

版本 C 最终要回答的问题：在长时程 humanoid 任务下，calibrated skill invocation gate 是否在所有评估维度（unsafe invocation、long-horizon coverage、OOD robustness、task success rate、与现代 baseline 的对比）上都优于规则、原始 critic、global conformal、LLM safety check 和 safe RL critic？版本 C 的答案如果在大部分维度是 yes，CASA 就是一个完整可发表到顶会的 humanoid safety system。如果某些维度不是 yes，论文 claim 退化为"CASA 在某某具体维度上更好"，而不是 universal winner——这是版本 C 接受的失败回退方案。

版本 C 是 CASA 故事的完整形态，但它的所有结论都依赖版本 A 和版本 B 的核心结论先成立。版本 C 不替代版本 A 和版本 B，而是把它们扩展到 long-horizon humanoid safety system 的完整对照。
