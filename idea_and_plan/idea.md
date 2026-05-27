1. 概要

CASA 想解决的不是“机器人怎么把一个动作做好”，而是“机器人什么时候才应该启动这个动作”（when a humanoid skill is safe to invoke）。现在的人形控制器，例如 HOVER、SONIC，已经能让机器人走路、转身、上台阶、做手势。但是，这些控制器主要解决的是“动作怎么执行”（how to execute a skill），并不会主动判断“现在这个动作该不该执行”。比如：用户现在离机器人太近，挥手会不会打到人？台阶现在看不清楚，迈出去会不会摔？感知延迟和推理延迟变高了，继续高速走会不会跟不上控制周期？这些都不是动作能力本身的问题，而是技能调用时机（skill invocation）的问题。

CASA 可以理解成给人形机器人加了一个“安全顾问”（calibrated safety critic）。在机器人执行每个候选技能之前，这个安全顾问会先看一眼当前状态，包括机器人自己的姿态、速度、关节状态、周围人和障碍物、用户位置、任务阶段、感知延迟和推理延迟；同时也会看候选技能的具体参数，例如走路速度、转身角度、手势幅度、台阶高度。然后它会估计：如果现在执行这个技能，未来几秒内出问题的概率是多少。这里的“出问题”不是泛泛而谈，而是由独立安全判定器（safety oracle）定义的安全违规（safety violation），比如碰撞、近碰撞、摔倒、近摔倒、用户距离违规、不安全手势、台阶失败、用户丢失或运行时超时。更进一步，CASA 不只给一个总风险分数，还会判断风险大概出现在动作的哪个阶段：是动作刚开始就危险，还是动作执行到中后段才危险。这通过生存分析 / 风险率模型（survival / hazard model）来做。比如一个手势动作如果前 1 秒风险很高，可能说明机器人还没站稳；如果后 2 秒风险变高，可能说明用户正在靠近机器人。关键是，神经网络直接输出的概率不能直接当成真实概率使用。CASA 会用 conformal 风险校准（conformal risk control）让这个分数有统计保证，而且不同技能分开校准；遇到训练中没见过的奇怪场景，CASA 还会通过分布外检测（OOD detection）触发停下、站稳、重新观察或等待这样的保守动作。


---

2. Related Work

现有相关工作大致可以分成四条线。第一条线是人形机器人底层控制和动作能力（humanoid control and motion capability）。HOVER、SONIC、GMT 这类工作主要解决“怎么让人形机器人把动作做好”，提升 whole-body control 和 motion tracking 能力，让机器人可以更稳定地走路、转身、做复杂全身动作。Switch 则更关注“动作之间怎么切换”（motion-level skill switching），比如从站立切到行走，或者从一个动态动作切到另一个动态动作。但 CASA 问的是另一个问题：这些动作什么时候应该被调用？ 也就是说，HOVER / SONIC / Switch 是底层动作执行器（skill executor），CASA 是动作执行之前的安全判断层。底层控制器负责把动作做出来，CASA 负责判断现在做这个动作是否安全。第二条线是行为树、状态机和任务规划（Behavior Tree / FSM / task planning），比如 BehaviorTree.CPP、ROSPlan、PDDL，可以把任务组织成“导航 → 停止 → 讲解 → 继续导航”这样的流程，也能写 fallback 或 recovery 节点。但是它们通常依赖人工规则，很难覆盖连续变化的风险。比如用户离机器人 0.8 米、机器人速度 0.3 m/s、手势幅度 60 度时是否安全，这种连续、多维、组合变化的场景，很难只靠 if-else 写清楚。

第三条线是 safe RL 中的 safety critic，这条线是 CASA 最容易被混淆的对手，必须讲清楚区别。safe RL（如 Recovery RL、CPO、Lagrangian PPO）也学一个“安全打分器”——通常叫 safety Q-function 或 cost critic——来判断策略是否会进入危险状态。表面上看，把 safe RL 的 cost critic 改造成“输入状态 + skill embedding，输出风险”，似乎就能做 CASA 想做的事。但实际上有三个具体障碍，让 safe RL 不能直接套到 humanoid 技能调用问题上。第一是 action space 不匹配：safe RL 的 cost Q-function 通常假设 action 是连续控制量（关节力矩、目标速度），不天然处理“先选哪个离散技能、再传一组连续技能参数”的混合 action 空间，而 CASA 显式把 skill type 和 skill 参数（手势幅度、走路速度、台阶高度）都编码到输入里。第二是 没有 distribution-free 的风险保证：safe RL 给的是点估计 Q-value 或 cost expectation，决策时靠“Q 大于阈值就拒绝”这样的启发式，阈值怎么选没有理论支撑，而 CASA 用 conformal risk control 直接控制“错误放行危险技能的经验错误率 ≤ α”，是和分布假设无关的统计保证。第三是 长时程任务下风险累积失控：safe RL 通常只保证单步 cost 在期望意义上不超过预算，连续做 K 次技能调用后联合违规概率最多累积到 1 − (1−α)^K，而 CASA 通过 per-skill Mondrian calibration 和 online conformal 直接处理这种 sequential coverage decay 问题。简单说，safe RL 解决的是“一个学好的策略在执行过程中不要出事”，CASA 解决的是“一个已经存在的固定技能在当前状态被调用是否会出事，并且这个判断本身要可信、要可累积”。

第四条线是 conformal prediction 和 failure prediction。Conformal prediction（Vovk 2005、Angelopoulos & Bates 2023 综述）让模型输出的风险分数有统计意义上的覆盖保证；Lindemann 2023 等近期工作已经把 conformal 用到机器人轨迹安全验证上。Survival analysis 在自动驾驶事故预测、机械臂抓取失败预测里也有成熟应用。CASA 把这两条线结合到人形机器人技能调用问题中：先预测某个技能未来是否会导致 safety violation、什么时候发生，再用 per-skill calibration 让不同技能的风险分数各自可信。综合来看，HOVER / SONIC 解决动作怎么做，Switch 解决动作怎么切，BT / FSM / PDDL 解决任务流程怎么组织，safe RL 解决端到端策略执行过程中的安全，conformal + survival 提供通用风险预测和校准，而 CASA 解决的是人形长时程任务中已有技能什么时候可以安全调用。


---

3. 方法与创新点

CASA 的核心方法是学习一个技能参数化安全评价器（skill-parameterized safety critic）。给定当前状态 s_t 和候选技能 π_i，CASA 预测：如果现在执行这个技能，在未来时间窗口 H 内发生安全违规的概率是多少。这里的 H 取“技能预计执行时间 + 一段安全缓冲”，比如 gesture 大概需要 2 秒，那 H 就看未来 3 秒内会不会出问题；stair 大概 2.5 秒，H 取 3.5 秒，每个技能各自一个 H。部署时的整体决策流程是这样：给定当前状态和一组候选技能，先经过 Hard Contract Filter 用便宜的人工规则剔除明显违规的候选；剩下的候选送入 Safety Critic 估计风险，并由 Survival / Hazard Head 输出 horizon 内每段时间的风险率；然后用 per-skill 的 conformal 阈值判断是否在校准的安全区间内；最后通过 OOD ensemble 检查在分布外或高不确定情况下退回保守动作，最终输出 allow、reject 或 fallback。

CASA 的输入主要分三类。第一类是机器人自身状态（robot state），包括身体姿态、速度、关节状态、脚底接触、tracking error，以及一些人形机器人特有的稳定性指标，比如 CoM、ZMP、capture point、support polygon、torso pitch / roll 等，这些信息能帮助模型判断机器人当前是否站稳、是否接近失衡。第二类是环境和用户状态（environment and human state），包括障碍物距离、人群密度、最近人体距离、用户是否可见、用户与机器人的相对位置、台阶置信度、是否处于窄通道等，这些信息决定了当前动作会不会影响人机安全。第三类是任务和运行时状态（task and runtime state），包括当前任务阶段、候选技能类型、技能参数、推理延迟、感知延迟和控制周期超时率。技能不能只用一个 skill id 表示，因为同样是 gesture，大幅度挥手和小幅度指向展品的风险完全不同；同样是 walk，高速走和低速走的风险也不同。所以 CASA 的输入是 skill type embedding 加 skill 参数向量一起送入网络。

CASA 有四个创新点。第一个是 formulation 创新：把 humanoid 技能调用安全建模为“状态 × 技能参数 → horizon 内逐段风险”的预测问题，这是一个具体的 formulation 选择，不是普通 safety Q-function。和 safety Q-function 相比，输入端显式建模“离散技能 + 连续技能参数”的混合 action 空间，让 critic 能区分同类技能的不同参数变体；输出端不是单点 Q-value，而是 horizon 内的 hazard 时间分布，包含“风险什么时候发生”。这两个 formulation 选择不是为了好看，而是为了让后面的 per-skill calibration 和 time-localized fallback 真正能落地——没有这个 formulation，per-skill 校准没东西可校准，time-localized fallback 也无从触发。第二个是 survival / hazard 时间结构：普通 critic 只告诉你“会不会出事”，CASA 告诉你“大概什么时候会出事”，用离散时间 hazard 模型预测 horizon 中每一段的瞬时风险率。这样既能给一个总累积风险，又能解释风险的时间结构——是刚启动就危险、中段才危险、还是后段才危险——也支持执行一半发现后段风险升高时提前中断的 time-localized fallback。第三个是按技能分组的 conformal 风险校准（per-skill Mondrian conformal risk control）：不同技能的风险分布天然差很多，stair 的 base violation rate 可能比 stop 高一个数量级，gesture 的风险又和用户距离高度相关，所以 CASA 不把所有技能混在一起校准，而是每种技能分别标定风险阈值，理论上给出 distribution-free 的 false negative rate ≤ α 保证，即错误放行危险技能的经验风险有上限。第四个是 OOD 触发的保守退化（OOD-triggered fallback）：通过 ensemble variance 检测训练分布外的状态（人群密度过高、延迟异常、技能参数超出训练范围），一旦判定 OOD，CASA 不强行给自信判断，而是触发 stop、stabilize、wait、reobserve 等保守动作。整体来看，CASA 不是再造一个控制器，而是在已有控制器外面加一层“技能调用安全判断”。它解决的是：已有技能什么时候该用，什么时候不该用，什么时候应该先停下来再观察。


---

4. 实验设计

实验第一阶段只做仿真，重点不是比谁动作更漂亮，而是比谁更少在错误时机调用技能。任务场景设计为一个长时程人形任务（long-horizon humanoid task），而不是简单的“到展品点讲解”。例如：机器人从起点出发，走向物品台，穿过动态人群，上一级 15cm 台阶，接近用户，在用户面前停稳，做指示性手势或 handover，然后转身、下台阶、穿过人群返回起点。这个任务中包含 walk、turn、stop、gesture、stair、handover、wait、recovery 等多个技能，覆盖 9 到 15 个连续技能调用决策点，能够测试连续技能调用中的安全判断能力。这个任务也能体现人形机器人的特点：上下 15cm 台阶需要双足稳定性判断；接近用户后做 gesture 或 handover 需要 whole-body coordination；转身时如果还拿着物体或靠近用户，就需要考虑 CoM、ZMP、support polygon 等稳定性约束。这些都不是普通轮式机器人可以完全替代的。

对比方法包括：Direct Executor，按任务脚本直接调用技能，不做任何安全判断；Hard Contract Only，只使用人工规则过滤明显危险动作；Raw Safety Critic，训练普通安全评价器但不做校准；Temperature-scaled Critic，使用温度缩放做概率校准；Quantile Regression Critic，直接预测高分位风险；CVaR-aware Critic，用风险敏感方式做决策；Recovery RL / safe RL baseline，用 safe RL 中的 safety Q-function 或 recovery policy 作为对比，正面验证前面 Related Work 中说的“safe RL critic 不够”的三个具体障碍；LLM Safety Check baseline，用 LLM 直接判断技能是否可执行（具体设计见下方）；以及本文方法 CASA，包含 skill-parameterized safety critic、survival / hazard head、per-skill Mondrian conformal calibration 和 OOD fallback。Switch-style motion graph 处于不同抽象层级（motion-level skill switching），仅在 Related Work 中讨论，不作为主 baseline。其中 LLM Safety Check baseline 的具体做法是：把每一步的状态以结构化文本（机器人速度、torso tilt、最近人体距离、用户距离、台阶置信度、推理延迟、当前任务阶段、候选技能类型和参数）送给 GPT-4o 或本地 Llama-3-70B，让模型用 chain-of-thought 推理输出 allow / reject / defer 加风险等级和理由，并辅以 5 个 few-shot 示例覆盖 allow / reject / defer 三种典型情况，包括“看起来安全但实际危险”和“看起来危险但实际安全”两类 hard case。这个 baseline 用来正面回答审稿人一定会问的“既然 LLM 这么强，为什么不直接用 LLM 当 safety checker”——预期 LLM 在语义常识上够用，但在连续几何细节（具体距离、速度、倾角的边界）和长时程一致性（多次决策的累积错误）上不如校准后的 critic。

评价指标分为三类。第一类是安全指标（safety metrics），由独立 safety oracle 根据仿真 ground truth 判定，不可用 CASA 自身输出当指标，包括错误放行危险技能的次数（unsafe-invocation count，第一关键指标）、ground-truth safety violation rate、碰撞 / 近碰撞、摔倒 / 近摔倒、用户距离违规、不安全手势、台阶失败、运行时超时等。第二类是任务指标（task metrics），包括任务成功率、任务完成时间、被拒绝的技能调用次数、fallback 次数、恢复成功率。这里要注意，CASA 可能不是最快的，目标是用可接受的速度换取更少的安全违规。第三类是 critic 指标（critic metrics），包括 AUROC、AUPRC、Brier score、Expected Calibration Error、false negative rate、conformal coverage 和 reliability diagram。最关键的实验结果应该是 long-horizon coverage decay：不只看单次技能调用是否安全，还要看连续 10 次、20 次、30 次技能调用后，不同方法的安全覆盖率（cumulative empirical coverage）如何衰减。这里 cumulative empirical coverage 的操作定义是：截至第 k 次决策，所有被放行执行的技能中实际未发生 violation 的比例，目标接近 1 − α（例如 α = 0.1 对应 90% 覆盖）；我们对比 marginal calibration、Bonferroni 分配（每步 α / K）和 CASA online conformal 三种 allocation 策略。

预期结果是：Direct Executor 速度可能最快，但安全违规最多；Hard Contract Only 能过滤明显危险动作，但对复杂连续风险处理不足；Raw Critic 能减少部分违规，但风险分数不可靠；Temperature scaling、Quantile Regression、CVaR 等方法可以改善校准，但难以处理不同技能之间风险分布差异和长时程累积；Recovery RL 验证 safe RL critic 在混合 action 空间和长时程下的具体障碍；LLM Safety Check 在语义层可用但在几何细节和长时程一致性上不足；CASA 通过 per-skill Mondrian conformal calibration、survival / hazard 时间结构和 OOD fallback，在降低安全违规、控制 false negative rate、保持 long-horizon coverage 方面表现更好。最终实验要证明：校准后的风险预测（calibrated risk prediction）比原始神经网络风险分数更适合用于人形机器人技能调用决策——特别是在长时程任务中，CASA 能在错误放行率受控的同时，维持可接受的任务完成率。 换句话说，CASA 的价值不是让机器人“更会做动作”，而是让机器人在长时程任务中更知道“什么时候该做，什么时候不该做”。

Phase 0：先跑通 SONIC 原始链路
目标很简单：先确认 SONIC 在 MuJoCo 里能稳定跑。
主要做：
跑通 SONIC 的 MuJoCo sim2sim / deploy demo
确认 walk、idle、turn、gesture 能正常执行
确认 ZMQ 通信不崩
确认机器人能连续运行
同时做一个关键检查：
测试 5cm / 8cm / 10cm / 12cm / 15cm box
看 SONIC 到底能不能跨过去
选择一个既不是 100% 成功、也不是 100% 失败的高度
作为第一版 stair-placeholder
这一阶段的意义：
先确认底层执行器能用，不然 CASA 没法接。

---
Phase 1：封装 SONIC 技能接口
目标：把 SONIC 的底层命令包装成统一的 skill API。
第一版只做这些技能：
walk
turn / face user
gesture
passive stop / wait
stair-placeholder
统一成：
Skill type
Skill parameters
Execution duration
Execution status
Termination reason
例如：
walk(vx, vy, vyaw, duration)
gesture(amplitude, frequency, side, duration)
stair(box_height, approach_speed, duration)
wait(duration)
这一阶段的意义：
CASA 不直接操作 SONIC 底层命令，而是判断这些 skill 是否允许被调用。

---
Phase 2：建立 Safety Oracle 和日志系统
目标：自动判断一次 skill 调用是否安全。
Oracle 要判断：
是否碰撞
是否近碰撞
是否摔倒
是否接近摔倒
用户距离是否太近
gesture 是否不安全
stair 是否失败
runtime 是否超时
每次 skill 调用后记录：
调用前状态
skill 类型
skill 参数
未来几秒是否出事
什么时候出事
出了什么类型的问题
这一阶段的意义：
没有 oracle，就没有训练标签，也没有评价指标。

---
Phase 3：跑 SONIC-only baseline
目标：先不加 CASA，只让 SONIC 按任务脚本执行。
任务可以设计成：
起点
→ walk 穿过障碍区域
→ turn 面向用户
→ stop
→ gesture
→ walk 接近 box
→ stair-placeholder
→ turn
→ walk 返回
记录：
任务成功率
unsafe invocation 次数
碰撞 / 摔倒 / 不安全 gesture / stair failure 次数
完成时间
失败视频
这一阶段的意义：
得到最重要的 baseline：SONIC-only。
后面所有方法都要和它比。

---
Phase 4：自动收集 CASA 训练数据
目标：自动生成 skill invocation dataset。
数据形式是：
当前状态 + 候选 skill + skill 参数 → 未来是否安全
第一版不强行做完整 counterfactual 回滚，而是采用 Hybrid 策略：
主数据：
    随机化初始状态 + 单 branch rollout

小规模分析数据：
    少量 counterfactual subset
也就是说，大规模训练数据这样采：
随机生成一个状态
随机选一个 skill 和参数
执行未来 H 秒
用 safety oracle 自动打标签
保存样本
同时做少量：
同一个状态下测试 walk / turn / gesture / stair / wait
看不同 skill 的风险差异
这一阶段的意义：
先把数据稳定采出来，不把项目卡死在复杂的 state rollback 上。

---
Phase 4.5：小数据 quick train，确认数据能学
目标：在正式采 5 万到 10 万数据之前，先用 5000 条数据训练一个 mini critic。
判断标准：
如果 AUROC ≥ 0.65：
    说明数据有学习信号，可以扩量采集。

如果 AUROC ≈ 0.55：
    说明模型几乎随机猜，不能继续盲目采数据。
如果失败，要回头查：
是不是输入特征不够
是不是 oracle 标签太吵
是不是危险样本太少
是不是某些 skill 数据不平衡
这一阶段的意义：
防止采完 10 万条数据才发现根本学不到东西。

---
Phase 5：训练 CASA Safety Critic
目标：训练核心模型。
输入：
机器人状态
用户状态
环境状态
运行时状态
任务阶段
skill type
skill parameters
输出：
未来 H 秒内出事的风险
第一版先训练 Raw Critic：
state × skill parameter → risk
然后加 Hazard Head：
不仅判断会不会出事
还判断大概什么时候出事
例如：
gesture 风险可能出现在前 1 秒
walk 风险可能出现在后 2 秒
stair 风险可能出现在踩 box 的中段
这一阶段的意义：
CASA 开始具备“判断这个 skill 现在能不能调用”的能力。

---
Phase 6：加入 Conformal Calibration 和 OOD Fallback
目标：让风险分数可信，而不是只输出一个神经网络分数。
先做 per-skill conformal calibration：
walk 一个阈值
turn 一个阈值
gesture 一个阈值
stair 一个阈值
passive 一个阈值
原因：
不同 skill 的风险分布不一样。
gesture 的风险和用户距离有关。
stair 的风险和台阶高度、姿态稳定性有关。
walk 的风险和障碍物、人群、速度有关。
不能所有 skill 共用一个阈值。
再做 long-horizon sequential calibration：
marginal conformal
Bonferroni allocation
online conformal
用来比较连续 20 次 skill 调用之后，安全覆盖率怎么衰减。
然后加入 OOD fallback：
如果场景超出训练分布
或者模型不确定
就不强行 allow
而是 fallback 到 stop / wait / reobserve
这一阶段的意义：
CASA 不只是会预测风险，而是能给出更可信、更保守的安全调用决策。

---
Phase 7：完整实验和对比
目标：证明 CASA 真的比 baseline 更安全。
主要对比：
SONIC-only
SONIC + Hard Contract
SONIC + Raw Critic
SONIC + Temperature Scaling
SONIC + Global Conformal
SONIC + Safety Q-function
SONIC + LLM Safety Check
SONIC + CASA
主要指标：
unsafe invocation count
ground-truth violation rate
false negative rate
task success rate
fallback count
rejection count
completion time
long-horizon coverage decay
最终要证明：
CASA 比 SONIC-only 更安全
CASA 比手写规则更灵活
CASA 比 raw critic 更可信
CASA 比 global conformal 更适合多技能
CASA 比 safety Q-function 更能控制长时程风险
CASA 比 LLM safety checker 更稳定
这一阶段的意义：
完成论文主实验。