1. 概要

版本 A 是 CASA 的最小可发表版本。它把"机器人什么时候才应该启动这个动作"这件事剥离出来，作为一个独立的、可验证的研究问题先做扎实，而不是一上来就把 hazard、OOD、sequential conformal、long-horizon coverage、LLM baseline、safe RL baseline 全部塞进同一篇论文。版本 A 的核心 claim 只有一个：在 humanoid skill invocation 这个问题上，**神经网络直接输出的风险分数不能直接当成真实概率使用，按技能分组做 conformal 风险校准（per-skill Mondrian conformal）比共用一个全局阈值或者只依赖原始 critic 分数更适合 humanoid 技能调用**。换句话说，版本 A 想用一个干净的实验证明：calibrated risk score 比 raw neural critic score 更适合用作 humanoid skill invocation gate，而且按技能分组校准比全局校准更适合 humanoid 多技能场景。

版本 A 把 CASA 定位为一个最小的"安全顾问"层：在机器人执行每一个候选技能之前，CASA 先看一眼当前状态（机器人姿态、速度、关节状态、周围障碍物、用户位置、任务阶段、感知与推理延迟）和候选技能的具体参数（走路速度、转身角度、手势幅度），然后给出一个未来固定窗口 H 内发生安全违规的风险分数；这个分数不直接拿来用，而是先经过一层按技能分组的 conformal 校准，再与 calibration 阈值比较，输出 allow 或 reject。版本 A 不预测风险发生在动作的哪个阶段，也不主动检测分布外状态，更不做长时程累积错误的控制；这些都留给版本 B 和版本 C。版本 A 的 reject 之后的 fallback 也很简单：固定退到 passive stop 或 passive wait，不做 reobserve、不做 graceful mid-execution interrupt。

版本 A 之所以这样裁剪，是因为一篇可发表论文最关键的不是塞满所有创新点，而是能否把一个核心 claim 用一组干净、可重复、可解释的实验讲清楚。所以版本 A 的工程目标也很明确：在 SONIC + MuJoCo 链路上，跑通 4 个 skill 的完整 skill invocation pipeline，自动采集 50k 量级的 skill invocation samples，训练一个 skill-parameterized safety critic，做 per-skill conformal calibration，并和 4 个直接对手（SONIC-only、Hard Contract、Raw Critic、Global Conformal）做严格对比。


---

2. Related Work

版本 A 的对手范围比完整版小，related work 主要分三条线。

第一条线是 humanoid 底层控制和动作能力（humanoid control and motion capability）。HOVER、SONIC、GMT 这类工作主要解决"怎么让人形机器人把动作做好"，提升 whole-body control 和 motion tracking 能力，让机器人可以更稳定地走路、转身、做复杂全身动作。Switch 则更关注"动作之间怎么切换"。版本 A 不重新训练任何底层 controller，直接把 SONIC 作为固定的 skill executor 使用。CASA 与这条线是上下游互补关系：HOVER / SONIC 是底层动作执行器，CASA 是动作执行之前的安全判断层。底层控制器负责把动作做出来，CASA 负责判断现在做这个动作是否安全。版本 A 不与这条线竞争，也不需要它做对照实验，而是直接复用它作为执行器。

第二条线是行为树、状态机和任务规划（Behavior Tree / FSM / PDDL），它们可以把任务组织成"导航 → 停止 → 讲解 → 继续导航"这样的流程，也能写 fallback 或 recovery 节点。版本 A 的 Hard Contract baseline 就是这条线最直接的代表：用人工写的几何与运行时规则（最近障碍距离、用户距离、torso 倾角、控制周期超时率、skill 参数边界）作为前置过滤器。版本 A 想正面证明的事就是：**对连续变化的、多维耦合的、参数化技能调用风险，人工规则在 reject rate 受控时不可避免会漏检**。所以 Hard Contract 在 A 版本中必须作为强 baseline 真实实现，而不是 strawman。

第三条线是 conformal prediction 与神经网络风险预测的可信化。Conformal prediction（Vovk 2005、Angelopoulos & Bates 2023 综述）让模型输出的风险分数有统计意义上的覆盖保证；Lindemann 2023 等近期工作已经把 conformal 用到机器人轨迹安全验证上。版本 A 想做的是这条线在 humanoid skill invocation 这个具体问题上的第一次系统应用：不是把 conformal 用到 trajectory-level，而是用到 skill-level；不是用全局 conformal，而是用按技能分组的 Mondrian conformal。版本 A 不与 conformal sequential prediction、ACI、Faithful CP 等长时程变体直接对照，这些属于版本 C。

版本 A 不与 safe RL（CPO、Lagrangian PPO、Recovery RL）做主对照，也不与 LLM safety check 做主对照。版本 A 的 related work 中会简要提到这两条线，并明确把它们的"完整对照"放到版本 C，理由是这两条线的 baseline 一旦做得不充分就会被审稿人指为不公平比较，而把 CPO 或 Lagrangian PPO 训练到收敛本身就是一个独立的大工程。版本 A 选择不背这个工程债。


---

3. 方法与创新点

版本 A 的方法核心是一个 skill-parameterized safety critic，加一层 per-skill Mondrian conformal calibration，外加一个并联的 Hard Contract Filter 作为快速规则过滤。整体决策流程是：给定当前状态和一组候选技能，每个候选先经过 Hard Contract Filter，明显违规的直接被剔除；剩下的候选送入 safety critic 估计未来 H 秒内的违规概率，并经过对应技能的 conformal 阈值比较；如果通过，则 allow；否则 fallback 到 passive stop / passive wait。版本 A 的 H 按 skill 分别取"技能预计执行时间 + 一段安全缓冲"，例如 gesture 约 2 秒，H 取 3 秒；walk 单步 invocation 约 1.5 秒，H 取 2.5 秒。

版本 A 的 critic 输入主要分三类。第一类是机器人自身状态，包括身体姿态、速度、关节状态、脚底接触、tracking error，以及一些 humanoid 特有的稳定性指标，例如 CoM、ZMP proxy、capture point proxy、support polygon proxy、torso pitch / roll 等。第二类是环境和用户状态，包括最近障碍距离、最近人体距离、用户是否可见、用户与机器人的相对位置、是否处于窄通道等。第三类是任务和运行时状态，包括当前任务阶段、候选技能类型、技能参数、推理延迟、感知延迟和控制周期超时率。技能不能只用一个 skill id 表示，因为同样是 gesture，大幅度挥手和小幅度指向的风险完全不同；同样是 walk，高速走和低速走的风险也不同。所以版本 A 的 critic 输入是 skill type embedding 加 skill 参数向量一起送入网络，输出一个标量 violation probability。

版本 A 的 critic 不输出 hazard 时间结构，也不做 ensemble。它就是一个干净的 (state × skill type × skill params → P(violation in H))。

版本 A 的核心创新点有两个。第一个是 formulation 创新：**把 humanoid 技能调用安全建模为"状态 × 技能参数 → 固定窗口内违规概率"的预测问题**。和把 safety 直接挂在 trajectory level、控制 level 或者 end-to-end policy level 的做法相比，这种 formulation 把"现在该不该调用这个 skill"这个决策点显式分离出来，让 safety critic 可以独立训练、独立校准、独立评估，同时不要求接管底层控制器。也和普通的 safety Q-function 不同：输入端显式建模"离散 skill type + 连续 skill 参数"的混合 action 空间，让 critic 能区分同类技能的不同参数变体，例如让 critic 知道 amplitude=20° 和 amplitude=90° 的 gesture 是同一个 skill type 但是两种不同的风险等级。这两个 formulation 选择不是为了好看，而是为了让后面的 per-skill calibration 真正能落地，没有这个 formulation，per-skill 校准没东西可校准。

第二个是按技能分组的 conformal 风险校准（per-skill Mondrian conformal risk control）。不同 humanoid skill 的风险分布天然差很多：gesture 的风险和用户距离高度相关，walk 的风险和障碍密度、人群速度有关，turn 的风险和侧向障碍距离有关，passive 的风险几乎只受运行时因素影响。如果把所有 skill 混在一起做 global conformal，得到的阈值要么对低风险 skill 过度保守（导致 walk 频繁被拒）、要么对高风险 skill 过度激进（导致 gesture 经常漏检）。版本 A 把 conformal 校准按 skill type 分组（walk / turn / gesture / passive 各一个阈值），让每个 skill 在自己的 calibration sub-distribution 上独立标定阈值，从而在每个 skill 上单独控制错误放行率（false negative rate）。在 distribution-free 假设下，这个保证是 per-skill 的 marginal coverage，不依赖具体网络架构和数据生成分布。版本 A 同时承诺一件事：因为 calibration set 必须在 Hard Contract 已经过滤的子分布上重新采，才能让 calibration distribution 和 deployment distribution 一致，避免 selection bias 破坏 exchangeability。

版本 A 不主张这两点是 conformal prediction 这个工具本身的创新；conformal 的算法部分已经被前人做过。版本 A 主张的是 **这个 formulation 第一次被系统应用到 humanoid skill invocation 上，并通过严格的 per-skill calibration 实验证明 calibrated risk 是 humanoid skill invocation gate 的更好选择**。


---

4. 实验设计

版本 A 的实验只做仿真，仿真器为 MuJoCo，底层执行器为 SONIC。任务设计为一个中等长度的 humanoid 交互任务，不含 stair：机器人从起点出发，沿着含动态障碍的走道行进，靠近用户后转身并面向用户，在合适的距离停下，做一个指示性手势，然后转身、沿原路返回起点。这个任务一共包含约 8 到 12 个连续技能调用决策点，覆盖 walk、turn、passive、gesture 四个 skill type，能体现"什么时候启动一个 skill"这件事的连续性，但不要求 long-horizon coverage decay（这是版本 C 的事）。版本 A 不依赖 stair skill，因为用 walk policy 翻越一块 box 不是真正的 humanoid stair skill，与其勉强加进来不如等版本 C 把 stair 单独处理。

版本 A 的对比方法只做 5 个，并且全部部署在同一个 SONIC executor 上，确保唯一变量是 skill invocation gate：第一个是 SONIC-only，按任务脚本直接调用技能，不做任何安全判断；第二个是 SONIC + Hard Contract，使用人工规则过滤明显违规的 skill 调用；第三个是 SONIC + Raw Critic，使用 skill-parameterized safety critic 输出的 raw 概率，按固定阈值（例如 0.5）做 reject；第四个是 SONIC + Global Conformal，使用同一个 critic，但 calibration 把所有 skill 合并成一个 group 标定阈值；第五个是 SONIC + CASA-A，本版本方法，per-skill Mondrian conformal calibration。这五个 baseline 的关系是严格嵌套的：从无 gate 到人工规则、从原始分数到全局校准、从全局校准到分组校准，每一步只引入一个变量，让最终结果能干净归因到 per-skill calibration 这个具体选择。

版本 A 的评价指标分为三类。第一类是安全指标，由独立 safety oracle 根据 MuJoCo ground truth 判定，不能用 CASA 自己输出当指标，包括错误放行危险技能的次数（unsafe-invocation count，第一关键指标）、ground-truth safety violation rate、碰撞 / 近碰撞、摔倒、用户距离违规、不安全手势、运行时超时。第二类是任务指标，包括任务成功率、任务完成时间、被拒绝的技能调用次数、fallback 次数；这里要明确，CASA 不追求"最快"，而是要在可接受任务成功率下显著降低 unsafe invocation。第三类是 critic 指标，包括 AUROC、AUPRC、Brier score、Expected Calibration Error、per-skill empirical false negative rate、reliability diagram（marginal 和 per-skill 两版）。版本 A 的主结果图必须包括一张 **rejection-risk Pareto 曲线**：x 轴为 rejection rate（从 0% 到 30%），y 轴为 unsafe invocation count 或 task success rate，五个 baseline 各画一条曲线；这张图比单一指标表更能说明 CASA 不是靠多 reject 来赢的。

版本 A 的预期结果是：SONIC-only 速度最快但 unsafe invocation 最多；Hard Contract 能挡掉明显违规但对连续变化的多维风险处理不足，typically 在 reject rate 受控时仍漏检；Raw Critic 能减少 unsafe invocation，但 reliability diagram 显示其概率分数与真实违规频率显著不一致（典型表现为 over-confidence），按固定阈值做 reject 在不同 skill 上 false negative rate 差异巨大；Global Conformal 整体覆盖率达标，但分技能看会出现某些 skill（典型为 gesture）显著超过 alpha，而其他 skill（典型为 passive）显著保守；CASA-A 在每个 skill 上都把 empirical false negative rate 控制在 alpha 附近，并在相近 rejection rate 下显著降低 unsafe invocation count。版本 A 不需要、也不应该尝试证明这一切来自 hazard、OOD、sequential 等机制——这些版本 A 没有。版本 A 唯一要证明的是：**per-skill calibrated risk 是 humanoid skill invocation gate 的更好选择**。

版本 A 最终要回答的问题：在已经存在的 humanoid skill executor 之上加一层校准过的 safety critic，能否在 unsafe invocation 和 task success 之间取得比 SONIC-only、Hard Contract、Raw Critic、Global Conformal 都更好的 trade-off？版本 A 的答案如果是 yes，那么这个 yes 是后续版本 B 和版本 C 的所有扩展（hazard、OOD、sequential、long-horizon、stair）的共同前提。版本 A 不证明这个前提，后面所有扩展都没有立足点。
