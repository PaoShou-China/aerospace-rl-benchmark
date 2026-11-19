# 行为克隆热启动（`bc_warmstart`）

**简体中文** · [English](algo-bc-warmstart.md)

## 1. 概述

基准可选地用传统基线控制器热启动 PPO，而非随机网络初始化（`algorithms/warmstart/bc_warmstart.py`）。流程：在任务 `conventional_controller` 下滚动训练环境，把 actor 的确定性策略均值（默认有界 Beta 策略的均值）回归到收集的专家动作，可选地把 critic 回归到同一批 rollout 的经验折扣回报，然后由 PPO 继续精炼。热启动按设计仅支持 PPO：回归目标是 PPO 的状态无关策略均值，且 `run_training` 只支持 PPO。

## 2. 数据收集

`collect_expert(env, steps)` 持有专用控制器实例，以 `env.conventional_controller(env)` 构造，并直接驱动它——环境本身仍是纯张量环境，无控制器状态、`extras` 中无控制器字段。在 $T=$`steps` 个向量化步中，收集器在 `torch.no_grad()` 下由当前观测计算专家动作，精确执行该动作，并把环境自动 reset 镜像到控制器有状态部分：
\begin{equation}
a_t=\pi^*(s_t),\qquad (s_{t+1},r_t,d_t)=\mathrm{env.step}(a_t),\qquad
\mathrm{ctrl.reset\_idx}(d_t),\quad d_t=\mathrm{terminated}_t\lor\mathrm{truncated}_t, \tag{1}
\end{equation}
首步前对所有副本做完整 `reset_idx`。因执行动作**就是**控制器指令，控制器内部自积分（对接燃料记账、quadx 修剪、fixedwing 积分、rendezvous 棘轮）与访问状态保持一致，存储对 $(s_t,\pi^*(s_t))$ 是专家自身闭环轨迹上的精确监督目标。专家动作在收集时校验：形状 $(N,A)$ 与设备须匹配环境，首步须有限并满足环境动作界（`|a| ≤ clip_actions + 1e-6`，默认界 $1$）；违反立即抛错。收集器返回设备张量 $\mathrm{obs}(T,N,d_{\mathrm{obs}})$、$\mathrm{act}(T,N,A)$（float32）、$\mathrm{rew}(T,N)$、$\mathrm{done}(T,N)$（bool，标记该步环境自动 reset），以及供筛选使用的 $\mathrm{terminated}(T,N)$ 与 $\mathrm{success}(T,N)$ 掩码。`steps=None` 表示跑完整回合（`env.max_steps`）。环境未声明 `conventional_controller` 的任务抛 `RuntimeError`。

## 3. 拟合

`fit_behavior_cloning` 把 rollout 展平为 $M=T\cdot N$ 对，用每 epoch 新置换和 `batch_size` 的 mini-batch 跑监督 epochs。actor 损失为模型确定性输出——策略均值（默认分布下 Beta 均值）——与专家动作的 MSE：
\begin{equation}
L_{\mathrm{actor}}=\frac1{|\mathcal B|}\sum_{i\in\mathcal B}\big\lVert\mu_\theta(s_i)-a_i^*\big\rVert_2^2, \tag{2}
\end{equation}
由 actor 参数上的独立 Adam 以 `lr` 优化。`fit_critic=True` 时 critic 回归到同一 rollout 的经验 $\gamma$ 折扣回报：
\begin{equation}
G_t=r_t+\gamma(1-d_t)G_{t+1},\qquad
L_{\mathrm{critic}}=\frac1{|\mathcal B|}\sum_{i\in\mathcal B}\big(V_\phi(s_i)-G_i\big)^2, \tag{3}
\end{equation}
由边界感知的反向递归计算（`returns_to_go`）：累加器在每个回合边界切断，成功（环境以 truncated 记录）与任务违约视为真实终止、终端价值为零，纯超时尾部被标记为 invalid 并从 critic 中排除（除非提供 bootstrap 值）。critic 参数上用第二个 Adam。两个模型的输入归一化器在 BC 数据上每批 `update_normalization()` 重新拟合，与 PPO 自身更新路径一致，故热启动也把输入白化校准到专家状态分布。探索标准差刻意不动：它是 PPO 的探索旋钮，不是 BC 目标。每 `max(1, epochs//10)` epochs 记录进度，拟合返回摘要（`transitions`、`epochs`、`batch_size`、`lr`、`fit_critic`、`wall_seconds`、`final_actor_mse`、`final_critic_mse`）。

## 4. 与 `run_training` 集成

热启动为 opt-in：逐任务入口经 `run_training(..., bc_warmstart={...})` 在 `BC_DEFAULTS` 上传入覆写 dict，故各任务 BC 配方可直接比较。`gamma=None` 解析为 PPO 配置折扣（仅用于 critic 的回报目标）。收集发生在环境构造与 checkpoint 续训之后、任何 PPO 更新之前；当 `resume` 实际加载 checkpoint 时自动跳过（checkpoint 会覆盖热启动）。拟合后环境重置，其回合计数（`completed_episodes`、`completed_successes`、`completed_returns` 及每个 `completed_metrics` 项）清零，故专家 rollout 抬高的计数不会泄漏进 PPO 报告，`run.json` 只反映 PPO 自己的回合。合并配置与拟合报告——以及实测 `collect_wall_seconds`——持久化到 `config.json` 的 `bc_warmstart` 段，BC 成本与 RL 训练预算分开报告（BC 收集步不计入 PPO 交互步）。

## 5. 默认超参

`BC_DEFAULTS`：`steps=512` 个向量化专家环境步、`epochs=100` 监督 epochs、`batch_size=8192`、`lr=10^{-3}`（actor——与 critic——参数上的独立 Adam）、`gamma=None`（$\to$ PPO 配置 $\gamma$）、`fit_critic=True`。`steps=None` 表示从环境读取整回长度（`env.max_steps`）；专家回合超过 512 步的任务（quadx、quadx_track、fixedwing、quadplane、rocket、missile、rendezvous）均使用该设置，确保专家数据覆盖完整轨迹而非仅开头几秒。长回合的大部分是低信息量的稳态飞行，故 `select` 只保留信息量最高的转移，做法对应演示数据筛选文献：按 `done` 掩码在每个副本内定位 episode，再对每个 episode 等分 8 个阶段（广覆盖比单纯挑“难”样本更重要——*Less is More: On Data Redundancy in VLA Training*），每副本预算跨 episode 与阶段分配。终端型任务（rocket、missile、docking、rendezvous）按 Temporal Surprise Score（TSS，专家动作变化幅度）排序（动作变化点即 *Keyframe-Focused Visual Imitation Learning* 中的 keyframe），并在 ±2 个时间邻居上做 max-pool，使关键帧以稠密簇保留；跟踪/巡航型任务（quadx、quadx_track、fixedwing、quadplane、satellite、transfer）使用分层均匀覆盖，因为冗余研究显示这类任务广覆盖优于难度排序。仅对**真实回合结束**（成功或任务违约，不含超时）的结束前若干步加分，保证精度关键的终端接近段不被稀疏化（对应 *ESPADA* 保留关键阶段）。默认 `drop_failed=True` 丢弃未成功的专家回合（若整个 split 无成功则回退为保留全部），对应去除次优演示的筛选文献；当成功回合占比低于 10% 时自动禁用过滤，避免近零成功率的专家把数据删光。Critic 目标先在完整轨迹上计算，再用同一索引 gather，故筛选不会使回报目标产生偏差；`select=None` 则训练全部采集步，`method='uniform'` / `'tss'` 可显式覆盖按任务的选择。数据集拟合分支（`bc_warmstart['dataset']`，即 controller-prior 研究的 `collect_bc.py` / `collect_dagger.py` → `train_arm.py` 管线）通过 `select_informative_indices_flat` 在每个 episode 内应用同一套筛选，并用向量化的 `returns_to_go_flat` 计算边界感知目标；episode 不相交的 train/val 划分保留。`bc_val_fraction>0` 会按副本留出 BC 验证集（仅监控，协议禁止早停），可用 `scripts/bc_selection_ablation.py` 在 held-out 专家动作上比较不同筛选方案。标准 $N=1024$ 副本下最多产生 $M=512\times1024=524{,}288$ 监督对，每 epoch 64 mini-batches，每网络 $100\times64=6400$ 梯度步。逐任务训练入口已默认启用筛选后的热启动（长回合任务 `BC_WARMSTART` 的 `steps=None`）。共享 BC checkpoint 带配方指纹（`BC_RECIPE_VERSION` + 解析后的 selector），配方变化时自动重训。BC 继承冻结专家上限——审计控制器成功率在 quadx/docking/rendezvous/transfer/satellite 为 100%，fixedwing 约 75%，rocket 约 84%——故热启动臂与专家成功率一并报告。

## 6. 有记录的限制：火箭

BC 热启动在火箭着陆任务上存在结构性困难。火箭 PDG 专家以终端 bang-bang 脉冲伺服剖面结束自杀燃烧，MSE 回归把该开关结构平均掉：蒸馏策略输出推力指令的时间平均而非脉冲，闭环不可执行。2026-10-24 在奖励演化 harness（`environments/rocket/evolve_reward.py`，generation g1）实测：所有 BC 臂 0% 成功，BC 闭环成功率 0%、零平台触地。当前筛选配方会录完整 60 s 回合并强调点火/着陆燃烧的变化点（rocket 用 `method='tss'`），但这不消除 bang-bang 回归限制；热启动火箭臂应作为经验问题，用 `scripts/bc_selection_ablation.py` / 闭环评测验证，不能视为必然收益。`evolve_reward.py` 仍可按臂关闭 BC（`_nb` 后缀）。这是 MSE 蒸馏近 bang-bang 专家的结构限制，不是其他任务平滑专家的缺陷。

## 7. 实现

- 文件：`algorithms/warmstart/bc_warmstart.py`（`collect_expert`、`returns_to_go` / `returns_to_go_flat`、`select_informative_pairs` / `..._indices_flat` / `..._indices_rollout`、`fit_behavior_cloning`、配方指纹与 sidecar）；接线在 `algorithms/train_utils.py:run_training`；筛选消融在 `scripts/bc_selection_ablation.py`。

| 常量 | 值 |
|---|---|
| `steps`（专家 rollout 长度 $T$） | 512，或 `None`（整回 + 信息筛选） |
| `epochs` / `batch_size` | 100 / 8192 |
| `lr`（独立 Adam，actor 与 critic） | $10^{-3}$ |
| `gamma`（critic RTG 折扣） | `None` $\to$ PPO 配置 $\gamma$ |
| `fit_critic` | `True` |
| $N{=}1024$ 时对数 / 每网络梯度步 | 524,288 / 6400 |
| obs / act dtype | float32，设备驻留 |
| 仅 PPO / 续训自动跳过 | 是 / 是 |

## 8. 参考

- Pomerleau, D. A. (1991), "Efficient Training of Artificial Neural Networks for Autonomous Navigation," Neural Computation —— 行为克隆。
- Ross, S., Gordon, G. J., & Bagnell, D. (2011), "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS, arXiv:1011.0686 —— 动机为专家轨迹上收集的协变量漂移分析。
- Açıkmeşe, B., & Ploen, S. R. (2007), "Convex Programming Approach to Powered Descent Guidance for Mars Landing," JGCD —— 第 6 节所引火箭专家所属 PDG 家族。
- Schulman, J. et al. (2017), "Proximal Policy Optimization Algorithms," arXiv:1707.06347 —— 精炼热启动的 RL 阶段。
- Wen, C. et al. (2021), "Keyframe-Focused Visual Imitation Learning," ICML —— 对专家动作变化点加权；keyframe/TSS 排序依据。
- "Less is More: On Data Redundancy in VLA Training," OpenReview —— Temporal Surprise Score，以及“广覆盖可优于难度筛选”的结论。
- "ESPADA: Execution Speedup via Semantics Aware Demonstration Data Downsampling for Imitation Learning," arXiv:2512.07371 —— 保留精度关键阶段、压缩过渡帧。
- "SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning," OpenReview —— 去除次优与冗余演示（`drop_failed` 依据）。
