# Rendezvous 基线：滚动时域 CWH 两脉冲制导（`RendezvousTwoImpulseBaseline`）

**简体中文** · [English](baseline-rendezvous-two-impulse.md)

## 1. 概述

交会任务的传统基线是在 Clohessy–Wiltshire–Hill（CWH）模型上递归重解的两脉冲终端制导律。每个控制步，在锁存保持点处的零努力脱靶量（ZEM）与零努力速度（ZEV）用环境自身积分器的**精确辛状态转移矩阵**传播（滑行弧上零模型误差），求解同时消除两项终端误差的脉冲对，并通过环境连续逐轴推力器把该对实现为最小能量最优制导程序的第一步。转移时间由加速度受限界播种，并在顺序点火无法在到达前完成时在线递增；不可行日程退化为速度归零。保持点位于初始视线 9 m 处，介于 5 m 禁飞球与 10 m 成功球之间。

## 2. 接口约定

- **观测**：仅 `'policy'` 组：原始 6 维 LVLH 相对状态 $o=[p,v]$（$x$ 径向、$y$ 沿迹、$z$ 交轨）。
- **构造期环境常量**：$\Delta t=1$ s，`max_steps`$=1000$，$m=500$ kg，$F_{\max}=10$ N/轴——即 $a_{\mathrm{axis}}=F_{\max}/m=0.02$ m/s² 每轴、$\Delta v_{\mathrm{step}}=a_{\mathrm{axis}}\Delta t=0.02$ m/s 每步——及任务半径 `keep_out`$=5$ m、`pos_goal`$=10$ m。平均角速度 $n=\sqrt{\mu/a^3}=1.1314\times10^{-3}$ rad/s（$\mu=3.986\times10^{14}$ m³/s²，$a=6.778\times10^{6}$ m）。
- **逐回合状态**（由 `reset_idx(mask)` 以环境 reset 掩码清零）：锁存保持点 $r_f$、到达日程 $T_{\mathrm{arr}}$ [步]、不可行标志与步计数器。
- **动作**：$u\in[-1,1]^3$，逐轴归一化推力 $F=F_{\max}u$，$\Delta t=1$ s 辛 Euler。两个"脉冲"均通过显式逐轴饱和映射实现为**有限**点火。

## 3. 控制律

**精确离散 STM**：环境辛 CWH plant 的无推力一步映射（对接姊妹文档式 (1)）为 $A_d\in\mathbb{R}^{6\times6}$。离散生成元 $\mathcal{L}=\log(A_d)/\Delta t$ 在构造时经 `scipy.linalg.logm` 提取一次（拒绝负实轴谱），任意逐副本时域 $T$ 的转移矩阵
\begin{equation}
\Phi(T)=\exp(\mathcal{L}T)=\begin{bmatrix}\Phi_{rr}&\Phi_{rv}\\\Phi_{vr}&\Phi_{vv}\end{bmatrix}\in\mathbb{R}^{6\times6}, \tag{1}
\end{equation}
在设备端用 `torch.matrix_exp` 批量计算。$A_d$ 的指数精确传播无推力 plant，不同于连续 CWH 指数。

**目标与日程**：回合首步在初始视线锁存保持点
\begin{equation}
r_f=\rho_h\frac{p_0}{\lVert p_0\rVert},\quad
\rho_h=\mathrm{keep\_out}+0.8(\mathrm{pos\_goal}-\mathrm{keep\_out})=9\ \mathrm{m}, \tag{2}
\end{equation}
终端速度为零。转移时间以加速度受限直线界播种
\begin{equation}
T_{\mathrm{arr}}\leftarrow\max\Big(\frac{2\sqrt{\lVert p_0\rVert/a_{\mathrm{axis}}}}{\Delta t},10\Big)\ \text{[步]}. \tag{3}
\end{equation}

**两脉冲求解**（每步、从当前状态、对剩余时间 $T_{\mathrm{go}}$）。零努力误差
\begin{align}
\mathrm{ZEM}&=r_f-(\Phi_{rr}p+\Phi_{rv}v), \tag{4a}\\
\mathrm{ZEV}&=-(\Phi_{vr}p+\Phi_{vv}v), \tag{4b}
\end{align}
脉冲对（当前位置脉冲、到达速度脉冲）
\begin{equation}
\Delta v_1=\Phi_{rv}^{-1}\mathrm{ZEM},\qquad
\Delta v_2=\mathrm{ZEV}-\Phi_{vv}\Delta v_1, \tag{5}
\end{equation}
在离散 plant 下精确满足 $r(T)=r_f$、$v(T)=0$。滚动时域重解吸收饱和时序、float32 观测量化与单步推力施加的 $\Delta t^2$ 位置滑移。

**病态保护**：$\Phi_{rv}$ 在 $nT=k\pi$（半轨道整数倍，远超本任务时域；据实现 docstring，所有 $T\le1000$ s 下 $\mathrm{cond}(\Phi_{rv})\le1.8$）接近奇异流形。两种防御：
\begin{align}
\text{若}\ |\det\Phi_{rv}|<\frac{10^{-10}}{n^3}:&\quad \Phi_{rv}\leftarrow\Phi_{rv}+\frac{10^{-6}}{n}I\quad\text{（Tikhonov）}, \tag{6a}\\
\text{若}\ \big|nT_{\mathrm{arr}}\Delta t-k\pi\big|<10^{-3},\ k=\mathrm{round}\big(\tfrac{nT_{\mathrm{arr}}\Delta t}{\pi}\big):&\quad T_{\mathrm{arr}}\leftarrow T_{\mathrm{arr}}+8\quad\text{（流形跳离）}. \tag{6b}
\end{align}

**点火顺序日程递增**：两次顺序点火须在到达前完成。点火时长 $\lVert\Delta v_i\rVert_\infty/\Delta v_{\mathrm{step}}$、余量 $M=10$ 步：
\begin{equation}
\mathrm{need}=t+\frac{\lVert\Delta v_1\rVert_\infty+\lVert\Delta v_2\rVert_\infty}{\Delta v_{\mathrm{step}}}+M;\quad
T_{\mathrm{arr}}\leftarrow\min(\mathrm{need},T_{\mathrm{cap}}),\ T_{\mathrm{cap}}=\mathrm{max\_steps}-8=992, \tag{7}
\end{equation}
当 $\mathrm{need}>T_{\mathrm{arr}}$ 时更新；若 $\mathrm{need}$ 超过 $T_{\mathrm{cap}}$ 则标记不可行。随后以更新后的 $T_{\mathrm{go}}$ 重解。

**脉冲 → 有限推力映射（OGL 首步）**：脉冲对在剩余时间内铺展为最小能量推力程序，其首步为 ZEM/ZEV 最优制导律
\begin{equation}
a=\frac{6\,\mathrm{ZEM}}{T_{\mathrm{go}}^2}-\frac{2\,\mathrm{ZEV}}{T_{\mathrm{go}}},\qquad
u=\operatorname{clip}(a/a_{\mathrm{axis}},-1,1)\ \text{（逐轴）}. \tag{8}
\end{equation}
CWH 重力偏置由 STM 携带在 ZEM/ZEV 内，重解每步修正残差。程序前段点火（脉冲 1 功）、中段滑行（剖面过零）、到达时制动进入保持点（脉冲 2 功）。

**退化模式**：日程不可行或已过期（$T_{\mathrm{arr}}\le t$）时，制导退化为速度归零
\begin{equation}
u=\operatorname{clip}(-v/\Delta v_{\mathrm{step}},-1,1). \tag{9}
\end{equation}

## 4. 参数

| 参数 | 值 | 来源 |
|---|---|---|
| 保持半径 $\rho_h$ | 9 m $=5+0.8(10-5)$ | `self.hold` |
| $a_{\mathrm{axis}}$、$\Delta v_{\mathrm{step}}$ | 0.02 m/s²、0.02 m/s 每步每轴 | `env.f_max/mass` |
| 时间播种 | $2\sqrt{\lVert p_0\rVert/a_{\mathrm{axis}}}/\Delta t$，下限 10 步 | `t_seed` |
| 余量 $M$ / 上限 $T_{\mathrm{cap}}$ | 10 步 / 992 步 | `ext_margin`、`t_cap` |
| Tikhonov 阈值 | $10^{-10}/n^3$（det）、$10^{-6}/n$（平移） | `_plan` |
| 流形跳离 | $|nT-k\pi|<10^{-3}$ ⇒ $+8$ 步 | `actions` |
| OGL 钳位 | 逐轴 $\pm a_{\mathrm{axis}}$ | `actions` |
| $n$ | $1.1314\times10^{-3}$ rad/s | `env.n` |

## 5. 实测表现

全部发布记录使用 `scripts/baseline_eval.py` 记账（每副本首回合、`env_kwargs={}`、1000 步时域）。更早的"4096 回合 100% / 1024/1024、回报 195.44、559 步"条目为 2026-10-22 冻结审计快照；检出中无 1024 或 4096 副本交会产物，不可复验，已由下列可验证记录取代。

**协议数据集 A**（`results/baseline_protocol/seed{7,13,21,101}/rendezvous/baseline.json`，每 seed 128 副本）：

| seed | 成功 | 成功/完成 | 平均回合回报 | env 步数 |
|---|---:|---:|---:|---:|
| 7 | 100% | 128/128 | 184.85 | 594 |
| 13 | 100% | 128/128 | 184.92 | 581 |
| 21 | 100% | 128/128 | 184.62 | 627 |
| 101（留出） | 100% | 128/128 | 184.88 | 649 |

协议 seeds 7/13/21：**100%（384/384）**，平均回报 184.79，平均 601 env 步，每个文件 `complete=true`。

**已发布单记录**：`results/rendezvous/baseline.json` 为 seed-13 协议记录：128 副本、seed 13、581 env 步、**100%（128/128）**、平均回合回报 **184.92**、`complete=true`。

**数据集 B**（`results/rendezvous/baselines/*_seed7.json`，256 副本、seed 7）：

| 臂 | CLI 键 | 成功 | 平均回报 | env 步数 |
|---|---|---|---:|---:|---:|
| `cwh_two_impulse`（默认） | — | 100%（256/256） | 184.91 | 594 |
| `cwh_compensated_pd` | `cwh_pd` | 100%（256/256） | 179.44 | 875 |
| `glideslope` | `glideslope` | 100%（256/256） | 180.96 | 805 |

三臂均解名义任务；`cwh_compensated_pd` 与 `glideslope` 窗口显著更大（875 / 805 vs 594 步）反映无两脉冲日程的较慢直线接近，而非不同成功判据。

冻结协议（`docs/evaluation_protocol.zh-CN.md` §1）记录终止优先级修复（终端步禁飞区违反即失败、永不算成功）不改变本基线：9 m 保持点在 5 m 禁飞球之外。冻结审计的 150–1900 m 场景网格（`eval_rendezvous_baseline.py`）未作为产物发布，故仅上述名义协议记录作为测量引用。

## 6. 已知局限

- 保持点锚定在**初始**视线；制导只重规划到达日程，从不重规划目标本身。
- 脉冲经逐轴饱和映射实现，故有限推力实现仅在饱和前精确；滚动时域重解是修正机制（有记录），而非解析饱和补偿。
- $\Phi_{rv}$ 在 $nT=k\pi$ 有真实奇异流形；由正则化与日程跳离（式 6）处理而非消除，时域上限使任务在设计上远离它。
- 无法装入时域的日程不被挽救；控制器退化为速度归零（式 9），回合判负。

## 7. 实现

- 控制器：`RendezvousTwoImpulseBaseline`（注册名 `cwh_two_impulse`、`BUILDERS['rendezvous']`）位于 `algorithms/baselines/baselines.py`；共享 STM helper `_cwh_symplectic_Ad`。
- 环境：`environments/satellite/rendezvous_docking/env.py`，类 `SpaceRendezvousEnv`（`conventional_controller = RendezvousTwoImpulseBaseline`）。
- 评测：`scripts/baseline_eval.py rendezvous`；场景网格 harness `environments/satellite/rendezvous_docking/eval_rendezvous_baseline.py`。

## 8. 参考

1. Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance System for Satellite Rendezvous", *Journal of the Aerospace Sciences* 27(9), 653–658 —— 相对运动模型及其转移矩阵。
2. Battin, R. H. *An Introduction to the Mathematics and Methods of Astrodynamics*（AIAA）—— ZEM/ZEV 最优制导首步所依据的最小能量拦截制导。
3. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement Learning in Spacecraft Proximity Operations", CVPR 2026 Workshop on AI4Space —— 任务定义。
