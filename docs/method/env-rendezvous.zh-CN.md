# 交会环境（`SpaceRendezvousEnv`）

**简体中文** · [English](env-rendezvous.md)

## 1. 概述

交会任务复现 SpaceGym 近距操作套件（Supreeth & Mehta，CVPR 2026 AI4Space Workshop，第 3.2 节）的中距逼近基准。追踪器初始距目标 1000–1500 m（目标在 400 km 圆轨道），须在避开 5 m 半径禁飞球、留在 2000 m 内、并遵守 4 kg 推进剂预算的条件下，接近到 10 m 以内且终端相对速度低于 0.5 m/s。相对运动在目标 LVLH 系下遵循 CWH 线性化方程，以 1 s 固定步长辛 Euler 积分；环境完全张量化（无引擎）。

## 2. 符号

| 符号 | 含义 | 值 |
|---|---|---|
| $\mu$ | 地球引力常数 | $3.986\times10^{14}$ m³/s² |
| $a$ | 目标轨道半长轴 | $6.778\times10^{6}$ m（400 km 圆轨道） |
| $n$ | 目标平均角速度 $\sqrt{\mu/a^{3}}$ | $1.1314\times10^{-3}$ rad/s |
| $m$ | 追踪器质量 | 500 kg |
| $F_{\max}$ | 单轴推力上限 | 10 N |
| $I_{sp}$ | RCS 比冲 | 300 s |
| $\Delta t$ | 积分步长 | 1 s |
| $p,v,F\in\mathbb{R}^{3}$ | LVLH 相对位置 / 速度 / 推力 | — |

LVLH 系以目标为原点：$x$ 径向向外，$y$ 沿迹，$z$ 交轨。

## 3. 动力学

带推力强迫的 CWH 方程：
\begin{align}
\ddot{x} &= 3n^{2}x + 2n\dot{y} + F_{x}/m, \tag{1a}\\
\ddot{y} &= -2n\dot{x} + F_{y}/m, \tag{1b}\\
\ddot{z} &= -n^{2}z + F_{z}/m. \tag{1c}
\end{align}
记 $G=\mathrm{diag}(3n^{2},0,-n^{2})$、$H=\big[\begin{smallmatrix}0&2n&0\\-2n&0&0\\0&0&0\end{smallmatrix}\big]$，$\dot v=Gp+Hv+F/m$。积分器为参考论文的辛 Euler 格式（Algorithm 1），先更新速度：
\begin{align}
v_{t+1} &= v_t + \Delta t(G p_t + H v_t + F_t/m), \tag{2a}\\
p_{t+1} &= p_t + \Delta t\,v_{t+1}. \tag{2b}
\end{align}
等价地，堆叠状态 $s=(p,v)$ 的一步为线性映射
\begin{equation}
\begin{bmatrix}p_{t+1}\\ v_{t+1}\end{bmatrix}
=
\underbrace{\begin{bmatrix}I+\Delta t^{2}G & \Delta t(I+\Delta t H)\\
\Delta t\,G & I+\Delta t\,H\end{bmatrix}}_{A_d}
\begin{bmatrix}p_t\\ v_t\end{bmatrix}
+
\underbrace{\frac{1}{m}\begin{bmatrix}\Delta t^{2}I\\ \Delta t\,I\end{bmatrix}}_{B_d}F_t ,
\tag{3}
\end{equation}
$\Delta t=1$ s。推进剂实时累计到每副本燃料计数器 $f_t$：
\begin{equation}
\Delta f_t=\frac{\lVert F_t\rVert\,\Delta t}{I_{sp} g_0},\qquad g_0=9.80665\ \mathrm{m/s^2}. \tag{4}
\end{equation}

## 4. 动作空间

| 属性 | 值 |
|---|---|
| 维度 | 3 |
| 范围 | $a\in[-1,1]^{3}$，逐元裁剪 |
| 缩放 | 逐轴 $F=F_{\max}\cdot a$（10 N/轴） |
| 燃料 | 按式 (4) 扣减；预算 $f_0=4$ kg |

三轴满推力持续整个 1000 步回合将消耗 $\sqrt{3}F_{\max}T/(I_{sp}g_0)\approx5.89$ kg，故预算只在近饱和轨迹上起作用；禁飞区与越界违反占主导。

## 5. 观测空间

观测为未归一化的 6 维 LVLH 相对状态
\begin{equation}
o_t=[\,p_t^{\top},\ v_t^{\top}\,]^{\top}\in\mathbb{R}^{6}, \tag{5}
\end{equation}
以 float32 置于 TensorDict 的 `policy` 键下。

## 6. 奖励

奖励为基于"制动距离增广距离"的伸缩式势函数塑形。定义势函数
\begin{equation}
\Phi(s)=-\frac{\lVert p\rVert+\lVert v\rVert^{2}/(2a_{\mathrm{ref}})}{d_{\mathrm{norm}}},\qquad
a_{\mathrm{ref}}=0.1\ \mathrm{m/s^{2}},\qquad d_{\mathrm{norm}}=d_0, \tag{6}
\end{equation}
其中 $d_0=\lVert p_0\rVert$ 为每回合初始距离，使归一化适配每次出生距离。参考减速度超过单轴推力权限 $F_{\max}/m=0.02$ m/s²，在加速接近阶段保留大部分进展信用，同时仍惩罚目标附近的不可恢复接近速度。每步奖励
\begin{equation}
r_t=w\big(\Phi(s_{t+1})-\Phi(s_t)\big)-\gamma\frac{\lVert F_t\rVert}{F_{\max}}-\tau,\qquad w=100,\ \gamma=0.01,\ \tau=0.02. \tag{7}
\end{equation}
逐步时间（悬停）惩罚 $\tau=0.02$ 是必需的：伸缩塑形是时不变的，在数百米处悬停几乎零成本，否则会成为有利可图的失败策略（完整 1000 步悬停约 $-20$，对 $+100$ 成功奖励；稠密时间成本与终端奖励的混合沿用 SpaceGym CVPRW 2026 / AAS 23-026 表述）。势函数塑形使回合求和伸缩：
\begin{equation}
\sum_{t=0}^{T-1}w\big(\Phi(s_{t+1})-\Phi(s_t)\big)=w\big(\Phi(s_T)-\Phi(s_0)\big), \tag{8}
\end{equation}
累计塑形与路径无关：接近与制动为正，后退为负，往返净零，悬停每步损失 $\tau$。二次速度项使奖励梯度与减速度 $a_{\mathrm{ref}}$ 下的最大制动逼近剖面一致。成功时加终端奖励 $R_{+}=+100$；违反施加一次性惩罚 $R_{-}=-100$（无逐步费用；progress 模式下基类的稠密 `keep_out_penalty=10` **不**启用，禁飞区穿透改由该终端惩罚计费）。100 分差于 2026-10-28 恢复：20 分差时终端事件对 2×128-MLP PPO 配方从不成为吸引子（两次完整运行中贪心成功率卡在约 0.5–2% 的出生运气下限），而 100 分差是唯一观测到达到 0.3–0.9 成功率的设置；其后期侵蚀由全批更新（`num_mini_batches=1`）与 400 次更新的 LR 半衰期抵消。

## 7. 成功、终止与截断

1. **违反（终止）**：$f_t\ge f_0$（燃料耗尽）**或** $\lVert p_t\rVert>2000$ m **或** $\lVert p_t\rVert<5$ m（禁飞区穿透）。置 `terminated`；施加 $R_{-}$。
2. **成功（截断）**：$\lVert p_t\rVert<10$ m **且** $\lVert v_t\rVert<0.5$ m/s **且** 无违反。置 `truncated` 与内部 `complete`；施加 $R_{+}$。永不置 `terminated`。
3. **超时（截断）**：步数达 1000（约 16.7 min）。置 `truncated`，无奖励无惩罚。

优先级冻结：终止步上的违反永远是失败——违反优先于成功。该排序至关重要：5 m 禁飞球位于 10 m 目标球**内部**，以捕获速度穿透禁飞区的状态否则会满足目标而被计为成功；在冻结规则下它是不带奖励的禁飞区失败。`extras['time_outs'] = truncated & ~terminated & ~complete`：仅纯超时，符合 Gymnasium 时限约定。

## 8. 初始分布

reset 时位置方向在单位球面均匀、距离在任务区间均匀；速度方向独立均匀、大小在 $[0,v_{\max}]$ 均匀：
\begin{equation}
p_0=d\hat u,\quad d\sim\mathcal U[1000,1500]\ \mathrm{m},\quad \hat u\sim\mathcal U(S^{2}),\qquad
v_0=s\hat w,\quad s\sim\mathcal U[0,1]\ \mathrm{m/s},\quad \hat w\sim\mathcal U(S^{2}), \tag{9}
\end{equation}
$\hat u,\hat w$ 独立。采样距离存为 $d_0=\lVert p_0\rVert$，作为该回合的奖励归一化量。燃料、步计数器与回合标志清零；未激活副本冻结且不消耗随机数。

## 9. 指标与 extras

回合完成时，`extras['episode']` 报告 `success_rate`（该步以 `complete` 结束的回合占比）与 `init_range_hi`；`extras['terminal_observations']` 携带 float32 终态观测。

## 10. 实现与关键常量

- 文件：`environments/satellite/rendezvous_docking/env.py`；类 `SpaceRendezvousEnv`（共享 `_CWHBase` 的子类）；任务 id `rendezvous`。
- 传统基线：`RendezvousTwoImpulseBaseline`（滚动时域 CWH 两脉冲制导）。

| 常量 | 值 |
|---|---|
| `max_steps` | 1000（缓冲 `max_episode_length` 1002） |
| `init_range` | (1000, 1500) m（2026-10-24 由 (800, 1200) m 放宽；基线原为 100%） |
| `init_vmax` | 1 m/s |
| `pos_goal` / `vel_goal` | 10 m / 0.5 m/s |
| `range_fail` | 2000 m |
| `keep_out` | 5 m |
| `fuel0` | 4 kg |
| `pos_norm`（$d_{\mathrm{norm}}$） | 每回合 $d_0$ |
| `brake_ref`（$a_{\mathrm{ref}}$） | 0.1 m/s² |
| `progress_weight` / `gamma` | 100 / 0.01 |
| `time_penalty`（$\tau$） | 每步 0.02 |
| `terminal_bonus` / `viol_penalty` | +100 / −100 |
| `keep_out_penalty` | 10（仅稠密模式；progress 模式使用终端违反惩罚） |

## 11. 参考

- Supreeth, H. S. G. & Mehta, "SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement Learning in Spacecraft Proximity Operations," *CVPR 2026 Workshop on AI4Space*（CC BY 4.0）；任务表、动力学与积分器复现自第 3 节（无公开参考实现）。
- Clohessy, W. H. & Wiltshire, R. S. (1960), "Terminal Guidance System for Satellite Rendezvous," *Journal of the Aerospace Sciences*, 27(9), 653–658.
