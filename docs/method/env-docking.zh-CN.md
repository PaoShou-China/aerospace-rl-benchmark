# 对接环境（`SpaceDockingEnv`）

**简体中文** · [English](env-docking.md)

## 1. 概述

对接任务复现 SpaceGym 近距操作套件（Supreeth & Mehta，CVPR 2026 AI4Space Workshop，第 3.1 节）的近距离最终逼近基准。追踪器初始距目标 60–120 m（目标在 400 km 圆轨道），须在 200 m 距离上限与 2 kg 推进剂预算内，接近到 1 m 以内且终端相对速度低于 0.1 m/s。相对运动在目标 LVLH 系下遵循 Clohessy–Wiltshire–Hill（CWH）线性化方程，以 1 s 固定步长辛 Euler 积分；环境完全张量化，所有副本在同一设备推进，无外部物理引擎。

## 2. 符号

| 符号 | 含义 | 值 |
|---|---|---|
| $\mu$ | 地球引力常数 | $3.986\times10^{14}$ m³/s² |
| $a$ | 目标轨道半长轴 | $6.778\times10^{6}$ m（400 km 圆轨道） |
| $n$ | 目标平均角速度，$\sqrt{\mu/a^{3}}$ | $1.1314\times10^{-3}$ rad/s |
| $m$ | 追踪器质量 | 500 kg |
| $F_{\max}$ | 单轴推力上限 | 10 N |
| $I_{sp}$ | RCS 比冲 | 300 s |
| $\Delta t$ | 积分步长 | 1 s |
| $p,v,F\in\mathbb{R}^{3}$ | LVLH 相对位置 / 速度 / 推力 | — |

LVLH 系以目标为原点：$x$ 径向向外，$y$ 沿迹，$z$ 交轨。

## 3. 动力学

带推力强迫的 CWH 方程（Clohessy & Wiltshire, 1960）：
\begin{align}
\ddot{x} &= 3n^{2}x + 2n\dot{y} + F_{x}/m, \tag{1a}\\
\ddot{y} &= -2n\dot{x} + F_{y}/m, \tag{1b}\\
\ddot{z} &= -n^{2}z + F_{z}/m. \tag{1c}
\end{align}
记 $G=\mathrm{diag}(3n^{2},0,-n^{2})$、$H=\big[\begin{smallmatrix}0&2n&0\\-2n&0&0\\0&0&0\end{smallmatrix}\big]$，加速度为 $\dot v=Gp+Hv+F/m$。积分器为参考论文的辛 Euler 格式（Algorithm 1），先更新速度：
\begin{align}
v_{t+1} &= v_t + \Delta t\,(G p_t + H v_t + F_t/m), \tag{2a}\\
p_{t+1} &= p_t + \Delta t\,v_{t+1}. \tag{2b}
\end{align}
等价地，堆叠状态 $s=(p,v)$ 的一步是线性映射
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
\Delta f_t=\frac{\lVert F_t\rVert\,\Delta t}{I_{sp}\,g_0},\qquad g_0=9.80665\ \mathrm{m/s^2}. \tag{4}
\end{equation}

## 4. 动作空间

| 属性 | 值 |
|---|---|
| 维度 | 3 |
| 范围 | $a\in[-1,1]^{3}$，逐元裁剪 |
| 缩放 | 逐轴 $F=F_{\max}\cdot a$（10 N/轴） |
| 燃料 | 按式 (4) 扣减；预算 $f_0=2$ kg |

三轴满推力持续整个 500 步回合将消耗 $\sqrt{3}F_{\max}T/(I_{sp}g_0)\approx2.94$ kg，因此预算只在近饱和轨迹上起作用；越界才是主要失败模式。

## 5. 观测空间

观测为未归一化的 6 维 LVLH 相对状态
\begin{equation}
o_t=[\,p_t^{\top},\ v_t^{\top}\,]^{\top}\in\mathbb{R}^{6}, \tag{5}
\end{equation}
以 float32 置于 TensorDict 的 `policy` 键下；不做任何归一化、缩放或坐标变换。

## 6. 奖励

奖励为基于"制动距离增广距离"的伸缩式势函数塑形。定义势函数
\begin{equation}
\Phi(s)=-\frac{\lVert p\rVert+\lVert v\rVert^{2}/(2a_{\mathrm{ref}})}{d_{\mathrm{norm}}},\qquad
a_{\mathrm{ref}}=0.6\frac{F_{\max}}{m}=0.012\ \mathrm{m/s^{2}},\qquad
d_{\mathrm{norm}}=100\ \mathrm{m}. \tag{6}
\end{equation}
每步奖励
\begin{equation}
r_t=w\big(\Phi(s_{t+1})-\Phi(s_t)\big)-\gamma\frac{\lVert F_t\rVert}{F_{\max}},\qquad w=100,\ \gamma=0.01. \tag{7}
\end{equation}
势函数塑形使回合求和伸缩：
\begin{equation}
\sum_{t=0}^{T-1}w\big(\Phi(s_{t+1})-\Phi(s_t)\big)=w\big(\Phi(s_T)-\Phi(s_0)\big), \tag{8}
\end{equation}
累计塑形与路径无关：悬停近似零成本，接近与制动为正，后退为负，往返净零。二次速度项使奖励梯度与减速度 $a_{\mathrm{ref}}$ 下的最大制动逼近剖面一致，抑制以不可恢复速度到达。成功时加终端奖励 $R_{+}=+100$；约束违反施加一次性惩罚 $R_{-}=-100$（无逐步费用）。

## 7. 成功、终止与截断

每步按优先级判定：

1. **违反（终止）**：$f_t\ge f_0$（燃料耗尽）**或** $\lVert p_t\rVert>200$ m。置 `terminated=True`；施加 $R_{-}$。
2. **成功（截断）**：$\lVert p_t\rVert<1$ m **且** $\lVert v_t\rVert<0.1$ m/s **且** 无违反。置 `truncated=True` 与内部 `complete`；施加 $R_{+}$。成功**不**置 `terminated`。
3. **超时（截断）**：步数达 500（约 8.3 min 仿真时间）。置 `truncated=True`，无奖励无惩罚。

优先级冻结：终止步上的约束违反永远是失败——违反优先于成功，同满足目标与违反的状态判为失败。`extras['time_outs'] = truncated & ~terminated & ~complete`，仅标记纯超时截断，符合 Gymnasium "时限不同于失败与达成目标"的约定。

## 8. 初始分布

reset 时位置方向在单位球面均匀、距离在任务区间均匀；速度方向独立均匀、大小在 $[0,v_{\max}]$ 均匀：
\begin{equation}
p_0=d\hat u,\quad d\sim\mathcal U[60,120]\ \mathrm{m},\quad \hat u\sim\mathcal U(S^{2}),\qquad
v_0=s\hat w,\quad s\sim\mathcal U[0,0.5]\ \mathrm{m/s},\quad \hat w\sim\mathcal U(S^{2}), \tag{9}
\end{equation}
$\hat u,\hat w$ 独立。燃料、步计数器与全部回合标志清零；未激活副本冻结且不消耗随机数。

## 9. 指标与 extras

回合完成时，`extras['episode']` 报告 `success_rate`（该步以 `complete` 结束的回合占比）与 `init_range_hi`；`extras['terminal_observations']` 携带 float32 终态观测。

## 10. 实现与关键常量

- 文件：`environments/satellite/rendezvous_docking/env.py`；类 `SpaceDockingEnv`（共享 `_CWHBase` 的子类）；任务 id `docking`。
- 传统基线：`DockingCWHLQRBaseline`（CWH-LQR 终端逼近 + 距离护盾 + 捕获网）。

| 常量 | 值 |
|---|---|
| `max_steps` | 500（缓冲 `max_episode_length` 502） |
| `init_range` | (60, 120) m（2026-10-24 由 (50, 100) m 放宽；基线原为 100%） |
| `init_vmax` | 0.5 m/s |
| `pos_goal` / `vel_goal` | 1 m / 0.1 m/s |
| `range_fail` | 200 m |
| `fuel0` | 2 kg |
| `pos_norm`（$d_{\mathrm{norm}}$） | 100 m |
| `brake_ref`（$a_{\mathrm{ref}}$） | 0.012 m/s² |
| `progress_weight` / `gamma` | 100 / 0.01 |
| `terminal_bonus` / `viol_penalty` | +100 / −100 |

## 11. 参考

- Supreeth, H. S. G. & Mehta, "SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement Learning in Spacecraft Proximity Operations," *CVPR 2026 Workshop on AI4Space*（CC BY 4.0）——任务表、动力学与积分器复现自第 3 节；无公开参考实现。
- Clohessy, W. H. & Wiltshire, R. S. (1960), "Terminal Guidance System for Satellite Rendezvous," *Journal of the Aerospace Sciences*, 27(9), 653–658.
