# Fixedwing-Waypoints 环境（`FixedwingWaypointsEnv`）

**简体中文** · [English](env-fixedwing-waypoints.md)

## 1. 概述

固定翼航路点任务：2.35 kg、1.6 m 翼展无人机从 $(0,0,10)$ m 以 20 m/s 出发，在 20–65 m 球壳内按序访问 4 个航路点（10 m 捕获球），时限 120 s；触地或出 100 m 球失败。由上游 PyFlyt 0.29.0 移植到 Genesis 张量化 plant。奖励与四旋翼任务同构。风为可选构造参数，静风 + 固定种子可逐位复现布局。

## 2. 符号

| 符号 | 含义 |
|---|---|
| $p,v\in\mathbb{R}^{3}$；$q=(w,x,y,z)$ | 世界位置 / 速度；姿态四元数 |
| $\omega_b,v_b\in\mathbb{R}^{3}$ | 体系角速度 / 线速度 |
| $w_k$，$k=0..3$；$i_t$ | 航路点；当前活动索引 |
| $a\in[-1,1]^{4}$ | 原始策略动作 |
| $\Delta t_\pi,\Delta t_c,\Delta t$ | 策略 $1/30$ s、控制 $1/120$ s、物理 $1/240$ s |
| $\rho$；$\bar q=\tfrac12\rho\lVert v_{\mathrm{air}}\rVert^{2}$ | 空气密度 1.225 kg/m³；动压 |

## 3. 动力学与控制层级

策略 30 Hz；每策略步执行 `inner_steps = 4` 个 120 Hz 控制周期，每周期一次作动更新 + 2 个 240 Hz 物理子步（每策略步 8 子步）。与四旋翼不同，无内环姿态环：策略直接驱动舵面与油门，唯一的子策略动力学是作动器滞后。

**舵面（240 Hz）** 5 个升力面（左/右副翼、平尾、垂尾、主翼）一阶滞后 $\tau_s=0.05$ s：
\begin{equation}
\delta_{t+1}=\delta_t+\frac{\Delta t}{\tau_s}\big(\delta^{\mathrm{cmd}}-\delta_t\big), \tag{1}
\end{equation}
采用 PyFlyt 叶素升力面模型：有限翼展升力线斜率、30% 襟翼弦比 + 粘性修正 $\eta=0.65$、零升迎角（翼/副翼 $-2^\circ$，尾翼 $0^\circ$）、正负失速（翼 $+14^\circ/-9^\circ$，尾翼 $\pm9^\circ$）、蒙皮摩擦阻力 $C_{d0}=0.01$；均在面局部空速 $v_{\mathrm{air}}=v-w$ 上评估，力 $\bar q S(c_l,c_d)$，俯仰力矩 $\bar q S c\,c_m$。

**发动机（240 Hz）** 油门一阶滞后 + 乘性噪声：
\begin{equation}
\theta_{t+1}=\theta_t+\frac{\Delta t}{\tau_m}\big(\theta^{\mathrm{cmd}}-\theta_t\big),\qquad
\theta_t\leftarrow\theta_t\big(1+0.02\,\xi_t\big),\qquad \xi_t\sim\mathcal N(1,1), \tag{2}
\end{equation}
$\tau_m=0.01$ s（每副本每物理步一次高斯采样）。螺旋桨推力 $F=\lvert\theta\rvert\theta\,F_{\max}$（$F_{\max}=18$ N）沿体 $+x$，反扭矩 $F(k_m/k_f)$，$k_m/k_f=7.94\times10^{-12}/3.16\times10^{-10}$。

## 4. 动作空间

裁剪到 $[-1,1]^4$ 后恒等仿射（$\ell=-\mathbf 1$，$h=+\mathbf 1$）。舵面按上游 `surface_assist` 混合，各面符号 $(+1,-1,+1,-1,0)$：

| 索引 | 符号 | 范围 | 含义 |
|---|---|---|---|
| 0 | $\delta_a$ | $[-1,1]$ | 差动副翼（左 $+\delta_a$，右 $-\delta_a$；$\pm30^\circ$） |
| 1 | $\delta_e/\delta_r$ | $[-1,1]$ | 升降舵 $+\delta_e$、方向舵 $-\delta_r$（反号；$\pm20^\circ$） |
| 2 | — | $[-1,1]$ | 死通道：映射到主翼，主翼偏度限为 $0^\circ$ |
| 3 | $\theta^{\mathrm{cmd}}$ | $[-1,1]\to[0,1]$ | 油门，进 plant 前重映射 $(a_3+1)/2$ |

$a_2$ 因上游设计而无效，有效动作维度为 3；该映射为保持一致性原样保留，不应"修正"。

## 5. 观测空间（39 维，float32）

| 切片 | 符号 | 单位 | 说明 |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | 体系角速度 |
| [3:7] | $q$ $(x,y,z,w)$ | — | 姿态四元数 |
| [7:10] | $v_b$ | m/s | 体系速度 |
| [10:13] | $p$ | m | 世界位置 |
| [13:17] | $a_{t-1}$ | — | 上一动作（重映射前） |
| [17:22] | $\delta$ | — | 舵面偏度状态（副翼 L/R、平/垂尾、主翼） |
| [22:23] | $\theta$ | — | 发动机油门状态 |
| [23:35] | $\Delta_k$ | m | 体系航路点偏差 $R(q)^\top(w_{i_t+k}-p)$ |
| [35:39] | $v_k$ | — | 航路点槽有效标志 |

## 6. 奖励

与四旋翼同构的势函数伸缩塑形：
\begin{equation}
\Phi(s)=-\Big(\big\lVert p-w_{i}\big\rVert + L(i)\Big),\quad i<4;\qquad
\Phi(s)=0\ \text{（全部捕获后）}, \tag{3}
\end{equation}
\begin{equation}
r_t=\Phi(s_{t+1})-\Phi(s_t)+15\,c_t-10\,\kappa_t, \tag{4}
\end{equation}
路径无关：绕点盘旋净零，成功回合恰收参考路线长度，坠毁/超时没收 $-\Phi(s_T)$。该奖励替换上游 $1/d$ 塑形（绕行可得无界回报）与 $\pm100$ 终端奖励（3600 步下信用分配不可达）。

## 7. 成功、终止与截断

1. **终止**：触地或 $\lVert p\rVert>100$ m；扣 $-10$。
2. **成功（截断）**：$i_t$ 达到 4；置 `complete`，不置 `terminated`。
3. **超时（截断）**：3600 步（120 s），仅对仍活动副本。

失败与末次捕获同子步时按终止处理。`extras['time_outs'] = truncated & ~terminated & ~complete`，仅纯超时被 bootstrap；坠毁、出界与已完成回合均截断 bootstrap。

## 8. 初始分布

$p_0=(0,0,10)$ m，单位姿态，$v_0=(20,0,0)$ m/s，零角速度。航路点极坐标采样：
\begin{equation}
\theta,\phi\sim\mathcal U[0,2\pi),\quad
r\sim\mathcal U[20,65]\ \mathrm{m},\quad
w=\big(r\sin\phi\cos\theta,\ r\sin\phi\sin\theta,\ \max(\lvert r\cos\phi\rvert,20)\big). \tag{6}
\end{equation}
相对上游 1–90 m 球壳、0.5 m 地板有两点依据：配平约束给出最低可控速度约 11 m/s，最小转弯半径 10–15 m，故距 100 m 出界球 10 m 内的目标无法转回；近地目标会迫使掠地捕获。20–65 m 球壳留下 35 m 出界裕度与 20 m 高度地板。10 m 捕获半径约等于一个最小转弯半径，替代上游 2 m（该转弯半径下近乎不可达，最佳脚本正对通过仍偏 8.6 m）。固定种子逐位复现。重置副本做 10×2 零指令 warmup。

## 9. 扰动

可选风参数与 QuadX 相同（`steady_wind`、`dryden`、`severity`；Dryden 参考标准差、尺度长度、极点与增益同前）。生成器每策略步（$1/30$ s）推进一次，子步间冻结；所有升力面对 $v_{\mathrm{air}}=v-w$ 评估。默认无风，静风动力学逐位不变。

## 10. 指标与 extras

每步设 `time_outs` 与 `terminal_observations`。有副本结束的步，`extras['episode']` 报告 `success_rate`、`waypoints_reached`、`crash_rate`、`out_of_bounds_rate`、`timeout_rate`；累计计数在 `completed_metrics` 与 `completed_returns` / `completed_episodes` / `completed_successes`。

## 11. 实现与关键常量

- 文件：`environments/fixedwing/env.py`，类 `FixedwingWaypointsEnv`（任务 id `fixedwing`）；plant `TensorPlant`，升力面 `environments/common/aerodynamics.py`（`LiftingSurfaceBatch`）。
- 传统控制器：`FixedwingAutopilotBaseline`（L1 横向 + TECS），类属性 `conventional_controller`。

| 常量 | 值 |
|---|---|
| 策略 / 控制 / 物理频率 | 30 / 120 / 240 Hz（`inner_steps=4`，2 子步） |
| `max_steps` | 3600（120 s）；`max_episode_length` 3602 |
| 质量 / 翼展 | 2.35 kg / 1.6 m |
| 最大推力 / 电机滞后 / 噪声比 | 18 N / 0.01 s / 0.02 |
| 舵面滞后 $\tau_s$ | 0.05 s |
| 舵面偏度限（副翼 L/R、平/垂尾、主翼） | $30^\circ,30^\circ,20^\circ,20^\circ,0^\circ$ |
| `CAPTURE_RADIUS` | 10.0 m |
| 出界半径 | 100 m |
| 航路点半径 / 高度地板 | $[20,65]$ m / 20 m |
| 出生状态 | $(0,0,10)$ m，$v_0=(20,0,0)$ m/s |
| `REACH_BONUS` / `CRASH_PENALTY` | +15 / −10 |
| `vel_clamp` | 关闭（`None`） |

## 12. 参考

- Tai, J. K. et al., *PyFlyt*, v0.29.0 — 上游任务定义、机体与升力面气动。
- Genesis contributors, *Genesis* — 刚体后端。
- Ng, Harada & Russell (1999) — 势函数塑形不变性。
- MIL-F-8785C — 低空 Dryden 湍流模型。
