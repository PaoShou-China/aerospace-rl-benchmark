# QuadX-Waypoints 环境（`QuadXWaypointsEnv`）

**简体中文** · [English](env-quadx-waypoints.md)

## 1. 概述

四旋翼航路点任务：Crazyflie 2.x 微型四旋翼（27 g，X 构型）从 $(0,0,1)$ m 静止起飞，在 5 m 飞行穹顶内按序访问 4 个随机航路点，进入 0.2 m 球即捕获，时限 20 s；触地或出界失败。由上游 PyFlyt 0.29.0 `QuadXWaypointsEnv` 移植到 Genesis 刚体后端并完全张量化。奖励为基于剩余路径长度的势函数伸缩塑形，另有捕获奖励与坠毁惩罚。风为可选构造参数，默认静风，固定种子可逐位复现航路点布局。

## 2. 符号

| 符号 | 含义 |
|---|---|
| $p,v\in\mathbb{R}^{3}$；$q=(w,x,y,z)$ | 世界位置 / 速度；姿态四元数 |
| $\omega_b,v_b\in\mathbb{R}^{3}$ | 体系角速度 / 线速度 |
| $w_k\in\mathbb{R}^{3}$，$k=0..3$ | 航路点位置；$i_t$ 为当前活动索引 |
| $a\in[-1,1]^{4}$；$\bar a$ | 原始策略动作及其物理映射 |
| $\Delta t_\pi,\Delta t_c,\Delta t$ | 策略 $1/40$ s、控制 $1/120$ s、物理 $1/240$ s |

## 3. 动力学与控制层级

三层速率：策略 40 Hz；每策略步执行 `inner_steps = 3` 个 120 Hz 控制周期；每控制周期做一次内环更新，再走 2 个 240 Hz 物理子步（每策略步 6 个子步）。

**内环 rate PID（120 Hz）** 跟踪指令角速度 $\omega^{\mathrm{cmd}}$，输出归一化力矩 $u\in[-1,1]^3$：
\begin{equation}
u_t=\mathrm{sat}\Big(k_p e_t + I_t + k_d\,(e_t-e_{t-1})/\Delta t_c\Big),\qquad
I_t=\mathrm{sat}\big(I_{t-1}+k_i\,e_t\,\Delta t_c\big), \tag{1}
\end{equation}
$e_t=\omega^{\mathrm{cmd}}-\omega_b$，$k_p=(0.04,0.04,0.08)$，$k_i=(5\times10^{-7},5\times10^{-7},2.7\times10^{-4})$，$k_d=(10^{-4},10^{-4},0)$，饱和限为 ±1。力矩指令与总距 $T\in[0,0.8]$ 经上游 X 构型 mixer
$M=\big[\begin{smallmatrix}-1&-1&-1&1\\1&1&-1&1\\1&-1&1&1\\-1&1&1&1\end{smallmatrix}\big]$
（电机顺序 $(+x,-y),(-x,+y),(+x,+y),(-x,-y)$）分配到 4 个电机，再按保分布方式缩放到 pwm 窗口 $[0.05,1]$。

**作动（240 Hz）** 电机状态一阶滞后 + 乘性噪声：
\begin{equation}
\theta_{t+1}=\theta_t+\frac{\Delta t}{\tau_m}(\pi_t-\theta_t),\qquad
\theta_t\leftarrow\theta_t\big(1+0.02\,\xi_t\big),\qquad \xi_t\sim\mathcal N(4,1), \tag{2}
\end{equation}
$\tau_m=0.01$ s，每个物理步每副本一次高斯采样（上游约定；非零均值作为常数油门直流增益吸收）。单电机推力二次律 $f_i=\lvert\theta_i\rvert\theta_i\,F_{\max}/4$（$F_{\max}=2.0$ N 总量），反扭矩 $\tau_{z,i}=f_i(k_m/k_f)s_i$，$k_m/k_f=7.94\times10^{-12}/3.16\times10^{-10}$，旋向 $s=(-1,-1,1,1)$。体阻力 $F_d=-\tfrac12\rho C_d A\,\lvert v_{\mathrm{air}}\rvert\circ v_{\mathrm{air}}$（$C_d=3.0$，每轴 $A=4\times10^{-4}$ m²）与角阻力 $\tau_d=-10^{-4}\lvert\omega_b\rvert\circ\omega_b$（接触时抑制）。求解器保护：$\lVert v\rVert$、$\lVert\omega\rVert$ 仅在超过 15 m/s、30 rad/s 时缩放。

## 4. 动作空间

原始动作裁剪到 $[-1,1]^4$ 后仿射映射 $\bar a=\ell+\tfrac{a+1}{2}(h-\ell)$，$\ell=(-\pi,-\pi,-\pi,0)$，$h=(\pi,\pi,\pi,0.8)$：

| 索引 | 符号 | 物理范围 | 含义 |
|---|---|---|---|
| 0 | $\omega_x^{\mathrm{cmd}}$ | $\pm\pi$ rad/s | 滚转角速度指令 |
| 1 | $\omega_y^{\mathrm{cmd}}$ | $\pm\pi$ rad/s | 俯仰角速度指令 |
| 2 | $\omega_z^{\mathrm{cmd}}$ | $\pm\pi$ rad/s | 偏航角速度指令 |
| 3 | $T$ | $[0,0.8]$ | 总距油门（pwm 单位） |

## 5. 观测空间（37 维，float32，`policy` 键）

| 切片 | 符号 | 单位 | 说明 |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | 体系角速度 |
| [3:7] | $q$ $(x,y,z,w)$ | — | 姿态四元数 |
| [7:10] | $v_b$ | m/s | 体系速度 |
| [10:13] | $p$ | m | 世界位置 |
| [13:17] | $\bar a_{t-1}$ | — | 上一物理动作 |
| [17:21] | $\theta$ | — | 各电机状态 |
| [21:33] | $\Delta_k$ | m | 体系航路点偏差 $R(q)^\top(w_{i_t+k}-p)$，$k=0..3$ |
| [33:37] | $v_k$ | — | 4 个航路点槽有效标志 |

## 6. 奖励

令 $\ell_k=\lVert w_{k+1}-w_k\rVert$，剩余航段和 $L(i)=\sum_{k=i}^{2}\ell_k$，势函数
\begin{equation}
\Phi(s)=-\Big(\big\lVert p-w_{i}\big\rVert + L(i)\Big),\quad i<4;\qquad
\Phi(s)=0\ \text{（四个航路点全部捕获后）}. \tag{3}
\end{equation}
每步奖励
\begin{equation}
r_t=\Phi(s_{t+1})-\Phi(s_t)+15\,c_t-10\,\kappa_t, \tag{4}
\end{equation}
$c_t$ 为该步捕获数（120 Hz 检测，一步可捕获多个），$\kappa_t$ 为坠毁/出界指示。势差使塑形可伸缩（Ng et al. 1999）：绕点盘旋净收益恰为零，成功回合累计恰为参考路径长度；坠毁或超时没收未实现势 $-\Phi(s_T)$，提前结束永远不更划算。$+15$ 保持稀疏事件清晰，$-10$ 强化失败。

## 7. 成功、终止与截断

事件在 120 Hz 子步循环内基于步后状态锁存；时限在策略步后判定一次，故最后控制步不被截短。

1. **终止**：触地或 $\lVert p\rVert>5$ m；记 `terminated`，扣 $-10$。
2. **成功（截断）**：$i_t$ 达到 4；记 `truncated` 与内部 `complete`，不记 `terminated`。
3. **超时（截断）**：800 步（20 s），仅对仍活动副本置 `truncated`。

失败与最后一次捕获同子步发生时按终止处理。`extras['time_outs'] = truncated & ~terminated & ~complete`，仅纯超时。训练器只对 `time_outs` 做价值 bootstrap。

## 8. 初始分布

$p_0=(0,0,1)$ m、单位姿态、零速度（生成噪声默认关闭）。航路点按上游极坐标采样，每副本由计数器式设备 RNG 抽取：
\begin{equation}
\theta,\phi\sim\mathcal U[0,2\pi),\quad
r\sim\mathcal U[1.5,4.8]\ \mathrm{m},\quad
w=\big(r\sin\phi\cos\theta,\ r\sin\phi\sin\theta,\ \max(\lvert r\cos\phi\rvert,0.4)\big). \tag{6}
\end{equation}
航路点抽样是唯一 reset 随机性；固定种子逐位复现布局。重置副本先做 10×2 零指令 warmup（20 物理子步）。

## 9. 扰动

可选风：`steady_wind`（世界系定常风，m/s）、`dryden ∈ [0,1]`、`severity ∈ {light, moderate, severe}`（默认 `moderate`）。启用后 `WindDisturbance` 叠加定常分量与 Dryden 阵风（MIL-F-8785C 低空形式）：各轴一阶成形滤波，300 ft 参考标准差 light `(1.5,1.5,0.7)`、moderate `(3.0,3.0,1.4)`、severe `(6.0,6.0,3.0)` m/s，尺度长度 `(200,200,50)` m，参考空速 15 m/s，滤波器增益使稳态标准差恰为 $\sigma_k\cdot$`dryden`。生成器每策略步推进一次，子步间冻结；所有气动项对 $v_{\mathrm{air}}=v-w$ 评估。默认（`steady_wind=None`、`dryden=0`）无风缓冲，静风动力学逐位不变。

## 10. 指标与 extras

每步设 `extras['time_outs']` 与 `extras['terminal_observations']`（重置前 float32 观测）。有副本结束的步，`extras['episode']` 报告 `success_rate`、`waypoints_reached`、`crash_rate`、`out_of_bounds_rate`、`timeout_rate`；累计计数在 `completed_metrics`（`waypoints_reached`、`crashes`、`out_of_bounds`、`timeouts`）与 `completed_returns` / `completed_episodes` / `completed_successes`。

## 11. 实现与关键常量

- 文件：`environments/multirotor/env.py`，类 `QuadXWaypointsEnv`（任务 id `quadx`）；plant 在 `environments/common/plant.py`（`TensorPlant`），风在 `environments/common/disturbances.py`。
- 传统控制器：`QuadXSE3Baseline`（SE(3) 几何跟踪），类属性 `conventional_controller`。

| 常量 | 值 |
|---|---|
| 策略 / 控制 / 物理频率 | 40 / 120 / 240 Hz（`inner_steps=3`，2 子步） |
| `max_steps` | 800（20 s）；`max_episode_length` 802 |
| 质量 / 最大总推力 | 0.027 kg / 2.0 N |
| 电机滞后 $\tau_m$ / 噪声比 | 0.01 s / 0.02 |
| rate-PID $k_p,k_i,k_d$ | $(.04,.04,.08)$，$(5{\times}10^{-7},5{\times}10^{-7},2.7{\times}10^{-4})$，$(10^{-4},10^{-4},0)$ |
| pwm 窗口 | $[0.05,1]$；总距指令 $\le 0.8$ |
| `CAPTURE_RADIUS` | 0.2 m |
| 出界半径 | 5 m |
| 航路点半径 / 高度地板 | $[1.5,4.8]$ m / 0.4 m（2026-10-24 由 $[1.0,4.5]$ m 放宽，基线已达 100%） |
| 出生状态 | $(0,0,1)$ m 静止水平 |
| `REACH_BONUS` / `CRASH_PENALTY` | +15 / −10 |
| `vel_clamp` | 15 m/s，30 rad/s |

## 12. 参考

- Tai, J. K. et al., *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, v0.29.0 — 上游任务定义、机体模型与控制默认值。
- Genesis contributors, *Genesis* — 刚体仿真后端。
- Ng, A. Y., Harada, D., & Russell, S. (1999), "Policy invariance under reward transformations," *ICML* — 势函数塑形不变性。
- MIL-F-8785C — 可选风扰所用的低空 Dryden 湍流模型。
