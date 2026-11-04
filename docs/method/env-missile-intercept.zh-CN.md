# Missile Intercept 环境（`MissileInterceptEnv`）

**简体中文** · [English](env-missile-intercept.md)

## 1. 概述

项目自研的 6-DOF 尾控导弹拦截运动学 3-DOF 目标机任务。导弹为 150 kg 细长体（头部沿体 $+x$），带四片十字尾翼、体阻力、细长体法向力模型和自主助推—滑行发动机（每回合前 4 s 沿体 $+x$ 提供 20 kN，其后无动力滑行）。策略指令**体角速度**，由 120 Hz rate PID 经 skid-to-turn mixer 转为舵偏。目标以 250 m/s 在约 4 km 高度执行三类解析机动之一（直线、正弦蛇形、水平桶滚），在物理子步内闭式推进；Genesis 场景中的无碰撞视觉实体仅用于渲染。交战几何为单一固定中等工作点（2026-10-24 决定）：初始距离 2500 m、目标航向锥 $\pm37.5^{\circ}$、目标侧向加速度 7.5 g、蛇形频率 0.75 Hz。导弹在交战高度以火控前置量发射——瞄准目标沿缩放名义剩余飞行时间的直线外推，并以 250 m/s 沿该方位起飞——因此发射即近碰撞航向，制导只修正残差。

## 2. 符号

| 符号 | 含义 |
|---|---|
| $p,q,v,\omega$ | 导弹世界位置、姿态（内部 wxyz，观测 xyzw）、世界速度、角速度；$v_b,\omega_b$ 为体系量 |
| $p_T,v_T$ | 目标世界位置 / 速度；$V_T=\lVert v_T\rVert=250$ m/s |
| $r=p_T-p$，$d=\lVert r\rVert$ | 视线（LOS）向量与距离 |
| $v_{\text{rel}}=v_T-v$；$\lambda=(r\times v_{\text{rel}})/\lVert r\rVert^{2}$ | 相对速度；LOS 角速度 |
| $V_c=-(r\cdot v_{\text{rel}})/\lVert r\rVert$ | 接近速度 |
| $a\in[-1,1]^{3}$；$\bar a$ | 原始动作；缩放后的体角速度指令 [rad/s] |
| $h,n_\perp$ | 目标单位航向及其水平左法向 |
| $a_T$ | 目标侧向加速度，$7.5g=73.575$ m/s² |

## 3. 动力学与控制层级

策略 60 Hz（$\Delta t=1/60$ s），控制 120 Hz（`inner_steps=2`），物理 240 Hz（每控制期 2 子步，$h=1/240$ s）；回合上限 900 策略步（15 s）。

- **弹体**：150 kg 刚体。轴向体阻力 $F_x=-\tfrac12\rho C_{d,x}A_x\lvert v_x\rvert v_x$（$C_{d,x}=0.25$，$A_x=0.0314$ m²，$\rho=1.225$）；细长体法向力 $F_{\text{lat}}=-\tfrac12\rho\lVert v\rVert A_x C_{n\alpha}v_{\text{lat}}$（$C_{n\alpha}=16$），作用于压心 $x_{\text{cp}}=-0.45$ m（质心之后），同时提供转弯力、风标恢复力矩与转动阻尼；合成俯仰/偏航阻尼 $\tau=-c_{mq}\lVert v\rVert\omega_{\text{lat}}$（$c_{mq}=4.0$）抑制原本近乎无阻尼的舵面风标模态。速度钳位：$\lVert v\rVert\le1200$ m/s，$\lVert\omega\rVert\le40$ rad/s。
- **舵面**：四片十字尾翼位于 $x=-1.3$ m（弦长 0.45 m，展长 0.40 m，$C_{l\alpha}=3.0$，失速 $\pm40^{\circ}$，偏度限 $\pm35^{\circ}$，作动滞后 $\tau=0.02$ s）。
- **rate PID**：逐轴比例控制（$k_p=5.0$，$k_i=k_d=0$，输出限 ±1）于 120 Hz 作用在体角速度上，产生归一化（滚/俯/偏）舵指令；高 $k_p$ 补偿强风标弹簧（舵指令→稳态转弯率增益约 0.2–0.3 rad/s 每单位）。
- **Skid-to-turn mixer**：滚转传动比 $g=0.01$（滚转惯量约为俯/偏的 1/150，故滚转通道降传动以保 120 Hz 环路稳定），[左,右,上,下] 舵指令为 $(q_c+g p_c,\ q_c-g p_c,\ -r_c-g p_c,\ -r_c+g p_c)$，裁剪到 $[-1,1]$。
- **发动机**：首个活动物理步自主点火，以 $20\,000\cdot(1+0.01\mathcal N(0,1))$ N 沿体 $+x$ 燃烧 4 s 后滑行；动作不含油门/点火。

## 4. 动作空间

$a\in[-1,1]^3$ 逐元裁剪后仿射映射：

| 索引 | 原始范围 | 映射范围 | 含义 |
|---|---|---|---|
| 0 | $[-1,1]$ | $\pm6.0$ rad/s | 滚转角速度指令（`OMEGA_MAX`） |
| 1 | $[-1,1]$ | $\pm1.0$ rad/s | 俯仰角速度指令（`OMEGA_LAT`） |
| 2 | $[-1,1]$ | $\pm1.0$ rad/s | 偏航角速度指令（`OMEGA_LAT`） |

`OMEGA_LAT` 于 2026-10-24 由 0.5 提到 1.0 rad/s 以利于原生 PPO 学习：稳态协调转弯率约 $a/V\approx0.2$–0.5 rad/s，但末制导需要更高的**瞬态**权限，rate PID 会直接饱和舵面，1.0 rad/s 上限让策略在末段修正时拉得更狠。

## 5. 观测空间（34 维，float32，未归一化）

| 切片 | 符号 | 单位 | 说明 |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | 体角速度 |
| [3:7] | $q$ (xyzw) | — | 姿态四元数 |
| [7:10] | $v_b$ | m/s | 体系速度 |
| [10:13] | $p$ | m | 世界位置 |
| [13:16] | $\bar a_{t-1}$ | rad/s | 上一步角速度指令 |
| [16:20] | — | — | 4 片舵偏 |
| [20:21] | — | — | 发动机点火标志 |
| [21:22] | — | — | 发动机燃烧标志 |
| [22:23] | — | — | 剩余燃烧时间比例 |
| [23:26] | $r$（体） | m | 目标相对位置（体系） |
| [26:29] | $v_{\text{rel}}$（体） | m/s | 相对速度（体系） |
| [29:32] | $\lambda$（体） | rad/s | LOS 角速度（体系） |
| [32:33] | $V_c$ | m/s | 接近速度 |
| [33:34] | $\lVert v_T\rVert$ | m/s | 目标速度 |

## 6. 目标模型

目标为运动学模型，在每个物理子步按每副本仿真时间 $t$（每子步 +1/240 s）解析推进，$p_0$ 为 $t=0$ 位置。三类机动：

**直线（type 0）**
\begin{equation}
p_T(t)=p_0+V_T h\,t,\qquad v_T(t)=V_T h. \tag{1a}
\end{equation}

**正弦蛇形（type 1）** 蛇形角频率 $\omega_w=2\pi\times0.75$ rad/s、随机相位 $\phi$：
\begin{equation}
p_T(t)=p_0+V_T h\,t+n_\perp\frac{a_T}{\omega_w^{2}}\sin(\omega_w t+\phi),\qquad
v_T(t)=V_T h+n_\perp\frac{a_T}{\omega_w}\cos(\omega_w t+\phi). \tag{1b}
\end{equation}

**水平桶滚（type 2）** 半径 $R_b=V_T^{2}/a_T\approx849.5$ m，带符号角速度 $\omega_b=\pm a_T/V_T\approx\pm0.294$ rad/s：
\begin{equation}
p_T(t)=c+R_b(\cos\theta,\sin\theta,0),\quad v_T(t)=V_T(-\sin\theta,\cos\theta,0),\quad \theta=\omega_b t+\phi_b. \tag{1c}
\end{equation}
桶滚相位不自由：要求初始切向与出生航向一致（$v_T(0)\parallel h$）即固定 $\phi_b=\operatorname{atan2}(-h_x,h_y)$（$\omega_b<0$ 时加 $\pi$），圆心沿内法向一个半径处 $c=p_0-R_b(\cos\phi_b,\sin\phi_b,0)$；否则桶滚目标会朝任意方向起飞，交战退化为尾追。

## 7. 奖励

每 120 Hz 内步累计，仅活动副本：

\begin{align}
r_{\text{progress}} &= 0.01\,(d^{-}-d) && \text{伸缩式接近进展}, \tag{2a}\\
r_{\text{align}} &= 0.01\,(\hat v\cdot\hat r) && \text{速度/LOS 对齐}, \tag{2b}\\
r_{\text{kill}} &= +100 && \text{击杀时一次性}, \tag{2c}\\
r_{\text{miss}} &= -10\min(d_{\min}/500\,\text{m},1) && \text{不可恢复脱靶时一次性}. \tag{2d}
\end{align}
$d^{-}$ 为上一内步边界距离。对齐项是与 PN 一致的稠密制导塑形：类势、可局部计算，且不像对 $\lVert\lambda\rVert$ 的惩罚那样可通过飞离目标来刷分。

## 8. 成功、终止与截断

在**每个物理子步**（240 Hz）按优先级判定：

1. **击杀（成功）**：任一子步 $d<$`KILL_RADIUS`$=15$ m。置 `complete`，支付 $r_{\text{kill}}$，在所在策略步末截断。半径于 2026-10-24 由 8 提到 15 m：8 m 时行为克隆热启动的中位脱靶（$d=0$ 时 9.8 m）正好压在引信边界，PPO 一次约 $10^{-3}$ 的权重漂移就会清零所有击杀（实测阻塞项）；15 m 让热启动稳落门内。
2. **不可恢复脱靶（终止）**：最近接近点已过：$r\cdot v_{\text{rel}}>0$（距离增大）且 $t>2$ s。支付一次 $r_{\text{miss}}$。同一内步内击杀优先于脱靶。
3. **出界（终止）**：$\lVert p\rVert>15$ km 或 $z<0$。
4. **时限（截断）**：`count`$\ge900$ 策略步（15 s），在整个控制步之后判定（rocket 语义），击杀截断同样如此。

`extras['time_outs'] = truncated & ~terminated & ~complete`：仅标记纯时限截断，排除击杀、脱靶与越界，符合 Gymnasium bootstrap 约定。

## 9. 初始分布

每副本一次 `uniform(9)` 抽样覆盖全部随机量：

- **目标**：距原点固定 $R=2500$ m 的水平方位均匀 $[0,2\pi)$；高度 $z_T=4000+(u-0.5)\cdot400$ m（$\pm200$ m）。航向指向导弹，限定在锥内：$\text{heading}=\operatorname{atan2}(-p_{T,y},-p_{T,x})+\mathcal U[-37.5^{\circ},37.5^{\circ}]$。
- **机动**：类型在 $\{0,1,2\}$ 均匀；$a_T=7.5\times9.81$ m/s²；$\omega_w=2\pi\times0.75$ rad/s；蛇形相位 $[0,2\pi)$；桶滚方向均匀；桶滚相位按第 6 节固定。
- **导弹**：位置 $(0,0,4000+(u-0.5)\cdot100)$ m（高度抖动 $\pm50$ m）；滚转抖动 $\mathcal U[-0.1,0.1]$ rad；零角速度。火控前置量瞄准目标沿缩放名义剩余飞行时间的直线外推：
\begin{equation}
t_{\text{go}}=0.8\frac{R}{V_{M0}+V_T}=4.0\ \text{s},\quad
\text{aim}=\frac{p_T+V_T t_{\text{go}}h-p_M}{\lVert p_T+V_T t_{\text{go}}h-p_M\rVert},\quad
v_{M0}=250\cdot\text{aim}\ \text{m/s}, \tag{3}
\end{equation}
姿态将体 $+x$ 对准 $\text{aim}$ 再加滚转抖动。该前置量的垂向分量清零初始 LOS 角速度；`TGO_SCALE`$=0.8$（1.0 为名义闭式）在调参 harness 中削减过冲脱靶（两个不相交 1024 回合种子群上成功率 +0.8 pp；仅调参记录，未发布对应产物）。完整机动模型预测更差（飞行期间机动位移仅数十米，而直线前置量正确抵消了交叉几何）。
- **无 warmup**：与火箭不同，reset 不做零指令 warmup——助推发动机在首个活动物理步自主点火，warmup 会消耗推力并在首个策略步前改变出生状态；空中导弹也无需沉降接触。

## 10. 扰动

可选风：`steady_wind`（世界系常向量，m/s）、`dryden ∈ [0,1]`（0 关闭）、`severity ∈ {light, moderate, severe}`（MIL-F-8785C 低空参考标准差 $(1.5,1.5,0.7)$、$(3.0,3.0,1.4)$、$(6.0,6.0,3.0)$ m/s；尺度长度 $(200,200,50)$ m）。一阶阵风滤波每策略步推进一次；气动项对 $v_{\text{air}}=v-v_{\text{wind}}$ 评估。静风（默认）动力学逐位不变。

## 11. 指标与 extras

回合完成时，`extras['episode']` 报告 `success_rate`、`miss_distance`（240 Hz 跟踪的逐回合最小距离）以及各分量奖励均值 `rew_{progress,align,kill,miss}`；长期累加器 `completed_metrics` 另有 `kill_rate`、`miss_rate`、`timeout_rate`。`extras['terminal_observations']` 携带全副本 float32 终态观测；`extras['time_outs']` 为第 8 节超时掩码。

## 12. 实现与关键常量

- 文件：`environments/missile/env.py`，类 `MissileInterceptEnv`，任务 id `missile`；目标运动学在 `_target_state`，发射姿态在 `_aim_quat`。
- Plant：`environments/common/plant.py`（`TensorPlant`，vehicle `'missile'`）；参数 `environments/common/original_pyflyt/models/vehicles/missile/missile.yaml`；URDF SHA-256 由 `consistency.verify_missile_assets` 固定。
- 传统基线：`MissileAPNBaseline`（增广比例导引）。

| 常量 | 值 |
|---|---|
| `agent_hz` / `inner_steps` | 60 / 2（控制 120 Hz，物理 240 Hz） |
| `max_steps` | 900（15 s；缓冲 `max_episode_length` 902） |
| `num_actions` / 观测维度 | 3 / 34 |
| `OMEGA_MAX` / `OMEGA_LAT` | 6.0 / 1.0 rad/s |
| `KILL_RADIUS` | 15.0 m |
| `R_SPAWN` / `CONE` / `A_LAT` / `F_WEAVE` | 2500 m / 37.5° / 7.5 g / 0.75 Hz |
| `V_T` / `V_M0` / `Z_MID` | 250 / 250 m/s / 4000 m |
| `K_PROG` / `K_ALIGN` | 0.01 / 0.01 |
| `KILL_BONUS` / `MISS_PENALTY` / `MISS_REF` / `TGO_SCALE` | +100 / −10 / 500 m / 0.8 |
| 质量 / 助推推力 / 燃烧时间 | 150 kg / 20 000 N / 4.0 s |
| `vel_clamp` | 1200 m/s，40 rad/s |
| 边界 / 脱靶条件 | $\lVert p\rVert>15$ km 或 $z<0$ / $t>2$ s 后 $r\cdot v_{\text{rel}}>0$ |

## 13. 参考

- MIL-F-8785C（低空 Dryden 湍流模型），经 `environments/common/disturbances.py` 实现。任务本身为项目自研，无外部参考实现。
