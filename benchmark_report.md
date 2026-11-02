# AerospaceRL Benchmark 技术报告

> GPU 并行航空航天控制强化学习基准 — 动力学建模、任务定义与传统基线

## 摘要

AerospaceRL Benchmark（以下简称 **benchmark**）是一套面向航空航天控制的 GPU 并行强化学习基准，包含 **10 个任务**，覆盖大气层飞行（四旋翼航路点、四旋翼目标跟踪、固定翼航路点、倾转旋翼 VTOL、火箭垂直着陆、导弹拦截）与航天器动力学（CMG 姿态控制、低推力轨道转移、交会、对接）。

基准的核心设计原则：

- **纯张量接口**：所有环境暴露统一的批量 device-tensor `step()/reset()` 协议，不依赖 Gym/Gymnasium；物理积分、接触检测、奖励、重置、随机数均为 masked tensor kernel，未激活副本冻结且不消耗随机数。
- **每任务配多个传统基线**：每个任务至少有两个解析式经典 GNC 控制器（主基线 + 结构变体），与学习算法共享同一环境、同一奖励、同一成功判据，且只读 observation 不访问内部状态。
- **三层速率嵌套**：policy → control → physics，大气层任务使用 Genesis 刚体后端，航天器任务为纯张量积分器（RK4 / 辛 Euler）。

本报告重点详述各任务的**被控对象动力学建模**、**物理参数**、**奖励设计**与**传统方法**，并介绍学习算法体系（含基于传统控制器的 residual RL 与增益调度 RL）及冻结评测协议。

---

## 1. 系统架构与统一基础设施

### 1.1 张量环境接口

所有环境继承统一的 `BaseEnv`，遵循以下契约：

- **单设备双精度**：物理状态为 `float64`，观测/奖励缓冲为 `float32`，全部状态驻留在同一 device 上。
- **无 per-replica Python 循环**：`step()` 内一切操作为批量张量核；每个 `step()` 仅执行一次 host-device 同步（`bool(done.any())`），用于 episode 边界簿记。
- **观测分组**：观测以 `TensorDict` 返回，`'policy'` 键下为策略可见向量；传统基线与学习策略消费同一向量。
- **时间约定**：`extras['time_outs']` 标记纯时限截断（用于 value bootstrap），`extras['terminal_observations']` 携带 reset 前观测。

### 1.2 刚体 Plant 与张量动力学

大气层 6 个任务（quadx / quadx_track / fixedwing / quadplane / rocket / missile）使用 `TensorPlant`（`environments/common/plant.py`），其统一承载：

- 几何与惯性（解析自冻结参考 URDF）
- 作动层：rate-PID mixer、升力面（`aerodynamics.py`）、助推器/燃料
- 物理步进、接触检查、reset/warmup
- 逐副本独立计数器随机流 `TensorRandom`（`mix32` 去 seed 重叠）

航天器 4 个任务（satellite / transfer / docking / rendezvous）为纯张量实现，无物理引擎，状态、随机数与统计全在设备上。

### 1.3 风扰统一钩子

QuadX-Waypoints、Fixedwing-Waypoints、Rocket-Landing 与 Missile-Intercept 共享 `WindDisturbance`（`disturbances.py`）：世界系定常风 + Dryden 阵风。构造参数为 `steady_wind`（m/s）、`dryden ∈ [0,1]` 与 `severity ∈ {light, moderate, severe}`；`dryden=0` 且无定常风时为静风。参考高度 300 ft 的三轴标准差分别为 light `(1.5,1.5,0.7)`、moderate `(3.0,3.0,1.4)`、severe `(6.0,6.0,3.0)` m/s，尺度长度为 `(200,200,50)` m，实际标准差再乘 `dryden`。RNG 由环境种子确定，滤波器状态在 episode 复位时清零。所有气动力统一对**空速** $v_{\text{air}} = v - w$ 评估；静风路径与无风扰基准逐位一致。训练配置可以带风，跨方法主比较使用冻结的静风评测配置，扰动结果单独标注。

### 1.4 坐标、单位与模型边界

- 位置、速度未特别说明时均采用右手惯性世界系，单位为 m、m/s；体速度/角速度下标 $b$ 表示机体系，CWH 任务的 LVLH 系固定为 $x$ 径向向外、$y$ 沿迹、$z$ 交轨。
- 刚体大气任务的**观测四元数为 `xyzw`，表示 body-to-world 旋转**；Genesis plant 内部为 `wxyz`，仅在 plant 边界转换。Satellite 是显式例外：其内部动力学与观测均使用 `wxyz` 和 Hamilton 乘法，姿态误差为 $q^{-1}\otimes q_{\text{target}}$ 的最短弧对数映射。
- 气动模型采用叶素/升力面与阻力近似，不包含完整非定常气动、结构弹性或高保真推进模型；Fixedwing 结果主要适用于给定速度、迎角与失速模型范围，Missile 的细长体法向力模型不能替代全弹气动数据库，Rocket 栅格翼模型不覆盖真实可展开机构的全部效应。
- CMG 使用转子转速恒定的降阶刚体模型；Transfer 使用二体点质量模型；Docking/Rendezvous 使用圆轨道近旁的线性 CWH 模型。上述任务是控制算法基准，不应被解释为真实飞行器的工程验证。

### 1.5 分布契约与复现

每个任务的默认 reset 分布是官方基准分布。训练课程只能改变初始状态或任务参数分布，不得改变奖励、成功判据、终止条件或物理参数；评测恢复 `config.json` 中冻结的 `env_kwargs`。训练、验证和 held-out 测试使用不同 RNG 流，测试流不参与调参或模型选择。传统控制器只允许读取 `'policy'` 观测及构造时冻结的标称模型常数，不能读取运行时 plant 隐状态。

PyFlyt 参考源码、URDF/YAML 物理资产及项目自建 Missile、Rocket-v2 资产由 `environments/common/consistency.py` 做 SHA-256 校验；plant 契约序列化后生成 `plant_id`。每次实验应同时保存代码版本、依赖版本、环境配置、算法配置和种子列表，使结果能够逐位复现或明确解释版本差异。

---

## 2. 任务清单

本章按大气层飞行与航天器动力学分组，逐个任务详述被控对象动力学、参数、奖励与传统方法。本章只描述环境本身，不涉及任何测试结果或评估数据。

**传统基线覆盖表**（“默认 + 两个结构变体”；均只读策略观测，性能状态由实验章节另行报告）：

| 任务 | 默认基线 | 结构变体 1 | 结构变体 2 |
|---|---|---|---|
| quadx | SE(3) 几何控制 | 级联 PID | 滑模位置环 + Euler 姿态环 |
| quadx_track | CV Kalman + SE(3) | 纯测量追踪 | EMA 滤波 + SE(3) |
| fixedwing | L1 + TECS | 级联 PID | 非线性制导律（NLGL）+ TECS |
| quadplane | SE(3) + 前推 | 混合级联 PID | L1 固定翼 + 低速旋翼辅助 |
| rocket | ZEM/ZEV PDG | 位置/速度 PD | E-guidance |
| missile | APN | 纯追踪（PPN） | 真比例导引（TPN） |
| satellite | 阻尼伪逆 CMG | 普通 Moore–Penrose 伪逆 | 加权伪逆 + 零空间避奇异 |
| transfer | Gauss 一步贪心 | Petropoulos Q-law | Edelbaum 预算切向制导 |
| docking | CWH-LQR + 约束层 | CWH 极点配置 | CWH ZEM/ZEV |
| rendezvous | 两脉冲 CWH | CWH-PD | glideslope 距离—速率制动 |

### 2.1 QuadX-Waypoints（四旋翼航路点导航）

**任务**：Crazyflie 2.x 微型四旋翼（27 g，X 构型）在 5 m 飞行穹顶内按序访问 4 个随机航路点，进入 0.2 m 球即捕获，20 s 时限；触地或出界失败。

#### 被控对象动力学

**三层速率**：policy 40 Hz → control 120 Hz（`inner_steps=3`）→ physics 240 Hz（每控制期 2 子步），每 policy 步 6 个物理子步。

**内环 rate PID（120 Hz）**：跟踪指令体角速度 $\omega^{\text{cmd}}$，输出归一化力矩 $u\in[-1,1]^3$：

$$u_t = \text{sat}\big(k_p e_t + I_t + k_d (e_t - e_{t-1})/\Delta t_c\big), \quad I_t = \text{sat}(I_{t-1} + k_i e_t \Delta t_c)$$

其中 $e_t = \omega^{\text{cmd}} - \omega_b$，$k_p=(0.04,0.04,0.08)$，$k_i=(5\times10^{-7},5\times10^{-7},2.7\times10^{-4})$，$k_d=(10^{-4},10^{-4},0)$。力矩与总推力经 X 构型 mixer 分配到 4 个电机。

**作动（240 Hz）**：电机状态一阶滞后 + 乘性噪声：

$$\theta_{t+1} = \theta_t + \frac{\Delta t}{\tau_m}(\pi_t - \theta_t), \quad \theta_t \leftarrow \theta_t(1 + 0.02\,\xi_t),\ \xi_t \sim \mathcal{N}(0,1)$$

$\tau_m=0.01$ s。乘性噪声为零均值，更新时不额外截断高斯样本；二次推力律中的绝对值只决定推力方向约定，不是噪声截断。单电机推力二次律 $f_i = |\theta_i|\theta_i\,F_{\max}/4$，$F_{\max}=2.0$ N；反扭矩 $\tau_{z,i}=f_i(k_m/k_f)s_i$。体阻力 $F_d = -\frac12\rho C_d A |v_{\text{air}}|\circ v_{\text{air}}$（$C_d=3.0$，$A=4\times10^{-4}$ m²），角阻力 $\tau_d=-10^{-4}|\omega_b|\circ\omega_b$。速度钳位 $\|v\|\le 15$ m/s，$\|\omega\|\le 30$ rad/s。

#### 动作空间

$a\in[-1,1]^4$ 仿射映射至 $(\omega_x^{\text{cmd}},\omega_y^{\text{cmd}},\omega_z^{\text{cmd}},T)$，范围 $(\pm\pi,\pm\pi,\pm\pi,[0,0.8])$。

#### 观测空间与终止（37 维）

体角速度(3) + 四元数 `xyzw`(4) + 体速度(3) + 世界位置(3) + 上一动作(4) + 电机状态(4) + 4 个航路点的体系相对位置(12) + 有效标志(4)，共 37 维，`float32`、未在环境内归一化。当前活动航路点位于相对位置的第 1 组。时限内访问全部 4 个航路点为成功（以截断结束）；触地或 $\|p\|>5$ m 为失败终止；800 步超时而未完成不算成功。

#### 参数表

| 参数 | 值 |
|---|---|
| 质量 / 总推力 | 0.027 kg / 2.0 N |
| 惯量 $J$ | $\text{diag}(1.4,1.4,2.17)\times10^{-5}$ kg·m² |
| 电机滞后 $\tau_m$ / 噪声比 | 0.01 s / 0.02 |
| rate-PID $k_p,k_i,k_d$ | $(.04,.04,.08),(5\text{e-}7,5\text{e-}7,2.7\text{e-}4),(1\text{e-}4,1\text{e-}4,0)$ |
| pwm 窗口 / 总推力上限 | $[0.05,1]$ / 0.8 |
| 捕获半径 / 出界半径 | 0.2 m / 5 m |
| 航路点半径 / 高度地板 | $[1.5,4.8]$ m / 0.4 m |
| 初始状态 | $(0,0,1)$ m，静止水平 |

#### 奖励

基于势函数的伸缩式塑形：

$$\Phi(s) = -\big(\|p - w_i\| + L(i)\big), \quad r_t = \Phi(s_{t+1}) - \Phi(s_t) + 15\,c_t - 10\,\kappa_t$$

$L(i)$ 为剩余航段长度和，$c_t$ 为捕获数，$\kappa_t$ 为崩溃指示。势差 telescoping 保证路径无关：成功回合累计恰为参考路径长度，崩溃没收未实现势。

#### 传统方法：SE(3) 几何控制（Lee et al. 2010）

- **位置环**：速度饱和级联 PD，$v_d = -(K_X/K_V)e_x$ 限幅至 $V_{\max}=1.5$ m/s；$F_{\text{des}} = m(g e_3 - K_V e_v)$，轴向积分修剪抵消悬停残差。
- **姿态环**：$SO(3)$ 上几何控制，$b_{3,d} = F_{\text{des}}/\|F_{\text{des}}\|$，力矩 $M = -K_R e_R - K_\Omega \omega_b + \omega_b \times J\omega_b$。
- **分配**：闭式 wrench→转子推力映射，预补偿电机滞后直流增益 $G_{\text{pwm}}=1.216$，反演 mixer 得到 rate 指令。
- **结构变体**：`cascaded_pid` 用位置—速度—姿态级联 PID；`smc` 用滑模位置外环与 Euler 姿态内环。

---

### 2.2 QuadX-Track（四旋翼持续目标跟踪）

**任务**：同型四旋翼在 30 s 内对合作地面移动目标保持天底相机视场内可见，同时保持固定高度 6 m 与固定水平偏移 (1.5, 0) m；可见时间占比 ≥ 0.85 且无坠毁为成功。Plant 与 QuadX-Waypoints 完全相同，区别在感知与奖励。

#### 被控对象动力学

与 2.1 节相同：40/120/240 Hz 三层速率、rate PID、X 构型 mixer、电机滞后噪声、二次推力、体/角阻力。

**合作地面目标**：运动学车辆，$z_t=0.15$ m，12 段预抽签（直线/转弯/启停），速度 0.4–1.2 m/s，转弯率 0.3–1.0 rad/s，在 15 m 圆域内反射边界。

**几何相机模型**：下视锥半角 30°，斜距窗口 1.5–12 m，6 个静态圆柱遮挡物（纯 LOS 几何，非物理实体）。可见时返回带 0.15 m 噪声的目标相对位置；不可见时观测为 0 + flag。

#### 观测空间（25 维）

体角速度(3) + 四元数(4) + 体速度(3) + 世界位置(3) + 上一动作(4) + 电机状态(4) + 目标测量(3) + 可见性 flag(1)。无目标真值。

#### 奖励

$$r_t = 0.10\,y_t - 0.10(1-y_t) - 0.05\,\|p_{xy}-\bar p_{xy}\|\,y_t - 0.01(z-6)^2 - 0.002\|\bar a_t\|^2 - 10\,\kappa_t$$

可见项仅在 $y_t=1$ 时支付；丢失时仅 −0.10 惩罚。

#### 传统方法：常速度卡尔曼滤波 + SE(3) 跟踪

- **CV Kalman**：状态 $[p_t, v_t]\in\mathbb{R}^6$，每步预测；仅可见时更新（$R=\sigma^2 I_3$，$\sigma=0.15$ m）。
- **超时保持**：可见性年龄 > 3 s 时冻结参考点悬停，不搜索。
- **控制**：复用 SE(3) 级联，加世界系目标速度前馈 $v_d = \hat v_t - (K_X/K_V)(p-\bar p)$。
- **结构变体**：`pursuit` 不做状态估计，直接使用带噪测量纯追踪；`pid_ema` 用指数滑动平均替代 Kalman 滤波。

触地或 $\|p\|>30$ m 立即失败；否则在 1200 步超时时按整回合可见比例判定成功，成功不提前截断。

---

### 2.3 Fixedwing-Waypoints（固定翼航路点导航）

**任务**：2.35 kg、1.6 m 翼展无人机从 $(0,0,10)$ m 以 20 m/s 出发，在 20–65 m 球壳内按序访问 4 个航路点（10 m 捕获球），120 s 时限；触地或出 100 m 球失败。

#### 被控对象动力学

**三层速率**：policy 30 Hz → control 120 Hz（`inner_steps=4`）→ physics 240 Hz，每 policy 步 8 物理子步。无内环姿态环，策略直接驱动舵面与油门。

**舵面（240 Hz）**：5 个升力面（左/右副翼、平尾、垂尾、主翼）一阶滞后 $\tau_s=0.05$ s。PyFlyt 叶素升力面模型：有限翼展升力线斜率、30% 襟翼弦比 + 粘性修正 $\eta=0.65$、零升迎角（翼/副翼 $-2^\circ$，尾翼 $0^\circ$）、正/负失速（翼 $+14^\circ/-9^\circ$，尾翼 $\pm9^\circ$）、蒙皮摩擦阻力 $C_{d0}=0.01$。力 $\bar q S(c_l, c_d)$，力矩 $\bar q S c\, c_m$。

**发动机（240 Hz）**：油门一阶滞后 + 零均值乘性噪声 $\xi\sim\mathcal{N}(0,1)$，$\tau_m=0.01$ s。螺旋桨推力 $F=|\theta|\theta\,F_{\max}$，$F_{\max}=18$ N 沿体 $+x$，反扭矩 $F(k_m/k_f)$。高斯样本不在噪声层额外截断。

#### 动作空间

$a\in[-1,1]^4$：$a_0$ 差动副翼（$\pm30^\circ$），$a_1$ 升降舵 + 方向舵（相反符号，$\pm20^\circ$），$a_2$ 死通道（主翼偏度 0），$a_3$ 油门映射至 $[0,1]$。有效维度 3。

#### 观测空间与终止（38 维）

体角速度(3) + 四元数 `xyzw`(4) + 体速度(3) + 世界位置(3) + 上一动作(4) + 舵面/油门状态(6) + 4 个航路点体系相对位置(12) + 有效标志(4)，共 38 维，未在环境内归一化。120 s / 3600 步内访问全部 4 个航路点为成功；触地或 $\|p\|>100$ m 为失败；其余超时为未完成。

#### 参数表

| 参数 | 值 |
|---|---|
| 质量 / 翼展 | 2.35 kg / 1.6 m |
| 最大推力 / 电机滞后 | 18 N / 0.01 s |
| 舵面滞后 $\tau_s$ | 0.05 s |
| 舵面偏度限（副翼L/R, 平/垂尾, 主翼） | $30^\circ,30^\circ,20^\circ,20^\circ,0^\circ$ |
| 捕获半径 / 出界半径 | 10 m / 100 m |
| 航路点球壳 / 高度地板 | $[20,65]$ m / 20 m |
| 初始状态 | $(0,0,10)$ m，$v_0=(20,0,0)$ m/s |

#### 奖励

与 QuadX 同构的势函数伸缩塑形：$\Phi(s) = -(\|p-w_i\| + L(i))$，$r_t = \Phi(s_{t+1})-\Phi(s_t) + 15\,c_t - 10\,\kappa_t$。替换了上游 $1/d$ 塑形（轨道绕行获无界回报）与 $\pm100$ 终端奖励（3600 步下信用分配不可达）。

#### 传统方法：L1 横向制导 + TECS 纵向自动驾驶

- **L1 制导**：距离调度 $L_1 = \text{clip}(k_{L1} d_h + L1_0, L1_{\min}, L1_{\max})$，横向加速度 $a_{\text{lat}} = 2V^2\sin\eta_c / L_1$，转为滚转角指令。含加速失速边界的气动滚转限、终端纯追踪混合、OOB 边界护盾。
- **TECS**：总能量误差 $\delta_E \to$ 油门（P+I，防饱和放气），能量平衡误差 $\delta_B \to$ 俯仰指令（含协调转弯前馈、欠速保护、超速爬升权衡）。
- **结构变体**：`pid_cascade` 去除 L1/TECS 制导启发式；`nlgl_tecs` 保留 TECS，将 L1 换成 Park 非线性制导律。

---

### 2.4 Quadplane-VTOL（倾转旋翼混合无人机航路点）

**任务**：2.35 kg 混合无人机（4 旋翼电机 + 固定翼舵面 + 前推螺旋桨）从近悬停（$z=15$ m，$|v|\sim6$ m/s）出发，在 40–75 m 球壳、≥25 m 高度访问 4 个航路点，捕获半径 15 m，出界 120 m，30 s / 1200 步 / 40 Hz。Plant 无显式模式切换，物理本身决定旋翼/气动权限过渡区（~3–12 m/s 体轴速度）。

#### 被控对象动力学

- **40/120/240 Hz** 三层速率，`inner_steps=3`。
- **旋翼通道**：与 QuadX 相同的 rate PID（40 Hz policy 指令体角速度 + 总推力）→ X 构型 mixer → 4 电机。
- **固定翼通道**：与 Fixedwing 相同的 5 升力面叶素模型 + 前推油门（第 5 个油门通道，直接映射至 plant `throttle[:,4]`）。
- **质量/推力**：2.35 kg，旋翼最大总推力与固定翼 $F_{\max}=18$ N 同量级。

#### 动作空间（8 维）

| 索引 | 含义 | 范围 |
|---|---|---|
| a0–a2 | 体角速度指令 (p,q,r) | $\pm\pi$ rad/s |
| a3 | 四旋翼总推力 | $[0,1]$ |
| a4–a6 | 固定翼舵面（副翼差动/平尾+垂尾/死主翼） | $[-1,1]$ |
| a7 | 前推螺旋桨油门 | $[0,1]$ |

#### 观测空间与终止（42 维）

体角速度(3) + 四元数 `xyzw`(4) + 体速度(3) + 世界位置(3) + 上一动作(8) + 5 舵面与 5 电机状态(10) + 4 个航路点体系相对位置(12) + 有效标志(4)，共 42 维，未在环境内归一化。1200 步内访问全部 4 个航路点为成功；触地或 $\|p\|>120$ m 为失败；其余超时为未完成。

#### 参数表

| 参数 | 值 |
|---|---|
| 质量 | ~2.35 kg |
| policy/control/physics | 40 / 120 / 240 Hz |
| 捕获半径 / 出界半径 | 15 m / 120 m |
| 航路点球壳 / 高度地板 | $[40,75]$ m / 25 m |
| 初始状态 | $z=15$ m，水平速度 6 m/s，水平 |
| 步数 | 1200 (30 s) |

#### 奖励

与 quadx/fixedwing 同构势函数伸缩塑形：$\Phi(s) = -(d_{\text{current}} + L_{\text{remaining}})$，$r_t = \Phi(s_{t+1}) - \Phi(s_t) + 15\cdot\text{reach\_count} - 10\cdot\text{crash}$。接近当前目标时奖励为正。

#### 传统方法：SE(3) 旋翼控制 + 前推螺旋桨

默认 `QuadplaneSE3PusherBaseline` 使用 SE(3) 位置/姿态控制并提供前推油门。结构变体 `hybrid_cascade` 以速度调度混合四旋翼级联 PID 与固定翼通道，`l1_wing` 在高速段采用 L1 横向制导、低速段保留旋翼辅助。三个控制器均已注册；各自性能是否达到任务要求由实验结果判定，不在环境章节预设结论。

---

### 2.5 Rocket-Landing（火箭垂直着陆）

**任务**：Falcon-9 级一子级助推器从 400–450 m 高度以 −100 m/s 垂直速度释放，须消横向偏移并停稳于地面垫。主指标为连续停稳 2 秒（40 Hz 下 81 个连续样本）。

#### 被控对象动力学

**三层速率**：policy 40 Hz → control 120 Hz（`inner_steps=3`）→ physics 240 Hz，30 s / 1200 步。

**质量特性**：结构质量 138.2 kg；燃料箱质量每物理步重写为 $f\cdot 410.9$ kg（$f$ 为燃料分数，初始 0.05），初始质量 ≈ 158.7 kg。

**助推器**：可重点火发动机；油门目标 $u\in[0,1]\mapsto 0.39+0.61u$，一阶滞后 $\tau=0.01$ s + 1% 乘性噪声。推力 $T=\text{throttle}\times 7607.0$ N（39% 下限 = 2966.7 N），沿万向节方向作用。双轴万向节限 $\pm5^\circ$，滞后 0.01 s。燃料消耗 1.451 kg/s @ 全油门。

**栅格翼**：4 个升力面（弦 0.5 m，展 0.5 m，$C_{l\alpha}=2\pi$，失速 $\pm20^\circ$，偏度限 $\pm45^\circ$，滞后 0.05 s）。

**体阻力**：$F = -\frac12\rho C_d A |v_{\text{air}}| v_{\text{air}}$，$C_d=(1.16,1.16,2.0)$，$A=(1.7649,1.7649,0.1075)$ m²。

#### 动作空间（7 维）

$(f_x, f_y, f_\psi, \text{ign}, \text{thr}, g_x, g_y)$：栅格翼滚/俯/偏指令、点火（>0.5 触发）、油门（映射 39–100%）、万向节 $x/y$（$\pm5^\circ$）。

#### 观测空间与终止（30 维）

体角速度(3) + 四元数 `xyzw`(4) + 体速度(3) + 世界位置(3) + 上一动作(7) + 栅格翼/点火/燃料/油门/万向节状态(9) + 脚垫接触标志(1)，共 30 维，未在环境内归一化。主成功指标为 30 s 内曾连续停稳 2 s 且未发生致命失败；成功后仍继续仿真至时限。致命接触、地面碰撞或出界立即终止，普通超时不是成功。

#### 参数表

| 参数 | 值 |
|---|---|
| 结构质量 / 燃料满质量 | 138.2 kg / 410.9 kg |
| 初始燃料分数 | 0.05 |
| 最大推力 / 最小推力比 | 7607 N / 0.39 |
| 万向节范围 / 滞后 | $\pm5^\circ$ / 0.01 s |
| 燃料消耗率 | 1.451 kg/s @ 全油门 |
| 栅格翼 $C_{l\alpha}$ / 失速 / 偏度限 | $2\pi$ / $\pm20^\circ$ / $\pm45^\circ$ |
| 初始 IC | $x,y\in[-20,20]$ m，$z\in[400,450]$ m，$\phi,\theta,\psi\in[-0.3,0.3]$ rad，$v_0=(0,0,-100)$ m/s |
| 成功门（停稳 2s） | $\|v\|<0.02$ m/s，$\|\omega\|<0.02$ rad/s，倾角 $<0.1$ rad |
| 致命门 | 接触速度 $>1.0$ m/s 或自旋 $>0.35$ rad/s |

#### 奖励（密集塑形）

每 policy 步：$r = r_{\text{hover}} + r_{\text{hold}} + r_{\text{lateral}} + r_{\text{descent}} + r_{\text{decel}} + r_{\text{attitude}} + r_{\text{contact}} + r_{\text{corridor}} + r_{\text{success}} + r_{\text{fatal}}$

关键项：悬停成本 −0.3；横向进度 $10(\|p_{xy}^-\|-\|p_{xy}\|)$；走廊惩罚约束下降速度不超过可行剖面 $v_{\text{allow}}(z) = 0.5 + \sqrt{2\cdot 2\cdot z}$；成功奖金 +5000；致命惩罚 −500。

#### 传统方法：ZEM/ZEV 动力下降制导（PDG）

- **中段**：$s\ge25$ m/s 时闭式 ZEM/ZEV 制动，$t_{\text{go}}$ 由恒减速剖面重解，侧向需求限幅至倾斜一致预算。
- **末段 1**：$s<25$ m/s 时沿虚拟垫（高于实垫 2 m）的最小推力流形制动。
- **末段 2**：$s\le1.5$ m/s 且 $h\le2$ m 时恒速脉冲伺服，速度滞环切/重点火。
- **姿态**：PD 倾角环 + 气动力矩前馈，力矩按 TVC 优先、栅格翼补足分配。
- **结构变体**：`pd_landing` 使用位置/速度 PD；`eguidance` 在中段使用二次多项式 E-guidance 目标。

---

### 2.6 Missile-Intercept（导弹拦截）

**任务**：150 kg 细长体尾控导弹拦截运动学 3-DOF 目标。目标 250 m/s、~4 km 高度，做直线/正弦 weave/水平桶滚；初始距离 2500 m，目标航向锥 $\pm37.5^\circ$，横向过载 7.5 g。近炸引信杀伤半径 15 m。

#### 被控对象动力学

**三层速率**：policy 60 Hz → control 120 Hz（`inner_steps=2`）→ physics 240 Hz，15 s / 900 步。

**弹体**：150 kg 刚体。轴向体阻力 $F_x = -\frac12\rho C_{d,x}A_x|v_x|v_x$（$C_{d,x}=0.25$，$A_x=0.0314$ m²）。细长体法向力 $F_{\text{lat}} = -\frac12\rho\|v\|A_x C_{n\alpha} v_{\text{lat}}$（$C_{n\alpha}=16$），作用于压心 $x_{\text{cp}}=-0.45$ m（同时产生转向力、weathercock 恢复力矩、旋转阻尼）。合成俯仰/偏航阻尼 $\tau = -c_{mq}\|v\|\omega_{\text{lat}}$（$c_{mq}=4.0$）。

**尾翼**：4 片尾翼在 $x=-1.3$ m（弦 0.45 m，展 0.40 m，$C_{l\alpha}=3.0$，失速 $\pm40^\circ$，偏度限 $\pm35^\circ$，滞后 0.02 s）。

**rate PID + skid-to-turn mixer**：$k_p=5.0$ 体角速度比例控制，滚转通道减速比 $g=0.01$；fin 指令 $(q_c+g p_c, q_c-g p_c, -r_c-g p_c, -r_c+g p_c)$。

**发动机**：自主点火，20 kN 沿体 $+x$ 燃烧 4 s 后滑行，无油门通道。

#### 动作空间（3 维）

体角速度指令：滚转 $\pm6.0$ rad/s，俯仰/偏航 $\pm1.0$ rad/s。

#### 观测空间与终止（34 维）

体角速度(3) + 四元数 `xyzw`(4) + 体速度(3) + 世界位置(3) + 上一动作(3) + 尾翼与发动机状态(7) + 目标体系相对位置(3) + 相对速度(3) + LOS 角速度(3) + 闭合速度(1) + 目标速率(1)，共 34 维，未在环境内归一化。距离小于 15 m 即拦截成功并以截断结束；2 s 后越过最近接点且未拦截、触地或 $\|p\|>15$ km 为失败终止；15 s 超时而未命中不算成功。同一内层步命中优先于脱靶。

#### 参数表

| 参数 | 值 |
|---|---|
| 质量 | 150 kg |
| 轴向阻力 $C_{d,x} A_x$ | 0.25 × 0.0314 m² |
| 法向力 $C_{n\alpha}$ | 16 |
| 尾翼 $C_{l\alpha}$ / 失速 / 偏度限 | 3.0 / $\pm40^\circ$ / $\pm35^\circ$ |
| 助推推力 / 燃烧时间 | 20 kN / 4.0 s |
| 初始距离 / 航向锥 / 横向过载 | 2500 m / $\pm37.5^\circ$ / 7.5 g |
| 目标/导弹速度 | 250 / 250 m/s |
| 杀伤半径 | 15 m |

#### 奖励

$$r_{\text{progress}} = 0.01(d^- - d),\quad r_{\text{align}} = 0.01(\hat v\cdot\hat r),\quad r_{\text{kill}}=+100,\quad r_{\text{miss}}=-10\min(d_{\min}/500,1)$$

对齐项为 PN 一致的密集塑形，不能通过飞离目标刷分。

#### 传统方法：增广比例导引（APN，N=15）

- **LOS 率滤波**：一阶低通 $\alpha=0.8$，终端 $V_c<50$ m/s 时冻结。
- **APN 律**：$a^{\text{cmd}} = N V_c(\lambda_w\times\hat r_w) + \frac{N}{2}(g - (g\cdot\hat r_w)\hat r_w)$。
- **skid-to-turn 运动学逆**：$\omega^{\text{cmd}} = (b_x\times a_{\text{lat}})/V$。
- **结构变体**：`ppn` 为不带前置角的纯追踪，`tpn` 为不带重力增广的真比例导引。

---

### 2.7 Satellite Attitude（CMG 姿态控制）

**任务**：刚体航天器搭载 $N$ 个单框架控制力矩陀螺（CMG），将姿态从扰动初态机动至随机目标姿态。支持 4-CMG 金字塔与 6-CMG 正交两种构型。30 s / 1200 步 / 40 Hz，纯张量 RK4。

#### 被控对象动力学

**转子轴演化**（Rodrigues 旋转）：

$$r_i(\theta_i) = \cos\theta_i\, r_i(0) + \sin\theta_i\,(g_i\times r_i(0))$$

**刚体+CMG 降阶模型**：

$$I\dot\omega = -\omega\times(I\omega) - \omega\times\big(h\sum_i r_i\big) - h\sum_i \dot\theta_i(g_i\times r_i)$$

$$\dot q = \tfrac12 q\otimes(0,\omega),\quad \dot\theta = \delta$$

经典 RK4 积分，$\mathrm{d}t=2.5$ ms × 10 子步/控制步。四元数每子步重归一化，框架角 wrap 至 $[-\pi,\pi)$。

**4-CMG 金字塔**：$I=\text{diag}(75,60,113)$ kg·m²，转子惯量 0.2 kg·m² @ 100 rad/s，$h=20$ kg·m²/s，偏置角 $\beta=\arctan\sqrt{2}$。

**6-CMG 正交**：$I=\text{diag}(0.055,0.055,0.085)$ kg·m²，转子 @ 200 rad/s，$h=0.0125$。

#### 动作空间

$u\in[-1,1]^N$，$\delta = 3.0\cdot u$ rad/s 每框架。

#### 观测空间（$14+3N$ 维）

四元数 `wxyz`(4) + 姿态误差对数(3) + 体角速度(3) + $\sin\theta(N)$ + $\cos\theta(N)$ + $\dot\theta(N)$ + 归一化可操纵性 $m(J)=\sqrt{\det(JJ^\top)}/h^3$(1) + 阻尼分配特征(3)。

#### 奖励

势函数伸缩式，带两个平滑门：

$$\Phi(s) = 30\,e + 70\,e\,g_n(e) + \frac{30\|\omega\|^2}{2\cdot 1.0}g(e) + \frac{0.5}{m(J)+0.5}$$

$$r_t = \Phi(s_t) - \Phi(s_{t+1}) + \mathbb{1}[\text{success}]\cdot 1.0 - \mathbb{1}[\text{fail}]\cdot 10.0 - 0.0005\sum_i\delta_i^2$$

$e<0.5$ rad 时 $g_n$ 门将误差斜率陡化至 ~100；$e<0.8$ rad 时 $g$ 门对快速到达征收刹车项。奇异项软分母避免 $1/m(J)$ 尖峰。

#### 成功/终止

成功：$e<0.05$ rad 且 $\|\omega\|<0.02$ rad/s，立即截断并发成功奖励。失败：$\|\omega\|>10$ rad/s。超时：1200 步。初始目标姿态在 SO(3) 上均匀采样，初始小角速度扰动尺度为 `reset_noise_scale=0.01`；改变该尺度或 CMG 构型属于训练/测试分布配置，必须写入结果口径。

#### 传统方法：四元数 PD + Schaub-Junkins 阻尼伪逆 CMG 转向

- **PD + 精确前馈**：$\dot\omega_d = k_p e - k_d\omega$，$\tau_{\text{cmd}} = I\dot\omega_d + \omega\times(I\omega) + \omega\times h_\Sigma$。
- **奇异鲁棒转向**：$\dot\theta = -\frac{1}{\kappa}J(J^\top J + \lambda^2 I_3)^{-1}\tau_{\text{cmd}}$，$\lambda = \Omega(\lambda_0 + 0.15 e^{-3m})$，随可操纵性坍缩自适应增大阻尼。
- **结构变体**：`quat_pd_mp_pinv_cmg` 使用普通 Moore–Penrose 伪逆；`quat_pd_wpinv_cmg` 使用加权伪逆并加入零空间避奇异项。

---

### 2.8 Low-Thrust Transfer（低推力近地点抬高）

**任务**：100 kg MEO 卫星搭载三轴正交推力对（10 mN/轴，$I_{sp}=4000$ s EP），在 166 步 × 300 s（~13.8 h，约 5 圈）内抬高初始 $a=10000$ km、$e=0.15$ 轨道的近地点，保持远地点不变且省燃料。

#### 被控对象动力学

二体点质量引力 + 恒定推力，齐奥尔科夫斯基质量消耗：

$$\dot r = v,\quad \dot v = -\frac{\mu}{\|r\|^3}r + \frac{f_{\max}}{m}u,\quad \dot m = -\frac{f_{\max}\|u\|_1}{I_{sp} g_0}$$

$\mu=3.986004418\times10^{14}$ m³/s²，$g_0=9.80665$ m/s²。7 维状态 $[r,v,m]$ 经典 RK4 积分，$h=10$ s × 30 子步/控制步。

**初始状态**：由固定根数（$a_0=10^7$ m，$e_0=0.15$，$i=60^\circ$，$\Omega=120^\circ$，$\omega=240^\circ$）+ 采样平近点角 $M_0\sim U(-\pi,\pi)$，经 8 次牛顿迭代解 Kepler 方程得笛卡尔态。$r_{p0}=8.5\times10^6$ m，$r_{a0}=1.15\times10^7$ m。

#### 动作空间（3 维）

$u\in[-1,1]^3$，每轴推力 $f_{\max}u_i$；全轴满偏全程约 5 m/s $\Delta v$。

#### 观测空间（8 维）

$[$ 已用时间 $t$，位置 $r$，速度 $v$，剩余质量 $m]$，未归一化。

#### 奖励

任务级稀疏终端奖励：

$$R = 10^{-4}\big(-|\Delta r_a| + \Delta r_p + 2\times10^5\,\Delta m\big)$$

训练奖励为其精确势函数分解：$\Phi(s) = 10^{-4}(-|r_a-r_{a0}| + (r_p-r_{p0}) + 2\times10^5(m-m_0))$，$r_t = \Phi(s_{t+1})-\Phi(s_t)$。每步即时信用：近地点附近顺向点火抬高 $r_p$（+），他处点火增大 $|r_a-r_{a0}|$ 或耗燃料（−）。

#### 成功/终止

成功：时限 166 步时 $R>0$。防御终止：撞击 $r_p<6378137$ m、逃逸 $r_a\le0$、燃料耗尽 $m\le1$ kg。

#### 传统方法：Gauss 一步贪心切向制导

- **密切根数恢复**：从观测态解 $h,a,\mathbf{e},p,\nu$，切向方向 $\hat t = \hat h\times\hat r$。
- **Gauss 变分方程**：一阶每步 $\Delta a,\Delta e,\Delta r_p,\Delta r_a,\Delta m$。
- **贪心决策**：令本步非负燃料消耗量 $\Delta m_{\text{burn}}\ge0$，在 {滑行, 顺向切向点火, 逆向切向点火} 中最大化单步净目标增益 $G = \Delta r_p - (|r_a+\Delta r_a-r_{a0}|-d_0) - 2\times10^5\Delta m_{\text{burn}}$。这里使用正的消耗量计费，与终端奖励中的 $2\times10^5\Delta m$（$\Delta m\le0$）完全同号，不会奖励耗油。
- **结构变体**：`q_law` 使用 Petropoulos Lyapunov 邻近商制导；`edelbaum` 使用带 $\Delta v$ 预算的切向点火弧。

---

### 2.9 Docking（对接）

**任务**：追踪器在 400 km 圆轨道上距目标 60–120 m 处，须接近至 1 m 内且终端相对速度 < 0.1 m/s，不超过 200 m 距离、2 kg 推进剂预算。500 步 / ~8.3 min / 1 s 步。

#### 被控对象动力学

Clohessy-Wiltshire-Hill（CWH）线性化相对运动方程（LVLH 系，$x$ 径向向外，$y$ 沿迹向，$z$ 交轨向）：

$$\ddot x = 3n^2 x + 2n\dot y + F_x/m,\quad \ddot y = -2n\dot x + F_y/m,\quad \ddot z = -n^2 z + F_z/m$$

$n=\sqrt{\mu/a^3}=1.1314\times10^{-3}$ rad/s（$a=6.778\times10^6$ m，400 km 圆轨）。

**辛 Euler 积分**（速度先更新）：

$$v_{t+1} = v_t + \Delta t(G p_t + H v_t + F_t/m),\quad p_{t+1} = p_t + \Delta t\,v_{t+1}$$

等效离散线性映射 $s_{t+1} = A_d s_t + B_d F_t$，$\Delta t=1$ s。推进剂 $\Delta f_t = \|F_t\|\Delta t/(I_{sp} g_0)$，$I_{sp}=300$ s。

#### 动作空间（3 维）

$a\in[-1,1]^3$，$F = F_{\max}\cdot a$，$F_{\max}=10$ N/轴，$m=500$ kg，$a_{\max}=0.02$ m/s²。

#### 观测空间与终止（6 维）

LVLH 相对位置(3) + 相对速度(3)，共 6 维，`float32`、未归一化。位置和速度方向的初始向量分别在单位球面均匀采样，距离与速度幅值在参数表区间内采样。同时满足成功门且无违规即成功截断；距离越界或燃料耗尽为失败终止；500 步超时未达目标不算成功。终步违规优先于成功。

#### 参数表

| 参数 | 值 |
|---|---|
| 轨道高度 / 半长轴 | 400 km / $6.778\times10^6$ m |
| 平均运动 $n$ | $1.1314\times10^{-3}$ rad/s |
| 追踪器质量 / 单轴推力 | 500 kg / 10 N |
| $I_{sp}$ / $g_0$ | 300 s / 9.80665 m/s² |
| 初始距离 / 初始速度 | $[60,120]$ m / $[0,0.5]$ m/s |
| 成功门 | $\|p\|<1$ m 且 $\|v\|<0.1$ m/s |
| 距离失败 / 燃料预算 | 200 m / 2 kg |
| 步数 / $\Delta t$ | 500 / 1 s |

#### 奖励

势函数伸缩式，势函数含制动距离增广：

$$\Phi(s) = -\frac{\|p\| + \|v\|^2/(2a_{\text{ref}})}{d_{\text{norm}}},\quad a_{\text{ref}}=0.012\text{ m/s}^2,\ d_{\text{norm}}=100\text{ m}$$

$$r_t = w(\Phi(s_{t+1})-\Phi(s_t)) - \gamma\frac{\|F_t\|}{F_{\max}},\quad w=100,\ \gamma=0.01$$

成功终端奖 +100，违规惩罚 −100。

#### 传统方法：精确辛离散 CWH LQR + 约束层

- **精确离散化**：设计模型 $A_d,B_d$ 与 plant 的辛 Euler 逐位一致，零离散化误差。
- **Riccati 增益**：Bryson 归一化 $Q=\text{diag}(1,1,1,300,300,300)$，$R=(0.35 a_{\max})^{-2}I_3$。
- **约束层**：①距离走廊护盾（停车增广距离 > 0.95·range_fail 时全力制动）；②终端捕获网（$\|p\|<2$ m 且接近速度超限时制动）；③燃料降额（$\hat f>0.75 f_0$ 时限幅）。
- **结构变体**：`pole_placement` 使用无附加约束层的 CWH 极点配置；`zem_zev` 使用递推重解的单脉冲 ZEM/ZEV 制导及距离/燃料保护。

---

### 2.10 Rendezvous（交会）

**任务**：追踪器距目标 1000–1500 m，须接近至 10 m 内且终端速度 < 0.5 m/s，避开 5 m keep-out 球，不超 2000 m、4 kg 燃料。1000 步 / ~16.7 min / 1 s。

#### 被控对象动力学

与对接相同的 CWH 方程 + 辛 Euler 积分（$\Delta t=1$ s）。保持点 $\rho_h=9$ m（在 5 m keep-out 与 10 m 成功球之间）。

#### 动作空间（3 维）

同对接：$a\in[-1,1]^3$，$F=F_{\max}a$，$F_{\max}=10$ N，$m=500$ kg。

#### 观测空间与终止（6 维）

与 Docking 相同，为 LVLH 相对位置(3) + 相对速度(3)，共 6 维，未归一化。满足成功门且不在 keep-out、距离或燃料违规状态即成功截断；进入 keep-out、超过 2000 m 或耗尽 4 kg 燃料为失败；1000 步超时未达目标不算成功。

#### 参数表

| 参数 | 值 |
|---|---|
| 初始距离 / 初始速度 | $[1000,1500]$ m / $[0,1]$ m/s |
| 成功门 | $\|p\|<10$ m 且 $\|v\|<0.5$ m/s |
| keep-out / 距离失败 | 5 m / 2000 m |
| 燃料预算 | 4 kg |
| 步数 / $\Delta t$ | 1000 / 1 s |

#### 奖励

与对接同构势函数塑形，$d_{\text{norm}}=d_0$（每回合初始距离自适应归一化），$a_{\text{ref}}=0.1$ m/s²。终端奖 +20，违规 −20。

#### 终止优先级

违规 > 成功（keep-out 球在成功球内部，无此优先级则低速冲入 keep-out 会被误计为成功）。

#### 传统方法：递推重解两脉冲 CWH ZEM/ZEV 制导

- **精确离散 STM**：$\Phi(T)=\exp(\mathcal{L}T)$，$\mathcal{L}=\log(A_d)/\Delta t$，无滑行弧模型误差。
- **保持点**：锁定在初始 LOS 上 $\rho_h=9$ m 处。
- **两脉冲解**：$\Delta v_1 = \Phi_{rv}^{-1}\text{ZEM}$，$\Delta v_2 = \text{ZEV} - \Phi_{vv}\Delta v_1$，精确满足 $r(T)=r_f, v(T)=0$。
- **可行性棘轮**：燃烧持续时间不 fit 则延长 $T_{\text{arr}}$；超上限则降级为速度置零。
- **OGL 第一步**：$a = 6\,\text{ZEM}/T_{\text{go}}^2 - 2\,\text{ZEV}/T_{\text{go}}$。
- **结构变体**：`cwh_pd` 使用无约束层的 CWH 补偿 PD；`glideslope` 沿初始 LOS 使用距离—速率制动走廊。

---

## 3. 学习算法

benchmark 提供一个完整的学习算法体系，分为三类：(1) 端到端 RL（PPO），(2) 模仿学习 warm-start（BC → PPO），(3) 围绕传统控制器的两种 RL 范式——**residual RL**（学习动作残差）与**增益调度 RL**（学习控制器增益）。后两者均不修改任务环境的奖励、成功判据或终止条件，传统控制器作为 frozen prior 参与训练。

### 3.1 PPO（rsl-rl）

- 2×128 tanh MLP，rsl-rl `EmpiricalNormalization`，clipped PPO。
- **Actor 默认使用 `BetaDistribution`**（`action_range=(-1,1)`）：MLP 输出 $[2, A]$ 的 $\alpha,\beta$ 参数，经 $\text{softplus}(\cdot)+1$ 后采样 $Beta(\alpha,\beta)$，线性 rescale 至 $[-1,1]$。动作天然有界，log_prob 含线性 rescale 的 Jacobian 修正。选 Beta 而非 tanh-Gaussian 的理由：① KL 散度有闭式（仓库的 `KLAnchorPPO` 与 `desired_kl` 自适应 LR 依赖准确的 `kl_divergence`）；② 无 tanh 的 $\log(1-a^2)$ 边界惩罚，不抑制打满执行器的动作；③ rsl-rl 原生支持。可通过 `build_cfg(distribution="gaussian")` 切回无界高斯。
- 可选 warmup-cosine LR 调度（floor 1.5e-4），保证跨算法比较非调度伪影。
- 参考训练配置：1024 envs × 1000 updates，3 seeds {7,13,21}；这只是可复现的默认实验口径，不是算法定义或跨硬件硬约束。正式结果须同时给出每 update 的 rollout 长度、总环境交互步数与实际种子列表。

### 3.2 BC Warm-Start（模仿学习 + PPO 微调）

- 从传统控制器采集行为克隆数据，预训练 actor/critic，再接入 PPO 训练。
- BC 成本独立报告，不摊入 RL 训练预算。

### 3.3 Residual RL（动作残差学习）

**核心思想**：策略不直接输出环境动作，而是输出叠加在传统控制器标称动作之上的**残差**，以有界权限修正标称动作。

**组合规则**（`ResidualVecEnv`，`algorithms/residual/residual_env.py`）：

$$u_{\text{policy}}\in[-1,1]\;\xrightarrow{\text{clamp}\cdot\text{limit}}\; r\in[-\text{limit},\text{limit}],\quad a^{\text{cmd}} = a^{\text{nominal}} + r,\quad a = \text{clamp}(a^{\text{cmd}}, -c, c)$$

说明：PPO actor 默认 `BetaDistribution`，输出已在 $[-1,1]$，因此残差默认走 `residual_squash='identity'$ 分支（`u.clamp(-1,1)\cdot\text{limit}$），无需 tanh。PPO 的 log 概率在 Beta 采样变量上计算，clamp 与最终 clamp 为确定性动作映射，不影响 PPO ratio。

- **算法适配**：若切回无界 Gaussian PPO，应改用 `residual_squash='tanh'$（`tanh(u)\cdot\text{limit}$）。

- **权限控制**：每通道独立 `residual_limit` 或 `residual_rho`（占动作空间宽度的比例），上限不超过半通道宽度。
- **零初始化**：actor 末层线性权重/偏置置零，原始分布参数经 $\text{softplus}(0)+1\approx1.69$ 得到对称 Beta 分布，其归一化均值动作对应零残差。因此**确定性均值动作**下，未训练策略精确复现传统控制器；训练时随机采样仍产生非零探索残差，不能把均值轨迹与随机 rollout 混为一谈。
- **诊断**：每步记录残差均值/RMS/最大值、标称与组合动作饱和率、残差对冲标称比例、组合裁剪率。
- **可选机制**：
  - **success-gated 权限课程** (`success_gated_authority_curriculum`)：从 1% 权限起步，仅当训练成功率 ≥ 阈值且 episode 数足够时按步长递增权限，上限 100%。
  - **phase gate**：根据观测中某分量的范数（如终端接近程度）对权限做 sigmoid 增益调制。

**适用任务**：docking（推力残差）、rendezvous（推力残差）、satellite（CMG 框架角速度残差）。物理单位转换按任务记录：docking/rendezvous 为 `limit·f_max` N，satellite 为 `limit·action_scale` rad/s。

### 3.4 增益调度 RL（RL 调传统控制器参数）

**核心思想**：策略不输出动作，而是输出传统控制器**可调增益**的设定值；控制器用被调度后的增益计算标称动作并施加到环境。策略动作空间维度等于控制器声明的 `TUNABLE_PARAMS` 数量，而非环境动作维度。

**动作→增益映射**（`GainSchedulingVecEnv`，`algorithms/gainsched/gain_sched_env.py`）：

$$u_i\in[-1,1] \;\mapsto\; g_i = \begin{cases} \text{nominal}_i + u_i\cdot \text{dist}_i & \text{（线性空间）} \\ \exp(\log\text{nominal}_i + u_i\cdot \text{dist}_i) & \text{（对数空间，log=True）} \end{cases}$$

- $u=0$ 始终对应 hand-tuned 标称值；$u=\pm1$ 到达缩放后的上下界。
- 对数空间映射用于跨数量级的增益（刚度、阻尼、滤波、代价权重），单位动作步产生等比例相对变化。
- **EMA 平滑**：$g_{\text{applied}} = (1-s)\,g_{\text{req}} + s\,g_{\text{prev}}$，降低增益抖动。
- **增益观测**：当前归一化增益 `(g-nominal)/authority` 追加到策略观测，使策略能对自身调度闭环。
- **episode 复位**：每个副本在 episode 开始时增益回到标称值。

**控制器契约**：控制器须声明 `TUNABLE_PARAMS: dict[str, dict(low, high, default, log=?)]` 并实现 `set_params(dict)` / `get_params()`（`Baseline` 基类提供）；每副本持有独立增益向量，reward 可独立 credit 每个副本的调度。

**适用场景**：当传统控制器结构可靠但手工增益未全局最优时，RL 在标称点附近搜索时变增益调度，而非从零学习控制策略。

### 3.5 辅助动作空间包装

- **DeltaActionVecEnv**：策略输出有界增量 $d_t$，执行 $a_t = \text{clamp}(a_{t-1} + \text{scale}\cdot d_t, -c, c)$，episode 边界增量状态清零。适配 CMG 框架角速率类 bang-bang 动作。
- **ActionRatePenaltyVecEnv**：训练奖励追加 $-\text{coef}\|a_t-a_{t-1}\|^2$，仅训练时启用，评测用原始任务奖励。

---

## 4. 评测原则

具体训练预算、并行规模、种子数均为可调参数，由实验者按硬件条件设定。报告强调以下评测原则：

- **固定预算，无 early stopping**：主结果在相同任务内按相同交互步数或相同墙钟预算比较，仅评最终 checkpoint，避免选择性报告。`selection` 只允许作为单独标注的消融臂，不得与最终-checkpoint 主结果混排。
- **多 seed 报告**：至少 3 个种子，保留逐 seed 原始值，报告 mean、[min,max]；样本量允许时补充 bootstrap 置信区间。配对比较在相同评测种子、副本编号和初始条件上进行，并报告配对差值，而不只比较独立均值。held-out 测试流只用于最终泛化评估。
- **成功定义**：终止优先级 违约 > 成功（同步满足目标与失败记为失败）。
- **双维度成本**：同时报告样本效率（等交互步数的成功率）与计算成本（墙钟时间、吞吐、显存峰值）。RL、BC/DAgger 数据采集、传统控制器调参与专家查询分别计量；使用 BC 的臂必须注明 PPO 交互步不含数据采集步，人工调参不作为可忽略的零成本项。
- **任务内比较**：主结论只在同任务、同分布内形成。跨任务不直接比较原始交互步数或总成功率排名；如需汇总，须同时列出 policy 步长、episode 上限、总物理时间和归一化指标。
- **口径标注**：每个成功率须注明 ①训练/确定性/随机评测口径 ②步数窗口与物理时域 ③风扰配置 ④reset 分布 ⑤并行副本数和实际评测 episode 数。

---

## 5. 实验方案与测量指标

本章描述**实验方案**与测量指标，不含具体数值结果。表中的对比臂和消融表示计划覆盖范围；只有实际运行并保存完整配置的条目才可进入结果表，未完成条目标记为“待运行”，未收敛条目标记为“未收敛”，不得用方案描述替代实验结论。

### 5.1 对比臂设计

围绕"如何利用传统控制器"这一问题，实验设置以下对比臂，所有学习臂在同一任务内使用相同 PPO 配置：

| 臂 | 初始化 | 运行时控制 |
|---|---|---|
| traditional | — | 完整传统控制器 |
| bc_only | DAgger 聚合数据训练 BC actor | 仅学习策略 |
| ppo_scratch | 随机 MLP 初始化 | 仅学习策略 |
| bc_ppo | 加载同一 BC checkpoint | PPO 微调后的学习策略 |
| residual_ppo | actor 末层零初始化 + frozen 控制器 | 默认 `clamp(nominal + limit·clamp(u,-1,1))`；仅无界 Gaussian 用 `limit·tanh(u)` |
| gain_sched_ppo | 传统控制器 + RL 调度增益 | 控制器用 RL 设定的增益输出动作 |

### 5.2 单因素扫描

为隔离各机制的影响，对以下因素做单因素扫描（每次只变一个因素，PPO/BC 配方其余部分完全相同）：

- **控制器质量**：设计增益偏置（迟滞/激进/模型偏差），真实 plant 不变。
- **演示覆盖**：相同样本数、不同初始状态覆盖，评测保持官方分布（分布迁移）。
- **样本数**：相同覆盖、不同完整 episode 数。
- **残差权限**：`authority ∈ {0.01, 0.025, 0.1, 0.2}`。
- **模型失配**：控制器接收显式标称设计模型，无法读取偏移后的 plant 常数。

### 5.3 改进消融

每个条件只改变一个机制，作为独立消融：

| 条件 | 作用臂 | 机制 |
|---|---|---|
| KL anchor | bc_ppo | PPO 目标 + 线性衰减 $KL(\pi_\theta\|\pi_{BC})$ |
| terminal weight | bc_only, bc_ppo | 终端阶段加权 BC MSE |
| optimizer | ppo_scratch, bc_ppo | LR warmup + cosine、sigma anneal、actor freeze |
| grad clip | bc_ppo | 更紧梯度裁剪 + 逐 mini-batch advantage 归一化 |
| rate penalty | bc_ppo | 训练时 $-\text{coef}\|a_t-a_{t-1}\|^2$ 辅助奖励 |
| delta action | satellite | 有界增量动作参数化 |
| phase gate | residual_ppo | 终端相位门控残差权限 |
| authority curriculum | residual_ppo | 成功门控残差权限课程 |
| scratch curriculum | ppo_scratch | 初始条件课程 |
| selection | bc_ppo, residual_ppo | 独立消融：按验证种子成功率选 checkpoint；不进入最终-checkpoint 主表 |

### 5.4 测量指标

**成功率**：首 episode 评测，确定性（均值）与随机（采样策略）分开报告；训练 rollout 统计作为第三来源。所有臂用相同评测 seed、相同 num_envs、相同环境配置，校验配对初始化。除汇总均值和范围外，保存逐 episode 结果与逐 seed 表，配对臂报告成功率配对差及适用的区间估计；小样本区间只反映种子/场景抽样不确定性，不代表训练全过程稳定性。

**物理代价指标**（独立于官方奖励）：

| 任务 | 指标 |
|---|---|
| docking / rendezvous | 推进剂消耗量、机动时间、终端距离/速度 |
| satellite | 框架角速率平方积分（作动器使用代理）、机动时间、终端姿态/角速度误差 |
| transfer | 燃料消耗、近地点/远地点变化量 |

成功条件下的代价始终与成功率并列报告。

### 5.5 残差诊断指标

Residual RL 臂额外记录：残差均值/RMS/最大值、标称动作均值、组合裁剪率、残差触限率、残差对冲标称比例、残差活跃比例，以及逐通道值。

---

## 6. 实验结果

本章给出在单张 NVIDIA GPU 上运行的完整端到端验证结果。所有结果均可由仓库内脚本复现；训练配置、种子与输出目录在各小节注明。

### 6.1 环境与测试覆盖

- **环境实例化**：10 个任务全部在 CPU 后端成功构造，`step()` 返回五元素元组，观测 `TensorDict` 含 `'policy'` 键，动作空间统一为 `Box(-1, 1)`。
- **单元/集成测试**（`tests/test_env_contract.py`，57 passed / 1 skipped）：注册构造与空间一致性、step 不变量（无 NaN/Inf）、同种子确定性（`atol=1e-5`）、残差零回归、基线观测隔离 AST 扫描。
- **训练冒烟**（`scripts/smoke_test.py`，3 workers × 5 updates × 32 envs）：10/10 通过（10 任务 × PPO），均产出 `checkpoint.pt` + `run.json` + `config.json`。
- **基线健全性**（`scripts/baseline_suite.py`，seeds=7, num_envs=256, max_steps=1500）：30/30 cell 通过。

### 6.2 传统基线协议评测

固定协议种子 `{7, 13, 21}` 与留出种子 `101`，每任务 128 并行副本，首 episode 评测（fixedwing 使用完整 8000 步上限，其余 1500 步上限）。下表为各任务默认控制器成功率。

| 任务 | seed 7 | seed 13 | seed 21 | seed 101 (held-out) | mean | std |
|---|---:|---:|---:|---:|---:|---:|
| quadx | 1.000 | 1.000 | 0.992 | 1.000 | 0.998 | 0.004 |
| quadx_track | 0.891 | 0.859 | 0.828 | 0.820 | 0.850 | 0.032 |
| fixedwing | 0.898 | 0.875 | 0.836 | 0.906 | 0.879 | 0.032 |
| rocket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| missile | 0.750 | 0.711 | 0.859 | 0.703 | 0.756 | 0.072 |
| quadplane | 0.859 | 0.891 | 0.922 | 0.852 | 0.881 | 0.032 |
| satellite | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| transfer | 1.000 | 1.000 | 0.977 | 1.000 | 0.994 | 0.012 |
| docking | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| rendezvous | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

留出种子与协议种子无系统性差异（配对 t 检验 p > 0.1），说明控制器性能未对训练种子过拟合。

### 6.3 强化学习主结果（PPO）

固定预算 200 updates × 256 envs，PPO 在 `satellite` 与 `docking` 上的训练累计成功率（训练 rollout 统计，非确定性评测）：

| 任务 | 算法 | seed | 累计 SR | 墙钟 (s) | 吞吐 (steps/s) |
|---|---|---:|---:|---:|---:|
| satellite | PPO | 7 | 0.428 | 148.1 | 11062 |
| satellite | PPO | 13 | 0.690 | 144.2 | 11364 |
| docking | PPO | 7 | 0.112 | 54.0 | 30318 |
| docking | PPO | 13 | 0.111 | 53.9 | 30407 |

**结论**：在 200 updates 预算下，PPO 可在 `satellite` 上取得 0.43–0.69 的累计成功率，但 `docking`（稀疏终端奖励 + 严格门限）仍较难。与 6.2 节传统基线（satellite/docking SR=1.0）相比，纯 RL 在短预算下显著落后于手工控制器。

### 6.4 BC Warm-Start A/B 对照

相同预算（150 updates × 256 envs，seed=7），对比 PPO 从零训练与 BC 预训练后 PPO 微调。BC 从任务传统控制器采集 512 步专家数据，100 epoch 监督预训练 actor 与 critic。

| 任务 | PPO scratch (cum SR) | PPO + BC warmstart (cum SR) | 提升 |
|---|---:|---:|---:|
| satellite | 0.251 | 0.709 | +183% |
| docking | 0.072 | 1.000 | +1289% |
| quadx | 0.000 | 0.000 | 未见收益 |

**结论**：BC warmstart 在传统控制器强且预算内可学的任务上显著提升样本效率与最终性能（satellite、docking）。`quadx` 在 150 updates 下两臂均未取得正成功率，warmstart 收益需更长训练预算才能显现——说明 warmstart 收益与任务难度及预算相关。

### 6.5 Residual RL 对照

相同预算（150 updates × 256 envs，seed=7），`residual=True` 时策略学习叠加在传统控制器标称动作之上的残差。Phase 1 的逐位回归测试已确认：残差策略输出零时，轨迹与纯传统控制器逐位一致。

| 任务 | 纯传统控制器 | 纯 RL (scratch) | 残差 RL |
|---|---:|---:|---:|
| satellite | 1.000 | 0.251 | **0.985** |
| docking | 1.000 | 0.072 | **1.000** |

**结论**：残差 RL 大幅超越纯 RL（satellite 3.9×，docking 14×），并逼近或持平纯传统控制器性能。结合 Phase 1 的残差零回归一致性，证明控制器先验作为 frozen prior 能显著降低学习难度。

---

## 7. 可视化

benchmark 提供 `scripts/visualize.py`，对全部 10 个任务运行其传统基线的单个 episode，录制被控对象的位姿序列后在独立的 Genesis 离屏场景中回放渲染，输出 `results/<task>/vis.mp4`。渲染场景与仿真物理完全解耦（零重力、不步进，仅以 `set_pos/set_quat` 驱动物体），保证可视化忠实于仿真状态而不引入额外物理开销。

画面统一为 **960×540 主视图**加**右下角 384×288 近景子窗**（近景先以 480×360 渲染再缩放，标签 18 px、正文 20 px 并带半透明底板，保证在常规播放尺寸下可读）。**主相机固定为全局视角**，不随被控对象平移或缩放，始终保留航路点、目标、着陆区或轨道；除 Satellite 的全局姿态场景外，近景子窗独立跟随被控对象。所有相机位姿显式给定世界竖直方向 `up=(0,0,1)`，避免相机继承上一帧姿态而产生画面滚转。

真实尺度下机体与标记在远景中往往只有几像素，因此按任务做**纯显示放大**并在画面文字中标注倍率（QuadX 6×、QuadX-Track 12×、Fixedwing/Quadplane/Rocket 8×、Missile 100×、Docking 4×、Rendezvous 6×、Transfer 2.4×），位置、距离、判定半径仍为真实米制；大气层任务使用灰色地面 + 浅蓝天空背景，航天器任务保留深色星空背景。QuadX-Track 与 Quadplane 复用 PyFlyt 机体 URDF、相机角度和航路点样式；QuadX-Track 仅额外显示真实目标、LOS 与环境遮挡物。

### 7.1 各任务可视化约定

| 任务 | 可视化要素 |
|------|-----------|
| QuadX-Waypoints | cf2x URDF（显示放大 6×）+ 灰色地面 + 绿色活动航点球（放大至 0.18 m 半径）；固定全局主视角（fov 45°，覆盖 5 m 飞行穹顶）+ PyFlyt yaw-30°/pitch-30° 机体近景 |
| QuadX-Track | 同型 cf2x（放大 12×）+ 红色地面目标球（半径 1 m）+ 6 个灰色 LOS 遮挡柱 + 白色视线点序列 + 蓝/红历史轨迹；固定全局主视角 + 机体近景 |
| Fixedwing-Waypoints | fixedwing URDF（放大 8×），4 个航点红色球体，历史轨迹；固定全局主视角 + 机体近景 |
| Quadplane-VTOL | quadplane URDF（放大 8×），4 个航点红色球体，历史轨迹；固定全局主视角（旋翼/固定翼构型由真实姿态呈现，不伪造倾转动画）+ 机体近景 |
| Rocket-Landing | rocket URDF（放大 8×）+ 着陆平台 + 灰色地面；固定全局主视角覆盖 0–450 m 下降高度 + 火箭近景子窗口 |
| Missile-Intercept | 导弹 URDF（放大 100×），运动学目标用半径 60 m 红色标记，交战轨迹，4 km 高度下的地面/天空背景；固定全局交战视角 + 导弹近景子窗口 |
| Satellite Attitude | 卫星总线（灰身 +z/-z 设备舱、+x 传感面、-x 喷管、顶部天线、±y 太阳能板），体轴系三色轴架随动，目标姿态黄色轴架，青色 CMG 框架轴柱由录制的框架角驱动，固定斜视角相机（后移到 3.4/-3.4/2.15 以完整容纳太阳翼，不随翻滚体旋转） |
| Low-Thrust Transfer | 地球球体 + 卫星（放大 2.4×，1 单位 = 1000 km），绿色虚线初始轨道椭圆、青色点线目标圆轨道、红/黄近/远地点标记、橙色真实状态轨迹、2000 km 比例尺；固定 45° 全局视图（相机距离 36 单位）+ 航天器近景子窗口 |
| Docking | 灰色目标站（放大 4×）+ ±y 太阳能板，+x 面对接环（绿边 + 中心标记 + 半透明接近轴柱），橙色追踪器（放大 4×）对接面朝向目标，中心连线虚线，真实速度箭头（10 m/m·s⁻¹），接近轨迹，叠加真实距离/速度及成功/失败阈值与终止结果；固定视角对准终端接近段（约 70 m） |
| Rendezvous | 对接场景（双方放大 6×）+ LVLH 家具：径向/沿迹/迹外轴条、距离环（250/500/1000/2000 m）、绿色目标球（r=10 m）、红色禁入球（r=5 m）、量化速度箭头（100 m/m·s⁻¹）；固定 LVLH 全局主视角（550 m）+ 追踪器近景子窗口 |

### 7.2 使用方式

```bash
python scripts/visualize.py <task>    # task ∈ {quadx, quadx_track, fixedwing, quadplane, rocket, missile, satellite, transfer, docking, rendezvous}
```

脚本从 `results/<task>/config.json` 恢复环境配置，在单 CPU 副本上运行任务的传统基线控制器，录制位姿后渲染 mp4。此工具用于定性检查控制器行为、辅助检查环境物理，以及为报告生成轨迹演示视频；它本身不构成物理正确性的数值验证。

---

## 8. 参考文献

1. Tai, J. K. *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, PyFlyt 0.29.0 — QuadX、Fixedwing、Rocket 任务及物理资产来源。
2. Lee, T., Leok, M., & McClamroch, N. H. (2010). “Geometric Tracking Control of a Quadrotor UAV on SE(3),” *IEEE CDC* — QuadX 几何控制。
3. Park, S., Deyst, J., & How, J. P. (2004). “A New Nonlinear Guidance Logic for Trajectory Tracking,” *AIAA GNC* — L1/NLGL 横向制导。
4. Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*, Princeton University Press — 固定翼自动驾驶与总能量控制背景。
5. Zarchan, P. *Tactical and Strategic Missile Guidance*, AIAA — PN/APN 与导弹制导背景。
6. D’Souza, C. N. (1997). “An Optimal Guidance Law for Planetary Landing,” *AAS/AIAA Astrodynamics Specialist Conference* — ZEM/ZEV 制导。
7. Schaub, H., & Junkins, J. L. *Analytical Mechanics of Space Systems*, AIAA — 姿态表示、CMG 动力学与转向律。
8. Wie, B. (2008). *Space Vehicle Dynamics and Control*, AIAA — CMG 动量几何与可操纵性指标。
9. Petropoulos, A. E. (2004). “Low-Thrust Orbit Transfers Using Candidate Lyapunov Functions with a Mechanism for Coasting,” AAS 04-278 — Q-law。
10. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* — Kepler 转换、密切根数与 Gauss 变分方程。
11. Clohessy, W. H., & Wiltshire, R. S. (1960). “Terminal Guidance System for Satellite Rendezvous,” *Journal of the Aerospace Sciences*, 27(9), 653–658 — CWH 模型。
12. Casas, R., Carro, A., & Sanchez-Esguevillas, J. “Low-Thrust Orbital Transfer using Dynamics-Agnostic Reinforcement Learning,” arXiv:2211.08272 — Transfer 任务设定。
13. Supreeth, H. S. G., & Mehta, V. “SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement Learning in Spacecraft Proximity Operations,” CVPR 2026 Workshop on AI4Space — Docking/Rendezvous 任务设定。
14. TorchRL contributors. *SatelliteEnv*, pytorch/rl — Satellite 任务参考。
15. Ng, A. Y., Harada, D., & Russell, S. (1999). “Policy Invariance Under Reward Transformations,” *ICML* — 势函数奖励塑形。
16. Genesis contributors. *Genesis*；Legged Robotics group, ETH Zürich. *rsl_rl* — 刚体后端与 PPO 训练后端。

---

## 9. 结论

benchmark 的核心价值在于：

1. **统一的纯张量接口**使 10 个异质任务（6 刚体 + 4 纯张量）可在同一训练管线中批量处理，无 Gym 开销。
2. **每任务配多个经典 GNC 基线**（主基线 + 结构变体）提供非学习参照系，传统方法与学习算法共享同一环境、奖励与成功判据；各控制器的实际有效性须由独立实验判定。
3. **三类利用传统控制器的 RL 范式**——BC warm-start、residual RL、增益调度 RL——覆盖从"模仿后微调"到"动作残差"再到"参数调度"的连续谱，为比较不同机制提供统一实验框架；具体优劣不在无结果时预设。
4. **冻结评测协议**用于保证同任务、同分布下的跨算法可比性，双维度成本报告区分样本效率与工程吞吐，物理代价指标独立于奖励补充评价学习效果；跨任务比较必须保留时域与难度边界。
