# 卫星姿态基线：四元数 PD + 奇异鲁棒 CMG 操纵（`SatelliteCMGBaseline`）

**简体中文** · [English](baseline-satellite-cmg-pd.md)

## 1. 概述

CMG 姿态控制任务的传统基线是四元数比例—微分（PD）控制器，配精确陀螺前馈，经 Schaub–Junkins 型奇异鲁棒阻尼伪逆操纵律分配到控制力矩陀螺（CMG）集群。控制器指令框架**角速率**（环境原生动作），从不指令体力矩：plant 经构型相关雅可比把框架速率映射为力矩，故操纵律须在线反演该映射，并在集群接近奇异构型时保持框架速率有界。下文约定均已对照 `SatelliteAttitudeEnv`（`environments/satellite/attitude_control/env.py`）核实。

## 2. 接口约定

- **观测**：仅 `'policy'` 组，$o=[q,e,\omega,\sin\theta,\cos\theta,\dot\theta,m,a_{\mathrm{alloc}}]$，$q$ 为星体四元数、$e$ 为体系姿态误差对数映射、$\omega$ 为体速率、$\theta$ 为框架角（$N\in\{4,6\}$）、$m$ 为归一化可操作度、$a_{\mathrm{alloc}}$ 为控制器不使用的分配特征。观测维度 $14+3N$（4-CMG 为 26，6-CMG 为 32）。控制器消费 $e$（`o[:,4:7]`）、$\omega$（`o[:,7:10]`）、$(\sin\theta,\cos\theta)$（`o[:,10:10+2N]`）与 $m$（`o[:,10+3N]`）。
- **构造期环境常量**：`num_gimbals`、转子动量 `rotor_h_dyn` $h$、转速 `rotor_speed` $\Omega$、对角星体惯量 `inertia` $I$、参考转子轴 `rotor_axes_ref` $\{r_i^0\}$、预计算叉积 `g_cross_r0` $\{g_i\times r_i^0\}$、动作缩放 `action_scale` $s_a=3$ rad/s。
- **逐回合状态**：无。控制器跨回合无记忆；`reset_idx` 为基类空操作。设备端诊断累加器（钳位/非有限步数、可操作度下限、峰值储存动量）仅用于报告，从不回馈控制律。
- **动作**：$a\in[-1,1]^N$，环境转为框架速率 $\dot\theta=s_aa$；故 $|a_i|=1$ 恰为 $\pm3$ rad/s 速率界。plant 无硬框架止挡（角度在 $\pm\pi$ 回绕）。

## 3. 控制律

**约定**：四元数为 $(w,x,y,z)$ Hamilton，$\dot q=\tfrac12q\otimes(0,\omega_b)$，$\omega$ 在体系。姿态误差为体系误差四元数的最短弧对数映射
\begin{equation}
e=\log(q^{-1}\otimes q_t)\in\mathbb{R}^3, \tag{1}
\end{equation}
对静止目标满足 $\dot e\approx-\omega$。故绕 $e$ 的正体速率缩小误差，速率指令携带**正**比例符号。

**由观测框架角得集群几何**：$g_i\perp r_i^0$、$|g_i|=1$，Rodrigues 公式给出闭式转子动量方向与力矩雅可比列
\begin{align}
r_i(\theta_i)&=\cos\theta_i\,r_i^0+\sin\theta_i\,(g_i\times r_i^0), \tag{2a}\\
g_i\times r_i(\theta_i)&=\cos\theta_i\,(g_i\times r_i^0)-\sin\theta_i\,r_i^0. \tag{2b}
\end{align}
控制器重建缩放雅可比列 $\mathrm{gr}_i=\Omega(g_i\times r_i)$ 与储存集群动量 $h_\Sigma=h\sum_ir_i(\theta_i)$，与环境自身可操作度/力矩代码逐位一致（两种集群均经探针验证）。

**带精确前馈的 PD**：期望体系角加速度与 plant $I\dot\omega=-\omega\times(I\omega)-\omega\times h_\Sigma+\tau$ 上实现它的力矩为
\begin{align}
\dot\omega_d&=k_pe-k_d\omega, \tag{3}\\
\tau_{\mathrm{cmd}}&=I\dot\omega_d+\omega\times(I\omega)+\omega\times h_\Sigma, \tag{4}
\end{align}
即精确抵消陀螺与储存动量项的反馈线性化（$I$ 对角）。闭环误差动力学化为 $\dot\omega\approx k_pe-k_d\omega$、$\dot e\approx-\omega$。

**奇异鲁棒操纵**：plant 给出 $\tau=-h\sum_i\dot\theta_i(g_i\times r_i)=-\kappa\mathrm{gr}^\top\dot\theta$，$\kappa=h/\Omega$。实现的阻尼伪逆
\begin{equation}
\dot\theta=-\frac{1}{\kappa}\mathrm{gr}\big(\mathrm{gr}^\top\mathrm{gr}+\lambda^2I_3\big)^{-1}\tau_{\mathrm{cmd}},\qquad
\lambda=\Omega\tilde\lambda,\quad \tilde\lambda(m)=\lambda_0+0.15e^{-3m}, \tag{5}
\end{equation}
$m$ 为观测归一化可操作度。$\mathrm{gr}=\Omega G$、$\lambda=\Omega\tilde\lambda$ 时式 (5) 与经典形式 $\dot\theta=-(1/h)G(GG^\top+\tilde\lambda^2I_3)^{-1}\tau_{\mathrm{cmd}}$ 代数等价。自适应阻尼随可操作度崩塌而增大，用转矩精度换奇异附近的框架速率有界（Schaub–Junkins 奇异鲁棒操纵）；据实现 docstring，远离奇异时力矩中继误差约 0.05%，全阻尼时约 3%。阻尼 $3\times3$ 系统因 $\tilde\lambda\ge\lambda_0>0$ 保持对称正定；$\tau_{\mathrm{cmd}}$、$\dot\theta$、$m$ 经 `nan_to_num`，使非有限输入转向零速率而不污染求解。

**指令映射**：
\begin{equation}
a=\operatorname{clip}(\dot\theta/s_a,-1,1),\qquad s_a=3\ \mathrm{rad/s}, \tag{6}
\end{equation}
以 float32 返回。每个被钳步计入 `metrics['clamp_envsteps']`。

## 4. 参数

| 参数 | 值 | 来源 |
|---|---|---|
| $k_p$（姿态增益） | 3.0 | `SatelliteCMGBaseline.kp` |
| $k_d$（速率增益） | 4.0 | `SatelliteCMGBaseline.kd` |
| $\lambda_0$（基础阻尼） | 0.02 | `SatelliteCMGBaseline.lam0` |
| 阻尼调度 | $\tilde\lambda=\lambda_0+0.15e^{-3m}$ | `actions()` |
| $s_a$（动作缩放） | 3.0 rad/s | `env.action_scale` |
| 4-CMG plant | $I=\mathrm{diag}(75,60,113)$ kg·m²、$h=20$ N·m·s、$\Omega=100$ rad/s | `SatelliteAttitudeEnv` |
| 6-CMG plant | $I=\mathrm{diag}(0.055,0.055,0.085)$ kg·m²、$h=1.25\times10^{-2}$ N·m·s、$\Omega=200$ rad/s | `SatelliteAttitudeEnv` |
| 控制频率 | 40 Hz（RK4，$10\times2.5$ ms 子步） | `SatelliteAttitudeEnv` |
| 成功容差 | 姿态误差 $<0.05$ rad，$|\omega|<0.02$ rad/s | `SatelliteAttitudeEnv` |

## 5. 实测表现

全部发布记录使用 `scripts/baseline_eval.py` 记账（每副本首回合、`env_kwargs={}`、1200 步时域）。检出中无 1024 副本卫星产物，故更早的"1024 回合 100%"声明撤回；可验证数字如下：

**协议数据集 A**（`results/baseline_protocol/seed{7,13,21,101}/satellite/baseline.json`，每 seed 128 副本）：

| seed | 成功 | 成功/完成 | 平均回合回报 | env 步数 |
|---|---:|---:|---:|---:|
| 7 | 100% | 128/128 | 66.09 | 330 |
| 13 | 100% | 128/128 | 64.17 | 326 |
| 21 | 100% | 128/128 | 65.47 | 304 |
| 101（留出） | 100% | 128/128 | 63.66 | 340 |

协议 seeds 7/13/21：**100%（384/384）**，平均回报 65.24，平均 320 env 步，每个文件 `complete=true`。

**已发布单记录**：`results/satellite/baseline.json` 为 seed-13 协议记录：128 副本、seed 13、326 env 步、**100%（128/128）**、平均回合回报 **64.17**、`complete=true`。

**数据集 B**（`results/satellite/baselines/*_seed7.json`，256 副本、seed 7；同四元数 PD 路径、不同操纵律）：

| 臂 | 注册名 | 成功 | 平均回报 | env 步数 |
|---|---|---:|---:|---:|
| 阻尼伪逆（本臂） | `quat_pd_damped_cmg`（默认） | 100%（256/256） | 64.85 | 330 |
| 普通 Moore–Penrose | `quat_pd_mp_pinv_cmg` | 100%（256/256） | 64.54 | 366 |
| 加权阻尼 + 零空间 | `quat_pd_wpinv_cmg` | 100%（256/256） | 65.03 | 337 |

冻结控制器审计（`docs/evaluation_protocol.zh-CN.md` §2）记录该控制器在奇异、180° 大角度机动与 6-CMG 审计套件上完全成功，并列为该任务的冻结行为克隆数据源。这些逐场景审计结果由 `scripts/satellite_steering_compare.py` 测得，其输出 `results/satellite/steering_compare.{json,md}` **未发布**；[baseline-satellite-mp-pinv.zh-CN.md](baseline-satellite-mp-pinv.zh-CN.md) 中场景级数字为运行记录而非可复验产物。

## 6. 已知局限

- 阻尼逆刻意在奇异附近用转矩保真度换框架速率有界（有记录中继误差远场约 0.05% vs 全阻尼约 3%）；它不把集群**导离**奇异（无零运动/梯度项），只降低逆的敏感性。
- $\pm s_a$ 的框架速率饱和由最终钳位吸收并计数，不做补偿；长时间饱和机动在力矩方向上是开环的。
- 律调校于构造时从环境取得的冻结集群几何与惯量；无在线辨识。
- 诊断指标为全局累加器，非逐回合状态。
- 保留本控制器 PD 路径与增益、仅把阻尼逆换为普通 Moore–Penrose 伪逆的操纵律消融见 [baseline-satellite-mp-pinv.zh-CN.md](baseline-satellite-mp-pinv.zh-CN.md)（`SatelliteOrdinaryPinvBaseline`，`quat_pd_mp_pinv_cmg`）。

## 7. 实现

- 控制器：`SatelliteCMGBaseline`（注册名 `quat_pd_damped_cmg`、`BUILDERS['satellite']`）位于 `algorithms/baselines/baselines.py`。
- 环境：`environments/satellite/attitude_control/env.py`，类 `SatelliteAttitudeEnv`（`conventional_controller = SatelliteCMGBaseline`）。
- 评测：`scripts/baseline_eval.py satellite`。

## 8. 参考

1. Schaub, H. & Junkins, J. L. *Analytical Mechanics of Space Systems*（AIAA Education Series）—— 奇异鲁棒（阻尼伪逆）CMG 操纵律。
2. Wie, B. (2008). *Space Vehicle Dynamics and Control*，AIAA —— CMG 动量几何与可操作度（奇异）度量。
3. TorchRL contributors. *SatelliteEnv*（pytorch/rl，MIT）—— 参考任务定义（奖励权重、容差、reset 分布）。
