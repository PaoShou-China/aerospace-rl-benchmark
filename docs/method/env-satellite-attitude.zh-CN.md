# 控制力矩陀螺卫星姿态控制（`satellite`）

**简体中文** · [English](env-satellite-attitude.md)

## 1. 概述

刚性星体搭载 $N$ 个单框架控制力矩陀螺（CMG）集群。支持两种构型，由构造参数 `num_cmgs` 选择（默认 $N=4$）：

- **4-CMG 金字塔**：每个星体面一个框架，斜角 $\beta=54.7356^{\circ}=\arctan\sqrt{2}$，星体惯量 $I=\operatorname{diag}(75,60,113)$ kg·m²，转子惯量 0.2 kg·m²、转速 100 rad/s，每台 CMG 储存转子动量 $h=20$ kg·m²/s。
- **6-CMG 正交集群**：每个星体面一个 CMG，$I=\operatorname{diag}(0.055,0.055,0.085)$ kg·m²，转子 200 rad/s，每台 $h=0.0125$ kg·m²/s。

每次 reset 星体姿态、体角速度与框架角都在动量中性归零位附近施加 $10^{-2}$ 尺度噪声扰动，目标姿态在 $\mathrm{SO}(3)$ 上均匀采样。智能体指令框架角速率，使星体转向目标，同时绕开集群内部奇异，后者由动量雅可比的归一化可操作度 $m(J)=\sqrt{\det(JJ^{\top})}/h^{3}$ 量化。所有副本在单设备上以批量张量推进，无物理引擎。

## 2. 符号

| 符号 | 含义 |
|---|---|
| $q\in\mathbb{S}^3$ | 星体姿态四元数，标量在前 $(w,x,y,z)$ |
| $\omega\in\mathbb{R}^3$ | 体系星体角速度 |
| $\theta\in[-\pi,\pi)^N$ | 框架角 |
| $\delta=\dot\theta\in\mathbb{R}^N$ | 指令框架角速率 |
| $g_i$ | CMG $i$ 的固定单位框架轴 |
| $r_i(\theta_i)$ | CMG $i$ 的单位转子动量方向 |
| $h$ | 单转子角动量大小（`rotor_h_dyn`） |
| $e=\lVert\log(q^{-1}q^\star)\rVert$ | 相对目标 $q^\star$ 的姿态误差角 |
| $m(J)$ | 归一化集群可操作度 |

## 3. 动力学

转子轴绕框架轴做 Rodrigues 旋转（由构型保证 $g_i\perp r_i(0)$）：

$$r_i(\theta_i)=\cos\theta_i\,r_i(0)+\sin\theta_i\,(g_i\times r_i(0)). \tag{1}$$

转子转速固定、框架角速率为指令时，降阶刚体 + CMG 模型为

$$I\dot\omega=-\omega\times(I\omega)-\omega\times\Big(h\sum_{i=1}^{N}r_i\Big)-h\sum_{i=1}^{N}\delta_i(g_i\times r_i), \tag{2}$$

$$\dot q=\tfrac12 q\otimes(0,\omega),\qquad \dot\theta=\delta. \tag{3}$$

式 (2)–(3) 用经典 RK4、$\mathrm{d}t=2.5$ ms、每控制步 10 子步积分，即 40 Hz 智能体频率（torchrl 参考以 100 Hz 跑相同物理）。每子步后四元数重归一化，框架角回绕到 $[-\pi,\pi)$。已终止/截断副本被掩码冻结且不消耗随机数。

## 4. 动作空间

| 属性 | 值 |
|---|---|
| 形状 | $(N,)$，$N=4$ 或 6 |
| 范围 | $u\in[-1,1]^N$（硬裁剪） |
| 缩放 | 每框架 $\delta=3.0\cdot u$ rad/s |

## 5. 观测空间

float32、未加界，置于 `policy` 键下。总维度 $14+3N$：$N=4$ 为 26，$N=6$ 为 32。下表切片为 $N=4$；$N=6$ 时三个框架块扩为 $[10{:}16],[16{:}22],[22{:}28]$，尾部移至 28 与 $[29{:}32]$。

| 切片 | 维度 | 内容 |
|---|---|---|
| `[0:4]` | 4 | 星体四元数 $q$，标量在前 $(w,x,y,z)$ |
| `[4:7]` | 3 | 姿态误差对数向量 $\log(q^{-1}q^\star)$（最短弧） |
| `[7:10]` | 3 | 体角速度 $\omega$ |
| `[10:14]` | $N$ | $\sin\theta_i$ |
| `[14:18]` | $N$ | $\cos\theta_i$ |
| `[18:22]` | $N$ | 框架角速率 $\dot\theta_i$ |
| `[22:23]` | 1 | 归一化可操作度 $m(J)=\sqrt{\det(JJ^{\top})}/h^{3}$ |
| `[23:26]` | 3 | 阻尼单位力矩分配 $J_u^{\top}(J_uJ_u^{\top}+I_3)^{-1}e_{\log}$ |

分配特征为经单位列雅可比 $J_u=J/h$ 沿姿态误差轴产生力矩的最小范数、单位阻尼框架角速率向量；它暴露固定的 CMG 几何（原始状态的双线性映射），但不泄露任何控制律。

## 6. 奖励

奖励为势函数式，含两个姿态误差平滑门：

$$g(e)=\sigma\Big(\frac{0.8-e}{0.15}\Big),\qquad g_n(e)=\sigma\Big(\frac{0.5-e}{0.15}\Big), \tag{4}$$

以及正的剩余距离势

$$\Phi(s)=30\,e+70\,e\,g_n(e)+\frac{30\lVert\omega\rVert^{2}}{2\cdot1.0}g(e)+\frac{0.5}{m(J)+0.5}. \tag{5}$$

每步奖励

$$r_t=\underbrace{\Phi(s_t)-\Phi(s_{t+1})}_{\text{伸缩进展}}+\underbrace{\mathbb 1[\text{success}]\cdot1.0-\mathbb 1[\text{fail}]\cdot10.0}_{\text{一次性终端奖励/惩罚}}-0.0005\sum_{i=1}^{N}\delta_i^{2}. \tag{6}$$

势函数在每次 reset 的新初始状态上重估，故塑形和伸缩为 $\Phi_0-\Phi_T$ 且与路径无关（Ng et al. 1999），无跨回合泄漏。远离目标时塑形退化为 $30e$ 误差进展项；进入 0.5 rad 后 $g_n$ 门把误差斜率加陡到约 100（否则捕获机动在 PPO 优势噪声下不可见）；进入 0.8 rad 后制动项对快速到达加价并奖励削减 $\lVert\omega\rVert$。奇异项由软分母限制在 $[0.5/2.05,\ 0.5/0.5]$，避免参考形式 $0.5/m(J)$ 在奇异面附近的 $10^{5}$ 级尖峰。四台框架满幅指令每步成本 0.018。

## 7. 成功、终止与截断

每步仅对活动副本检查：

- **成功（终端，无 bootstrap）**：$e<0.05$ rad **且** $\lVert\omega\rVert<0.02$ rad/s。标记 `truncated` 与 `complete`，获一次性 $+1.0$ 稳定奖励并停止。
- **失败（终端）**：$\lVert\omega\rVert>10$ rad/s（失控自旋；正常机动峰值 1–3 rad/s）。标记 `terminated` 并施加一次性 $-10.0$ 惩罚——否则伸缩奖励对失败中性。
- **超时（截断，bootstrap）**：`count >= 1200` 步（30 s）。

`extras['time_outs'] = truncated ∧ ¬terminated ∧ ¬complete`，故成功与失败被学习器视为真终端，仅真正超时触发 $\gamma V(s')$ bootstrap。`max_episode_length=1202`。

## 8. 初始分布

- 目标姿态：经 Shoemake 采样在 $\mathrm{SO}(3)$ 均匀，$q^\star=(\sqrt{1-u_1}\sin2\pi u_2,\ \sqrt{1-u_1}\cos2\pi u_2,\ \sqrt{u_1}\sin2\pi u_3,\ \sqrt{u_1}\cos2\pi u_3)$，$u\sim U(0,1)^3$。
- 星体姿态：$q_0=\operatorname{normalize}((1,0,0,0)+10^{-2}\eta)$，$\eta\sim U(-1,1)^4$。
- 体角速度：$\omega_0=10^{-2}\eta$，$\eta\sim U(-1,1)^3$。
- 框架：$\theta_0=\theta_{\text{home}}+10^{-2}\eta$，动量中性归零位在 $N=4$ 为 $(\pi,0,\pi,0)$、$N=6$ 为 $(\pi,\pi,\pi,0,0,0)$，使 $\sum_i r_i(\theta_{\text{home}})=0$ 精确成立（静止航天器储存约零动量；停在 $\theta\approx0$ 会留下守恒偏置动量，使约 20% 随机目标无法稳定）。
- $\dot\theta_0=0$；塑形势初始化为 $\Phi(s_0)$。

## 9. 指标与 extras

每步 extras：`time_outs`（bootstrap 掩码）与 `terminal_observations`（reset 前观测）。任一副本结束时，`extras['episode']` 报告该步完成副本的 `success_rate`。累计缓冲为 `returns`、`completed_returns`、`completed_episodes`、`completed_successes`。

## 10. 实现与关键常量

类 `SatelliteAttitudeEnv` 位于 `environments/satellite/attitude_control/env.py`；注册说明见包 `__init__.py`。构造默认：`num_envs=64`、`backend='cpu'`、`seed=7`、`num_cmgs=4`。

| 常量 | 值 |
|---|---|
| 控制频率 / 积分器 | 40 Hz；RK4，$\mathrm{d}t=2.5$ ms × 10 子步 |
| `max_steps` | 1200（30 s） |
| 动作裁剪 / 缩放 | $\pm1$ / ×3.0 rad/s |
| $k_1$（误差进展） | 30.0 |
| $k_{1b}$（近端增强）/ 门 / 软度 | 70.0 / 0.5 rad / 0.15 |
| $k_2$（制动）/ 门 / 软度 / $\alpha_{\rm ref}$ | 30.0 / 0.8 rad / 0.15 / 1.0 |
| 奇异权重 / 软度 | 0.5 / 0.5 |
| 控制成本 | $5\times10^{-4}\lVert\delta\rVert^2$ |
| 成功容差 | $e<0.05$ rad，$\lVert\omega\rVert<0.02$ rad/s |
| 稳定奖励 / 失败阈值 / 失败惩罚 | $+1.0$ / $10$ rad/s / $-10.0$ |
| reset 噪声尺度 | $10^{-2}$（2026-10-24 由 $10^{-3}$ 提高；基线原为 100%——初始 $\lVert\omega\rVert$ 最高约 0.017 rad/s，现恰低于 0.02 成功容差） |
| 4-CMG：$I$ / 转子 / $h$ | $(75,60,113)$ / 0.2 kg·m² @ 100 rad/s / 20 |
| 6-CMG：$I$ / 转子 / $h$ | $(0.055,0.055,0.085)$ / $6.25\times10^{-5}$ kg·m² @ 200 rad/s / 0.0125 |

## 11. 参考

- torchrl `SatelliteEnv`（pytorch/rl，MIT）——集群几何、物理与容差的任务参考。
- Wie, B. (2008). *Space Vehicle Dynamics and Control*, AIAA Education Series —— CMG 动量几何与可操作度奇异度量。
- Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations —— 势函数塑形伸缩论证。
- Schaub, H., Junkins, J. L. —— 四元数 PD + 奇异鲁棒阻尼 CMG 操纵，用作传统控制器基线。
