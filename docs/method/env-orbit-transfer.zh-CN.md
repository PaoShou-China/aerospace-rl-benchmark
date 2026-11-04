# 低推力近地点抬升轨道转移（`transfer`）

**简体中文** · [English](env-orbit-transfer.md)

## 1. 概述

一颗 100 kg 中地球轨道卫星装有三对正交推力器（每轴 10 mN、连续、$I_{\mathrm{sp}}=4000$ s 电推进），在 166 个控制步（每步 300 s，共 $49{,}800$ s ≈ 13.8 h，约五圈）内抬升初始 $a=10{,}000$ km、$e=0.15$ 轨道（近地点 $r_p=8500$ km、远地点 $r_a=11{,}500$ km）的近地点半径，同时保持远地点半径不变、推进剂消耗低。每次 reset 初始平近点角在 $[-\pi,\pi]$ 均匀重采样；其余根数固定为 $i=60^{\circ}$、$\Omega=120^{\circ}$、$\omega=240^{\circ}$。该任务逐参数复现 gym-satellite-trajectory 的 `PerigeeRaisingEnv`。所有副本在单设备上以批量张量推进，无物理引擎。

## 2. 符号

| 符号 | 含义 |
|---|---|
| $r,v\in\mathbb{R}^3$ | GCRF 位置 [m] 与速度 [m/s] |
| $m$ | 剩余质量 [kg] |
| $\mu$ | 地球引力常数 $3.986004418\times10^{14}$ m³/s² |
| $u\in[-1,1]^3$ | 逐轴推力指令 |
| $f_{\max}=0.01$ N | 单轴推力大小 |
| $r_a,r_p$ | 密切远地点 / 近地点半径 [m] |
| $r_{a0},r_{p0}$ | reset 状态下的远 / 近地点半径 |
| $a,e,i,\Omega,\omega,M$ | 经典 Kepler 根数 |

## 3. 动力学

运动方程为二体点质量引力加每个控制步内保持恒定的推力，并用 Tsiolkovsky 质量消耗：

$$\dot r=v,\qquad \dot v=-\frac{\mu}{\lVert r\rVert^{3}}r+\frac{f_{\max}}{m}u,\qquad \dot m=-\frac{f_{\max}\lVert u\rVert_{1}}{I_{\mathrm{sp}}g_0}, \tag{1}$$

$g_0=9.80665$ m/s²。注意质量消耗正比于指令的 $\ell_1$ 范数：同时点火的每个轴都耗推进剂。7 维状态 $[r,v,m]$ 用经典 RK4、$h=10$ s、每 300 s 控制步 30 个子步积分（参考实现用 Orekit 自适应 Dormand–Prince 8(5,3) 积分同一 ODE；定步长下每圈相位误差远低于奖励粒度）。未激活副本冻结。

初始笛卡尔状态由固定根数与采样的平近点角求 Kepler 方程 $M=E-e\sin E$ 得偏近点角，从 $E_0=M$ 做 8 次固定 Newton–Raphson 迭代：

$$E\leftarrow E-\frac{E-e\sin E-M}{1-e\cos E}, \tag{2}$$

再由近焦点状态经 $R_3(\Omega)R_1(i)R_3(\omega)$ 旋转到 GCRF（Vallado）。由任意笛卡尔状态恢复密切半径：$a=(2/\lVert r\rVert-\lVert v\rVert^2/\mu)^{-1}$ 与偏心率矢量，$r_a=a(1+e)$、$r_p=a(1-e)$。

## 4. 动作空间

| 属性 | 值 |
|---|---|
| 形状 | $(3,)$，每个体轴一项 |
| 范围 | $u\in[-1,1]^3$（硬裁剪） |
| 效果 | 逐轴推力 $f_{\max}u_i$；加速度 $f_{\max}u_i/m$（$m_0$ 时每轴约 $10^{-4}$ m/s²） |

单轴整回合满推力在 100 kg 下约提供 $5$ m/s 的 $\Delta v$。

## 5. 观测空间

float32、未加界的原始状态，置于 `policy` 键下，与参考实现一致（参考声明 $\pm1.1\times$ 特征界 $[T,r_a,r_a,r_a,v_{\max},v_{\max},v_{\max},m_0]$ 但从不归一化）：

| 切片 | 维度 | 内容 |
|---|---|---|
| `[0:1]` | 1 | 已执行任务时间 $t=k\cdot300$ s |
| `[1:4]` | 3 | GCRF 位置 $r$ [m] |
| `[4:7]` | 3 | GCRF 速度 $v$ [m/s] |
| `[7:8]` | 1 | 剩余质量 $m$ [kg] |

## 6. 奖励

任务级目标为参考实现的稀疏终端奖励，仅在步数上限结算一次：

$$R=10^{-4}\big(-\lvert\Delta r_a\rvert+\Delta r_p+2\times10^{5}\Delta m\big), \tag{3}$$

半径以米、质量以 kg：近地点每抬升 1 km 得 $+0.1$，远地点绝对漂移每 km 得 $-0.1$（两个方向都罚），且 $\Delta m=m-m_0\le0$ 随推进剂消耗减小，故每 kg 推进剂扣 $-20$（参考 Table I 权重）。

训练奖励为式 (3) 的精确势函数伸缩分解。令

$$\Phi(s)=10^{-4}\big(-\lvert r_a-r_{a0}\rvert+(r_p-r_{p0})+2\times10^{5}(m-m_0)\big), \tag{4}$$

其在每个新初始条件上恒为零，则每步奖励

$$r_t=\Phi(s_{t+1})-\Phi(s_t), \tag{5}$$

回合求和恰等于终端 $R$（时域固定且 $\gamma=1$，故 PBRS 回报无折扣且严格不变），而每步都得到即时、可归因的信用：在远地点附近顺行点火抬升 $r_p$（$+r_t$）；别处点火增大 $\lvert r_a-r_{a0}\rvert$ 或消耗 $m$（$-r_t$）。终端 $R$ 保留为仅指标量（`terminal_reward`）、成功判据与回合日志。该分解在标准 64 副本批次下很关键：稀疏终端奖励每次 PPO 更新仅经约 12 个回合边界到达学习器，无法训练。

## 7. 成功、终止与截断

- **超时 / 结算**：`count >= 166` 步（$49{,}800$ s）。若副本活动且未违反，回合以 `truncated` 结束并记录终端 $R$。
- **成功**：任务级成功为参考目标在最终轨道上的取值：`complete = grant ∧ (R > 0)`。（参考实现在任意 done 时报告 `is_success=True`，在仅超时终止下为空泛；此处 `completed_successes` 统计终端奖励为正的回合。）
- **防御性终止**（标记 `terminated`；在参考 10 mN 推力下不可达，用于防止超参变化污染指标）：撞击 $r_p<6{,}378{,}137$ m（WGS84 赤道半径）；逃逸 $r_a\le0$（轨道能量非正）；质量耗尽 $m\le1$ kg。
- **bootstrap 语义**：该固定时域任务的步数上限是真正终端，故 `extras['time_outs']` 全为 False，不做 $\gamma V(s')$ 超时 bootstrap。`max_episode_length=168`。

## 8. 初始分布

- 平近点角 $M_0\sim U(-\pi,\pi)$，每次 reset 重采样。
- 固定根数 $a_0=10^{7}$ m、$e_0=0.15$、$i_0=\pi/3$、$\Omega_0=2\pi/3$、$\omega_0=4\pi/3$；笛卡尔状态由第 3 节得出，$r_{p0}=8.5\times10^{6}$ m、$r_{a0}=1.15\times10^{7}$ m（记录自实际初始密切状态）。$e_0$ 于 2026-10-24 由参考的 0.1 提高（基线原为 100%）。
- $m_0=100$ kg；已执行时间 $t=0$；势函数 $\Phi(s_0)=0$。

## 9. 指标与 extras

每步 extras：`time_outs`（全 False）与 `terminal_observations`。完成回合累计：`perigee_raise_km`$=(r_p-r_{p0})/10^3$、`apo_drift_km`$=\lvert r_a-r_{a0}\rvert/10^3$、`propellant_kg`$=m_0-m$、`terminal_reward`。任一副本结束时，`extras['episode']` 报告逐完成副本均值 `success_rate`、`perigee_raise_km`、`apo_drift_km`、`propellant_kg`。累计缓冲为 `returns`、`completed_returns`、`completed_episodes`、`completed_successes`。

## 10. 实现与关键常量

类 `LowThrustTransferEnv` 位于 `environments/satellite/orbit_transfer/env.py`；注册说明见包 `__init__.py`。构造默认：`num_envs=64`、`backend='cpu'`、`seed=7`。

| 常量 | 值 |
|---|---|
| $\mu$ | $3.986004418\times10^{14}$ m³/s²（WGS84） |
| $f_{\max}$ / $I_{\mathrm{sp}}$ / $g_0$ | $0.01$ N / $4000$ s / $9.80665$ m/s² |
| $m_0$ | $100$ kg |
| $(a_0,e_0,i_0,\Omega_0,\omega_0)$ | $(10^{7}$ m, $0.15$, $\pi/3$, $2\pi/3$, $4\pi/3)$ |
| 控制步 $\mathrm{d}t$ | $300$ s |
| 积分器 | RK4，每控制步 30 子步 × $10$ s |
| `max_steps` | 166（$49{,}800$ s，约 5 圈） |
| 奖励尺度 | $10^{-4}(-\lvert\Delta r_a\rvert+\Delta r_p+2\times10^{5}\Delta m)$ |
| 违反阈值 | $r_p<6.378137\times10^{6}$ m；$r_a\le0$；$m\le1$ kg |
| Kepler 求解 | 从 $E_0=M$ 做 8 次 Newton–Raphson |
| 参考观测界（仅声明） | $1.1\times[49800\ \mathrm{s},\ r_a,r_a,r_a,v_{\max},v_{\max},v_{\max},100\ \mathrm{kg}]$ |

## 11. 参考

- gym-satellite-trajectory，`PerigeeRaisingEnv`（github.com/zampanteymedio/gym-satellite-trajectory）——任务参考，逐参数复现。
- Casas, R., Carro, A., Sánchez-Esguevillas, J. "Low-Thrust Orbital Transfer using Dynamics-Agnostic Reinforcement Learning", arXiv:2211.08272 —— 奖励权重（Table I）与任务参数。
- Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —— Kepler 到笛卡尔转换（Newton–Raphson 偏近点角求解）与密切根数恢复。
