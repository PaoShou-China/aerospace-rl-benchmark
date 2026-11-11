# 火箭着陆环境（`RocketLandingEnv`）

**简体中文** · [English](env-rocket-landing.md)

## 1. 概述

火箭着陆任务忠实移植自 PyFlyt 0.29.0 `RocketLandingEnv`（随仓库冻结的 `original_pyflyt` 中 `gym_envs/rocket_envs/rocket_landing_env.py` + `rocket_base_env.py`）：一枚 Falcon-9 级一级助推器在 400–450 m 高度、−100 m/s 垂直速度释放，须下降、消除水平偏移并稳定落在地面平台上。与上游唯一不同的是刚体物理引擎（Genesis 而非 PyBullet）；任务定义（奖励、动力学模型、回合逻辑、初始条件）逐字沿用上游。

奖励为 2026-10-28 的 "v5" 逐策略步稠密塑形（benchmark v2 使用该奖励，配 40 Hz 策略频率与 2 s 稳定成功判据）。v4 的仅俯冲塑形被 5 m/s 坠冲策略刷分：65.5M 转场、零成功、正回报。v5 增加了可行下降速度走廊与一次性致命惩罚；环境在共享 Genesis 场景上完全张量化。

## 2. 符号

| 符号 | 含义 |
|---|---|
| $p\in\mathbb{R}^{3}$ | 助推器原点世界位置 [m]；平台位于原点 |
| $p_{xy}$ | 水平分量 $(p_x,p_y)$ |
| $q$ | 姿态四元数（内部 wxyz，观测 xyzw） |
| $v,\omega$ | 世界系线速度 / 角速度；$v_b,\omega_b$ 体系 |
| $(\phi,\theta,\psi)$ | 滚转、俯仰、偏航（$q$ 的 Euler 角） |
| $f\in[0,1]$ | 燃料比例（出生值 0.05） |
| $a\in[-1,1]^{7}$ | 原始动作；$\bar a$ 其在上游 Box 中的仿射像 |

## 3. 动力学与控制层级

benchmark **v2** 下三层速率：**策略 40 Hz**（$\Delta t=1/40$ s；研究冻结快照的速率——60 Hz 无法整除 120 Hz 控制环）、**控制 120 Hz**（每策略步 `inner_steps=3` 个控制周期，动作保持）、**物理 240 Hz**（每控制周期 2 子步，$h=1/240$ s）于 Genesis 刚体求解器上逐子步更新作动。所有速率从 plant 读回（`agent_hz` 为构造参数；两个支持版本默认 40 Hz）。

机体模型为上游 F9 级助推器，按 $10^{3}$ 缩放自 Falcon 9 一级（参考推进剂 123.5 t RP-1 + 287.4 t LOX = 410.9 t，参考推力 7.607 MN）：

- **质量特性**：结构质量 138.2 kg（`base_link` 91.0 kg、`booster_link` 47.0 kg、四片栅格翼各 0.05 kg）。`fueltank_link` 质量每物理步改写为 $f\cdot410.9$ kg，惯量 $f\cdot\mathrm{diag}(1678,1678,7.01)$ kg·m²，故出生质量 $138.2+0.05\times410.9\approx158.7$ kg（满燃料最大 549.1 kg）。
- **助推器**：可重启发动机，每物理步当点火指令超过 0.5 时置点火锁存。油门目标为重映射 $u\in[0,1]\mapsto0.39+0.61u$，经一阶滞后（$\tau=0.01$ s）跟踪并每物理步乘 $1+0.01\mathcal N(0,1)$ 噪声。推力 $T=\text{throttle}\times7607.0$ N（39% 下限 = 2966.7 N），作用于 `booster_link`，方向为万向节方向 $(\sin g_y,\ -\sin g_x\cos g_y,\ \cos g_x\cos g_y)$；两轴万向节限 $\pm5^{\circ}$，滞后 $\tau=0.01$ s。满油门耗油 1.451 kg/s；$f=0$ 后推力为零。
- **栅格翼**：四个升力面（`fin_pos_x, fin_neg_x, fin_pos_y, fin_neg_y`；弦 0.5 m、展 0.5 m、$C_{l\alpha}=2\pi$、$\eta=0.65$、失速 $\pm20^{\circ}$、偏度限 $\pm45^{\circ}$、滞后 $\tau=0.05$ s）。鳍指令 $(a_x,a_y,a_\psi)$ 混合为 $(a_y{+}a_\psi,\ a_y{-}a_\psi,\ a_x{-}a_\psi,\ a_x{+}a_\psi)$，裁剪到 $[-1,1]$。
- **体阻力**：$F=-\tfrac12\rho C_d A\lvert v_{\text{air}}\rvert v_{\text{air}}$，$\rho=1.225$ kg/m³，$C_d=(1.16,1.16,2.0)$，$A=(1.7649,1.7649,0.1075)$ m²，作用于 `base_link`。`aero="realistic"` 时 $\rho$ 取该高度 ISA 密度，升力面升/阻力系数乘压缩性因子，发动机推力随环境压力下降而增大，舵机速率限幅 200 deg/s 与 1% 死区，观测带 IMU/GPS 噪声与一步延迟，并为助推器启用 Genesis 接触约束求解器（realistic 模式下所有大气任务均启用），使着陆台能够真实支撑、倾倒并提供摩擦耗散，而非仅做接触检测；v2 成功还要求连续稳定接触 0.5 s，而非单次触地采样。

reset 时先做 10 次零指令 warmup（20 物理子步）以沉降接触与作动器。

## 4. 动作空间

策略输出 $a\in[-1,1]^{7}$，逐元裁剪（`clip_actions=1.0`）后仿射映射到上游 Box，下界 $(-1,-1,-1,0,0,-1,-1)$、上界 $(1,1,1,1,1,1,1)$。该仿射映射是刻意的：零均值高斯策略因此约 50% 的步点火，与上游探索一致。

| 索引 | 原始范围 | 映射范围 | 物理含义 |
|---|---|---|---|
| 0 | $[-1,1]$ | $[-1,1]$ | 鳍滚转指令 $a_x$ |
| 1 | $[-1,1]$ | $[-1,1]$ | 鳍俯仰指令 $a_y$ |
| 2 | $[-1,1]$ | $[-1,1]$ | 鳍偏航指令 $a_\psi$ |
| 3 | $[-1,1]$ | $[0,1]$ | 点火（每物理步 >0.5 即点火） |
| 4 | $[-1,1]$ | $[0,1]$ | 油门（重映射到 39–100% 推力区间） |
| 5 | $[-1,1]$ | $[-1,1]$ | 万向节 $x$（$\pm5^{\circ}$） |
| 6 | $[-1,1]$ | $[-1,1]$ | 万向节 $y$（$\pm5^{\circ}$） |

## 5. 观测空间（30 维，float32）

| 切片 | 符号 | 单位 | 说明 |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | 体角速度 |
| [3:7] | $q$ (xyzw) | — | 姿态四元数 |
| [7:10] | $v_b$ | m/s | 体系线速度 |
| [10:13] | $p$ | m | 世界位置 |
| [13:20] | $\bar a_{t-1}$ | — | 上一映射动作（第 4 节范围） |
| [20:24] | — | — | 栅格翼作动（4） |
| [24:25] | — | — | 发动机点火标志 |
| [25:26] | $f$ | — | 燃料比例 |
| [26:27] | — | — | 油门状态 |
| [27:29] | $g_x,g_y$ | — | 万向节指令（度缩放前） |
| [29:30] | — | — | 平台接触标志 |

## 6. 奖励

每**策略**步一个稠密奖励，由整步状态差计算（$\Delta t=1/40$ s），分量对应 `REWARD_KEYS`：

\begin{align}
r_{\text{hover}} &= -0.3, \tag{1a}\\
r_{\text{hold}} &= \frac{0.005}{\lVert p_{xy}\rVert+0.1}\ \le 0.05, \tag{1b}\\
r_{\text{lateral}} &= 10\big(\lVert p_{xy}^{-}\rVert-\lVert p_{xy}\rVert\big), \tag{1c}\\
r_{\text{descent}} &= 0.1(z^{-}-z), \tag{1d}\\
r_{\text{decel}} &= 0.1(\Delta v_z+1)e^{-\max(z,0)}\sigma(v_z),\quad \sigma(v_z)=\begin{cases}+1 & v_z<0\\ -1 & \text{否则}\end{cases} \tag{1e}\\
r_{\text{attitude}} &= -0.5\lvert\omega_{b,z}\rvert-0.5\sqrt{\phi^{2}+\theta^{2}}, \tag{1f}\\
r_{\text{contact}} &= \mathbb 1[\text{任一平台接触}]\cdot(0.5-1.0\lvert v_{z,c}\rvert), \tag{1g}\\
r_{\text{corridor}} &= -0.5\,\mathrm{relu}\big(-v_z-(0.5+\sqrt{2\cdot2\,z})\big), \tag{1h}\\
r_{\text{success}} &= 5000\cdot\mathbb 1[\text{本步首次稳定}], \tag{1i}\\
r_{\text{fatal}} &= -500\cdot\mathbb 1[\text{新终止转场}], \tag{1j}
\end{align}
其中 $(\cdot)^{-}$ 为策略步起始值，$\Delta v_z$ 为整步垂直速度变化，$v_{z,c}$ 为首次接触内步的世界下降速度。总奖励 $r=\sum_k r_k$；分量和累计到 `reward_components` 并以 `rew_*` 指标报告。设计依据（$\gamma=0.995$ 的折现回报分析）：成功奖励满足 $B=5000\gg\gamma c/(1-\gamma)\approx70$（$c$ 为最大可持续逐步接触流），故延迟可达成功永不划算；走廊 (1h) 由保守可行剖面 $v_{\text{allow}}(z)$ 界定下降速度，一次性致命成本 (1j) 使坠冲回报为负而非逃逸剩余悬停成本；保持项每步上限 0.05，低于 0.3 悬停成本，故驻留也无法刷分。所有系数可经 `reward` dict 覆写，并记录进 `cfg` / `run.json` 供演化扫描臂使用。

## 7. 成功、终止与截断

每 120 Hz 控制周期按以下顺序判定：

1. **致命接触（终止）**：除（火箭, 平台）外的任何接触，或 $z<0$；或某平台接触步的**上一**内步状态速度 $>1.0$ m/s 或自旋 $>0.35$ rad/s。
2. **出界（终止）**：$\lVert p_{xy}\rVert>200$ m 或 $z>500$ m。
3. **成功（不终止、不截断）——v2 稳定判据**：主指标为 `ever_stable_2s`：飞行器须在平台上连续稳定 2 s。策略步样本**稳定**当且仅当有平台接触、无致命状态、速度 $<0.02$ m/s、自旋 $<0.02$ rad/s、倾角 $\sqrt{\phi^2+\theta^2}<0.1$ rad；中断则重置连续计数，致命/越界终止使终态样本无效。40 Hz 下需 `ceil(2*40)+1=81` 个连续样本（旧 60 Hz 为 121）。达到连续长度即锁存 `ever_stable_2s` 并支付一次性奖励；回合**继续**，稳定接触流（式 1g）继续支付至时限。旧的瞬时平台接触锁存（`ever_success`，另有 `terminal_single_point` / `terminal_stable_2s`）仍作诊断报告。真实尺度 v1 使用第 13 节单独的稳定定义。
4. **时限（截断）**：`count >= 1200` 策略步（30 s），在整个控制步之后判定；唯一的截断来源。

平台接触判定用上一内步速度，因为上游在触地后仍继续仿真：致命门与成功门之间的触地状态会继续并可能稳定。`extras['time_outs'] = truncated & ~terminated & ~complete`，标记纯时限截断供价值 bootstrap，排除失败与已稳定回合。

## 8. 初始分布

每副本一次 6 维均匀抽样，燃料比例 $f_0=0.05$：

\begin{equation}
x,y\sim\mathcal U[-20,20]\ \text{m},\quad z\sim\mathcal U[400,450]\ \text{m},\quad
\phi,\theta,\psi\sim\mathcal U[-0.3,0.3]\ \text{rad},\quad v_0=(0,0,-100)\ \text{m/s},\quad\omega_0=0. \tag{2}
\end{equation}
上游的随机投放速度选项是死代码（键名拼错为 `randoimize_drop`），故只有 −100 m/s 投放速度生效；该怪癖逐字复现。

## 9. 扰动

可选风：`steady_wind`（世界系常向量，m/s）、`dryden∈[0,1]`（0 关闭）、`severity∈{light, moderate, severe}`（MIL-F-8785C 低空参考标准差 $(1.5,1.5,0.7)$、$(3.0,3.0,1.4)$、$(6.0,6.0,3.0)$ m/s；尺度长度 $(200,200,50)$ m）。一阶阵风滤波每策略步推进一次；气动项对 $v_{\text{air}}=v-v_{\text{wind}}$ 评估。静风（默认）动力学逐位不变。

## 10. 指标与 extras

回合完成时，`extras['episode']` 报告 `success_rate`、`pad_touch_rate`、首触遥测 `touch_speed`、`touch_ang_vel`、`touch_tilt`（仅监测，首触不参与回合逻辑），以及分量奖励均值 `rew_{hover,hold,lateral,descent,decel,attitude,contact,success}`；长期累加器在 `completed_metrics`。`extras['terminal_observations']` 携带全副本 float32 终态观测；`extras['time_outs']` 为第 7 节超时掩码。

## 11. 实现与关键常量

- 文件：`environments/rocket/env.py`；类 `RocketLandingEnv`；奖励在模块函数 `rocket_shaping`；任务 id `rocket`。
- Plant：`environments/common/plant.py`（`TensorPlant`，vehicle `'rocket'`）；参数 `environments/common/original_pyflyt/models/vehicles/rocket/rocket.yaml`。
- 传统基线：`RocketPDGBaseline`（ZEM/ZEV 动力下降制导 + TVC/鳍姿态 PD）。

| 常量 | 值 |
|---|---|
| `agent_hz` / `inner_steps` | 40 / 3（控制 120 Hz，物理 240 Hz） |
| `max_steps` | v2：1200（30 s）；真实尺度 v1：2400（60 s） |
| `num_actions` / 观测维度 | 7 / 30 |
| `HOVER_COST` / `HOLD_COEF` | 0.3 / 0.005 |
| `LATERAL_COEF` / `DESCENT_COEF` / `DECEL_COEF` | 10.0 / 0.1 / 0.1 |
| `YAW_COEF` / `TILT_COEF` | 0.5 / 0.5 |
| `CONTACT_BASE` / `CONTACT_VEL_COEF` / `SUCCESS_BONUS` | 0.5 / 1.0 / 5000.0 |
| `ic_xy_half` / `ic_z_min` / `ic_euler_half` | 20 m / 400 m（跨 50）/ 0.3 rad |
| 最小 / 最大推力；万向节范围 | 2966.7 / 7607.0 N；$\pm5^{\circ}$ |
| 燃料容量 / 速率 / 出生比例 | 410.9 kg / 1.451 kg/s / 0.05 |
| 助推器 / 万向节 $\tau$；油门噪声 | 0.01 s / 0.01 s；1% 乘性 |
| 成功门 | $|v|<0.02$ m/s，$|\omega|<0.02$ rad/s，倾角 $<0.1$ rad |
| 致命门 / 边界 | 接触速度 $>1.0$ m/s 或自旋 $>0.35$ rad/s；$\lVert p_{xy}\rVert>200$ m，$z>500$ m，$z<0$ |

## 12. 参考

- PyFlyt 0.29.0，随仓库打包于 `environments/common/original_pyflyt/`：`gym_envs/rocket_envs/rocket_landing_env.py` 与 `rocket_base_env.py`（任务定义）、`core/drones/rocket.py`（机体模型）、`core/abstractions/boosters.py`（点火与油门重映射）。
- MIL-F-8785C（低空 Dryden 湍流模型），经 `environments/common/disturbances.py` 实现。

## 13. 真实尺度终端着陆版本（`v1`）

`benchmark_version='v1'` 为真实尺度公开数据工程模型；旧缩放版 v1 已移除。v2 仍是独立的缩放遗留基准。该模型未经 SpaceX 飞行数据验证，只仿真终端动力着陆，不含进入点火、超声速气动、推进剂晃动、结构柔性或着陆腿压溃。

| 项目 | 名义值 / 约定 | 来源 |
|---|---|---|
| 一级 + 级间段 | 41.2 + 6.5 m；直径 3.66 m | SpaceX Falcon 9 用户手册几何 |
| 干质量 | 25,600 kg | 公开社区估计；22–28 t 不确定度 |
| 残余推进剂 | 采样 6,000–10,000 kg；LOX:RP-1 = 2.33 | 情景选择，非官方储备 |
| 发动机 | 一台 Merlin 1D+，海平面最大 845 kN | 公开发动机数据 |
| 油门 / Isp | 57–100%；海平面 282 s | 公开推力范围与发动机数据 |
| 最大质量流 | 305.55 kg/s | 由 `T/(Isp*g0)` 导出 |
| 贮箱模型 | 独立 LOX/RP-1 柱，变质量/惯量/质心 | 工程开系统近似 |
| TVC / 栅格翼 | ±5° / 四面，±20° | 公开几何 + 工程作动先验 |
| 平台 / 支脚 | 半径 26 m；基座原点静止于 z=0.65 m | 仿真器任务几何 |

7 维动作与 30 维观测接口不变。燃料观测为剩余 LOX+RP-1 总量除以 410.9 t。初始脚高 0.8–1.5 km，下降率 60–120 m/s，水平偏移至多 100 m，倾角至多 3°。时限 60 s。平台接触快于 6 m/s、自旋高于 0.35 rad/s 或倾角高于 15° 为致命。成功及其一次性奖励均要求连续 2 s 速度 <0.1 m/s、自旋 <0.02 rad/s、倾角 <5°；该版本瞬时接触不再支付成功奖励。

体气动、鳍系数与刚性足接触均为显式工程先验。平台支撑的角/侧向阻尼代表腿足能量耗散，并非实测着陆腿模型。真实尺度传统控制器使用高度调度动力下降、直立 TVC/鳍姿态控制、终端关机与触地关车锁存。其不确定性应以参数失配/域随机化检验，而非解读为已验证的 F9 数据。

## 14. 基准 v2：接触模型与停稳判据

v1 基准的 Genesis 着陆接触配置存在结构性问题：三条平底碰撞条触台后持续抖动，即使发动机关闭的轻柔触地也会被弹回；瞬时 ever-success 会把抖动期间短暂满足阈值的状态记为成功，使评分偏离"火箭着陆并停稳"的任务目标。v2 改用球形垫接触，并以连续 2 秒停稳作为判据。

### 14.1 接触验证

在 128 个随机轻柔触地初始状态上，发动机关闭、舵面中立、默认求解器、仿真 10 s：

| 接触诊断 | 原版三条 | v2 六球形垫 |
|---|---:|---:|
| 接触后抬升超过 1 cm | 108/128 | 0/128 |
| 致命结束 | 88/128 | 0/128 |
| 曾满足瞬时成功条件 | 85/128 | 128/128 |

完整首回合评测（每列 4×128，共 512 回合），以连续 2 秒停稳为主指标：

| 条件与策略 | 开发集 | 独立集 |
|---|---:|---:|
| nominal / seed | 508/512 | 504/512 |
| nominal / PDG | 512/512 | 512/512 |
| shifted / seed | 281/512 | 293/512 |
| shifted / PDG | 290/512 | 314/512 |

逐环境接触记录与源文件 SHA-256 见
[`evidence/rocket_v2/`](../evidence/rocket_v2/)。

### 14.2 main 中对应位置

- **接触资产**：`environments/common/original_pyflyt/models/vehicles/rocket/rocket_v2.urdf`，每条腿用两个半径 0.025 m 的球形碰撞垫。原 `rocket.urdf` 保留。
- **物理模型**：`environments/common/plant.py` 与 `environments/rocket/rocket_benchmark.py`，逐回合物理扰动用独立计数器随机流积分，种子偏移 104729。
- **任务评分**：`environments/rocket/env.py` 使用 v2，主指标 `ever_stable_2s`；奖励、致命判据、初始分布、30 维观测、7 维动作与 30 s 时域保持不变。
- **评测**：`scripts/rocket_benchmark_eval.py` 按 4×128 批次统计完整首回合。

### 14.3 连续停稳指标

每个策略步结束时检查：平台接触、速度范数 < 0.02 m/s、角速度范数 < 0.02 rad/s、滚/俯合成倾角 < 0.1 rad、无致命或出界终止。连续满足条件的首末样本跨度须达到 2 秒：**40 Hz 下 81 点**（60 Hz 下 121 点）。接触中断或任一条件违反即重置连续计数器。

### 14.4 逐回合物理扰动

| 参数 | nominal | shifted | stress |
|---|---|---|---|
| 推力缩放 | 1 | [0.92, 1.08] | [0.85, 1.15] |
| 表面力/力矩缩放 | 1 | [0.6, 1.4] | [0.3, 1.7] |
| 发动机与万向节时间常数（s） | 0.01 | [0.005, 0.03] | [0.005, 0.06] |
| 两轴万向节偏置（deg） | 0 | [-0.3, 0.3] | [-1, 1] |
| 水平定常风（m/s） | 0 | [0, 5] | [0, 10] |

main 既有的 `steady_wind` / `dryden` 接口保留；额外配置的风叠加在逐回合风之上。

### 14.5 使用

```bash
python scripts/rocket_benchmark_eval.py --backend gpu --variation nominal --out results/rocket-v2-nominal
python scripts/rocket_benchmark_eval.py --backend gpu --variation shifted --out results/rocket-v2-shifted
```

默认开发种子为 1000007、2000007、3000007、4000007。报告含逐回合记录、完整性、样本数、Wilson 95% 区间、初始状态与源哈希。checkpoint 评测使用 `scripts/eval.py`。
