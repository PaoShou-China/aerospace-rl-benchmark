# 卫星姿态消融：四元数 PD + 普通 Moore–Penrose CMG 操纵（`SatelliteOrdinaryPinvBaseline`）

**简体中文** · [English](baseline-satellite-mp-pinv.md)

## 1. 概述

本基线是冻结经典卫星控制器的操纵律消融。它保留与 `SatelliteCMGBaseline`（`quat_pd_damped_cmg`）**相同**的四元数误差、**相同**的 PD 增益（`kp=3`、`kd=4`）、**相同**的精确陀螺/储存动量前馈与**相同**的框架速率钳位，只改变力矩→框架速率映射：奇异鲁棒阻尼逆换成 3×N CMG 雅可比的**普通 Moore–Penrose 伪逆**。

目的是对操纵律做干净的 A/B。不加入阻尼调度、零空间运动、奇异规避/逃离、预测控制或参考时间缩放。集群接近奇异时允许普通逆请求极大速率；环境 ±3 rad/s 钳位生效，由此产生的力矩短缺被**测量**而非补救。

## 2. 复用的四元数 PD 路径（约定审计）

两种操纵律均继承 `algorithms/baselines/baselines.py` 中的 `_SatelliteCMGPDBase`，故以下为**同一段代码**而非副本：

- 观测：仅 `'policy'` 组，$[q,e,\omega,\sin\theta,\cos\theta,\dot\theta,m,a_{\rm alloc}]$；
- 四元数 $(w,x,y,z)$ Hamilton，体系 $\omega$，$\dot q=\tfrac12q\otimes(0,\omega_b)$；
- 姿态误差 $e=\log(q^{-1}\otimes q_t)$ 在体系，$\dot e\approx-\omega$，故 $\dot\omega_d=+k_pe-k_d\omega$；
- plant 力矩 $\tau=-h\sum_i\dot\theta_i(g_i\times r_i)=-\kappa\mathrm{gr}^\top\dot\theta$，$\mathrm{gr}_i=\Omega(g_i\times r_i)$、$\kappa=h/\Omega$；
- 雅可比列由观测角重建 $\mathrm{gr}_i=\Omega[\cos\theta_i(g_i\times r_i^0)-\sin\theta_ir_i^0]$，对 4/6-CMG 与 `env._manipulability` 逐位一致；
- PD 力矩 $\tau_{\rm cmd}=I\dot\omega_d+\omega\times(I\omega)+\omega\times h_\Sigma$，$I$ 对角，$h_\Sigma=h\sum_ir_i(\theta_i)$；
- 动作 $a=\mathrm{clip}(\dot\theta/3,-1,1)\in\mathbb{R}^N$（float32），$|a_i|=1$ 恰为 ±3 rad/s 速率界。

重构后已复核：相同 seed 与观测下阻尼臂的动作与奖励与重构前实现**逐位一致**。

## 3. 普通 Moore–Penrose 操纵

力矩映射 3×N 雅可比 $G$（列 $g_i\times r_i$）。$\mathrm{gr}=\Omega G$、$\kappa=h/\Omega$：

\begin{equation}
\dot\theta=-\frac{1}{\kappa}\operatorname{pinv}(\mathrm{gr}^\top)\tau_{\rm cmd},\qquad
\operatorname{pinv}(A)=V\Sigma^+U^\top \tag{1}
\end{equation}
$\Sigma^+$ 截断低于数值秩阈值的奇异值。代码为 `torch.linalg.pinv(gr.transpose(1,2), rtol=..., atol=...)`，可直接用于任意 $N\ge3$。

**数值秩阈值（每次报告记录）**：

| 集群 | `rtol` | `atol` | 规则 |
|---|---|---|---|
| 4-CMG | `4*eps(float64)` = 8.8818e-16 | 0 | LAPACK/`pinv` 默认 `max(m,n)*eps` |
| 6-CMG | `6*eps(float64)` = 1.3323e-15 | 0 | 同上 |

阈值相对于最大奇异值，故不依赖 `gr` 的转速缩放。因阈值在机器精度，真正近奇异但满秩的构型**不**被截断：逆把小奇异方向放大 $\sigma_{\max}/\sigma_{\min}$，由速率钳位而非秩阈值吸收。精确秩亏（如 4-CMG 屋顶奇异）被截断并给出最小范数速率解。

**无奇异处理**：实现刻意不含阻尼调度、零空间项、梯度逃离、MPC 或时间缩放。

## 4. 速率限制与实际力矩记账

每条指令按环境方式精确钳位 $a=\mathrm{clip}(\dot\theta/3,-1,1)$，再以交付速率 $\dot\theta_{\rm applied}=a_{\rm float32}\cdot3$ 在**步前框架角**计算**实际**力矩：

\begin{equation}
\tau_{\rm act}=-\kappa\mathrm{gr}^\top\dot\theta_{\rm applied},\qquad
\tau_{\rm err}=\tau_{\rm act}-\tau_{\rm cmd}. \tag{2}
\end{equation}

报告指标包括 $\lVert\tau_{\rm err}\rVert$ 均值/最大、占集群峰值力矩权限 $3hN$ 的比例（每框架 $h|g_i\times r_i|\dot\theta_{\max}=3h$，因 $|g_i\times r_i|=1$ 且 $\mathrm{gr}$ 内转速与 $\kappa$ 相消）、任一框架饱和的 env 步占比、饱和框架指令占比、峰值**钳位前**请求速率。同记账施于使用相同 PD 指令作力矩参考的 RL 臂，故比较同口径。力矩在步前角（控制器所用同角）求值；plant 在 RK4 子步上积分速率。

## 5. 测试场景

每集群（`num_cmgs=4` 与 `6`）：

| 场景 | 初始条件 | 目标 |
|---|---|---|
| `random_target` | 官方 reset 分布 | $\mathrm{SO}(3)$ 均匀随机 |
| `near_singular` | 动量中性近奇异框架构型 | $\mathrm{SO}(3)$ 均匀随机 |
| `singular_180` | 精确奇异（4-CMG）/ 近奇异（6-CMG） | 绕体 $x$ 轴 180° |

4-CMG 动量中性奇异构型为屋顶奇异 $\theta_i=\pi/2-\text{skew}\approx0.61548$ rad（所有 $i$）：四个力矩方向塌缩到 $-z$（秩 1），$\sum_ir_i=0$ 精确。近奇异变体交替框架偏 ±0.01 rad（$\sigma_{\min}/\sigma_{\max}\approx0.0071$）。6-CMG 近奇异构型数值求得（动量中性罚）：
$\theta=(-2.177243665,1.378914139,-0.192070305,0.273303902,-1.297523787,-0.606455308)$，$\sum_ir_i\approx2\times10^{-4}$、$\sigma_{\min}/\sigma_{\max}\approx6.1\times10^{-5}$。

## 6. 结果

来源：`results/satellite/steering_compare.{json,md}`，由
```bash
python scripts/satellite_steering_compare.py --backend cpu --num-envs 64 \
    --arms damped,mp_pinv,ppo --cmgs 4,6
```
产生。**两个输出文件均未随仓库发布**，下表为该 harness 调用的运行记录（每臂 64 副本、seed 7、每副本一回合、1200 步上限、0.025 s 控制周期下 30 s）。无法从仓库产物复验；可复验的官方协议记录是表后的数据集 B JSON。所有臂见同一目标流与同一共享 PD 力矩参考。`sat envstep` 为任一框架指令饱和的步占比，`sat gimbal` 为饱和框架指令占比，力矩误差为 $\lVert\tau_{\rm act}-\tau_{\rm cmd}\rVert$ 瞬时均值及其占集群峰值权限（$3hN$）比例。失败为 `runaway/timeout`。

### 4-CMG（金字塔，$\kappa=0.2$，峰值权限 240 N·m）

| 场景 | 臂 | 成功 | 稳定步（均值/p90） | 稳定 s | 终态姿态误差 | sat envstep | sat gimbal | 力矩误差 (N·m) | % 权限 | 失败 |
|---|---|---|---|---|---|---|---|---|---|---|
| random_target | damped | 1.000 | 246.6 / 298.2 | 6.16 | 0.0195 | 0.344 | 0.244 | 86.02 | 39.5% | 0/0 |
| random_target | mp_pinv | 1.000 | 255.2 / 317.7 | 6.38 | 0.0198 | 0.370 | 0.295 | 99.49 | 46.3% | 0/0 |
| near_singular | damped | 1.000 | 249.6 / 295.4 | 6.24 | 0.0199 | 0.346 | 0.267 | 88.46 | 40.4% | 0/0 |
| near_singular | mp_pinv | 1.000 | 262.9 / 321.4 | 6.57 | 0.0199 | 0.385 | 0.314 | 112.8 | 52.0% | 0/0 |
| singular_180 | damped | 0.000 | — | — | 3.1416 | 0.000 | 0.000 | 706.9 | 294.5% | 0/64 |
| singular_180 | mp_pinv | 0.000 | — | — | 3.1416 | 0.000 | 0.000 | 706.9 | 294.5% | 0/64 |

峰值钳位前速率请求：damped 115 rad/s，ordinary 3 597 rad/s（`random_target`）与 5 049 rad/s（`near_singular`）。±3 rad/s 钳位是两臂交付速率的界。

### 6-CMG（正交，$\kappa=6.25\times10^{-5}$，峰值权限 0.225 N·m）

| 场景 | 臂 | 成功 | 稳定步（均值/p90） | 稳定 s | 终态姿态误差 | sat envstep | sat gimbal | 力矩误差 (N·m) | % 权限 | 失败 |
|---|---|---|---|---|---|---|---|---|---|---|
| random_target | damped | 1.000 | 233.7 / 283.4 | 5.84 | 0.0199 | 0.267 | 0.190 | 0.0526 | 26.3% | 0/0 |
| random_target | mp_pinv | 1.000 | 234.6 / 282.2 | 5.87 | 0.0198 | 0.272 | 0.211 | 0.0538 | 26.9% | 0/0 |
| near_singular | damped | 1.000 | 235.8 / 279.7 | 5.89 | 0.0199 | 0.274 | 0.203 | 0.0547 | 27.1% | 0/0 |
| near_singular | mp_pinv | 1.000 | 241.8 / 287.0 | 6.05 | 0.0199 | 0.294 | 0.227 | 0.0643 | 31.7% | 0/0 |
| singular_180 | damped | 1.000 | 271.0 / 271.0 | 6.78 | 0.0200 | 0.354 | 0.252 | 0.0790 | 35.1% | 0/0 |
| singular_180 | mp_pinv | 1.000 | 284.0 / 284.0 | 7.10 | 0.0197 | 0.398 | 0.312 | 0.0972 | 43.2% | 0/0 |

力矩误差百分比以瞬时峰值权限归一，而非指令，故 >100% 意味着共享 PD 参考本身要求超过该状态集群可交付量；经典臂保持在 26–52%。两集群绝对力矩**不可比**——6-CMG plant 是更小卫星（$I=0.055$ kg·m²，每转子动量 0.0125 vs 20 N·m·s）。

峰值钳位前速率请求：damped 153 / 165 / 115 rad/s，普通 556 / 274 694 / 233 086 rad/s（三场景）。近奇异情形普通逆请求约 $10^3$–$10^5$ 倍钳位；仅钳位使指令有限。此处无 4-CMG 超时，因 6-CMG 构型满秩（$\sigma_{\min}/\sigma_{\max}\approx6.1\times10^{-5}$）。

**已发布官方协议记录（可复验）**：数据集 B 文件 `results/satellite/baselines/*_seed7.json` 使用相同 `scripts/baseline_eval.py` 记账（256 副本、seed 7、每副本首回合、1200 步时域、名义 4-CMG 任务），读者可从检出复现：

| 臂 | 注册名 | 成功 | 平均回报 | env 步数 |
|---|---|---:|---:|---:|
| 阻尼伪逆（冻结基线） | `quat_pd_damped_cmg` | 100%（256/256） | 64.85 | 330 |
| 普通 Moore–Penrose（本臂） | `quat_pd_mp_pinv_cmg` | 100%（256/256） | 64.54 | 366 |
| 加权阻尼 + 零空间 | `quat_pd_wpinv_cmg` | 100%（256/256） | 65.03 | 337 |

故消融代价体现于每副本首回合所需窗口（330 → 366 env 步，+10.9%）；成功率不变，仍 100%。力矩误差与饱和差异仅由上述未发布的场景 harness 记录。

**PPO 臂**：无 checkpoint——`results/satellite_ppo/` 在检出中缺失——故 PPO 报告为不可用，而非给出数字。

**秩诊断**：低于 `pinv` 秩阈值的 env 步占比处处为 0.0，除精确 4-CMG 屋顶奇异（为 1.0，$\sigma_{\min}=0$ 时逆按预期截断）。故近奇异情形普通逆**不**被秩阈值挽救；放大是真实的，只由速率钳位界定。

**小结**：两律都解出每个非退化 4-CMG 与全部 6-CMG 场景。用普通逆替代阻尼在 4-CMG `random_target` 上代价 +3.5% 稳定步与 +16% 平均力矩误差，4-CMG `near_singular` +5.3%/+28%，6-CMG `random_target` +0.4%/+2.3%，6-CMG `near_singular` +2.6%/+17.6%，6-CMG `singular_180` +4.8%/+23%。代价表现为大得多的钳位前速率请求与略高饱和，成功率不变。精确 4-CMG 屋顶奇异对两经典臂都是真实失败（缺失轴上的 PD 指令不在雅可比值域内），无逃离机制记录。

## 7. 解释与失败情形

- 远离奇异时两逆给出相同最小范数解，成功与终态姿态误差完全一致。平均力矩误差仍不同（4-CMG `random_target` 99.5 vs 86.0 N·m，+16%），因普通臂在机动的奇异穿越中速率摆动被钳位截断（饱和 env 步 0.370 vs 0.344），而阻尼臂更接近可交付力矩。
- 接近奇异时普通逆请求与 $1/\sigma_{\min}$ 成正比的速率（4-CMG 峰值钳位前请求 3.6–5.0 krad/s，6-CMG 最高 2.7e5 rad/s，而 damped ≤165 rad/s）。±3 rad/s 钳位使交付力矩偏离指令，报告力矩误差/饱和比例刻画该损失。每个满秩场景成功率不受影响，代价是稳定步多几个百分点与略高饱和框架比例。
- 精确奇异且指令在雅可比值域外（4-CMG 屋顶绕 x 轴 180°）时，两经典律在缺失轴上指令零力矩，1200 步后超时。这是正常失败，不加入逃离/恢复机制。

## 8. 实现与注册

- 共享 PD 基类 + 阻尼臂：`algorithms/baselines/baselines.py`，`_SatelliteCMGPDBase`、`SatelliteCMGBaseline`。
- 普通臂：`SatelliteOrdinaryPinvBaseline`，注册名 `quat_pd_mp_pinv_cmg`，注册于 `BASELINE_VARIANTS['satellite']`。
- CLI：`scripts/baseline_eval.py`（`--baseline quat_pd_mp_pinv_cmg`、`--num-cmgs 4|6`）与 `scripts/satellite_steering_compare.py`。
- 复用的 RL 臂：PPO 槽（`results/satellite_ppo/checkpoint.pt`）当前为空，故 harness 报告 PPO 不可用而非数字。RL 臂可选：checkpoint 缺失或 CMG 数与训练不符时 harness 自动跳过。

## 9. 运行说明

```bash
# 单臂基线，4-CMG（默认）或 6-CMG
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg --num-cmgs 6

# 完整操纵律比较（damped / ordinary / PPO）
python scripts/satellite_steering_compare.py --backend cpu --num-envs 64 \
    --arms damped,mp_pinv,ppo --cmgs 4,6
```

训练调用不触碰 `results/satellite/`，冻结的基线/审计产物得以保留。`satellite_steering_compare.py` 在 checkpoint 缺失或 CMG 数不符时自动跳过 RL 臂。

## 10. 参考

1. Moore, E. H. (1920); Penrose, R. (1955) —— 广义逆。
2. Golub, G. H. & Van Loan, C. F. *Matrix Computations* —— SVD 秩阈值与最小范数最小二乘。
3. Schaub, H. & Junkins, J. L. *Analytical Mechanics of Space Systems* —— 阻尼比较臂所用奇异鲁棒操纵律。
