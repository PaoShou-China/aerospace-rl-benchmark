# Docking 基线：辛精确 CWH 离散 LQR + 约束层（`DockingCWHLQRBaseline`）

**简体中文** · [English](baseline-docking-cwh-lqr.md)

## 1. 概述

对接任务的传统基线是在 Clohessy–Wiltshire–Hill（CWH）plant 的**精确离散化**上设计的离散线性二次调节器（LQR），并附加三个显式约束处理层：距离走廊护盾、终端捕获网与燃料降额。由于设计模型逐位复现环境的辛 Euler 积分器，对分段常值加速度指令，LQR 模型相对 plant **离散化误差为零**；全部近似只来自约束层与逐轴输出饱和。

## 2. 接口约定

- **观测**：仅 `'policy'` 组：原始 6 维 LVLH 相对状态 $o=[p,v]$，$x$ 径向、$y$ 沿迹、$z$ 交轨。
- **构造期环境常量**：追踪器质量 $m=500$ kg，逐轴推力限 $F_{\max}=10$ N（逐轴权限 $a_{\max}=F_{\max}/m=0.02$ m/s²），控制周期 $\Delta t=1$ s，$I_{sp}=300$ s，$g_0=9.80665$ m/s²，成功容差 `pos_goal`$=1$ m、`vel_goal`$=0.1$ m/s，距离失败界 `range_fail`$=200$ m，燃料预算 `fuel0`$=2$ kg，时域 `max_steps`$=500$ 步，平均角速度 $n=\sqrt{\mu/a^3}=1.1314\times10^{-3}$ rad/s（$\mu=3.986\times10^{14}$ m³/s²，$a=6.778\times10^{6}$ m，400 km 圆轨道目标）。
- **逐回合状态**：控制器自行积分实测燃料 $\hat f$ 与剩余步计数器；二者及层统计由 `reset_idx(mask)` 清零，评测循环每步以环境 reset 掩码调用。
- **动作**：$u\in[-1,1]^3$，环境映射为逐轴推力 $F=F_{\max}u$；燃料累计 $\Delta f=\lVert F\rVert_2\Delta t/(I_{sp}g_0)$。

## 3. 控制律

**设计模型（精确辛离散化）**：连续 CWH plant 为 $\dot v=Gp+Hv+a$，$G=\mathrm{diag}(3n^2,0,-n^2)$、$H=\big[\begin{smallmatrix}0&2n&0\\-2n&0&0\\0&0&0\end{smallmatrix}\big]$，环境用速度优先的辛 Euler 步积分。堆叠状态 $x=(p,v)$ 的匹配离散模型
\begin{equation}
x_{t+1}=A_dx_t+B_da_t,\quad
A_d=\begin{bmatrix}I+\Delta t^2G&\Delta t(I+\Delta t H)\\\Delta t G&I+\Delta t H\end{bmatrix},\quad
B_d=\begin{bmatrix}\Delta t^2I\\\Delta t I\end{bmatrix}, \tag{1}
\end{equation}
与 plant 完全一致（helper `_cwh_lqr`，`method='symplectic'`；`'taylor2'` 变体仅用于复现旧有增益）。

**Riccati 增益（按任务容差 Bryson 归一化）**
\begin{equation}
Q=\mathrm{diag}\Big(\frac{1}{p_{\mathrm{goal}}^2}I_3,\frac{w_{\mathrm{vel}}}{v_{\mathrm{goal}}^2}I_3\Big)
=\mathrm{diag}(1,1,1,300,300,300),\quad
R=\frac{1}{(\kappa a_{\max})^2}I_3,\ \kappa=0.35,\ w_{\mathrm{vel}}=3, \tag{2}
\end{equation}
\begin{equation}
P=\mathrm{DARE}(A_d,B_d,Q,R)\ (\texttt{scipy.solve\_discrete\_are}),\quad
K=(R+B_d^\top PB_d)^{-1}B_d^\top PA_d,\quad
a_{\mathrm{lqr}}=-Kx. \tag{3}
\end{equation}
该归一化下纯位置反馈在成功边界恰好需求 $\kappa a_{\max}$（35% 权限，即 65% 终端裕度），$w_{\mathrm{vel}}=3$ 使闭环接近临界阻尼；据实现 docstring，穿越 $|p|=1$ m 时验证捕获裕度为 $|v|\approx0.066$ m/s，比 `vel_goal` 低 34%。远场接近因物理而饱和（0.02 m/s² 权限制动约 100 m 接近量），由逐轴输出饱和吸收为 bang–brake 段。

**约束层 1 —— 距离走廊护盾**：唯一状态约束是 $|p|<$`range_fail`（对接无禁飞球）。外向径向速率 $v_{\mathrm{out}}=v\cdot\hat p$、假设制动权限 $a_b=a_{\max}/2$，制动增广距离
\begin{equation}
d_{\mathrm{stop}}=\lVert p\rVert+\frac{\max(v_{\mathrm{out}},0)^2}{2a_b}. \tag{4}
\end{equation}
若 $v_{\mathrm{out}}>0$ 且 $d_{\mathrm{stop}}>0.95\,$`range_fail`，指令被满权限制动覆盖 $a\leftarrow-a_{\max}\hat p$；走廊内 $|p|>0.8\,$`range_fail` 时任何外向推力分量被投影掉，$a\cdot\hat p>0$ 时 $a\leftarrow a-(a\cdot\hat p)\hat p$。

**约束层 2 —— 终端捕获网**：$|p|<2p_{\mathrm{goal}}$ 内若接近速度已超速度容差，则制动使成功球以低于 `vel_goal` 穿越：
\begin{equation}
a\leftarrow+a_{\max}\hat p\quad\text{若}\quad
\lVert p\rVert<2p_{\mathrm{goal}}\ \text{且}\ -(v\cdot\hat p)>v_{\mathrm{goal}}. \tag{5}
\end{equation}

**约束层 3 —— 燃料降额**：控制器按 plant 方式精确积分实测冲量 $\hat f\leftarrow\hat f+\lVert u\rVert_2F_{\max}\Delta t/(I_{sp}g_0)$。$\hat f$ 超过 $0.75\,$`fuel0` 后，逐步推力范数被限，使剩余预算覆盖剩余时域 $N_{\mathrm{left}}$：
\begin{equation}
F_{\mathrm{allow}}=\max\Big(\frac{(\mathrm{fuel0}-\hat f)I_{sp}g_0}{N_{\mathrm{left}}\Delta t}m,\ 0.5F_{\max}\Big),\quad
a\leftarrow a\frac{F_{\mathrm{allow}}}{\lVert a\rVert m}\ \text{若}\ \lVert a\rVert m>F_{\mathrm{allow}}, \tag{6}
\end{equation}
绝不低于半权限。实现中 $m$ 缩放上限在参考包线内松弛到 $0.5F_{\max}$ 下限，故该层是防御性保护；冻结协议记录基线燃料远低于 2 kg 预算（最大 0.912 kg，见 `docs/evaluation_protocol.zh-CN.md` §1）。

**输出映射**
\begin{equation}
u=\operatorname{clip}(am/F_{\max},-1,1)\ \text{（逐轴）}, \tag{7}
\end{equation}
饱和轴计入 `stat_sat`。

## 4. 参数

| 参数 | 值 | 来源 |
|---|---|---|
| $n$ | $\sqrt{\mu/a^3}=1.1314\times10^{-3}$ rad/s | 环境常量 |
| $a_{\max}=F_{\max}/m$ | $10/500=0.02$ m/s² 每轴 | `env.f_max`、`env.mass` |
| $Q$ | $\mathrm{diag}(1,1,1,300,300,300)$ | 式 (2)，`pos_goal=1`、`vel_goal=0.1`、$w_{\mathrm{vel}}=3$ |
| $R$ | $(0.35a_{\max})^{-2}I_3\approx2.04\times10^4I_3$ | `kappa=0.35` |
| 护盾阈值 | 走廊 $0.8r_{\mathrm{fail}}=160$ m；制动 $0.95r_{\mathrm{fail}}=190$ m；$a_b=0.5a_{\max}$ | `__init__` |
| 捕获网 | $\|p\|<2$ m、接近速度 $>0.1$ m/s | `__init__` |
| 燃料降额 | 软阈值 $0.75$`fuel0`$=1.5$ kg；下限 $0.5F_{\max}$ | `__init__` |
| 时域 | 500 步，$\Delta t=1$ s | `env.max_steps` |
| `legacy` 臂 | `q_pos=10^{-4}, q_vel=1, r_acc=1`、`taylor2`、无约束层 | 构造标志（非默认） |

## 5. 实测表现

全部发布记录使用 `scripts/baseline_eval.py` 记账（每副本首回合、标称 plant）。2026-11-12 重测前，默认臂曾因新增的空间站实体外壳跌到 0.4%：新的几何把质心对准原点解释为鼻锥穿入舱体，而 $w_{\mathrm{vel}}=3$ 在 0.02 m/s² 权限下也制动过晚。修正后（鼻锥停在端口平面的终端保持点 + $w_{\mathrm{vel}}=15$）恢复满成功率。

**协议数据集 A**（`results/baseline_protocol/seed{7,13,21,101}/docking/baseline.json`，每 seed 128 副本、500 步时域）：

| seed | 成功 | 成功/完成 | 平均回合回报 | env 步数 |
|---|---:|---:|---:|---:|
| 7 | 100% | 128/128 | 190.16 | 288 |
| 13 | 100% | 128/128 | 190.49 | 288 |
| 21 | 100% | 128/128 | 191.13 | 288 |
| 101（留出） | 100% | 128/128 | 191.95 | 288 |

协议 seeds 7/13/21 合并：**100%（384/384）**，平均回报 190.59，每个文件 `complete=true`。

**已发布单记录**：`results/docking/baseline.json` 为 seed-13 协议记录：128 副本、seed 13、288 env 步、**100%（128/128）**、平均回合回报 **190.49**、`complete=true`。

**数据集 B**（`results/docking/baselines/*_seed7.json`，128 副本、seed 7）：

| 臂 | CLI 键 | 成功 | 平均回报 | env 步数 |
|---|---|---|---:|---:|---:|
| `cwh_lqr`（默认） | — | 100%（128/128） | 190.16 | 288 |
| `cwh_pole_placement` | `pole_placement` | 88.3%（113/128） | 166.22 | 192 |
| `zem_zev` | `zem_zev` | 100%（128/128） | 190.77 | 320 |

三臂仅在终端制导律上不同；均用同一 `SpaceDockingEnv`、1 m / 0.1 m/s 捕获容差与 200 m 距离界，故步数差异（192–320）是捕获时间不同，而非不同任务。`pole_placement` 没有终端保持点，仍把质心驱向原点，因此 11.7% 的回合让鼻锥穿入站体；这是纯极点配置相对默认臂的预期代价，而非调参失败。

冻结控制器审计（`docs/evaluation_protocol.zh-CN.md` §2）记录该控制器在场景网格上完全成功并冻结为行为克隆数据源；也指出终止优先级修复（终端步违反优先于成功）对本基线无实质影响，因其燃料远低于预算。审计的逐场景产物未发布，且发布 JSON 不含燃料或逐场景指标（`metric_counts={}`），故 §1 的 0.912 kg 最大燃料为 2026-10-22 审计记录，而非可从这些文件复现的值。

## 6. 已知局限

- LQR 核心假设无约束；全部状态/燃料约束由式 (4)–(6) 的启发式覆盖层执行，其阈值（0.8/0.95 距离比例、$a_b=a_{\max}/2$、75% 燃料点、半权限下限）是设计选择而非优化值。
- 远场接近为饱和下的开环 bang–brake 段；LQR 只塑形终端段。
- 燃料降额上限（式 6）在参考包线内由 $0.5F_{\max}$ 下限主导，实际是非约束性安全层。
- 增益在构造时对冻结 plant 常量一次算定；无对质量或轨道变化的在线自适应。

## 7. 实现

- 控制器：`DockingCWHLQRBaseline`（注册名 `cwh_lqr`、`BUILDERS['docking']`）位于 `algorithms/baselines/baselines.py`；共享增益 helper `_cwh_lqr`。
- 环境：`environments/satellite/rendezvous_docking/env.py`，类 `SpaceDockingEnv`（`conventional_controller = DockingCWHLQRBaseline`）。
- 评测：`scripts/baseline_eval.py docking`；深入/消融 harness `environments/satellite/rendezvous_docking/eval_docking_lqr.py`。

## 8. 参考

1. Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance System for Satellite Rendezvous", *Journal of the Aerospace Sciences* 27(9), 653–658 —— 相对运动模型。
2. Bryson, A. E. & Ho, Y.-C. (1975). *Applied Optimal Control* —— 按容差的 $Q/R$ 归一化（Bryson 规则）。
3. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement Learning in Spacecraft Proximity Operations", CVPR 2026 Workshop on AI4Space —— 任务定义与积分器。
