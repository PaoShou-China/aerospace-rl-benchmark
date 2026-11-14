# 低推力转移基线：一步贪心 Gauss 切向制导（`TransferGaussBaseline`）

**简体中文** · [English](baseline-transfer-gauss.md)

## 1. 概述

低推力近地点抬升任务的传统基线是基于 Gauss 变分方程的一步贪心制导律。每个控制步，由观测笛卡尔状态恢复密切轨道根数；在饱和**切向**推力下解析构造近地点与远地点半径的一阶逐步变化；控制器在三个候选中选择——滑行、顺行切向点火、逆行切向点火——最大化任务参考终端目标的一步增益（扣除推进剂成本）。远地点附近顺行切向点火在（一阶）固定 $r_a$ 下抬高 $r_p$——经典近地点抬升点火——而近地点附近同样推力主要抬升 $r_a$，故贪心规则自然把推力集中在远地点附近，偶尔**逆行**点火修正远地点过冲（有记录的预期行为）。

## 2. 接口约定

- **观测**：仅 `'policy'` 组：原始未归一化 8 维状态 $o=[t,r,v,m]$（经过时间、GCRF 位置、速度、质量）。控制器消费 `o[:,1:4]`、`o[:,4:7]`、`o[:,7:8]`。
- **构造期环境常量**：$\mu$、控制周期 $\Delta t=300$ s、逐轴推力限 $F_{\max}=10$ mN、$I_{sp}=4000$ s、$g_0=9.80665$ m/s²、出生轨道参考拱点半径 $r_{p0}=a_0(1-e_0)$、$r_{a0}=a_0(1+e_0)$（$a_0=10^7$ m、$e_0=0.15$；reset 只重采样平近点角，故参考半径恒定）。
- **逐回合状态**：无；`reset_idx` 为基类空操作。律为纯状态反馈。
- **动作**：$a\in[-1,1]^3$，环境映射为逐轴推力 $F=F_{\max}a$，质量流 $\dot m=-F_{\max}\lVert a\rVert_1/(I_{sp}g_0)$。控制器输出 $a=s\hat t$，$s\in\{-1,0,+1\}$、$\hat t$ 为式 (2) 的切向单位向量，故只决定滑行/点火极性；方向由密切几何固定。

## 3. 控制律

**由观测状态得密切根数**：$r=\lVert r\rVert$，
\begin{equation}
h=r\times v,\quad
a=\Big(\tfrac{2}{r}-\tfrac{\lVert v\rVert^2}{\mu}\Big)^{-1},\quad
\mathbf e=\frac{v\times h}{\mu}-\frac{r}{r},\quad
e=\lVert\mathbf e\rVert,\quad
p=\frac{\lVert h\rVert^2}{\mu},\quad
v_r=\frac{v\cdot r}{r}. \tag{1}
\end{equation}
真近点角由 $O(e)$ 对 $(e\sin\nu,e\cos\nu)=(hv_r/\mu,p/r-1)$ 恢复：
\begin{equation}
\nu=\operatorname{atan2}\Big(\frac{hv_r}{\mu},\frac{p}{r}-1\Big), \tag{2}
\end{equation}
无除法：$e\to0$ 时该对同时消失，$\sin\nu,\cos\nu$ 保持有界且互相一致（近圆数值稳定）。切向点火方向 $\hat t=\hat h\times\hat r$。

**一步根数变化（Gauss 速率）**：饱和切向点火下，$S_\Delta=(F_{\max}/m)\Delta t$ 为每步单位质量速度冲量，Gauss 变分方程给出一阶逐步变化
\begin{align}
\Delta a&=\frac{2a^2}{h}\frac{p}{r}S_\Delta, \tag{3a}\\
\Delta e&=\frac{(p+r)\cos\nu+re}{h}S_\Delta, \tag{3b}\\
\Delta r_p&=(1-e)\Delta a-a\Delta e,\qquad \Delta r_a=(1+e)\Delta a+a\Delta e, \tag{3c}\\
\Delta m&=\frac{F_{\max}\lVert\hat t\rVert_1}{I_{sp}g_0}\Delta t, \tag{3d}
\end{align}
$\Delta m$ 为三对正交推力器消耗的推进剂（环境的 $\lVert a\rVert_1$ 质量流约定作用于单位向量 $\hat t$），在观测质量上求值。

**贪心决策**：任务参考终端目标
\begin{equation}
J=(r_p-r_{p0})-|r_a-r_{a0}|-2\times10^5(m_0-m), \tag{4}
\end{equation}
（环境终端奖励 $R=10^{-4}(-|\Delta r_a|+\Delta r_p+2\times10^5\Delta m)$ 的未缩放孪生，论文 Table I 权重）。令 $d_0=|r_a-r_{a0}|$，两极性的一步增益（精确含 $|r_a-r_{a0}|$ 折点）为
\begin{align}
G_+&=\Delta r_p-\big(|r_a+\Delta r_a-r_{a0}|-d_0\big)-2\times10^5\Delta m, \tag{5a}\\
G_-&=-\Delta r_p-\big(|r_a-\Delta r_a-r_{a0}|-d_0\big)-2\times10^5\Delta m, \tag{5b}
\end{align}
决策为 $\{0,G_+,G_-\}$（滑行、顺行、逆行）上的 argmax：
\begin{equation}
a=s\hat t,\qquad
s=\begin{cases}+1 & G_+>\max(0,G_-)\\ -1 & G_->\max(0,G_+)\\ 0 & \text{否则（滑行）}\end{cases}. \tag{6}
\end{equation}
故仅当预测目标增益超过推进剂成本 $2\times10^5\Delta m$ 才点火；$r_a>r_{a0}$ 时折点项可使逆行候选胜出，产生有记录的远地点修正点火。可选构造参数 `target_rp` [m] 在 $a(1-e)\ge$ `target_rp` 后强制滑行；注册配置保持 `None`。

## 4. 参数

| 参数 | 值 | 来源 |
|---|---|---|
| $\mu$ | $3.986004418\times10^{14}$ m³/s² | `env.mu`（WGS84） |
| $F_{\max}$ | 0.01 N 每推力器对 | `env.f_max` |
| $I_{sp}$、$g_0$ | 4000 s、9.80665 m/s² | `env.isp`、`env.g0` |
| $m_0$ | 100 kg | `env.mass0` |
| 参考轨道 | $a_0=10^7$ m、$e_0=0.15$ ⇒ $r_{p0}=8.5\times10^6$ m、$r_{a0}=1.15\times10^7$ m | `env.sma0`、`env.ecc0` |
| 推进剂权重 | $2\times10^5$ m/kg（目标 $J$ 权重） | `prop_cost` |
| 控制步 / 时域 | $\Delta t=300$ s、166 步（约 5 圈） | `env.dt`、`env.max_steps` |
| `target_rp` | `None`（禁用） | 构造默认 |

## 5. 实测表现

全部发布记录使用 `scripts/baseline_eval.py` 记账（每副本首回合、标称 plant、166 步时域）。检出中无 1024 副本转移产物，故更早的"1024 env / seed 7 / 终端 R ≈0.86"声明撤回；2026-11-12 重测的可验证数字如下：

**协议数据集 A**（`results/baseline_protocol/seed{7,13,21,101}/transfer/baseline.json`，每 seed 128 副本）：

| seed | 成功 | 成功/完成 | 平均回合回报 | env 步数 |
|---|---:|---:|---:|---:|
| 7 | 100% | 128/128 | 0.7680 | 192 |
| 13 | 100% | 128/128 | 0.7686 | 192 |
| 21 | 100% | 128/128 | 0.7690 | 192 |
| 101（留出） | 100% | 128/128 | 0.7681 | 192 |

协议 seeds 7/13/21：均值 **100.0% ± 0.0**（384/384），`complete=true`。任务级成功判据为最终轨道上 `R > 0`（环境 `terminal_reward` 即伸缩回合回报）；2026-11-12 的调优（内部目标近地点抬高 300 km，见 §6）同时把此前勉强为正的回报提升到 0.77。

**已发布单记录**：`results/transfer/baseline.json` 为 seed-13 协议记录：128 副本、seed 13、192 步、**100%（128/128）**、平均回合回报 **0.7686**、`complete=true`。

**数据集 B**（`results/transfer/baselines/gauss_tangential_seed7.json`，128 副本、seed 7）：**100%（128/128）**，平均回合回报 **0.7680**。

**逐回合物理记账**（数据集 B `task_metrics`，128 回合均值）：近地点抬升 **10.69 km**、远地点漂移 **0.178 km**、推进剂 **1.42×10⁻² kg**。内部 8.8×10⁶ m 的近地点目标使贪心律果断越过 `R>0` 门槛，最终 `R` 远离零，而非调优前 0.007 的勉强裕量。

冻结控制器审计（`docs/evaluation_protocol.zh-CN.md` §2）将该律列为完全成功并冻结为该任务行为克隆数据源；审计的逐场景产物未发布。

## 6. 已知局限

- 规则对 $J$ 为一步贪心：不是全局/多圈优化器，除式 (3) 的线性化单步预测外无未来点火窗口模型。
- 线性化假设逐步根数变化小；10 mN 参考推力下处处满足，但律无步内积分误差修正环。
- 逆行分支只通过 $|r_a-r_{a0}|$ 折点修正远地点过冲；律中无硬远地点约束。
- 近圆退化解析处理（式 2）但未消除：$e\to0$ 时拱线方向在 float32 观测噪声地板以下不可分辨，门控退化为有界、锚定目标的抖动而非发散。任务偏心率跨 $[0.02,0.1]$，高于该地板五个数量级（实现 docstring）。

## 7. 实现

- 控制器：`TransferGaussBaseline`（注册名 `gauss_tangential`、`BUILDERS['transfer']`）位于 `algorithms/baselines/baselines.py`。
- 环境：`environments/satellite/orbit_transfer/env.py`，类 `LowThrustTransferEnv`（`conventional_controller = TransferGaussBaseline`）。
- 评测：`scripts/baseline_eval.py transfer`。

## 8. 参考

1. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —— 密切根数恢复与 Gauss 形式变分方程。
2. Battin, R. H. *An Introduction to the Mathematics and Methods of Astrodynamics*（AIAA）—— Gauss 行星方程。
3. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital Transfer using Dynamics-Agnostic Reinforcement Learning"（arXiv:2211.08272, IEEE）及配套 `PerigeeRaisingEnv` —— 任务定义与 Table I 目标权重。
