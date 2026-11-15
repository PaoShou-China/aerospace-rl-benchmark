# 低推力转移基线：Petropoulos Q-law 反馈制导（`TransferQLawBaseline`）

**简体中文** · [English](baseline-transfer-qlaw.md)

## 1. 概述

低推力近地点抬升任务的第二个经典基线是 Petropoulos Q-law（AAS 04-278 / AAS 05-130），低推力多圈轨道转移中被引最多的闭环反馈制导。由密切轨道根数相对参考的加权、时间尺度归一化平方误差构造类 Lyapunov 邻近商 $Q$；Gauss 变分方程被用两次：一次以每个根数误差的最大可达变化率 $\dot z_{xx}$ 归一化（对推力方向与真近点角取双最大），一次选择使 $Q$ 下降最快的推力方向 $\hat u^*=-\Phi^\top Q_z/\lVert\Phi^\top Q_z\rVert$。**有效度**检验在当前 $Q$ 下降率相对当前轨道最佳近点角可交付值很差时滑行。

**任务适配（如实声明，不声称等价于自由增益形式）**：任务为"抬升近地点、约束远地点漂移"，无唯一目标轨道，故 Q-law 参考被显式配置：近地点参考 $r_{p,\mathrm{ref}}=r_{p0}+$`RP_RAISE` 为公开固定参数，远地点参考保持出生远地点 $r_{a,\mathrm{ref}}=r_{a0}$，使 $e$ 误差项惩罚远地点漂移。参考对映射到目标根数对 $(a_T,e_T)$（式 4）。因任务无需改变轨道面，律实现于平面 $(a,e)$ 子集，$i,\Omega,\omega$ 自由；原始形式的近地点罚函数 $P$ 不启用（回合内 $r_p\gg$ 地球半径，且此处抬升 $r_p$ 即为目标）。这是面向任务的 Q-law 配置，不声称与无目标增益形式等价。

## 2. 接口约定

与 Gauss 基线相同（同环境、同缓冲）：

- **观测**：原始 8 维 $o=[t,r,v,m]$；控制器消费 `o[:,1:4]`、`o[:,4:7]`、`o[:,7:8]`。
- **环境常量**：$\mu$、$F_{\max}=10$ mN 每轴对、$I_{sp}=4000$ s、$g_0$、出生半径 $r_{p0}=8.5\times10^6$ m、$r_{a0}=1.15\times10^7$ m（取自 `env.mu/f_max/isp/g0/sma0/ecc0`）。
- **逐回合状态**：无；纯状态反馈。
- **动作**：$a\in[-1,1]^3$；环境施加逐轴推力 $F=F_{\max}a$ 与质量流 $\dot m=-F_{\max}\lVert a\rVert_1/(I_{sp}g_0)$。控制器输出单位范数平面内点火方向或恰为 $0$（滑行），故逐轴界与 L1 燃料模型按构造满足。

## 3. 控制律

**密切根数**：逐字使用 Gauss 基线写法的式 (1)，包括无除法真近点角恢复 $\nu=\operatorname{atan2}(hv_r/\mu,p/r-1)$（近圆安全）。

**邻近商（Petropoulos AAS 04-278 $Q$ 式；AAS 05-130 改进）**：平面子集 $Z=[a,e]$，
\begin{equation}
Q=\sum_{z\in\{a,e\}}W_zS_z(Z)\left(\frac{d(z,z_T)}{\dot z_{xx}}\right)^2,\qquad
d(a,a_T)=a-a_T,\ d(e,e_T)=e-e_T, \tag{1}
\end{equation}
$W_p=0$（本任务不启用近地点罚 $P$），半长轴缩放函数
\begin{equation}
S_a=\left(1+\left(\frac{a-a_T}{ma_T}\right)^n\right)^{1/r},\qquad S_e=1,\qquad (m,n,r)=(3,4,2) \tag{2}
\end{equation}
取 Petropoulos 名义值。$\dot z_{xx}$ 为根数 $z$ 在当前密切轨道上对**推力方向与真近点角两者**的最大变化率；此处用 $N_\theta=2048$ 点真近点角网格扫描 Gauss 速率求值。对 $a$ 的解析参考
\begin{equation}
\dot a_{xx}=\frac{2a^2}{h}(1+e) \tag{3}
\end{equation}
（近地点顺行切向点火）精确，仅作网格扫描的 harness 校验；$e$ 无此类闭式，$\dot e_{xx}$ 取同一网格。

**参考映射**：公开近地点参考与冻结远地点参考，
\begin{equation}
a_T=\frac{r_{a,\mathrm{ref}}+r_{p,\mathrm{ref}}}{2},\quad
e_T=\frac{r_{a,\mathrm{ref}}-r_{p,\mathrm{ref}}}{r_{a,\mathrm{ref}}+r_{p,\mathrm{ref}}},\quad
r_{p,\mathrm{ref}}=r_{p0}+\texttt{RP\_RAISE},\ r_{a,\mathrm{ref}}=r_{a0}. \tag{4}
\end{equation}

**推力方向（AAS 04-278）**：$Q_z=\nabla_ZQ$（精确偏导，含 $\dot z_{xx}$ 对 $(a,e)$ 的依赖，经网格极大值的自动微分获得——由包络定理几乎处处有效并经 FD 验证），$\Phi(Z)$ 为 Gauss 矩阵在径向/切向系中的 $(a,e)$ 行，
\begin{equation}
\Phi=\begin{bmatrix}\frac{2a^2}{h}e\sin\nu&\frac{2a^2}{h}\frac{p}{r}\\[2pt]\frac{p\sin\nu}{h}&\frac{(p+r)\cos\nu+re}{h}\end{bmatrix},\qquad
\hat u^*=-\frac{\Phi^\top Q_z}{\lVert\Phi^\top Q_z\rVert}, \tag{5}
\end{equation}
使 $\dot Q=Q_z^\top\Phi\hat u^*$ 在当前近点角下对推力方向最小。惯性动作 $a=u_r\hat r+u_t\hat t$，$\hat t=\hat h\times\hat r$；面外分量恰为零（float64 下验证至 $3\times10^{-8}$，harness A3）。

**有效度滑行（AAS 04-278 机制）**：当前近点角可达 $Q$ 下降率与当前轨道任意近点角最佳值之比
\begin{equation}
\eta=\frac{\lVert\Phi(\nu)^\top Q_z\rVert}{\max_\theta\lVert\Phi(\theta)^\top Q_z\rVert},\qquad
a=\begin{cases}\hat u^* & \eta\ge\eta_{\min}\\ 0 & \eta<\eta_{\min}\text{（滑行）}\end{cases}, \tag{6}
\end{equation}
公共推力加速度因子 $F_{\max}/m$ 在比值中相消，故 $m$ 不进门控。$\eta_{\min}$ 为固定常规阈值——无学习、无调度。

## 4. 参数

| 参数 | 值 | 来源 |
|---|---|---|
| $\mu$ | $3.986004418\times10^{14}$ m³/s² | `env.mu`（WGS84） |
| $F_{\max}$ | 0.01 N 每推力器对 | `env.f_max` |
| $I_{sp}$、$g_0$ | 4000 s、9.80665 m/s² | `env.isp`、`env.g0` |
| $m_0$ | 100 kg | `env.mass0` |
| 参考轨道 | $r_{p0}=8.5\times10^6$ m、$r_{a0}=1.15\times10^7$ m | `env.sma0`、`env.ecc0` |
| $r_{p,\mathrm{ref}}$ | $r_{p0}+$ `RP_RAISE`（调参后冻结） | 式 (4) |
| $r_{a,\mathrm{ref}}$ | $r_{a0}$（冻结） | 式 (4) |
| $W_a$、$W_e$ | 1、`W_E`（调参后冻结） | 式 (1) |
| $S_a$ 的 $(m,n,r)$ | (3,4,2) | Petropoulos 名义 |
| $\eta_{\min}$ | `ETA_MIN`（调参后冻结） | 式 (6) |
| $N_\theta$（$\dot z_{xx}$ 网格） | 2048 | 实现 |
| 控制步 / 时域 | $\Delta t=300$ s、166 步（约 5 圈） | `env.dt`、`env.max_steps` |

调参值由 harness 的 `phase_b()` 在独立调参 seeds 13/21 上选择（规则：最大平均终端 R，平局按 $|\Delta r_a|$，再按推进剂），官方 seed-7 评测前冻结为类常量；选中值见 §6。

## 5. 实现与注册

- 控制器：`TransferQLawBaseline`（`name='q_law_petropoulos'`），注册项 `BASELINE_VARIANTS['transfer']['q_law']`；经 `build_baseline('transfer', env, variant='q_law')` 构建；位于 `algorithms/baselines/baselines.py`。
- 评测：`scripts/baseline_eval.py transfer --baseline q_law`（每副本首回合；已发布产物 `results/transfer/baselines/q_law_petropoulos_seed7.json`，256 副本、seed 7、166 步，含完整控制器 `config()`）。`--baseline` 键（`q_law`）与类 `name`（`q_law_petropoulos`）不同；`baseline_eval.py` 以类 `name` 命名输出，即默认输出目录下 `baseline_q_law_petropoulos.json`。**该 1024 副本 harness 文件未发布**；检出中只有 256 副本数据集 B JSON。
- 验证/调参/比较 harness：`environments/satellite/orbit_transfer/eval_transfer_qlaw.py`（自包含、无 CLI；`OFFICIAL_ENVS=1024`、`TUNE_ENVS=512`）：
  - `phase_a()` —— 小推力正确性（7 项检查，除注明外均达或接近机器精度）：A1 输出惯性动作的根数速率 $\dot a,\dot e$ 与精确雅可比收缩 $(F_{\max}/m)\nabla_v(a,e)\cdot u$（代数恒等，重力对密切根数无瞬时影响）；A2 $\nabla Q$ 与 $Q(a,e)$ 中心 FD；A3 框架不变量（面外约 0、单位范数、逐轴界）；A4 远地点点火为纯顺行切向（$u_t>0.99$、$u_r\approx0$），1% 油门单步探针在 100% 副本上复现 $\Delta r_p,\Delta r_a$ 符号（中位幅值误差约 5%，一阶预测 vs 300 s 非线性步）；A5 $Q$ 沿点火从不增加；A6 网格 $\dot a_{xx}$ vs 解析式 (3)。
  - `phase_b()` —— 在 seeds 13/21 上对（`RP_RAISE`、$W_e$、$\eta_{\min}$）做调参网格（每格 512 副本、27 格）。
  - `phase_c()` —— 冻结律的官方协议评测（默认 1024 副本）。
  - `phase_d()` —— Gauss / Q-law / PPO 三路比较，含逐臂成功率、终端 R、近地点增益、远地点漂移、推进剂、$\Delta v$ 与点火占空比。

  harness 写 `results/transfer/qlaw_report.json` 与 `results/transfer/comparison.json`；**两文件均未发布**，故下文 phase-B/phase-D 数字为运行记录而非可复验产物。

## 6. 实测表现

**小推力验证（`phase_a()`，7 项全通过）**。A1 根数速率对精确雅可比收缩一致度 $4.5\times10^{-16}$（$\dot a$）/ $2.0\times10^{-15}$（$\dot e$）；A2 $\nabla Q$ 偏导 vs 中心 FD：$1.6\times10^{-11}$ / $3.3\times10^{-11}$；A3 最大面外分量 $3.4\times10^{-8}$、最大单位范数偏差 $3.7\times10^{-8}$；A4 远地点点火恰为顺行切向（$u_t=1$、$u_r=-0$），近地点恰为逆行切向，1% 油门单步探针对 $\Delta r_p$ 与 $\Delta r_a$ 均 100% 符号一致（中位幅值误差 5.3% / 6.0%，一阶预测 vs 300 s 非线性步）；A5 沿点火零 $Q$ 增加步；A6 网格 $\dot a_{xx}$ vs 式 (3)：$3.7\times10^{-16}$。

**调参（`phase_b()`，seeds 13/21、512 env、27 点网格；harness 记录，未发布）**。选择规则：最大平均终端 R，平局按 $|\Delta r_a|$，再按推进剂。选中 $(\texttt{RP\_RAISE},W_e,\eta_{\min})=(20\ \mathrm{km},1.0,0.1)$，$R=1.1510$、100% 成功、$\Delta r_p=15.52$ km、$|\Delta r_a|=0.17$ km、$0.0192$ kg。网格行为良好：`RP_RAISE`$=10$ km 把 $\Delta r_p$ 卡在 10 km 目标（R ≈0.62）；$40$ km 时 $\Delta r_p$ 不变（5 圈时域是约束）但大 $W_e$ 让 $|\Delta r_a|$ 漂到 8–11 km（R ≤0.12）；$\eta_{\min}=0.3$ 使点火饥饿（占空比 0.74，$\Delta r_p$ 未达目标）。调参值与已发布类常量一致——无需再改。

**已发布数据集 B（seed 7、256 副本、每副本一回合）**。`results/transfer/baselines/gauss_tangential_seed7.json` 与 `q_law_petropoulos_seed7.json`，均 `complete=true`：

| 臂 | 成功 | 平均终端 R | $\Delta r_p$ | $|\Delta r_a|$ | 推进剂 |
|---|---|---:|---:|---:|---:|
| Gauss 贪心 | 100%（256/256） | 0.0074 | 0.134 km | 0.0375 km | $1.17\times10^{-4}$ kg |
| **Q-law（本臂）** | **100%（256/256）** | **1.1523** | **15.524 km** | **0.161 km** | 0.0192 kg |

两行都满足任务级成功判据 `R>0`；平均回合回报等于终端 `R`，因环境奖励正是该终端目标的精确势函数伸缩。相对调参 seed 记录（R 1.1510），seed-7 的 1.1523 显示无调参集过拟合。两臂物理上并不接近：Q-law 抬升近地点约为 Gauss 贪心律的 116 倍，推进剂约 164 倍，远地点漂移约 4.3 倍。有效度门使 Q-law 占空比近 1（harness 记录 0.985），因该目标下近远地点切向推力几乎总是有效；而 Gauss 贪心律显式计价推进剂，在点火一步增益未超推进剂价格时滑行——10 mN 参考推力下几乎总是如此。早期草稿引用的 harness 侧 $\Delta v$（4.91 m/s）与已发布推进剂质量不一致；0.0192 kg 在 $I_{sp}=4000$ s 下的火箭方程值约 7.5 m/s，但两者均非发布产物，此处均不作基准数字。

**与学习臂（PPO）比较**：`results/transfer/comparison.json` 与 `qlaw_report.json` **未发布**，检出中无 transfer 的 PPO checkpoint。32.8 M 转场预算（1024 env × 1000 updates × 32 steps/env）是 `docs/evaluation_protocol.zh-CN.md` 的等交互目标；不是已完成运行，此处不声称任何比较数字。

## 7. 已知局限

- Q-law 是反馈律而非轨迹优化器：五圈时域足以收敛到 $(a_T,e_T)$，但除有效度门外不尝试点火弧段调度。
- $(a,e)$ 平面子集无法表达面外目标；任务不需要。
- 远地点参考冻结于 $r_{a0}$ 时，$e$ 误差项对远地点漂移双向对称惩罚；它是漂移界，而非抬升远地点能力。
- 网格极大 $\dot z_{xx}$ 及经极大值的自动微分偏导几乎处处精确（包络定理），并在 harness 点 FD 验证；在网格极大器跳变的（测度零）近点角处，偏导为胜出分支的单侧极限。

## 8. 参考

1. Petropoulos, A. E. "Low-Thrust Orbit Transfers Using Candidate Lyapunov Functions with a Mechanism for Coasting," AAS/AIAA Spaceflight Mechanics Meeting, AAS 04-278, 2004 —— $Q$、$\dot z_{xx}$、$\hat u^*=-\Phi^\top Q_z/\lVert\Phi^\top Q_z\rVert$、有效度滑行、$S_a$ 的 $(m,n,r)=(3,4,2)$。
2. Petropoulos, A. E. "Refinements to the Q-law for Low-Thrust Orbit Transfers," AAS 05-130, 2005 —— 罚/缩放改进。
3. Kumar, S. et al. "Closed-Loop Control Law for Low Thrust Orbit Transfer with Guaranteed Stability"（arXiv:2511.23014）—— $Q$ 结构、Gauss 矩阵 $\Phi$ 与 $\dot z_{xx}$ 双最大的现代表述，用于交叉核对实现。
4. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —— 密切根数恢复；Battin, R. H. *An Introduction to the Mathematics and Methods of Astrodynamics*（AIAA）—— Gauss 行星方程。
5. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital Transfer using Dynamics-Agnostic Reinforcement Learning"（arXiv:2211.08272）—— 任务定义。
