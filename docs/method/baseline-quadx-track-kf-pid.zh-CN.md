# QuadX-Track 基线：常速度卡尔曼滤波 + 固定偏移 SE(3) 跟踪（`QuadXTrackKFBaseline`）

**简体中文** · [English](baseline-quadx-track-kf-pid.md)

## 1. 概述

QuadX-Track 要求 Crazyflie 2.x 在 30 s 内把合作地面车辆保持在天底相机的视场内，同时保持固定高度与固定水平偏移。目标仅间歇可见：视场、斜距与遮挡限制会丢失检测，检测带 0.15 m 位置噪声。

传统基线为标准基于模型解法：**线性常速度卡尔曼滤波**（Lee et al. 2010）在丢失期间预测，后接仓库的 **SE(3) 几何跟踪控制器**，参考为**固定高度 + 固定水平偏移**并带速度前馈；公开的固定超时保持规则覆盖长时间中断。无主动搜索、无信息规划、无学习、无安全层。

## 2. 接口约定

- **原始传感信息与学习方法完全相同**。控制器只读 `'policy'` 组：$\omega_b$ `[0:3]`、姿态 $q$ `[3:7]`、$v_b$ `[7:10]`、$p$ `[10:13]`、当前相对测量 $\zeta_t-p$ `[21:24]`、可见标志 $y_t$ `[24]`。$y_t=0$ 时测量为零占位，滤波只预测。目标真值（`env.target_pos`、`env.target_heading`、航段计划）从不读取。
- **构造期模型知识**：控制周期 $\Delta t=1/40$ s、相机噪声 $\sigma=0.15$ m、高度与偏移、cf2x URDF 的 plant 常量为公开设计参数。视场足迹与 LOS 几何只通过可见标志使用，不重新推导遮挡。
- **内部状态**：滤波均值/协方差、可见年龄、冻结保持点、SE(3) 轴向修剪。`reset_idx(mask)` 随环境 reset 掩码清零。
- **输出**：原生动作 $a\in[-1,1]^4$（三轴体角速度指令与总距）。

## 3. 常速度卡尔曼滤波

状态 $x=[p_t^\top,v_t^\top]^\top\in\mathbb{R}^6$（目标位置与速度，世界系）。

**预测**每个策略步执行，含丢失步：
\begin{equation}
x^-=Fx,\quad P^-=FPF^\top+Q,\quad
F=\begin{bmatrix}I_3&\Delta t I_3\\0&I_3\end{bmatrix},\quad
Q=q_a\begin{bmatrix}\tfrac{\Delta t^3}{3}I_3&\tfrac{\Delta t^2}{2}I_3\\[2pt]\tfrac{\Delta t^2}{2}I_3&\Delta t I_3\end{bmatrix},
\end{equation}
$q_a=\sigma_a^2$ 为白加速度功率谱密度（`SIGMA_ACCEL`）。

**量测更新**仅在 $y_t=1$：
\begin{equation}
y=p+\zeta_t,\quad H=[I_3\ 0],\quad
S=HP^-H^\top+R,\quad K=P^-H^\top S^{-1},\quad x=x^-+K(y-Hx^-),\quad P=(I-KH)P^-,
\end{equation}
$R=\sigma^2I_3$（`SIGMA_MEAS` 等于已知传感器 sigma），更新后 $P$ 对称化。$y_t=0$ 时完全跳过更新，状态即预测。

**初始化**首次检测时以 $x=[y^\top,0^\top]^\top$、$P_0=\mathrm{diag}(\sigma^2I_3,\sigma_v^2I_3)$ 播种，$\sigma_v=0.5$ m/s，不使用目标真值。环境保证 reset 时必有一次检测，故滤波在首个控制步初始化。

**固定超时保持**：可见年龄在每个丢失步递增；超过 `HOLD_TIMEOUT=3 s` 时参考冻结在该刻估计，
\begin{equation}
\bar p\leftarrow\big[\hat p_{t,x}+\delta_x,\ \hat p_{t,y}+\delta_y,\ \texttt{ALT\_REF}\big]\quad\text{（超时步），}
\end{equation}
飞行器悬停至新检测到来后恢复滤波。保持期间不执行搜索、覆盖、信息增益或重规划策略。

## 4. 参考与控制

参考
\begin{equation}
\bar p=\big[\hat p_{t,x}+\delta_x,\ \hat p_{t,y}+\delta_y,\ \texttt{ALT\_REF}\big],\quad \delta=(1.5,0)\ \mathrm{m},\quad \texttt{ALT\_REF}=6\ \mathrm{m}.
\end{equation}

复用 QuadX 基线的 SE(3) 级联，仅加一项世界系目标速度前馈：
\begin{equation}
v_d=\hat v_t-\frac{K_X}{K_V}(p-\bar p),\quad
v_d\leftarrow v_d\frac{V_{\max}}{\max(\lVert v_d\rVert,V_{\max})},
\end{equation}
随后为相同的期望力、姿态、力矩与旋翼分配：$F_{\mathrm{des}}=m(g e_3-K_V(v-v_d))$、$b_{3,d}=F_{\mathrm{des}}/\lVert F_{\mathrm{des}}\rVert$、$M=-K_R e_R-K_\Omega\omega_b+\omega_b\times J\omega_b$ 与闭式旋翼映射。偏航调节到零，与静止航路点控制器相同。前馈消除纯位置 PD 追赶 0.4–1.2 m/s 目标时的稳态滞后；增益仍为固定常量。

## 5. 固定参数与独立调参集

| 常量 | 值 | 作用 |
|---|---|---|
| `SIGMA_ACCEL` $\sigma_a$ | 1.5 m/s²（选用） | 白加速度标准差 |
| `SIGMA_MEAS` $\sigma$ | 0.15 m | 量测噪声标准差（已知传感器规格） |
| `VEL_PRIOR_STD` | 0.5 m/s | 首次检测速度先验 |
| `HOLD_TIMEOUT` | 3.0 s | 公开固定超时保持 |
| `V_MAX` | 3.0 m/s | 跟踪限速 |
| `KX` / `KV` | 4.0 / 2.0（选用） | 外环位置/速度增益 |
| `KR` / `KOM` / `KI_POS` / `I_MAX` | 继承 | 来自 `QuadXSE3Baseline` 的 attitude/修剪增益 |
| $\delta$ / `ALT_REF` | (1.5, 0) m / 6 m | 固定水平偏移与高度 |

**选择协议**：`environments/multirotor/eval_quadx_track_kf.py` 在**独立调参种子 101**（与报告评测种子 7 不相交）上评估小网格：
$\sigma_a\in\{0.5,1.5,4.0\}$ m/s²、$(K_X,K_V)\in(4,2),(4,4)$，六种组合各 256 副本、每副本一个完整 30 s 回合。选择分数为 $\text{成功率}+0.5\cdot\text{可见比例}-0.2\cdot\text{跟踪误差}$（按成功率、再按误差打破平局）。选中值写入 `results/quadx_track/kf_tuning.json` 并冻结为报告评测所用类默认值。

**调参结果（seed 101，每组合 256 回合；harness 写 `results/quadx_track/kf_tuning.json`，未随仓库发布，下表为记录的调参输出）**

| $\sigma_a$ | $K_X$ | $K_V$ | 成功 | 可见比例 | 跟踪误差 [m] | 分数 |
|---|---|---|---|---|---|---|
| 0.5 | 4 | 2 | 0.805 | 0.909 | 0.278 | 1.204 |
| 0.5 | 4 | 4 | 0.805 | 0.912 | 0.242 | 1.212 |
| **1.5** | **4** | **2** | **0.820** | **0.916** | **0.256** | **1.227** |
| 1.5 | 4 | 4 | 0.805 | 0.913 | 0.227 | 1.216 |
| 4.0 | 4 | 2 | 0.781 | 0.900 | 0.252 | 1.181 |
| 4.0 | 4 | 4 | 0.766 | 0.891 | 0.222 | 1.167 |

选中行 $\sigma_a=1.5$ m/s²、$K_X=4$、$K_V=2$；六行均在最优 0.02 内，故非尖锐最优点。网格刻意很小：重点是可复现、有记录的选择，而非穷举扫描。运行时无参数自适应，所有报告回合使用同一常量。

## 6. 实测表现

协议数据集 B：`results/quadx_track/baselines/cv_kf_pid_seed7.json`（256 副本、seed 7、每副本一个完整 1200 步 30 s 回合）。控制器只见公开观测——与学习策略相同的 25 维向量、零占位与可见标志。数据集 A（128 副本 × seeds 7/13/21/101）为 89.1 / 85.9 / 82.8 / 82.0%，均值 85.9±3.1%；已发布 `results/quadx_track/baseline.json` 是该集合的 seed-21 成员（82.8%）。下表为 seed-7 数据集 B 记录。

| 指标 | 值 |
|---|---|
| 成功率（可见比例 ≥0.85 且未坠毁） | 0.8516（218/256 首回合） |
| 可见时间比例 | 0.9219 |
| 每回合丢失时间 | 2.343 s |
| 最长连续丢失 | 2.187 s |
| 平均定点跟踪误差（可见步） | 0.2626 m |
| 碰撞率（触地） | 0.000 |
| 出界率 | 0.000 |
| 未成功的时限率 | 0.1484 |
| 每回合飞行距离 | 23.80 m |
| 每回合丢失原因步数（视场 / 斜距 / 遮挡） | 28.1 / 0.0 / 65.6 |
| 平均回合回报 | 85.68 |

解读：基线在 30 s 回合中 92% 时间保持目标可见，并以 0.26 m RMS 保持 1.5 m 定点。全部丢失均为几何原因——目标离开足迹（视场，28 步）或经过静态遮挡物后（66 步）；斜距窗口从不失败，因固定 6 m 高度使斜距接近 6 m。最长丢失（2.19 s）低于 3 s 保持超时，故冻结保持规则在报告运行中从未触发；错过 0.85 成功线的 15% 回合只是可见时间略多于 4.5 s 的损失，而非坠毁。

## 7. 已知局限

- 滤波假设常速度目标；转弯与启停仅由过程噪声 $\sigma_a$ 处理，故剧烈机动时估计滞后且标称协方差偏乐观。
- 长时间丢失时 3 s 后参考冻结；若目标在被遮挡或超距时离开足迹，基线不搜索、不能重新捕获，回合记录丢失时间。
- 参考偏移与高度固定；控制器不用高度换足迹覆盖。
- SE(3) 内环为与航路点任务相同的静止设定律，仅加速度前馈；无轨迹优化或遮挡感知规划。
- 控制器只读公开观测，无法知道目标真实模式或计划。

## 8. 实现

- 控制器：`QuadXTrackKFBaseline`（`name='cv_kf_pid'`）位于 `algorithms/baselines/baselines.py`，注册于 `BUILDERS['quadx_track']`，由 `QuadXTrackEnv.conventional_controller` 暴露。
- 评测：`scripts/baseline_eval.py quadx_track`（产物 `results/quadx_track/baseline.json`，seed 21 / 128 副本）；数据集 B 记录 `results/quadx_track/baselines/cv_kf_pid_seed7.json`（seed 7 / 256 副本）。
- 调参：`environments/multirotor/eval_quadx_track_kf.py`（写 `results/quadx_track/kf_tuning.json`，未发布）。
- 环境：`docs/method/env-quadx-track.zh-CN.md`（相机模型、目标计划、奖励、指标）。

## 9. 参考

- Lee, T., Leok, M., McClamroch, N. H., "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *IEEE CDC*, 2010 —— 复用的 SE(3) 律。
- Welch, G., Bishop, G., "An Introduction to the Kalman Filter," UNC-Chapel Hill TR 95-041 —— 标准线性常速度滤波。
- Tai, J. K. et al., *PyFlyt*, v0.29.0 —— cf2x plant 与内环 rate PID。
