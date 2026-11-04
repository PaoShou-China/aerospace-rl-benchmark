# QuadX-Waypoints —— 位置—姿态级联 PID 基线

**简体中文** · [English](baseline-quadx-cascade-pid.md)

## 1. 概述

本文给出 QuadX-Waypoints 的第二个传统臂，与 SE(3) 几何基线并列，不修改环境、奖励、成功判据、捕获顺序或任一学习臂。它是速率指令四旋翼的教科书级联：**外环位置 PID** 把活动航路点误差转为期望加速度，重力补偿与固定偏航参考把加速度转为**期望滚转、俯仰与总距**，**姿态 P/PD** 把 Euler 姿态误差转为**体角速度指令**，由环境既有 120 Hz rate PID 执行。动作是环境四维速率+总距缩放的精确逆。

环路层级：

```
位置 PID (40 Hz) -> a_des -> 滚转/俯仰/推力 -> 姿态 P/PD (40 Hz)
    -> 体角速度指令 -> 环境 rate PID (120 Hz) -> 电机 mixer (240 Hz)
```

控制器与 SE(3) 基线使用相同观测、相同冻结 cf2x plant 常量与相同活动航路点（`env.current_target`）。无学习、无预测、无航路点前瞻、无轨迹前馈、无指令安全滤波；唯一记忆是每副本一个钳位位置误差积分。输出与积分饱和为普通 PID 的 `clip`。

## 2. 接口约定

- **观测**：只读 `'policy'` 组：$\omega_b$（`[0:3]`）、四元数 $(x,y,z,w)$（`[3:7]`）、$v_b$（`[7:10]`）、$p$（`[10:13]`）。活动航路点 $p^\star$ 取自 `env.current_target`，与 SE(3) 臂相同；观测亦在 slot-0 体系偏差中编码该量，故无特权 plant 状态。
- **构造期模型知识**：$\Delta t=1/\texttt{agent\_hz}=1/40$ s 及参考 URDF/YAML 的冻结 Crazyflie 2.x 常量：质量、pwm=1 总推力、环境总距上限、实测电机滞后/噪声直流增益。
- **内部状态**：每副本世界系位置误差积分 $I\in\mathbb{R}^3$ 与上次航路点索引（用于交接时清 $I$）。`reset_idx(mask)` 随环境 reset 掩码清零，无其它记忆。
- **输出**：$a\in[-1,1]^4$：三轴体角速度指令（$\pm\pi$ rad/s）与总距（$[0,0.8]$）。float64 计算、跨副本批量，返回 float32。

## 3. 控制律

### 3.1 外环位置 PID

令 $e_p=p-p^\star$、世界速度 $v_w=R v_b$。位置误差在世界系积分
\begin{equation}
I_{k+1}=\mathrm{clip}\big(I_k+K_I e_p\Delta t,\,-I_{\max},\,I_{\max}\big), \tag{1}
\end{equation}
但当上一未饱和指令已裁剪 $\lVert a_{\mathrm{raw},k}\rVert>A_{\max}$ 时冻结更新（条件抗饱和）。微分取实测速度而非设定，故航路点交接不产生微分冲击：
\begin{equation}
a_{\mathrm{raw}}=-K_P e_p-K_D v_w+I,\qquad
a_{\mathrm{des}}=a_{\mathrm{raw}}\frac{A_{\max}}{\max(\lVert a_{\mathrm{raw}}\rVert,A_{\max})}. \tag{2}
\end{equation}
外环是位置 PID，输出为期望加速度（$K_P$ s⁻²、$K_I$ s⁻³、$K_D$ s⁻¹）。环境推进航路点索引时 $I$ 清零，该推进走环境正常捕获逻辑——控制器不改顺序、不加捕获测试。

### 3.2 重力补偿、期望姿态与总距

比力指令加重力
\begin{equation}
a_{\mathrm{cmd}}=a_{\mathrm{des}}+g e_3, \tag{3}
\end{equation}
在固定偏航参考 $\psi_r$ 的航向系中分解
\begin{equation}
u=\begin{pmatrix}
a_{\mathrm{cmd},x}\cos\psi_r+a_{\mathrm{cmd},y}\sin\psi_r\\
-a_{\mathrm{cmd},x}\sin\psi_r+a_{\mathrm{cmd},y}\cos\psi_r\\
a_{\mathrm{cmd},z}
\end{pmatrix}. \tag{4}
\end{equation}
对 plant 的 ZYX Euler 约定，推力轴在航向系为 $b_3=(\sin\theta\cos\phi,-\sin\phi,\cos\theta\cos\phi)$，故
\begin{equation}
\theta_d=\operatorname{atan2}(u_x,u_z),\qquad
\phi_d=\operatorname{atan2}(-u_y,\sqrt{u_x^2+u_z^2}),\qquad
f=m\lVert a_{\mathrm{cmd}}\rVert. \tag{5}
\end{equation}
符号随 plant：正 $x$ 加速度指令正俯仰（$b_{3,x}>0$），正 $y$ 加速度指令负滚转（$b_{3,y}>0$），正 $z$ 加速度抬高总距。偏航参考固定 $\psi_r=0$，姿态环驱动实际偏航到该值。因 $a_{\mathrm{cmd},z}\ge g-A_{\max}>0$，两个期望角都在 $(-\pi/2,\pi/2)$ 内，无需额外姿态参考钳位。

### 3.3 姿态 P/PD 到体角速度指令

令 $(\phi,\theta,\psi)$ 为 $R$ 的 ZYX Euler 角，$\mathrm{wrap}(\cdot)$ 映射到 $(-\pi,\pi]$：
\begin{equation}
e_{\mathrm{att}}=\mathrm{wrap}\begin{pmatrix}\phi_d-\phi\\ \theta_d-\theta\\ \psi_r-\psi\end{pmatrix},\qquad
\omega^{\mathrm{cmd}}=K_{\mathrm{att}}\odot e_{\mathrm{att}}-K_d\odot\omega_b,\qquad
\omega^{\mathrm{cmd}}\leftarrow\mathrm{clip}(\omega^{\mathrm{cmd}},-\pi,\pi), \tag{6}
\end{equation}
$K_{\mathrm{att}}$（s⁻¹）、$K_d$（无量纲）。速率阻尼作用于实测体角速度，是姿态 PD 的 "D"；内环仍是环境 PID，级联不复制也不替换它。内环远快于外环（比例路径速率时间常数约 $10^{-2}$ s），姿态环表现为时间常数 $1/K_{\mathrm{att}}$ 的一阶系统加小 $K_d$ 阻尼。

### 3.4 动作逆映射

plant 在电机状态 $\theta_i$ 处给出单旋翼推力 $f_i=\theta_i^2 F_{\max}^{\mathrm{tot}}/4$；总距指令 $\pi\in[0,0.8]$ 经一阶电机滞后（$\tau_m=0.01$ s）与均值 $\mathbb E[1+0.02\mathcal N(4,1)]=1.08$ 的乘性噪声。指令到电机状态的均值稳态增益为实测直流增益 $G_{\mathrm{pwm}}=1.216$，故给出 $f$ 的总距
\begin{equation}
T=\mathrm{clip}\Big(\frac{1}{G_{\mathrm{pwm}}}\sqrt{\frac{f}{F_{\max}^{\mathrm{tot}}}},\,0,\,T_{\max}\Big),\qquad T_{\max}=0.8. \tag{7}
\end{equation}
环境动作缩放是仿射的 $\bar a=\ell+\tfrac{a+1}{2}(h-\ell)$，$\ell=(-\pi,-\pi,-\pi,0)$、$h=(\pi,\pi,\pi,0.8)$；逆映射
\begin{equation}
a=\Big(\frac{\omega^{\mathrm{cmd}}}{\pi},\ \frac{2T}{T_{\max}}-1\Big)\ \mathrm{clipped\ to}\ [-1,1]^4. \tag{8}
\end{equation}
式 (7)–(8) 在一处固定单位（rad/s 与 pwm）、符号（体系右手速率、总距 $\ge0$）与缩放（二次推力与实测直流增益）。累计裁剪比例记入 `sat_acc`、`sat_rate`、`sat_thrust`，仅作诊断不作反馈。

## 4. 参数

| 常量 | 值 | 作用 |
|---|---|---|
| `MASS` $m$ | 0.027 kg | 机体质量（cf2x） |
| `G` | 9.81 m/s² | 重力 |
| `TOTAL_THRUST` $F_{\max}^{\mathrm{tot}}$ | 2.0 N | pwm=1 总推力 |
| `THR_MAX` $T_{\max}$ | 0.8 | 环境总距上限 |
| `PWM_GAIN` $G_{\mathrm{pwm}}$ | 1.216 | 噪声均值 1.08 下实测直流增益 |
| `KP_POS` $K_P$ | 3.0 s⁻² | 位置比例 |
| `KI_POS` $K_I$ | 0.10 s⁻³ | 位置积分 |
| `KD_POS` $K_D$ | 4.0 s⁻¹ | 实测速度阻尼 |
| `A_MAX` $A_{\max}$ | 4.0 m/s² | 期望加速度范数钳位 |
| `I_MAX` $I_{\max}$ | 0.5 m/s² | 积分钳位 |
| `KP_ATT` $K_{\mathrm{att}}^{r,p}$ | 4.0 s⁻¹ | 滚/俯姿态 P |
| `KD_ATT` $K_d^{r,p}$ | 0.35 | 滚/俯速率阻尼 |
| `KP_YAW` $K_{\mathrm{att}}^{y}$ | 2.0 s⁻¹ | 偏航姿态 P |
| `KD_YAW` $K_d^{y}$ | 0.20 | 偏航速率阻尼 |
| `YAW_REF` $\psi_r$ | 0 rad | 固定偏航参考 |
| `RATE_MAX` | $\pi$ rad/s | 环境速率指令窗口 |
| `dt` $\Delta t$ | $1/40$ s | 取自 `env.agent_hz` |

**未触碰**的内环常量为环境 rate-PID 增益 $((.04,.04,.08),(5{\times}10^{-7},5{\times}10^{-7},2.7{\times}10^{-4}), (10^{-4},10^{-4},0))$ 与电机 mixer；它们属于 plant。完整动作/观测布局见 [env-quadx-waypoints.zh-CN.md](env-quadx-waypoints.zh-CN.md)。

## 5. 实测表现

以下均为本仓库 GPU 后端、静风（无 `results/quadx/config.json`，即默认 env kwargs）、`scripts/baseline_eval.py` 无偏记账：每副本一个完整首回合，不丢弃飞行中回合。仅发布的 quadx 记录为 `baseline.json`（32 副本/seed 7）、`baseline_cascaded_pid.json`（512 副本/seed 7）与 `results/quadx/baselines/` 下数据集 B（256 副本/seed 7）；检出中无 1024 副本产物。

### 5.1 已发布对比记录

| 臂 | 副本 | 成功 | 坠毁 / 出界 / 超时 | 窗口 | 平均回报 |
|---|---|---|---|---|---|
| SE(3) 几何（`baseline.json`，seed 7） | 32 | 90.6%（29/32） | 0 / 0 / 3 | 800 步 | 66.09 |
| 级联 PID（`baseline_cascaded_pid.json`，seed 7） | 512 | **11.9%**（61/512） | 8.8% / 1.0% / 77.8% | 800 步 | 14.14 |
| 级联 PID（`baselines/cascaded_pid_seed7.json`，数据集 B） | 256 | **99.2%** | 0 / 0 / 0.8% | 800 步 | 72.74 |
| SE(3) 几何（`baselines/se3_geometric_seed7.json`，数据集 B） | 256 | 100.0% | 0 / 0 / 0 | 648 步 | 72.86 |
| SMC（`baselines/smc_seed7.json`，数据集 B） | 256 | 100.0% | 0 / 0 / 0 | 557 步 | 72.86 |

512 副本的 `baseline_cascaded_pid.json` 是**陈旧产物**（77.8% 超时、4 航路点中均达 0.75），早于控制器/plant 修复；同一控制器在当前代码上于 256 副本/seed 7 达 99.2%。不要把 11.9% 当作当前测量。更早的"官方 1024 env"值（99.12%、均值 3.99/4、9 次超时）无法从检出复现，已撤回；§5.3 与 §6 保留选增益时记录的调参扫描数字，但对应 `results/quadx/cascade_pid_tests.json` 未发布。

当前代码上级联（数据集 B，256/seed 7）落后几何臂 0.8 pp，平均回报基本相当。其记录的失败只有两次超时（0.8%）：这些副本追赶边界航路点，恰在 20 s 上限之后到达，计为未成功。**无坠毁、无出界终止**，即方向/权限行为健全，剩余差距是过渡时间，这是纯阻尼位置 PID 缺少 SE(3) 臂速度上限与前馈的预期代价。SE(3) 数据集 B 窗口不同（648 vs 800）仅因为级联运行须等待其超时。

### 5.2 控制方向检查（`results/quadx/cascade_pid_tests.json`）

由 `environments/multirotor/test_quadx_cascade_pid.py` 生成但**未随仓库发布**；下列为调参时记录的 harness 输出，重跑 harness 可得。T0 从稳定水平悬停、零速度读取一次控制调用，故总距比较不受出生下落步污染。值为 $[-1,1]^4$ 环境动作（速率/$\pi$ 与 $2T/T_{\max}-1$）：

| 指令 | 实测 | 检查 |
|---|---|---|
| 悬停，体角速度 | 0.0000 | $\lvert\omega\rvert<10^{-3}$ |
| $+x$ 目标 | 俯仰动作 $+0.3776$ | 俯仰 $>0$ |
| $-x$ 目标 | 俯仰动作 $-0.3776$ | 俯仰 $<0$ |
| $+y$ 目标 | 滚转动作 $-0.3776$ | 滚转 $<0$ |
| $-y$ 目标 | 滚转动作 $+0.3776$ | 滚转 $>0$ |
| $+z$ 目标 | 总距动作 $-0.1451$（悬停 $-0.2518$） | 总距上升 |
| $-z$ 目标 | 总距动作 $-0.3765$ | 总距下降 |
| 偏航 $=+0.5$ rad | 偏航速率动作 $-0.3183$（$-1.0$ rad/s） | 偏航速率 $<0$ |

符号全部与 plant 一致：正 $x$ 误差抬头、正 $y$ 误差左滚、正 $z$ 误差抬高总距、正偏航误差驱动偏航速率朝固定参考减小。

### 5.3 悬停、单航路点、四航路点

- **悬停**（256 env，400 步=10 s，直接 plant 步进以免设定被捕获消耗）：漂移均值 1.29 cm、最大 **5.04 cm**，姿态最大 0.000 rad，稳态总距动作 $-0.2488$（油门 0.300，即含积分修剪的重力补偿值）。积分消除均值增益偏差且无可测姿态偏移。
- **单航路点 +x**，偏移 $(2.5,0,0.6)$ m（64 env，400 步）：末端误差均值 0.111 m、最大 **0.119 m**；位移 $(+2.395,+0.000,+0.568)$ m，$x$/$z$ 方向正确，交叉耦合低于 0.1 mm。
- **单航路点 +y**，偏移 $(0,2.5,0.6)$ m（32 env，300 步）：末端误差均值 0.105 m、最大 **0.118 m**；位移 $(-0.000,+2.400,+0.573)$ m。
- **四航路点**，调参时 512 env/seed 7：成功 **99.41%**，均值 3.994/4，0 坠毁、0 出界、3 超时（调参记录，产物未发布）。该控制器当前数据集 B 记录为 256 env/seed 7 的 99.2%。

## 6. 调参记录

只调外环位置环；姿态 P/PD 与偏航增益保持首个物理合理值，因内环快一个数量级。每个候选在 `run_four_waypoints` 中跑完整四航路点回合并记录裁剪饱和比例 `sat_acc`。

| 候选 | Env | 成功 | 均值 wp | 出界 | 超时 | `sat_acc` | 回报 |
|---|---|---|---|---|---|---|---|
| **$K_D=4.0$，$A_{\max}=4.0$（选用）** | 512 | **99.41%** | 3.994 | 0 | 3 | 0.168 | 72.53 |
| $K_D=2.5$，$A_{\max}=4.0$（首猜） | 512 | 72.85% | 3.373 | 139 | 0 | 0.476 | 58.31 |
| $K_D=2.5$，$A_{\max}=4.0$（首猜） | 256 | 69.53% | 3.258 | 78 | 0 | 0.487 | 56.14 |
| $K_P=4.0$，$K_D=4.0$ | 512 | 96.88% | 3.945 | 16 | 0 | 0.455 | 71.32 |
| $K_D=4.0$，$A_{\max}=5.0$ | 512 | 94.73% | 3.947 | 0 | 27 | 0.115 | 71.81 |

说明：

1. **$K_D=2.5$（初始）**：$K_P=3$ 时阻尼比 $\zeta=2.5/(2\sqrt3)=0.72$，接近 4.5–4.8 m 边界航路点过快，穿过球体并在反向指令制动前离开 5 m 穹顶，导致 27–30% 出界与低回报（256/512 env 均如此，失败模式确定而非噪声）。
2. **$K_D=4.0$（选用）**：$\zeta=1.15$，仅阻尼项即消除过冲；出界降到 0、512 env 成功率升到 99.4%；当前数据集 B 记录为 256 env 的 99.2%，余两次超时为慢速边界过渡而非发散。这就是 §7 讨论的无速度上限替代方案。
3. **$K_P=4.0$**：更硬比例路径在 512 env 重新引入 16 次出界且回报下降，弃用。
4. **$A_{\max}=5.0$**：更大加速度权限无益，把出界失败转为 27 次超时（94.7%），接近又太快无法在捕获半径内稳定，弃用，保留 4.0。
5. **$K_I=0.10$** 未扫描：$G_{\mathrm{pwm}}$ 前馈下稳态偏差已小，0.1 在 10 s 测试内即消除悬停偏移，且 0.5 m/s² 积分钳位防止航路点瞬态绕积。
6. **姿态环**（$K_{\mathrm{att}}=4$ s⁻¹、$K_d=0.35$、偏航 $2.0/0.20$）由 T0 方向符号与悬停测试（0.000 rad 姿态偏移、1.29 cm 平均漂移）验证，未再调：快速工作由环境 rate PID 完成，四航路点运行中唯一失败的自由度是外环阻尼。

## 7. 已知局限

- **无速度上限**：原 SE(3) 臂加 $V_{\max}=1.5$ m/s 巡航上限，因未饱和 PD 外环朝 4.5 m 航路点可达约 6 m/s 并冲过 5 m 穹顶。本臂刻意保留纯位置 PID 形式，改用足够大的阻尼增益（$K_D=4.0$ s⁻¹，$K_P=3$ s⁻² 时 $\zeta\approx1.15$），使接近速度随航路点误差缩小而塌缩。调参显示默认 $K_D=2.5$ 会出穹顶，$K_D=4.0$ 在不加速度参考或安全滤波的前提下消除之；代价是更慢的过渡（§6）。
- **航路点交接**：环境航路点索引变化时清积分；活动航路点恒为 `env.current_target`。控制器不前瞻、不换航路点、不改捕获顺序。
- **姿态参数化**：姿态环用 ZYX Euler 误差，在基准包线内有效（水平出生；$A_{\max}/g<0.41$ 故指令倾角低于 22°）。它不是 SE(3) 臂那样的全局 $SO(3)$ 姿态律；超出 Euler 图卡的注入初始姿态不在预期工作范围。
- **仅均值增益前馈**：$G_{\mathrm{pwm}}$ 修正平均作动增益；逐步乘性噪声与残余 rate-PID 积分/微分留给反馈，位置积分吸收剩余悬停偏差。
- **饱和只记账不规避**：加速度、速率或推力指令裁剪时，控制器记录比例并继续，不围绕饱和重规划。

## 8. 实现

- **控制器**：`QuadXCascadePIDBaseline` 位于 [baselines.py](../../algorithms/baselines/baselines.py)，`name='cascaded_pid'`，注册为命名 A/B 变体 `BASELINE_VARIANTS['quadx']['cascaded_pid']`；默认 `build_baseline('quadx', env)` 仍返回未变的 SE(3) 臂。
- **评测**：通用入口已支持命名变体：
  ```bash
  python scripts/baseline_eval.py quadx                       # SE(3), baseline.json
  python scripts/baseline_eval.py quadx --baseline cascaded_pid  # -> baseline_cascaded_pid.json
  ```
  脚本默认 1024 env/seed 7；已发布记录使用 §5.1 的副本数，环境构造（存在时恢复 `results/quadx/config.json`）、奖励与成功记账均相同。
- **测试**：`environments/multirotor/test_quadx_cascade_pid.py` 运行方向符号检查、悬停保持、+x/+y 单航路点收敛与 512 env 四航路点检查；`--sweep` 另跑小增益扫描。已发布记录由 `scripts/baseline_eval.py` 产生；harness 写 `results/quadx/cascade_pid_tests.json`（未发布）。
- **公平比较命令**：学习臂为已发布入口，未改动：
  ```bash
  python environments/multirotor/train_quadx.py     # ALGORITHM = 'ppo'
  python scripts/eval.py quadx                      # 确定性 checkpoint 评测
  python scripts/baseline_eval.py quadx             # SE(3)
  python scripts/baseline_eval.py quadx --baseline cascaded_pid
  ```
  四臂共享任务、奖励与 `4/4 航路点 < 0.2 m` 成功判据；比较时须显式固定副本数与种子（脚本默认 1024 / seed 7）。

## 9. 参考

- S. Bouabdallah, P. Murrieri, R. Siegwart, "Design and Control of an Indoor Micro Quadrotor," *IEEE ICRA*, 2004 —— 小型四旋翼的位置/姿态/速率级联。
- D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," *IEEE ICRA*, 2011 —— 含重力补偿与固定偏航姿态参考的级联结构。
- T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *IEEE CDC*, 2010 —— 配套 SE(3) 臂与几何姿态替代。
- J. K. Tai et al., *PyFlyt*, v0.29.0 —— cf2x 机体模型、120 Hz rate PID、电机 mixer、滞后与噪声。
