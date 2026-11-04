# QuadX 基线：SE(3) 几何控制（`QuadXSE3Baseline`）

**简体中文** · [English](baseline-quadx-se3.md)

## 1. 概述

QuadX-Waypoints 任务要求 Crazyflie 2.x 微型四旋翼（27 g，X 构型）在 5 m 飞行穹顶内按序访问 4 个随机航路点，进入 0.2 m 球即捕获，20 s 时限、40 Hz 策略；触地或出界失败。本任务的传统基线为 Lee、Leok、McClamroch（2010）的几何跟踪控制，即欠驱动四旋翼航路点跟踪的标准基于模型解法：带重力前馈的 PD 位置环给出期望力矢量；总距为其在体推力轴上的投影；直接在 $SO(3)$ 上定义姿态环使推力轴对准期望力；所得 wrench 闭式分配到 4 个旋翼。为适配基准 plant 做了两项实用增强：尊重飞行穹顶的速度饱和外环，以及吸收作动链电机噪声直流增益的轴向积分修剪（含抗饱和）。控制器确定性、解析——无训练、无调参环、无在线优化器。

## 2. 接口约定

- **观测**：只读 `'policy'` 组：体角速度 $\omega_b$（`[0:3]`）、四元数 $(x,y,z,w)$（`[3:7]`）、体速度 $v_b$（`[7:10]`）、世界位置 $p$（`[10:13]`）。活动航路点 $p^\star$ 取自 `env.current_target`；同一量也以 slot-0 体系偏差加本机位置编码在观测中（fixedwing 基线正做此重建），故无特权状态。
- **构造期模型知识**：控制周期 $\Delta t=1/\texttt{agent\_hz}=1/40$ s；plant 常量（质量、惯量、旋翼臂长、$k_m/k_f$、推力窗口、内环 rate-PID 比例增益）为 Crazyflie 2.x 参考 URDF/YAML 的冻结类常量，对应经典 GNC 设计者本会掌握的模型知识；不读任何内部 plant 状态缓冲。
- **内部状态**：每副本一个标量轴向修剪积分；`reset_idx(mask)` 随环境 reset 掩码清零，无其它记忆。
- **输出**：环境原生动作 $a\in[-1,1]^4$：三轴体角速度指令（物理范围 $\pm\pi$ rad/s）与总距（物理范围 $[0,0.8]$）。全部计算为 `env.device` 上 float64、跨副本批量；返回动作为 float32。

## 3. 控制律

**位置环（速度饱和级联）** 令 $e_x=p-p^\star$、世界速度 $v=R v_b$，外环给出限速接近速度
\begin{equation}
\bar v_d=-\frac{K_X}{K_V}e_x,\qquad
v_d=\bar v_d\frac{V_{\max}}{\max(\lVert\bar v_d\rVert,V_{\max})},\qquad
e_v=v-v_d, \tag{1}
\end{equation}
未饱和时退化为经典 PD（$-K_V e_v=-K_V v-K_X e_x$），过渡段限速 $V_{\max}$。期望力与总距
\begin{equation}
F_{\mathrm{des}}=m(g e_3-K_V e_v),\qquad
f_{\mathrm{raw}}=F_{\mathrm{des}}^\top b_3+I,\qquad
f=\mathrm{clip}(f_{\mathrm{raw}},f_{\min},f_{\max}), \tag{2}
\end{equation}
$b_3=R e_3$ 为世界系体推力轴，$f_{\min}=4F_{\max}^{\mathrm{tot}}\pi_{\min}^2=0.02$ N，$f_{\max}=F_{\max}^{\mathrm{tot}}T_{\max}^2=1.28$ N。轴向修剪积分抵消悬停残差
\begin{equation}
\dot I=-K_I(e_x^\top b_3)\mathbb 1\{\lVert e_x\rVert<0.5\ \mathrm{m}\},\qquad I\in[-I_{\max},I_{\max}], \tag{3}
\end{equation}
带抗饱和门控：$f_{\mathrm{raw}}>f_{\max}$ 时 $\dot I$ 被钳为非正、$f_{\mathrm{raw}}<f_{\min}$ 时钳为非负，使积分器在加剧饱和的方向冻结。

**期望姿态（偏航固定为零）**
\begin{equation}
b_{3,d}=\frac{F_{\mathrm{des}}}{\lVert F_{\mathrm{des}}\rVert},\qquad
b_1=\frac{e_1-(e_1^\top b_{3,d})b_{3,d}}{\lVert\cdot\rVert},\qquad
R_d=[\,b_1,\ b_{3,d}\times b_1,\ b_{3,d}\,], \tag{4}
\end{equation}
$\lVert F_{\mathrm{des}}\rVert\le10^{-6}$ 时回退到 $b_3$，$b_{3,d}\approx\pm e_1$ 时退回 $e_2$ 投影。

**$SO(3)$ 姿态环** 采用 vee 映射 $\mathrm{vee}(S)=\tfrac12(S_{32}-S_{23},S_{13}-S_{31},S_{21}-S_{12})$：
\begin{equation}
e_R=\tfrac12\mathrm{vee}(R_d^\top R-R^\top R_d),\qquad
e_\omega=\omega_b-R^\top R_d\omega_d=\omega_b, \tag{5}
\end{equation}
因航路点设定静止（$\omega_d=0$）。力矩律
\begin{equation}
M=-K_R e_R-K_\Omega\omega_b+\omega_b\times(J\omega_b). \tag{6}
\end{equation}

**分配** 旋翼臂 $(a_{y,i})=(-1,1,1,-1)$、$(a_{x,i})=(1,-1,1,-1)$（电机顺序 $(+x,-y),(-x,+y),(+x,+y),(-x,-y)$），旋向 $s=(-1,-1,1,1)$，臂长 $d$，$c=k_m/k_f$：
\begin{equation}
t_i=\frac f4+\frac{M_x a_{y,i}-M_y a_{x,i}}{4d}+\frac{M_z s_i}{4c},\qquad
t_i\leftarrow\mathrm{clip}(t_i,F_{\max}^{\mathrm{tot}}\pi_{\min}^2,F_{\max}^{\mathrm{tot}}/4), \tag{7}
\end{equation}
精确复现 wrench $(f,M)$（臂/旋向列互相正交且平方范数为 4），裁剪到单旋翼窗口 $[0.005,0.5]$ N。推力经 plant 二次推力模型映射为 pwm，并用实测电机滞后直流增益 $G_{\mathrm{pwm}}=1.216$ 预补偿：
\begin{equation}
\pi_i=\frac{1}{G_{\mathrm{pwm}}}\sqrt{\frac{4t_i}{F_{\max}^{\mathrm{tot}}}},\qquad
T=\mathrm{clip}(\tfrac14\textstyle\sum_i\pi_i,0,T_{\max}),\qquad
d_i=\pi_i-T. \tag{8}
\end{equation}

**速率指令合成** 环境 120 Hz 内环为 rate PID，其比例路径给出归一化力矩 $u\approx k_p(\omega^{\mathrm{cmd}}-\omega_b)$，经 $\pi_i=T+u_x a_{y,i}+u_y(-a_{x,i})+u_z s_i$ 混入旋翼。反演（正交）mixer 与比例增益：
\begin{equation}
\tau=\frac14\sum_i d_i(a_{y,i},-a_{x,i},s_i),\qquad
\omega^{\mathrm{cmd}}=\omega_b+\tau\oslash k_p,\qquad
\omega^{\mathrm{cmd}}\leftarrow\mathrm{clip}(\omega^{\mathrm{cmd}},-\pi,\pi). \tag{9}
\end{equation}
动作
\begin{equation}
a=\Big(\frac{\omega^{\mathrm{cmd}}}{\pi},\ \frac{2T}{T_{\max}}-1\Big)\ \mathrm{clipped\ to}\ [-1,1]^4. \tag{10}
\end{equation}
累计裁剪指令比例记入 `sat_total` / `sat_rotor` / `sat_rate` 供离线饱和统计。

## 4. 参数

| 常量 | 值 | 作用 |
|---|---|---|
| `MASS` $m$ | 0.027 kg | 机体质量（cf2x URDF） |
| `G` | 9.81 m/s² | 重力 |
| `INERTIA` $J$ | $\mathrm{diag}(1.4,1.4,2.17)\times10^{-5}$ kg·m² | 基座对角惯量 |
| `ARM` $d$ | 0.028 m | 旋翼到体 $z$ 轴距离 |
| `KM_KF` $c$ | $7.94\times10^{-12}/3.16\times10^{-10}\approx0.02513$ | 反扭矩/推力比 |
| `TOTAL_THRUST` $F_{\max}^{\mathrm{tot}}$ | 2.0 N | pwm=1 时总推力 |
| `PWM_MIN` / `PWM_MAX` | 0.05 / 1.0 | 单旋翼 pwm 窗口 |
| `THR_MAX` $T_{\max}$ | 0.8 | 环境总距上限 |
| `KP_RATE` $k_p$ | $(0.04,0.04,0.08)$ | plant 内环 rate-PID 比例增益 |
| `PWM_GAIN` $G_{\mathrm{pwm}}$ | 1.216 | 噪声均值 1.08 下实测电机滞后直流增益 |
| `KX` $K_X$ | 4.0 s⁻² | 位置比例增益 |
| `KV` $K_V$ | 3.0 s⁻¹ | 速度比例增益 |
| `V_MAX` $V_{\max}$ | 1.5 m/s | 巡航限速 |
| `KR` $K_R$ | $8\times10^{-4}$ N·m/rad | 姿态比例增益 |
| `KOM` $K_\Omega$ | $2.5\times10^{-4}$ N·m·s/rad | 体角速度阻尼 |
| `KI_POS` $K_I$ | 0.15 N/m | 轴向位置误差积分增益 |
| `I_MAX` $I_{\max}$ | 0.35 N | 积分钳位（约 1.3× 悬停推力） |
| 近端门 | 0.5 m | 修剪仅在捕获邻域内生效 |
| `dt` $\Delta t$ | $1/40$ s | 取自 `env.agent_hz` |

## 5. 实测表现

| 协议 | 来源 | 结果 |
|---|---|---|
| 数据集 A（128 副本 × seeds 7/13/21/101，每副本首回合） | `results/baseline_protocol/seed*/quadx/baseline.json` | 各 seed 100.0 / 100.0 / 99.2 / 100.0%；384 回合均值 99.7±0.5% |
| 数据集 B（256 副本，seed 7） | `results/quadx/baselines/se3_geometric_seed7.json` | 100% 成功，平均回合回报 72.86 |
| `baseline_eval.py quadx` 已发布记录（32 副本，seed 7，800 步窗口） | `results/quadx/baseline.json` | 29/32 首回合成功（0.906），平均回报 66.09，`complete=true` |

已发布的 `results/quadx/baseline.json` 是 32 副本 smoke 规模记录，不是 1024 副本运行；99.7% 来自四 seed 数据集 A。旧 1024 数值（如 1024/1024、回报 70.91、523 步）无法从当前检出复现，已撤回。

功能测试 `environments/multirotor/test_quadx_se3.py` 认证（失败即抛错）：无仿真下精确 wrench→旋翼分配（T0）；跨批大小 $\ge0.90$ 航路点成功率（T1）；1024 env 域随机生成及注入对抗初始位置/姿态下 $\ge0.85$（T2）；速率窗口、单旋翼与总推力饱和瞬态下的闭环恢复（T3）；部分批重置下修剪积分清零与再收敛（T4）。

## 6. 已知局限

- 速度上限 (1) 是对纯 PD 的刻意偏离：未饱和时外环朝 4.5 m 航路点加速到约 $(K_X/K_V)\lVert e_x\rVert\approx6$ m/s，冲过 5 m 穹顶。
- 修剪积分只用于抵消毫米级悬停残差；门控在 $\lVert e_x\rVert<0.5$ m，因为数米级过渡误差会把修剪积到钳位（约 1.3× 悬停推力）并抬高每次接近。
- $G_{\mathrm{pwm}}$ 前馈只修正**平均**作动增益（乘性噪声下电机滞后不动点 $\mathbb E[1+0.02\mathcal N(4,1)]=1.08$）；逐步噪声不补偿。
- 设定在每个控制步按静止处理：$\omega_d=0$，指令偏航固定为零，无轨迹前馈。
- 恶劣瞬态下指令饱和（总推力、单旋翼 pwm、$\pm\pi$ 速率窗口）；控制器只记录饱和比例，不重规划。

## 7. 实现

- 控制器：`algorithms/baselines/baselines.py`，类 `QuadXSE3Baseline`（`name='se3_geometric'`），经 `build_baseline('quadx', env)` 实例化；环境在 `environments/multirotor/env.py` 将其声明为 `conventional_controller`。
- 评测：`scripts/baseline_eval.py quadx`（产物 `results/quadx/baseline.json`）；功能 harness `environments/multirotor/test_quadx_se3.py`。
- 环境参考：`docs/method/env-quadx-waypoints.md`（动作约定、rate-PID 内环、电机混合、作动滞后与噪声）。

## 8. 参考

- T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *IEEE CDC*, 2010 —— 力/力矩律 (2)、(4)–(6)。
- Tai, J. K. et al., *PyFlyt*, v0.29.0 —— 上游 Crazyflie 2.x 机体模型、rate-PID 内环与电机 mixer。
