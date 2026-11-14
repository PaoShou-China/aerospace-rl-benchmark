# Rocket-Landing 基线：解析 ZEM/ZEV 动力下降制导（`RocketPDGBaseline`）

**简体中文** · [English](baseline-rocket-pdg.md)

## 1. 概述

火箭着陆任务的传统基线是**完全解析**的动力下降制导与控制栈：中段下降用闭式零努力脱靶量/零努力速度（ZEM/ZEV）制动律；末端为两阶段自杀燃烧，跟踪到虚拟平台制动流形；终端为恒速脉冲伺服；内环为 PD 姿态环并提供气动力矩前馈，输出分配到发动机万向节（TVC）与四片栅格翼。**没有在线优化器**——每个策略步所有量均闭式重解，故该方法无求解器、无求解失败模式；超出作动权限的需求在相应通道饱和（解析律的确定性回退），设备端饱和/滑行/点火切换统计即求解器诊断的诚实类比。控制器为全批量张量核：无逐环境 Python 循环、无主机同步。

## 2. 接口约定

- **输入**：仅 `RocketLandingEnv` 的 `'policy'` 观测组（30 维，float32，内部转 float64）：体角速度 `[0:3]`、姿态四元数 xyzw `[3:7]`、体系速度 `[7:10]`、世界位置 `[10:13]`、上一原始动作 `[13:20]`、栅格翼作动 `[20:24]`、点火标志 `[24]`、燃料比例 `[25]`、油门状态 `[26]`、万向节状态 `[27:29]`、平台接触标志 `[29]`。
- **输出**：环境约定 $[-1,1]^7$ 原始动作 $(f_x,f_y,f_\psi,\texttt{ign},\texttt{thr},g_x,g_y)$。环境将 $[-1,1]^7$ 仿射映射到下界 $(-1,-1,-1,0,0,-1,-1)$ 的 Box，故基线输出点火 $\pm1$、油门 $2u-1$（$u\in[0,1]$）；plant 再把 $u$ 重映射到 $[0.39,1]T_{\max}$。
- **构造**：机体常量（$T_{\max}$、最小推力比、干/燃料质量、万向节范围、鳍系数与力臂）在构造时取自环境/URDF 规格，对应经典 GNC 设计者的模型知识；从不读内部环境状态缓冲。
- **逐回合状态**：无——点火滞环由**观测到的**点火标志驱动，故默认空 `reset_idx` 足够；仅诊断计数器（`steps, sat_max, coast, ignitions, infeasible, floor, fin_sat`）跨回合保留。
- **速率**：策略 40 Hz / 控制 120 Hz（每策略步 3 个内环控制周期）/ 物理 240 Hz；30 s（1200 步）回合上限。

## 3. 制导与控制律

令 $m=M_{\text{dry}}+fM_{\text{fuel}}$，$v_w$ 为世界速度，$s=\max(0,-v_{w,z})$ 为下降速度，$r_f=(0,0,2.575\ \text{m})$ 为平台静止点，$g=9.81$ m/s²。

### 3.1 中段 ZEM/ZEV 制动（$s\ge S_{\text{gate}}=25$ m/s）

每步以最大净制动加速度 60% 的恒定减速度剖面重解剩余时间（早制动使气流——鳍对齐目标——近垂直；剩余 40% 权限吸收横向倾斜分量与阻力模型误差），随后为朝 $r_f$、零终端速度的 ZEM/ZEV 加速度（速度项**相减**：制动朝目标运动、加固远离目标运动）：

$$
a_{\text{net}}=\max\!\Big(\tfrac{T_{\max}}{m}-g,1\Big),\quad
a_{\text{brake}}=0.6\,a_{\text{net}},\quad
t_{go}=\operatorname{clamp}\!\Big(\frac{s-V_{\text{touch}}}{a_{\text{brake}}},0.5,30\Big),\quad
\mathrm{ZEM}=r_f-p-v_wt_{go},\quad
a^{\text{cmd}}=\frac{6\,\mathrm{ZEM}}{t_{go}^2}-\frac{2\,v_w}{t_{go}}. \tag{1}
$$

垂直通道取下限、横向需求按倾斜一致预算封顶（约 $6^\circ$，在满鳍气动保持限约 $9.6^\circ$ 内）；推力指向加速度加重力且从不奖励向下需求（发动机只能沿体 $+z$ 推）；推力在 $T_{\max}$ 饱和；$T_{\min}=0.39T_{\max}$ 附近点火滞环自点火：

$$
|a^{\text{cmd}}_{xy}|\le\operatorname{clamp}\!\big(0.11\max(a^{\text{cmd}}_z+g,0),1,4.5\big),\quad
a^{\text{thr}}=a^{\text{cmd}}+ge_z\ (a^{\text{thr}}_z\ge0),\quad
T_{\text{cmd}}=\operatorname{clamp}\!\big(m\lVert a^{\text{thr}}\rVert,0,T_{\max}\big), \tag{2}
$$

$$
\text{当 } T_{\text{cmd}}\ge1.03T_{\min}\text{ 点火};\qquad \text{当 } T_{\text{cmd}}\ge0.97T_{\min}\text{ 保持点火}. \tag{3}
$$

### 3.2 末端阶段 1：虚拟平台最小推力流形（$s<25$ m/s）

最小推力超过重力（158.7 kg 出生质量下 $T_{\min}/(mg)=1.91$），故点火发动机无法跟随缓慢下降；末端沿抬高 $H_{\text{virt}}=2$ m 的**虚拟**平台最小推力流形制动。令 $h_{\text{agl}}=\max(p_z-2.575,0)$：

$$
a_{\min}=\max\!\Big(\tfrac{0.39T_{\max}}{m}-g,0.5\Big),\quad
A_{\text{end}}=a_{\min}+\varepsilon_{\text{end}}\ (\varepsilon_{\text{end}}=1.0),\quad
h_v=\max(h_{\text{agl}}-H_{\text{virt}},0.05), \tag{4}
$$

$$
s_t^2=V_{\text{touch}}^2+2A_{\text{end}}h_v,\qquad
a_{\text{req}}=\frac{v_{w,z}^2-s_t^2}{2h_v}+g+A_{\text{end}}. \tag{5}
$$

式 (5) 的 $+A_{\text{end}}$ 斜率项是关键：它使律**跟踪**流形（速度误差按 $e\approx e_0h/h_0$ 衰减），而非仅在流形上终止。$\varepsilon_{\text{end}}$ 裕度让指令推力高于 $T_{\min}$ 约 5%，使 plant 乘性推力噪声（$1+0.01\mathcal N$）既不能削到最小推力下限，也不能过度制动导致熄火—复燃抖动。流形以下发动机滑行——$T_{\min}>mg$ 时发动机无法比流形下降更慢，故滑行**就是**线下分支。

### 3.3 末端阶段 2：终端脉冲伺服（$s\le1.5$ m/s，$h_{\text{agl}}\le2$ m）

脉冲门内垂直需求为常量 $a_{\text{vert}}=g+A_{\text{end}}$，点火为速度滞环——$s<S_{\text{off}}=0.30$ m/s 熄火，$s>S_{\text{on}}=0.40$ m/s 复燃。点火时净加速度 $+a_{\min}$（仅制动）；熄火时 $-g$。速度带不随高度收缩，故在策略周期量化下触地速度按构造有界于约 $[0.15,0.65]$ m/s，低于 1.0 m/s 致命预接触阈值。平台接触强制熄火，但速度带伺服**不**在触地后永久锁死：微弹跳再发射在下落途中重入带，$S_{\text{on}}$ 复燃把它制动回平台（试过触地后硬熄火锁存，会把回弹变成 1.36 m/s 致命再接触）。刻意无上升侧复燃：发动机只能向上推，上升时点火是助推而非接住（试过：1024 env 规模下持续弹跳—助推振荡，约 25% 超时）。脉冲阶段横向律只阻尼横向**速度**，封顶 $A_{\text{lat,pulse}}=1.5$ m/s²；阶段 1 横向律为 $a_{\text{lat}}=-1.5v_{w,xy}-p_{xy}$，按式 (2) 封顶。

### 3.4 油门、姿态 PD 与作动分配

油门指令 $u=\operatorname{clamp}((T_{\text{cmd}}/T_{\max}-0.39)/0.61,0,1)$，输出为 $2u-1$。期望推力轴 $b_3=a^{\text{thr}}/\lVert a^{\text{thr}}\rVert$（需求消失或 $b_{3,z}\le0.15$ 时回退 $e_z$，保持跨滑行/点火过渡的制导倾斜）；令 $u_b=R^\top b_3$，倾斜误差 $e_{\text{tilt}}=(-u_{b,y},u_{b,x},0)$。栅格翼等效刚度 $k_{\text{aero}}V^2$ 的气动恢复弹簧把体 $+z$ 拉向逆气流轴；前馈在倾斜轴上抵消它，封顶于鳍保持能力（线性弹簧模型在约 $9.6^\circ$ 失速配平点外失效）：

$$
\tau_{\text{ff}}=\operatorname{clamp}\!\big(k_{\text{aero}}V^2e_{\text{flow}},\pm0.95K_{\text{fin}}z_{\text{fin}}V^2\big),\qquad
\tau^{\text{cmd}}=\big(K_{P,T}e_{\text{tilt},xy}-K_{D,T}\omega_{b,xy},-K_{D,Z}\omega_{b,z}\big)-\tau_{\text{ff}}. \tag{6}
$$

世界系横向力预算分给三个作动器；推力倾斜与万向节侧力的贡献从**观测到的**油门与万向节状态重建，故鳍只补真实残差，封顶为其力权限 30%（$d_{\text{fin}}=a_{\text{fin}}^{\text{body}}m/(K_{\text{fin}}V^2)$）。偏转鳍在 $z_{\text{fin}}$ 力臂上也会给体加力矩；发动机点火时该耦合力矩前馈补偿进 $\tau^{\text{cmd}}$（滑行时留给气动弹簧，因鳍把力矩锁定于力）。力矩按上限分配——先快速 TVC，鳍补余量——$k_{\text{gim}}=hT_{\text{now}}\delta g_{\max}$、$k_{\text{fin}}=K_{\text{fin}}z_{\text{fin}}V^2$：

$$
\tau_{\text{gim}}=\operatorname{clamp}(\tau^{\text{cmd}},\pm0.95k_{\text{gim}}),\qquad
\tau_{\text{fin}}=\operatorname{clamp}(\tau^{\text{cmd}}-\tau_{\text{gim}},\pm0.35k_{\text{fin}}). \tag{7}
$$

符号已核对的作动映射输出 $f_x=\tau_{\text{fin},y}/k_{\text{fin}}+d_{\text{fin},x}$、$f_y=-\tau_{\text{fin},x}/k_{\text{fin}}+d_{\text{fin},y}$、$g_{x,y}=-\tau_{\text{gim},x,y}/k_{\text{gim}}$（发动机关机时为零）、$f_\psi=-K_{D,Z}\omega_{b,z}/(K_{\text{yaw}}V^2)$。

## 4. 参数

全部为 `RocketPDGBaseline` 的精确代码常量。

| 常量 | 值 | 含义 |
|---|---|---|
| `M_DRY` | 138.2 kg | 结构质量（基座 91 + 助推 47 + 4 鳍 ×0.05） |
| `FUEL_FULL` | 410.9 kg | 燃料比例 1 时贮箱质量（出生燃料 5%） |
| `T_MAX` | 7607.0 N | 助推最大推力 |
| `MIN_RATIO` | 2966.7/7607.0 ≈0.390 | 最小/最大推力比 |
| `GIMBAL_RAD` | 5° | 两轴万向节范围 |
| `G` | 9.81 m/s² | 重力 |
| `FIN_K` | 0.22668 | 单位偏度鳍力矩 [N·m/unit/(m/s)²] |
| `FIN_YAW_K` | 0.1587 | 四鳍差动偏航力矩系数 |
| `AERO_K` | 3.6 | 鳍恢复弹簧刚度 [N·m/rad/(m/s)²] |
| `V_TOUCH` | 0.35 m/s | 脉冲带中心 / 流形终端速度 |
| `TGO_FLOOR` | 0.5 s | ZEM/ZEV 奇异保护（上限钳 30 s） |
| `S_GATE` | 25 m/s | 低于此空速末端跟踪器取代式 (1) |
| `EPS_END` | 1.0 m/s² | 末端相对最小推力净制动的减速裕度 |
| `H_VIRT` | 2.0 m | 虚拟平台抬高 |
| `S_PULSE` | 1.5 m/s | 脉冲伺服速度门 |
| `S_OFF` / `S_ON` | 0.30 / 0.40 m/s | 脉冲滞环熄火 / 复燃阈值 |
| `H_PULSE` | 2.0 m | 脉冲伺服高度上限 |
| `A_LAT_PULSE` | 1.5 m/s² | 脉冲阶段横向加速度上限（约 $1.6^\circ$ 倾斜） |
| `LAT_TILT` | 0.11 | 横向上限占垂直加速度比例（约 $6^\circ$） |
| `A_LAT_MAX` | 4.5 m/s² | 横向加速度绝对上限 |
| `KP_T` | 12000 N·m/rad | 倾斜力矩增益 |
| `KD_T` | 6000 N·m·s/rad | 倾斜速率阻尼（约 ζ 0.8） |
| `KD_Z` | 3.0 N·m·s/rad | 偏航自旋阻尼 |
| `REST_Z` | 2.575 m | 平台顶 0.15 m + 脚底偏置 2.425 m |

## 5. 实测表现

成功判据现为 v2 benchmark 指标 `ever_stable_2s`：平台接触且 $\lVert v\rVert<0.02$ m/s、$\lVert\omega\rVert<0.02$ rad/s、倾角 $<0.1$ rad 连续 2 s（40 Hz 下 81 个策略样本），预接触速度 $>1.0$ m/s 或自旋 $>0.35$ rad/s 为致命。瞬时逐接触事件仍驱动稳定奖励。

| 来源 / 协议 | 结果 |
|---|---|
| `results/rocket/baseline.json` —— 官方 `scripts/baseline_eval.py rocket`，128 env、seed 13、GPU、1216 env 步（每副本一个首回合） | 成功率 **100%**（128/128，`ever_stable_2s`），平均回合回报 900.2，`complete=true` |
| `results/rocket/baselines/zem_zev_pdg_seed7.json` —— 数据集 B，128 env、seed 7、1216 env 步 | 成功率 **100%**，平均回合回报 900.15 |
| `results/rocket/baselines/pd_landing_seed7.json` —— 数据集 B，128 env、seed 7 | 成功率 100%，平均回合回报 7.27 |
| `results/rocket/baselines/eguidance_seed7.json` —— 数据集 B，128 env、seed 7 | 成功率 39.1%，平均回合回报 −3304.8 |
| `results/rocket/pdg_test_report.json`，场景 A–E —— `rocket_pdg_test.py`，1024 副本、1250 步窗口、seed 7、GPU（harness 产物，**未发布**） | 场景 A（上游随机初态）：SR 0.4395；平台触地率 0.731；触地速度 p10/p50/p90 = 0.077/0.282/1.382 m/s。B（角点初态）：SR 0.3363。C/D/E（名义 z 425 m；低 z 400 m；±3 m/s 横向漂移）：0.4785 / 0.4961 / 0.4717 |

2026-11-12 复测最初测得三个臂全部 **0%**：2026-11-05 的“禁用接触约束求解”被应用到了所有机型，助推器虽然检测到平台却得不到接触冲量，穿过平台后无法满足停稳判据。现在 legacy 模式下的火箭保持刚体接触求解器开启（`plant.py`），恢复了稳定触地（`touch_speed ≈ 0.001 m/s`）与上表的 100% 记录。

历史 60 Hz 记录（1024 env、1800 步、成功 0.8281、回报 3968.2）早于 40 Hz 策略切换与 2 s 稳定指标；与当前行不可比，仅作历史保留。

同 harness 制导诊断（解析律的求解器统计类比），场景 A：最大推力饱和占 0.89% 步，关机滑行 50.9%，不可行需求 0.89%，$t_{go}$ 触底 60.6%，每回合约 122 次点火脉冲，零燃料耗尽回合（出生燃料 20.5 kg 中平均用 7.0 kg），峰值过载 5.07 g 均值 / 5.39 g 最大；场景 B–E 同量级。残余失败以致命碰撞为主（50–66%）；出界与超时各低于 1%。

## 6. 已知局限

仅列代码注释或 README 记录的局限。

- **稳定阶段摇晃/翻倒致命**。主要残余失败是到达状态质量而非触地速度：实测触地自旋（均值 0.12–0.15 rad/s，最大达 2.2 rad/s）远高于 0.02 rad/s 稳定阈值。点火时唯一阻尼摇晃的作动路径是倾斜速率阻尼；调参记录 $K_{D,T}=8000$ 单独略好（83.9%）但与软带配合变差，10000 过冲（79.2%）。
- **权限饱和制动瞬态**作为有记录的残余失败保留（根 README），表现为上述约 0.9% 饱和与不可行需求占比。
- **脉冲带量化**：触地速度保证受量化约束（约 0.25 m/s 每 25 ms 策略样本，即一个 40 Hz 策略步保持三个 120 Hz 控制步）。
- **气动模型有效性**：线性鳍弹簧模型仅在约 $10^\circ$ 倾斜内成立（鳍在约 $9.6^\circ$ 配平点外失速），故前馈封顶；大倾角指令超出设计包线。
- **单向发动机**：$T_{\min}/(mg)=1.91$ 时发动机只能制动；终端下降必然是脉冲串，任何上升侧复燃都是助推（试过的弹跳—助推失败模式），故回弹接住仅在下落侧。
- **饱和是回退**：超出权限的需求使相应通道饱和；无可行性投影。

## 7. 实现

- 控制器：`RocketPDGBaseline`（`name='zem_zev_pdg'`）位于 `algorithms/baselines/baselines.py`，注册于 `BUILDERS['rocket']`；由 `build_baseline('rocket', env)` 实例化，在 `environments/rocket/env.py::RocketLandingEnv` 声明为 `conventional_controller`。
- 评测：`scripts/baseline_eval.py rocket`（官方协议，写 `results/rocket/baseline.json`）；深入 harness `environments/rocket/rocket_pdg_test.py`（写 `results/rocket/pdg_test_report.json`，未发布）。

## 8. 参考

1. Battin, R. H. *An Introduction to the Mathematics and Methods of Astrodynamics*，AIAA —— ZEM/ZEV 制导背景。
2. Açıkmeşe, B. & Ploen, S. R. (2007), "Convex Programming Approach to Powered Descent Guidance for Mars Landing," *J. Guidance, Control, and Dynamics* 30(5) —— PDG 语境；本基线是无优化器的对应物。
3. Tai, J. K. *PyFlyt*（0.29.0）—— 上游任务、机体模型与着陆判据。
