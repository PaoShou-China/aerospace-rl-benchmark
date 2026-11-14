# Fixedwing 基线：L1 制导 + TECS 自动驾驶（`FixedwingAutopilotBaseline`）

**简体中文** · [English](baseline-fixedwing-l1-tecs.md)

## 1. 概述

Fixedwing-Waypoints 要求 2.35 kg、1.6 m 翼展无人机从 $(0,0,10)$ m 以 20 m/s 出发，在 120 s、30 Hz 策略下访问原点周围 20–65 m 球壳（高度地板 20 m）内的 4 个航路点；进入 10 m 球即捕获，触地或离开 100 m 出界球失败。传统基线为常规固定翼自驾：**L1 横向制导**（Park, Deyst & Howe 2004）把到活动航路点的视线角转为滚转指令，简化 **TECS 纵向环**（Lambregts 1983）把比总能量误差分配给油门、能量平衡误差分配给俯仰指令；内环滚转/俯仰经环境直接舵面接口闭合。因机体功率受限（最大推力 18 N < 重力约 23 N，最小转弯半径 10–15 m），发布的控制律加入有记录的终端收敛增强：距离调度 L1、加速失速边界的气动滚转限、高度优先滚转门、终端纯追踪混合、出界护盾与超速爬升权衡。

## 2. 接口约定

- **观测**：只读 `'policy'` 组：$\omega_b$ `[0:3]`、四元数 `[3:7]`、$v_b$ `[7:10]`、$p$ `[10:13]`、到 4 个航路点槽的体系偏差 $\Delta_k$ `[23:35]`；活动航路点重建为 $w=p+R\Delta_0$。控制器无空速传感器，用 $V=\lVert v_b\rVert$（体系地速）作空速；静风时二者一致，有风时 plant 用 $v_{\text{air}}=v-w$ 但控制器不用。下文报告基线记录均为静风（`env_kwargs={}`）。
- **构造期知识**：全部增益与限幅取自 `algorithms/baselines/fixedwing_autopilot.json`；控制周期 $\Delta t=1/30$ s（`agent_hz`）。角度用环境 Euler 约定（x 前、y 左、z 上）：正滚转 = 右倾，正俯仰 = 低头。
- **内部状态**：每副本两个标量积分器（油门能量积分、俯仰配平积分），`reset_idx(mask)` 清零。
- **输出**：原生动作 $a\in[-1,1]^4$：$a_0$ 差动副翼（$a_0>0$ 右滚），$a_1$ 平尾（$+$ 低头；反号垂尾耦合当扰动），$a_2\equiv0$（上游死通道：主翼偏度限为零），$a_3$ 油门重映射 $\theta^{\mathrm{cmd}}=(a_3+1)/2$，推力 $\theta^2\cdot18$ N。float64 批量计算，返回 float32。

## 3. 控制律

记 $\sigma(x)=1/(1+e^{-x})$、$\mathrm{wrap}(\cdot)$ 到 $(-\pi,\pi]$；$\phi,\theta$ 为 $R$ 的 ZYX Euler 滚/俯角；$d_3=\lVert\Delta_0\rVert$、$h_{\mathrm{err}}=w_z-p_z$、$\rho_n=\lVert p\rVert$、$\rho_h=\lVert p_{xy}\rVert$。

**横向瞄准点**：爬升段包容权重 $w_{\mathrm{loit}}=\sigma((\rho_h-\rho_{\mathrm{loit}})/3)\sigma((h_{\mathrm{err}}-h_{\mathrm{loit}})/3)$ 发布时禁用（$h_{\mathrm{loit}}=10^4$ m，因而 $w_{\mathrm{loit}}\equiv0$）；瞄准距离与方位混合航路点与原点视线：
\begin{equation}
d_h=(1-w_{\mathrm{loit}})\lVert w_{xy}-p_{xy}\rVert+w_{\mathrm{loit}}\rho_h,\quad
\lambda=\mathrm{wrap}\big(\lambda_w+w_{\mathrm{loit}}\mathrm{wrap}(\lambda_o-\lambda_w)\big), \tag{1}
\end{equation}
$\lambda_w=\mathrm{atan2}(w_y-p_y,w_x-p_x)$、$\lambda_o=\mathrm{atan2}(-p_y,-p_x)$。航迹 $\chi=\mathrm{atan2}(v_y,v_x)$（$\lVert v_{xy}\rVert>2$ m/s 时）否则取 $R$ 航向；$\eta=\mathrm{wrap}(\lambda-\chi)$，钳位 $\eta_c\in[-\pi/2,\pi/2]$，使身后航路点指令满强度转弯而非经 $\sin\eta$ 消失。

**L1 制导（距离调度）**：
\begin{equation}
L_1=\mathrm{clip}(k_{L1}d_h+L1_0,\ L1_{\min},\ L1_{\max}),\quad
a_{\mathrm{lat}}=\frac{2V^2\sin\eta_c}{L1}\mathbb 1\{d_h>3\ \mathrm{m}\},\quad
\phi_{L1}=-\arctan\frac{a_{\mathrm{lat}}}{g}, \tag{2}
\end{equation}
含 3 m 近场死区（目标近乎正上方）与环境滚转符号约定（左侧目标 $\eta>0$ 需右手负滚转）。

**滚转限**：加速失速边界 $n_{\max}=(V/v_{\mathrm{stall}})^2$ 给出气动滚转限，爬升时由 $w_{\mathrm{cl}}=\sigma((h_{\mathrm{err}}-h_{\mathrm{cl}})/3)$ 收紧：
\begin{equation}
\phi_{\max}^{\mathrm{aero}}=\min\big(\arccos\,\mathrm{clip}((v_{\mathrm{stall}}/V)^2,0,1),\ \phi_{\max}\big),\quad
\phi_{\lim}=\phi_{\max}^{\mathrm{aero}}+w_{\mathrm{cl}}\big(\min(\phi_{\max}^{\mathrm{aero}},\phi_{\mathrm{cl}})-\phi_{\max}^{\mathrm{aero}}\big). \tag{3}
\end{equation}

**终端纯追踪混合与出界护盾**：
\begin{equation}
\phi_{pp}=-k_{pp}\eta,\quad
w_{pp}=\sigma((d_{pp}-d_3)/3)\sigma((|\eta|-\eta_{pp})/(-0.1))(1-w_{\mathrm{loit}}),\quad
\phi^{\mathrm{cmd}}=\mathrm{clip}(w_{pp}\phi_{pp}+(1-w_{pp})\phi_{L1},-\phi_{\lim},\phi_{\lim}). \tag{4}
\end{equation}
超过 $\rho_{\mathrm{sh}}=72$ m 时边界护盾混入朝家方位比例指令，覆盖爬升收紧后的滚转限（仅受气动限钳制）：径向内向速度 $\dot\rho^-=-(p^\top v)/\rho_n$、家方位角 $\eta_o=\mathrm{wrap}(\lambda_o-\chi)$，
\begin{equation}
w_{\mathrm{sh}}=\sigma((\rho_n-72)/4)\sigma((\dot\rho^- -0.5)/1),\quad
\phi^{\mathrm{cmd}}\leftarrow(1-w_{\mathrm{sh}})\phi^{\mathrm{cmd}}+w_{\mathrm{sh}}(-k_{\mathrm{sh}}\eta_o). \tag{5}
\end{equation}
（源码注释称门控为"接近边界且向外运动"；实现中第二因子实际检验径向**内向**速度与 0.5 m/s。）滚转内环为滚转角 PD + 速率阻尼：$a_0=k_\phi\mathrm{wrap}(\phi^{\mathrm{cmd}}-\phi)-k_{d,\phi}\omega_{b,x}$。

**速度计划**：参考随航迹误差、距离低于 $d_{\mathrm{slow}}$ 与爬升门控航段而减速：
\begin{equation}
v_\eta=\mathrm{clip}(v_{\mathrm{ref}}-k_\eta|\eta|,v_{\min},v_{\mathrm{ref}}),\quad
v_d=\mathrm{clip}(v_{\min}+k_d(d_3-d_{\mathrm{slow}}),v_{\min},v_{\mathrm{ref}}),\quad
v_{\mathrm{cl}}=v_{\mathrm{ref}}+(v_{\mathrm{cl,ref}}-v_{\mathrm{ref}})w_{\mathrm{cl}}, \tag{6}
\end{equation}
\begin{equation}
v_{\mathrm{eff}}=\min(v_\eta,v_d,v_{\mathrm{cl}}),\qquad
\delta_v=\frac{v_{\mathrm{eff}}^2-V^2}{2g}. \tag{7}
\end{equation}

**TECS 能量分配**：比总能量与能量平衡误差
\begin{equation}
\delta_E=\mathrm{clip}(h_{\mathrm{err}}+\delta_v,-25,25),\qquad
\delta_B=h_{\mathrm{err}}-\delta_v. \tag{8}
\end{equation}
油门（$\delta_E$ 上 P+I）带盈余门控抗饱和放气 $\dot I_{\mathrm{aw}}=k_{\mathrm{aw}}(V-v_{\mathrm{eff}})^{+}\mathbb 1\{\delta_E<0\}$：
\begin{equation}
I\leftarrow\mathrm{clip}(I+k_{E,i}\delta_E\Delta t-\dot I_{\mathrm{aw}}\Delta t,0,I_{\max}),\quad
\theta=\mathrm{clip}(\theta_{\mathrm{trim}}+k_E\delta_E+I,0,1). \tag{9}
\end{equation}
legacy 上 $I_{\max}=0.2$（天花板 $0.35+0.2+0.2=0.75$）。realistic 预设的定距桨
前进比损失使该天花板在 14–16 m/s 时不足以提供爬升功率，因此该预设下控制器把
$I_{\max}$ 提到 $0.45$、可到满油门；legacy 表不变（全局提高实测 79.7%，低于已发布
的 91%，主要由越界导致）。
俯仰指令（$\delta_B$ 上 P）带倾斜飞行迎角前馈、欠速能量保护与超速爬升权衡：
\begin{equation}
\theta^{\mathrm{cmd}}=\theta_{\mathrm{trim}}-k_B\delta_B
-\Big(\frac{1}{\max(\cos\phi,0.2)}-1\Big)\alpha_{\mathrm{trim}}
+k_{\mathrm{sp}}(v_{\mathrm{eff}}-V)^{+}
-k_{\mathrm{up}}(V-v_{\mathrm{eff}})^{+}w_{\mathrm{cl}}, \tag{10}
\end{equation}
钳位到 $[-\theta_{\mathrm{up}}(V),\theta_{\mathrm{down}}]$（负俯仰=抬头），带速度调度抬头权限 $\theta_{\mathrm{up}}(V)=\theta_{\mathrm{up}}^{\max}\mathrm{clip}((V-v_{lo})/(v_{hi}-v_{lo}),0,1)$。

**俯仰内环（PID + 协调转弯前馈）**：速率阻尼只作用于相对协调转弯俯仰率 $q_{\mathrm{coord}}=\omega_{b,z}\tan\phi$ 的偏差（纯偏航运动学恒等式）：
\begin{equation}
I_\theta\leftarrow\mathrm{clip}(I_\theta+k_{\theta,i}(\theta^{\mathrm{cmd}}-\theta)\Delta t,-0.35,0.35),\quad
a_1=k_\theta(\theta^{\mathrm{cmd}}-\theta)-k_{d,\theta}(\omega_{b,y}-q_{\mathrm{coord}})+I_\theta. \tag{11}
\end{equation}

**动作组装**：$a=(a_0,a_1,0,2\theta-1)$，裁剪到 $[-1,1]^4$。

## 4. 参数

全部取自 `algorithms/baselines/fixedwing_autopilot.json`（配置名 `l1_tecs`）。

| 组 | 参数 | 值 |
|---|---|---|
| 横向 | `l1_distance` / `l1_min_distance`；`l1_dist_gain` / `l1_dist_offset` | 40.0 m / 12.0 m；0.6 / 8.0 m |
| 横向 | `max_bank_deg`；`v_stall_guard`；`k_roll` / `kd_roll` | 60°；11.5 m/s；3.0 / 0.5 |
| 纵向 | `v_ref` / `v_turn_min`；`k_speed_bleed` / `k_dist_slow` / `d_slow_ref` | 20.0 / 14.0 m/s；2.5 / 0.35 / 10.0 m |
| 纵向 | `k_energy` / `k_energy_i` / `integral_max` / `throttle_trim`；`k_aw_speed` | 0.008 / 0.005 / 0.2（realistic 下 0.45）/ 0.35；0.012 |
| 纵向 | `k_balance` / `k_speed_keep` / `k_geo_up` | 0.008 / 0.05 / 0.6 |
| 纵向 | `k_pp` / `d_pp_entry` / `eta_pp_gate_deg` | 1.2 / 35.0 m / 40° |
| 纵向 | `bank_climb_deg` / `v_climb_ref` / `h_climb_gate` | 45° / 14.0 m/s / 5.0 m |
| 纵向 | `pn_shield` / `v_shield_min` / `k_shield` | 72.0 m / 0.5 m/s / 2.0 |
| 纵向 | `h_loiter_gate` / `pn_loiter` | $10^4$ m（禁用）/ 35.0 m |
| 纵向 | `pitch_trim_deg` / `alpha_trim_deg`；`pitch_down_max_deg` | −2°（抬头）/ 2.5°；12° |
| 纵向 | `pitch_up_max_deg` / `pitch_up_v_min` / `pitch_up_v_full` | 16° / 11.0 / 15.0 m/s |
| 纵向 | `k_pitch` / `kd_pitch` / `k_pitch_i` | 4.0 / 0.45 / 2.0 |

## 5. 实测表现

| 协议 | 来源 | 结果 |
|---|---|---|
| 冻结协议调参快照（harness `environments/fixedwing/eval_fixedwing_autopilot.py`：1024 env、3700 步窗口 > 3600 步时域；训练 seed 7，dev seeds 1000–1003，test seeds 2000–2003） | 2026-10-22 记录，在 21.3% → 74.9% 调参迭代之后；**非发布产物** | 约 74.6% 训练 / 74.4% dev / 73.1% test |
| 发布配置的调参记录（2026-10-22 扫描中 `pitch_up_max_deg` 10 → 16） | `fixedwing_autopilot.json` 注释 | 出界 23.9% → 1.8%，坠毁率 0.0；18° 实测 94.9% vs 16° 94.7%（保留 16° 以留失速裕度） |
| 协议数据集 A（128 副本 × seeds 7/13/21/101，3600 步窗口） | `results/baseline_protocol/seed*/fixedwing/baseline.json` | seeds 89.8 / 87.5 / 83.6 / 90.6%；384 回合均值 87.0±3.2% |
| 数据集 B，128 副本 × seeds 7/13/21/101，完整 3616 步窗口 | `results/fixedwing/baselines/l1_tecs_seed*.json` | 均值 87.0±3.2%，平均回报 208.84，`complete=true`；出界 2.2% |
| 已发布单记录 | `results/fixedwing/baseline.json`（128 副本、seed 13、3616 步窗口） | 87.5%（112/128 首回合），平均回报 209.9，`complete=true` |

2026-11-12 复测对每个单元使用完整时域，旧的 1500 步右删失不再适用；数据集 A/B 现在共享同一副本数与窗口。

## 6. 已知局限

- 残余失败模式（1.8%）：极高目标仍会在爬升中超时。冻结协议把残余记为结构限制——出界终局极限环（常数 $L_1$ 在 10 m 捕获球外 $r\approx L_1/(2\sin\eta)$ 维持稳定轨道，促成式 (2)）。
- 机体功率受限（最大推力 18 N < 重力 23 N）：持续大坡度下沉 0.5–1 m/s，故有滚转门 (3)；油门侧超速调速器尝试后弃用（低于配平的输出上限会破坏积分器的配平估计；能量误差钳位禁止在计划空速下给爬升功率）。
- 爬升段包容（$w_{\mathrm{loit}}$）在训练集测得净负后禁用（`h_loiter_gate` $=10^4$）；距离调度减速与 $k_{\mathrm{up}}\in\{0.8,1.0\}$ 同样弃用。参数仅在训练/开发初始条件上调（训练 seed 7，dev seeds 1000–1003）。
- 通道 $a_1$ 以反号驱动平尾与垂尾（方向舵耦合当扰动）；$a_2$ 上游设计为死通道，恒为 0。plant 支持风场（`v_air=v-wind`，`steady_wind`/`dryden` 构造参数）且训练入口默认 30% Dryden，但本控制器无空速传感器、以地速作空速，故其发布基线记录为静风运行。

## 7. 实现

- 控制器：`FixedwingAutopilotBaseline`（`name='l1_tecs'`）位于 `algorithms/baselines/baselines.py`；参数 `algorithms/baselines/fixedwing_autopilot.json`；经 `build_baseline('fixedwing', env)` 实例化，在 `environments/fixedwing/env.py` 声明为 `conventional_controller`。
- 评测：`scripts/baseline_eval.py fixedwing`（产物 `results/fixedwing/baseline.json`）；数据集 B 产物 `results/fixedwing/baselines/l1_tecs_seed7.json`；场景网格/消融 harness `environments/fixedwing/eval_fixedwing_autopilot.py`（写 `results/fixedwing/autopilot_report.json`，未发布）。
- 环境参考：`docs/method/env-fixedwing-waypoints.zh-CN.md`（舵面混合、作动滞后、发动机模型、捕获与出界几何）。

## 8. 参考

- S. Park, J. Deyst, J. P. How, "A New Nonlinear Guidance Logic for Trajectory Tracking," *AIAA GNC*, 2004 —— L1 横向加速度律 (2)。
- A. A. Lambregts, 总能量控制（TECS），1983，取自源码引用的简化形式（`fixedwing_autopilot.json`）：能量误差 → 油门，能量平衡误差 → 俯仰 (8)–(10)。
- Tai, J. K. et al., *PyFlyt*, v0.29.0 —— 上游固定翼模型、舵面混合与作动约定。
