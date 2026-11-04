# Fixedwing 基线：常规级联 PID 自驾（`FixedwingPIDBaseline`）

**简体中文** · [English](baseline-fixedwing-pid.md)

## 1. 概述

`FixedwingPIDBaseline` 是为 Fixedwing-Waypoints 任务新增的第三个传统参考（20–65 m 球壳内 4 个随机航路点、20 m 高度地板、10 m 捕获半径、100 m 出界球、120 s / 3600 步、静风确定性评测）。它刻意**不是**制导启发式：与冻结的 L1+TECS 基线不同，它无距离调度、无终端纯追踪混合、无出界护盾、无能量优先逻辑。

实现的级联为

$$
\text{方位}\to\chi_{\mathrm{ref}}\to\phi_{\mathrm{ref}}\to\dot\phi_{\mathrm{ref}}\to\delta_a,\qquad
h_{\mathrm{err}}\to\theta_{\mathrm{ref}}\to\dot\theta_{\mathrm{ref}}\to\delta_e,\qquad
V_{\mathrm{err}}\to\delta_t,
$$

即航向保持（P/I）→ 滚转角 → 滚转率 → 副翼；高度保持（P/I）→ 俯仰角 → 俯仰率 → 升降舵；空速（P/I）→ 油门。所有增益恒定；控制器唯一状态是五个限幅积分器，每回合由 `reset_idx` 清零。

## 2. 接口约定（已对照 plant 核实）

- **观测组**：只读公开 `'policy'` 张量：$\omega_b$ `[0:3]`、四元数 xyzw `[3:7]`、$v_b$ `[7:10]`、$p$ `[10:13]`、到 4 个航路点槽的体系偏差 $\Delta_k$ `[23:35]`。活动航路点重建为 $w=p+R\Delta_0$；不触碰特权目标或风场。
- **空速 vs 地速**：观测无空速管/真空速通道。速度环用 $V=\lVert v_b\rVert$，即**地速**（体系速度范数）。静风（冻结评测条件）下等于真空速。启用可选 Dryden 扰动时它仍是地速，且刻意**不是**内部相对空气真值 $v_{\text{air}}=v-w$：控制器从不读风场。航向同样为地面航迹 $\mathrm{atan2}(v_y,v_x)$，水平速度低于 `v_min_valid` = 2 m/s 时回退到机头航向 $\psi$。
- **舵面符号**（x 前、y 左、z 上）：正滚转 = 右倾，正俯仰 = **低头**。故左转需负滚转参考，爬升需负俯仰参考。
- **动作通道** $a\in[-1,1]^4$：$a_0$ 差动副翼（$a_0>0$ 右滚）；$a_1$ 平尾（$a_1>0$ 低头）加反号垂尾耦合（视为扰动）；$a_2$ 为上游死主翼通道（偏度限零），**恒为 0**；$a_3$ 由环境重映射 $\delta_t=(a_3+1)/2$。
- **数值**：float64 于 `env.device`、跨副本批量；返回 float32。控制周期 $\Delta t=1/30$ s。

## 3. 控制律

$\mathrm{wrap}(\cdot)$ 到 $(-\pi,\pi]$；$\phi,\theta$ 为 $R$ 的 ZYX 分解滚/俯角；$r=\omega_{b,z}$；$\mathrm{clip}(x,l,u)=\min(\max(x,l),u)$。

**横向：航向 → 滚转 → 滚转率 → 副翼**

$$
\lambda=\mathrm{atan2}(w_y-p_y,w_x-p_x),\quad
\chi=\begin{cases}\mathrm{atan2}(v_y,v_x)&\lVert v_{xy}\rVert>2\\ \psi&\text{否则}\end{cases},\quad
e_\chi=\mathrm{wrap}(\lambda-\chi)
$$

$$
I_\chi\leftarrow\mathrm{clip}(I_\chi+k_{i\chi}e_\chi\Delta t,\pm I_{\chi,\max}),\quad
u_\chi=k_\chi e_\chi+I_\chi,\quad
\phi_{\mathrm{ref}}=\mathrm{clip}(-u_\chi,\pm\phi_{\max})
$$

$$
e_\phi=\mathrm{wrap}(\phi_{\mathrm{ref}}-\phi),\quad
\dot\phi_{\mathrm{ref}}=\mathrm{clip}(k_\phi e_\phi,\pm\dot\phi_{\max}),\quad
e_{\dot\phi}=\dot\phi_{\mathrm{ref}}-\omega_{b,x}
$$

$$
I_{\dot\phi}\leftarrow\mathrm{clip}(I_{\dot\phi}+k_{i\dot\phi}e_{\dot\phi}\Delta t,\pm I_{\dot\phi,\max}),\quad
a_0=\mathrm{clip}(k_{\dot\phi}e_{\dot\phi}+I_{\dot\phi},-1,1)
$$

**纵向：高度 → 俯仰 → 俯仰率 → 升降舵**

$$
h_{\mathrm{err}}=w_z-p_z,\quad
I_h\leftarrow\mathrm{clip}(I_h+k_{ih}h_{\mathrm{err}}\Delta t,\pm I_{h,\max}),\quad
u_h=k_h h_{\mathrm{err}}+I_h
$$

$$
\theta_{\mathrm{ref}}=\mathrm{clip}(\theta_{\mathrm{trim}}-u_h,-\theta_{\mathrm{up}},\theta_{\mathrm{down}}),\quad
e_\theta=\theta_{\mathrm{ref}}-\theta,\quad
\dot\theta_{\mathrm{ref}}=\mathrm{clip}(k_\theta e_\theta,\pm\dot\theta_{\max})
$$

$$
q_{\mathrm{coord}}=\omega_{b,z}\tan\phi,\qquad
e_{\dot\theta}=\dot\theta_{\mathrm{ref}}-(\omega_{b,y}-q_{\mathrm{coord}})
$$

$$
I_{\dot\theta}\leftarrow\mathrm{clip}(I_{\dot\theta}+k_{i\dot\theta}e_{\dot\theta}\Delta t,\pm I_{\dot\theta,\max}),\quad
a_1=\mathrm{clip}(k_{\dot\theta}e_{\dot\theta}+I_{\dot\theta},-1,1)
$$

$q_{\mathrm{coord}}=\omega_{b,z}\tan\phi$ 是倾斜转弯中体俯仰率与 Euler 俯仰率的纯运动学恒等式；减去它可防止内环把协调转弯当作俯仰扰动（否则持续倾斜时飞机会下沉并最终触地）。

**速度 → 油门**

$$
e_V=V_{\mathrm{ref}}-V,\quad
I_V\leftarrow\mathrm{clip}(I_V+k_{iV}e_V\Delta t,\pm I_{V,\max}),\quad
\delta_t=\mathrm{clip}(\delta_{\mathrm{trim}}+k_V e_V+I_V,\delta_{\min},\delta_{\max})
$$

**动作**：$a=(a_0,a_1,0,2\delta_t-1)$，裁剪到 $[-1,1]^4$。

**抗饱和与限幅**：每个环使用**条件积分**：当环输出饱和且误差仍推向饱和（`|raw| > limit && raw*error > 0`）时冻结积分器，另有硬积分钳位。参考受 $\phi_{\max}$、$\dot\phi_{\max}$、$\theta_{\mathrm{up/down}}$、$\dot\theta_{\max}$ 限制，舵面 $[-1,1]$，油门 $[\delta_{\min},\delta_{\max}]$。无失速保护、自适应调度、路径规划、NMPC 或学习补偿。

## 4. 参数

`algorithms/baselines/fixedwing_pid.json`（配置名 `pid_cascade`）。

| 组 | 参数 | 值 | 含义 |
|---|---|---|---|
| 横向 | `k_course` / `k_i_course` / `i_course_max_deg` | 0.9 / 0.05 / 7° | 航向误差 → 滚转参考 |
| 横向 | `max_bank_deg` | 50° | 滚转参考钳位 |
| 横向 | `k_roll_angle` / `max_roll_rate_deg_s` | 2.0 / 50°/s | 滚转角 → 滚转率参考 |
| 横向 | `k_roll_rate` / `k_i_roll_rate` / `i_roll_rate_max` | 0.40 / 0.25 / 0.30 | 滚转率 → 副翼 |
| 纵向 | `k_altitude` / `k_i_altitude` / `i_altitude_max_deg` | 0.025 / 0.0012 / 10° | 高度误差 → 俯仰参考 |
| 纵向 | `pitch_trim_deg` | −2°（抬头） | 配平俯仰参考 |
| 纵向 | `pitch_up_max_deg` / `pitch_down_max_deg` | 18° / 12° | 俯仰参考权限 |
| 纵向 | `k_pitch_angle` / `max_pitch_rate_deg_s` | 2.0 / 50°/s | 俯仰角 → 俯仰率参考 |
| 纵向 | `k_pitch_rate` / `k_i_pitch_rate` / `i_pitch_rate_max` | 0.80 / 0.50 / 0.90 | 俯仰率 → 升降舵 |
| 速度 | `v_ref` / `v_min_valid` | 16.0 / 2.0 m/s | 空速参考 / 航迹跟踪有效下限 |
| 速度 | `k_speed` / `k_i_speed` / `i_speed_max` | 0.12 / 0.04 / 0.30 | 速度误差 → 油门 |
| 速度 | `throttle_trim` / `throttle_min` / `throttle_max` | 0.30 / 0.15 / 1.0 | 油门偏置与钳位 |

## 5. 调参流程与场景分离

- **调参场景**（`scenarios` 模式）：注入直线、左转、右转、爬升、下降与恒空速航线，在调参种子 7 上评估；从配平巡航状态起步（20 m/s 出生覆盖为 16 m/s），航线各段长于一最小转弯半径（约 22 m）。
- **正式比较**（`benchmark` 模式）：留出初始条件 seeds **2000–2003**、1024 副本、静风、确定性动作——与 `scripts/eval.py` 相同记账（窗口内全部回合完成，每副本保证一个完整回合）。

仅做小范围扫描（其余增益固定）：

1. 滚转限 55°/60° 与滚转率 90°/s 弃用：机体（最大推力 18 N < 重力 23 N）在持续 55–60° 倾斜时无法保持高度；14 m/s 变体坠毁。
2. 16 m/s 下 45° vs 50°：50° 每回合捕获更多航路点且零坠毁。
3. 俯仰率积分限从 0.35 提到 0.90：45–50° 转弯需要持续升降舵配平，0.35 无法提供，导致约 2.7 m/s 下沉触地。
4. 协调转弯俯仰率补偿 `q_coord` 在观察到每次持续倾斜都稳定掉高后加入。

## 6. 实测表现

**已发布记录（当前检出）**：数据集 B（`results/fixedwing/baselines/*.json`，256 副本 / seed 7 / 1500 步窗口）与数据集 A（`results/baseline_protocol/seed*/fixedwing/baseline.json`，128 副本 × seeds 7/13/21/101 / 3600 步窗口）：

| 控制器 | 数据集 B 成功（完成首回合） | 数据集 B 出界 | 数据集 A 均值（4 seeds） |
|---|---|---|---|
| L1+TECS（`l1_tecs`） | 98.6%（216 完成 / 256） | 2.5% | 87.0±3.2% |
| PID 级联（`pid_cascade`） | 91.6%（179 / 256） | 10.1% | 不在协议数据集 A |

数据集 B 的 1500 步窗口对回合集做右删失，故分母为完成的首回合数而非 256；数据集 A 用完整 3600 步时域，是跨方法参考。

**留出比较（调参时 harness，未发布）**：harness `environments/fixedwing/eval_fixedwing_pid.py benchmark`，每次 1024 副本、静风、确定性动作、seeds 2000–2003（调参均未见）；窗口内全部完成的回合均计入（scripts/eval.py 记账）。PID 增益仅在场景航线 + seed 7 上调，故下表对 PID 臂是样本外。harness 写 `results/fixedwing_pid/comparison.json`，**未发布**。

| 控制器 | 成功 | 航路点/回合 | 任务时间 (s) | 坠毁 | 出界 | 超时 |
|---|---|---|---|---|---|---|
| L1+TECS（冻结） | 93.5% [93.0, 94.3] | 3.82 | 28.4 | 0.0% | 3.0% | 3.6% |
| **PID 级联（本臂）** | 80.4% [79.4, 81.3] | 3.54 | 35.0 | 0.0% | 10.8% | 8.8% |
| PPO（1024×1000，seed 7） | 待训练完成 | | | | | |

原始逐 seed 数据（未发布）：`results/fixedwing_pid/comparison.json`；轨迹（seed 2000，8 副本各首回合）在 `trajectories_<controller>_seed2000.{json,png}`。

与 L1+TECS 差距（−13.1 pp）的解读：PID 臂几乎只通过 100 m 球附近的出界极限环（10.8% vs 3.0%）与超时（8.8% vs 3.6%）丢回合——其坠毁率 0.0%，与 L1+TECS 相同。这是 §7 的预期结构后果：纯航向保持环配固定滚转钳位无法总在出界球前拉停外冲，而 L1 的距离相关追踪项在目标接近时收紧转弯。任务时间约长 6.6 s（35.0 vs 28.4 s），原因相同——转弯更宽更晚。

场景（调参切分，每场景单回合，seed 7，静风；完整数据在 `results/fixedwing_pid/scenarios_pid.json` / `scenarios_l1_tecs.json`，均未发布）：

| 场景 | PID 判定 | PID 亮点 | L1+TECS 亮点 |
|---|---|---|---|
| 直线 | 通过 | 4.4 s，高度 RMS 2.3 m，航向 RMS 0.2°，V 15.8 | 4.3 s，高度 RMS 1.3 m |
| 左转 | 通过 | 8.6 s，倾斜 −53°，最低高度 11.2 m | 7.1 s，倾斜 −61°，最低 13.4 m |
| 右转 | 通过 | 8.2 s，倾斜 +51°，最低 16.7 m | 7.1 s，倾斜 +58°，最低 15.2 m |
| 爬升 | 通过 | 4.7 s，净 +4.1 m，俯仰 −13° | 4.4 s，净 +4.0 m |
| 下降 | 通过 | 4.2 s，净 −8.1 m，俯仰 +6.8° | 4.1 s，净 −6.2 m |
| 速度 14/18/22 | 通过 | RMS 内跟踪 0.6/0.2/0.2 m/s | 这些场景速度由环境固定 |

转弯场景显示纯航向保持环的已知结构代价：飞机先在捕获球外绕一圈再穿下一个航路点（整场景航向 RMS 约 73°），而 L1 的追踪项会切角。两控制器分别通过 5/5（L1）与 8/8（PID，含三个空速设定点）。

## 7. 已知局限

- **无距离调度**：纯航向保持环无法切入航路点内圈；连续航路点间距小于一个转弯半径时，可能在捕获球外形成稳定轨道（左/右转场景需多绕一圈才能捕获）。随机 20–65 m 任务分布对此容忍，但这是主要残余失败模式。
- **功率受限机体**：16 m/s 下持续 50° 倾斜掉高约 0.5–1 m/s；高度环只能在 12°/18° 俯仰权限内以速度换高度，故大转弯有 5–6 m 下沉。
- **高度积分较慢**（`k_i_altitude` = 0.0012）使平飞存在约 0.5 m/s 的小下垂，直到俯仰率积分建立所需配平；这是刻意的——更快高度积分会在调参场景引起俯仰/高度振荡。
- **无失速保护、NMPC、学习补偿或自适应**（设计如此）；控制器也忽略 L1/TECS 启发式（无终端追踪、无出界护盾），故 100 m 边界附近仍可能出界。
- 通道 $a_2$ 上游为死通道且恒为零；控制器不假设其有任何权限。

## 8. 实现与运行

- 控制器：`FixedwingPIDBaseline`（`name='pid_cascade'`）位于 `algorithms/baselines/baselines.py`，注册为 `BASELINE_VARIANTS['fixedwing']['pid_cascade']`；参数 `algorithms/baselines/fixedwing_pid.json`。以 `build_baseline('fixedwing', env, 'pid_cascade')` 实例化。
- Harness：`environments/fixedwing/eval_fixedwing_pid.py`（`scenarios` / `benchmark` / `all`）。

```bash
# 1. 调参 / 健全性场景（直线、左/右转、爬升、下降、空速）
python environments/fixedwing/eval_fixedwing_pid.py scenarios --out results/fixedwing_pid

# 2. 正式留出比较（PID + L1+TECS + 已发布 PPO 如存在）
python environments/fixedwing/eval_fixedwing_pid.py benchmark \
    --num-envs 1024 --seeds 2000 2001 2002 2003 \
    --controllers l1_tecs pid_cascade ppo --out results/fixedwing_pid

# 3. 经共享基线 harness 的单基线检查
python scripts/baseline_eval.py fixedwing --baseline pid_cascade --seed 101
```

产物（`results/fixedwing_pid/` 下，均未发布）：`scenarios_pid.json`、`scenarios_l1_tecs.json`、`comparison.json`、`comparison.md`，以及逐控制器 `trajectories_<controller>_seed<k>.json` 与俯视 PNG（x–y，颜色=高度）。

学习臂（PPO）从 `results/fixedwing_ppo/checkpoint.pt`（未发布）加载训练 `config.json`；缺失 checkpoint 时报告并跳过，而非静默替换。

## 9. 参考

- G. F. Franklin, J. D. Powell, A. Emami-Naeini, *Feedback Control of Dynamic Systems* —— 级联 / 抗饱和结构（条件积分、参考与舵面钳位）。
- T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control* —— 倾斜飞行协调转弯运动学恒等式 $q_{\mathrm{coord}}=r\tan\phi$。
- J. K. Tai et al., *PyFlyt*, v0.29.0 —— 上游固定翼 plant、舵面混合与动作映射（$a_2$ 死通道）。
