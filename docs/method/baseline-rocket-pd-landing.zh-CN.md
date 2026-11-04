# Rocket-Landing 基线：常规 PD 着陆（`RocketPDLandingBaseline`）

**简体中文** · [English](baseline-rocket-pd-landing.md)

## 1. 概述

火箭着陆任务的传统基线是刻意简化的常规级联 PD 着陆律：位置/速度 PD 生成期望加速度，期望加速度转为期望推力大小与推力轴，姿态/速率 PD 生成力矩需求，由既有 TVC + 栅格翼控制分配接口跟踪。无优化器、无学习模块、无制导求解：垂直剖面为经典制动距离/阈值高度律（`v_ref(h) = -min(V_MAX, K_H * h)`），发动机为该剖面附近的固定速度带滞环，点火为一次性制动距离边沿触发。全部常量固定在类内；调参只在 `environments/rocket/rocket_pd_tune.py` 的专用场景中做过一次（第 5 节），从不逐初态手工调。

它刻意**不是** PDG 基线：无 ZEM/ZEV 最优制导求解、无气动前馈、无鳍/万向节力分配。该臂的意义是为学习策略以及相同条件下的 PDG 提供朴素 PD 标尺。

## 2. 接口约定

- **输入**：仅 `RocketLandingEnv` 的 `'policy'` 观测组（30 维，float64）。读取通道：体速率 `wb` `[0:3]`、姿态四元数 xyzw `[3:7]`、体速度 `vb` `[7:10]`、世界位置 `p` `[10:13]`、发动机状态 `ignited` `[24]`、燃料 `[25]`、油门状态 `[26]`、万向节状态 `[27:29]`、平台接触标志 `[29]`。世界速度由 `R(q) @ vb` 用 `_quat_to_R`（xyzw，标量在后）重建。
- **输出**：环境 7 维原始动作约定 `[fin_x, fin_y, fin_yaw, ignition, throttle, gimbal_x, gimbal_y]`，Box `[-1,-1,-1,0,0,-1,-1]..[1]`（环境把原始 $[-1,1]$ 仿射映射到该 Box；点火/油门留在 $[0,1]$）。
- **内部状态**：每副本一个布尔 `started`（一次性制动点火已触发），回合 reset 时清零。

## 3. 控制律

每策略步（60 Hz，每步 2 个内环控制步）：

1. **几何**：`h = (p_z - REST_Z).clamp_min(0)`，`v_w = R @ vb`，下降速度 `v_dn = max(0, -v_z)`，平台偏移 `p_xy`。
2. **横向（下降中段）**：`a_xy = -KP_XY * p_xy - KD_XY * v_xy`，饱和到 `lat_cap = clamp(0.14 * max(f_z, g), 1, 5)` m/s²（倾斜预算：最小推力流形下 0.14 rad 约 8°，在约 9.6° 鳍失速拐点内）。低于终端高度 `H_TERM = 2 m` 时去掉位置项（最小推力下追位置会导致增长—反向的腿滑失败），改为纯速度阻尼 `a_xy = clamp(-KV_LAT * v_xy, ±A_LAT_PULSE)`。
3. **垂直**：`v_ref_dn = min(V_MAX, K_H * h)`（0.5×h 剖面，上限 20 m/s），`a_z = KV_Z * (v_ref_dn - v_dn)`，比力 `f_z = a_z + g`。
4. **发动机带**：一次性边沿触发 `first = (h <= h_brake + M_ON) & (v_dn > V_BAND) & ~started`，制动距离 `h_brake = v_dn²/(2*a_brake)`、`a_brake = BRAKE_FRAC * T_MAX / m - g`，`M_ON = 15 m` 裕度使点火时保留 5% 推力余量。运行中由该剖面附近固定速度带滞环维持：低于 `v_ref_dn - V_BAND` 熄火（终端带 `V_TERM_OFF/ON` 在 `H_TERM` 下），高于 `v_ref_dn + V_BAND` 复燃（终端 `V_TERM_ON`）。公开平台接触标志即平台上的发动机关（`ign &= ~pad_c`），且**无触地后永久熄火锁存**：速度带在稳定阶段保持活跃，故回弹在下落途中被接住（速度带**就是**回弹捕获器）。上升时从不点火（`v_dn = max(0, -v_z)`）——助推而非接住。
5. **推力矢量**：`sigma = (a_xy, f_z)`，`T = clamp(m|sigma|, T_MIN, T_MAX)`，油门按既有约定 `u_thr = (T/T_MAX - 0.39)/0.61`。
6. **姿态参考与力矩**：期望推力轴由**点火时发动机产生的比推力矢量**构造，`s = clamp(|sigma|, T_MIN/m, T_MAX/m)`，垂直分量 `f_vert = sqrt(s² - |a_xy|²)`，即 `sin(tilt) = a_xy / s`。力矩需求 `tau = KP_ATT * e_tilt - KD_ATT * w_xy`（另加 `KD_YAW` 偏航速率阻尼），经 plant 既有固定增益 TVC/栅格翼接口分配（`gx = -tau_gim/k_gim`、`fx/fy = -tau_fin/k_fin`，各通道饱和于 0.95）。参考跨滑行/点火过渡保持（下次点火用同一钳位），故脉冲串从不追逐它无法跟踪的直立设定点。

第 6 步的参考是关键细节：直接引用 `f_z`（`atan(a_xy/f_z)`）会让最小推力钳位把横向环增益放大 `s/f_z` 倍——脉冲段约 2 倍过增益（`s ~ 20.5` vs `f_z ~ 9.8` 比力），每次复燃持有最多 8.7° 的**陈旧倾斜**。倾斜的最小推力脉冲随后让腿在平台上滑动，形成增长—反向极限环（0% 成功）。用 `sin(tilt) = a_xy/s` 后，单位 plant 增益下交付横向加速度处处等于指令。

## 4. 试验排除的方案（记录在案以免重蹈）

- **触地后永久熄火锁存**：首次平台接触后硬 `ign &= ~touched` 能让轻触着陆（中位约 0.46 m/s），但放任接触求解器自激微弹跳增长（±0.1 → +0.79/−0.85 m/s），直到火箭被弹出并以 >1 m/s 再接触（致命）。活跃速度带（下落途中 `V_TERM_ON` 复燃）改为接住回弹；PDG 在同一 plant 上 85.5% 的成功率证明弹出并非硬上限。
- **硬熄火高度**：`H_CUT = 0.05 m` 关机（"不要推穿接触"）是灾难性的（成功 0.008）：越过该点火箭从约 0.05 m 自由落体，以约 0.96 m/s 再接触。`T_MIN` 燃穿接触步是**缓冲**而非障碍。
- **连续距离条件**：把制动距离条件作为**保持**条件（点火期间 `h <= h_need + M_ON`）会与终端带冲突（其熄火阈值高于复燃阈值 → 一开一关抖动、终端爬行）。它必须是边沿触发；其后一切归速度带。
- **脉冲段内追位置**：`T_MIN` > 重力时发动机只能制动；最小推力下的位置误差会产生增长—反向极限环与腿滑（正是 `H_TERM` 以下改为纯速度阻尼的动机，与 PDG 笔记一致）。

## 5. 调参场景与记录

`environments/rocket/rocket_pd_tune.py`（报告 `results/rocket_pd_tune/tune_report.json`，**未发布**）：每候选 128 副本 × 一回合（1900 步安全上限，覆盖环境 1200 步 30 s 时域），静风、调参 seed 101（区别于官方 seed 7），围绕已发布常量做单因子扫描并加组合最优确认。排序：成功率、然后更少的 >1 m/s 触地、更少的 >0.35 rad/s 自旋、更低中位触地速度。以下为诊断运行保留的增益扫描记录（64-env 调参 harness，seed 101）：

| 变更 | 成功 | 说明 |
|---|---|---|
| 倾斜参考 = `f_z`（修复前） | 0.000 | 2 倍横向过增益，腿滑极限环 |
| 倾斜参考 = `a_xy/s`（修复） | 0.578 | 恢复单位横向增益 |
| + `KD_ATT` 6000 | 0.500 | 无气动前馈：环路需要更强阻尼 |
| + `KD_ATT` 10000 | 0.453 | 过冲，一如 PDG 笔记 |

完整网格结果与移植后的优胜者在报告 JSON 中。网格结论（调参 seed，每项 128 副本）：已发布常量接近最优；除 `A_LAT_PULSE` 1.5 → 1.0（0.484 → 0.531，官方确认后采用）外，每个单旋钮变动均为中性或更差。叠加的组合最优候选崩坏（成功 0.008，80% 回合触地自旋 >0.35）——终端带、脉冲与姿态增益相互破坏，这正是确认运行要捕获的。最终发布常量为第 3 节引用的类属性。

## 6. 测试记录

已发布官方协议记录（`results/rocket/baselines/*_seed7.json`，256 副本 / seed 7 / 1200 步，静风，v2 指标 `ever_stable_2s`）：

| 臂 | 成功（`ever_stable_2s`） | 平均回报 | 平台触地率 |
|---|---|---|---|
| PDG（`zem_zev_pdg`） | **100%** | 907.31 | 1.000 |
| PD（`pd_landing`） | **100%** | 22.34 | 1.000 |

两臂也报告 `ever_success = 1.0`、`terminal_stable_2s = 1.0`，且无 `success_then_failure` / `stable_then_failure`。回报差距是奖励形状产物，而非成功差距：v5 走廊项对 PD 臂更快的下降剖面罚得更重（`rew_corridor ≈ −1301k` vs `−1081k`），尽管二者都稳定。更早的短窗口记录（PDG 0.8555 / PD 0.8438，回报 4125.71 / 4063.64）在不同成功指标下产生，已撤回。

已发布的 `A_LAT_PULSE = 1.0` 来自调参网格（调参 seed 上 0.531 vs 0.484），并在官方 harness 上确认（较 1.5 提升 1.6 pp）。

## 7. 实现与注册

- 控制器：`RocketPDLandingBaseline`（注册名 `pd_landing`）位于 `algorithms/baselines/baselines.py`，注册于 `BASELINE_VARIANTS['rocket']['pd_landing']`；经 `build_baseline('rocket', env, variant='pd_landing')` 实例化。
- 环境：`environments/rocket/env.py`，类 `RocketLandingEnv`（默认 `conventional_controller` 为 PDG 臂）。
- 评测：

```bash
python scripts/baseline_eval.py rocket --baseline pd_landing
python environments/rocket/rocket_pd_tune.py
```

## 8. 参考

1. Tai, J. K. et al., *PyFlyt*, v0.29.0 —— 上游火箭任务、机身模型与着陆判据。
2. Açıkmeşe, B. & Ploen, S. R. (2007), "Convex Programming Approach to Powered Descent Guidance for Mars Landing," *J. Guidance, Control, and Dynamics* 30(5) —— 对照的 PDG 基线。
3. Wie, B. *Space Vehicle Dynamics and Control*（AIAA）—— 级联 PD 姿态控制与推力矢量分配。
