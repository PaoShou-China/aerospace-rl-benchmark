# Missile-Intercept 基线：增广比例导引（`MissileAPNBaseline`）

**简体中文** · [English](baseline-missile-apn.md)

## 1. 概述

Missile-Intercept 的传统基线为增广比例导引（APN），经 skid-to-turn 运动学逆映射到环境的体角速度指令接口。该律把比例导引项（导航常数 $N$）与半 $N$ 重力补偿项组合，对实测视线（LOS）角速度低通滤波，并把指令横向加速度转为体角速度指令，由 plant 中 120 Hz rate PID 经尾舵 mixer 跟踪。制导解析且完全设备端批量；唯一内部状态是每副本 LOS 角速度滤波器，回合复位时清零。

交战条件为环境唯一的固定中等工作点（2026-10-24 决定）：初始距离 2500 m、目标航向锥相对导弹方位 $\pm37.5^{\circ}$、目标侧向加速度 7.5 g、蛇形频率 0.75 Hz、目标与导弹速度 $V_T=V_{M0}=250$ m/s、约 4 km 高度、15 m 近炸引信杀伤半径、15 s 回合上限。导弹带火控前置量发射（沿缩放名义剩余飞行时间 $t_{go}=0.8R/(V_{M0}+V_T)$ 的直线外推），故发射即近碰撞航向，制导只修正残余误差。

## 2. 接口约定

- **输入**：仅 `MissileInterceptEnv` 的 `'policy'` 观测组（34 维，float32；内部转 float64）。读取通道：四元数 xyzw `[3:7]`、$v_b$ `[7:10]`、体系目标相对位置 $r_b$ `[23:26]`、体系相对速度 `[26:29]`、体系 LOS 角速度 $\lambda_b$ `[29:32]`、接近速度 $V_c$ `[32]`。
- **输出**：$[-1,1]^3$ 原始动作，环境解释为归一化体角速度指令并仿射映射到 $(\omega_x,\omega_y,\omega_z)$，逐轴限幅 $(\texttt{OMEGA\_MAX},\texttt{OMEGA\_LAT},\texttt{OMEGA\_LAT})=(6.0,1.0,1.0)$ rad/s。plant rate PID 经 skid-to-turn mixer 转为舵指令；无油门通道（助推—滑行发动机自主）。
- **构造**：构造时只从环境模块取速率指令限幅；不读任何内部环境状态缓冲。
- **逐回合状态**：`reset_idx(mask)` 将 $\hat\lambda$ 在掩码副本清零；测试 harness 每步由 `env.reset_buf` 驱动（跳过会跨回合泄漏滤波状态并使首步发散）。

## 3. 制导与控制律

### 3.1 LOS 角速度滤波

观测的体系 LOS 角速度经一阶低通，系数 $\alpha=0.8$，在终端区 $V_c<50$ m/s（LOS 角速度量测噪声爆炸）冻结：

$$
\hat\lambda\leftarrow\hat\lambda+\alpha(\lambda_b-\hat\lambda),\qquad
\alpha=\begin{cases}0.8,&V_c\ge50\ \text{m/s}\\0,&V_c<50\ \text{m/s}\end{cases} \tag{1}
$$

滤波后速率旋到世界系 $\lambda_w=R\hat\lambda$，LOS 单位向量 $\hat r_w=R(r_b/\lVert r_b\rVert)$，$\lVert r_b\rVert\ge10^{-3}$ m。

### 3.2 增广比例导引

指令加速度为 PN 项加半 $N$ 重力补偿在垂直 LOS 平面上的投影，$g=(0,0,-9.81)$ m/s²：

$$
a^{\text{cmd}}=N V_c(\lambda_w\times\hat r_w)+\frac{N}{2}\big(g-(g\cdot\hat r_w)\hat r_w\big). \tag{2}
$$

叉积 $\lambda_w\times\hat r_w$ 是关键：$\lambda_w=(r\times v_{\text{rel}})/\lVert r\rVert^2$ 是 LOS **角速度**（旋转轴），垂直于 LOS 但不是加速度方向；指令须绕 LOS 旋转 90° 进入 LOS 末端扫过的平面。丢掉叉积会使指令倾斜并产生持续横向发散（仓库内已验证：脱靶单调增大）。

### 3.3 Skid-to-turn 运动学逆

指令在体系表达并丢弃推力轴（$b_x$）分量——导弹助推后不能沿 $b_x$ 加速：

$$
a_{\text{lat}}=\big(I-b_xb_x^\top\big)R^\top a^{\text{cmd}}. \tag{3}
$$

以转弯率 $a/V$ 把机头转向指令横向加速度方向即可相应弯曲速度矢量，给出体角速度指令

$$
\omega^{\text{cmd}}=\frac{b_x\times a_{\text{lat}}}{V},\qquad
V=\max(\lVert v_b\rVert,50\ \text{m/s}). \tag{4}
$$

### 3.4 输出归一化

速率指令按逐轴限幅归一化并裁剪：

$$
a_{\text{raw}}=\operatorname{clamp}\Big(\frac{\omega^{\text{cmd}}}{(6.0,1.0,1.0)},-1,1\Big). \tag{5}
$$

式 (5) 饱和是制导需求超出速率指令包线时的确定性回退；下游 rate PID 直接饱和舵面。

## 4. 参数

| 常量 | 值 | 含义 |
|---|---|---|
| `N` | 15.0 | 导航常数（从文档中的 N=4 / N=10 谱系重调，见 §5） |
| `LOS_FILTER` | 0.8 | 一阶 LOS 角速度滤波系数 $\alpha$ |
| 滤波冻结 | $V_c<50$ m/s | 终端 LOS 噪声保护（$\alpha=0$） |
| 速度下限 | $V\ge50$ m/s | 运动学逆 (4) 分母保护 |
| 距离钳位 | $\lVert r_b\rVert\ge10^{-3}$ m | LOS 单位向量奇异保护 |
| `OMEGA_MAX` | 6.0 rad/s | 滚转速率指令限（环境常量，构造时读取） |
| `OMEGA_LAT` | 1.0 rad/s | 俯仰/偏航速率限（2026-10-24 由 0.5 提到 1.0 以增终端瞬态权限） |

环境固定工作点（背景，非基线调参）：`R_SPAWN`=2500 m、`CONE`=±37.5°、`A_LAT`=7.5 g、`F_WEAVE`=0.75 Hz、`V_T`=`V_M0`=250 m/s、`Z_MID`=4000 m、`KILL_RADIUS`=15 m、`TGO_SCALE`=0.8、回合上限 900 策略步（15 s）。

## 5. 实测表现

成功即近炸引信杀伤（任一物理子步脱靶 <15 m）。所有发布记录使用 `scripts/baseline_eval.py` 记账（每副本首回合、固定中等工作点、`env_kwargs={}`）。环境 `difficulty` 构造参数已于 2026-10-24 移除，早先按难度参数化的表不可复现，此处不再引用。

**协议数据集 A**（`results/baseline_protocol/seed{7,13,21,101}/missile/baseline.json`，每 seed 128 副本）：

| seed | 杀伤率 | 杀伤/完成 | 平均回合回报 | env 步数 |
|---|---:|---:|---:|---:|
| 7 | 75.0% | 96 / 128 | 100.53 | 591 |
| 13 | 71.1% | 91 / 128 | 95.62 | 656 |
| 21 | 85.9% | 110 / 128 | 113.61 | 575 |
| 101（留出） | 70.3% | 90 / 128 | 94.91 | 597 |

协议 seeds 7/13/21：**77.3% ± 7.7**（297/384），平均回报 103.25，每个文件 `complete=true`。相对样本量，逐 seed 离散（70.3–85.9%）较大；单个 128 副本记录不应被读作精确能力估计。

**已发布单记录**：`results/missile/baseline.json` 为 seed-13 协议记录：128 副本、seed 13、656 env 步、**杀伤率 0.7109（91/128）**、平均回合回报 **95.62**、`complete=true`。

**数据集 B**（`results/missile/baselines/*_seed7.json`，256 副本、seed 7、637 env 步）：

| 臂 | CLI 键 | 杀伤率 | 杀伤/完成 | 平均回报 |
|---|---|---|---:|---:|---:|
| `apn_rate`（本臂，N=15） | —（默认） | **76.2%** | 195 / 256 | 101.90 |
| `tpn` | `tpn` | 76.2% | 195 / 256 | 102.13 |
| `ppn` | `ppn` | 2.0% | 5 / 256 | 27.21 |

在此固定快照上 `tpn`（无半 $N$ 重力增广的真 PN）与 APN 相当；`ppn`（无前置量的纯追踪）几乎处处失败。数据集 B 还报告 `apn_rate` 的 `timeout_rate=0.0`，即失败是真正越过最近接近点而非 15 s 超时。

**调参记录（未发布）**：`MissileAPNBaseline` 代码注释记录固定工作点上 1024 回合 seed-7 调参运行中 N=15 得 SR 0.812、N=10 为 0.770，N≥18 过冲并破坏 LOS 环。检出中无对应 JSON 产物，故为设计笔记而非基准记录。

**验收门（非测量）**：`environments/missile/test_missile_apn.py` T3 驱动 256 副本至多 3000 env 步，仅断言混合机动杀伤率 ≥0.45 且直线目标 ≥0.85；不存产物，不是结果。

## 6. 已知局限

- **直线射击假设**：APN 假设非机动目标；残余失败集中在蛇形与桶滚机动类型（环境 reset 时均匀抽 `t_type`）。逐机动分解由 `environments/missile/eval_missile_scenarios.py` 生成，但其输出 `results/missile/scenarios.json` **未发布**，故此处不引用逐类型失败切分。该机动目标残余正是 RL 预期攻击的差距。
- **终端 LOS 角速度噪声**：$V_c<50$ m/s 以下滤波冻结，终端修正依赖冻结估计。
- **导航常数敏感**：N≥18 过冲并破坏 LOS 环（代码内调参说明）。
- **速率指令包线**：制导权限受俯仰/偏航速率限约束，超出即式 (5) 饱和，无可可行性投影。

## 7. 实现

- 控制器：`MissileAPNBaseline`（`name='apn_rate'`）位于 `algorithms/baselines/baselines.py`，经 `@register_baseline('missile')` 注册；由 `build_baseline('missile', env)` 实例化，在 `environments/missile/env.py::MissileInterceptEnv` 声明为 `conventional_controller`。
- 评测：`scripts/baseline_eval.py missile`（官方协议，写已发布的 `results/missile/baseline.json`）；数据集 B 臂由 `scripts/baseline_suite.py` / `scripts/baseline_eval.py missile --baseline <tpn|ppn>` 生成于 `results/missile/baselines/`。测试套件 `environments/missile/test_missile_apn.py`（T1 舵符号、T2 速率环、T3 闭环拦截门；默认 `cpu` 后端）；场景 harness `environments/missile/eval_missile_scenarios.py`（写 `results/missile/scenarios.json`，**未发布**）。

## 8. 参考

1. Zarchan, P. *Tactical and Strategic Missile Guidance*, AIAA —— 比例导引与带重力补偿的增广比例导引。
2. Siouris, G. M. *Missile Guidance and Control Systems*, Springer —— PN 机械化与 skid-to-turn 机体约定。
3. 任务谱系（项目自研环境，参考）：BVR Gym（arXiv:2403.17533）与 LAG（arXiv:2506.14164）的交战及过载/LOS 角速度约定。
