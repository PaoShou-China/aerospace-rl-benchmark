# QuadX-Track 环境（`QuadXTrackEnv`）

**简体中文** · [English](env-quadx-track.md)

> **状态：已实现** —— 环境、`train_quadx_track.py` 入口、KF+SE(3) 基线以及 `baseline_eval.py` / `eval.py` 两处注册均已存在并运行过；这不是需要新建民用观测环境的计划任务。

## 1. 概述

QuadX-Track 是四旋翼家族的持续观测任务。Crazyflie 2.x 微型四旋翼（27 g，X 构型）需在 30 s、40 Hz 策略频率下，把**合作地面移动目标**保持在稳定下视相机的视场内，同时维持固定高度与固定水平偏移。目标在地面按预抽签的直线、转弯、启停段在 15 m 半径区域内行驶。

Plant、内环 rate PID、电机 mixer、4 维速率+总距动作接口、求解器保护与 40 Hz 策略频率**完全沿用 QuadX-Waypoints**（`TensorPlant('quadx')`）。任务差别仅在感知与奖励。感知为几何计算而非渲染：下视锥、斜距窗口、静态圆柱 LOS 遮挡与各向同性位置噪声。无拦截、接触、武器或碰撞目标。

## 2. 符号

| 符号 | 含义 |
|---|---|
| $p,v$；$q=(w,x,y,z)$ | 载机世界位置 / 速度；姿态四元数 |
| $\omega_b,v_b$ | 体系角速度 / 线速度 |
| $p_t$，$\theta_t$，$s_t$ | 目标位置（世界）、航向、地速 |
| $\alpha$ | 相机半角 30° |
| $d_h,d_3$ | 水平 / 斜距 |
| $y_t\in\{0,1\}$ | 可见标志 |
| $\zeta_t$ | 世界轴下相对载机的位置测量 |
| $\bar p=p_t^{\mathrm{ref}}$ | 跟踪参考点 |
| $\Delta t_\pi,\Delta t_c,\Delta t$ | 策略 1/40 s，控制 1/120 s，物理 1/240 s |

## 3. Plant 与控制层级

与 QuadX-Waypoints 相同。策略 40 Hz；每策略步执行 `inner_steps=3` 个 120 Hz 控制周期，每周期一次 rate PID 更新 + 2 个 240 Hz 物理子步（共 6 子步）。电机滞后/噪声、二次推力、逐轴阻力与 15 m/s、30 rad/s 求解器保护均不变。

## 4. 合作地面目标

目标为定高 $z_t=0.15$ m 的运动学地面车辆。reset 时每副本一次均匀抽样固定出生半径 $r\sim\mathcal U[0.5,2]$ m、出生角、初始航向及 `NUM_SEGMENTS=12` 段未来航段。每段携带

$$
\text{mode}\in\{\text{straight},\text{turn},\text{stop-go}\},\quad
T_k\sim\mathcal U[2,5]\ \mathrm{s},\quad
s_k\sim\mathcal U[0.4,1.2]\ \mathrm{m/s},\quad
\omega_k=\pm\mathcal U[0.3,1.0]\ \mathrm{rad/s},
$$

stop-go 时 $s_k=0$，straight 时 $\omega_k=0$。目标每策略步（$\Delta t_\pi=25$ ms）推进：

$$
p_t \leftarrow p_t + s_k[\cos\theta_t,\ \sin\theta_t,\ 0]^\top\Delta t_\pi,\qquad
\theta_t \leftarrow \theta_t+\omega_k\Delta t_\pi .
$$

航段计时器倒计时，到期装载下一预抽段。最短段 2 s，故每策略步至多跨一个边界，更新为单次 gather。模式参数仅在 reset 抽样，`step` 内从不抽样，使未激活副本不入随机流。

目标在圆形作业区 $r\le15$ m 反射：位置投影回圆，航向关于径向法向反射。这使 30 s 回合留在载机 30 m 出界球内且不消耗在线随机数。

## 5. 几何相机模型

相机装在稳定云台上，光轴为世界 $-z$，高差 $\Delta z$ 处地面足迹半径为 $\Delta z\tan\alpha$。检测需**同时**满足：

1. **视场**：$z-z_t>0$ 且 $d_h\le(z-z_t)\tan\alpha$。
2. **斜距窗口**：$1.5\ \mathrm{m}\le d_3\le12\ \mathrm{m}$。
3. **视线**：$p$ 到 $p_t$ 的线段不与静态圆柱遮挡物相交。对位于 $(c_x,c_y)$、半径 $r_o$、高 $h_o$ 的遮挡物，令 $\tau=\mathrm{clip}(((c-p)\cdot(p_t-p))/\lVert p_t-p\rVert_{xy}^2,0,1)$；当 $\lVert p_{xy}+\tau(p_t-p)_{xy}-c\rVert<r_o$ 且 $z+\tau(z_t-z)<h_o$ 时 LOS 被挡（视线在柱顶以下穿过圆柱）。

遮挡物 $(\cdot,\cdot,r_o,h_o)$：

| x | y | 半径 | 高度 |
|---|---|---|---|
| 4.5 | 3.0 | 1.0 | 5.0 |
| −3.5 | 5.5 | 1.2 | 6.0 |
| 6.0 | −4.0 | 1.0 | 4.5 |
| −6.0 | −3.0 | 1.1 | 5.5 |
| 1.0 | −6.5 | 1.0 | 4.0 |
| −1.5 | 8.5 | 1.0 | 6.5 |

它们**纯为视线几何**：不是 plant 中的物理实体，无碰撞、无受力，不能用于拦截或接触目标。任务无碰撞目标；报告的碰撞指标是载机自身触地。

可见时相机返回

$$
\zeta_t=p_t+\varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2 I_3),\quad \sigma=0.15\ \mathrm{m},
$$

世界轴下**相对载机**，即观测存 $\zeta_t-p$。噪声用独立计数器流（`seed+7919`），不扰动 plant 电机噪声序列。任一条件不满足时观测恰为 $\zeta_t-p=0$ 且 $y_t=0$。

## 6. 动作空间

与 QuadX-Waypoints 相同：裁剪到 $[-1,1]^4$，仿射映射到 $(\omega_x^{\mathrm{cmd}},\omega_y^{\mathrm{cmd}},\omega_z^{\mathrm{cmd}},T)$，物理范围 $(\pm\pi,\pm\pi,\pm\pi,[0,0.8])$。所有控制器共享该接口。

## 7. 观测空间与公开信息契约（25 维，float32）

| 切片 | 符号 | 单位 | 说明 |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | 体角速度 |
| [3:7] | $q$ (x,y,z,w) | — | 姿态四元数 |
| [7:10] | $v_b$ | m/s | 体系速度 |
| [10:13] | $p$ | m | 世界位置 |
| [13:17] | $\bar a_{t-1}$ | mixed | 上一物理动作 |
| [17:21] | $\theta$ | — | 各电机状态 |
| [21:24] | $\zeta_t-p$ | m | 当前可见目标测量（世界轴）；**不可见时为 0** |
| [24:25] | $y_t$ | — | 可见标志 |

观测**不含目标真值**：无目标位置、速度、模式、计划或上次可见时间。不可见步除标志外与"此刻无测量"不可区分。PPO 入口以其原有 memoryless 网络消费该向量；传统基线消费同一向量，其卡尔曼滤波属于方法本身而非额外传感信息。

## 8. 奖励

参考点 $\bar p=(p_{t,x}+\delta_x,\ p_{t,y}+\delta_y,\ \texttt{ALT\_REF})$，$\delta=(1.5,0)$ m，高度 6 m，每策略步：

$$
r_t = 0.10\,y_t - 0.10(1-y_t)
      - 0.05\lVert p_{xy}-\bar p_{xy}\rVert y_t
      - 0.01(z-\texttt{ALT\_REF})^2
      - 0.002\lVert\bar a_t\rVert^2
      - 10\,\kappa_t ,
$$

$\kappa_t\in\{0,1\}$ 为坠毁/出界。跟踪项仅在可见时支付；丢失时唯一信号是 $-0.10$ 惩罚。奖励是环境侧量，不是观测通道，也不通过它向控制器暴露目标状态。

## 9. 终止、截断与成功

1. **终止**：触地或 $\lVert p\rVert>30$ m。置 `terminated`。
2. **截断**：1200 策略步（30 s）。
3. **成功**：在时限截断时，回合可见时间占比 $\ge0.85$ 且未坠毁则置 `complete`。成功被记录但不提前截断，故 `extras['time_outs']` 包含全部时限截断，价值 bootstrap 与其他任务一致。

## 10. 初始分布

载机在 $(0,0,6)$ m 水平静止出生。目标在载机地面投影 0.5–2 m 半径内出生，保证首帧必为检测。重置副本执行 plant 的 10×2 零指令 warmup，随后以该首帧测量初始化观测。固定种子逐位复现出生、目标计划与传感噪声。

## 11. 指标

每完成回合累计：

| 指标 | 定义 |
|---|---|
| `visible_fraction` | 可见策略步 / 回合步数 |
| `lost_seconds` | 回合总不可见时间 |
| `max_dropout_seconds` | 最长连续不可见区间 |
| `track_error` | 可见步上 $\lVert p_{xy}-\bar p_{xy}\rVert$ 均值 |
| `flight_distance` | 路径长度积分（m） |
| `collisions` | 触地终止 |
| `out_of_bounds` | 出球终止 |
| `timeouts` | 未成功的时限截断 |
| `fov_loss_steps`, `range_loss_steps`, `occlusion_steps` | 丢失原因计数（每不可见步按该优先级计一个原因） |

`scripts/baseline_eval.py` 与 `scripts/eval.py` 将累计计数器除以完成回合数，在 `task_metrics` 下报告。

## 12. 实现与关键常量

- 文件：`environments/multirotor/track_env.py`，类 `QuadXTrackEnv`（任务 id `quadx_track`），由 `environments/multirotor/__init__.py` 导出；plant 在 `environments/common/plant.py`。
- 传统控制器：`QuadXTrackKFBaseline`（`conventional_controller`）。
- 训练入口：`environments/multirotor/train_quadx_track.py`（PPO）。调参/测试 harness：`eval_quadx_track_kf.py`、`test_quadx_track.py`。

| 常量 | 值 |
|---|---|
| 策略/控制/物理频率 | 40 / 120 / 240 Hz（`inner_steps=3`，2 子步） |
| `max_steps` / 回合 | 1200（30 s）；`max_episode_length=1202` |
| 相机半角 / 传感窗口 | 30°；1.5–12 m 斜距 |
| 位置噪声 | 每轴 0.15 m |
| 高度 / 水平偏移 | 6.0 m；(1.5, 0.0) m |
| 目标速度 / 转弯率 / 航段 | 0.4–1.2 m/s；0.3–1.0 rad/s；12 段，每段 2–5 s |
| 目标区域 / 载机边界 | 15 m / 30 m |
| 成功可见占比 | 0.85 |
| 奖励权重 | +0.10 可见，−0.10 丢失，−0.05/m 跟踪，−0.01/m² 高度，−0.002 动作²，−10 坠毁 |
| `vel_clamp` | 15 m/s，30 rad/s |

## 13. RL 训练协议（PPO）

学习臂为**未修改的仓库 PPO 实现**：`rsl_rl` `OnPolicyRunner` + rsl-rl `MLPModel` actor-critic（2×128 tanh，无循环层），状态无关 Beta 策略（`BetaDistribution`，有界并缩放到 $[-1,1]$），每次更新 100 向量化步、5 epochs、4 minibatches、固定 lr $3\times10^{-4}$、熵 0.01。入口启用行为克隆热启动（`BC_WARMSTART = dict(steps=512, epochs=100, batch_size=8192, lr=1e-3, fit_critic=True)`），`WIND=None` 保持静风。

它消费第 7 节 25 维观测（当前自身状态、当前测量、可见标志；不可见时测量为 0）、同一 4 维动作接口、同一奖励、同一 reset 分布与同一评测场景。

**交互预算**：`NUM_ENVS=1024`，`TRAIN_ENV_STEPS=20,000` 向量化策略步，即 `UPDATES=200` 次 PPO 更新、$20{,}000\times1024=20.48$ M 转场。训练 seed 7。`eval.py` 默认 seed 7，但**未发布 PPO 运行**：`results/quadx_track/ppo/` 不存在，故无 checkpoint/config/eval 可复现学习臂数字。已发布传统记录为 `results/quadx_track/baseline.json`（128 副本，seed 21，1200 步窗口，106/128=82.8%，平均回报 81.73），与下方协议 seed 21 行逐字节一致；任务内消融集为 256 副本 seed 7 数据集 B。

```bash
python environments/multirotor/train_quadx_track.py            # PPO 臂
python scripts/eval.py quadx_track --out results/quadx_track/ppo
python scripts/baseline_eval.py quadx_track
```

`eval.py` 默认确定性策略（均值动作）、1024 副本、seed 7，每个副本至少完成一回合后停止（安全上限 `--max-steps 20000`）。它从所指向的 checkpoint 目录恢复训练 `env_kwargs`（此处为 `results/quadx_track/ppo/config.json`），**并非已发布文件**，故 `results/quadx_track/ppo/eval.json` 未公开。

**已发布记录**：每回合跑满 1200 步窗口（成功在时限判定、不截断），故各行 `steps` 即窗口长度。数据集 A 为 128 副本首回合协议；B 为 256 副本 seed 7 消融集。

| 记录 | 副本 / 种子 | 成功率 | 平均回报 |
|---|---|---|---|
| A, seed 7（`cv_kf_pid`） | 128 / 7 | 89.1%（114/128） | 89.67 |
| A, seed 13 | 128 / 13 | 85.9%（110/128） | 88.13 |
| A, seed 21 | 128 / 21 | 82.8%（106/128） | 81.73 |
| A, seed 101（留出） | 128 / 101 | 82.0%（105/128） | 86.47 |
| 已发布 `results/quadx_track/baseline.json` | 128 / 21 | 82.8%（106/128） | 81.73 |
| B `baselines/cv_kf_pid_seed7.json` | 256 / 7 | 85.2%（218/256） | 85.68 |
| B `baselines/pid_ema_seed7.json` | 256 / 7 | 75.8%（194/256） | 75.98 |
| B `baselines/pursuit_seed7.json` | 256 / 7 | 75.8%（194/256） | 71.85 |

协议种子 7/13/21 平均成功率 85.9%、平均回报 86.51。这些数字仅属传统 CV+KF+PID 臂；无学习臂产物伴随。

## 14. 参考

- Tai, J. K. et al., *PyFlyt*, v0.29.0 —— 上游 Crazyflie 2.x plant、rate PID 与 mixer 原样复用。
- Genesis contributors, *Genesis* —— 刚体后端。
- Lee, T., Leok, M., McClamroch, N. H. (2010) —— 传统基线使用的 SE(3) 几何跟踪。
