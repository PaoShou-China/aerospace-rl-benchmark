# 传统（非学习）基线完整测试报告

## 测试口径

- 冻结评测协议：官方 benchmark 初始分布、静风、**标称 plant（域随机化关闭）**、128 并行环境、每个并行副本只采信**首回合**（`FirstEpisodes`，与 `scripts/baseline_eval.py` 同一记账）。
- 每个任务 × 每个控制器在协议种子 {7, 13, 21} 各跑一次，另跑留出种子 101 一次；成功率 = `completed_successes / completed_episodes`。
- 95% 区间为 pooling 三个协议种子后的 Wilson 区间（样本量 ≈ 3×副本数 回合）。
- 训练侧 `env_kwargs`（如固定翼 Dryden 风）**不继承**：基线一律静风评测。

## 总表（协议种子 7/13/21 均值，附留出种子 101）

| 任务 | 控制器 | 成功率 mean±sd | [min, max] | 95% CI | 留出 101 | mean return | 回合 | 步数窗口 | 平均墙钟 s |
|---|---|---|---|---|---|---|---|---|---|
| quadx | cascaded_pid | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 72.6 | 384 | 544 | 11 |
| quadx | smc | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 72.6 | 384 | 576 | 11 |
| quadx | se3_geometric | **99.7%** ± 0.5 | [99.2, 100.0] | [98.5, 100.0] | 100.0% | 72.5 | 384 | 672 | 14 |
| quadx_track | cv_kf_pid | **79.2%** ± 3.5 | [75.8, 82.8] | [74.8, 82.9] | 75.0% | 84.7 | 384 | 1216 | 30 |
| quadx_track | pid_ema | **74.7%** ± 4.0 | [70.3, 78.1] | [70.2, 78.8] | 78.1% | 78.1 | 384 | 1216 | 30 |
| quadx_track | pursuit | **74.2%** ± 4.3 | [69.5, 78.1] | [69.6, 78.3] | 78.1% | 72.8 | 384 | 1216 | 29 |
| fixedwing | nlgl_tecs | **94.3%** ± 1.2 | [93.0, 95.3] | [91.5, 96.2] | 90.6% | 217.6 | 384 | 3616 | 135 |
| fixedwing | l1_tecs | **87.0%** ± 3.2 | [83.6, 89.8] | [83.2, 90.0] | 90.6% | 208.8 | 384 | 3616 | 130 |
| fixedwing | pid_cascade | **76.8%** ± 4.6 | [73.4, 82.0] | [72.3, 80.8] | 76.6% | 193.2 | 384 | 3616 | 120 |
| rocket | pd_landing | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 7.3 | 384 | 1216 | 40 |
| rocket | zem_zev_pdg | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 900.2 | 384 | 1216 | 41 |
| rocket | eguidance | **34.4%** ± 5.1 | [28.9, 39.1] | [29.8, 39.3] | 33.6% | -3304.8 | 384 | 1216 | 46 |
| missile | apn_rate | **77.6%** ± 8.1 | [71.1, 86.7] | [73.2, 81.5] | 71.9% | 103.5 | 384 | 672 | 17 |
| missile | tpn | **77.6%** ± 7.4 | [71.9, 85.9] | [73.2, 81.5] | 71.9% | 103.7 | 384 | 576 | 17 |
| missile | ppn | **2.3%** ± 0.8 | [1.6, 3.1] | [1.2, 4.4] | 2.3% | 27.6 | 384 | 928 | 26 |
| quadplane | se3_pusher | **98.2%** ± 0.5 | [97.7, 98.4] | [96.3, 99.1] | 96.1% | 274.4 | 384 | 1216 | 27 |
| quadplane | l1_wing | **49.2%** ± 5.6 | [43.0, 53.9] | [44.3, 54.2] | 39.1% | 217.3 | 384 | 1216 | 23 |
| quadplane | hybrid_cascade | **26.8%** ± 2.4 | [24.2, 28.9] | [22.6, 31.5] | 22.7% | 167.5 | 384 | 1216 | 21 |
| satellite | quat_pd_damped_cmg | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 65.2 | 384 | 352 | 7 |
| satellite | quat_pd_mp_pinv_cmg | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 64.9 | 384 | 384 | 8 |
| satellite | quat_pd_wpinv_cmg | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 65.4 | 384 | 352 | 7 |
| transfer | edelbaum | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 0.2 | 384 | 192 | 3 |
| transfer | gauss_tangential | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 0.8 | 384 | 192 | 3 |
| transfer | q_law_petropoulos | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 1.0 | 384 | 192 | 3 |
| docking | cwh_lqr | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 190.6 | 384 | 288 | 0 |
| docking | zem_zev | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 191.2 | 384 | 320 | 1 |
| docking | cwh_pole_placement | **89.6%** ± 3.0 | [87.5, 93.0] | [86.1, 92.3] | 86.7% | 169.3 | 384 | 224 | 0 |
| rendezvous | cwh_compensated_pd | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 179.3 | 384 | 928 | 2 |
| rendezvous | cwh_two_impulse | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 184.8 | 384 | 640 | 3 |
| rendezvous | glideslope | **100.0%** ± 0.0 | [100.0, 100.0] | [99.0, 100.0] | 100.0% | 180.8 | 384 | 864 | 1 |

## 失败模式（取协议种子中最差那次的回合平均率）

| 任务 | 控制器 | crash | 出界 | 超时 | 致命撞击 | 动作有限值 |
|---|---|---|---|---|---|---|
| quadx | cascaded_pid | 0.0% | 0.0% | 0.0% | — | ✓ |
| quadx | smc | 0.0% | 0.0% | 0.0% | — | ✓ |
| quadx | se3_geometric | 0.0% | 0.7% | 0.0% | — | ✓ |
| quadx_track | cv_kf_pid | — | 0.0% | 8.5% | — | ✓ |
| quadx_track | pid_ema | — | 0.0% | 23.4% | — | ✓ |
| quadx_track | pursuit | — | 0.0% | 24.2% | — | ✓ |
| fixedwing | nlgl_tecs | 0.0% | 7.0% | 1.0% | — | ✓ |
| fixedwing | l1_tecs | 0.0% | 2.2% | 5.2% | — | ✓ |
| fixedwing | pid_cascade | 0.0% | 12.9% | 6.8% | — | ✓ |
| rocket | pd_landing | — | — | — | — | ✓ |
| rocket | zem_zev_pdg | — | — | — | — | ✓ |
| rocket | eguidance | — | — | — | — | ✓ |
| missile | apn_rate | — | — | 0.0% | — | ✓ |
| missile | tpn | — | — | 0.0% | — | ✓ |
| missile | ppn | — | — | 3.9% | — | ✓ |
| quadplane | se3_pusher | 2.0% | 0.0% | 0.7% | — | ✓ |
| quadplane | l1_wing | 0.0% | 0.0% | 57.0% | — | ✓ |
| quadplane | hybrid_cascade | 0.0% | 4.7% | 71.1% | — | ✓ |
| satellite | quat_pd_damped_cmg | — | — | — | — | ✓ |
| satellite | quat_pd_mp_pinv_cmg | — | — | — | — | ✓ |
| satellite | quat_pd_wpinv_cmg | — | — | — | — | ✓ |
| transfer | edelbaum | — | — | — | — | ✓ |
| transfer | gauss_tangential | — | — | — | — | ✓ |
| transfer | q_law_petropoulos | — | — | — | — | ✓ |
| docking | cwh_lqr | — | — | — | — | ✓ |
| docking | zem_zev | — | — | — | — | ✓ |
| docking | cwh_pole_placement | — | — | — | — | ✓ |
| rendezvous | cwh_compensated_pd | — | — | — | — | ✓ |
| rendezvous | cwh_two_impulse | — | — | — | — | ✓ |
| rendezvous | glideslope | — | — | — | — | ✓ |

## 方法族谱

| 任务 | 控制器 | 方法 |
|---|---|---|
| quadx | cascaded_pid | 位置→姿态级联 PID（Euler） |
| quadx | smc | 滑模位置外环（指数趋近律+边界层）+ Euler 姿态级联 |
| quadx | se3_geometric | Lee SE(3) 几何控制（默认） |
| quadx_track | cv_kf_pid | CV 卡尔曼滤波估计目标 + SE(3) 跟踪（默认） |
| quadx_track | pid_ema | 一阶 EMA 目标滤波 + SE(3) 跟踪 |
| quadx_track | pursuit | 无滤波纯追踪（原始含噪相对测量） |
| fixedwing | nlgl_tecs | Park 非线性制导律（NLGL）+ TECS |
| fixedwing | l1_tecs | L1 横向制导 + TECS 纵向（默认） |
| fixedwing | pid_cascade | 常规级联 PID 姿态/高度/速度 |
| rocket | pd_landing | 纯 PD 着陆（无制导层） |
| rocket | zem_zev_pdg | ZEM/ZEV 中段 + 虚拟平台 PDG + 速度带脉冲伺服（默认） |
| rocket | eguidance | Apollo E-guidance 二次多项式加速制导 + 终端脉冲伺服 |
| missile | apn_rate | 增广比例导引 APN + half-N 重力补偿（默认） |
| missile | tpn | 真比例导引 TPN（无 (N/2)g 增广） |
| missile | ppn | 纯追踪导引（无前置量） |
| quadplane | se3_pusher | 四旋翼 SE(3) 全程 + 前向推进器分配（默认） |
| quadplane | l1_wing | 机翼模式 L1 横向 + 四旋翼低速助推 |
| quadplane | hybrid_cascade | 模式切换混合自驾（低速四旋翼/高速机翼），含机翼升力卸荷项 |
| satellite | quat_pd_damped_cmg | 四元数 PD + Schaub-Junkins 阻尼伪逆 CMG（默认） |
| satellite | quat_pd_mp_pinv_cmg | 四元数 PD + 普通 Moore-Penrose 伪逆 CMG |
| satellite | quat_pd_wpinv_cmg | 四元数 PD + Oh-Fleming 加权阻尼最小范数 + Vadali 零空间奇异规避 |
| transfer | edelbaum | Edelbaum 解析 Δv 预算开环切向推力 |
| transfer | gauss_tangential | Gauss 行星方程切向推力单步贪心（默认） |
| transfer | q_law_petropoulos | Petropoulos Q-law Lyapunov 制导 |
| docking | cwh_lqr | CWH 全状态 LQR + 走廊/捕获网/燃料护栏（默认） |
| docking | zem_zev | CWH ZEM/ZEV 单脉冲滚动时域制导 + 燃料/距离护栏 |
| docking | cwh_pole_placement | CWH 极点配置（无附加约束层） |
| rendezvous | cwh_compensated_pd | CWH 补偿 PD 直趋原点 |
| rendezvous | cwh_two_impulse | CWH 滚动时域两脉冲终端制导（默认） |
| rendezvous | glideslope | 距离-速率刹车走廊（glideslope）V-bar 逼近 |

## 每任务最优经典方法

| 任务 | 最优（协议种子均值） | 次优 | 最弱 | 方法间极差 |
|---|---|---|---|---|
| quadx | cascaded_pid (100.0%) | smc (100.0%) | se3_geometric (99.7%) | 0.3 pp |
| quadx_track | cv_kf_pid (79.2%) | pid_ema (74.7%) | pursuit (74.2%) | 4.9 pp |
| fixedwing | nlgl_tecs (94.3%) | l1_tecs (87.0%) | pid_cascade (76.8%) | 17.4 pp |
| rocket | pd_landing (100.0%) | zem_zev_pdg (100.0%) | eguidance (34.4%) | 65.6 pp |
| missile | apn_rate (77.6%) | tpn (77.6%) | ppn (2.3%) | 75.3 pp |
| quadplane | se3_pusher (98.2%) | l1_wing (49.2%) | hybrid_cascade (26.8%) | 71.4 pp |
| satellite | quat_pd_damped_cmg (100.0%) | quat_pd_mp_pinv_cmg (100.0%) | quat_pd_wpinv_cmg (100.0%) | 0.0 pp |
| transfer | edelbaum (100.0%) | gauss_tangential (100.0%) | q_law_petropoulos (100.0%) | 0.0 pp |
| docking | cwh_lqr (100.0%) | zem_zev (100.0%) | cwh_pole_placement (89.6%) | 10.4 pp |
| rendezvous | cwh_compensated_pd (100.0%) | cwh_two_impulse (100.0%) | glideslope (100.0%) | 0.0 pp |

