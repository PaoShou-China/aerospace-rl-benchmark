# 传统（非学习）基线完整测试报告

## 测试口径

- 冻结评测协议：官方 benchmark 初始分布、静风、1024 并行环境、每个并行副本只采信**首回合**（`FirstEpisodes`，与 `scripts/baseline_eval.py` 同一记账）。
- 每个任务 × 每个控制器在协议种子 {7, 13, 21} 各跑一次，另跑留出种子 101 一次；成功率 = `completed_successes / completed_episodes`。
- 95% 区间为 pooling 三个协议种子后的 Wilson 区间（样本量 ≈ 3×1024 回合）。
- 训练侧 `env_kwargs`（如固定翼 Dryden 风）**不继承**：基线一律静风评测。

## 总表（协议种子 7/13/21 均值，附留出种子 101）

| 任务 | 控制器 | 成功率 mean±sd | [min, max] | 95% CI | 留出 101 | mean return | 回合 | 步数窗口 | 平均墙钟 s |
|---|---|---|---|---|---|---|---|---|---|
| quadx | se3_geometric | **100.0%** ± 0.1 | [99.9, 100.0] | [99.8, 100.0] | 99.9% | 72.7 | 3072 | 762 | 93 |
| quadx | smc | **99.8%** ± 0.1 | [99.7, 99.9] | [99.6, 99.9] | 99.9% | 72.7 | 3072 | 619 | 77 |
| quadx | cascaded_pid | **99.1%** ± 0.1 | [99.0, 99.1] | [98.6, 99.3] | 99.0% | 72.6 | 3072 | 800 | 85 |
| quadx_track | cv_kf_pid | **84.0%** ± 0.9 | [83.4, 85.1] | [82.7, 85.3] | 83.4% | 84.9 | 3072 | 1200 | 61 |
| quadx_track | pid_ema | **78.3%** ± 0.9 | [77.2, 78.9] | [76.8, 79.7] | 78.6% | 77.7 | 3072 | 1200 | 60 |
| quadx_track | pursuit | **77.9%** ± 1.2 | [76.6, 78.9] | [76.4, 79.3] | 78.2% | 72.7 | 3072 | 1200 | 59 |
| fixedwing | nlgl_tecs | **91.2%** ± 1.2 | [89.8, 91.9] | [90.1, 92.1] | 93.0% | 214.5 | 3072 | 3600 | 578 |
| fixedwing | l1_tecs | **86.8%** ± 0.5 | [86.2, 87.0] | [85.5, 87.9] | 89.4% | 210.6 | 3072 | 3600 | 542 |
| fixedwing | pid_cascade | **72.9%** ± 1.5 | [71.7, 74.6] | [71.4, 74.5] | 75.5% | 190.4 | 3072 | 3600 | 486 |
| rocket | zem_zev_pdg | **100.0%** ± 0.1 | [99.9, 100.0] | [99.8, 100.0] | 100.0% | 893.6 | 3072 | 1200 | 108 |
| rocket | pd_landing | **99.9%** ± 0.1 | [99.8, 100.0] | [99.7, 100.0] | 99.9% | 6.3 | 3072 | 1200 | 108 |
| rocket | eguidance | **34.0%** ± 0.2 | [33.8, 34.3] | [32.4, 35.7] | 33.2% | -3371.3 | 3072 | 1200 | 175 |
| missile | tpn | **78.4%** ± 1.1 | [77.6, 79.7] | [76.9, 79.8] | 76.0% | 104.6 | 3072 | 640 | 33 |
| missile | apn_rate | **77.9%** ± 1.1 | [77.1, 79.2] | [76.4, 79.4] | 75.5% | 104.0 | 3072 | 670 | 37 |
| missile | ppn | **2.5%** ± 0.1 | [2.3, 2.6] | [2.0, 3.1] | 1.8% | 27.9 | 3072 | 900 | 50 |
| quadplane | se3_pusher | **87.6%** ± 1.3 | [86.4, 89.0] | [86.4, 88.7] | 87.8% | 270.1 | 3072 | 1200 | 153 |
| quadplane | l1_wing | **47.9%** ± 2.0 | [45.8, 49.6] | [46.2, 49.7] | 45.6% | 220.5 | 3072 | 1200 | 125 |
| quadplane | hybrid_cascade | **26.0%** ± 1.2 | [25.1, 27.3] | [24.5, 27.6] | 25.8% | 171.7 | 3072 | 1200 | 112 |
| satellite | quat_pd_damped_cmg | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 64.1 | 3072 | 343 | 7 |
| satellite | quat_pd_mp_pinv_cmg | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 63.8 | 3072 | 377 | 9 |
| satellite | quat_pd_wpinv_cmg | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 64.3 | 3072 | 337 | 7 |
| transfer | edelbaum | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 0.2 | 3072 | 166 | 2 |
| transfer | gauss_tangential | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 0.8 | 3072 | 166 | 2 |
| transfer | q_law_petropoulos | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 1.2 | 3072 | 166 | 3 |
| docking | cwh_lqr | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 190.3 | 3072 | 277 | 1 |
| docking | cwh_pole_placement | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 191.0 | 3072 | 236 | 0 |
| docking | zem_zev | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 191.8 | 3072 | 318 | 1 |
| rendezvous | cwh_compensated_pd | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 179.3 | 3072 | 954 | 2 |
| rendezvous | cwh_two_impulse | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 184.8 | 3072 | 653 | 4 |
| rendezvous | glideslope | **100.0%** ± 0.0 | [100.0, 100.0] | [99.9, 100.0] | 100.0% | 180.9 | 3072 | 857 | 2 |

## 失败模式（取协议种子中最差那次的回合平均率）

| 任务 | 控制器 | crash | 出界 | 超时 | 致命撞击 | 动作有限值 |
|---|---|---|---|---|---|---|
| quadx | se3_geometric | 0.0% | 0.1% | 0.0% | — | ✓ |
| quadx | smc | 0.0% | 0.3% | 0.0% | — | ✓ |
| quadx | cascaded_pid | 0.0% | 0.0% | 1.0% | — | ✓ |
| quadx_track | cv_kf_pid | — | 0.0% | 16.6% | — | ✓ |
| quadx_track | pid_ema | — | 0.0% | 22.8% | — | ✓ |
| quadx_track | pursuit | — | 0.0% | 23.4% | — | ✓ |
| fixedwing | nlgl_tecs | 0.0% | 7.2% | 1.1% | — | ✓ |
| fixedwing | l1_tecs | 0.0% | 1.5% | 4.4% | — | ✓ |
| fixedwing | pid_cascade | 0.0% | 9.1% | 10.1% | — | ✓ |
| rocket | zem_zev_pdg | — | — | — | — | ✓ |
| rocket | pd_landing | — | — | — | — | ✓ |
| rocket | eguidance | — | — | — | — | ✓ |
| missile | tpn | — | — | 0.0% | — | ✓ |
| missile | apn_rate | — | — | 0.0% | — | ✓ |
| missile | ppn | — | — | 1.9% | — | ✓ |
| quadplane | se3_pusher | 0.1% | 0.2% | 13.2% | — | ✓ |
| quadplane | l1_wing | 0.0% | 0.1% | 54.0% | — | ✓ |
| quadplane | hybrid_cascade | 0.0% | 5.8% | 68.9% | — | ✓ |
| satellite | quat_pd_damped_cmg | — | — | — | — | ✓ |
| satellite | quat_pd_mp_pinv_cmg | — | — | — | — | ✓ |
| satellite | quat_pd_wpinv_cmg | — | — | — | — | ✓ |
| transfer | edelbaum | — | — | — | — | ✓ |
| transfer | gauss_tangential | — | — | — | — | ✓ |
| transfer | q_law_petropoulos | — | — | — | — | ✓ |
| docking | cwh_lqr | — | — | — | — | ✓ |
| docking | cwh_pole_placement | — | — | — | — | ✓ |
| docking | zem_zev | — | — | — | — | ✓ |
| rendezvous | cwh_compensated_pd | — | — | — | — | ✓ |
| rendezvous | cwh_two_impulse | — | — | — | — | ✓ |
| rendezvous | glideslope | — | — | — | — | ✓ |

## 方法族谱

| 任务 | 控制器 | 方法 |
|---|---|---|
| quadx | se3_geometric | Lee SE(3) 几何控制（默认） |
| quadx | smc | 滑模位置外环（指数趋近律+边界层）+ Euler 姿态级联 |
| quadx | cascaded_pid | 位置→姿态级联 PID（Euler） |
| quadx_track | cv_kf_pid | CV 卡尔曼滤波估计目标 + SE(3) 跟踪（默认） |
| quadx_track | pid_ema | 一阶 EMA 目标滤波 + SE(3) 跟踪 |
| quadx_track | pursuit | 无滤波纯追踪（原始含噪相对测量） |
| fixedwing | nlgl_tecs | Park 非线性制导律（NLGL）+ TECS |
| fixedwing | l1_tecs | L1 横向制导 + TECS 纵向（默认） |
| fixedwing | pid_cascade | 常规级联 PID 姿态/高度/速度 |
| rocket | zem_zev_pdg | ZEM/ZEV 中段 + 虚拟平台 PDG + 速度带脉冲伺服（默认） |
| rocket | pd_landing | 纯 PD 着陆（无制导层） |
| rocket | eguidance | Apollo E-guidance 二次多项式加速制导 + 终端脉冲伺服 |
| missile | tpn | 真比例导引 TPN（无 (N/2)g 增广） |
| missile | apn_rate | 增广比例导引 APN + half-N 重力补偿（默认） |
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
| docking | cwh_pole_placement | CWH 极点配置（无附加约束层） |
| docking | zem_zev | CWH ZEM/ZEV 单脉冲滚动时域制导 + 燃料/距离护栏 |
| rendezvous | cwh_compensated_pd | CWH 补偿 PD 直趋原点 |
| rendezvous | cwh_two_impulse | CWH 滚动时域两脉冲终端制导（默认） |
| rendezvous | glideslope | 距离-速率刹车走廊（glideslope）V-bar 逼近 |

## 每任务最优经典方法

| 任务 | 最优（协议种子均值） | 次优 | 最弱 | 方法间极差 |
|---|---|---|---|---|
| quadx | se3_geometric (100.0%) | smc (99.8%) | cascaded_pid (99.1%) | 0.9 pp |
| quadx_track | cv_kf_pid (84.0%) | pid_ema (78.3%) | pursuit (77.9%) | 6.1 pp |
| fixedwing | nlgl_tecs (91.2%) | l1_tecs (86.8%) | pid_cascade (72.9%) | 18.2 pp |
| rocket | zem_zev_pdg (100.0%) | pd_landing (99.9%) | eguidance (34.0%) | 66.0 pp |
| missile | tpn (78.4%) | apn_rate (77.9%) | ppn (2.5%) | 75.9 pp |
| quadplane | se3_pusher (87.6%) | l1_wing (47.9%) | hybrid_cascade (26.0%) | 61.6 pp |
| satellite | quat_pd_damped_cmg (100.0%) | quat_pd_mp_pinv_cmg (100.0%) | quat_pd_wpinv_cmg (100.0%) | 0.0 pp |
| transfer | edelbaum (100.0%) | gauss_tangential (100.0%) | q_law_petropoulos (100.0%) | 0.0 pp |
| docking | cwh_lqr (100.0%) | cwh_pole_placement (100.0%) | zem_zev (100.0%) | 0.0 pp |
| rendezvous | cwh_compensated_pd (100.0%) | cwh_two_impulse (100.0%) | glideslope (100.0%) | 0.0 pp |

