# 方法文档

**简体中文** · [English](README.md)

面向 AerospaceRL Benchmark 中每个 RL 环境与每个算法（学习或经典）的论文式方法参考。
每篇文档定义符号，将动力学 / 奖励 / 控制律写成编号方程，列出精确的观测与动作布局，
并给出代码中每个常量的取值。若包 docstring 或 README 与实现不一致，以代码为准。

每篇方法文档都有英文全文与中文对照（`*.zh-CN.md`）；中文页为同结构精炼版，
保留关键方程、常量、观测/动作表和判据，完整推导以英文页为准。

## 环境

| 文档 | 任务（id） | 一句话摘要 |
|---|---|---|
| [env-quadx-waypoints.zh-CN.md](env-quadx-waypoints.zh-CN.md) | QuadX-Waypoints（`quadx`） | Crazyflie 级四旋翼，在 5 m 穹顶内依次捕获 4 个航点，势函数望远镜式奖励 |
| [env-quadx-track.zh-CN.md](env-quadx-track.zh-CN.md) | QuadX-Track（`quadx_track`） | Crazyflie 级四旋翼，在视场/距离/遮挡/噪声约束下持续观测合作地面移动目标 |
| [env-fixedwing-waypoints.zh-CN.md](env-fixedwing-waypoints.zh-CN.md) | Fixedwing-Waypoints（`fixedwing`） | 2.35 kg 固定翼，在 20–65 m 壳层内捕获 4 个航点，含失速的叶素气动 |
| [env.py](../../environments/vtol/env.py) | Quadplane-VTOL（`quadplane`） | 2.35 kg 混合无人机（旋翼 + 固定翼舵面 + 推进器），旋翼→气动权限自然过渡；模块 docstring 即方法参考 |
| [env-rocket-landing.zh-CN.md](env-rocket-landing.zh-CN.md) | Rocket-Landing（`rocket`） | F9 级助推器动力着陆，benchmark v2（40 Hz，停稳 2 s）：v5 稠密奖励，成功不触发截断 |
| [env-missile-intercept.zh-CN.md](env-missile-intercept.zh-CN.md) | Missile-Intercept（`missile`） | 6-DOF 尾舵导弹对抗运动学机动目标，近炸引信击杀 |
| [env-satellite-attitude.zh-CN.md](env-satellite-attitude.zh-CN.md) | CMG 姿态控制（`satellite`） | 刚性星体 + 4/6-CMG 群，奇异感知的望远镜式奖励 |
| [env-orbit-transfer.zh-CN.md](env-orbit-transfer.zh-CN.md) | 低推力转移（`transfer`） | 二体近地点抬升，10 mN/轴电推进，PBRS 望远镜式叠加到终端轨道根数奖励 |
| [env-docking.zh-CN.md](env-docking.zh-CN.md) | Docking（`docking`） | CWH 最终逼近 60–120 m，离散辛积分，1 m / 0.1 m/s 成功门限 |
| [env-rendezvous.zh-CN.md](env-rendezvous.zh-CN.md) | Rendezvous（`rendezvous`） | CWH 中距逼近 1000–1500 m，含 5 m keep-out 球，10 m / 0.5 m/s 目标 |

## 学习算法

| 文档 | 内容 |
|---|---|
| [algo-ppo.zh-CN.md](algo-ppo.zh-CN.md) | rsl-rl PPO，2×128 tanh MLP，共享 warmup-cosine 学习率调度，公平对照协议 |
| [algo-bc-warmstart.zh-CN.md](algo-bc-warmstart.zh-CN.md) | 从经典基线做行为克隆热启动（仅 PPO），数据契约与拟合流程 |

## 经典（非学习）基线

每个环境在 `BUILDERS` 中注册一个默认控制器，并在 `BASELINE_VARIANTS` 中提供
CLI 可选的替代控制律（`scripts/baseline_eval.py --baseline <key>`）。即
**10 个默认 + 20 个不同变体控制器 = 30 个控制器臂**
（`algorithms/baselines/baselines.py`）；22 个 `--baseline` 键中有两个是默认臂的别名
（下表已标注），因此表格列出 32 行，覆盖这 30 个不同控制器。`--baseline` 键用反引号给出。

### 3.1 每任务默认控制器（`BUILDERS`）

| 文档 | 任务 | 默认方法 |
|---|---|---|
| [baseline-quadx-se3.zh-CN.md](baseline-quadx-se3.zh-CN.md) | quadx | SE(3) 几何跟踪控制（Lee et al. 2010）+ 积分配平 |
| [baseline-quadx-track-kf-pid.zh-CN.md](baseline-quadx-track-kf-pid.zh-CN.md) | quadx_track | 匀速卡尔曼滤波 + 固定偏移 SE(3) 跟踪，含公开超时保持 |
| [baseline-fixedwing-l1-tecs.zh-CN.md](baseline-fixedwing-l1-tecs.zh-CN.md) | fixedwing | L1 横向制导（Park-Deyst-Howe 2004）+ TECS 纵向自驾 |
| [variants_quadplane.py](../../algorithms/baselines/variants_quadplane.py)（`QuadplaneSE3PusherBaseline`） | quadplane | 全航点任务用四旋翼 SE(3) 几何控制，推进器油门闭合机体系 x 地速误差（已验证：128 副本 × seeds 7/13/21 上 98.2%±0.5，留出 seed 101 = 96.1%） |
| [baseline-rocket-pdg.zh-CN.md](baseline-rocket-pdg.zh-CN.md) | rocket | ZEM/ZEV 动力下降制导、虚拟平台制动流形、脉冲伺服 |
| [baseline-missile-apn.zh-CN.md](baseline-missile-apn.zh-CN.md) | missile | 增广比例导引映射为体轴角速率指令 |
| [baseline-satellite-cmg-pd.zh-CN.md](baseline-satellite-cmg-pd.zh-CN.md) | satellite | 四元数 PD + Schaub-Junkins 阻尼伪逆 CMG 操纵 |
| [baseline-transfer-gauss.zh-CN.md](baseline-transfer-gauss.zh-CN.md) | transfer | Gauss 单步贪心切向制导，目标函数计入推进剂代价 |
| [baseline-docking-cwh-lqr.zh-CN.md](baseline-docking-cwh-lqr.zh-CN.md) | docking | 辛精确离散 CWH LQR + 距离护栏 / 捕获网 / 燃料降额 |
| [baseline-rendezvous-two-impulse.zh-CN.md](baseline-rendezvous-two-impulse.zh-CN.md) | rendezvous | 递归重解的两脉冲 CWH ZEM/ZEV 制导，含可行性护栏 |

`RendezvousCWHPDBaseline` 另注册在独立键 `rendezvous_cwhpd` 下
（同一任务的第二个控制器，不能通过 `--baseline` 访问）。

### 3.2 任务内变体（`BASELINE_VARIANTS`）

| 任务 | `--baseline` 键 | 控制器 | 方法参考 |
|---|---|---|---|
| quadx | `cascaded_pid` | `QuadXCascadePIDBaseline` | [baseline-quadx-cascade-pid.zh-CN.md](baseline-quadx-cascade-pid.zh-CN.md)：位置 PID → 姿态 P/PD 级联，叠加在环境原生角速率 PID 上 |
| quadx | `smc` | `QuadXSMCBaseline` | [variants_quadx.py](../../algorithms/baselines/variants_quadx.py)：滑模面 `s = e_v + λe_p` + 饱和指数趋近律，Euler 姿态级联 |
| quadx_track | `pursuit` | `QuadXTrackPursuitBaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py)：原始测量纯追踪，无估计器 |
| quadx_track | `pid_ema` | `QuadXTrackPIDEMABaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py)：EMA 滤波目标 + SE(3) 级联 |
| fixedwing | `pid_cascade` | `FixedwingPIDBaseline` | [baseline-fixedwing-pid.zh-CN.md](baseline-fixedwing-pid.zh-CN.md)：航向→滚转→滚转率、高度→俯仰→俯仰率、空速 PI |
| fixedwing | `nlgl_tecs` | `FixedwingNLGLBaseline` | [variants_fw_rocket.py](../../algorithms/baselines/variants_fw_rocket.py)：Park 非线性制导律 + TECS 纵向栈 |
| rocket | `pd_landing` | `RocketPDLandingBaseline` | [baseline-rocket-pd-landing.zh-CN.md](baseline-rocket-pd-landing.zh-CN.md)：制动距离/阈值高度 PD 着陆，滞环发动机带，单次点火 |
| rocket | `eguidance` | `RocketEGuidanceBaseline` | [variants_fw_rocket.py](../../algorithms/baselines/variants_fw_rocket.py)：Apollo E-guidance（二次多项式瞄准）中段制导 |
| missile | `ppn` | `MissilePPNBaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py)：纯追踪，无前置量 |
| missile | `tpn` | `MissileTPNBaseline` | [variants_air.py](../../algorithms/baselines/variants_air.py)：真比例导引，无重力增广 |
| quadplane | `hybrid_cascade` | `QuadplaneHybridBaseline` | [baselines.py](../../algorithms/baselines/baselines.py)：速度调度的旋翼/机翼模式切换，含机翼升力卸荷项；为 live-wing plant 重调（26.8% 成功率，消融臂） |
| quadplane | `l1_wing` | `QuadplaneL1WingBaseline` | [variants_quadplane.py](../../algorithms/baselines/variants_quadplane.py)：机翼模式 L1 横向制导 + 旋翼低速辅助（49.2%，消融臂） |
| satellite | `quat_pd_damped_cmg` | `SatelliteCMGBaseline` | [baseline-satellite-cmg-pd.zh-CN.md](baseline-satellite-cmg-pd.zh-CN.md)：默认臂，可通过键选择 |
| satellite | `quat_pd_mp_pinv_cmg` | `SatelliteOrdinaryPinvBaseline` | [baseline-satellite-mp-pinv.zh-CN.md](baseline-satellite-mp-pinv.zh-CN.md)：同一 PD，普通 Moore–Penrose 伪逆操纵 |
| satellite | `quat_pd_wpinv_cmg` | `SatelliteWeightedPinvBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py)：加权伪逆 + 零空间奇异规避操纵 |
| transfer | `gauss_tangential` | `TransferGaussBaseline` | [baseline-transfer-gauss.zh-CN.md](baseline-transfer-gauss.zh-CN.md)：默认臂，可通过键选择 |
| transfer | `q_law` | `TransferQLawBaseline` | [baseline-transfer-qlaw.zh-CN.md](baseline-transfer-qlaw.zh-CN.md)：Petropoulos Q-law 反馈制导，平面 (a, e) 子集 + effectivity 滑行 |
| transfer | `edelbaum` | `TransferEdelbaumBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py)：Edelbaum 开环 Δv 预算，推力沿当地速度 |
| docking | `pole_placement` | `DockingCWHPlacementBaseline` | [baselines.py](../../algorithms/baselines/baselines.py)：CWH 全状态反馈，标准极点配置，无额外层 |
| docking | `zem_zev` | `DockingZEMZEVBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py)：CWH ZEM/ZEV 单脉冲制导 + 燃料/距离护栏 |
| rendezvous | `cwh_pd` | `RendezvousCWHPDBaseline` | [baselines.py](../../algorithms/baselines/baselines.py)：CWH 补偿 PD 直趋原点，无约束层 |
| rendezvous | `glideslope` | `RendezvousGlideslopeBaseline` | [variants_space.py](../../algorithms/baselines/variants_space.py)：沿初始视线的距离变化率制动走廊（glideslope） |

两行“默认臂，可通过键选择”是 3.1 在 CLI 名称下的重复项：它们不是额外控制器。
全部 30 个不同控制器的逐臂成功率、Wilson 区间与失败模式表见
[baselines.zh-CN.md](../baselines.zh-CN.md)。
