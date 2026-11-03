# AerospaceRL Benchmark：GPU 并行航空航天控制环境

[English](README.md) | **简体中文** | [项目网页](https://paoshou-china.github.io/aerospace-rl-benchmark/)

面向航空航天控制的轻量强化学习基准，包含 **10 个任务：6 个大气飞行任务与 4 个航天器任务**。所有环境采用统一的批量设备张量接口，无需 Gym/Gymnasium 包装。每个任务提供与学习策略共享动作约定的经典 GNC 控制器，可独立评测，也可为行为克隆热启动提供示范。

项目网页支持中英文切换，展示任务回放、经典基线、设计要点与复现入口。浏览器会记住语言选择；任务名称、代码标识和单位保持统一。

## 设计与接口

- **大气飞行任务**：基于 URDF 刚体模型与 Genesis 后端，执行机构、接触和任务逻辑采用批量张量计算。
- **航天器任务**：使用显式张量动力学，无需刚体引擎。
- **统一设备约定**：物理状态为 `float64`，观测与奖励缓冲区为 `float32`，动作必须位于 `env.device`。
- **并行机制**：不使用逐副本 Python 循环；物理热路径不进行主机同步。每次环境 `step()` 仍有一次 `bool(done.any())` 同步，用于回合边界与重置记账；日志可能产生额外标量同步。
- **冻结副本**：失活副本被冻结，不消耗随机数。

技术主张及其验证条件见 [technical-highlights](docs/technical-highlights.md)。

## 环境一览

| 任务 | 类 | 动力学与控制 | 观测 / 动作 | 策略步长或频率 | 回合上限 | 成功判据 |
|---|---|---|---|---|---|---|
| 四旋翼航点 QuadX-Waypoints | `QuadXWaypointsEnv` | Crazyflie 2.x URDF，240 Hz，速率 PID | 37 / 4 | 40 Hz | 800 步，20 s | 捕获全部 4 个航点，距离 < 0.2 m |
| 四旋翼跟踪 QuadX-Track | `QuadXTrackEnv` | URDF + 运动学地面目标 + 几何相机 | 25 / 4 | 40 Hz | 1200 步，30 s | 时限处可见时间比例 ≥ 0.85 且无坠毁；成功不提前终止 |
| 固定翼航点 Fixedwing-Waypoints | `FixedwingWaypointsEnv` | 2.35 kg URDF，240 Hz，叶素气动 | 39 / 4 | 30 Hz | 3600 步，120 s | 捕获全部 4 个航点，距离 < 10 m |
| 复合翼 Quadplane-VTOL | `QuadplaneWaypointsEnv` | 四旋翼电机、固定翼翼面与前推电机，240 Hz | 47 / 8 | 40 Hz | 1200 步，30 s | 捕获全部 4 个航点，距离 < 15 m |
| 火箭着陆 Rocket-Landing | `RocketLandingEnv` | F9 级助推器 URDF，240 Hz，栅格舵与推力矢量 | 30 / 7 | 40 Hz | 1200 步，30 s | 接触平台，速度 < 0.02 m/s、角速度 < 0.02 rad/s、倾角 < 0.1 rad，连续满足 2 s（81 个样本）；成功不提前终止 |
| 拦截 Missile-Intercept | `MissileInterceptEnv` | 150 kg URDF，240 Hz，尾翼气动、助推—滑行发动机与运动学目标 | 34 / 3 | 60 Hz | 900 步，15 s | 任意物理子步脱靶距离 < 15 m |
| CMG 姿态控制 Satellite Attitude | `SatelliteAttitudeEnv` | 刚体 + 4/6 个 CMG，RK4 2.5 ms | 26/32 / 4/6 | 40 Hz | 1200 步，30 s | 每步检查：姿态误差 < 0.05 rad 且角速度 < 0.02 rad/s；角速度 > 10 rad/s 失败 |
| 低推力转移 Low-Thrust Transfer | `LowThrustTransferEnv` | 二体引力 + 三轴 10 mN 电推进，RK4 10 s | 8 / 3 | 300 s/步 | 166 步，约 5 圈 | 时限处终端轨道要素回报 R > 0 |
| 对接 Docking | `SpaceDockingEnv` | CWH，辛欧拉积分 1 s | 6 / 3 | 1 s/步 | 500 步，约 8.3 min | 距离 < 1 m 且速度 < 0.1 m/s |
| 交会 Rendezvous | `SpaceRendezvousEnv` | CWH，辛欧拉积分 1 s | 6 / 3 | 1 s/步 | 1000 步，约 16.7 min | 距离 < 10 m、速度 < 0.5 m/s，满足禁飞区约束 |

各任务的观测、动作、奖励、初始分布与终止条件见对应环境包的 `__init__.py` 文档字符串；公式、参数和算法说明索引见 [方法文档](docs/method/README.md)。

## 目录结构

```text
environments/                 # 任务、动力学、资产与独立训练入口
  multirotor/                 # 四旋翼航点与跟踪
  fixedwing/                  # 固定翼航点
  vtol/                       # 复合翼
  rocket/                     # 火箭着陆
  missile/                    # 拦截
  satellite/
    attitude_control/         # CMG 姿态控制
    orbit_transfer/           # 低推力转移
    rendezvous_docking/       # 交会与对接
  common/                     # plant、气动、扰动、随机数与共享工具
algorithms/
  ppo/                        # rsl-rl PPO 配置
  warmstart/                  # 行为克隆热启动
  baselines/                  # 经典控制器与注册表
scripts/                      # 通用训练、评测、汇总与可视化
docs/                         # 方法、协议与技术说明
experiments/controller_priors/ # 控制器先验对比实验
index.html                    # GitHub Pages 双语项目网页
```

## 算法与训练记录

训练入口通过 `algorithms/train_utils.py` 调用 **rsl-rl PPO**。默认策略为状态无关的 Beta 分布，动作按环境要求映射至 `[-1, 1]`；网络使用两层 128 单元 tanh MLP 和经验归一化。`build_cfg(distribution='gaussian')` 可切换为高斯策略。

`run_training(lr_cosine=True)` 可启用预热余弦学习率。未启用时使用各入口指定的固定学习率：已发布入口通常为 `3e-4`，低推力转移为 `2.5e-4`；库默认值为 `1e-3`。每次运行记录任务配置、实际 transition 数、物理步数和设备端完成指标。

## 经典控制基线

控制器位于 [baselines.py](algorithms/baselines/baselines.py)，只读取 `'policy'` 观测组，并在构造时接收环境常量；不直接读取环境内部状态缓冲区。

以下为随仓库发布的参考结果，**不是统一协议下的跨任务排名，也不是多种子均值**。副本数、种子、窗口和成功版本必须同时报告。

| 任务 | 经典基线 | 参考结果与评测条件 |
|---|---|---|
| QuadX-Waypoints | SE(3) 几何控制 + 积分修正 | 90.6%，32 副本，seed 7，800 步；256 副本 seed 7 为 100%；128 副本、seeds 7/13/21 的均值为 99.7% |
| QuadX-Waypoints | 位置 PID → 姿态 P/PD 级联 | 99.2%，256 副本，seed 7，800 步；旧的 11.9% 文件仅为修正前历史记录 |
| QuadX-Track | 匀速 KF + 固定偏移 SE(3) + 3 s 超时保持 | 82.8%，128 副本，seed 21，1200 步；可见时间 90.2%，偏移误差 0.258 m |
| Fixedwing-Waypoints | L1 横向 + TECS 纵向控制 | 97.0%，128 副本，seed 13，1500 步，99 个完成回合；未覆盖完整 120 s 时限，`complete=false` |
| Quadplane-VTOL | SE(3) + 前推电机 | 89.1%，128 副本，seed 13，1200 步；混合级联与 L1-wing 在 256 副本 seed 7 下为 26.6% / 49.6% |
| Rocket-Landing | ZEM/PN 制动 + 虚拟平台流形 + 脉冲伺服 PDG | v2：100%，128 副本，seed 13，1200 步，40 Hz，连续停稳 2 s；平均回报 903.5 |
| Satellite (CMG) | 四元数 PD + 阻尼伪逆 | 100%，128 副本，seed 13，326 步；普通 Moore–Penrose 变体为 100%（256 副本，seed 7） |
| Transfer | Gauss 单步贪心切向制导 | 100%，128 副本，seed 13，166 步，终端回报 0.0067 |
| Docking | CWH 离散 LQR + 安全盾 / 燃料降额 | 100%，128 副本，seed 13，263 步 |
| Rendezvous | CWH 双脉冲 ZEM/PN，递归求解 | 100%，128 副本，seed 13，581 步，平均回报 184.9 |
| Missile-Intercept | 增广比例导引 APN | 71.1%，128 副本，seed 13，656 步 |

默认 `baseline_eval.py` 参数为 1024 副本、20000 步、seed 7，**与上述历史文件的实际配置不同**。多种子 128 副本协议及分析见 [基线测试报告](docs/baseline_test_report.md)。详细消融与分场景入口见 [英文 README](README.md#classical-non-learned-baselines)。

## 安装

参考环境为 Linux、Python 3.10.20、CUDA 12.8 版 PyTorch 2.11.0、Genesis 1.4.4、rsl-rl-lib 5.3.0、NumPy 2.2.6、SciPy 1.15.3；参考硬件为 RTX 3090。精确依赖见 [requirements.txt](requirements.txt)。

```bash
git clone https://github.com/PaoShou-China/aerospace-rl-benchmark.git
cd aerospace-rl-benchmark
python -m pip install -r requirements.txt
```

Genesis 刚体任务使用 GPU 后端时需要 CUDA 版 PyTorch。四个纯张量航天器任务及经典控制器计算支持 CPU-only PyTorch；大气任务的环境后端要求仍须单独满足。无需为本套件额外安装 Gymnasium / PettingZoo 包装。

## 使用方法

独立训练入口的参数直接在文件中配置，不接收 CLI 参数：

```bash
python environments/multirotor/train_quadx.py
python environments/multirotor/train_quadx_track.py
python environments/fixedwing/train_fixedwing.py
python environments/vtol/train_quadplane.py
python environments/rocket/train_rocket.py
python environments/missile/train_missile.py
python environments/satellite/attitude_control/train_satellite.py
python environments/satellite/orbit_transfer/train_transfer.py
python environments/satellite/rendezvous_docking/train_docking.py
python environments/satellite/rendezvous_docking/train_rendezvous.py
```

通用 CLI 训练入口只注册 `quadx`、`fixedwing`、`rocket`、`missile`、`satellite`、`transfer`、`docking`、`rendezvous`；`quadx_track` 和 `quadplane` 请使用上面的独立入口。

```bash
# 快速训练、检查点评测、经典基线评测
python scripts/train.py quadx
python scripts/eval.py quadx
python scripts/eval.py quadx_track --out results/quadx_track/ppo
python scripts/eval.py quadplane
python scripts/baseline_eval.py quadx
python scripts/baseline_eval.py quadx --baseline cascaded_pid
python scripts/baseline_eval.py quadx_track
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg
python scripts/satellite_steering_compare.py
```

### 结果文件

| 路径 | 内容 |
|---|---|
| `results/<task>/baseline.json` | 默认经典控制器参考结果，实际副本数、种子和窗口记录于文件内 |
| `results/<task>/baselines/<controller>_seed7.json` | 任务内消融，通常为 256 副本、seed 7 |
| `results/<task>/run.json`、`config.json`、`eval.json` | 已归档的训练、配置与评测结果 |
| `results/baseline_protocol/seed{7,13,21,101}/<task>/baseline.json` | 多种子 128 副本协议 |

并非每个任务均发布了全部文件。引用实验前，应核查相应文档与实际文件是否存在。

## 运行约定

- `rew_buf` 与 `obs_buf` 为 `float32`；内部物理状态为 `float64`。
- `extras['time_outs']` 表示纯时限截断，用于 rsl-rl 的时限自举。
- `extras['terminal_observations']` 保存重置前观测。
- `completed_episodes`、`completed_successes`、`completed_returns` 在设备端累积回合统计；`complete` 为逐副本成功标志。
- 动作须位于 `env.device`；隐式主机 / 设备复制会报错。

## 风扰动

QuadX-Waypoints、Fixedwing-Waypoints、Quadplane-VTOL、Rocket-Landing 和 Missile-Intercept 支持世界系恒定风 `steady_wind`（m/s）及 Dryden 阵风。`dryden` 为预设强度的 0–1 比例，`severity` 为 `light`、`moderate` 或 `severe`。气动计算使用相对空气速度 `v_air = v - wind`。

当前入口中，quadx / fixedwing / quadplane 使用 `dryden=0.3, severity="moderate"` 且无恒定风；rocket 为 `dryden=0.0`；missile 未配置风；quadx_track 使用静风且构造函数不暴露风参数。航天器任务在真空中运行。评测从已保存配置恢复 `env_kwargs`，不会隐式改为静风。

## 火箭 benchmark v2

v2 使用独立的六球形脚垫接触资产，以**连续停稳 2 秒**为主要判据。原始 `rocket.urdf` 保留为哈希锁定的对照资产。旧 v1 的瞬时成功率 82.8% / 84.1% 不可直接与 v2 比较。

当前 main 采用 40 Hz 策略、120 Hz 控制和 240 Hz 物理频率；50 Hz 无法整除 120 Hz 控制环。版本、频率和源码哈希应随结果保存。详见 [v2 中文说明](docs/rocket_benchmark_v2_zh.md)。

```bash
python scripts/rocket_benchmark_eval.py --backend gpu --variation nominal --out results/rocket-v2-nominal
python scripts/rocket_benchmark_eval.py --backend gpu --variation shifted --out results/rocket-v2-shifted
# 无策略、发动机关闭的接触诊断，不计为策略成功率
python scripts/rocket_passive_contact.py --out results/passive-v2.json
python -m unittest discover -s tests -v
```

## 后续规划

新增任务遵循“张量环境与规格 → 经典控制器 → 独立训练入口 → 结果归档 → 文档更新”的流程，规划均为单智能体任务。

| 优先级 | 规划任务 |
|---|---|
| 1 | 动态平台火箭着陆、故障对接、固定翼自动着陆、四旋翼吊载；拦截任务已实现 |
| 2 | 地形跟随、航天器巡检、火箭返场、多旋翼穿门竞速、高超声速滑翔 |
| 3 | GEO 位置保持、火箭入轨、非合作目标消旋与捕获、移动平台精确降落 |

统一风扰动模块与字典式经典控制器注册表已实现。完整规划说明见 [英文路线图](README.md#roadmap-planned-control-tasks)；规划项不代表已实现功能。

## 参考资料与方法说明

参考文献保留原始英文题名，便于检索。任务来源包括 PyFlyt、TorchRL SatelliteEnv、低推力轨道转移论文及 SpaceGym；动力学与控制参考包括 Clohessy–Wiltshire、Wie、Vallado、Lee 等的 SE(3) 控制和 Park 等的 L1 制导；实现依赖 Genesis 与 rsl-rl。

完整的 11 条原始文献、各任务来源与对应关系见 [英文参考文献表](README.md#references)，详细数学定义见 [方法文档索引](docs/method/README.md)。本文与英文 README 共用代码入口及参考结果；评测版本差异以结果文件和任务规格为准。
