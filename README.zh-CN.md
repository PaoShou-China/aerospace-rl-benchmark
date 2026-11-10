<p align="center">
<a href="https://paoshou-china.github.io/aerospace-rl-benchmark/"><img src="docs/assets/aerospacerl-cover.webp" alt="AerospaceRL Benchmark" width="100%"></a>
</p>

<p align="center"><strong>航空航天控制 · GPU 并行仿真 · 强化学习</strong></p>

<p align="center"><a href="README.md">English</a> · <a href="#快速开始">快速开始</a> · <a href="#任务概览">任务概览</a> · <a href="#文档">文档</a> · <a href="#致谢">致谢</a></p>

<p align="center"><a href="https://paoshou-china.github.io/aerospace-rl-benchmark/"><strong>访问项目主页与任务演示 ↗</strong></a><br><sub>封面为品牌插画；任务演示为实际仿真回放。</sub></p>

---

**面向航空航天控制的 GPU 并行强化学习环境。**
十个单智能体任务——六个大气飞行器基于 URDF 刚体后端，四个航天任务基于显式张量动力学——共用同一套批量设备张量接口，不依赖 Gym / Gymnasium 封装。

所有任务遵循相同的 `step` / `reset` / 缓冲区约定，在同一奖励与成功判据下各配一个经典 GNC 基线；物理、接触、奖励、重置与随机数均为带掩码的张量核。回合结束判断等主机决策仍可能触发 CPU/GPU 同步；加速路径通过缓存与分段 CUDA Graph 减少重复调度。

## 快速开始

```bash
git clone https://github.com/PaoShou-China/aerospace-rl-benchmark.git
cd aerospace-rl-benchmark
python -m pip install -r requirements.txt

python environments/multirotor/train_quadx.py   # 训练（参数写在文件内）
python scripts/eval.py quadx                     # 评测检查点
python scripts/baseline_eval.py quadx            # 评测经典基线
```

六个 Genesis 任务在 GPU 后端上需要 CUDA 版 PyTorch；四个张量动力学任务与全部经典基线也可在纯 CPU 版 PyTorch 上运行。精确版本、环境契约与可选的 Dryden 风扰接口见
[environment-contract.zh-CN.md](docs/environment-contract.zh-CN.md)；完整训练入口、运行产物与目录结构见下方[仓库结构](#仓库结构)一节。

## CUDA 加速

十个任务均支持可选的执行加速，覆盖专家数据采集、行为克隆（BC）与 PPO 训练。任务方程、奖励、模型精度与训练预算保持原有配置；代码中没有按 GPU 型号或显存容量分支。

在仓库根目录通过任务原有的 `main` 函数开启训练加速（以 Rocket-Landing 为例）：

```python
from environments.rocket.train_rocket import main
main(performance=True, out_dir="results/cuda_rocket", clean=False)
```

评估得到的检查点：

```bash
python scripts/eval.py rocket --performance --out results/cuda_rocket
```

这些选项默认关闭；不受支持的 PPO 配置沿用上游更新实现。

### 执行层

`performance=True` 时，环境步进由少量 CUDA Graph 重放完成（`algorithms/performance.py`）：

1. **Genesis 性能模式。** Genesis 编译针对场景特化的静态数组内核；对这些单刚体（6 自由度）飞行器，接触求解改用单内核的 *monolith* 牛顿分支。通用内核固定使用 *decomposed* 分支，其迭代循环在 SM 9.0 以下的 GPU（RTX 30/40 系列）上由主机驱动、每次迭代同步一次。两个分支使用同一物理问题和求解容差，但求解路径不同，可能产生浮点差异；完整轨迹与训练行为的等价性不能仅由容差相同推出。
2. **含 Genesis 内核的整子步图。** Quadrants 内核与 Torch 运算录入同一张 CUDA Graph：捕获期间把 Quadrants 自身的图发射改为普通发射，保留其主机参数缓冲所在的启动上下文，并按 Genesis 自身的节奏重放其周期性错误检查与主机步数计数。
3. **分段步进图。** 每次 `env.step` 在主机决策点（暂停恢复、重置、回合日志）处拆分为原地更新持久缓冲的分段（`_step_begin`、`_control_step`、`_step_end`、`_reset_replicas`；航天器任务为 `_step_core`），每段重放为一张图。捕获前会审计第二次预热与捕获本身：整数计数器被重放，其他主机状态变化（张量属性被替换而非原地更新）会让该分段保持即时执行并给出提示，而不会重放过期状态。
4. **可选的算子融合**（`AEROSPACE_FUSE=1`）。Inductor 将逐元素的动力学、控制、观测与积分运算融合为 Triton 内核，再由分段图录制。融合会合并乘加并可能重排归约（float32 舍入差异），并在启动时增加一次性编译。

| 环境变量 | 默认 | 作用 |
| --- | --- | --- |
| `AEROSPACE_GENESIS_PERFORMANCE` | `1` | `0` 保留 Genesis 通用内核（与此前逐位一致；第 2、3 层退回仅含 Torch 的分段） |
| `AEROSPACE_NATIVE_CAPTURE` | `1` | `0` 将 Genesis 调用留在 CUDA Graph 之外 |
| `AEROSPACE_SEGMENT_GRAPHS` | `1` | `0` 关闭分段步进图 |
| `AEROSPACE_FUSE` | `0` | `1` 开启算子融合（第 4 层） |
| `AEROSPACE_SEGMENT_DEBUG` | `0` | `1` 报告分段重新预热的原因 |

## 任务概览

### 大气飞行 · 6 个任务

| 任务 | 观测维度 | 动作维度 | 策略频率 / 步长 | 成功判据摘要 |
|---|---:|---:|---|---|
| QuadX-Waypoints | 37 | 4 | 40 Hz | 4/4 航点误差 < 0.2 m |
| QuadX-Track | 25 | 4 | 40 Hz | 可见时间占比 ≥ 0.85，未坠毁 |
| Fixedwing-Waypoints | 39 | 4 | 30 Hz | 4/4 航点误差 < 10 m |
| Quadplane-VTOL | 47 | 8 | 40 Hz | 4/4 航点误差 < 15 m |
| Rocket-Landing | 30 | 7 | 40 Hz | 连续停稳 2 s；完整 v2 判据见任务说明 |
| Missile-Intercept | 34 | 3 | 60 Hz | 脱靶距离 < 15 m |

### 航天器 · 4 个任务

| 任务 | 观测维度 | 动作维度 | 策略频率 / 步长 | 成功判据摘要 |
|---|---:|---:|---|---|
| Satellite attitude (CMG) | 26 | 4 | 40 Hz | 姿态误差 < 0.05 rad，\|ω\| < 0.02 rad/s |
| Low-thrust transfer | 8 | 3 | 300 s/步 | 末端轨道根数回报 |
| Docking | 6 | 3 | 1 s/步 | 距离 < 1 m，速度 < 0.1 m/s |
| Rendezvous | 6 | 3 | 1 s/步 | 距离 < 10 m，速度 < 0.5 m/s，遵守禁飞球 |

完整动力学、回合长度、观测布局与奖励分解见
[tasks.zh-CN.md](docs/tasks.zh-CN.md) 与
[method/README.zh-CN.md](docs/method/README.zh-CN.md)。

基线成功率及其评测条件见
[baselines.zh-CN.md](docs/baselines.zh-CN.md)。数值按任务、按控制器分别报告，不构成跨任务排名。火箭 v1 数值为瞬时成功率，与 v2 的稳定停留判据不可比。

## 仓库结构

```text
environments/               # 任务定义
├── multirotor/             # QuadX-Waypoints + QuadX-Track
├── fixedwing/              # Fixedwing-Waypoints
├── vtol/                   # Quadplane-VTOL
├── rocket/                 # Rocket-Landing
├── missile/                # Missile-Intercept
├── satellite/              # attitude_control, orbit_transfer, rendezvous_docking
└── common/                 # plant, dynamics, aero, contacts, disturbances
algorithms/                 # 动作生成
├── train_utils.py          # 各入口共用的 run_training()
├── ppo/                    # rsl-rl PPO 配置构造器
├── warmstart/              # 从基线做行为克隆 warm start
├── residual/               # residual-RL 动作包装
└── baselines/              # 经典 GNC 控制器，每任务一个
experiments/                # controller-prior 研究（BC、residual、增益调度）
scripts/                    # 通用 train / eval / baseline / sweep / summarize 工具
docs/                       # 方法级文档、协议
showcase/                   # GitHub Pages 落地页用的回放视频
tests/                      # 单元、环境契约与基准测试
```

任务级深度 harness 与各自环境同目录（如
`environments/missile/eval_missile_scenarios.py`、
`environments/rocket/rocket_pdg_test.py`、
`environments/satellite/rendezvous_docking/eval_docking_lqr.py`）。

### 训练入口

训练入口自包含：所有参数硬编码在文件中，无 CLI 参数。修改常量后运行：

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

评测 checkpoint 或经典基线：

```bash
python scripts/eval.py quadx                # 恢复 results/<task>/config.json
python scripts/baseline_eval.py quadx       # 任意已注册任务
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg
python scripts/aggregate_5seed.py           # 多种子协议汇总
```

旧版 CLI 聚合器仍可用于快速扫描
（`python scripts/train.py quadx`；注册 quadx、fixedwing、rocket、missile、
satellite、transfer、docking、rendezvous —— `quadx_track` 与 `quadplane`
使用上面的任务级入口）。

### 运行产物

每次运行写入 `results/<task>/`：`run.json`（配置、真实转场数、物理步计数、
设备端完成指标）、`config.json`、`checkpoint.pt`，评测后还有 `eval.json`。
基线产物为 `baseline.json`（默认控制器）与
`baselines/<controller>_seed7.json`（任务内消融）。

## 文档

| 文档 | 内容 |
|---|---|
| [tasks.zh-CN.md](docs/tasks.zh-CN.md) | 任务表、动力学、观测与动作布局 |
| [baselines.zh-CN.md](docs/baselines.zh-CN.md) | 经典控制器结果（含多种子数据与复现命令） |
| [method/README.zh-CN.md](docs/method/README.zh-CN.md) | 各任务、各算法的论文式方法文档（符号、方程、常量） |
| [environment-contract.zh-CN.md](docs/environment-contract.zh-CN.md) | 张量契约、安装、风扰 |
| [evaluation_protocol.zh-CN.md](docs/evaluation_protocol.zh-CN.md) | 评测协议与报告规则 |
| [benchmark_report.zh-CN.md](benchmark_report.zh-CN.md) | 基准技术报告 + 基础设施/可复现性附录 + 参考文献 |

## 致谢

> ### 特别致谢 · [百度伐谋](https://cloud.baidu.com/product/famou.html)
> 特别感谢 **百度伐谋**。[访问官方网站 ↗](https://cloud.baidu.com/product/famou.html)

感谢以下老师与同学对本项目的支持与帮助。

**老师：** 程林 · 龚胜平 · 师鹏

**同学：** 王翰 · 史运翰 · 龚钰轩 · 代雨航 · 祁科羽

**相关项目：** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) ·
[PyTorch](https://pytorch.org) ·
[rsl-rl](https://github.com/leggedrobotics/rsl_rl) ·
[PyFlyt](https://github.com/jbpalmer/PyFlyt) ·
[TorchRL](https://github.com/pytorch/rl) ·
[gym-satellite-trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) ·
SpaceGym（Supreeth & Mehta，CVPR 2026 Workshop on AI4Space）

任务定义与默认参数沿用基准技术报告的[参考文献](benchmark_report.zh-CN.md#8-参考文献)一节列出的公开工作；相应模型与基准的上游许可及引用要求适用。
