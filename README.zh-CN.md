<p align="center">
<a href="https://paoshou-china.github.io/aerospace-rl-benchmark/"><img src="docs/assets/aerospacerl-cover.webp" alt="AerospaceRL Benchmark" width="100%"></a>
</p>

<p align="center"><strong>航空航天控制 · GPU 并行仿真 · 强化学习</strong></p>

<p align="center"><a href="README.md">English</a> · <a href="#快速开始">快速开始</a> · <a href="#任务概览">任务概览</a> · <a href="#文档">文档</a> · <a href="#致谢">致谢</a></p>

<p align="center"><a href="https://paoshou-china.github.io/aerospace-rl-benchmark/"><strong>访问项目主页与任务演示 ↗</strong></a><br><sub>封面为品牌插画；任务演示为实际仿真回放。</sub></p>

---

**面向航空航天控制的 GPU 并行强化学习环境。**
十个单智能体任务——六个大气飞行器基于 URDF 刚体后端，四个航天任务基于显式张量动力学——共用同一套批量设备张量接口，不依赖 Gym / Gymnasium 封装。

所有任务遵循相同的 `step` / `reset` / 缓冲区约定，在同一奖励与成功判据下各配一个经典 GNC 基线；物理、接触、奖励、重置与随机数均为带掩码的张量核。每步唯一的 host 同步是 `bool(done.any())`。

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
[environment-contract.zh-CN.md](docs/environment-contract.zh-CN.md)；完整训练入口、运行产物与目录结构见
[repository-layout.zh-CN.md](docs/repository-layout.zh-CN.md)。

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

`performance=True` 时，环境步进由少量 CUDA Graph 重放完成，而不是逐个发射数千个小核（`algorithms/performance.py`）：

1. **Genesis 性能模式。** Genesis 编译针对场景特化的静态数组内核；对这些单刚体（6 自由度）飞行器，接触求解改用单内核的 *monolith* 牛顿分支。通用内核固定使用 *decomposed* 分支，其迭代循环在 SM 9.0 以下的 GPU（RTX 30/40 系列）上由主机驱动、每次迭代同步一次。两个分支求解同一问题、收敛到同一容差，轨迹在 float32 舍入层面一致，而非逐位相同。
2. **含 Genesis 内核的整子步图。** Quadrants 内核与 Torch 运算录入同一张 CUDA Graph：捕获期间把 Quadrants 自身的图发射改为普通发射，保留其主机参数缓冲所在的启动上下文，并按 Genesis 自身的节奏重放其周期性错误检查与主机步数计数。
3. **分段步进图。** 每次 `env.step` 在主机决策点（暂停恢复、重置、回合日志）处拆分为原地更新持久缓冲的分段（`_step_begin`、`_control_step`、`_step_end`、`_reset_replicas`；航天器任务为 `_step_core`），每段重放为一张图。捕获前会审计第二次预热与捕获本身：整数计数器被重放，其他主机状态变化（张量属性被替换而非原地更新）会让该分段保持即时执行并给出提示，而不会重放过期状态。
4. **可选的算子融合**（`AEROSPACE_FUSE=1`）。Inductor 将逐元素的动力学、控制、观测与积分运算融合为 Triton 内核，再由分段图录制。融合会合并乘加并可能重排归约（float32 舍入差异），并在启动时增加一次性编译。

第 2、3 层重放的正是即时执行所发射的内核：开启与关闭这两层的回放在全部十个任务上逐位相同，重构后的即时步进与此前版本也逐位相同。四个航天器任务的 PPO 损失与上一版本逐次更新完全一致。

| 环境变量 | 默认 | 作用 |
| --- | --- | --- |
| `AEROSPACE_GENESIS_PERFORMANCE` | `1` | `0` 保留 Genesis 通用内核（与此前逐位一致；第 2、3 层退回仅含 Torch 的分段） |
| `AEROSPACE_NATIVE_CAPTURE` | `1` | `0` 将 Genesis 调用留在 CUDA Graph 之外 |
| `AEROSPACE_SEGMENT_GRAPHS` | `1` | `0` 关闭分段步进图 |
| `AEROSPACE_FUSE` | `0` | `1` 开启算子融合（第 4 层） |
| `AEROSPACE_SEGMENT_DEBUG` | `0` | `1` 报告分段重新预热的原因 |

### 实测加速

相对上一加速版本（[`d7ead04`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/d7ead044bdc22ddb11b1adefa4569bdd3b7c8b5a)）的 PPO 更新吞吐，沿用其基准协议：1024 个并行环境、种子 7、各任务自带的 PPO 配置、不含 BC，取两次预热更新后五次更新的中位数（含采样、日志与 PPO 更新）。

| 任务 | `d7ead04` 秒/更新 | 本分支 秒/更新 | 加速比 | `AEROSPACE_FUSE=1` 秒/更新 | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QuadX-Waypoints | 2.415 | 0.579 | **4.17x** | 0.598 | 4.04x |
| QuadX-Track | 2.781 | 0.724 | **3.84x** | 0.786 | 3.54x |
| Fixedwing-Waypoints | 1.977 | 0.545 | **3.63x** | 0.632 | 3.13x |
| Quadplane-VTOL | 1.461 | 0.520 | **2.81x** | 0.443 | 3.30x |
| Rocket-Landing | 2.533 | 1.029 | **2.46x** | 0.862 | 2.94x |
| Missile-Intercept | 1.350 | 0.345 | **3.91x** | 0.328 | 4.12x |
| Satellite attitude (CMG) | 0.763 | 0.599 | **1.27x** | 0.337 | 2.26x |
| Low-thrust transfer | 0.150 | 0.132 | **1.14x** | 0.0810 | 1.85x |
| Docking | 0.0808 | 0.0365 | **2.21x** | 0.0492 | 1.64x |
| Rendezvous | 0.818 | 0.276 | **2.96x** | 0.299 | 2.74x |

两组在同一台机器上交替运行（各两轮，取较好的一轮，以减弱云主机上其他租户的干扰）。融合列为同一协议下的单次运行，存在约 ±15% 的波动：其卫星姿态与轨道转移任务的提升明显超出该波动，其余任务则不然。融合还会在启动时增加 1–3 分钟的一次性编译。

上一版本以同一协议相对 [`4439708`](https://github.com/PaoShou-China/aerospace-rl-benchmark/commit/4439708d3bfaf97c585be84727fd7df48eedefaa) 测得 1.88–15.29 倍，上表倍数在此基础上叠加。

从零开始（无 BC 预热）的 Quadx PPO 可能使 Genesis 接触求解器产生非有限约束力，Genesis 报告 `GenesisException: Invalid constraint forces causing 'nan'`。两种求解分支都会出现（30 次更新内，性能模式 3 个种子中 2 个、通用内核 3 个中 1 个），并非本次改动引入。完整训练的收敛时间与成功率等价性仍未验证。

测量环境：NVIDIA GeForce RTX 4080 SUPER（驱动 595.71）、Python 3.12.3、PyTorch 2.8.0+cu128、Genesis 1.4.4、Quadrants 1.3.3、rsl-rl 5.3.0。`requirements.txt` 中固定的 PyTorch 2.11 尚未针对这些加速路径验证。

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
| Satellite attitude (CMG) | 26 或 32 | 4 或 6 | 40 Hz | 姿态误差 < 0.05 rad，\|ω\| < 0.02 rad/s |
| Low-thrust transfer | 8 | 3 | 300 s/步 | 末端轨道根数回报 |
| Docking | 6 | 3 | 1 s/步 | 距离 < 1 m，速度 < 0.1 m/s |
| Rendezvous | 6 | 3 | 1 s/步 | 距离 < 10 m，速度 < 0.5 m/s，遵守禁飞球 |

完整动力学、回合长度、观测布局与奖励分解见
[tasks.zh-CN.md](docs/tasks.zh-CN.md) 与
[method/README.zh-CN.md](docs/method/README.zh-CN.md)。

基线成功率及其评测条件见
[baselines.zh-CN.md](docs/baselines.zh-CN.md)。数值按任务、按控制器分别报告，不构成跨任务排名。火箭 v1 数值为瞬时成功率，与 v2 的稳定停留判据不可比。

## 文档

| 文档 | 内容 |
|---|---|
| [tasks.zh-CN.md](docs/tasks.zh-CN.md) | 任务表、动力学、观测与动作布局 |
| [baselines.zh-CN.md](docs/baselines.zh-CN.md) | 经典控制器结果与可比性说明 |
| [method/README.zh-CN.md](docs/method/README.zh-CN.md) | 各任务、各算法的论文式方法文档（符号、方程、常量） |
| [environment-contract.zh-CN.md](docs/environment-contract.zh-CN.md) | 张量契约、安装、风扰 |
| [evaluation_protocol.zh-CN.md](docs/evaluation_protocol.zh-CN.md) | 评测协议与报告规则 |
| [baseline_test_report.zh-CN.md](docs/baseline_test_report.zh-CN.md) | 多种子基线复分析 |
| [technical-highlights.zh-CN.md](docs/technical-highlights.zh-CN.md) | 面向评审的基础设施说明 |
| [repository-layout.zh-CN.md](docs/repository-layout.zh-CN.md) | 目录结构、训练入口、运行产物 |
| [roadmap.zh-CN.md](docs/roadmap.zh-CN.md) | 计划任务与已有基础设施 |
| [references.zh-CN.md](docs/references.zh-CN.md) | 文献与上游项目 |
| [benchmark_report.zh-CN.md](benchmark_report.zh-CN.md) | 基准技术报告 |
| [rocket_benchmark_v2.zh-CN.md](docs/rocket_benchmark_v2.zh-CN.md) | 火箭 v2 判据与复现说明 |

## 致谢

> ### 特别致谢 · [百度伐谋](https://cloud.baidu.com/product/famou.html)
> 特别感谢 **百度伐谋**。[访问官方网站 ↗](https://cloud.baidu.com/product/famou.html)

感谢以下老师与同学对本项目的支持与帮助。

**老师：** 程林 · 龚胜平 · 师鹏

**同学：** 王翰 · 史运翰 · 龚钰轩 · 代雨航

**相关项目：** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) ·
[PyTorch](https://pytorch.org) ·
[rsl-rl](https://github.com/leggedrobotics/rsl_rl) ·
[PyFlyt](https://github.com/jbpalmer/PyFlyt) ·
[TorchRL](https://github.com/pytorch/rl) ·
[gym-satellite-trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) ·
SpaceGym（Supreeth & Mehta，CVPR 2026 Workshop on AI4Space）

任务定义与默认参数沿用 [references.zh-CN.md](docs/references.zh-CN.md) 列出的公开工作；相应模型与基准的上游许可及引用要求适用。
