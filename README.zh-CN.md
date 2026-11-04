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
