<p align="center">
  <a href="https://paoshou-china.github.io/aerospace-rl-benchmark/">
    <img src="docs/assets/aerospacerl-cover.webp" alt="AerospaceRL Benchmark — 航空航天控制项目封面" width="100%">
  </a>
  <br>
  <sub>品牌插画，并非仿真截图。</sub>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://paoshou-china.github.io/aerospace-rl-benchmark/">在线展示</a> ·
  <a href="docs/technical-highlights.md">技术要点</a> ·
  <a href="docs/evaluation_protocol.md">评测协议</a> ·
  <a href="benchmark_report.md">基准报告</a>
</p>

**面向航空航天控制的 GPU 并行强化学习环境。**
十个单智能体任务——六个大气飞行器基于 URDF 刚体后端，四个航天任务基于显式张量动力学——共用同一套批量设备张量接口，不依赖 Gym / Gymnasium 封装。

所有任务遵循相同的 `step` / `reset` / 缓冲区约定，在同一奖励与成功判据下各配一个经典 GNC 基线；物理、接触、奖励、重置与随机数均为带掩码的张量核。每步唯一的 host 同步是 `bool(done.any())`。

## 快速开始

```bash
python -m pip install -r requirements.txt

python environments/multirotor/train_quadx.py   # 训练（参数写在文件内）
python scripts/eval.py quadx                     # 评测检查点
python scripts/baseline_eval.py quadx            # 评测经典基线
```

六个 Genesis 任务在 GPU 后端上需要 CUDA 版 PyTorch；四个张量动力学任务与全部经典基线也可在纯 CPU 版 PyTorch 上运行。精确版本、环境契约与可选的 Dryden 风扰接口见
[docs/environment-contract.md](docs/environment-contract.md)；完整训练入口、运行产物与目录结构见
[docs/repository-layout.md](docs/repository-layout.md)。

## 任务概览

| 任务 | 分组 | 观测 / 动作 | 控制频率 | 成功判据 |
|---|---|---|---|---|
| QuadX-Waypoints | 大气 | 37 / 4 | 40 Hz | 4/4 航点误差 < 0.2 m |
| QuadX-Track | 大气 | 25 / 4 | 40 Hz | 可见时间占比 ≥ 0.85，未坠毁 |
| Fixedwing-Waypoints | 大气 | 39 / 4 | 30 Hz | 4/4 航点误差 < 10 m |
| Quadplane-VTOL | 大气 | 47 / 8 | 40 Hz | 4/4 航点误差 < 15 m |
| Rocket-Landing | 大气 | 30 / 7 | 40 Hz | 稳定着陆，\|v\| < 0.02 m/s，倾角 < 0.1 rad |
| Missile-Intercept | 大气 | 34 / 3 | 60 Hz | 脱靶距离 < 15 m |
| Satellite attitude (CMG) | 航天 | 26 / 32 · 4 / 6 | 40 Hz | 姿态误差 < 0.05 rad，\|ω\| < 0.02 rad/s |
| Low-thrust transfer | 航天 | 8 / 3 | 300 s/步 | 末端轨道根数回报 |
| Docking | 航天 | 6 / 3 | 1 s/步 | 距离 < 1 m，速度 < 0.1 m/s |
| Rendezvous | 航天 | 6 / 3 | 1 s/步 | 距离 < 10 m，速度 < 0.5 m/s，遵守禁飞球 |

完整动力学、回合长度、观测布局与奖励分解见
[docs/tasks.md](docs/tasks.md) 与
[docs/method/README.md](docs/method/README.md)。

基线成功率及其评测条件见
[docs/baselines.md](docs/baselines.md)。数值按任务、按控制器分别报告，不构成跨任务排名。火箭 v1 数值为瞬时成功率，与 v2 的稳定停留判据不可比。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/tasks.md](docs/tasks.md) | 任务表、动力学、观测与动作布局 |
| [docs/baselines.md](docs/baselines.md) | 经典控制器结果与可比性说明 |
| [docs/method/README.md](docs/method/README.md) | 各任务、各算法的论文式方法文档（符号、方程、常量） |
| [docs/environment-contract.md](docs/environment-contract.md) | 张量契约、安装、风扰 |
| [docs/evaluation_protocol.md](docs/evaluation_protocol.md) | 评测协议与报告规则 |
| [docs/baseline_test_report.md](docs/baseline_test_report.md) | 多种子基线复分析 |
| [docs/technical-highlights.md](docs/technical-highlights.md) | 面向评审的基础设施说明 |
| [docs/repository-layout.md](docs/repository-layout.md) | 目录结构、训练入口、运行产物 |
| [docs/roadmap.md](docs/roadmap.md) | 计划任务与已有基础设施 |
| [docs/references.md](docs/references.md) | 文献与上游项目 |
| [benchmark_report.md](benchmark_report.md) | 中文基准报告 |
| [docs/rocket_benchmark_v2_zh.md](docs/rocket_benchmark_v2_zh.md) | 火箭 v2 判据与复现说明 |

## 致谢

**导师：** Lin Cheng · Shengping Gong · Peng Shi

**同学：** Wang Han · Shi Yunhan · Gong Yuxuan · Dai Yuhang

**相关项目：** [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) ·
[PyTorch](https://pytorch.org) ·
[rsl-rl](https://github.com/leggedrobotics/rsl_rl) ·
[PyFlyt](https://github.com/jbpalmer/PyFlyt) ·
[TorchRL](https://github.com/pytorch/rl) ·
[gym-satellite-trajectory](https://github.com/zampanteymedio/gym-satellite-trajectory) ·
SpaceGym（Supreeth & Mehta，CVPR 2026 Workshop on AI4Space）

任务定义与默认参数沿用 [docs/references.md](docs/references.md) 列出的公开工作；相应模型与基准的上游许可及引用要求适用。
