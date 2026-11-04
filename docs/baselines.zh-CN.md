# 经典基线结果

**简体中文** · [English](baselines.md)

每个任务都附带一个非学习的 GNC 基线
（[algorithms/baselines/](../algorithms/baselines/baselines.py)），与策略共享同一环境、
同一奖励与同一成功门槛。控制器只读取 `'policy'` 观测组，构造时读取环境常数，
从不访问内部状态缓冲区。

| 任务 | 基线 | 参考成功率 |
|---|---|---|
| QuadX-Waypoints | SE(3) 几何控制 + 积分配平 | 90.6%（32 副本，seed 7）；多种子协议均值 99.7%（128 副本 × seed 7/13/21；留出 seed 101 = 100%） |
| QuadX-Waypoints | 位置 PID → 姿态级联 | 99.2%（256 副本，seed 7，0 碰撞 / 0 出界） |
| QuadX-Track | CV 卡尔曼滤波 + 固定偏移 SE(3) | 82.8%（128 副本，seed 21）；可见时间 90.2%，间距 0.258 m |
| Fixedwing-Waypoints | L1 横向 + TECS 纵向自驾 | 97.0%（128 副本，seed 13；窗口在 120 s 时限前结束） |
| Quadplane-VTOL | SE(3) + 前向推进器 | 89.1%（128 副本，seed 13） |
| Rocket-Landing | PDG：ZEM/PN 制动 + 虚拟平台流形 + 脉冲伺服 | v2 判据下 100%（40 Hz，连续停稳 2 s） |
| Satellite (CMG) | 四元数 PD + Schaub–Junkins 阻尼伪逆 | 100%（128 副本，seed 13） |
| Transfer | Gauss 单步贪心切向制导 | 100%（128 副本，seed 13） |
| Docking | CWH 离散 LQR + 护栏/燃料降额 | 100%（128 副本，seed 13） |
| Rendezvous | 两脉冲 CWH ZEM/PN，递归重解 | 100%（128 副本，seed 13） |
| Missile-Intercept | 增广比例导引（N = 15）→ 体轴角速率指令 | 71.1%（128 副本，seed 13） |

表中数字来自随仓库发布的 `results/<task>/baseline.json`；每个文件记录自己的副本数、
种子与步数窗口。任务内消融（256 副本，seed 7）位于 `results/<task>/baselines/`，
多种子协议复分析见 [baseline_test_report.zh-CN.md](baseline_test_report.zh-CN.md)。

两点解读值得保留：

- 这些是参照系，不是稻草人。固定翼与火箭基线保留了有记录的残余失败
  （末端极限环、权限饱和制动），学习策略正是被期望去攻击这些失败。
- 旧笔记中的火箭 v1 数字（82.8% / 84.1%）是**瞬时**成功率，与 v2 的稳定停稳判据
  不可比。背景与复现命令见 [rocket_benchmark_v2.zh-CN.md](rocket_benchmark_v2.zh-CN.md)。

成功率按任务、按控制器分别报告。它们不构成跨任务排名：任务难度、观测预算与成功判据
各不相同，数字也来自不同的副本数与种子。
