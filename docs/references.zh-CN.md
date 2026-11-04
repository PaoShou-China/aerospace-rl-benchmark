# 参考文献

**简体中文** · [English](references.md)

本套件复现或改编了以下已发表任务；每个环境 docstring 都列出了其默认参数的来源。

## 任务与飞行器模型

1. Tai, J. K. *PyFlyt: UAV Simulation and Reinforcement Learning Environments*
   —— 任务定义、奖励 shaping 与 cf2x / fixedwing / rocket 的 URDF-YAML 模型
   （PyFlyt 0.29.0）。
2. TorchRL 贡献者。*SatelliteEnv* —— CMG 姿态任务的奖励权重、容差与复位分布。
3. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital Transfer
   using Dynamics-Agnostic Reinforcement Learning"（arXiv:2211.08272）及其配套代码
   —— 近地点抬升任务，逐参数复现。
4. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark for
   Deep Reinforcement Learning in Spacecraft Proximity Operations", CVPR 2026
   Workshop on AI4Space (CC BY 4.0) —— 对接/交会动力学、任务表与奖励权重。

## 动力学与 GNC

5. Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance System for
   Satellite Rendezvous" —— 线性化相对运动模型。
6. Wie, B. (2008). *Space Vehicle Dynamics and Control* —— CMG 动量几何与奇异度度量。
7. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* ——
   开普勒根数到笛卡尔坐标转换与密切轨道恢复。
8. Lee, T., Leok, M. & McClamroch, N. H. (2010). "Geometric Tracking Control of
   a Quadrotor UAV on SE(3)", CDC —— QuadX 几何基线。
9. Park, S., Deyst, J. & How, J. P. (2004). "A New Nonlinear Guidance Logic for
   Trajectory Tracking", AIAA GNC —— 固定翼 L1 横向制导。

## 基础设施

10. Genesis 贡献者。*Genesis* —— 刚体引擎与张量 API。
11. Legged Robotics group, ETH Zürich. *rsl_rl* —— PPO 训练后端。
