# 近端策略优化（`algo='ppo'`）

**简体中文** · [English](algo-ppo.md)

## 1. 概述

AerospaceRL 基准的默认学习臂是 `rsl-rl` 库实现的 on-policy 近端策略优化（PPO）（`rsl_rl.runners.OnPolicyRunner` + `rsl_rl.algorithms.PPO`），经共享 harness `algorithms/train_utils.py:run_training` 驱动。批量 GPU 环境由 `RslVecEnv` 适配到 `rsl_rl.env.VecEnv` 协议：把环境 5 元组步进 `(obs, rew, terminated, truncated, extras)` 折叠为 rsl-rl 期望的 4 元组，`dones = terminated | truncated`；超时 bootstrap 仍经 `extras['time_outs']` 流动。每个任务入口（`environments/**/train_*.py`）共用同一配置构建器 `build_cfg`，故某次运行相对共享基线只在入口显式固定的变量上不同。默认策略/价值网络为 rsl-rl `MLPModel`（第 3 节）。

## 2. 目标与更新规则

每次迭代从 $N$ 个并行副本各采 `num_steps_per_env` $=32$ 步（每次更新 $32N$ 转场，默认 $N=1024$ 下划分为 $32N/4=8192$ 的 mini-batch）。优势由广义优势估计计算：
\begin{equation}
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\qquad
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}, \tag{1}
\end{equation}
$\gamma=0.99$、$\lambda=0.95$，存储前应用 rsl-rl 超时 bootstrap $r_t\leftarrow r_t+\gamma V(s_t)\mathbb 1[\mathrm{time\_out}_t]$。优势在全批归一化 $\hat A\leftarrow(\hat A-\bar{\hat A})/(\sigma_{\hat A}+10^{-8})$。actor 最大化裁剪代理目标
\begin{equation}
\mathcal L_{\mathrm{clip}}(\theta)=\hat{\mathbb E}_t\big[\min(\rho_t(\theta)\hat A_t,\mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)\big],\quad
\rho_t(\theta)=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{\mathrm{old}}}(a_t|s_t)), \tag{2}
\end{equation}
裁剪 $\epsilon=0.2$。价值损失为裁剪二次型
\begin{equation}
L_V=\hat{\mathbb E}_t\big[\max((V_\theta(s_t)-R_t)^2,(V_{\mathrm{clip}}(s_t)-R_t)^2)\big],\quad
V_{\mathrm{clip}}=V_{\mathrm{old}}+\mathrm{clip}(V_\theta-V_{\mathrm{old}},-\epsilon,\epsilon), \tag{3}
\end{equation}
$R_t=\hat A_t+V_{\mathrm{old}}(s_t)$（rsl-rl `use_clipped_value_loss=True`），总损失
\begin{equation}
L=-\mathcal L_{\mathrm{clip}}+c_vL_V-c_e\mathcal H[\pi_\theta], \tag{4}
\end{equation}
价值系数 $c_v=1.0$、熵系数 $c_e=0.001$。默认策略为有界、状态无关 Beta 分布（`BetaDistribution`，`action_range=(-1.0,1.0)`）：MLP 对每动作输出 $\alpha,\beta>1$，$Beta(\alpha,\beta)$ 样本线性缩放到 $[-1,1]$，对数概率携带缩放雅可比。`build_cfg` 也接受 `distribution='gaussian'`，选择无界 `GaussianDistribution`（`std_type='log'`），初始标准差 $\sigma_0=$`init_std`；Beta 分支忽略 `init_std`，后者仅为向后兼容保留。一次更新对联合 actor+critic 参数做单个 Adam（$\mathrm{lr}=10^{-3}$）的 $E=4$ epochs × $M=4$ mini-batches（每迭代 16 梯度步），逐模块梯度范数裁剪 1.0（rsl-rl 默认 `max_grad_norm`）。`build_cfg` 固定 `schedule='fixed'`（KL 自适应 rsl-rl 调度可用但默认禁用）；`desired_kl=0.01` 保留在配置中以利复现。

## 3. 网络结构

actor 与 critic 均为 rsl-rl `MLPModel`：两个 128 单元 tanh 隐层、线性头、输入端 running-statistics `EmpiricalNormalization`。

## 4. 共享学习率调度

`run_training(lr_cosine=True)` 应用 warmup-cosine 调度（`algorithms/train_utils.py` 中 `_warmup_cosine_decay`）：
\begin{equation}
\eta(s)=\eta_{\mathrm{init}}+(\eta_{\mathrm{peak}}-\eta_{\mathrm{init}})\frac{s}{w},\ s<w;\qquad
\eta(s)=\eta_{\mathrm{end}}+\tfrac12(\eta_{\mathrm{peak}}-\eta_{\mathrm{end}})\Big(1+\cos\pi\frac{s-w}{D-w}\Big),\ s\ge w, \tag{7}
\end{equation}
$\eta_{\mathrm{init}}=\eta_{\mathrm{peak}}$ 为配置学习率（默认 $10^{-3}$），余弦地板 $\eta_{\mathrm{end}}=1.5\times10^{-4}$，warmup 长度 $w=\max(1,\lfloor10^{-6}U\rfloor)$、衰减时域 $D=U$（预算 $U$ 次更新；标准 $U=1000$ 时 warmup 段仅一次更新，故调度实际为纯余弦衰减）。因 `schedule='fixed'` 从不重设 `alg.learning_rate`，harness 直接写优化器 `param_groups`，每 $\max(1,\lfloor U/100\rfloor)$ 次更新写一次，在块起始求值调度——包括首块之前，故任何更新都不会运行在调度从未规定的学习率上。

## 5. 公平比较协议

任务各臂共享 `build_cfg` 默认值；逐任务入口只覆写被测变量（如 `train_quadx.py` 固定 lr $3\times10^{-4}$、熵 0.01、`num_steps_per_env` 100、epochs 5；其 `init_std=1.0` 在默认 Beta 策略下无效；`train_docking.py` 固定 lr $3\times10^{-4}$、$\gamma=0.999$、熵 0.005）。已发布入口均不覆写 `distribution`，故当前所有运行使用 Beta 策略（Gaussian 时代的 `sigma_pin` / `lr_sigma_coupling` 旋钮已于 2026-11-02 从 `train_rendezvous.py` 退役，不再传入）。除非另行固定，运行使用 $N=1024$ 副本、seed 7、1000 次更新预算，即 $1000\times32\times1024\approx3.28\times10^7$ 转场；`run.json` 记录 `transitions`、`train_wall_seconds`、`train_throughput_steps_per_s`、`peak_vram_gib` 及回合计数，比较按冻结评测协议在等交互步（样本效率）与等 wall clock（吞吐）下报告。

## 6. 实现

- Harness：`algorithms/train_utils.py`（`build_cfg`、`run_training`、`RslVecEnv`）；算法：`rsl_rl.algorithms.PPO`（rsl-rl）配 rsl-rl `MLPModel` actor/critic。
- 每次运行产物：`checkpoint.pt` + 每 100 次更新的 `model_{it}.pt`（`save_interval=100`）、`run.json`、`config.json`，位于 `results/<task>/`；TensorBoard 日志（`logger='tensorboard'`）。

| 常量 | 值 |
|---|---|
| $\gamma$ / $\lambda$ / 裁剪 $\epsilon$ | 0.99 / 0.95 / 0.2 |
| 每 env 步数 / epochs / mini-batches | 32 / 4 / 4（$N{=}1024$ 时 mini-batch $8192$） |
| 学习率 / 调度 / desired KL | $10^{-3}$ / `'fixed'` / 0.01 |
| 熵系数 $c_e$ / 价值系数 $c_v$ / 策略分布 | 0.001 / 1.0 / `BetaDistribution` 缩放到 $[-1,1]$（`init_std` 仅对 `distribution='gaussian'` 生效） |
| 梯度范数裁剪 / 价值裁剪 | 1.0（rsl-rl 默认）/ 启用 |
| 网络 | 2×128 tanh MLP，running-stats `EmpiricalNormalization` |
| 余弦调度 | peak=lr，floor $1.5\times10^{-4}$，$w=\max(1,\lfloor10^{-6}U\rfloor)$，块 $\lfloor U/100\rfloor$ |
| 副本 / seed / 更新（默认预算） | 1024 / 7 / 1000 |

## 7. 参考

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017), "Proximal Policy Optimization Algorithms," arXiv:1707.06347 —— 裁剪代理目标。
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016), "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR —— GAE。
- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022), "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning," CoRL —— 此处使用的 rsl-rl 实现。
