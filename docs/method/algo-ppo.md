# Proximal Policy Optimization (`algo='ppo'`)

## 1. Overview

The default learning arm of the AerospaceRL benchmark is on-policy Proximal Policy Optimization (PPO)
as implemented by the `rsl-rl` library (`rsl_rl.runners.OnPolicyRunner` + `rsl_rl.algorithms.PPO`),
driven through the shared harness `algorithms/train_utils.py:run_training`. The batched GPU
environments are adapted to the `rsl_rl.env.VecEnv` protocol by `RslVecEnv`, which collapses the
env's 5-tuple step `(obs, rew, terminated, truncated, extras)` into the 4-tuple rsl-rl expects with
`dones = terminated | truncated`; time-out bootstrapping still flows through `extras['time_outs']`.
Every per-task entry (`environments/**/train_*.py`) consumes the same configuration builder
`build_cfg`, so a run differs from the shared baseline only in the variables that entry explicitly
pins. The default policy/value network is the *flash-residual trunk* (Section 3), an
architecture-matched transplant of the FlashSAC network structure onto the PPO update path.

## 2. Objective and update rule

Each iteration collects `num_steps_per_env` $=32$ steps from each of the $N$ parallel replicas
($32N$ transitions per update, split into mini-batches of $32N/4 = 8192$ at the default $N=1024$).
Advantages are computed by generalized advantage estimation,
\begin{equation}
\delta_t = r_t + \gamma\,(1-d_t)\,V(s_{t+1}) - V(s_t), \qquad
\hat A_t = \delta_t + \gamma\lambda\,(1-d_t)\,\hat A_{t+1}, \tag{1}
\end{equation}
with $\gamma=0.99$, $\lambda=0.95$, and the time-out bootstrap of rsl-rl applied before storage,
$r_t \leftarrow r_t + \gamma\,V(s_t)\,\mathbb 1[\mathrm{time\_out}_t]$. Advantages are normalized
over the whole batch, $\hat A \leftarrow (\hat A - \bar{\hat A})/(\sigma_{\hat A}+10^{-8})$. The
actor maximizes the clipped surrogate
\begin{equation}
\mathcal L_{\mathrm{clip}}(\theta)=\hat{\mathbb E}_t\Big[\min\big(\rho_t(\theta)\hat A_t,\;
\mathrm{clip}(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon)\hat A_t\big)\Big],\qquad
\rho_t(\theta)=\exp\big(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{\mathrm{old}}}(a_t|s_t)\big), \tag{2}
\end{equation}
with clip $\epsilon=0.2$. The value loss is the clipped quadratic
\begin{equation}
L_V = \hat{\mathbb E}_t\Big[\max\big((V_\theta(s_t)-R_t)^2,\;
(V_{\mathrm{clip}}(s_t)-R_t)^2\big)\Big],\qquad
V_{\mathrm{clip}} = V_{\mathrm{old}} + \mathrm{clip}(V_\theta - V_{\mathrm{old}},-\epsilon,\epsilon), \tag{3}
\end{equation}
with $R_t=\hat A_t + V_{\mathrm{old}}(s_t)$ (rsl-rl `use_clipped_value_loss=True`), and the total
loss is
\begin{equation}
L = -\mathcal L_{\mathrm{clip}} + c_v\,L_V - c_e\,\mathcal H[\pi_\theta], \tag{4}
\end{equation}
with value coefficient $c_v = 1.0$ and entropy coefficient $c_e = 0.001$. The policy is a
state-independent Gaussian (`GaussianDistribution`, `std_type='log'`) with initial standard
deviation $\sigma_0 = 0.5$. One update performs $E=4$ epochs over $M=4$ mini-batches (16 gradient
steps per iteration) of a single Adam optimizer ($\mathrm{lr}=10^{-3}$) over the joint actor+critic
parameters, with per-module gradient-norm clipping at $1.0$ (rsl-rl default `max_grad_norm`).
`build_cfg` fixes `schedule='fixed'` (the KL-adaptive rsl-rl schedule is available but disabled by
default); `desired_kl=0.01` is retained in the config for reproducibility.

## 3. Network architecture (the FlashSAC trunk was removed)

Both actor and critic are the rsl-rl `MLPModel`: two hidden layers of 128 units with tanh
activation, a linear head, and running-statistics `EmpiricalNormalization` on the input. The
`FlashResMLPModel` arm — FlashSAC's residual `UnitLinear` trunk with RMSNorm and unit-weight
renormalization, selected by `arch='flash_residual'` in `algorithms/ppo/ppo_flashnet.py` — was
removed on 2026-10-28 by user decision: under the frozen 1024-env single-seed protocol its docking
arm scored 0.0% on the held-out evaluation where the MLP arm reached 85.0%, and its rocket arm
0.0%. `build_cfg` therefore has no `arch` / `trunk_hidden` / `critic_trunk_hidden` knob, and
`algorithms/ppo/ppo_flashnet.py` no longer exists.

## 4. Shared learning-rate schedule

For cross-algorithm A/B arms, `run_training(lr_cosine=True)` applies the identical warmup-cosine
schedule that FlashSAC's `_set_lr` uses (`algorithms/flashsac/flash_sac.py:_warmup_cosine_decay`):
\begin{equation}
\eta(s) = \eta_{\mathrm{init}} + (\eta_{\mathrm{peak}}-\eta_{\mathrm{init}})\,\frac{s}{w},\quad s<w;\qquad
\eta(s) = \eta_{\mathrm{end}} + \tfrac12(\eta_{\mathrm{peak}}-\eta_{\mathrm{end}})
\Big(1+\cos\pi\,\tfrac{s-w}{D-w}\Big),\quad s\ge w, \tag{7}
\end{equation}
with $\eta_{\mathrm{init}}=\eta_{\mathrm{peak}}$ equal to the configured learning rate (default
$10^{-3}$), cosine floor $\eta_{\mathrm{end}}=1.5\times10^{-4}$, warmup length
$w=\max(1,\lfloor 10^{-6} U\rfloor)$ and decay horizon $D=U$ for a budget of $U$ updates (at the
standard $U=1000$ the warmup segment is a single update at the peak, so the schedule is effectively
pure cosine decay). Since `schedule='fixed'` never reapplies `alg.learning_rate`, the harness
writes the optimizer's `param_groups` directly, once per chunk of $\max(1,\lfloor U/100\rfloor)$
updates, evaluating the schedule at the chunk's start — including before the first chunk, so no
update ever runs at a learning rate the schedule never prescribes. Arms under this schedule are
reported as "same formula, chunked (PPO) vs per-update (FlashSAC) application".

## 5. Fair-comparison protocol

All arms of a task share `build_cfg`'s defaults; a per-task entry overrides exactly the variables
under test (e.g. `train_quadx.py` pins lr $3\times10^{-4}$, entropy $0.01$, $\sigma_0=1.0$,
`num_steps_per_env` 100, epochs 5; `train_docking.py` pins lr $3\times10^{-4}$, $\gamma=0.999$,
entropy $0.005$). Unless pinned otherwise, runs use $N=1024$ replicas, seed 7, and a budget of
1000 updates, i.e. $1000\times32\times1024 \approx 3.28\times10^{7}$ transitions; `run.json`
records `transitions`, `train_wall_seconds`, `train_throughput_steps_per_s` and `peak_vram_gib`
alongside the episode counters, and comparisons are reported at equal interaction steps (sample
efficiency) and equal wall clock (throughput) per the frozen evaluation protocol.

## 6. Implementation

- Harness: `algorithms/train_utils.py` (`build_cfg`, `run_training`, `RslVecEnv`); algorithm:
  `rsl_rl.algorithms.PPO` (rsl-rl) over rsl-rl `MLPModel` actors/critics.
- Artifacts per run: `checkpoint.pt` + `model_{it}.pt` every 100 updates (`save_interval=100`),
  `run.json`, `config.json` under `results/<task>/`; logging via TensorBoard (`logger='tensorboard'`).

| Constant | Value |
|---|---|
| $\gamma$ / $\lambda$ / clip $\epsilon$ | 0.99 / 0.95 / 0.2 |
| steps per env / epochs / mini-batches | 32 / 4 / 4 (mini-batch $8192$ at $N{=}1024$) |
| learning rate / schedule / desired KL | $10^{-3}$ / `'fixed'` / 0.01 |
| entropy coef $c_e$ / value coef $c_v$ / init $\sigma_0$ | 0.001 / 1.0 / 0.5 (log-std) |
| grad-norm clip / value clipping | 1.0 (rsl-rl default) / enabled |
| trunk | 2 blocks, expansion 4, actor $d{=}128$ / critic $d{=}256$, `use_bn=False` |
| cosine schedule | peak $=$ lr, floor $1.5\times10^{-4}$, $w=\max(1,\lfloor10^{-6}U\rfloor)$, chunk $\lfloor U/100\rfloor$ |
| replicas / seed / updates (default budget) | 1024 / 7 / 1000 |

## 7. References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017), "Proximal Policy
  Optimization Algorithms," arXiv:1707.06347 — clipped surrogate objective.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016), "High-Dimensional
  Continuous Control Using Generalized Advantage Estimation," ICLR — GAE.
- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022), "Learning to Walk in Minutes Using
  Massively Parallel Deep Reinforcement Learning," CoRL — the rsl-rl implementation used here.
- Kim et al. (2026), "FlashSAC," arXiv:2604.04539 — source of the transplanted trunk architecture
  and its norm-control semantics.
