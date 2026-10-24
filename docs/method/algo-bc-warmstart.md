# Behavior-Cloning Warm Start (`bc_warmstart`)

## 1. Overview

The benchmark optionally warm-starts PPO from the classical baseline controllers instead of a
random network initialization (`algorithms/warmstart/bc_warmstart.py`). The procedure rolls the
training env under the task's `conventional_controller`, regresses the actor's Gaussian mean onto
the collected expert actions, optionally regresses the critic onto the empirical discounted
returns-to-go of the same rollouts, and then lets PPO refine from there. The warm start is
PPO-only by design: FlashSAC's tanh-bounded actor head is a different regression target and its
runner is not touched (`run_training` raises `ValueError` if `bc_warmstart` is combined with
`algo='flash_sac'`).

## 2. Data collection

`collect_expert(env, steps)` owns a dedicated controller instance, constructed as
`env.conventional_controller(env)`, and drives it directly — the env itself stays a plain tensor
environment with no controller state and no controller field in `extras`. At each of the
$T=$ `steps` vectorized steps the collector computes the expert action from the current observation
under `torch.no_grad()`, executes exactly that action, and mirrors the env's auto-resets onto the
controller's stateful parts:
\begin{equation}
a_t = \pi^*(s_t),\qquad (s_{t+1}, r_t, d_t) = \mathrm{env.step}(a_t),\qquad
\mathrm{ctrl.reset\_idx}(d_t),\quad d_t = \mathrm{terminated}_t \lor \mathrm{truncated}_t, \tag{1}
\end{equation}
with a full `reset_idx` on all replicas before the first step. Because the executed action *is* the
controller's command, the controller's internal self-integrations (docking fuel bookkeeping, quadx
trim, fixedwing integrals, rendezvous ratchet) stay consistent with the visited states, and the
stored pairs $(s_t, \pi^*(s_t))$ are exact supervised targets on the expert's own closed-loop
trajectory. Expert actions are validated on collection: shape $(N, A)$ and device must match the
env, and the first step must be finite and respect the env's action bound
(`|a| ≤ clip_actions + 1e-6`, bound default $1$); violations raise immediately. The collector
returns device-resident tensors $\mathrm{obs}\,(T,N,d_{\mathrm{obs}})$,
$\mathrm{act}\,(T,N,A)$ (float32), $\mathrm{rew}\,(T,N)$ and $\mathrm{done}\,(T,N)$ (bool, marking
the env auto-reset at that step). A task whose env declares no `conventional_controller` raises
`RuntimeError`.

## 3. Fitting

`fit_behavior_cloning` flattens the rollout into $M = T\cdot N$ pairs and runs supervised epochs
with a fresh per-epoch permutation and mini-batches of size `batch_size`. The actor loss is the MSE
between the model's deterministic output — the Gaussian mean — and the expert action,
\begin{equation}
L_{\mathrm{actor}} = \frac1{|\mathcal B|}\sum_{i\in\mathcal B}
\big\lVert \mu_\theta(s_i) - a_i^* \big\rVert_2^2, \tag{2}
\end{equation}
optimized by a separate Adam over the actor's parameters at `lr`. When `fit_critic=True` the critic
is regressed onto the empirical $\gamma$-discounted returns-to-go of the same rollouts,
\begin{equation}
G_t = r_t + \gamma\,(1-d_t)\,G_{t+1},\qquad
L_{\mathrm{critic}} = \frac1{|\mathcal B|}\sum_{i\in\mathcal B}\big(V_\phi(s_i) - G_i\big)^2, \tag{3}
\end{equation}
computed by a backward recursion whose accumulator is cut at every episode boundary (`returns_to_go`),
with a second Adam over the critic's parameters. Both models' empirical input normalizers are
refit on the BC data with `update_normalization()` per batch, mirroring what PPO's own update path
does, so the warm start also calibrates the input whitening to the expert's state distribution. The
exploration standard deviation is deliberately untouched: it is PPO's exploration knob, not a BC
target. Progress is logged every `max(1, epochs//10)` epochs, and the fit returns a summary
(`transitions`, `epochs`, `batch_size`, `lr`, `fit_critic`, `wall_seconds`, `final_actor_mse`,
`final_critic_mse`).

## 4. Integration with `run_training`

The warm start is opt-in: a per-task entry passes a dict of overrides on top of `BC_DEFAULTS` via
`run_training(..., bc_warmstart={...})`, so every task's BC recipe stays directly comparable.
`gamma=None` resolves to the PPO config's discount (used only for the critic's returns-to-go
targets). Collection happens after env construction and checkpoint resume but before any PPO
update; it is automatically skipped when `resume` actually loads a checkpoint, since the checkpoint
would overwrite the warm start anyway. After fitting, the env is reset and its episode counters
(`completed_episodes`, `completed_successes`, `completed_returns`, and every
`completed_metrics` entry) are zeroed, so the counters the expert rollout inflated never leak into
PPO's reporting and `run.json` reflects PPO's own episodes only. The merged configuration and fit
report — plus the measured `collect_wall_seconds` — are persisted in `config.json` under the
`bc_warmstart` section, and BC costs are reported separately from the RL training budget (BC
collection steps are not counted among PPO's interaction steps).

## 5. Default hyperparameters

`BC_DEFAULTS`: `steps=512` vectorized env steps of expert data, `epochs=100` supervised epochs,
`batch_size=8192`, `lr=10^{-3}` (a separate Adam over the actor's — and critic's — parameters),
`gamma=None` ($\to$ the PPO config's $\gamma$), `fit_critic=True`. At the standard $N=1024$
replicas this yields $M = 512\times1024 = 524{,}288$ supervised pairs, 64 mini-batches per epoch
and $100\times64 = 6400$ gradient steps per network. The per-task train entries currently ship
with `BC_WARMSTART=None` (warm start disabled); enabling it requires only a dict, e.g.
`bc_warmstart={}` for the shared defaults. BC inherits the frozen expert's ceiling — the audited
controller success rates are 100% on quadx/docking/rendezvous/transfer/satellite, $\sim$75% on
fixedwing and $\sim$84% on rocket — so warm-started arms are reported alongside the expert success
rate.

## 6. Documented limitation: rocket

BC warm start is known to fail on the rocket-landing task. The rocket's PDG expert ends its
suicide burn with a terminal bang-bang pulse servo profile, and MSE regression averages that
switching structure away: the distilled policy emits the time-average of the thrust command instead
of the pulse, which is not executable closed-loop. Measured 2026-10-24 in the reward-evolution
harness (`environments/rocket/evolve_reward.py`, generation g1): all BC arms finished at 0% success
with a measured BC closed-loop success rate of 0% and zero pad touches. Generation g2 therefore
drops BC entirely (arms suffixed `_nb`, training from the zero-init do-nothing policy), and
`train_rocket.py` ships with `BC_WARMSTART=None`. This is a structural limitation of
MSE-distilling a near-bang-bang expert, not a defect of the other tasks' smooth experts.

## 7. Implementation

- File: `algorithms/warmstart/bc_warmstart.py` (`collect_expert`, `returns_to_go`,
  `fit_behavior_cloning`, `BC_DEFAULTS`); wiring in `algorithms/train_utils.py:run_training`.

| Constant | Value |
|---|---|
| `steps` (expert rollout length $T$) | 512 |
| `epochs` / `batch_size` | 100 / 8192 |
| `lr` (separate Adam, actor and critic) | $10^{-3}$ |
| `gamma` (critic RTG discount) | `None` $\to$ PPO config's $\gamma$ |
| `fit_critic` | `True` |
| pairs at $N{=}1024$ / gradient steps per network | 524,288 / 6400 |
| obs / act dtypes | float32, device-resident |
| PPO-only / auto-skip on resume | yes / yes |

## 8. References

- Pomerleau, D. A. (1991), "Efficient Training of Artificial Neural Networks for Autonomous
  Navigation," Neural Computation — behavior cloning.
- Ross, S., Gordon, G. J., & Bagnell, D. (2011), "A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning," AISTATS, arXiv:1011.0686 — the covariate-shift
  analysis motivating on-expert-trajectory collection.
- Açıkmeşe, B., & Ploen, S. R. (2007), "Convex Programming Approach to Powered Descent Guidance
  for Mars Landing," JGCD — the PDG family of the rocket expert referenced in Section 6.
- Schulman, J. et al. (2017), "Proximal Policy Optimization Algorithms," arXiv:1707.06347 — the
  RL stage that refines the warm start.
