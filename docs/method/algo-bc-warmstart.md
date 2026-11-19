# Behavior-Cloning Warm Start (`bc_warmstart`)

**English** · [简体中文](algo-bc-warmstart.zh-CN.md)

## 1. Overview

The benchmark optionally warm-starts PPO from the classical baseline controllers instead of a
random network initialization (`algorithms/warmstart/bc_warmstart.py`). The procedure rolls the
training env under the task's `conventional_controller`, regresses the actor's deterministic
policy mean (the mean of the default bounded Beta policy) onto the collected expert actions,
optionally regresses the critic onto the empirical discounted returns-to-go of the same rollouts,
and then lets PPO refine from there. The warm start is PPO-only by design: PPO's state-independent
policy mean is the regression target, and `run_training` only supports PPO.

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
$\mathrm{act}\,(T,N,A)$ (float32), $\mathrm{rew}\,(T,N)$, $\mathrm{done}\,(T,N)$ (bool, marking
the env auto-reset at that step), plus $\mathrm{terminated}\,(T,N)$ and $\mathrm{success}\,(T,N)$
masks used by the curation step. `steps=None` rolls one full episode horizon
(`env.max_steps`). A task whose env declares no `conventional_controller` raises `RuntimeError`.

## 3. Fitting

`fit_behavior_cloning` flattens the rollout into $M = T\cdot N$ pairs and runs supervised epochs
with a fresh per-epoch permutation and mini-batches of size `batch_size`. The actor loss is the MSE
between the model's deterministic output — the policy mean (the Beta mean under the default
distribution) — and the expert action,
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
computed by a boundary-aware backward recursion (`returns_to_go`): the accumulator is cut at every
episode boundary, success (recorded by the env as a truncation) and task violations are treated as
true terminals with zero terminal value, and plain timeout tails are marked invalid and excluded
from the critic unless a bootstrap value is supplied. A second Adam optimizes the critic's
parameters. Both models' empirical input normalizers are
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
`gamma=None` ($\to$ the PPO config's $\gamma$), `fit_critic=True`. `steps=None` collects one full
episode horizon from the env (`env.max_steps`); tasks whose expert episode outlasts 512 steps
(quadx, quadx_track, fixedwing, quadplane, rocket, missile, rendezvous) use it so the expert data
covers the complete trajectory instead of only its first seconds. Because most of a long episode is
low-information steady flight, `select` keeps only the most informative transitions, following the
demonstration-curation literature: episodes are located per replica (from the done mask) and each
episode is split into 8 equal phases (broad coverage matters more than difficulty alone — *Less is
More: On Data Redundancy in VLA Training*), with the per-replica budget distributed across its
episodes and phases. Terminal-type tasks (rocket, missile, docking, rendezvous) rank steps by the
Temporal Surprise Score (TSS, the magnitude of the expert's action change — *Less is More*; action
changepoints are the keyframes of *Keyframe-Focused Visual Imitation Learning*), max-pooled over ±2
temporal neighbours so keyframes stay dense clusters; tracking/cruise tasks (quadx, quadx_track,
fixedwing, quadplane, satellite, transfer) use stratified-uniform coverage, because the redundancy
study finds broad coverage beats difficulty ranking there. A bonus on the steps just before a *true*
episode end (success or task violation, not timeouts) keeps the precision-critical terminal approach
dense (cf. *ESPADA*). Failed expert episodes are dropped by default (`drop_failed=True`, falling back
to keeping everything when a split has no success), as the curation literature recommends removing
suboptimal demonstrations; a safety floor disables filtering when fewer than 10% of episode ends
succeeded, so a near-zero-success expert cannot wipe the dataset. Critic targets are computed on the FULL trajectory first and gathered with
the same indices, so selection never biases the return-to-go. Set `select=None` to train on every
collected step; `method='uniform'` / `'tss'` explicitly overrides the per-task choice. The dataset-fit
branch (`bc_warmstart['dataset']`, used by the controller-prior study's `collect_bc.py` /
`collect_dagger.py` → `train_arm.py` pipeline) applies the same per-episode selection through
`select_informative_indices_flat` and computes boundary-aware targets with the vectorised
`returns_to_go_flat`; the episode-disjoint train/val split is preserved. `bc_val_fraction>0` withholds
that fraction of replicas as a BC validation set (monitoring only — the protocol forbids early
stopping) so curation variants can be compared on held-out expert actions via
`scripts/bc_selection_ablation.py`. At the standard $N=1024$
replicas this yields at most $M = 512\times1024 = 524{,}288$ supervised pairs, 64 mini-batches per
epoch and $100\times64 = 6400$ gradient steps per network. The per-task train entries ship with the
curated warm start enabled (`BC_WARMSTART` with `steps=None` for long-horizon tasks). Cached shared
BC checkpoints carry a recipe fingerprint (`BC_RECIPE_VERSION` + resolved selector) and are refitted
automatically when the curation recipe changes. BC inherits the frozen expert's ceiling — the audited
controller success rates are 100% on quadx/docking/rendezvous/transfer/satellite, $\sim$75% on
fixedwing and $\sim$84% on rocket — so warm-started arms are reported alongside the expert success
rate.

## 6. Documented limitation: rocket

BC warm start is known to be structurally hard on the rocket-landing task. The rocket's PDG expert
ends its suicide burn with a terminal bang-bang pulse servo profile, and MSE regression averages that
switching structure away: the distilled policy emits the time-average of the thrust command instead
of the pulse, which is not executable closed-loop. Measured 2026-10-24 in the reward-evolution
harness (`environments/rocket/evolve_reward.py`, generation g1): all BC arms finished at 0% success
with a measured BC closed-loop success rate of 0% and zero pad touches. The curation recipe now
records the full 60 s episode and emphasises the ignition/landing-burn changepoints (`method='tss'`
for rocket), but this does not remove the bang-bang regression limit; treat warm-started rocket arms
as an empirical question and check them with `scripts/bc_selection_ablation.py` / closed-loop eval,
not as a guaranteed gain. `evolve_reward.py` can still disable BC per arm (`_nb` suffix). This is a
limitation of MSE-distilling a near-bang-bang expert, not a defect of the other tasks' smooth
experts.

## 7. Implementation

- File: `algorithms/warmstart/bc_warmstart.py` (`collect_expert`, `returns_to_go` /
  `returns_to_go_flat`, `select_informative_pairs` / `..._indices_flat` /
  `..._indices_rollout`, `fit_behavior_cloning`, recipe fingerprint + sidecar);
  wiring in `algorithms/train_utils.py:run_training`; curation ablation in
  `scripts/bc_selection_ablation.py`.

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
- Wen, C. et al. (2021), "Keyframe-Focused Visual Imitation Learning," ICML — upweighting expert
  action changepoints; basis for the keyframe/TSS ranking.
- "Less is More: On Data Redundancy in VLA Training," OpenReview — temporal surprise score and the
  finding that broad coverage can beat difficulty-based frame selection.
- "ESPADA: Execution Speedup via Semantics Aware Demonstration Data Downsampling for Imitation
  Learning," arXiv:2512.07371 — preserving precision-critical phases while dropping transit frames.
- "SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning,"
  OpenReview — removing suboptimal and redundant demonstrations (basis for `drop_failed`).
