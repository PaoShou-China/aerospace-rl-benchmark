# FlashSAC (`algo='flash_sac'`)

## 1. Overview

The off-policy arm is a port of FlashSAC (Kim et al., 2026, arXiv:2604.04539) to the benchmark's
batched GPU environment protocol (`algorithms/flashsac/flash_sac.py`). FlashSAC is a soft
actor-critic variant that sharply reduces the number of gradient updates relative to SAC —
one update per interaction step — and compensates with larger norm-controlled networks, keeping
bootstrapped critic learning stable by explicitly bounding weight, feature and gradient norms. The
port is faithful to the official implementation (`github.com/holiday-robot/flashsac`); documented
non-algorithmic deviations are a device-resident ring replay buffer in the suite's own env protocol
(no gym, no CPU round-trips anywhere in the loop) and fp16 AMP wired but disabled by default.
`FlashSACRunner` exposes the same surface as rsl-rl's `OnPolicyRunner` (`learn` / `save` / `load`),
so both algorithms consume the identical batched env protocol, checkpoint naming and reporting
pipeline and results stay directly comparable.

## 2. SAC objective

The policy maximizes the entropy-regularized return
$J(\pi)=\mathbb E\big[\sum_t \gamma^t\big(r_t + \alpha\,\mathcal H(\pi(\cdot|s_t))\big)\big]$ with
$\gamma=0.99$. The actor is a tanh-squashed Gaussian with state-dependent standard deviation: given
trunk features, $u\sim\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2)$, $a=\tanh u$, with the tanh
Jacobian correction
\begin{equation}
\log\pi_\theta(a|s)=\textstyle\sum_i\Big[\log\mathcal N(u_i;\mu_i,\sigma_i)
-2\big(\log 2 - u_i - \mathrm{softplus}(-2u_i)\big)\Big], \tag{1}
\end{equation}
and a bounded log-std head, $\log\sigma = \ell_{\min} + \tfrac12(\ell_{\max}-\ell_{\min})
(1+\tanh\ell_{\mathrm{raw}})$ with $[\ell_{\min},\ell_{\max}]=[-10,\,2]$. The critic is a clipped
double ($J=2$) categorical critic fused into one batched ensemble module (`einsum` over the
ensemble dim): each member outputs log-probabilities $p_j(z|s,a)$ over $B=101$ uniformly spaced
atoms $z_i\in[-5,5]$, with scalar readout $Q_j(s,a)=\sum_i p_j(z_i|s,a)\,z_i$. The critic is
trained by cross-entropy against a projected soft Bellman target built from the target networks:
with $a'\sim\pi_\theta(\cdot|s')$ and $j^*=\arg\min_j \bar Q_j(s',a')$ selecting the min-$Q$ member
whose log-probs $\bar p_{j^*}$ define the target distribution,
\begin{equation}
\mathcal T z_i = \mathrm{clip}\Big(r + \gamma^{n}\,(1-d)\,\big(z_i - \alpha\log\pi_\theta(a'|s')\big),\,-5,\,5\Big),\qquad
m = \textstyle\sum_i \bar p_{j^*}(z_i)\,\Pi\big(\mathcal T z_i\big), \tag{2}
\end{equation}
where $\Pi$ is the two-hot projection onto the bracketing atoms (atom spacing $0.1$), $n=1$ is the
n-step fold length, and $d$ marks true terminations only (time-limit truncations keep
bootstrapping). The loss is $L_Q = -\frac1J\sum_j\hat{\mathbb E}\big[\sum_i m_i\log
p_j(z_i|s,a)\big]$, with the prediction and target forward passes evaluated on one concatenated
batch $[s;s'],[a;a']$ under batch statistics so both passes share identical BatchNorm statistics.
The actor is updated on reparameterized samples with the temperature detached,
\begin{equation}
L_\pi = \hat{\mathbb E}_{s,\,a\sim\pi_\theta}\Big[\alpha\log\pi_\theta(a|s)
- \min\big(Q_1(s,a),Q_2(s,a)\big)\Big], \tag{3}
\end{equation}
and the temperature $\alpha=\exp(\log\alpha)$ is auto-tuned by Adam on
$L_\alpha = \alpha\,(\bar{\mathcal H} - \mathcal H_{\mathrm{target}})$ toward the per-dimension
Gaussian target entropy
\begin{equation}
\mathcal H_{\mathrm{target}} = \tfrac{A}{2}\log\big(2\pi e\,\sigma_*^2\big),\qquad \sigma_* = 0.15, \tag{4}
\end{equation}
with $A$ the action dimension and $\alpha$ initialized to $0.01$.

## 3. Norm-control mechanisms

Three mechanisms bound the norms of the network (all applied to actor and both critics, at
initialization and after every optimizer step via `normalize_parameters`):

1. **Weight norm control.** Every linear layer is a bias-free `UnitLinear`, orthogonally
   initialized (gain 1), whose weight rows are re-normalized to unit $\ell_2$ norm,
   $W_{i:}\leftarrow W_{i:}/\lVert W_{i:}\rVert_2$.
2. **Feature norm control.** A `UnitBatchNorm` (momentum $0.01$, $\mathrm{eps}=10^{-5}$; batch
   statistics in training mode, running statistics at eval) follows every linear layer, and a
   `UnitRMSNorm` ($\mathrm{eps}=10^{-6}$) precedes the prediction heads. The affine scale/shift of
   each BatchNorm is constrained to $\lVert(\gamma,\beta)\rVert_2=\sqrt d$ and the RMSNorm scale to
   $\lVert g\rVert_2=\sqrt d$. Trunk blocks are residual MLPs with expansion 4:
   $h\leftarrow h + \mathrm{ReLU}\,\mathrm{BN}\,W_2\,\mathrm{ReLU}\,\mathrm{BN}\,W_1 h$.
3. **Gradient norm control** is realized through the first two: unit-norm weights and normalized
   features bound every block's Jacobian, keeping backpropagated gradient norms in the stable
   regime; no separate gradient clip is applied in the update path.

In addition, **reward normalization** keeps the critic's targets matched to its support: rewards
are scaled by the running RMS of the discounted return,
\begin{equation}
\tilde r = r \Big/ \max\Big(\sqrt{\mathrm{Var}[G]+10^{-8}},\; \max|G|/5\Big), \tag{5}
\end{equation}
where $G$ is the per-replica running discounted return and the $\max|G|/5$ floor guarantees
normalized returns stay inside the $[-5,5]$ support.

## 4. Update scheme

Each iteration performs exactly one vectorized environment step (all $N$ replicas, $N$ transitions)
plus `updates_per_step` $=1$ gradient update — the low update-to-data regime that makes FlashSAC
wall-clock fast. Transitions are stored in a device-resident ring buffer of capacity $10^{6}$
(uniform sampling, batch $2048$), with n-step return folding ($n=1$) whose chain is cut at any
episode boundary and bootstraps from the chain-end next observation with factor
$1-\mathbb 1[\mathrm{terminated\ in\ chain}]$. Since the suite's envs reset inside `step()`, the
buffer stores the pre-reset `extras['terminal_observations']` as next observation on done steps,
and `terminated = dones & ~time_outs` distinguishes true terminations (bootstrap cut) from
time-limit truncations (bootstrap kept). Updates start once the buffer holds
$\ge 10^{4}$ transitions. The critic updates every gradient step; the actor and temperature update
every second gradient step (`actor_update_period=2`). Target critic parameters follow an EMA,
$\bar\theta \leftarrow \bar\theta + \tau(\theta-\bar\theta)$ with $\tau=0.01$, applied after every
critic step (`torch._foreach_lerp_`). All three Adam optimizers (actor, critic, temperature) share
one warmup-cosine learning-rate schedule — the same `_warmup_cosine_decay` function the PPO arm
uses (eq. (7) of `algo-ppo.md`) — evaluated at every gradient update with peak
$3\times10^{-4}$, floor $1.5\times10^{-4}$, and warmup length
$w=\max(1,\lfloor 10^{-6}\times D\rfloor)$ of the $D =$ iterations $\times$ `updates_per_step`
total updates. Exploration uses *zeta noise-repeat*: each replica's Gaussian exploration noise is
held constant for $n$ consecutive interaction steps with $n$ re-drawn from a truncated zeta
distribution, $\Pr(n)\propto n^{-2}$, $n\in[1,16]$, so batched replicas produce temporally coherent
exploration instead of white noise; evaluation actions are the deterministic mean $\tanh\mu(s)$.

## 5. Implementation details

Everything runs on the env's device (CUDA in the benchmark) in float32; the actor and critic
optimizers are fused Adams on CUDA. AMP (`GradScaler`) is wired behind `use_amp` and defaults to
off. Checkpoints (`model_{it}.pt` every 100 iterations, plus the harness's final `checkpoint.pt`)
persist the actor/critic/target-critic weights, the log-temperature, all three optimizer states,
the reward normalizer state and the update counter; the replay buffer and RNG streams are *not*
persisted, so a resumed run is a parameter/optimizer warm start rather than a bit-exact mid-run
resume. `run.json` reports `transitions` $=$ updates $\times$ $N$ (one vectorized step per
iteration), `train_wall_seconds`, `train_throughput_steps_per_s` and `peak_vram_gib` on the same
footing as the PPO arm.

## 6. Implementation

- File: `algorithms/flashsac/flash_sac.py` (`UnitLinear`, `UnitBatchNorm`, `UnitRMSNorm`,
  `FlashSACActor`, `FlashSACDoubleCritic`, `NormalTanhPolicy`, `RewardNormalizer`, `ReplayBuffer`,
  `FlashSAC`, `FlashSACRunner`, `flash_sac_cfg`); wiring in `algorithms/train_utils.py:run_training`.

| Constant | Value |
|---|---|
| buffer capacity / min length / batch | $10^{6}$ / $10^{4}$ / 2048 |
| updates per step / actor period / n-step | 1 / 2 / 1 |
| $\gamma$ / target EMA $\tau$ | 0.99 / 0.01 |
| critic bins / support | 101 / $[-5,5]$ (`g_max=5`) |
| actor / critic trunk | $2\times128$ / $2\times256$ residual blocks, expansion 4 |
| temperature init / target $\sigma_*$ / log-std bounds | 0.01 / 0.15 / $[-10,2]$ |
| zeta noise $\mu$ / max hold | 2 / 16 |
| lr schedule | peak $3\times10^{-4}$ $\to$ $1.5\times10^{-4}$ cosine, $w=\max(1,\lfloor10^{-6}D\rfloor)$ |
| reward normalization | running-RMS of discounted return, floored by $\max|G|/5$ |
| AMP | wired, default off |

## 7. References

- Kim et al. (2026), "FlashSAC," arXiv:2604.04539, RSS'26 — algorithm and defaults ported here.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018), "Soft Actor-Critic," ICML,
  arXiv:1801.01290 — entropy-regularized objective and temperature auto-tuning.
- Fujimoto, S., van Hoof, H., & Meger, D. (2018), "Addressing Function Approximation Error in
  Actor-Critic Methods," ICML, arXiv:1802.09477 — clipped double Q.
- Bellemare, M. G., Dabney, W., & Munos, R. (2017), "A Distributional Perspective on Reinforcement
  Learning," ICML, arXiv:1707.06887 — categorical value distribution and two-hot projection.
- Lillicrap, T. P. et al. (2016), "Continuous Control with Deep Reinforcement Learning," ICLR,
  arXiv:1509.02971 — EMA target networks.
