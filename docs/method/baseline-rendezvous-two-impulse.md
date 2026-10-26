# Rendezvous Baseline: Receding-Horizon CWH Two-Impulse Guidance (`RendezvousTwoImpulseBaseline`)

## 1. Overview

The classical baseline for the rendezvous task is a recursively re-solved
two-impulse terminal guidance law on the Clohessy–Wiltshire–Hill (CWH)
model. Every control step, the zero-effort miss (ZEM) and zero-effort
velocity (ZEV) at a latched hold point are propagated with the **exact
symplectic state-transition matrix** of the env's own integrator (zero
model error on coasting arcs), the impulse pair nulling both terminal
errors is solved, and the pair is realized through the env's continuous
per-axis thrusters as the first step of the minimum-energy optimal
guidance program. The transfer time is seeded from an
acceleration-limited bound and ratcheted online whenever the sequential
burns no longer fit before arrival; infeasible schedules degrade to
velocity nulling. The hold point sits at 9 m on the initial line of
sight, between the 5 m keep-out sphere and the 10 m success sphere.

## 2. Interface contract

- **Observations.** Only the `'policy'` group: the raw 6-dimensional LVLH
  relative state $o=[p,\ v]$ ($x$ radial, $y$ along-track, $z$
  cross-track).
- **Environment constants at construction.** $\Delta t = 1$ s,
  `max_steps` $=1000$, $m=500$ kg, $F_{\max}=10$ N/axis — i.e.
  $a_{\mathrm{axis}}=F_{\max}/m=0.02$ m/s² per axis and
  $\Delta v_{\mathrm{step}} = a_{\mathrm{axis}}\Delta t = 0.02$ m/s per
  step — and the task radii `keep_out` $=5$ m, `pos_goal` $=10$ m. The
  mean motion is $n=\sqrt{\mu/a^{3}}=1.1314\times10^{-3}$ rad/s
  ($\mu=3.986\times10^{14}$ m³/s², $a=6.778\times10^{6}$ m).
- **Per-episode state** (cleared by `reset_idx(mask)`, called with the
  env's reset mask): the latched hold point $r_f$, the arrival schedule
  $T_{\mathrm{arr}}$ [steps], an infeasibility flag, and a step counter.
- **Action.** $u\in[-1,1]^3$, per-axis normalized thrust, $F=F_{\max}u$,
  symplectic Euler at $\Delta t=1$ s. Both "impulses" are realized as
  *finite* burns through the explicit per-axis saturation map.

## 3. Control law

**Exact discrete STM.** The unthrusted one-step map of the env's
symplectic CWH plant (Eq. (1) of the docking companion document) is
$A_d\in\mathbb{R}^{6\times6}$. The discrete generator
$\mathcal{L} = \log(A_d)/\Delta t$ is extracted once at construction via
`scipy.linalg.logm` (rejecting negative-real-axis spectra), and the
transition matrix for any per-replica horizon $T$,
\begin{equation}
\Phi(T) = \exp(\mathcal{L}\,T)
= \begin{bmatrix}\Phi_{rr} & \Phi_{rv}\\ \Phi_{vr} & \Phi_{vv}\end{bmatrix}
\in\mathbb{R}^{6\times6},
\tag{1}
\end{equation}
is computed batched on device with `torch.matrix_exp`. Exponentials of
$A_d$ propagate the unthrusted plant exactly, unlike the continuous CWH
exponential.

**Target and schedule.** On the first step of an episode the hold point
is latched on the initial line of sight,
\begin{equation}
r_f = \rho_h\,\frac{p_0}{\lVert p_0\rVert},\qquad
\rho_h = \mathrm{keep\_out} + 0.8\,(\mathrm{pos\_goal}-\mathrm{keep\_out}) = 9\ \mathrm{m},
\tag{2}
\end{equation}
with null terminal velocity. The transfer time is seeded with the
acceleration-limited straight-line bound
\begin{equation}
T_{\mathrm{arr}} \leftarrow \max\!\Big(\frac{2\sqrt{\lVert p_0\rVert/a_{\mathrm{axis}}}}{\Delta t},\ 10\Big)
\ \ \text{[steps]} .
\tag{3}
\end{equation}

**Two-impulse solve** (every step, from the current state, for the
remaining time $T_{\mathrm{go}}$). With the zero-effort errors
\begin{align}
\mathrm{ZEM} &= r_f - (\Phi_{rr}\,p + \Phi_{rv}\,v),
\tag{4a}\\
\mathrm{ZEV} &= -(\Phi_{vr}\,p + \Phi_{vv}\,v),
\tag{4b}
\end{align}
the impulse pair (position pulse now, velocity pulse at arrival) is
\begin{equation}
\Delta v_1 = \Phi_{rv}^{-1}\,\mathrm{ZEM},
\qquad
\Delta v_2 = \mathrm{ZEV} - \Phi_{vv}\,\Delta v_1 ,
\tag{5}
\end{equation}
which satisfies $r(T)=r_f$, $v(T)=0$ exactly under the discrete plant.
The receding-horizon re-solve absorbs saturation timing, float32
observation quantization, and the $\Delta t^2$ position slip of a
one-step thrust application.

**Ill-conditioning guard.** $\Phi_{rv}$ approaches a singular manifold at
$nT = k\pi$ (half-orbital multiples, far beyond this task's horizon; per
the implementation docstring, $\mathrm{cond}(\Phi_{rv})\le1.8$ for all
$T\le1000$ s). Two defenses are implemented:
\begin{align}
\text{if}\ |\det\Phi_{rv}| < \frac{10^{-10}}{n^{3}}:&\quad
\Phi_{rv} \leftarrow \Phi_{rv} + \frac{10^{-6}}{n}\,I
\quad\text{(Tikhonov)},
\tag{6a}\\
\text{if}\ \big|n\,T_{\mathrm{arr}}\Delta t - k\pi\big| < 10^{-3},\
k=\mathrm{round}\!\Big(\frac{n\,T_{\mathrm{arr}}\Delta t}{\pi}\Big):&\quad
T_{\mathrm{arr}} \leftarrow T_{\mathrm{arr}} + 8
\quad\text{(manifold snap)} .
\tag{6b}
\end{align}

**Burn-order schedule ratchet.** The two sequential burns must fit before
arrival. With burn durations
$\lVert\Delta v_i\rVert_\infty/\Delta v_{\mathrm{step}}$ and slack margin
$M=10$ steps,
\begin{equation}
\mathrm{need} = t + \frac{\lVert\Delta v_1\rVert_\infty + \lVert\Delta v_2\rVert_\infty}{\Delta v_{\mathrm{step}}} + M;
\qquad
T_{\mathrm{arr}} \leftarrow \min(\mathrm{need},\,T_{\mathrm{cap}}),
\quad T_{\mathrm{cap}} = \mathrm{max\_steps}-8 = 992 ,
\tag{7}
\end{equation}
whenever $\mathrm{need} > T_{\mathrm{arr}}$; if $\mathrm{need}$
exceeds $T_{\mathrm{cap}}$ the episode is flagged infeasible. The plan is
then re-solved with the updated $T_{\mathrm{go}}$.

**Impulse → finite-thrust mapping (OGL first step).** The impulse pair is
spread over the remaining time as the minimum-energy thrust program,
whose first step is the ZEM/ZEV optimal guidance law
\begin{equation}
a = \frac{6\,\mathrm{ZEM}}{T_{\mathrm{go}}^{2}} - \frac{2\,\mathrm{ZEV}}{T_{\mathrm{go}}},
\qquad
u = \operatorname{clip}\!\big(a/a_{\mathrm{axis}},\,-1,\,1\big)\ \text{(per axis)} .
\tag{8}
\end{equation}
The CWH gravity bias is carried inside ZEM/ZEV by the STM, and the
re-solve corrects the residual each step. The program burns early
(pulse-1 work), coasts mid-course where the profile crosses zero, and
brakes into the hold point at arrival (pulse-2 work).

**Degraded mode.** If the schedule is infeasible or has expired
($T_{\mathrm{arr}}\le t$), the guidance degrades to velocity nulling,
\begin{equation}
u = \operatorname{clip}\!\big(-v/\Delta v_{\mathrm{step}},\,-1,\,1\big) .
\tag{9}
\end{equation}

## 4. Parameters

| Parameter | Value | Source |
|---|---|---|
| hold radius $\rho_h$ | 9 m $= 5 + 0.8\,(10-5)$ | `self.hold` |
| $a_{\mathrm{axis}}$, $\Delta v_{\mathrm{step}}$ | 0.02 m/s², 0.02 m/s per step per axis | `env.f_max/mass` |
| time seed | $2\sqrt{\lVert p_0\rVert/a_{\mathrm{axis}}}/\Delta t$, floor 10 steps | `t_seed` |
| slack $M$ / cap $T_{\mathrm{cap}}$ | 10 steps / 992 steps | `ext_margin`, `t_cap` |
| Tikhonov thresholds | $10^{-10}/n^{3}$ (det), $10^{-6}/n$ (shift) | `_plan` |
| manifold snap | $|nT - k\pi| < 10^{-3}$ $\Rightarrow$ $+8$ steps | `actions` |
| OGL clamp | per-axis $\pm a_{\mathrm{axis}}$ | `actions` |
| $n$ | $1.1314\times10^{-3}$ rad/s | `env.n` |

## 5. Measured performance

- Root README, frozen baseline table: **100% success over 4096 episodes**
  across the benchmark envelope.
- Checked-in run `results/rendezvous/baseline.json` (official
  `baseline_eval` protocol: 1024 envs, seed 7, unbiased per-replica
  episode accounting): **success rate 1.0 (1024/1024)**, mean episode
  return **195.44**, 559 env steps, 10.75 s wall time.
- The frozen protocol (`docs/evaluation_protocol.md`, §1) records that
  the termination-priority fix (terminal-step keep-out violation is a
  failure, never a success) leaves this baseline unchanged: the 9 m hold
  point lies outside the 5 m keep-out sphere.

## 6. Known limitations

- The hold point is anchored to the *initial* line of sight; the guidance
  re-plans only the arrival schedule, never the target itself.
- Impulses are realized through a per-axis saturation map, so the
  finite-thrust realization is exact only up to saturation; the
  receding-horizon re-solve is the correction mechanism (documented), not
  an analytic saturation compensation.
- $\Phi_{rv}$ has a genuine singular manifold at $nT=k\pi$; it is handled
  by regularization and schedule snapping (Eqs. 6) rather than
  eliminated, and the horizon cap keeps the task far from it by design.
- Schedules that cannot fit the horizon are not rescued; the controller
  degrades to velocity nulling (Eq. 9) and the episode is forfeited.

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class
  `RendezvousTwoImpulseBaseline` (registry name `cwh_two_impulse`,
  `BUILDERS['rendezvous']`); shared STM helper `_cwh_symplectic_Ad`.
- Environment: `environments/satellite/rendezvous_docking/env.py`, class
  `SpaceRendezvousEnv` (`conventional_controller = RendezvousTwoImpulseBaseline`).
- Evaluation: `scripts/baseline_eval.py rendezvous`; scenario-grid harness
  `environments/satellite/rendezvous_docking/eval_rendezvous_baseline.py`.

## 8. References

1. Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance System
   for Satellite Rendezvous", *Journal of the Aerospace Sciences* 27(9),
   653–658 — the relative-motion model and its transition matrix.
2. Battin, R. H. *An Introduction to the Mathematics and Methods of
   Astrodynamics* (AIAA) — minimum-energy intercept guidance underlying
   the ZEM/ZEV optimal-guidance first step.
3. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark
   for Deep Reinforcement Learning in Spacecraft Proximity Operations",
   CVPR 2026 Workshop on AI4Space — the task definition.
