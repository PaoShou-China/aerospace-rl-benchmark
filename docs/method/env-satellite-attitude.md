# Satellite Attitude Control with Control Moment Gyros (`satellite`)

## 1. Overview

A rigid spacecraft bus carries a cluster of $N$ single-gimbal control moment
gyros (CMGs). Two cluster configurations are supported, selected by the
constructor argument `num_cmgs` (default $N=4$):

- **4-CMG pyramid.** One gimbal per bus face with skew angle
  $\beta = 54.7356^{\circ} = \arctan\sqrt{2}$, bus inertia
  $I = \operatorname{diag}(75, 60, 113)\ \mathrm{kg\,m^2}$, rotor inertia
  $0.2\ \mathrm{kg\,m^2}$ spun at $100\ \mathrm{rad/s}$, giving stored rotor
  momentum $h = 20\ \mathrm{kg\,m^2/s}$ per CMG.
- **6-CMG orthogonal cluster.** One CMG per bus face,
  $I = \operatorname{diag}(0.055, 0.055, 0.085)\ \mathrm{kg\,m^2}$, rotors at
  $200\ \mathrm{rad/s}$ with $h = 0.0125\ \mathrm{kg\,m^2/s}$ per CMG.

At every reset the bus attitude, body rate and gimbal angles are perturbed by
$10^{-3}$-scale noise about a momentum-neutral home state and a target
attitude is sampled uniformly on $\mathrm{SO}(3)$. The agent commands gimbal
rates to slew the bus onto the target while steering around internal cluster
singularities, quantified by the normalized manipulability
$m(J) = \sqrt{\det(JJ^{\top})}/h^{3}$ of the momentum Jacobian. All replicas
advance as batched tensors on a single device; there is no physics engine.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $q \in \mathbb{S}^3$ | bus attitude quaternion, scalar-first $(w, x, y, z)$ |
| $\omega \in \mathbb{R}^3$ | bus angular rate in body frame |
| $\theta \in [-\pi,\pi)^N$ | gimbal angles |
| $\delta = \dot\theta \in \mathbb{R}^N$ | commanded gimbal rates |
| $g_i$ | fixed unit gimbal axis of CMG $i$ |
| $r_i(\theta_i)$ | unit rotor-momentum direction of CMG $i$ |
| $h$ | per-rotor angular momentum magnitude (`rotor_h_dyn`) |
| $e = \lVert \log(q^{-1} q^\star) \rVert$ | attitude error angle to target $q^\star$ |
| $m(J)$ | normalized cluster manipulability |

## 3. Dynamics

Rotor axes evolve by Rodrigues rotation about the gimbal axes
($g_i \perp r_i(0)$ by construction):

$$r_i(\theta_i) = \cos\theta_i\, r_i(0) + \sin\theta_i\, (g_i \times r_i(0)). \tag{1}$$

With rotor speeds pinned and gimbal rates commanded, the reduced-order
rigid-body + CMG model is

$$I\,\dot\omega = -\,\omega \times (I\omega) \;-\; \omega \times \Big(h \sum_{i=1}^{N} r_i\Big) \;-\; h \sum_{i=1}^{N} \delta_i\, (g_i \times r_i), \tag{2}$$

$$\dot q = \tfrac{1}{2}\, q \otimes (0, \omega), \qquad \dot\theta = \delta. \tag{3}$$

Equations (2)–(3) are integrated with classical RK4 at $\mathrm{d}t =
2.5\ \mathrm{ms}$ for 10 substeps per control step, i.e. a 40 Hz agent rate
(the torchrl reference runs the same physics at 100 Hz). The quaternion is
renormalized after every substep and the gimbal angles are wrapped to
$[-\pi, \pi)$. Replicas that have already terminated or truncated are masked
frozen and consume no random numbers.

## 4. Action Space

| Property | Value |
|---|---|
| Shape | $(N,)$, $N = 4$ or $6$ |
| Range | $u \in [-1, 1]^N$ (hard clip) |
| Scaling | $\delta = 3.0 \cdot u$ rad/s per gimbal |

## 5. Observation Space

Float32, unbounded, exposed under the `'policy'` TensorDict key. Total
dimension $14 + 3N$: 26 for $N = 4$, 32 for $N = 6$. Slice ranges below are
for $N = 4$; for $N = 6$ the three gimbal blocks widen to
$[10{:}16], [16{:}22], [22{:}28]$ and the tail shifts to $28$ and $[29{:}32]$.

| Slice | Dim | Content |
|---|---|---|
| `[0:4]` | 4 | bus quaternion $q$, scalar-first $(w,x,y,z)$ |
| `[4:7]` | 3 | attitude-error log vector $\log(q^{-1} q^\star)$ (shortest arc) |
| `[7:10]` | 3 | body rate $\omega$ |
| `[10:14]` | $N$ | $\sin\theta_i$ |
| `[14:18]` | $N$ | $\cos\theta_i$ |
| `[18:22]` | $N$ | gimbal rates $\dot\theta_i$ |
| `[22:23]` | 1 | normalized manipulability $m(J) = \sqrt{\det(JJ^{\top})}/h^{3}$ |
| `[23:26]` | 3 | damped unit-torque allocation $J_u^{\top}(J_u J_u^{\top} + I_3)^{-1} e_{\log}$ |

The allocation feature is the minimum-norm, unit-damped gimbal-rate vector
that would produce torque along the attitude-error axis through the
unit-column Jacobian $J_u = J/h$; it exposes the fixed CMG geometry (a
bilinear map of the raw state) without revealing any control law.

## 6. Reward

The reward is potential-based with two smooth gates on the attitude error,

$$g(e) = \sigma\!\Big(\frac{0.8 - e}{0.15}\Big), \qquad g_n(e) = \sigma\!\Big(\frac{0.5 - e}{0.15}\Big), \tag{4}$$

and the positive distance-to-go potential

$$\Phi(s) = 30\, e \;+\; 70\, e\, g_n(e) \;+\; \frac{30\, \lVert\omega\rVert^{2}}{2 \cdot 1.0}\, g(e) \;+\; \frac{0.5}{m(J) + 0.5}. \tag{5}$$

The per-step reward is

$$r_t = \underbrace{\Phi(s_t) - \Phi(s_{t+1})}_{\text{telescoping progress}} \;+\; \underbrace{\mathbb{1}[\text{success}] \cdot 1.0 \;-\; \mathbb{1}[\text{fail}] \cdot 10.0}_{\text{one-time terminal bonus/penalty}} \;-\; 0.0005 \sum_{i=1}^{N} \delta_i^{2}. \tag{6}$$

Because the potential is re-evaluated at the fresh initial state on every
reset, the shaped sum telescopes to $\Phi_0 - \Phi_T$ regardless of path
(Ng et al. 1999), with no cross-episode leakage. Far from the target the
shaping reduces to the $30\,e$ error-progress term; inside $0.5$ rad the
$g_n$ gate steepens the error slope to ${\sim}100$ (the capture maneuver is
otherwise invisible under PPO advantage noise); inside $0.8$ rad the brake
term surcharges arriving fast and rewards shedding $\lVert\omega\rVert$. The
singularity term is bounded in $[0.5/2.05,\ 0.5/0.5]$ by the soft
denominator, avoiding the $10^{5}$-scale spikes of the reference form
$0.5/m(J)$ near singular surfaces. A full-scale command on all four gimbals
costs $0.018$ per step.

## 7. Success, Termination, Truncation

Checked on every step, on active replicas only:

- **Success (terminal, no bootstrap).** $e < 0.05$ rad **and**
  $\lVert\omega\rVert < 0.02$ rad/s. The replica is marked `truncated` and
  `complete`, receives the one-time $+1.0$ settle bonus, and stops.
- **Failure (terminal).** $\lVert\omega\rVert > 10$ rad/s (runaway spin;
  legitimate slews peak at 1–3 rad/s). Marked `terminated` with a one-time
  $-10.0$ penalty — without it the telescoping reward is failure-neutral.
- **Timeout (truncation, bootstrap).** `count >= 1200` steps (30 s).

`extras['time_outs'] = truncated ∧ ¬terminated ∧ ¬complete`, so success and
failure are treated as true terminals by the learner while only genuine
timeouts trigger the $\gamma V(s')$ bootstrap. `max_episode_length = 1202`.

## 8. Initial Distribution

- Target attitude: uniform on $\mathrm{SO}(3)$ via Shoemake sampling,
  $q^\star = (\sqrt{1-u_1}\sin 2\pi u_2,\ \sqrt{1-u_1}\cos 2\pi u_2,\ \sqrt{u_1}\sin 2\pi u_3,\ \sqrt{u_1}\cos 2\pi u_3)$, $u \sim U(0,1)^3$.
- Bus attitude: $q_0 = \operatorname{normalize}\big((1,0,0,0) + 10^{-3}\, \eta\big)$, $\eta \sim U(-1,1)^4$.
- Body rate: $\omega_0 = 10^{-3}\, \eta$, $\eta \sim U(-1,1)^3$.
- Gimbals: $\theta_0 = \theta_{\text{home}} + 10^{-3}\, \eta$, with the
  momentum-neutral home $\theta_{\text{home}} = (\pi, 0, \pi, 0)$ for $N=4$
  and $(\pi, \pi, \pi, 0, 0, 0)$ for $N=6$, chosen so that
  $\sum_i r_i(\theta_{\text{home}}) = 0$ exactly (a quiescent spacecraft
  stores ~zero momentum; parking at $\theta \approx 0$ leaves a conserved
  bias momentum that makes ${\sim}20\%$ of random targets unrestable).
- $\dot\theta_0 = 0$; the shaping potential is initialized to
  $\Phi(s_0)$.

## 9. Metrics and Extras

Per-step extras: `time_outs` (bootstrap mask, §7) and
`terminal_observations` (pre-reset observation for bootstrapping). When any
replica finishes, `extras['episode']` reports the `success_rate` over the
replicas done at that step. Cumulative buffers expose `returns`,
`completed_returns`, `completed_episodes` and `completed_successes`.

## 10. Implementation

Class `SatelliteAttitudeEnv` in
`environments/satellite/attitude_control/env.py`; registration docstring in
the package `__init__.py`. Constructor defaults: `num_envs=64`,
`backend='cpu'`, `seed=7`, `num_cmgs=4`.

| Constant | Value |
|---|---|
| control rate / integrator | 40 Hz; RK4, $\mathrm{d}t = 2.5$ ms $\times$ 10 substeps |
| `max_steps` | 1200 (30 s) |
| action clip / scale | $\pm 1$ / $\times 3.0$ rad/s |
| $k_1$ (error progress) | 30.0 |
| $k_{1b}$ (near boost) / gate / softness | 70.0 / 0.5 rad / 0.15 |
| $k_2$ (brake) / gate / softness / $\alpha_{\rm ref}$ | 30.0 / 0.8 rad / 0.15 / 1.0 |
| singularity weight / soft | 0.5 / 0.5 |
| control cost | $5\times10^{-4}\,\lVert\delta\rVert^2$ |
| success tolerances | $e < 0.05$ rad, $\lVert\omega\rVert < 0.02$ rad/s |
| settle bonus / fail threshold / fail penalty | $+1.0$ / $10$ rad/s / $-10.0$ |
| reset noise scale | $10^{-3}$ |
| 4-CMG: $I$ / rotor / $h$ | $(75,60,113)$ / $0.2\ \mathrm{kg\,m^2}$ @ 100 rad/s / 20 |
| 6-CMG: $I$ / rotor / $h$ | $(0.055,0.055,0.085)$ / $6.25\times10^{-5}\ \mathrm{kg\,m^2}$ @ 200 rad/s / 0.0125 |

## 11. References

- torchrl `SatelliteEnv` (pytorch/rl, MIT) — task reference for cluster
  geometry, physics and tolerances.
- Wie, B. (2008). *Space Vehicle Dynamics and Control*, AIAA Education
  Series — CMG momentum geometry and the manipulability singularity metric.
- Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward
  transformations — potential-based shaping telescoping argument.
- Schaub, H., Junkins, J. L. — quaternion PD + singularity-robust damped CMG
  steering, used as the conventional-controller baseline.
