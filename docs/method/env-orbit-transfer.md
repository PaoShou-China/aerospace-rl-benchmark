# Low-Thrust Perigee-Raising Orbital Transfer (`transfer`)

## 1. Overview

A $100\ \mathrm{kg}$ medium-Earth-orbit satellite equipped with three
orthogonal thruster pairs ($10\ \mathrm{mN}$ per axis, continuous,
$I_{\mathrm{sp}} = 4000\ \mathrm{s}$ electric propulsion) fires for 166
control steps of $300\ \mathrm{s}$ each ($49{,}800\ \mathrm{s} \approx 13.8$
h, about five revolutions) to raise the pericenter radius of an initial
$a = 10{,}000\ \mathrm{km}$, $e = 0.1$ orbit — perigee radius
$r_p = 9000\ \mathrm{km}$, apogee radius $r_a = 11{,}000\ \mathrm{km}$ —
while keeping the apocenter radius fixed and propellant use low. The initial
mean anomaly is resampled uniformly on $[-\pi, \pi]$ at every reset; the
remaining elements are fixed at $i = 60^{\circ}$, $\Omega = 120^{\circ}$,
$\omega = 240^{\circ}$. The task reproduces the `PerigeeRaisingEnv` of
gym-satellite-trajectory parameter-for-parameter. All replicas advance as
batched tensors on a single device; there is no physics engine.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $r, v \in \mathbb{R}^3$ | position [m] and velocity [m/s] in GCRF |
| $m$ | remaining mass [kg] |
| $\mu$ | Earth gravitational parameter, $3.986004418\times10^{14}\ \mathrm{m^3/s^2}$ |
| $u \in [-1,1]^3$ | per-axis thrust command |
| $f_{\max} = 0.01\ \mathrm{N}$ | per-axis thrust magnitude |
| $r_a, r_p$ | osculating apogee / perigee radii [m] |
| $r_{a0}, r_{p0}$ | apogee / perigee radii at the reset state |
| $a, e, i, \Omega, \omega, M$ | classical Keplerian elements |

## 3. Dynamics

The equations of motion are two-body point-mass gravity plus constant thrust
held over each control step, with Tsiolkovsky mass drain:

$$\dot r = v, \qquad \dot v = -\frac{\mu}{\lVert r \rVert^{3}}\, r + \frac{f_{\max}}{m}\, u, \qquad \dot m = -\frac{f_{\max}\, \lVert u \rVert_{1}}{I_{\mathrm{sp}}\, g_0}, \tag{1}$$

with $g_0 = 9.80665\ \mathrm{m/s^2}$. Note the mass drain is proportional to
the $\ell_1$ norm of the command: every axis fired simultaneously draws
propellant. The 7-dim state $[r, v, m]$ is integrated with classical RK4 at
$h = 10\ \mathrm{s}$ over 30 substeps per $300\ \mathrm{s}$ control step (the
reference integrates the same ODE with Orekit's adaptive Dormand-Prince
8(5,3); the fixed-step per-revolution phase error stays far below the reward
granularity). Masked (inactive) replicas are frozen.

The initial Cartesian state is obtained from the fixed elements and the
sampled mean anomaly by solving Kepler's equation
$M = E - e\sin E$ for the eccentric anomaly with 8 fixed Newton–Raphson
iterations from $E_0 = M$,

$$E \leftarrow E - \frac{E - e\sin E - M}{1 - e\cos E}, \tag{2}$$

followed by the perifocal state and the $R_3(\Omega) R_1(i) R_3(\omega)$
rotation into GCRF (Vallado). Osculating radii are recovered from any
Cartesian state via $a = \left(2/\lVert r\rVert - \lVert v\rVert^2/\mu\right)^{-1}$
and the eccentricity vector, $r_a = a(1+e)$, $r_p = a(1-e)$.

## 4. Action Space

| Property | Value |
|---|---|
| Shape | $(3,)$, one entry per body axis |
| Range | $u \in [-1, 1]^3$ (hard clip) |
| Effect | per-axis thrust $f_{\max} u_i$; acceleration $f_{\max} u_i / m$ ($\approx 10^{-4}\ \mathrm{m/s^2}$ per axis at $m_0$) |

A single axis held at full scale for the whole episode delivers
${\approx} 5\ \mathrm{m/s}$ of $\Delta v$ at $100\ \mathrm{kg}$.

## 5. Observation Space

Float32, unbounded raw state under the `'policy'` TensorDict key, exactly as
the reference (the reference declares $\pm 1.1\times$ characteristic bounds —
$[T, r_a, r_a, r_a, v_{\max}, v_{\max}, v_{\max}, m_0]$ — but never
normalizes):

| Slice | Dim | Content |
|---|---|---|
| `[0:1]` | 1 | elapsed mission time $t = k \cdot 300\ \mathrm{s}$ |
| `[1:4]` | 3 | position $r$ in GCRF [m] |
| `[4:7]` | 3 | velocity $v$ in GCRF [m/s] |
| `[7:8]` | 1 | remaining mass $m$ [kg] |

## 6. Reward

The task-level objective is the reference's sparse terminal reward, granted
once at the step limit,

$$R = 10^{-4}\,\big( -\lvert \Delta r_a \rvert + \Delta r_p + 2\times10^{5}\, \Delta m \big), \tag{3}$$

with radii in meters and mass in kg: $+0.1$ per km of perigee radius raised,
$-0.1$ per km of absolute apogee-radius drift (either direction is
penalized), and, since $\Delta m = m - m_0 \le 0$ decreases as propellant is
spent, $-20$ per kg of propellant (the reference's Table I weights).

The training reward is the exact potential-based telescoping decomposition of
(3). With

$$\Phi(s) = 10^{-4}\,\big( -\lvert r_a - r_{a0} \rvert + (r_p - r_{p0}) + 2\times10^{5}\,(m - m_0) \big), \tag{4}$$

which is identically zero at every fresh initial condition, the per-step
reward is

$$r_t = \Phi(s_{t+1}) - \Phi(s_t), \tag{5}$$

so the episode sum equals the terminal $R$ exactly (the horizon is fixed and
$\gamma = 1$, so the PBRS return is undiscounted and exactly invariant) while
every step receives immediate, action-attributable credit: burning prograde
near apogee raises $r_p$ ($+r_t$); burning elsewhere raises
$\lvert r_a - r_{a0} \rvert$ or drains $m$ ($-r_t$). The terminal $R$ itself
is retained as a metrics-only quantity (`terminal_reward`), as the success
criterion, and in the episode log. This decomposition matters at the
standard 64-env batch, where a sparse terminal grant reaches the learner
through only ${\sim}12$ episode boundaries per PPO update and does not train.

## 7. Success, Termination, Truncation

- **Timeout / grant.** `count >= 166` steps ($49{,}800\ \mathrm{s}$). If the
  replica is active and unviolated, the episode ends as `truncated` and the
  terminal $R$ is recorded.
- **Success.** Task-level success is the reference objective evaluated at the
  final orbit: `complete = grant ∧ (R > 0)`. (The reference reports
  `is_success = True` at any done, which is vacuous under timeout-only
  termination; here `completed_successes` counts episodes with positive
  terminal reward.)
- **Defensive terminations** (marked `terminated`; unreachable at the
  reference $10\ \mathrm{mN}$ thrust, guarding the metric against
  hyperparameter changes): impact, $r_p < 6{,}378{,}137\ \mathrm{m}$ (WGS84
  equatorial radius); escape, $r_a \le 0$ (non-positive orbital energy);
  mass depletion, $m \le 1\ \mathrm{kg}$.
- **Bootstrap semantics.** The step limit is a true terminal for this
  fixed-horizon task, so `extras['time_outs']` is all-False and no
  $\gamma V(s')$ timeout bootstrap is applied. `max_episode_length = 168`.

## 8. Initial Distribution

- Mean anomaly $M_0 \sim U(-\pi, \pi)$, resampled at every reset.
- Fixed elements $a_0 = 10^{7}\ \mathrm{m}$, $e_0 = 0.1$, $i_0 = \pi/3$,
  $\Omega_0 = 2\pi/3$, $\omega_0 = 4\pi/3$; the Cartesian state follows from
  §3, giving $r_{p0} = 9\times10^{6}\ \mathrm{m}$,
  $r_{a0} = 1.1\times10^{7}\ \mathrm{m}$ (recorded from the actual initial
  osculating state).
- $m_0 = 100\ \mathrm{kg}$; elapsed time $t = 0$; potential $\Phi(s_0) = 0$.

## 9. Metrics and Extras

Per-step extras: `time_outs` (all-False, §7) and `terminal_observations`.
Accumulated over completed episodes: `perigee_raise_km`
$= (r_p - r_{p0})/10^3$, `apo_drift_km` $= \lvert r_a - r_{a0}\rvert/10^3$,
`propellant_kg` $= m_0 - m$, and `terminal_reward`. When any replica
finishes, `extras['episode']` reports the per-done-replica means of
`success_rate`, `perigee_raise_km`, `apo_drift_km` and `propellant_kg`.
Cumulative buffers expose `returns`, `completed_returns`,
`completed_episodes` and `completed_successes`.

## 10. Implementation

Class `LowThrustTransferEnv` in
`environments/satellite/orbit_transfer/env.py`; registration docstring in the
package `__init__.py`. Constructor defaults: `num_envs=64`, `backend='cpu'`,
`seed=7`.

| Constant | Value |
|---|---|
| $\mu$ | $3.986004418\times10^{14}\ \mathrm{m^3/s^2}$ (WGS84) |
| $f_{\max}$ / $I_{\mathrm{sp}}$ / $g_0$ | $0.01\ \mathrm{N}$ / $4000\ \mathrm{s}$ / $9.80665\ \mathrm{m/s^2}$ |
| $m_0$ | $100\ \mathrm{kg}$ |
| $(a_0, e_0, i_0, \Omega_0, \omega_0)$ | $(10^{7}\ \mathrm{m},\ 0.1,\ \pi/3,\ 2\pi/3,\ 4\pi/3)$ |
| control step $\mathrm{d}t$ | $300\ \mathrm{s}$ |
| integrator | RK4, 30 substeps $\times$ $10\ \mathrm{s}$ per control step |
| `max_steps` | 166 ($49{,}800\ \mathrm{s}$, ${\sim}5$ revolutions) |
| reward scale | $10^{-4}(-\lvert\Delta r_a\rvert + \Delta r_p + 2\times10^{5}\Delta m)$ |
| violation thresholds | $r_p < 6.378137\times10^{6}\ \mathrm{m}$; $r_a \le 0$; $m \le 1\ \mathrm{kg}$ |
| Kepler solver | 8 Newton–Raphson iterations from $E_0 = M$ |
| reference obs bounds (declared only) | $1.1\times[49800\ \mathrm{s},\ r_a, r_a, r_a, v_{\max}, v_{\max}, v_{\max},\ 100\ \mathrm{kg}]$ |

## 11. References

- gym-satellite-trajectory,
  `PerigeeRaisingEnv` (github.com/zampanteymedio/gym-satellite-trajectory) —
  task reference, reproduced parameter-for-parameter.
- Casas, R., Carro, A., Sánchez-Esguevillas, J. "Low-Thrust Orbital Transfer
  using Dynamics-Agnostic Reinforcement Learning", arXiv:2211.08272 (IEEE) —
  reward weights (Table I) and mission parameters.
- Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —
  Kepler-to-Cartesian conversion (Newton–Raphson eccentric-anomaly solve) and
  osculating-element recovery.
