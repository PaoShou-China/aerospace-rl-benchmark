# Fixedwing-Waypoints Environment (`FixedwingWaypointsEnv`)

## 1. Overview

The Fixedwing-Waypoints task is the fixed-wing navigation task of the AerospaceRL benchmark, ported from
the upstream PyFlyt `FixedwingWaypointsEnv` (PyFlyt 0.29.0) onto the Genesis rigid-body backend with a
fully tensorized plant. A 2.35 kg, 1.6 m span unmanned aircraft spawns at $(0,0,10)$ m with an initial
airspeed of 20 m/s and must visit four waypoints sampled in a 20–65 m shell around the origin, in order,
within the 120 s time limit. A waypoint is captured on entering a 10 m sphere around it; the episode fails
on ground contact or on leaving the 100 m out-of-bounds sphere. Rewards use the same potential-based
telescoping shaping as the quadrotor task. Wind is an optional constructor-side disturbance; calm air with
a fixed seed reproduces a waypoint layout bit-for-bit.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $p,v\in\mathbb{R}^{3}$; $q=(w,x,y,z)$ | world position / velocity; attitude quaternion |
| $\omega_b,v_b\in\mathbb{R}^{3}$ | body-frame angular / linear velocity |
| $w_k\in\mathbb{R}^{3}$, $k=0..3$ | waypoint positions; $i_t$ active index |
| $a\in[-1,1]^{4}$ | raw policy action |
| $\Delta t_\pi,\Delta t_c,\Delta t$ | periods: policy $1/30$ s, control $1/120$ s, physics $1/240$ s |
| $\rho$; $\bar q=\tfrac12\rho\lVert v_{\mathrm{air}}\rVert^{2}$ | air density, 1.225 kg/m³; dynamic pressure |

## 3. Dynamics and control hierarchy

The policy runs at 30 Hz. Each policy step executes `inner_steps = 4` control periods at 120 Hz, each
applying the actuation update followed by two Genesis physics substeps at 240 Hz (eight physics substeps
per policy step). Unlike the quadrotor task there is no inner attitude loop: the policy drives the control
surfaces and the throttle directly, and the only sub-policy dynamics are the actuator lags.

**Surfaces (240 Hz).** Five lifting surfaces — left/right ailerons, horizontal tail, vertical tail, and
main wing — follow first-order deflection lags with $\tau_s=0.05$ s,
\begin{equation}
\delta_{t+1}=\delta_t+\frac{\Delta t}{\tau_s}\big(\delta^{\mathrm{cmd}}-\delta_t\big), \tag{1}
\end{equation}
and produce forces and moments from the PyFlyt blade-element lifting-surface model: finite-wing lift-curve
slope, flap effectiveness from the 30% flap-to-chord ratio with viscosity correction $\eta=0.65$, zero-lift
angle $\alpha_0$ ($-2^\circ$ on wing/ailerons, $0^\circ$ on the tails), positive and negative stall
($+14^\circ/-9^\circ$ wing, $\pm9^\circ$ tails), and skin-friction drag $C_{d0}=0.01$, all evaluated at the
surface-local air-relative velocity $v_{\mathrm{air}}=v-w$ with forces $\bar q S(c_l,c_d)$ and pitching
moment $\bar q S c\, c_m$.

**Engine (240 Hz).** The throttle state follows a first-order lag with multiplicative noise,
\begin{equation}
\theta_{t+1}=\theta_t+\frac{\Delta t}{\tau_m}\big(\theta^{\mathrm{cmd}}-\theta_t\big),\qquad
\theta_t\leftarrow\theta_t\big(1+0.02\,\xi_t\big),\qquad \xi_t\sim\mathcal N(1,1), \tag{2}
\end{equation}
with $\tau_m=0.01$ s (one Gaussian draw per replica per physics step, upstream convention). Propeller
thrust $F=\lvert\theta\rvert\theta\,F_{\max}$ with $F_{\max}=18$ N acts along the body $x$ axis, together
with the reaction torque $F\,(k_m/k_f)$, $k_m/k_f=7.94\times10^{-12}/3.16\times10^{-10}$.

## 4. Action space

Raw actions are clipped to $[-1,1]^4$ and passed through the identity affine map ($\ell=-\mathbf 1$,
$h=+\mathbf 1$). Surface commands follow the upstream `surface_assist` mixing with per-surface signs
$(+1,-1,+1,-1,0)$:

| Index | Symbol | Range | Physical meaning |
|---|---|---|---|
| 0 | $\delta_a$ | $[-1,1]$ | differential ailerons (left $+\delta_a$, right $-\delta_a$; $\pm30^\circ$) |
| 1 | $\delta_e/\delta_r$ | $[-1,1]$ | elevator $+\delta_e$ and rudder $-\delta_r$ (opposite signs; $\pm20^\circ$) |
| 2 | — | $[-1,1]$ | dead channel: mapped to the main wing, whose deflection limit is $0^\circ$ |
| 3 | $\theta^{\mathrm{cmd}}$ | $[-1,1]\to[0,1]$ | throttle, remapped $\theta^{\mathrm{cmd}}=(a_3+1)/2$ before the plant |

Channel $a_2$ is inert by upstream design (the main wing surface has a zero deflection limit), so the
effective action dimensionality is 3; the mapping is retained verbatim for parity and must not be "fixed".

## 5. Observation space

Float32 vector of dimension 39, returned under the `policy` key of a TensorDict. Deltas to invalid waypoint
slots are zeroed.

| Slice | Symbol | Unit | Description |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | body angular velocity |
| [3:7] | $q$ as $(x,y,z,w)$ | — | attitude quaternion |
| [7:10] | $v_b$ | m/s | body-frame velocity |
| [10:13] | $p$ | m | world position |
| [13:17] | $a_{t-1}$ | — | last action (pre-throttle-remap, $[-1,1]^4$) |
| [17:22] | $\delta$ | — | surface deflection states (ail L/R, h-/v-tail, main wing) |
| [22:23] | $\theta$ | — | engine throttle state |
| [23:35] | $\Delta_k$ | m | body-frame deltas $R(q)^\top(w_{i_t+k}-p)$, $k=0..3$, flattened |
| [35:39] | $v_k$ | — | validity flags of the four waypoint slots (0/1) |

## 6. Reward

With leg lengths $\ell_k=\lVert w_{k+1}-w_k\rVert$ and remaining-leg suffix $L(i)=\sum_{k=i}^{2}\ell_k$,
the potential is
\begin{equation}
\Phi(s)=-\Big(\big\lVert p-w_{i}\big\rVert + L(i)\Big),\qquad i<4;\qquad
\Phi(s)=0\ \text{once all four waypoints are captured}, \tag{3}
\end{equation}
and the per-step reward is
\begin{equation}
r_t=\Phi(s_{t+1})-\Phi(s_t)+15\,c_t-10\,\kappa_t, \tag{4}
\end{equation}
with $c_t$ the number of captures during step $t$ (capture is tested at 120 Hz) and $\kappa_t\in\{0,1\}$
the crash/out-of-bounds indicator. Being a difference of a state potential, the shaping telescopes,
\begin{equation}
\sum_{t=0}^{T-1}\big(\Phi(s_{t+1})-\Phi(s_t)\big)=\Phi(s_T)-\Phi(s_0), \tag{5}
\end{equation}
so cumulative shaping is path-independent (Ng, Harada & Russell, 1999): orbiting a waypoint nets exactly
zero, a successful episode collects exactly the reference route length
$\lVert p_0-w_0\rVert+\sum_k\ell_k$, and a crash or timeout forfeits all unrealized potential $-\Phi(s_T)$
— ending the episode early is never cheaper than flying. The $+15$ capture bonus keeps a crisp sparse event
and the $-10$ penalty sharpens failure. This replaces the upstream shaped reward, whose $1/d$ term pays
unbounded return for orbiting and whose terminal $\pm100$ is unreachable by credit assignment over a
3600-step episode.

## 7. Success, termination, and truncation

Events latch inside the 120 Hz substep loop on the post-substep state; the time limit is evaluated once per
policy step afterwards:

1. **Termination.** Ground contact, or $\lVert p\rVert>100$ m. Sets `terminated`; incurs the $-10$ penalty.
2. **Success (truncation).** $i_t$ reaches 4. Sets `truncated` and the internal `complete` flag; success
   does not set `terminated`.
3. **Timeout (truncation).** Step count reaches 3600 (120 s). Sets `truncated` for still-active replicas only.

If a failure and the final capture coincide in one substep, both flags latch and the episode is treated as
terminated. `extras['time_outs']` is `truncated & ~terminated & ~complete`: pure timeouts only. The
benchmark trainers bootstrap the value function exactly where `time_outs` is set — FlashSAC cuts the
bootstrap on `done & ~time_outs` (using the pre-reset `extras['terminal_observations']` as bootstrap
state), rsl-rl PPO via its standard `time_outs` GAE handling — so crashes, out-of-bounds, and *completed*
episodes all cut the bootstrap, and only genuine timeouts are bootstrapped.

## 8. Initial state distribution

Spawns are deterministic: $p_0=(0,0,10)$ m, identity attitude, $v_0=(20,0,0)$ m/s, zero angular velocity.
Waypoints follow the upstream polar sampling, drawn per replica from a counter-based device RNG:
\begin{equation}
\theta,\phi\sim\mathcal U[0,2\pi),\quad
r\sim\mathcal U[20,\,65]\ \mathrm{m},\quad
w=\big(r\sin\phi\cos\theta,\; r\sin\phi\sin\theta,\; \max(\lvert r\cos\phi\rvert,\,20\ \mathrm{m})\big). \tag{6}
\end{equation}
The shell and altitude floor depart from the upstream 1–90 m shell with a 0.5 m floor for two reasons
documented in the source: trim constraints give a minimum controllable speed of about 11 m/s and hence a
10–15 m minimum turn radius, so targets within 10 m of the 100 m out-of-bounds sphere cannot be turned back
inside; and near-ground targets forced ground-skimming captures. The 20–65 m shell leaves a 35 m
out-of-bounds margin and a 20 m altitude floor. The 10 m capture radius — about one minimum turn radius —
replaces the upstream 2 m, which is quasi-unreachable at this turn radius (the best scripted head-on pass
missed by 8.6 m). The waypoint draw is the only reset randomness; a fixed seed reproduces a layout
bit-for-bit. Reset replicas run a 10-by-2 zero-command warmup (20 physics substeps).

## 9. Disturbances

Optional wind is configured through the constructor kwargs `steady_wind` (world-frame constant velocity,
m/s), `dryden` (intensity in $[0,1]$), and `severity` (`'light'`/`'moderate'`/`'severe'`, default
`'moderate'`). When enabled, `WindDisturbance` superimposes the steady component with a Dryden gust
(MIL-F-8785C low-altitude form): per-axis first-order shaping filters driven by white noise from a seeded
deterministic generator, with reference sigmas at 300 ft of $(1.5,1.5,0.7)$, $(3.0,3.0,1.4)$,
$(6.0,6.0,3.0)$ m/s per severity class scaled by `dryden`, scale lengths $(200,200,50)$ m, poles
$a_k=\exp(-\Delta t_\pi V/L_k)$ at reference airspeed $V=15$ m/s, and filter gain
$\sigma_k\sqrt{1-a_k^2}$, making the stationary per-axis standard deviation exactly $\sigma_k\cdot$`dryden`.
The generator advances once per policy step ($\Delta t_\pi=1/30$ s) and the wind is frozen across the inner
substeps; all lifting surfaces evaluate $v_{\mathrm{air}}=v-w$. By default (`steady_wind=None`, `dryden=0`)
no wind buffer exists and the calm-air dynamics are bit-for-bit unchanged.

## 10. Metrics and extras

`extras['time_outs']` and `extras['terminal_observations']` (pre-reset float32 observations) are set every
step. On steps where at least one replica finishes, `extras['episode']` reports `success_rate`,
`waypoints_reached` (mean over the finishing replicas), `crash_rate`, `out_of_bounds_rate`, and
`timeout_rate`. Cumulative counters live in `completed_metrics` (`waypoints_reached`, `crashes`,
`out_of_bounds`, `timeouts`) alongside `completed_returns`, `completed_episodes`, and
`completed_successes`.

## 11. Implementation

- File: `environments/fixedwing/env.py`, class `FixedwingWaypointsEnv` (task id `fixedwing`); plant in
  `environments/common/plant.py` (`TensorPlant`), surfaces in `environments/common/aerodynamics.py`
  (`LiftingSurfaceBatch`), wind in `environments/common/disturbances.py`.
- Conventional controller: `FixedwingAutopilotBaseline` (L1 lateral guidance + TECS), exposed as the class
  attribute `conventional_controller`.

| Constant | Value |
|---|---|
| policy / control / physics rates | 30 / 120 / 240 Hz (`inner_steps = 4`, 2 substeps) |
| `max_steps` | 3600 (120 s); `max_episode_length` 3602 |
| mass / wing span | 2.35 kg / 1.6 m |
| max thrust / motor lag $\tau_m$ / noise ratio | 18 N / 0.01 s / 0.02 |
| surface lag $\tau_s$ | 0.05 s |
| deflection limits (ail L/R, h-/v-tail, wing) | $30^\circ,30^\circ,20^\circ,20^\circ,0^\circ$ |
| `CAPTURE_RADIUS` | 10.0 m |
| out-of-bounds radius | 100 m |
| waypoint radius / altitude floor | $[20,65]$ m / 20 m |
| spawn | $(0,0,10)$ m, $v_0=(20,0,0)$ m/s |
| `REACH_BONUS` / `CRASH_PENALTY` | +15 / −10 |
| `vel_clamp` | disabled (`None`) |

## 12. References

- Tai, J. K. et al., *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, v0.29.0 — upstream
  task definition, vehicle model, and lifting-surface aerodynamics.
- Genesis contributors, *Genesis* — rigid-body simulation backend.
- Ng, A. Y., Harada, D., & Russell, S. (1999), "Policy invariance under reward transformations," *ICML* —
  potential-based shaping invariance.
- MIL-F-8785C, *Flying Qualities of Piloted Airplanes* — low-altitude Dryden turbulence model used by the
  optional wind disturbance.
