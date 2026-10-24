# QuadX-Waypoints Environment (`QuadXWaypointsEnv`)

## 1. Overview

The QuadX-Waypoints task is the quadrotor navigation task of the AerospaceRL benchmark, ported from the
upstream PyFlyt `QuadXWaypointsEnv` (PyFlyt 0.29.0) onto the Genesis rigid-body backend with a fully
tensorized plant. A Crazyflie 2.x micro-quadrotor (27 g, X configuration, four rotors) spawns at rest at
$(0,0,1)$ m inside a 5 m flight dome and must visit four randomly sampled waypoints in order, each captured
by entering a 0.2 m sphere, within the 20 s time limit; ground contact or leaving the dome fails the
episode. Rewards follow a potential-based telescoping shaping on the remaining path length, with an
explicit per-waypoint bonus and a crash penalty. Wind is an optional constructor-side disturbance; by
default the air is calm and a fixed seed reproduces a waypoint layout bit-for-bit.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $p,v\in\mathbb{R}^{3}$; $q=(w,x,y,z)$ | world position / velocity; attitude quaternion |
| $\omega_b,v_b\in\mathbb{R}^{3}$ | body-frame angular / linear velocity |
| $w_k\in\mathbb{R}^{3}$, $k=0..3$ | waypoint positions; $i_t$ active index |
| $a\in[-1,1]^{4}$; $\bar a$ | raw policy action; its physical image |
| $\Delta t_\pi,\Delta t_c,\Delta t$ | periods: policy $1/40$ s, control $1/120$ s, physics $1/240$ s |
| $\rho$ | air density, 1.225 kg/m³ |

## 3. Dynamics and control hierarchy

Three nested rates are used. The policy runs at 40 Hz. Each policy step executes `inner_steps = 3` control
periods at 120 Hz, and each control period applies one inner-loop controller update followed by two Genesis
physics substeps at 240 Hz, i.e. six physics substeps per policy step.

**Inner loop (120 Hz).** A rate PID tracks the commanded body rates $\omega^{\mathrm{cmd}}$ with normalized
torque output $u\in[-1,1]^3$:
\begin{equation}
u_t=\mathrm{sat}\Big(k_p e_t + I_t + k_d\,(e_t-e_{t-1})/\Delta t_c\Big),\qquad
I_t=\mathrm{sat}\big(I_{t-1}+k_i\,e_t\,\Delta t_c\big), \tag{1}
\end{equation}
with $e_t=\omega^{\mathrm{cmd}}-\omega_b$, $k_p=(0.04,0.04,0.08)$,
$k_i=(5\times10^{-7},5\times10^{-7},2.7\times10^{-4})$, $k_d=(10^{-4},10^{-4},0)$, and unit saturation
limits. The torque command and the collective $T\in[0,0.8]$ are mixed onto the four rotors,
$\tilde\pi = M\,[u^\top, T]^\top$, with the upstream X-quad mixer
$M=\big[\begin{smallmatrix}-1&-1&-1&1\\1&1&-1&1\\1&-1&1&1\\-1&1&1&1\end{smallmatrix}\big]$
(motor order $(+x,-y),(-x,+y),(+x,+y),(-x,-y)$), followed by the upstream spread-preserving rescale into
the pwm window $[0.05,1]$.

**Actuation (240 Hz).** Each motor state follows a first-order lag and multiplicative noise,
\begin{equation}
\theta_{t+1}=\theta_t+\frac{\Delta t}{\tau_m}(\pi_t-\theta_t),\qquad
\theta_t\leftarrow\theta_t\big(1+0.02\,\xi_t\big),\qquad \xi_t\sim\mathcal N(4,1), \tag{2}
\end{equation}
with $\tau_m=0.01$ s; $\xi$ is a single Gaussian draw per replica per physics step, shared across the four
motors (upstream convention; the nonzero mean is absorbed as a constant throttle dc gain). Per-rotor thrust
is quadratic in the motor state, $f_i=\lvert\theta_i\rvert\theta_i\,F_{\max}/4$ with $F_{\max}=2.0$ N
total, and produces reaction torque $\tau_{z,i}=f_i\,(k_m/k_f)\,s_i$ with
$k_m/k_f=7.94\times10^{-12}/3.16\times10^{-10}$ and spin signs $s=(-1,-1,1,1)$. Componentwise quadratic
body drag $F_d=-\tfrac12\rho C_d A\,\lvert v_{\mathrm{air}}\rvert\circ v_{\mathrm{air}}$ ($C_d=3.0$,
$A=4\times10^{-4}$ m² per axis) and angular drag $\tau_d=-10^{-4}\lvert\omega_b\rvert\circ\omega_b$
(suppressed while in contact) complete the plant. As a solver guard, $\lVert v\rVert$ and
$\lVert\omega\rVert$ are rescaled to 15 m/s and 30 rad/s only when exceeded (`vel_clamp`), far above
nominal waypoint flight.

## 4. Action space

Raw actions are clipped to $[-1,1]^4$ and mapped affinely, $\bar a = \ell + \tfrac{a+1}{2}(h-\ell)$, with
$\ell=(-\pi,-\pi,-\pi,0)$, $h=(\pi,\pi,\pi,0.8)$:

| Index | Symbol | Physical range | Meaning |
|---|---|---|---|
| 0 | $\omega_x^{\mathrm{cmd}}$ | $\pm\pi$ rad/s | roll-rate command |
| 1 | $\omega_y^{\mathrm{cmd}}$ | $\pm\pi$ rad/s | pitch-rate command |
| 2 | $\omega_z^{\mathrm{cmd}}$ | $\pm\pi$ rad/s | yaw-rate command |
| 3 | $T$ | $[0,\,0.8]$ | collective throttle (pwm units) |

## 5. Observation space

Float32 vector of dimension 37, returned under the `policy` key of a TensorDict. Deltas to invalid waypoint
slots are zeroed.

| Slice | Symbol | Unit | Description |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | body angular velocity |
| [3:7] | $q$ as $(x,y,z,w)$ | — | attitude quaternion |
| [7:10] | $v_b$ | m/s | body-frame velocity |
| [10:13] | $p$ | m | world position |
| [13:17] | $\bar a_{t-1}$ | (mixed) | last physical action (rate commands, collective) |
| [17:21] | $\theta$ | — | per-rotor motor states |
| [21:33] | $\Delta_k$ | m | body-frame deltas $R(q)^\top(w_{i_t+k}-p)$, $k=0..3$, flattened |
| [33:37] | $v_k$ | — | validity flags of the four waypoint slots (0/1) |

## 6. Reward

Let $\ell_k=\lVert w_{k+1}-w_k\rVert$ be the leg lengths and $L(i)=\sum_{k=i}^{2}\ell_k$ the remaining-leg
suffix. The potential is
\begin{equation}
\Phi(s)=-\Big(\big\lVert p-w_{i}\big\rVert + L(i)\Big),\qquad i<4;\qquad
\Phi(s)=0\ \text{once all four waypoints are captured}. \tag{3}
\end{equation}
The per-step reward is
\begin{equation}
r_t=\Phi(s_{t+1})-\Phi(s_t)+15\,c_t-10\,\kappa_t, \tag{4}
\end{equation}
where $c_t$ counts waypoint captures during step $t$ (capture is tested at 120 Hz, so one 40 Hz step may
capture several) and $\kappa_t\in\{0,1\}$ indicates a crash or out-of-bounds event. Being a difference of a
state potential, the shaping telescopes:
\begin{equation}
\sum_{t=0}^{T-1}\big(\Phi(s_{t+1})-\Phi(s_t)\big)=\Phi(s_T)-\Phi(s_0). \tag{5}
\end{equation}
Cumulative shaping is therefore path-independent: orbiting a waypoint at constant distance nets exactly
zero, and a fully successful episode collects precisely the reference route length
$\lVert p_0-w_0\rVert+\sum_k\ell_k$, independently of the trajectory flown — the potential-based invariance
of Ng, Harada & Russell (1999). A crash or timeout forfeits the unrealized potential $-\Phi(s_T)$, so
ending the episode early is never cheaper than flying. The $+15$ bonus keeps waypoint capture a crisp
sparse event; the $-10$ penalty sharpens failure.

## 7. Success, termination, and truncation

Events latch inside the 120 Hz substep loop on the post-substep state; the time limit is evaluated once per
policy step afterwards, so the final control step is never cut short:

1. **Termination.** Ground contact, or $\lVert p\rVert>5$ m. Sets `terminated`; incurs the $-10$ penalty.
2. **Success (truncation).** $i_t$ reaches 4. Sets `truncated` and the internal `complete` flag; success
   does not set `terminated`.
3. **Timeout (truncation).** Step count reaches 800 (20 s). Sets `truncated` for still-active replicas only.

If a failure and the final capture coincide in one substep, both flags latch and the episode is treated as
terminated. `extras['time_outs']` is `truncated & ~terminated & ~complete`: pure timeouts only. The
benchmark trainers bootstrap the value function exactly where `time_outs` is set — FlashSAC cuts the
bootstrap on `done & ~time_outs` (using the pre-reset `extras['terminal_observations']` as bootstrap
state), rsl-rl PPO via its standard `time_outs` GAE handling — so crashes, out-of-bounds, and *completed*
episodes all cut the bootstrap, and only genuine timeouts are bootstrapped.

## 8. Initial state distribution

Spawns are deterministic: $p_0=(0,0,1)$ m, identity attitude, zero linear and angular velocity (spawn-noise
attributes are manual domain-randomization knobs and are zero in the benchmark). Waypoints follow the
upstream polar sampling, drawn per replica from a counter-based device RNG:
\begin{equation}
\theta,\phi\sim\mathcal U[0,2\pi),\quad
r\sim\mathcal U[1.5,\,4.8]\ \mathrm{m},\quad
w=\big(r\sin\phi\cos\theta,\; r\sin\phi\sin\theta,\; \max(\lvert r\cos\phi\rvert,\,0.4\ \mathrm{m})\big). \tag{6}
\end{equation}
The waypoint draw is the only reset randomness; a fixed seed reproduces a layout bit-for-bit. Reset
replicas run a 10-by-2 zero-command warmup (20 physics substeps) to settle the plant.

## 9. Disturbances

Optional wind is configured through the constructor kwargs `steady_wind` (world-frame constant velocity,
m/s), `dryden` (intensity in $[0,1]$), and `severity` (`'light'`/`'moderate'`/`'severe'`, default
`'moderate'`). When enabled, `WindDisturbance` superimposes the steady component with a Dryden gust
(MIL-F-8785C low-altitude form): per-axis first-order shaping filters driven by white noise from a seeded
deterministic generator, with reference sigmas at 300 ft of $(1.5,1.5,0.7)$, $(3.0,3.0,1.4)$,
$(6.0,6.0,3.0)$ m/s per severity class scaled by `dryden`, scale lengths $(200,200,50)$ m, poles
$a_k=\exp(-\Delta t_\pi V/L_k)$ at reference airspeed $V=15$ m/s, and filter gain
$\sigma_k\sqrt{1-a_k^2}$, making the stationary per-axis standard deviation exactly $\sigma_k\cdot$`dryden`.
The generator advances once per policy step and the wind is frozen across the inner substeps; all
aerodynamic terms evaluate $v_{\mathrm{air}}=v-w$. By default (`steady_wind=None`, `dryden=0`) no wind
buffer exists and the calm-air dynamics are bit-for-bit unchanged.

## 10. Metrics and extras

`extras['time_outs']` and `extras['terminal_observations']` (pre-reset float32 observations) are set every
step. On steps where at least one replica finishes, `extras['episode']` reports `success_rate`,
`waypoints_reached` (mean over the finishing replicas), `crash_rate`, `out_of_bounds_rate`, and
`timeout_rate`. Cumulative counters live in `completed_metrics` (`waypoints_reached`, `crashes`,
`out_of_bounds`, `timeouts`) alongside `completed_returns`, `completed_episodes`, and
`completed_successes`.

## 11. Implementation

- File: `environments/multirotor/env.py`, class `QuadXWaypointsEnv` (task id `quadx`); plant in
  `environments/common/plant.py` (`TensorPlant`), wind in `environments/common/disturbances.py`.
- Conventional controller: `QuadXSE3Baseline` (SE(3) geometric tracking), exposed as the class attribute
  `conventional_controller`.

| Constant | Value |
|---|---|
| policy / control / physics rates | 40 / 120 / 240 Hz (`inner_steps = 3`, 2 substeps) |
| `max_steps` | 800 (20 s); `max_episode_length` 802 |
| mass / max total thrust | 0.027 kg / 2.0 N |
| motor lag $\tau_m$ / noise ratio | 0.01 s / 0.02 |
| rate-PID $k_p,k_i,k_d$ | $(.04,.04,.08)$, $(5{\times}10^{-7},5{\times}10^{-7},2.7{\times}10^{-4})$, $(10^{-4},10^{-4},0)$ |
| pwm window | $[0.05,1]$; collective command $\le 0.8$ |
| `CAPTURE_RADIUS` | 0.2 m |
| out-of-bounds radius | 5 m |
| waypoint radius / altitude floor | $[1.5,4.8]$ m / 0.4 m (widened from $[1.0,4.5]$ m on 2026-10-24; baseline was at 100% success) |
| spawn | $(0,0,1)$ m, at rest, level |
| `REACH_BONUS` / `CRASH_PENALTY` | +15 / −10 |
| `vel_clamp` ($\lVert v\rVert$, $\lVert\omega\rVert$) | 15 m/s, 30 rad/s |

## 12. References

- Tai, J. K. et al., *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, v0.29.0 — upstream
  task definition, vehicle model, and control defaults.
- Genesis contributors, *Genesis* — rigid-body simulation backend.
- Ng, A. Y., Harada, D., & Russell, S. (1999), "Policy invariance under reward transformations," *ICML* —
  potential-based shaping invariance.
- MIL-F-8785C, *Flying Qualities of Piloted Airplanes* — low-altitude Dryden turbulence model used by the
  optional wind disturbance.
