# Rendezvous Environment (`SpaceRendezvousEnv`)

## 1. Overview

The rendezvous task reproduces the medium-range approach benchmark of the
SpaceGym proximity-operations suite (Supreeth & Mehta, CVPR 2026 Workshop on
AI4Space, Section 3.2). A chaser, initialized 800–1200 m from a target on a
circular 400 km low-Earth orbit, must close to within 10 m with terminal
relative speed below 0.5 m/s, while avoiding a 5 m radius keep-out sphere,
staying within 2000 m, and respecting a 4 kg propellant budget. Relative motion
follows the Clohessy–Wiltshire–Hill (CWH) linearized equations in the target's
local-vertical/local-horizontal (LVLH) frame, integrated by symplectic Euler
at a fixed 1 s step; the environment is fully tensorized (no engine).

## 2. Notation

| Symbol | Meaning | Value |
|---|---|---|
| $\mu$ | Earth gravitational parameter | $3.986\times10^{14}$ m³/s² |
| $a$ | target orbit semi-major axis | $6.778\times10^{6}$ m (400 km circular) |
| $n$ | target mean motion, $\sqrt{\mu/a^{3}}$ | $1.1314\times10^{-3}$ rad/s |
| $m$ | chaser mass | 500 kg |
| $F_{\max}$ | per-axis thrust limit | 10 N |
| $I_{sp}$ | RCS specific impulse | 300 s |
| $\Delta t$ | integration step | 1 s |
| $p,v,F\in\mathbb{R}^{3}$ | LVLH relative position / velocity / thrust | — |

The LVLH frame is centered on the target: $x$ radially outward, $y$ along-track, $z$ cross-track.

## 3. Dynamics

The CWH equations (Clohessy & Wiltshire, 1960) with thrust forcing are
\begin{align}
\ddot{x} &= 3n^{2}x + 2n\dot{y} + F_{x}/m, \tag{1a}\\
\ddot{y} &= -2n\dot{x} + F_{y}/m, \tag{1b}\\
\ddot{z} &= -n^{2}z + F_{z}/m. \tag{1c}
\end{align}
Writing $G=\mathrm{diag}(3n^{2},0,-n^{2})$ and
$H=\big[\begin{smallmatrix}0&2n&0\\-2n&0&0\\0&0&0\end{smallmatrix}\big]$, the
acceleration is $\dot v = Gp + Hv + F/m$. The integrator is the reference paper's
symplectic Euler scheme (Algorithm 1), velocity updated first:
\begin{align}
v_{t+1} &= v_t + \Delta t\,(G p_t + H v_t + F_t/m), \tag{2a}\\
p_{t+1} &= p_t + \Delta t\,v_{t+1}. \tag{2b}
\end{align}
Equivalently, one step of the stacked state $s=(p,v)$ is the linear map
\begin{equation}
\begin{bmatrix}p_{t+1}\\ v_{t+1}\end{bmatrix}
=
\underbrace{\begin{bmatrix}I+\Delta t^{2}G & \Delta t\,(I+\Delta t\,H)\\
\Delta t\,G & I+\Delta t\,H\end{bmatrix}}_{A_d}
\begin{bmatrix}p_t\\ v_t\end{bmatrix}
+
\underbrace{\frac{1}{m}\begin{bmatrix}\Delta t^{2}I\\ \Delta t\,I\end{bmatrix}}_{B_d}F_t ,
\tag{3}
\end{equation}
with $\Delta t=1$ s. Propellant consumption is tracked in real time and
accumulated into the per-replica fuel counter $f_t$,
\begin{equation}
\Delta f_t = \frac{\lVert F_t\rVert\,\Delta t}{I_{sp}\,g_0},
\qquad g_0 = 9.80665\ \mathrm{m/s^2}. \tag{4}
\end{equation}

## 4. Action space

| Property | Value |
|---|---|
| Dimension | 3 |
| Range | $a\in[-1,1]^{3}$, clipped elementwise |
| Scaling | $F = F_{\max}\cdot a$ per axis (10 N/axis) |
| Fuel | $\Delta f$ per Eq. (4); budget $f_0 = 4$ kg |

A full three-axis burn sustained for the entire 1000-step episode consumes
$\sqrt{3}\,F_{\max}\,T/(I_{sp}g_0)\approx 5.89$ kg, so the budget binds only for
near-saturated profiles; keep-out and range violations dominate.

## 5. Observation space

The observation is the raw, unnormalized 6-dimensional LVLH relative state
\begin{equation}
o_t = [\,p_t^{\top},\ v_t^{\top}\,]^{\top}\in\mathbb{R}^{6}, \tag{5}
\end{equation}
returned as float32 under the `policy` key of a TensorDict.

## 6. Reward

The reward is a telescoping potential-based shaping on the
brake-distance-augmented range. Define the potential
\begin{equation}
\Phi(s) = -\frac{\lVert p\rVert + \lVert v\rVert^{2}/(2a_{\mathrm{ref}})}{d_{\mathrm{norm}}},
\qquad a_{\mathrm{ref}} = 0.1\ \mathrm{m/s^{2}}, \qquad d_{\mathrm{norm}} = d_0, \tag{6}
\end{equation}
where $d_0=\lVert p_0\rVert$ is the per-episode initial separation, adapting the
normalization to each spawn distance. The reference deceleration exceeds the per-axis
thrust authority $F_{\max}/m=0.02$ m/s², preserving most of the progress credit
during the accelerate-toward-target phase while still penalizing unrecoverable
approach speeds near the target. The per-step reward is
\begin{equation}
r_t = w\,\big(\Phi(s_{t+1}) - \Phi(s_t)\big) - \gamma\,\frac{\lVert F_t\rVert}{F_{\max}},
\qquad w = 100,\quad \gamma = 0.01. \tag{7}
\end{equation}
The shaping is potential-based, so the episode sum telescopes,
\begin{equation}
\sum_{t=0}^{T-1} w\,\big(\Phi(s_{t+1})-\Phi(s_t)\big) = w\,\big(\Phi(s_T)-\Phi(s_0)\big), \tag{8}
\end{equation}
and the cumulative shaping is path-independent: hovering costs approximately
zero, closing and braking pay positively, retreating pays negatively, and
round trips net zero. The quadratic velocity term aligns the reward gradient
with the maximum-brake approach profile at deceleration $a_{\mathrm{ref}}$.
On success a terminal bonus $R_{+}=+20$ is added; a violation incurs a one-time
penalty $R_{-}=-20$ (no per-step fee). The 20-point terminal spread (rather
than the 100 used for docking) reduces advantage noise near the success/timeout
decision boundary while preserving the incentive ordering.

## 7. Success, termination, and truncation

1. **Violation (termination).** $f_t \ge f_0$ (fuel exhaustion) **or**
   $\lVert p_t\rVert > 2000$ m **or** $\lVert p_t\rVert < 5$ m (keep-out
   penetration). Sets `terminated`; applies $R_{-}$.
2. **Success (truncation).** $\lVert p_t\rVert < 10$ m **and**
   $\lVert v_t\rVert < 0.5$ m/s **and** no violation. Sets `truncated` and
   the internal `complete` flag; applies $R_{+}$. Never sets `terminated`.
3. **Timeout (truncation).** step count reaches 1000 ($\approx 16.7$ min).
   Sets `truncated` with neither bonus nor penalty.

The priority rule is frozen: a violation on the terminal step is always a
failure, never a success — violation outranks success. The ordering is
load-bearing because the 5 m keep-out sphere lies *inside* the 10 m goal ball:
a capture-speed penetration of the no-fly zone would otherwise satisfy the goal
and be counted a success; under the frozen rule it is a keep-out failure with
no bonus. `extras['time_outs']` is `truncated & ~terminated & ~complete`: pure
timeouts only, excluding successes, matching the Gymnasium time-limit convention.

## 8. Initial state distribution

At reset, the position direction is uniform on the unit sphere and the
separation uniform in the task range; the velocity has an independent uniform
direction with magnitude uniform on $[0, v_{\max}]$:
\begin{equation}
p_0 = d\,\hat u,\quad d\sim\mathcal{U}[800,1200]\ \mathrm{m},\quad \hat u\sim\mathcal{U}(S^{2}),\qquad
v_0 = s\,\hat w,\quad s\sim\mathcal{U}[0,1]\ \mathrm{m/s},\quad \hat w\sim\mathcal{U}(S^{2}), \tag{9}
\end{equation}
with $\hat u,\hat w$ independent. The sampled separation is stored as
$d_0=\lVert p_0\rVert$ and serves as that episode's reward normalizer. Fuel, step
counter, and episode flags are zeroed; inactive replicas are frozen and consume no random numbers.

## 9. Metrics and extras

On any completed episode, `extras['episode']` reports `success_rate` (fraction of episodes ending this step
with `complete` set) and `init_range_hi`; `extras['terminal_observations']` carries the float32 terminal observation.

## 10. Implementation

- File: `environments/satellite/rendezvous_docking/env.py`; class `SpaceRendezvousEnv`
  (subclass of the shared `_CWHBase`); task id `rendezvous`
- Conventional baseline: `RendezvousTwoImpulseBaseline` (receding-horizon CWH two-impulse guidance)

| Constant | Value |
|---|---|
| `max_steps` | 1000 (`max_episode_length` buffer 1002) |
| `init_range` | (800, 1200) m |
| `init_vmax` | 1 m/s |
| `pos_goal` / `vel_goal` | 10 m / 0.5 m/s |
| `range_fail` | 2000 m |
| `keep_out` | 5 m |
| `fuel0` | 4 kg |
| `pos_norm` ($d_{\mathrm{norm}}$) | per-episode $d_0$ |
| `brake_ref` ($a_{\mathrm{ref}}$) | 0.1 m/s² |
| `progress_weight` / `gamma` | 100 / 0.01 |
| `terminal_bonus` / `viol_penalty` | +20 / −20 |

## 11. References

- Supreeth, H. S. G. & Mehta, "SpaceGym: A Gymnasium-Based Benchmark for Deep
  Reinforcement Learning in Spacecraft Proximity Operations," *CVPR 2026
  Workshop on AI4Space* (CC BY 4.0); task table, dynamics, and integrator
  reproduced from Section 3 (no public reference implementation exists).
- Clohessy, W. H. & Wiltshire, R. S. (1960), "Terminal Guidance System for
  Satellite Rendezvous," *Journal of the Aerospace Sciences*, 27(9), 653–658.
