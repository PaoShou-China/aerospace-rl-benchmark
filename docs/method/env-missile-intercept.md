# Missile Intercept Environment (`MissileInterceptEnv`)

## 1. Overview

The missile-intercept task is a project-authored 6-DOF tail-controlled missile
engaging a kinematic 3-DOF target aircraft. The missile is a 150 kg slender
body (nose along body $+x$) with four cross tail fins, body drag, a
slender-body normal-force model, and an autonomous boost-coast engine (20 kN
along body $+x$ for the first 4 s of each episode, then unpowered coast). The
policy commands **body rates**, which a 120 Hz rate PID converts to fin
deflections through a skid-to-turn mixer. The target flies one of three
analytic maneuver classes — straight line, sinusoidal weave, or horizontal
barrel roll — at 250 m/s near 4 km altitude, advanced in closed form inside
the physics substep loop; a collision-free visual entity mirrors it in the
Genesis scene for rendering only. The engagement geometry is a single fixed
medium operating point (decision of 2026-10-24): initial range 2500 m,
target-heading cone $\pm 37.5^{\circ}$, target lateral acceleration 7.5 g,
weave frequency 0.75 Hz. The missile spawns at the engagement altitude with
fire-control lead — it aims at the straight-line extrapolation of the target
over a scaled nominal time-to-go and launches at 250 m/s along that bearing —
so the launch starts on a near collision course and the guidance only corrects
residual errors.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $p, q, v, \omega$ | missile world position, attitude (wxyz internally, exposed xyzw), world velocity, angular velocity; $v_b,\omega_b$ body-frame |
| $p_T, v_T$ | target world position / velocity; $V_T = \lVert v_T\rVert = 250$ m/s |
| $r = p_T - p$, $d = \lVert r\rVert$ | line-of-sight (LOS) vector and range |
| $v_{\text{rel}} = v_T - v$; $\lambda = (r\times v_{\text{rel}})/\lVert r\rVert^{2}$ | relative velocity; LOS rate |
| $V_c = -(r\cdot v_{\text{rel}})/\lVert r\rVert$ | closing speed |
| $a\in[-1,1]^{3}$ | raw policy action; $\bar a$ the scaled body-rate command [rad/s] |
| $h, n_\perp$ | target unit heading and its horizontal left-normal |
| $a_T$ | target lateral acceleration, $7.5\,g = 73.575$ m/s² |

## 3. Dynamics and control hierarchy

**Rates:** policy 60 Hz ($\Delta t = 1/60$ s), control 120 Hz
(`inner_steps = 2`), physics 240 Hz (two substeps per control period,
$h = 1/240$ s); episode limit 900 policy steps (15 s).

- **Airframe.** 150 kg rigid body. Axial body drag
  $F_x = -\tfrac12\rho\, C_{d,x} A_x |v_x| v_x$ ($C_{d,x}=0.25$,
  $A_x = 0.0314$ m², $\rho = 1.225$ kg/m³). Slender-body normal force
  $F_{\text{lat}} = -\tfrac12\rho\lVert v\rVert A_x C_{n\alpha}\,
  v_{\text{lat}}$ ($C_{n\alpha}=16$), applied at the center of pressure
  $x_{\text{cp}} = -0.45$ m aft of the CG with the transverse flow at the CP —
  simultaneously the turning force, the weathercock restoring moment, and a
  rotational damping term. Synthetic pitch/yaw damping
  $\tau = -c_{mq}\lVert v\rVert\,\omega_{\text{lat}}$ ($c_{mq}=4.0$) damps the
  otherwise nearly undamped fin weathercock mode. Speed clamps:
  $\lVert v\rVert\le 1200$ m/s, $\lVert\omega\rVert\le 40$ rad/s (`vel_clamp`).
- **Fins.** Four cross tail surfaces at $x=-1.3$ m (chord 0.45 m, span
  0.40 m, $C_{l\alpha}=3.0$, stall $\pm40^{\circ}$, deflection limit
  $\pm35^{\circ}$, actuator lag $\tau=0.02$ s).
- **Rate PID.** Per-axis proportional controller ($k_p = 5.0$, $k_i = k_d = 0$,
  output limit $\pm1$) on body rates at 120 Hz, producing a normalized
  (roll, pitch, yaw) fin command; the high $k_p$ compensates the strong
  weathercock spring (fin-cmd → steady-turn-rate gain ~0.2–0.3 rad/s per unit).
- **Skid-to-turn mixer.** With roll gearing $g = 0.01$ (roll inertia is ~150×
  smaller than pitch/yaw, so the roll channel is geared down to keep the
  120 Hz loop stable), the fin commands for [left, right, top, bottom] are
  $(q_c + g\,p_c,\ q_c - g\,p_c,\ -r_c - g\,p_c,\ -r_c + g\,p_c)$ for rate-PID
  outputs $(p_c, q_c, r_c)$, clamped to $[-1,1]$.
- **Engine.** Ignites autonomously on the first active physics step and burns
  at $20\,000\cdot(1 + 0.01\,\mathcal N(0,1))$ N along body $+x$ until the 4 s
  burn clock elapses; then coasts. The action carries no throttle/ignition.

## 4. Action space

The policy emits $a\in[-1,1]^{3}$, clipped elementwise and affinely mapped to
body-rate commands:

| Index | Raw range | Mapped range | Physical meaning |
|---|---|---|---|
| 0 | $[-1,1]$ | $\pm 6.0$ rad/s | roll-rate command (`OMEGA_MAX`) |
| 1 | $[-1,1]$ | $\pm 1.0$ rad/s | pitch-rate command (`OMEGA_LAT`) |
| 2 | $[-1,1]$ | $\pm 1.0$ rad/s | yaw-rate command (`OMEGA_LAT`) |

`OMEGA_LAT` was raised from 0.5 to 1.0 rad/s on 2026-10-24 for vanilla-PPO
learnability: the steady coordinated-turn rate is $a/V \approx 0.2$–0.5 rad/s,
but terminal guidance needs *transient* authority above it — the rate PID
simply saturates the fins, so a 1.0 rad/s cap lets the policy pull harder
during the endgame correction.

## 5. Observation space

A single float32 vector of dimension 34 under the `policy` key (unnormalized;
policy-side empirical normalization owns scaling):

| Slice | Symbol | Unit | Description |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | body angular velocity |
| [3:7] | $q$ (xyzw) | — | attitude quaternion |
| [7:10] | $v_b$ | m/s | body-frame velocity |
| [10:13] | $p$ | m | world position |
| [13:16] | $\bar a_{t-1}$ | rad/s | previous rate commands |
| [16:20] | — | — | fin actuations (4) |
| [20:21] | — | — | engine ignited flag |
| [21:22] | — | — | engine burning flag |
| [22:23] | — | — | burn-time-left fraction |
| [23:26] | $r$ (body) | m | target relative position, body frame |
| [26:29] | $v_{\text{rel}}$ (body) | m/s | relative velocity, body frame |
| [29:32] | $\lambda$ (body) | rad/s | LOS rate, body frame |
| [32:33] | $V_c$ | m/s | closing speed |
| [33:34] | $\lVert v_T\rVert$ | m/s | target speed |

## 6. Target model

The target is kinematic, advanced analytically at every physics substep from
the per-replica simulation time $t$ (incremented by $1/240$ s per substep),
with $p_0$ its position at $t = 0$. Three maneuver classes:

**Straight (type 0):**
\begin{equation}
p_T(t) = p_0 + V_T h\, t, \qquad v_T(t) = V_T h. \tag{1a}
\end{equation}

**Sinusoidal weave (type 1):** with weave angular rate
$\omega_w = 2\pi\times 0.75$ rad/s and random phase $\phi$,
\begin{equation}
p_T(t) = p_0 + V_T h\, t + n_\perp\,\frac{a_T}{\omega_w^{2}}\sin(\omega_w t + \phi),
\qquad
v_T(t) = V_T h + n_\perp\,\frac{a_T}{\omega_w}\cos(\omega_w t + \phi). \tag{1b}
\end{equation}

**Horizontal barrel roll (type 2):** a horizontal circle of radius
$R_b = V_T^{2}/a_T \approx 849.5$ m traversed at signed angular rate
$\omega_b = \pm a_T/V_T \approx \pm 0.294$ rad/s,
\begin{equation}
p_T(t) = c + R_b\,(\cos\theta, \sin\theta, 0),\qquad
v_T(t) = V_T\,(-\sin\theta, \cos\theta, 0),\qquad
\theta = \omega_b t + \phi_b. \tag{1c}
\end{equation}

The barrel phase is not free: requiring the initial tangent to match the spawn
heading ($v_T(0) \parallel h$) pins $\phi_b = \operatorname{atan2}(-h_x, h_y)$
(plus $\pi$ when $\omega_b < 0$), and the center sits one radius along the
inward normal, $c = p_0 - R_b(\cos\phi_b,\sin\phi_b,0)$; otherwise the barrel
target would start flying in an arbitrary direction and the engagement would
degenerate into a tail chase.

## 7. Reward

Accumulated per 120 Hz inner step, only while the replica is active, with
components matching the `reward_components` keys:

\begin{align}
r_{\text{progress}} &= 0.01\,(d^{-} - d) && \text{telescoping closing progress}, \tag{2a}\\
r_{\text{align}} &= 0.02\,(\hat v\cdot \hat r) && \text{velocity/LOS alignment}, \tag{2b}\\
r_{\text{ctrl}} &= -0.01\,\lVert \bar a\rVert^{2} && \text{rate-command effort}, \tag{2c}\\
r_{\text{kill}} &= +50 && \text{once, on kill}, \tag{2d}\\
r_{\text{miss}} &= -10 && \text{once, on unrecoverable miss}, \tag{2e}
\end{align}

with $d^{-}$ the range at the previous inner-step boundary. The alignment term
is the dense, PN-consistent guidance shaping: potential-like and locally
computable, and unlike a penalty on $\lVert\lambda\rVert$ it cannot be gamed
by flying away from the target.

## 8. Success, termination, and truncation

Evaluated at **every physics substep** (240 Hz), in the following priority
order:

1. **Kill (success).** $d < $ `KILL_RADIUS` $= 15$ m at any substep. Sets the
   `complete` flag, pays $r_{\text{kill}}$, and truncates at the end of the
   enclosing policy step. The radius was raised from 8 to 15 m on 2026-10-24:
   at 8 m the behavior-cloned warm start's median miss (9.8 m at $d=0$) sat on
   the fuse boundary, so any ~$10^{-3}$ weight drift from a PPO update wiped
   out all kills (measured blocker); 15 m puts the warm start comfortably
   inside the gate.
2. **Unrecoverable miss (termination).** The closest-approach point passes:
   $r\cdot v_{\text{rel}} > 0$ (range increasing) while $t > 2$ s. Pays
   $r_{\text{miss}}$ once. A kill preempts a miss within the same inner step.
3. **Out of bounds (termination).** $\lVert p\rVert > 15$ km or $z < 0$.
4. **Time limit (truncation).** `count` $\ge 900$ policy steps (15 s), applied
   after the full control step (rocket semantics), as is the kill truncation.

`extras['time_outs']` is `truncated & ~terminated & ~complete`: it flags pure
time-limit truncations only, excluding kills, misses, and bound violations,
matching the Gymnasium bootstrap convention.

## 9. Initial distribution

One uniform(9) draw per replica covers all random quantities:

- **Target:** horizontal bearing uniform on $[0,2\pi)$ at fixed range
  $R = 2500$ m from the origin; altitude $z_T = 4000 + (u-0.5)\cdot 400$ m
  ($\pm 200$ m). Heading points back toward the missile within a fixed cone:
  $\text{heading} = \operatorname{atan2}(-p_{T,y}, -p_{T,x}) +
  \mathcal U[-37.5^{\circ}, 37.5^{\circ}]$.
- **Maneuver:** type uniform on $\{0,1,2\}$; $a_T = 7.5\times 9.81$ m/s²;
  $\omega_w = 2\pi\times 0.75$ rad/s; weave phase uniform on $[0,2\pi)$;
  barrel direction uniform; barrel phase pinned as in Section 6.
- **Missile:** position $(0, 0, 4000 + (u-0.5)\cdot 100)$ m (altitude jitter
  $\pm 50$ m); roll jitter $\mathcal U[-0.1, 0.1]$ rad; zero angular rate.
  Fire-control lead aims at the straight-line extrapolation of the target over
  the scaled nominal time-to-go,
  \begin{equation}
  t_{\text{go}} = 0.8\,\frac{R}{V_{M0} + V_T} = 4.0\ \text{s},\qquad
  \text{aim} = \frac{p_T + V_T t_{\text{go}} h - p_M}{\lVert p_T + V_T t_{\text{go}} h - p_M\rVert},
  \qquad v_{M0} = 250\cdot\text{aim}\ \text{m/s}, \tag{3}
  \end{equation}
  with attitude taking body $+x$ to $\text{aim}$ plus the roll jitter. The
  perpendicular component of this lead nulls the initial LOS rate;
  `TGO_SCALE` $= 0.8$ (1.0 = nominal closed-form) trimmed overshoot misses,
  +0.8 pp success rate on two disjoint 1024-episode seed populations; a full
  maneuver-model prediction performed worse (the maneuver displacement over the
  flight is tens of metres, while the straight-line lead correctly cancels the
  crossing geometry).
- **No warmup.** Unlike the rocket, reset performs no zero-command warmup: the
  boost engine fires autonomously on the first active physics step, so a warmup
  would burn thrust and shift the spawn state before the first policy step; an
  airborne missile has no contacts to settle.

## 10. Disturbances

Optional wind via the constructor kwargs `steady_wind` (world-frame constant
3-vector, m/s), `dryden` (intensity in $[0,1]$, 0 disables), and `severity`
(`'light' | 'moderate' | 'severe'`; MIL-F-8785C low-altitude reference sigmas
$(1.5,1.5,0.7)$, $(3.0,3.0,1.4)$, $(6.0,6.0,3.0)$ m/s; scale lengths
$(200,200,50)$ m). The first-order gust filters advance once per policy step;
aerodynamic terms evaluate $v_{\text{air}} = v - v_{\text{wind}}$. Calm air
(the default) leaves the dynamics bit-for-bit unchanged.

## 11. Metrics and extras

On any completed episode, `extras['episode']` reports `success_rate`,
`miss_distance` (the per-episode minimum range, tracked at 240 Hz), and the
per-component reward means `rew_{progress, align, ctrl, kill, miss}`;
long-run accumulators in `completed_metrics` add `kill_rate`, `miss_rate`, and
`timeout_rate`. `extras['terminal_observations']` carries the float32 terminal
observation for all replicas; `extras['time_outs']` is the timeout mask of
Section 8.

## 12. Implementation

- File: `environments/missile/env.py`; class `MissileInterceptEnv`; task id
  `missile`. Target kinematics in `_target_state`; launch attitude in
  `_aim_quat`.
- Plant: `environments/common/plant.py` (`TensorPlant`, vehicle `'missile'`);
  parameters `environments/common/original_pyflyt/models/vehicles/missile/missile.yaml`;
  URDF SHA-256 pinned by `consistency.verify_missile_assets`.
- Conventional baseline: `MissileAPNBaseline` (augmented proportional
  navigation).

| Constant | Value |
|---|---|
| `agent_hz` / `inner_steps` | 60 / 2 (control 120 Hz, physics 240 Hz) |
| `max_steps` | 900 (15 s; buffer `max_episode_length` 902) |
| `num_actions` / obs dim | 3 / 34 |
| `OMEGA_MAX` / `OMEGA_LAT` | 6.0 / 1.0 rad/s |
| `KILL_RADIUS` | 15.0 m |
| `R_SPAWN` / `CONE` / `A_LAT` / `F_WEAVE` | 2500 m / 37.5° / 7.5 g / 0.75 Hz |
| `V_T` / `V_M0` / `Z_MID` | 250 / 250 m/s / 4000 m |
| `K_PROG` / `K_ALIGN` / `K_CTRL` | 0.01 / 0.02 / 0.01 |
| `KILL_BONUS` / `MISS_PENALTY` / `TGO_SCALE` | +50 / −10 / 0.8 |
| mass / boost thrust / burn time | 150 kg / 20 000 N / 4.0 s |
| `vel_clamp` | 1200 m/s, 40 rad/s |
| bounds / miss condition | $\lVert p\rVert > 15$ km or $z < 0$ / $r\cdot v_{\text{rel}} > 0$ after $t > 2$ s |

## 13. References

- MIL-F-8785C (low-altitude Dryden turbulence model), via
  `environments/common/disturbances.py`. The task itself is project-authored;
  no external reference implementation exists.
