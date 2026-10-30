# Rocket Landing Environment (`RocketLandingEnv`)

## 1. Overview

The rocket-landing task is an upstream-faithful port of the PyFlyt 0.29.0
`RocketLandingEnv` (`gym_envs/rocket_envs/rocket_landing_env.py` +
`rocket_base_env.py` of the bundled `original_pyflyt` reference): a
Falcon-9-class first-stage booster, released at 400–450 m altitude with a
−100 m/s vertical velocity, must descend, null its lateral offset, and settle
onto a ground pad. Only the rigid-body physics engine differs from upstream
(Genesis instead of PyBullet); the task definition — reward, dynamics model,
episode logic, initial conditions — follows upstream verbatim.

The reward is the 2026-10-28 "v5" dense per-policy-step shaping (benchmark v2
uses this reward together with the 40 Hz policy rate and the 2 s settled
success criterion). v4's dive-only shaping was farmed by a 5 m/s crash-dive
policy: 65.5M transitions, zero successes, positive return. v5 adds a feasible
descent-speed corridor and a one-shot fatal penalty; the environment is fully
tensorized on a shared Genesis scene.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $p\in\mathbb{R}^{3}$ | world-frame position of the booster origin [m]; the pad sits at the origin |
| $p_{xy}$ | horizontal component $(p_x,p_y)$ |
| $q$ | attitude quaternion (wxyz internally, exposed xyzw) |
| $v,\omega$ | world-frame linear / angular velocity; $v_b,\omega_b$ body-frame |
| $(\phi,\theta,\psi)$ | roll, pitch, yaw (Euler angles of $q$) |
| $f\in[0,1]$ | fuel fraction (spawn value 0.05) |
| $a\in[-1,1]^{7}$ | raw policy action; $\bar a$ its affine image in the upstream Box |

## 3. Dynamics and control hierarchy

Three nested rates drive the simulation under benchmark **v2**: **policy 40 Hz**
($\Delta t = 1/40$ s; the rate of the research freeze snapshot — 60 Hz does not
divide the 120 Hz control loop), **control 120 Hz** (`inner_steps = 3` control
periods per policy step, action held constant), and **physics 240 Hz** (two
substeps per control period, $h = 1/240$ s) on the Genesis rigid-body solver
with per-substep actuation updates. All rates are read back from the plant
(`agent_hz` is a constructor kwarg; `benchmark_version='v1'` restores the
legacy 60 Hz configuration for historical comparisons).

The vehicle model is the upstream F9-class booster scaled by $10^{3}$ from
the Falcon 9 first stage (reference propellant 123.5 t RP-1 + 287.4 t LOX =
410.9 t, reference thrust 7.607 MN):

- **Mass properties.** Structural mass 138.2 kg (`base_link` 91.0 kg,
  `booster_link` 47.0 kg, four grid fins at 0.05 kg). The `fueltank_link` mass
  is rewritten every physics step to $f\cdot 410.9$ kg with inertia
  $f\cdot\mathrm{diag}(1678,1678,7.01)$ kg·m², so the spawn mass is
  $138.2 + 0.05\times 410.9 \approx 158.7$ kg (fully fuelled maximum 549.1 kg).
- **Booster.** Reignitable engine: on each physics step the ignition latch is
  set iff the ignition command exceeds 0.5. The throttle target is the remap
  $u\in[0,1]\mapsto 0.39 + 0.61\,u$, tracked by a first-order lag
  ($\tau = 0.01$ s) and multiplied by $1 + 0.01\,\mathcal N(0,1)$ noise each
  physics step. Thrust $T = \text{throttle}\times 7607.0$ N (39% floor =
  2966.7 N) acts at `booster_link` along the gimballed direction
  $(\sin g_y,\ -\sin g_x\cos g_y,\ \cos g_x\cos g_y)$; the two-axis gimbal is
  limited to $\pm 5^{\circ}$ with lag $\tau = 0.01$ s. Fuel burns at 1.451 kg/s
  at full throttle; thrust is zero once $f = 0$.
- **Grid fins.** Four lifting surfaces (`fin_pos_x, fin_neg_x, fin_pos_y,
  fin_neg_y`; chord 0.5 m, span 0.5 m, $C_{l\alpha}=2\pi$, $\eta=0.65$, stall
  $\pm 20^{\circ}$, deflection limit $\pm 45^{\circ}$, lag $\tau=0.05$ s). The
  finlet commands $(a_x,a_y,a_\psi)$ mix as
  $(a_y{+}a_\psi,\ a_y{-}a_\psi,\ a_x{-}a_\psi,\ a_x{+}a_\psi)$, clamped to
  $[-1,1]$.
- **Body drag.** $F = -\tfrac12 \rho\, C_d A\, |v_{\text{air}}|\,v_{\text{air}}$
  with $\rho = 1.225$ kg/m³, $C_d = (1.16, 1.16, 2.0)$,
  $A = (1.7649, 1.7649, 0.1075)$ m², applied at `base_link`.

On reset, a 10-iteration zero-command warmup (20 physics substeps) settles
contacts and actuators before the first observation.

## 4. Action space

The policy emits $a\in[-1,1]^{7}$, clipped elementwise (`clip_actions = 1.0`)
and affinely mapped onto the upstream Box with lows $(-1,-1,-1,0,0,-1,-1)$ and
highs $(1,1,1,1,1,1,1)$. The affine map is deliberate: a zero-mean Gaussian
policy then fires the engine on ~50% of steps, matching upstream exploration.

| Index | Raw range | Mapped range | Physical meaning |
|---|---|---|---|
| 0 | $[-1,1]$ | $[-1,1]$ | finlet roll command $a_x$ |
| 1 | $[-1,1]$ | $[-1,1]$ | finlet pitch command $a_y$ |
| 2 | $[-1,1]$ | $[-1,1]$ | finlet yaw command $a_\psi$ |
| 3 | $[-1,1]$ | $[0,1]$ | ignition (fires iff $>0.5$, per physics step) |
| 4 | $[-1,1]$ | $[0,1]$ | throttle (remapped to the 39–100% thrust range) |
| 5 | $[-1,1]$ | $[-1,1]$ | gimbal $x$ ($\pm 5^{\circ}$) |
| 6 | $[-1,1]$ | $[-1,1]$ | gimbal $y$ ($\pm 5^{\circ}$) |

## 5. Observation space

A single float32 vector of dimension 30 under the `policy` key:

| Slice | Symbol | Unit | Description |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | body angular velocity |
| [3:7] | $q$ (xyzw) | — | attitude quaternion |
| [7:10] | $v_b$ | m/s | body-frame linear velocity |
| [10:13] | $p$ | m | world position |
| [13:20] | $\bar a_{t-1}$ | — | last mapped action (Section 4 ranges) |
| [20:24] | — | — | grid-fin actuations (4) |
| [24:25] | — | — | engine ignited flag |
| [25:26] | $f$ | — | fuel fraction |
| [26:27] | — | — | throttle state |
| [27:29] | $g_x, g_y$ | — | gimbal commands (pre-degree scaling) |
| [29:30] | — | — | pad contact flag |

## 6. Reward

One dense reward per **policy** step, computed from whole-step state deltas
($\Delta t = 1/40$ s), with components matching `REWARD_KEYS`:

\begin{align}
r_{\text{hover}} &= -0.3, \tag{1a}\\
r_{\text{hold}} &= \frac{0.005}{\lVert p_{xy}\rVert + 0.1}\ \le 0.05, \tag{1b}\\
r_{\text{lateral}} &= 10\,\big(\lVert p_{xy}^{-}\rVert - \lVert p_{xy}\rVert\big), \tag{1c}\\
r_{\text{descent}} &= 0.1\,(z^{-} - z), \tag{1d}\\
r_{\text{decel}} &= 0.1\,(\Delta v_z + 1)\,e^{-\max(z,0)}\,\sigma(v_z),
\quad \sigma(v_z) = \begin{cases}+1 & v_z < 0\\ -1 & \text{otherwise,}\end{cases} \tag{1e}\\
r_{\text{attitude}} &= -0.5\,|\omega_{b,z}| - 0.5\,\sqrt{\phi^{2}+\theta^{2}}, \tag{1f}\\
r_{\text{contact}} &= \mathbb{1}[\text{any pad contact}]\cdot\big(0.5 - 1.0\,|v_{z,c}|\big), \tag{1g}\\
r_{\text{corridor}} &= -0.5\,\mathrm{relu}\big(-v_z - (0.5 + \sqrt{2\cdot 2\,z})\big), \tag{1h}\\
r_{\text{success}} &= 5000\cdot\mathbb{1}[\text{first settle this step}], \tag{1i}\\
r_{\text{fatal}} &= -500\cdot\mathbb{1}[\text{new terminal transition}], \tag{1j}
\end{align}

where $(\cdot)^{-}$ denotes the policy-step start, $\Delta v_z$ the whole-step
change in vertical velocity, and $v_{z,c}$ the world descent speed at the first
contact inner step. The total is $r = \sum_k r_k$; per-component sums are
accumulated into `reward_components` and reported as `rew_*` metrics. Design
rationale (discounted-return analysis at $\gamma = 0.995$): the success bonus
satisfies $B = 5000 \gg \gamma c/(1-\gamma) \approx 70$ for the maximum
sustainable per-step contact stream $c$, so delaying a reachable success is
never profitable; the corridor (1h) bounds the descent speed by the
conservative feasible profile $v_{\text{allow}}(z)$ and the one-shot fatal cost
(1j) makes a crash-dive return negative instead of letting it escape the
remaining hover cost; the hold term is capped
at 0.05 per step, below the 0.3 hover cost, so station-keeping cannot be farmed
either. All coefficients are constructor-overridable via the `reward` dict and
recorded into `cfg` / `run.json` for the evolution sweep arms.

## 7. Success, termination, and truncation

Evaluated on every 120 Hz control period, in the following order:

1. **Fatal contact (termination).** Any contact other than (rocket, pad), or
   $z < 0$; or a pad-contact step on which the **previous** inner-step state
   has speed $> 1.0$ m/s or spin $> 0.35$ rad/s.
2. **Out of bounds (termination).** $\lVert p_{xy}\rVert > 200$ m or
   $z > 500$ m.
3. **Success (no termination, no truncation) — v2 settled criterion.** The
   primary metric is `ever_stable_2s`: the vehicle must remain on the pad in
   a settled state for 2 s continuously. A policy-step sample is *settled* iff
   it has pad contact, no fatal state, speed $< 0.02$ m/s, spin $< 0.02$ rad/s
   and tilt $\sqrt{\phi^2+\theta^2} < 0.1$ rad; a gap resets the streak, and a
   fatal/bounds termination invalidates the terminal sample. At 40 Hz this
   requires `ceil(2*40)+1 = 81` consecutive samples (121 at the legacy 60 Hz).
   Reaching the streak latches `ever_stable_2s` and pays the one-shot bonus;
   the episode **continues** and the settled contact stream (Eq. 1g) keeps
   paying until the time limit. The old instantaneous pad-contact latch
   (`ever_success`, plus `terminal_single_point` / `terminal_stable_2s`) is
   still reported as a diagnostic, but **v1 instantaneous success rates are
   not comparable to the v2 score**.
4. **Time limit (truncation).** `count` $\ge 1200$ policy steps (30 s),
   applied after the full control step; the only truncation source.

Pad-contact judgement uses the previous inner-step velocity because upstream
keeps simulating after a touch: touchdown states between the fatal and success
gates simply continue and may settle. `extras['time_outs']` is
`truncated & ~terminated & ~complete`, flagging pure time-limit truncations for
value bootstrapping and excluding both failures and settled episodes.

## 8. Initial distribution

Per replica, from one 6-dimensional uniform draw, with fuel fraction
$f_0 = 0.05$:

\begin{equation}
x, y \sim \mathcal U[-20, 20]\ \text{m},\quad
z \sim \mathcal U[400, 450]\ \text{m},\quad
\phi,\theta,\psi \sim \mathcal U[-0.3, 0.3]\ \text{rad},\quad
v_0 = (0,0,-100)\ \text{m/s},\quad \omega_0 = 0. \tag{2}
\end{equation}

Upstream's randomized drop velocities are dead code there (the option key is
misspelled `randoimize_drop`), so only the −100 m/s drop speed applies; the
quirk is reproduced verbatim.

## 9. Disturbances

Optional wind via the constructor kwargs `steady_wind` (world-frame constant
3-vector, m/s), `dryden` (intensity in $[0,1]$, 0 disables), and `severity`
(`'light' | 'moderate' | 'severe'`; MIL-F-8785C low-altitude reference sigmas
$(1.5,1.5,0.7)$, $(3.0,3.0,1.4)$, $(6.0,6.0,3.0)$ m/s; scale lengths
$(200,200,50)$ m). The first-order gust filters advance once per policy step;
aerodynamic terms evaluate $v_{\text{air}} = v - v_{\text{wind}}$. Calm air
(the default) leaves the dynamics bit-for-bit unchanged.

## 10. Metrics and extras

On any completed episode, `extras['episode']` reports `success_rate`,
`pad_touch_rate`, the first-touch telemetry `touch_speed`, `touch_ang_vel`,
`touch_tilt` (monitoring only; first touch is not part of the episode logic),
and per-component reward means `rew_{hover, hold, lateral, descent, decel,
attitude, contact, success}`; long-run accumulators live in
`completed_metrics`. `extras['terminal_observations']` carries the float32
terminal observation for all replicas; `extras['time_outs']` is the timeout
mask of Section 7.

## 11. Implementation

- File: `environments/rocket/env.py`; class `RocketLandingEnv`; reward in the
  module function `rocket_shaping`; task id `rocket`.
- Plant: `environments/common/plant.py` (`TensorPlant`, vehicle `'rocket'`);
  parameters `environments/common/original_pyflyt/models/vehicles/rocket/rocket.yaml`.
- Conventional baseline: `RocketPDGBaseline` (ZEM/ZEV powered-descent guidance
  with TVC/fin attitude PD).

| Constant | Value |
|---|---|
| `agent_hz` / `inner_steps` | 40 / 3 (control 120 Hz, physics 240 Hz; v1: 60 / 2) |
| `max_steps` | 1200 (30 s; buffer `max_episode_length` 1202; v1: 1800) |
| `num_actions` / obs dim | 7 / 30 |
| `HOVER_COST` / `HOLD_COEF` | 0.3 / 0.005 |
| `LATERAL_COEF` / `DESCENT_COEF` / `DECEL_COEF` | 10.0 / 0.1 / 0.1 |
| `YAW_COEF` / `TILT_COEF` | 0.5 / 0.5 |
| `CONTACT_BASE` / `CONTACT_VEL_COEF` / `SUCCESS_BONUS` | 0.5 / 1.0 / 5000.0 |
| `ic_xy_half` / `ic_z_min` / `ic_euler_half` | 20 m / 400 m (span 50) / 0.3 rad |
| min / max thrust; gimbal range | 2966.7 / 7607.0 N; $\pm5^{\circ}$ |
| fuel capacity / rate / spawn fraction | 410.9 kg / 1.451 kg/s / 0.05 |
| booster / gimbal $\tau$; throttle noise | 0.01 s / 0.01 s; 1% multiplicative |
| success gate | $|v|<0.02$ m/s, $|\omega|<0.02$ rad/s, tilt $<0.1$ rad |
| fatal gate / bounds | contact speed $>1.0$ m/s or spin $>0.35$ rad/s; $\lVert p_{xy}\rVert>200$ m, $z>500$ m, $z<0$ |

## 12. References

- PyFlyt 0.29.0, bundled as `environments/common/original_pyflyt/`:
  `gym_envs/rocket_envs/rocket_landing_env.py` and `rocket_base_env.py` (task
  definition), `core/drones/rocket.py` (vehicle model),
  `core/abstractions/boosters.py` (ignition and throttle remap).
- MIL-F-8785C (low-altitude Dryden turbulence model), via
  `environments/common/disturbances.py`.
