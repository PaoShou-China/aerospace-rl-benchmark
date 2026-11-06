# AerospaceRL Benchmark Technical Report

**English** · [简体中文](benchmark_report.zh-CN.md)

> GPU-parallel reinforcement-learning benchmark for aerospace control — dynamics modeling, task definitions and classical baselines

## Abstract

AerospaceRL Benchmark (hereafter **benchmark**) is a GPU-parallel reinforcement
learning benchmark for aerospace control with **10 tasks**, covering atmospheric
flight (quadrotor waypoints, quadrotor target tracking, fixed-wing waypoints,
tilt-rotor VTOL, rocket vertical landing, missile interception) and spacecraft
dynamics (CMG attitude control, low-thrust orbit transfer, rendezvous, docking).

Core design principles:

- **Pure tensor interface**: every environment exposes a uniform batched
  device-tensor `step()/reset()` protocol and does not depend on
  Gym/Gymnasium; physics integration, contact detection, rewards, resets and
  random numbers are masked tensor kernels, and inactive replicas freeze without
  consuming random numbers.
- **Multiple classical baselines per task**: each task has at least two analytic
  classical GNC controllers (a default baseline plus structural variants), sharing
  the same environment, reward and success criterion as learning algorithms, and
  reading only observations rather than internal state.
- **Three nested rates**: policy → control → physics; atmospheric tasks use the
  Genesis rigid-body backend, while spacecraft tasks use pure tensor integrators
  (RK4 / symplectic Euler).

This report focuses on each task's **plant dynamics modeling**, **physical
parameters**, **reward design** and **classical methods**, and introduces the
learning-algorithm suite (including residual RL and gain-scheduling RL built on
classical controllers) and the frozen evaluation protocol.

---

## 1. System architecture and shared infrastructure

### 1.1 Tensor environment interface

All environments inherit a common `BaseEnv` and follow this contract:

- **Single device, double precision**: physical state is `float64`, observation
  and reward buffers are `float32`, and all state resides on the same device.
- **No per-replica Python loop**: everything inside `step()` is a batched tensor
  kernel; each `step()` performs exactly one host-device synchronization
  (`bool(done.any())`) for episode-boundary bookkeeping.
- **Observation groups**: observations are returned as a `TensorDict`, with the
  policy-visible vector under `'policy'`; classical baselines and learned policies
  consume the same vector.
- **Time conventions**: `extras['time_outs']` marks pure time-limit truncation
  (for value bootstrapping); `extras['terminal_observations']` carries the
  pre-reset observation.

### 1.2 Rigid-body plant and tensor dynamics

The 6 atmospheric tasks (quadx / quadx_track / fixedwing / quadplane / rocket /
missile) use `TensorPlant` (`environments/common/plant.py`), which uniformly
carries:

- geometry and inertia (parsed from the frozen reference URDF)
- actuation layer: rate-PID mixer, lifting surfaces (`aerodynamics.py`),
  booster/fuel
- physics stepping, contact checks, reset/warmup
- per-replica independent counter random streams `TensorRandom` (`mix32` removes
  seed overlap)

The 4 spacecraft tasks (satellite / transfer / docking / rendezvous) are pure
tensor implementations with no physics engine; state, random numbers and
statistics all stay on device.

### 1.3 Unified wind-disturbance hook

QuadX-Waypoints, Fixedwing-Waypoints, Quadplane-VTOL, Rocket-Landing and
Missile-Intercept (5 atmospheric tasks) share `WindDisturbance`
(`disturbances.py`): world-frame steady wind + Dryden gusts. Constructor
parameters are `steady_wind` (m/s), `dryden ∈ [0,1]` and
`severity ∈ {light, moderate, severe}`; `dryden=0` with no steady wind is calm
air. At the 300 ft reference altitude the three-axis standard deviations are
light `(1.5,1.5,0.7)`, moderate `(3.0,3.0,1.4)` and severe `(6.0,6.0,3.0)` m/s,
with scale lengths `(200,200,50)` m; the actual standard deviation is multiplied
by `dryden`. RNG is determined by the environment seed, and filter state is
cleared on episode reset. All aerodynamic forces are uniformly evaluated against
**airspeed** $v_{\text{air}} = v - w$; the calm-air path is bit-identical to the
no-wind benchmark. QuadX-Track's constructor does not accept wind parameters and
is always calm. Training entries hard-code these default wind configurations:
quadx / fixedwing / quadplane `dryden=0.3`, rocket `dryden=0.0` (same convention
as the calm-air baseline), missile no wind parameters. `scripts/eval.py`
restores the run's `env_kwargs` from `results/<task>/config.json` when present,
so evaluation follows the training wind configuration. Cross-method primary
comparisons must explicitly fix the same wind configuration (calm air or the same
Dryden tier per task) and report disturbance results separately.

### 1.4 Coordinates, units and model boundaries

- Unless stated otherwise, positions and velocities use a right-handed inertial
  world frame in m and m/s; body velocity/angular rate with subscript $b$ means
  body frame; CWH tasks fix the LVLH frame as $x$ radial outward, $y$ along-track,
  $z$ cross-track.
- For rigid-body atmospheric tasks, the **observation quaternion is `xyzw` and
  represents a body-to-world rotation**; the Genesis plant internally uses
  `wxyz` and converts only at the plant boundary. Satellite is an explicit
  exception: its internal dynamics and observations use `wxyz` and Hamilton
  multiplication, and attitude error is the shortest-arc logarithmic map of
  $q^{-1}\otimes q_{\text{target}}$.
- Aerodynamic models use blade-element/lifting-surface and drag approximations,
  not full unsteady aerodynamics, structural elasticity or high-fidelity
  propulsion models; Fixedwing results mainly apply within the given speed,
  angle-of-attack and stall-model range, Missile's slender-body normal-force
  model cannot replace a full-airframe aerodynamic database, and Rocket's grid-fin
  model does not cover all effects of a real deployable mechanism.
- CMG uses a reduced-order rigid-body model with constant rotor speed; Transfer
  uses a two-body point-mass model; Docking/Rendezvous use the linear CWH model
  near a circular orbit. These tasks are control-algorithm benchmarks and must
  not be interpreted as engineering validation of real vehicles.

### 1.5 Distribution contract and reproduction

Each task's default reset distribution is the official benchmark distribution.
A training curriculum may change only initial-state or task-parameter
distributions, not rewards, success criteria, termination conditions or physical
parameters; evaluation restores the frozen `env_kwargs` from `config.json`.
Training, validation and held-out testing use different RNG streams, and the test
stream never participates in tuning or model selection. Classical controllers may
read only `'policy'` observations and nominal model constants frozen at
construction; they may not read runtime plant hidden state.

PyFlyt reference source, URDF/YAML physical assets and the project-created
Missile and Rocket-v2 assets are SHA-256 verified by
`environments/common/consistency.py`; the plant contract is serialized into a
`plant_id`. Every experiment should save the code version, dependency versions,
environment configuration, algorithm configuration and seed list together, so
results can be reproduced bit-for-bit or version differences explained
explicitly.

---

## 2. Task list

This chapter groups tasks by atmospheric flight and spacecraft dynamics and
details each task's plant dynamics, parameters, rewards and classical methods.
It describes only the environments and contains no test results or evaluation
data.

**Classical baseline coverage table** ("default + two structural variants";
all read policy observations only; performance status is reported separately in
the experimental chapter):

| Task | Default baseline | Structural variant 1 | Structural variant 2 |
|---|---|---|---|
| quadx | SE(3) geometric control | cascaded PID | sliding-mode position loop + Euler attitude loop |
| quadx_track | CV Kalman + SE(3) | raw-measurement pursuit | EMA filter + SE(3) |
| fixedwing | L1 + TECS | cascaded PID | nonlinear guidance law (NLGL) + TECS |
| quadplane | SE(3) + pusher | hybrid cascaded PID | L1 fixed-wing + low-speed rotor assist |
| rocket | ZEM/ZEV PDG | position/velocity PD | E-guidance |
| missile | APN | pure pursuit (PPN) | true proportional navigation (TPN) |
| satellite | damped pseudo-inverse CMG | ordinary Moore–Penrose pseudo-inverse | weighted pseudo-inverse + null-space singularity avoidance |
| transfer | Gauss one-step greedy | Petropoulos Q-law | Edelbaum budget tangential guidance |
| docking | CWH-LQR + constraint layer | CWH pole placement | CWH ZEM/ZEV |
| rendezvous | two-impulse CWH | CWH-PD | glideslope range–rate braking |

### 2.1 QuadX-Waypoints (quadrotor waypoint navigation)

**Task**: a Crazyflie 2.x micro quadrotor (27 g, X configuration) visits 4 random
waypoints in order inside a 5 m flight dome, capturing each within a 0.2 m
sphere, with a 20 s limit; ground contact or out-of-bounds fails.

#### Plant dynamics

**Three rates**: policy 40 Hz → control 120 Hz (`inner_steps=3`) → physics
240 Hz (2 substeps per control period), 6 physics substeps per policy step.

**Inner rate PID (120 Hz)**: tracks the commanded body angular rate
$\omega^{\text{cmd}}$ and outputs normalized torque $u\in[-1,1]^3$:

$$u_t = \text{sat}\big(k_p e_t + I_t + k_d (e_t - e_{t-1})/\Delta t_c\big), \quad I_t = \text{sat}(I_{t-1} + k_i e_t \Delta t_c)$$

where $e_t = \omega^{\text{cmd}} - \omega_b$, $k_p=(0.04,0.04,0.08)$,
$k_i=(5\times10^{-7},5\times10^{-7},2.7\times10^{-4})$,
$k_d=(10^{-4},10^{-4},0)$. Torque and total thrust are allocated to 4 motors
through an X-configuration mixer.

**Actuation (240 Hz)**: motor state first-order lag + multiplicative noise:

$$\theta_{t+1} = \theta_t + \frac{\Delta t}{\tau_m}(\pi_t - \theta_t), \quad \theta_t \leftarrow \theta_t(1 + 0.02\,\xi_t),\ \xi_t \sim \mathcal{N}(0,1)$$

$\tau_m=0.01$ s. The multiplicative noise is zero-mean, and Gaussian samples are
not additionally truncated at the noise layer; the absolute value in the
quadratic thrust law only fixes the thrust-direction convention, not noise
truncation. Single-motor quadratic thrust law
$f_i = |\theta_i|\theta_i\,F_{\max}/4$, $F_{\max}=2.0$ N; reaction torque
$\tau_{z,i}=f_i(k_m/k_f)s_i$. Body drag
$F_d = -\frac12\rho C_d A |v_{\text{air}}|\circ v_{\text{air}}$
($C_d=3.0$, $A=4\times10^{-4}$ m²), angular drag
$\tau_d=-10^{-4}|\omega_b|\circ\omega_b$. Velocity clamps
$\|v\|\le 15$ m/s, $\|\omega\|\le 30$ rad/s.

#### Action space

$a\in[-1,1]^4$ affinely maps to
$(\omega_x^{\text{cmd}},\omega_y^{\text{cmd}},\omega_z^{\text{cmd}},T)$ with
ranges $(\pm\pi,\pm\pi,\pm\pi,[0,0.8])$.

#### Observation space and termination (37 dims)

Body angular rate (3) + quaternion `xyzw` (4) + body velocity (3) + world position
(3) + previous action (4) + motor state (4) + body-frame relative positions of 4
waypoints (12) + validity flags (4), 37 dims, `float32`, not normalized inside
the environment. The current active waypoint is the first relative-position
group. Visiting all 4 waypoints within the limit is success (truncation);
ground contact or $\|p\|>5$ m is failure termination; timeout at 800 steps
without completion does not count as success.

#### Parameter table

| Parameter | Value |
|---|---|
| Mass / total thrust | 0.027 kg / 2.0 N |
| Inertia $J$ | $\text{diag}(1.4,1.4,2.17)\times10^{-5}$ kg·m² |
| Motor lag $\tau_m$ / noise ratio | 0.01 s / 0.02 |
| rate-PID $k_p,k_i,k_d$ | $(.04,.04,.08),(5\text{e-}7,5\text{e-}7,2.7\text{e-}4),(1\text{e-}4,1\text{e-}4,0)$ |
| pwm window / total thrust cap | $[0.05,1]$ / 0.8 |
| Capture radius / out-of-bounds radius | 0.2 m / 5 m |
| Waypoint shell / altitude floor | $[1.5,4.8]$ m / 0.4 m |
| Initial state | $(0,0,1)$ m, at rest and level |

#### Reward

Potential-based telescoping shaping:

$$\Phi(s) = -\big(\|p - w_i\| + L(i)\big), \quad r_t = \Phi(s_{t+1}) - \Phi(s_t) + 15\,c_t - 10\,\kappa_t$$

$L(i)$ is the remaining-leg length sum, $c_t$ the number of captures and
$\kappa_t$ the crash indicator. The telescoping potential difference guarantees
path independence: a successful episode accumulates exactly the reference path
length, and a crash forfeits unrealized potential.

#### Classical method: SE(3) geometric control (Lee et al. 2010)

- **Position loop**: saturated-velocity cascaded PD,
  $v_d = -(K_X/K_V)e_x$ clipped to $V_{\max}=1.5$ m/s;
  $F_{\text{des}} = m(g e_3 - K_V e_v)$, with an axis-wise integral trim canceling
  hover residual.
- **Attitude loop**: geometric control on $SO(3)$,
  $b_{3,d} = F_{\text{des}}/\|F_{\text{des}}\|$, torque
  $M = -K_R e_R - K_\Omega \omega_b + \omega_b \times J\omega_b$.
- **Allocation**: closed-form wrench→rotor-thrust mapping, pre-compensating the
  motor-lag DC gain $G_{\text{pwm}}=1.216$, then inverting the mixer to obtain
  rate commands.
- **Structural variants**: `cascaded_pid` uses position–velocity–attitude
  cascaded PID; `smc` uses a sliding-mode position outer loop and Euler attitude
  inner loop.

---

### 2.2 QuadX-Track (quadrotor persistent target tracking)

**Task**: the same quadrotor keeps a cooperative ground moving target visible
inside a nadir camera's field of view for 30 s, while holding a fixed 6 m altitude
and fixed horizontal offset (1.5, 0) m; a visibility fraction ≥ 0.85 with no crash
is success. The plant is identical to QuadX-Waypoints; perception and reward
differ.

#### Plant dynamics

As in §2.1: 40/120/240 Hz three rates, rate PID, X-configuration mixer, motor lag
and noise, quadratic thrust, body/angular drag.

**Cooperative ground target**: a kinematic vehicle at $z_t=0.15$ m, 12
pre-sampled segments (straight/turn/start-stop), speed 0.4–1.2 m/s, turn rate
0.3–1.0 rad/s, with reflecting boundaries inside a 15 m circle.

**Geometric camera model**: 30° half-angle downward cone, slant-range window
1.5–12 m, 6 static cylindrical occluders (pure LOS geometry, not physical
bodies). When visible, returns the target relative position with 0.15 m noise;
when not visible, the observation is 0 + flag.

#### Observation space (25 dims)

Body angular rate (3) + quaternion (4) + body velocity (3) + world position (3) +
previous action (4) + motor state (4) + target measurement (3) + visibility
flag (1). No target ground truth.

#### Reward

$$r_t = 0.10\,y_t - 0.10(1-y_t) - 0.05\,\|p_{xy}-\bar p_{xy}\|\,y_t - 0.01(z-6)^2 - 0.002\|\bar a_t\|^2 - 10\,\kappa_t$$

The visibility term pays only when $y_t=1$; loss incurs only the −0.10 penalty.

#### Classical method: constant-velocity Kalman filter + SE(3) tracking

- **CV Kalman**: state $[p_t, v_t]\in\mathbb{R}^6$, predicted each step; updated
  only when visible ($R=\sigma^2 I_3$, $\sigma=0.15$ m).
- **Timeout hold**: when visibility age > 3 s, freeze the reference point and
  hover instead of searching.
- **Control**: reuse the SE(3) cascade with world-frame target-velocity feedforward
  $v_d = \hat v_t - (K_X/K_V)(p-\bar p)$.
- **Structural variants**: `pursuit` performs no state estimation and uses noisy
  measurements directly; `pid_ema` replaces the Kalman filter with an exponential
  moving average.

Ground contact or $\|p\|>30$ m fails immediately; otherwise, at the 1200-step
timeout, success is judged from the whole-episode visibility fraction, and success
is not truncated early.

---

### 2.3 Fixedwing-Waypoints (fixed-wing waypoint navigation)

**Task**: a 2.35 kg, 1.6 m wingspan UAV starts from $(0,0,10)$ m at 20 m/s and
visits 4 waypoints in order inside a 20–65 m spherical shell (10 m capture
sphere), with a 120 s limit; ground contact or leaving the 100 m sphere fails.

#### Plant dynamics

**Three rates**: policy 30 Hz → control 120 Hz (`inner_steps=4`) → physics
240 Hz, 8 physics substeps per policy step. There is no inner attitude loop; the
policy directly drives control surfaces and throttle.

**Control surfaces (240 Hz)**: 5 lifting surfaces (left/right aileron,
horizontal tail, vertical tail, main wing) with first-order lag $\tau_s=0.05$ s.
PyFlyt blade-element lifting-surface model: finite-span lift-line slope, 30%
flap chord ratio + viscous correction $\eta=0.65$, zero-lift angles (wing/aileron
$-2^\circ$, tail $0^\circ$), positive/negative stall (wing $+14^\circ/-9^\circ$,
tail $\pm9^\circ$), skin-friction drag $C_{d0}=0.01$. Force
$\bar q S(c_l, c_d)$, torque $\bar q S c\, c_m$.

**Engine (240 Hz)**: throttle first-order lag + zero-mean multiplicative noise
$\xi\sim\mathcal{N}(0,1)$, $\tau_m=0.01$ s. Propeller thrust
$F=|\theta|\theta\,F_{\max}$, $F_{\max}=18$ N along body $+x$, reaction torque
$F(k_m/k_f)$. Gaussian samples are not additionally truncated at the noise layer.

#### Action space

$a\in[-1,1]^4$: $a_0$ differential aileron ($\pm30^\circ$), $a_1$ elevator +
rudder (opposite signs, $\pm20^\circ$), $a_2$ dead channel (main-wing camber 0),
$a_3$ throttle mapped to $[0,1]$. Effective dimension 3.

#### Observation space and termination (38 dims)

Body angular rate (3) + quaternion `xyzw` (4) + body velocity (3) + world position
(3) + previous action (4) + control-surface/throttle state (6) + body-frame
relative positions of 4 waypoints (12) + validity flags (4), 38 dims, not
normalized inside the environment. Visiting all 4 waypoints within
120 s / 3600 steps is success; ground contact or $\|p\|>100$ m is failure; other
timeouts are incomplete.

#### Parameter table

| Parameter | Value |
|---|---|
| Mass / wingspan | 2.35 kg / 1.6 m |
| Max thrust / motor lag | 18 N / 0.01 s |
| Control-surface lag $\tau_s$ | 0.05 s |
| Surface deflection limits (aileron L/R, tail, main wing) | $30^\circ,30^\circ,20^\circ,20^\circ,0^\circ$ |
| Capture radius / out-of-bounds radius | 10 m / 100 m |
| Waypoint shell / altitude floor | $[20,65]$ m / 20 m |
| Initial state | $(0,0,10)$ m, $v_0=(20,0,0)$ m/s |

#### Reward

The same potential-telescoping shaping as QuadX:
$\Phi(s) = -(\|p-w_i\| + L(i))$,
$r_t = \Phi(s_{t+1})-\Phi(s_t) + 15\,c_t - 10\,\kappa_t$. This replaces upstream's
$1/d$ shaping (which gives unbounded return for orbiting) and $\pm100$ terminal
rewards (whose credit assignment is unreachable over 3600 steps).

#### Classical method: L1 lateral guidance + TECS longitudinal autopilot

- **L1 guidance**: distance-scheduled
  $L_1 = \text{clip}(k_{L1} d_h + L1_0, L1_{\min}, L1_{\max})$, lateral acceleration
  $a_{\text{lat}} = 2V^2\sin\eta_c / L_1$, converted to a roll command. Includes an
  aerodynamic roll limit with an accelerated-stall boundary, terminal pure-pursuit
  blending, and an OOB boundary shield.
- **TECS**: total-energy error $\delta_E \to$ throttle (P+I, anti-windup bleed),
  energy-balance error $\delta_B \to$ pitch command (with coordinated-turn
  feedforward, underspeed protection and overspeed climb trade-off).
- **Structural variants**: `pid_cascade` removes the L1/TECS guidance heuristics;
  `nlgl_tecs` keeps TECS and replaces L1 with Park's nonlinear guidance law.

---

### 2.4 Quadplane-VTOL (tilt-rotor hybrid UAV waypoints)

**Task**: a 2.35 kg hybrid UAV (4 rotor motors + fixed-wing surfaces + forward
pusher propeller) starts near hover ($z=15$ m, $|v|\sim6$ m/s), visits 4
waypoints in a 40–75 m shell at ≥25 m altitude, capture radius 15 m,
out-of-bounds 120 m, 30 s / 1200 steps / 40 Hz. The plant has no explicit mode
switch; physics itself determines the rotor/aerodynamic authority transition
region (~3–12 m/s body-axis speed).

#### Plant dynamics

- **40/120/240 Hz** three rates, `inner_steps=3`.
- **Rotor channel**: the same rate PID as QuadX (40 Hz policy commands body
  angular rate + total thrust) → X-configuration mixer → 4 motors.
- **Fixed-wing channel**: the same 5-surface blade-element model as Fixedwing +
  forward pusher throttle (a 5th throttle channel mapped directly to plant
  `throttle[:,4]`).
- **Mass/thrust**: 2.35 kg; rotor maximum total thrust is on the same order as
  fixed-wing $F_{\max}=18$ N.

#### Action space (8 dims)

| Index | Meaning | Range |
|---|---|---|
| a0–a2 | body angular-rate command (p,q,r) | $\pm\pi$ rad/s |
| a3 | quadrotor total thrust | $[0,1]$ |
| a4–a6 | fixed-wing surfaces (differential aileron/elevator+rudder/dead main wing) | $[-1,1]$ |
| a7 | forward pusher throttle | $[0,1]$ |

#### Observation space and termination (42 dims)

Body angular rate (3) + quaternion `xyzw` (4) + body velocity (3) + world position
(3) + previous action (8) + 5 surface and 5 motor states (10) + body-frame
relative positions of 4 waypoints (12) + validity flags (4), 42 dims, not
normalized inside the environment. Visiting all 4 waypoints within 1200 steps is
success; ground contact or $\|p\|>120$ m is failure; other timeouts are
incomplete.

#### Parameter table

| Parameter | Value |
|---|---|
| Mass | ~2.35 kg |
| policy/control/physics | 40 / 120 / 240 Hz |
| Capture radius / out-of-bounds radius | 15 m / 120 m |
| Waypoint shell / altitude floor | $[40,75]$ m / 25 m |
| Initial state | $z=15$ m, horizontal speed 6 m/s, level |
| Steps | 1200 (30 s) |

#### Reward

The same potential-telescoping shaping as quadx/fixedwing:
$\Phi(s) = -(d_{\text{current}} + L_{\text{remaining}})$,
$r_t = \Phi(s_{t+1}) - \Phi(s_t) + 15\cdot\text{reach\_count} - 10\cdot\text{crash}$.
Reward is positive while approaching the current target.

#### Classical method: SE(3) rotor control + forward pusher

The default `QuadplaneSE3PusherBaseline` uses SE(3) position/attitude control and
provides forward-pusher throttle. The structural variant `hybrid_cascade` uses
speed-scheduled hybrid quadrotor cascaded PID and fixed-wing channels, and
`l1_wing` uses L1 lateral guidance at high speed with rotor assist at low speed.
All three controllers are registered; whether each meets the task requirement is
decided by experimental results, not preset in the environment chapter.

---

### 2.5 Rocket-Landing (rocket vertical landing)

**Task**: a Falcon-9-class first-stage booster is released from 400–450 m at a
−100 m/s vertical velocity and must null lateral offset and settle on the ground
pad. The primary metric is a continuous 2-second settle (81 consecutive samples
at 40 Hz).

#### Plant dynamics

**Three rates**: policy 40 Hz → control 120 Hz (`inner_steps=3`) → physics
240 Hz, 30 s / 1200 steps.

**Mass properties**: structural mass 138.2 kg; the fuel-tank mass is rewritten
each physics step as $f\cdot 410.9$ kg ($f$ is the fuel fraction, initially
0.05), giving an initial mass ≈158.7 kg.

**Booster**: restartable engine; throttle target
$u\in[0,1]\mapsto 0.39+0.61u$, first-order lag $\tau=0.01$ s + 1% multiplicative
noise. Thrust $T=\text{throttle}\times 7607.0$ N (39% floor = 2966.7 N), acting
along the gimbal direction. Two-axis gimbal limit $\pm5^\circ$, lag 0.01 s. Fuel
consumption 1.451 kg/s at full throttle.

**Grid fins**: 4 lifting surfaces (chord 0.5 m, span 0.5 m, $C_{l\alpha}=2\pi$,
stall $\pm20^\circ$, deflection limit $\pm45^\circ$, lag 0.05 s).

**Body drag**: $F = -\frac12\rho C_d A |v_{\text{air}}| v_{\text{air}}$,
$C_d=(1.16,1.16,2.0)$, $A=(1.7649,1.7649,0.1075)$ m².

#### Action space (7 dims)

$(f_x, f_y, f_\psi, \text{ign}, \text{thr}, g_x, g_y)$: grid-fin roll/pitch/yaw
commands, ignition (>0.5 triggers), throttle (maps 39–100%), gimbal $x/y$
($\pm5^\circ$).

#### Observation space and termination (30 dims)

Body angular rate (3) + quaternion `xyzw` (4) + body velocity (3) + world position
(3) + previous action (7) + grid-fin/ignition/fuel/throttle/gimbal state (9) +
pad-contact flag (1), 30 dims, not normalized inside the environment. Success
means a continuous 2 s settle within 30 s without a fatal failure; after success
simulation continues to the limit. Fatal contact, ground collision or
out-of-bounds terminate immediately; ordinary timeout is not success.

#### Parameter table

| Parameter | Value |
|---|---|
| Structural mass / full fuel mass | 138.2 kg / 410.9 kg |
| Initial fuel fraction | 0.05 |
| Max thrust / min thrust ratio | 7607 N / 0.39 |
| Gimbal range / lag | $\pm5^\circ$ / 0.01 s |
| Fuel consumption rate | 1.451 kg/s at full throttle |
| Grid-fin $C_{l\alpha}$ / stall / deflection limit | $2\pi$ / $\pm20^\circ$ / $\pm45^\circ$ |
| Initial IC | $x,y\in[-20,20]$ m, $z\in[400,450]$ m, $\phi,\theta,\psi\in[-0.3,0.3]$ rad, $v_0=(0,0,-100)$ m/s |
| Success gate (2 s settle) | $\|v\|<0.02$ m/s, $\|\omega\|<0.02$ rad/s, tilt $<0.1$ rad |
| Fatal gate | contact speed $>1.0$ m/s or spin $>0.35$ rad/s |

#### Reward (dense shaping)

Per policy step: $r = r_{\text{hover}} + r_{\text{hold}} + r_{\text{lateral}} + r_{\text{descent}} + r_{\text{decel}} + r_{\text{attitude}} + r_{\text{contact}} + r_{\text{corridor}} + r_{\text{success}} + r_{\text{fatal}}$

Key terms: hover cost −0.3; lateral progress
$10(\|p_{xy}^-\|-\|p_{xy}\|)$; corridor penalty bounding descent speed by the
feasible profile $v_{\text{allow}}(z) = 0.5 + \sqrt{2\cdot 2\cdot z}$; success
bonus +5000; fatal penalty −500.

#### Classical method: ZEM/ZEV powered-descent guidance (PDG)

- **Midcourse**: closed-form ZEM/ZEV braking when $s\ge25$ m/s, with $t_{\text{go}}$
  re-solved from a constant-deceleration profile and lateral demand clipped to a
  tilt-consistent budget.
- **Terminal 1**: minimum-thrust manifold braking along a virtual pad (2 m above
  the real pad) when $s<25$ m/s.
- **Terminal 2**: constant-velocity pulse servo when $s\le1.5$ m/s and $h\le2$ m,
  with velocity hysteresis toggling/re-igniting the engine.
- **Attitude**: PD tilt loop + aerodynamic torque feedforward, with torque
  allocated TVC-first and grid fins supplementing.
- **Structural variants**: `pd_landing` uses position/velocity PD; `eguidance`
  uses quadratic-polynomial E-guidance targeting in midcourse.

---

### 2.6 Missile-Intercept (missile interception)

**Task**: a 150 kg slender-body tail-controlled missile intercepts a kinematic
3-DOF target. The target flies at 250 m/s at ~4 km altitude, performing
straight/sinusoidal weave/horizontal barrel roll; initial range 2500 m, target
heading cone $\pm37.5^\circ$, lateral overload 7.5 g. Proximity-fuze lethal
radius 15 m.

#### Plant dynamics

**Three rates**: policy 60 Hz → control 120 Hz (`inner_steps=2`) → physics
240 Hz, 15 s / 900 steps.

**Airframe**: 150 kg rigid body. Axial body drag
$F_x = -\frac12\rho C_{d,x}A_x|v_x|v_x$ ($C_{d,x}=0.25$, $A_x=0.0314$ m²).
Slender-body normal force
$F_{\text{lat}} = -\frac12\rho\|v\|A_x C_{n\alpha} v_{\text{lat}}$
($C_{n\alpha}=16$), acting at the center of pressure $x_{\text{cp}}=-0.45$ m
(generating steering force, weathercock restoring torque and rotational
damping). Combined pitch/yaw damping
$\tau = -c_{mq}\|v\|\omega_{\text{lat}}$ ($c_{mq}=4.0$).

**Tail fins**: 4 fins at $x=-1.3$ m (chord 0.45 m, span 0.40 m,
$C_{l\alpha}=3.0$, stall $\pm40^\circ$, deflection limit $\pm35^\circ$, lag
0.02 s).

**rate PID + skid-to-turn mixer**: $k_p=5.0$ body angular-rate proportional
control, roll channel reduction ratio $g=0.01$; fin commands
$(q_c+g p_c, q_c-g p_c, -r_c-g p_c, -r_c+g p_c)$.

**Engine**: autonomous ignition, 20 kN along body $+x$ for 4 s then coasting;
no throttle channel.

#### Action space (3 dims)

Body angular-rate commands: roll $\pm6.0$ rad/s, pitch/yaw $\pm1.0$ rad/s.

#### Observation space and termination (34 dims)

Body angular rate (3) + quaternion `xyzw` (4) + body velocity (3) + world position
(3) + previous action (3) + tail-fin and engine state (7) + body-frame target
relative position (3) + relative velocity (3) + LOS angular rate (3) + closing
speed (1) + target speed (1), 34 dims, not normalized inside the environment.
Range below 15 m is a successful intercept and ends with truncation; passing the
closest approach by 2 s without intercepting, ground contact or $\|p\|>15$ km is
failure termination; timeout at 15 s without a hit is not success. A hit in the
same inner step takes priority over a miss.

#### Parameter table

| Parameter | Value |
|---|---|
| Mass | 150 kg |
| Axial drag $C_{d,x} A_x$ | 0.25 × 0.0314 m² |
| Normal force $C_{n\alpha}$ | 16 |
| Tail-fin $C_{l\alpha}$ / stall / deflection limit | 3.0 / $\pm40^\circ$ / $\pm35^\circ$ |
| Boost thrust / burn time | 20 kN / 4.0 s |
| Initial range / heading cone / lateral overload | 2500 m / $\pm37.5^\circ$ / 7.5 g |
| Target/missile speed | 250 / 250 m/s |
| Lethal radius | 15 m |

#### Reward

$$r_{\text{progress}} = 0.01(d^- - d),\quad r_{\text{align}} = 0.01(\hat v\cdot\hat r),\quad r_{\text{kill}}=+100,\quad r_{\text{miss}}=-10\min(d_{\min}/500,1)$$

The alignment term is PN-consistent dense shaping and cannot be farmed by flying
away from the target.

#### Classical method: augmented proportional navigation (APN, N=15)

- **LOS-rate filtering**: first-order low-pass $\alpha=0.8$; frozen when terminal
  $V_c<50$ m/s.
- **APN law**: $a^{\text{cmd}} = N V_c(\lambda_w\times\hat r_w) + \frac{N}{2}(g - (g\cdot\hat r_w)\hat r_w)$.
- **skid-to-turn kinematic inverse**: $\omega^{\text{cmd}} = (b_x\times a_{\text{lat}})/V$.
- **Structural variants**: `ppn` is pure pursuit without lead; `tpn` is true
  proportional navigation without gravity augmentation.

---

### 2.7 Satellite Attitude (CMG attitude control)

**Task**: a rigid spacecraft carrying $N$ single-gimbal control moment gyros
(CMGs) maneuvers from a perturbed initial attitude to a random target attitude.
Both 4-CMG pyramid and 6-CMG orthogonal configurations are supported. 30 s /
1200 steps / 40 Hz, pure tensor RK4.

#### Plant dynamics

**Rotor-axis evolution** (Rodrigues rotation):

$$r_i(\theta_i) = \cos\theta_i\, r_i(0) + \sin\theta_i\,(g_i\times r_i(0))$$

**Rigid body + CMG reduced-order model**:

$$I\dot\omega = -\omega\times(I\omega) - \omega\times\big(h\sum_i r_i\big) - h\sum_i \dot\theta_i(g_i\times r_i)$$

$$\dot q = \tfrac12 q\otimes(0,\omega),\quad \dot\theta = \delta$$

Classical RK4 integration, $\mathrm{d}t=2.5$ ms × 10 substeps/control step.
Quaternions are renormalized each substep, and gimbal angles wrap to
$[-\pi,\pi)$.

**4-CMG pyramid**: $I=\text{diag}(75,60,113)$ kg·m², rotor inertia 0.2 kg·m² at
100 rad/s, $h=20$ kg·m²/s, skew angle $\beta=\arctan\sqrt{2}$.

**6-CMG orthogonal**: $I=\text{diag}(0.055,0.055,0.085)$ kg·m², rotor at
200 rad/s, $h=0.0125$.

#### Action space

$u\in[-1,1]^N$, $\delta = 3.0\cdot u$ rad/s per gimbal.

#### Observation space ($14+3N$ dims)

Quaternion `wxyz` (4) + attitude-error logarithm (3) + body angular rate (3) +
$\sin\theta(N)$ + $\cos\theta(N)$ + $\dot\theta(N)$ + normalized manipulability
$m(J)=\sqrt{\det(JJ^\top)}/h^3$ (1) + damped-allocation features (3).

#### Reward

Potential telescoping with two smooth gates:

$$\Phi(s) = 30\,e + 70\,e\,g_n(e) + \frac{30\|\omega\|^2}{2\cdot 1.0}g(e) + \frac{0.5}{m(J)+0.5}$$

$$r_t = \Phi(s_t) - \Phi(s_{t+1}) + \mathbb{1}[\text{success}]\cdot 1.0 - \mathbb{1}[\text{fail}]\cdot 10.0 - 0.0005\sum_i\delta_i^2$$

$g_n$ steepens the error slope to ~100 when $e<0.5$ rad; $g$ charges a braking
term for fast arrival when $e<0.8$ rad. The singular term's soft denominator
avoids $1/m(J)$ spikes.

#### Success/termination

Success: $e<0.05$ rad and $\|\omega\|<0.02$ rad/s, immediately truncated with a
success reward. Failure: $\|\omega\|>10$ rad/s. Timeout: 1200 steps. The initial
target attitude is sampled uniformly on SO(3), and the initial small angular-rate
perturbation scale is `reset_noise_scale=0.01`; changing this scale or the CMG
configuration is a training/test distribution setting and must be recorded in the
result conventions.

#### Classical method: quaternion PD + Schaub–Junkins damped pseudo-inverse CMG steering

- **PD + exact feedforward**: $\dot\omega_d = k_p e - k_d\omega$,
  $\tau_{\text{cmd}} = I\dot\omega_d + \omega\times(I\omega) + \omega\times h_\Sigma$.
- **Singularity-robust steering**:
  $\dot\theta = -\frac{1}{\kappa}J(J^\top J + \lambda^2 I_3)^{-1}\tau_{\text{cmd}}$,
  $\lambda = \Omega(\lambda_0 + 0.15 e^{-3m})$, with damping increasing as
  manipulability collapses.
- **Structural variants**: `quat_pd_mp_pinv_cmg` uses an ordinary Moore–Penrose
  pseudo-inverse; `quat_pd_wpinv_cmg` uses a weighted pseudo-inverse with a
  null-space singularity-avoidance term.

---

### 2.8 Low-Thrust Transfer (low-thrust perigee raising)

**Task**: a 100 kg MEO satellite carrying three orthogonal thrust pairs
(10 mN/axis, $I_{sp}=4000$ s EP) raises the perigee of an initial
$a=10000$ km, $e=0.15$ orbit within 166 steps × 300 s (~13.8 h, about 5 orbits),
keeping apogee fixed while saving fuel.

#### Plant dynamics

Two-body point-mass gravity + constant thrust, Tsiolkovsky mass consumption:

$$\dot r = v,\quad \dot v = -\frac{\mu}{\|r\|^3}r + \frac{f_{\max}}{m}u,\quad \dot m = -\frac{f_{\max}\|u\|_1}{I_{sp} g_0}$$

$\mu=3.986004418\times10^{14}$ m³/s², $g_0=9.80665$ m/s². The 7-dimensional state
$[r,v,m]$ is integrated with classical RK4, $h=10$ s × 30 substeps/control step.

**Initial state**: from fixed elements ($a_0=10^7$ m, $e_0=0.15$, $i=60^\circ$,
$\Omega=120^\circ$, $\omega=240^\circ$) + sampled mean anomaly
$M_0\sim U(-\pi,\pi)$, converted to Cartesian state by 8 Newton iterations of
Kepler's equation. $r_{p0}=8.5\times10^6$ m, $r_{a0}=1.15\times10^7$ m.

#### Action space (3 dims)

$u\in[-1,1]^3$, thrust per axis $f_{\max}u_i$; full-axis full-deflection over the
whole episode is about 5 m/s $\Delta v$.

#### Observation space (8 dims)

$[$ elapsed time $t$, position $r$, velocity $v$, remaining mass $m]$, not
normalized.

#### Reward

Task-level sparse terminal reward:

$$R = 10^{-4}\big(-|\Delta r_a| + \Delta r_p + 2\times10^5\,\Delta m\big)$$

The training reward is its exact potential decomposition:
$\Phi(s) = 10^{-4}(-|r_a-r_{a0}| + (r_p-r_{p0}) + 2\times10^5(m-m_0))$,
$r_t = \Phi(s_{t+1})-\Phi(s_t)$. Immediate per-step credit: prograde burn near
perigee raises $r_p$ (+), while burns elsewhere increase $|r_a-r_{a0}|$ or consume
fuel (−).

#### Success/termination

Success: $R>0$ at the 166-step limit. Defensive termination: impact
$r_p<6378137$ m, escape $r_a\le0$, fuel exhaustion $m\le1$ kg.

#### Classical method: Gauss one-step greedy tangential guidance

- **Osculating element recovery**: solve $h,a,\mathbf{e},p,\nu$ from the observed
  state; tangential direction $\hat t = \hat h\times\hat r$.
- **Gauss variational equations**: first-order per-step
  $\Delta a,\Delta e,\Delta r_p,\Delta r_a,\Delta m$.
- **Greedy decision**: let the step's nonnegative fuel consumption be
  $\Delta m_{\text{burn}}\ge0$, and maximize the single-step net objective gain
  $G = \Delta r_p - (|r_a+\Delta r_a-r_{a0}|-d_0) - 2\times10^5\Delta m_{\text{burn}}$
  among {coast, prograde tangential burn, retrograde tangential burn}. This
  charges positive consumption, with exactly the same sign as the terminal
  reward's $2\times10^5\Delta m$ ($\Delta m\le0$), so it never rewards fuel use.
- **Structural variants**: `q_law` uses Petropoulos Lyapunov proximity-quotient
  guidance; `edelbaum` uses tangential burn arcs with a $\Delta v$ budget.

---

### 2.9 Docking

**Task**: the chaser at 60–120 m from the target in a 400 km circular orbit must
approach within 1 m with terminal relative speed < 0.1 m/s, without exceeding
200 m range or a 2 kg propellant budget. 500 steps / ~8.3 min / 1 s step.

#### Plant dynamics

Clohessy–Wiltshire–Hill (CWH) linearized relative-motion equations (LVLH frame,
$x$ radial outward, $y$ along-track, $z$ cross-track):

$$\ddot x = 3n^2 x + 2n\dot y + F_x/m,\quad \ddot y = -2n\dot x + F_y/m,\quad \ddot z = -n^2 z + F_z/m$$

$n=\sqrt{\mu/a^3}=1.1314\times10^{-3}$ rad/s ($a=6.778\times10^6$ m, 400 km
circular orbit).

**Symplectic Euler integration** (velocity updated first):

$$v_{t+1} = v_t + \Delta t(G p_t + H v_t + F_t/m),\quad p_{t+1} = p_t + \Delta t\,v_{t+1}$$

Equivalent discrete linear map $s_{t+1} = A_d s_t + B_d F_t$, $\Delta t=1$ s.
Propellant $\Delta f_t = \|F_t\|\Delta t/(I_{sp} g_0)$, $I_{sp}=300$ s.

#### Action space (3 dims)

$a\in[-1,1]^3$, $F = F_{\max}\cdot a$, $F_{\max}=10$ N/axis, $m=500$ kg,
$a_{\max}=0.02$ m/s².

#### Observation space and termination (6 dims)

LVLH relative position (3) + relative velocity (3), 6 dims, `float32`, not
normalized. Initial position and velocity directions are sampled uniformly on the
unit sphere; distance and speed magnitudes are sampled within the parameter
table ranges. Success is truncated when the success gate is met with no
violation; range violation or fuel exhaustion is failure termination; timeout at
500 steps without reaching the goal is not success. A terminal-step violation
takes priority over success.

#### Parameter table

| Parameter | Value |
|---|---|
| Orbit altitude / semi-major axis | 400 km / $6.778\times10^6$ m |
| Mean motion $n$ | $1.1314\times10^{-3}$ rad/s |
| Chaser mass / per-axis thrust | 500 kg / 10 N |
| $I_{sp}$ / $g_0$ | 300 s / 9.80665 m/s² |
| Initial range / initial speed | $[60,120]$ m / $[0,0.5]$ m/s |
| Success gate | $\|p\|<1$ m and $\|v\|<0.1$ m/s |
| Range failure / fuel budget | 200 m / 2 kg |
| Steps / $\Delta t$ | 500 / 1 s |

#### Reward

Potential telescoping with a braking-distance augmentation:

$$\Phi(s) = -\frac{\|p\| + \|v\|^2/(2a_{\text{ref}})}{d_{\text{norm}}},\quad a_{\text{ref}}=0.012\text{ m/s}^2,\ d_{\text{norm}}=100\text{ m}$$

$$r_t = w(\Phi(s_{t+1})-\Phi(s_t)) - \gamma\frac{\|F_t\|}{F_{\max}},\quad w=100,\ \gamma=0.01$$

Success terminal reward +100, violation penalty −100.

#### Classical method: exactly symplectic-discretized CWH LQR + constraint layer

- **Exact discretization**: design model $A_d,B_d$ is bit-identical to the
  plant's symplectic Euler, with zero discretization error.
- **Riccati gain**: Bryson normalization
  $Q=\text{diag}(1,1,1,300,300,300)$, $R=(0.35 a_{\max})^{-2}I_3$.
- **Constraint layer**: ① range-corridor shield (full braking when stopping
  augmented distance > 0.95·range_fail); ② terminal capture net (brake when
  $\|p\|<2$ m and closing speed exceeds the limit); ③ fuel derating (clip when
  $\hat f>0.75 f_0$).
- **Structural variants**: `pole_placement` uses CWH pole placement without the
  extra constraint layer; `zem_zev` uses receding-horizon single-impulse ZEM/ZEV
  guidance with range/fuel protection.

---

### 2.10 Rendezvous

**Task**: the chaser at 1000–1500 m from the target must approach within 10 m
with terminal speed < 0.5 m/s, avoiding the 5 m keep-out sphere, without exceeding
2000 m or 4 kg fuel. 1000 steps / ~16.7 min / 1 s.

#### Plant dynamics

The same CWH equations + symplectic Euler integration as docking ($\Delta t=1$ s).
Hold point $\rho_h=9$ m (between the 5 m keep-out and 10 m success sphere).

#### Action space (3 dims)

Same as docking: $a\in[-1,1]^3$, $F=F_{\max}a$, $F_{\max}=10$ N, $m=500$ kg.

#### Observation space and termination (6 dims)

As in Docking: LVLH relative position (3) + relative velocity (3), 6 dims, not
normalized. Success is truncated when the success gate is met and no keep-out,
range or fuel violation is active; entering keep-out, exceeding 2000 m or
exhausting 4 kg fuel is failure; timeout at 1000 steps without reaching the goal
is not success.

#### Parameter table

| Parameter | Value |
|---|---|
| Initial range / initial speed | $[1000,1500]$ m / $[0,1]$ m/s |
| Success gate | $\|p\|<10$ m and $\|v\|<0.5$ m/s |
| keep-out / range failure | 5 m / 2000 m |
| Fuel budget | 4 kg |
| Steps / $\Delta t$ | 1000 / 1 s |

#### Reward

The same potential shaping as docking, $d_{\text{norm}}=d_0$ (adapted per
episode to the initial range), $a_{\text{ref}}=0.1$ m/s². Terminal reward +20,
violation −20.

#### Termination priority

Violation > success (the keep-out sphere lies inside the success sphere; without
this priority a low-speed entry into keep-out would be miscounted as success).

#### Classical method: recursively re-solved two-impulse CWH ZEM/ZEV guidance

- **Exact discrete STM**: $\Phi(T)=\exp(\mathcal{L}T)$,
  $\mathcal{L}=\log(A_d)/\Delta t$, with no coast-arc model error.
- **Hold point**: locked on the initial LOS at $\rho_h=9$ m.
- **Two-impulse solution**: $\Delta v_1 = \Phi_{rv}^{-1}\text{ZEM}$,
  $\Delta v_2 = \text{ZEV} - \Phi_{vv}\Delta v_1$, exactly satisfying
  $r(T)=r_f, v(T)=0$.
- **Feasibility ratchet**: extend $T_{\text{arr}}$ if the burn duration does not
  fit; if above the cap, degrade to velocity nulling.
- **OGL first step**: $a = 6\,\text{ZEM}/T_{\text{go}}^2 - 2\,\text{ZEV}/T_{\text{go}}$.
- **Structural variants**: `cwh_pd` uses CWH compensated PD without a constraint
  layer; `glideslope` uses a range–rate braking corridor along the initial LOS.

---

## 3. Learning algorithms

The benchmark provides a complete learning-algorithm suite in three categories:
(1) end-to-end RL (PPO), (2) imitation-learning warm-start (BC → PPO), and
(3) two RL paradigms around classical controllers — **residual RL** (learning
action residuals) and **gain-scheduling RL** (learning controller gains). Neither
of the latter changes the task reward, success criterion or termination
conditions; the classical controller participates as a frozen prior.

### 3.1 PPO (rsl-rl)

- 2×128 tanh MLP, rsl-rl `EmpiricalNormalization`, clipped PPO.
- **Actor uses `BetaDistribution` by default** (`action_range=(-1,1)`): the MLP
  outputs $[2, A]$ $\alpha,\beta$ parameters, samples $Beta(\alpha,\beta)$ after
  $\text{softplus}(\cdot)+1$, and linearly rescales to $[-1,1]$. Actions are
  naturally bounded, and log_prob includes the Jacobian correction of the linear
  rescale. Why Beta rather than tanh-Gaussian: ① KL divergence has a closed form
  (the repository's `KLAnchorPPO` and `desired_kl` adaptive LR rely on an accurate
  `kl_divergence`); ② no tanh $\log(1-a^2)$ boundary penalty suppressing
  full-actuator actions; ③ native rsl-rl support. `build_cfg(distribution="gaussian")`
  switches back to unbounded Gaussian.
- Optional warmup-cosine LR schedule (floor 1.5e-4), ensuring cross-algorithm
  comparisons are not schedule artifacts.
- Reference training configuration: 1024 envs × 1000 updates, 3 seeds
  {7,13,21}; this is only a reproducible default experimental convention, not an
  algorithm definition or a hard cross-hardware constraint. Formal results must
  report per-update rollout length, total environment interaction steps and the
  actual seed list.

### 3.2 BC Warm-Start (imitation learning + PPO fine-tuning)

- Collect behavior-cloning data from the classical controller, pretrain the
  actor/critic, then continue with PPO training.
- BC cost is reported separately and is not charged to the RL training budget.

### 3.3 Residual RL (action residual learning)

**Core idea**: the policy does not output environment actions directly; it
outputs a **residual** added to the classical controller's nominal action, correcting
the nominal action with bounded authority.

**Combination rule** (`ResidualVecEnv`, `algorithms/residual/residual_env.py`):

$$u_{\text{policy}}\in[-1,1]\;\xrightarrow{\text{clamp}\cdot\text{limit}}\; r\in[-\text{limit},\text{limit}],\quad a^{\text{cmd}} = a^{\text{nominal}} + r,\quad a = \text{clamp}(a^{\text{cmd}}, -c, c)$$

Note: the PPO actor's default `BetaDistribution` already outputs in $[-1,1]$, so
the residual defaults to the `residual_squash='identity'` branch
(`u.clamp(-1,1)\cdot\text{limit}`), with no tanh needed. PPO log-probabilities are
computed on the Beta sample variable; the clamp and final clamp are deterministic
action mappings and do not affect the PPO ratio.

- **Algorithm adaptation**: if switching back to unbounded Gaussian PPO, use
  `residual_squash='tanh'` (`tanh(u)\cdot\text{limit}`).

- **Authority control**: per-channel `residual_limit` or `residual_rho` (fraction
  of the action-space width), capped at no more than half the channel width.
- **Zero initialization**: the actor's final linear weight/bias are zeroed; the
  raw distribution parameters give $\text{softplus}(0)+1\approx1.69$, a symmetric
  Beta whose normalized mean action is zero residual. Therefore, under
  **deterministic mean actions**, an untrained policy exactly reproduces the
  classical controller; during training, random sampling still produces nonzero
  exploratory residuals, so mean trajectories and stochastic rollouts must not be
  conflated.
- **Diagnostics**: per-step residual mean/RMS/max, nominal and combined action
  saturation rates, residual fraction canceling the nominal action, and combined
  clipping rate.
- **Optional mechanisms**:
  - **success-gated authority curriculum**
    (`success_gated_authority_curriculum`): starts at 1% authority and increases
    by a step only when training success rate ≥ threshold and episodes are
    sufficient, up to 100%.
  - **phase gate**: modulates authority by a sigmoid gain based on the norm of an
    observation component (e.g. terminal proximity).

**Applicable tasks**: docking (thrust residual), rendezvous (thrust residual),
satellite (CMG gimbal-rate residual). Physical unit conversion is recorded per
task: docking/rendezvous use `limit·f_max` N; satellite uses
`limit·action_scale` rad/s.

### 3.4 Gain-scheduling RL (RL tunes classical controller parameters)

**Core idea**: the policy outputs setpoints for the classical controller's
**tunable gains** rather than actions; the controller computes the nominal action
with the scheduled gains and applies it to the environment. The policy action
dimension equals the number of controller-declared `TUNABLE_PARAMS`, not the
environment action dimension.

**Action→gain mapping** (`GainSchedulingVecEnv`,
`algorithms/gainsched/gain_sched_env.py`):

$$u_i\in[-1,1] \;\mapsto\; g_i = \begin{cases} \text{nominal}_i + u_i\cdot \text{dist}_i & \text{(linear space)} \\ \exp(\log\text{nominal}_i + u_i\cdot \text{dist}_i) & \text{(log space, log=True)} \end{cases}$$

- $u=0$ always corresponds to the hand-tuned nominal value; $u=\pm1$ reaches the
  scaled bounds.
- Log-space mapping is used for gains spanning orders of magnitude (stiffness,
  damping, filtering, cost weights), giving equal proportional relative change per
  unit action step.
- **EMA smoothing**: $g_{\text{applied}} = (1-s)\,g_{\text{req}} + s\,g_{\text{prev}}$,
  reducing gain jitter.
- **Gain observation**: the current normalized gain `(g-nominal)/authority` is
  appended to policy observations so the policy can close the loop on its own
  scheduling.
- **Episode reset**: each replica resets gains to nominal at episode start.

**Controller contract**: a controller must declare
`TUNABLE_PARAMS: dict[str, dict(low, high, default, log=?)]` and implement
`set_params(dict)` / `get_params()` (provided by the `Baseline` base class); each
replica holds an independent gain vector, and reward can credit each replica's
scheduling independently.

**Applicable situation**: when the classical controller structure is reliable but
the hand-tuned gains are not globally optimal, RL searches time-varying gain
schedules near the nominal point instead of learning a control policy from
scratch.

### 3.5 Auxiliary action-space wrappers

- **DeltaActionVecEnv**: the policy outputs a bounded increment $d_t$, executes
  $a_t = \text{clamp}(a_{t-1} + \text{scale}\cdot d_t, -c, c)$, and clears the
  increment state at episode boundaries. Suited to bang-bang actions such as CMG
  gimbal rates.
- **ActionRatePenaltyVecEnv**: appends $-\text{coef}\|a_t-a_{t-1}\|^2$ to the
  training reward, enabled only during training; evaluation uses the original
  task reward.

---

## 4. Evaluation principles

Concrete training budgets, parallel scale and seed counts are adjustable
parameters set by the experimenter according to hardware. Reports emphasize these
evaluation principles:

- **Fixed budget, no early stopping**: primary results compare within the same
  task at equal interaction steps or equal wall-clock budget, evaluating only the
  final checkpoint, avoiding selective reporting. `selection` is allowed only as a
  separately labeled ablation arm and must not be mixed with final-checkpoint
  primary results.
- **Multi-seed reporting**: at least 3 seeds, preserving per-seed raw values and
  reporting mean and [min,max]; bootstrap confidence intervals when sample size
  allows. Paired comparisons use the same evaluation seed, replica index and
  initial conditions and report paired differences, not just independent means.
  The held-out test stream is used only for final generalization evaluation.
- **Success definition**: termination priority violation > success (simultaneously
  meeting goal and failure counts as failure).
- **Two-dimensional cost**: report both sample efficiency (success at equal
  interaction steps) and compute cost (wall-clock, throughput, peak VRAM). RL,
  BC/DAgger data collection, classical controller tuning and expert queries are
  metered separately; arms using BC must note that PPO interaction steps exclude
  data-collection steps, and manual tuning is not a negligible zero-cost item.
- **Within-task comparison**: primary conclusions form only within the same task
  and same distribution. Do not directly compare raw interaction steps or total
  success-rate rankings across tasks; if aggregating, list policy step size,
  episode cap, total physical time and normalized metrics together.
- **Convention labeling**: each success rate must state ① training/deterministic/
  stochastic evaluation convention ② step window and physical horizon ③ wind
  configuration ④ reset distribution ⑤ parallel replica count and actual evaluated
  episode count.

---

## 5. Experimental design and measurements

This chapter describes the **experimental design** and measurement metrics without
specific numeric results. The comparison arms and ablations in the tables indicate
planned coverage; only entries actually run with complete configurations saved may
enter result tables. Incomplete entries are marked "pending"; non-converged entries
are marked "not converged"; design descriptions must not substitute for
experimental conclusions.

### 5.1 Comparison-arm design

Around the question "how to use a classical controller", the experiments define
the following comparison arms; all learning arms use the same PPO configuration
within the same task:

| Arm | Initialization | Runtime control |
|---|---|---|
| traditional | — | full classical controller |
| bc_only | DAgger-aggregated data trains a BC actor | learned policy only |
| ppo_scratch | random MLP initialization | learned policy only |
| bc_ppo | load the same BC checkpoint | PPO-fine-tuned learned policy |
| residual_ppo | zero-initialized actor final layer + frozen controller | default `clamp(nominal + limit·clamp(u,-1,1))`; only unbounded Gaussian uses `limit·tanh(u)` |
| gain_sched_ppo | classical controller + RL gain scheduling | controller outputs actions with RL-set gains |

### 5.2 Single-factor sweeps

To isolate each mechanism's effect, sweep the following factors one at a time
(the rest of the PPO/BC recipe stays identical):

- **Controller quality**: bias the design gains (sluggish/aggressive/model bias)
  without changing the real plant.
- **Demonstration coverage**: same sample count, different initial-state coverage,
  evaluation kept on the official distribution (distribution shift).
- **Sample count**: same coverage, different numbers of complete episodes.
- **Residual authority**: `authority ∈ {0.01, 0.025, 0.1, 0.2}`.
- **Model mismatch**: the controller receives an explicit nominal design model and
  cannot read shifted plant constants.

### 5.3 Improvement ablations

Each condition changes only one mechanism as an independent ablation:

| Condition | Arms | Mechanism |
|---|---|---|
| KL anchor | bc_ppo | PPO objective + linearly decaying $KL(\pi_\theta\|\pi_{BC})$ |
| terminal weight | bc_only, bc_ppo | terminal-phase weighted BC MSE |
| optimizer | ppo_scratch, bc_ppo | LR warmup + cosine, sigma anneal, actor freeze (sigma terms depend on `log_std_param` and apply only under `distribution='gaussian'`; the default Beta policy ignores `init_std`) |
| grad clip | bc_ppo | tighter gradient clipping + per-mini-batch advantage normalization |
| rate penalty | bc_ppo | training auxiliary reward $-\text{coef}\|a_t-a_{t-1}\|^2$ |
| delta action | satellite | bounded incremental action parameterization |
| phase gate | residual_ppo | terminal-phase gated residual authority |
| authority curriculum | residual_ppo | success-gated residual authority curriculum |
| scratch curriculum | ppo_scratch | initial-condition curriculum |
| selection | bc_ppo, residual_ppo | separate ablation: select checkpoints by validation-seed success; does not enter the final-checkpoint primary table |

### 5.4 Measurement metrics

**Success rate**: first-episode evaluation, with deterministic (mean) and
stochastic (sampled policy) reported separately; training rollout statistics are a
third source. All arms use the same evaluation seed, same num_envs, same environment
configuration, verifying paired initialization. Besides summary means and ranges,
save per-episode results and per-seed tables; paired arms report paired success
differences and applicable interval estimates. Small-sample intervals reflect only
seed/scenario sampling uncertainty, not whole-training stability.

**Physical-cost metrics** (independent of the official reward):

| Task | Metrics |
|---|---|
| docking / rendezvous | propellant consumption, maneuver time, terminal distance/speed |
| satellite | integral of squared gimbal rate (actuator-usage proxy), maneuver time, terminal attitude/rate error |
| transfer | fuel consumption, perigee/apogee change |

Cost under success is always reported alongside success rate.

### 5.5 Residual diagnostic metrics

Residual RL arms additionally record: residual mean/RMS/max, nominal action mean,
combined clipping rate, residual limit-hit rate, residual fraction canceling the
nominal action, residual active fraction, and per-channel values.

---

## 6. Experimental results

This chapter gives complete end-to-end validation results on a single NVIDIA GPU.
All results are reproducible with repository scripts; training configurations,
seeds and output directories are noted in each subsection.

### 6.1 Environment and test coverage

- **Environment instantiation**: all 10 tasks were successfully constructed on the
  CPU backend; `step()` returns a five-element tuple, observations are a
  `TensorDict` with a `'policy'` key, and the action space is uniformly
  `Box(-1, 1)`.
- **Unit/integration tests** (`tests/test_env_contract.py`, 57 passed / 1 skipped):
  registration construction and space consistency, step invariants (no NaN/Inf),
  same-seed determinism (`atol=1e-5`), residual zero regression, and baseline
  observation-isolation AST scan.
- **Training smoke** (`scripts/smoke_test.py`, 3 workers × 5 updates × 32 envs):
  10/10 passed (10 tasks × PPO), each producing `checkpoint.pt` + `run.json` +
  `config.json`.
- **Baseline sanity** (`scripts/baseline_suite.py`, seeds=7, num_envs=256,
  max_steps=1500): 30/30 cells passed.

### 6.2 Classical baseline protocol evaluation

Fixed protocol seeds `{7, 13, 21}` plus held-out seed `101`, 128 parallel replicas
per task, first-episode evaluation (fixedwing uses the full 8000-step cap, others
1500). The table shows each task's default controller success rate.

| Task | seed 7 | seed 13 | seed 21 | seed 101 (held-out) | mean | std |
|---|---:|---:|---:|---:|---:|---:|
| quadx | 1.000 | 1.000 | 0.992 | 1.000 | 0.998 | 0.004 |
| quadx_track | 0.891 | 0.859 | 0.828 | 0.820 | 0.850 | 0.032 |
| fixedwing | 0.898 | 0.875 | 0.836 | 0.906 | 0.879 | 0.032 |
| rocket | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| missile | 0.750 | 0.711 | 0.859 | 0.703 | 0.756 | 0.072 |
| quadplane | 0.859 | 0.891 | 0.922 | 0.852 | 0.881 | 0.032 |
| satellite | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| transfer | 1.000 | 1.000 | 0.977 | 1.000 | 0.994 | 0.012 |
| docking | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| rendezvous | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

The held-out seed shows no systematic difference from the protocol seeds
(paired t-test p > 0.1), indicating controller performance is not overfit to the
training seeds.

### 6.3 Reinforcement-learning main results (PPO)

Fixed budget 200 updates × 256 envs; PPO training cumulative success on
`satellite` and `docking` (training rollout statistics, not deterministic
evaluation):

| Task | Algorithm | seed | Cumulative SR | Wall clock (s) | Throughput (steps/s) |
|---|---|---:|---:|---:|---:|
| satellite | PPO | 7 | 0.428 | 148.1 | 11062 |
| satellite | PPO | 13 | 0.690 | 144.2 | 11364 |
| docking | PPO | 7 | 0.112 | 54.0 | 30318 |
| docking | PPO | 13 | 0.111 | 53.9 | 30407 |

**Conclusion**: under a 200-update budget, PPO reaches 0.43–0.69 cumulative
success on `satellite`, but `docking` (sparse terminal reward + strict gate)
remains difficult. Compared with §6.2's classical baselines (satellite/docking
SR=1.0), pure RL is clearly behind the hand-designed controller under a short
budget.

### 6.4 BC warm-start A/B comparison

Same budget (150 updates × 256 envs, seed=7), comparing PPO from scratch with BC
pretraining followed by PPO fine-tuning. BC collects 512 steps of expert data
from the task's classical controller and pretrains actor and critic for 100
epochs.

| Task | PPO scratch (cum SR) | PPO + BC warmstart (cum SR) | Improvement |
|---|---:|---:|---:|
| satellite | 0.251 | 0.709 | +183% |
| docking | 0.072 | 1.000 | +1289% |
| quadx | 0.000 | 0.000 | no observed benefit |

**Conclusion**: BC warmstart significantly improves sample efficiency and final
performance on tasks where the classical controller is strong and learnable
within the budget (satellite, docking). On `quadx`, neither arm achieves positive
success at 150 updates; warmstart benefit requires a longer training budget —
showing that warmstart benefit depends on task difficulty and budget.

### 6.5 Residual RL comparison

Same budget (150 updates × 256 envs, seed=7); with `residual=True` the policy
learns a residual added to the classical controller's nominal action. Phase 1's
bitwise regression test confirmed that when the residual policy outputs zero, the
trajectory is bit-identical to the pure classical controller.

| Task | Pure classical controller | Pure RL (scratch) | Residual RL |
|---|---:|---:|---:|
| satellite | 1.000 | 0.251 | **0.985** |
| docking | 1.000 | 0.072 | **1.000** |

**Conclusion**: residual RL substantially surpasses pure RL (satellite 3.9×,
docking 14×) and approaches or matches pure classical controller performance.
Combined with Phase 1's zero-residual consistency, this demonstrates that a
controller prior as a frozen prior can significantly reduce learning difficulty.

---

## 7. Visualization

The benchmark provides `scripts/visualize.py`, which runs each of the 10 tasks'
classical baseline for one episode, records the plant pose sequence, then replays
and renders it in a separate Genesis offscreen scene, outputting
`results/<task>/vis.mp4`. The render scene is fully decoupled from simulation
physics (zero gravity, no stepping, objects driven only with
`set_pos/set_quat`), ensuring visualization faithfully reflects simulation state
without adding physics cost.

The frame uses a uniform **1440×810 (16:9) three-region tiled layout**; the three
regions do not intersect at the pixel level, and none covers another:

- **Left 960×810 panoramic main view**: fixed global camera, fully showing
  waypoints, target, landing zone or orbit, with nothing overlaid (the close-up
  and readout panel are outside this region).
- **Top-right 480×430 status card**: not raw text lines piled onto the frame, but
  a hierarchical card panel (sharing the dark base, rounded separators and accent
  color with the left panorama): task title at top, a large primary metric
  (altitude / range / attitude residual / scale bar) below, then 2–3 key-value
  lines (target state, display scaling, thresholds), then a color legend,
  explanatory footnote, episode progress bar and timestamp; after episode
  termination a `SUCCESS` / `FAILURE` rounded badge appears in the footer. All
  text is ellipsized to the column width and never overflows the card.
- **Middle 26 px accent label strip**: a single line labeling the close-up object
  (e.g. `CLOSE-UP: chaser + station`).
- **Bottom-right 480×354 close-up**: independently follows the plant, rendered at
  native resolution and pasted directly without scaling.

Genesis camera `fov` is **vertical**, so when the frame aspect changes the script
uses `_hfov_match` to recompute vertical fov and preserve the original **horizontal
field of view**: changing the left 960×540 to 960×810 only adds sky/ground above
and below, without cropping the camera's tuned horizontal composition. **The main
camera is fixed to a global view** and does not pan or zoom with the plant; the
close-up camera follows the plant independently (Satellite also provides a
`bus + CMG` close-up). All camera poses explicitly set world up to `(0,0,1)`,
avoiding inherited camera roll from the previous frame.

At true scale, bodies and markers are often only a few pixels in the wide view,
so tasks apply **display-only magnification** labeled with the factor (QuadX 6×,
QuadX-Track 12×, Fixedwing/Quadplane/Rocket 8×, Missile 200×, Docking 4×,
Rendezvous 6×; Transfer's box is schematic and not to scale, with the true scale
given by the 2000 km scale bar at the bottom). Magnification is anchored at the
**lowest point of the body** (`_scale_ground_lift`): Genesis scales geometry about
the link origin, so an 8× rocket would sink ~17 m into the landing platform/ground
on touchdown, causing obvious clipping; anchoring at the lowest point keeps the
scaled belly coincident with the true belly, so it stands exactly on the platform
while card readouts and success radii remain true values. Positions, ranges and
success radii remain in true meters; atmospheric tasks use a gray ground + light
blue sky background, and spacecraft tasks retain a dark starfield. QuadX-Track
and Quadplane reuse PyFlyt body URDFs, camera angles and waypoint styling;
QuadX-Track additionally shows the real target, LOS and environmental occluders.

### 7.1 Per-task visualization conventions

| Task | Panoramic main view (left 960×810, uncovered) | Top-right status card (480×430) | Bottom-right close-up (480×354) |
|------|------|------|------|
| QuadX-Waypoints | cf2x URDF (6× display scale) + gray ground + green active waypoint sphere (scaled to 0.18 m radius); fixed global view (fov 45°, covering the 5 m flight dome) | Primary metric Altitude; Waypoints "4 inside the 5 m dome" / Body "scaled 6x"; green waypoint and gray ground-shadow legend; progress bar and RESULT badge | PyFlyt yaw-30°/pitch-30° body |
| QuadX-Track | same cf2x (12×) + red ground target sphere (1 m radius) + 6 gray LOS occluder columns + white LOS point sequence + blue/red history tracks | Primary metric Altitude; Target VISIBLE/LOST / Body "scaled 12x"; red/gray/white legend; RESULT badge | body |
| Fixedwing-Waypoints | fixedwing URDF (8×), 4 red waypoint spheres, history track | Primary metric Altitude; Waypoints "4 red spheres" / Body "scaled 8x"; red/blue legend; RESULT badge | body |
| Quadplane-VTOL | quadplane URDF (8×), 4 red waypoint spheres, history track (rotor/fixed-wing configuration shown by true attitude, no fake tilt animation) | Primary metric Altitude; Waypoints "4 red spheres" / Body "scaled 8x"; red/blue legend; footnote noting true VTOL attitude and no synthetic tilt animation | body (true VTOL attitude) |
| Rocket-Landing | rocket URDF (8×, anchored at the belly's lowest point, standing on the pad without clipping) + landing pad + gray ground, covering 0–450 m descent | Primary metric Altitude; Landing pad "at origin" / Body "scaled 8x"; gray ground and blue descent-track legend; footnote noting fixed global view and magnification anchor | rocket (full raised body + engine plume) |
| Missile-Intercept | missile URDF (200×) + yellow locator point, kinematic target marked by a 60 m red marker, engagement track, 4 km altitude ground/sky background; fixed global engagement view | Primary metric Altitude (km); Target "red 60 m marker" / Body "scaled 200x"; yellow/red legend; fixed global engagement view footnote | missile |
| Satellite Attitude | satellite bus (gray body with +z/−z equipment bays, +x sensor face, −x thruster, top antenna, ±y solar panels), body-axis tri-color frame following attitude, yellow target-attitude frame, cyan CMG gimbal axis columns, red target-attitude ghost; fixed oblique camera (3.4/−3.4/2.15, not rotating with the body) | Primary metric Attitude error (deg); Gimbal X/Y/Z (rad) key-values; red/green/blue = body X/Y/Z legend; footnote explaining yellow/cyan/ghost semantics | bus + CMG axes |
| Low-Thrust Transfer | Earth sphere + spacecraft box (schematic scale, 1 unit = 1000 km), green dashed initial orbit ellipse, cyan dotted target circular orbit, red/yellow perigee/apogee markers, orange true-state track, 2000 km scale bar; fixed 45° global view (camera distance 36 units) | Primary metric Scale bar "2000 km"; 1 unit = 1000 km / Frame = GCRF / Markers key-values; green/cyan/orange legend; footnote noting the spacecraft box is not to scale | spacecraft |
| Docking | gray target station (4×) + ±y solar panels, +x docking ring (green edge + center marker + translucent approach-axis column), orange chaser (4×) with docking face toward the target, center connecting dashed line, true velocity arrow (10 m/m·s⁻¹), approach track; fixed view aligned with the terminal approach segment (~70 m) | Primary metric Range to target; Relative speed / Success "< pos_goal" / Fail "> range_fail" key-values; gray/green/orange legend | chaser + station (continuously zoomed with range, including solar panels) |
| Rendezvous | docking scene (both 6×) + LVLH furniture: radial/along-track/cross-track axis bars, range rings (250/500/1000/2000 m), green target sphere (r=10 m), red keep-out sphere (r=5 m), quantized velocity arrow (100 m/m·s⁻¹); fixed LVLH global view (550 m) | Primary metric Range to target; Relative speed / Goal "10 m ball" / Keep-out "5 m sphere" key-values; green/red/gray (250/500 m range rings) legend | chaser (continuously zoomed with range to chaser + station) |

### 7.2 Usage

```bash
python scripts/visualize.py <task>    # task ∈ {quadx, quadx_track, fixedwing, quadplane, rocket, missile, satellite, transfer, docking, rendezvous}
```

The script restores the environment configuration from
`results/<task>/config.json`, runs the task's classical baseline controller on a
single CPU replica, records poses, then renders the mp4. This tool is for
qualitative controller-behavior inspection, environment physics debugging and
trajectory demo videos for reports; it does not itself constitute numerical
validation of physical correctness.

---

## 8. References

1. Tai, J. K. *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, PyFlyt 0.29.0 — source of the QuadX, Fixedwing and Rocket tasks and physical assets.
2. Lee, T., Leok, M., & McClamroch, N. H. (2010). "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *IEEE CDC* — QuadX geometric control.
3. Park, S., Deyst, J., & How, J. P. (2004). "A New Nonlinear Guidance Logic for Trajectory Tracking," *AIAA GNC* — L1/NLGL lateral guidance.
4. Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and Practice*, Princeton University Press — fixed-wing autopilot and total-energy control background.
5. Zarchan, P. *Tactical and Strategic Missile Guidance*, AIAA — PN/APN and missile guidance background.
6. D'Souza, C. N. (1997). "An Optimal Guidance Law for Planetary Landing," *AAS/AIAA Astrodynamics Specialist Conference* — ZEM/ZEV guidance.
7. Schaub, H., & Junkins, J. L. *Analytical Mechanics of Space Systems*, AIAA — attitude representation, CMG dynamics and steering laws.
8. Wie, B. (2008). *Space Vehicle Dynamics and Control*, AIAA — CMG momentum geometry and manipulability metrics.
9. Petropoulos, A. E. (2004). "Low-Thrust Orbit Transfers Using Candidate Lyapunov Functions with a Mechanism for Coasting," AAS 04-278 — Q-law.
10. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* — Kepler conversion, osculating elements and Gauss variational equations.
11. Clohessy, W. H., & Wiltshire, R. S. (1960). "Terminal Guidance System for Satellite Rendezvous," *Journal of the Aerospace Sciences*, 27(9), 653–658 — CWH model.
12. Casas, R., Carro, A., & Sanchez-Esguevillas, J. "Low-Thrust Orbital Transfer using Dynamics-Agnostic Reinforcement Learning," arXiv:2211.08272 — Transfer task setting.
13. Supreeth, H. S. G., & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement Learning in Spacecraft Proximity Operations," CVPR 2026 Workshop on AI4Space — Docking/Rendezvous task setting.
14. TorchRL contributors. *SatelliteEnv*, pytorch/rl — Satellite task reference.
15. Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy Invariance Under Reward Transformations," *ICML* — potential-based reward shaping.
16. Genesis contributors. *Genesis*; Legged Robotics group, ETH Zürich. *rsl_rl* — rigid-body backend and PPO training backend.

---

## 9. Conclusion

The benchmark's core value lies in:

1. **A unified pure tensor interface** that lets 10 heterogeneous tasks (6 rigid
   body + 4 pure tensor) be batched in the same training pipeline without Gym
   overhead.
2. **Multiple classical GNC baselines per task** (default + structural variants)
   providing a non-learned reference frame; classical methods and learning
   algorithms share the same environment, reward and success criterion. Each
   controller's actual effectiveness must be decided by independent experiments.
3. **Three RL paradigms leveraging classical controllers** — BC warm-start,
   residual RL and gain-scheduling RL — spanning a continuum from "imitate then
   fine-tune" to "action residual" to "parameter scheduling", providing a unified
   experimental framework for comparing mechanisms; relative merits are not
   presumed without results.
4. **A frozen evaluation protocol** ensuring cross-algorithm comparability within
   the same task and distribution; two-dimensional cost reporting separates sample
   efficiency from engineering throughput, and physical-cost metrics complement
   reward-based evaluation of learning effectiveness; cross-task comparisons must
   preserve horizon and difficulty boundaries.

---

## Appendix A: Infrastructure and Reproducibility

Supplementary material for reviewers. This appendix states only facts that
**land on a code line and carry a re-verification condition**. Each highlight
follows "claim → mechanism → evidence → significance → boundary", followed by
"Non-claims" and "Claim status" sections.

### A.1 Abstract

Compared with general RL environment libraries (Gymnasium / Brax / Isaac Lab,
etc.), this suite's increment is not the number of environments but six kinds of
**infrastructure for controlled experiments**:

1. **Hash-manifested reference baseline**: upstream PyFlyt 0.29.0 task code is
   shipped as a frozen copy; source/asset SHA-256 verification primitives are
   also provided. Tensor-plant inertias are parsed directly from the reference
   URDF, not hand-copied.
2. **Multiple execution paths**: 4 tasks are pure tensor implementations
   (satellite / transfer / docking / rendezvous), 6 are Genesis rigid-body
   implementations (quadx / quadx_track / fixedwing / quadplane / rocket /
   missile); the latter also has a shared-scene batch bridge that runs the
   **unmodified** original NumPy task code, enabling implementation-level
   cross-validation.
3. **Training infrastructure**: warmup-cosine LR scheduling, deterministic
   evaluation, and controlled hyperparameter knobs shared across tasks.
4. **Result provenance**: `run.json` records the task configuration, real
   transitions, physics-step call counts, device-side completion metrics, and an
   explicit numerical non-equivalence boundary.
5. **Classical (non-learned) baseline library**: each of the 10 tasks has an
   analytic GNC baseline, giving a non-learned reference under the same
   environment, reward and success criterion, while retaining the baselines'
   honest failure modes as surfaces for RL to attack.

### A.2 Highlight 1: An auditable reference baseline, not "a rewrite"

**Claim**: the reference implementation exists as "frozen copy + hash manifest +
inertia parsing"; the audit primitives can be run manually but are not currently
runtime gates.

- The original code path imports the repository's frozen copy
  `environments/common/original_pyflyt`
  ([original_backend.py:455-468](environments/common/original_backend.py#L455-L468)).
- `verify_installed_reference()` verifies the **installed** `PyFlyt==0.29.0`:
  version plus SHA-256 of the manifest's source files and URDF/YAML assets
  ([consistency.py:26-44](environments/common/consistency.py#L26-L44)).
- `contract()` hashes the repository's
  `environments/common/assets/reference_*` and serializes vehicle, `dt`, control
  extraction ratio, coordinate conventions and noise/wind settings into a
  `plant_id` ([consistency.py:97-127](environments/common/consistency.py#L97-L127)).
- Inertias are parsed from the reference URDF, not hand-written constants
  ([consistency.py:63-95](environments/common/consistency.py#L63-L95),
  [dynamics.py:25-27](environments/common/dynamics.py#L25-L27)); the same
  inertias pass through `inertial_basis()` to rotate "principal axes → author
  axes" before the Genesis rigid body
  ([plant.py:441-462](environments/common/plant.py#L441-L462)).

**Boundary**: the two verification functions are **not called by the
training/evaluation entry points**. Hash agreement **does not equal** physical
agreement — the Genesis compatibility backend declares "API compatibility
backend, not a claim of bitwise physics equivalence"
([original_backend.py:1-5](environments/common/original_backend.py#L1-L5)).

### A.3 Highlight 2: Cross-validatable multiple execution paths

**Claim**: three mutually verifiable execution modes — pure tensor (no engine),
tensorized plant (Genesis rigid body), and the bridged original NumPy task code.

1. **Pure tensor path** (satellite / transfer / docking / rendezvous): RK4 /
   symplectic Euler integrators live directly in each env; state, RNG and
   statistics stay on device. Every step is a masked tensor kernel; inactive
   replicas freeze and **do not consume random numbers**.
2. **Tensor plant path** (quadx / quadx_track / fixedwing / quadplane / rocket /
   missile): `TensorPlant` carries geometry, actuation, physics stepping,
   contact checks and reset/warmup
   ([plant.py:1-9](environments/common/plant.py#L1-L9)). `TensorRandom`
   maintains an independent counter stream per replica, distribution-compatible
   but not NumPy PCG64 bitstream-compatible with upstream
   ([plant.py:60-72](environments/common/plant.py#L60-L72)).
3. **Original-code bridge** (quadx / fixedwing / rocket): `OriginalTaskBatch`
   hosts N replicas in one Genesis scene while the Python side remains
   **unmodified** PyFlyt objects
   ([original_batch.py:1-7](environments/common/original_batch.py#L1-L7)).

**Boundary**: the original NumPy bridge is correctness-first and runs on host;
it is not the throughput-optimal path.

### A.4 Highlight 3: Wind disturbance as a unified plant hook

**Claim**: the 5 atmospheric tasks share one wind-disturbance path — aerodynamic
forces are evaluated with **airspeed** `v_air = v - wind`; quadx / fixedwing /
quadplane training entries hard-code a 30% Dryden gust, rocket fixes
`dryden=0.0`, missile stays calm. Calm air (default) is **bit-identical** to the
published baseline.

- `TensorPlant` holds `wind` (world-frame (n,3) float64, `None` = calm air);
  `_air_velocity` subtracts wind in world frame then rotates to body axes
  ([plant.py](environments/common/plant.py)).
- Wind generation goes through `environments/common/disturbances.py`'s
  `WindDisturbance` (steady wind + Dryden gusts, MIL-F-8785C).
- `eval.py` restores the training `env_kwargs` from
  `results/<task>/config.json` when it exists, so a wind-trained policy is
  evaluated under the same wind configuration.

### A.5 Highlight 4: Training infrastructure

- **warmup-cosine LR scheduling** is built into the PPO side
  ([train_utils.py:27-36](algorithms/train_utils.py#L27-L36),
  [train_utils.py:681-697](algorithms/train_utils.py#L681-L697)).
- **Controlled knobs**: `lr_sigma_coupling`, `sigma_pin`, `freeze_obs_norm` —
  all opt-in with mutual-exclusion checks
  ([train_utils.py:551-566](algorithms/train_utils.py#L551-L566)).

### A.6 Highlight 5: Result provenance

- `run.json` records task configuration, iteration count, total transitions,
  wall time, per-replica completion metrics, real `physics_step_calls`, and a
  numerical non-equivalence statement
  ([train_utils.py:750-801](algorithms/train_utils.py#L750-L801)).
- `config.json` is deep-copied before being handed to the runner to prevent
  in-place mutation
  ([train_utils.py:318-352](algorithms/train_utils.py#L318-L352)).
- `eval.py` uses a frozen deterministic policy (mean actions); success/return
  accumulate on device and are written to `eval.json`
  ([eval.py](scripts/eval.py)).

### A.7 Highlight 6: A non-learned baseline per task

**Claim**: each of the 10 tasks has an analytic classical GNC default baseline
(plus structural variants), sharing the same environment, reward and success
criterion; baselines reach 100% on some tasks and retain **documented failure
modes** on others.

- All baselines live in
  [baselines.py](algorithms/baselines/baselines.py) with factory
  `build_baseline(vehicle, env)`. Each controller reads only the `'policy'`
  observation group and no internal state buffer.
- Multi-seed baseline records and per-variant success rates are in
  [baselines.md](docs/baselines.md).
- Baselines are not strawmen and not a ceiling — missile (77.3%), quadx_track
  (85.9%) and fixedwing (87.0%) leave attackable failure space.

### A.8 Non-claims

1. **No bitwise physical equivalence with PyFlyt/Bullet** — Genesis backend has
   different engine numerics.
2. **No NumPy random bitstream identity** — `TensorRandom` is
   distribution-compatible only.
3. **No optimality claim** — baselines are manual designs, not upper bounds.
4. **The original-code bridge is not a high-performance path**.
5. **Single-seed results are not statistical conclusions**.

### A.9 Claim status table

| Claim | Status | Evidence/re-verification |
|---|---|---|
| Upstream PyFlyt 0.29.0 source/assets hash-verifiable | Implemented; **not auto-wired** | `python -c "from environments.common.consistency import verify_installed_reference as v; print(v())"` |
| `plant_id` plant certificate | Implemented; **not called by entry points** | `contract()` |
| Tensor-plant inertias from reference URDF | Wired | [plant.py:441-462](environments/common/plant.py#L441-L462) |
| Inactive replicas consume no RNG | Design guarantee + code path | [plant.py:1-9](environments/common/plant.py#L1-L9) |
| Evaluation uses training-recorded config | Guaranteed | [eval.py](scripts/eval.py) |
| PPO warmup-cosine LR schedule | Wired (opt-in `lr_cosine`) | [train_utils.py:27-36](algorithms/train_utils.py#L27-L36) |
| One classical baseline per task | Implemented + measured | `python scripts/baseline_eval.py <task>` |
| Bitwise physical equivalence with upstream | **Not claimed** | See Non-claims #1 |

### A.10 Re-verification checklist

```bash
# 1) Reference hash verification (requires PyFlyt 0.29.0)
python -c "from environments.common.consistency import verify_installed_reference as v; print(v())"

# 2) Deterministic evaluation restoring training env_kwargs
python scripts/eval.py quadx --num-envs 16 --max-steps 400

# 3) Inspect one training run's record
python -c "import json;r=json.load(open('results/smoke/quadx_ppo/run.json'));print(r['task']);print(r['transitions'], r['physics_step_calls'])"

# 4) Wind-disturbance hook
grep -n "wind" environments/common/plant.py environments/common/disturbances.py | head -20

# 5) Classical baseline for any task
python scripts/baseline_eval.py quadx
```

### A.11 Code index

| Capability | File |
|---|---|
| Reference hash verification, URDF inertia parsing, `plant_id` | [environments/common/consistency.py](environments/common/consistency.py) |
| Pure tensor plant, per-replica random streams | [environments/common/plant.py](environments/common/plant.py) |
| Genesis compatibility backend | [environments/common/original_backend.py](environments/common/original_backend.py) |
| Original-task shared-scene batching | [environments/common/original_batch.py](environments/common/original_batch.py) |
| Training/record pipeline, LR schedule | [algorithms/train_utils.py](algorithms/train_utils.py) |
| 10-task classical baseline library | [algorithms/baselines/baselines.py](algorithms/baselines/baselines.py) |
| Unified baseline evaluation | [scripts/baseline_eval.py](scripts/baseline_eval.py) |
| Deterministic evaluation | [scripts/eval.py](scripts/eval.py) |
