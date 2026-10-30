# QuadX-Track Environment (`QuadXTrackEnv`)

**Status: implemented** — the environment, the `train_quadx_track.py` entry, the
KF+SE(3) baseline, and both `baseline_eval.py` / `eval.py` registrations exist
and have been run; this is not a planned or future task requiring a new civil
observation environment.

## 1. Overview

QuadX-Track is the persistent-observation task of the AerospaceRL quadrotor family. A Crazyflie 2.x
micro-quadrotor (27 g, X configuration) must keep a **cooperative ground moving target** inside a
nadir-stabilized camera's field of view while holding a fixed altitude and a fixed horizontal offset from
the target, for a 30 s episode at a 40 Hz policy rate. The target drives on the ground plane with a
pre-drawn schedule of straight, turning and stop-go segments inside a 15 m radius area.

The plant, inner rate PID, motor mixer, 4-D rate+collective action interface, solver guard and 40 Hz
policy rate are **exactly the QuadX-Waypoints stack** (`TensorPlant('quadx')`, see
`docs/method/env-quadx-waypoints.md`). The task differs only in what is sensed and rewarded. Sensing is
geometric rather than rendered: a downward cone, a slant-range window, line-of-sight occlusion by static
cylinders, and isotropic position noise. There is no interception, contact, weapon or collision objective.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $p,v\in\mathbb{R}^{3}$; $q=(w,x,y,z)$ | vehicle world position / velocity; attitude quaternion |
| $\omega_b,v_b\in\mathbb{R}^{3}$ | body-frame angular / linear velocity |
| $p_t\in\mathbb{R}^{3}$, $\theta_t$, $s_t$ | target position (world), heading, ground speed |
| $\alpha$ | camera half-angle, 30° |
| $d_h,d_3$ | horizontal and slant vehicle–target range |
| $y_t\in\{0,1\}$ | visibility flag |
| $\zeta_t\in\mathbb{R}^{3}$ | position measurement in world axes, relative to the vehicle |
| $\bar p=p_t^{\mathrm{ref}}$ | tracking reference point |
| $\Delta t_\pi,\Delta t_c,\Delta t$ | policy 1/40 s, control 1/120 s, physics 1/240 s |

## 3. Plant and control hierarchy

Identical to QuadX-Waypoints. The policy runs at 40 Hz. Each policy step executes `inner_steps = 3`
control periods at 120 Hz; each control period applies the plant's rate-PID update followed by two Genesis
physics substeps at 240 Hz, i.e. six physics substeps per policy step. Motor lag/noise, quadratic thrust,
componentwise drag and the 15 m/s / 30 rad/s solver guard are unchanged.

## 4. Cooperative ground target

The target is a kinematic ground vehicle at constant height $z_t = 0.15$ m. At reset, one uniform draw per
replica fixes the spawn radius $r\sim\mathcal U[0.5,2]$ m, spawn angle, initial heading, and a schedule of
`NUM_SEGMENTS = 12` future segments. Each segment carries

$$
\text{mode}\in\{\text{straight},\text{turn},\text{stop-go}\},\quad
T_k\sim\mathcal U[2,5]\ \mathrm{s},\quad
s_k\sim\mathcal U[0.4,1.2]\ \mathrm{m/s},\quad
\omega_k=\pm\mathcal U[0.3,1.0]\ \mathrm{rad/s},
$$

with $s_k=0$ for stop-go and $\omega_k=0$ for straight. The target advances once per policy step
($\Delta t_\pi = 25$ ms) with

$$
p_t \leftarrow p_t + s_k\begin{bmatrix}\cos\theta_t\\ \sin\theta_t\\0\end{bmatrix}\Delta t_\pi,
\qquad
\theta_t \leftarrow \theta_t + \omega_k\,\Delta t_\pi .
$$

A segment timer counts down; when it expires the next pre-drawn segment is loaded. Because the shortest
segment is 2 s, at most one boundary is crossed per policy step, so the update is a single gather. Mode
parameters are drawn at reset only, never inside `step`, which keeps inactive replicas out of the random
stream.

The target reflects off the circular operating area $r\le 15$ m: the position is projected back onto the
circle and the heading is reflected about the radial normal. This keeps a 30 s episode inside the
vehicle's 30 m out-of-bounds sphere without consuming randomness online.

## 5. Geometric camera model

The camera is mounted on a stabilized gimbal, so its optical axis is world $-z$ and its ground footprint
at altitude difference $\Delta z$ has radius $\Delta z\tan\alpha$. A detection requires **all** of:

1. **Field of view.** $z - z_t > 0$ and $d_h \le (z-z_t)\tan\alpha$.
2. **Slant-range window.** $1.5\ \mathrm{m} \le d_3 \le 12\ \mathrm{m}$.
3. **Line of sight.** No static cylindrical occluder intersects the segment from $p$ to $p_t$. For an
   occluder at $(c_x,c_y)$ with radius $r_o$ and height $h_o$, let
   $\tau=\mathrm{clip}\big(((c-p)\cdot(p_t-p))/\lVert p_t-p\rVert_{xy}^2,0,1\big)$; the LOS is blocked
   when $\lVert p_{xy}+\tau(p_t-p)_{xy}-c\rVert < r_o$ and $z+\tau(z_t-z) < h_o$ (the LOS passes through
   the cylinder below its top).

The occluders $(\cdot,\cdot,r_o,h_o)$ are:

| x | y | radius | height |
|---|---|---|---|
| 4.5 | 3.0 | 1.0 | 5.0 |
| −3.5 | 5.5 | 1.2 | 6.0 |
| 6.0 | −4.0 | 1.0 | 4.5 |
| −6.0 | −3.0 | 1.1 | 5.5 |
| 1.0 | −6.5 | 1.0 | 4.0 |
| −1.5 | 8.5 | 1.0 | 6.5 |

They are **pure line-of-sight geometry**: they are not physical entities in the plant, carry no collision
or force, and cannot be used to intercept or contact the target. The task has no collision objective; the
reported collision metric is the vehicle's own ground contact.

If visible, the camera returns

$$
\zeta_t = p_t + \varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2 I_3),\quad \sigma=0.15\ \mathrm{m},
$$

in world axes **relative to the vehicle**, i.e. the observation stores $\zeta_t-p$. The noise uses a
separate counter-based stream (`seed + 7919`) so it never perturbs the plant's motor-noise sequence. If
any condition fails, the observation stores exactly $\zeta_t-p=0$ and $y_t=0$.

## 6. Action space

Identical to QuadX-Waypoints: raw actions clipped to $[-1,1]^4$ and mapped affinely to
$(\omega_x^{\mathrm{cmd}},\omega_y^{\mathrm{cmd}},\omega_z^{\mathrm{cmd}},T)$ with physical ranges
$(\pm\pi,\pm\pi,\pm\pi,[0,0.8])$. All controllers use this same interface.

## 7. Observation space and the public-information contract

Float32 vector of dimension 25 under the `policy` key of a TensorDict:

| Slice | Symbol | Unit | Description |
|---|---|---|---|
| [0:3] | $\omega_b$ | rad/s | body angular velocity |
| [3:7] | $q$ as $(x,y,z,w)$ | — | attitude quaternion |
| [7:10] | $v_b$ | m/s | body-frame velocity |
| [10:13] | $p$ | m | world position |
| [13:17] | $\bar a_{t-1}$ | mixed | last physical action |
| [17:21] | $\theta$ | — | per-rotor motor states |
| [21:24] | $\zeta_t-p$ | m | current visible target measurement, world axes; **0 when invisible** |
| [24:25] | $y_t$ | — | visibility flag |

The observation contains **no target truth**: no target position, velocity, mode, schedule, or time since
last seen. An invisible step is indistinguishable from "no measurement at this instant" apart from the
flag. The PPO and FlashSAC entries consume exactly this vector with their original memoryless networks.
The classical baseline consumes the same vector; its Kalman filter is part of that method, not extra
sensor information.

## 8. Reward

With the reference point $\bar p = (p_{t,x}+\delta_x,\;p_{t,y}+\delta_y,\;\texttt{ALT\_REF})$,
$\delta=(1.5,0)$ m and altitude 6 m, per policy step:

$$
r_t = 0.10\,y_t - 0.10\,(1-y_t)
      - 0.05\,\lVert p_{xy}-\bar p_{xy}\rVert\,y_t
      - 0.01\,(z-\texttt{ALT\_REF})^2
      - 0.002\,\lVert\bar a_t\rVert^2
      - 10\,\kappa_t ,
$$

where $\kappa_t\in\{0,1\}$ is a crash/out-of-bounds event. The tracking term is paid only while visible:
when the target is lost, the only signal is the $-0.10$ loss penalty. The reward is an environment-side
quantity; it is not an observation channel, and no target state is exposed to a controller through it.

## 9. Termination, truncation, success

1. **Termination.** Ground contact, or $\lVert p\rVert > 30$ m. Sets `terminated`, clears the episode.
2. **Truncation.** 1200 policy steps (30 s).
3. **Success.** At the time-limit truncation, `complete` is set when the episode's visible-time fraction
   is at least 0.85 and the vehicle has not crashed. Success is recorded but does not cut the episode
   short; `extras['time_outs']` therefore contains all time-limit truncations, and the value bootstrap
   follows the same convention as the other tasks.

## 10. Initial state distribution

The vehicle spawns level and at rest at $(0,0,6)$ m. The target spawns on the ground within a 0.5–2 m
radius of the vehicle's ground projection, so the first camera frame is guaranteed to be a detection.
Reset replicas run the plant's 10-by-2 zero-command warmup and then seed the observation with that first
measurement. A fixed seed reproduces the spawn, the target schedule and the sensor noise bit-for-bit.

## 11. Metrics

Per completed episode the environment accumulates

| Metric | Definition |
|---|---|
| `visible_fraction` | visible policy steps / episode steps |
| `lost_seconds` | total invisible time in the episode |
| `max_dropout_seconds` | longest continuous invisible interval |
| `track_error` | mean $\lVert p_{xy}-\bar p_{xy}\rVert$ over visible steps |
| `flight_distance` | integrated path length, m |
| `collisions` | ground-contact terminations |
| `out_of_bounds` | out-of-sphere terminations |
| `timeouts` | time-limit truncations without success |
| `fov_loss_steps`, `range_loss_steps`, `occlusion_steps` | loss-cause counts (one cause per invisible step, in that priority) |

`scripts/baseline_eval.py` and `scripts/eval.py` divide the cumulative counters by the number of
completed episodes and report them under `task_metrics`.

## 12. Implementation

- File: `environments/multirotor/track_env.py`, class `QuadXTrackEnv` (task id `quadx_track`); exported
  from `environments/multirotor/__init__.py`. Plant in `environments/common/plant.py`.
- Conventional controller: `QuadXTrackKFBaseline`, exposed as the class attribute
  `conventional_controller`.
- Training entry: `environments/multirotor/train_quadx_track.py` (PPO / FlashSAC).
- Tuning and test harnesses: `eval_quadx_track_kf.py`, `test_quadx_track.py`.

| Constant | Value |
|---|---|
| policy/control/physics rates | 40 / 120 / 240 Hz (`inner_steps=3`, 2 substeps) |
| `max_steps` / episode | 1200 (30 s); `max_episode_length = 1202` |
| camera half-angle / sensor window | 30°; 1.5–12 m slant range |
| position noise | 0.15 m per axis |
| altitude / horizontal offset | 6.0 m; (1.5, 0.0) m |
| target speed / turn rate / segments | 0.4–1.2 m/s; 0.3–1.0 rad/s; 12 segments of 2–5 s |
| target area / vehicle bounds | 15 m / 30 m |
| success visible fraction | 0.85 |
| reward weights | +0.10 visible, −0.10 lost, −0.05/m track, −0.01/m² altitude, −0.002 action², −10 crash |
| `vel_clamp` | 15 m/s, 30 rad/s |

## 13. RL comparison protocol (original PPO vs original FlashSAC)

Both learned arms are the **unmodified repository implementations**:

- PPO: `rsl_rl` `OnPolicyRunner` with the task's shipped `FlashResTrunk` actor-critic (no recurrent
  layer), 100 vectorized steps per update, 5 epochs, 4 minibatches, lr $3\times10^{-4}$, entropy
  $0.01$, initial action std 1.0.
- FlashSAC: `algorithms/flashsac/flash_sac.py` (norm-controlled actor/critic, tanh-Gaussian policy,
  automatic temperature, one vectorized step per iteration), using its original `flash_sac_cfg`
  defaults. No filter, predictor, GRU/LSTM, safety layer, reward shaping or observation augmentation is
  added by the entry.

Both consume the 25-D observation of Section 7 (current own state, current measurement, visibility
flag; zero measurement when invisible), the same 4-D action interface, the same reward (Section 8), the
same reset distribution and the same evaluation scenarios.

**Shared interaction budget.** `TRAIN_ENV_STEPS = 6,000` vectorized policy steps at 1024 replicas
(6.14 M transitions) for both arms:

| Arm | Iterations | Env steps per iteration | Transitions |
|---|---|---|---|
| PPO | 60 updates | 100 | 6.14 M |
| FlashSAC | 6,000 iterations | 1 | 6.14 M |

The algorithms' own gradient-update counts and wall-clock differ by construction; this table equalizes
the data budget, which is the quantity that matters for the comparison. The seed is 7 for training and
for evaluation, so the learned arms and the classical baseline see the same scenario population. The
budget is deliberately small: it is the largest equal budget that completes inside a bounded wall-clock
on the shared, saturated GPU this benchmark is developed on. PPO therefore gets 60 updates and FlashSAC
6,000 gradient updates (its replay buffer needs 10,000 transitions before the first update, i.e. ~10
iterations at 1024 replicas).

**Commands.**

```bash
python environments/multirotor/train_quadx_track.py            # PPO arm
python environments/multirotor/train_quadx_track_flashsac.py   # FlashSAC arm
python scripts/eval.py quadx_track --out results/quadx_track/ppo
python scripts/eval.py quadx_track --out results/quadx_track/flash_sac
python scripts/baseline_eval.py quadx_track
```

Evaluation is deterministic (Gaussian mean action), 1024 replicas, seed 7, until every replica has
completed at least one episode; `scripts/eval.py` restores the training `env_kwargs` from
`results/quadx_track/<algo>/config.json`. Artifacts: `results/quadx_track/ppo/eval.json`,
`results/quadx_track/flash_sac/eval.json`, `results/quadx_track/baseline.json`.

## 14. References

- Tai, J. K. et al., *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, v0.29.0 — upstream
  Crazyflie 2.x plant, rate-PID and mixer reused unchanged.
- Genesis contributors, *Genesis* — rigid-body backend.
- Lee, T., Leok, M., McClamroch, N. H. (2010) — SE(3) geometric tracking used by the classical baseline.
