# Fixedwing Baseline: Conventional Cascaded PID Autopilot (`FixedwingPIDBaseline`)

## 1. Overview

`FixedwingPIDBaseline` is a textbook cascaded PID autopilot added as a third classical
reference for the Fixedwing-Waypoints task (4 random waypoints in a 20–65 m shell, 20 m
altitude floor, 10 m capture radius, 100 m out-of-bounds sphere, 120 s / 3600-step horizon,
calm-air deterministic evaluation). It is deliberately *not* a guidance heuristic: unlike
the frozen L1+TECS baseline it has no distance scheduling, no terminal pure-pursuit blend,
no out-of-bounds shield, and no energy-priority logic.

The implemented stack is

$$
\text{bearing}\to\chi_{\mathrm{ref}}\to\phi_{\mathrm{ref}}\to\dot\phi_{\mathrm{ref}}\to\delta_a,\qquad
h_{\mathrm{err}}\to\theta_{\mathrm{ref}}\to\dot\theta_{\mathrm{ref}}\to\delta_e,\qquad
V_{\mathrm{err}}\to\delta_t,
$$

i.e. course-hold (P/I) → bank-angle → bank-rate → aileron, altitude-hold (P/I) →
pitch-angle → pitch-rate → elevator, and airspeed (P/I) → throttle. Every gain is constant;
the only controller state is five clamp-limited integrators cleared per episode by
`reset_idx`.

## 2. Interface contract (verified against the plant)

- **Observation group.** Only the public `'policy'` tensor is read: body rates
  $\omega_b$ (`[0:3]`), quaternion xyzw (`[3:7]`), body velocity $v_b$ (`[7:10]`), world
  position $p$ (`[10:13]`), and the body-frame deltas $\Delta_k$ to the four waypoint slots
  (`[23:35]`). The active waypoint is reconstructed as $w = p + R\Delta_0$; no privileged
  target or wind field is touched.
- **Airspeed vs. ground speed.** The observation exposes no pitot/true-airspeed channel.
  The speed loop uses $V = \lVert v_b\rVert$, which is the **ground speed** (body-frame
  velocity norm). In calm air — the frozen evaluation condition — it equals true airspeed.
  With the optional Dryden disturbance enabled it remains a ground-speed measurement and is
  deliberately *not* the internal air-relative truth $v_{\mathrm{air}} = v - w$: the
  controller never reads the wind field. Course is likewise the ground track
  $\mathrm{atan2}(v_y, v_x)$, falling back to heading $\psi$ when the horizontal speed is
  below `v_min_valid` = 2 m/s.
- **Surface signs** (x forward, y left, z up): positive roll = right bank, positive pitch =
  **nose down**. A left turn therefore needs a negative bank reference and a climb needs a
  negative pitch reference.
- **Action channels** $a\in[-1,1]^4$: $a_0$ differential ailerons ($a_0>0$ rolls right);
  $a_1$ horizontal tail ($a_1>0$ nose-down) plus an opposite-sign vertical-tail coupling
  treated as a disturbance; $a_2$ is the dead upstream main-wing channel (zero deflection
  limit) and is driven with **exactly 0**; $a_3$ is remapped by the env as
  $\delta_t = (a_3+1)/2$.
- **Numerics.** float64 on `env.device`, batched over replicas; float32 action returned.
  Control period $\Delta t = 1/30$ s.

## 3. Control law

$\mathrm{wrap}(\cdot)$ wraps to $(-\pi,\pi]$; $\phi,\theta$ are the roll/pitch angles from
the ZYX decomposition of $R$; $r=\omega_{b,z}$; $\mathrm{clip}(x,l,u)=\min(\max(x,l),u)$.

**Lateral: course → bank → bank rate → aileron.**

$$
\lambda = \mathrm{atan2}(w_y-p_y,\,w_x-p_x),\qquad
\chi = \begin{cases}\mathrm{atan2}(v_y,v_x) & \lVert v_{xy}\rVert > 2\\ \psi & \text{else}\end{cases},\qquad
e_\chi = \mathrm{wrap}(\lambda-\chi)
$$

$$
I_\chi \leftarrow \mathrm{clip}(I_\chi + k_{i\chi} e_\chi \Delta t,\ \pm I_{\chi,\max}),\qquad
u_\chi = k_\chi e_\chi + I_\chi,\qquad
\phi_{\mathrm{ref}} = \mathrm{clip}(-u_\chi,\ \pm\phi_{\max})
$$

$$
e_\phi = \mathrm{wrap}(\phi_{\mathrm{ref}}-\phi),\qquad
\dot\phi_{\mathrm{ref}} = \mathrm{clip}(k_\phi e_\phi,\ \pm\dot\phi_{\max}),\qquad
e_{\dot\phi} = \dot\phi_{\mathrm{ref}}-\omega_{b,x}
$$

$$
I_{\dot\phi}\leftarrow\mathrm{clip}(I_{\dot\phi}+k_{i\dot\phi}e_{\dot\phi}\Delta t,\ \pm I_{\dot\phi,\max}),\qquad
a_0 = \mathrm{clip}(k_{\dot\phi} e_{\dot\phi}+I_{\dot\phi},\ -1,\ 1)
$$

**Longitudinal: altitude → pitch → pitch rate → elevator.**

$$
h_{\mathrm{err}}=w_z-p_z,\qquad
I_h\leftarrow\mathrm{clip}(I_h+k_{ih}h_{\mathrm{err}}\Delta t,\ \pm I_{h,\max}),\qquad
u_h=k_h h_{\mathrm{err}}+I_h
$$

$$
\theta_{\mathrm{ref}}=\mathrm{clip}(\theta_{\mathrm{trim}}-u_h,\ -\theta_{\mathrm{up}},\ \theta_{\mathrm{down}}),\qquad
e_\theta=\theta_{\mathrm{ref}}-\theta,\qquad
\dot\theta_{\mathrm{ref}}=\mathrm{clip}(k_\theta e_\theta,\ \pm\dot\theta_{\max})
$$

$$
q_{\mathrm{coord}}=\omega_{b,z}\tan\phi,\qquad
e_{\dot\theta}=\dot\theta_{\mathrm{ref}}-(\omega_{b,y}-q_{\mathrm{coord}})
$$

$$
I_{\dot\theta}\leftarrow\mathrm{clip}(I_{\dot\theta}+k_{i\dot\theta}e_{\dot\theta}\Delta t,\ \pm I_{\dot\theta,\max}),\qquad
a_1=\mathrm{clip}(k_{\dot\theta}e_{\dot\theta}+I_{\dot\theta},\ -1,\ 1)
$$

The $q_{\mathrm{coord}}=\omega_{b,z}\tan\phi$ term is the pure kinematic identity between
body pitch rate and Euler pitch rate in a banked turn; subtracting it stops the inner loop
from treating a coordinated turn as a pitch disturbance (without it the aircraft sinks and
eventually hits the ground in sustained banks).

**Speed → throttle.**

$$
e_V=V_{\mathrm{ref}}-V,\qquad
I_V\leftarrow\mathrm{clip}(I_V+k_{iV}e_V\Delta t,\ \pm I_{V,\max}),\qquad
\delta_t=\mathrm{clip}(\delta_{\mathrm{trim}}+k_V e_V+I_V,\ \delta_{\min},\ \delta_{\max})
$$

**Action.** $a=(a_0,\ a_1,\ 0,\ 2\delta_t-1)$, clipped to $[-1,1]^4$.

**Anti-windup and limits.** Every loop uses *conditional integration*: the integrator is
frozen for the steps in which the loop output is saturated and the error still pushes
further into saturation (`|raw| > limit && raw*error > 0`), on top of hard integral clamps.
References are limited by $\phi_{\max}$, $\dot\phi_{\max}$, $\theta_{\mathrm{up/down}}$,
$\dot\theta_{\max}$, and surfaces by $[-1,1]$; throttle by $[\delta_{\min},\delta_{\max}]$.
No stall-protection, adaptive scheduling, path planning, NMPC or learning compensation is
present.

## 4. Parameters

`algorithms/baselines/fixedwing_pid.json` (config name `pid_cascade`).

| Group | Parameter | Value | Meaning |
|---|---|---|---|
| lateral | `k_course` / `k_i_course` / `i_course_max_deg` | 0.9 / 0.05 / 7° | course error → bank reference |
| lateral | `max_bank_deg` | 50° | bank reference clamp |
| lateral | `k_roll_angle` / `max_roll_rate_deg_s` | 2.0 / 50°/s | bank angle → bank-rate reference |
| lateral | `k_roll_rate` / `k_i_roll_rate` / `i_roll_rate_max` | 0.40 / 0.25 / 0.30 | bank rate → aileron |
| longitudinal | `k_altitude` / `k_i_altitude` / `i_altitude_max_deg` | 0.025 / 0.0012 / 10° | altitude error → pitch reference |
| longitudinal | `pitch_trim_deg` | −2° (nose up) | trim pitch reference |
| longitudinal | `pitch_up_max_deg` / `pitch_down_max_deg` | 18° / 12° | pitch reference authority |
| longitudinal | `k_pitch_angle` / `max_pitch_rate_deg_s` | 2.0 / 50°/s | pitch angle → pitch-rate reference |
| longitudinal | `k_pitch_rate` / `k_i_pitch_rate` / `i_pitch_rate_max` | 0.80 / 0.50 / 0.90 | pitch rate → elevator |
| speed | `v_ref` / `v_min_valid` | 16.0 / 2.0 m/s | airspeed reference / course-tracking validity floor |
| speed | `k_speed` / `k_i_speed` / `i_speed_max` | 0.12 / 0.04 / 0.30 | speed error → throttle |
| speed | `throttle_trim` / `throttle_min` / `throttle_max` | 0.30 / 0.15 / 1.0 | throttle bias and clamp |

## 5. Tuning procedure and scene separation

Tuning and formal testing are separate:

- **Tuning scenes** (`scenarios` mode): injected straight, left-turn, right-turn, climb,
  descent and constant-airspeed routes, evaluated on the tuning seed 7. They start from a
  trimmed cruise state (20 m/s spawn is overridden to 16 m/s) and use compact routes whose
  legs are longer than the ≈22 m minimum turn radius.
- **Formal comparison** (`benchmark` mode): random waypoints on held-out initial-condition
  seeds **2000–2003**, 1024 replicas, calm air, deterministic actions — the same accounting
  as `scripts/eval.py` (all episodes completed inside the window, one full episode per
  replica guaranteed).

Small-range sweeps only (all other gains fixed):

1. Bank limit 55°/60° and roll-rate 90°/s were rejected: in sustained 55–60° banks the
   airframe (max thrust 18 N < weight 23 N) cannot hold altitude; 14 m/s variants crashed.
2. 45° vs 50° bank at 16 m/s: 50° captured more waypoints per episode at zero crash rate.
3. The pitch-rate integrator limit had to be raised from 0.35 to 0.90: a 45–50° turn needs a
   sustained elevator trim that a 0.35 limit could not supply, causing a ≈2.7 m/s sink into
   the ground.
4. The coordinated-turn pitch-rate compensation `q_coord` (above) was added after observing
   a steady altitude loss in every sustained bank.

## 6. Measured performance

Held-out comparison, harness `environments/fixedwing/eval_fixedwing_pid.py benchmark`,
1024 replicas per run, calm air, deterministic actions, seeds 2000–2003
(none seen during tuning); every episode completed inside the window counts
(scripts/eval.py accounting). PID gains were tuned only on the scenario routes +
seed 7, so this table is out-of-sample for the PID arm.

| controller | success | waypoints/ep | mission time (s) | crash | OOB | timeout |
|---|---|---|---|---|---|---|
| L1+TECS (frozen) | 93.5% [93.0, 94.3] | 3.82 | 28.4 | 0.0% | 3.0% | 3.6% |
| **PID cascade (this arm)** | 80.4% [79.4, 81.3] | 3.54 | 35.0 | 0.0% | 10.8% | 8.8% |
| FlashSAC (1024×1000, seed 7) | 0.0% [0.0, 0.0] | 0.02 | n/a | 97.6% | 2.4% | 0.0% |
| PPO (1024×1000, seed 7) | pending training completion | | | | | |

Raw per-seed data: `results/fixedwing_pid/comparison.json`; trajectories
(first episode each of 8 replicas, seed 2000) in
`trajectories_<controller>_seed2000.{json,png}`.

Reading of the gap to L1+TECS (−13.1 pp): the PID arm loses episodes almost
exclusively through OOB limit-cycling near the 100 m sphere (10.8% vs 3.0%) and
timeouts (8.8% vs 3.6%) — its crash rate is 0.0%, identical to L1+TECS. This is
the expected structural consequence of §7: a pure course-hold loop with a fixed
bank clamp cannot always arrest the outbound dive before the OOB sphere, while
L1's distance-dependent pursuit term tightens the turn as the target nears.
Mission time is ~6.6 s longer (35.0 vs 28.4 s) for the same reason — wider,
later turns.

Scenarios (tuning split, single episode each, seed 7, calm air; full data in
`results/fixedwing_pid/scenarios_pid.json` / `scenarios_l1_tecs.json`):

| scene | PID verdict | PID highlights | L1+TECS highlights |
|---|---|---|---|
| straight | pass | 4.4 s, alt RMS 2.3 m, course RMS 0.2°, V 15.8 | 4.3 s, alt RMS 1.3 m |
| left turn | pass | 8.6 s, bank −53°, min alt 11.2 m | 7.1 s, bank −61°, min alt 13.4 m |
| right turn | pass | 8.2 s, bank +51°, min alt 16.7 m | 7.1 s, bank +58°, min alt 15.2 m |
| climb | pass | 4.7 s, +4.1 m net, pitch −13° | 4.4 s, +4.0 m net |
| descent | pass | 4.2 s, −8.1 m net, pitch +6.8° | 4.1 s, −6.2 m net |
| speed 14/18/22 | pass | tracked within 0.6/0.2/0.2 m/s RMS | speed is env-fixed in these scenes |

Turn scenes show the known structural cost of the pure course-hold loop: the aircraft
orbits once outside the capture sphere before threading the next waypoint
(course RMS ≈ 73° over the whole scene), while L1's pursuit term cuts the corner.
Both controllers score 5/5 (L1) and 8/8 (PID incl. three airspeed setpoints).

## 7. Known limitations

- **No distance scheduling.** A pure course-hold loop cannot cut inside a waypoint;
  consecutive waypoints closer than one turn radius can produce a stable orbit outside the
  capture sphere (the left/right-turn scenes need one extra orbit to capture). The random
  20–65 m task distribution is tolerant of this, but it is the dominant residual failure mode.
- **Power-limited airframe.** Sustained 50° banks at 16 m/s lose roughly 0.5–1 m/s of
  altitude; the altitude loop can only trade speed for height within the 12°/18° pitch
  authority, so deep turns show a 5–6 m dip.
- **Slow altitude integral** (`k_i_altitude` = 0.0012) leaves a small (~0.5 m/s) level-flight
  sag until the pitch-rate integrator builds the required trim; this is intentional — a
  faster altitude integral caused pitch/altitude oscillation in the tuning scenes.
- **No stall protection, NMPC, learning compensation or adaptation**, by design; the
  controller also ignores the L1/TECS heuristics (no terminal pursuit, no OOB shield), which
  is why OOB failures remain possible near the 100 m boundary.
- Channel $a_2$ is dead upstream and fixed at zero; the controller does not assume any
  authority on it.

## 8. Implementation and run instructions

- Controller: `algorithms/baselines/baselines.py`, class `FixedwingPIDBaseline`
  (`name = 'pid_cascade'`), registered as
  `BASELINE_VARIANTS['fixedwing']['pid_cascade']`; parameters in
  `algorithms/baselines/fixedwing_pid.json`. Instantiate with
  `build_baseline('fixedwing', env, 'pid_cascade')`.
- Harness: `environments/fixedwing/eval_fixedwing_pid.py` (`scenarios` / `benchmark` / `all`).

```bash
# 1. tuning / sanity scenes (straight, left/right turn, climb, descent, airspeed)
python environments/fixedwing/eval_fixedwing_pid.py scenarios --out results/fixedwing_pid

# 2. formal held-out comparison (PID + L1+TECS + shipped PPO / FlashSAC if present)
python environments/fixedwing/eval_fixedwing_pid.py benchmark \
    --num-envs 1024 --seeds 2000 2001 2002 2003 \
    --controllers l1_tecs pid_cascade ppo flashsac --out results/fixedwing_pid

# 3. single-baseline check through the shared baseline harness
python scripts/baseline_eval.py fixedwing --baseline pid_cascade --seed 101
```

Artifacts (under `results/fixedwing_pid/`): `scenarios_pid.json`,
`scenarios_l1_tecs.json`, `comparison.json`, `comparison.md`, and per-controller
`trajectories_<controller>_seed<k>.json` plus top-down PNG plots (x–y, colour = altitude).

The learned arms are loaded from `results/fixedwing_ppo/checkpoint.pt` and
`results/fixedwing_flashsac/checkpoint.pt` with the training `config.json`; missing
checkpoints are reported and skipped rather than silently substituted.

## 9. References

- G. F. Franklin, J. D. Powell, A. Emami-Naeini, *Feedback Control of Dynamic Systems* —
  cascade / anti-windup structure used here (conditional integration, reference and surface
  clamps).
- T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control* — coordinated-turn
  kinematic identity $q_{\mathrm{coord}} = r\tan\phi$ for banked flight.
- J. K. Tai et al., *PyFlyt*, v0.29.0 — upstream fixed-wing plant, surface mixing and action
  mapping ($a_2$ dead channel).
