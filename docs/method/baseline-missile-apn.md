# Missile-Intercept Baseline: Augmented Proportional Navigation (`MissileAPNBaseline`)

## 1. Overview

The classical baseline for the Missile-Intercept task is augmented
proportional navigation (APN) mapped onto the environment's body-rate
command interface through the skid-to-turn kinematic inverse. The law
combines a proportional-navigation term with navigation constant $N$ and a
half-$N$ gravity-compensation term, low-pass-filters the measured
line-of-sight (LOS) rate, and converts the commanded lateral acceleration
into body-rate commands that a 120 Hz rate PID in the plant tracks through
the tail-fin mixer. The guidance is analytic and fully batched on device;
the sole internal state is the per-replica LOS-rate filter, cleared on
episode reset.

The engagement it runs against is the environment's single fixed medium
operating point (decision of 2026-10-24): initial range 2500 m,
target-heading cone $\pm 37.5^{\circ}$ around the missile bearing, target
lateral acceleration 7.5 g, weave frequency 0.75 Hz, target and missile
speeds $V_T = V_{M0} = 250$ m/s at $\sim$4 km altitude, 15 m proximity-fuse
kill radius, and a 15 s episode limit. The missile is launched with
fire-control lead (straight-line extrapolation over the scaled nominal
time-to-go $t_{go} = 0.8\,R/(V_{M0} + V_T)$), so the shot starts on a near
collision course and the guidance corrects only residual errors.

## 2. Interface contract

- **Input.** Only the `'policy'` observation group of
  `MissileInterceptEnv` (34 dims, float32; cast to float64 internally). The
  channels read are: attitude quaternion (xyzw) `[3:7]`, body velocity
  $v_b$ `[7:10]`, target relative position in body frame $r_b$ `[23:26]`,
  relative velocity in body frame `[26:29]`, LOS rate in body frame
  $\lambda_b$ `[29:32]`, and closing speed $V_c$ `[32]`.
- **Output.** A raw action in $[-1,1]^3$, interpreted by the environment as
  normalized body-rate commands and affinely mapped to
  $(\omega_x, \omega_y, \omega_z)$ with per-axis limits
  $(\texttt{OMEGA\_MAX}, \texttt{OMEGA\_LAT}, \texttt{OMEGA\_LAT}) =
  (6.0, 1.0, 1.0)$ rad/s. The plant's rate PID converts these to fin
  commands through the skid-to-turn mixer; there is no throttle channel
  (the boost-coast engine is autonomous).
- **Construction.** Only the rate-command limits are taken from the
  environment module at construction; no internal environment state buffer
  is read.
- **Per-episode state.** `reset_idx(mask)` zeroes the LOS-rate filter
  $\hat\lambda$ on the masked replicas; the test harness drives this from
  `env.reset_buf` every step (skipping it leaks filter state across
  episodes and diverges on the first step).

## 3. Guidance and control law

### 3.1 LOS-rate filtering

The observed body-frame LOS rate passes through a first-order low-pass
filter with coefficient $\alpha = 0.8$, frozen in the terminal regime
$V_c < 50$ m/s where the LOS-rate measurement noise blows up:

$$
\hat\lambda \leftarrow \hat\lambda + \alpha\,(\lambda_b - \hat\lambda),\qquad
\alpha = \begin{cases} 0.8, & V_c \ge 50\ \text{m/s} \\ 0, & V_c < 50\ \text{m/s} \end{cases} \tag{1}
$$

The filtered rate is rotated to the world frame,
$\lambda_w = R\,\hat\lambda$, and the LOS unit vector is formed as
$\hat r_w = R\,(r_b / \lVert r_b\rVert)$ with
$\lVert r_b\rVert \ge 10^{-3}$ m.

### 3.2 Augmented proportional navigation

The commanded acceleration is the PN term plus half-$N$ gravity
compensation projected onto the plane perpendicular to the LOS, with
$g = (0, 0, -9.81)$ m/s²:

$$
a^{\text{cmd}} = N\,V_c\,(\lambda_w \times \hat r_w)
\;+\; \frac{N}{2}\,\big(g - (g \cdot \hat r_w)\,\hat r_w\big). \tag{2}
$$

The cross product $\lambda_w \times \hat r_w$ is load-bearing:
$\lambda_w = (r \times v_{\text{rel}})/\lVert r\rVert^2$ is the LOS
*angular velocity* (rotation axis), which is perpendicular to the LOS but
is not the acceleration direction; the command must be rotated 90° about
the LOS into the plane the LOS tip sweeps through. Dropping the cross
product tilts the command and produces persistent cross-range divergence
(verified in-repo: monotonically growing miss).

### 3.3 Skid-to-turn kinematic inverse

The command is expressed in the body frame and its thrust-axis ($b_x$)
component is dropped — the missile cannot accelerate along $b_x$ after
boost:

$$
a_{\text{lat}} = \big(I - b_x b_x^\top\big)\,R^\top a^{\text{cmd}}. \tag{3}
$$

Rotating the nose toward the commanded lateral-acceleration direction at
the turn rate $a/V$ curves the velocity vector accordingly, giving the
body-rate command

$$
\omega^{\text{cmd}} = \frac{b_x \times a_{\text{lat}}}{V},\qquad
V = \max(\lVert v_b\rVert,\, 50\ \text{m/s}). \tag{4}
$$

### 3.4 Output normalization

The rate command is normalized by the per-axis limits and clipped:

$$
a_{\text{raw}} = \operatorname{clamp}\!\Big(
\frac{\omega^{\text{cmd}}}{(6.0,\, 1.0,\, 1.0)},\, -1,\, 1\Big). \tag{5}
$$

Saturation of (5) is the deterministic fallback when the guidance demand
exceeds the rate-command envelope; the rate PID downstream simply saturates
the fins.

## 4. Parameters

All values are the exact code constants of `MissileAPNBaseline` and the
environment module it reads at construction.

| Constant | Value | Meaning |
|---|---|---|
| `N` | 15.0 | navigation constant (retuned from the documented N=4 / N=10 lineage; see Section 5) |
| `LOS_FILTER` | 0.8 | first-order LOS-rate filter coefficient $\alpha$ |
| filter freeze | $V_c < 50$ m/s | terminal LOS-rate noise guard ($\alpha = 0$) |
| speed floor | $V \ge 50$ m/s | denominator guard of the kinematic inverse (4) |
| range clamp | $\lVert r_b\rVert \ge 10^{-3}$ m | LOS-unit-vector singularity guard |
| `OMEGA_MAX` | 6.0 rad/s | roll-rate command limit (env constant, read at construction) |
| `OMEGA_LAT` | 1.0 rad/s | pitch/yaw rate command limit (raised 0.5 → 1.0 on 2026-10-24 for terminal transient authority) |

The fixed medium operating point of the environment (context, not tuned by
the baseline): `R_SPAWN` = 2500 m, `CONE` = ±37.5°, `A_LAT` = 7.5 g,
`F_WEAVE` = 0.75 Hz, `V_T` = `V_M0` = 250 m/s, `Z_MID` = 4000 m,
`KILL_RADIUS` = 15 m, `TGO_SCALE` = 0.8, episode limit 900 policy steps
(15 s).

## 5. Measured performance

All numbers are quoted from repository artifacts with their protocols;
success is a proximity-fuse kill (miss distance < 15 m at any physics
substep).

**Current configuration (N = 15, fixed medium operating point).**

| Source / protocol | Result |
|---|---|
| `results/missile/baseline.json` — official `scripts/baseline_eval.py missile`, 1024 envs, seed 7, GPU, 433 env steps (1037 completed episodes) | kill rate **0.8303** (861/1037), mean episode return 90.49 |
| In-code tuning note (`MissileAPNBaseline` comment), 1024-episode runs, seed 7, fixed medium operating point | N = 15: SR 0.812; N = 10: SR 0.770; N ≥ 18 overshoots and diverges the LOS loop |
| `environments/missile/test_missile_apn.py` T3 (acceptance gate, not a measurement) | asserts mixed-type SR ≥ 0.45 and straight-target SR ≥ 0.85 at the fixed medium point |

**Historical measurements (difficulty-parameterized environment, earlier
baseline configuration, seed 7).** The following predate the 2026-10-24
fixed-medium environment and the N = 15 retune; they are quoted with their
original protocol labels.

| Source / protocol | Result |
|---|---|
| `environments/missile/README.md` table, `test_missile_apn.py` T3, difficulty d = 0, 256 envs | SR 0.551 mixed / 1.000 straight; median miss (kills) 4.7 m |
| same, T5 sweep, 128 envs/cell | d = 0: SR 0.562 (straight 1.000, $r_0$ 1504 m); d = 0.5: 0.454 (0.775, 2503 m); d = 1.0: 0.062 (0.135, 3502 m) |
| same, official `baseline_eval.py missile`, benchmark d = 1.0, 1024 envs | SR 0.0576 (59/1025 episodes), mean episode return −75.0 |
| `results/missile/scenarios.json` — `eval_missile_scenarios.py` difficulty × maneuver-type grid, 256 envs per cell, ≤256 episodes per cell, seed 7 | d = 0: SR 0.7305 (straight 1.000, weave 0.886, barrel 0.423; median miss of kills 10.6 m, of all 12.6 m); d = 0.5: SR 0.6641 (1.000 / 0.977 / 0.116; 9.1 / 12.8 m); d = 1.0: SR 0.1318 (0.205 / 0.191 / 0.000; 10.4 / 84.6 m; barrel all-episode median miss 1036.3 m) |

Note: the prose summary of the scenario grid in
`environments/missile/README.md` (overall SR 0.551 / 0.492 / 0.066 and
median miss 7.4 / 8.1 / 84.7 m at d = 0 / 0.5 / 1.0) disagrees with the
on-disk `scenarios.json` artifact quoted above; the JSON artifact is the
authoritative record of that harness run.

## 6. Known limitations

Only limitations documented in the code comments or READMEs are listed.

- **Straight-line shot assumption.** The APN law assumes a non-maneuvering
  target; its residual failures concentrate on the weave and barrel-roll
  maneuver types at high difficulty (missile README). At d = 1.0 the tight
  high-severity barrel circle defeats the straight-line shot solution
  entirely (barrel SR 0.000, median miss 1036 m in the scenario grid). This
  maneuvering-target residual is the gap RL is expected to attack.
- **Terminal LOS-rate noise.** The filter freezes below $V_c < 50$ m/s;
  terminal corrections rely on the frozen estimate.
- **Navigation-constant sensitivity.** N ≥ 18 overshoots and diverges the
  LOS loop (in-code tuning note).
- **Rate-command envelope.** Guidance authority is bounded by the
  pitch/yaw rate limit; demands beyond it saturate (5) with no feasibility
  projection.
- **Stale labels.** The class docstring and both READMEs still describe the
  law as "N = 4"; the implemented constant is N = 15 (code wins). The
  missile README's action-space section likewise predates the
  `OMEGA_LAT` 0.5 → 1.0 change and the removal of the difficulty knob.

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class
  `MissileAPNBaseline` (`name = 'apn_rate'`), registered via
  `@register_baseline('missile')`; instantiated by
  `build_baseline('missile', env)` and declared as
  `conventional_controller` on
  `environments/missile/env.py::MissileInterceptEnv`.
- Evaluation: `scripts/baseline_eval.py missile` (official protocol, writes
  `results/missile/baseline.json`); test suite
  `environments/missile/test_missile_apn.py` (T1 fin-sign, T2 rate-loop,
  T3 closed-loop intercept gates); scenario harness
  `environments/missile/eval_missile_scenarios.py` (writes
  `results/missile/scenarios.json`).

## 8. References

1. Zarchan, P. *Tactical and Strategic Missile Guidance*, AIAA —
   proportional navigation and augmented proportional navigation with
   gravity compensation.
2. Siouris, G. M. *Missile Guidance and Control Systems*, Springer — PN
   mechanization and skid-to-turn airframe conventions.
3. Task lineage (project-authored environment, informed by): BVR Gym
   (arXiv:2403.17533) and LAG (arXiv:2506.14164) for the engagement and
   overload/LOS-rate conventions.
