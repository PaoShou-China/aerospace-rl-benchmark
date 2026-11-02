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

Success is a proximity-fuse kill (miss distance < 15 m at any physics
substep). All shipped records use the `scripts/baseline_eval.py` accounting
(first episode per replica, fixed medium operating point, `env_kwargs = {}`).
The environment's `difficulty` constructor kwarg was removed on 2026-10-24,
so the earlier difficulty-parameterized tables are no longer reproducible
and are not quoted here.

**Protocol dataset A**
(`results/baseline_protocol/seed{7,13,21,101}/missile/baseline.json`,
128 replicas per seed):

| seed | kill rate | kills / finished | mean episode return | env steps |
|---|---:|---:|---:|---:|
| 7 | 75.0% | 96 / 128 | 100.53 | 591 |
| 13 | 71.1% | 91 / 128 | 95.62 | 656 |
| 21 | 85.9% | 110 / 128 | 113.61 | 575 |
| 101 (held-out) | 70.3% | 90 / 128 | 94.91 | 597 |

Protocol seeds 7/13/21: **77.3% ± 7.7** (297/384), mean return 103.25,
`complete = true` in every file. The seed-to-seed spread (70.3–85.9%) is
large relative to the sample size; a single 128-replica record must not be
read as a precise capability estimate.

**Shipped single record.** `results/missile/baseline.json` is the seed-13
protocol record: 128 replicas, seed 13, 656 env steps, **kill rate
0.7109 (91/128)**, mean episode return **95.62**, `complete = true`.

**Dataset B** (`results/missile/baselines/*_seed7.json`, 256 replicas,
seed 7, 637 env steps):

| arm | CLI key | kill rate | kills / finished | mean return |
|---|---|---:|---:|---:|
| `apn_rate` (this, N = 15) | — (default) | **76.2%** | 195 / 256 | 101.90 |
| `tpn` | `tpn` | 76.2% | 195 / 256 | 102.13 |
| `ppn` | `ppn` | 2.0% | 5 / 256 | 27.21 |

`tpn` (true PN without the half-$N$ gravity augmentation) matches APN on
this fixed snapshot; `ppn` (pure pursuit, no lead) fails almost everywhere.
Dataset B also reports `timeout_rate = 0.0` for `apn_rate`, i.e. the
failures are genuine passes of the closest-approach point rather than
15 s timeouts.

**Tuning record (not shipped).** The `MissileAPNBaseline` code comment
records 1024-episode seed-7 tuning runs at the fixed operating point in
which N = 15 gave SR 0.812 versus 0.770 at N = 10, and N ≥ 18 overshot and
diverged the LOS loop. No JSON artifact for those runs is in the checkout,
so they are design notes, not benchmark records.

**Acceptance gate (not a measurement).**
`environments/missile/test_missile_apn.py` T3 drives 256 replicas for up
to 3000 env steps and asserts only that the mixed-maneuver kill rate is
≥ 0.45 and the straight-target rate ≥ 0.85. It stores no artifact and is
not a result.

## 6. Known limitations

Only limitations documented in the code comments or READMEs are listed.

- **Straight-line shot assumption.** The APN law assumes a non-maneuvering
  target; its residual failures concentrate on the weave and barrel-roll
  maneuver types (the env draws `t_type` uniformly at reset). The
  per-maneuver breakdown is produced by
  `environments/missile/eval_missile_scenarios.py`, but its output
  `results/missile/scenarios.json` is **not shipped**, so no per-type
  failure split is quoted here. This maneuvering-target residual is the
  gap RL is expected to attack.
- **Terminal LOS-rate noise.** The filter freezes below $V_c < 50$ m/s;
  terminal corrections rely on the frozen estimate.
- **Navigation-constant sensitivity.** N ≥ 18 overshoots and diverges the
  LOS loop (in-code tuning note).
- **Rate-command envelope.** Guidance authority is bounded by the
  pitch/yaw rate limit; demands beyond it saturate (5) with no feasibility
  projection.

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class
  `MissileAPNBaseline` (`name = 'apn_rate'`), registered via
  `@register_baseline('missile')`; instantiated by
  `build_baseline('missile', env)` and declared as
  `conventional_controller` on
  `environments/missile/env.py::MissileInterceptEnv`.
- Evaluation: `scripts/baseline_eval.py missile` (official protocol, writes
  `results/missile/baseline.json`, which is shipped). Dataset-B arms are
  produced by `scripts/baseline_suite.py` /
  `scripts/baseline_eval.py missile --baseline <tpn|ppn>` under
  `results/missile/baselines/`. Test suite
  `environments/missile/test_missile_apn.py` (T1 fin-sign, T2 rate-loop,
  T3 closed-loop intercept gates; default backend `cpu`); scenario harness
  `environments/missile/eval_missile_scenarios.py` (writes
  `results/missile/scenarios.json`, **not shipped**).

## 8. References

1. Zarchan, P. *Tactical and Strategic Missile Guidance*, AIAA —
   proportional navigation and augmented proportional navigation with
   gravity compensation.
2. Siouris, G. M. *Missile Guidance and Control Systems*, Springer — PN
   mechanization and skid-to-turn airframe conventions.
3. Task lineage (project-authored environment, informed by): BVR Gym
   (arXiv:2403.17533) and LAG (arXiv:2506.14164) for the engagement and
   overload/LOS-rate conventions.
