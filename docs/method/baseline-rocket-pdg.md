# Rocket-Landing Baseline: Analytic ZEM/ZEV Powered-Descent Guidance (`RocketPDGBaseline`)

## 1. Overview

The classical baseline for the Rocket-Landing task is a fully analytic powered-descent guidance and control stack: a
closed-form zero-effort-miss / zero-effort-velocity (ZEM/ZEV) braking law for the mid-course descent, a two-stage
suicide-burn endgame tracked onto a virtual-pad braking manifold, a constant-rate terminal pulse servo, and a
proportional–derivative (PD) attitude inner loop with aerodynamic-moment feed-forward, allocated onto the engine
gimbal (TVC) and the four grid fins. There is **no online optimizer** — every quantity is re-solved in closed form at
every policy step, so the method has no solver and no solver-failure mode; demands beyond actuator authority saturate
their channel, the deterministic fallback of an analytic law, and the on-device saturation / coast / ignition-toggle
statistics are the honest analogs of solver diagnostics. The controller is fully batched tensor kernels: no
per-environment Python loop, no host synchronization.

## 2. Interface contract

- **Input.** Only the `'policy'` observation group of `RocketLandingEnv` (30 dims, float32; cast to float64
  internally): body angular velocity `[0:3]`, attitude quaternion xyzw `[3:7]`, body-frame velocity `[7:10]`, world
  position `[10:13]`, previous raw action `[13:20]`, fin actuation states `[20:24]`, ignited flag `[24]`, fuel
  fraction `[25]`, throttle state `[26]`, gimbal states `[27:29]`, pad-contact flag `[29]`.
- **Output.** A raw action in $[-1,1]^7$ in the environment's convention
  $(f_x, f_y, f_\psi, \texttt{ign}, \texttt{thr}, g_x, g_y)$. The environment affinely maps $[-1,1]^7$ onto its Box
  with lows $(-1,-1,-1,0,0,-1,-1)$, so the baseline emits ignition $\pm 1$ and throttle $2u-1$, $u\in[0,1]$; the plant
  remaps $u$ onto $[0.39, 1]\,T_{\max}$.
- **Construction.** Vehicle constants ($T_{\max}$, min-thrust ratio, dry and fuel mass, gimbal range, fin coefficients
  and moment arms) are taken from the environment/URDF specification at construction time, mirroring a classical GNC
  designer's model knowledge; no internal environment state buffer is ever read.
- **Per-episode state.** None: the ignition hysteresis is driven by the *observed* ignited flag, so the default no-op
  `reset_idx` suffices; only the diagnostic counters (`steps, sat_max, coast, ignitions, infeasible, floor, fin_sat`)
  persist across episodes.
- **Rates.** Policy 60 Hz / control 120 Hz (2 inner control periods per policy step) / physics 240 Hz; 30 s
  (1800-step) episode limit.

## 3. Guidance and control law

Let $m = M_{\text{dry}} + f\,M_{\text{fuel}}$, $v_w$ the world-frame velocity, $s = \max(0, -v_{w,z})$ the descent
speed, $r_f = (0,0,2.575\,\text{m})$ the pad rest point, $g = 9.81$ m/s².

### 3.1 Mid-course ZEM/ZEV braking ($s \ge S_{\text{gate}} = 25$ m/s)

Time-to-go is re-solved each step from a constant-deceleration profile at 60% of the maximum net braking acceleration
(braking early keeps the airflow — the fins' alignment target — near vertical; the spare 40% authority absorbs the
lateral tilt component and drag-model error), and the ZEM/ZEV acceleration toward $r_f$ with zero terminal velocity
follows (the velocity term *subtracts*: it brakes motion toward the target and stiffens motion away from it):

$$
a_{\text{net}} = \max\!\Big(\tfrac{T_{\max}}{m} - g,\; 1\Big),\quad
a_{\text{brake}} = 0.6\,a_{\text{net}},\quad
t_{go} = \operatorname{clamp}\!\Big(\frac{s - V_{\text{touch}}}{a_{\text{brake}}},\; 0.5,\; 30\Big),\quad
\mathrm{ZEM} = r_f - p - v_w\,t_{go},\quad
a^{\text{cmd}} = \frac{6\,\mathrm{ZEM}}{t_{go}^2} - \frac{2\,v_w}{t_{go}}. \tag{1}
$$

The vertical channel is floored and the lateral demand capped to a tilt-consistent budget ($\sim 6^{\circ}$, inside
the $\sim 9.6^{\circ}$ full-fin aerodynamic hold limit); the thrust-pointing acceleration adds gravity and never
rewards a downward demand (the engine can only push along body $+z$); thrust saturates at $T_{\max}$; and an ignition
hysteresis about $T_{\min} = 0.39\,T_{\max}$ self-lights the engine:

$$
|a^{\text{cmd}}_{xy}| \le \operatorname{clamp}\!\big(0.11\,\max(a^{\text{cmd}}_z {+} g, 0),\, 1,\, 4.5\big),\qquad
a^{\text{thr}} = a^{\text{cmd}} + g\,e_z\ (a^{\text{thr}}_z \ge 0),\qquad
T_{\text{cmd}} = \operatorname{clamp}\!\big(m\lVert a^{\text{thr}}\rVert,\, 0,\, T_{\max}\big), \tag{2}
$$

$$
\text{light when } T_{\text{cmd}} \ge 1.03\,T_{\min};\qquad
\text{stay lit while } T_{\text{cmd}} \ge 0.97\,T_{\min}. \tag{3}
$$

### 3.2 Endgame stage 1: virtual-pad min-thrust manifold ($s < 25$ m/s)

Min thrust exceeds weight (at the 158.7 kg spawn mass $T_{\min}/(mg) = 1.91$), so a lit engine cannot follow a slow
descent; the endgame brakes along the min-thrust manifold of a *virtual* pad raised $H_{\text{virt}} = 2$ m above the
real one. With $h_{\text{agl}} = \max(p_z - 2.575, 0)$:

$$
a_{\min} = \max\!\Big(\tfrac{0.39\,T_{\max}}{m} - g,\; 0.5\Big),\qquad
A_{\text{end}} = a_{\min} + \varepsilon_{\text{end}}\;(\varepsilon_{\text{end}} = 1.0),\qquad
h_v = \max(h_{\text{agl}} - H_{\text{virt}},\, 0.05), \tag{4}
$$
$$
s_t^2 = V_{\text{touch}}^2 + 2\,A_{\text{end}}\,h_v,\qquad
a_{\text{req}} = \frac{v_{w,z}^2 - s_t^2}{2\,h_v} + g + A_{\text{end}}. \tag{5}
$$

The $+A_{\text{end}}$ slope term in (5) is load-bearing: it makes the law track the manifold (the speed error decays
like $e \approx e_0\,h/h_0$) rather than merely terminate on it. The $\varepsilon_{\text{end}}$ margin rides the
commanded thrust ~5% above $T_{\min}$ so the plant's multiplicative thrust noise ($1 + 0.01\,\mathcal N$) can neither
clip the min-thrust floor nor out-brake the profile into cut–relight chatter. Below the manifold the engine coasts —
with $T_{\min} > mg$ the engine cannot descend slower than the manifold, so coasting *is* the below-line branch.

### 3.3 Endgame stage 2: terminal pulse servo ($s \le 1.5$ m/s, $h_{\text{agl}} \le 2$ m)

Inside the pulse gate the vertical demand is the constant $a_{\text{vert}} = g + A_{\text{end}}$ and ignition is a
speed hysteresis — cut when $s < S_{\text{off}} = 0.30$ m/s, relight when $s > S_{\text{on}} = 0.40$ m/s. While lit
the net acceleration is $+a_{\min}$ (brake only); while cut it is $-g$. The band is fixed in speed rather than
shrinking with height, so with the policy-period quantization the touchdown speed is bounded to roughly
$[0.15, 0.65]$ m/s by construction, against the 1.0 m/s fatal pre-contact threshold. Pad contact forces the engine
off, but the band servo is *not* latched off after touchdown: a micro-bounce re-launch re-enters the band on the way
down and the $S_{\text{on}}$ relight brakes it back onto the pad (a hard cut-after-touchdown latch was probed and
turned the rebound into a fatal 1.36 m/s re-contact). There is deliberately no rising-side relight: the engine can
only push up, so lighting during an ascent is a boost, not a catch (probed: a sustained bounce–boost oscillator with
~25% timeouts at 1024-env scale). The pulse-phase lateral law damps lateral *velocity* only, capped at
$A_{\text{lat,pulse}} = 1.5$ m/s²; the stage-1 lateral law is $a_{\text{lat}} = -1.5\,v_{w,xy} - p_{xy}$, capped as
in (2).

### 3.4 Throttle, attitude PD, and actuator allocation

Throttle command $u = \operatorname{clamp}((T_{\text{cmd}}/T_{\max} - 0.39)/0.61,\, 0,\, 1)$, emitted as raw action
$2u - 1$. The desired thrust axis is $b_3 = a^{\text{thr}}/\lVert a^{\text{thr}}\rVert$ (falling back to $e_z$ when
the demand vanishes or tips below $b_{3,z} \le 0.15$, keeping the guidance tilt across coast/burn transitions); with
$u_b = R^\top b_3$ the tilt error is $e_{\text{tilt}} = (-u_{b,y},\, u_{b,x},\, 0)$. The grid fins act as an
aerodynamic restoring spring of stiffness $k_{\text{aero}} V^2$ pulling body $+z$ toward the retrograde flow axis; a
feed-forward cancels it about the tilt axes, capped at the fin hold capability (the linear spring model breaks down
past the ~9.6° fin-stall trim point):

$$
\tau_{\text{ff}} = \operatorname{clamp}\!\big(k_{\text{aero}} V^2 e_{\text{flow}},\;
\pm 0.95\,K_{\text{fin}} z_{\text{fin}} V^2\big), \qquad
\tau^{\text{cmd}} = \big(K_{P,T}\, e_{\text{tilt},xy} - K_{D,T}\, \omega_{b,xy},\;
- K_{D,Z}\, \omega_{b,z}\big) - \tau_{\text{ff}}. \tag{6}
$$

The world-frame lateral force budget is split across three actuators; the thrust-tilt and gimbal side-force
contributions are reconstructed from the *observed* throttle and gimbal states, so the fins only make up the true
residual, capped at 30% of their force authority
($d_{\text{fin}} = a_{\text{fin}}^{\text{body}}\,m/(K_{\text{fin}} V^2)$). A deflected fin also torques the body at
the $z_{\text{fin}}$ lever; with the engine lit this coupling torque is pre-compensated into $\tau^{\text{cmd}}$
(while coasting it is left to the aero spring, since fins lock torque to force). Torque is allocated by ceiling —
fast TVC first, fins for the remainder — with $k_{\text{gim}} = h\,T_{\text{now}}\,\delta g_{\max}$ and
$k_{\text{fin}} = K_{\text{fin}} z_{\text{fin}} V^2$:

$$
\tau_{\text{gim}} = \operatorname{clamp}(\tau^{\text{cmd}},\, \pm 0.95\,k_{\text{gim}}),\qquad
\tau_{\text{fin}} = \operatorname{clamp}(\tau^{\text{cmd}} - \tau_{\text{gim}},\, \pm 0.35\,k_{\text{fin}}). \tag{7}
$$

The sign-verified actuator map emits $f_x = \tau_{\text{fin},y}/k_{\text{fin}} + d_{\text{fin},x}$,
$f_y = -\tau_{\text{fin},x}/k_{\text{fin}} + d_{\text{fin},y}$,
$g_{x,y} = -\tau_{\text{gim},x,y}/k_{\text{gim}}$ (zero when the engine is off), and
$f_\psi = -K_{D,Z}\omega_{b,z}/(K_{\text{yaw}} V^2)$.

## 4. Parameters

All values are the exact code constants of `RocketPDGBaseline`.

| Constant | Value | Meaning |
|---|---|---|
| `M_DRY` | 138.2 kg | structural mass (base 91 + booster 47 + 4 fins × 0.05) |
| `FUEL_FULL` | 410.9 kg | fuel-tank mass at fuel fraction 1 (spawn fuel 5%) |
| `T_MAX` | 7607.0 N | booster maximum thrust |
| `MIN_RATIO` | 2966.7/7607.0 ≈ 0.390 | min/max thrust ratio |
| `GIMBAL_RAD` | 5° | two-axis gimbal range |
| `G` | 9.81 m/s² | gravity |
| `FIN_K` | 0.22668 | finlet torque per unit deflection [N·m/unit/(m/s)²] |
| `FIN_YAW_K` | 0.1587 | 4-fin differential yaw torque coefficient |
| `AERO_K` | 3.6 | fin restoring-spring stiffness [N·m/rad/(m/s)²] |
| `V_TOUCH` | 0.35 m/s | pulse-band centre / manifold terminal speed |
| `TGO_FLOOR` | 0.5 s | ZEM/ZEV singularity guard (upper clamp 30 s) |
| `S_GATE` | 25 m/s | airspeed below which the endgame tracker replaces (1) |
| `EPS_END` | 1.0 m/s² | endgame decel margin over min-thrust net braking |
| `H_VIRT` | 2.0 m | virtual-pad raise |
| `S_PULSE` | 1.5 m/s | pulse-servo speed gate |
| `S_OFF` / `S_ON` | 0.30 / 0.40 m/s | pulse hysteresis cut / relight thresholds |
| `H_PULSE` | 2.0 m | pulse-servo altitude ceiling |
| `A_LAT_PULSE` | 1.5 m/s² | pulse-phase lateral accel cap (~1.6° tilt) |
| `LAT_TILT` | 0.11 | lateral cap as a fraction of vertical accel (~6°) |
| `A_LAT_MAX` | 4.5 m/s² | absolute lateral accel cap |
| `KP_T` | 12000 N·m/rad | tilt torque gain |
| `KD_T` | 6000 N·m·s/rad | tilt rate damping (≈ ζ 0.8) |
| `KD_Z` | 3.0 N·m·s/rad | yaw spin damping |
| `REST_Z` | 2.575 m | pad top 0.15 m + leg bottom offset 2.425 m |

## 5. Measured performance

All numbers are quoted from repository artifacts with their protocols; the success criterion is the upstream settling
check (pad contact with $\lVert v\rVert < 0.02$ m/s, $\lVert\omega\rVert < 0.02$ rad/s, tilt < 0.1 rad; pre-contact
speed > 1.0 m/s or spin > 0.35 rad/s is fatal).

| Source / protocol | Result |
|---|---|
| `results/rocket/baseline.json` — official `scripts/baseline_eval.py rocket`, 1024 envs, seed 7, GPU, 1800 env steps (one full 30 s episode per replica, 1024 episodes) | success rate **0.8281** (848/1024), mean episode return 3968.2 |
| `results/rocket/pdg_test_report.json`, scenario A (random upstream ICs) — `rocket_pdg_test.py`, 1024 replicas, 1250-step window, seed 7, GPU | SR 0.4395; pad-touch rate 0.731; touch speed p10/p50/p90 = 0.077/0.282/1.382 m/s; touch tilt mean 0.025 rad; touch spin mean 0.154 rad/s |
| same, scenario B (corner ICs: xy ±20 m × z {400, 450} × tilt {0, ±0.3}) | SR 0.3363; touch rate 0.489; touch speed p50 0.146 m/s |
| same, scenarios C/D/E (nominal z = 425 m; low-z 400 m; ±3 m/s lateral velocity, all xy = 0) | SR 0.4785 / 0.4961 / 0.4717; touch rate 0.776 / 0.776 / 0.769 |

Guidance diagnostics from the same harness (the analytic-law analogs of solver statistics), scenario A: max-thrust
saturation 0.89% of steps, engine-off coast 50.9%, infeasible demand 0.89%, $t_{go}$ floor hits 60.6%, ~122 ignition
pulses per episode, zero fuel-out episodes (mean fuel used 7.0 kg of the 20.5 kg spawn fuel), peak overload 5.07 g
mean / 5.39 g max; scenarios B–E are of the same order. Residual failures are dominated by fatal collisions (50–66%);
out-of-bounds and timeout are each below 1%.

## 6. Known limitations

Only limitations documented in the code comments or READMEs are listed.

- **Settle-phase rocking/topple deaths.** The dominant residual failure mode is arrival-state quality, not touchdown
  speed: measured touch spins (mean 0.12–0.15 rad/s, max up to 2.2 rad/s) sit far above the 0.02 rad/s settle
  threshold. Tilt-rate damping is the only actuator path that damps rocking while lit; the tuning notes record
  $K_{D,T} = 8000$ marginally better alone (83.9%) but degraded with the soft band, 10000 overshooting (79.2%).
- **Authority-saturated braking transients** are retained as documented residual failures (root README), visible as
  the ~0.9% saturation and infeasible-demand fractions above.
- **Pulse-band quantization.** The touchdown-speed guarantee is quantization-bounded (~0.25 m/s per 25 ms control
  sample, i.e. roughly ±0.12 m/s per 60 Hz policy step).
- **Aerodynamic model validity.** The linear fin-spring model holds only to ~10° of tilt (fins stall beyond the
  ~9.6° trim point), so the feed-forward is capped; large tilt commands are outside the design envelope.
- **One-way engine.** With $T_{\min}/(mg) = 1.91$ the engine can only brake; the terminal descent is necessarily a
  pulse train, and any rising-side relight is a boost (the probed bounce–boost failure mode), so the rebound catch
  is falling-side only.
- **Saturation is the fallback.** Any demand beyond authority saturates the corresponding channel; there is no
  feasibility projection.

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class `RocketPDGBaseline` (`name = 'zem_zev_pdg'`), registered in
  `BUILDERS` under `'rocket'`; instantiated by `build_baseline('rocket', env)` and declared as
  `conventional_controller` on `environments/rocket/env.py::RocketLandingEnv`.
- Evaluation: `scripts/baseline_eval.py rocket` (official protocol, writes `results/rocket/baseline.json`); deep-dive
  harness `environments/rocket/rocket_pdg_test.py` (writes `results/rocket/pdg_test_report.json`).

## 8. References

1. Battin, R. H. *An Introduction to the Mathematics and Methods of Astrodynamics*. AIAA — ZEM/ZEV guidance background.
2. Açıkmeşe, B. & Ploen, S. R. (2007). "Convex Programming Approach to Powered Descent Guidance for Mars Landing",
   *Journal of Guidance, Control, and Dynamics* 30(5) — PDG context; this baseline is the optimizer-free counterpart.
3. Tai, J. K. *PyFlyt: UAV Simulation and RL Environments* (0.29.0) — upstream task, vehicle model, landing criteria.
