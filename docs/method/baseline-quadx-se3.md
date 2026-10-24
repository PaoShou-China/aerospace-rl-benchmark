# QuadX Baseline: SE(3) Geometric Control (`QuadXSE3Baseline`)

## 1. Overview

The QuadX-Waypoints task requires a Crazyflie 2.x micro-quadrotor (27 g, X configuration) to visit four
randomly placed waypoints in order inside a 5 m flight dome, capturing each by entering a 0.2 m sphere,
within a 20 s horizon at a 40 Hz policy rate; ground contact or leaving the dome fails the episode. The
classical baseline for this task is the geometric tracking controller of Lee, Leok and McClamroch (2010),
the standard model-based solution for underactuated quadrotor waypoint tracking. A proportional-derivative
position loop with gravity feed-forward shapes a desired force vector; the collective thrust is its
projection onto the body thrust axis; an attitude loop defined directly on $SO(3)$ aligns the thrust axis
with the desired force; and the resulting wrench is allocated onto the four rotors in closed form. Two
practical augmentations adapt the published law to the benchmark plant: a velocity-saturated outer loop
that respects the flight dome, and an axial integral trim with anti-windup that absorbs the motor-noise
dc gain of the actuation chain. The controller is deterministic and analytic — no training, no tuning
loop, no online optimizer.

## 2. Interface contract

- **Observations.** The controller reads only the `'policy'` observation group: body rates
  $\omega_b$ (slice $[0{:}3]$), attitude quaternion $(x,y,z,w)$ ($[3{:}7]$), body velocity $v_b$
  ($[7{:}10]$), and world position $p$ ($[10{:}13]$). The active waypoint $p^\star$ is taken from
  `env.current_target`; the same quantity is encoded in the policy observation as the slot-0 body-frame
  waypoint delta plus the ownship position (the fixedwing baseline performs exactly this reconstruction),
  so no privileged state is involved.
- **Construction-time model knowledge.** The control period is read from the env,
  $\Delta t = 1/\texttt{agent\_hz} = 1/40$ s. Plant constants (mass, inertia, rotor arm, $k_m/k_f$,
  thrust window, inner rate-PID proportional gains) are frozen class constants of the Crazyflie 2.x
  reference URDF/YAML, mirroring the model knowledge a classical GNC designer would have; no internal
  plant state buffer is read.
- **Internal state.** One scalar axial-trim integral per replica. `reset_idx(mask)` clears it on the
  env's reset mask; there is no other memory.
- **Output.** The env's native action $a\in[-1,1]^4$: three body-rate commands (physical range
  $\pm\pi$ rad/s) and the collective throttle (physical range $[0,\,0.8]$). All computation is float64
  on `env.device`, batched over replicas; the returned action is float32.

## 3. Control law

**Position loop (velocity-saturated cascade).** With $e_x = p - p^\star$ and world velocity
$v = R\,v_b$, the outer loop commands a speed-capped approach velocity
\begin{equation}
\bar v_d = -\frac{K_X}{K_V}\,e_x,\qquad
v_d = \bar v_d\,\frac{V_{\max}}{\max(\lVert\bar v_d\rVert,\,V_{\max})},\qquad
e_v = v - v_d, \tag{1}
\end{equation}
so the unsaturated law recovers the classical PD ($-K_V e_v = -K_V v - K_X e_x$) while transits are
limited to $V_{\max}$. The desired force and collective are
\begin{equation}
F_{\mathrm{des}} = m\big(g\,e_3 - K_V e_v\big),\qquad
f_{\mathrm{raw}} = F_{\mathrm{des}}^\top b_3 + I,\qquad
f = \mathrm{clip}\big(f_{\mathrm{raw}},\,f_{\min},\,f_{\max}\big), \tag{2}
\end{equation}
with $b_3 = R e_3$ the body thrust axis in world frame, $f_{\min} = 4\,F_{\max}^{\mathrm{tot}}\,\pi_{\min}^2 = 0.02$ N
and $f_{\max} = F_{\max}^{\mathrm{tot}}\,T_{\max}^2 = 1.28$ N. The axial trim integral cancels hover residuals,
\begin{equation}
\dot I = -K_I\,(e_x^\top b_3)\,\mathbb 1\{\lVert e_x\rVert < 0.5\ \mathrm{m}\},\qquad
I \in [-I_{\max},\,I_{\max}], \tag{3}
\end{equation}
with anti-windup gating: $\dot I$ is clamped to be non-positive while $f_{\mathrm{raw}} > f_{\max}$ and
non-negative while $f_{\mathrm{raw}} < f_{\min}$, so the integrator freezes against the direction that
would deepen an active saturation.

**Desired attitude (yaw fixed at zero).**
\begin{equation}
b_{3,d} = \frac{F_{\mathrm{des}}}{\lVert F_{\mathrm{des}}\rVert},\qquad
b_1 = \frac{e_1 - (e_1^\top b_{3,d})\,b_{3,d}}{\lVert\,\cdot\,\rVert},\qquad
R_d = \big[\,b_1,\; b_{3,d}\times b_1,\; b_{3,d}\,\big], \tag{4}
\end{equation}
with a fallback to $b_3$ when $\lVert F_{\mathrm{des}}\rVert \le 10^{-6}$ and to the $e_2$ projection when
$b_{3,d} \approx \pm e_1$ degenerates the heading projection.

**Attitude loop on $SO(3)$.** With the vee map
$\mathrm{vee}(S) = \tfrac12(S_{32}-S_{23},\,S_{13}-S_{31},\,S_{21}-S_{12})$,
\begin{equation}
e_R = \tfrac12\,\mathrm{vee}\!\big(R_d^\top R - R^\top R_d\big),\qquad
e_\omega = \omega_b - R^\top R_d\,\omega_d = \omega_b, \tag{5}
\end{equation}
since the waypoint setpoint is stationary ($\omega_d = 0$). The moment law is
\begin{equation}
M = -K_R\,e_R - K_\Omega\,\omega_b + \omega_b \times (J\,\omega_b). \tag{6}
\end{equation}

**Allocation.** With rotor arms $(a_{y,i}) = (-1,1,1,-1)$, $(a_{x,i}) = (1,-1,1,-1)$ (motor order
$(+x,-y),(-x,+y),(+x,+y),(-x,-y)$), spin signs $s = (-1,-1,1,1)$, arm length $d$ and
$c = k_m/k_f$, the per-rotor thrusts
\begin{equation}
t_i = \frac{f}{4} + \frac{M_x\,a_{y,i} - M_y\,a_{x,i}}{4d} + \frac{M_z\,s_i}{4c},\qquad
t_i \leftarrow \mathrm{clip}\big(t_i,\,F_{\max}^{\mathrm{tot}}\pi_{\min}^2,\,F_{\max}^{\mathrm{tot}}/4\big), \tag{7}
\end{equation}
reproduce the wrench $(f, M)$ exactly (the arm/spin columns are mutually orthogonal with squared norm 4),
clipped to the per-rotor window $[0.005,\,0.5]$ N. Thrusts map to pwm commands through the plant's
quadratic thrust model, pre-compensated by the measured motor-lag dc gain $G_{\mathrm{pwm}} = 1.216$:
\begin{equation}
\pi_i = \frac{1}{G_{\mathrm{pwm}}}\sqrt{\frac{4\,t_i}{F_{\max}^{\mathrm{tot}}}},\qquad
T = \mathrm{clip}\big(\tfrac14\textstyle\sum_i \pi_i,\; 0,\; T_{\max}\big),\qquad
d_i = \pi_i - T. \tag{8}
\end{equation}

**Rate-command synthesis.** The env's 120 Hz inner loop is a rate PID whose proportional path delivers
the normalized torque $u \approx k_p\,(\omega^{\mathrm{cmd}} - \omega_b)$, mixed onto the rotors as
$\pi_i = T + u_x a_{y,i} + u_y(-a_{x,i}) + u_z s_i$. Inverting the (orthogonal) mixer and the
proportional gain,
\begin{equation}
\tau = \frac14 \sum_i d_i\,(a_{y,i},\,-a_{x,i},\,s_i),\qquad
\omega^{\mathrm{cmd}} = \omega_b + \tau \oslash k_p,\qquad
\omega^{\mathrm{cmd}} \leftarrow \mathrm{clip}\big(\omega^{\mathrm{cmd}},\,-\pi,\,\pi\big), \tag{9}
\end{equation}
so that the plant's remixing reconstructs the intended pwm vector up to the plant's own spread-preserving
clip and its (small) integral/derivative residuals. The action is
\begin{equation}
a = \Big(\frac{\omega^{\mathrm{cmd}}}{\pi},\; \frac{2T}{T_{\max}} - 1\Big)\ \mathrm{clipped\ to}\ [-1,1]^4. \tag{10}
\end{equation}

Cumulative clipped-command fractions (total thrust, per-rotor, rate) are accumulated in
`sat_total` / `sat_rotor` / `sat_rate` for offline saturation accounting.

## 4. Parameters

| Constant | Value | Role |
|---|---|---|
| `MASS` $m$ | 0.027 kg | vehicle mass (cf2x URDF) |
| `G` $g$ | 9.81 m/s² | gravity |
| `INERTIA` $J$ | $\mathrm{diag}(1.4,\,1.4,\,2.17)\times10^{-5}$ kg·m² | base-link diagonal inertia |
| `ARM` $d$ | 0.028 m | rotor distance from the body $z$ axis |
| `KM_KF` $c$ | $7.94\times10^{-12}/3.16\times10^{-10} \approx 0.02513$ | reaction-torque / thrust ratio |
| `TOTAL_THRUST` $F_{\max}^{\mathrm{tot}}$ | 2.0 N | total thrust at pwm $=1$ |
| `PWM_MIN` / `PWM_MAX` | 0.05 / 1.0 | plant per-rotor pwm window |
| `THR_MAX` $T_{\max}$ | 0.8 | env collective command ceiling |
| `KP_RATE` $k_p$ | $(0.04,\,0.04,\,0.08)$ | plant inner rate-PID proportional gains |
| `PWM_GAIN` $G_{\mathrm{pwm}}$ | 1.216 | measured motor-lag dc gain under noise mean 1.08 |
| `KX` $K_X$ | 4.0 s⁻² | position proportional gain |
| `KV` $K_V$ | 3.0 s⁻¹ | velocity proportional gain |
| `V_MAX` $V_{\max}$ | 1.5 m/s | cruise speed cap |
| `KR` $K_R$ | $8\times10^{-4}$ N·m/rad | attitude proportional gain |
| `KOM` $K_\Omega$ | $2.5\times10^{-4}$ N·m·s/rad | body-rate damping gain |
| `KI_POS` $K_I$ | 0.15 N/m | axial position-error integral gain |
| `I_MAX` $I_{\max}$ | 0.35 N | integral clamp ($\approx 1.3\times$ hover thrust) |
| near gate | 0.5 m | trim active only inside the capture neighborhood |
| `dt` $\Delta t$ | $1/40$ s | from `env.agent_hz` |

## 5. Measured performance

| Protocol | Source | Result |
|---|---|---|
| Official `baseline_eval` protocol (1024 envs, seed 7), aggregate reference | root `README.md` table | 100% success over 5120 episodes |
| `scripts/baseline_eval.py quadx` (1024 envs, seed 7, one full episode per replica, no selection bias) | `results/quadx/baseline.json` | 1024/1024 episodes successful (success rate 1.0), mean episode return 70.91, 523 env steps, 57.07 s wall |

The functional test suite `environments/multirotor/test_quadx_se3.py` certifies (failures raise): exact
wrench→rotor allocation without simulation (T0); $\ge 0.90$ benchmark waypoint success across batch sizes
(T1); $\ge 0.85$ under domain-randomized spawns at 1024 envs and under injected adversarial initial
positions/attitudes (T2); closed-loop recovery through rate-window, per-rotor and total-thrust saturation
transients (T3); and trim-integral clearing with re-convergence under partial batch resets (T4).

## 6. Known limitations

Documented in the source:

- The velocity cap (1) is a deliberate departure from the plain PD law: unsaturated, the outer loop
  accelerates to $\approx (K_X/K_V)\lVert e_x\rVert \approx 6$ m/s toward a 4.5 m waypoint and overshoots
  the 5 m dome.
- The trim integral exists only to cancel millimetre-scale hover residuals; it is gated to
  $\lVert e_x\rVert < 0.5$ m because a multi-metre transit error would wind the trim to its clamp
  ($\approx 1.3\times$ hover thrust) and loft every approach.
- The $G_{\mathrm{pwm}}$ feed-forward corrects the *mean* actuation gain (motor-lag fixed point under the
  multiplicative noise $\mathbb E[1 + 0.02\,\mathcal N(4,1)] = 1.08$); per-step noise is not compensated.
- The setpoint is tracked as stationary per control step: $\omega_d = 0$, commanded yaw fixed at zero, no
  trajectory feed-forward.
- Under hostile transients the commands saturate (total thrust, per-rotor pwm, $\pm\pi$ rate window); the
  controller records saturation fractions rather than re-planning around them.

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class `QuadXSE3Baseline` (`name = 'se3_geometric'`),
  instantiated via `build_baseline('quadx', env)`; the env declares it as `conventional_controller` in
  `environments/multirotor/env.py`.
- Evaluation: `scripts/baseline_eval.py quadx` (artifact `results/quadx/baseline.json`); functional
  harness `environments/multirotor/test_quadx_se3.py`.
- Environment reference: `docs/method/env-quadx-waypoints.md` (action convention, rate-PID inner loop,
  motor mixing, actuation lag and noise).

## 8. References

- T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *49th IEEE
  Conference on Decision and Control*, 2010 — force/torque law (2), (4)–(6).
- Tai, J. K. et al., *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, v0.29.0 — upstream
  Crazyflie 2.x vehicle model, rate-PID inner loop and motor mixer.
