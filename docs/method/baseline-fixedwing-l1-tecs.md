# Fixedwing Baseline: L1 Guidance + TECS Autopilot (`FixedwingAutopilotBaseline`)

## 1. Overview

The Fixedwing-Waypoints task requires a 2.35 kg, 1.6 m span unmanned aircraft — spawned at $(0,0,10)$ m at
20 m/s — to visit four waypoints sampled in a 20–65 m shell (altitude floor 20 m) around the origin within
a 120 s horizon at a 30 Hz policy rate; a waypoint is captured on entering a 10 m sphere, and ground
contact or leaving the 100 m out-of-bounds sphere fails the episode. The classical baseline is a
conventional fixed-wing autopilot: **L1 lateral guidance** (Park, Deyst and Howe 2004) turns the
line-of-sight angle to the active waypoint into a bank-angle command, and a simplified **TECS longitudinal
loop** (Lambregts 1983) assigns the specific total-energy error to the throttle and the energy-balance
error to the pitch command; inner roll/pitch loops close on the env's direct surface interface. Because the
airframe is power-limited (max thrust 18 N < weight ≈ 23 N, minimum turn radius 10–15 m), the shipped law
adds documented terminal-convergence augmentations: distance-scheduled L1, an aerodynamic bank limit from
the accelerated-stall boundary, an altitude-priority bank gate, a terminal pure-pursuit blend, an
out-of-bounds shield, and an overspeed climb trade.

## 2. Interface contract

- **Observations.** The controller reads only the `'policy'` observation group: body rates $\omega_b$
  ($[0{:}3]$), quaternion $(x,y,z,w)$ ($[3{:}7]$), body velocity $v_b$ ($[7{:}10]$), world position $p$
  ($[10{:}13]$), and the body-frame deltas $\Delta_k$ to the four waypoint slots ($[23{:}35]$); the active
  waypoint is reconstructed as $w = p + R\,\Delta_0$. No wind field exists in the tensorized plant, so
  airspeed equals ground speed, $V = \lVert v_b\rVert$.
- **Construction-time knowledge.** All gains and limits are loaded from
  `algorithms/baselines/fixedwing_autopilot.json`; the control period is $\Delta t = 1/30$ s (`agent_hz`).
  Angles use the env's euler convention (x fwd, y left, z up): positive roll = right bank, positive pitch =
  nose down.
- **Internal state.** Two scalar integrators per replica (throttle energy integral, pitch trim integral),
  cleared by `reset_idx(mask)`.
- **Output.** The env's native action $a\in[-1,1]^4$: $a_0$ differential ailerons ($a_0>0$ rolls right),
  $a_1$ horizontal tail ($+$, nose-down; opposite-sign vertical-tail coupling treated as a disturbance),
  $a_2 \equiv 0$ (dead upstream channel: the main wing has a zero deflection limit), $a_3$ throttle remapped
  as $\theta^{\mathrm{cmd}} = (a_3+1)/2$ with thrust $= \theta^2 \cdot 18$ N. Computation is float64 on
  `env.device`, batched; the returned action is float32.

## 3. Control law

$\sigma(x) = 1/(1+e^{-x})$ denotes the logistic gate; $\mathrm{wrap}(\cdot)$ wraps to $(-\pi,\pi]$;
$\phi,\theta$ are roll/pitch from the ZYX euler decomposition of $R$; $d_3 = \lVert\Delta_0\rVert$,
$h_{\mathrm{err}} = w_z - p_z$, $\rho_n = \lVert p\rVert$, $\rho_h = \lVert p_{xy}\rVert$.

**Lateral aim point.** The climb-phase containment weight
$w_{\mathrm{loit}} = \sigma\big((\rho_h - \rho_{\mathrm{loit}})/3\big)\,\sigma\big((h_{\mathrm{err}} - h_{\mathrm{loit}})/3\big)$
is shipped disabled ($h_{\mathrm{loit}} = 10^4$ m, hence $w_{\mathrm{loit}} \equiv 0$); the aim distance and
bearing blend waypoint and origin lines-of-sight,
\begin{equation}
d_h = (1-w_{\mathrm{loit}})\,\lVert w_{xy} - p_{xy}\rVert + w_{\mathrm{loit}}\,\rho_h,\qquad
\lambda = \mathrm{wrap}\big(\lambda_w + w_{\mathrm{loit}}\,\mathrm{wrap}(\lambda_o - \lambda_w)\big), \tag{1}
\end{equation}
$\lambda_w = \mathrm{atan2}(w_y-p_y,\,w_x-p_x)$, $\lambda_o = \mathrm{atan2}(-p_y,\,-p_x)$. The course is
$\chi = \mathrm{atan2}(v_y,v_x)$ when $\lVert v_{xy}\rVert > 2$ m/s, else the heading from $R$;
$\eta = \mathrm{wrap}(\lambda - \chi)$, clamped to $\eta_c \in [-\pi/2, \pi/2]$ so a waypoint behind the
aircraft commands a full-strength turn instead of vanishing through $\sin\eta$.

**L1 guidance, distance-scheduled.**
\begin{equation}
L_1 = \mathrm{clip}\big(k_{L1}\,d_h + L1_0,\; L1_{\min},\; L1_{\max}\big),\qquad
a_{\mathrm{lat}} = \frac{2V^2\sin\eta_c}{L1}\,\mathbb 1\{d_h > 3\ \mathrm{m}\},\qquad
\phi_{L1} = -\arctan\!\frac{a_{\mathrm{lat}}}{g}, \tag{2}
\end{equation}
with a 3 m near-field deadband (the target nearly overhead) and the env's bank sign convention (a target
to the left, $\eta > 0$, requires right-hand-negative bank).

**Bank limits.** The accelerated-stall boundary $n_{\max} = (V/v_{\mathrm{stall}})^2$ gives the aerodynamic
bank limit, tightened during climbs by $w_{\mathrm{cl}} = \sigma\big((h_{\mathrm{err}} - h_{\mathrm{cl}})/3\big)$:
\begin{equation}
\phi_{\max}^{\mathrm{aero}} = \min\!\Big(\arccos\,\mathrm{clip}\big((v_{\mathrm{stall}}/V)^2, 0, 1\big),\ \phi_{\max}\Big),\qquad
\phi_{\lim} = \phi_{\max}^{\mathrm{aero}} + w_{\mathrm{cl}}\big(\min(\phi_{\max}^{\mathrm{aero}}, \phi_{\mathrm{cl}}) - \phi_{\max}^{\mathrm{aero}}\big). \tag{3}
\end{equation}

**Terminal pure-pursuit blend and OOB shield.**
\begin{equation}
\phi_{pp} = -k_{pp}\,\eta,\qquad
w_{pp} = \sigma\big((d_{pp} - d_3)/3\big)\,\sigma\big((|\eta| - \eta_{pp})/(-0.1)\big)\,(1 - w_{\mathrm{loit}}),\qquad
\phi^{\mathrm{cmd}} = \mathrm{clip}\big(w_{pp}\,\phi_{pp} + (1-w_{pp})\,\phi_{L1},\; -\phi_{\lim},\; \phi_{\lim}\big). \tag{4}
\end{equation}
Beyond $\rho_{\mathrm{sh}} = 72$ m the boundary shield blends in a proportional home-bearing command that
overrides the climb-capped bank limit (clamped only by the aerodynamic limit): with radial inward speed
$\dot\rho^{-} = -(p^\top v)/\rho_n$ and home angle $\eta_o = \mathrm{wrap}(\lambda_o - \chi)$,
\begin{equation}
w_{\mathrm{sh}} = \sigma\big((\rho_n - 72)/4\big)\,\sigma\big((\dot\rho^{-} - 0.5)/1\big),\qquad
\phi^{\mathrm{cmd}} \leftarrow (1 - w_{\mathrm{sh}})\,\phi^{\mathrm{cmd}} + w_{\mathrm{sh}}\,(-k_{\mathrm{sh}}\,\eta_o). \tag{5}
\end{equation}
(The source comment describes the gate as "close to the boundary AND moving outward"; as implemented, the
second factor tests the radial *inward* speed against $0.5$ m/s.) The roll inner loop is a PD on bank angle
with rate damping, $a_0 = k_\phi\,\mathrm{wrap}(\phi^{\mathrm{cmd}} - \phi) - k_{d,\phi}\,\omega_{b,x}$.

**Speed schedule.** The reference bleeds with track error, with range below $d_{\mathrm{slow}}$, and on climb-gated legs:
\begin{equation}
v_\eta = \mathrm{clip}\big(v_{\mathrm{ref}} - k_{\eta}|\eta|,\, v_{\min},\, v_{\mathrm{ref}}\big),\quad
v_d = \mathrm{clip}\big(v_{\min} + k_{d}(d_3 - d_{\mathrm{slow}}),\, v_{\min},\, v_{\mathrm{ref}}\big),\quad
v_{\mathrm{cl}} = v_{\mathrm{ref}} + (v_{\mathrm{cl,ref}} - v_{\mathrm{ref}})\,w_{\mathrm{cl}}, \tag{6}
\end{equation}
\begin{equation}
v_{\mathrm{eff}} = \min(v_\eta,\, v_d,\, v_{\mathrm{cl}}),\qquad
\delta_v = \frac{v_{\mathrm{eff}}^2 - V^2}{2g}. \tag{7}
\end{equation}

**TECS energy allocation.** Specific total-energy and energy-balance errors,
\begin{equation}
\delta_E = \mathrm{clip}\big(h_{\mathrm{err}} + \delta_v,\, -25,\, 25\big),\qquad
\delta_B = h_{\mathrm{err}} - \delta_v. \tag{8}
\end{equation}
Throttle (P + I on $\delta_E$) with surplus-gated anti-windup bleed
$\dot I_{\mathrm{aw}} = k_{\mathrm{aw}}(V - v_{\mathrm{eff}})^{+}\,\mathbb 1\{\delta_E < 0\}$:
\begin{equation}
I \leftarrow \mathrm{clip}\big(I + k_{E,i}\,\delta_E\,\Delta t - \dot I_{\mathrm{aw}}\,\Delta t,\; 0,\; I_{\max}\big),\qquad
\theta = \mathrm{clip}\big(\theta_{\mathrm{trim}} + k_E\,\delta_E + I,\; 0,\; 1\big). \tag{9}
\end{equation}
Pitch command (P on $\delta_B$) with banked-flight alpha feed-forward, underspeed energy protection, and
the overspeed climb trade:
\begin{equation}
\theta^{\mathrm{cmd}} = \theta_{\mathrm{trim}} - k_B\,\delta_B
  - \Big(\frac{1}{\max(\cos\phi,\,0.2)} - 1\Big)\alpha_{\mathrm{trim}}
  + k_{\mathrm{sp}}\,(v_{\mathrm{eff}} - V)^{+}
  - k_{\mathrm{up}}\,(V - v_{\mathrm{eff}})^{+}\,w_{\mathrm{cl}}, \tag{10}
\end{equation}
clamped to $[-\theta_{\mathrm{up}}(V),\ \theta_{\mathrm{down}}]$ (negative pitch = nose up) with
speed-scheduled nose-up authority
$\theta_{\mathrm{up}}(V) = \theta_{\mathrm{up}}^{\max}\,\mathrm{clip}\big((V - v_{lo})/(v_{hi} - v_{lo}), 0, 1\big)$.

**Pitch inner loop (PID, coordinated-turn feed-forward).** Rate damping acts only on the deviation from
the coordinated-turn pitch rate $q_{\mathrm{coord}} = \omega_{b,z}\tan\phi$ (pure-yaw kinematic identity):
\begin{equation}
I_\theta \leftarrow \mathrm{clip}\big(I_\theta + k_{\theta,i}\,(\theta^{\mathrm{cmd}} - \theta)\,\Delta t,\, -0.35,\, 0.35\big),\qquad
a_1 = k_\theta\,(\theta^{\mathrm{cmd}} - \theta) - k_{d,\theta}\,(\omega_{b,y} - q_{\mathrm{coord}}) + I_\theta. \tag{11}
\end{equation}

**Action assembly.** $a = \big(a_0,\ a_1,\ 0,\ 2\theta - 1\big)$, clipped to $[-1,1]^4$.

## 4. Parameters

All values from `algorithms/baselines/fixedwing_autopilot.json` (config name `l1_tecs`).

| Group | Parameters | Values |
|---|---|---|
| lateral | `l1_distance` / `l1_min_distance`; `l1_dist_gain` / `l1_dist_offset` | 40.0 m / 12.0 m; 0.6 / 8.0 m |
| lateral | `max_bank_deg`; `v_stall_guard`; `k_roll` / `kd_roll` | 60°; 11.5 m/s; 3.0 / 0.5 |
| longitudinal | `v_ref` / `v_turn_min`; `k_speed_bleed` / `k_dist_slow` / `d_slow_ref` | 20.0 / 14.0 m/s; 2.5 / 0.35 / 10.0 m |
| longitudinal | `k_energy` / `k_energy_i` / `integral_max` / `throttle_trim`; `k_aw_speed` | 0.008 / 0.005 / 0.2 / 0.35; 0.012 |
| longitudinal | `k_balance` / `k_speed_keep` / `k_geo_up` | 0.008 / 0.05 / 0.6 |
| longitudinal | `k_pp` / `d_pp_entry` / `eta_pp_gate_deg` | 1.2 / 35.0 m / 40° |
| longitudinal | `bank_climb_deg` / `v_climb_ref` / `h_climb_gate` | 45° / 14.0 m/s / 5.0 m |
| longitudinal | `pn_shield` / `v_shield_min` / `k_shield` | 72.0 m / 0.5 m/s / 2.0 |
| longitudinal | `h_loiter_gate` / `pn_loiter` | $10^4$ m (disabled) / 35.0 m |
| longitudinal | `pitch_trim_deg` / `alpha_trim_deg`; `pitch_down_max_deg` | −2° (nose up) / 2.5°; 12° |
| longitudinal | `pitch_up_max_deg` / `pitch_up_v_min` / `pitch_up_v_full` | 16° / 11.0 / 15.0 m/s |
| longitudinal | `k_pitch` / `kd_pitch` / `k_pitch_i` | 4.0 / 0.45 / 2.0 |

## 5. Measured performance

| Protocol | Source | Result |
|---|---|---|
| Frozen protocol (harness `environments/fixedwing/eval_fixedwing_autopilot.py`: 1024 envs, 3700-step windows > 3600-step horizon; train seed 7, dev seeds 1000–1003, test seeds 2000–2003) | root `README.md` table; `docs/evaluation_protocol.md` (frozen 2026-10-22, after a documented 21.3% → 74.9% tuning iteration) | ≈ 74.6% train / 74.4% dev / 73.1% test |
| Tuning record of the shipped config (`pitch_up_max_deg` 10 → 16 in the 2026-10-22 sweep) | comment field of `fixedwing_autopilot.json` | out-of-bounds 23.9% → 1.8% at 0.0 crash rate; 18° measured 94.9% vs 94.7% at 16° (16° kept for stall margin) |
| `scripts/baseline_eval.py fixedwing` (1024 envs, seed 7, one full episode per replica, no selection bias) | `results/fixedwing/baseline.json` | 994/1025 finished episodes successful (success rate 0.9698), mean episode return 222.36, 1386 env steps, 224.17 s wall |

These figures belong to different protocols and tuning snapshots and are not mutually reconciled in the
repo.

## 6. Known limitations

As documented in the source (config comment, env docstrings, protocol doc):

- Residual failure mode (1.8%): very high targets still time out climbing. The frozen protocol records the
  residual as a structural limitation — out-of-bounds endgame limit cycles (a constant $L_1$ sustains a
  stable orbit at $r \approx L_1/(2\sin\eta)$ outside the 10 m capture sphere, motivating (2)).
- The airframe is power-limited (max thrust 18 N < weight 23 N): sustained steep banks sink 0.5–1 m/s,
  hence the bank gate (3); throttle-side overspeed governors were tried and rejected (an output cap below
  trim destroys the integrator's trim estimate; an energy-error clamp forbids climb power at the scheduled
  airspeed).
- Climb-phase containment ($w_{\mathrm{loit}}$) is shipped disabled (`h_loiter_gate` $= 10^4$) after
  measuring net-negative on the train split; range-scheduled speed bleed and $k_{\mathrm{up}}\in\{0.8, 1.0\}$
  were likewise rejected. Parameters were tuned only on training/development initial conditions (train
  seed 7, dev seeds 1000–1003).
- Channel $a_1$ drives the horizontal and vertical tails with opposite signs (rudder coupling treated as a
  disturbance); $a_2$ is dead by upstream design and driven with exactly 0. No wind field exists in the
  tensorized plant; airspeed equals ground speed (calm-air benchmark).

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class `FixedwingAutopilotBaseline`
  (`name = 'l1_tecs'`); parameters: `algorithms/baselines/fixedwing_autopilot.json`; instantiated via
  `build_baseline('fixedwing', env)` and declared as `conventional_controller` in
  `environments/fixedwing/env.py`.
- Evaluation: `scripts/baseline_eval.py fixedwing` (artifact `results/fixedwing/baseline.json`);
  scenario-grid/ablation harness `environments/fixedwing/eval_fixedwing_autopilot.py` (writes
  `results/fixedwing/autopilot_report.json`).
- Environment reference: `docs/method/env-fixedwing-waypoints.md` (action surface mixing, actuator lags,
  engine model, capture and out-of-bounds geometry).

## 8. References

- S. Park, J. Deyst, J. P. How, "A New Nonlinear Guidance Logic for Trajectory Tracking," *AIAA Guidance,
  Navigation, and Control Conference*, 2004 — L1 lateral acceleration law (2).
- A. A. Lambregts, total-energy control (TECS), 1983, in the simplified form cited by the source
  (`fixedwing_autopilot.json`): energy error → throttle, energy-balance error → pitch (8)–(10).
- Tai, J. K. et al., *PyFlyt: UAV Simulation and Reinforcement Learning Environments*, v0.29.0 — upstream
  fixed-wing model, surface mixing and actuation convention.
