# QuadX-Waypoints — Position–Attitude Cascaded PID Baseline

## 1. Overview

This document specifies a second classical arm for the QuadX-Waypoints task, added
alongside the SE(3) geometric baseline without modifying the environment, reward, success gate,
capture order or either learned arm. It is the textbook cascade for a rate-command
quadrotor: an **outer position PID** turns the active-waypoint error into a desired
acceleration, gravity compensation and a fixed yaw reference turn that acceleration into
**desired roll, pitch and collective thrust**, and an **attitude P/PD** turns the Euler
attitude error into **body-rate commands** which the environment's existing 120 Hz rate PID
executes. The action is the exact inverse of the env's four-dimensional rate+collective
scaling.

The loop hierarchy is therefore

```
position PID (40 Hz)  ->  a_des  ->  roll/pitch/thrust  ->  attitude P/PD (40 Hz)
    ->  body-rate command  ->  env rate PID (120 Hz)  ->  motor mixer (240 Hz)
```

The controller uses the same observations as the SE(3) baseline, the same frozen cf2x plant
constants, and the same active waypoint (`env.current_target`). It contains no learning, no
prediction, no waypoint lookahead, no trajectory feed-forward and no command-safety filter;
the only memory is one clamped position-error integral per replica. Output and integral
saturation are the ordinary $\mathrm{clip}$ operations of a PID implementation.

## 2. Interface contract

- **Observations.** Only the `'policy'` observation group: body rates $\omega_b$ ($[0{:}3]$),
  attitude quaternion $(x,y,z,w)$ ($[3{:}7]$), body velocity $v_b$ ($[7{:}10]$) and world
  position $p$ ($[10{:}13]$). The active waypoint $p^\star$ is read from
  `env.current_target`, the same quantity the SE(3) arm uses; the policy observation also
  encodes it in the slot-0 body-frame delta, so no privileged plant state is involved.
- **Construction-time model knowledge.** $\Delta t = 1/\texttt{agent\_hz}=1/40$ s and the
  frozen Crazyflie 2.x constants of the reference URDF/YAML: mass, total thrust at pwm $=1$,
  the env collective ceiling and the measured motor-lag/noise dc gain.
- **Internal state.** One world-frame position-error integral $I\in\mathbb{R}^3$ per replica
  and the last seen waypoint index (used to clear $I$ on a waypoint handoff).
  `reset_idx(mask)` clears both on the env's reset mask. There is no other memory.
- **Output.** The env action $a\in[-1,1]^4$: three body-rate commands (physical range
  $\pm\pi$ rad/s) and the collective throttle (physical range $[0,\,0.8]$). All arithmetic is
  float64 on `env.device`, batched over replicas; the returned action is float32.

## 3. Control law

### 3.1 Outer position PID

Let $e_p = p-p^\star$ and let $v_w = R\,v_b$ be the world velocity. The controller integrates
the position error in the world frame,

\begin{equation}
I_{k+1} = \mathrm{clip}\!\left(I_k + K_I\,e_p\,\Delta t,\,-I_{\max},\,I_{\max}\right), \tag{1}
\end{equation}

but freezes the update while the previous unsaturated command was already clipped,
$\lVert a_{\mathrm{raw},k}\rVert > A_{\max}$ (conditional anti-windup). The derivative is taken
on the measured velocity, not on the setpoint, so a waypoint handoff does not produce a
derivative kick:

\begin{equation}
a_{\mathrm{raw}} = -K_P\,e_p - K_D\,v_w + I,\qquad
a_{\mathrm{des}} = a_{\mathrm{raw}}\,
\frac{A_{\max}}{\max(\lVert a_{\mathrm{raw}}\rVert,\,A_{\max})}. \tag{2}
\end{equation}

Thus the outer loop is a PID on position whose output is a desired acceleration
($K_P$ in s⁻², $K_I$ in s⁻³, $K_D$ in s⁻¹). $I$ is cleared whenever the env advances its
waypoint index, which happens through the env's normal capture logic — the controller neither
changes the order nor adds a capture test.

### 3.2 Gravity compensation, desired attitude and collective

The specific-force command adds gravity,

\begin{equation}
a_{\mathrm{cmd}} = a_{\mathrm{des}} + g\,e_3, \tag{3}
\end{equation}

and is resolved in the heading frame of the fixed yaw reference $\psi_r$,

\begin{equation}
u = \begin{pmatrix}
a_{\mathrm{cmd},x}\cos\psi_r + a_{\mathrm{cmd},y}\sin\psi_r\\
-a_{\mathrm{cmd},x}\sin\psi_r + a_{\mathrm{cmd},y}\cos\psi_r\\
a_{\mathrm{cmd},z}
\end{pmatrix}. \tag{4}
\end{equation}

For the ZYX Euler convention of the plant, the thrust axis is
$b_3=(\sin\theta\cos\phi,\;-\sin\phi,\;\cos\theta\cos\phi)$ in the heading frame, so the
desired roll and pitch and the total thrust are

\begin{equation}
\theta_d = \operatorname{atan2}(u_x,u_z),\qquad
\phi_d = \operatorname{atan2}\!\left(-u_y,\sqrt{u_x^2+u_z^2}\right),\qquad
f = m\,\lVert a_{\mathrm{cmd}}\rVert. \tag{5}
\end{equation}

The signs follow the plant: a positive $x$ acceleration commands positive pitch
($b_{3,x}>0$), a positive $y$ acceleration commands negative roll ($b_{3,y}>0$), and a
positive $z$ acceleration raises the collective. The yaw reference is held at
$\psi_r=0$; the attitude loop below drives the actual yaw to it. Because
$a_{\mathrm{cmd},z}\ge g-A_{\max}>0$, both desired angles stay inside $(-\pi/2,\pi/2)$ and no
additional attitude-reference clamp is needed.

### 3.3 Attitude P/PD to body-rate commands

Let $(\phi,\theta,\psi)$ be the ZYX Euler angles of $R$ and let $\mathrm{wrap}(\cdot)$ map an
angle to $(-\pi,\pi]$. The attitude loop produces the body-rate reference

\begin{equation}
e_{\mathrm{att}} = \mathrm{wrap}\!\begin{pmatrix}\phi_d-\phi\\ \theta_d-\theta\\ \psi_r-\psi\end{pmatrix},\qquad
\omega^{\mathrm{cmd}} = K_{\mathrm{att}}\odot e_{\mathrm{att}} - K_d\odot\omega_b,\qquad
\omega^{\mathrm{cmd}} \leftarrow \mathrm{clip}(\omega^{\mathrm{cmd}},-\pi,\pi), \tag{6}
\end{equation}

with $K_{\mathrm{att}}=(K_{\mathrm{att}}^{r},K_{\mathrm{att}}^{p},K_{\mathrm{att}}^{y})$
in s⁻¹ and $K_d=(K_d^{r},K_d^{p},K_d^{y})$ dimensionless. The rate damping term acts on the
measured body rate and is the "D" of the attitude PD; the inner rate loop itself remains the
environment's PID, so the cascade does not duplicate or replace it. With the inner loop much
faster than the outer one (its proportional path gives a rate time constant of order
$10^{-2}$ s), the attitude loop behaves as a first-order system with time constant
$1/K_{\mathrm{att}}$ plus the small $K_d$ damping.

### 3.4 Inverse action mapping

The plant delivers per-rotor thrust $f_i=\theta_i^2 F_{\max}^{\mathrm{tot}}/4$ at motor state
$\theta_i$; the collective command $\pi\in[0,0.8]$ is the target of a first-order motor lag
($\tau_m=0.01$ s) followed by multiplicative noise with mean
$\mathbb{E}[1+0.02\,\mathcal N(4,1)]=1.08$. The mean steady-state gain from collective command
to motor state is the measured dc gain $G_{\mathrm{pwm}}=1.216$, so the collective that
delivers $f$ is

\begin{equation}
T = \mathrm{clip}\!\left(\frac{1}{G_{\mathrm{pwm}}}\sqrt{\frac{f}{F_{\max}^{\mathrm{tot}}}},\;0,\;T_{\max}\right),
\qquad T_{\max}=0.8. \tag{7}
\end{equation}

The env action scaling is affine in the raw action, $\bar a=\ell+\tfrac{a+1}{2}(h-\ell)$, with
$\ell=(-\pi,-\pi,-\pi,0)$ and $h=(\pi,\pi,\pi,0.8)$; inverting it gives

\begin{equation}
a = \left(\frac{\omega^{\mathrm{cmd}}}{\pi},\;\frac{2T}{T_{\max}}-1\right)
\;\mathrm{clipped\ to}\;[-1,1]^4. \tag{8}
\end{equation}

Equations (7)–(8) fix the units (rad/s and pwm), the signs (body-frame right-hand rates,
collective $\ge 0$) and the scaling (quadratic thrust and the measured dc gain) in one place.
Cumulative clipped-command fractions are accumulated in `sat_acc`, `sat_rate` and
`sat_thrust` for offline saturation accounting; they are diagnostics, not feedback.

## 4. Parameters

| Constant | Value | Role |
|---|---|---|
| `MASS` $m$ | 0.027 kg | vehicle mass (cf2x URDF/YAML) |
| `G` $g$ | 9.81 m/s² | gravity |
| `TOTAL_THRUST` $F_{\max}^{\mathrm{tot}}$ | 2.0 N | total thrust at pwm $=1$ |
| `THR_MAX` $T_{\max}$ | 0.8 | env collective command ceiling |
| `PWM_GAIN` $G_{\mathrm{pwm}}$ | 1.216 | measured motor-lag dc gain under noise mean 1.08 |
| `KP_POS` $K_P$ | 3.0 s⁻² | position proportional gain |
| `KI_POS` $K_I$ | 0.10 s⁻³ | position integral gain |
| `KD_POS` $K_D$ | 4.0 s⁻¹ | measured-velocity damping |
| `A_MAX` $A_{\max}$ | 4.0 m/s² | desired-acceleration norm clamp |
| `I_MAX` $I_{\max}$ | 0.5 m/s² | integral clamp |
| `KP_ATT` $K_{\mathrm{att}}^{r,p}$ | 4.0 s⁻¹ | roll/pitch attitude P gain |
| `KD_ATT` $K_d^{r,p}$ | 0.35 | roll/pitch rate damping |
| `KP_YAW` $K_{\mathrm{att}}^{y}$ | 2.0 s⁻¹ | yaw attitude P gain |
| `KD_YAW` $K_d^{y}$ | 0.20 | yaw rate damping |
| `YAW_REF` $\psi_r$ | 0 rad | fixed yaw reference |
| `RATE_MAX` | $\pi$ rad/s | env rate-command window |
| `dt` $\Delta t$ | $1/40$ s | from `env.agent_hz` |

The inner-loop constants that are **not** touched are the env's rate-PID gains
$(k_p,k_i,k_d)=((.04,.04,.08),\,(5{\times}10^{-7},5{\times}10^{-7},2.7{\times}10^{-4}),
\,(10^{-4},10^{-4},0))$ and the motor mixer; they are part of the plant. The full action and
observation layout is in [env-quadx-waypoints.md](env-quadx-waypoints.md).

## 5. Measured performance

All numbers below are from this repository on the GPU backend, calm air (no
`results/quadx/config.json`, so the default env kwargs), seed 7, and the
unbiased accounting of `scripts/baseline_eval.py`: one completed episode per
replica, no in-flight episode dropped. The two classical arms were re-run back
to back on the same machine; the learned arms are the unchanged shipped entries.

### 5.1 Official 1024-env comparison

| Arm | Episodes | Success | Mean waypoints | Crash / OOB / timeout | Window | Mean return | Wall |
|---|---|---|---|---|---|---|---|
| SE(3) geometric (`baseline.json`) | 1025 | **100.0%** | 4.000 | 0 / 0 / 0 | 583 steps | 72.30 | 509 s |
| Cascaded PID (`baseline_cascaded_pid.json`) | 1024 | **99.12%** | 3.991 | 0 / 0 / 9 | 800 steps | 72.53 | 629 s |
| PPO (`train_quadx.py`, unchanged) | — | reuse shipped checkpoint with `scripts/eval.py quadx` | | | | | |
| FlashSAC (`train_quadx.py`, unchanged) | — | reuse shipped checkpoint with `scripts/eval.py quadx` | | | | | |

The cascade is 0.9 pp behind the geometric arm and reaches the same mean
return. Its only failures are 9 timeouts (0.88%): those replicas chase a
boundary waypoint, arrive just after the 20 s episode cap, and are counted as
unsuccessful. There are **no crashes and no out-of-bounds terminations**, i.e.
the direction/authority behaviour is sound and the residual gap is transit time,
which is the expected price of the plain damped position PID without the SE(3)
arm's velocity cap and feed-forward. Both files live in `results/quadx/`; the
SE(3) window differs (583 vs 800 steps) only because the cascade run must wait
for its timeouts.

### 5.2 Control-direction checks (`results/quadx/cascade_pid_tests.json`)

T0 starts from a settled level hover with zero velocity and reads one control
call, so the collective comparison is not contaminated by the spawn fall step.
Values are the env action in $[-1,1]^4$ (rates/$\pi$ and $2T/T_{\max}-1$):

| Command | Measured | Check |
|---|---|---|
| hover, body rates | 0.0000 | $\lvert\omega\rvert<10^{-3}$ |
| $+x$ target | pitch action $+0.3776$ | pitch $>0$ |
| $-x$ target | pitch action $-0.3776$ | pitch $<0$ |
| $+y$ target | roll action $-0.3776$ | roll $<0$ |
| $-y$ target | roll action $+0.3776$ | roll $>0$ |
| $+z$ target | collective action $-0.1451$ (hover $-0.2518$) | collective up |
| $-z$ target | collective action $-0.3765$ | collective down |
| yaw $=+0.5$ rad | yaw-rate action $-0.3183$ ($-1.0$ rad/s) | yaw rate $<0$ |

All signs match the plant: a positive $x$ error pitches nose-up, a positive $y$
error rolls left, a positive $z$ error raises the collective, and a positive yaw
error drives the yaw rate negative toward the fixed reference.

### 5.3 Hover, single waypoint, four waypoints

- **Hover** (256 envs, 400 steps $=10$ s, direct plant stepping so the setpoint
  is not consumed by capture): drift mean 1.29 cm, max **5.04 cm**, attitude
  max 0.000 rad, settled collective action $-0.2488$ (throttle 0.300, i.e. the
  gravity-compensation value with the integral trim). The integral settles the
  mean-gain bias without measurable attitude excursion.
- **Single waypoint +x**, offset $(2.5,0,0.6)$ m (64 envs, 400 steps): final
  error mean 0.111 m, max **0.119 m**; travelled $(+2.395,+0.000,+0.568)$ m, i.e.
  the $x$ and $z$ directions are correct and cross-coupling is below 0.1 mm.
- **Single waypoint +y**, offset $(0,2.5,0.6)$ m (32 envs, 300 steps): final
  error mean 0.105 m, max **0.118 m**; travelled $(-0.000,+2.400,+0.573)$ m.
- **Four waypoints**, 512 envs / seed 7 during tuning: success **99.41%**,
  mean 3.994/4, 0 crashes, 0 OOB, 3 timeouts; official 1024-env row is in
  §5.1.


## 6. Tuning record

Only the outer position loop was tuned; the attitude P/PD and the yaw gains were
left at their first physically motivated values because the inner rate loop is
an order of magnitude faster. Each candidate is a full four-waypoint episode in
`run_four_waypoints`, seed 7, one episode per replica, clipped saturation
fraction `sat_acc` reported for reference.

| Candidate | Envs | Success | Mean wp | OOB | Timeouts | `sat_acc` | Return |
|---|---|---|---|---|---|---|---|
| **$K_D=4.0$, $A_{\max}=4.0$ (chosen)** | 512 | **99.41%** | 3.994 | 0 | 3 | 0.168 | 72.53 |
| $K_D=2.5$, $A_{\max}=4.0$ (first guess) | 512 | 72.85% | 3.373 | 139 | 0 | 0.476 | 58.31 |
| $K_D=2.5$, $A_{\max}=4.0$ (first guess) | 256 | 69.53% | 3.258 | 78 | 0 | 0.487 | 56.14 |
| $K_P=4.0$, $K_D=4.0$ | 512 | 96.88% | 3.945 | 16 | 0 | 0.455 | 71.32 |
| $K_D=4.0$, $A_{\max}=5.0$ | 512 | 94.73% | 3.947 | 0 | 27 | 0.115 | 71.81 |

Narrative:

1. **$K_D=2.5$ (initial).** With $K_P=3$, the damping ratio is
   $\zeta=2.5/(2\sqrt3)=0.72$: the approach to a 4.5–4.8 m boundary waypoint is
   fast enough that the vehicle crosses the sphere and leaves the 5 m dome
   before the reverse command can brake it. This produced 27–30% OOB and the
   low returns in the table, at both 256 and 512 envs (the failure mode is
   deterministic, not noise).
2. **$K_D=4.0$ (chosen).** $\zeta=1.15$; the damping term alone removes the
   overshoot. OOB falls to 0 and success rises to 99.4% at 512 envs; the
   official 1024-env run is 99.12% with the 9 residual timeouts: those are slow
   boundary transits, not divergences. This is the no-velocity-cap alternative
   discussed in §7.
3. **$K_P=4.0$.** Stiffer proportional path re-introduces 16 OOB at 512 envs
   and lowers the return; rejected.
4. **$A_{\max}=5.0$.** More acceleration authority does not help: it converts
   the OOB failures into 27 timeouts (success 94.7%) because the approach is
   again too fast to settle inside the capture radius; rejected, $A_{\max}=4.0$
   kept.
5. **$K_I=0.10$** was not swept: with $G_{\mathrm{pwm}}$ feed-forward the
   steady-state bias is small, and 0.1 already removes the hover offset within
   the 10 s test while the 0.5 m/s² integral clamp keeps waypoint transients
   from winding up.
6. **Attitude loop** ($K_{\mathrm{att}}=4$ s⁻¹, $K_d=0.35$, yaw $2.0/0.20$) was
   checked by the T0 direction signs and the hover test (0.000 rad attitude
   excursion, 1.29 cm mean drift) and not re-tuned: the env's own rate PID does
   the fast work, and the outer loop's damping was the only failing degree of
   freedom in the four-waypoint runs.


## 7. Known limitations

- **No velocity cap.** The original SE(3) arm adds a $V_{\max}=1.5$ m/s cruise cap because an
  unsaturated PD outer loop reaches roughly $(K_X/K_V)\lVert e_x\rVert\approx 6$ m/s toward a
  4.5 m waypoint and overshoots the 5 m dome. This arm deliberately keeps the plain
  position-PID form; instead, the damping gain is chosen large enough
  ($K_D=4.0$ s⁻¹, damping ratio $\zeta=K_D/(2\sqrt{K_P})\approx1.15$ for $K_P=3$ s⁻²) that
  the approach speed collapses as the waypoint error shrinks. Tuning showed
  that the default $K_D=2.5$ produced out-of-dome failures; $K_D=4.0$ removed them without
  adding a velocity reference or safety filter. The trade-off is a slower transit, discussed
  in §6.
- **Waypoint handoff.** The integral is cleared on the env's waypoint-index change; the
  active waypoint is always `env.current_target`. The controller does not look ahead, choose
  a different waypoint or alter the capture order.
- **Attitude parametrization.** The attitude loop uses ZYX Euler errors, which are valid
  over the benchmark envelope (level spawn; command tilt below $22^\circ$ because
  $A_{\max}/g<0.41$). It is not a global $SO(3)$ attitude law, unlike the SE(3) arm; injected
  initial attitudes beyond the Euler chart are not part of its intended operating range.
- **Mean-gain feed-forward only.** $G_{\mathrm{pwm}}$ corrects the mean actuation gain; the
  per-step multiplicative noise and the residual rate-PID integral/derivative are left to
  feedback. The position integral absorbs the remaining hover bias.
- **Saturation is accounted, not avoided.** If the acceleration, rate or thrust command
  clips, the controller records the fraction and continues; there is no re-planning around
  saturation.

## 8. Implementation

- **Controller.** `QuadXCascadePIDBaseline` in
  [baselines.py](../../algorithms/baselines/baselines.py), `name = 'cascaded_pid'`.
  Registered as a named A/B variant:
  `BASELINE_VARIANTS['quadx']['cascaded_pid']`; the default `build_baseline('quadx', env)`
  still returns the SE(3) arm unchanged.
- **Evaluation.** The generic entry already supports named variants:
  ```bash
  python scripts/baseline_eval.py quadx                       # SE(3), baseline.json
  python scripts/baseline_eval.py quadx --baseline cascaded_pid  # -> baseline_cascaded_pid.json
  ```
  Both runs use 1024 envs, seed 7, the same env construction (including restored
  `results/quadx/config.json` when present), the same reward and the same success accounting.
- **Tests.** `environments/multirotor/test_quadx_cascade_pid.py` runs the direction-sign
  checks, hover station keeping, +x/+y single-waypoint convergence and a 512-env
  four-waypoint check; `--sweep` additionally runs the small gain sweep. The official
  1024-env record is produced by `scripts/baseline_eval.py`. The harness writes
  `results/quadx/cascade_pid_tests.json`.
- **Fair-comparison commands.** The learned arms are the shipped entries, unchanged:
  ```bash
  python environments/multirotor/train_quadx.py     # ALGORITHM = 'ppo' | 'flash_sac'
  python scripts/eval.py quadx                      # deterministic checkpoint evaluation
  python scripts/baseline_eval.py quadx             # SE(3)
  python scripts/baseline_eval.py quadx --baseline cascaded_pid
  ```
  All four arms share the task, the 1024-env/seed-7 protocol, the reward and the
  `4/4 waypoints < 0.2 m` success gate.

## 9. References

- S. Bouabdallah, P. Murrieri, R. Siegwart, "Design and Control of an Indoor Micro Quadrotor,"
  *IEEE ICRA*, 2004 — the position/attitude/rate cascade for a small quadrotor.
- D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors,"
  *IEEE ICRA*, 2011 — cascade structure with gravity compensation and a fixed-yaw attitude
  reference.
- T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3),"
  *IEEE CDC*, 2010 — the companion SE(3) arm and the geometric attitude alternative.
- J. K. Tai et al., *PyFlyt: UAV Simulation and Reinforcement Learning Environments*,
  v0.29.0 — cf2x vehicle model, 120 Hz rate PID, motor mixer, lag and noise.
