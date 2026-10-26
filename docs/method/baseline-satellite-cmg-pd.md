# Satellite Attitude Baseline: Quaternion PD + Singularity-Robust CMG Steering (`SatelliteCMGBaseline`)

## 1. Overview

The classical baseline for the CMG attitude-control task is a quaternion
proportional–derivative (PD) controller with exact gyroscopic feed-forward,
allocated onto the control-moment-gyro (CMG) cluster by a
singularity-robust damped pseudo-inverse steering law of the
Schaub–Junkins type. The controller commands gimbal *rates* (the env's
native action), never body torques: the plant maps gimbal rates to torque
through the configuration-dependent Jacobian of the cluster, so the
steering law must invert that map online while keeping gimbal rates
bounded as the cluster approaches a singular configuration. All
conventions below are verified against `SatelliteAttitudeEnv`
(`environments/satellite/attitude_control/env.py`).

## 2. Interface contract

- **Observations.** The controller reads only the `'policy'` observation
  group, whose layout is
  $o = [q,\ e,\ \omega,\ \sin\theta,\ \cos\theta,\ \dot\theta,\ m,\ a_{\mathrm{alloc}}]$
  with $q\in\mathbb{R}^{4}$ the bus attitude quaternion, $e\in\mathbb{R}^{3}$
  the body-frame attitude-error log map, $\omega\in\mathbb{R}^{3}$ the body
  rate, $\theta\in\mathbb{R}^{N}$ the gimbal angles ($N\in\{4,6\}$),
  $m$ the normalized cluster manipulability, and $a_{\mathrm{alloc}}$ an
  allocation feature the controller does not use. The implemented
  observation dimension is $11+4N$ (27 for the 4-CMG cluster, 35 for the
  6-CMG cluster). The controller consumes $e$ (`o[:, 4:7]`),
  $\omega$ (`o[:, 7:10]`), $(\sin\theta,\cos\theta)$
  (`o[:, 10:10+2N]`) and $m$ (`o[:, 10+3N]`).
- **Environment constants at construction.** `num_gimbals`, the dynamic
  rotor momentum `rotor_h_dyn` $h$, the rotor spin speed `rotor_speed`
  $\Omega$, the diagonal bus inertia `inertia` $I$, the reference rotor
  axes `rotor_axes_ref` $\{r_i^0\}$, the precomputed cross products
  `g_cross_r0` $\{g_i\times r_i^0\}$, and the command scaling
  `action_scale` $s_a = 3$ rad/s.
- **Per-episode state.** None. The controller is memoryless across
  episodes; `reset_idx` is the base-class no-op. Device-side diagnostic
  accumulators (clamp/nonfinite step counts, manipulability floor,
  peak stored momentum) persist for reporting only and never feed the
  control law.
- **Action.** $a\in[-1,1]^{N}$, converted by the env to gimbal rates
  $\dot\theta = s_a\,a$; $|a_i|=1$ is therefore exactly the
  $\pm 3$ rad/s rate bound. The plant has no hard gimbal stops (angles
  wrap at $\pm\pi$).

## 3. Control law

**Conventions.** Quaternions are $(w,x,y,z)$ Hamilton with
$\dot q = \tfrac12\,q\otimes(0,\omega_b)$, $\omega$ expressed in the body
frame. The attitude error is the shortest-arc log map of the
body-frame error quaternion,
\begin{equation}
e \;=\; \log\!\big(q^{-1}\otimes q_t\big)\in\mathbb{R}^{3},
\tag{1}
\end{equation}
which obeys $\dot e \approx -\omega$ for a static target. A positive body
rate about $e$ therefore shrinks the error, so the rate command carries a
**positive** proportional sign (Eq. 3 below).

**Cluster geometry from the observed gimbal angles.** With
$g_i\perp r_i^0$ and $|g_i|=1$, Rodrigues' formula gives the rotor
momentum directions and torque-Jacobian columns in closed form,
\begin{align}
r_i(\theta_i) &= \cos\theta_i\, r_i^0 + \sin\theta_i\,(g_i\times r_i^0),
\tag{2a}\\
g_i\times r_i(\theta_i) &= \cos\theta_i\,(g_i\times r_i^0) - \sin\theta_i\,r_i^0 .
\tag{2b}
\end{align}
The controller reconstructs the scaled Jacobian columns
$\mathrm{gr}_i = \Omega\,(g_i\times r_i)$ and the stored cluster momentum
$h_{\Sigma} = h\sum_i r_i(\theta_i)$ bit-exactly against the env's own
manipulability/torque code (probe-verified for both clusters).

**PD with exact feed-forward.** The desired body-frame angular
acceleration and the torque required to realize it on the plant
$I\dot\omega = -\omega\times(I\omega) - \omega\times h_{\Sigma} + \tau$
are
\begin{align}
\dot\omega_d &= k_p\,e - k_d\,\omega ,
\tag{3}\\
\tau_{\mathrm{cmd}} &= I\,\dot\omega_d \;+\; \omega\times(I\omega)
\;+\; \omega\times h_{\Sigma} ,
\tag{4}
\end{align}
i.e. feedback linearization with exact cancellation of the gyroscopic and
stored-momentum terms ($I$ diagonal). The closed-loop error dynamics
reduce to $\dot\omega \approx k_p e - k_d\omega$, $\dot e\approx-\omega$.

**Singularity-robust steering.** The plant produces
$\tau = -h\sum_i \dot\theta_i\,(g_i\times r_i) = -\kappa\,\mathrm{gr}^{\top}\dot\theta$
with $\kappa = h/\Omega$. The implemented damped pseudo-inverse of this
map is
\begin{equation}
\dot\theta
\;=\;
-\frac{1}{\kappa}\,\mathrm{gr}\,
\big(\mathrm{gr}^{\top}\mathrm{gr} + \lambda^{2} I_3\big)^{-1}\,
\tau_{\mathrm{cmd}},
\qquad
\lambda = \Omega\,\tilde\lambda,\quad
\tilde\lambda(m) = \lambda_0 + 0.15\,e^{-3m},
\tag{5}
\end{equation}
where $m$ is the observed normalized manipulability. With
$\mathrm{gr} = \Omega\,G$ ($G$ the unit-scaled Jacobian) and
$\lambda = \Omega\tilde\lambda$, Eq. (5) is algebraically identical to the
classical form
$\dot\theta = -(1/h)\,G\,(GG^{\top}+\tilde\lambda^{2}I_3)^{-1}\tau_{\mathrm{cmd}}$.
The adaptive damping grows as the manipulability collapses, trading torque
accuracy for bounded gimbal rates near singular configurations
(Schaub–Junkins singularity-robust steering); per the implementation
docstring, the torque-relay error is $\sim$0.05% far from singularities
and $\sim$3% fully damped. The damped $3\times3$ system stays symmetric
positive definite since $\tilde\lambda \ge \lambda_0 > 0$; $\tau_{\mathrm{cmd}}$,
$\dot\theta$ and $m$ pass through `nan_to_num` so nonfinite inputs steer
toward zero rate rather than poisoning the solve.

**Command mapping.**
\begin{equation}
a \;=\; \operatorname{clip}\!\big(\dot\theta / s_a,\,-1,\,1\big),
\qquad s_a = 3\ \mathrm{rad/s},
\tag{6}
\end{equation}
returned as `float32`. Every clipped step is counted in
`metrics['clamp_envsteps']`.

## 4. Parameters

| Parameter | Value | Source |
|---|---|---|
| $k_p$ (attitude gain) | 3.0 | `SatelliteCMGBaseline.kp` |
| $k_d$ (rate gain) | 4.0 | `SatelliteCMGBaseline.kd` |
| $\lambda_0$ (base damping) | 0.02 | `SatelliteCMGBaseline.lam0` |
| damping schedule | $\tilde\lambda = \lambda_0 + 0.15\,e^{-3m}$ | `actions()` |
| $s_a$ (action scale) | 3.0 rad/s | `env.action_scale` |
| 4-CMG plant | $I=\mathrm{diag}(75,60,113)$ kg·m², $h=20$ N·m·s, $\Omega=100$ rad/s | `SatelliteAttitudeEnv` |
| 6-CMG plant | $I=\mathrm{diag}(0.055,0.055,0.085)$ kg·m², $h=1.25\times10^{-2}$ N·m·s, $\Omega=200$ rad/s | `SatelliteAttitudeEnv` |
| control rate | 40 Hz (RK4, $10\times2.5$ ms substeps) | `SatelliteAttitudeEnv` |
| success tolerance | att. err $<0.05$ rad, $|\omega|<0.02$ rad/s | `SatelliteAttitudeEnv` |

## 5. Measured performance

Under the official `baseline_eval` protocol (1024 parallel envs, seed 7,
unbiased per-replica episode accounting), the root README's frozen
baseline table reports a **100% success rate over 1024 episodes** for the
nominal 4-CMG task. The frozen controller audit
(`docs/evaluation_protocol.md`, §2) additionally records the controller as
fully successful on the singularity, 180°-slew and 6-CMG audit suites and
lists it as the frozen behavior-cloning data source for this task.

## 6. Known limitations

- The damped inverse deliberately trades torque fidelity for bounded
  gimbal rates near singular configurations (documented relay error
  $\sim$0.05% far-field vs $\sim$3% fully damped); it does not steer the
  cluster *away* from singularities (no null-motion/gradient term), it
  only de-sensitizes the inverse.
- Gimbal-rate saturation at $\pm s_a$ is absorbed by the final clamp and
  counted, not compensated; long saturated slews are therefore
  open-loop in the torque direction.
- The law is tuned to the frozen cluster geometry and inertia taken from
  the env at construction; it carries no online identification.
- Diagnostic metrics are global accumulators, not per-episode state.

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class
  `SatelliteCMGBaseline` (registry name `quat_pd_damped_cmg`,
  `BUILDERS['satellite']`).
- Environment: `environments/satellite/attitude_control/env.py`, class
  `SatelliteAttitudeEnv` (`conventional_controller = SatelliteCMGBaseline`).
- Evaluation: `scripts/baseline_eval.py satellite`.

## 8. References

1. Schaub, H. & Junkins, J. L. *Analytical Mechanics of Space Systems*
   (AIAA Education Series) — singularity-robust (damped pseudo-inverse)
   CMG steering law.
2. Wie, B. (2008). *Space Vehicle Dynamics and Control*, AIAA — CMG
   momentum geometry and the manipulability (singularity) metric.
3. TorchRL contributors. *SatelliteEnv* (pytorch/rl, MIT) — reference
   task definition (reward weights, tolerances, reset distribution).
