# QuadX-Track Baseline: Constant-Velocity Kalman Filter + Fixed-Offset SE(3) Tracking (`QuadXTrackKFBaseline`)

## 1. Overview

The QuadX-Track task requires a Crazyflie 2.x micro-quadrotor to keep a cooperative ground vehicle in a
nadir camera's field of view for 30 s while holding a fixed altitude and a fixed horizontal offset. The
target is only observed intermittently: field-of-view, slant-range and occlusion limits drop detections,
and detections carry 0.15 m position noise.

The classical baseline is the standard model-based solution: a **linear constant-velocity Kalman filter**
(Lee et al. 2010) that predicts through dropouts, followed by the repository's **SE(3) geometric tracking
controller** with a **fixed altitude and fixed horizontal offset** reference and velocity feed-forward.
A public fixed timeout-hold rule covers long outages. There is no active search, no information planning,
no learning and no safety layer.

## 2. Interface contract

- **Raw sensor information is identical to the learned methods.** The controller reads only the `'policy'`
  observation group: body rates $\omega_b$ `[0:3]`, attitude $q$ `[3:7]`, body velocity $v_b$ `[7:10]`,
  world position $p$ `[10:13]`, the current relative measurement $\zeta_t-p$ `[21:24]`, and the
  visibility flag $y_t$ `[24]`. When $y_t=0$ the measurement is the fixed zero placeholder; the filter
  then predicts only. The target truth (`env.target_pos`, `env.target_heading`, segment schedule) is never
  read.
- **Construction-time model knowledge.** Control period $\Delta t=1/40$ s, camera noise $\sigma=0.15$ m,
  altitude and offset, and the plant constants of the cf2x URDF are public design parameters. The flow
  footprint and LOS geometry are used only through the visibility flag, not by re-deriving occlusions.
- **Internal state.** Filter mean/covariance, visibility age, a frozen-hold point and the SE(3) axial trim.
  `reset_idx(mask)` clears all of it on the environment's reset mask.
- **Output.** The native action $a\in[-1,1]^4$ (three body-rate commands and collective throttle).

## 3. Constant-velocity Kalman filter

State $x=[p_t^\top,v_t^\top]^\top\in\mathbb{R}^6$ (target position and velocity, world frame).

**Prediction** every policy step, including dropout steps:
\begin{equation}
x^- = Fx,\qquad P^- = FPF^\top + Q,\qquad
F=\begin{bmatrix}I_3&\Delta t\,I_3\\0&I_3\end{bmatrix},\qquad
Q=q_a\begin{bmatrix}\tfrac{\Delta t^3}{3}I_3&\tfrac{\Delta t^2}{2}I_3\\[2pt]\tfrac{\Delta t^2}{2}I_3&\Delta t\,I_3\end{bmatrix},
\end{equation}
with $q_a=\sigma_a^2$ the white-acceleration power spectral density (`SIGMA_ACCEL`).

**Measurement update** only when $y_t=1$:
\begin{equation}
y=p+\zeta_t,\qquad H=[I_3\;\;0],\qquad
S=HP^-H^\top+R,\qquad K=P^-H^\top S^{-1},\qquad
x=x^-+K(y-Hx^-),\qquad P=(I-KH)P^-,
\end{equation}
with $R=\sigma^2I_3$ (`SIGMA_MEAS` = the known sensor sigma) and $P$ symmetrized after the update.
When $y_t=0$ the update is skipped entirely; the state is the prediction.

**Initialization.** At the first detection the filter is seeded with
$x=[y^\top,0^\top]^\top$ and $P_0=\mathrm{diag}(\sigma^2I_3,\sigma_v^2I_3)$, $\sigma_v=0.5$ m/s. No
target truth is used. The environment guarantees a detection at reset, so the filter is initialized on
the first control step.

**Fixed timeout hold.** The visibility age increments on every dropout step. If it exceeds
`HOLD_TIMEOUT = 3 s`, the reference is frozen at the estimate held at that instant,
\begin{equation}
\bar p \leftarrow \big[\hat p_{t,x}+\delta_x,\;\hat p_{t,y}+\delta_y,\;\texttt{ALT\_REF}\big]
\quad\text{at the timeout step,}
\end{equation}
and the vehicle hovers there until a new detection arrives, at which point the filter resumes. No search,
coverage, information-gain or replanning policy is executed during the hold.

## 4. Reference and control

The reference is
\begin{equation}
\bar p=\big[\hat p_{t,x}+\delta_x,\;\hat p_{t,y}+\delta_y,\;\texttt{ALT\_REF}\big],
\qquad \delta=(1.5,0)\ \mathrm{m},\quad \texttt{ALT\_REF}=6\ \mathrm{m}.
\end{equation}

The SE(3) cascade of the QuadX baseline is reused with one addition, a world-frame target-velocity
feed-forward:
\begin{equation}
v_d=\hat v_t-\frac{K_X}{K_V}(p-\bar p),\qquad
v_d\leftarrow v_d\,\frac{V_{\max}}{\max(\lVert v_d\rVert,V_{\max})},
\end{equation}
followed by the same desired-force, attitude, moment and rotor-allocation equations
$F_{\mathrm{des}}=m(g e_3-K_V(v-v_d))$, $b_{3,d}=F_{\mathrm{des}}/\lVert F_{\mathrm{des}}\rVert$,
$M=-K_R e_R-K_\Omega\omega_b+\omega_b\times J\omega_b$, and the closed-form rotor map. Yaw is regulated
to zero, exactly as in the stationary waypoint controller. The feed-forward removes the steady lag that a
pure position PD would have while chasing a 0.4–1.2 m/s target; the gains remain fixed constants.

## 5. Fixed parameters and independent tuning set

All values are class constants; none is adapted online.

| Constant | Value | Role |
|---|---|---|
| `SIGMA_ACCEL` $\sigma_a$ | 1.5 m/s² (selected) | white-acceleration std |
| `SIGMA_MEAS` $\sigma$ | 0.15 m | measurement noise std (known sensor spec) |
| `VEL_PRIOR_STD` | 0.5 m/s | first-detection velocity prior |
| `HOLD_TIMEOUT` | 3.0 s | public fixed timeout hold |
| `V_MAX` | 3.0 m/s | tracking speed cap |
| `KX` / `KV` | 4.0 / 2.0 (selected) | outer position / velocity gains |
| `KR` / `KOM` / `KI_POS` / `I_MAX` | inherited | SE(3) attitude / trim gains from `QuadXSE3Baseline` |
| $\delta$ / `ALT_REF` | (1.5, 0) m / 6 m | fixed horizontal offset and altitude |

**Selection protocol.** `environments/multirotor/eval_quadx_track_kf.py` evaluates a small grid on an
**independent tuning seed (101)**, disjoint from the reported evaluation seed (7):

$$
\sigma_a\in\{0.5,1.5,4.0\}\ \mathrm{m/s^2},\qquad (K_X,K_V)\in(4,2),(4,4),
$$

six combinations, 256 replicas each, one full 30 s episode per replica. The selection score is
$\text{success rate}+0.5\cdot\text{visible fraction}-0.2\cdot\text{track error}$ (ties by success, then
track error). The selected values are written to `results/quadx_track/kf_tuning.json` and frozen as the
class defaults used in the reported evaluation.

**Tuning result (seed 101, 256 episodes per combination, `results/quadx_track/kf_tuning.json`).**

| $\sigma_a$ | $K_X$ | $K_V$ | success | visible frac. | track error [m] | score |
|---|---|---|---|---|---|---|
| 0.5 | 4 | 2 | 0.805 | 0.909 | 0.278 | 1.204 |
| 0.5 | 4 | 4 | 0.805 | 0.912 | 0.242 | 1.212 |
| **1.5** | **4** | **2** | **0.820** | **0.916** | **0.256** | **1.227** |
| 1.5 | 4 | 4 | 0.805 | 0.913 | 0.227 | 1.216 |
| 4.0 | 4 | 2 | 0.781 | 0.900 | 0.252 | 1.181 |
| 4.0 | 4 | 4 | 0.766 | 0.891 | 0.222 | 1.167 |

The selected row is $\sigma_a=1.5$ m/s², $K_X=4$, $K_V=2$; all six rows stay within 0.02 of the best
score, so the choice is not a sharp optimum. The grid is deliberately small: the point is a
reproducible, documented selection, not an exhaustive sweep. No parameter is adapted at run time; the
same constants are used for every reported episode.

## 6. Measured performance

Reported evaluation: `scripts/baseline_eval.py quadx_track`, 1024 replicas, seed 7, GPU, one full 30 s
episode per replica (`results/quadx_track/baseline.json`). The controller saw only the public
observation — the same 25-D vector, zero placeholder and visibility flag the learned policies receive.

| Metric | Value |
|---|---|
| Success rate (visible fraction ≥ 0.85, no crash) | 0.851 (871/1024 episodes) |
| Visible-time fraction | 0.923 |
| Lost time per episode | 2.31 s |
| Longest dropout | 2.16 s |
| Mean standoff tracking error (visible steps) | 0.266 m |
| Collision rate (ground contact) | 0.000 |
| Out-of-bounds rate | 0.000 |
| Time-limit rate without success | 0.149 |
| Flight distance per episode | 24.17 m |
| Loss-cause steps per episode (FOV / range / occlusion) | 31.2 / 0.0 / 61.3 |
| Mean episode return | 85.9 |

Reading: the baseline keeps the target visible 92% of the 30 s episode and holds the fixed 1.5 m
standoff to 0.27 m RMS. All losses are geometric — the target leaves the footprint (FOV, 31 steps) or
passes behind a static occluder (61 steps); the range window never fails, because the fixed 6 m altitude
keeps the slant range near 6 m. The longest dropout (2.16 s) stays below the 3 s hold timeout, so the
frozen-hold rule never fires in the reported run; the 15% of episodes that miss the 0.85 success bar do
so by losing slightly more than 4.5 s of visibility, not by crashing.

## 7. Known limitations

- The filter assumes a constant-velocity target; turns and stop-go are handled only by the process noise
  $\sigma_a$, so the estimate lags during aggressive manoeuvres and the nominal covariance is optimistic.
- During a long dropout the reference is frozen after 3 s. If the target leaves the footprint while
  occluded or out of range, the baseline does not search and cannot reacquire; the episode records the
  lost time.
- The reference offset and altitude are fixed; the controller does not trade altitude for footprint
  coverage.
- The SE(3) inner loop is the same stationary-setpoint law as the waypoint task; only the velocity
  feed-forward is added. No trajectory optimization or occlusion-aware planning is performed.
- The controller reads only the public observation; it cannot know the target's true mode or schedule.

## 8. Implementation

- Controller: `algorithms/baselines/baselines.py`, class `QuadXTrackKFBaseline` (`name='cv_kf_pid'`),
  registered in `BUILDERS['quadx_track']`; exposed by `QuadXTrackEnv.conventional_controller`.
- Evaluation: `scripts/baseline_eval.py quadx_track` (artifact `results/quadx_track/baseline.json`).
- Tuning: `environments/multirotor/eval_quadx_track_kf.py` (artifact `results/quadx_track/kf_tuning.json`).
- Environment: `docs/method/env-quadx-track.md` (camera model, target schedule, reward, metrics).

## 9. References

- Lee, T., Leok, M., McClamroch, N. H., "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *49th
  IEEE Conference on Decision and Control*, 2010 — the SE(3) law reused here.
- Welch, G., Bishop, G., "An Introduction to the Kalman Filter," UNC-Chapel Hill TR 95-041 — the standard
  linear constant-velocity filter.
- Tai, J. K. et al., *PyFlyt*, v0.29.0 — cf2x plant and inner rate PID.
