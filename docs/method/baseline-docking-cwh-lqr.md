# Docking Baseline: Symplectic-Exact CWH Discrete LQR with Constraint Layers (`DockingCWHLQRBaseline`)

## 1. Overview

The classical baseline for the docking task is a discrete linear–quadratic
regulator (LQR) designed on an *exact* discretization of the
Clohessy–Wiltshire–Hill (CWH) plant, augmented with three explicit
constraint-handling layers: a range-corridor shield, a terminal capture
net, and a fuel derate. Because the design model reproduces the env's
symplectic Euler integrator bit-for-bit, the LQR model carries **zero
discretization error** against the plant for piecewise-constant
acceleration commands; all approximation enters only through the
constraint layers and the per-axis output saturation.

## 2. Interface contract

- **Observations.** Only the `'policy'` group: the raw 6-dimensional LVLH
  relative state $o = [p,\ v]$, $x$ radial, $y$ along-track, $z$
  cross-track.
- **Environment constants at construction.** Chaser mass $m=500$ kg,
  per-axis thrust limit $F_{\max}=10$ N (per-axis authority
  $a_{\max}=F_{\max}/m=0.02$ m/s²), control period $\Delta t=1$ s,
  $I_{sp}=300$ s, $g_0=9.80665$ m/s², success tolerances
  `pos_goal` $=1$ m and `vel_goal` $=0.1$ m/s, range-failure bound
  `range_fail` $=200$ m, fuel budget `fuel0` $=2$ kg, horizon
  `max_steps` $=500$ steps, and the mean motion
  $n=\sqrt{\mu/a^{3}}=1.1314\times10^{-3}$ rad/s ($\mu=3.986\times10^{14}$
  m³/s², $a=6.778\times10^{6}$ m, 400 km circular target orbit).
- **Per-episode state.** The controller integrates its own realized fuel
  $\hat f$ and a remaining-step counter; both, together with the layer
  statistics, are cleared by `reset_idx(mask)`, which the evaluation loop
  calls with the env's reset mask after every step.
- **Action.** $u\in[-1,1]^3$, mapped by the env to per-axis thrust
  $F = F_{\max} u$; fuel accumulates as
  $\Delta f = \lVert F\rVert_2\,\Delta t/(I_{sp}g_0)$.

## 3. Control law

**Design model (exact symplectic discretization).** The continuous CWH
plant is $\dot v = Gp + Hv + a$ with
$G=\mathrm{diag}(3n^{2},0,-n^{2})$,
$H=\big[\begin{smallmatrix}0&2n&0\\-2n&0&0\\0&0&0\end{smallmatrix}\big]$,
integrated by the env with a velocity-first symplectic Euler step. The
matching discrete model for the stacked state $x=(p,v)$ is
\begin{equation}
x_{t+1} = A_d x_t + B_d a_t,\qquad
A_d = \begin{bmatrix} I + \Delta t^{2}G & \Delta t\,(I+\Delta t\,H)\\
\Delta t\,G & I+\Delta t\,H\end{bmatrix},\quad
B_d = \begin{bmatrix}\Delta t^{2}I\\ \Delta t\,I\end{bmatrix},
\tag{1}
\end{equation}
which coincides with the plant exactly (helper `_cwh_lqr`,
`method='symplectic'`; a `'taylor2'` variant is retained only to
reproduce the legacy incumbent gains).

**Riccati gain (Bryson-normalized by the task tolerances).**
\begin{equation}
Q = \mathrm{diag}\!\Big(\frac{1}{p_{\mathrm{goal}}^{2}}I_3,\,
\frac{w_{\mathrm{vel}}}{v_{\mathrm{goal}}^{2}}I_3\Big)
= \mathrm{diag}(1,1,1,300,300,300),
\qquad
R = \frac{1}{(\kappa\,a_{\max})^{2}}I_3,\quad \kappa = 0.35,\quad w_{\mathrm{vel}}=3,
\tag{2}
\end{equation}
\begin{equation}
P = \mathrm{DARE}(A_d,B_d,Q,R)\ \ (\texttt{scipy.solve\_discrete\_are}),\qquad
K = (R + B_d^{\top} P B_d)^{-1} B_d^{\top} P A_d,\qquad
a_{\mathrm{lqr}} = -Kx .
\tag{3}
\end{equation}
With this normalization the pure position feedback demands exactly
$\kappa\,a_{\max}$ at the success boundary (35% authority, i.e. 65%
terminal headroom), and $w_{\mathrm{vel}}=3$ places the closed loop near
critical damping; per the implementation docstring, the verified capture
margin is $|v|\approx0.066$ m/s when crossing $|p|=1$ m, 34% under
`vel_goal`. The far-field approach saturates by physics (braking
$\sim$100 m of closure at 0.02 m/s² authority), which the per-axis output
saturation absorbs as a bang–brake phase.

**Constraint layer 1 — range-corridor shield.** The only state constraint
is $|p|<$ `range_fail` (docking has no keep-out sphere). With outward
radial rate $v_{\mathrm{out}} = v\cdot\hat p$ and assumed braking
authority $a_b = a_{\max}/2$, the stopping-augmented range is
\begin{equation}
d_{\mathrm{stop}} = \lVert p\rVert + \frac{\max(v_{\mathrm{out}},0)^{2}}{2a_b}.
\tag{4}
\end{equation}
If $v_{\mathrm{out}}>0$ and $d_{\mathrm{stop}} > 0.95\,$`range_fail` the
command is overridden by a full-authority brake,
$a \leftarrow -a_{\max}\hat p$; inside the corridor
$|p| > 0.8\,$`range_fail` any outward thrust component is projected out,
$a \leftarrow a - (a\cdot\hat p)\hat p$ when $a\cdot\hat p>0$.

**Constraint layer 2 — terminal capture net.** Inside
$|p| < 2\,p_{\mathrm{goal}}$, if the closing speed already exceeds the
velocity tolerance, the approach is braked so the success sphere is
crossed below `vel_goal`:
\begin{equation}
a \leftarrow +a_{\max}\hat p
\qquad\text{if}\quad
\lVert p\rVert < 2\,p_{\mathrm{goal}}\ \ \text{and}\ \ -(v\cdot\hat p) > v_{\mathrm{goal}} .
\tag{5}
\end{equation}

**Constraint layer 3 — fuel derate.** The controller integrates its own
realized impulse exactly as the plant does,
$\hat f \leftarrow \hat f + \lVert u\rVert_2 F_{\max}\Delta t/(I_{sp}g_0)$.
Once $\hat f$ exceeds $0.75\,$`fuel0`, the per-step thrust norm is capped
so the remaining budget covers the remaining horizon $N_{\mathrm{left}}$:
\begin{equation}
F_{\mathrm{allow}} =
\max\!\Big(\frac{(\mathrm{fuel0}-\hat f)\,I_{sp}\,g_0}
{N_{\mathrm{left}}\,\Delta t}\,m,\; 0.5\,F_{\max}\Big),
\qquad
a \leftarrow a\,\frac{F_{\mathrm{allow}}}{\lVert a\rVert\,m}
\ \ \text{if}\ \ \lVert a\rVert\,m > F_{\mathrm{allow}},
\tag{6}
\end{equation}
never below half authority. As implemented, the $m$-scaled cap relaxes to
the $0.5F_{\max}$ floor across the reference envelope, so the layer is a
defensive guard; the frozen protocol records baseline fuel usage well
below the 2 kg budget (maximum 0.912 kg, `docs/evaluation_protocol.md` §1).

**Output mapping.**
\begin{equation}
u = \operatorname{clip}\!\big(a\,m/F_{\max},\,-1,\,1\big)\ \text{(per axis)},
\tag{7}
\end{equation}
with saturated axes counted in `stat_sat`.

## 4. Parameters

| Parameter | Value | Source |
|---|---|---|
| $n$ | $\sqrt{\mu/a^3}=1.1314\times10^{-3}$ rad/s | env constants |
| $a_{\max}=F_{\max}/m$ | $10/500 = 0.02$ m/s² per axis | `env.f_max`, `env.mass` |
| $Q$ | $\mathrm{diag}(1,1,1,300,300,300)$ | Eq. (2), `pos_goal=1`, `vel_goal=0.1`, $w_{\mathrm{vel}}=3$ |
| $R$ | $(0.35\,a_{\max})^{-2}I_3 \approx 2.04\times10^{4}I_3$ | `kappa=0.35` |
| shield thresholds | corridor $0.8\,r_{\mathrm{fail}}=160$ m; brake $0.95\,r_{\mathrm{fail}}=190$ m; $a_b=0.5\,a_{\max}$ | `__init__` |
| capture net | $|p|<2$ m, closing speed $>0.1$ m/s | `__init__` |
| fuel derate | soft threshold $0.75\,$`fuel0` $=1.5$ kg; floor $0.5F_{\max}$ | `__init__` |
| horizon | 500 steps, $\Delta t = 1$ s | `env.max_steps` |
| `legacy` arm | `q_pos=10^{-4}, q_vel=1, r_acc=1$, `taylor2`, no layers | constructor flag (not default) |

## 5. Measured performance

Under the official `baseline_eval` protocol (1024 parallel envs, seed 7),
the root README's frozen baseline table reports a **100% success rate**
for the docking task. The frozen controller audit
(`docs/evaluation_protocol.md`, §2) records the controller as fully
successful across its scenario grid and frozen as the behavior-cloning
data source; it also notes the termination-priority fix (violation wins
over success on the terminal step) is immaterial for this baseline because
its fuel usage stays far below the budget.

## 6. Known limitations

- The LQR core assumes no constraints; all state/fuel constraints are
  enforced by the heuristic override layers of Eqs. (4)–(6), whose
  thresholds (0.8/0.95 range fractions, $a_b = a_{\max}/2$, 75% fuel
  point, half-authority floor) are design choices, not optimized values.
- The far-field approach is an open-loop bang–brake phase under
  saturation; the LQR shapes only the terminal segment.
- The fuel-derate cap as implemented (Eq. 6) is dominated by its
  $0.5F_{\max}$ floor over the reference envelope and is effectively a
  non-binding safety layer.
- The gain is computed once at construction for the frozen plant
  constants; there is no online adaptation to mass or orbit changes.

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class
  `DockingCWHLQRBaseline` (registry name `cwh_lqr`,
  `BUILDERS['docking']`); shared gain helper `_cwh_lqr`.
- Environment: `environments/satellite/rendezvous_docking/env.py`, class
  `SpaceDockingEnv` (`conventional_controller = DockingCWHLQRBaseline`).
- Evaluation: `scripts/baseline_eval.py docking`; deep-dive/ablation
  harness `environments/satellite/rendezvous_docking/eval_docking_lqr.py`.

## 8. References

1. Clohessy, W. H. & Wiltshire, R. S. (1960). "Terminal Guidance System
   for Satellite Rendezvous", *Journal of the Aerospace Sciences* 27(9),
   653–658 — the relative-motion model.
2. Bryson, A. E. & Ho, Y.-C. (1975). *Applied Optimal Control* —
   tolerance-based $Q/R$ normalization (Bryson's rule).
3. Supreeth, H. S. G. & Mehta, V. "SpaceGym: A Gymnasium-Based Benchmark
   for Deep Reinforcement Learning in Spacecraft Proximity Operations",
   CVPR 2026 Workshop on AI4Space — the task definition and integrator.
