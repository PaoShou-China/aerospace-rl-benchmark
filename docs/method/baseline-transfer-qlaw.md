# Low-Thrust Transfer Baseline: Petropoulos Q-Law Feedback Guidance (`TransferQLawBaseline`)

**English** · [简体中文](baseline-transfer-qlaw.zh-CN.md)

## 1. Overview

The second classical baseline for the low-thrust perigee-raising task is
Petropoulos' Q-law (AAS 04-278 / AAS 05-130), the most cited closed-loop
feedback guidance for low-thrust multi-revolution orbit transfer. A
Lyapunov-like proximity quotient $Q$ is built from the weighted,
time-scale-normalized squared errors of the osculating orbital elements
against a reference, and the Gauss variational equations are used twice:
once to normalize each element error by its maximum achievable rate of
change $\dot z_{xx}$ (a double maximum over the thrust direction and the
true anomaly), and once to pick the thrust direction that decreases $Q$
as fast as possible, $\hat u^{*}=-\Phi^{T}Q_z/\lVert\Phi^{T}Q_z\rVert$.
An *effectivity* test coasts the spacecraft whenever the instantaneous
$Q$-decrease rate is poor compared with what the current orbit could
deliver at its best anomaly.

**Task adaptation (declared, not claimed equivalent to a free-gain
formulation).** The task is "raise perigee, bound the apogee drift"
with no unique target orbit, so the Q-law reference is configured for it
explicitly: the perigee reference $r_{p,\mathrm{ref}}=r_{p0}+$
`RP_RAISE` is a public fixed parameter, and the apogee reference is kept
at the spawn apogee $r_{a,\mathrm{ref}}=r_{a0}$ so that the $e$-error
term penalizes apogee drift. The reference pair maps to a target
element pair $(a_T,e_T)$ (Eq. 4). Because the mission needs no plane
change, the law is implemented on the planar $(a,e)$ element subset with
$i,\Omega,\omega$ left free; the perigee penalty function $P$ of the
original formulation is not activated ($r_p\gg$ Earth radius throughout
the episode, and raising $r_p$ is the objective here). This is a
task-facing Q-law configuration, not a claim of equivalence with the
target-free gain formulation.

## 2. Interface contract

Identical to the Gauss baseline (same env, same buffers):

- **Observations.** Raw 8-dimensional $o=[t,\ r,\ v,\ m]$; the controller
  consumes $o[:, 1:4]$, $o[:, 4:7]$, $o[:, 7:8]$.
- **Environment constants.** $\mu$, $F_{\max}=10$ mN per axis pair,
  $I_{sp}=4000$ s, $g_0$, and the spawn radii $r_{p0}=8.5\times10^{6}$ m,
  $r_{a0}=1.15\times10^{7}$ m from `env.mu/f_max/isp/g0/sma0/ecc0`.
- **Per-episode state.** None; pure state feedback.
- **Action.** $a\in[-1,1]^3$; the env applies per-axis thrust
  $F=F_{\max}a$ and mass flow
  $\dot m=-F_{\max}\lVert a\rVert_1/(I_{sp}g_0)$. The controller emits
  either a unit-norm burn direction in the orbit plane or exactly $0$
  (coast), so the per-axis bound and the L1 fuel model are respected by
  construction.

## 3. Control law

**Osculating elements.** Eq. (1) of the Gauss-baseline write-up is used
verbatim, including the division-free true-anomaly recovery
$\nu=\operatorname{atan2}(h v_r/\mu,\ p/r-1)$ (near-circular safe).

**Proximity quotient (Petropoulos AAS 04-278, Eq. for $Q$; AAS 05-130
refinements).** On the planar subset $Z=[a,e]$,

\begin{equation}
Q = \sum_{z\in\{a,e\}} W_z\, S_z(Z)\left(\frac{d(z,z_T)}{\dot z_{xx}}\right)^{2},
\qquad
d(a,a_T)=a-a_T,\quad d(e,e_T)=e-e_T ,
\tag{1}
\end{equation}

with $W_p=0$ (perigee penalty $P$ not activated on this task) and the
semi-major-axis scaling function

\begin{equation}
S_a = \left(1+\left(\frac{a-a_T}{m\,a_T}\right)^{n}\right)^{1/r},
\qquad S_e = 1,
\qquad (m,n,r)=(3,4,2)
\tag{2}
\end{equation}

at Petropoulos' nominal values. $\dot z_{xx}$ is the maximum rate of
change of element $z$ over **both** the thrust direction and the true
anomaly on the current osculating orbit; here it is evaluated by an
$N_\theta=2048$-point true-anomaly grid scan of the Gauss rates. For
$a$ the analytic reference

\begin{equation}
\dot a_{xx} = \frac{2a^{2}}{h}\,(1+e)
\tag{3}
\end{equation}

(prograde transverse burn at perigee) is exact and is used only as a
harness check of the grid scan; no such closed form exists for $e$, so
$\dot e_{xx}$ is taken from the same grid.

**Reference mapping.** With the public perigee reference and the frozen
apogee reference,

\begin{equation}
a_T = \frac{r_{a,\mathrm{ref}}+r_{p,\mathrm{ref}}}{2},
\qquad
e_T = \frac{r_{a,\mathrm{ref}}-r_{p,\mathrm{ref}}}
            {r_{a,\mathrm{ref}}+r_{p,\mathrm{ref}}},
\qquad
r_{p,\mathrm{ref}} = r_{p0}+\texttt{RP\_RAISE},\quad
r_{a,\mathrm{ref}} = r_{a0} .
\tag{4}
\end{equation}

**Thrust direction (AAS 04-278).** Writing $Q_z=\nabla_Z Q$ (exact
partials, including the dependence of $\dot z_{xx}$ on $(a,e)$, obtained
by automatic differentiation through the grid maxima — valid almost
everywhere by the envelope theorem, and FD-verified) and $\Phi(Z)$ the
$(a,e)$ rows of the Gauss matrix in the radial/transverse frame,

\begin{equation}
\Phi = \begin{bmatrix}
\frac{2a^{2}}{h}\,e\sin\nu & \frac{2a^{2}}{h}\,\frac{p}{r}\\[2pt]
\frac{p\sin\nu}{h} & \frac{(p+r)\cos\nu + r e}{h}
\end{bmatrix},
\qquad
\hat u^{*} = -\frac{\Phi^{T}Q_z}{\lVert\Phi^{T}Q_z\rVert},
\tag{5}
\end{equation}

so that $\dot Q = Q_z^{T}\Phi\,\hat u^{*}$ is minimized over thrust
directions at the current anomaly. The inertial action is then
$a = u_r\hat r + u_t\hat t$ with $\hat t=\hat h\times\hat r$; the
out-of-plane component is exactly zero (verified to $3\times10^{-8}$ in
float64, harness A3).

**Effectivity coasting (AAS 04-278 mechanism).** The $Q$-decrease rate
achievable at the current anomaly is compared with the best the current
orbit could deliver at any anomaly,

\begin{equation}
\eta = \frac{\lVert\Phi(\nu)^{T}Q_z\rVert}
            {\max_{\theta}\lVert\Phi(\theta)^{T}Q_z\rVert},
\qquad
a = \begin{cases}
\hat u^{*} & \eta \ge \eta_{\min}\\
0 & \eta < \eta_{\min}\ \text{(coast)}
\end{cases}
\tag{6}
\end{equation}

(the common thrust-acceleration factor $F_{\max}/m$ cancels in the
ratio, so $m$ does not enter the gate). $\eta_{\min}$ is a fixed,
conventional threshold — no learning, no scheduling.

## 4. Parameters

| Parameter | Value | Source |
|---|---|---|
| $\mu$ | $3.986004418\times10^{14}$ m³/s² | `env.mu` (WGS84) |
| $F_{\max}$ | 0.01 N per thruster pair | `env.f_max` |
| $I_{sp}$, $g_0$ | 4000 s, 9.80665 m/s² | `env.isp`, `env.g0` |
| $m_0$ | 100 kg | `env.mass0` |
| reference orbit | $r_{p0}=8.5\times10^{6}$ m, $r_{a0}=1.15\times10^{7}$ m | `env.sma0`, `env.ecc0` |
| $r_{p,\mathrm{ref}}$ | $r_{p0}+$ `RP_RAISE` (tuned, frozen) | Eq. (4) |
| $r_{a,\mathrm{ref}}$ | $r_{a0}$ (frozen) | Eq. (4) |
| $W_a$, $W_e$ | 1, `W_E` (tuned, frozen) | Eq. (1) |
| $(m,n,r)$ of $S_a$ | (3, 4, 2) | Petropoulos nominal |
| $\eta_{\min}$ | `ETA_MIN` (tuned, frozen) | Eq. (6) |
| $N_\theta$ ($\dot z_{xx}$ grid) | 2048 | implementation |
| control step / horizon | $\Delta t = 300$ s, 166 steps ($\approx 5$ revolutions) | `env.dt`, `env.max_steps` |

Tuned values are selected on the independent tuning seeds 13/21 by
`phase_b()` of the harness (selection rule: max mean terminal R,
tie-break $|\Delta r_a|$, then propellant) and frozen as class constants
before the official seed-7 evaluation; see §6 for the selected values.

## 5. Implementation and registration

- Controller: `algorithms/baselines/baselines.py`, class
  `TransferQLawBaseline` (`name = 'q_law_petropoulos'`), registry entry
  `BASELINE_VARIANTS['transfer']['q_law']`; built via
  `build_baseline('transfer', env, variant='q_law')`.
- Evaluation: `scripts/baseline_eval.py transfer --baseline q_law`
  (first episode per replica; the shipped artifact is
  `results/transfer/baselines/q_law_petropoulos_seed7.json`, 256 replicas,
  seed 7, 166 steps, with the full controller `config()`). The
  `--baseline` key (`q_law`) and the class `name`
  (`q_law_petropoulos`) differ; `baseline_eval.py` names the output file
  from the class `name`, i.e. `baseline_q_law_petropoulos.json` in the
  default output directory. **That 1024-replica harness file is not
  shipped**; only the 256-replica dataset-B JSON is in the checkout.
- Verification / tuning / comparison harness:
  `environments/satellite/orbit_transfer/eval_transfer_qlaw.py`
  (self-contained, no CLI; `OFFICIAL_ENVS = 1024`, `TUNE_ENVS = 512`):

  - `phase_a()` — small-thrust correctness (7 checks, all at or near
    machine precision unless noted):
    A1 thrust-induced element rates $\dot a,\dot e$ of the emitted
    inertial action vs the exact Jacobian contraction
    $(F_{\max}/m)\,\nabla_v(a,e)\cdot u$ (an algebraic identity, since
    gravity has zero instantaneous effect on osculating elements);
    A2 $\nabla Q$ vs central FD of $Q(a,e)$;
    A3 frame invariants (out-of-plane $\approx 0$, unit norm, per-axis
    bound);
    A4 apogee burn is pure prograde transverse ($u_t>0.99$,
    $u_r\approx0$) and a 1%-throttle one-step probe reproduces the
    sign of $\Delta r_p,\Delta r_a$ on 100% of replicas (median
    magnitude error $\approx 5$%, first-order prediction vs 300 s
    nonlinear step);
    A5 $Q$ never increases along a burn;
    A6 grid $\dot a_{xx}$ vs the analytic Eq. (3).
  - `phase_b()` — tuning grid over (`RP_RAISE`, $W_e$, $\eta_{\min}$)
    on seeds 13/21 (512 replicas per cell, 27 cells).
  - `phase_c()` — official-protocol evaluation of the frozen law
    (1024 replicas by default).
  - `phase_d()` — three-way comparison Gauss / Q-law / PPO
    with per-arm success rate, terminal R, perigee gain, apogee drift,
    propellant, $\Delta v$ and burn duty.

  The harness writes `results/transfer/qlaw_report.json` and
  `results/transfer/comparison.json`; **neither file is shipped**, so the
  phase-B/phase-D numbers below are run-time records, not re-verifiable
  artifacts.

## 6. Measured performance

**Small-thrust verification (`phase_a()`, all 7 checks pass).**
A1 element-rate agreement $4.5\times10^{-16}$ ($\dot a$) /
$2.0\times10^{-15}$ ($\dot e$) against the exact Jacobian contraction;
A2 $\nabla Q$ partials vs central FD: $1.6\times10^{-11}$ /
$3.3\times10^{-11}$; A3 max out-of-plane component
$3.4\times10^{-8}$, max unit-norm deviation $3.7\times10^{-8}$; A4
apogee burn exactly prograde transverse ($u_t=1$, $u_r=-0$), perigee
exactly retrograde transverse, and the 1%-throttle one-step probe gives
100% sign agreement on both $\Delta r_p$ and $\Delta r_a$ (median
magnitude error 5.3% / 6.0%, first-order prediction vs the 300 s
nonlinear step); A5 zero $Q$-increasing steps along the burn; A6 grid
$\dot a_{xx}$ vs Eq. (3): $3.7\times10^{-16}$.

**Tuning (`phase_b()`, seeds 13/21, 512 envs, 27-point grid; harness record, not shipped).**
Selection rule: max mean terminal R, tie-break $|\Delta r_a|$, then
propellant. Selected $(\texttt{RP\_RAISE}, W_e, \eta_{\min}) =
(20\,\mathrm{km}, 1.0, 0.1)$ with $R=1.1510$, 100% success,
$\Delta r_p=15.52$ km, $|\Delta r_a|=0.17$ km, $0.0192$ kg. The grid is
well-behaved: `RP_RAISE`$=10$ km caps $\Delta r_p$ at the 10 km target
(R $\approx0.62$); $40$ km leaves $\Delta r_p$ unchanged (the 5-rev
horizon is the binding constraint) but large $W_e$ then lets
$|\Delta r_a|$ drift to 8-11 km (R $\le 0.12$); $\eta_{\min}=0.3$
starves the burn (duty 0.74, $\Delta r_p$ short of target). The tuned
values coincide with the shipped class constants — no further change.

**Shipped dataset B (seed 7, 256 replicas, one episode per replica).**
`results/transfer/baselines/gauss_tangential_seed7.json` and
`results/transfer/baselines/q_law_petropoulos_seed7.json`, both
`complete = true`:

| Arm | success | mean terminal R | $\Delta r_p$ | $|\Delta r_a|$ | propellant |
|---|---|---:|---:|---:|---:|
| Gauss greedy | 100% (256/256) | 0.0074 | 0.134 km | 0.0375 km | $1.17\times10^{-4}$ kg |
| **Q-law (this)** | **100% (256/256)** | **1.1523** | **15.524 km** | **0.161 km** | 0.0192 kg |

Both rows satisfy the task-level success criterion `R > 0`; the mean
episode return equals the terminal `R` because the environment's reward
is the exact potential-based telescoping of that terminal objective.
Against the tuning-seed record (R 1.1510) the seed-7 score 1.1523 shows
no tuning-set overfit. The two arms are not close in physical terms:
Q-law raises the perigee by ~116× more than the Gauss greedy law, at
~164× the propellant and ~4.3× the apocenter drift. The effectivity gate
keeps the Q-law duty near one (the harness recorded 0.985) because
near-apogee transverse thrust is almost always effective for this
objective, whereas the Gauss greedy rule prices propellant explicitly and
coasts whenever a burn's one-step gain does not clear the propellant
price — which is almost always at the 10 mN reference thrust. The
harness-side $\Delta v$ figure quoted in earlier drafts (4.91 m/s) is
inconsistent with the shipped propellant mass; the rocket-equation value
for 0.0192 kg at $I_{sp}=4000$ s is ≈7.5 m/s, but neither figure is a
shipped artifact and neither is used as a benchmark number here.

**Comparison with the learned arm (PPO):** `results/transfer/comparison.json`
and `qlaw_report.json` are **not shipped**, and no PPO checkpoint for
transfer exists in the checkout. The 32.8 M-transition budget
(1024 envs × 1000 updates × 32 steps/env) is the equal-interaction target
from `docs/evaluation_protocol.md`; it is not a completed run and no
comparison number is claimed here.

## 7. Known limitations

- Q-law is a feedback law, not a trajectory optimizer: the five-revolution
  horizon is long enough for the law to converge toward $(a_T,e_T)$ but
  no burn-arc scheduling beyond the effectivity gate is attempted.
- The $(a,e)$ planar subset cannot represent out-of-plane objectives;
  the task requires none.
- With the apogee reference frozen at $r_{a0}$, the $e$-error term
  penalizes apogee drift symmetrically in both directions; it is a
  drift bound, not an apogee-raising capability.
- The grid-maximum $\dot z_{xx}$ and the autodiff partials through the
  maximum are exact almost everywhere (envelope theorem) and are
  FD-verified at the harness points; at the (measure-zero) anomaly where
  the grid maximizer jumps, the partial is the one-sided limit of the
  winning branch.

## 8. References

1. Petropoulos, A. E. "Low-Thrust Orbit Transfers Using Candidate
   Lyapunov Functions with a Mechanism for Coasting," AAS/AIAA
   Spaceflight Mechanics Meeting, AAS 04-278, 2004 — $Q$, $\dot z_{xx}$,
   $\hat u^{*}=-\Phi^{T}Q_z/\lVert\Phi^{T}Q_z\rVert$, effectivity
   coasting, $S_a$ with $(m,n,r)=(3,4,2)$.
2. Petropoulos, A. E. "Refinements to the Q-law for Low-Thrust Orbit
   Transfers," AAS/AIAA Spaceflight Mechanics Meeting, AAS 05-130, 2005
   — penalty/scaling refinements.
3. Kumar, S. et al. "Closed-Loop Control Law for Low Thrust Orbit
   Transfer with Guaranteed Stability" (arXiv:2511.23014) — modern
   restatement of the $Q$ structure, Gauss matrix $\Phi$ and the
   $\dot z_{xx}$ double maximum used to cross-check the implementation.
4. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —
   osculating-element recovery; Battin, R. H. *An Introduction to the
   Mathematics and Methods of Astrodynamics* (AIAA) — Gauss planetary
   equations.
5. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital
   Transfer using Dynamics-Agnostic Reinforcement Learning"
   (arXiv:2211.08272) — task definition.
