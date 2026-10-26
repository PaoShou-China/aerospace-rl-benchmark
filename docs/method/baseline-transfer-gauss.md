# Low-Thrust Transfer Baseline: One-Step Greedy Gauss Tangential Guidance (`TransferGaussBaseline`)

## 1. Overview

The classical baseline for the low-thrust perigee-raising task is a
one-step greedy guidance law built on the Gauss variational equations.
Each control step, the osculating orbit elements are recovered from the
observed Cartesian state, the first-order per-step changes of the
perigee and apogee radii under a saturated *transverse* thrust are formed
analytically, and the controller chooses among three candidates — coast,
prograde transverse burn, retrograde transverse burn — by maximizing the
one-step gain of the task's reference terminal objective, net of the
propellant cost. A prograde transverse burn near apogee raises $r_p$ at
(first-order) fixed $r_a$ — the classical perigee-raise burn — while near
perigee the same thrust mostly lifts $r_a$, so the greedy rule naturally
concentrates thrust around apogee and occasionally fires *retrograde* to
repair an apogee overshoot (documented intended behavior).

## 2. Interface contract

- **Observations.** Only the `'policy'` group: the raw, unnormalized
  8-dimensional state $o=[t,\ r,\ v,\ m]$ (elapsed time, GCRF position,
  velocity, mass). The controller consumes `o[:, 1:4]`, `o[:, 4:7]`,
  `o[:, 7:8]`.
- **Environment constants at construction.** $\mu$ (`env.mu`), the
  control period $\Delta t=300$ s (`env.dt`), the per-axis thrust limit
  $F_{\max}=10$ mN (`env.f_max`), $I_{sp}=4000$ s, $g_0=9.80665$ m/s²,
  and the reference apsis radii of the spawn orbit,
  $r_{p0}=a_0(1-e_0)$, $r_{a0}=a_0(1+e_0)$ with $a_0=10^{7}$ m,
  $e_0=0.15$ (`env.sma0`, `env.ecc0`; the env re-samples only the mean
  anomaly at reset, so the reference radii are constant).
- **Per-episode state.** None; `reset_idx` is the base-class no-op. The
  law is a pure state feedback.
- **Action.** $a\in[-1,1]^3$, mapped by the env to per-axis thrust
  $F = F_{\max}\,a$ with mass flow
  $\dot m = -F_{\max}\lVert a\rVert_1/(I_{sp}g_0)$. The controller emits
  $a = s\,\hat t$ with $s\in\{-1,0,+1\}$ and $\hat t$ the transverse unit
  vector of Eq. (2), so only the coast/burn polarity is decided; the
  direction is fixed by the osculating geometry.

## 3. Control law

**Osculating elements from the observed state.** With $r=\lVert r\rVert$,
\begin{equation}
h = r\times v,\quad
a = \Big(\tfrac{2}{r}-\tfrac{\lVert v\rVert^{2}}{\mu}\Big)^{-1},\quad
\mathbf{e} = \frac{v\times h}{\mu}-\frac{r}{r},\quad
e=\lVert\mathbf{e}\rVert,\quad
p = \frac{\lVert h\rVert^{2}}{\mu},\quad
v_r = \frac{v\cdot r}{r}.
\tag{1}
\end{equation}
The true anomaly is recovered from the $O(e)$ pair
$(e\sin\nu,\,e\cos\nu) = (h v_r/\mu,\ p/r - 1)$ through
\begin{equation}
\nu = \operatorname{atan2}\!\Big(\frac{h\,v_r}{\mu},\ \frac{p}{r}-1\Big),
\tag{2}
\end{equation}
which divides by nothing: as $e\to0$ the pair vanishes together, so
$\sin\nu,\cos\nu$ stay bounded and mutually consistent (near-circular
numerical stability). The transverse burn direction is
$\hat t = \hat h\times\hat r$.

**One-step element changes (Gauss rates).** For a saturated transverse
burn, $S_{\Delta} = (F_{\max}/m)\,\Delta t$ is the per-step velocity
impulse per unit mass, and the Gauss variational equations give the
first-order per-step changes
\begin{align}
\Delta a &= \frac{2a^{2}}{h}\,\frac{p}{r}\,S_{\Delta},
\tag{3a}\\
\Delta e &= \frac{(p+r)\cos\nu + r e}{h}\,S_{\Delta},
\tag{3b}\\
\Delta r_p &= (1-e)\,\Delta a - a\,\Delta e,\qquad
\Delta r_a = (1+e)\,\Delta a + a\,\Delta e,
\tag{3c}\\
\Delta m &= \frac{F_{\max}\,\lVert\hat t\rVert_1}{I_{sp}\,g_0}\,\Delta t ,
\tag{3d}
\end{align}
with $\Delta m$ the propellant consumed by the three orthogonal thruster
pairs (the env's $\lVert a\rVert_1$ mass-flow convention applied to the
unit vector $\hat t$), evaluated on the observed mass.

**Greedy decision.** The reference terminal objective of the task is
\begin{equation}
J \;=\; (r_p - r_{p0}) \;-\; |r_a - r_{a0}| \;-\; 2\times10^{5}\,(m_0 - m),
\tag{4}
\end{equation}
(the unscaled twin of the env's terminal reward
$R = 10^{-4}(-|\Delta r_a| + \Delta r_p + 2\times10^{5}\,\Delta m)$, the
paper's Table I weights). Writing $d_0 = |r_a - r_{a0}|$, the one-step
gains of the two burn polarities, including the $|r_a-r_{a0}|$ kink
exactly, are
\begin{align}
G_{+} &= \phantom{-}\Delta r_p - \big(|r_a + \Delta r_a - r_{a0}| - d_0\big) - 2\times10^{5}\,\Delta m,
\tag{5a}\\
G_{-} &= -\Delta r_p - \big(|r_a - \Delta r_a - r_{a0}| - d_0\big) - 2\times10^{5}\,\Delta m,
\tag{5b}
\end{align}
and the decision is the argmax over $\{0,\,G_{+},\,G_{-}\}$ (coast,
prograde, retrograde):
\begin{equation}
a \;=\; s\,\hat t,\qquad
s = \begin{cases}
+1 & G_+ > \max(0, G_-)\\
-1 & G_- > \max(0, G_+)\\
0 & \text{otherwise (coast)} .
\end{cases}
\tag{6}
\end{equation}
A burn is therefore taken only when its predicted objective gain exceeds
the propellant cost $2\times10^{5}\,\Delta m$; when $r_a > r_{a0}$ the
kink term can make the retrograde candidate win, producing the documented
apogee-repair burns. An optional constructor argument `target_rp` [m]
forces coast once $a(1-e)\ge$ `target_rp`; the registered configuration
leaves it at `None`.

## 4. Parameters

| Parameter | Value | Source |
|---|---|---|
| $\mu$ | $3.986004418\times10^{14}$ m³/s² | `env.mu` (WGS84) |
| $F_{\max}$ | 0.01 N per thruster pair | `env.f_max` |
| $I_{sp}$, $g_0$ | 4000 s, 9.80665 m/s² | `env.isp`, `env.g0` |
| $m_0$ | 100 kg | `env.mass0` |
| reference orbit | $a_0 = 10^7$ m, $e_0 = 0.15$ $\Rightarrow$ $r_{p0}=8.5\times10^{6}$ m, $r_{a0}=1.15\times10^{7}$ m | `env.sma0`, `env.ecc0` |
| propellant weight | $2\times10^{5}$ m/kg (objective $J$ weight) | `prop_cost` |
| control step / horizon | $\Delta t = 300$ s, 166 steps ($\approx 5$ revolutions) | `env.dt`, `env.max_steps` |
| `target_rp` | `None` (disabled) | constructor default |

## 5. Measured performance

Under the official `baseline_eval` protocol (1024 parallel envs, seed 7),
the root README's frozen baseline table reports **100% success** (the
task-level criterion $R>0$ at the final orbit) with **terminal
$R \approx 0.86$**. The frozen controller audit
(`docs/evaluation_protocol.md`, §2) records the law as fully successful
and frozen as the behavior-cloning data source for this task.

## 6. Known limitations

- The rule is one-step greedy with respect to $J$: it is not a
  global/multi-revolution optimizer and carries no model of future burn
  windows beyond the linearized single-step prediction of Eqs. (3).
- The linearization assumes small per-step element changes; at the
  10 mN reference thrust this is uniformly satisfied, but the law has no
  correction loop for integration error over a step.
- The retrograde branch repairs apogee overshoot only through the
  $|r_a - r_{a0}|$ kink; nothing in the law enforces a hard apogee
  constraint.
- Near-circular degeneration is handled analytically (Eq. 2) but not
  eliminated: as $e\to0$ the apsidal direction becomes unresolvable below
  the float32 observation-noise floor and the gate degrades to bounded,
  objective-anchored chatter rather than divergence. The task's
  eccentricity spans $[0.02, 0.1]$, five orders of magnitude above that
  floor (implementation docstring).

## 7. Implementation

- Controller: `algorithms/baselines/baselines.py`, class
  `TransferGaussBaseline` (registry name `gauss_tangential`,
  `BUILDERS['transfer']`).
- Environment: `environments/satellite/orbit_transfer/env.py`, class
  `LowThrustTransferEnv` (`conventional_controller = TransferGaussBaseline`).
- Evaluation: `scripts/baseline_eval.py transfer`.

## 8. References

1. Vallado, D. A. *Fundamentals of Astrodynamics and Applications* —
   osculating-element recovery and the Gauss form of the variational
   equations.
2. Battin, R. H. *An Introduction to the Mathematics and Methods of
   Astrodynamics* (AIAA) — Gauss planetary equations.
3. Casas, R., Carro, A. & Sanchez-Esguevillas, J. "Low-Thrust Orbital
   Transfer using Dynamics-Agnostic Reinforcement Learning"
   (arXiv:2211.08272, IEEE) and the companion `PerigeeRaisingEnv` — the
   task definition and the Table I objective weights reproduced here.
