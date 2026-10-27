# Satellite Attitude Ablation: Quaternion PD + Ordinary Moore–Penrose CMG Steering (`SatelliteOrdinaryPinvBaseline`)

## 1. Overview

This baseline is the steering-law ablation of the frozen classical satellite
controller. It keeps the **same** quaternion error, the **same** PD gains
(`kp=3`, `kd=4`), the **same** exact gyroscopic/stored-momentum feed-forward
and the **same** gimbal-rate clamp as
`SatelliteCMGBaseline` (`quat_pd_damped_cmg`). Only the torque-to-gimbal-rate
map changes: the singularity-robust damped inverse is replaced by the
**ordinary Moore–Penrose pseudoinverse** of the 3×N CMG Jacobian.

The purpose is a clean A/B of the steering law. No damping schedule, no
null-space motion, no singularity avoidance / escape, no predictive control
and no reference time scaling is added. When the cluster approaches
singularity the ordinary inverse is allowed to request very large rates;
the env's ±3 rad/s clamp is applied, and the resulting torque shortfall is
measured rather than remedied.

## 2. Reused quaternion-PD path (convention audit)

Both steering laws inherit `_SatelliteCMGPDBase` in
`algorithms/baselines/baselines.py`, so the following are **literally the
same code**, not copies:

- observations: only the `'policy'` group,
  $[q, e, \omega, \sin\theta, \cos\theta, \dot\theta, m, a_{\rm alloc}]$;
- quaternions are $(w,x,y,z)$ Hamilton with body-frame $\omega$ and
  $\dot q=\tfrac12 q\otimes(0,\omega_b)$;
- attitude error $e=\log(q^{-1}\otimes q_t)$ in the body frame, with
  $\dot e\approx-\omega$, hence $\dot\omega_d = +k_p e - k_d\omega$;
- plant torque is $\tau=-h\sum_i\dot\theta_i\,(g_i\times r_i)$, i.e.
  $\tau=-\kappa\,\mathrm{gr}^{\top}\dot\theta$ with
  $\mathrm{gr}_i=\Omega\,(g_i\times r_i)$ and $\kappa=h/\Omega$;
- Jacobian columns are reconstructed from the observed angles as
  $\mathrm{gr}_i=\Omega[\cos\theta_i(g_i\times r_i^0)-\sin\theta_i r_i^0]$,
  bit-exact against `env._manipulability` for 4- and 6-CMG;
- PD torque
  $\tau_{\rm cmd}=I\dot\omega_d+\omega\times(I\omega)+\omega\times h_\Sigma$,
  with $I$ diagonal and $h_\Sigma=h\sum_i r_i(\theta_i)$;
- action $a=\mathrm{clip}(\dot\theta/3,-1,1)\in\mathbb{R}^N$ (float32), so
  $|a_i|=1$ is exactly the ±3 rad/s rate bound.

The audit was re-verified after refactoring: on the same seed and observations
the damped arm's actions and rewards are **bit-identical** to the pre-refactor
implementation.

## 3. Ordinary Moore–Penrose steering

The torque map has the 3×N Jacobian $G$ (columns $g_i\times r_i$). With
$\mathrm{gr}=\Omega G$ and $\kappa=h/\Omega$,

\begin{equation}
\dot\theta \;=\; -\frac{1}{\kappa}\,\operatorname{pinv}\!\big(\mathrm{gr}^{\top}\big)\,\tau_{\rm cmd},
\qquad
\operatorname{pinv}(A)=V\Sigma^{+}U^{\top}
\tag{1}
\end{equation}

where $\Sigma^{+}$ truncates singular values below the numerical rank
threshold. In code this is `torch.linalg.pinv(gr.transpose(1,2), rtol=..., atol=...)`
(`gr` stores the N Jacobian rows), which works for any $N\ge 3$ directly from
the matrix dimensions.

**Numerical rank threshold (recorded in every report):**

| cluster | `rtol` | `atol` | rule |
|---|---|---|---|
| 4-CMG | `4*eps(float64)` = 8.8818e-16 | 0 | LAPACK/`pinv` default `max(m,n)*eps` |
| 6-CMG | `6*eps(float64)` = 1.3323e-15 | 0 | same |

The threshold is relative to the largest singular value, so it does not depend
on the rotor-speed scaling of `gr`. Because the threshold is at machine
precision, genuinely near-singular but full-rank configurations are **not**
truncated: the inverse amplifies the small singular direction by
$\sigma_{\max}/\sigma_{\min}$ and the rate clamp, not the rank threshold,
absorbs it. Exact rank deficiency (e.g. the 4-CMG roof singularity) is
truncated and produces the minimum-norm rate solution.

**No singularity handling.** The implementation deliberately contains no
damping schedule, null-space term, gradient escape, MPC or time scaling.

## 4. Rate limit and actual-torque accounting

Every command is clipped exactly as the env does it,
$a=\mathrm{clip}(\dot\theta/3,-1,1)$, then the delivered rate
$\dot\theta_{\rm applied}=a_{\rm float32}\cdot 3$ is used to compute the
**actual** torque at the pre-step gimbal angles:

\begin{equation}
\tau_{\rm act} = -\kappa\,\mathrm{gr}^{\top}\dot\theta_{\rm applied},
\qquad
\tau_{\rm err}=\tau_{\rm act}-\tau_{\rm cmd}.
\tag{2}
\end{equation}

Reported metrics include the mean/max $\lVert\tau_{\rm err}\rVert$, the error
as a fraction of the cluster's summed peak torque authority $3hN$ (per gimbal
$h\,|g_i\times r_i|\,\dot\theta_{\max}=3h$, since $|g_i\times r_i|=1$ and the
rotor speed inside $\mathrm{gr}$ cancels $\kappa$), the fraction of env-steps
with any gimbal saturated, the fraction of saturated gimbal commands, and the
peak **pre-clamp** requested rate. The same accounting is applied to the RL
arms using the same PD command as the torque reference, so the comparison is
like-for-like. The torque is evaluated at the pre-step angles (the same angle
used by the controller); the plant integrates the rates over the RK4 substeps.

## 5. Test scenarios

Per cluster (`num_cmgs=4` and `6`):

| scenario | initial condition | target |
|---|---|---|
| `random_target` | official reset distribution | uniform random SO(3) |
| `near_singular` | momentum-neutral near-singular gimbal configuration | uniform random SO(3) |
| `singular_180` | exactly singular (4-CMG) / near-singular (6-CMG) | 180° about body x |

The 4-CMG momentum-neutral singular configuration is the roof singularity
$\theta_i=\pi/2-\text{skew}\approx0.61548$ rad for all $i$: all four torque
directions collapse onto $-z$ (rank 1) and $\sum_i r_i=0$ exactly. The
near-singular variant offsets alternate gimbals by ±0.01 rad
($\sigma_{\min}/\sigma_{\max}\approx0.0071$). The 6-CMG near-singular
configuration was found numerically with a momentum-neutral penalty:
$\theta=(-2.177243665,1.378914139,-0.192070305,0.273303902,-1.297523787,-0.606455308)$,
with $\sum_i r_i\approx2\times10^{-4}$ and
$\sigma_{\min}/\sigma_{\max}\approx6.1\times10^{-5}$.

## 6. Results

Source: `results/satellite/steering_compare.{json,md}`, produced by

```bash
python scripts/satellite_steering_compare.py --backend cpu --num-envs 64 \
    --arms damped,mp_pinv,flash_sac --cmgs 4,6
```

64 replicas per arm, seed 7, one episode per replica, 1200-step cap (30 s at
the 0.025 s control period). All arms see the same target stream and the same
shared PD torque reference. `sat envstep` is the fraction of steps with any
gimbal command saturated, `sat gimbal` the fraction of saturated gimbal
commands, and the torque error is the mean instantaneous
$\lVert\tau_{\rm act}-\tau_{\rm cmd}\rVert$ with its fraction of the cluster's
peak authority ($3hN$). Failures are `runaway/timeout`.

### 4-CMG (pyramid, $\kappa=0.2$, peak authority 240 N·m)

| scenario | arm | success | settle steps (mean/p90) | settle s | final att err | sat envstep | sat gimbal | torque err (N·m) | % authority | failures |
|---|---|---|---|---|---|---|---|---|---|---|
| random_target | damped | 1.000 | 246.6 / 298.2 | 6.16 | 0.0195 | 0.344 | 0.244 | 86.02 | 39.5% | 0/0 |
| random_target | mp_pinv | 1.000 | 255.2 / 317.7 | 6.38 | 0.0198 | 0.370 | 0.295 | 99.49 | 46.3% | 0/0 |
| random_target | flash_sac | 0.000 | — | — | 2.2693 | 0.000 | 0.000 | 569.9 | 237.5% | 0/64 |
| near_singular | damped | 1.000 | 249.6 / 295.4 | 6.24 | 0.0199 | 0.346 | 0.267 | 88.46 | 40.4% | 0/0 |
| near_singular | mp_pinv | 1.000 | 262.9 / 321.4 | 6.57 | 0.0199 | 0.385 | 0.314 | 112.8 | 52.0% | 0/0 |
| near_singular | flash_sac | 0.000 | — | — | 2.2448 | 0.000 | 0.000 | 547.3 | 228.0% | 0/64 |
| singular_180 | damped | 0.000 | — | — | 3.1416 | 0.000 | 0.000 | 706.9 | 294.5% | 0/64 |
| singular_180 | mp_pinv | 0.000 | — | — | 3.1416 | 0.000 | 0.000 | 706.9 | 294.5% | 0/64 |
| singular_180 | flash_sac | 0.000 | — | — | 0.8152 | 0.000 | 0.000 | 491.5 | 204.8% | 0/64 |

Peak pre-clamp rate request: damped 115 rad/s, ordinary 3 597 rad/s
(`random_target`) and 5 049 rad/s (`near_singular`). The ±3 rad/s clamp is
what bounds the delivered rate in both arms.

### 6-CMG (orthogonal, $\kappa=6.25\times10^{-5}$, peak authority 0.225 N·m)

| scenario | arm | success | settle steps (mean/p90) | settle s | final att err | sat envstep | sat gimbal | torque err (N·m) | % authority | failures |
|---|---|---|---|---|---|---|---|---|---|---|
| random_target | damped | 1.000 | 233.7 / 283.4 | 5.84 | 0.0199 | 0.267 | 0.190 | 0.0526 | 26.3% | 0/0 |
| random_target | mp_pinv | 1.000 | 234.6 / 282.2 | 5.87 | 0.0198 | 0.272 | 0.211 | 0.0538 | 26.9% | 0/0 |
| near_singular | damped | 1.000 | 235.8 / 279.7 | 5.89 | 0.0199 | 0.274 | 0.203 | 0.0547 | 27.1% | 0/0 |
| near_singular | mp_pinv | 1.000 | 241.8 / 287.0 | 6.05 | 0.0199 | 0.294 | 0.227 | 0.0643 | 31.7% | 0/0 |
| singular_180 | damped | 1.000 | 271.0 / 271.0 | 6.78 | 0.0200 | 0.354 | 0.252 | 0.0790 | 35.1% | 0/0 |
| singular_180 | mp_pinv | 1.000 | 284.0 / 284.0 | 7.10 | 0.0197 | 0.398 | 0.312 | 0.0972 | 43.2% | 0/0 |

The torque-error percentage is normalised by the instantaneous peak authority,
not by the command, so values above 100% (FlashSAC) mean the shared PD
reference itself asks for more than the cluster could ever deliver at that
state; the classical arms stay at 26–52%. Absolute torque is **not**
comparable across the two clusters — the 6-CMG plant is a much smaller
satellite ($I=0.055$ kg·m², per-rotor momentum 0.0125 versus 20 N·m·s).

Peak pre-clamp rate request: damped 153 / 165 / 115 rad/s versus ordinary
556 / 274 694 / 233 086 rad/s for the three scenarios. In the near-singular
cases the ordinary inverse asks for rates $\sim10^3$–$10^5\times$ the clamp;
only the clamp keeps the command finite. The 4-CMG timeouts do not arise here
because the 6-CMG configuration is full rank
($\sigma_{\min}/\sigma_{\max}\approx6.1\times10^{-5}$).

**PPO arm:** no checkpoint exists (`results/satellite_ppo` was removed after
an aborted run under GPU contention), so PPO is reported as unavailable, not
as a number. FlashSAC uses `results/satellite_flashsac/checkpoint.pt`
(1000 vectorized env updates) and was only trained for 4-CMG.

**Rank diagnostics.** The fraction of env-steps below the `pinv` rank
threshold is 0.0 everywhere except the exact 4-CMG roof singularity (1.0,
where $\sigma_{\min}=0$ and the inverse truncates as intended). The ordinary
inverse therefore is *not* rescued by the rank threshold in the near-singular
cases; the amplification is real and is only bounded by the rate clamp.

**Summary.** Both laws solve every non-degenerate 4-CMG and all 6-CMG
scenarios. Replacing damping by the ordinary inverse costs
+3.5 % settling steps and +16 % mean torque error on 4-CMG
`random_target`, +5.3 % / +28 % on 4-CMG `near_singular`, +0.4 % / +2.3 % on
6-CMG `random_target`, +2.6 % / +17.6 % on 6-CMG `near_singular`, and
+4.8 % / +23 % on the 6-CMG `singular_180`. The price shows up as a much
larger pre-clamp rate request and slightly higher saturation, while success is
unchanged. The exact 4-CMG roof singularity is a genuine failure for both
classical arms (the PD command about the missing axes is not in the Jacobian
range), recorded without any escape mechanism. FlashSAC fails all 4-CMG
scenarios (terminal attitude error 2.24–2.27 rad on random/near-singular,
0.82 rad at the 180° start), all by timeout.

## 7. Interpretation and failure cases

- Far from singularities both inverses give the same minimum-norm solution;
  success and terminal attitude error agree exactly. Their mean torque error
  still differs (99.5 vs 86.0 N·m on 4-CMG `random_target`, +16 %) because the
  ordinary arm's rate swings during the singular passes of the manoeuvre are
  cut by the clamp (saturated env-steps 0.370 vs 0.344), whereas the damped
  arm stays closer to the deliverable torque.
- Close to singularity the ordinary inverse requests rates proportional to
  $1/\sigma_{\min}$ (peak pre-clamp request 3.6–5.0 krad/s at 4-CMG, up to
  2.7e5 rad/s at 6-CMG, versus ≤165 rad/s damped). The ±3 rad/s clamp then
  makes the delivered torque deviate from the command, and the reported torque
  error / saturation fraction captures the loss. Success is unaffected in
  every full-rank scenario, at the cost of a few percent more settling steps
  and a slightly higher saturated-gimbal fraction.
- At an exact singularity with a command outside the Jacobian range (the
  180°-about-x target at the 4-CMG roof), both classical laws command zero
  torque about the missing axes and time out after 1200 steps. This is
  recorded as a normal failure; no escape/recovery mechanism is added.
- The shipped FlashSAC schedule (1000 vectorized env steps, 4-CMG only)
  completes no episode within the 1200-step horizon; its failures are
  reported in the comparison, not patched. Its terminal attitude error is
  2.24–2.27 rad from a random start and 0.82 rad from the 180° start, i.e. it
  moves the satellite but does not reach the tolerance.

## 8. Implementation and registration

- Shared PD base + damped arm: `algorithms/baselines/baselines.py`,
  `_SatelliteCMGPDBase`, `SatelliteCMGBaseline`.
- Ordinary arm: `SatelliteOrdinaryPinvBaseline`, registry name
  `quat_pd_mp_pinv_cmg`, registered in `BASELINE_VARIANTS['satellite']`.
- CLI: `scripts/baseline_eval.py` (`--baseline quat_pd_mp_pinv_cmg`,
  `--num-cmgs 4|6`) and `scripts/satellite_steering_compare.py`.
- Reused RL arms: `results/satellite_flashsac/checkpoint.pt` (FlashSAC,
  4-CMG) is loaded through the existing `FlashSACRunner`; the PPO slot
  (`results/satellite_ppo/checkpoint.pt`) is currently empty, so the harness
  reports PPO as unavailable instead of a number. Both are optional: the
  harness auto-skips an arm whose checkpoint is missing or was trained for a
  different CMG count.

## 9. Run instructions

```bash
# single-arm baseline, 4-CMG (default) or 6-CMG
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg --num-cmgs 6

# full steering-law comparison (damped / ordinary / PPO / FlashSAC);
# the CPU variant is what produced section 6 (GPU was busy with 6 trainings)
python scripts/satellite_steering_compare.py
python scripts/satellite_steering_compare.py --backend cpu --num-envs 64 \
    --arms damped,mp_pinv,flash_sac --cmgs 4,6

# original PPO and FlashSAC into separate result directories
python - <<'PY'
from algorithms.train_utils import run_training
from environments.satellite.attitude_control import SatelliteAttitudeEnv
common = dict(num_envs=1024, updates=1000, seed=7, backend='gpu', clean=True,
              learning_rate=3e-4, lr_schedule='fixed', gamma=0.999,
              num_steps_per_env=150, num_mini_batches=5,
              num_learning_epochs=40, entropy_coef=0.0, init_std=1.0,
              clip_param=0.2)
run_training('satellite_ppo', SatelliteAttitudeEnv,
             out_dir='results/satellite_ppo', **common)
run_training('satellite_flashsac', SatelliteAttitudeEnv, algo='flash_sac',
             out_dir='results/satellite_flashsac', **common)
PY
```

The two training calls do not touch `results/satellite/`, so the frozen
baseline/audit artifacts are preserved. `satellite_steering_compare.py`
auto-skips an RL arm when its checkpoint is missing or was trained for a
different CMG count.

## 10. References

1. Moore, E. H. (1920); Penrose, R. (1955) — generalized inverse.
2. Golub, G. H. & Van Loan, C. F. *Matrix Computations* — SVD rank threshold
   and minimum-norm least-squares.
3. Schaub, H. & Junkins, J. L. *Analytical Mechanics of Space Systems* —
   singularity-robust steering law used by the damped comparison arm.
