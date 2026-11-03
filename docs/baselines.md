# Classical Baseline Results

Every task ships a non-learned GNC baseline
([algorithms/baselines/](../algorithms/baselines/baselines.py)) on the same
environment, reward and success bar as the policy. Controllers read only the
`'policy'` observation group, take env constants at construction, and never
touch internal state buffers.

| Task | Baseline | Reference success |
|---|---|---|
| QuadX-Waypoints | SE(3) geometric control + integral trim | 90.6% (32 replicas, seed 7); 99.7% mean over the multi-seed protocol (128 replicas × seeds 7/13/21; held-out seed 101 = 100%) |
| QuadX-Waypoints | position PID → attitude cascade | 99.2% (256 replicas, seed 7, 0 crash / OOB) |
| QuadX-Track | CV Kalman filter + fixed-offset SE(3) | 82.8% (128 replicas, seed 21); 90.2% visible time, 0.258 m standoff |
| Fixedwing-Waypoints | L1 lateral + TECS longitudinal autopilot | 97.0% (128 replicas, seed 13; the window ends before the 120 s limit) |
| Quadplane-VTOL | SE(3) + forward pusher | 89.1% (128 replicas, seed 13) |
| Rocket-Landing | PDG: ZEM/PN braking + virtual-pad manifold + pulse servo | 100% under the v2 criterion (40 Hz, continuously settled for 2 s) |
| Satellite (CMG) | quaternion PD + Schaub–Junkins damped pseudo-inverse | 100% (128 replicas, seed 13) |
| Transfer | Gauss one-step greedy tangential guidance | 100% (128 replicas, seed 13) |
| Docking | CWH discrete LQR + shield/fuel derating | 100% (128 replicas, seed 13) |
| Rendezvous | two-impulse CWH ZEM/PN, recursive re-solve | 100% (128 replicas, seed 13) |
| Missile-Intercept | augmented PN (N = 15) → body-rate commands | 71.1% (128 replicas, seed 13) |

Numbers are the shipped `results/<task>/baseline.json` artifacts; each file
records its own replica count, seed and step window. Within-task ablations
(256 replicas, seed 7) sit in `results/<task>/baselines/`, and the multi-seed
protocol reanalysis is in
[baseline_test_report.md](baseline_test_report.md).

Two readings worth keeping:

- These are references, not strawmen. The fixed-wing and rocket baselines keep
  documented residual failures (endgame limit cycles, authority-saturated
  braking) that a learned policy is expected to attack.
- The rocket v1 figures in older notes (82.8% / 84.1%) are **instantaneous**
  success rates and are not comparable to the v2 settled criterion. Background
  and reproduction commands: [rocket_benchmark_v2_zh.md](rocket_benchmark_v2_zh.md).

Success rates are reported per task and per controller. They do not define a
cross-task ranking: task difficulty, observation budgets and success criteria
differ, and the numbers come from different replica counts and seeds.
