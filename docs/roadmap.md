# Roadmap

Every future task follows the same recipe: tensorized env + spec docstring →
classical baseline registered in `algorithms/baselines/` → self-contained
`train_<task>.py` → `results/<task>/` → README update. All tasks are
single-agent.

## Priority 1

| Task | Environment | Baseline | Interest |
|---|---|---|---|
| Droneship Landing | Rocket-Landing + 6-DOF moving pad (sea-wave spectrum) | PDG + lag-compensated retargeting | moving touchdown target |
| Docking-Fault | Docking + stuck/dead thrusters + measurement noise | LQR + control reallocation | fault injection, safe RL |
| Autoland | glide slope + crosswind + flare, runway-contact criteria | glide-slope guidance + TECS | autopilot already exists |
| Slung Payload | QuadX + cable-suspended pendulum load | SE(3) + swing damping | underactuated coupling |

## Priority 2

| Task | Environment | Baseline | Interest |
|---|---|---|---|
| Terrain Following | DEM sampling, radar-altimeter observations | TF-lateral guidance | local observations |
| Inspection | CWH + field-of-view constraint, coverage reward | ZEM/PN fly-around | long horizon |
| RTB Boost-back | ascent + 180° flip + landing, ~90 s | gravity turn + two-phase PDG | long-horizon shaping |
| HGV Glide | hypersonic glide, heat-flux / load-factor corridor | BTT corridor guidance | constraint satisfaction |
| Racing Gate | gate sequence, time penalty + speed cap | minimum-time tracking | aggressive flight |

## Priority 3

| Task | Environment | Baseline | Interest |
|---|---|---|---|
| GEO Station-keeping | perturbed two-body (J2 + sun/moon), ±0.05° deadband | impulsive Euler–Hill corrections | very low control budget |
| Ascent to Orbit | gravity turn, dynamic-pressure / heat-flux limits | linear-tangent guidance | completes launch → orbit → rendezvous |
| Detumble & Capture | tumbling non-cooperative target, contact dynamics | angular-momentum matching | contact-rich manipulation |
| Precision Landing | moving cart pad + simulated vision latency | MPC-style tracking | latency-robust observation |

Landed infrastructure: the unified `WindDisturbance` in
`environments/common/disturbances.py` and the baseline registry
(`@register_baseline('<task>')` — one line wires a new controller into
`scripts/baseline_eval.py`).
