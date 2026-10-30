# Rocket-Landing Baseline: Conventional PD Landing (`RocketPDLandingBaseline`)

## 1. Overview

The classical baseline for the Rocket-Landing task is a deliberately simple,
conventional cascaded PD landing law: a position/velocity PD generates a
desired acceleration, the desired acceleration is converted to a desired
thrust magnitude and thrust axis, and an attitude/rate PD generates the
torque demand that the existing TVC + grid-fin control-allocation interface
tracks.  There is no optimizer, no learning module, no guidance solve: the
vertical profile is the classic brake-distance/threshold-height law
(`v_ref(h) = -min(V_MAX, K_H * h)`), the engine is a fixed-speed-band
hysteresis around that profile, and ignition is a one-shot braking-distance
edge trigger.  All constants are fixed in the class; tuning happened once in
the dedicated scenario of `environments/rocket/rocket_pd_tune.py` (Section 5),
never per-initial-state by hand.

It is intentionally *not* the PDG baseline (`RocketPDGBaseline`): no
ZEM/ZEV optimal-guidance solve, no aerodynamic feed-forward, no fin/gimbal
force allocation.  The point of the arm is a plain PD yardstick for the
learned policies and for PDG under identical conditions.

## 2. Interface contract

- **Input.** Only the `'policy'` observation group of `RocketLandingEnv`
  (30 dims, float64).  Channels read: body rate `wb` `[0:3]`, attitude
  quaternion xyzw `[3:7]`, body velocity `vb` `[7:10]`, world position `p`
  `[10:13]`, engine state `ignited` `[24]`, fuel `[25]`, throttle state `[26]`,
  gimbal states `[27:29]`, pad-contact flag `[29]`.  World velocity is
  reconstructed as `R(q) @ vb` with `_quat_to_R` (xyzw, scalar-last).
- **Output.** The env's raw 7-dim action convention: `[fin_x, fin_y,
  fin_yaw, ignition, throttle, gimbal_x, gimbal_y]` with the box
  `[-1,-1,-1,0,0,-1,-1]..[1]` (the env affinely maps raw `[-1,1]` onto that
  box; ignition/throttle stay in `[0,1]`).
- **Internal state.** One boolean per replica: `started` (the one-shot brake
  ignition has fired), cleared on episode reset.

## 3. Control law

Per policy step (60 Hz, 2 inner control steps each):

1. **Geometry.** `h = (p_z - REST_Z).clamp_min(0)`, `v_w = R @ vb`,
   descent speed `v_dn = max(0, -v_z)`, pad offset `p_xy`.
2. **Lateral (mid-descent).** `a_xy = -KP_XY * p_xy - KD_XY * v_xy`,
   saturated to `lat_cap = clamp(0.14 * max(f_z, g), 1, 5)` m/s^2 (a tilt
   budget: at the min-thrust manifold 0.14 rad is ~8 deg, inside the ~9.6 deg
   fin-stall knee).  Below the terminal height `H_TERM = 2 m` the position
   term is dropped (a position chase at min thrust is the grow-and-reverse
   leg-skate failure) and replaced by pure velocity damping:
   `a_xy = clamp(-KV_LAT * v_xy, +-A_LAT_PULSE)`.
3. **Vertical.** `v_ref_dn = min(V_MAX, K_H * h)` (0.5 * h profile,
   20 m/s cap), `a_z = KV_Z * (v_ref_dn - v_dn)`, specific force `f_z = a_z + g`.
4. **Engine band.** One-shot edge trigger
   `first = (h <= h_brake + M_ON) & (v_dn > V_BAND) & ~started` with the
   braking distance `h_brake = v_dn^2 / (2 * a_brake)`, `a_brake =
   BRAKE_FRAC * T_MAX / m - g` and the `M_ON = 15 m` margin giving a 5%
   thrust reserve at ignition.  While running, a fixed speed-band hysteresis
   around the profile holds it: cut below `v_ref_dn - V_BAND` (terminal band
   `V_TERM_OFF/ON` under `H_TERM`), relight above `v_ref_dn + V_BAND`
   (terminal `V_TERM_ON`).  The public pad-contact flag is the engine switch
   on the pad (`ign &= ~pad_c`) and there is **no permanent post-touch
   cutoff latch**: the band stays alive through the settle phase, so a
   rebound is caught on its way down (the band IS the rebound catcher).  The
   engine never lights while ascending (`v_dn = max(0, -v_z)`) -- boost not
   catch.
5. **Thrust vector.** `sigma = (a_xy, f_z)`, `T = clamp(m|sigma|, T_MIN,
   T_MAX)`, throttle mapped through the existing convention
   `u_thr = (T/T_MAX - 0.39)/0.61`.
6. **Attitude reference and torque.** The desired thrust axis is built from
   the *specific-thrust vector the engine produces when lit*,
   `s = clamp(|sigma|, T_MIN/m, T_MAX/m)`, with vertical component
   `f_vert = sqrt(s^2 - |a_xy|^2)`, i.e. `sin(tilt) = a_xy / s`.  The torque
   demand is `tau = KP_ATT * e_tilt - KD_ATT * w_xy` (+ `KD_YAW` yaw-rate
   damping), allocated through the plant's existing fixed-gain TVC/grid-fin
   interface (`gx = -tau_gim / k_gim`, `fx/fy = -tau_fin / k_fin`, saturated
   at 0.95 of each channel).  The reference is kept across coast/burn
   transitions (the next light is on the same clamp), so the pulse train
   never chases an upright setpoint it cannot track.

Step 6's reference is the load-bearing detail: referencing `f_z` directly
(`atan(a_xy/f_z)`) leaves the min-thrust clamp to multiply the lateral loop
gain by `s/f_z` -- ~2x overgained in the pulse leg (`s ~ 20.5` vs
`f_z ~ 9.8` specific force), which held up to 8.7 deg of *stale tilt* across
each relight; the tilted min-thrust pulses then skated the legs across the
pad in a grow-and-reverse limit cycle (0% success).  With `sin(tilt) =
a_xy/s` the delivered lateral acceleration equals the command at unit plant
gain everywhere.

## 4. What the probes ruled out (recorded so they stay ruled out)

- **Permanent post-touch engine latch.** A hard `ign &= ~touched` cutoff
  after the first pad contact lands soft touches (~0.46 m/s median) but lets
  the contact solver's self-excited micro-bounce grow unchecked
  (+-0.1 -> +0.79/-0.85 m/s) until the rocket is ejected and re-contacts at
  >1 m/s = fatal.  The live band (relight at `V_TERM_ON` on the way down)
  catches the rebound instead; PDG's 85.5% on the same plant proves
  ejection is not a hard ceiling.
- **Hard cutoff altitude.** An `H_CUT = 0.05 m` engine kill ("do not push
  through the contact") is catastrophic (success 0.008): past the cut the
  rocket free-falls from ~0.05 m and re-contacts at ~0.96 m/s.  `T_MIN`
  burning through the contact step is a *cushion*, not an obstacle.
- **Continuous distance condition.** Making the brake-distance condition the
  *hold* condition (`h <= h_need + M_ON` while on) fights the terminal band
  (its cut threshold sits above the relight threshold -> 1-on-1-off chatter,
  terminal crawl).  It must be an edge trigger; the band owns everything
  after.
- **Position chasing inside the pulse leg.** With `T_MIN` > weight the
  engine can only brake; a position error at min thrust produces a
  grow-and-reverse limit cycle and leg skating (the failure that motivated
  velocity-only damping below `H_TERM`, matching the PDG note).

## 5. Tuning scenario and records

`environments/rocket/rocket_pd_tune.py` (report:
`results/rocket_pd_tune/tune_report.json`): 128 replicas x one episode
(1900 steps) per candidate, calm air, tuning seed 101 (distinct from the
official seed 7), one-factor-at-a-time around the shipped constants plus a
combined-best confirmation.  Ranking: success rate, then fewer >1 m/s
touches, fewer >0.35 rad/s spins, lower median touch speed.  Gain-sweep
records kept from the diagnosis runs (64-env tuning harness, seed 101):

| change | success | note |
| --- | --- | --- |
| tilt ref = `f_z` (pre-fix) | 0.000 | 2x lateral overgain, leg-skate limit cycle |
| tilt ref = `a_xy/s` (fix) | 0.578 | unit lateral gain restored |
| + `KD_ATT` 6000 | 0.500 | no aero FF: loop wants the stronger damping |
| + `KD_ATT` 10000 | 0.453 | overshoots, as in the PDG note |

Full grid results and the transplanted winner are in the report JSON.  Grid
outcome (tuning seed, 128 replicas each): the shipped constants are
near-optimal; every single-knob move is neutral or worse except
`A_LAT_PULSE` 1.5 -> 1.0 (0.484 -> 0.531), which was adopted after official
confirmation.  The stacked combined-best candidate collapses (success
0.008, touch spin > 0.35 in 80% of episodes) -- the terminal-band, pulse and
attitude gains interact destructively, which is exactly what the
confirmation run exists to catch.  Final shipped constants are the class
attributes quoted in Section 3.

## 6. Test records

Official harness `scripts/baseline_eval.py rocket --num-envs 256 --seed 7`
(calm air, one completed episode per replica, results in `results/rocket/`):

| arm | success | mean return | pad-touch rate | touch speed | touch spin | touch tilt |
| --- | --- | --- | --- | --- | --- | --- |
| PDG (`zem_zev_pdg`) | **0.8555** | 4125.71 | 0.617 | 0.374 m/s | 0.121 rad/s | 0.0195 rad |
| PD (`pd_landing`) | **0.8438** | 4063.64 | 0.695 | 0.340 m/s | 0.118 rad/s | 0.0171 rad |

(`task_metrics` in each JSON; touch speed/spin/tilt are means over episodes
whose done-step still has pad contact.)  The shipped `A_LAT_PULSE = 1.0`
came from the tune grid (0.531 vs 0.484 on the tuning seed) and was
confirmed on the official harness (+1.6 pp over 1.5).  The simple PD's
touch quality matches or beats PDG's; the residual ~1 pp success gap is
within the 256-episode noise band.
