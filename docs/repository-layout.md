# Repository Layout

```text
environments/               # everything that defines a task
├── multirotor/             # QuadX-Waypoints + QuadX-Track
├── fixedwing/              # Fixedwing-Waypoints
├── vtol/                   # Quadplane-VTOL
├── rocket/                 # Rocket-Landing
├── missile/                # Missile-Intercept
├── satellite/              # attitude_control, orbit_transfer, rendezvous_docking
└── common/                 # plant, dynamics, aero, contacts, disturbances
algorithms/                 # everything that produces actions
├── train_utils.py          # shared run_training() used by every entry
├── ppo/                    # rsl-rl PPO config builder
├── warmstart/              # behavior-cloning warm start from baselines
├── residual/               # residual-RL action wrapper
└── baselines/              # classical GNC controllers, one per task
experiments/                # controller-prior studies (BC, residual, gain scheduling)
scripts/                    # generic train / eval / baseline / sweep / summarize tools
docs/                       # method-level write-ups, protocol, technical highlights
showcase/                   # replay clips used by the GitHub Pages landing page
tests/                      # unit, env-contract and benchmark tests
```

Per-task deep-dive harnesses live next to their environments (for example
`environments/missile/eval_missile_scenarios.py`,
`environments/rocket/rocket_pdg_test.py`,
`environments/satellite/rendezvous_docking/eval_docking_lqr.py`).

## Training entries

Training entries are self-contained: every parameter is hard-coded in the file,
there are no CLI arguments. Edit the constants and run:

```bash
python environments/multirotor/train_quadx.py
python environments/multirotor/train_quadx_track.py
python environments/fixedwing/train_fixedwing.py
python environments/vtol/train_quadplane.py
python environments/rocket/train_rocket.py
python environments/missile/train_missile.py
python environments/satellite/attitude_control/train_satellite.py
python environments/satellite/orbit_transfer/train_transfer.py
python environments/satellite/rendezvous_docking/train_docking.py
python environments/satellite/rendezvous_docking/train_rendezvous.py
```

Evaluate a checkpoint, or the classical baseline:

```bash
python scripts/eval.py quadx                # restores results/<task>/config.json
python scripts/eval.py quadplane
python scripts/baseline_eval.py quadx       # any registered task
python scripts/baseline_eval.py satellite --baseline quat_pd_mp_pinv_cmg
python scripts/aggregate_5seed.py           # multi-seed protocol summary
```

The legacy CLI aggregator still works for quick sweeps
(`python scripts/train.py quadx`; it registers quadx, fixedwing, rocket,
missile, satellite, transfer, docking, rendezvous — `quadx_track` and
`quadplane` use their per-task entries above).

## Run artifacts

Each run writes `results/<task>/`: `run.json` (config, real transition counts,
physical-step counts, device-side completion metrics), `config.json`,
`checkpoint.pt`, plus `eval.json` when evaluated. Baseline artifacts are
`baseline.json` (default controller) and `baselines/<controller>_seed7.json`
(within-task ablations).
