# Satellite

Spacecraft tasks on explicit tensor dynamics (no physics engine). All three
subtasks share the batched `step`/`reset`/buffer contract of the other
categories and are grouped here by mission phase.

## Tasks

| Subdirectory | Task (id) | Dynamics | Episode |
|---|---|---|---|
| [`attitude_control/`](attitude_control/) | Satellite CMG attitude control (`satellite`) | rigid body + 4/6-CMG cluster, RK4 2.5 ms | 1200 steps = 30 s |
| [`orbit_transfer/`](orbit_transfer/) | Low-thrust orbit raising (`transfer`) | two-body gravity + 3-axis 10 mN EP, RK4 10 s | 166 steps ≈ 5 revs |
| [`rendezvous_docking/`](rendezvous_docking/) | CWH docking (`docking`) + rendezvous (`rendezvous`) | Clohessy-Wiltshire-Hill, symplectic Euler 1 s | 500 / 1000 steps |

Each subdirectory holds the environment (`env.py`, full task spec in the
package docstring), the self-contained training entry (`train_<task>.py`,
constants in-file, no CLI arguments) and task-specific deep-dive harnesses
(`eval_docking_lqr.py`, `eval_rendezvous_baseline.py`).

## Run

From the repository root:

```bash
python environments/satellite/attitude_control/train_satellite.py   # -> results/satellite/
python environments/satellite/orbit_transfer/train_transfer.py      # -> results/transfer/
python environments/satellite/rendezvous_docking/train_docking.py   # -> results/docking/
python environments/satellite/rendezvous_docking/train_rendezvous.py  # -> results/rendezvous/

python environments/satellite/rendezvous_docking/eval_docking_lqr.py        # LQR deep dive
python environments/satellite/rendezvous_docking/eval_rendezvous_baseline.py  # ZEM/PN deep dive

python scripts/eval.py satellite        # deterministic checkpoint evaluation
python scripts/baseline_eval.py satellite  # classical baseline; also
                                       #   transfer | docking | rendezvous
```
