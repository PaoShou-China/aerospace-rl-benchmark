# Checkpoint regression and initialization diagnosis

Date: 2026-10-28. No network weights were updated. All diagnostics use seed 7,
64 replicas, deterministic policy inference, seven ordered gates, and a 20 s limit.
Replicas share a fixed initial state; these are not 64 distinct initial conditions.

## Original start, before environment edits

Position [-5, 4.7, 0.61], identity orientation, zero velocities/motor/PID state.
`original_start.json` contains all 23 checkpoint evaluations, including the
duplicate final checkpoint and model_3999.pt. Iterations below count completed
updates (model_2400.pt contains 2401 completed updates).

| Completed updates | Mean gates passed | Failure |
| --- | --- | --- |
| 2000, 2001 | 1 | Out of bounds |
| 2101 | 0.234375 | Gate collision |
| 2201, 2301 | 1 | Ground contact |
| 2401, 2501 | 0 | Gate collision |
| 2601 through 4000, all saved checkpoints | 0 | Ground/gate collision |

First observed drop: between 2001 and 2101, followed by recovery.
Persistent zero-gate performance at saved checkpoints begins between 2301 and
2401. There are no saved updates inside these intervals, so exact onset is unknown.
This locates regression on this test start, not its causal mechanism.

## New fixed initialization

Full-track evaluation now uses the midpoint between Gate7 and Gate1, applying
the same entry-side projection, altitude floor, collision check and fallback as
training initialization, but with no random position jitter. Target order remains
Gate1 through Gate7. Position is [-1.275, 2.975, 2.365] m. Velocity and angular
velocity are zero; motor/PID states are reset, not hover-trimmed.

During verification we found that the shared quat_from_euler helper computes
0.5*cos(angle) instead of cos(0.5*angle), and likewise for sine. For yaw-only
initialization normalization therefore doubles the intended heading. The shared
file was not modified. splits/env.py now constructs the unit yaw quaternion
directly for both segment training resets and fixed full-track resets.
The new quaternion wxyz is approximately [0.765946, 0, 0, -0.642905].
Existing checkpoints and replay-buffer attitudes were learned/stored under the
old initialization; they are not retroactively corrected.

`segment_start.json` records an intermediate diagnostic retaining the old heading
conversion: both checkpoints timed out with zero gates. It is not the final protocol.
`segment_start_corrected_yaw.json` records the final implemented initialization:

| Model | Mean gates | Completion | Failure | Mean duration |
| --- | --- | --- | --- | --- |
| 2000 updates | 0 | 0% | Gate collision, 100% | 1.275 s |
| 4000 updates | 0 | 0% | Ground contact, 100% | 4.823 s |

Changing initialization alone did not restore successful traversal. No additional
training was run. Future evaluation JSON includes test_initialization metadata.

## Changes for review

- env.py: optional deterministic midpoint in _hover_states; default training
  position jitter retained; locally corrected yaw quaternion; full-track reset
  reuses deterministic segment initialization; evaluation protocol metadata added.
- diagnose_checkpoints.py: read-only checkpoint sweep; writes the requested JSON
  report without copying or replacing model files.
- diagnostics/: raw before/after reports and this explanation.

To evaluate the final two checkpoints again under the current environment:

```powershell
E:\miniconda\envs\arbnhmk\python.exe environments/multirotor/splits/diagnose_checkpoints.py --final-only --report environments/multirotor/splits/diagnostics/recheck.json
```

Omit --final-only to sweep all saved checkpoints. The script uses the current
environment, so it does not reproduce the historical original-start protocol.
