# Opt-in civilian scenario evaluation

Recovered from the local civilian working-tree delta against `151054b`, then
adapted to `2346cae`. The older delta was reconstructed separately rather than
applied wholesale over newer common-plant and landing code.

`environments.make(..., civil_profile=...)` accepts:

- `reference`: no scenario overrides;
- `development` and `hard`: fixed engineering stress hypotheses, not measured
  flight-data calibration or certified operating limits;
- `navigation_outage`: hard conditions plus held policy navigation packets for
  QuadX, orbital rendezvous and orbital transfer only. Transfer's onboard clock
  and propellant accounting remain live. Rewards and physical state are not held.

Profiles override matching explicit environment kwargs; omitted profiles preserve
the normal factory behavior. Wind direction is seeded deterministically and each
call returns independent nested configuration objects. There are nine civilian
profiles and unsupported task/profile combinations fail before construction.

## Evaluation

```
python scripts/eval_civil.py rendezvous --profile hard \
  --checkpoint PATH/model.pt --config PATH/config.json \
  --seed 23 --num-envs 16 --backend cpu --out PATH/report.json
```

The script strictly restores a standard `MLPModel` actor and tracks exactly the
first completed episode of each vector lane. It records model/config/initial-state
hashes, effective overrides, first-episode success and accuracy metrics. Run the
same seed separately for each profile to obtain paired evaluation cohorts.
Nonfinite actions/results and incomplete cohorts fail rather than producing a
success report. A report is not evidence of new training or independent validation
of a recovered checkpoint. Custom archived actor classes and source-package
release loaders are deliberately unsupported in this integration.

## Numerical compatibility

Tracking-camera noise is configurable and defaults to its original value.
Orbital-transfer eclipse thrust scales acceleration and fuel flow together using
an idealized fixed-Sun cylindrical shadow, with eclipse disabled by default.
The recovered sub-ULP propellant accumulation fix also applies without a profile:
it integrates consumed fuel independently before reconstructing wet mass, and
uses that accumulated quantity for fuel cost and reported consumption. This fixes
float32 wet-mass rounding but can change transfer rewards/metrics versus older
runs. Do not compare old transfer checkpoints' returns as if numerically identical.

## Deliberately deferred

No recovered common plant, rocket environment, original-backend, landing-asset or
release-policy changes were copied over the latest baseline. Individual-foot load
feedback, foot-support diagnostics and archived policy loaders need a separate
compatibility review with the full-scale landing integration. Existing optional
rocket metadata is reported only when present; no pad-load capability is implied.

## CPU checks

```
python -m unittest discover -s tests -p 'test_civil_*.py' -v
```

Tests cover deterministic independent configs, invalid input, reference-state
compatibility, eclipse force/fuel scaling, small fuel burns, masked resets,
navigation packet holding, live transfer clock/fuel, and first-episode accuracy.
They do not certify GPU performance or flight realism.
