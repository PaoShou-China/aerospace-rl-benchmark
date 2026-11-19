# Phy-AI offline touchdown experiment

This is a historically isolated research prototype, not a live simulator adapter.
It does not register a benchmark policy, alter current environments, or claim
improved landing success. The frozen physics model and its fixed target belong
to the original 2026-10-29 experiment, not the latest main interfaces.

## Run the structural tests

From this directory, using an environment with PyTorch, pytest and NumPy:

```sh
python -m pytest -p no:cacheprovider tests -q
```

`requirements.lock.txt` records the historical Mac environment; it is provenance,
not a cross-platform installation lock. Tests import the local `phyai` and
`frozen` packages via `tests/conftest.py`; do not install them globally.
Replay scripts require a separately supplied historical dataset, which is
intentionally not included. Aggregate reports are historical evidence, not
results obtained on current main. See `README_TRANSFER.md` for architecture,
limitations and source provenance, and `docs/REMAINDER_ERRATUM_ZH.md` for the
corrected uncertainty remainder argument.

## Isolation and evidence

- No weights, raw trajectories, credentials or environment endpoints.
- No new simulation interactions or training updates in this recovery.
- Contact-model gate G0 remains unmet; real-time and closed-loop performance
  are not established.
- `SHA256SUMS.json` verifies the imported source/evidence snapshot. It excludes
  this integration README and the new `reports/cloud_integration_validation.json`.
