# Civilian integration and current-main boundary

Imported from `origin/Yxuan-Gong` commit `a6b1973` onto civilian-sync
`2346cae`. This is a historical research package, not a current-main trained
controller. No new success rate, deployment qualification or generalization
claim is made.

## Frozen reproduction path

Use `scripts/run.py prepare-eval --out <new-directory>` then the documented
frozen evaluator. Core files, `train.py`, `run.py`, seeds, archived episode
records and tensor storage are unchanged. The historical archive hash and
checkpoint file hashes are retained in `civilian_integration.json`.
The bundled archive excludes the unused `original_pyflyt/pz_envs` subtree and its unused reference source manifest;
all retained file payloads have identical hashes. Tar owner metadata is
normalized. Two checkpoints have only absolute training-directory strings
redacted in pickle metadata; tensor/optimizer storage bytes are unchanged.
Thus original whole-file hashes do not apply to the sanitized containers.
`provenance.json` and `FILE_MANIFEST.json` describe the shipped bytes.
The earlier packaging/integration JSON records are historical checks, not
validation results for this integration. Original NOTICE statements about
unchanged checkpoint bytes and a complete upstream archive are superseded
by this explicit integration record.

## Current-main compatibility boundary

Do **not** point the frozen evaluator at the current repository environment.
The frozen import namespace is `common` / `pyflyt_genesis.rocket`; current
main uses `environments.rocket` and `algorithms`. Both expose 30 observations
and 7 actions, but matching shapes do not establish dynamics compatibility.
The frozen evaluator expects a four-value `step` and pre-reset terminal
observations; current main returns the Gymnasium five-value tuple with
separate termination/truncation. Current main also has a different rocket
contact model, sustained success criteria and benchmark-version options.
The learned flight/contact operator and phase heights remain fitted to the
frozen system. An import alias or tuple shim alone would silently mislabel
evidence and is intentionally not supplied.

A future, separately labeled port must select and record benchmark version,
validate observation units/frames and geometry, adapt termination and
pre-reset observations, and recalibrate/retrain the physical operator.
Use current-main `tests/test_rocket_integration.py` and
`tests/test_rocket_benchmark.py` as environment-contract checks, then run new
paired first-episode cohorts under a new report name. Never overwrite
`results/physical_belief_evaluation.json` with that experiment.
The confidence-sequence plan is historical pending research, not evidence
of a currently running job. Observable-survival integration is separate.
