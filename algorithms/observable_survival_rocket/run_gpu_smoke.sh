#!/usr/bin/env bash
# Explicit foreground execution only. No credentials, installs or background jobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:?Usage: bash run_gpu_smoke.sh /absolute/path/to/new/output}"
python "$HERE/preflight.py"
python -m pytest "$HERE/tests" -q
python "$HERE/prepare_experiment.py" --repo "$REPO" --out "$OUT"
timeout "${TIME_LIMIT:-45m}" python -u "$OUT/bundle/scripts/run.py" train \
  --config "$OUT/smoke.json" --out "$OUT/run_smoke" --skip-evaluation
printf '\nSmoke training finished; no benchmark improvement is inferred.\n'
printf 'Inspect %s/run_smoke/status.json and logs before longer runs.\n' "$OUT"
