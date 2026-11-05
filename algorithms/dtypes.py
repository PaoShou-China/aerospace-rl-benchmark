"""Shared hot-path compute dtype.

Kept outside the ``environments`` package so that ``algorithms.baselines`` can
import it without triggering ``environments/__init__`` (which imports the task
environments that in turn import the baselines).
"""

import torch

# Hot-path compute dtype, fixed to float32: consumer GPUs (e.g. RTX 3090)
# run FP64 at 1/64 of FP32 throughput.  Measured 2026-11-04: quadx baseline
# success and episode return match the old fp64 path to ~1e-7.
COMPUTE_DTYPE = torch.float32
