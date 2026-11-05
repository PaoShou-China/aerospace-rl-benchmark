"""Immutable deterministic policy snapshots with optional CUDA graph replay."""
from __future__ import annotations

import torch

from algorithms.performance import TensorGraph


class InferencePolicy:
    """Freeze an RSL MLP for serving without changing its weights or precision.

    Construct after loading a checkpoint and calling actor.eval(). The snapshot
    is independent of subsequent training/checkpoint loads on the source actor.
    Call with the original observation mapping or a pre-concatenated tensor.
    Returned actions own their storage and remain valid after the next call.
    One instance serves serial Python calls; successive CUDA streams are ordered.
    CPU uses eager execution, and new CUDA batch shapes warm up and recapture.
    """

    def __init__(self, actor):
        from rsl_rl.models import MLPModel

        if type(actor) is not MLPModel or actor.is_recurrent:
            raise TypeError("InferencePolicy requires an unextended RSL MLPModel")
        if any(module.training for module in actor.modules()):
            raise ValueError("call actor.eval() before creating an inference snapshot")
        self.obs_groups = tuple(actor.obs_groups)
        # Use the library's own deterministic export, including its normalizer
        # and Beta/Gaussian output transform. No checkpoint weights are folded.
        self._model = actor.as_jit().eval().requires_grad_(False)
        param = next(self._model.parameters())
        self.device, self.dtype = param.device, param.dtype
        self.backend = "cuda_graph" if self.device.type == "cuda" else "eager"
        self._graph = TensorGraph(self._model)
        self._last_stream = None
        self._signature = None

    @torch.no_grad()
    def __call__(self, observations):
        if torch.is_tensor(observations):
            features = observations
        elif len(self.obs_groups) == 1:
            features = observations[self.obs_groups[0]]
        else:
            features = torch.cat([observations[key] for key in self.obs_groups], dim=-1)
        if features.device != self.device or features.dtype != self.dtype:
            raise ValueError("observations must match the snapshot device and dtype")
        if self.device.type != "cuda":
            return self._model(features)
        current = torch.cuda.current_stream(self.device)
        if self._last_stream is not None and current != self._last_stream:
            current.wait_stream(self._last_stream)
        signature = (tuple(features.shape), features.dtype, features.device)
        if signature != self._signature and self._last_stream is not None:
            # A shape change retires the preceding graph's borrowed buffers.
            self._last_stream.synchronize()
        self._signature = signature
        self._last_stream = current
        return self._graph(features).clone()
