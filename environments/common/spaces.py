"""Lightweight action/observation space descriptions (no gymnasium dependency).

Only the subset of the gymnasium ``Box`` contract used by this repo is
implemented: scalar or per-dimension ``low``/``high`` bounds, a flat
``shape``, a floating ``dtype``, and ``sample`` / ``contains``.  The
policy-facing action space is uniformly ``[-1, 1]`` across every task
(``clip_actions == 1`` everywhere); ``low``/``high`` are torch tensors so
they live on the env device and can be sliced per-channel.

This module is the single source of truth for space metadata.  Environments
must not re-declare bounds in ad-hoc attributes; they derive ``action_space``
/ ``observation_space`` from :class:`BaseEnv`.
"""

from __future__ import annotations


import torch


class Box:
    """A possibly-bounded flat tensor box.

    ``low`` and ``high`` may be scalars or one-dimensional tensors of length
    ``shape[0]``; they are broadcast to a per-dimension vector on the given
    device.  Infinite bounds (``-inf`` / ``inf``) are allowed and reproduce
    gymnasium's unbounded-box semantics for observation spaces.
    """

    def __init__(self, low, high, shape, dtype=torch.float32, device="cpu"):
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) != 1:
            raise ValueError(f"Box supports flat (1-D) shapes only, got {shape}")
        n = int(shape[0])
        self._shape = (n,)
        self._dtype = dtype
        lo = torch.as_tensor(low, dtype=dtype, device=device)
        hi = torch.as_tensor(high, dtype=dtype, device=device)
        if lo.numel() == 1:
            lo = lo.expand(n).clone()
        if hi.numel() == 1:
            hi = hi.expand(n).clone()
        if lo.shape != (n,) or hi.shape != (n,):
            raise ValueError(
                f"low/high must be scalar or length {n}, got {tuple(lo.shape)} / {tuple(hi.shape)}"
            )
        self.low = lo
        self.high = hi
        self.device = device

    @property
    def shape(self):
        """Return the flat shape tuple ``(n,)`` of the space."""
        return self._shape

    @property
    def dtype(self):
        """Return the element ``torch.dtype`` of samples from this space."""
        return self._dtype

    def sample(self, batch_shape=(), generator=None):
        """Uniform sample in ``[low, high]``; infinite bounds sample N(0,1)."""
        n = self._shape[0]
        shape = tuple(batch_shape) + (n,)
        lo = self.low
        hi = self.high
        finite = torch.isfinite(lo) & torch.isfinite(hi)
        out = torch.empty(shape, dtype=self._dtype, device=self.device)
        if bool(finite.all()):
            u = torch.rand(shape, dtype=self._dtype, device=self.device, generator=generator)
            out = lo + u * (hi - lo)
        elif bool(finite.any()):
            u = torch.rand(shape, dtype=self._dtype, device=self.device, generator=generator)
            uniform = lo + u * (hi - lo)
            normal = torch.randn(shape, dtype=self._dtype, device=self.device, generator=generator)
            out = torch.where(finite, uniform, normal)
        else:
            out = torch.randn(shape, dtype=self._dtype, device=self.device, generator=generator)
        return out

    def contains(self, x):
        """Return True when ``x`` lies within every per-dimension bound."""
        x = torch.as_tensor(x, dtype=self._dtype, device=self.device)
        # A 0-dim sample (``x.shape == ()``) would raise IndexError on
        # ``x.shape[-1]``; it is only a member of a 1-D box of size 1.
        if x.dim() == 0:
            if self._shape[0] != 1:
                return False
            x = x.reshape(self._shape)
        elif x.shape[-1] != self._shape[0]:
            return False
        return bool(((x >= self.low) & (x <= self.high)).all())

    def __repr__(self):
        lo = float(self.low.min()) if self.low.numel() else float("nan")
        hi = float(self.high.max()) if self.high.numel() else float("nan")
        return f"Box(low={lo}, high={hi}, shape={self._shape}, dtype={self._dtype})"

    def __eq__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        return (
            self._shape == other._shape
            and torch.equal(self.low, other.low)
            and torch.equal(self.high, other.high)
        )
