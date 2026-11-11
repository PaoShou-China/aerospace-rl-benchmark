"""Opt-in observation sensor model for the realistic fidelity preset.

The benchmark path exposes the exact plant state.  Real autopilots see a
noisy, biased, delayed estimate instead.  This module adds that layer at the
observation boundary only -- the plant itself is untouched:

- white noise on body rates, attitude, body velocity and position;
- a slow gyro bias random walk;
- optional seeker noise on the missile's relative-position / relative-velocity
  / LOS-rate block;
- a one-policy-step transport delay (``latency=1``).

All randomness comes from a dedicated counter-based ``TensorRandom`` stream
seeded away from the plant stream, so enabling sensors never shifts the
motor/wind noise sequence.  ``reset(mask)`` re-zeroes bias and marks the
fresh replicas so they see their first observation without a stale delay.
"""

import math

import torch

from .dynamics import COMPUTE_DTYPE
from .plant import TensorRandom

_SENSOR_SEED_OFFSET = 0x5EED5EED


def _quat_mul_xyzw(a, b):
    """Hamilton product of two ``(..., 4)`` quaternions in ``xyzw`` order."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        -1,
    )


class SensorModel:
    """Per-replica Gaussian sensor errors plus a one-step observation delay."""

    def __init__(
        self,
        num_envs,
        device,
        seed,
        *,
        gyro_sigma=0.005,
        attitude_sigma=0.005,
        velocity_sigma=0.1,
        position_sigma=0.2,
        bias_step=1e-4,
        latency=1,
        seeker_position_sigma=0.0,
        seeker_velocity_sigma=0.0,
        seeker_los_sigma=0.0,
        dt=1.0 / 60.0,
        reference_dt=1.0 / 60.0,
    ):
        if latency not in (0, 1):
            raise ValueError("SensorModel supports latency of 0 or 1 policy steps")
        self.num_envs = num_envs
        self.device = device
        # Sigmas are calibrated per sample at ``reference_dt``; converting to
        # a continuous-time density keeps the noise bandwidth comparable when
        # the task runs at a different control rate.
        self._noise_scale = math.sqrt(float(reference_dt) / float(dt))
        self._bias_scale = math.sqrt(float(dt) / float(reference_dt))
        self.gyro_sigma = float(gyro_sigma)
        self.attitude_sigma = float(attitude_sigma)
        self.velocity_sigma = float(velocity_sigma)
        self.position_sigma = float(position_sigma)
        self.bias_step = float(bias_step)
        self.latency = int(max(latency, 0))
        self.seeker_position_sigma = float(seeker_position_sigma)
        self.seeker_velocity_sigma = float(seeker_velocity_sigma)
        self.seeker_los_sigma = float(seeker_los_sigma)
        # Per-replica domain-randomization multiplier on every noise source
        # (1.0 = nominal).  A noisier or cleaner sensor is a common sim-to-real
        # mismatch; scaling here never touches the plant RNG stream.
        self.sensor_scale = torch.ones((num_envs, 1), device=device, dtype=COMPUTE_DTYPE)
        # Dedicated stream: sensors must not perturb the plant's RNG sequence.
        self.rng = TensorRandom(num_envs, device, int(seed) ^ _SENSOR_SEED_OFFSET)
        self.bias = torch.zeros((num_envs, 3), device=device, dtype=COMPUTE_DTYPE)
        self._delay = None
        self._fresh = torch.ones(num_envs, device=device, dtype=torch.bool)

    # -- noise helpers ------------------------------------------------------
    def _normal3(self, mask):
        """Three independent unit normals per replica (only ``mask`` advances)."""
        u = self.rng.uniform(6, mask)
        g = torch.sqrt(-2.0 * torch.log(u[:, 0::2])) * torch.cos(6.283185307179586 * u[:, 1::2])
        return g.to(COMPUTE_DTYPE) * mask[:, None]

    def _noise(self, sigma, mask):
        return self._normal3(mask) * (float(sigma) * self._noise_scale) * self.sensor_scale

    def set_scale(self, scale):
        """Scale all noise sigmas and the bias random walk (float or per-env).

        Larger means a noisier sensor; 0 gives a noiseless sensor (the bias
        random walk is frozen too).  The scale is applied at sample time, so
        it can be changed per episode without rebuilding the model.
        """
        if isinstance(scale, (int, float)):
            self.sensor_scale.fill_(max(float(scale), 0.0))
        else:
            value = (
                scale.to(self.sensor_scale.device).to(COMPUTE_DTYPE).reshape(-1, 1).clamp_min(0.0)
            )
            self.sensor_scale.copy_(value)

    # -- public API ---------------------------------------------------------
    def reset(self, mask):
        """Re-zero bias and mark replicas as fresh (first obs not delayed)."""
        self.bias.masked_fill_(mask[:, None], 0.0)
        self._fresh.masked_fill_(mask, True)

    def apply(self, obs, mask, seeker_slice=None, los_slice=None):
        """Return a noisy, optionally delayed copy of the observation buffer.

        ``obs`` is ``(n, dim)``; only rows in ``mask`` consume random numbers.
        ``seeker_slice`` selects a 6-vector ``[rel_pos, rel_vel]`` and
        ``los_slice`` a 3-vector LOS rate; both are optional.
        """
        out = obs.clone()
        active = mask
        bias_noise = (
            self._normal3(active) * (self.bias_step * self._bias_scale) * self.sensor_scale
        )
        self.bias.add_(bias_noise)
        self.bias.clamp_(-0.02, 0.02)
        # header layout shared by every atmospheric task:
        # [wb(3), q_xyzw(4), vb(3), p(3), ...]
        out[:, 0:3] += (self.bias + self._noise(self.gyro_sigma, active)) * active[:, None]
        dtheta = self._noise(self.attitude_sigma, active)
        dq = torch.cat((0.5 * dtheta, torch.ones_like(dtheta[:, :1])), -1)
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        quat = _quat_mul_xyzw(out[:, 3:7], dq)
        out[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out[:, 7:10] += self._noise(self.velocity_sigma, active) * active[:, None]
        out[:, 10:13] += self._noise(self.position_sigma, active) * active[:, None]
        if seeker_slice is not None:
            start, end = seeker_slice
            if end - start >= 6:
                noise = self._noise(self.seeker_position_sigma, active)
                out[:, start : start + 3] += noise * active[:, None]
                noise = self._noise(self.seeker_velocity_sigma, active)
                out[:, start + 3 : start + 6] += noise * active[:, None]
        if los_slice is not None:
            start, end = los_slice
            if end - start != 3:
                raise ValueError("los_slice must select exactly three components")
            noise = self._noise(self.seeker_los_sigma, active)
            out[:, start:end] += noise * active[:, None]
        if self.latency <= 0:
            return out
        if self._delay is None:
            self._delay = out.clone()
            self._fresh.fill_(False)
            return out
        previous = self._delay.clone()
        self._delay.copy_(torch.where(active[:, None], out, self._delay))
        stale = active & ~self._fresh
        out.copy_(torch.where(stale[:, None], previous, out))
        self._fresh.masked_fill_(active, False)
        return out
