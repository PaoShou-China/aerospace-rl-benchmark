"""Supplemental landing metrics; never replace the environment success flag."""
from __future__ import annotations
import math
import numpy as np


class StableLandingMonitor:
    """Track first episodes at the 40-Hz policy sample rate.

    A sampled dwell is not a continuous-time guarantee. A two-second settled
    streak followed by a crash is NOT durable success. Update terminal samples
    before auto-reset and exclude later episodes.
    """
    def __init__(self, count: int, dt: float = .025, hold_seconds: float = 2.):
        if count < 1 or dt <= 0 or hold_seconds <= 0:
            raise ValueError('positive count, dt and hold time required')
        self.count, self.dt = count, float(dt)
        self.required_steps = int(math.ceil(hold_seconds / dt - 1e-10))
        self.consecutive = np.zeros(count, dtype=np.int64)
        self.longest = np.zeros(count, dtype=np.int64)
        self.ever_official = np.zeros(count, dtype=bool)
        self.ever_fatal = np.zeros(count, dtype=bool)
        self.ever_out = np.zeros(count, dtype=bool)
        self.total = np.zeros(count, dtype=np.int64)
        self.finished = np.zeros(count, dtype=bool)

    def update(self, settled, official, fatal, out_of_bounds, active=None):
        vals = [np.asarray(v, dtype=bool) for v in
                (settled, official, fatal, out_of_bounds)]
        if any(v.shape != (self.count,) for v in vals):
            raise ValueError('monitor input shape must equal (num_envs,)')
        settled, official, fatal, out = vals
        mask = ~self.finished
        if active is not None:
            mask &= np.asarray(active, dtype=bool)
        valid_settle = settled & ~fatal & ~out
        self.consecutive[mask] = np.where(valid_settle[mask], self.consecutive[mask] + 1, 0)
        self.longest[mask] = np.maximum(self.longest[mask], self.consecutive[mask])
        self.ever_official[mask] |= official[mask]
        self.ever_fatal[mask] |= fatal[mask]
        self.ever_out[mask] |= out[mask]
        self.total[mask] += 1

    def finish(self, index: int) -> dict:
        if self.finished[index]:
            raise ValueError('first episode was already finalized')
        self.finished[index] = True
        terminal_hold = self.consecutive[index] >= self.required_steps
        return {
            'sampled_terminal_dwell_seconds': float(self.consecutive[index] * self.dt),
            'sampled_longest_dwell_seconds': float(self.longest[index] * self.dt),
            'sampled_two_second_terminal_settle': bool(terminal_hold),
            'durable_success': bool(terminal_hold and not self.ever_fatal[index]
                                    and not self.ever_out[index]),
            'official_success_ever': bool(self.ever_official[index]),
            'official_then_fatal_or_out': bool(self.ever_official[index]
                                               and (self.ever_fatal[index] or self.ever_out[index])),
            'monitored_policy_samples': int(self.total[index]),
            'metric_sampling_hz': 1 / self.dt,
        }
