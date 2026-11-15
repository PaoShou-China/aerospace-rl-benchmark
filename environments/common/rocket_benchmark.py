"""Versioned rocket contact, hidden episode variation and landing metrics.

This module has no Genesis dependency so protocol checks can run on CPU.
The parameters below belong to the simulator, never the policy observation.
"""

import math
import torch

from .dynamics import COMPUTE_DTYPE


VARIATIONS = {
    "nominal": dict(
        thrust=(1.0, 1.0),
        fin=(1.0, 1.0),
        booster_tau=(0.01, 0.01),
        gimbal_tau=(0.01, 0.01),
        gimbal_bias_deg=(0.0, 0.0),
        wind=(0.0, 0.0),
    ),
    "shifted": dict(
        thrust=(0.92, 1.08),
        fin=(0.6, 1.4),
        booster_tau=(0.005, 0.03),
        gimbal_tau=(0.005, 0.03),
        gimbal_bias_deg=(-0.3, 0.3),
        wind=(0.0, 5.0),
    ),
    "stress": dict(
        thrust=(0.85, 1.15),
        fin=(0.3, 1.7),
        booster_tau=(0.005, 0.06),
        gimbal_tau=(0.005, 0.06),
        gimbal_bias_deg=(-1.0, 1.0),
        wind=(0.0, 10.0),
    ),
}
VARIATION_SEED_OFFSET = 104729


class RocketVariation:
    """Per-replica hidden rocket parameters drawn from a named preset.

    Overrides thrust scale, fin authority, booster/gimbal actuator time
    constants, gimbal-bias angles and ambient wind.  Every buffer is a
    device-resident tensor of shape ``(n, ...)`` and is sampled from a
    per-replica uniform stream provided by the caller's RNG.
    """
    def __init__(self, n, device, rng, preset):
        if preset not in VARIATIONS:
            raise ValueError(f"Unknown rocket variation: {preset}")
        self.ranges = VARIATIONS[preset]
        self.rng = rng
        one = torch.ones(n, device=device, dtype=COMPUTE_DTYPE)
        self.thrust = one.clone()
        self.fin = one.clone()
        self.booster_tau = one * 0.01
        self.gimbal_tau = one * 0.01
        self.gimbal_bias = torch.zeros((n, 2), device=device, dtype=COMPUTE_DTYPE)
        self.wind = torch.zeros((n, 3), device=device, dtype=COMPUTE_DTYPE)
        # Match the research v2 stream: construction draw, then reset draw.
        self.sample(torch.ones(n, device=device, dtype=torch.bool))

    def sample(self, mask):
        """Draw new per-replica parameter values for every active replica.

        Consumes 8 uniform samples per active replica: 4 scalar multipliers,
        2 gimbal-bias axes, and 2 wind magnitude/heading pairs.
        """
        u = self.rng.uniform(8, mask)

        def lerp(key, x):
            """Map ``x in [0, 1]`` linearly onto ``ranges[key]``."""
            low, high = self.ranges[key]
            return low + (high - low) * x

        for col, key in enumerate(("thrust", "fin", "booster_tau", "gimbal_tau")):
            getattr(self, key).copy_(torch.where(mask, lerp(key, u[:, col]), getattr(self, key)))
        bias = (
            torch.stack((lerp("gimbal_bias_deg", u[:, 4]), lerp("gimbal_bias_deg", u[:, 5])), -1)
            * math.pi
            / 180
        )
        self.gimbal_bias.copy_(torch.where(mask[:, None], bias, self.gimbal_bias))
        speed, heading = lerp("wind", u[:, 6]), 2 * math.pi * u[:, 7]
        wind = torch.stack(
            (speed * torch.cos(heading), speed * torch.sin(heading), torch.zeros_like(speed)), -1
        )
        self.wind.copy_(torch.where(mask[:, None], wind, self.wind))


class LandingMetrics:
    """Consecutive post-policy-step observations; a gap resets the dwell.

    N samples span (N-1)/Hz seconds, hence 121 at 60 Hz and 81 at 40 Hz.
    Fatal/bounds termination invalidates the terminal sample. Time-limit
    truncation does not. No reset/warmup sample contributes to the dwell.
    """

    def __init__(self, n, device, agent_hz, velocity_limit=0.02, spin_limit=0.02, tilt_limit=0.1):
        self.required_samples = math.ceil(2.0 * agent_hz) + 1
        self.velocity_limit = velocity_limit
        self.spin_limit = spin_limit
        self.tilt_limit = tilt_limit
        self.streak = torch.zeros(n, device=device, dtype=torch.long)
        self.ever_stable = torch.zeros(n, device=device, dtype=torch.bool)
        self.terminal_settled = self.ever_stable.clone()

    def update(self, contact, velocity, omega, tilt, failed):
        """Update dwell counters from the latest post-step observation."""
        settled = (
            contact
            & ~failed
            & (velocity.norm(dim=-1) < self.velocity_limit)
            & (omega.norm(dim=-1) < self.spin_limit)
            & (tilt < self.tilt_limit)
        )
        self.terminal_settled.copy_(settled)
        self.streak.copy_(torch.where(settled, self.streak + 1, 0))
        self.ever_stable |= self.streak >= self.required_samples

    def values(self, ever_success, failed):
        """Return a dict of per-replica landing metric boolean flags."""
        return {
            "ever_success": ever_success,
            "ever_stable_2s": self.ever_stable,
            "terminal_single_point": self.terminal_settled,
            "terminal_stable_2s": self.streak >= self.required_samples,
            "success_then_failure": ever_success & failed,
            "stable_then_failure": self.ever_stable & failed,
        }

    def reset(self, mask):
        """Clear dwell counters for replicas indicated by ``mask``."""
        self.streak.masked_fill_(mask, 0)
        self.ever_stable.masked_fill_(mask, False)
        self.terminal_settled.masked_fill_(mask, False)
