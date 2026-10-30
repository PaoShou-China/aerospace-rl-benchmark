"""Versioned rocket contact, hidden episode variation and landing metrics.

This module has no Genesis dependency so protocol checks can run on CPU.
The parameters below belong to the simulator, never the policy observation.
"""
import math
import torch


VARIATIONS = {
    'nominal': dict(thrust=(1., 1.), fin=(1., 1.), booster_tau=(.01, .01),
                    gimbal_tau=(.01, .01), gimbal_bias_deg=(0., 0.), wind=(0., 0.)),
    'shifted': dict(thrust=(.92, 1.08), fin=(.6, 1.4), booster_tau=(.005, .03),
                    gimbal_tau=(.005, .03), gimbal_bias_deg=(-.3, .3), wind=(0., 5.)),
    'stress': dict(thrust=(.85, 1.15), fin=(.3, 1.7), booster_tau=(.005, .06),
                   gimbal_tau=(.005, .06), gimbal_bias_deg=(-1., 1.), wind=(0., 10.)),
}
VARIATION_SEED_OFFSET = 104729


class RocketVariation:
    def __init__(self, n, device, rng, preset):
        if preset not in VARIATIONS:
            raise ValueError(f'Unknown rocket variation: {preset}')
        self.ranges = VARIATIONS[preset]
        self.rng = rng
        one = torch.ones(n, device=device, dtype=torch.float64)
        self.thrust = one.clone()
        self.fin = one.clone()
        self.booster_tau = one * .01
        self.gimbal_tau = one * .01
        self.gimbal_bias = torch.zeros((n, 2), device=device, dtype=torch.float64)
        self.wind = torch.zeros((n, 3), device=device, dtype=torch.float64)
        # Match the research v2 stream: construction draw, then reset draw.
        self.sample(torch.ones(n, device=device, dtype=torch.bool))

    def sample(self, mask):
        u = self.rng.uniform(8, mask)
        def lerp(key, x):
            low, high = self.ranges[key]
            return low + (high - low) * x
        for col, key in enumerate(('thrust', 'fin', 'booster_tau', 'gimbal_tau')):
            getattr(self, key).copy_(torch.where(mask, lerp(key, u[:, col]), getattr(self, key)))
        bias = torch.stack((lerp('gimbal_bias_deg', u[:, 4]),
                            lerp('gimbal_bias_deg', u[:, 5])), -1) * math.pi / 180
        self.gimbal_bias.copy_(torch.where(mask[:, None], bias, self.gimbal_bias))
        speed, heading = lerp('wind', u[:, 6]), 2 * math.pi * u[:, 7]
        wind = torch.stack((speed * torch.cos(heading), speed * torch.sin(heading),
                            torch.zeros_like(speed)), -1)
        self.wind.copy_(torch.where(mask[:, None], wind, self.wind))


class LandingMetrics:
    """Consecutive post-policy-step observations; a gap resets the dwell.

    N samples span (N-1)/Hz seconds, hence 121 at 60 Hz and 81 at 40 Hz.
    Fatal/bounds termination invalidates the terminal sample. Time-limit
    truncation does not. No reset/warmup sample contributes to the dwell.
    """
    def __init__(self, n, device, agent_hz):
        self.required_samples = math.ceil(2. * agent_hz) + 1
        self.streak = torch.zeros(n, device=device, dtype=torch.long)
        self.ever_stable = torch.zeros(n, device=device, dtype=torch.bool)
        self.terminal_settled = self.ever_stable.clone()

    def update(self, contact, velocity, omega, tilt, failed):
        settled = contact & ~failed & (velocity.norm(dim=-1) < .02) & (
            omega.norm(dim=-1) < .02) & (tilt < .1)
        self.terminal_settled.copy_(settled)
        self.streak.copy_(torch.where(settled, self.streak + 1, 0))
        self.ever_stable |= self.streak >= self.required_samples

    def values(self, ever_success, failed):
        return {
            'ever_success': ever_success,
            'ever_stable_2s': self.ever_stable,
            'terminal_single_point': self.terminal_settled,
            'terminal_stable_2s': self.streak >= self.required_samples,
            'success_then_failure': ever_success & failed,
            'stable_then_failure': self.ever_stable & failed,
        }

    def reset(self, mask):
        self.streak.masked_fill_(mask, 0)
        self.ever_stable.masked_fill_(mask, False)
        self.terminal_settled.masked_fill_(mask, False)
