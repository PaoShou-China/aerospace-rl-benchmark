"""Unified single-agent wind/disturbance injection for the env side.

One interface, three components, so every future task wires disturbances in
the same way and dynamics code never grows ad-hoc wind logic:

- steady constant wind (direction + speed, per-replica)
- Dryden turbulence (MIL-F-8785C low-altitude form): first-order shaping
  filters driven by unit white noise, one per body axis; the steady-state
  per-axis std equals the severity-class sigma scaled by ``dryden``
- a seeded deterministic RNG stream per instance

Usage inside an env (dynamics stay untouched -- just add the returned
velocity to the airflow term, i.e. v_air = v_vehicle - v_wind):

    from environments.common.disturbances import WindDisturbance

    self._wind = WindDisturbance(num_envs, device, seed,
                                 steady_wind=cfg.steady_wind,
                                 dryden=cfg.dryden, dt=cfg.physics_dt)
    ...  # in reset():
    self._wind.reset(env_ids)
    ...  # in step(), after state integration:
    wind = self._wind.step()            # (num_envs, 3) world-frame [m/s]

The module is pure torch, device-resident, and stateless across calls
except for its own filter states; it never touches env internals.

Randomness is a counter-based stream PER REPLICA (the same scheme as
``plant.TensorRandom``): ``seed_i = mix32(seed ^ i * GOLDEN)`` and a private
sample counter, so advancing a subset via ``step(env_ids)`` never shifts the
noise sequence another replica will see (a single shared CPU generator made
the samples depend on how many *other* replicas had been stepped).
"""

import math

import torch

from .dynamics import COMPUTE_DTYPE


# murmur3 fmix32; bijective modulo 2**32 (same finalizer as plant.TensorRandom).
def _mix32(x):
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) & 0xFFFFFFFF
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    return x ^ (x >> 16)


_GOLDEN = 0x9E3779B1


class WindDisturbance:
    """Per-replica steady wind + Dryden gust generator.

    Args
    ----
    num_envs        number of parallel replicas.
    device          torch device for all buffers.
    seed            seed for the counter-based per-replica RNG stream.
    steady_wind     (3,) world-frame constant wind [m/s] (broadcast to all
                    replicas), or None for calm air.
    dryden          turbulence intensity in [0, 1] -- 0 disables the gust
                    filters, 1 is the full MIL-F-8785C low-altitude
                    intensity for the given severity class.
    severity        Dryden turbulence severity class 'light' | 'moderate' |
                    'severe' (sigma reference levels at 300 ft).
    dt              integration period of the shaping filters [s]; pass the
                    env physics step so the gust bandwidth scales correctly.
    airspeed        reference airspeed for the filter time constants [m/s]
                    (the same V is used for every replica; tasks that know
                    per-replica airspeed can pass it to step()).
    """

    # MIL-F-8785C low-altitude reference sigmas (300 ft) per severity [m/s]
    _SIGMAS = {
        "light": (1.5, 1.5, 0.7),  # (u, v, w)
        "moderate": (3.0, 3.0, 1.4),
        "severe": (6.0, 6.0, 3.0),
    }
    # low-altitude turbulence scale lengths [m]
    _LENGTHS = (200.0, 200.0, 50.0)  # (Lu, Lv, Lw)

    def __init__(
        self,
        num_envs,
        device="cpu",
        seed=7,
        steady_wind=None,
        dryden=0.0,
        severity="moderate",
        dt=1.0 / 240,
        airspeed=15.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.dt = float(dt)
        self.V = max(float(airspeed), 1e-3)
        self._damp = float(min(max(dryden, 0.0), 1.0))
        self._seed = int(seed)

        if steady_wind is None:
            self._steady = torch.zeros(num_envs, 3, device=device)
        else:
            self._steady = (
                torch.as_tensor(steady_wind, dtype=torch.float32, device=device)
                .expand(num_envs, 3)
                .clone()
            )

        sig = self._SIGMAS[severity]
        self._sigma = torch.tensor(sig, device=device) * self._damp
        self._length = torch.tensor(self._LENGTHS, device=device)

        # second-order discrete state-space per axis, built once from the
        # continuous Dryden poles:  s_u = -V/Lu ;  s_v,w = -V/Lv,w (doubled)
        taus = self._length / self.V  # (3,)
        a = torch.exp(-self.dt / taus)  # pole radii
        # per-axis first-order shaping: steady-state std matches sigma exactly
        self._gain = self._sigma * torch.sqrt((1.0 - a * a).clamp_min(1e-12))
        # first-order shaping is enough at env step rates; store pole + state
        self._pole = a
        self._state = torch.zeros(3, num_envs, device=device)
        # Counter-based per-replica RNG: every replica owns a stream keyed by
        # mix32(seed ^ i*GOLDEN) plus a private counter, so masked steps advance
        # only the selected replicas' noise sequences.
        i = torch.arange(num_envs, device=device, dtype=torch.int64)
        self._rng_seeds = _mix32((self._seed & 0xFFFFFFFF) ^ ((i * _GOLDEN) & 0xFFFFFFFF))
        self._rng_counter = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self._step_count = 0

    # -- RNG ----------------------------------------------------------------
    def _uniform(self, width, mask):
        """``width`` uniforms in [0, 1) per replica; only ``mask`` rows advance."""
        k = self._rng_counter[:, None] + torch.arange(
            width, device=self.device, dtype=torch.int64
        )
        x = (self._rng_seeds[:, None] + (k + 1) * 1664525 + 1013904223) & 0xFFFFFFFF
        x = ((x ^ (x >> 16)) * 2246822519) & 0xFFFFFFFF
        x = ((x ^ (x >> 13)) * 3266489917) & 0xFFFFFFFF
        x = x ^ (x >> 16)
        self._rng_counter.add_(mask.to(torch.int64) * width)
        return (x.to(COMPUTE_DTYPE) + 0.5) / 4294967296.0

    def _gauss(self, mask):
        """Three unit gaussians per replica from that replica's own stream.

        ``mask`` selects the replicas that CONSUME samples; every other replica
        keeps its counter, so a masked step cannot shift its future noise.
        """
        u = self._uniform(6, mask)  # 3 Box-Muller pairs, one per body axis
        g = torch.sqrt(-2.0 * torch.log(u[:, 0::2])) * torch.cos(2 * math.pi * u[:, 1::2])
        return g.to(self._state.dtype)

    def _index(self, env_ids):
        """Normalize ``env_ids`` (boolean mask or index tensor) to 1-D indices."""
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if ids.dtype == torch.bool:
            ids = ids.nonzero(as_tuple=False).flatten()
        return ids.to(torch.long)

    # -- public API ----------------------------------------------------------
    def reset(self, env_ids=None):
        """Zero the filter states for (re)started replicas."""
        if env_ids is None:
            self._state.zero_()
            return
        idx = self._index(env_ids)
        self._state[:, idx] = 0.0

    def step(self, env_ids=None):
        """Advance the filters and return wind [m/s].

        ``env_ids=None`` advances every replica and returns ``(num_envs, 3)``.
        A boolean mask or an index tensor advances ONLY those replicas (the
        other filters stay frozen) and returns their rows, ``(len(ids), 3)``,
        so a caller can scatter it into its per-env airflow directly.
        """
        self._step_count = getattr(self, "_step_count", 0) + 1
        if env_ids is None:
            sel = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            g = self._gauss(sel).T  # (3, N)
            self._state = self._pole.unsqueeze(1) * self._state + self._gain.unsqueeze(1) * g
            return self._steady + self._state.T  # (N, 3)
        idx = self._index(env_ids)
        sel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sel[idx] = True
        g = self._gauss(sel)  # (N, 3); only the selected rows consume samples
        sub_state = (
            self._pole.unsqueeze(1) * self._state[:, idx]
            + self._gain.unsqueeze(1) * g[idx].T
        )
        self._state[:, idx] = sub_state
        return self._steady[idx] + sub_state.T  # (K, 3)

    def steady(self):
        """Steady component only, (N, 3) -- useful for baseline probes."""
        return self._steady.clone()
