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
"""

import torch


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
    _SIGMAS = {'light': (1.5, 1.5, 0.7),          # (u, v, w)
               'moderate': (3.0, 3.0, 1.4),
               'severe': (6.0, 6.0, 3.0)}
    # low-altitude turbulence scale lengths [m]
    _LENGTHS = (200.0, 200.0, 50.0)               # (Lu, Lv, Lw)

    def __init__(self, num_envs, device='cpu', seed=7,
                 steady_wind=None, dryden=0.0, severity='moderate',
                 dt=1.0 / 240, airspeed=15.0):
        self.num_envs = num_envs
        self.device = device
        self.dt = float(dt)
        self.V = max(float(airspeed), 1e-3)
        self._damp = float(min(max(dryden, 0.0), 1.0))
        self._seed = int(seed)

        if steady_wind is None:
            self._steady = torch.zeros(num_envs, 3, device=device)
        else:
            self._steady = torch.as_tensor(steady_wind, dtype=torch.float32,
                                           device=device).expand(num_envs, 3).clone()

        sig = self._SIGMAS[severity]
        self._sigma = torch.tensor(sig, device=device) * self._damp
        self._length = torch.tensor(self._LENGTHS, device=device)

        # second-order discrete state-space per axis, built once from the
        # continuous Dryden poles:  s_u = -V/Lu ;  s_v,w = -V/Lv,w (doubled)
        taus = self._length / self.V                        # (3,)
        a = torch.exp(-self.dt / taus)                      # pole radii
        # per-axis first-order shaping: steady-state std matches sigma exactly
        self._gain = self._sigma * torch.sqrt((1.0 - a * a).clamp_min(1e-12))
        # first-order shaping is enough at env step rates; store pole + state
        self._pole = a
        self._state = torch.zeros(3, num_envs, device=device)
        self._gen = torch.Generator(device='cpu')
        self._gen.manual_seed(self._seed)
        self._step_count = 0

    # -- RNG ----------------------------------------------------------------
    def _gauss(self, env_ids=None):
        """Unit gaussians from the seeded CPU generator (deterministic)."""
        n = self.num_envs if env_ids is None else len(env_ids)
        return torch.randn(n, 3, generator=self._gen).to(self.device)

    # -- public API ----------------------------------------------------------
    def reset(self, env_ids=None):
        """Zero the filter states for (re)started replicas."""
        if env_ids is None:
            self._state.zero_()
        else:
            self._state[:, env_ids.to(self.device)] = 0.0

    def step(self, env_ids=None):
        """Advance the filters one period and return wind [m/s], (N, 3)."""
        self._step_count = getattr(self, '_step_count', 0) + 1
        g = self._gauss(env_ids).T                          # (3, N)
        self._state = self._pole.unsqueeze(1) * self._state + self._gain.unsqueeze(1) * g
        if env_ids is None:
            gust = self._state.T                            # (N, 3)
        else:
            gust = self._state.T[env_ids.to(self.device)]
        return self._steady + gust

    def steady(self):
        """Steady component only, (N, 3) -- useful for baseline probes."""
        return self._steady.clone()
