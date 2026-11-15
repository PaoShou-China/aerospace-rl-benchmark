"""Unified single-agent wind/disturbance injection for the env side.

One interface, three components, so every future task wires disturbances in
the same way and dynamics code never grows ad-hoc wind logic:

- steady wind (direction + speed, per-replica), optionally sheared with a
  power-law profile in altitude
- Dryden turbulence (MIL-F-8785C low-altitude form) in two selectable
  flavours: the legacy first-order shaping filter per body axis
  (``model="first_order"``, bit-for-bit benchmark path) or the exact
  second-order lateral/vertical filters with a first-order longitudinal
  filter (``model="dryden"``).  Both normalise the output so the
  steady-state per-axis std equals the severity-class sigma scaled by
  ``dryden``.
- a seeded deterministic RNG stream per instance

Usage inside an env (dynamics stay untouched -- just add the returned
velocity to the airflow term, i.e. v_air = v_vehicle - v_wind):

    from environments.common.disturbances import WindDisturbance

    self._wind = WindDisturbance(num_envs, device, seed,
                                 steady_wind=cfg.steady_wind,
                                 dryden=cfg.dryden, dt=cfg.physics_dt,
                                 model="dryden", shear_exponent=0.14)
    ...  # in reset():
    self._wind.reset(env_ids)
    ...  # in step(), after state integration:
    wind = self._wind.step(altitude=altitude)  # (num_envs, 3) world [m/s]

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
    time_varying    rebuild the discrete filters from the per-replica airspeed
                    and altitude every ``update_interval`` seconds.  The
                    MIL-F-8785C scale lengths and intensities depend on height
                    and the filter bandwidth on V/L, so a climbing or
                    accelerating vehicle sees the correct gust spectrum.
                    With ``False`` the filters are built once at construction
                    (the published benchmark path).
    update_interval cadence [s] of the time-varying filter rebuild.
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
        model="first_order",
        shear_exponent=0.0,
        shear_reference_height=10.0,
        reference_altitude=None,
        time_varying=False,
        update_interval=0.2,
    ):
        if model not in ("first_order", "dryden"):
            raise ValueError(f"unknown turbulence model {model!r}")
        self.num_envs = num_envs
        self.device = device
        self.dt = float(dt)
        self.V = max(float(airspeed), 1e-3)
        self._damp = float(min(max(dryden, 0.0), 1.0))
        self._seed = int(seed)
        self.model = model
        self.time_varying = bool(time_varying)
        self._update_interval = max(float(update_interval), self.dt)
        self._elapsed = 0.0
        # Optional power-law mean-wind profile (Hellmann exponent).  The
        # supplied steady_wind is interpreted at ``shear_reference_height``;
        # alpha=0 (legacy default) leaves it constant with altitude.
        self._shear_alpha = float(shear_exponent)
        self._shear_height = max(float(shear_reference_height), 1e-3)

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
        self._sigma_ref = self._sigma.clone()
        # Reference height [ft] for the MIL low-altitude scaling.  A caller
        # that passes ``reference_altitude`` fixes the level at which the
        # severity sigmas apply; otherwise the standard 300 ft level is used.
        ref_m = 91.44 if reference_altitude is None else float(reference_altitude)
        self._ref_altitude_ft = min(max(ref_m / 0.3048, 10.0), 1000.0)
        if reference_altitude is None:
            lengths = self._LENGTHS
        else:
            # MIL-F-8785C low-altitude scale lengths: L_w = h and
            # L_u = L_v = h / (0.177 + 0.000823 h)^1.2, h in feet.  Clamp to
            # the 10-1000 ft validity band.
            h_ft = self._ref_altitude_ft
            l_uv_ft = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2
            lengths = (l_uv_ft * 0.3048, l_uv_ft * 0.3048, h_ft * 0.3048)
        self._length = torch.tensor(lengths, device=device)
        # Per-replica flight condition for the time-varying filters.
        self._airspeed = torch.full((num_envs,), self.V, device=device)
        self._altitude = torch.full(
            (num_envs,), (91.44 if reference_altitude is None else float(reference_altitude)),
            device=device,
        )
        if model == "first_order":
            if self.time_varying:
                self._pole = torch.zeros(num_envs, 3, device=device)
                self._gain = torch.zeros(num_envs, 3, device=device)
                self._build_first_order_tv()
            else:
                # Legacy path: one pole per axis, exactly as published.
                taus = self._length / self.V  # (3,)
                a = torch.exp(-self.dt / taus)  # pole radii
                # per-axis first-order shaping: steady-state std matches sigma exactly
                self._gain = self._sigma * torch.sqrt((1.0 - a * a).clamp_min(1e-12))
                self._pole = a
            self._state = torch.zeros(3, num_envs, device=device)
        else:
            if self.time_varying:
                self._A = torch.zeros(num_envs, 3, 2, 2, device=device)
                self._B = torch.zeros(num_envs, 3, 2, device=device)
                self._C = torch.zeros(num_envs, 3, 2, device=device)
                self._var_table = self._dryden_variance_table()
                self._build_dryden_tv()
            else:
                # MIL-F-8785C Dryden: u is first order, v/w are second order with a
                # numerator zero.  Build the exact zero-order-hold discrete system
                # once and normalise its output gain by the stationary variance, so
                # every axis has steady-state std == sigma regardless of dt/V/L.
                self._A, self._B, self._C = self._build_dryden()
            self._state2 = torch.zeros(3, num_envs, 2, device=device)
        # Counter-based per-replica RNG: every replica owns a stream keyed by
        # mix32(seed ^ i*GOLDEN) plus a private counter, so masked steps advance
        # only the selected replicas' noise sequences.
        i = torch.arange(num_envs, device=device, dtype=torch.int64)
        self._rng_seeds = _mix32((self._seed & 0xFFFFFFFF) ^ ((i * _GOLDEN) & 0xFFFFFFFF))
        self._rng_counter = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self._step_count = 0

    def _build_dryden(self):
        """Exact ZOH discretisation of the MIL-F-8785C Dryden filters.

        Returns ``(A, B, C)`` shaped ``(3, 2, 2)``, ``(3, 2)`` and
        ``(3, 2)``.  Axis 0 (u) is the first-order longitudinal filter;
        axes 1-2 (v, w) are the second-order lateral/vertical filters

            H(s) = sigma sqrt(2 L / (pi V)) (1 + sqrt(3)(L/V)s) / (1 + (L/V)s)^2

        The output row is scaled so that unit-variance discrete white noise
        produces a stationary standard deviation of exactly ``sigma``.
        """
        a_out = torch.zeros(3, 2, 2, dtype=torch.float64)
        b_out = torch.zeros(3, 2, dtype=torch.float64)
        c_out = torch.zeros(3, 2, dtype=torch.float64)
        for axis in range(3):
            length = float(self._length[axis])
            sigma = float(self._sigma[axis])
            tau = length / self.V
            k = sigma * math.sqrt(2.0 * length / (math.pi * self.V))
            if axis == 0:
                a = torch.tensor([[-1.0 / tau]], dtype=torch.float64)
                b = torch.tensor([[1.0]], dtype=torch.float64)
                c = torch.tensor([[k]], dtype=torch.float64)
            else:
                a = torch.tensor(
                    [[0.0, 1.0], [-1.0 / tau**2, -2.0 / tau]], dtype=torch.float64
                )
                b = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
                c = torch.tensor([[k, k * math.sqrt(3.0) * tau]], dtype=torch.float64)
            n = a.shape[0]
            # ZOH via the augmented matrix exponential:
            #   expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]]
            aug = torch.zeros(n + 1, n + 1, dtype=torch.float64)
            aug[:n, :n] = a
            aug[:n, n] = b[:, 0]
            disc = torch.matrix_exp(aug * self.dt)
            ad, bd = disc[:n, :n], disc[:n, n]
            # Stationary covariance P = Ad P Ad^T + Bd Bd^T, solved as the
            # Kronecker system (I - Ad (x) Ad) vec(P) = vec(Bd Bd^T).
            eye = torch.eye(n * n, dtype=torch.float64)
            lhs = eye - torch.kron(ad, ad)
            rhs = torch.outer(bd, bd).reshape(-1)
            p = torch.linalg.solve(lhs, rhs).reshape(n, n)
            # ``c`` already carries sigma; divide by the unit-noise response
            # std to pin the stationary output std to sigma.
            variance = float(c @ p @ c.T)
            c = c * (sigma / math.sqrt(max(variance, 1e-30)))
            a_out[axis, :n, :n] = ad
            b_out[axis, :n] = bd
            c_out[axis, :n] = c[0]
        return (
            a_out.to(self._steady.device, COMPUTE_DTYPE),
            b_out.to(self._steady.device, COMPUTE_DTYPE),
            c_out.to(self._steady.device, COMPUTE_DTYPE),
        )

    # -- time-varying filters ------------------------------------------------
    def _flight_conditions(self):
        """Per-replica MIL-F-8785C scale lengths [m] and sigmas [m/s].

        Below 1000 ft the standard gives ``L_w = h``,
        ``L_u = L_v = h / (0.177 + 0.000823 h)^1.2`` (h in feet) and
        ``sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4`` while
        ``sigma_w`` stays at its reference value.  Both are clamped to the
        10-1000 ft validity band.
        """
        h_ft = (self._altitude.clamp_min(0.1) / 0.3048).clamp(10.0, 1000.0)
        l_uv_ft = h_ft / (0.177 + 0.000823 * h_ft).pow(1.2)
        lengths = torch.stack((l_uv_ft, l_uv_ft, h_ft), -1) * 0.3048
        ref = 0.177 + 0.000823 * self._ref_altitude_ft
        scale_uv = ref / (0.177 + 0.000823 * h_ft)
        sigma = self._sigma_ref[None, :].expand_as(lengths).clone()
        sigma[:, :2] = sigma[:, :2] * scale_uv[:, None].pow(0.4)
        return sigma, lengths

    def _build_first_order_tv(self):
        """Rebuild the per-replica first-order gust filters from V and h."""
        sigma, lengths = self._flight_conditions()
        tau = lengths / self._airspeed[:, None].clamp_min(1.0)
        pole = torch.exp(-self.dt / tau)
        self._pole.copy_(pole)
        self._gain.copy_(sigma * torch.sqrt((1.0 - pole.square()).clamp_min(1e-12)))

    def _dryden_variance_table(self, size=256):
        """Stationary output variance of the unit-normalised second-order filter.

        In normalised time ``s = t / tau`` the Dryden lateral/vertical filter is
        ``A = [[0,1],[-1,-2]]``, ``B = [0,1]``, ``C = [1, sqrt(3)]`` and its
        ZOH discretisation depends only on ``r = dt/tau``.  Tabulating
        ``g(r) = var(C x)`` (log-spaced) lets the per-step rebuild normalise
        the output to ``sigma`` with tensor-only interpolation, which keeps the
        whole update free of host branches during graph capture.
        """
        log_r = torch.linspace(
            math.log(1e-4), math.log(10.0), size, dtype=torch.float64
        )
        g = torch.empty(size, dtype=torch.float64)
        eye = torch.eye(4, dtype=torch.float64)
        c = torch.tensor([1.0, math.sqrt(3.0)], dtype=torch.float64)
        for i, lr in enumerate(log_r):
            r = float(torch.exp(lr))
            e = math.exp(-r)
            ad = e * torch.tensor([[1.0 + r, r], [-r, 1.0 - r]], dtype=torch.float64)
            bd = torch.tensor([1.0 - e * (1.0 + r), r * e], dtype=torch.float64)
            q = torch.outer(bd, bd)
            p = torch.linalg.solve(eye - torch.kron(ad, ad), q.reshape(-1)).reshape(2, 2)
            g[i] = (c @ p @ c).clamp_min(1e-30)
        return log_r.to(self.device), torch.log(g).to(self.device)

    def _g_lookup(self, r):
        """Interpolate the tabulated stationary variance in log space."""
        table, values = self._var_table
        log_r = torch.log(r.clamp_min(1e-12)).to(table.dtype)
        idx = torch.searchsorted(table, log_r).clamp(1, table.numel() - 1) - 1
        w = ((log_r - table[idx]) / (table[idx + 1] - table[idx])).clamp(0.0, 1.0)
        return torch.exp(values[idx] * (1.0 - w) + values[idx + 1] * w).to(r.dtype)

    def _build_dryden_tv(self):
        """Rebuild the per-replica Dryden filters from V and h (tensor-only)."""
        sigma, lengths = self._flight_conditions()
        tau = lengths / self._airspeed[:, None].clamp_min(1.0)
        r = (self.dt / tau).clamp(1e-12, 10.0)
        e = torch.exp(-r)
        a = self._A
        a[..., 0, 0] = e * (1.0 + r)
        a[..., 0, 1] = e * r
        a[..., 1, 0] = -e * r
        a[..., 1, 1] = e * (1.0 - r)
        b = self._B
        b[..., 0] = 1.0 - e * (1.0 + r)
        b[..., 1] = r * e
        # Axis 0 (u) is first order; its stationary variance is (1-e)/(1+e).
        g0 = ((1.0 - e[:, 0]) / (1.0 + e[:, 0]).clamp_min(1e-12)).clamp_min(1e-30)
        g2 = self._g_lookup(r[:, 1:])
        gain = sigma / torch.cat((g0[:, None], g2), -1).sqrt()
        c = self._C
        c.zero_()
        c[..., 0] = gain
        c[:, 1:, 1] = gain[:, 1:] * math.sqrt(3.0)
        # The u axis uses the scalar first-order pair Ad = e, Bd = 1-e.
        a[:, 0, 0, 1] = 0.0
        a[:, 0, 1, 0] = 0.0
        a[:, 0, 1, 1] = 0.0
        b[:, 0, 1] = 0.0
        a[:, 0, 0, 0] = e[:, 0]
        b[:, 0, 0] = 1.0 - e[:, 0]

    def _refresh_filters(self, airspeed, altitude):
        """Update the stored flight condition and rebuild the discrete filters."""
        if airspeed is not None:
            self._airspeed.copy_(
                torch.as_tensor(airspeed, device=self.device, dtype=COMPUTE_DTYPE)
                .reshape(-1)
                .clamp_min(1.0)
            )
        if altitude is not None:
            self._altitude.copy_(
                torch.as_tensor(altitude, device=self.device, dtype=COMPUTE_DTYPE).reshape(-1)
            )
        if self.model == "first_order":
            self._build_first_order_tv()
        else:
            self._build_dryden_tv()

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
        return g.to(self._steady.dtype)

    def _index(self, env_ids):
        """Normalize ``env_ids`` (boolean mask or index tensor) to 1-D indices."""
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if ids.dtype == torch.bool:
            ids = ids.nonzero(as_tuple=False).flatten()
        return ids.to(torch.long)

    def _steady_at(self, altitude, idx=None):
        """Mean wind, optionally scaled by the power-law shear profile."""
        steady = self._steady if idx is None else self._steady[idx]
        if self._shear_alpha == 0.0 or altitude is None:
            return steady
        h = torch.as_tensor(altitude, device=self.device, dtype=COMPUTE_DTYPE).reshape(-1)
        if h.shape[0] != steady.shape[0]:
            h = h[idx]
        scale = (h.clamp_min(1.0) / self._shear_height).pow(self._shear_alpha)
        return steady * scale[:, None]

    # -- public API ----------------------------------------------------------
    def reset(self, env_ids=None):
        """Zero the filter states for (re)started replicas."""
        if self.model == "first_order":
            state = self._state
            if env_ids is None:
                state.zero_()
                return
            if (torch.is_tensor(env_ids) and env_ids.dtype == torch.bool
                    and env_ids.device == state.device):
                # Boolean masks reset in place without a host-synchronizing nonzero().
                state.masked_fill_(env_ids.reshape(1, -1), 0.0)
                return
            state[:, self._index(env_ids)] = 0.0
            return
        state = self._state2
        if env_ids is None:
            state.zero_()
            return
        if (torch.is_tensor(env_ids) and env_ids.dtype == torch.bool
                and env_ids.device == state.device):
            state.masked_fill_(env_ids.reshape(1, -1, 1), 0.0)
            return
        state[:, self._index(env_ids)] = 0.0

    def step(self, env_ids=None, altitude=None, airspeed=None):
        """Advance the filters and return wind [m/s].

        ``env_ids=None`` advances every replica and returns ``(num_envs, 3)``.
        A boolean mask or an index tensor advances ONLY those replicas (the
        other filters stay frozen) and returns their rows, ``(len(ids), 3)``,
        so a caller can scatter it into its per-env airflow directly.
        ``altitude`` (optional, per replica) enables the power-law mean-wind
        shear profile; ``None`` leaves the steady component unchanged.
        ``airspeed`` (optional, per replica) and ``altitude`` also drive the
        time-varying Dryden filters when ``time_varying=True``.
        """
        self._step_count = getattr(self, "_step_count", 0) + 1
        if self.time_varying:
            self._refresh_filters(airspeed, altitude)
        if env_ids is None:
            sel = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            g = self._gauss(sel)  # (N, 3)
            if self.model == "first_order":
                if self.time_varying:
                    pole, gain = self._pole.T, self._gain.T
                else:
                    pole, gain = self._pole.unsqueeze(1), self._gain.unsqueeze(1)
                # In place: the filter state keeps its storage across steps.
                self._state.copy_(pole * self._state + gain * g.T)
                gust = self._state.T
            else:
                if self.time_varying:
                    x = torch.einsum("nsij,snj->sni", self._A, self._state2)
                    x = x + self._B.permute(1, 0, 2) * g.T.unsqueeze(-1)
                    c = self._C.permute(1, 0, 2)
                else:
                    x = torch.einsum("sij,snj->sni", self._A, self._state2)
                    x = x + self._B[:, None, :] * g.T.unsqueeze(-1)
                    c = self._C[:, None, :]
                self._state2.copy_(x)
                gust = (c * x).sum(-1).T
            return self._steady_at(altitude) + gust  # (N, 3)
        idx = self._index(env_ids)
        sel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sel[idx] = True
        g = self._gauss(sel)  # (N, 3); only the selected rows consume samples
        if self.model == "first_order":
            if self.time_varying:
                pole, gain = self._pole[idx].T, self._gain[idx].T
            else:
                pole, gain = self._pole.unsqueeze(1), self._gain.unsqueeze(1)
            sub_state = pole * self._state[:, idx] + gain * g[idx].T
            self._state[:, idx] = sub_state
            gust = sub_state.T
        else:
            if self.time_varying:
                x = torch.einsum("ksij,skj->ski", self._A[idx], self._state2[:, idx])
                x = x + self._B[idx].permute(1, 0, 2) * g[idx].T.unsqueeze(-1)
                c = self._C[idx].permute(1, 0, 2)
            else:
                x = torch.einsum("sij,skj->ski", self._A, self._state2[:, idx])
                x = x + self._B[:, None, :] * g[idx].T.unsqueeze(-1)
                c = self._C[:, None, :]
            self._state2[:, idx] = x
            gust = (c * x).sum(-1).T
        return self._steady_at(altitude, idx) + gust  # (K, 3)

    def steady(self):
        """Steady component only, (N, 3) -- useful for baseline probes."""
        return self._steady.clone()
