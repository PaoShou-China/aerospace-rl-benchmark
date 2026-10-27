"""Falcon 9 first-stage rocket landing environment.

Architecture: Genesis is the SINGLE rigid body integrator + contact solver.
A pure PyTorch tensor module provides:
  - atmosphere (US76), mass props (open-system LOX/RP-1 columns),
  - engine thrust/mdot/ignition, TVC gimbal, grid fins (audit-2 CG-relative flow),
  - body aero, RCS, variable-mass Idot·ω term.

At every physics substep (dt=0.001 s, 1 kHz):
  1. Advance actuator states (ignition ramp, throttle, gimbal PD, fins/legs deploy)
  2. Compute engine thrust, mdot, fuel depletion → auto-shutdown
  3. Evaluate atmosphere (rho, p, a_sound, T) at current world z
  4. Rotate world state → body frame; run aero / fins / RCS
  5. Assemble body-frame F_total, T_total about the CoM (includes r_nozzle × F_thrust)
  6. mass_props_batch → (M, com_body, I_tr, I_ax, ...) → composite mass/inertia/CoM
  7. Genesis set_links_mass + set_links_inertia + set_links_COM
  8. apply_links_external_wrench(force=F_world_from_body, torque=T_com - İ·ω, idx=0)
  9. scene.step()  → Genesis integrates rigid body + contact

Audit-1 … audit-9 fixes applied (see falcon9_dynamics.py docstring).

Genesis 1.4.4 constraints discovered:
  - merge_fixed_links=True gives one rigid body; apply_links_external_wrench
    with links_idx_local=<int> works correctly only then.
  - With merge_fixed_links=False, apply_links_external_wrench silently ignores
    force when any links_idx_local value is specified (Genesis bug).
  - Therefore we use merge_fixed_links=True and apply composite F, T to the
    single link. Per-tank fuel visualisation is not tracked here but per-env
    fuel mass is tracked in Python tensors.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import genesis as gs
import torch
from torch import Tensor

from .falcon9_params import Falcon9Params, P, SCENARIO_PRESETS, ScenarioDef
from .falcon9_dynamics import (
    atmosphere_eval, mass_props_batch, engine_deriv_batch, ignition_ramp_tick,
    gimbal_to_direction, gimbal_saturate, gimbal_deriv_batch,
    assemble_forces_torques, rotate_body_to_world, rotate_world_to_body,
)
from .falcon9_reward import (
    falcon9_shaping, default_reward_cfg, landing_thresholds, _body_tilt_from_quat,
)


# ---------------------------------------------------------------------------
# Engine state (per-env batched)
# ---------------------------------------------------------------------------
@dataclass
class Falcon9EngineState:
    throttle_act: Tensor       # (N,) [0, 1] rate-limited throttle
    throttle_cmd: Tensor      # (N,) commanded throttle
    eng_on: Tensor            # (N,) bool
    ignition_timer: Tensor    # (N,) s since ignition command (negative when off)
    relights_used: Tensor     # (N,) int
    # Gimbal
    g0_cmd: Tensor            # (N,) rad — commanded TVC deflection
    g1_cmd: Tensor            # (N,) rad
    g0_act: Tensor            # (N,) rad about body +X
    g1_act: Tensor            # (N,) rad about body +Y
    gr0: Tensor               # (N,) rad/s
    gr1: Tensor               # (N,) rad/s
    # Fuel
    m_lox: Tensor             # (N,) kg
    m_rp1: Tensor             # (N,) kg
    m_rcs_n2: Tensor          # (N,) kg


@dataclass
class Falcon9ActuatorState:
    fins_deployed: Tensor          # (N,) bool
    fin_deflection: Tensor         # (N, 4) rad  — zero for v1 (passive aero)
    rcs_on: Tensor                 # (N,) bool
    tau_rcs_body_cmd: Tensor       # (N, 3) N·m body-frame torque
    legs_deployed: Tensor          # (N,) bool


# ---------------------------------------------------------------------------
# Genesis-backed environment
# ---------------------------------------------------------------------------
class Falcon9Env:
    """Real-scale Falcon 9 first-stage landing environment.

    Args:
        num_envs: batch size
        backend: 'gpu' or 'cpu'
        p: Falcon9Params instance
        urdf_path: path to URDF (defaults to rocket_falcon9_stage1.urdf)
        pos: initial world position (base_link origin)
        wind_world: (N, 3) m/s world-frame wind
    """

    def __init__(self,
                 num_envs: int = 1,
                 backend: str = 'gpu',
                 p: Falcon9Params = P,
                 urdf_path: str | None = None,
                 pos: Optional[tuple] = None,
                 quat: Optional[tuple] = None,
                 vel: Optional[tuple] = None,
                 ang: Optional[tuple] = None,
                 wind_world: tuple = (0.0, 0.0, 0.0),
                 scenario: Optional[str] = None,
                 seed: int = 1,
                 run_idx: int = 0,
                 dispersion: bool = True,
                 ):
        self.p = p
        self.g0 = p.world.g0
        self.dt = p.timing.physics_dt     # 0.001 s (1 kHz)
        self.policy_dt = p.timing.policy_dt  # 0.02 s  (50 Hz)
        self.n_sub = p.timing.substeps_per_policy
        self.N = num_envs
        self.device = 'cuda' if backend == 'gpu' else 'cpu'
        self.scenario_name = scenario or p.default_scenario
        self._scenario_seed = seed
        self._scenario_run_idx = run_idx
        self._scenario_dispersion = dispersion

        # Genesis init
        # Genesis init — dt=0.001s (1 kHz), policy runs at 50 Hz with 20 substeps.
        # iterations/ls_iterations bumped from Genesis defaults (25/50) to 100/200
        # because Falcon 9 (29 t, 41 m) hits contact at ~10× the benchmark's toy-rock
        # impact velocities. Without more solver iterations, edge-contact on a tilted
        # cylinder produces ill-conditioned Jacobians → Invalid constraint forces / NaN.
        # Torsional/rolling friction are enabled so the four footpads dissipate the
        # rocking motion of an off-vertical touchdown (a point contact otherwise has
        # no rolling resistance at all and the body rocks forever).
        gs.init(backend=gs.gpu if backend == 'gpu' else gs.cpu, logging_level='warning')
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=self.dt, gravity=(0, 0, -self.g0)),
            rigid_options=gs.options.RigidOptions(
                batch_links_info=True,
                enable_self_collision=False,
                integrator=gs.integrator.implicitfast,
                iterations=100,
                ls_iterations=200,
                enable_torsional_friction=True,
                enable_rolling_friction=True,
                noslip_iterations=5,
            ),
        )

        # Ground plane (z=0)
        self.ground = self.scene.add_entity(gs.morphs.Plane())

        # Falcon 9 stage — merge_fixed_links=True → 1 rigid body
        urdf_path = urdf_path or (
            __file__.replace('falcon9_env.py',
                             'original_pyflyt/models/vehicles/rocket/rocket_falcon9_stage1.urdf')
        )
        scen = SCENARIO_PRESETS[self.scenario_name]
        cos_t = math.cos(scen.tilt0 / 2); sin_t = math.sin(scen.tilt0 / 2)
        # Genesis needs concrete tuple defaults for the URDF morph; we prime
        # with the scenario preset's nominal state and reset_from_scenario
        # overwrites everything after scene.build.
        _pos_prime   = (0.0, 0.0, scen.h0)              if pos  is None else tuple(pos)
        _quat_prime  = (cos_t, sin_t, 0.0, 0.0)         if quat is None else tuple(quat)
        _vel_prime   = (0.0, 0.0, scen.vz0)             if vel  is None else tuple(vel)
        _ang_prime   = (0.0, 0.0, 0.0)                   if ang  is None else tuple(ang)

        self.vehicle = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=_pos_prime, quat=_quat_prime, fixed=False, align=False,
                merge_fixed_links=True, recompute_inertia=True, scale=1.0,
            ),
            material=gs.materials.Rigid(
                friction=0.6, friction_torsional=0.5, friction_rolling=0.5,
                coup_friction=0.5, coup_softness=0.001, coup_restitution=0.0,
                contact_resistance=1e7,
            ),
        )

        self.scene.build(n_envs=num_envs)
        # With merge_fixed_links=True there is exactly ONE link → idx 0
        assert len(self.vehicle.links) == 1, \
            f"Expected 1 link (merge_fixed_links=True), got {len(self.vehicle.links)}"
        self.idx_root = 0

        # Initial state — build from the scenario preset by default; explicit
        # pos/quat/vel/ang override that preset's values. After scene.build we
        # run reset_from_scenario to apply per-env dispersion (seeded), so the
        # tensors below are only used to prime Genesis' state bookkeeping.
        scen = SCENARIO_PRESETS[self.scenario_name]
        cos_t = math.cos(scen.tilt0 / 2); sin_t = math.sin(scen.tilt0 / 2)
        # Default: upright, zero lateral, prescribed h0/vz0 from preset
        _pos = pos if pos is not None else (0.0, 0.0, scen.h0)
        _quat = quat if quat is not None else (cos_t, sin_t, 0.0, 0.0)
        _vel = vel if vel is not None else (0.0, 0.0, scen.vz0)
        _ang = ang if ang is not None else (0.0, 0.0, 0.0)

        initial_pos = torch.tensor(list(_pos), dtype=torch.float32, device=self.device)
        if initial_pos.ndim == 1: initial_pos = initial_pos.repeat(num_envs, 1)
        initial_quat = torch.tensor(list(_quat), dtype=torch.float32, device=self.device)
        if initial_quat.ndim == 1: initial_quat = initial_quat.repeat(num_envs, 1)
        initial_vel = torch.tensor(list(_vel), dtype=torch.float32, device=self.device)
        if initial_vel.ndim == 1: initial_vel = initial_vel.repeat(num_envs, 1)
        initial_ang = torch.tensor(list(_ang), dtype=torch.float32, device=self.device)
        if initial_ang.ndim == 1: initial_ang = initial_ang.repeat(num_envs, 1)

        self.vehicle.set_pos(initial_pos)
        self.vehicle.set_quat(initial_quat)
        self.vehicle.set_dofs_velocity(torch.cat([initial_vel, initial_ang], dim=-1))

        # Actuator state (allocated on self.device from the start)
        dev = self.device
        ep = p.engine; pp = p.propellant
        prop0 = scen.prop0
        m_rp1_init = prop0 / (1.0 + pp.MIX_RATIO)
        m_lox_init = prop0 - m_rp1_init
        self.engine = Falcon9EngineState(
            throttle_act=torch.full((num_envs,), ep.ENG_THR_MIN, device=dev),
            throttle_cmd=torch.full((num_envs,), ep.ENG_THR_MIN, device=dev),
            eng_on=torch.zeros(num_envs, dtype=torch.bool, device=dev),
            ignition_timer=torch.full((num_envs,), -1.0, device=dev),
            relights_used=torch.zeros(num_envs, dtype=torch.int64, device=dev),
            g0_cmd=torch.zeros(num_envs, device=dev),
            g1_cmd=torch.zeros(num_envs, device=dev),
            g0_act=torch.zeros(num_envs, device=dev),
            g1_act=torch.zeros(num_envs, device=dev),
            gr0=torch.zeros(num_envs, device=dev),
            gr1=torch.zeros(num_envs, device=dev),
            m_lox=torch.full((num_envs,), m_lox_init, device=dev),
            m_rp1=torch.full((num_envs,), m_rp1_init, device=dev),
            m_rcs_n2=torch.full((num_envs,), pp.RCS_N2, device=dev),
        )
        self.act = Falcon9ActuatorState(
            fins_deployed=torch.full((num_envs,), bool(scen.fins), dtype=torch.bool, device=dev),
            fin_deflection=torch.zeros(num_envs, 4, device=dev),
            rcs_on=torch.zeros(num_envs, dtype=torch.bool, device=dev),
            tau_rcs_body_cmd=torch.zeros(num_envs, 3, device=dev),
            legs_deployed=torch.zeros(num_envs, dtype=torch.bool, device=dev),
        )

        # Wind (can be updated per env during reset)
        self.wind_world = torch.tensor(
            list(wind_world), device=self.device, dtype=torch.float32
        ).repeat(num_envs, 1)

        # ------------------------------------------------------------------
        # Reward + termination bookkeeping
        # ------------------------------------------------------------------
        self.reward_cfg = default_reward_cfg()
        self.thr = landing_thresholds()
        self.max_steps = self.thr['episode_seconds'] * self.policy_dt ** -1  # 2000

        # Per-env episode buffers
        self.count = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        self.reward = torch.zeros(num_envs, device=self.device)
        self.terminated = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.truncated = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.complete = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        # Contact bookkeeping
        self.contact_pad = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.contact_vz_first = torch.zeros(num_envs, device=self.device)
        self.touch_any = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.touch_vz_first = torch.zeros(num_envs, device=self.device)

        self._last_reward_components: dict = {k: torch.zeros(num_envs) for k in
                                              ('hover', 'hold', 'lateral', 'descent',
                                               'decel', 'attitude', 'contact', 'success')}

        # ------------------------------------------------------------------
        # Prime Genesis state with the scenario preset and apply seeded dispersion.
        # Scenario selection / dispersion seed can be overridden at construction.
        # ------------------------------------------------------------------
        self.reset_from_scenario(
            scenario_name=self.scenario_name,
            seed=self._scenario_seed,
            run_idx=self._scenario_run_idx,
            dispersion=self._scenario_dispersion,
        )

    # ------------------------------------------------------------------
    # Scenario initialisation / domain randomisation
    # ------------------------------------------------------------------
    def reset_from_scenario(
            self,
            scenario_name: Optional[str] = None,
            seed: Optional[int] = None,
            run_idx: Optional[int] = None,
            dispersion: Optional[bool] = None,
            env_ids: Optional[list[int]] = None,
            *,
            pos: Optional[Tensor | tuple | list] = None,
            quat: Optional[Tensor | tuple | list] = None,
            vel: Optional[Tensor | tuple | list] = None,
            ang: Optional[Tensor | tuple | list] = None,
            m_lox: Optional[Tensor | float] = None,
            m_rp1: Optional[Tensor | float] = None,
            wind_world: Optional[Tensor | tuple | list] = None,
    ) -> None:
        """Reset (a subset of) environments to a scenario preset.

        Mirrors booster-lander-simulator/scenario.c:34-82 — builds the initial
        state from a ScenarioDef and applies seeded per-env dispersion
        (altitude ±6%, velocity ±12-20 m/s, lateral ±30-250 m, tilt ±2°,
        angular-rate ±0.02 rad/s). Explicit ``pos`` / ``vel`` / ... arguments
        override the corresponding component of the preset (no dispersion
        applied to an overridden channel).

        Dispersion is the default behaviour for a fresh episode; pass
        ``dispersion=False`` for a single deterministic run of the preset.
        """
        sn = scenario_name or self.scenario_name
        if sn not in SCENARIO_PRESETS:
            raise KeyError(
                f"Unknown scenario {sn!r}. Choose one of: {list(SCENARIO_PRESETS)}"
            )
        sd = SCENARIO_PRESETS[sn]
        ids = list(range(self.N)) if env_ids is None else list(env_ids)
        idx = torch.tensor(ids, device=self.device, dtype=torch.int64)
        n = len(ids)
        dev = self.device
        use_disp = self._scenario_dispersion if dispersion is None else dispersion
        seed_i = self._scenario_seed if seed is None else seed
        run_i = self._scenario_run_idx if run_idx is None else run_idx
        pp = self.p.propellant

        # ---- RNG (numpy-free, torch.Generator on target device) ----
        def _g(section: int) -> torch.Generator:
            g = torch.Generator(device='cpu')
            g.manual_seed(int(seed_i) * 1000 + int(run_i) * 10 + int(section))
            return g

        g_h   = _g(1); g_lat = _g(2); g_ly  = _g(3)
        g_vz  = _g(4); g_vx  = _g(5); g_vy  = _g(6)
        g_tilt= _g(7); g_wx  = _g(8); g_wy  = _g(9); g_wz = _g(11)
        g_az  = _g(12); g_prop = _g(10)

        def _disps(g: torch.Generator, sigma: float) -> Tensor:
            """Batch of normal samples with sigma=0 → zeros."""
            if not use_disp or sigma <= 0:
                return torch.zeros(n, device='cpu')
            return torch.randn(n, generator=g) * sigma

        # ---- Propellant load (with ±2% dispersion) ----
        prop_scale = 1.0 + (_disps(g_prop, 0.02) if use_disp else torch.zeros(n))
        prop0_b = sd.prop0 * prop_scale
        m_rp1_b = prop0_b / (1.0 + pp.MIX_RATIO)
        m_lox_b = prop0_b - m_rp1_b

        # ---- Position: horizontal offset + altitude dispersion ----
        # Vertical: vehicle body origin sits at world-z = h0 + CoM(body frame)
        # We don't know mp.com yet so set body-origin z = h0 + analytic com.
        _, com_a, _, _, _, _ = mass_props_batch(
            m_lox_b.to(dev), m_rp1_b.to(dev), self.p,
        )
        h0_b = sd.h0 + _disps(g_h, sd.h0_sigma_frac * sd.h0)
        lx_b = sd.lateral + _disps(g_lat, sd.lat_sigma)
        ly_b = _disps(g_ly, sd.lat_sigma)

        pos_b = torch.stack([
            lx_b.to(dev),
            ly_b.to(dev),
            (h0_b + com_a.detach().cpu()).to(dev),
        ], dim=-1)

        # ---- Velocity (world frame) ----
        vx_b = _disps(g_vx, sd.velxy_sigma)
        vy_b = _disps(g_vy, sd.velxy_sigma)
        vz_b = sd.vz0 + _disps(g_vz, sd.vz0_sigma)
        vel_b = torch.stack([vx_b.to(dev), vy_b.to(dev), vz_b.to(dev)], dim=-1)

        # ---- Attitude (tilt + random azimuth) ----
        tilt_b = abs(sd.tilt0) + _disps(g_tilt, sd.tilt_sigma).abs()
        if use_disp:
            az_b = 2 * math.pi * torch.rand(n, generator=g_az)
        else:
            az_b = torch.zeros(n)
        # body-z axis in world frame
        bz_w = torch.stack([
            torch.sin(tilt_b) * torch.cos(az_b),
            torch.sin(tilt_b) * torch.sin(az_b),
            torch.cos(tilt_b),
        ], dim=-1).to(dev)
        # Convert (body_z_world) → unit quaternion from world +z to body_z_world
        q_b = torch.empty(n, 4, device=dev)
        q_b[:, 0] = 1.0 + bz_w[:, 2]
        q_b[:, 1] = -bz_w[:, 1]
        q_b[:, 2] = bz_w[:, 0]
        q_b[:, 3] = 0.0
        q_b = torch.nn.functional.normalize(q_b, dim=-1)

        # ---- Angular velocity ----
        ang_b = torch.stack([
            _disps(g_wx, sd.angvel_sigma),
            _disps(g_wy, sd.angvel_sigma),
            _disps(g_wz, sd.angvel_sigma),
        ], dim=-1).to(dev)

        # ---- Apply explicit overrides (no dispersion applied to overridden channels) ----
        def _override(field, val):
            if val is None: return field
            t = torch.as_tensor(val, dtype=torch.float32, device=dev)
            if t.ndim == 1 and t.shape[0] == field.shape[-1]:
                return t.unsqueeze(0).repeat(n, 1)
            return t.to(dev).float()
        pos_b = _override(pos_b, pos)
        q_b = _override(q_b, quat)
        vel_b = _override(vel_b, vel)
        ang_b = _override(ang_b, ang)
        if m_lox is not None:
            if isinstance(m_lox, (int, float)): m_lox_b = torch.full((n,), float(m_lox))
            else: m_lox_b = torch.as_tensor(m_lox, dtype=torch.float32)
        if m_rp1 is not None:
            if isinstance(m_rp1, (int, float)): m_rp1_b = torch.full((n,), float(m_rp1))
            else: m_rp1_b = torch.as_tensor(m_rp1, dtype=torch.float32)
        m_lox_b = m_lox_b.to(dev); m_rp1_b = m_rp1_b.to(dev)

        # ---- Wind (power-law profile, mean reference at deck height) ----
        if wind_world is not None:
            w_b = torch.as_tensor(wind_world, dtype=torch.float32, device=dev)
            if w_b.ndim == 1: w_b = w_b.unsqueeze(0).repeat(n, 1)
            self.wind_world[idx] = w_b
        else:
            self.wind_world[idx] = 0.0

        # ---- Push to Genesis ----
        self.vehicle.set_pos(pos_b, envs_idx=idx)
        self.vehicle.set_quat(q_b, envs_idx=idx)
        self.vehicle.set_dofs_velocity(torch.cat([vel_b, ang_b], dim=-1), envs_idx=idx)

        # ---- Engine / actuator reset (mirrors legacy reset defaults) ----
        eng = self.engine; act = self.act
        ep = self.p.engine; pp_ = self.p.propellant
        eng.throttle_act[idx] = ep.ENG_THR_MIN
        eng.throttle_cmd[idx] = ep.ENG_THR_MIN
        eng.eng_on[idx] = False
        eng.ignition_timer[idx] = -1.0
        eng.relights_used[idx] = 0
        eng.g0_cmd[idx] = 0.0; eng.g1_cmd[idx] = 0.0
        eng.g0_act[idx] = 0.0; eng.g1_act[idx] = 0.0
        eng.gr0[idx] = 0.0; eng.gr1[idx] = 0.0
        eng.m_lox[idx] = m_lox_b
        eng.m_rp1[idx] = m_rp1_b
        eng.m_rcs_n2[idx] = pp_.RCS_N2

        act.fins_deployed[idx] = bool(sd.fins)
        act.fin_deflection[idx] = 0.0
        act.rcs_on[idx] = False
        act.tau_rcs_body_cmd[idx] = 0.0
        act.legs_deployed[idx] = False

        # ---- Episode bookkeeping ----
        self.count[idx] = 0
        self.reward[idx] = 0.0
        self.terminated[idx] = False
        self.truncated[idx] = False
        self.complete[idx] = False
        self.contact_pad[idx] = False
        self.contact_vz_first[idx] = 0.0
        self.touch_any[idx] = False
        self.touch_vz_first[idx] = 0.0
        for k in self._last_reward_components:
            self._last_reward_components[k][idx] = 0.0

        # Push composite mass + inertia + CoM to Genesis
        self._apply_mass_props(envs_idx=idx)

    # ------------------------------------------------------------------
    # Reset (legacy passthrough — forwards to reset_from_scenario)
    # ------------------------------------------------------------------
    def reset(self,
              env_ids: Optional[list[int]] = None,
              pos=None, quat=None, vel=None, ang=None,
              m_lox=None, m_rp1=None, wind_world=None):
        """Legacy reset API — forwards to reset_from_scenario with overrides."""
        self.reset_from_scenario(
            env_ids=env_ids,
            pos=pos, quat=quat, vel=vel, ang=ang,
            m_lox=m_lox, m_rp1=m_rp1, wind_world=wind_world,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_mass_props(self, envs_idx=None):
        """Push composite mass / inertia / CoM into Genesis.

        The URDF morph is built with align=False, so the link frame keeps the
        authored body axes and Genesis lets us write mass, inertia (about the
        CoM, in body axes) and CoM offset at runtime. Passing align=True would
        re-anchor the frame on the CoM and make these writes raise.
        """
        M, com_body, I_tr, I_ax, _, _ = mass_props_batch(
            self.engine.m_lox, self.engine.m_rp1, self.p
        )
        idx = torch.arange(self.N, device=self.device) if envs_idx is None else envs_idx
        n = idx.numel()

        # (n, 1) mass
        self.vehicle.set_links_mass(M[idx, None], links_idx_local=0, envs_idx=idx)

        # (n, 1, 3, 3) diagonal inertia about the CoM, body axes
        inert = torch.zeros(n, 1, 3, 3, device=self.device, dtype=M.dtype)
        inert[:, 0, 0, 0] = I_tr[idx]
        inert[:, 0, 1, 1] = I_tr[idx]
        inert[:, 0, 2, 2] = I_ax[idx]
        self.vehicle.set_links_inertia(inert, links_idx_local=0, envs_idx=idx)

        # (n, 1, 3) CoM offset from the link origin (body frame, z along the axis)
        com = torch.zeros(n, 1, 3, device=self.device, dtype=M.dtype)
        com[:, 0, 2] = com_body[idx]
        self.vehicle.set_links_COM(com, links_idx_local=0, envs_idx=idx)

    # ------------------------------------------------------------------
    # Substep (single dt=0.002 s, 500 Hz)
    # ------------------------------------------------------------------
    def _substep(self):
        p = self.p; ep = p.engine
        eng = self.engine; act = self.act

        # Genesis state (world frame)
        pos = self.vehicle.get_pos()          # (N, 3)
        quat = self.vehicle.get_quat()        # (N, 4) wxyz
        vel = self.vehicle.get_vel()          # (N, 3)
        ang = self.vehicle.get_ang()          # (N, 3)

        # --- 1. Ignition ramp (smoothstep TEA-TEB flash) ---
        eng.ignition_timer, ramp = ignition_ramp_tick(
            eng.ignition_timer, eng.eng_on, self.dt, p
        )

        # --- 2. Throttle dynamics (first-order lag + shutdown decay) ---
        thr_err = eng.throttle_cmd - eng.throttle_act
        thr_rate = thr_err / ep.ENG_THR_TAU
        thr_rate = thr_rate.clamp(-ep.ENG_THR_RATE, ep.ENG_THR_RATE)
        eng.throttle_act = (eng.throttle_act + thr_rate * self.dt).clamp(0.0, 1.0)
        eng.throttle_act = torch.where(
            eng.eng_on,
            eng.throttle_act,
            eng.throttle_act * torch.exp(
                torch.tensor(-self.dt / ep.ENG_SHUT_TAU, device=self.device)
            ),
        )

        # --- 3. Gimbal second-order PD (servo tracks the commanded deflection) ---
        gacc0, gacc1 = gimbal_deriv_batch(
            eng.g0_act, eng.g1_act, eng.gr0, eng.gr1,
            eng.g0_cmd, eng.g1_cmd, p,
        )
        eng.gr0 = eng.gr0 + gacc0 * self.dt
        eng.gr1 = eng.gr1 + gacc1 * self.dt
        eng.g0_act = eng.g0_act + eng.gr0 * self.dt
        eng.g1_act = eng.g1_act + eng.gr1 * self.dt
        eng.g0_act, eng.g1_act, eng.gr0, eng.gr1 = gimbal_saturate(
            eng.g0_act, eng.g1_act, eng.gr0, eng.gr1, p
        )

        # --- 4. Atmosphere + engine thrust / mdot ---
        _, pressure, _, _ = atmosphere_eval(pos[:, 2], p)
        thrust_per, mdot_total, mdot_lox, mdot_rp1, _ = engine_deriv_batch(
            eng.throttle_cmd, eng.throttle_act, ramp, eng.eng_on, pressure, p
        )

        # --- 5. Fuel depletion + auto-shutdown ---
        eng.m_lox = (eng.m_lox - mdot_lox * self.dt).clamp_min(0.0)
        eng.m_rp1 = (eng.m_rp1 - mdot_rp1 * self.dt).clamp_min(0.0)
        deplete = (eng.m_lox <= 1.0) | (eng.m_rp1 <= 1.0)
        eng.eng_on = eng.eng_on & ~deplete

        # RCS command is zero once the N2 bottles run dry
        rcs_cmd = torch.where(
            (eng.m_rcs_n2 > 0.0).unsqueeze(-1), act.tau_rcs_body_cmd,
            torch.zeros_like(act.tau_rcs_body_cmd),
        )

        # --- 6. Fins / legs deploy logic ---
        act.fins_deployed = (pos[:, 2] < p.fins.FIN_DEPLOY_H) & (
            eng.eng_on | (pos[:, 2] < p.legs.LEG_DEPLOY_H)
        )
        act.legs_deployed = pos[:, 2] < p.legs.LEG_DEPLOY_H

        # --- 7. Assemble forces/torques (body frame, torques about the CoM) ---
        ft = assemble_forces_torques(
            eng.m_lox, eng.m_rp1,
            pos, quat, vel, ang,
            eng.throttle_act, eng.throttle_cmd,
            ramp, eng.eng_on,
            eng.g0_act, eng.g1_act,
            act.fins_deployed, act.fin_deflection,
            rcs_cmd, self.wind_world,
            dt=self.dt, p=p,
        )

        # --- 8. RCS propellant depletion ---
        eng.m_rcs_n2 = (eng.m_rcs_n2 - ft.mdot_rcs_n2 * self.dt).clamp_min(0.0)

        # --- 9. Open-system İ·ω torque (source dynamics.c:240-242) ---
        # Genesis integrates a closed-system rigid body, so the -İ·ω term from
        # fuel leaving the vehicle must be injected explicitly. It is a pure
        # couple, hence frame-independent: -İ·ω in body axes.
        _, _, I_tr, I_ax, Idot_tr_coeff, Idot_ax_coeff = mass_props_batch(
            eng.m_lox, eng.m_rp1, p
        )
        w_body = rotate_world_to_body(quat, ang)
        Idot_tr = Idot_tr_coeff[0] * (-mdot_lox) + Idot_tr_coeff[1] * (-mdot_rp1)
        Idot_ax = Idot_ax_coeff * (-mdot_total)
        I_dot_omega = torch.stack(
            [Idot_tr * w_body[:, 0], Idot_tr * w_body[:, 1], Idot_ax * w_body[:, 2]],
            dim=-1,
        )

        # --- 10. Apply to Genesis: F in world, pure body-frame torque at the CoM ---
        F_total_world = rotate_body_to_world(quat, ft.F_body)
        T_total_body = ft.T_body - I_dot_omega
        self.vehicle.apply_links_external_wrench(
            force=F_total_world,
            torque=T_total_body,
            links_idx_local=0,
            ref=gs.link_ref_frame.link_COM, local=True,
        )

        # --- 11. Push updated composite mass / inertia / CoM ---
        self._apply_mass_props()

        # --- 12. Genesis substep (integrate rigid body + contact) ---
        self.scene.step()

    # ------------------------------------------------------------------
    # Step (one policy tick = 10 substeps @ 500 Hz → 50 Hz)
    # ------------------------------------------------------------------
    def step(self,
             throttle_cmd: Optional[Tensor] = None,
             eng_on: Optional[Tensor] = None,
             gimbal_cmd: Optional[Tensor] = None,
             rcs_torque_cmd: Optional[Tensor] = None,
             fin_cmd: Optional[Tensor] = None,
             ) -> dict:
        """Run n_sub physics substeps. Return state + reward + done flags."""
        N = self.N
        dev = self.device

        if throttle_cmd is None:
            throttle_cmd = torch.full((N,), self.p.engine.ENG_THR_MIN, device=dev)
        if eng_on is None:
            eng_on = torch.zeros(N, dtype=torch.bool, device=dev)
        if gimbal_cmd is None:
            gimbal_cmd = torch.zeros(N, 2, device=dev)
        if rcs_torque_cmd is None:
            rcs_torque_cmd = torch.zeros(N, 3, device=dev)
        if fin_cmd is None:
            fin_cmd = torch.zeros(N, self.p.fins.FIN_COUNT, device=dev)

        gmax = self.p.engine.ENG_GIMBAL_MAX
        gimbal_cmd = gimbal_cmd.to(dev).clamp(-gmax, gmax)
        fmax = self.p.fins.FIN_DEFL_MAX
        fin_cmd = fin_cmd.to(dev).clamp(-fmax, fmax)

        self.engine.throttle_cmd = throttle_cmd.to(dev)
        self.engine.eng_on = eng_on.to(dev)
        # Commanded TVC; the gimbal servo (substep 3) drives g0_act/g1_act toward it
        self.engine.g0_cmd = gimbal_cmd[:, 0].clone()
        self.engine.g1_cmd = gimbal_cmd[:, 1].clone()
        self.act.tau_rcs_body_cmd = rcs_torque_cmd.to(dev)
        self.act.fin_deflection = fin_cmd

        # Snapshots at START of policy step
        active = ~(self.terminated | self.truncated)
        pos_s = self.vehicle.get_pos().clone()
        vel_s = self.vehicle.get_vel().clone()
        # contact_pad tracks "had ANY contact in this policy step" (for shaping)
        self.contact_pad.fill_(False)
        # contact_vz_first tracks "vz at first contact in THIS policy step"
        self.contact_vz_first.fill_(0.0)
        # touch_any and touch_vz_first persist across steps — only reset on episode reset

        # --- 10 substeps ---
        for _ in range(self.n_sub):
            vel_before = self.vehicle.get_vel()[:, 2].clone()   # PRE-substep vz (world)
            self._substep()

            # Per-substep contact check (any env, while active)
            contacts = self.vehicle.get_links_net_contact_force()  # (N, N_links, 3)
            fc_norm = contacts.norm(dim=-1).squeeze(-1)          # (N,) after squeeze
            sub_contact = fc_norm > 1e6                           # significant contact
            # Use PRE-substep velocity (before contact impulse) — AUDIT-5 fix
            vz_now = vel_before

            first_touch_episode = sub_contact & ~self.touch_any & active  # FIRST EVER in episode
            self.contact_vz_first = torch.where(
                sub_contact & ~self.contact_pad,   # first within THIS policy step
                vz_now, self.contact_vz_first,
            )
            self.contact_pad |= sub_contact & active                     # ANY contact this step
            self.touch_vz_first = torch.where(first_touch_episode, vz_now, self.touch_vz_first)
            self.touch_any |= sub_contact & active

        # --- Post-step state snapshot ---
        pos_e = self.vehicle.get_pos().clone()
        quat_e = self.vehicle.get_quat().clone()
        vel_e = self.vehicle.get_vel().clone()
        ang_e = self.vehicle.get_ang().clone()
        w_body = rotate_world_to_body(quat_e, ang_e)

        self.count += 1

        # --- Termination logic ---
        # Bounds
        bounds = (pos_e[:, :2].norm(dim=-1) > self.thr['bounds_xy']) | \
                 (pos_e[:, 2] > self.thr['bounds_z_high'])          # z high = escaping
        bounds &= active

        # Below ground (fatal)
        below_ground = (pos_e[:, 2] < self.thr['bounds_z_low']) & active

        # Pad contact thresholds — use FIRST impact velocity for fatal check
        body_speed = vel_e.norm(dim=-1)
        touch_active = self.touch_any & active
        # Impact velocity at first contact EVER in this episode (pre-contact)
        impact_speed = self.touch_vz_first.abs()   # world-frame descent speed at impact
        fatal_pad = touch_active & (
            (impact_speed > self.thr['fatal_pad_speed']) |
            (w_body.norm(dim=-1) > self.thr['fatal_pad_spin'])
        )
        # Success: post-contact settled state (body speed low, spin low, tilt low)
        success_pad = touch_active & ~fatal_pad & \
                      (body_speed < self.thr['success_vz_max']) & \
                      (w_body.norm(dim=-1) < self.thr['success_spin_max']) & \
                      (_body_tilt_from_quat(quat_e) < self.thr['success_tilt_max'])

        self.complete |= success_pad & active

        # Terminated = fatal OR bounds OR below ground
        self.terminated |= (fatal_pad | bounds | below_ground) & active

        # Truncated = max episode length
        active2 = ~(self.terminated | self.truncated)
        self.truncated |= (self.count >= self.max_steps) & active2

        # --- Reward ---
        r_total, r_comps = falcon9_shaping(
            self.reward_cfg,
            pos_s, pos_e,
            vel_e[:, 2], vel_s[:, 2],
            quat_e, w_body,
            self.contact_pad,
            self.contact_vz_first,
            success_pad,                 # newly complete this step
        )
        self.reward = r_total
        self._last_reward_components = {k: v.detach().clone() for k, v in r_comps.items()}

        # --- Read state dict ---
        state = self._read_state()
        state['reward'] = self.reward.detach().clone()
        state['terminated'] = self.terminated.detach().clone()
        state['truncated'] = self.truncated.detach().clone()
        state['complete'] = self.complete.detach().clone()
        state['touch_any'] = self.touch_any.detach().clone()
        state['body_speed'] = body_speed.detach().clone()
        state['count'] = self.count.detach().clone()
        return state

    def _read_state(self) -> dict:
        return {
            'pos_world':        self.vehicle.get_pos().detach().clone(),
            'quat_wxyz_world':  self.vehicle.get_quat().detach().clone(),
            'vel_world':        self.vehicle.get_vel().detach().clone(),
            'ang_world':        self.vehicle.get_ang().detach().clone(),
            'm_lox':            self.engine.m_lox.detach().clone(),
            'm_rp1':            self.engine.m_rp1.detach().clone(),
            'm_rcs_n2':         self.engine.m_rcs_n2.detach().clone(),
            'throttle':         self.engine.throttle_act.detach().clone(),
            'eng_on':           self.engine.eng_on.detach().clone(),
            'gimbal':           torch.stack(
                                    [self.engine.g0_act, self.engine.g1_act], dim=-1
                                ).detach().clone(),
            'gimbal_rate':      torch.stack(
                                    [self.engine.gr0, self.engine.gr1], dim=-1
                                ).detach().clone(),
            'fins_deployed':    self.act.fins_deployed.detach().clone(),
            'legs_deployed':    self.act.legs_deployed.detach().clone(),
            'wind_world':       self.wind_world.detach().clone(),
        }

    def close(self):
        pass
