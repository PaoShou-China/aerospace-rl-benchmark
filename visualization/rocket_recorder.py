"""Opt-in, single-episode observer; never advances or changes the plant.

The first version intentionally supports one replica for diagnostic collection.
GPU-to-host snapshots are paid only when this observer is explicitly attached.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import subprocess

import numpy as np

from .rocket_data import Episode, MODELS, ROOT, SCHEMA, asset_geometry, foot_height


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def scalar(value):
    return array(value).reshape(-1)[0].item()


class RocketRecorder:
    def __init__(self, env, *, seed, source, config, terminal_height=20.):
        if env.num_envs != 1:
            raise ValueError('Diagnostic recorder requires num_envs=1; batch recording is not implemented')
        if not math.isfinite(terminal_height) or terminal_height <= 0:
            raise ValueError('terminal_height must be positive')
        self.samples, self.events, self.policy_steps = [], [], []
        self.seen, self.reasons = set(), set()
        self.policy_index = 0
        self.ended = self.stop_samples = False
        p = env.plant
        capacity = float(p.params['booster_params']['total_fuel'])
        self.dt = float(p._geometry._dt)
        self.policy_hz, self.inner_steps = int(env.agent_hz), int(env.inner_steps)
        if not np.isclose(self.dt * 2 * self.inner_steps, 1/self.policy_hz):
            raise ValueError('Control/physics rate contract changed')
        versions = {}
        for dist in ('genesis-world', 'torch', 'numpy'):
            try:
                versions[dist] = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                versions[dist] = 'unavailable'
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = 'unavailable-archive-without-git', None
        self.metadata = dict(task='rocket', source=source, source_commit=commit, source_dirty=dirty,
            seed=seed, env_id=0, episode_id=f'seed-{seed}-first', device=str(env.device),
            solver_dtype=str(p.drone.get_pos().dtype), versions=versions,
            effective_env_config=dict(env.cfg), requested_config=config,
            position_frame='ENU', velocity_frame='ENU', angular_velocity_frame='ENU',
            quaternion_order='wxyz', orientation='body_to_world', units='SI',
            policy_hz=self.policy_hz, control_hz=self.policy_hz*self.inner_steps,
            physics_hz=1/self.dt, sample_rate_hz=1/self.dt,
            episode_limit_steps=int(env.max_steps), warmup_excluded=True,
            terminal_height_m=float(terminal_height), terminal_definition='first descending foot clearance <= threshold',
            fuel_fraction_reference='full_tank', fuel_capacity_kg=capacity,
            initial_fuel_mass_kg=float(scalar(p.fuel))*capacity,
            gimbal_range_rad=math.radians(p.params['booster_params']['gimbal_range_degrees']))
        self.metadata.update(asset_geometry())
        assets = [MODELS/'vehicles/rocket/rocket.urdf', MODELS/'vehicles/rocket/rocket.yaml', MODELS/'landing_pad.urdf']
        self.metadata['assets_sha256'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in assets}
        self._snapshot(env, 0.)

    def _event(self, kind, time, **values):
        if kind in self.seen:
            return
        self.seen.add(kind)
        self.events.append(dict(t_s=float(time), type=kind, **values))

    def _snapshot(self, env, time):
        p = env.plant
        pos, q, v, w = (array(a)[0] for a in p.state())
        s = dict(t_s=float(time), position_m=pos.tolist(), quaternion_wxyz=q.tolist(),
            linear_velocity_mps=v.tolist(), angular_velocity_radps=w.tolist(),
            fin_actual=array(p.surfaces.actuation)[0].tolist(),
            gimbal_rad=(array(p.gimbal)[0]*self.metadata['gimbal_range_rad']).tolist(),
            throttle_actual=float(scalar(p.throttle)), ignited=bool(scalar(p.ignited)),
            fuel_fraction=float(scalar(p.fuel)), fuel_mass_kg=float(scalar(p.fuel))*self.metadata['fuel_capacity_kg'],
            pad_contact=bool(scalar(p.pad_contact)), ground_contact=bool(scalar(p.ground_contact)),
            complete=bool(scalar(env.complete)), terminated=bool(scalar(env.terminated)), truncated=bool(scalar(env.truncated)))
        self.samples.append(s)
        if v[2] < 0 and foot_height(s, self.metadata) <= self.metadata['terminal_height_m']:
            self._event('terminal_start', time)
        return s

    def begin_policy(self, env, raw, mapped):
        if self.ended:
            return
        self.policy_index += 1
        self.policy_steps.append(dict(t_s=(self.policy_index-1)/self.policy_hz,
            observation=array(env.obs_buf)[0].tolist(), action_raw=array(raw)[0].tolist(),
            action_mapped=array(mapped)[0].tolist()))

    def after_physics(self, env, active, pre_state, inner, substep):
        if self.ended or self.stop_samples or not bool(scalar(active)):
            return
        time = (self.policy_index-1)/self.policy_hz + (inner*2+substep+1)*self.dt
        s = self._snapshot(env, time)
        for field, kind in (('pad_contact', 'pad_contact'), ('ground_contact', 'ground_contact')):
            if s[field]:
                self._event(kind, time, time_resolution_s=self.dt,
                    velocity_pre_physics_world_mps=array(pre_state[2])[0].tolist(),
                    velocity_post_physics_world_mps=s['linear_velocity_mps'])

    def after_control(self, env, active, inner, speed, spin, tilt, reasons):
        if self.ended or self.stop_samples or not bool(scalar(active)):
            return
        time = (self.policy_index-1)/self.policy_hz + (inner+1)*2*self.dt
        if bool(scalar(env.plant.pad_contact)):
            self._event('pad_contact_gate', time, gate_speed_mps=float(scalar(speed)),
                        gate_spin_radps=float(scalar(spin)), tilt_rad=float(scalar(tilt)))
        for key in ('complete', 'terminated', 'truncated'):
            self.samples[-1][key] = bool(scalar(getattr(env, key)))
        if self.samples[-1]['complete']:
            self._event('settled', time)
        self.reasons.update(k for k, value in reasons.items() if bool(scalar(value)))
        if self.samples[-1]['terminated']:
            self._event('terminated', time, reasons=sorted(self.reasons))
            self.stop_samples = True

    def after_policy(self, env):
        """Called before reset; preserve the last physical contact flags."""
        if self.ended:
            return
        self.policy_steps[-1]['reward'] = float(scalar(env.reward))
        self.policy_steps[-1]['reward_components_cumulative'] = {k: float(scalar(v)) for k, v in env.reward_components.items()}
        for key in ('complete', 'terminated', 'truncated'):
            self.samples[-1][key] = bool(scalar(getattr(env, key)))
        if bool(scalar(env.reset_buf)):
            if self.samples[-1]['truncated']:
                self._event('time_limit', self.samples[-1]['t_s'])
            self.ended = True

    def episode(self):
        if len(self.samples) < 2:
            raise ValueError('No rollout samples were recorded')
        if not self.ended:
            self._event('collection_limit', self.samples[-1]['t_s'])
        last = self.samples[-1]
        result = dict(complete=last['complete'], terminated=last['terminated'], truncated=last['truncated'],
                      collection_limited=not self.ended, failure_reasons=sorted(self.reasons))
        return Episode(dict(schema=SCHEMA, demo=False, metadata=self.metadata,
                            samples=self.samples, events=sorted(self.events, key=lambda e: e['t_s']),
                            policy_steps=self.policy_steps, result=result))
