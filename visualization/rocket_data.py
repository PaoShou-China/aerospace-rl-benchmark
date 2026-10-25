"""Versioned rocket logs, geometry calibration and timestamp-based metrics.

Only NumPy is needed here. Simulation and rendering dependencies are lazy.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = 'aerospace-visual-state/v1'
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'environments/common/original_pyflyt/models'
VECTOR_FIELDS = {'position_m': 3, 'quaternion_wxyz': 4, 'linear_velocity_mps': 3,
                 'angular_velocity_radps': 3, 'gimbal_rad': 2, 'fin_actual': 4}
FLAGS = ('pad_contact', 'ground_contact', 'complete', 'terminated', 'truncated')


def rotation(q):
    """Body-to-world rotation from a normalized-or-not wxyz quaternion."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid quaternion')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def slerp(a, b, u):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    angle = math.acos(dot)
    return (math.sin((1-u)*angle)*a + math.sin(u*angle)*b) / math.sin(angle)


def asset_geometry(models=MODELS):
    """Read actual fixed landing-leg corners; exclude the decorative flame."""
    rocket = Path(models) / 'vehicles/rocket/rocket.urdf'
    root = ET.fromstring(rocket.read_text().split('</robot>')[0] + '</robot>')
    corners = []
    for joint in root.findall('joint'):
        child = joint.find('child').get('link')
        if not child.startswith('leg_'):
            continue
        if joint.get('type') != 'fixed' or joint.find('parent').get('link') != 'base_link':
            raise ValueError('Landing-leg topology changed; recalibrate the display geometry')
        origin = joint.find('origin')
        p = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        r, pitch, yaw = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        R = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
        collision = root.find(f"link[@name='{child}']/collision")
        if collision.find('origin') is not None:
            co = collision.find('origin')
            if co.get('xyz', '0 0 0') != '0 0 0' or co.get('rpy', '0 0 0') != '0 0 0':
                raise ValueError('Landing-leg collision origin changed; recalibrate geometry')
        size = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
        corners.extend((p + R @ (np.asarray(signs)*size/2)).tolist()
                       for signs in itertools.product((-1, 1), repeat=3))
    if not corners:
        raise ValueError('No landing-leg geometry found')
    pad = ET.parse(Path(models) / 'landing_pad.urdf').getroot()
    cylinder = pad.find('link/collision/geometry/cylinder')
    pad_center_z = .1  # matches TensorPlant / original_backend at this source version
    pad_top = pad_center_z + float(cylinder.get('length'))/2
    offset = -float(np.min(np.asarray(corners)[:, 2]))
    return dict(foot_points_body_m=corners, model_foot_offset_m=offset,
                pad_position_m=[0., 0., pad_center_z], pad_surface_z_m=pad_top,
                pad_radius_m=float(cylinder.get('radius')),
                target_position_m=[0., 0., pad_top + offset])


def foot_height(sample, metadata):
    p = np.asarray(sample['position_m'], float)
    corners = np.asarray(metadata['foot_points_body_m'], float)
    world = corners @ rotation(sample['quaternion_wxyz']).T + p
    pad = np.asarray(metadata['pad_position_m'])
    on_pad = np.linalg.norm(world[:, :2]-pad[:2], axis=1) <= metadata['pad_radius_m']
    surface = np.where(on_pad, metadata['pad_surface_z_m'], 0.)
    return float(np.min(world[:, 2]-surface))


def validate(data):
    if data.get('schema') != SCHEMA:
        raise ValueError(f'Expected schema {SCHEMA}')
    m = data.get('metadata', {})
    if m.get('position_frame') != 'ENU' or m.get('quaternion_order') != 'wxyz':
        raise ValueError('Only ENU / wxyz rocket logs are supported')
    samples = data.get('samples', [])
    if len(samples) < 2:
        raise ValueError('At least two samples are required')
    times = np.asarray([s['t_s'] for s in samples], float)
    if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Sample times must start at zero and strictly increase')
    for key, n in VECTOR_FIELDS.items():
        a = np.asarray([s[key] for s in samples], float)
        if a.shape != (len(samples), n) or not np.isfinite(a).all():
            raise ValueError(f'Invalid {key}')
    norms = np.linalg.norm([s['quaternion_wxyz'] for s in samples], axis=1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-10):
        raise ValueError('Invalid quaternion norm')
    for s in samples:
        for key in ('fuel_fraction', 'fuel_mass_kg', 'throttle_actual'):
            if not math.isfinite(s[key]):
                raise ValueError(f'Nonfinite {key}')
        if s['fuel_mass_kg'] < 0 or not 0 <= s['fuel_fraction'] <= 1:
            raise ValueError('Fuel is outside its physical domain')
        if any(type(s[key]) is not bool for key in FLAGS):
            raise ValueError('Contact and result flags must be booleans')
    for e in data.get('events', []):
        if not isinstance(e.get('type'), str) or not math.isfinite(e.get('t_s', float('nan'))) or not 0 <= e['t_s'] <= times[-1]:
            raise ValueError('Event time/type is invalid')
    target = np.asarray(m['target_position_m'], float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Invalid target')
    for key in ('fuel_capacity_kg', 'initial_fuel_mass_kg', 'terminal_height_m'):
        if not math.isfinite(m[key]) or m[key] <= 0:
            raise ValueError(f'Invalid {key}')
    feet = np.asarray(m['foot_points_body_m'], float)
    pad = np.asarray(m['pad_position_m'], float)
    if feet.ndim != 2 or feet.shape[1] != 3 or not len(feet) or not np.isfinite(feet).all():
        raise ValueError('Invalid foot geometry')
    if pad.shape != (3,) or not np.isfinite(pad).all() or not math.isfinite(m['pad_surface_z_m']) or not math.isfinite(m['pad_radius_m']) or m['pad_radius_m'] <= 0:
        raise ValueError('Invalid pad geometry')
    if not np.isclose(m['initial_fuel_mass_kg'], samples[0]['fuel_mass_kg']):
        raise ValueError('Initial fuel mass disagrees with the first sample')
    if m.get('velocity_frame', 'ENU') != 'ENU':
        raise ValueError('Velocity must be expressed in ENU')
    return data


class Episode:
    def __init__(self, data):
        self.data = validate(data)
        self.metadata = data['metadata']
        self.times = np.array([s['t_s'] for s in data['samples']])

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            path = path / 'trajectory.json'
        return cls(json.loads(path.read_text(encoding='utf-8')))

    @property
    def duration(self):
        return float(self.times[-1])

    def event_time(self, *kinds):
        return min((e['t_s'] for e in self.data.get('events', []) if e['type'] in kinds), default=None)

    def at(self, time):
        time = float(np.clip(time, 0, self.duration))
        j = min(int(np.searchsorted(self.times, time, side='right')), len(self.times)-1)
        i = max(0, j-1)
        a, b = self.data['samples'][i], self.data['samples'][j]
        u = (time-self.times[i])/(self.times[j]-self.times[i]) if j != i else 0.
        # Flags are left-held between samples, exact at event/sample boundaries.
        out = dict(b if time == self.times[j] else a)
        out['t_s'] = time
        for key in VECTOR_FIELDS:
            out[key] = (slerp(a[key], b[key], u) if key == 'quaternion_wxyz' else
                        np.asarray(a[key])*(1-u)+np.asarray(b[key])*u).tolist()
        for key in ('fuel_fraction', 'fuel_mass_kg', 'throttle_actual'):
            out[key] = a[key]*(1-u)+b[key]*u
        return out

    def metrics(self, time):
        s = self.at(time)
        e = np.asarray(s['position_m'])-self.metadata['target_position_m']
        contact = self.event_time('pad_contact', 'ground_contact')
        settled = self.event_time('settled')
        terminal = self.event_time('terminal_start')
        visible_contact = contact if contact is not None and s['t_s'] >= contact else None
        stop = visible_contact if visible_contact is not None else s['t_s']
        initial = self.metadata['initial_fuel_mass_kg']
        return dict(sample=s, position_error_m=e.tolist(), horizontal_error_m=float(np.linalg.norm(e[:2])),
                    error_3d_m=float(np.linalg.norm(e)), speed_mps=float(np.linalg.norm(s['linear_velocity_mps'])),
                    foot_height_m=foot_height(s, self.metadata), fuel_used_kg=initial-s['fuel_mass_kg'],
                    fuel_used_percent=100*(initial-s['fuel_mass_kg'])/initial,
                    landing_elapsed_s=stop, contact_time_s=visible_contact,
                    settled_time_s=settled if settled is not None and s['t_s'] >= settled else None,
                    terminal_elapsed_s=max(0., stop-terminal) if terminal is not None and s['t_s'] >= terminal else None)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'trajectory.json').write_text(json.dumps(self.data, indent=2, allow_nan=False), encoding='utf-8')
        (directory/'manifest.json').write_text(json.dumps(dict(schema=SCHEMA, demo=self.data.get('demo'),
            metadata=self.metadata), indent=2, allow_nan=False), encoding='utf-8')
        arrays = {k: np.asarray([s[k] for s in self.data['samples']]) for k in self.data['samples'][0]}
        np.savez_compressed(directory/'states.npz', **arrays)
        (directory/'events.jsonl').write_text(''.join(json.dumps(e, allow_nan=False)+'\n' for e in self.data.get('events', [])), encoding='utf-8')
        summary = dict(self.metrics(self.duration))
        summary.update(self.data.get('result', {}))
        (directory/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


def playback_times(duration, fps=30, speed=1., start=0., end=None):
    """Exclude the end boundary; encoded duration differs by < one video frame."""
    end = duration if end is None else end
    if not all(math.isfinite(v) for v in (fps, speed, start, end)) or fps <= 0 or speed <= 0 or not 0 <= start < end <= duration:
        raise ValueError('Invalid playback range, fps or speed')
    n = max(1, int(math.ceil((end-start)*fps/speed)))
    return start + np.arange(n)*speed/fps
