"""Genesis offscreen visualization of one baseline episode per task.

Usage: python visualize.py quadx|fixedwing|rocket|missile|satellite|transfer|docking|rendezvous

Runs the task's classical baseline on a single CPU replica (same env
construction as eval.py/baseline_eval.py, incl. results/<task>/config.json
restoration) while recording the controlled object's pose each agent step,
then replays the recording in a visual-only Genesis scene and renders an
mp4 to results/<task>/vis.mp4.

Scene conventions per task:
  quadx       cf2x URDF on a dark slate ground plane, opaque green sphere at
              the active waypoint (PyFlyt marker scaling), yellow nose arrow
              along body +x, translucent prop disks over the four motor mounts
              (display-only, no physics), and a fixed PyFlyt debug-view
              camera scaled to the ~9 cm drone
  fixedwing   fixedwing URDF, red spheres at the 4 waypoints, camera follows
  rocket      rocket URDF + landing pad on a ground plane, camera follows
  satellite   recognizable bus (grey body, +/-z equipment compartments, +x
              sensor face, -x nozzle, top antenna, +/-y solar panels with a
              light grid), body-frame axis triad (X red / Y green / Z blue)
              rotating with the bus, target-attitude triad (yellow) driven by
              the recorded target quaternion, cyan CMG rotor-axis cylinders
              driven by the recorded gimbal angles (the only CMG state the
              env exposes; rotor angles/speeds are NOT provided so no rotor
              spin is faked), red ghost box at the target attitude, static
              star decoration and a FIXED oblique camera (never follows the
              tumbling body), all driven by the recorded wxyz quaternions
  transfer    Earth sphere + satellite (1 unit = 1000 km), 3D view fixed on
              Earth covering the full orbit: green dashed initial-orbit
              ellipse, cyan dotted target (circularized at the initial
              apogee radius), red/blue perigee/apogee markers of the initial
              orbit, orange trail of the real state history, and a 2000 km
              scale bar. The satellite box is enlarged for visibility; the
              true spacecraft scale is the scale bar.
  docking     target station (grey body + ±y-attached solar panels) with a
              docking port on the +x face (green ring edge, centre marker,
              translucent approach-axis cylinder); the orange chaser keeps
              its docking face turned toward the target with its own port
              ring + centre marker, a dashed centre-to-centre line of sight,
              a true-state velocity arrow (100 m per m/s) and an approach
              trail. Overlay burns the live range / speed (real state,
              metres) plus the actual success/fail thresholds; a labeled
              CLOSE-UP inset shows the port region; the terminal banner is
              the env's own termination outcome (SUCCESS / FAILURE / TIMEOUT)
  rendezvous  docking scene plus LVLH display furniture: radial/along-track/
              cross-track axis bars, range rings (250/500/1000/2000 m, all in
              metres), green goal ball (r = 10 m, the real pos_goal), the
              red keep-out sphere (r = 5 m) and a quantized velocity arrow
              (arrow pool in metres, 100 m per m/s of true relative speed),
              with the live range / speed values burned into each frame.
              The camera keeps a FIXED direction in the LVLH frame (no
              rotation with the chaser); only its distance follows the
              range so target and chaser always share the frame.
  missile     missile URDF on the km-scale 4 km-altitude engagement, red
              marker for the kinematic target aircraft (posed per frame from
              the recorded target state), engagement trail, and a follow
              camera looking at the missile-target mid-point so both stay
              framed throughout the intercept
Trails are preallocated small spheres; unused markers are parked far away.
The render scene shares no physics with the sim: it is built with zero
gravity and never stepped, entities are driven with set_pos/set_quat only.
"""
import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(_ROOT))          # exposes environments.* and algorithms.*

import numpy as np
import torch
import genesis as gs

from environments.multirotor import QuadXWaypointsEnv
from environments.fixedwing import FixedwingWaypointsEnv
from environments.rocket import RocketLandingEnv
from environments.missile import MissileInterceptEnv
from environments.satellite.attitude_control import SatelliteAttitudeEnv
from environments.satellite.orbit_transfer import LowThrustTransferEnv
from environments.satellite.rendezvous_docking import SpaceDockingEnv, SpaceRendezvousEnv

from algorithms.baselines.baselines import build_baseline
from algorithms.train_utils import RslVecEnv

ENVS = {'quadx': QuadXWaypointsEnv, 'fixedwing': FixedwingWaypointsEnv, 'rocket': RocketLandingEnv,
        'missile': MissileInterceptEnv,
        'satellite': SatelliteAttitudeEnv, 'transfer': LowThrustTransferEnv,
        'docking': SpaceDockingEnv, 'rendezvous': SpaceRendezvousEnv}
MODELS = Path(__file__).resolve().parent.parent / 'environments' / 'common' / 'original_pyflyt' / 'models'
TRAIL_MARKERS = 256
ORBIT_POINTS = 96
HIDDEN = np.array((0., 0., -1e5))  # park spot for unused trail markers
# rendezvous display furniture (all display-only, metres of the CWH LVLH frame)
RING_RADII = (250., 500., 1000., 2000.)   # range rings; 2000 m = range_fail
RING_SEGMENTS = 48
RING_PLANE = 0.                            # rings lie in the x-y (radial/along-track) plane
VEL_ARROW_SCALE = 100.                     # arrow length in metres per m/s of true relative speed
ARROW_SIZES = (5., 10., 20., 40., 80., 160.)  # quantized arrow pool [m] (set_scale unsupported)


def build_env(task):
    """Single-replica CPU env, same construction as baseline_eval.py."""
    out = Path(__file__).resolve().parent.parent / 'results' / task
    cfg = json.loads((out / 'config.json').read_text()) if (out / 'config.json').exists() else {}
    env_kwargs = dict(cfg.get('env_kwargs') or {})
    return ENVS[task](num_envs=1, backend='cpu', seed=7, **env_kwargs), cfg


class _JsonStd(list):
    """Per-dim init std as a JSON-serializable list rsl_rl still accepts.

    config.json round-trips the training init_std to a plain list, which
    crashes rsl_rl's `init_std * torch.ones(output_dim)`; same fix as
    train_rocket.py."""
    def __mul__(self, other):
        return torch.as_tensor(self, dtype=other.dtype, device=other.device) * other
    __rmul__ = __mul__


class _CkptPolicy:
    """Adapter wrapping a trained checkpoint's deterministic actor in the
    baseline controller interface (actions / reset_idx)."""

    def __init__(self, task, env, cfg):
        from rsl_rl.runners import OnPolicyRunner
        out = Path(__file__).resolve().parent.parent / 'results' / task
        ckpt = out / 'checkpoint.pt'
        assert ckpt.exists(), f'missing {ckpt}'
        dist = cfg.get('actor', {}).get('distribution_cfg', {})
        if isinstance(dist.get('init_std'), list):
            dist['init_std'] = _JsonStd(dist['init_std'])
        if cfg.get('algo') == 'flash_sac':
            from algorithms.flashsac.flash_sac import FlashSACRunner
            runner = FlashSACRunner(env, cfg, log_dir='/tmp/vis_tb', device=str(env.device))
            runner.load(str(ckpt))
            self._fn = runner.deterministic_action
        else:
            runner = OnPolicyRunner(RslVecEnv(env), cfg, log_dir='/tmp/vis_tb', device=str(env.device))
            runner.load(str(ckpt))
            self._fn = runner.alg.actor.eval()

    def actions(self, obs):
        return self._fn(obs)

    def reset_idx(self, mask):
        pass


class _BaselinePolicy:
    """Uniform adapter around the classical baseline."""

    def __init__(self, task, env):
        self._c = build_baseline(task, env)

    def actions(self, obs):
        return self._c.actions(obs)

    def reset_idx(self, mask):
        self._c.reset_idx(mask)


@torch.no_grad()
def record(task, env, policy, max_steps):
    """Roll one episode with the given policy (baseline or checkpoint actor),
    capturing the controlled pose per agent step.

    Returns (frames, extras): frames is a list of per-step pose dicts with
    plain-numpy 'pos' (3,) and 'quat' (4,) wxyz; extras carries task markers
    (targets, keep-out radius, ...) for the scene builder.
    """
    frames = []
    extras = {}
    if task == 'quadx':
        extras['targets'] = env.targets.cpu().numpy()[0]
    elif task == 'fixedwing':
        extras['targets'] = env.targets.cpu().numpy()[0]
    elif task in ('docking', 'rendezvous'):
        extras['keep_out'] = env.keep_out
        extras['pos_goal'] = env.pos_goal
        extras['vel_goal'] = env.vel_goal
        extras['range_fail'] = env.range_fail
    elif task == 'satellite':
        # display-only CMG geometry (the env's own published cluster constants);
        # gimbal ANGLES come from the recorded state each frame
        extras['gimbal_axes'] = env.gimbal_axes.cpu().numpy()
        extras['rotor_axes_ref'] = env.rotor_axes_ref.cpu().numpy()
    elif task == 'transfer':
        # display-only orbital geometry (never touches the sim): fixed initial
        # Keplerian elements + the per-episode apogee/perigee radii the reward
        # is actually computed from
        extras['orbit'] = dict(sma=env.sma0, ecc=env.ecc0, inc=env.inc0,
                               raan=env.raan0, argp=env.argp0,
                               ra0=float(env.ra0.cpu()[0]), rp0=float(env.rp0.cpu()[0]))

    def snap():
        if task == 'missile':
            frames.append({'pos': env.plant.drone.get_pos().cpu().numpy()[0],
                           'quat': env.plant.drone.get_quat().cpu().numpy()[0],
                           'tgt': env.target_pos.cpu().numpy()[0].copy()})
        elif task in ('quadx', 'fixedwing', 'rocket'):
            frames.append({'pos': env.plant.drone.get_pos().cpu().numpy()[0],
                           'quat': env.plant.drone.get_quat().cpu().numpy()[0]})
        elif task == 'satellite':
            frames.append({'pos': np.zeros(3), 'quat': env.q.cpu().numpy()[0].copy(),
                           'target_quat': env.target_quat.cpu().numpy()[0].copy(),
                           'gimbal': env.gimbal.cpu().numpy()[0].copy()})
        elif task == 'transfer':
            frames.append({'pos': env.r.cpu().numpy()[0].copy() * 1e-6, 'quat': None})  # 1 unit = 1000 km
        else:  # docking / rendezvous: CWH relative frame, target at the origin
            f = {'pos': env.p.cpu().numpy()[0].copy(), 'quat': None,
                 'vel': env.v.cpu().numpy()[0].copy()}
            frames.append(f)

    obs = env.get_observations()
    snap()
    steps = 0
    while steps < max_steps:
        actions = policy.actions(obs)
        obs, _, term, trunc, _ = env.step(actions)
        reset_buf = term | trunc
        policy.reset_idx(reset_buf)
        steps += 1
        if bool(reset_buf.any()):  # episode ended; state was already reset -- keep the last snap
            # terminal banner from the env's OWN termination bookkeeping (the
            # flags are read off extras before reset_idx clears them): success
            # only via the episode success_rate, timeout only via time_outs
            to = bool(env.extras.get('time_outs', torch.zeros(1, dtype=torch.bool)).any())
            ep = env.extras.get('episode')
            sr = float(ep['success_rate']) if ep else 0.
            extras['terminal'] = ('SUCCESS' if sr > .5 else
                                  'TIMEOUT (no capture)' if to else 'FAILURE')
            break
        snap()
    extras['steps'] = steps
    return frames, extras


def _urdf(scene, path, **kw):
    """Load a URDF visually, sanitizing the upstream rocket's malformed XML
    (duplicated closing tag; same fix as common.original_backend.loadURDF)."""
    import tempfile
    import xml.etree.ElementTree as ET
    raw = Path(path).read_text()
    root = ET.fromstring(raw.split('</robot>')[0] + '</robot>')
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((Path(path).parent / mesh.attrib['filename']).resolve()))
    tmp = Path(tempfile.gettempdir()) / f'vis_{Path(path).name}'
    ET.ElementTree(root).write(tmp)
    return scene.add_entity(gs.morphs.URDF(file=str(tmp), collision=False, **kw))


def _marker(scene, radius, color, pos=(0, 0, 0)):
    return scene.add_entity(gs.morphs.Sphere(pos=tuple(pos), radius=radius, fixed=False, collision=False),
                            surface=gs.surfaces.Plastic(color=color))


def _box(scene, size, color, pos=(0, 0, 0)):
    return scene.add_entity(gs.morphs.Box(pos=tuple(pos), size=tuple(size), fixed=False, collision=False),
                            surface=gs.surfaces.Plastic(color=color))


def _trail(scene, radius, color):
    return [_marker(scene, radius, color, (0, 0, -1e5)) for _ in range(TRAIL_MARKERS)]


def _update_trail(markers, path):
    """Replay the whole path each frame through TRAIL_MARKERS spheres."""
    n = min(len(path), TRAIL_MARKERS)
    picked = np.linspace(0, len(path) - 1, n).astype(int) if len(path) > 1 else np.array([0])
    for i, m in enumerate(markers):
        p = path[picked[i]] if i < n else HIDDEN
        m.set_pos(tuple(map(float, p)))


def _follow(cam, p, offset, lookat=None):
    p = np.asarray(p, dtype=float)
    cam.set_pose(pos=tuple(p + np.asarray(offset)), lookat=tuple(np.asarray(lookat, dtype=float) if lookat is not None else p))


def _dir_quat(d):
    """wxyz quaternion rotating the box long axis (z) onto direction d."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (1., 0., 0., 0.)
    d = d / n
    ax = np.cross((0., 0., 1.), d)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return (1., 0., 0., 0.) if d[2] > 0 else (0., 1., 0., 0.)
    ax = ax / s
    ang = .5 * math.acos(np.clip(d[2], -1., 1.))
    return (math.cos(ang), *(ax * math.sin(ang)))


def _quat_x_to(d):
    """wxyz quaternion rotating the local +X axis onto direction d (the
    chaser's docking face is +X, so this orients the chaser toward the
    target station)."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (1., 0., 0., 0.)
    d = d / n
    ax = np.cross((1., 0., 0.), d)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return (1., 0., 0., 0.) if d[0] > 0 else (0., 0., 1., 0.)
    ax = ax / s
    ang = .5 * math.acos(np.clip(d[0], -1., 1.))
    return (math.cos(ang), *(ax * math.sin(ang)))


def _cyl(scene, radius, length, color):
    """Thin cylinder marker whose long axis can be oriented via set_quat
    (Genesis cylinders default to the local z axis)."""
    return scene.add_entity(gs.morphs.Cylinder(pos=HIDDEN, radius=radius,
                                               height=length, fixed=False, collision=False),
                            surface=gs.surfaces.Plastic(color=color))


def _quat_z_to(d):
    """wxyz quaternion rotating the local +Z axis onto direction d."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (1., 0., 0., 0.)
    d = d / n
    ax = np.cross((0., 0., 1.), d)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return (1., 0., 0., 0.) if d[2] > 0 else (0., 1., 0., 0.)
    ax = ax / s
    ang = .5 * math.acos(np.clip(d[2], -1., 1.))
    return (math.cos(ang), *(ax * math.sin(ang)))


def _quat_mul_np(a, b):
    """Hamilton product of two wxyz quaternions (numpy)."""
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2))


def _quat_rot(q, v):
    """Rotate vector v (3,) by wxyz quaternion q (display-only numpy math)."""
    qv = np.append(0., v)
    c = np.array((q[0], -q[1], -q[2], -q[3]))  # conjugate of (w, x, y, z)
    return _quat_mul_np(_quat_mul_np(q, qv), c)[1:]


def _quat_angle(q1, q2):
    """Shortest-arc angle [rad] between two wxyz quaternions."""
    d = abs(float(np.dot(q1, q2)))
    return 2. * math.acos(min(1., d))


def _perp_basis(d):
    """Orthonormal (u, w) pair perpendicular to unit vector d."""
    ref = np.array((0., 0., 1.)) if abs(d[2]) < .9 else np.array((1., 0., 0.))
    u = np.cross(d, ref); u = u / np.linalg.norm(u)
    return u, np.cross(d, u)


def _inset(main_rgb, close_rgb, label):
    """Paste the labeled close-up view bottom-right onto the main frame
    (kept small so it never covers the docking action in the main view)."""
    from PIL import Image, ImageDraw, ImageFont
    close = Image.fromarray(close_rgb).resize((256, 192))
    dr = ImageDraw.Draw(close)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    dr.rectangle((0, 0, 255, 191), outline=(255, 255, 255), width=2)
    dr.rectangle((0, 0, 175, 18), fill=(0, 0, 0))
    dr.text((4, 2), label, fill=(255, 230, 80), font=font)
    main = Image.fromarray(main_rgb)
    main.paste(close, (640 - 262, 480 - 198))
    return np.asarray(main)


def _overlay(rgb, lines):
    """Burn legend text into a rendered frame (display-only)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    y = 8
    for ln in lines:
        dr.text((11, y + 1), ln, fill=(0, 0, 0), font=font)   # shadow
        dr.text((10, y), ln, fill=(255, 255, 255), font=font)
        y += 19
    return np.asarray(img)


def build_scene(task, extras, first):
    """Visual-only scene; entities returned in a dict for the replay loop."""
    if not gs._initialized:  # pure-tensor tasks never construct a physics scene
        gs.init(backend=gs.cpu, logging_level='warning')
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(gravity=(0, 0, 0)),
                     renderer=gs.renderers.Rasterizer())
    ents = {}
    p0 = first['pos']
    if task == 'quadx':
        # PyFlyt-style visualization (mirrors PyFlyt 0.29 quadx_envs): plain
        # grey ground, the cf2x URDF loaded as-is with no added geometry, an
        # opaque solid green sphere at the active waypoint (radius = goal reach
        # distance / 4, exactly PyFlyt's waypoint_handler scaling), and a
        # fixed camera at PyFlyt's debug-view angles (yaw 30, pitch -30).
        # The drone is ~9 cm across, so cameraDistance is scaled down from
        # PyFlyt's interactive 5 m to keep the vehicle readable.
        scene.add_entity(gs.morphs.Plane(),
                         surface=gs.surfaces.Plastic(color=(.55, .55, .55, 1.)))
        ents['body'] = _urdf(scene, MODELS / 'vehicles/cf2x/cf2x.urdf')
        ents['goal'] = _marker(scene, .035, (0., 1., 0., 1.), extras['target'])
        ents['attach'] = []
        t = np.asarray(extras['target'], dtype=float)
        yaw, pitch, dist = math.radians(30.), math.radians(-30.), 0.85
        off = np.array((dist * math.cos(pitch) * math.cos(yaw),
                        -dist * math.cos(pitch) * math.sin(yaw),
                        -dist * math.sin(pitch)))
        cam = scene.add_camera(res=(960, 540), pos=tuple(t + off),
                               lookat=tuple(t), fov=40, far=50)
    elif task == 'fixedwing':
        scene.add_entity(gs.morphs.Plane())
        ents['body'] = _urdf(scene, MODELS / 'vehicles/fixedwing/fixedwing.urdf')
        ents['wps'] = [_marker(scene, 1., (.9, .2, .2, .9), t) for t in extras['targets']]
        ents['trail'] = _trail(scene, .15, (.2, .4, .9, .5))
        cam = scene.add_camera(res=(640, 480), pos=(0, -12, 6), lookat=tuple(p0), fov=50, far=400)
    elif task == 'rocket':
        scene.add_entity(gs.morphs.Plane())
        _urdf(scene, MODELS / 'landing_pad.urdf', pos=(0, 0, .1))
        ents['body'] = _urdf(scene, MODELS / 'vehicles/rocket/rocket.urdf')
        ents['trail'] = _trail(scene, .4, (.2, .4, .9, .5))
        cam = scene.add_camera(res=(640, 480), pos=(380, -380, 430), lookat=(0, 0, 225), fov=60, far=3000)
    elif task == 'satellite':
        # Recognizable bus: grey body + equipment compartments on the +/-z
        # faces (up/down), sensor head on +x (front), nozzle on -x (rear),
        # antenna on top, solar panels on +/-y with a light grid. Panels and
        # details are rigidly attached to the bus: each stores (entity,
        # local_pos, local_quat) and is posed per frame from the recorded
        # quaternion. All geometry here is display-only (collision=False,
        # scene never stepped) and only built when visualize.py is run.
        ents['body'] = _box(scene, (.8, .8, .8), (.72, .75, .8, 1.))
        ents['attach'] = []
        def _attach(ent, pos, quat=(1., 0., 0., 0.)):
            ents['attach'].append((ent, np.asarray(pos, float), tuple(quat)))
        # up/down equipment compartments distinguish body +z from -z
        _attach(_box(scene, (.5, .5, .14), (.45, .48, .55, 1.)), (0, 0, .46))
        _attach(_box(scene, (.5, .5, .14), (.45, .48, .55, 1.)), (0, 0, -.46))
        # front (+x) sensor head and rear (-x) nozzle distinguish +x from -x
        _attach(_box(scene, (.12, .3, .3), (.35, .38, .42, 1.)), (.46, 0, 0))
        _attach(_cyl(scene, .1, .22, (.25, .25, .28, 1.)), (-.5, 0, 0), _quat_x_to((-1., 0., 0.)))
        # top antenna: boom + gold dish
        _attach(_cyl(scene, .025, .35, (.8, .8, .82, 1.)), (0, 0, .68))
        _attach(_cyl(scene, .13, .05, (.9, .75, .3, 1.)), (0, 0, .88))
        # +/-y solar panels: dark panel, yoke to the bus, light grid lines
        for s in (1., -1.):
            _attach(_box(scene, (.85, 2.1, .04), (.12, .22, .5, 1.)), (0, s * 1.45, 0))
            _attach(_box(scene, (.3, .4, .12), (.55, .58, .62, 1.)), (0, s * .55, 0))
            for k in (-.68, 0., .68):
                _attach(_box(scene, (.87, .05, .06), (.07, .13, .32, 1.)), (0, s * 1.45 + k, 0))
            for k in (-.28, .28):
                _attach(_box(scene, (.05, 2.1, .06), (.07, .13, .32, 1.)), (k, s * 1.45, 0))
        # body-frame triad (X red, Y green, Z blue) rotating with the bus
        for d, col in (((1., 0., 0.), (.9, .25, .25, .95)),
                       ((0., 1., 0.), (.25, .8, .3, .95)),
                       ((0., 0., 1.), (.3, .5, .95, .95))):
            _attach(_box(scene, (.045, .045, 1.3), col), np.array(d) * .65, _quat_z_to(d))
        # target-attitude triad (yellow), posed from the recorded target quat
        ents['triad_target'] = [
            (_box(scene, (.03, .03, 1.1), (.95, .8, .2, .9)),
             np.array(d) * .55, _quat_z_to(d))
            for d in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
        # red ghost box at the target attitude (legacy marker)
        ents['goal'] = _box(scene, (.8, .8, .8), (.9, .25, .2, .3))
        # one cyan cylinder per CMG, oriented along the instantaneous rotor
        # momentum axis r_i(theta) (real gimbal state; no faked rotor spin)
        ga = extras['gimbal_axes']
        ents['cmg'] = [_cyl(scene, .035, .55, (.2, .9, .95, .95)) for _ in ga]
        # static star decoration (fixed positions, never moves)
        rng = np.random.default_rng(3)
        for p in rng.normal(size=(70, 3)):
            p = p / np.linalg.norm(p) * 22.
            _marker(scene, .06, (1., 1., 1., .8), p)
        # FIXED oblique camera: never follows the tumbling body, so the bus
        # stays clear and stable throughout the episode
        cam = scene.add_camera(res=(640, 480), pos=(2.6, -2.6, 1.6),
                               lookat=(0, 0, 0), fov=45, far=100)
    elif task == 'transfer':
        # 3D dynamics (i = pi/3), so a 3D view centered on Earth. Orbit lines
        # and markers are display-only geometry derived from the recorded
        # initial elements; nothing here feeds back into the sim.
        o = extras['orbit']
        # perifocal -> inertial rotation, same convention as env._kepler_to_cartesian
        cO, sO = math.cos(o['raan']), math.sin(o['raan'])
        ci, si = math.cos(o['inc']), math.sin(o['inc'])
        cw, sw = math.cos(o['argp']), math.sin(o['argp'])
        R = np.array([[cO*cw-sO*sw*ci, -cO*sw-sO*cw*ci, sO*si],
                      [sO*cw+cO*sw*ci, -sO*sw+cO*cw*ci, -cO*si],
                      [sw*si, cw*si, ci]])
        a, e = o['sma'], o['ecc']
        b = a * math.sqrt(1. - e * e)
        E = np.linspace(0., 2. * np.pi, ORBIT_POINTS, endpoint=False)
        pf = np.stack((a * (np.cos(E) - e), b * np.sin(E), np.zeros_like(E)), -1)
        scale = 1e-6  # 1 scene unit = 1000 km
        ents['orbit0'] = [  # dashed green initial orbit (true ellipse)
            _marker(scene, .07 if i % 2 == 0 else .03, (.2, .8, .3, .9), p)
            for i, p in enumerate(pf @ R.T * scale)]
        # dotted cyan target orbit: circle at the initial apogee radius
        # (ideal perigee-raising end state, same plane at reference thrust)
        circ = np.stack((o['ra0'] * np.cos(E), o['ra0'] * np.sin(E), np.zeros_like(E)), -1)
        ents['orbit_t'] = [_marker(scene, .04, (.2, .8, .9, .9), p @ R.T * scale) for p in circ]
        # perigee (red) / apogee (yellow) of the initial orbit
        ents['peri'] = _marker(scene, .22, (.9, .2, .2, 1.), R[:, 0] * a * (1. - e) * scale)
        ents['apo'] = _marker(scene, .22, (.95, .85, .2, 1.), R[:, 0] * -a * (1. + e) * scale)
        ents['earth'] = _marker(scene, 6.371, (.15, .35, .75, 1.))
        ents['body'] = _box(scene, (.5, .5, 1.), (1., .65, .1, 1.))  # enlarged spacecraft
        ents['trail'] = _trail(scene, .12, (1., .65, .1, 1.))
        # fixed camera centered on Earth, tilted ~45 deg off the orbital
        # plane so the ellipse shape stays visible for the whole episode
        n = R[:, 2] / np.linalg.norm(R[:, 2])
        u = np.cross(n, (0., 0., 1.)); u /= np.linalg.norm(u)
        campos = -30. * u + 30. * n  # ~45 deg above the orbital plane
        cam = scene.add_camera(res=(640, 480), pos=tuple(campos), lookat=(0, 0, 0), fov=45, far=300)
        # 2000 km scale bar (2 scene units), below the frame center
        view = -campos / np.linalg.norm(campos)
        right = np.cross(view, (0., 0., 1.)); right /= np.linalg.norm(right)
        up = np.cross(right, view)
        base = -up * 15. - right
        ents['scalebar'] = [_marker(scene, .1, (1., 1., 1., .9), base + right * i) for i in range(5)]
    elif task == 'missile':
        # km-scale air-intercept scene at ~4 km altitude: missile URDF, red
        # marker for the kinematic target aircraft (posed per frame from the
        # recorded target state), engagement trail, wide follow camera
        ents['body'] = _urdf(scene, MODELS / 'vehicles/missile/missile.urdf')
        ents['target'] = _marker(scene, 8., (.9, .2, .2, .9), tuple(first['tgt']))
        ents['trail'] = _trail(scene, 1.5, (.2, .4, .9, .5))
        cam = scene.add_camera(res=(640, 480), pos=tuple(p0 + np.array((0., -800., 300.))),
                               lookat=tuple(p0), fov=50, far=20000)
    else:  # docking / rendezvous
        # Target station (grey) with two solar panels ATTACHED to the ±y
        # faces (no longer intersecting the core) and a docking port on the
        # +x face: green ring edge + centre marker + translucent axis
        # cylinder marking the approach corridor. The chaser (orange) gets
        # its own port ring on the face turned toward the target and a
        # forward-direction arrow; all marker positions follow the recorded
        # relative state each frame.
        big = task == 'rendezvous'
        s = 3. if big else 1.5          # station body edge [m]
        c = 2. if big else 1.           # chaser body edge [m]
        px = s / 2.                     # docking-port face offset from origin
        ents['station'] = _box(scene, (s, s, s), (.75, .78, .8, 1.))
        pl, pt = (7., .3) if big else (2.6, .1)
        ents['panel_a'] = _box(scene, (1.6 * s if big else 2.2, pl, pt),
                               (.25, .4, .9, 1.), (0, s / 2. + pl / 2., 0))
        ents['panel_b'] = _box(scene, (1.6 * s if big else 2.2, pl, pt),
                               (.25, .4, .9, 1.), (0, -s / 2. - pl / 2., 0))
        # target docking port on the +x face: ring edge, centre marker, axis
        rr = .7 if big else .35  # port ring radius [m]
        ents['t_ring'] = [_marker(scene, .12 if big else .05, (.2, .9, .4, .95),
                                  (px, rr * math.cos(2 * math.pi * i / 12.),
                                   rr * math.sin(2 * math.pi * i / 12.)))
                          for i in range(12)]
        ents['t_center'] = _marker(scene, .16 if big else .07, (.1, .95, .3, 1.), (px, 0, 0))
        ents['t_axis'] = _cyl(scene, .12 if big else .05, 3.5 * s if big else 2.2,
                              (.2, .9, .4, .28))
        ents['body'] = _box(scene, (c, c, c), (1., .6, .1, 1.))
        # chaser port ring (8 spheres) + centre marker, positioned on the
        # target-facing (+X local) face each frame
        ents['c_ring'] = [_marker(scene, .07 if big else .04, (1., .8, .2, .95))
                          for _ in range(8)]
        ents['c_center'] = _marker(scene, .09 if big else .05, (1., .85, .3, 1.))
        # dashed line-of-sight pool between the two centres
        ents['link'] = [_marker(scene, 1.2 if big else .04, (.95, .95, .95, .5))
                        for _ in range(20)]
        ents['trail'] = _trail(scene, .35 if big else .12, (1., .6, .1, .9))
        if extras.get('keep_out'):
            ents['keep_out'] = _marker(scene, extras['keep_out'], (.9, .2, .2, .12))
        # relative-velocity arrow pool (true speed * 100 m per m/s, quantized)
        ents['arrows'] = {L: _box(scene, (2.5, 2.5, L) if big else (1.2, 1.2, L),
                                  (1., .9, .25, .95), HIDDEN)
                          for L in ARROW_SIZES}
        if big:  # rendezvous-only LVLH furniture
            # success goal ball (r = pos_goal = 10 m, the real criterion)
            ents['goal'] = _marker(scene, extras['pos_goal'], (.2, .85, .3, .15))
            # LVLH axis bars through the target: x radial (red), y along-track
            # (green), z cross-track (blue); each spans +/-120 m
            ents['axes'] = (_box(scene, (240., 2., 2.), (.85, .35, .35, .55), (0, 0, 0)),
                            _box(scene, (2., 240., 2.), (.35, .8, .35, .55), (0, 0, 0)),
                            _box(scene, (2., 2., 240.), (.35, .55, .9, .55), (0, 0, 0)))
            # range rings in the radial/along-track plane (x-y, z = RING_PLANE),
            # all in metres; the outermost is the 2000 m range-fail boundary
            ents['rings'] = [_marker(scene, 2.5, (.85, .8, .5, .28),
                                     (r * math.cos(2 * math.pi * i / RING_SEGMENTS),
                                      r * math.sin(2 * math.pi * i / RING_SEGMENTS),
                                      RING_PLANE))
                             for r in RING_RADII for i in range(RING_SEGMENTS)]
        off = (18., 5., 4.) if task == 'rendezvous' else (12., 3., 2.5)
        cam = scene.add_camera(res=(640, 480), lookat=(0, 0, 0), fov=55, far=3000)
        # labeled close-up camera on the docking port region (composited as an
        # inset; the main view always stays full-frame)
        ents['close_cam'] = scene.add_camera(
            res=(320, 240), pos=(px + 2.5 * s, 1.6 * s, .9 * s), lookat=(px, 0, 0),
            fov=55, far=300)
    scene.build(n_envs=1)
    return scene, ents, cam


def render(task, frames, extras, out_path, fps):
    scene, ents, cam = build_scene(task, extras, frames[0])
    trail = ents.get('trail')
    path = []
    rgb_frames = []
    big = task == 'rendezvous'
    space = task in ('docking', 'rendezvous')
    c_size = 2. if big else 1.
    px = 1.5 if big else .75                    # target port face offset
    close_rng = 60. if big else 30.             # inset tracks the chaser below this
    arrows = ents.get('arrows', {})
    link = ents.get('link', [])
    if space:
        # static target approach-axis marker: translucent +x cylinder from
        # the port face outward
        L = 3.5 * (3. if big else 1.5)
        ents['t_axis'].set_pos((px + L / 2., 0., 0.))
        ents['t_axis'].set_quat((math.sqrt(.5), 0., math.sqrt(.5), 0.))
    if task == 'rendezvous':
        # Stable LVLH view: the viewing DIRECTION is fixed once from the
        # initial line of sight, so the frame never rotates with the chaser;
        # only camera distance scales with the live range (clamped) and the
        # lookat slides along the target-chaser line, keeping both in frame
        # while the keep-out / goal balls stay visible on final approach.
        p0 = np.asarray(frames[0]['pos'], dtype=float)
        r0 = np.linalg.norm(p0)
        u = p0 / r0 if r0 > 1e-9 else np.array((0., -1., 0.))
        side = np.cross(u, (0., 0., 1.))
        ns = np.linalg.norm(side)
        side = side / ns if ns > 1e-9 else np.array((1., 0., 0.))
        view_dir = u + .55 * side + .35 * np.array((0., 0., 1.))
        view_dir = view_dir / np.linalg.norm(view_dir)
    terminal = extras.get('terminal')
    for fi, f in enumerate(frames):
        ents['body'].set_pos(tuple(map(float, f['pos'])))
        if f['quat'] is not None:
            ents['body'].set_quat(tuple(map(float, f['quat'])))
        if task == 'missile':
            ents['target'].set_pos(tuple(map(float, f['tgt'])))
        if task == 'satellite':
            ents['goal'].set_quat(tuple(map(float, f['target_quat'])))
            q, qt = np.asarray(f['quat'], float), np.asarray(f['target_quat'], float)
            # rigidly attached bus details + body triad follow the bus quat
            for ent, lp, lq in ents['attach']:
                ent.set_pos(tuple(map(float, _quat_rot(q, lp))))
                ent.set_quat(tuple(map(float, _quat_mul_np(q, lq))))
            # target triad follows the recorded target attitude
            for ent, lp, lq in ents['triad_target']:
                ent.set_pos(tuple(map(float, _quat_rot(qt, lp))))
                ent.set_quat(tuple(map(float, _quat_mul_np(qt, lq))))
            # CMG rotor momentum axes from the REAL gimbal state:
            # r_i(theta) = cos(th) r_ref + sin(th) (g x r_ref)
            ga, rref = extras['gimbal_axes'], extras['rotor_axes_ref']
            gim = f['gimbal']
            for i, ent in enumerate(ents['cmg']):
                d = math.cos(gim[i]) * rref[i] + math.sin(gim[i]) * np.cross(ga[i], rref[i])
                d = d / np.linalg.norm(d)
                ent.set_pos(tuple(map(float, _quat_rot(q, ga[i] * .52))))
                ent.set_quat(tuple(map(float, _quat_z_to(_quat_rot(q, d)))))
        elif trail is not None:
            path.append(f['pos'])
            _update_trail(trail, path)
        p = np.asarray(f['pos'], dtype=float)
        if task == 'quadx':
            # PyFlyt-style: the URDF renders as-is (no display attachments),
            # camera fixed at the debug-view pose (yaw 30, pitch -30 over the
            # initial waypoint) instead of following the body
            rgb_frames.append(cam.render(rgb=True, force_render=True)[0])
            continue
        if space:
            # chaser attitude: docking face (+X local) turned toward the
            # target; port ring / centre marker on that face; dashed
            # line-of-sight between the two centres, kept short of both
            # bodies so it never covers a port
            dist = float(np.linalg.norm(p))
            d = -p / dist if dist > 1e-9 else np.array((1., 0., 0.))
            ents['body'].set_quat(_quat_x_to(d))
            front = p + d * (c_size / 2.)
            uu, ww = _perp_basis(d)
            ring_r = .6 if big else .3
            for i, m in enumerate(ents['c_ring']):
                a = 2 * math.pi * i / len(ents['c_ring'])
                m.set_pos(tuple(front + (math.cos(a) * uu + math.sin(a) * ww) * ring_r))
            ents['c_center'].set_pos(tuple(front))
            for i, m in enumerate(link):
                m.set_pos(tuple(p * (.12 + .76 * i / (len(link) - 1))))
            # relative-velocity arrow: pool entry with true length
            # |v| * VEL_ARROW_SCALE (100 m per m/s), oriented along v
            v = f.get('vel', np.zeros(3))
            speed = float(np.linalg.norm(v))
            for L, m in arrows.items():
                m.set_pos(tuple(HIDDEN))
            if speed > 1e-3 and arrows:
                ge = [sz for sz in ARROW_SIZES if sz >= speed * VEL_ARROW_SCALE]
                L = min(ge) if ge else max(ARROW_SIZES)
                if L > .5 * dist:  # never let the arrow cross the target region
                    L = None
            else:
                L = None
            if L is not None:
                a = arrows[L]
                a.set_pos(tuple(p + v / speed * L * .5))
                a.set_quat(_dir_quat(v))
            # labeled close-up inset: port-focused, sliding to the chaser on
            # final approach; the main view always stays full-frame
            s_ = 3. if big else 1.5
            cpos = tuple(p * .5 + np.array((px, 0., 0.))) if dist < close_rng else (px, 0., 0.)
            ents['close_cam'].set_pose(pos=(px + 2.5 * s_, 1.6 * s_, .9 * s_), lookat=cpos)
        if task == 'rendezvous':
            r = float(np.linalg.norm(p))
            d = float(np.clip(1.8 * max(r, 8.), 45., 2600.))
            lookat = p * .45
            cam.set_pose(pos=tuple(lookat + view_dir * d), lookat=tuple(lookat))
            rgb = cam.render(rgb=True, force_render=True)[0]
            lines = [
                'Rendezvous  CWH LVLH frame, target at origin  [metres]',
                'X radial   Y along-track   Z cross-track',
                'green ball r=10 m goal | red ball r=5 m keep-out | rings 250/500/1000/2000 m',
                f'range = {r:.0f} m   |v| = {speed:.2f} m/s   (yellow arrow: 100 m per m/s)',
            ]
            if terminal is not None and fi == len(frames) - 1:
                lines.append(f'RESULT: {terminal}')
            rgb = _overlay(rgb, lines)
            rgb = _inset(rgb, ents['close_cam'].render(rgb=True, force_render=True)[0],
                         'CLOSE-UP: docking port')
            rgb_frames.append(rgb)
            continue
        if task == 'docking':
            r = np.linalg.norm(p)
            ph = p / r if r > 1e-9 else np.array((0., 0., 1.))
            side = np.cross(ph, (0., 0., 1.))
            n = np.linalg.norm(side)
            side = side / n if n > 1e-9 else np.array((1., 0., 0.))
            # camera parked behind the chaser along the target line, small
            # lateral offset: chaser, station and docking axis all stay in
            # frame; fixed offsets, no zoom, no jumps
            k1, k2, k3 = 12., 3., 2.5
            cam.set_pose(pos=tuple(p + ph * k1 + side * k2 + (0., 0., k3)),
                         lookat=tuple(p * .5))
            rgb = cam.render(rgb=True, force_render=True)[0]
            lines = [
                'Docking  CWH LVLH frame, target at origin  [metres]',
                f'success: range < {extras["pos_goal"]} m & |v| < {extras["vel_goal"]} m/s'
                f'  |  fail: range > {extras["range_fail"]} m',
                f'range = {float(r):.2f} m   |v| = {speed:.3f} m/s   (yellow arrow: 100 m per m/s)',
                'grey station + green port ring/axis = target | orange + yellow port = chaser',
            ]
            if terminal is not None and fi == len(frames) - 1:
                lines.append(f'RESULT: {terminal}')
            rgb = _overlay(rgb, lines)
            rgb = _inset(rgb, ents['close_cam'].render(rgb=True, force_render=True)[0],
                         'CLOSE-UP: docking port')
            rgb_frames.append(rgb)
        elif task in ('fixedwing', 'rocket'):
            r = np.linalg.norm(p)
            ph = p / r if r > 1e-9 else np.array((0., 0., 1.))
            side = np.cross(ph, (0., 0., 1.))
            n = np.linalg.norm(side)
            side = side / n if n > 1e-9 else np.array((1., 0., 0.))
            if task == 'rocket':
                # Stable oblique follow view. The viewing azimuth is FIXED
                # ((+1,-1,0)/sqrt(2), no rotation with the rocket's lateral
                # drift, so the frame never swings or jumps). The lookat sits
                # mid-way between the rocket and the pad (xy follows the
                # rocket, z = pz/2) and the follow distance grows smoothly
                # with altitude (D = 80 m + pz), so the FULL rocket, the pad
                # and the whole descent corridor stay inside the fov=60
                # frame at every altitude; near touchdown the camera settles
                # ~80 m out at a low oblique angle showing the pad contact.
                # No zoom, no discontinuities: every term is a continuous
                # function of the rocket position.
                D = 80. + max(p[2], 0.)
                lookat = np.array((p[0], p[1], 0.5 * p[2]))
                dirv = np.array((1., -1., 0.)) / math.sqrt(2.)
                cpos = lookat + dirv * (0.9 * D)
                cpos[2] += 0.38 * D
                cam.set_pose(pos=tuple(cpos), lookat=tuple(lookat))
            else:  # fixedwing: unchanged legacy behavior (same offsets as docking)
                k1, k2, k3 = 12., 3., 2.5
                cam.set_pose(pos=tuple(p + ph * k1 + side * k2 + (0., 0., k3)),
                             lookat=tuple(p * .5))
            rgb_frames.append(cam.render(rgb=True, force_render=True)[0])
        elif task == 'transfer':
            # fixed camera on the central body: following the spacecraft would
            # hide the orbit shape; the full trajectory must stay in frame
            rgb = cam.render(rgb=True, force_render=True)[0]
            rgb = _overlay(rgb, [
                'Orbital transfer  inertial GCRF frame  [1 scene unit = 1000 km]',
                'green dashed = initial orbit | cyan dotted = target (circular @ initial apogee)',
                'red = initial perigee | yellow = initial apogee | orange trail = true state history',
                'satellite box enlarged for visibility; true scale = the 2000 km bar (bottom)',
            ])
            rgb_frames.append(rgb)
        elif task == 'satellite':
            # fixed oblique camera (set at build time): no follow, no shake
            att = math.degrees(_quat_angle(f['quat'], f['target_quat']))
            g = ' '.join(f'{a:+.2f}' for a in f['gimbal'])
            rgb = _overlay(cam.render(rgb=True, force_render=True)[0], [
                'Satellite CMG slew  body: X red / Y green / Z blue',
                'yellow = target triad | cyan = CMG axes | ghost = target',
                f'att err = {att:.1f} deg   gimbal [rad] = [{g}]',
            ])
            if terminal is not None and fi == len(frames) - 1:
                rgb = _overlay(rgb, [f'RESULT: {terminal}'])
            rgb_frames.append(rgb)
        elif task == 'missile':
            # follow the missile from between it and the target; the lookat
            # rides the missile-target mid-point so both stay framed at
            # km-scale ranges
            tgt = np.asarray(f['tgt'], dtype=float)
            mid = .5 * (p + tgt)
            _follow(cam, p, tuple(.7 * (mid - p) + np.array((0., 0., 100.))), lookat=mid)
            rgb_frames.append(cam.render(rgb=True, force_render=True)[0])
        elif task != 'quadx':  # quadx is rendered in its own branch above
            _follow(cam, p, (-9, -9, 5))
            rgb_frames.append(cam.render(rgb=True, force_render=True)[0])
    scene.destroy()
    import imageio
    imageio.mimsave(out_path, rgb_frames, fps=fps, codec='libx264', quality=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vehicle', choices=list(ENVS))
    parser.add_argument('--max-steps', type=int, default=4200,
                        help='safety cap on agent steps of the recorded episode')
    parser.add_argument('--fps', type=int, default=30, help='playback fps of the mp4')
    parser.add_argument('--out', type=Path, default=None, help='mp4 path (default results/<task>/vis.mp4)')
    parser.add_argument('--ckpt', action='store_true',
                        help='replay the trained checkpoint (results/<task>/checkpoint.pt) '
                             'instead of the classical baseline')
    args = parser.parse_args()

    out = args.out or Path(__file__).resolve().parent.parent / 'results' / args.vehicle / \
        ('vis_ckpt.mp4' if args.ckpt else 'vis.mp4')
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    env, cfg = build_env(args.vehicle)
    policy = _CkptPolicy(args.vehicle, env, cfg) if args.ckpt else _BaselinePolicy(args.vehicle, env)
    frames, extras = record(args.vehicle, env, policy, args.max_steps)
    env.close()

    render(args.vehicle, frames, extras, out, args.fps)
    print(f'{args.vehicle}: recorded {len(frames)} agent steps -> {out}')


if __name__ == '__main__':
    main()
