"""Genesis offscreen visualization of one baseline episode per task.

Usage: python visualize.py quadx|quadx_track|fixedwing|quadplane|rocket|missile|satellite|transfer|docking|rendezvous

Runs the task's classical baseline on a single CPU replica (same env
construction as eval.py/baseline_eval.py, incl. results/<task>/config.json
restoration) while recording the controlled object's pose each agent step,
then replays the recording in a visual-only Genesis scene and renders an
mp4 to results/<task>/vis.mp4.

Scene conventions: every main camera is fixed at a global task view so targets,
waypoints, the landing site and orbit context never leave the frame due to
vehicle motion. The composed 1440x810 frame is 16:9 and has three disjoint
zones: the panorama keeps its full 960x810 frame on the left (nothing is drawn
on top of it), a 480x430 state card sits top-right, and a 480x354 close-up
below it (separated by a 26 px label bar) follows the controlled object at
close range. The state card is a styled panel (task title, hero metric, key/
value rows, colour legend, note, episode progress and a terminal RESULT
pill), not a stack of raw readout lines. Bodies and markers are enlarged
(display-only) where the true scale would leave them a few pixels wide; each
card states the magnification so the render is never mistaken for a scale
drawing. Genesis' camera `fov` is vertical, so when the frame aspect changes
the vertical fov is recomputed to keep the original HORIZONTAL coverage
(_hfov_match): the taller frame only adds sky/ground above and below, the
left/right framing the cameras were set up for is preserved.

Per-task scene contents:
  quadx       cf2x URDF on a grey ground plane under a light sky, opaque green
              sphere at the active waypoint (PyFlyt marker scaling), body
              enlarged 6x so the ~9 cm drone is readable, and a
              PyFlyt-angle close-up inset
  quadx_track cf2x URDF enlarged 12x, red ground target, true line of sight
              and the env's cylindrical LOS occluders, close-up inset
  fixedwing   fixedwing URDF enlarged 8x, red spheres at the 4 waypoints,
              close-up inset
  quadplane   PyFlyt quadplane URDF enlarged 8x, red waypoint spheres,
              close-up inset
  rocket      rocket URDF enlarged 8x + landing pad on a ground plane,
              close-up inset; hard-stops are handled here: the display scale is
              anchored at the airframe's LOWEST point (_scale_ground_lift), so
              the 8x body stands on the pad instead of sinking ~17 m into it
              when the true base is only centimetres above the pad; historical
              trail never covers the body
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
              CLOSE-UP inset follows the chaser and continuously widens from
              a chaser close-up to a chaser+station pair frame (solar panels
              included) as the range closes; the drawn chaser is held at a
              minimum separation so the enlarged bodies never intersect at
              contact (overlay range/speed stay true). The terminal banner is
              the env's own termination outcome (SUCCESS / FAILURE / TIMEOUT)
  rendezvous  docking scene plus LVLH display furniture: radial/along-track/
              cross-track axis bars, range rings (250/500/1000/2000 m, all in
              metres), green goal ball (r = 10 m, the real pos_goal), the
              red keep-out sphere (r = 5 m) and a quantized velocity arrow
              (arrow pool in metres, 100 m per m/s of true relative speed),
              with the live range / speed values burned into each frame.
              The main camera is fixed in the LVLH frame; the close-up inset
              moves independently and widens to the chaser+station pair frame
              as the range closes.
  missile     missile URDF enlarged 200x + a yellow locator dot riding on it
              on the km-scale 4 km-altitude engagement, red marker for the
              kinematic target aircraft (posed per frame from the recorded
              target state), engagement trail, a fixed global engagement
              camera panned along the engagement line so the missile/target
              pair never hides behind the inset, and a missile close-up inset
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
sys.path.insert(0, str(_ROOT))  # exposes environments.* and algorithms.*

import numpy as np  # noqa: E402
import torch  # noqa: E402
import genesis as gs  # noqa: E402

from environments.multirotor import QuadXTrackEnv, QuadXWaypointsEnv  # noqa: E402
from environments.multirotor.track_env import OCCLUDERS  # noqa: E402
from environments.fixedwing import FixedwingWaypointsEnv  # noqa: E402
from environments.vtol import QuadplaneWaypointsEnv  # noqa: E402
from environments.rocket import RocketLandingEnv  # noqa: E402
from environments.missile import MissileInterceptEnv  # noqa: E402
from environments.satellite.attitude_control import SatelliteAttitudeEnv  # noqa: E402
from environments.satellite.orbit_transfer import LowThrustTransferEnv  # noqa: E402
from environments.satellite.rendezvous_docking import (  # noqa: E402
    SpaceDockingEnv,
    SpaceRendezvousEnv,
)

from algorithms.baselines.baselines import build_baseline  # noqa: E402
from algorithms.train_utils import RslVecEnv  # noqa: E402

ENVS = {
    "quadx": QuadXWaypointsEnv,
    "quadx_track": QuadXTrackEnv,
    "fixedwing": FixedwingWaypointsEnv,
    "quadplane": QuadplaneWaypointsEnv,
    "rocket": RocketLandingEnv,
    "missile": MissileInterceptEnv,
    "satellite": SatelliteAttitudeEnv,
    "transfer": LowThrustTransferEnv,
    "docking": SpaceDockingEnv,
    "rendezvous": SpaceRendezvousEnv,
}
MODELS = (
    Path(__file__).resolve().parent.parent
    / "environments"
    / "common"
    / "original_pyflyt"
    / "models"
)
TRAIL_MARKERS = 256
ORBIT_POINTS = 96
HIDDEN = np.array((0.0, 0.0, -1e5))  # park spot for unused trail markers
# Layout: the panorama keeps a tall 960x810 frame on the left and nothing is
# ever pasted on top of it. The right column is a styled state card (top), a
# one-line label bar, and the close-up (bottom), so the three zones are
# disjoint: the panorama is never occluded, and the readout never covers the
# close-up. The composed 1440x810 canvas is 16:9 instead of the earlier 8:3,
# which kept the panorama very letterboxed in a normal player.
MAIN_RES = (960, 810)
STATE_RES = (480, 430)
LABEL_H = 26
CLOSE_RES = (480, 810 - STATE_RES[1] - LABEL_H)  # 480x354
FRAME_RES = (MAIN_RES[0] + STATE_RES[0], 810)  # 1440x810 (16:9)
CARD_BG = (22, 25, 31)
CARD_BORDER = (52, 58, 70)
CARD_TRACK = (44, 50, 60)
TEXT_COLOR = (238, 242, 247)
DIM_COLOR = (156, 166, 180)
FAINT_COLOR = (112, 122, 136)
ACCENT_COLOR = (255, 199, 64)
LEGEND_COLORS = {
    "green": (90, 200, 90),
    "red": (226, 74, 60),
    "yellow": (240, 205, 60),
    "cyan": (80, 200, 225),
    "orange": (235, 145, 55),
    "blue": (95, 130, 235),
    "grey": (150, 155, 165),
    "white": (240, 242, 245),
}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# rendezvous display furniture (all display-only, metres of the CWH LVLH frame)
RING_RADII = (250.0, 500.0, 1000.0, 2000.0)  # range rings; 2000 m = range_fail
RING_SEGMENTS = 48
RING_PLANE = 0.0  # rings lie in the x-y (radial/along-track) plane
VEL_ARROW_SCALE = 100.0  # arrow length in metres per m/s of true relative speed
ARROW_SIZES = (
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
)  # quantized arrow pool [m] (set_scale unsupported)
DOCKING_ARROW_SCALE = 10.0  # 10 m per m/s keeps arrows inside the 120 m approach
DOCKING_ARROW_SIZES = (2.0, 5.0, 10.0, 20.0, 40.0)
# outdoor scenes use a light sky background instead of the default near-black
# one: with a near-horizon camera the ground stops at the camera far plane and
# the uncovered screen corners then read as a dark wedge
SKY_COLOR = (0.55, 0.65, 0.80)
GROUND_COLOR = (0.55, 0.55, 0.55, 1.0)
GROUND_SIZE = 20000.0  # metres; large enough that the ground reaches the horizon
WORLD_UP = (0.0, 0.0, 1.0)  # keep every inset free of camera roll



def build_env(task):
    """Single-replica CPU env, same construction as baseline_eval.py."""
    out = Path(__file__).resolve().parent.parent / "results" / task
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    env_kwargs = dict(cfg.get("env_kwargs") or {})
    return ENVS[task](num_envs=1, backend="cpu", seed=7, **env_kwargs), cfg


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
        from rsl_rl.runners import OnPolicyRunner  # noqa: E402

        out = Path(__file__).resolve().parent.parent / "results" / task
        ckpt = out / "checkpoint.pt"
        assert ckpt.exists(), f"missing {ckpt}"
        dist = cfg.get("actor", {}).get("distribution_cfg", {})
        if isinstance(dist.get("init_std"), list):
            dist["init_std"] = _JsonStd(dist["init_std"])
        if cfg.get("algo") == "flash_sac":
            raise RuntimeError(
                "flash_sac checkpoints are no longer supported (algorithm removed); "
                f"re-train with PPO to visualize {ckpt}"
            )
        runner = OnPolicyRunner(
            RslVecEnv(env), cfg, log_dir="/tmp/vis_tb", device=str(env.device)
        )
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
    if task in ("quadx", "fixedwing", "quadplane"):
        extras["targets"] = env.targets.cpu().numpy()[0]
        if task == "quadx":
            extras["target"] = extras["targets"][0]
    elif task == "quadx_track":
        extras["occluders"] = OCCLUDERS
    elif task in ("docking", "rendezvous"):
        extras["keep_out"] = env.keep_out
        extras["pos_goal"] = env.pos_goal
        extras["vel_goal"] = env.vel_goal
        extras["range_fail"] = env.range_fail
    elif task == "satellite":
        # display-only CMG geometry (the env's own published cluster constants);
        # gimbal ANGLES come from the recorded state each frame
        extras["gimbal_axes"] = env.gimbal_axes.cpu().numpy()
        extras["rotor_axes_ref"] = env.rotor_axes_ref.cpu().numpy()
    elif task == "transfer":
        # display-only orbital geometry (never touches the sim): fixed initial
        # Keplerian elements + the per-episode apogee/perigee radii the reward
        # is actually computed from
        extras["orbit"] = dict(
            sma=env.sma0,
            ecc=env.ecc0,
            inc=env.inc0,
            raan=env.raan0,
            argp=env.argp0,
            ra0=float(env.ra0.cpu()[0]),
            rp0=float(env.rp0.cpu()[0]),
        )

    def snap():
        if task == "missile":
            frames.append(
                {
                    "pos": env.plant.drone.get_pos().cpu().numpy()[0],
                    "quat": env.plant.drone.get_quat().cpu().numpy()[0],
                    "tgt": env.target_pos.cpu().numpy()[0].copy(),
                }
            )
        elif task in ("quadx", "fixedwing", "quadplane", "rocket"):
            frames.append(
                {
                    "pos": env.plant.drone.get_pos().cpu().numpy()[0],
                    "quat": env.plant.drone.get_quat().cpu().numpy()[0],
                }
            )
            if task == "quadx":
                frames[-1]["goal"] = env.targets[0, env.target_index[0]].cpu().numpy().copy()
        elif task == "quadx_track":
            frames.append(
                {
                    "pos": env.plant.drone.get_pos().cpu().numpy()[0],
                    "quat": env.plant.drone.get_quat().cpu().numpy()[0],
                    "tgt": env.target_pos.cpu().numpy()[0].copy(),
                    "visible": bool(env.vis.cpu()[0]),
                }
            )
        elif task == "satellite":
            frames.append(
                {
                    "pos": np.zeros(3),
                    "quat": env.q.cpu().numpy()[0].copy(),
                    "target_quat": env.target_quat.cpu().numpy()[0].copy(),
                    "gimbal": env.gimbal.cpu().numpy()[0].copy(),
                }
            )
        elif task == "transfer":
            frames.append(
                {"pos": env.r.cpu().numpy()[0].copy() * 1e-6, "quat": None}
            )  # 1 unit = 1000 km
        else:  # docking / rendezvous: CWH relative frame, target at the origin
            f = {
                "pos": env.p.cpu().numpy()[0].copy(),
                "quat": None,
                "vel": env.v.cpu().numpy()[0].copy(),
            }
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
            to = bool(env.extras.get("time_outs", torch.zeros(1, dtype=torch.bool)).any())
            ep = env.extras.get("episode")
            sr = float(ep["success_rate"]) if ep else 0.0
            extras["terminal"] = (
                "SUCCESS" if sr > 0.5 else "TIMEOUT (no capture)" if to else "FAILURE"
            )
            break
        snap()
    extras["steps"] = steps
    return frames, extras


def _urdf(scene, path, scale=1.0, **kw):
    """Load a URDF visually, sanitizing the upstream rocket's malformed XML
    (duplicated closing tag; same fix as common.original_backend.loadURDF)."""
    import copy  # noqa: E402
    import tempfile  # noqa: E402
    import xml.etree.ElementTree as ET  # noqa: E402

    raw = Path(path).read_text()
    root = ET.fromstring(raw.split("</robot>")[0] + "</robot>")
    # PyFlyt's quadplane URDF specifies only collision geometry. PyBullet can
    # display it, but Genesis needs an explicit visual node in the temporary
    # render-only copy. Reuse the exact asset geometry; never edit the source.
    for link in root.findall("link"):
        if link.find("visual") is None and link.find("collision") is not None:
            visual = ET.Element("visual")
            for child in link.find("collision"):
                visual.append(copy.deepcopy(child))
            link.append(visual)
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((Path(path).parent / mesh.attrib["filename"]).resolve()))
    tmp = Path(tempfile.gettempdir()) / f"vis_{Path(path).name}"
    ET.ElementTree(root).write(tmp)
    return scene.add_entity(
        gs.morphs.URDF(file=str(tmp), scale=scale, collision=False, **kw)
    )



def _urdf_lowest_z(path):
    """Lowest z of a URDF's geometry BEFORE scaling, walking fixed joints.

    Per-link primitive bounds alone are not enough: the rocket's three landing
    legs are boxes on joints at z=-2.4, i.e. 4 cm lower than the body cylinder.
    Only primitives are inspected (cylinder/box); meshes are ignored.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(Path(path).read_text().split("</robot>")[0] + "</robot>")
    links = root.findall("link")
    if not links:
        return 0.0
    parent_of, joint_z = {}, {}
    for joint in root.findall("joint"):
        if joint.find("child") is None or joint.find("parent") is None:
            continue
        child = joint.find("child").attrib["link"]
        parent_of[child] = joint.find("parent").attrib["link"]
        origin = joint.find("origin")
        joint_z[child] = (
            float(origin.attrib.get("xyz", "0 0 0").split()[2]) if origin is not None else 0.0
        )

    def link_z(name):
        z, guard = 0.0, 0
        while name in parent_of and guard < 64:
            z += joint_z[name]
            name = parent_of[name]
            guard += 1
        return z

    lowest = 0.0
    for link in links:
        z = link_z(link.attrib["name"])
        # collision geometry only: the rocket's `flame` link is a visual-only
        # plume reaching 4.4 m below the link origin and must not anchor it
        geoms = link.findall("collision")
        for geom in geoms:
            origin = geom.find("origin")
            z_geom = z + (
                float(origin.attrib.get("xyz", "0 0 0").split()[2]) if origin is not None else 0.0
            )
            geometry = geom.find("geometry")
            if geometry is None:
                continue
            for g in geometry:
                if g.tag == "cylinder":
                    extent = float(g.attrib["length"]) / 2.0
                elif g.tag == "box":
                    extent = float(g.attrib["size"].split()[2]) / 2.0
                elif g.tag == "sphere":
                    extent = float(g.attrib["radius"])
                else:
                    continue
                lowest = min(lowest, z_geom - extent)
    return lowest


def _scale_ground_lift(lowest_z, scale):
    """Vertical offset that keeps a display-enlarged body out of the ground.

    Genesis scales URDF geometry about each link origin, so a body whose real
    bottom clears the pad by centimetres would have its scaled bottom
    (scale-1)*|lowest_z| below the pad -- the enlarged airframe sinks into it.
    Scale about the LOWEST point instead: the offset below puts the displayed
    bottom back on the true (unscaled) bottom, so the model stands on the pad.
    """
    if scale <= 1.0 or lowest_z >= 0.0:
        return 0.0
    return -lowest_z * (scale - 1.0)


def _ground(scene, size=GROUND_SIZE):
    """Flat grey ground wide enough to reach the horizon of every camera."""
    return scene.add_entity(
        gs.morphs.Plane(plane_size=(size, size)),
        surface=gs.surfaces.Plastic(color=GROUND_COLOR),
    )


def _marker(scene, radius, color, pos=(0, 0, 0)):
    return scene.add_entity(
        gs.morphs.Sphere(pos=tuple(pos), radius=radius, fixed=False, collision=False),
        surface=gs.surfaces.Plastic(color=color),
    )


def _box(scene, size, color, pos=(0, 0, 0)):
    return scene.add_entity(
        gs.morphs.Box(pos=tuple(pos), size=tuple(size), fixed=False, collision=False),
        surface=gs.surfaces.Plastic(color=color),
    )


def _trail(scene, radius, color):
    return [_marker(scene, radius, color, (0, 0, -1e5)) for _ in range(TRAIL_MARKERS)]


def _update_trail(markers, path):
    """Replay historical positions without covering the current entity."""
    history = path[:-1]
    n = min(len(history), TRAIL_MARKERS)
    picked = (
        np.linspace(0, len(history) - 1, n).astype(int)
        if len(history) > 1
        else np.array([0], dtype=int)
    )
    for i, m in enumerate(markers):
        p = history[picked[i]] if i < n else HIDDEN
        m.set_pos(tuple(map(float, p)))


def _follow(cam, p, offset, lookat=None):
    p = np.asarray(p, dtype=float)
    cam.set_pose(
        pos=tuple(p + np.asarray(offset)),
        lookat=tuple(np.asarray(lookat, dtype=float) if lookat is not None else p),
        up=WORLD_UP,
    )


def _dir_quat(d):
    """wxyz quaternion rotating the box long axis (z) onto direction d."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    ax = np.cross((0.0, 0.0, 1.0), d)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0) if d[2] > 0 else (0.0, 1.0, 0.0, 0.0)
    ax = ax / s
    ang = 0.5 * math.acos(np.clip(d[2], -1.0, 1.0))
    return (math.cos(ang), *(ax * math.sin(ang)))


def _quat_x_to(d):
    """wxyz quaternion rotating the local +X axis onto direction d (the
    chaser's docking face is +X, so this orients the chaser toward the
    target station)."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    ax = np.cross((1.0, 0.0, 0.0), d)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0) if d[0] > 0 else (0.0, 0.0, 1.0, 0.0)
    ax = ax / s
    ang = 0.5 * math.acos(np.clip(d[0], -1.0, 1.0))
    return (math.cos(ang), *(ax * math.sin(ang)))


def _cyl(scene, radius, length, color, pos=HIDDEN):
    """Thin cylinder marker whose long axis can be oriented via set_quat
    (Genesis cylinders default to the local z axis)."""
    return scene.add_entity(
        gs.morphs.Cylinder(pos=tuple(pos), radius=radius, height=length, fixed=False, collision=False),
        surface=gs.surfaces.Plastic(color=color),
    )


def _quat_z_to(d):
    """wxyz quaternion rotating the local +Z axis onto direction d."""
    d = np.asarray(d, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    d = d / n
    ax = np.cross((0.0, 0.0, 1.0), d)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0) if d[2] > 0 else (0.0, 1.0, 0.0, 0.0)
    ax = ax / s
    ang = 0.5 * math.acos(np.clip(d[2], -1.0, 1.0))
    return (math.cos(ang), *(ax * math.sin(ang)))


def _quat_mul_np(a, b):
    """Hamilton product of two wxyz quaternions (numpy)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _quat_rot(q, v):
    """Rotate vector v (3,) by wxyz quaternion q (display-only numpy math)."""
    qv = np.append(0.0, v)
    c = np.array((q[0], -q[1], -q[2], -q[3]))  # conjugate of (w, x, y, z)
    return _quat_mul_np(_quat_mul_np(q, qv), c)[1:]


def _quat_angle(q1, q2):
    """Shortest-arc angle [rad] between two wxyz quaternions."""
    d = abs(float(np.dot(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def _perp_basis(d):
    """Orthonormal (u, w) pair perpendicular to unit vector d."""
    ref = np.array((0.0, 0.0, 1.0)) if abs(d[2]) < 0.9 else np.array((1.0, 0.0, 0.0))
    u = np.cross(d, ref)
    u = u / np.linalg.norm(u)
    return u, np.cross(d, u)


def _font(size, bold=True):
    from PIL import ImageFont  # noqa: E402

    path = FONT_PATH if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _fit(draw, text, font, max_px):
    """Trim a string with an ellipsis so it never leaves its column."""
    if draw.textlength(text, font=font) <= max_px:
        return text
    while text and draw.textlength(text + "\u2026", font=font) > max_px:
        text = text[:-1]
    return text + "\u2026"


def _hfov_match(fov, new_res, old_res=(960, 540)):
    """Vertical fov that keeps the old horizontal field of view.

    Genesis' ``fov`` is vertical. Changing the render aspect (e.g. from 960x540
    to a taller 960x810 panorama) would otherwise crop the horizontal framing
    the cameras were set up for, so widen the vertical fov by the aspect ratio
    instead: same left/right coverage, more sky/ground above and below.
    """
    aspect_old = old_res[0] / old_res[1]
    aspect_new = new_res[0] / new_res[1]
    t = math.tan(math.radians(fov) / 2.0) * aspect_old / aspect_new
    return math.degrees(2.0 * math.atan(t))


def _state_card(state, height=STATE_RES[1]):
    """Styled state readout: title, hero metric, key/value rows, legend, footer.

    Every string is pre-authored per task and trimmed to its column, so the
    card can never overflow into the label bar or the close-up.
    """
    from PIL import Image, ImageDraw  # noqa: E402

    w = STATE_RES[0]
    img = Image.new("RGB", (w, height), CARD_BG)
    dr = ImageDraw.Draw(img)
    dr.rounded_rectangle((0, 0, w - 1, height - 1), radius=10, outline=CARD_BORDER)
    x0, x1 = 18, w - 18

    title_f = _font(19)
    tag_f = _font(10, bold=False)
    dr.text((x0, 14), _fit(dr, state["task"], title_f, x1 - x0 - 60),
            font=title_f, fill=TEXT_COLOR)
    tw = dr.textlength("STATE", font=tag_f)
    dr.text((x1 - tw, 22), "STATE", font=tag_f, fill=FAINT_COLOR)
    y = 44
    dr.line((x0, y, x1, y), fill=CARD_BORDER)
    y += 18

    # hero metric: small label over one large value
    lab_f = _font(10, bold=False)
    hero_f = _font(40)
    dr.text((x0, y), state["hero"][0].upper(), font=lab_f, fill=FAINT_COLOR)
    dr.text((x0, y + 17), _fit(dr, state["hero"][1], hero_f, x1 - x0),
            font=hero_f, fill=TEXT_COLOR)
    y += 82
    dr.line((x0, y, x1, y), fill=CARD_BORDER)
    y += 16

    row_lab_f = _font(12, bold=False)
    row_val_f = _font(14)
    for label, value in state.get("rows", ()):
        dr.text((x0, y + 2), label, font=row_lab_f, fill=DIM_COLOR)
        value = _fit(dr, value, row_val_f, (x1 - x0) * 0.62)
        dr.text((x1 - dr.textlength(value, font=row_val_f), y), value,
                font=row_val_f, fill=TEXT_COLOR)
        y += 28

    legend = state.get("legend") or ()
    if legend:
        y += 4
        dr.text((x0, y), "LEGEND", font=_font(9, bold=False), fill=FAINT_COLOR)
        y += 16
        leg_f = _font(11, bold=False)
        for color, text in legend:
            fill = LEGEND_COLORS.get(color, LEGEND_COLORS["white"])
            dr.rounded_rectangle((x0, y + 3, x0 + 9, y + 12), radius=2, fill=fill)
            dr.text((x0 + 18, y + 1),
                    _fit(dr, text, leg_f, x1 - x0 - 24), font=leg_f, fill=DIM_COLOR)
            y += 21

    note = state.get("note")
    footer_y = height - 72
    if note:
        dr.text((x0, min(y + 6, footer_y - 22)),
                _fit(dr, note, _font(10, bold=False), x1 - x0),
                font=_font(10, bold=False), fill=FAINT_COLOR)

    # footer: episode progress plus the env's own terminal outcome
    result = state.get("result")
    if result:
        pill_f = _font(11)
        color = {
            "SUCCESS": (46, 150, 82),
            "FAILURE": (170, 62, 58),
        }.get(result, (176, 128, 44))
        pw = dr.textlength(result, font=pill_f) + 22
        dr.rounded_rectangle((x1 - pw, footer_y, x1, footer_y + 24),
                             radius=12, fill=color)
        dr.text((x1 - pw + 11, footer_y + 5), result, font=pill_f,
                fill=(245, 248, 250))
    dr.text((x0, footer_y + 6), "EPISODE", font=_font(9, bold=False),
            fill=FAINT_COLOR)
    track_y = footer_y + 38
    dr.rounded_rectangle((x0, track_y, x1, track_y + 6), radius=3, fill=CARD_TRACK)
    frac = float(state.get("progress") or 0.0)
    if frac > 0.0:
        dr.rounded_rectangle((x0, track_y, x0 + max(6, int((x1 - x0) * frac)),
                              track_y + 6), radius=3, fill=ACCENT_COLOR)
    dr.text((x0, track_y + 12), state.get("clock", ""),
            font=_font(10, bold=False), fill=DIM_COLOR)
    return img


def _frame(main_rgb, close_rgb, state, label=None):
    """Tile panorama | state card / label bar / close-up without any overlap.

    main_rgb keeps its native 960x810 frame at the far left; the right column
    holds the readout card on top and the close-up below, separated by a
    one-line label bar. Every pixel belongs to exactly one zone, so the
    panorama is never covered and the readout never sits on the close-up.
    """
    from PIL import Image  # noqa: E402

    canvas = Image.new("RGB", FRAME_RES, (10, 11, 13))
    canvas.paste(Image.fromarray(main_rgb), (0, 0))
    canvas.paste(_state_card(state), (MAIN_RES[0], 0))
    if close_rgb is not None:
        # ---- label bar ----
        bar = Image.new("RGB", (STATE_RES[0], LABEL_H), (28, 32, 39))
        from PIL import ImageDraw  # noqa: E402

        db = ImageDraw.Draw(bar)
        db.rectangle((0, 0, 2, LABEL_H - 1), fill=ACCENT_COLOR)
        db.text((14, 5), label or "CLOSE-UP", fill=ACCENT_COLOR, font=_font(11))

        # ---- close-up (bottom right) ----
        close = Image.fromarray(close_rgb)
        if close.size != CLOSE_RES:
            close = close.resize(CLOSE_RES, Image.LANCZOS)
        canvas.paste(bar, (MAIN_RES[0], STATE_RES[1]))
        canvas.paste(close, (MAIN_RES[0], STATE_RES[1] + LABEL_H))

    # ---- zone separators ----
    # All separators live inside the right column so the 960x810 panorama is
    # left byte-for-byte intact (no dark line even on its last two columns).
    from PIL import ImageDraw  # noqa: E402

    dcs = ImageDraw.Draw(canvas)
    dcs.rectangle((MAIN_RES[0], 0, MAIN_RES[0] + 1, FRAME_RES[1] - 1), fill=(8, 9, 11))
    if close_rgb is not None:
        dcs.rectangle(
            (MAIN_RES[0], STATE_RES[1] - 2, FRAME_RES[0] - 1, STATE_RES[1] - 1),
            fill=(8, 9, 11),
        )
        dcs.rectangle(
            (MAIN_RES[0], STATE_RES[1] + LABEL_H - 2, FRAME_RES[0] - 1,
             STATE_RES[1] + LABEL_H - 1),
            fill=(8, 9, 11),
        )
    return np.asarray(canvas)



def build_scene(task, extras, first):
    """Visual-only scene; entities returned in a dict for the replay loop."""
    if not gs._initialized:  # pure-tensor tasks never construct a physics scene
        gs.init(backend=gs.cpu, logging_level="warning")
    # the four space tasks keep the default dark background (stars, orbit
    # lines); every atmospheric task gets a light sky so horizon corners are
    # never a black wedge
    outdoor = task in ("quadx", "quadx_track", "fixedwing", "quadplane", "rocket", "missile")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(gravity=(0, 0, 0)),
        renderer=gs.renderers.Rasterizer(),
        vis_options=gs.options.VisOptions(background_color=SKY_COLOR) if outdoor else None,
    )
    ents = {}
    p0 = first["pos"]
    if task == "quadx":
        # PyFlyt-style visualization (mirrors PyFlyt 0.29 quadx_envs): plain
        # grey ground, the cf2x URDF (enlarged 6x so the ~9 cm airframe is
        # readable at dome scale), an opaque solid green sphere at the active
        # waypoint (PyFlyt's waypoint_handler scaling, enlarged to stay visible),
        # and a fixed global camera. The close-up inset keeps the true camera
        # geometry of PyFlyt's yaw-30 / pitch-30 debug view.
        _ground(scene)
        ents["body"] = _urdf(scene, MODELS / "vehicles/cf2x/cf2x.urdf", scale=6.0)
        ents["goal"] = _marker(scene, 0.18, (0.0, 1.0, 0.0, 1.0), extras["target"])
        cam = scene.add_camera(
            res=MAIN_RES, pos=(6, -8, 5.5), lookat=(0, 0, 2.0),
            fov=_hfov_match(45, MAIN_RES), far=2000
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=5000
        )
    elif task == "quadx_track":
        # Reuse the PyFlyt cf2x model (enlarged 8x for readability). Only the
        # target, true LOS and the environment's geometric occluders are added.
        _ground(scene)
        ents["body"] = _urdf(scene, MODELS / "vehicles/cf2x/cf2x.urdf", scale=12.0)
        ents["target"] = _marker(scene, 1.0, (0.9, 0.15, 0.1, 1.0), first["tgt"])
        ents["occluders"] = []
        for x, y, radius, height in extras["occluders"]:
            ent = _cyl(
                scene, radius, height, (0.35, 0.35, 0.38, 0.65), (x, y, height / 2.0)
            )
            ents["occluders"].append(ent)
        ents["los"] = [_marker(scene, 0.05, (1.0, 1.0, 1.0, 0.65)) for _ in range(24)]
        ents["trail"] = _trail(scene, 0.08, (0.2, 0.4, 0.9, 0.6))
        ents["target_trail"] = _trail(scene, 0.08, (0.9, 0.15, 0.1, 0.7))
        cam = scene.add_camera(
            res=MAIN_RES, pos=(19, -19, 12.5), lookat=(0, 0, 4.5),
            fov=_hfov_match(55, MAIN_RES), far=3000
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=5000
        )
    elif task == "fixedwing":
        _ground(scene)
        ents["body"] = _urdf(
            scene, MODELS / "vehicles/fixedwing/fixedwing.urdf", scale=8.0
        )
        ents["wps"] = [_marker(scene, 2.5, (0.9, 0.2, 0.2, 0.9), t) for t in extras["targets"]]
        ents["trail"] = _trail(scene, 0.5, (0.2, 0.4, 0.9, 0.6))
        # fixed global camera centred on the waypoint cluster; the panorama is
        # its own zone in the composed frame, so nothing is panned away
        cam = scene.add_camera(
            res=MAIN_RES, pos=(85, -85, 70), lookat=(0, 0, 30),
            fov=_hfov_match(55, MAIN_RES), far=5000
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=5000
        )
    elif task == "quadplane":
        # Same PyFlyt waypoint scene convention as Fixedwing, with the native
        # quadplane URDF (enlarged 6x); no synthetic tilt/mode animation is
        # introduced.
        _ground(scene)
        ents["body"] = _urdf(
            scene, MODELS / "vehicles/quadplane/quadplane.urdf", scale=8.0
        )
        ents["wps"] = [_marker(scene, 2.5, (0.9, 0.2, 0.2, 0.9), t) for t in extras["targets"]]
        ents["trail"] = _trail(scene, 0.5, (0.2, 0.4, 0.9, 0.6))
        # same fixed global framing as Fixedwing
        cam = scene.add_camera(
            res=MAIN_RES, pos=(100, -100, 74), lookat=(0, 0, 32),
            fov=_hfov_match(55, MAIN_RES), far=5000
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=5000
        )
    elif task == "rocket":
        _ground(scene)
        _urdf(scene, MODELS / "landing_pad.urdf", pos=(0, 0, 0.1))
        # 5.4 m airframe at ~450 m altitude would be a few pixels: enlarged 8x.
        # The display scale is anchored at the airframe's lowest point (computed
        # after scene.build below), otherwise the 8x body would sink ~17 m into
        # the pad as it lands (the true base only clears the pad by centimetres).
        ents["body"] = _urdf(scene, MODELS / "vehicles/rocket/rocket.urdf", scale=8.0)
        ents["trail"] = _trail(scene, 0.5, (0.2, 0.4, 0.9, 0.6))
        cam = scene.add_camera(
            res=MAIN_RES, pos=(350, -350, 420), lookat=(0, 0, 250),
            fov=_hfov_match(60, MAIN_RES), far=8000
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=8000
        )

    elif task == "satellite":
        # Recognizable bus: grey body + equipment compartments on the +/-z
        # faces (up/down), sensor head on +x (front), nozzle on -x (rear),
        # antenna on top, solar panels on +/-y with a light grid. Panels and
        # details are rigidly attached to the bus: each stores (entity,
        # local_pos, local_quat) and is posed per frame from the recorded
        # quaternion. All geometry here is display-only (collision=False,
        # scene never stepped) and only built when visualize.py is run.
        ents["body"] = _box(scene, (0.8, 0.8, 0.8), (0.72, 0.75, 0.8, 1.0))
        ents["attach"] = []

        def _attach(ent, pos, quat=(1.0, 0.0, 0.0, 0.0)):
            ents["attach"].append((ent, np.asarray(pos, float), tuple(quat)))

        # up/down equipment compartments distinguish body +z from -z
        _attach(_box(scene, (0.5, 0.5, 0.14), (0.45, 0.48, 0.55, 1.0)), (0, 0, 0.46))
        _attach(_box(scene, (0.5, 0.5, 0.14), (0.45, 0.48, 0.55, 1.0)), (0, 0, -0.46))
        # front (+x) sensor head and rear (-x) nozzle distinguish +x from -x
        _attach(_box(scene, (0.12, 0.3, 0.3), (0.35, 0.38, 0.42, 1.0)), (0.46, 0, 0))
        _attach(
            _cyl(scene, 0.1, 0.22, (0.25, 0.25, 0.28, 1.0)),
            (-0.5, 0, 0),
            _quat_x_to((-1.0, 0.0, 0.0)),
        )
        # top antenna: boom + gold dish
        _attach(_cyl(scene, 0.025, 0.35, (0.8, 0.8, 0.82, 1.0)), (0, 0, 0.68))
        _attach(_cyl(scene, 0.13, 0.05, (0.9, 0.75, 0.3, 1.0)), (0, 0, 0.88))
        # +/-y solar panels: dark panel, yoke to the bus, light grid lines
        for s in (1.0, -1.0):
            _attach(_box(scene, (0.85, 2.1, 0.04), (0.12, 0.22, 0.5, 1.0)), (0, s * 1.45, 0))
            _attach(_box(scene, (0.3, 0.4, 0.12), (0.55, 0.58, 0.62, 1.0)), (0, s * 0.55, 0))
            for k in (-0.68, 0.0, 0.68):
                _attach(
                    _box(scene, (0.87, 0.05, 0.06), (0.07, 0.13, 0.32, 1.0)), (0, s * 1.45 + k, 0)
                )
            for k in (-0.28, 0.28):
                _attach(_box(scene, (0.05, 2.1, 0.06), (0.07, 0.13, 0.32, 1.0)), (k, s * 1.45, 0))
        # body-frame triad (X red, Y green, Z blue) rotating with the bus
        for d, col in (
            ((1.0, 0.0, 0.0), (0.9, 0.25, 0.25, 0.95)),
            ((0.0, 1.0, 0.0), (0.25, 0.8, 0.3, 0.95)),
            ((0.0, 0.0, 1.0), (0.3, 0.5, 0.95, 0.95)),
        ):
            _attach(_box(scene, (0.045, 0.045, 1.3), col), np.array(d) * 0.65, _quat_z_to(d))
        # target-attitude triad (yellow), posed from the recorded target quat
        ents["triad_target"] = [
            (
                _box(scene, (0.03, 0.03, 1.1), (0.95, 0.8, 0.2, 0.9)),
                np.array(d) * 0.55,
                _quat_z_to(d),
            )
            for d in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ]
        # red ghost box at the target attitude (legacy marker)
        ents["goal"] = _box(scene, (0.8, 0.8, 0.8), (0.9, 0.25, 0.2, 0.3))
        # one cyan cylinder per CMG, oriented along the instantaneous rotor
        # momentum axis r_i(theta) (real gimbal state; no faked rotor spin)
        ga = extras["gimbal_axes"]
        ents["cmg"] = [_cyl(scene, 0.035, 0.55, (0.2, 0.9, 0.95, 0.95)) for _ in ga]
        # static star decoration (fixed positions, never moves)
        rng = np.random.default_rng(3)
        for p in rng.normal(size=(70, 3)):
            p = p / np.linalg.norm(p) * 22.0
            _marker(scene, 0.06, (1.0, 1.0, 1.0, 0.8), p)
        # FIXED oblique camera: never follows the tumbling body, so the bus
        # stays clear and stable throughout the episode
        cam = scene.add_camera(
            res=MAIN_RES, pos=(3.4, -3.4, 2.15), lookat=(0, 0, 0),
            fov=_hfov_match(45, MAIN_RES), far=100
        )
        # The main camera is the fixed global attitude view; the close-up is a
        # second fixed camera closer to the bus so the CMG axes and body triad
        # stay readable. Panels may sweep out of the close-up frame during a
        # large slew, which is what a close-up does; the panorama keeps them.
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=100
        )
    elif task == "transfer":
        # 3D dynamics (i = pi/3), so a 3D view centered on Earth. Orbit lines
        # and markers are display-only geometry derived from the recorded
        # initial elements; nothing here feeds back into the sim.
        o = extras["orbit"]
        # perifocal -> inertial rotation, same convention as env._kepler_to_cartesian
        cO, sO = math.cos(o["raan"]), math.sin(o["raan"])
        ci, si = math.cos(o["inc"]), math.sin(o["inc"])
        cw, sw = math.cos(o["argp"]), math.sin(o["argp"])
        R = np.array(
            [
                [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                [sw * si, cw * si, ci],
            ]
        )
        a, e = o["sma"], o["ecc"]
        b = a * math.sqrt(1.0 - e * e)
        E = np.linspace(0.0, 2.0 * np.pi, ORBIT_POINTS, endpoint=False)
        pf = np.stack((a * (np.cos(E) - e), b * np.sin(E), np.zeros_like(E)), -1)
        scale = 1e-6  # 1 scene unit = 1000 km
        ents["orbit0"] = [  # dashed green initial orbit (true ellipse)
            _marker(scene, 0.07 if i % 2 == 0 else 0.03, (0.2, 0.8, 0.3, 0.9), p)
            for i, p in enumerate(pf @ R.T * scale)
        ]
        # dotted cyan target orbit: circle at the initial apogee radius
        # (ideal perigee-raising end state, same plane at reference thrust)
        circ = np.stack((o["ra0"] * np.cos(E), o["ra0"] * np.sin(E), np.zeros_like(E)), -1)
        ents["orbit_t"] = [
            _marker(scene, 0.04, (0.2, 0.8, 0.9, 0.9), p @ R.T * scale) for p in circ
        ]
        # perigee (red) / apogee (yellow) of the initial orbit
        ents["peri"] = _marker(scene, 0.22, (0.9, 0.2, 0.2, 1.0), R[:, 0] * a * (1.0 - e) * scale)
        ents["apo"] = _marker(scene, 0.22, (0.95, 0.85, 0.2, 1.0), R[:, 0] * -a * (1.0 + e) * scale)
        ents["earth"] = _marker(scene, 6.371, (0.15, 0.35, 0.75, 1.0))
        ents["body"] = _box(scene, (1.2, 1.2, 2.4), (1.0, 0.65, 0.1, 1.0))  # enlarged spacecraft
        ents["trail"] = _trail(scene, 0.12, (1.0, 0.65, 0.1, 1.0))
        # fixed camera centered on Earth, tilted ~45 deg off the orbital
        # plane so the ellipse shape stays visible for the whole episode
        n = R[:, 2] / np.linalg.norm(R[:, 2])
        u = np.cross(n, (0.0, 0.0, 1.0))
        u /= np.linalg.norm(u)
        campos = -36.0 * u + 36.0 * n  # ~45 deg above the orbital plane
        cam = scene.add_camera(
            res=MAIN_RES, pos=tuple(campos), lookat=(0, 0, 0),
            fov=_hfov_match(45, MAIN_RES), far=300
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=300
        )
        # 2000 km scale bar (2 scene units), below the frame center
        view = -campos / np.linalg.norm(campos)
        right = np.cross(view, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, view)
        base = -up * 15.0 - right
        ents["scalebar"] = [
            _marker(scene, 0.18, (1.0, 1.0, 1.0, 0.95), base + right * i) for i in range(5)
        ]
    elif task == "missile":
        # km-scale air-intercept scene at ~4 km altitude: missile URDF (enlarged
        # 200x, otherwise a 3 m body is sub-pixel at engagement range), red
        # marker for the kinematic target aircraft (posed per frame from the
        # recorded target state), a yellow locator dot riding on the missile
        # (the narrow white airframe alone is easy to lose against the ground),
        # engagement trail, fixed global camera.
        # A far ground plane plus the light sky background give the engagement
        # an unambiguous horizon instead of an empty dark frame.
        _ground(scene, 100000.0)
        ents["body"] = _urdf(scene, MODELS / "vehicles/missile/missile.urdf", scale=200.0)
        ents["me"] = _marker(scene, 55.0, (1.0, 0.9, 0.25, 1.0), tuple(p0))
        ents["target"] = _marker(scene, 60.0, (0.9, 0.2, 0.2, 0.9), tuple(first["tgt"]))
        ents["trail"] = _trail(scene, 10.0, (0.2, 0.4, 0.9, 0.5))
        mid0 = 0.5 * (np.asarray(p0) + np.asarray(first["tgt"]))
        # Pan the look-at point along the engagement line (+x) so the pair sits
        # left of the bottom-right inset window for the whole approach.
        cam = scene.add_camera(
            res=MAIN_RES,
            pos=tuple(mid0 + np.array((0.0, -4500.0, 1300.0))),
            lookat=tuple(mid0),
            fov=_hfov_match(50, MAIN_RES),
            far=20000,
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(45, CLOSE_RES), far=20000
        )
    else:  # docking / rendezvous
        # Target station (grey) with two solar panels ATTACHED to the ±y
        # faces (no longer intersecting the core) and a docking port on the
        # +x face: green ring edge + centre marker + translucent axis
        # cylinder marking the approach corridor. The chaser (orange) gets
        # its own port ring on the face turned toward the target and a
        # forward-direction arrow; all marker positions follow the recorded
        # relative state each frame.
        big = task == "rendezvous"
        # Local bodies are magnified (display-only) because a 1.5 m station at
        # approach range is a handful of pixels; positions, ranges, goals and
        # keep-out radii stay in true metres and the overlay states the factor.
        k = 6.0 if big else 4.0
        s = (3.0 if big else 1.5) * k  # station body edge [m]
        c = (2.0 if big else 1.0) * k  # chaser body edge [m]
        px = s / 2.0  # docking-port face offset from origin
        ents["station"] = _box(scene, (s, s, s), (0.75, 0.78, 0.8, 1.0))
        pl, pt = ((7.0, 0.3) if big else (2.6, 0.1))
        pl, pt = pl * k, pt * k
        ents["panel_a"] = _box(
            scene,
            (1.6 * s if big else 2.2 * k, pl, pt),
            (0.25, 0.4, 0.9, 1.0),
            (0, s / 2.0 + pl / 2.0, 0),
        )
        ents["panel_b"] = _box(
            scene,
            (1.6 * s if big else 2.2 * k, pl, pt),
            (0.25, 0.4, 0.9, 1.0),
            (0, -s / 2.0 - pl / 2.0, 0),
        )
        # target docking port on the +x face: ring edge, centre marker, axis
        rr = (0.7 if big else 0.35) * k  # port ring radius [m]
        ents["t_ring"] = [
            _marker(
                scene,
                (0.12 if big else 0.05) * k,
                (0.2, 0.9, 0.4, 0.95),
                (px, rr * math.cos(2 * math.pi * i / 12.0), rr * math.sin(2 * math.pi * i / 12.0)),
            )
            for i in range(12)
        ]
        ents["t_center"] = _marker(
            scene, (0.16 if big else 0.07) * k, (0.1, 0.95, 0.3, 1.0), (px, 0, 0)
        )
        ents["t_axis"] = _cyl(
            scene,
            (0.12 if big else 0.05) * k,
            3.5 * s if big else 2.2 * k,
            (0.2, 0.9, 0.4, 0.28),
        )
        ents["body"] = _box(scene, (c, c, c), (1.0, 0.6, 0.1, 1.0))
        # chaser port ring (8 spheres) + centre marker, positioned on the
        # target-facing (+X local) face each frame
        ents["c_ring"] = [
            _marker(scene, (0.07 if big else 0.04) * k, (1.0, 0.8, 0.2, 0.95))
            for _ in range(8)
        ]
        ents["c_center"] = _marker(
            scene, (0.09 if big else 0.05) * k, (1.0, 0.85, 0.3, 1.0)
        )
        # dashed line-of-sight pool between the two centres
        ents["link"] = [
            _marker(scene, 1.2 if big else 0.2, (0.95, 0.95, 0.95, 0.55))
            for _ in range(20)
        ]
        ents["trail"] = _trail(
            scene, 2.0 if big else 0.35, (1.0, 0.6, 0.1, 0.9)
        )
        if extras.get("keep_out"):
            ents["keep_out"] = _marker(scene, extras["keep_out"], (0.9, 0.2, 0.2, 0.12))
        # relative-velocity arrow pool (quantized; true speed * per-task scale)
        if big:
            ents["arrows"] = {
                L: _box(scene, (2.5, 2.5, L), (1.0, 0.9, 0.25, 0.95), HIDDEN)
                for L in ARROW_SIZES
            }
        else:
            ents["arrows"] = {
                L: _box(scene, (3.6, 3.6, L), (1.0, 0.9, 0.25, 0.95), HIDDEN)
                for L in DOCKING_ARROW_SIZES
            }
        if big:  # rendezvous-only LVLH furniture
            # success goal ball (r = pos_goal = 10 m, the real criterion)
            ents["goal"] = _marker(scene, extras["pos_goal"], (0.2, 0.85, 0.3, 0.15))
            # LVLH axis bars through the target: x radial (red), y along-track
            # (green), z cross-track (blue); each spans +/-120 m
            ents["axes"] = (
                _box(scene, (240.0, 2.0, 2.0), (0.85, 0.35, 0.35, 0.55), (0, 0, 0)),
                _box(scene, (2.0, 240.0, 2.0), (0.35, 0.8, 0.35, 0.55), (0, 0, 0)),
                _box(scene, (2.0, 2.0, 240.0), (0.35, 0.55, 0.9, 0.55), (0, 0, 0)),
            )
            # range rings in the radial/along-track plane (x-y, z = RING_PLANE),
            # all in metres; the outermost is the 2000 m range-fail boundary
            ents["rings"] = [
                _marker(
                    scene,
                    2.5,
                    (0.85, 0.8, 0.5, 0.28),
                    (
                        r * math.cos(2 * math.pi * i / RING_SEGMENTS),
                        r * math.sin(2 * math.pi * i / RING_SEGMENTS),
                        RING_PLANE,
                    ),
                )
                for r in RING_RADII
                for i in range(RING_SEGMENTS)
            ]
        if task == "rendezvous":
            # fixed LVLH view centered on the target: the approach zone (rings
            # 250/500 m, goal and keep-out) stays readable; the far spawn only
            # appears in the chaser inset until the chaser enters this zone
            u = np.asarray(p0) / np.linalg.norm(p0)
            side = np.cross(u, (0.0, 0.0, 1.0))
            side /= np.linalg.norm(side)
            view = 0.35 * u + 0.75 * side + 0.45 * np.array((0.0, 0.0, 1.0))
            view /= np.linalg.norm(view)
            cam_pos = 550.0 * view
            look = (0.0, 0.0, 0.0)
        else:
            # docking: fixed view of the terminal approach (the 60-120 m spawn
            # is covered by the chaser inset until it enters the frame)
            cam_pos = np.array((42.0, -46.0, 30.0))
            look = (0.0, 0.0, 0.0)
        cam = scene.add_camera(
            res=MAIN_RES, pos=tuple(cam_pos), lookat=look,
            fov=_hfov_match(60, MAIN_RES), far=5000
        )
        ents["close_cam"] = scene.add_camera(
            res=CLOSE_RES, fov=_hfov_match(55, CLOSE_RES), far=3000
        )
    scene.build(n_envs=1)
    if task == "rocket":
        # display-only: after build the visual geometry is final, so the anchor
        # offset for the 8x airframe can be measured exactly
        ents["body_lift"] = _scale_ground_lift(
            _urdf_lowest_z(MODELS / "vehicles/rocket/rocket.urdf"), 8.0
        )
    return scene, ents, cam


def render(task, frames, extras, out_path, fps):
    scene, ents, cam = build_scene(task, extras, frames[0])
    trail = ents.get("trail")
    path = []
    target_path = []
    rgb_frames = []
    big = task == "rendezvous"
    space = task in ("docking", "rendezvous")
    k = 6.0 if big else 4.0
    c_size = (2.0 if big else 1.0) * k
    px = 0.5 * ((3.0 if big else 1.5) * k)  # target port face offset
    arrows = ents.get("arrows", {})
    link = ents.get("link", [])
    arrow_scale = VEL_ARROW_SCALE if big else DOCKING_ARROW_SCALE
    arrow_sizes = ARROW_SIZES if big else DOCKING_ARROW_SIZES
    if space:
        # static target approach-axis marker: translucent +x cylinder from
        # the port face outward
        L = 3.5 * ((3.0 if big else 1.5) * k) if big else 2.2 * k
        ents["t_axis"].set_pos((px + L / 2.0, 0.0, 0.0))
        ents["t_axis"].set_quat((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0))
    terminal = extras.get("terminal")
    n_frames = len(frames)

    def _state_task(name, hero, rows=(), legend=(), note=None, fi=0, result=None):
        frac, clock = (fi + 1) / n_frames, (
            f"t = {fi / 30.0:.1f} s / {n_frames / 30.0:.1f} s"
        )
        out = dict(task=name, hero=hero, rows=list(rows), legend=list(legend),
                   note=note, progress=frac, clock=clock)
        if result:
            out["result"] = result
        return out

    for fi, f in enumerate(frames):
        # display-only lift (rocket): keeps the enlarged airframe off the pad
        pos = tuple(map(float, f["pos"]))
        lift = ents.get("body_lift", 0.0)
        ents["body"].set_pos((pos[0], pos[1], pos[2] + lift))
        if f["quat"] is not None:
            ents["body"].set_quat(tuple(map(float, f["quat"])))
        if task == "missile":
            ents["target"].set_pos(tuple(map(float, f["tgt"])))
        if task == "quadx_track":
            ents["target"].set_pos(tuple(map(float, f["tgt"])))
        if task == "satellite":
            ents["goal"].set_quat(tuple(map(float, f["target_quat"])))
            q, qt = np.asarray(f["quat"], float), np.asarray(f["target_quat"], float)
            # rigidly attached bus details + body triad follow the bus quat
            for ent, lp, lq in ents["attach"]:
                ent.set_pos(tuple(map(float, _quat_rot(q, lp))))
                ent.set_quat(tuple(map(float, _quat_mul_np(q, lq))))
            # target triad follows the recorded target attitude
            for ent, lp, lq in ents["triad_target"]:
                ent.set_pos(tuple(map(float, _quat_rot(qt, lp))))
                ent.set_quat(tuple(map(float, _quat_mul_np(qt, lq))))
            # CMG rotor momentum axes from the REAL gimbal state:
            # r_i(theta) = cos(th) r_ref + sin(th) (g x r_ref)
            ga, rref = extras["gimbal_axes"], extras["rotor_axes_ref"]
            gim = f["gimbal"]
            for i, ent in enumerate(ents["cmg"]):
                d = math.cos(gim[i]) * rref[i] + math.sin(gim[i]) * np.cross(ga[i], rref[i])
                d = d / np.linalg.norm(d)
                ent.set_pos(tuple(map(float, _quat_rot(q, ga[i] * 0.52))))
                ent.set_quat(tuple(map(float, _quat_z_to(_quat_rot(q, d)))))
        elif trail is not None:
            path.append(f["pos"])
            _update_trail(trail, path)
        p = np.asarray(f["pos"], dtype=float)
        if task == "quadx":
            ents["goal"].set_pos(tuple(map(float, f["goal"])))
            # Fixed global main view; only the inset follows the micro-drone.
            yaw, pitch, dist = math.radians(30.0), math.radians(-30.0), 1.15
            off = np.array(
                (
                    dist * math.cos(pitch) * math.cos(yaw),
                    -dist * math.cos(pitch) * math.sin(yaw),
                    -dist * math.sin(pitch),
                )
            )
            ents["close_cam"].set_pose(
                pos=tuple(p + off), lookat=tuple(p), up=WORLD_UP
            )
            state = _state_task(
                "QuadX-Waypoints",
                ("Altitude", f"{p[2]:.2f} m"),
                [("Waypoints", "4 inside the 5 m dome"), ("Body", "scaled 6x")],
                [("green", "active waypoint"), ("grey", "ground shadow")],
                "fixed global camera · PyFlyt 30/30 close-up",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                cam.render(rgb=True, force_render=True)[0],
                ents["close_cam"].render(rgb=True, force_render=True)[0],
                state,
                "CLOSE-UP: quadx",
            )
            rgb_frames.append(rgb)
            continue
        if task == "quadx_track":
            tgt = np.asarray(f["tgt"], dtype=float)
            target_path.append(tgt)
            _update_trail(ents["target_trail"], target_path)
            for i, marker in enumerate(ents["los"]):
                marker.set_pos(tuple(p + (tgt - p) * (i + 1) / (len(ents["los"]) + 1)))
            yaw, pitch, dist = math.radians(30.0), math.radians(-30.0), 2.2
            off = np.array(
                (
                    dist * math.cos(pitch) * math.cos(yaw),
                    -dist * math.cos(pitch) * math.sin(yaw),
                    -dist * math.sin(pitch),
                )
            )
            ents["close_cam"].set_pose(
                pos=tuple(p + off), lookat=tuple(p), up=WORLD_UP
            )
            state = _state_task(
                "QuadX-Track",
                ("Altitude", f"{p[2]:.2f} m"),
                [("Target", "VISIBLE" if f["visible"] else "LOST"),
                 ("Body", "scaled 12x")],
                [("red", "ground target"), ("grey", "LOS occluder"),
                 ("white", "line of sight")],
                "fixed global view · true target state",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                cam.render(rgb=True, force_render=True)[0],
                ents["close_cam"].render(rgb=True, force_render=True)[0],
                state,
                "CLOSE-UP: quadx_track",
            )
            rgb_frames.append(rgb)
            continue
        if space:
            # chaser attitude: docking face (+X local) turned toward the
            # target; port ring / centre marker on that face; dashed
            # line-of-sight between the two centres, kept short of both
            # bodies so it never covers a port
            dist = float(np.linalg.norm(p))
            # The bodies are drawn enlarged (see the overlay magnification
            # note), so at terminal range the true separation would push the
            # orange chaser inside the grey station. Only the DRAWN chaser is
            # held at a minimum separation; ranges/speeds in the overlay and
            # the recorded trail stay in true metres.
            min_sep = 0.5 * (c_size + ((3.0 if big else 1.5) * k)) + 0.4 * k
            pd = p * max(1.0, min_sep / dist) if dist > 1e-9 else p
            d = -pd / np.linalg.norm(pd) if dist > 1e-9 else np.array((1.0, 0.0, 0.0))
            ents["body"].set_pos(tuple(map(float, pd)))
            ents["body"].set_quat(_quat_x_to(d))
            front = pd + d * (c_size / 2.0)
            uu, ww = _perp_basis(d)
            ring_r = (0.6 if big else 0.3) * k
            for i, m in enumerate(ents["c_ring"]):
                a = 2 * math.pi * i / len(ents["c_ring"])
                m.set_pos(tuple(front + (math.cos(a) * uu + math.sin(a) * ww) * ring_r))
            ents["c_center"].set_pos(tuple(front))
            for i, m in enumerate(link):
                m.set_pos(tuple(pd * (0.12 + 0.76 * i / (len(link) - 1))))
            # relative-velocity arrow: pool entry with true length
            # |v| * per-task arrow scale, oriented along v
            v = f.get("vel", np.zeros(3))
            speed = float(np.linalg.norm(v))
            for L, m in arrows.items():
                m.set_pos(tuple(HIDDEN))
            if speed > 1e-3 and arrows:
                ge = [sz for sz in arrow_sizes if sz >= speed * arrow_scale]
                L = min(ge) if ge else max(arrow_sizes)
                if L > 0.5 * float(np.linalg.norm(pd)):
                    # never let the arrow cross the target region
                    L = None
            else:
                L = None
            if L is not None:
                a = arrows[L]
                a.set_pos(tuple(pd + v / speed * L * 0.5))
                a.set_quat(_dir_quat(v))
            # The inset follows the chaser and, once the station is close
            # enough that the pair has to be read together, widens to frame the
            # target too (solar panels included) instead of cropping it away;
            # far away it is a pure chaser close-up. Both the look-at point and
            # the stand-off distance are continuous in the range, so the inset
            # never cuts to a new framing mid-approach.
            span = float(np.linalg.norm(pd))
            e = ((3.0 if big else 1.5) * k) / 2.0 + (7.0 if big else 2.6) * k
            off = min(2.1 * (0.5 * span + e), 10.0 * c_size)
            off = max(4.5 * c_size, off)
            w = 0.5 + 0.5 * min(span / (4.0 * e), 1.0)  # chaser -> pair midpoint
            look = pd * w
            direction = np.array((1.0, -1.0, 0.5))
            direction /= np.linalg.norm(direction)
            ents["close_cam"].set_pose(
                pos=tuple(look + off * direction), lookat=tuple(look), up=WORLD_UP
            )
        if task == "rendezvous":
            r = float(np.linalg.norm(p))
            rgb = cam.render(rgb=True, force_render=True)[0]
            state = _state_task(
                "Rendezvous · CWH LVLH",
                ("Range to target", f"{r:.0f} m"),
                [("Relative speed", f"{speed:.2f} m/s"),
                 ("Goal", "10 m ball"),
                 ("Keep-out", "5 m sphere")],
                [("green", "goal 10 m"), ("red", "keep-out 5 m"),
                 ("grey", "range rings 250 / 500 m")],
                "target at LVLH origin · bodies scaled 6x · arrow 100 m per m/s",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                rgb,
                ents["close_cam"].render(rgb=True, force_render=True)[0],
                state,
                "CLOSE-UP: chaser",
            )
            rgb_frames.append(rgb)
            continue
        if task == "docking":
            r = float(np.linalg.norm(p))
            rgb = cam.render(rgb=True, force_render=True)[0]
            state = _state_task(
                "Docking · CWH LVLH",
                ("Range to target", f"{float(r):.2f} m"),
                [("Relative speed", f"{speed:.3f} m/s"),
                 ("Success", f"< {extras['pos_goal']} m & < {extras['vel_goal']} m/s"),
                 ("Fail", f"> {extras['range_fail']} m")],
                [("grey", "target station"), ("green", "docking port"),
                 ("orange", "chaser")],
                "bodies scaled 4x · velocity arrow 10 m per m/s",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                rgb,
                ents["close_cam"].render(rgb=True, force_render=True)[0],
                state,
                "CLOSE-UP: chaser + station",
            )
            rgb_frames.append(rgb)
        elif task in ("fixedwing", "quadplane", "rocket"):
            png = cam.render(rgb=True, force_render=True)[0]
            offset = (
                np.array((70.0, -70.0, 45.0))
                if task == "rocket"
                else np.array((20.0, -20.0, 12.5))
                if task == "quadplane"
                else np.array((19.0, -19.0, 12.0))
            )
            # follow the DISPLAYED body (rocket's scaled airframe is lifted so
            # its base sits on the pad), never the raw link origin
            lift = np.array((0.0, 0.0, ents.get("body_lift", 0.0)))
            ents["close_cam"].set_pose(
                pos=tuple(p + offset + lift), lookat=tuple(p + lift), up=WORLD_UP
            )
            scale = {"fixedwing": 8, "quadplane": 8, "rocket": 8}[task]
            names = {
                "fixedwing": "Fixedwing-Waypoints",
                "quadplane": "Quadplane · VTOL",
                "rocket": "Rocket landing",
            }
            if task == "rocket":
                rows = [("Landing pad", "at origin"), ("Body", f"scaled {scale}x")]
                legend = [("grey", "ground plane"), ("blue", "descent path")]
                note = "fixed global view of the 0-450 m descent · 8x model anchored on the pad"
            else:
                rows = [("Waypoints", "4 red spheres"), ("Body", f"scaled {scale}x")]
                legend = [("red", "waypoint target"), ("blue", "flown path")]
                note = (
                    "live VTOL attitude · no synthetic tilt animation"
                    if task == "quadplane"
                    else "fixed global view of the waypoint circuit"
                )
            state = _state_task(
                names[task],
                ("Altitude", f"{p[2]:.1f} m"),
                rows,
                legend,
                note,
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                png,
                ents["close_cam"].render(rgb=True, force_render=True)[0],
                state,
                f"CLOSE-UP: {task}",
            )
            rgb_frames.append(rgb)
        elif task == "transfer":
            # fixed camera on the central body: following the spacecraft would
            # hide the orbit shape; the full trajectory must stay in frame
            rgb = cam.render(rgb=True, force_render=True)[0]
            state = _state_task(
                "Orbital transfer · GCRF",
                ("Scale bar", "2000 km"),
                [("1 unit", "1000 km"), ("Frame", "inertial GCRF"),
                 ("Markers", "perigee / apogee")],
                [("green", "initial orbit"), ("cyan", "target circle"),
                 ("orange", "trajectory")],
                "spacecraft box oversized (not to scale)",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            ents["close_cam"].set_pose(
                pos=tuple(p + np.array((4.2, -4.2, 2.1))),
                lookat=tuple(p),
                up=WORLD_UP,
            )
            rgb = _frame(
                rgb,
                ents["close_cam"].render(rgb=True, force_render=True)[0],
                state,
                "CLOSE-UP: spacecraft",
            )
            rgb_frames.append(rgb)
        elif task == "satellite":
            # main camera is the fixed global attitude view; the close-up is a
            # separate fixed camera nearer the bus so the axes stay readable
            att = math.degrees(_quat_angle(f["quat"], f["target_quat"]))
            state = _state_task(
                "Satellite · CMG slew",
                ("Attitude error", f"{att:.1f} deg"),
                [(f"Gimbal {ax}", f"{v:+.2f} rad")
                 for ax, v in zip("XYZ", f["gimbal"])],
                [("red", "body X"), ("green", "body Y"), ("blue", "body Z")],
                "yellow = target triad · cyan = CMG axis · ghost = target",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            ents["close_cam"].set_pose(
                pos=(2.0, -2.0, 1.4), lookat=(0.0, 0.0, 0.0), up=WORLD_UP
            )
            rgb_frames.append(
                _frame(
                    cam.render(rgb=True, force_render=True)[0],
                    ents["close_cam"].render(rgb=True, force_render=True)[0],
                    state,
                    "CLOSE-UP: bus + CMG",
                )
            )
        elif task == "missile":
            # Fixed global engagement view; only the inset follows the missile.
            ents["me"].set_pos(tuple(p))
            png = cam.render(rgb=True, force_render=True)[0]
            ents["close_cam"].set_pose(
                pos=tuple(p + np.array((700.0, -700.0, 420.0))),
                lookat=tuple(p),
                up=WORLD_UP,
            )
            state = _state_task(
                "Missile intercept",
                ("Altitude", f"{p[2] / 1000.0:.2f} km"),
                [("Target", "red 60 m marker"), ("Body", "scaled 200x")],
                [("yellow", "missile + locator"), ("red", "target marker")],
                "fixed global engagement view (0-4 km)",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                png,
                ents["close_cam"].render(rgb=True, force_render=True)[0],
                state,
                "CLOSE-UP: missile",
            )
            rgb_frames.append(rgb)
        elif task != "quadx":  # quadx is rendered in its own branch above
            # catch-all (not reachable for the supported tasks): raw panorama
            _follow(cam, p, (-9, -9, 5))
            rgb_frames.append(cam.render(rgb=True, force_render=True)[0])
    scene.destroy()
    import imageio  # noqa: E402

    imageio.mimsave(
        out_path,
        rgb_frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,  # keep the native 1440x810 frames unstretched
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vehicle", choices=list(ENVS))
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4200,
        help="safety cap on agent steps of the recorded episode",
    )
    parser.add_argument("--fps", type=int, default=30, help="playback fps of the mp4")
    parser.add_argument(
        "--out", type=Path, default=None, help="mp4 path (default results/<task>/vis.mp4)"
    )
    parser.add_argument(
        "--ckpt",
        action="store_true",
        help="replay the trained checkpoint (results/<task>/checkpoint.pt) "
        "instead of the classical baseline",
    )
    args = parser.parse_args()

    out = args.out or Path(__file__).resolve().parent.parent / "results" / args.vehicle / (
        "vis_ckpt.mp4" if args.ckpt else "vis.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    env, cfg = build_env(args.vehicle)
    policy = (
        _CkptPolicy(args.vehicle, env, cfg) if args.ckpt else _BaselinePolicy(args.vehicle, env)
    )
    frames, extras = record(args.vehicle, env, policy, args.max_steps)
    env.close()

    render(args.vehicle, frames, extras, out, args.fps)
    print(f"{args.vehicle}: recorded {len(frames)} agent steps -> {out}")


if __name__ == "__main__":
    main()
