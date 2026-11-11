"""Genesis offscreen visualization of one baseline episode per task.

Usage: python visualize.py quadx|quadx_track|fixedwing|quadplane|rocket|missile|satellite|transfer|docking|rendezvous

Runs the task's classical baseline on a single CPU replica (same env
construction as eval.py/baseline_eval.py, incl. results/<task>/config.json
restoration) while recording the controlled object's pose each agent step,
then replays the recording in a visual-only Genesis scene and renders an
mp4 to results/<task>/vis.mp4.

Scene conventions: each main camera anchors a global task view so targets,
waypoints, the landing site and orbit context never leave the frame. quadx,
fixedwing and quadplane add a slow zero-phase pan (quadx on the drone + active
waypoint midpoint, the winged tasks on the waypoint circuit with the look-at
eased toward the aircraft) so the vehicle cannot drift behind the overlay. The
composed 1440x810 frame is a full-bleed 16:9 panorama: a glass state panel
floats bottom-left and a 460x259 close-up inset bottom-right, both with
generous margins so the subject in the middle of the frame stays clear. The
panel is a styled panel (task title, hero metric, key/value rows, colour
legend, note, episode progress and a terminal RESULT pill), not a stack of raw
readout lines. Bodies and markers are enlarged (display-only) where the true
scale would leave them a few pixels wide; each panel states the magnification
so the render is never mistaken for a scale drawing. Genesis' camera `fov` is
vertical, so when the frame aspect changes the vertical fov is recomputed to
keep the original HORIZONTAL coverage
(_hfov_match): the taller frame only adds sky/ground above and below, the
left/right framing the cameras were set up for is preserved.

Per-task scene contents:
  quadx       Crazyflie-2.1 model (PCB stack, carbon arms, 7 mm motors, 45 mm
              props) on a grass ground plane under a light sky, opaque green
              sphere at the active waypoint (PyFlyt marker scaling), body
              enlarged 9x so the ~9 cm drone is readable, and a
              PyFlyt-angle close-up inset
  quadx_track same model enlarged 14x, red ground target, true line of sight
              and the env's cylindrical LOS occluders, close-up inset
  fixedwing   Cessna-172-like high-wing trainer (1.8 m span) enlarged 10x, red
              spheres at the 4 waypoints, body-mounted left-wingtip inset
  quadplane   VTOL fixed wing (pod, high wing, two lift booms with 4 rotors,
              pusher prop) enlarged 10x, red waypoint spheres, left-wingtip inset
  rocket      Falcon-9-style booster (white tank, black interstage, grid fins,
              four legs, nine engine bells) enlarged 16x + landing pad on a
              ground plane, fixed ground-telephoto tracking inset; the
              display scale is anchored at the airframe's LOWEST point
              (_scale_ground_lift), so the 16x body stands on the pad instead
              of sinking ~40 m into it when the true base is only centimetres
              above the pad; historical trail never covers the body
  satellite   three-axis comsat bus (gold MLI body, radiators, dish, antenna
              boom, +/-y solar panels with cell frames), body-frame axis triad
              (X red / Y green / Z blue) rotating with the bus, target-
              attitude triad (yellow) driven by the recorded target
              quaternion, cyan CMG rotor-axis cylinders driven by the recorded
              gimbal angles (the only CMG state the env exposes; rotor
              angles/speeds are NOT provided so no rotor spin is faked), red
              ghost box at the target attitude, static star decoration and a
              FIXED oblique camera (never follows the tumbling body), all
              driven by the recorded wxyz quaternions
  transfer    Earth sphere + small satellite (gold bus, two solar wings;
              1 unit = 1000 km), 3D view fixed on Earth covering the full
              orbit: green dashed initial-orbit ellipse, cyan dotted target
              (circularized at the initial apogee radius), red/blue perigee/
              apogee markers of the initial orbit, orange trail of the real
              state history, violet osculating-orbit prediction and moving
              apsides recomputed from each recorded r/v state, live perigee /
              apogee / inclination HUD, and a 2000 km scale bar. The satellite
              is enlarged for visibility; the true scale is the scale bar.
  docking     space-station target (pressurised core, docking node, solar
              wings, radiators) with a docking port on the +x face (green ring
              edge, centre marker, translucent approach-axis cylinder); the
              Crew-Dragon-like chaser keeps its docking face turned toward the
              target with its own port ring + centre marker, a dashed
              centre-to-centre line of sight, a true-state velocity arrow
              (100 m per m/s) and an approach trail. Overlay burns the live
              range / speed (real state, metres) plus the actual success/fail
              thresholds; a labeled CLOSE-UP inset follows the chaser and
              continuously widens from a chaser close-up to a chaser+station
              pair frame (solar panels included) as the range closes; the
              drawn chaser is held at a minimum separation so the enlarged
              bodies never intersect at contact (overlay range/speed stay
              true). The terminal banner is the env's own termination outcome
              (SUCCESS / FAILURE / TIMEOUT)
  rendezvous  docking scene plus LVLH display furniture: radial/along-track/
              cross-track axis bars, range rings (250/500/1000/2000 m, all in
              metres), green goal ball (r = 10 m, the real pos_goal), the
              red keep-out sphere (r = 5 m) and a quantized velocity arrow
              (arrow pool in metres, 100 m per m/s of true relative speed),
              with the live range / speed values burned into each frame.
              The main camera is fixed in the LVLH frame; the close-up inset
              moves independently and widens to the chaser+station pair frame
              as the range closes.
  missile     AIM-120-class missile (ogive nose, strakes, X tail fins)
              enlarged 200x + a yellow locator dot riding on it on the km-scale
              4 km-altitude engagement, red marker for the kinematic target
              aircraft (posed per frame from the recorded target state),
              engagement trail, a fixed global engagement camera panned along
              the engagement line so the missile/target pair never hides
              behind the inset, and a missile close-up inset
Trails are preallocated small spheres; unused markers are parked far away.
The render scene shares no physics with the sim: it is built with zero
gravity and never stepped, entities are driven with set_pos/set_quat only.
Airframes are project-authored, material-grouped trimesh models (see the
"high-fidelity display models" section) sized to the simulator's own
bounding volumes; the recorded pose still comes from the environment. Detail
passes add the parts that read at inset range: PCB chips/wires and blade-tip
paint on the micro-drone; door seams, nav/strobe/beacon lights, antennas and
exhaust stacks on the Cessna; winglets, cowls, camera turret and hinge lines
on the quadplane; raceway clamps, RCS pods, grid-fin actuators, leg pistons
and soot streaks on the booster; seeker dome, band and access panels on the
missile; MLI seams, star-tracker hoods, thruster quads, dish struts, radiator
louvers and array hinges on the spacecraft; and IDSS petals, windows,
SuperDraco pods and trunk fins on the chaser. Moving hardware is split into
independent local-frame rigs: quadrotor and fixed-wing propellers follow the
recorded throttle with display-rate reduction to avoid video strobing;
fixed-wing surfaces, rocket grid fins, missile fins and the rocket TVC cluster
follow the plant's actual first-order actuation states; engine glow uses real
ignition/burning flags; and spacecraft exhaust dots oppose the recorded thrust
vector. No landing-leg or docking-lock sequence is invented when the
environment exposes no corresponding state.
"""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(_ROOT))  # exposes environments.* and algorithms.*

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
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
# Layout: a single 16:9 cinematic frame. The panorama is full-bleed; the HUD
# glass panel sits bottom-left and the close-up inset bottom-right, both with
# generous margins, so the subject in the middle of the frame stays clear. The
# old split-panel layout (tall panorama + opaque card column) read as a
# dashboard screenshot; a full-frame replay with floating glass reads as a
# mission film while keeping every number legible.
MAIN_RES = (1440, 810)
CLOSE_RES = (460, 259)  # 16:9 inset; native resolution of the second camera
CLOSE_HERO_RES = (960, 540)  # standalone close-up replay resolution
FRAME_RES = MAIN_RES
CARD_BG = (8, 13, 19)
CARD_BORDER = (54, 82, 102)
CARD_TRACK = (34, 48, 60)
TEXT_COLOR = (238, 244, 250)
DIM_COLOR = (158, 176, 194)
FAINT_COLOR = (104, 126, 147)
ACCENT_COLOR = (127, 227, 255)  # mission-control ice blue
MINT_COLOR = (154, 242, 205)
AMBER_COLOR = (239, 178, 106)
BAD_COLOR = (255, 124, 140)
LEGEND_COLORS = {
    "green": (95, 224, 160),
    "red": (255, 124, 140),
    "yellow": (239, 198, 116),
    "cyan": (127, 227, 255),
    "orange": (239, 178, 106),
    "violet": (244, 114, 208),
    "blue": (122, 162, 247),
    "grey": (158, 176, 194),
    "white": (238, 244, 250),
}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
# per-task film identity: code / category / accent / outdoor scene
TASK_META = {
    "quadx": ("QX-01", "ATMOSPHERIC / MULTIROTOR", ACCENT_COLOR, True),
    "quadx_track": ("QX-02", "ATMOSPHERIC / TRACKING", ACCENT_COLOR, True),
    "fixedwing": ("FW-03", "ATMOSPHERIC / FIXED WING", ACCENT_COLOR, True),
    "quadplane": ("QP-04", "ATMOSPHERIC / HYBRID VTOL", ACCENT_COLOR, True),
    "rocket": ("RK-05", "ATMOSPHERIC / PROPULSIVE LANDING", ACCENT_COLOR, True),
    "missile": ("MI-06", "ATMOSPHERIC / INTERCEPT", ACCENT_COLOR, True),
    "satellite": ("SAT-07", "SPACECRAFT / CMG ATTITUDE", MINT_COLOR, False),
    "transfer": ("LTT-08", "SPACECRAFT / LOW THRUST", MINT_COLOR, False),
    "docking": ("DCK-09", "SPACECRAFT / FINAL APPROACH", MINT_COLOR, False),
    "rendezvous": ("RDV-10", "SPACECRAFT / MEDIUM APPROACH", MINT_COLOR, False),
}
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
SKY_COLOR = (0.135, 0.200, 0.265)
GROUND_COLOR = (0.062, 0.078, 0.092, 1.0)
GRASS_GROUND = (0.040, 0.062, 0.030, 1.0)
DRY_GROUND = (0.070, 0.065, 0.038, 1.0)
SEA_GROUND = (0.014, 0.034, 0.050, 1.0)
GRASS_GRID = (0.115, 0.175, 0.110, 1.0)
DRY_GRID = (0.165, 0.160, 0.105, 1.0)
GROUND_SIZE = 20000.0  # metres; large enough that the ground reaches the horizon
WORLD_UP = (0.0, 0.0, 1.0)  # keep every inset free of camera roll
# Per-task photographic backdrops (assets/backgrounds). Each plate is sky or
# space only: the depth-derived far mask composites it strictly behind the
# geometry, so the scene's own ground, grid and vehicles stay untouched.
# Value = (file name, exposure gain, crop centre for atmospheric plates).
# The crop centre picks which slice of the photo fills the visible sky band:
# bottom-only slices are flat haze on the top-down cameras, so aim at the
# cloud/smoke layer instead. Missing files fall back to the procedural sky.
BACKDROP_DIR = Path(__file__).resolve().parent.parent / "assets" / "backgrounds"
BACKDROPS = {
    "quadx": ("quadx.jpg", 0.74, 0.50),
    "quadx_track": ("quadx_track.jpg", 0.80, 0.72),
    "fixedwing": ("fixedwing.jpg", 0.78, 0.55),
    "quadplane": ("quadplane.jpg", 0.82, 0.45),
    "rocket": ("rocket.jpg", 0.78, 0.55),
    "missile": ("missile.jpg", 0.70, 0.55),
    "satellite": ("satellite.jpg", 0.62, 0.50),
    "transfer": ("transfer.jpg", 0.52, 0.50),
    "docking": ("docking.jpg", 0.66, 0.50),
    "rendezvous": ("rendezvous.jpg", 0.58, 0.50),
}
# Ground cover: (theme, patch wavelength [m], exclusion radius [m]). The post
# pass reconstructs world XY from the depth buffer and repaints upward-facing
# pixels near z=0 as grass (all atmospheric tasks) or dry grass / sea.
GROUND_DETAIL = {
    "quadx": ("grass", 3.0, 0.0),
    "quadx_track": ("grass", 5.0, 0.0),
    "fixedwing": ("grass", 80.0, 0.0),
    "quadplane": ("grass", 80.0, 0.0),
    "rocket": ("grass", 50.0, 2.4),
    "missile": ("grass", 900.0, 0.0),
}



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
            mu=float(env.mu),
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
            frames[-1]["throttle"] = env.plant.throttle.cpu().numpy()[0].copy()
            frames[-1]["surfaces"] = env.plant.surfaces.actuation.cpu().numpy()[0].copy()
            frames[-1]["aux"] = env.plant.auxiliary().cpu().numpy()[0].copy()
        elif task in ("quadx", "fixedwing", "quadplane", "rocket"):
            frames.append(
                {
                    "pos": env.plant.drone.get_pos().cpu().numpy()[0],
                    "quat": env.plant.drone.get_quat().cpu().numpy()[0],
                }
            )
            frames[-1]["throttle"] = env.plant.throttle.cpu().numpy()[0].copy()
            if task != "quadx":
                frames[-1]["surfaces"] = env.plant.surfaces.actuation.cpu().numpy()[0].copy()
            if task == "rocket":
                frames[-1]["aux"] = env.plant.auxiliary().cpu().numpy()[0].copy()
            if task in ("quadx", "fixedwing", "quadplane"):
                # active waypoint for the GCS overlay; the index increments on
                # reach, so clamp keeps the final pre-reset snap in range
                ti = min(int(env.target_index[0]), env.targets.shape[1] - 1)
                frames[-1]["goal"] = env.targets[0, ti].cpu().numpy().copy()
                frames[-1]["wpt"] = ti
        elif task == "quadx_track":
            frames.append(
                {
                    "pos": env.plant.drone.get_pos().cpu().numpy()[0],
                    "quat": env.plant.drone.get_quat().cpu().numpy()[0],
                    "tgt": env.target_pos.cpu().numpy()[0].copy(),
                    "visible": bool(env.vis.cpu()[0]),
                }
            )
            # prop motion rigs read the recorded motor commands, so this branch
            # must publish them too or the blades stay frozen
            frames[-1]["throttle"] = env.plant.throttle.cpu().numpy()[0].copy()
        elif task == "satellite":
            frames.append(
                {
                    "pos": np.zeros(3),
                    "quat": env.q.cpu().numpy()[0].copy(),
                    "target_quat": env.target_quat.cpu().numpy()[0].copy(),
                    "gimbal": env.gimbal.cpu().numpy()[0].copy(),
                    # body rates, needed by the STK-style attitude sphere
                    "omega": env.omega.cpu().numpy()[0].copy(),
                }
            )
        elif task == "transfer":
            frames.append(
                {
                    "pos": env.r.cpu().numpy()[0].copy() * 1e-6,
                    "vel": env.v.cpu().numpy()[0].copy() * 1e-6,
                    "action": env.action.cpu().numpy()[0].copy(),
                    "quat": None,
                }
            )  # 1 unit = 1000 km; velocity uses the same length unit per second
        else:  # docking / rendezvous: CWH relative frame, target at the origin
            f = {
                "pos": env.p.cpu().numpy()[0].copy(),
                "quat": None,
                "vel": env.v.cpu().numpy()[0].copy(),
                "action": env.action.cpu().numpy()[0].copy()
                / (float(env.f_max[0]) if torch.is_tensor(env.f_max) else float(env.f_max)),
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


def _scale_ground_lift(lowest_z, scale):
    """Vertical offset that keeps a display-enlarged body out of the ground.

    The mesh model is scaled about its own origin, so a body whose real bottom
    clears the pad by centimetres would have its scaled bottom
    (scale-1)*|lowest_z| below the pad -- the enlarged airframe sinks into it.
    Scale about the LOWEST point instead: the offset below puts the displayed
    bottom back on the true (unscaled) bottom, so the model stands on the pad.
    """
    if scale <= 1.0 or lowest_z >= 0.0:
        return 0.0
    return -lowest_z * (scale - 1.0)


def _ground(scene, size=GROUND_SIZE, color=GROUND_COLOR):
    """Flat ground wide enough to reach the horizon of every camera."""
    return scene.add_entity(
        gs.morphs.Plane(plane_size=(size, size)),
        surface=gs.surfaces.Plastic(color=color),
    )


def _marker(scene, radius, color, pos=(0, 0, 0), emissive=None, opacity=None,
            roughness=None):
    kwargs = {"color": color}
    if emissive is not None:
        kwargs["emissive"] = tuple(emissive)
    if opacity is not None:
        kwargs["opacity"] = float(opacity)
    if roughness is not None:
        kwargs["roughness"] = float(roughness)
    return scene.add_entity(
        gs.morphs.Sphere(pos=tuple(pos), radius=radius, fixed=False, collision=False),
        surface=gs.surfaces.Plastic(**kwargs),
    )


def _flame_chain(scene, radii, steps=None, parts=1, seed=3):
    """Continuous emissive plume with a ragged silhouette.

    One chain of heavily overlapping spheres gives the smooth surface a real
    exhaust needs, but a perfectly regular taper reads as a painted cone. So
    each sphere gets a small random radius jitter and a lateral offset, the
    colour ramp runs white-hot bell -> yellow -> orange -> ember red, and the
    first rings carry oversized white shock diamonds. Translucent shells were
    tried and rejected: the rasteriser composites transparency against the
    background, washing the plume out to beige over the sky.
    """
    n = len(radii)
    steps = steps or [0.30 * r for r in radii]
    rng = np.random.default_rng(seed)
    stops = (
        (0.00, (1.00, 0.88, 0.60)),
        (0.06, (1.00, 0.66, 0.22)),
        (0.20, (1.00, 0.42, 0.06)),
        (0.52, (0.96, 0.20, 0.020)),
        (1.00, (0.40, 0.05, 0.004)),
    )
    marks, offs = [], []
    for k, r in enumerate(radii):
        t = k / max(n - 1, 1)
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-6)
                col = np.array(c0) + (np.array(c1) - np.array(c0)) * np.clip(f, 0.0, 1.0)
                break
        rr = r * (1.0 + 0.13 * (float(rng.random()) - 0.5))
        # slow radial ripple: the flame edge boils instead of staying a cone
        rr *= 1.0 + 0.06 * math.sin(k * 1.7)
        marks.append(
            _marker(scene, rr, (0.0, 0.0, 0.0, 1.0), HIDDEN,
                    emissive=tuple(float(v) for v in col), opacity=1.0)
        )
        offs.append(
            (
                (float(rng.random()) - 0.5) * 0.22 * r,
                (float(rng.random()) - 0.5) * 0.22 * r,
            )
        )
    return {
        "marks": marks, "offs": offs, "dz": [0.0] * len(marks),
        "phase": [0.0] * len(marks), "steps": steps, "rings": n, "parts": 1,
    }


def _box(scene, size, color, pos=(0, 0, 0)):
    return scene.add_entity(
        gs.morphs.Box(pos=tuple(pos), size=tuple(size), fixed=False, collision=False),
        surface=gs.surfaces.Plastic(color=color),
    )


def _trail(scene, radius, color):
    return [_marker(scene, radius, color, (0, 0, -1e5)) for _ in range(TRAIL_MARKERS)]


def _update_trail(markers, path, clearance=0.0):
    """Replay historical positions without covering the current entity.

    `clearance` (scene metres) drops the newest history points inside the
    current body's displayed envelope: on the 16x rocket the last beads would
    otherwise sit on the airframe, and the space chasers are 5 / 14 m long.
    """
    history = path[:-1]
    if clearance > 0.0 and history and len(path) > 1:
        cur = np.asarray(path[-1], float)
        keep = np.linalg.norm(np.asarray(history, float) - cur, axis=1) > clearance
        history = [p for p, k in zip(history, keep) if k]
    n = min(len(history), TRAIL_MARKERS)
    if n == 0:
        for m in markers:
            m.set_pos(tuple(HIDDEN))
        return
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


def _cyl(scene, radius, length, color, pos=HIDDEN, quat=None, emissive=None,
         opacity=None):
    """Thin cylinder marker whose long axis can be oriented via set_quat
    (Genesis cylinders default to the local z axis)."""
    kw = {} if quat is None else {"quat": tuple(quat)}
    surf = {"color": color}
    if emissive is not None:
        surf["emissive"] = tuple(emissive)
    if opacity is not None:
        surf["opacity"] = float(opacity)
    return scene.add_entity(
        gs.morphs.Cylinder(
            pos=tuple(pos), radius=radius, height=length, fixed=False, collision=False, **kw
        ),
        surface=gs.surfaces.Plastic(**surf),
    )


# ---------------- high-fidelity display models ----------------
# The simulator URDFs are deliberately simple (PyFlyt's fixed wing and
# quadplane are boxes, the rocket is a bare cylinder, cf2x is one collada
# shell). The replay scene has zero gravity and is never stepped, so the
# DISPLAYED airframe can be a project-authored model that reads like the real
# vehicle while the pose still comes from the recorded simulator state.
# Models are built from trimesh primitives in the vehicle frame, merged per
# material (one mesh entity per colour) and driven by a single pose through
# _Rig, so they behave exactly like the one-body entity they replace.
class _Rig:
    """One logical body split into per-material entities; one pose drives all."""

    def __init__(self, ents):
        self.ents = ents
        self.lowest = 0.0  # lowest local z before scaling (rocket ground lift)

    def set_pos(self, pos):
        for ent in self.ents:
            ent.set_pos(pos)

    def set_quat(self, quat):
        for ent in self.ents:
            ent.set_quat(quat)


_MODEL_DIR = Path(tempfile.gettempdir()) / "pyflyt_vis_models"
# colour, roughness, metallic
_MODEL_MATS = {
    "white": ((0.86, 0.88, 0.91, 1.0), 0.55, 0.05),
    "shell": ((0.70, 0.72, 0.76, 1.0), 0.50, 0.10),
    "carbon": ((0.055, 0.060, 0.070, 1.0), 0.55, 0.15),
    "pcb": ((0.050, 0.075, 0.120, 1.0), 0.50, 0.10),
    "motor": ((0.180, 0.190, 0.220, 1.0), 0.40, 0.60),
    "metal": ((0.70, 0.72, 0.76, 1.0), 0.30, 0.90),
    "propw": ((0.88, 0.89, 0.90, 1.0), 0.45, 0.05),
    "propd": ((0.070, 0.070, 0.080, 1.0), 0.45, 0.10),
    "battery": ((0.090, 0.130, 0.300, 1.0), 0.45, 0.10),
    "label": ((0.82, 0.85, 0.88, 1.0), 0.60, 0.00),
    "led": ((0.95, 0.20, 0.16, 1.0), 0.35, 0.00),
    "ledg": ((0.20, 0.95, 0.35, 1.0), 0.35, 0.00),
    "ledb": ((0.25, 0.75, 1.00, 1.0), 0.35, 0.00),
    "blue": ((0.075, 0.170, 0.470, 1.0), 0.45, 0.10),
    "red": ((0.62, 0.075, 0.075, 1.0), 0.45, 0.10),
    "dark": ((0.035, 0.040, 0.050, 1.0), 0.55, 0.10),
    "grey": ((0.30, 0.32, 0.35, 1.0), 0.50, 0.30),
    "black": ((0.012, 0.014, 0.018, 1.0), 0.60, 0.05),
    "gold": ((0.72, 0.54, 0.20, 1.0), 0.38, 0.65),
    "solar": ((0.045, 0.085, 0.240, 1.0), 0.30, 0.20),
    "orange": ((0.86, 0.42, 0.065, 1.0), 0.45, 0.10),
    "band": ((0.72, 0.60, 0.15, 1.0), 0.45, 0.10),
    "glass": ((0.045, 0.060, 0.080, 1.0), 0.15, 0.35),
    "chip": ((0.020, 0.020, 0.025, 1.0), 0.50, 0.10),
    "wire": ((0.16, 0.05, 0.05, 1.0), 0.55, 0.05),
    # propeller motion blur: a thin translucent disc under the blades
    "blur": ((0.88, 0.90, 0.93, 0.13), 0.35, 0.0),
}


def _rot(axis, deg):
    return trimesh.transformations.rotation_matrix(math.radians(deg), axis)


def _m_xf(mesh, pos=(0.0, 0.0, 0.0), rot=None):
    if rot is not None:
        mesh.apply_transform(rot)
    mesh.apply_translation(pos)
    return mesh


def _m_box(size, pos=(0.0, 0.0, 0.0), rot=None):
    return _m_xf(trimesh.creation.box(extents=size), pos, rot)


def _m_cyl(radius, height, pos=(0.0, 0.0, 0.0), rot=None, n=24):
    return _m_xf(trimesh.creation.cylinder(radius=radius, height=height, sections=n), pos, rot)


def _m_cone(radius, height, pos=(0.0, 0.0, 0.0), rot=None, n=24):
    return _m_xf(trimesh.creation.cone(radius=radius, height=height, sections=n), pos, rot)


def _m_sph(radius, pos=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), sub=2):
    mesh = trimesh.creation.icosphere(subdivisions=sub, radius=radius)
    mesh.apply_scale(scale)
    return _m_xf(mesh, pos)


def _m_rod(p0, p1, radius, n=10):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=n)
    mesh.apply_transform(trimesh.geometry.align_vectors((0.0, 0.0, 1.0), d / length))
    return _m_xf(mesh, 0.5 * (p0 + p1))


def _m_plate(poly, z0, z1):
    """Thin prism from a flat planform (wings, stabilisers, fins)."""
    pts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    return trimesh.convex.convex_hull(np.asarray(pts, float))


def _m_vplate(poly_xz, y0, y1):
    """Thin prism from an x-z profile (vertical tails, missile fins)."""
    pts = [(x, y0, z) for x, z in poly_xz] + [(x, y1, z) for x, z in poly_xz]
    return trimesh.convex.convex_hull(np.asarray(pts, float))


def _m_loft(sections, n=24):
    """Closed elliptical fuselage: sections = [(x, ry, rz, z_centre)]."""
    verts, faces = [], []
    for x, ry, rz, zc in sections:
        for k in range(n):
            a = 2.0 * math.pi * k / n
            verts.append((x, ry * math.cos(a), zc + rz * math.sin(a)))
    m = len(sections)
    for i in range(m - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            c, d = (i + 1) * n + k, (i + 1) * n + (k + 1) % n
            faces.append((a, c, d))
            faces.append((a, d, b))
    c0 = len(verts)
    verts.append((sections[0][0], 0.0, sections[0][3]))
    c1 = len(verts)
    verts.append((sections[-1][0], 0.0, sections[-1][3]))
    base = (m - 1) * n
    for k in range(n):
        faces.append((c0, (k + 1) % n, k))
        faces.append((c1, base + k, base + (k + 1) % n))
    mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def _m_prop(radius, hub=0.006, blades=2, width=0.30, thick=0.0018, pitch=20.0, phase=0.0):
    """Thin twisted propeller in the local x-y plane (axis z)."""
    parts = [_m_cyl(hub, thick * 2.6, n=12)]
    tanp = math.tan(math.radians(pitch))
    for b in range(blades):
        pts = []
        for r, w in ((hub * 0.9, radius * width * 0.45), (radius, radius * width)):
            for sy in (-1.0, 1.0):
                zc = -sy * 0.5 * w * tanp
                pts += [(r, sy * 0.5 * w, zc - 0.5 * thick), (r, sy * 0.5 * w, zc + 0.5 * thick)]
        blade = trimesh.convex.convex_hull(np.asarray(pts, float))
        blade.apply_transform(_rot((0.0, 0.0, 1.0), 360.0 / blades * b + phase))
        parts.append(blade)
    return trimesh.util.concatenate(parts)


def _m_prop_tips(radius, phase=0.0, width=0.022, thick=0.0024):
    """Contrasting blade tips: two small blocks at the propeller disc edge."""
    parts = []
    for a in (phase, phase + 180.0):
        block = _m_box((0.014, width, thick), (radius, 0.0, 0.0))
        block.apply_transform(_rot((0.0, 0.0, 1.0), a))
        parts.append(block)
    return trimesh.util.concatenate(parts)


def _model(scene, name, groups, scale=1.0, pos=(0.0, 0.0, 0.0)):
    """Load a multi-material model as one mesh entity per material."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ents, lowest, by_key = [], 0.0, {}
    for key, meshes in groups.items():
        mesh = trimesh.util.concatenate(meshes)
        mesh.remove_unreferenced_vertices()
        lowest = min(lowest, float(np.min(mesh.bounds[:, 2])))
        path = _MODEL_DIR / f"{name}_{key}.obj"
        mesh.export(path)
        color, rough, metallic = _MODEL_MATS[key]
        ent = scene.add_entity(
            gs.morphs.Mesh(
                file=str(path), scale=scale, pos=tuple(pos),
                collision=False, fixed=False,
            ),
            surface=gs.surfaces.Plastic(
                color=color, roughness=rough, metallic=metallic, smooth=False
            ),
        )
        by_key[key] = ent
        ents.append(ent)
    rig = _Rig(ents)
    rig.lowest = lowest
    rig.by_key = by_key
    return rig


def _axis_quat(axis, angle):
    """wxyz quaternion for a local hinge/spin axis and angle in radians."""
    axis = np.asarray(axis, float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    h = 0.5 * float(angle)
    return np.array((math.cos(h), *(axis * math.sin(h))))


def _prop_groups(radius, blade, tip, pitch=20.0, tip_width=0.022, tip_thick=0.0024):
    """Independent blade set; display-rate reduction prevents video strobing."""
    return {
        blade: [_m_prop(radius, pitch=pitch)],
        tip: [_m_prop_tips(radius, width=tip_width, thick=tip_thick)],
        # translucent motion-blur disc: hides the per-frame blade stepping
        "blur": [_m_cyl(radius * 0.985, 0.0012, (0.0, 0.0, 0.0), n=40)],
    }


def _add_motion(ents, rig, pos, scale, kind, index=0, base=(1, 0, 0, 0),
                axis=(0, 0, 1), limit=0.0, sign=1.0, phase=0.0):
    """Register an independently hinged rig in the parent body's local frame."""
    ents.setdefault("motion", []).append(
        dict(rig=rig, pos=np.asarray(pos, float), scale=float(scale), kind=kind,
             index=int(index), base=np.asarray(base, float), axis=np.asarray(axis, float),
             limit=float(limit), sign=float(sign), phase=float(phase),
             disc=rig.by_key.get("blur"))
    )


def _hinge_mesh(mesh, hinge, base=(1, 0, 0, 0), rot=None):
    """Bake a body-frame mesh into coordinates relative to its rotated hinge."""
    if rot is not None:
        mesh.apply_transform(rot)
    mesh.apply_translation(-_quat_rot(np.asarray(base, float), np.asarray(hinge, float)))
    return mesh


def _update_motion(ents, f, phases):
    """Pose moving parts from recorded actuator states, never new dynamics."""
    q = f.get("quat")
    if q is None:
        return
    q = np.asarray(q, float)
    origin = np.asarray(f["pos"], float) + np.array((0.0, 0.0, ents.get("body_lift", 0.0)))
    throttle = np.asarray(f.get("throttle", np.zeros(1)), float)
    surfaces = np.asarray(f.get("surfaces", np.zeros(8)), float)
    for i, part in enumerate(ents.get("motion", [])):
        kind = part["kind"]
        if kind == "prop":
            # Sweep a fixed angle per RENDERED frame (not per simulated second):
            # the highlight cut resamples whole episodes, so a physical-time rate
            # would advance ~2 rad between composite frames and alias into noise.
            # The translucent disc in the prop rig covers the stepping, so the
            # blades can run faster than a quarter turn per frame without the
            # two-blade strobe reading as a wobble.
            command = abs(float(throttle[min(part["index"], len(throttle) - 1)]))
            step = 0.0 if command < 0.02 else 0.90 + 1.30 * min(command, 1.0)
            phases[i] += part["sign"] * step
            angle = phases[i] + part["phase"]
            local_q = _quat_mul_np(part["base"], _axis_quat(part["axis"], angle))
        elif kind == "tvc":
            angles = np.asarray(f.get("aux", np.zeros(9)), float)[-2:] * math.radians(5.0)
            local_q = _quat_mul_np(
                _axis_quat((1, 0, 0), angles[0]), _axis_quat((0, 1, 0), angles[1])
            )
        else:
            command = float(surfaces[min(part["index"], len(surfaces) - 1)])
            angle = part["sign"] * command * part["limit"]
            local_q = _quat_mul_np(part["base"], _axis_quat(part["axis"], angle))
        part["rig"].set_pos(tuple(origin + _quat_rot(q, part["pos"] * part["scale"])))
        part["rig"].set_quat(tuple(_quat_mul_np(q, local_q)))
        if kind == "prop" and part.get("disc") is not None and command < 0.05:
            # parked blades should not sit under a motion-blur disc
            part["disc"].set_pos(tuple(HIDDEN))


def _surf_rows(surfaces):
    """Live control-surface deflections (deg) for the aircraft HUD panel."""
    ail_l, ail_r, ele, rud = (float(v) for v in np.asarray(surfaces, float)[:4])
    return [
        ("Aileron L / R", f"{ail_l * 30.0:+.0f} / {ail_r * 30.0:+.0f} deg"),
        ("Elev / Rudder", f"{ele * 20.0:+.0f} / {rud * 20.0:+.0f} deg"),
    ]


def _missile_rows(surfaces):
    """Live tail-fin deflections (deg), plant order left/right/top/bottom."""
    left, right, top, bottom = (float(v) * 35.0 for v in np.asarray(surfaces, float)[:4])
    return [
        ("Fins L / R", f"{left:+.0f} / {right:+.0f} deg"),
        ("Fins T / B", f"{top:+.0f} / {bottom:+.0f} deg"),
    ]


def _update_thruster(markers, origin, action, spacing):
    """Short inertial-frame exhaust trace aligned opposite the recorded thrust."""
    action = np.asarray(action, float)
    magnitude = float(np.linalg.norm(action))
    origin = np.asarray(origin, float)
    if magnitude < 0.02:
        for marker in markers:
            marker.set_pos(tuple(HIDDEN))
        return
    direction = action / magnitude
    count = max(1, min(len(markers), int(math.ceil(magnitude * len(markers)))))
    for i, marker in enumerate(markers):
        point = origin - direction * spacing * (i + 1) if i < count else HIDDEN
        marker.set_pos(tuple(map(float, point)))


def _build_quadrotor():
    """Crazyflie 2.1: PCB stack, four carbon arms, 7 mm motors, 45 mm props."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add("pcb", _m_box((0.050, 0.050, 0.0022), (0.0, 0.0, 0.0016)))
    add("pcb", _m_box((0.038, 0.038, 0.0016), (0.0, 0.0, -0.0072)))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            add("metal", _m_cyl(0.0014, 0.0072, (sx * 0.020, sy * 0.020, -0.0028)))
    for sx, sy in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
        tx, ty = sx * 0.028, sy * 0.028
        ang = math.degrees(math.atan2(ty, tx))
        add(
            "carbon",
            _m_box((0.040, 0.0050, 0.0015), (0.5 * tx, 0.5 * ty, 0.0018),
                   _rot((0.0, 0.0, 1.0), ang)),
        )
        add("motor", _m_cyl(0.0044, 0.0115, (tx, ty, 0.0095)))
        add("metal", _m_cyl(0.0048, 0.0030, (tx, ty, 0.0162)))
        add("metal", _m_cyl(0.0009, 0.0040, (tx, ty, 0.0195)))
        # phase wires running out along each arm; blades are independent rigs
        add("wire", _m_rod((sx * 0.017, sy * 0.017, 0.0024),
                           (tx - sx * 0.0025, ty - sy * 0.0025, 0.0042), 0.00055))
    # board hardware: battery strap, USB, header, chips, corner screws
    add("carbon", _m_box((0.032, 0.0035, 0.0060), (0.0, -0.001, 0.0080)))
    for cx, cy, cw, ch in ((0.006, -0.012, 0.010, 0.008),
                           (-0.012, 0.008, 0.007, 0.007),
                           (0.010, 0.004, 0.006, 0.010)):
        add("chip", _m_box((cw, ch, 0.0016), (cx, cy, 0.0031)))
    add("metal", _m_box((0.008, 0.005, 0.0032), (-0.020, -0.006, 0.0027)))
    add("gold", _m_box((0.020, 0.0025, 0.0012), (0.0, 0.016, 0.0026)))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            add("metal", _m_cyl(0.0012, 0.0008, (sx * 0.0215, sy * 0.0215, 0.0029), n=8))
    add("glass", _m_sph(0.0018, (0.026, 0.0, 0.0032), sub=1))
    add("battery", _m_box((0.031, 0.017, 0.0052), (0.0, -0.004, 0.0078)))
    add("label", _m_box((0.030, 0.004, 0.0054), (0.0, -0.010, 0.0078)))
    add("pcb", _m_box((0.012, 0.008, 0.0040), (0.014, 0.012, 0.0060)))
    add("metal", _m_cyl(0.0006, 0.020, (-0.019, 0.019, -0.006), _rot((0.0, 1.0, 0.0), 40)))
    add("led", _m_sph(0.0012, (0.024, -0.024, -0.004), sub=1))
    add("ledb", _m_sph(0.0012, (-0.024, 0.024, -0.004), sub=1))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            add(
                "carbon",
                _m_box((0.0022, 0.0022, 0.024), (sx * 0.0205, sy * 0.0205, -0.0115),
                       _rot((0.0, 1.0, 0.0), -sx * 14.0)),
            )
    return g


def _add_quadrotor_motion(scene, ents, scale):
    positions = ((1, -1), (1, 1), (-1, -1), (-1, 1))
    spins = (-1, -1, 1, 1)  # TensorPlant quadx motor order
    for i, ((sx, sy), spin) in enumerate(zip(positions, spins)):
        blade = "propw" if sx * sy > 0 else "propd"
        tip = "propd" if blade == "propw" else "propw"
        rig = _model(
            scene, f"cf_prop{i}",
            _prop_groups(0.0225, blade, tip, 16.0, tip_width=0.009, tip_thick=0.0016),
            scale=scale
        )
        _add_motion(
            ents, rig, (sx * 0.028, sy * 0.028, 0.0212), scale,
            "prop", index=i, sign=spin, phase=0.63 * i,
        )


def _build_fixedwing():
    """Cessna-172-like high-wing trainer matching PyFlyt's 1.4 m x 1.8 m frame."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add(
        "white",
        _m_loft([
            (0.735, 0.020, 0.020, 0.000),
            (0.700, 0.052, 0.058, 0.004),
            (0.610, 0.078, 0.090, 0.010),
            (0.470, 0.094, 0.108, 0.016),
            (0.230, 0.100, 0.116, 0.020),
            (-0.090, 0.096, 0.112, 0.022),
            (-0.380, 0.074, 0.090, 0.030),
            (-0.610, 0.043, 0.058, 0.042),
            (-0.745, 0.018, 0.026, 0.052),
        ]),
    )
    add("shell", _m_cyl(0.063, 0.030, (0.712, 0.0, 0.004), _rot((0.0, 1.0, 0.0), 90)))
    add("metal", _m_cone(0.040, 0.070, (0.760, 0.0, 0.004), _rot((0.0, 1.0, 0.0), 90)))
    add("glass", _m_sph(1.0, (0.285, 0.0, 0.090), (0.195, 0.080, 0.055)))
    for sy in (-1.0, 1.0):
        add("glass", _m_box((0.150, 0.006, 0.032), (0.330, sy * 0.090, 0.060)))
        add("blue", _m_box((0.560, 0.003, 0.016), (0.150, sy * 0.0995, 0.014)))
        # cabin door seams + handle
        add("dark", _m_box((0.012, 0.004, 0.080), (0.440, sy * 0.093, 0.032)))
        add("dark", _m_box((0.012, 0.004, 0.070), (-0.020, sy * 0.093, 0.030)))
        add("metal", _m_box((0.018, 0.006, 0.008), (0.400, sy * 0.095, 0.040)))
        add("dark", _m_sph(0.004, (0.380, sy * 0.096, -0.012), sub=1))
        # exhaust stacks under the cowling
        add("dark", _m_cyl(0.011, 0.045, (0.615, sy * 0.048, -0.050),
                           _rot((1.0, 0.0, 0.0), 24.0), n=12))
    # antennas + beacon / tail light
    add("metal", _m_rod((-0.120, 0.0, 0.128), (-0.150, 0.0, 0.208), 0.004))
    add("metal", _m_rod((-0.300, 0.0, -0.078), (-0.325, 0.0, -0.140), 0.004))
    add("led", _m_sph(0.012, (-0.690, 0.0, 0.306), sub=1))
    add("label", _m_sph(0.010, (-0.800, 0.0, 0.070), sub=1))
    for s in (-1.0, 1.0):
        # _m_plate hulls its polygon, so a single notched planform would FILL
        # the aileron bay and bury the moving surface. Two convex panels leave
        # a real bay; the aileron rig (shell colour) sits in it.
        inboard = _m_plate(
            [(0.140, 0.020 * s), (-0.180, 0.020 * s),
             (-0.180, 0.550 * s), (0.068, 0.550 * s)],
            0.150, 0.178,
        )
        outboard = _m_plate(
            [(0.068, 0.550 * s), (-0.100, 0.550 * s),
             (-0.100, 0.900 * s), (0.020, 0.900 * s)],
            0.150, 0.178,
        )
        for panel in (inboard, outboard):
            panel.apply_transform(_rot((1.0, 0.0, 0.0), 3.0 * s))
            add("white", panel)
        # navigation / strobe lights at the tips, landing light on the left LE
        add("led" if s < 0 else "ledg", _m_sph(0.014, (0.000, 0.906 * s, 0.212), sub=1))
        add("label", _m_sph(0.008, (0.010, 0.906 * s, 0.212), sub=1))
        if s < 0:
            add("metal", _m_box((0.030, 0.026, 0.016), (0.138, -0.520, 0.190)))
            add("label", _m_box((0.018, 0.018, 0.010), (0.150, -0.520, 0.190)))
        add("metal", _m_rod((0.055, 0.100 * s, -0.070), (0.000, 0.470 * s, 0.145), 0.008))
        # tricycle gear
        add("metal", _m_rod((0.050, 0.105 * s, -0.045), (0.020, 0.215 * s, -0.170), 0.010))
        add("carbon", _m_cyl(0.036, 0.022, (0.020, 0.215 * s, -0.185),
                             _rot((1.0, 0.0, 0.0), 90), n=18))
        add("carbon", _m_cyl(0.032, 0.018, (0.020, 0.215 * s, -0.185),
                             _rot((1.0, 0.0, 0.0), 90), n=18))
        add("metal", _m_cyl(0.013, 0.026, (0.020, 0.215 * s, -0.185),
                            _rot((1.0, 0.0, 0.0), 90), n=14))
    add("metal", _m_rod((0.550, 0.0, -0.060), (0.550, 0.0, -0.165), 0.010))
    add("carbon", _m_cyl(0.030, 0.018, (0.550, 0.0, -0.180),
                         _rot((1.0, 0.0, 0.0), 90), n=18))
    add("blue", _m_vplate(
        [(-0.600, 0.052), (-0.680, 0.052), (-0.680, 0.300), (-0.635, 0.300)],
        -0.012, 0.012,
    ))
    add("white", _m_plate(
        [(-0.600, -0.300), (-0.700, -0.300), (-0.700, 0.300), (-0.600, 0.300)],
        0.060, 0.080,
    ))
    add("blue", _m_plate(
        [(-0.600, -0.300), (-0.660, -0.300), (-0.660, 0.300), (-0.600, 0.300)],
        0.060, 0.081,
    ))
    return g


def _add_fixedwing_motion(scene, ents, scale):
    prop_base = _axis_quat((0, 1, 0), math.pi / 2.0)
    prop = _model(
        scene, "cessna_prop",
        _prop_groups(0.165, "grey", "white", 24.0, tip_width=0.030, tip_thick=0.004),
        scale=scale,
    )
    _add_motion(ents, prop, (0.733, 0.0, 0.004), scale, "prop", base=prop_base)
    for s in (-1.0, 1.0):
        base = _axis_quat((1, 0, 0), math.radians(3.0 * s))
        rot = _rot((1.0, 0.0, 0.0), 3.0 * s)
        hinge = (-0.100, 0.0, 0.164)
        white = _hinge_mesh(_m_plate(
            [(-0.100, 0.550 * s), (-0.180, 0.550 * s),
             (-0.180, 0.780 * s), (-0.100, 0.780 * s)], 0.150, 0.178,
        ), hinge, base, rot)
        tip = _hinge_mesh(_m_plate(
            [(-0.100, 0.780 * s), (-0.180, 0.780 * s),
             (-0.180, 0.900 * s), (-0.100, 0.900 * s)], 0.150, 0.179,
        ), hinge, base, _rot((1.0, 0.0, 0.0), 3.0 * s))
        # blue trim along the trailing edge: the painted edge makes even a
        # few-degree deflection legible at inset scale
        band = _hinge_mesh(_m_plate(
            [(-0.148, 0.550 * s), (-0.180, 0.550 * s),
             (-0.180, 0.900 * s), (-0.148, 0.900 * s)], 0.1505, 0.1775,
        ), hinge, base, _rot((1.0, 0.0, 0.0), 3.0 * s))
        rig = _model(
            scene, f"cessna_ail{int(s)}",
            {"shell": [white], "blue": [tip, band]}, scale=scale,
        )
        _add_motion(
            ents, rig, hinge, scale, "surface", index=0 if s > 0 else 1,
            base=base, axis=(0, 1, 0), limit=math.radians(30.0), sign=-1.0,
        )
    hinge = (-0.700, 0.0, 0.070)
    elevator = _hinge_mesh(_m_plate(
        [(-0.700, -0.300), (-0.790, -0.300), (-0.790, 0.300), (-0.700, 0.300)],
        0.060, 0.080,
    ), hinge)
    elev_band = _hinge_mesh(_m_plate(
        [(-0.758, -0.300), (-0.790, -0.300), (-0.790, 0.300), (-0.758, 0.300)],
        0.0602, 0.0798,
    ), hinge)
    rig = _model(scene, "cessna_elevator",
                 {"shell": [elevator], "blue": [elev_band]}, scale=scale)
    _add_motion(ents, rig, hinge, scale, "surface", index=2,
                axis=(0, 1, 0), limit=math.radians(20.0), sign=-1.0)
    hinge = (-0.680, 0.0, 0.052)
    rudder = _hinge_mesh(_m_vplate(
        [(-0.680, 0.052), (-0.760, 0.052), (-0.745, 0.300), (-0.680, 0.300)],
        -0.012, 0.012,
    ), hinge)
    rud_band = _hinge_mesh(_m_vplate(
        [(-0.735, 0.052), (-0.760, 0.052), (-0.745, 0.300), (-0.735, 0.300)],
        -0.0122, 0.0122,
    ), hinge)
    rig = _model(scene, "cessna_rudder",
                 {"shell": [rudder], "blue": [rud_band]}, scale=scale)
    _add_motion(ents, rig, hinge, scale, "surface", index=3,
                axis=(0, 0, 1), limit=math.radians(20.0), sign=-1.0)


def _build_quadplane():
    """VTOL fixed wing: pod, high wing, two lift booms with 4 rotors, pusher."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add(
        "white",
        _m_loft([
            (0.640, 0.014, 0.014, 0.000),
            (0.600, 0.045, 0.045, 0.000),
            (0.510, 0.070, 0.076, 0.002),
            (0.300, 0.084, 0.092, 0.006),
            (0.000, 0.086, 0.096, 0.008),
            (-0.350, 0.074, 0.086, 0.012),
            (-0.520, 0.048, 0.060, 0.014),
            (-0.640, 0.020, 0.028, 0.016),
        ]),
    )
    add("glass", _m_sph(1.0, (0.330, 0.0, 0.080), (0.145, 0.058, 0.042)))
    for s in (-1.0, 1.0):
        # Same convex-hull constraint as the fixedwing: two panels, not a notch.
        inboard = _m_plate(
            [(-0.200, 0.020 * s), (-0.500, 0.020 * s),
             (-0.500, 0.550 * s), (-0.243, 0.550 * s)],
            0.092, 0.122,
        )
        outboard = _m_plate(
            [(-0.243, 0.550 * s), (-0.425, 0.550 * s),
             (-0.425, 0.900 * s), (-0.270, 0.900 * s)],
            0.092, 0.122,
        )
        for panel in (inboard, outboard):
            panel.apply_transform(_rot((1.0, 0.0, 0.0), 2.0 * s))
            add("white", panel)
        # blue leading-edge strip + tip cap (same dihedral as the wing)
        strip = _m_plate(
            [(-0.200, 0.020 * s), (-0.216, 0.020 * s),
             (-0.286, 0.900 * s), (-0.270, 0.900 * s)],
            0.118, 0.124,
        )
        strip.apply_transform(_rot((1.0, 0.0, 0.0), 2.0 * s))
        add("blue", strip)
        # lift boom + two motor nacelles (white cowls, dark motors, tip paint)
        add("carbon", _m_rod((0.290, 0.500 * s, 0.112), (-1.040, 0.500 * s, 0.112), 0.018))
        for mx in (0.050, -0.950):
            add("carbon", _m_cyl(0.030, 0.070, (mx, 0.500 * s, 0.140), n=14))
            add("white", _m_cyl(0.036, 0.036, (mx, 0.500 * s, 0.150), n=16))
            add("motor", _m_cyl(0.022, 0.030, (mx, 0.500 * s, 0.180), n=14))
            add("metal", _m_cyl(0.004, 0.030, (mx, 0.500 * s, 0.205), n=10))
        add("carbon", _m_box((0.026, 0.020, 0.060), (0.050, 0.500 * s, 0.085)))
        add("carbon", _m_box((0.026, 0.020, 0.060), (-0.950, 0.500 * s, 0.085)))
        # winglet at the tip (same dihedral); ailerons are independent rigs
        winglet = _m_vplate(
            [(-0.400, 0.122), (-0.428, 0.122), (-0.432, 0.230), (-0.402, 0.230)],
            (0.890 * s), (0.902 * s),
        )
        winglet.apply_transform(_rot((1.0, 0.0, 0.0), 2.0 * s))
        add("blue", winglet)
        # twin fin + stabiliser
        add("blue", _m_vplate(
            [(-1.020, 0.080), (-1.090, 0.080), (-1.090, 0.310), (-1.055, 0.310)],
            (-0.012 + 0.500 * s), (0.012 + 0.500 * s),
        ))
        add("metal", _m_rod((0.150, 0.105 * s, -0.060), (-0.120, 0.140 * s, -0.135), 0.008))
        add("carbon", _m_cyl(0.024, 0.016, (-0.120, 0.140 * s, -0.145),
                             _rot((1.0, 0.0, 0.0), 90), n=14))
    add("white", _m_plate(
        [(-1.020, -0.500), (-1.100, -0.500), (-1.100, 0.500), (-1.020, 0.500)],
        0.245, 0.270,
    ))
    add("shell", _m_cyl(0.040, 0.050, (-0.668, 0.0, 0.016), _rot((0.0, 1.0, 0.0), 90)))
    # EO/IR turret under the nose, pitot, antennas, gear hubs
    add("metal", _m_cyl(0.020, 0.030, (0.500, 0.0, -0.086), n=14))
    add("glass", _m_sph(0.026, (0.500, 0.0, -0.102), sub=2))
    add("metal", _m_rod((0.630, 0.0, 0.006), (0.715, 0.0, 0.006), 0.003))
    add("metal", _m_rod((-0.150, 0.0, 0.104), (-0.175, 0.0, 0.170), 0.004))
    add("metal", _m_rod((-0.380, 0.0, -0.068), (-0.400, 0.0, -0.125), 0.004))
    add("metal", _m_rod((0.470, 0.0, -0.070), (0.470, 0.0, -0.150), 0.008))
    add("carbon", _m_cyl(0.026, 0.016, (0.470, 0.0, -0.160),
                         _rot((1.0, 0.0, 0.0), 90), n=14))
    add("metal", _m_cyl(0.010, 0.018, (0.470, 0.0, -0.160),
                        _rot((1.0, 0.0, 0.0), 90), n=12))
    return g


def _add_quadplane_motion(scene, ents, scale):
    # TensorPlant order: front-right, rear-left, front-left, rear-right.
    motors = ((0.050, -0.500), (-0.950, 0.500), (0.050, 0.500), (-0.950, -0.500))
    for i, ((x, y), spin) in enumerate(zip(motors, (1, 1, -1, -1))):
        rig = _model(
            scene, f"qp_lift_prop{i}",
            _prop_groups(0.150, "grey", "propw", 14.0, tip_width=0.020, tip_thick=0.002),
            scale=scale,
        )
        _add_motion(ents, rig, (x, y, 0.198), scale, "prop", index=i,
                    sign=spin, phase=0.63 * i)
    prop_base = _axis_quat((0, 1, 0), math.pi / 2.0)
    prop = _model(
        scene, "qp_pusher_prop",
        _prop_groups(0.155, "grey", "propw", 18.0, tip_width=0.020, tip_thick=0.002),
        scale=scale,
    )
    _add_motion(ents, prop, (-0.700, 0.0, 0.016), scale, "prop", index=4, base=prop_base)
    for s in (-1.0, 1.0):
        base = _axis_quat((1, 0, 0), math.radians(2.0 * s))
        hinge = (-0.425, 0.0, 0.108)
        white = _hinge_mesh(_m_plate(
            [(-0.425, 0.550 * s), (-0.470, 0.550 * s),
             (-0.470, 0.800 * s), (-0.425, 0.800 * s)], 0.092, 0.122,
        ), hinge, base, _rot((1.0, 0.0, 0.0), 2.0 * s))
        tip = _hinge_mesh(_m_plate(
            [(-0.425, 0.800 * s), (-0.470, 0.800 * s),
             (-0.470, 0.900 * s), (-0.425, 0.900 * s)], 0.092, 0.123,
        ), hinge, base, _rot((1.0, 0.0, 0.0), 2.0 * s))
        band = _hinge_mesh(_m_plate(
            [(-0.452, 0.550 * s), (-0.470, 0.550 * s),
             (-0.470, 0.800 * s), (-0.452, 0.800 * s)], 0.0925, 0.1215,
        ), hinge, base, _rot((1.0, 0.0, 0.0), 2.0 * s))
        rig = _model(
            scene, f"qp_ail{int(s)}",
            {"shell": [white], "blue": [tip, band]}, scale=scale,
        )
        _add_motion(ents, rig, hinge, scale, "surface", index=0 if s > 0 else 1,
                    base=base, axis=(0, 1, 0), limit=math.radians(30.0), sign=-1.0)
    hinge = (-1.100, 0.0, 0.2575)
    elevator = _hinge_mesh(_m_plate(
        [(-1.100, -0.460), (-1.190, -0.460), (-1.190, 0.460), (-1.100, 0.460)],
        0.245, 0.270,
    ), hinge)
    caps = []
    for s in (-1.0, 1.0):
        caps.append(_hinge_mesh(_m_plate(
            [(-1.100, 0.460 * s), (-1.190, 0.460 * s),
             (-1.190, 0.500 * s), (-1.100, 0.500 * s)], 0.245, 0.271,
        ), hinge))
    rig = _model(scene, "qp_elevator", {"white": [elevator], "blue": caps}, scale=scale)
    _add_motion(ents, rig, hinge, scale, "surface", index=2,
                axis=(0, 1, 0), limit=math.radians(20.0), sign=-1.0)
    for s in (-1.0, 1.0):
        hinge = (-1.090, 0.500 * s, 0.080)
        rudder = _hinge_mesh(_m_vplate(
            [(-1.090, 0.080), (-1.180, 0.080), (-1.150, 0.310), (-1.090, 0.310)],
            (-0.012 + 0.500 * s), (0.012 + 0.500 * s),
        ), hinge)
        rig = _model(scene, f"qp_rudder{int(s)}", {"blue": [rudder]}, scale=scale)
        _add_motion(ents, rig, hinge, scale, "surface", index=3,
                    axis=(0, 0, 1), limit=math.radians(20.0), sign=-1.0)


def _build_rocket():
    """Falcon-9-style booster: white tank, black interstage, grid fins, octaweb."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add("white", _m_cyl(0.190, 3.40, (0.0, 0.0, 0.680), n=32))  # z -1.02 .. 2.38
    add("grey", _m_cyl(0.188, 1.03, (0.0, 0.0, -1.535), n=32))  # sooted lower tank
    add("grey", _m_cyl(0.205, 0.40, (0.0, 0.0, -2.250), n=32))  # octaweb skirt
    add("black", _m_cyl(0.191, 0.32, (0.0, 0.0, 2.220), n=32))  # interstage
    add("metal", _m_cyl(0.196, 0.03, (0.0, 0.0, 2.050), n=32))
    add("white", _m_cyl(0.197, 0.020, (0.0, 0.0, 2.392), n=32))  # interstage rim
    add("black", _m_cyl(0.192, 0.030, (0.0, 0.0, 1.320), n=32))  # tank band
    # raceway + conduit clamps
    add("grey", _m_box((0.028, 0.030, 3.30), (-0.196, 0.0, 0.750)))
    for z in (-0.75, -0.35, 0.05, 0.45, 0.85, 1.25, 1.65):
        add("grey", _m_box((0.040, 0.044, 0.026), (-0.194, 0.0, z)))
    # cold-gas RCS pods just below the interstage
    for k in range(4):
        pod = _m_box((0.075, 0.060, 0.090), (0.245, 0.0, 1.760))
        pod.apply_transform(_rot((0.0, 0.0, 1.0), 45.0 + 90.0 * k))
        add("grey", pod)
        nozzle = _m_cone(0.020, 0.045, (0.300, 0.0, 1.760),
                         _rot((0.0, 1.0, 0.0), 90), n=10)
        nozzle.apply_transform(_rot((0.0, 0.0, 1.0), 45.0 + 90.0 * k))
        add("metal", nozzle)
    # grid-fin actuators stay on the tank; the fins themselves hinge independently
    for k in range(4):
        actuator = _m_cyl(0.030, 0.075, (0.205, 0.0, 1.930), n=12)
        actuator.apply_transform(_rot((0.0, 0.0, 1.0), 45.0 + 90.0 * k))
        add("metal", actuator)
    # four deployed landing legs + feet + pistons
    for k in range(4):
        ang = math.radians(45.0 + 90.0 * k)
        c, s = math.cos(ang), math.sin(ang)
        add("carbon", _m_rod((0.200 * c, 0.200 * s, -2.100),
                             (0.560 * c, 0.560 * s, -2.610), 0.032))
        add("metal", _m_rod((0.245 * c, 0.245 * s, -2.030),
                            (0.505 * c, 0.505 * s, -2.520), 0.011))
        add("grey", _m_box((0.085, 0.085, 0.055), (0.215, 0.0, -2.085),
                           _rot((0.0, 0.0, 1.0), 45.0 + 90.0 * k)))
        add("grey", _m_box((0.110, 0.110, 0.030), (0.575 * c, 0.575 * s, -2.630),
                           _rot((0.0, 0.0, 1.0), 45.0 + 90.0 * k)))
    # soot streaks down the lower tank
    for k in range(6):
        a = 15.0 + 60.0 * k
        streak = _m_box((0.008, 0.055, 0.95), (0.188, 0.0, -1.53))
        streak.apply_transform(_rot((0.0, 0.0, 1.0), a))
        add("black", streak)
    add("grey", _m_cyl(0.200, 0.050, (0.0, 0.0, -2.330), n=32))  # heat shield ring
    return g


def _add_rocket_motion(scene, ents, scale):
    for k in range(4):
        angle = 45.0 + 90.0 * k
        base = _axis_quat((0, 0, 1), math.radians(angle))
        hinge = (0.205, 0.0, 2.020)
        fin = [_m_box((0.020, 0.290, 0.230), (0.315, 0.0, 2.020))]
        for dz in (-0.075, 0.0, 0.075):
            fin.append(_m_box((0.024, 0.270, 0.018), (0.315, 0.0, 2.020 + dz)))
        for dy in (-0.085, 0.0, 0.085):
            fin.append(_m_box((0.024, 0.018, 0.210), (0.315, dy, 2.020)))
        fin = [_hinge_mesh(m, hinge, base, _rot((0, 0, 1), angle)) for m in fin]
        rig = _model(scene, f"f9_gridfin{k}", {"grey": fin}, scale=scale)
        _add_motion(ents, rig, hinge, scale, "surface", index=k, base=base,
                    axis=(0, 1, 0), limit=math.radians(45.0), sign=-1.0)
    hinge = (0.0, 0.0, -2.280)
    bells = [_hinge_mesh(
        _m_cone(0.105, 0.260, (0.0, 0.0, -2.380), _rot((1, 0, 0), 180)), hinge
    )]
    for k in range(8):
        a = math.radians(22.5 + 45.0 * k)
        bells.append(_hinge_mesh(_m_cone(
            0.062, 0.200, (0.142 * math.cos(a), 0.142 * math.sin(a), -2.430),
            _rot((1, 0, 0), 180), n=16,
        ), hinge))
    rig = _model(scene, "f9_tvc_cluster", {"metal": bells}, scale=scale)
    _add_motion(ents, rig, hinge, scale, "tvc")


def _build_missile():
    """AIM-120-class air-to-air missile: ogive nose, strakes, X tail fins."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add("shell", _m_cyl(0.100, 2.55, (0.0, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=28))
    add("shell", _m_loft([
        (1.275, 0.100, 0.100, 0.0),
        (1.360, 0.092, 0.092, 0.0),
        (1.450, 0.068, 0.068, 0.0),
        (1.520, 0.032, 0.032, 0.0),
        (1.545, 0.012, 0.012, 0.0),
    ], n=28))
    add("grey", _m_cyl(0.101, 0.020, (1.272, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=28))
    add("glass", _m_cone(0.034, 0.075, (1.560, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=20))
    add("metal", _m_cyl(0.020, 0.012, (1.600, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=12))
    add("metal", _m_cone(0.086, 0.150, (-1.350, 0.0, 0.0), _rot((0.0, 1.0, 0.0), -90), n=20))
    add("dark", _m_cone(0.062, 0.090, (-1.390, 0.0, 0.0), _rot((0.0, 1.0, 0.0), -90), n=16))
    for deg in (45.0, 135.0, 225.0, 315.0):
        strake = _m_vplate(
            [(-0.150, 0.098), (0.450, 0.098), (0.450, 0.165), (-0.150, 0.165)],
            0.0, 0.012,
        )
        strake.apply_transform(_rot((1.0, 0.0, 0.0), deg))
        add("shell", strake)
    for x in (0.620, -0.620):
        add("band", _m_cyl(0.102, 0.030, (x, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=24))
    # fin root fairings, access panels and an umbilical door
    for deg in (45.0, 135.0, 225.0, 315.0):
        root = _m_box((0.320, 0.055, 0.050), (-1.130, 0.0, 0.098))
        root.apply_transform(_rot((1.0, 0.0, 0.0), deg))
        add("shell", root)
    for x, w, h, zc in ((0.900, 0.070, 0.045, 0.070),
                        (0.180, 0.090, 0.055, -0.060),
                        (-0.480, 0.070, 0.040, 0.070)):
        add("grey", _m_box((w, 0.010, h), (x, 0.0995, zc)))
    add("dark", _m_box((0.050, 0.030, 0.070), (0.720, -0.090, -0.040)))
    return g


def _add_missile_motion(scene, ents, scale):
    # X-fin display positions mapped to the plant's left/top/right/bottom channels.
    for k, (deg, channel) in enumerate(zip((45.0, 135.0, 225.0, 315.0), (0, 2, 1, 3))):
        base = _axis_quat((1, 0, 0), math.radians(deg))
        hinge = (-1.240, 0.0, 0.098)
        fin = _hinge_mesh(_m_vplate(
            [(-1.470, 0.098), (-1.020, 0.098), (-1.060, 0.360), (-1.470, 0.300)],
            0.0, 0.016,
        ), hinge, base, _rot((1, 0, 0), deg))
        rig = _model(scene, f"amraam_fin{k}", {"shell": [fin]}, scale=scale)
        _add_motion(ents, rig, hinge, scale, "surface", index=channel, base=base,
                    axis=(1, 0, 0), limit=math.radians(35.0), sign=-1.0)


def _build_satellite():
    """Three-axis comsat bus: gold MLI, radiators, dish, two solar wings."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add("shell", _m_box((0.620, 0.620, 0.860)))
    add("gold", _m_box((0.640, 0.640, 0.300), (0.0, 0.0, 0.020)))
    add("gold", _m_box((0.020, 0.400, 0.520), (-0.315, 0.0, 0.0)))
    # MLI blanket seams: thin raised gold bands around the bus
    for zz in (-0.250, 0.270):
        add("gold", _m_box((0.628, 0.628, 0.012), (0.0, 0.0, zz)))
    add("grey", _m_box((0.660, 0.660, 0.055), (0.0, 0.0, 0.445)))
    add("grey", _m_box((0.660, 0.660, 0.055), (0.0, 0.0, -0.445)))
    for sz in (-1.0, 1.0):
        add("shell", _m_box((0.440, 0.400, 0.020), (0.0, 0.0, sz * 0.440)))
        for ly in (-0.120, 0.0, 0.120):
            add("dark", _m_box((0.420, 0.012, 0.022), (0.0, ly, sz * 0.449)))
    # +x sensor face: dish on a short boom, supported by three struts
    add("metal", _m_cyl(0.028, 0.150, (0.385, 0.0, 0.160), _rot((0.0, 1.0, 0.0), 90)))
    add("white", _m_cone(0.240, 0.100, (0.505, 0.0, 0.160), _rot((0.0, 1.0, 0.0), -90), n=28))
    add("metal", _m_cyl(0.018, 0.060, (0.535, 0.0, 0.160), _rot((0.0, 1.0, 0.0), 90), n=12))
    for a in (0.0, 120.0, 240.0):
        ang = math.radians(a)
        add("metal", _m_rod(
            (0.320, 0.160 + 0.10 * math.cos(ang), 0.160 + 0.10 * math.sin(ang)),
            (0.465, 0.160 + 0.205 * math.cos(ang), 0.160 + 0.205 * math.sin(ang)),
            0.007,
        ))
    add("grey", _m_box((0.100, 0.220, 0.160), (0.325, -0.150, -0.120)))
    # top antenna boom + gold dish
    add("metal", _m_cyl(0.020, 0.220, (0.0, 0.120, 0.585)))
    add("gold", _m_cone(0.130, 0.055, (0.0, 0.120, 0.715), _rot((1.0, 0.0, 0.0), 180), n=24))
    add("grey", _m_cyl(0.045, 0.060, (0.0, 0.120, 0.700), n=16))
    # star trackers with sun hoods and thrusters on the -x face
    for sy, sz in ((0.16, 0.20), (-0.16, -0.10)):
        add("glass", _m_cyl(0.055, 0.070, (-0.330, sy, sz), _rot((0.0, 1.0, 0.0), 90), n=16))
        add("grey", _m_box((0.060, 0.150, 0.150), (-0.365, sy, sz)))
    for sx, sy in ((0.18, 0.18), (0.18, -0.18), (-0.18, 0.18), (-0.18, -0.18)):
        add("metal", _m_cone(0.028, 0.060, (-0.330, sy, -0.470), _rot((1.0, 0.0, 0.0), 180), n=12))
    # +/-y solar wings
    for s in (-1.0, 1.0):
        add("metal", _m_box((0.120, 0.380, 0.070), (0.0, s * 0.480, 0.0)))
        add("solar", _m_box((0.850, 2.100, 0.030), (0.0, s * 1.450, 0.0)))
        for k in (-0.700, 0.0, 0.700):
            add("blue", _m_box((0.870, 0.035, 0.040), (0.0, s * 1.450 + k, 0.0)))
        for k in (-0.280, 0.280):
            add("blue", _m_box((0.035, 2.100, 0.040), (k, s * 1.450, 0.0)))
        add("metal", _m_cyl(0.030, 0.100, (0.0, s * 0.640, 0.0), _rot((1.0, 0.0, 0.0), 90)))
        # hinge / panel-joint lines across the wing
        for hy in (0.95, 1.45, 1.95):
            add("gold", _m_box((0.860, 0.018, 0.036), (0.0, s * hy, 0.0)))
    return g


def _build_smallsat():
    """Orbit-transfer spacecraft: compact gold bus with two solar wings."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add("gold", _m_box((0.900, 0.900, 1.400)))
    for zz in (-0.360, 0.360):
        add("band", _m_box((0.908, 0.908, 0.016), (0.0, 0.0, zz)))
    add("white", _m_box((0.940, 0.940, 0.060), (0.0, 0.0, 0.730)))
    add("white", _m_box((0.940, 0.940, 0.060), (0.0, 0.0, -0.730)))
    add("white", _m_box((0.020, 0.560, 0.860), (-0.455, 0.0, 0.0)))  # side radiator
    add("glass", _m_cyl(0.055, 0.060, (-0.470, 0.240, 0.320), _rot((0.0, 1.0, 0.0), 90), n=14))
    add("metal", _m_cyl(0.050, 0.320, (0.520, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90)))
    add("white", _m_cone(0.300, 0.140, (0.720, 0.0, 0.0), _rot((0.0, 1.0, 0.0), -90), n=24))
    add("metal", _m_cyl(0.012, 0.120, (0.760, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=10))
    add("metal", _m_sph(0.020, (0.825, 0.0, 0.0), sub=1))
    for s in (-1.0, 1.0):
        add("metal", _m_box((0.180, 0.520, 0.100), (0.0, s * 0.680, 0.0)))
        add("solar", _m_box((0.900, 1.450, 0.045), (0.0, s * 1.620, 0.0)))
        for k in (-0.480, 0.0, 0.480):
            add("blue", _m_box((0.920, 0.040, 0.055), (0.0, s * 1.620 + k, 0.0)))
        add("blue", _m_box((0.040, 1.450, 0.055), (0.0, s * 1.620, 0.0)))
        for hy in (1.18, 1.62, 2.06):
            add("gold", _m_box((0.910, 0.018, 0.060), (0.0, s * hy, 0.0)))
    for sx, sy in ((0.28, 0.28), (0.28, -0.28), (-0.28, 0.28), (-0.28, -0.28)):
        add("metal", _m_cone(0.045, 0.090, (sx, sy, -0.780), _rot((1.0, 0.0, 0.0), 180), n=12))
    return g


def _build_station():
    """Space-station target: pressurised core, docking node, solar wings."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add("white", _m_cyl(0.300, 0.860, (-0.020, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=32))
    # module seams, gold MLI wrap and handrails along the core
    for x in (-0.360, 0.020, 0.330):
        add("shell", _m_cyl(0.304, 0.016, (x, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=32))
    add("gold", _m_cyl(0.303, 0.150, (-0.180, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=32))
    for a in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0):
        ang = math.radians(a)
        add("metal", _m_rod((-0.300, 0.318 * math.cos(ang), 0.318 * math.sin(ang)),
                            (0.300, 0.318 * math.cos(ang), 0.318 * math.sin(ang)), 0.006))
    add("grey", _m_cyl(0.310, 0.060, (0.400, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=32))
    add("gold", _m_cyl(0.190, 0.170, (0.420, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=24))
    add("metal", _m_cyl(0.200, 0.055, (0.487, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=24))
    add("dark", _m_cyl(0.125, 0.020, (0.520, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=20))
    # docking ring petals
    for k in range(8):
        a = 45.0 * k
        petal = _m_box((0.035, 0.050, 0.020), (0.515, 0.205, 0.0))
        petal.apply_transform(_rot((1.0, 0.0, 0.0), a))
        add("grey", petal)
    add("grey", _m_cyl(0.240, 0.320, (-0.610, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=24))
    add("metal", _m_cyl(0.090, 0.120, (-0.800, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=16))
    for s in (-1.0, 1.0):
        add("metal", _m_box((0.100, 0.420, 0.060), (0.0, s * 0.520, 0.0)))
        add("solar", _m_box((0.800, 1.000, 0.035), (0.0, s * 1.250, 0.0)))
        for k in (-0.300, 0.0, 0.300):
            add("blue", _m_box((0.820, 0.035, 0.045), (0.0, s * 1.250 + k, 0.0)))
        for k in (-0.360, 0.360):
            add("blue", _m_box((0.035, 1.000, 0.045), (k, s * 1.250, 0.0)))
        # radiator wings on the core, with louver lines
        add("white", _m_box((0.480, 0.030, 0.460), (0.0, s * 0.310, 0.0)))
        for lz in (-0.120, 0.0, 0.120):
            add("dark", _m_box((0.440, 0.034, 0.012), (0.0, s * 0.310, lz)))
    # short truss with a radiator and aft RCS quads
    for sy in (-0.14, 0.14):
        for sz in (-0.14, 0.14):
            add("metal", _m_rod((-0.760, sy, sz), (-0.980, sy, sz), 0.012))
    for a in (0, 90, 180, 270):
        ang = math.radians(a)
        add("metal", _m_rod(
            (-0.760, 0.14 * math.cos(ang), 0.14 * math.sin(ang)),
            (-0.980, 0.14 * math.cos(ang + math.pi / 2.0),
             0.14 * math.sin(ang + math.pi / 2.0)), 0.007))
    add("white", _m_box((0.030, 0.340, 0.340), (-0.995, 0.0, 0.0)))
    for sy, sz in ((0.08, 0.08), (0.08, -0.08), (-0.08, 0.08), (-0.08, -0.08)):
        add("metal", _m_cone(0.022, 0.050, (-1.020, sy, sz),
                             _rot((0.0, 1.0, 0.0), -90), n=10))
    # antenna masts with small gold dishes
    add("metal", _m_cyl(0.010, 0.260, (0.120, 0.120, 0.420), n=10))
    add("gold", _m_cone(0.085, 0.040, (0.120, 0.120, 0.560), _rot((1.0, 0.0, 0.0), 180), n=18))
    add("metal", _m_cyl(0.008, 0.220, (-0.150, -0.120, -0.400), n=10))
    add("gold", _m_cone(0.065, 0.030, (-0.150, -0.120, -0.520), n=16))
    add("metal", _m_rod((-0.400, 0.0, 0.340), (-0.400, 0.0, -0.340), 0.022))
    add("gold", _m_box((0.220, 0.220, 0.220), (0.180, 0.0, -0.360)))
    add("label", _m_box((0.230, 0.230, 0.030), (0.180, 0.0, -0.470)))
    return g


def _build_chaser():
    """Crew-Dragon-like chaser: capsule, nose docking ring, service trunk."""
    g = {}

    def add(key, mesh):
        g.setdefault(key, []).append(mesh)

    add("white", _m_loft([
        (0.460, 0.105, 0.105, 0.0),
        (0.415, 0.150, 0.150, 0.0),
        (0.345, 0.215, 0.215, 0.0),
        (0.250, 0.300, 0.300, 0.0),
        (0.140, 0.365, 0.365, 0.0),
        (0.020, 0.395, 0.395, 0.0),
        (0.000, 0.398, 0.398, 0.0),
    ], n=28))
    add("dark", _m_cyl(0.400, 0.035, (0.0, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=28))
    # IDSS docking adapter: metal ring, dark hatch, eight capture petals
    add("metal", _m_cyl(0.190, 0.080, (0.490, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=24))
    add("orange", _m_cyl(0.198, 0.016, (0.470, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=24))
    add("dark", _m_cyl(0.130, 0.014, (0.532, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=20))
    for k in range(8):
        a = 2.0 * math.pi * k / 8.0
        petal = _m_box((0.030, 0.045, 0.018),
                       (0.528, 0.170 * math.cos(a), 0.170 * math.sin(a)))
        petal.apply_transform(_rot((1.0, 0.0, 0.0), math.degrees(a)))
        add("grey", petal)
    # three windows on the capsule
    for sy, sz in ((0.0, 0.250), (0.235, -0.085), (-0.235, -0.085)):
        add("glass", _m_box((0.050, 0.075, 0.020), (0.300, sy, sz)))
    # SuperDraco pods around the capsule shoulder
    for k in range(8):
        a = 22.5 + 45.0 * k
        pod = _m_box((0.042, 0.040, 0.072), (0.105, 0.390, 0.0))
        pod.apply_transform(_rot((1.0, 0.0, 0.0), a))
        add("shell", pod)
    add("solar", _m_cyl(0.390, 0.440, (-0.260, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=28))
    # one half of the trunk carries the radiator (the other half is solar)
    add("white", _m_box((0.380, 0.250, 0.014), (-0.260, 0.0, 0.386)))
    for k in range(4):
        a = 45.0 + 90.0 * k
        add("dark", _m_box((0.430, 0.012, 0.012), (-0.260, 0.392, 0.0),
                           _rot((1.0, 0.0, 0.0), a)))
    for x in (-0.140, -0.360):
        add("orange", _m_cyl(0.396, 0.024, (x, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=28))
    for k in range(4):
        a = 45.0 + 90.0 * k
        add("orange", _m_box((0.070, 0.100, 0.130), (-0.440, 0.400, 0.0),
                             _rot((1.0, 0.0, 0.0), a)))
    for sy in (-1.0, 1.0):
        for sx in (-0.200, 0.120):
            add("orange", _m_box((0.100, 0.090, 0.100), (sx, sy * 0.360, 0.180)))
    add("metal", _m_cyl(0.160, 0.060, (-0.500, 0.0, 0.0), _rot((0.0, 1.0, 0.0), 90), n=20))
    return g


def _ring(scene, radius, color, center=(0, 0, 0), n=72, plane="xy", tube=0.05):
    """Continuous display ring built from exact-length cylinder segments.

    The previous version used a ring of spheres, which read as a dotted
    annotation. Real line segments make orbit / range / docking furniture look
    engineered instead of sketched, at the cost of a few dozen static bodies.
    """
    cx, cy, cz = (float(v) for v in center)
    out = []
    for i in range(n):
        a0 = 2.0 * math.pi * i / n
        a1 = 2.0 * math.pi * (i + 1) / n
        if plane == "yz":
            p0 = np.array((cx, cy + radius * math.cos(a0), cz + radius * math.sin(a0)))
            p1 = np.array((cx, cy + radius * math.cos(a1), cz + radius * math.sin(a1)))
        elif plane == "xz":
            p0 = np.array((cx + radius * math.cos(a0), cy, cz + radius * math.sin(a0)))
            p1 = np.array((cx + radius * math.cos(a1), cy, cz + radius * math.sin(a1)))
        else:
            p0 = np.array((cx + radius * math.cos(a0), cy + radius * math.sin(a0), cz))
            p1 = np.array((cx + radius * math.cos(a1), cy + radius * math.sin(a1), cz))
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        out.append(
            _cyl(scene, tube, L, color, tuple(0.5 * (p0 + p1)), _quat_z_to(d / L))
        )
    return out


def _polyline(scene, points, radius, color, closed=False):
    """Static polyline (orbit, boundary, scale bar) made of cylinder segments."""
    pts = [np.asarray(p, dtype=float) for p in points]
    if closed:
        pts = pts + [pts[0]]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        out.append(_cyl(scene, radius, L, color, tuple(0.5 * (a + b)), _quat_z_to(d / L)))
    return out


def _orbit_elements(f, mu):
    """Osculating elements and a predicted orbit from one recorded GCRF state."""
    r = np.asarray(f["pos"], float) * 1e6  # scene unit -> m
    v = np.asarray(f["vel"], float) * 1e6  # scene unit / s -> m / s
    rm = float(np.linalg.norm(r))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    e_vec = np.cross(v, h) / mu - r / rm
    e = float(np.linalg.norm(e_vec))
    a = 1.0 / (2.0 / rm - float(np.dot(v, v)) / mu)
    e_hat = e_vec / e if e > 1e-9 else r / rm
    h_hat = h / hn
    q_hat = np.cross(h_hat, e_hat)
    E = np.linspace(0.0, 2.0 * np.pi, ORBIT_POINTS, endpoint=False)
    b = a * math.sqrt(max(0.0, 1.0 - e * e))
    points = (
        np.outer(a * (np.cos(E) - e), e_hat)
        + np.outer(b * np.sin(E), q_hat)
    ) * 1e-6
    return {
        "sma": a,
        "ecc": e,
        "ra": a * (1.0 + e),
        "rp": a * (1.0 - e),
        "inc": math.acos(float(np.clip(h[2] / hn, -1.0, 1.0))),
        "points": points,
        "peri": e_hat * a * (1.0 - e) * 1e-6,
        "apo": -e_hat * a * (1.0 + e) * 1e-6,
    }


def _grid(scene, extent, spacing, color=(0.175, 0.285, 0.345, 1.0), tube=None):
    """Ground survey grid: depth cue plus a sense of scale for every outdoor task."""
    n = max(1, int(extent / spacing))
    tube = tube if tube is not None else max(0.012, spacing * 0.012)
    out = []
    for i in range(-n, n + 1):
        p = i * spacing
        out.append(
            _box(scene, (2.0 * extent, tube, 0.02), color, (0.0, p, 0.018))
        )
        out.append(
            _box(scene, (tube, 2.0 * extent, 0.02), color, (p, 0.0, 0.018))
        )
    return out


def _beam(scene, base, height, color, radius):
    """Thin vertical guide light used to mark waypoints / targets in depth."""
    x, y, z = (float(v) for v in base)
    return _cyl(scene, radius, height, color, (x, y, z + height / 2.0))


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


def _fmt_time(t):
    m = int(max(0.0, t) // 60)
    return f"{m:02d}:{max(0.0, t) - 60 * m:04.1f}"


def _rgba(color, alpha):
    return (int(color[0]), int(color[1]), int(color[2]), int(alpha))


def _mono(size, bold=True):
    from PIL import ImageFont  # noqa: E402

    path = FONT_MONO_BOLD if bold else FONT_MONO
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return _font(size, bold)


def _split_unit(value):
    """Split ``0.70 m`` into value/unit so the unit can be set smaller."""
    parts = value.rsplit(" ", 1)
    if len(parts) == 2 and parts[1] and len(parts[1]) <= 5 and not parts[1][0].isdigit():
        return parts[0], parts[1]
    return value, ""


def _sparkline(draw, box, values, color, label, idx=None):
    """Telemetry trace with a playhead: faint future, bright elapsed part and
    a live value readout at the cursor (Open MCT-style strip chart)."""
    if not values or len(values) < 2:
        return
    x0, y0, x1, y1 = box
    lo, hi = float(min(values)), float(max(values))
    if hi - lo < 1e-9:
        hi, lo = hi + 1.0, lo - 1.0
    pad = (hi - lo) * 0.16
    lo, hi = lo - pad, hi + pad
    n = len(values)
    idx = n - 1 if idx is None else int(max(0, min(n - 1, idx)))

    def pt(i):
        return (x0 + (x1 - x0) * i / (n - 1),
                y1 - (y1 - y0) * (values[i] - lo) / (hi - lo))

    pts = [pt(i) for i in range(n)]
    draw.text((x0, y0 - 13), label, font=_mono(8, bold=False), fill=FAINT_COLOR)
    cur = values[idx]
    fmt = f"{cur:,.1f}" if abs(cur) >= 10 else f"{cur:,.2f}"
    draw.text((x1, y0 - 13), fmt, font=_mono(9), fill=_rgba(color, 235), anchor="ra")
    # baseline grid
    for k in (0.0, 0.5, 1.0):
        yy = y1 - (y1 - y0) * k
        draw.line((x0, yy, x1, yy), fill=(255, 255, 255, 12), width=1)
    # whole trace (future included) very faint, elapsed part bright
    draw.polygon(pts + [(x1, y1), (x0, y1)], fill=_rgba(color, 12))
    draw.line(pts, fill=_rgba(color, 70), width=1, joint="curve")
    past = pts[: idx + 1]
    if len(past) > 1:
        draw.polygon(past + [(past[-1][0], y1), (x0, y1)], fill=_rgba(color, 30))
        draw.line(past, fill=_rgba(color, 70), width=5, joint="curve")
        draw.line(past, fill=_rgba(color, 240), width=1, joint="curve")
    cx, cy = pts[idx]
    draw.line((cx, y0 - 2, cx, y1), fill=(238, 244, 250, 80), width=1)
    draw.ellipse((cx - 4.0, cy - 4.0, cx + 4.0, cy + 4.0), outline=_rgba(color, 220))
    draw.ellipse((cx - 2.2, cy - 2.2, cx + 2.2, cy + 2.2), fill=_rgba(color, 255))


def _timeline(draw, box, frac, marks, color):
    """Mission time conductor: track, elapsed fill, phase ticks and playhead."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    mid = (y0 + y1) / 2.0
    draw.rounded_rectangle((x0, mid - 2, x1, mid + 2), radius=2,
                           fill=_rgba(CARD_TRACK, 220))
    if frac > 0.0:
        draw.rounded_rectangle((x0, mid - 2, x0 + max(4, w * frac), mid + 2),
                               radius=2, fill=_rgba(color, 220))
    for m, _phase in marks or ():
        if m <= 0.0 or m >= 1.0:
            continue
        x = x0 + w * m
        col = _rgba(color, 200) if m <= frac else _rgba(FAINT_COLOR, 120)
        draw.line((x, mid - 5, x, mid + 5), fill=col, width=1)
    x = x0 + w * float(max(0.0, min(1.0, frac)))
    draw.line((x, y0, x, y1), fill=(238, 244, 250, 220), width=2)
    draw.polygon(((x - 3, y0 - 1), (x + 3, y0 - 1), (x, y0 + 4)),
                 fill=(238, 244, 250, 220))


def _backdrop(task, h, w, hy):
    """Load the task's photographic sky/space plate, exposure-matched.

    Atmospheric plates are cropped to the visible sky band: these cameras look
    down, so mapping the whole photo to the frame would hide the cloud band
    behind the ground and leave only flat zenith blue. ``centre`` selects the
    slice of the photo that fills that band. Space plates cover the full frame.
    Cached per (file, frame size) with a 16 px horizon tolerance; returns None
    when the asset is absent, which keeps the procedural sky path working in
    asset-free checkouts.
    """
    spec = BACKDROPS.get(task)
    if spec is None:
        return None
    name, gain, centre = spec
    outdoor = TASK_META.get(task, ("", "", ACCENT_COLOR, True))[3]
    band = max(8, int(round(hy * h / 2.0)) * 2) if outdoor else h
    key = (name, h, w)
    cached = _BACKDROP_CACHE.get(key)
    # the panning cameras drift the horizon by a few pixels; keep the previous
    # resize within a 16 px tolerance so a slow pan never re-crops per frame
    if cached is not None and abs(cached[0] - band) <= 16:
        return cached[1]
    path = BACKDROP_DIR / name
    if not path.exists():
        _BACKDROP_CACHE[key] = (band, None)
        return None
    from PIL import Image  # noqa: E402

    with Image.open(path) as im:
        im = im.convert("RGB")
        sw, sh = im.size
        crop_h = max(1, int(round(sh * band / h)))
        top = int(round(centre * sh - crop_h / 2.0))
        top = max(0, min(sh - crop_h, top))
        im = im.crop((0, top, sw, top + crop_h)).resize((w, band), Image.LANCZOS)
    sky = np.asarray(im, np.float32) / 255.0 * gain
    if band < h:  # extend the horizon row under the ground (unseen)
        sky = np.concatenate((sky, np.repeat(sky[-1:], h - band, axis=0)), axis=0)
    sky = np.clip(sky, 0.0, 1.0)
    _BACKDROP_CACHE[key] = (band, sky)
    return sky


def _hash2(ix, iy, seed):
    """Deterministic 2D hash in [0, 1]; wraps as int64 on purpose."""
    h = ix * np.int64(374761393) + iy * np.int64(668265263)
    h = h + np.int64(seed) * np.int64(1442695041)
    h = (h ^ (h >> 13)) * np.int64(1274126177)
    h = h ^ (h >> 16)
    return (h & np.int64(0x7FFFFFFF)).astype(np.float32) * (1.0 / 2147483647.0)


def _noise2(x, y, seed):
    """Smooth 2D value noise for world-space ground mottling."""
    ix = np.floor(x)
    iy = np.floor(y)
    fx = (x - ix).astype(np.float32)
    fy = (y - iy).astype(np.float32)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    a = _hash2(ix, iy, seed)
    b = _hash2(ix + 1, iy, seed)
    c = _hash2(ix, iy + 1, seed)
    e = _hash2(ix + 1, iy + 1, seed)
    top = a + (b - a) * ux
    bot = c + (e - c) * ux
    return top + (bot - top) * uy


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ground_texture(a, depth, sky_mask, task, cache):
    """Repaint the flat ground plane as grassland.

    Every atmospheric task uses the same grass cover (the sea theme is kept as
    an option but no task selects it). The rasteriser has no texture stack, so
    the ground reads as one flat colour. Reconstructing world XY from the depth
    buffer lets this layer stay locked to the terrain through camera pans
    instead of sliding with the screen. Only upward-facing pixels close to z=0
    are repainted; the rocket landing pad is excluded by radius.
    """
    spec = GROUND_DETAIL.get(task)
    if spec is None or depth is None or _LAST_NORMAL is None or _LAST_CAM is None:
        return a
    rays = cache.get("rays")
    if rays is None:
        return a
    n = _LAST_NORMAL.astype(np.float32) / 127.5 - 1.0
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    theme, scale, exclude = spec
    cpos, look, _, fov, _ = _LAST_CAM
    fwd = np.asarray(look, float) - np.asarray(cpos, float)
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    # Genesis depth is measured along the optical axis; convert it to ray
    # distance so reconstructed ground points land on z=0.
    denom = np.clip(np.sum(rays * fwd[None, None, :], -1), 0.15, 1.0)
    ray_len = depth / denom
    pos = np.asarray(cpos, np.float32)[None, None, :] + rays * ray_len[..., None]
    # world size of one pixel at that depth: detail octaves finer than this
    # alias into speckle, so they are faded out with distance. Divide by the
    # incidence cosine (ray z for a flat floor): a pixel near the horizon
    # spans tens of metres along the view ray, and without this correction the
    # texture stretches into the horizontal banding that reads as scan lines.
    inc = np.clip(np.abs(rays[..., 2]), 0.05, 1.0)
    pix = ray_len * (2.0 * math.tan(math.radians(fov) * 0.5) / a.shape[0]) / inc
    x, y = pos[..., 0], pos[..., 1]
    # depth quantisation grows with range: a fixed metre tolerance drops whole
    # pixel rows out of the repaint at grazing distance, which shows up as
    # horizontal scan lines across the meadow
    zmax = np.maximum(max(0.05, scale * 0.012), ray_len * 0.004)
    g = (~sky_mask) & (n[..., 2] > 0.55) & (np.abs(pos[..., 2]) < zmax)
    if exclude:
        g = g & ((x * x + y * y) > exclude * exclude)
    if not g.any():
        return a
    cache["ground_mask"] = g

    if theme == "sea":
        # elongated along one axis: wind-driven streaks rather than blobs
        m = 0.60 * _noise2(x / scale, y / (scale * 0.30), 41)
        m += 0.28 * _noise2(x / (scale * 0.25) + 11.0, y / (scale * 0.09) - 3.0, 43)
        m += 0.12 * _noise2(x / (scale * 0.08) - 7.0, y / (scale * 0.04) + 5.0, 47)
        t = _smoothstep(0.28, 0.74, m)
        dark = np.array((0.006, 0.026, 0.048), np.float32)
        light = np.array((0.028, 0.085, 0.115), np.float32)
        target = dark + (light - dark) * t[..., None]
        crest = np.clip((m - 0.60) / 0.40, 0.0, 1.0) ** 2
        target = target + crest[..., None] * np.array((0.030, 0.048, 0.055), np.float32)
    else:
        dry = theme == "dry"

        def octave(wavelength, seed, ox, oy):
            lam = scale * wavelength
            w = np.clip(1.0 - pix / (0.30 * lam), 0.0, 1.0)
            return _noise2(x / lam + ox, y / lam + oy, seed), w

        # meadow -> patches -> clumps -> tufts -> blades: every octave fades
        # out once it drops below the pixel footprint, so distant grass stays
        # smooth instead of shimmering
        field_raw, fw = octave(6.0, 61, 3.0, -9.0)
        field = 0.5 + (field_raw - 0.5) * fw
        n2, w2 = octave(1.0, 7, 0.0, 0.0)
        n3, w3 = octave(0.30, 19, 13.0, -7.0)
        n4, w4 = octave(0.10, 31, -5.0, 3.0)
        n5, w5 = octave(0.035, 37, 23.0, -11.0)
        amp = 0.40 * w2 + 0.28 * w3 + 0.20 * w4 + 0.12 * w5
        m = (
            0.40 * n2 * w2 + 0.28 * n3 * w3 + 0.20 * n4 * w4 + 0.12 * n5 * w5
        ) / np.maximum(amp, 1e-3)
        t = _smoothstep(0.34, 0.66, m)
        if dry:
            dark = np.array((0.085, 0.072, 0.038), np.float32)
            light = np.array((0.150, 0.135, 0.072), np.float32)
        else:
            dark = np.array((0.105, 0.160, 0.065), np.float32)
            light = np.array((0.235, 0.320, 0.115), np.float32)
        target = dark + (light - dark) * t[..., None]
        if not dry:
            # broad straw-coloured field patches break up the green without
            # reading as a checkerboard
            mix = 0.34 * _smoothstep(0.46, 0.78, field)
            straw = np.array((0.275, 0.245, 0.130), np.float32)
            target = target * (1.0 - mix[..., None]) + straw * mix[..., None]
            # micro light/shade from the clump layer: grass is never uniform
            target = target * (0.93 + 0.14 * n4 + 0.08 * n5 * w5)[..., None]
            # broad cloud-light variation rolling across the meadow
            target = target * (0.88 + 0.24 * field)[..., None]
            # grazing incidence stretches every patch into a long horizontal
            # smear: fade the whole pattern out towards the horizon instead of
            # drawing streaks that read as scan lines
            flat = _smoothstep(0.12, 0.55, inc)
            flat = 0.35 + 0.65 * flat
            base = dark * 0.5 + light * 0.5
            target = base + (target - base) * flat[..., None]

    # follow the rasteriser's light level, but compress the range: a dusk shot
    # must not turn the meadow into a black hole
    lum = a.mean(-1, keepdims=True)
    shade = np.clip(lum / 0.060, 0.75, 1.25)
    target = target * (0.55 + 0.45 * shade)
    if "ground_grain" not in cache:
        rng = np.random.default_rng(29)
        cache["ground_grain"] = rng.normal(
            0.0, 1.0, (a.shape[0], a.shape[1], 1)
        ).astype(np.float32)
    target = target + cache["ground_grain"] * 0.006
    return np.clip(a * (1.0 - g[..., None]) + target * g[..., None], 0.0, 1.0)


_PLUME = None  # (nozzle, axis, length, radius, tick) set by the rocket loop


def _plume_overlay(a, cam, plume):
    """Additive screen-space flame glow projected from the 3D nozzle.

    The rasterised plume is opaque geometry, so it always reads as a clean
    airbrushed cone. A real exhaust is mostly *light*: a white-hot core, a
    saturated orange body and a wide soft halo, with a ragged, boiling edge.
    Those are cheap to paint in screen space: project the nozzle and the plume
    axis with the frame's own camera, then add a procedural taper whose edge
    is broken up by the same value noise used for the ground, so it flickers
    with the frames. Keeps the 3D core underneath for occlusion and depth.
    """
    if cam is None or plume is None:
        return a
    cpos, look, up, fov, _res = cam
    h, w = a.shape[:2]
    nozzle, axis, length, radius, tick = plume
    nozzle = np.asarray(nozzle, float)
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    cpos = np.asarray(cpos, float)
    fwd = np.asarray(look, float) - cpos
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, np.asarray(up, float))
    right /= max(np.linalg.norm(right), 1e-9)
    up2 = np.cross(right, fwd)
    fpx = (h * 0.5) / math.tan(math.radians(fov) * 0.5)

    def project(p):
        rel = np.asarray(p, float) - cpos
        z = float(np.dot(rel, fwd))
        if z <= 1.0:
            return None
        return (
            w * 0.5 + fpx * float(np.dot(rel, right)) / z,
            h * 0.5 - fpx * float(np.dot(rel, up2)) / z,
            z,
        )

    p0 = project(nozzle)
    p1 = project(nozzle + axis * length)
    if p0 is None or p1 is None:
        return a
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 2.0:
        return a
    dx, dy = dx / L, dy / L
    ex, ey = -dy, dx
    halfw = max(2.0, fpx * radius / p0[2])
    pad = int(halfw * 2.4 + 6)
    x0 = int(max(0, min(p0[0], p1[0]) - pad))
    x1 = int(min(w, max(p0[0], p1[0]) + pad))
    y0 = int(max(0, min(p0[1], p1[1]) - pad))
    y1 = int(min(h, max(p0[1], p1[1]) + pad))
    if x1 <= x0 or y1 <= y0:
        return a
    ys, xs = np.mgrid[y0:y1, x0:x1]
    u = ((xs - p0[0]) * dx + (ys - p0[1]) * dy) / L
    v = ((xs - p0[0]) * ex + (ys - p0[1]) * ey) / halfw
    # the plume is cut where it hits the pad
    u_max = 1.0
    if axis[2] < -0.2:
        u_max = float(np.clip((nozzle[2] - 0.15) / (length * -axis[2]), 0.0, 1.0))
    m = (u > 0.0) & (u < u_max + 0.02)
    if not m.any():
        return a
    # diameter: throat at the bell, widest a third of the way down, then taper
    prof = (0.34 + 0.66 * _smoothstep(0.0, 0.30, u)) * (
        1.0 - np.clip((u - 0.30) / 0.70, 0.0, 1.0)
    ) ** 0.55
    r_v = np.abs(v) / np.maximum(prof, 1e-3)
    core = np.clip(1.0 - r_v ** 2, 0.0, 1.0) ** 1.5
    halo = np.exp(-((np.abs(v) / np.maximum(prof, 1e-3) * 0.75) ** 2)) * 0.30
    # boiling edge: low-frequency noise scrolling down the plume
    noise = _value_noise(u * 3.0, v * 0.9 - tick * 0.06, 5, 3, 23)
    core = core * (0.55 + 0.75 * noise)
    stops_u = (0.00, 0.10, 0.35, 0.70, 1.00)
    stops_c = np.array(
        ((1.00, 0.97, 0.86), (1.00, 0.82, 0.40), (1.00, 0.52, 0.10),
         (0.96, 0.22, 0.02), (0.40, 0.04, 0.003)),
        np.float32,
    )
    uu = np.clip(u, 0.0, 1.0)
    col = np.stack(
        [np.interp(uu, stops_u, stops_c[:, i]) for i in range(3)], axis=-1
    ).astype(np.float32)
    fade = np.clip((u_max - u) / 0.06, 0.0, 1.0)
    fade *= np.clip(u / 0.02, 0.0, 1.0)  # soft throat, no square cap at the bell
    gain = (core * 0.95 + halo * 0.55) * fade
    gain *= m
    sub = a[y0:y1, x0:x1]
    a[y0:y1, x0:x1] = np.clip(sub + col * gain[..., None], 0.0, 1.0)
    return a


def _post(rgb, depth, task):
    """Depth-aware sky, atmosphere, bloom, vignette and film grade.

    The rasteriser gives a flat background colour and a hard horizon. Replacing
    the far-depth pixels with the task's backdrop (or a vertical sky gradient
    when the plate is missing) and fading distant ground into the horizon is
    what turns the render from a screenshot into a film.
    """
    from PIL import Image, ImageFilter  # noqa: E402

    outdoor = TASK_META.get(task, ("", "", ACCENT_COLOR, True))[3]
    a = rgb.astype(np.float32) / 255.0
    raw01 = a.copy()
    h, w = a.shape[:2]
    key = (task, h, w)
    cache = _POST_CACHE.setdefault(key, {})
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    # the horizon sits wherever the far-depth background starts; derive it
    # from the depth mask instead of assuming a fixed camera pitch
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        far = float(np.nanmax(depth))
        sky_mask = depth >= far * 0.9995
        if outdoor:
            # moved bodies/plumes can keep a stale far depth; only treat pixels
            # that still match the flat background colour as sky
            corners = np.concatenate(
                [raw01[:4, :4].reshape(-1, 3), raw01[:4, -4:].reshape(-1, 3)]
            )
            bgc = np.median(corners, axis=0)
            sky_mask &= np.abs(raw01 - bgc).max(-1) < 0.06
        d = np.nan_to_num(depth, nan=far, posinf=far)
        rows = np.where(sky_mask, np.arange(h, dtype=np.float32)[:, None], -1.0)
        hy = float(rows.max(0).mean()) / max(1, h) if sky_mask.any() else 0.32
        hy = float(np.clip(hy, 0.10, 0.85))
    else:
        sky_mask = np.zeros((h, w), bool)
        d = np.full((h, w), 1.0, np.float32)
        hy = 0.35

    if abs(cache.get("sky_hy", -1.0) - hy) > 0.004 or "sky" not in cache:
        plate = _backdrop(task, h, w, hy)
        if plate is not None:
            sky = plate
            # Fog blends into the plate's own colour at the horizon row, not a
            # global average, so the seam follows cloud/limb gradients.
            row = int(np.clip(hy * h, 0, h - 1))
            bg = plate[row][None, :, :].copy()
        elif outdoor:
            # polar dusk: near-black zenith, teal mid, muted slate horizon
            top = np.array((0.006, 0.014, 0.026), np.float32)
            mid = np.array((0.030, 0.070, 0.104), np.float32)
            hor = np.array((0.135, 0.200, 0.265), np.float32)
            t = np.clip(yy / max(hy, 0.10), 0.0, 1.0)
            sky = np.where(t < 0.42,
                           top + (mid - top) * (t / 0.42),
                           mid + (hor - mid) * ((t - 0.42) / 0.58))
            sky = np.broadcast_to(sky[:, None, :], (h, w, 3)).copy()
            # low sun: compact disc, tight inner halo, wide atmospheric wash
            sx, sy = 0.74, max(0.08, hy * 0.56)
            core = np.exp(-(((xx - sx) ** 2) * 5200.0 + ((yy - sy) ** 2) * 5200.0))
            halo = np.exp(-(((xx - sx) ** 2) * 24.0 + ((yy - sy) ** 2) * 70.0))
            wash = np.exp(-(((xx - sx) ** 2) * 7.0 + ((yy - sy) ** 2) * 32.0))
            sky += np.array((1.0, 0.86, 0.64), np.float32) * core[..., None] * 1.10
            sky += np.array((0.82, 0.52, 0.28), np.float32) * halo[..., None] * 0.30
            sky += np.array((0.30, 0.20, 0.12), np.float32) * wash[..., None] * 0.05
            bg = np.array((0.135, 0.200, 0.265), np.float32)
        else:
            top = np.array((0.006, 0.011, 0.020), np.float32)
            bot = np.array((0.020, 0.040, 0.062), np.float32)
            sky = top + (bot - top) * yy[..., None]
            sky = np.broadcast_to(sky, (h, w, 3)).copy()
            # faint nebula wash so deep space is not a dead flat black
            neb = np.exp(-(((xx - 0.24) ** 2) * 5.0 + ((yy - 0.28) ** 2) * 9.0))
            sky += np.array((0.10, 0.20, 0.30), np.float32) * neb[..., None] * 0.20
            bg = np.array((0.01, 0.02, 0.03), np.float32)
        cache["sky"] = sky
        cache["bg"] = bg
        cache["sky_hy"] = hy
        cache["plate"] = plate is not None

    sky = cache["sky"]
    m = Image.fromarray((sky_mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.GaussianBlur(2))
    alpha = np.asarray(m, np.float32)[..., None] / 255.0
    a = a * (1.0 - alpha) + sky * alpha

    # faint warm seam right above the horizon (kept very subtle; the fog
    # gradient carries the atmospheric falloff)
    if outdoor and sky_mask.any():
        below_ground = np.roll(~sky_mask, -10, axis=0)
        limb = sky_mask & below_ground
        if limb.any():
            a[limb] += np.array((0.035, 0.022, 0.012), np.float32)

    # screen-space key / rim / specular from the rasteriser normal buffer
    normal = _LAST_NORMAL
    if normal is not None and normal.shape[:2] == a.shape[:2]:
        n = normal.astype(np.float32) / 127.5 - 1.0
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        L = np.array((0.38, 0.55, 0.74), np.float32)
        L /= np.linalg.norm(L)
        lam = np.clip(np.sum(n * L, -1), 0.0, 1.0)
        valid = (~sky_mask).astype(np.float32)
        a *= (1.0 + (0.24 * lam - 0.09) * valid)[..., None]
        fill = np.clip(np.sum(n * -L, -1), 0.0, 1.0)
        a += (fill * 0.05 * valid)[..., None] * np.array((0.30, 0.48, 0.75), np.float32)

        rays = cache.get("rays")
        cam_tag = (
            None if _LAST_CAM is None else
            tuple(np.round(_LAST_CAM[0], 0)) + tuple(np.round(_LAST_CAM[1], 0))
        )
        if rays is None or cache.get("rays_cam") != cam_tag:
            rays = None
            if _LAST_CAM is not None:
                cpos, look, up, fov, _ = _LAST_CAM
                fwd = look - cpos
                fwd /= max(np.linalg.norm(fwd), 1e-9)
                right = np.cross(fwd, up)
                right /= max(np.linalg.norm(right), 1e-9)
                up2 = np.cross(right, fwd)
                tv = math.tan(math.radians(fov) / 2.0)
                th = tv * w / h
                xs = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
                ys = 1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0
                rays = (
                    fwd[None, None, :]
                    + right[None, None, :] * (xs * th)[None, :, None]
                    + up2[None, None, :] * (ys * tv)[:, None, None]
                )
                rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
            # moving camera (slow pan): rebuild in place, never accumulate
            cache["rays"] = rays
            cache["rays_cam"] = cam_tag
        if rays is not None:
            v = -rays
            ndv = np.clip(np.abs(np.sum(n * v, -1)), 0.0, 1.0)
            rim = ((1.0 - ndv) ** 2.4) * valid
            a += rim[..., None] * np.array((0.10, 0.22, 0.36), np.float32)
            hv = L + v
            hv /= np.maximum(np.linalg.norm(hv, axis=-1, keepdims=True), 1e-6)
            spec = (np.clip(np.sum(n * hv, -1), 0.0, 1.0) ** 30.0) * valid
            a += spec[..., None] * np.array((0.10, 0.09, 0.075), np.float32)
        a = np.clip(a, 0.0, 1.0)

    a = _ground_texture(a, d, sky_mask, task, cache)

    if not sky_mask.all() and depth is not None:
        # aerial haze belongs to the ground/horizon only: vehicles, markers and
        # emissive plumes must stay crisp instead of being washed into the plate
        ground = cache.get("ground_mask")
        if ground is None or ground.shape[:2] != a.shape[:2]:
            ground = ~sky_mask
        # Haze is anchored at the *far* ground quantile, not the median: the
        # near field stays crisp and dark, only the horizon dissolves. Using a
        # low quantile here washed the whole floor into the sky colour.
        ref = float(np.percentile(d[ground], 96)) if ground.any() else 1.0
        ref = max(ref, 1e-6)
        if outdoor:
            fog = 1.0 - np.exp(-((d / (ref * 0.85)) ** 1.7))
            # a photographic plate carries its own horizon haze, so let the
            # far ground dissolve into it more fully than the flat gradient
            cap = 0.90 if cache.get("plate") else 0.78
            if task == "missile":
                # the whole frame is grassland seen from 4 km: keep the meadow
                # readable instead of dissolving it into a dark storm horizon
                cap *= 0.45
            fog = np.clip(fog, 0.0, 0.92) * cap * ground
            a = a * (1.0 - fog)[..., None] + cache["bg"] * fog[..., None]
        else:
            haze = 1.0 - np.exp(-((d / (ref * 1.2)) ** 1.5))
            haze = np.clip(haze, 0.0, 0.5) * 0.10 * ground
            a = a * (1.0 - haze)[..., None] + cache["bg"] * haze[..., None]

    if not outdoor:
        a = np.clip(a * 1.10, 0.0, 1.0)

    if task == "transfer":
        a = _apply_earth(a, depth, cache)

    # film grade: gentle S-curve, cool shadows, warm highlights
    lum = a.mean(-1, keepdims=True)
    a = a + (a - lum) * 0.10
    a = np.clip((a - 0.5) * 1.055 + 0.5, 0.0, 1.0)
    a = a * 0.985 + np.array((0.004, 0.007, 0.012), np.float32)

    # bloom: threshold the highlights, blur them at 1/4 res, add back
    bright = np.clip(a - 0.76, 0.0, 1.0)
    small = Image.fromarray((bright * 255.0 * 1.5).clip(0, 255).astype(np.uint8))
    small = small.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(6))
    glow = np.asarray(small.resize((w, h), Image.BILINEAR), np.float32) / 255.0
    warm_px = (a[..., 0] - a[..., 2]) > 0.12
    bloom_tint = np.where(
        warm_px[..., None],
        np.array((0.95, 0.55, 0.22), np.float32),
        np.array((0.50, 0.62, 0.78), np.float32),
    )
    a = a + glow * bloom_tint * 0.44

    # separate warm bloom: emissive plumes and sunset clouds get a soft orange
    # halo instead of the cool HUD bloom above
    warm = np.clip((a[..., 0] - a[..., 2]) * 1.4, 0.0, 1.0)
    warm = warm * np.clip(a[..., 0] - 0.55, 0.0, 1.0)
    wsmall = Image.fromarray((warm * 255.0).clip(0, 255).astype(np.uint8))
    wsmall = wsmall.resize((max(1, w // 6), max(1, h // 6)), Image.BILINEAR)
    wsmall = wsmall.filter(ImageFilter.GaussianBlur(7))
    wglow = np.asarray(wsmall.resize((w, h), Image.BILINEAR), np.float32) / 255.0
    a = a + wglow[..., None] * np.array((0.55, 0.24, 0.055), np.float32) * 0.40

    # screen-space flame light for the rocket landing burn (both cameras)
    a = _plume_overlay(a, _LAST_CAM, _PLUME)

    if "vig" not in cache:
        r2 = (xx - 0.5) ** 2 * 1.15 + (yy - 0.5) ** 2
        cache["vig"] = (1.0 - 0.38 * np.clip(r2 / 0.42, 0.0, 1.0))[..., None]
        rng = np.random.default_rng(11)
        cache["grain"] = rng.normal(0.0, 1.0, (h, w, 1)).astype(np.float32)
    a = a * cache["vig"] + cache["grain"] * 0.010
    return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


_POST_CACHE = {}
_BACKDROP_CACHE = {}
_CLOSEUP_ONLY = False
_LAST_DEPTH = None
_LAST_NORMAL = None
_LAST_CAM = None
_CLOSE_DEPTH = None
_CLOSE_NORMAL = None
_CLOSE_CAM = None
_EARTH_R = 6.371
_EARTH_SUN = np.array((0.92, -0.10, 0.34), np.float32)


def _value_noise(u, v, nx, ny, seed):
    """Periodic-in-u bilinear value noise on the equirectangular map."""
    g = np.random.default_rng(seed).random((ny + 1, nx + 1)).astype(np.float32)
    x = u * nx
    y = np.clip(v, 0.0, 0.999999) * ny
    x0 = np.floor(x).astype(np.int64) % nx
    y0 = np.clip(np.floor(y).astype(np.int64), 0, ny - 1)
    fx = (x - np.floor(x)).astype(np.float32)
    fy = (y - np.floor(y)).astype(np.float32)
    a00 = g[y0, x0]
    a10 = g[y0, (x0 + 1) % nx]
    a01 = g[y0 + 1, x0]
    a11 = g[y0 + 1, (x0 + 1) % nx]
    return (a00 * (1 - fx) * (1 - fy) + a10 * fx * (1 - fy)
            + a01 * (1 - fx) * fy + a11 * fx * fy)


def _fbm(u, v, seed, octaves=5, base=(6, 3)):
    total = np.zeros_like(u, dtype=np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        nx = base[0] * 2 ** o
        ny = max(2, base[1] * 2 ** o)
        total = total + amp * _value_noise(u, v, nx, ny, seed + 17 * o)
        norm += amp
        amp *= 0.55
    return total / norm


def _smooth(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _earth_layer(h, w, cam_state):
    """Cacheable procedural Earth: surface colour, hit mask and ray distance.

    Genesis' rasteriser has no texture stack for this scene, so the planet is
    identified analytically (ray/sphere intersection against the depth buffer)
    and shaded with a value-noise continent map, cloud layer, day/night
    terminator, ocean specular and an atmospheric limb. Camera and planet are
    fixed in the transfer shot, so this is computed once and re-projected every
    frame; the depth test keeps the spacecraft and trail in front.
    """
    cpos, look, up, fov, _ = cam_state
    cpos = np.asarray(cpos, np.float32)
    fwd = np.asarray(look, np.float32) - cpos
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    right = np.cross(fwd, np.asarray(up, np.float32))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up2 = np.cross(right, fwd)
    tv = math.tan(math.radians(fov) / 2.0)
    th = tv * w / h
    xs = ((np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0)[None, :, None]
    ys = (1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0)[:, None, None]
    rays = fwd[None, None, :] + right[None, None, :] * xs * th + up2[None, None, :] * ys * tv
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)

    b = np.sum(rays * cpos, -1)
    c = float(np.dot(cpos, cpos)) - _EARTH_R ** 2
    disc = b * b - c
    hit = disc > 0.0
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    hit &= t > 0.0
    p = cpos + rays * t[..., None]
    n = p / _EARTH_R
    lat = np.arcsin(np.clip(n[..., 2], -1.0, 1.0)) / (0.5 * np.pi)
    u = (np.arctan2(n[..., 1], n[..., 0]) / (2.0 * np.pi)) % 1.0
    vv = 0.5 + 0.5 * lat

    land_n = _fbm(u, vv, 11, octaves=5, base=(56, 28))
    land = _smooth(0.485, 0.535, land_n)
    biome = _fbm(u + 0.37, vv * 1.4, 23, octaves=4, base=(40, 20))
    desert = _smooth(0.50, 0.68, biome)
    surf = np.stack(
        (
            0.008 + 0.008 * land_n,
            0.026 + 0.018 * land_n,
            0.078 + 0.040 * land_n,
        ),
        -1,
    ).astype(np.float32)  # deep ocean
    green = np.array((0.062, 0.140, 0.058), np.float32)
    sand = np.array((0.265, 0.200, 0.095), np.float32)
    rock = np.array((0.110, 0.110, 0.095), np.float32)
    ground = green * (1.0 - desert)[..., None] + sand * desert[..., None]
    ground = ground * (1.0 - 0.55 * _smooth(0.62, 0.80, vv))[..., None] + rock * 0.25
    surf = surf * (1.0 - land[..., None]) + ground * land[..., None]
    # polar ice, with a noisy edge so the caps are not latitude bands
    ice = _smooth(0.80, 0.92, np.abs(lat) + 0.06 * (land_n - 0.5))
    surf = surf * (1.0 - ice[..., None]) + np.array((0.62, 0.68, 0.76), np.float32) * ice[..., None]
    # cloud deck: stretched along longitude, broken by latitude bands
    cloud_n = _fbm(u * 1.25 + 0.21, vv * 1.05, 37, octaves=6, base=(48, 24))
    bands = 0.80 + 0.20 * np.sin(lat * 9.0 + 0.6) * np.cos(lat * 3.2)
    cloud = _smooth(0.560, 0.740, cloud_n * (0.75 + 0.25 * bands))
    surf = surf * (1.0 - 0.50 * cloud[..., None]) + 0.50 * cloud[..., None] * np.array(
        (0.70, 0.74, 0.80), np.float32
    )

    L = _EARTH_SUN / float(np.linalg.norm(_EARTH_SUN))
    lam = np.clip(np.sum(n * L, -1), 0.0, 1.0)
    day = 0.030 + 1.45 * lam
    surf = surf * day[..., None]
    term = np.clip(1.0 - np.abs(lam - 0.10) / 0.16, 0.0, 1.0)
    surf += term[..., None] * np.array((0.15, 0.055, 0.018), np.float32)
    view = -rays
    hv = L + view
    hv /= np.maximum(np.linalg.norm(hv, axis=-1, keepdims=True), 1e-6)
    spec = (np.clip(np.sum(n * hv, -1), 0.0, 1.0) ** 90.0) * (1.0 - land) * 0.22
    surf += spec[..., None] * np.array((0.55, 0.66, 0.86), np.float32)
    mu = np.clip(np.sum(n * view, -1), 0.0, 1.0)
    rim = (1.0 - mu) ** 3.0
    surf += rim[..., None] * np.array((0.10, 0.24, 0.52), np.float32) * 1.35
    surf = np.clip(surf, 0.0, 1.0)

    from PIL import Image, ImageFilter  # noqa: E402

    m = Image.fromarray((hit.astype(np.float32) * 255.0).astype(np.uint8))
    blur = np.asarray(m.filter(ImageFilter.GaussianBlur(5)), np.float32) / 255.0
    outer = np.clip(blur - hit.astype(np.float32), 0.0, 1.0)
    return surf, hit, t.astype(np.float32), outer


def _apply_earth(a, depth, cache):
    if _LAST_CAM is None or depth is None:
        return a
    if "earth" not in cache:
        cache["earth"] = _earth_layer(a.shape[0], a.shape[1], _LAST_CAM)
    surf, hit, t, outer = cache["earth"]
    tol = np.maximum(0.025 * t, 0.03)
    # alpha reaches 1 exactly at the rasterised surface (depth == t); the
    # previous form centred the ramp on t and left the planet at 50 %, which
    # let the flat Plastic blue show through the procedural shading
    alpha = np.clip((depth - t + 4.0 * tol) / (4.0 * tol), 0.0, 1.0) * hit
    a = a * (1.0 - alpha[..., None]) + surf * alpha[..., None]
    keep = np.clip((depth - t + 4.0 * tol) / (4.0 * tol), 0.0, 1.0)
    a = a + outer[..., None] * keep[..., None] * np.array((0.14, 0.34, 0.66), np.float32)
    return a


def _main_render(cam):
    """Main-camera render retaining depth, normals and the camera pose.

    The rasteriser exposes a normal buffer but Genesis only accepts dynamic
    lights on the (CUDA-only) batch renderer. Reading normals here lets
    ``_post`` add the missing key / rim / specular response in screen space.
    """
    global _LAST_DEPTH, _LAST_NORMAL, _LAST_CAM
    out = cam.render(rgb=True, depth=True, normal=True, force_render=True)
    _LAST_DEPTH = out[1] if len(out) > 1 else None
    _LAST_NORMAL = out[3] if len(out) > 3 else None
    try:
        _LAST_CAM = (
            np.asarray(cam.pos, dtype=float).reshape(-1)[:3],
            np.asarray(cam.lookat, dtype=float).reshape(-1)[:3],
            np.asarray(cam.up, dtype=float).reshape(-1)[:3],
            float(cam.fov),
            tuple(cam.res),
        )
    except Exception:
        _LAST_CAM = None
    return out[0]


def _close_render(cam):
    """Close-up render retaining its own depth, normals and camera pose.

    The standalone close-up replays go through the same _post pipeline as the
    main frame, so the depth/normal buffers of the close camera must be kept
    alongside the RGB image.
    """
    global _CLOSE_DEPTH, _CLOSE_NORMAL, _CLOSE_CAM
    out = cam.render(rgb=True, depth=True, normal=True, force_render=True)
    _CLOSE_DEPTH = out[1] if len(out) > 1 else None
    _CLOSE_NORMAL = out[3] if len(out) > 3 else None
    try:
        _CLOSE_CAM = (
            np.asarray(cam.pos, dtype=float).reshape(-1)[:3],
            np.asarray(cam.lookat, dtype=float).reshape(-1)[:3],
            np.asarray(cam.up, dtype=float).reshape(-1)[:3],
            float(cam.fov),
            tuple(cam.res),
        )
    except Exception:
        _CLOSE_CAM = None
    return out[0]


def _post_close(rgb, task):
    """Run the close-up through the same grade/background/ground pass."""
    global _LAST_DEPTH, _LAST_NORMAL, _LAST_CAM
    saved = (_LAST_DEPTH, _LAST_NORMAL, _LAST_CAM)
    _LAST_DEPTH, _LAST_NORMAL, _LAST_CAM = _CLOSE_DEPTH, _CLOSE_NORMAL, _CLOSE_CAM
    try:
        return _post(rgb, _CLOSE_DEPTH, task)
    finally:
        _LAST_DEPTH, _LAST_NORMAL, _LAST_CAM = saved


def _frame(main_rgb, close_rgb, state, label=None, depth=None):
    """Cinematic full-bleed frame: graded render, glass HUD, close-up inset."""
    if _CLOSEUP_ONLY and close_rgb is not None:
        return _post_close(close_rgb, state.get("task_key", ""))
    if close_rgb is not None:
        close_rgb = _post_close(close_rgb, state.get("task_key", ""))
    from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

    if depth is None:
        depth = _LAST_DEPTH
    task = state.get("task_key", "")
    accent = TASK_META.get(task, ("", "", ACCENT_COLOR, True))[2]
    rgb = _post(main_rgb, depth, task)
    canvas = Image.fromarray(rgb).convert("RGB")
    W, H = canvas.size

    # subtle top/bottom cinematic falloff keeps overlaid type readable
    shade = Image.new("L", (1, H))
    sd = ImageDraw.Draw(shade)
    for y in range(H):
        top = max(0.0, 1.0 - y / 170.0) ** 2 * 0.26
        bot = max(0.0, (y - (H - 190)) / 190.0) ** 2 * 0.36
        sd.point((0, y), fill=int(min(1.0, top + bot) * 255))
    shade = shade.resize((W, H))
    canvas = Image.composite(Image.new("RGB", (W, H), (2, 5, 8)), canvas, shade)

    mg = 36
    # ---------------- close-up inset (bottom right) ----------------
    if close_rgb is not None:
        iw, ih = CLOSE_RES
        ix, iy = W - mg - iw, H - mg - ih
        close = Image.fromarray(close_rgb).resize(CLOSE_RES, Image.LANCZOS)
        mask = Image.new("L", CLOSE_RES, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, iw - 1, ih - 1), radius=9, fill=255)
        shadow = Image.new("RGBA", (iw + 40, ih + 40), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            (18, 22, iw + 18, ih + 26), radius=12, fill=(0, 0, 0, 150)
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(14))
        canvas.paste(Image.new("RGB", (iw + 40, ih + 40), (0, 0, 0)), (ix - 20, iy - 14),
                     shadow.split()[3])
        canvas.paste(close, (ix, iy), mask)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rounded_rectangle((ix, iy, ix + iw, iy + ih), radius=9,
                             outline=_rgba(accent, 80), width=1)
        # corner ticks
        L = 16
        for cx, cy, dx, dy in (
            (ix - 7, iy - 7, 1, 1), (ix + iw + 7, iy - 7, -1, 1),
            (ix - 7, iy + ih + 7, 1, -1), (ix + iw + 7, iy + ih + 7, -1, -1),
        ):
            od.line((cx, cy, cx + dx * L, cy), fill=_rgba(accent, 150), width=1)
            od.line((cx, cy, cx, cy + dy * L), fill=_rgba(accent, 150), width=1)
        tag = (label or "CLOSE-UP").upper()
        tf = _mono(9)
        tw = od.textlength(tag, font=tf)
        od.rounded_rectangle((ix + 12, iy + 12, ix + 24 + tw, iy + 34), radius=4,
                             fill=(4, 9, 13, 200), outline=_rgba(accent, 60))
        od.text((ix + 18, iy + 18), tag, font=tf, fill=_rgba(accent, 235))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")

    # ---------------- HUD glass panel (bottom left) ----------------
    pw, ph = 540, 300
    px, py = mg, H - mg - ph
    panel_mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(panel_mask).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=10, fill=255)
    under = canvas.crop((px, py, px + pw, py + ph)).filter(ImageFilter.GaussianBlur(12))
    under = Image.blend(under, Image.new("RGB", (pw, ph), CARD_BG), 0.70)
    canvas.paste(under, (px, py), panel_mask)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rounded_rectangle((px, py, px + pw, py + ph), radius=10,
                        outline=_rgba(accent, 65), width=1)
    d.rectangle((px + 1, py + 12, px + 3, py + ph - 12), fill=_rgba(accent, 210))

    x0, x1 = px + 22, px + pw - 22
    code, cat = TASK_META.get(task, ("--", "", accent, True))[:2]
    d.text((x0, py + 16), code, font=_mono(10), fill=_rgba(accent, 240))
    d.text((x0 + 52, py + 16), cat, font=_mono(8, bold=False), fill=FAINT_COLOR)

    # live / terminal state pill
    result = state.get("result")
    pill = result or "REPLAY"
    pcol = {"SUCCESS": (46, 150, 92), "FAILURE": (170, 62, 66)}.get(pill, (40, 96, 118))
    pfont = _mono(9)
    ptw = d.textlength(pill, font=pfont)
    d.rounded_rectangle((x1 - ptw - 34, py + 13, x1, py + 35), radius=11,
                        fill=_rgba(pcol, 225))
    d.ellipse((x1 - ptw - 26, py + 21, x1 - ptw - 20, py + 27), fill=(240, 255, 250, 230))
    d.text((x1 - ptw - 15, py + 19), pill, font=pfont, fill=(240, 248, 250, 245))

    d.text((x0, py + 40), _fit(d, state["task"], _font(27), x1 - x0 - 10),
           font=_font(27), fill=TEXT_COLOR)

    # hero metric, unit in a smaller face
    d.text((x0, py + 82), state["hero"][0].upper(), font=_mono(8, bold=False),
           fill=FAINT_COLOR)
    value, unit = _split_unit(state["hero"][1])
    vf = _font(46)
    d.text((x0, py + 94), _fit(d, value, vf, 250), font=vf, fill=TEXT_COLOR)
    if unit:
        vw = d.textlength(_fit(d, value, vf, 250), font=vf)
        d.text((x0 + vw + 7, py + 116), unit, font=_mono(13), fill=_rgba(accent, 235))

    spark = state.get("spark")
    if spark:
        _sparkline(d, (x0 + 300, py + 100, x1, py + 154), spark[0], accent, spark[1],
                   idx=state.get("spark_idx"))

    d.line((x0, py + 168, x1, py + 168), fill=(255, 255, 255, 26), width=1)
    row_lab, row_val = _font(10, bold=False), _mono(10)
    yy = py + 180
    for row in list(state.get("rows", ()))[:3]:
        lab, val = row[0], row[1]
        vcol = row[2] if len(row) > 2 else TEXT_COLOR
        d.text((x0, yy), lab, font=row_lab, fill=DIM_COLOR)
        val = _fit(d, val, row_val, (x1 - x0) * 0.55)
        d.text((x1 - d.textlength(val, font=row_val), yy - 1), val,
               font=row_val, fill=vcol)
        yy += 21

    if state.get("note"):
        d.text((x0, py + 244), _fit(d, state["note"], _mono(8, bold=False), x1 - x0),
               font=_mono(8, bold=False), fill=FAINT_COLOR)

    frac = float(state.get("progress") or 0.0)
    _timeline(d, (x0 + 2, py + ph - 20, x1 - 2, py + ph - 6), frac,
              state.get("marks"), accent)
    d.text((x0, py + ph - 38), "EPISODE", font=_mono(7, bold=False), fill=FAINT_COLOR)
    d.text((x0 + 62, py + ph - 39), state.get("clock", ""),
           font=_mono(9, bold=False), fill=DIM_COLOR)

    # ---------------- brand + telemetry header ----------------
    d.text((mg, 34), "AEROSPACERL", font=_mono(10), fill=(238, 244, 250, 225))
    d.text((mg + 92, 34), "//  GENESIS", font=_mono(10, bold=False), fill=_rgba(accent, 210))
    d.line((mg, 52, mg + 196, 52), fill=(255, 255, 255, 34), width=1)
    d.text((mg, 59), "MISSION REPLAY", font=_mono(8, bold=False), fill=DIM_COLOR)

    clock = state.get("clock", "")
    phase = state.get("phase", "")
    d.text((W - mg, 34), clock, font=_mono(10), fill=(238, 244, 250, 225), anchor="ra")
    if phase:
        d.text((W - mg, 50), phase, font=_mono(8, bold=False), fill=_rgba(accent, 215), anchor="ra")
    d.text((W - mg, 66), "60 FPS  ·  GENESIS", font=_mono(7, bold=False),
           fill=FAINT_COLOR, anchor="ra")

    canvas = Image.alpha_composite(canvas.convert("RGBA"), ov)
    gcs = state.get("gcs")
    if gcs is not None:  # mission-planner layer rides on the composed frame
        canvas = _gcs_overlay(canvas, task, gcs)
    return np.asarray(canvas.convert("RGB"))


# ---------------- GCS overlay (pure-PIL mission-planner layer) --------------
# Everything below draws on the composed frame; the Genesis scene, cameras and
# the _post grade are untouched. Layout derives from MAIN_RES so the layer
# survives a resolution change. Three borrowed mission-ops displays live here:
# Orbiter's Docking MFD (CVEL/TVEL/CDST, _dock_rows), STK's attitude sphere
# (_att_sphere) and an Open MCT-style time conductor + strip chart
# (_timeline / _sparkline with playhead).
def _quat_euler(q):
    """wxyz quaternion -> (roll, pitch, yaw), aerospace convention."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _attitude(task, frames, fi):
    """(roll, pitch, yaw) for the navball / heading tape / plan triangle.

    transfer / docking / rendezvous record quat=None: the transfer navball
    falls back to the numerical flight-path angle + inertial track, the CWH
    tasks to the drawn chaser's target-facing attitude (the same _quat_x_to
    the replay poses the body with), so the dial always shows a real state.
    """
    q = frames[fi].get("quat")
    if q is not None:
        return _quat_euler(q)
    p = np.asarray(frames[fi]["pos"], float)
    if task == "transfer":
        i0, i1 = max(0, fi - 1), min(len(frames) - 1, fi + 1)
        v = np.asarray(frames[i1]["pos"], float) - np.asarray(frames[i0]["pos"], float)
        nv, r = float(np.linalg.norm(v)), float(np.linalg.norm(p))
        if nv < 1e-12 or r < 1e-12:
            return 0.0, 0.0, 0.0
        v = v / nv
        fpa = math.asin(float(np.clip(np.dot(v, p / r), -1.0, 1.0)))
        return 0.0, fpa, math.atan2(v[1], v[0])
    n = float(np.linalg.norm(p))
    if n < 1e-9:
        return 0.0, 0.0, 0.0
    return _quat_euler(_quat_x_to(-p / n))


def _chip(d, rect, accent, label=None):
    """Glass chip matching the HUD/inset furniture: dark translucent fill,
    1 px accent border at ~30 %, corner brackets, micro-caps mono label."""
    x0, y0, x1, y1 = rect
    d.rounded_rectangle(rect, radius=8, fill=(10, 16, 22, 140),
                        outline=_rgba(accent, 76), width=1)
    L = 9
    for cx, cy, dx, dy in (
        (x0 - 4, y0 - 4, 1, 1), (x1 + 4, y0 - 4, -1, 1),
        (x0 - 4, y1 + 4, 1, -1), (x1 + 4, y1 + 4, -1, -1),
    ):
        d.line((cx, cy, cx + dx * L, cy), fill=_rgba(accent, 140), width=1)
        d.line((cx, cy, cx, cy + dy * L), fill=_rgba(accent, 140), width=1)
    if label:
        d.text((x0 + 13, y0 + 7), label, font=_mono(8, bold=False), fill=FAINT_COLOR)


def _dash_line(d, p0, p1, fill, width=1, dash=5, gap=4):
    """Single dashed segment; PIL has no dash support of its own."""
    x0, y0 = p0
    x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-6:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    t = 0.0
    while t < L:
        t2 = min(t + dash, L)
        d.line((x0 + ux * t, y0 + uy * t, x0 + ux * t2, y0 + uy * t2),
               fill=fill, width=width)
        t = t2 + gap


def _dash_poly(d, pts, fill, width=1, dash=5, gap=4):
    for a, b in zip(pts[:-1], pts[1:]):
        _dash_line(d, a, b, fill, width, dash, gap)


def _dash_circle(d, center, r, fill, width=1, n=60, duty=0.55):
    """Dashed ring from gapped arc segments (target orbit on the plan view)."""
    x, y = center
    seg = 360.0 / n
    for i in range(n):
        d.arc((x - r, y - r, x + r, y + r), start=i * seg,
              end=i * seg + seg * duty, fill=fill, width=width)


def _adi_geometry(hero):
    """Dial centre / radius in MAIN_RES pixels (satellite flies a hero ADI)."""
    s = MAIN_RES[1] / 810.0
    r = int((92 if hero else 72) * s)
    cx = int((150 if hero else 128) * MAIN_RES[0] / 1440.0)
    cy = int((200 if hero else 168) * s)
    return cx, cy, r


def _draw_adi(d, ov, accent, roll, pitch, hero):
    """Attitude director indicator: sky/ground ball, pitch ladder, roll scale.

    The sky half and the ladder live on one layer rotated by the bank angle,
    the way a mechanical AI's ball moves under the fixed aircraft symbol.
    """
    from PIL import Image, ImageDraw

    cx, cy, r = _adi_geometry(hero)
    s = MAIN_RES[1] / 810.0
    _chip(d, (cx - r - 14, cy - r - 24, cx + r + 14, cy + r + 12), accent, "ADI")
    dia = 2 * r
    k = r / 50.0  # px per degree of pitch: the ladder spans +/-40 deg
    pitch_px = float(np.clip(math.degrees(pitch) * k, -r, r))
    ball = Image.new("RGBA", (dia, dia), (0, 0, 0, 0))
    bd = ImageDraw.Draw(ball)
    bd.rectangle((-dia, -dia, 2 * dia, r + pitch_px), fill=_rgba(accent, 66))
    f7 = _mono(max(7, int(8 * s)), bold=False)
    for deg in range(-40, 41, 10):
        v = r - deg * k
        half = 30 if deg == 0 else 22 if deg % 20 == 0 else 13
        bd.line((r - half, v, r - 5, v), fill=(238, 244, 250, 165), width=1)
        bd.line((r + 5, v, r + half, v), fill=(238, 244, 250, 165), width=1)
        if deg in (-40, -20, 20, 40):
            t = str(abs(deg))
            bd.text((r - half - 4, v), t, font=f7, fill=(238, 244, 250, 150), anchor="rm")
            bd.text((r + half + 4, v), t, font=f7, fill=(238, 244, 250, 150), anchor="lm")
    ball = ball.rotate(math.degrees(roll), resample=Image.BILINEAR, center=(r, r))
    dial = Image.new("RGBA", (dia, dia), (0, 0, 0, 0))
    ImageDraw.Draw(dial).ellipse((0, 0, dia - 1, dia - 1), fill=(30, 22, 14, 210))
    dial = Image.alpha_composite(dial, ball)
    circ = Image.new("L", (dia, dia), 0)
    ImageDraw.Draw(circ).ellipse((0, 0, dia - 1, dia - 1), fill=255)
    ov.paste(dial, (cx - r, cy - r), circ)
    # fixed roll scale along the top arc, bank pointer riding on it
    for adeg in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
        a = math.radians(adeg)
        r0 = r - (11 if adeg % 30 == 0 else 7)
        d.line((cx + r0 * math.sin(a), cy - r0 * math.cos(a),
                cx + (r - 3) * math.sin(a), cy - (r - 3) * math.cos(a)),
               fill=(238, 244, 250, 140), width=1)
    a = float(roll)
    d.polygon(
        [(cx + (r - 2) * math.sin(a), cy - (r - 2) * math.cos(a)),
         (cx + (r - 13) * math.sin(a - 0.055), cy - (r - 13) * math.cos(a - 0.055)),
         (cx + (r - 13) * math.sin(a + 0.055), cy - (r - 13) * math.cos(a + 0.055))],
        fill=_rgba(AMBER_COLOR, 235),
    )
    # fixed aircraft symbol
    wspan = int(r * 0.42)
    d.line((cx - wspan, cy, cx - 9, cy), fill=_rgba(AMBER_COLOR, 245), width=2)
    d.line((cx + 9, cy, cx + wspan, cy), fill=_rgba(AMBER_COLOR, 245), width=2)
    d.line((cx - wspan, cy, cx - wspan, cy + 5), fill=_rgba(AMBER_COLOR, 245), width=2)
    d.line((cx + wspan, cy, cx + wspan, cy + 5), fill=_rgba(AMBER_COLOR, 245), width=2)
    d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=_rgba(AMBER_COLOR, 245), width=2)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=_rgba(accent, 90), width=1)


def _draw_heading(d, accent, yaw, hero):
    """Ribbon compass under the ADI: 15 deg ticks, cardinal letters, caret."""
    cx, cy, r = _adi_geometry(hero)
    s = MAIN_RES[1] / 810.0
    w = int((266 if hero else 240) * MAIN_RES[0] / 1440.0)
    x0 = cx - (r + 14)  # left edge aligned with the ADI chip
    x1 = x0 + w
    y0 = cy + r + 18
    y1 = y0 + int(34 * s)
    _chip(d, (x0, y0, x1, y1), accent, None)
    hdg = (90.0 - math.degrees(yaw)) % 360.0  # world yaw (from +x) -> compass
    cxx = (x0 + x1) / 2.0
    px_deg = (x1 - x0 - 20) / 120.0  # +/-60 deg of tape
    f7 = _mono(max(7, int(8 * s)), bold=False)
    for h in range(0, 360, 15):
        off = ((h - hdg + 540.0) % 360.0) - 180.0
        if abs(off) > 60.0:
            continue
        x = cxx + off * px_deg
        if h % 90 == 0:
            d.text((x, y1 - 20), "NESW"[h // 90], font=f7,
                   fill=_rgba(accent, 230), anchor="mm")
            d.line((x, y1 - 12, x, y1 - 6), fill=(238, 244, 250, 170), width=1)
        elif h % 45 == 0:
            d.line((x, y1 - 10, x, y1 - 6), fill=(238, 244, 250, 130), width=1)
        else:
            d.line((x, y1 - 8, x, y1 - 6), fill=(238, 244, 250, 90), width=1)
    d.polygon([(cxx - 5, y0 + 14), (cxx + 5, y0 + 14), (cxx, y0 + 20)],
              fill=_rgba(AMBER_COLOR, 240))
    txt = f"{hdg:03.0f}"
    f9 = _mono(max(8, int(9 * s)))
    tw = d.textlength(txt, font=f9)
    d.rounded_rectangle((cxx - tw / 2 - 5, y0 + 1, cxx + tw / 2 + 5, y0 + 13),
                        radius=3, fill=(10, 16, 22, 200), outline=_rgba(accent, 80))
    d.text((cxx, y0 + 7), txt, font=f9, fill=TEXT_COLOR, anchor="mm")


def _stage(task, telem, frames, extras, fi):
    """Stage pill labels + index of the current one for the top strip."""
    n = len(frames)
    s = float(telem["series"][fi])
    last = fi >= n - 1
    ok = extras.get("terminal") == "SUCCESS"
    if task == "quadx":
        parts = str(telem["phase"][fi]).split()  # "WAYPOINT k / n"
        k, nn = (parts[1], parts[3]) if len(parts) >= 4 else ("1", "?")
        labels = ["LAUNCH", f"CRUISE WPT {k}/{nn}", "COMPLETE"]
        cur = 2 if last else (0 if float(telem["times"][fi]) < 1.0 else 1)
    elif task == "quadx_track":
        labels = ["SEARCH", "TRACK", "COMPLETE"]
        cur = 2 if last else (1 if frames[fi].get("visible") else 0)
    elif task in ("fixedwing", "quadplane"):
        labels = ["LAUNCH", f"WPT {int(frames[fi].get('wpt', 0)) + 1}/4", "DONE"]
        cur = 2 if last else (0 if float(telem["times"][fi]) < 1.0 else 1)
    elif task == "rocket":
        labels = ["DESCENT", "TERMINAL", "LANDED"]
        cur = 2 if (last or s < 2.0) else (1 if s <= 150.0 else 0)
    elif task == "missile":
        labels = ["MIDCOURSE", "TERMINAL", "INTERCEPT"]
        cur = 2 if last else (0 if s > 2000.0 else 1)
    elif task == "satellite":
        labels = ["SLEW", "SETTLE", "LOCKED"]
        cur = 0 if s > 10.0 else (1 if s > 1.0 else 2)
    elif task == "transfer":
        labels = ["BURN", "COAST", "CIRCULARIZED"]
        cur = 2 if last else (0 if fi < n // 3 else 1)
    else:  # docking / rendezvous: CWH range thresholds
        labels = ["APPROACH", "PROXIMITY", "CAPTURE"]
        cur = 2 if (s < 2.0 or (last and ok)) else (1 if s <= 50.0 else 0)
    return labels, cur


def _draw_stages(d, task, accent, telem, frames, extras, fi):
    """Connected pill strip, top-center: done / current / pending styling."""
    W = MAIN_RES[0]
    s = MAIN_RES[1] / 810.0
    labels, cur = _stage(task, telem, frames, extras, fi)
    f8 = _mono(max(7, int(8 * s)))
    pad, hh, gap = 10.0 * s, 18.0 * s, 15.0 * s
    widths = [d.textlength(lb, font=f8) + 2 * pad for lb in labels]
    x = (W - (sum(widths) + gap * (len(labels) - 1))) / 2.0
    y = 20.0 * s
    for i, (lb, wd) in enumerate(zip(labels, widths)):
        if i:
            d.line((x - gap, y + hh / 2, x, y + hh / 2), fill=_rgba(accent, 70), width=1)
        box = (x, y, x + wd, y + hh)
        if i < cur:  # done
            d.rounded_rectangle(box, radius=hh / 2, fill=(10, 16, 22, 120),
                                outline=_rgba(accent, 110), width=1)
            col = _rgba(accent, 205)
        elif i == cur:  # current: solid accent + soft glow, dark text
            d.rounded_rectangle((x - 3, y - 3, x + wd + 3, y + hh + 3),
                                radius=hh / 2 + 3, fill=_rgba(accent, 30))
            d.rounded_rectangle(box, radius=hh / 2, fill=_rgba(accent, 232))
            col = (10, 16, 22, 255)
        else:  # pending
            d.rounded_rectangle(box, radius=hh / 2, fill=(10, 16, 22, 60),
                                outline=_rgba(FAINT_COLOR, 55), width=1)
            col = _rgba(FAINT_COLOR, 160)
        d.text((x + pad, y + hh / 2), lb, font=f8, fill=col, anchor="lm")
        x += wd + gap


def _dock_rows(f):
    """Orbiter Docking-MFD quantities from the recorded relative state.

    CVEL is the line-of-sight closing rate (positive when closing), TVEL the
    lateral rate that has to be nulled to stay on the corridor, and CDST the
    off-axis distance from the station's +x docking axis.
    """
    p = np.asarray(f["pos"], float)
    v = np.asarray(f.get("vel", np.zeros(3)), float)
    r = float(np.linalg.norm(p))
    los = p / max(r, 1e-9)
    cvel = -float(np.dot(v, los))
    tvel = float(np.linalg.norm(v - float(np.dot(v, los)) * los))
    cdst = float(np.linalg.norm(p[1:]))
    return [
        ("CVEL closing", f"{cvel:+.3f} m/s", MINT_COLOR if cvel >= 0.0 else BAD_COLOR),
        ("TVEL lateral", f"{tvel:.3f} m/s", MINT_COLOR if tvel < 0.05 else AMBER_COLOR),
        ("CDST off-axis", f"{cdst:.2f} m"),
    ]


def _att_sphere(d, ov, accent, frames, fi):
    """STK-style attitude sphere for the CMG task: target-frame grid, the live
    body triad, an angular-rate vector and the attitude-error readout."""
    sx, sy = MAIN_RES[0] / 1440.0, MAIN_RES[1] / 810.0
    x0, y0, x1, y1 = int(944 * sx), int(88 * sy), int(1404 * sx), int(408 * sy)
    _chip(d, (x0, y0, x1, y1), accent, "ATTITUDE SPHERE")
    cx, cy = (x0 + x1) / 2.0, y0 + 152 * sy
    R = 104 * sy
    f = frames[fi]
    q = np.asarray(f["quat"], float)
    qt = np.asarray(f["target_quat"], float)
    qrel = _quat_mul_np(np.array((qt[0], -qt[1], -qt[2], -qt[3])), q)
    fwd = np.array((0.45, -0.80, 0.40))
    fwd /= np.linalg.norm(fwd)
    right = np.cross((0.0, 0.0, 1.0), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)

    def proj(v):
        v = np.asarray(v, float)
        return (cx + R * float(np.dot(v, right)),
                cy - R * float(np.dot(v, up)), float(np.dot(v, fwd)))

    f8 = _mono(max(7, int(8 * sy)), bold=False)
    d.ellipse((cx - R, cy - R, cx + R, cy + R), fill=(12, 18, 26, 46),
              outline=_rgba(accent, 100), width=1)
    target_col = (239, 198, 116)

    def great(axis):
        pts = []
        for i in range(73):
            a = 2.0 * math.pi * i / 72.0
            v = ((math.cos(a), math.sin(a), 0.0) if axis == "xy"
                 else (math.cos(a), 0.0, math.sin(a)) if axis == "xz"
                 else (0.0, math.cos(a), math.sin(a)))
            pts.append(proj(v))
        for a, b in zip(pts[:-1], pts[1:]):
            if a[2] > 0.0 and b[2] > 0.0:
                d.line((a[0], a[1], b[0], b[1]), fill=_rgba(target_col, 115), width=1)
            elif a[2] < 0.0 and b[2] < 0.0:
                _dash_line(d, a[:2], b[:2], _rgba(target_col, 48), width=1, dash=2, gap=3)

    for ax in ("xy", "xz", "yz"):
        great(ax)
    # target triad (yellow) and live body triad (red/green/blue)
    for i, (name, col) in enumerate((("X", target_col), ("Y", target_col), ("Z", target_col))):
        e = np.eye(3)[i]
        px, py, depth = proj(e * 1.14)
        d.line((cx, cy, px, py), fill=_rgba(col, 200 if depth > 0 else 70), width=2)
        lx, ly, _ = proj(e * 1.30)
        d.text((lx, ly), name, font=f8, fill=_rgba(col, 190), anchor="mm")
    for i, (name, col) in enumerate((("x", LEGEND_COLORS["red"]),
                                     ("y", LEGEND_COLORS["green"]),
                                     ("z", LEGEND_COLORS["blue"]))):
        v = _quat_rot(qrel, np.eye(3)[i])
        px, py, depth = proj(v * 1.08)
        alpha = 245 if depth > 0 else 110
        d.line((cx, cy, px, py), fill=_rgba(col, alpha), width=2)
        lx, ly, _ = proj(v * 1.24)
        d.text((lx, ly), name, font=f8, fill=_rgba(col, alpha), anchor="mm")
    # attitude-error arc between the target Z and the live body z
    zb = _quat_rot(qrel, np.array((0.0, 0.0, 1.0)))
    arc = []
    for i in range(17):
        t = i / 16.0
        v = (1.0 - t) * np.array((0.0, 0.0, 1.0)) + t * zb
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            arc.append(proj(v / n * 1.02))
    for a, b in zip(arc[:-1], arc[1:]):
        _dash_line(d, a[:2], b[:2], _rgba(accent, 170), width=1, dash=3, gap=2)
    # angular-rate vector, normalised to 0.5 rad/s at the limb
    omega = f.get("omega")
    rate_deg = 0.0
    if omega is not None:
        w = _quat_rot(qrel, np.asarray(omega, float))
        wn = float(np.linalg.norm(w))
        rate_deg = math.degrees(wn)
        if wn > 1e-9:
            tip = proj(w / wn * min(1.0, wn / 0.5))
            d.line((cx, cy, tip[0], tip[1]), fill=_rgba(ACCENT_COLOR, 230), width=2)
            d.ellipse((tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3),
                      outline=_rgba(ACCENT_COLOR, 230))
            d.text((tip[0] + 6, tip[1] - 6), "w", font=f8, fill=_rgba(ACCENT_COLOR, 230))
    err = math.degrees(_quat_angle(q, qt))
    d.text((x0 + 13, y1 - 42), f"ERR {err:5.1f} deg", font=f8, fill=_rgba(accent, 220))
    d.text((x1 - 13, y1 - 42), f"|w| {rate_deg:4.1f} deg/s", font=f8,
           fill=DIM_COLOR, anchor="ra")
    d.text((x0 + 13, y1 - 24), "yellow target · RGB body · cyan rate",
           font=_mono(max(6, int(7 * sy)), bold=False), fill=FAINT_COLOR)


def _plan_context(task, frames, extras):
    """Fixed per-episode map frame for the plan view (centroid + 1.25x extent).

    Computed once from the full frame list so the map never jitters or
    re-scales mid-replay. Map east/north are world x/y; the CWH tasks map
    east=along-track, north=radial with the station at the origin; transfer
    positions are converted from 1000-km units to km.
    """
    if task == "satellite":
        return None  # attitude task: the hero ADI replaces the map
    pos = np.array([f["pos"] for f in frames], dtype=float)
    ctx = {"units": "m", "wps": None, "tgt": None, "earth": 0.0,
           "goal_ring": 0.0, "rings": [], "occluders": ()}
    pts = [np.zeros((1, 2))]
    if task in ("docking", "rendezvous"):
        ctx["path"] = np.stack((pos[:, 1], pos[:, 0]), axis=-1)
        # typed station-centred rings: goal / keep-out / 50+100 m references
        ctx["rings"] = [(float(extras["pos_goal"]), "goal")]
        if extras.get("keep_out"):
            ctx["rings"].append((float(extras["keep_out"]), "keep"))
        ctx["rings"] += [(50.0, "ref"), (100.0, "ref")]
        pts.append(np.array(((100.0, 0.0), (-100.0, 0.0), (0.0, 100.0), (0.0, -100.0))))
    elif task == "transfer":
        ctx["path"] = pos[:, :2] * 1000.0
        ctx["units"] = "km"
        ctx["earth"] = 6371.0
        ctx["goal_ring"] = float(extras["orbit"]["ra0"]) * 1e-3  # m -> km
    else:
        ctx["path"] = pos[:, :2].copy()
    pts.append(ctx["path"])
    if task in ("quadx", "fixedwing", "quadplane"):
        ctx["wps"] = np.asarray(extras["targets"], dtype=float)[:, :2]
        pts.append(ctx["wps"])
    if task in ("quadx_track", "missile"):
        ctx["tgt"] = np.array([f["tgt"] for f in frames], dtype=float)[:, :2]
        pts.append(ctx["tgt"])
    if task == "quadx_track":
        ctx["occluders"] = tuple(
            (float(x), float(y), float(r)) for x, y, r, _ in extras["occluders"]
        )
        pts.append(
            np.array([(x, y) for x, y, _ in ctx["occluders"]], dtype=float).reshape(-1, 2)
        )
    if ctx["goal_ring"]:
        r = ctx["goal_ring"]
        pts.append(np.array(((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r))))
    allp = np.concatenate(pts)
    c = allp.mean(axis=0)
    ctx["center"] = c
    ctx["half"] = max(float(np.abs(allp - c).max()), 1e-6) * 1.25
    return ctx


def _draw_plan(d, ov, task, accent, ctx, frames, fi, yaw):
    """North-up 2D plan view: grid, scale bar, north arrow, route and trail."""
    sx, sy = MAIN_RES[0] / 1440.0, MAIN_RES[1] / 810.0
    px0, py0, px1, py1 = int(944 * sx), int(88 * sy), int(1404 * sx), int(408 * sy)
    _chip(d, (px0, py0, px1, py1), accent, None)
    f8 = _mono(max(7, int(8 * sy)), bold=False)
    d.text((px0 + 13, py0 + 9), "PLAN VIEW", font=f8, fill=FAINT_COLOR)
    mx0, my0 = px0 + int(16 * sx), py0 + int(30 * sy)
    mx1, my1 = px1 - int(16 * sx), py1 - int(46 * sy)
    mcxx, mcyy = (mx0 + mx1) / 2.0, (my0 + my1) / 2.0
    scale = min(mx1 - mx0, my1 - my0) / 2.0 / ctx["half"]
    c = ctx["center"]
    path = ctx["path"]
    cur = path[fi]

    def mp(pt):
        return (mcxx + (pt[0] - c[0]) * scale, mcyy - (pt[1] - c[1]) * scale)

    if task in ("docking", "rendezvous"):
        hdr = f"AT {cur[0]:+.1f}  RAD {cur[1]:+.1f} M"
    else:
        hdr = f"E {cur[0]:+.1f}  N {cur[1]:+.1f} {'KM' if ctx['units'] == 'km' else 'M'}"
    d.text((px1 - 13, py0 + 9), hdr, font=f8, fill=_rgba(accent, 215), anchor="ra")
    # faint grid at 1/2/5 x 10^n steps
    raw = 2.0 * ctx["half"] / 6.0
    p10 = 10.0 ** math.floor(math.log10(max(raw, 1e-12)))
    step = next(m * p10 for m in (1.0, 2.0, 5.0, 10.0) if m * p10 >= raw)
    xw0, xw1 = c[0] - (mcxx - mx0) / scale, c[0] + (mx1 - mcxx) / scale
    yw0, yw1 = c[1] - (my1 - mcyy) / scale, c[1] + (mcyy - my0) / scale
    gx = math.ceil(xw0 / step) * step
    while gx <= xw1:
        x = mp((gx, 0.0))[0]
        d.line((x, my0, x, my1), fill=(255, 255, 255, 11), width=1)
        gx += step
    gy = math.ceil(yw0 / step) * step
    while gy <= yw1:
        y = mp((0.0, gy))[1]
        d.line((mx0, y, mx1, y), fill=(255, 255, 255, 11), width=1)
        gy += step
    origin = mp((0.0, 0.0))
    if ctx["earth"]:  # transfer: Earth disc + dashed target circle
        rr = ctx["earth"] * scale
        d.ellipse((origin[0] - rr, origin[1] - rr, origin[0] + rr, origin[1] + rr),
                  fill=(28, 62, 116, 150), outline=_rgba((122, 162, 247), 120))
    if ctx["goal_ring"]:
        _dash_circle(d, origin, ctx["goal_ring"] * scale, _rgba(accent, 90))
    for r, kind in ctx["rings"]:
        rr = r * scale
        box = (origin[0] - rr, origin[1] - rr, origin[0] + rr, origin[1] + rr)
        lx, ly = origin[0], origin[1] - rr - 7  # micro label on the ring top
        if kind == "goal":
            d.ellipse(box, outline=_rgba(MINT_COLOR, 210), width=1)
            d.text((lx, ly), f"GOAL {r:g}m", font=f8, fill=_rgba(MINT_COLOR, 180), anchor="mm")
        elif kind == "keep":
            _dash_circle(d, origin, rr, _rgba(BAD_COLOR, 160))
            d.text((lx, ly), f"KEEP {r:g}m", font=f8, fill=_rgba(BAD_COLOR, 160), anchor="mm")
        else:
            _dash_circle(d, origin, rr, (255, 255, 255, 44))
            d.text((lx, ly), f"{r:g}m", font=f8, fill=FAINT_COLOR, anchor="mm")
    for ox, oy, orad in ctx["occluders"]:
        rr = max(2.0, orad * scale)
        d.ellipse((mp((ox, oy))[0] - rr, mp((ox, oy))[1] - rr,
                   mp((ox, oy))[0] + rr, mp((ox, oy))[1] + rr),
                  outline=(255, 255, 255, 40))
    # planned route (dim dashed) and remaining legs (faint)
    f = frames[fi]
    if ctx["wps"] is not None:
        _dash_poly(d, [mp(p) for p in ctx["wps"]], _rgba(DIM_COLOR, 90), 1, 5, 4)
        k = int(f.get("wpt", 0))
        _dash_poly(d, [mp(cur)] + [mp(p) for p in ctx["wps"][k:]],
                   _rgba(accent, 70), 1, 4, 5)
    if ctx["tgt"] is not None:
        _dash_poly(d, [mp(p) for p in ctx["tgt"]], _rgba(BAD_COLOR, 80), 1, 5, 4)
    if task in ("rocket", "docking", "rendezvous"):
        _dash_line(d, mp(cur), origin, _rgba(accent, 60), 1, 4, 5)
    # flown trail (accent solid; CWH gets a soft glow underlay)
    stride = max(1, (fi + 1) // 600)
    trail_pts = [mp(p) for p in path[: fi + 1 : stride]]
    cwh = task in ("docking", "rendezvous")
    if len(trail_pts) > 1:
        if cwh:
            d.line(trail_pts, fill=_rgba(accent, 60), width=4, joint="curve")
        d.line(trail_pts, fill=_rgba(accent, 235 if cwh else 210), width=2, joint="curve")
    # start square, final-goal diamond, active-target ring, missile link
    s0 = mp(path[0])
    d.rectangle((s0[0] - 4, s0[1] - 4, s0[0] + 4, s0[1] + 4),
                outline=(238, 244, 250, 170), width=1)
    goal = None
    if ctx["wps"] is not None:
        goal = ctx["wps"][-1]
    elif ctx["tgt"] is not None:
        goal = ctx["tgt"][-1]
    elif task in ("rocket", "docking", "rendezvous"):
        goal = (0.0, 0.0)
    if goal is not None:
        g = mp(goal)
        d.polygon([(g[0], g[1] - 6), (g[0] + 6, g[1]), (g[0], g[1] + 6), (g[0] - 6, g[1])],
                  outline=_rgba(MINT_COLOR, 200))
    active = None
    if task in ("quadx", "fixedwing", "quadplane"):
        active = np.asarray(f["goal"], float)[:2]
    elif task in ("quadx_track", "missile"):
        active = np.asarray(f["tgt"], float)[:2]
    elif task == "rocket":
        active = (0.0, 0.0)
    if active is not None:
        a = mp(active)
        col = BAD_COLOR if task in ("quadx_track", "missile") else AMBER_COLOR
        d.ellipse((a[0] - 7, a[1] - 7, a[0] + 7, a[1] + 7), outline=_rgba(col, 220), width=2)
    if task == "missile":
        d.line((mp(cur), mp(np.asarray(f["tgt"], float)[:2])),
               fill=_rgba(AMBER_COLOR, 110), width=1)
    if task == "rocket":
        d.line((origin[0] - 7, origin[1], origin[0] + 7, origin[1]),
               fill=_rgba(accent, 160), width=1)
        d.line((origin[0], origin[1] - 7, origin[0], origin[1] + 7),
               fill=_rgba(accent, 160), width=1)
    # ownship heading triangle; the quat-less tasks steer by velocity / facing
    if task in ("docking", "rendezvous"):
        p = np.asarray(f["pos"], float)
        v = np.array((-p[1], -p[0]))  # faces the station; plan coords swapped
    elif task == "transfer":
        i0, i1 = max(0, fi - 1), min(len(frames) - 1, fi + 1)
        v = path[i1] - path[i0]
    else:
        v = np.array((math.cos(yaw), math.sin(yaw)))
    if np.linalg.norm(v) < 1e-9:
        v = np.array((1.0, 0.0))
    v = v / np.linalg.norm(v)
    dx, dy = float(v[0]), -float(v[1])  # screen direction (north up)
    cp = mp(cur)
    tn, tb = (13.0, 7.0) if cwh else (9.0, 5.0)
    bx_, by_ = cp[0] - dx * tb, cp[1] - dy * tb
    d.polygon([(cp[0] + dx * tn, cp[1] + dy * tn),
               (bx_ - dy * tb, by_ + dx * tb), (bx_ + dy * tb, by_ - dx * tb)],
              fill=_rgba(accent, 240))
    # north arrow + scale bar
    nx, ny = mx1 - 8, my0 + 20
    d.line((nx, ny, nx, ny - 12), fill=_rgba(accent, 200), width=1)
    d.polygon([(nx - 3, ny - 8), (nx + 3, ny - 8), (nx, ny - 14)], fill=_rgba(accent, 200))
    d.text((nx, ny + 3), "N", font=f8, fill=DIM_COLOR, anchor="ma")
    raw_len = (mx1 - mx0) * 0.30 / scale
    p10 = 10.0 ** math.floor(math.log10(max(raw_len, 1e-12)))
    L = max((m * p10 for m in (1.0, 2.0, 5.0) if m * p10 <= raw_len), default=p10)
    bx, by = mx0 + 2, py1 - int(24 * sy)
    d.line((bx, by, bx + L * scale, by), fill=(238, 244, 250, 190), width=2)
    d.line((bx, by - 3, bx, by + 3), fill=(238, 244, 250, 190), width=1)
    d.line((bx + L * scale, by - 3, bx + L * scale, by + 3), fill=(238, 244, 250, 190), width=1)
    if ctx["units"] == "km":
        lab = f"{L:g} km"
    else:
        lab = f"{L / 1000.0:g} km" if L >= 1000.0 else f"{L:g} m"
    d.text((bx + L * scale + 7, by), lab, font=f8, fill=DIM_COLOR, anchor="lm")
    if cwh:  # terminal-phase detail lives in the chaser-centred zoom inset
        _draw_plan_zoom(d, ov, accent, ctx, frames, fi, (px0, py0, py1))


def _draw_plan_zoom(d, ov, accent, ctx, frames, fi, panel):
    """Chaser-centred zoom sub-inset, bottom-left of the CWH plan panel.

    Window = max(20 m, 3.5x the current chaser-station span), recentred on the
    chaser every frame; the outer map keeps its fixed episode extent. Drawing
    into a private image gives hard clipping at the inset border for free.
    """
    from PIL import Image, ImageDraw

    s = MAIN_RES[1] / 810.0
    px0, _, py1 = panel
    zw, zh = int(150 * s), int(110 * s)
    zx0 = px0 + int(16 * s)
    zy0 = py1 - int(46 * s) - zh  # bottom flush with the map window
    path = ctx["path"]
    cur = path[fi]
    rng = float(np.linalg.norm(cur))  # station sits at the plan origin
    half = max(20.0, 3.5 * rng) / 2.0
    zscale = (min(zw, zh - 14) - 10) / 2.0 / half
    zcx, zcy = zw / 2.0, 7.0 + (zh - 14) / 2.0  # 14 px header strip

    def zm(pt):
        return (zcx + (pt[0] - cur[0]) * zscale, zcy - (pt[1] - cur[1]) * zscale)

    zim = Image.new("RGBA", (zw, zh), (0, 0, 0, 0))
    zd = ImageDraw.Draw(zim)
    zd.rectangle((0, 0, zw - 1, zh - 1), fill=(10, 16, 22, 215))
    o = zm((0.0, 0.0))
    for r, kind in ctx["rings"]:  # same station-centred rings, zoomed
        rr = r * zscale
        box = (o[0] - rr, o[1] - rr, o[0] + rr, o[1] + rr)
        if kind == "goal":
            zd.ellipse(box, outline=_rgba(MINT_COLOR, 215), width=1)
        elif kind == "keep":
            _dash_circle(zd, o, rr, _rgba(BAD_COLOR, 165))
        else:
            _dash_circle(zd, o, rr, (255, 255, 255, 48))
    zd.line([zm(p) for p in path[: fi + 1]], fill=_rgba(accent, 70), width=4)
    zd.line([zm(p) for p in path[: fi + 1]], fill=_rgba(accent, 240), width=2)
    zd.rectangle((o[0] - 3, o[1] - 3, o[0] + 3, o[1] + 3),
                 outline=(238, 244, 250, 190), width=1)
    p = np.asarray(frames[fi]["pos"], float)
    v = np.array((-p[1], -p[0]))  # chaser faces the station; plan axes swapped
    if np.linalg.norm(v) < 1e-9:
        v = np.array((1.0, 0.0))
    v = v / np.linalg.norm(v)
    dx, dy = float(v[0]), -float(v[1])
    cp = zm(cur)
    bx_, by_ = cp[0] - dx * 7, cp[1] - dy * 7
    zd.polygon([(cp[0] + dx * 12, cp[1] + dy * 12), (bx_ - dy * 7, by_ + dx * 7),
                (bx_ + dy * 7, by_ - dx * 7)], fill=_rgba(accent, 245))
    ov.alpha_composite(zim, (zx0, zy0))
    f7 = _mono(max(7, int(8 * s)), bold=False)
    d.rectangle((zx0, zy0, zx0 + zw - 1, zy0 + zh - 1), outline=_rgba(accent, 90), width=1)
    d.text((zx0 + 8, zy0 + 4), "ZOOM", font=f7, fill=FAINT_COLOR)
    d.text((zx0 + zw - 8, zy0 + 4), f"R {rng:.1f} m", font=f7,
           fill=_rgba(accent, 215), anchor="ra")


def _fmt_range(m):
    m = float(m)
    return f"{m / 1000.0:.2f}km" if m >= 1000.0 else f"{m:.1f}m"


def _project(pt):
    """World point -> (sx, sy, fwd depth) through the last main camera.

    _LAST_CAM is refreshed by _main_render after every slow-pan set_pose, so
    the pose here is always the one the current panorama was rendered with.
    """
    if _LAST_CAM is None:
        return None
    cpos, look, up, fov, res = _LAST_CAM
    fwd = np.asarray(look, float) - np.asarray(cpos, float)
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        return None
    fwd = fwd / n
    right = np.cross(fwd, np.asarray(up, float))
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        return None
    right = right / rn
    up2 = np.cross(right, fwd)
    rel = np.asarray(pt, float) - np.asarray(cpos, float)
    z = float(np.dot(rel, fwd))
    if z < 1e-3:
        return None  # behind the camera
    tv = math.tan(math.radians(fov) / 2.0)
    w, h = res
    th = tv * w / h
    sx = (0.5 + float(np.dot(rel, right)) / (z * th) * 0.5) * w
    sy = (0.5 - float(np.dot(rel, up2)) / (z * tv) * 0.5) * h
    return sx, sy, z


def _occluded(sx, sy, dist):
    """Depth-buffer test: True when the sampled surface is >10 % closer than
    the tag anchor, i.e. scene geometry covers the target (dashed bracket)."""
    dep = _LAST_DEPTH
    if dep is None:
        return False
    a = np.asarray(dep, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    h, w = a.shape
    ix, iy = int(sx), int(sy)
    if not (0 <= ix < w and 0 <= iy < h):
        return False
    dv = float(a[iy, ix])
    if not math.isfinite(dv) or dv >= float(np.nanmax(a)) * 0.9995:
        return False  # sky pixel: nothing in front
    return dv < 0.9 * dist


def _tag(d, sx, sy, color, label, dim):
    """Corner-bracket box + anchor dot + leadered mono label at a screen point."""
    s = MAIN_RES[1] / 810.0
    b, arm = int(22 * s), int(9 * s)
    col = _rgba(color, 110 if dim else 235)
    for cx, cy, dx, dy in ((sx - b, sy - b, 1, 1), (sx + b, sy - b, -1, 1),
                           (sx - b, sy + b, 1, -1), (sx + b, sy + b, -1, -1)):
        if dim:
            _dash_line(d, (cx, cy), (cx + dx * arm, cy), col, 2, 3, 3)
            _dash_line(d, (cx, cy), (cx, cy + dy * arm), col, 2, 3, 3)
        else:
            d.line((cx, cy, cx + dx * arm, cy), fill=col, width=2)
            d.line((cx, cy, cx, cy + dy * arm), fill=col, width=2)
    d.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=col)
    f8 = _mono(max(7, int(8 * s)))
    tw = d.textlength(label, font=f8)
    lx = sx + b + 10
    edge = sx + b
    if lx + tw + 6 > MAIN_RES[0] - 40:
        lx = sx - b - 10 - tw - 6  # flip to the left of the box at the edge
        edge = sx - b
    ly = max(40.0, sy - b - 13)
    d.line((edge, sy - b, lx + (0 if lx > sx else tw + 6), ly + 10),
           fill=_rgba(color, 120), width=1)
    d.rectangle((lx - 2, ly, lx + tw + 6, ly + 12), fill=(10, 16, 22, 160))
    d.text((lx + 2, ly + 2), label, font=f8, fill=col)


def _draw_target_tags(d, task, accent, frames, fi):
    """3D target brackets in the main panorama (satellite tags nothing)."""
    W, H = MAIN_RES
    f = frames[fi]
    p = np.asarray(f["pos"], float)
    tags = []
    if task == "quadx":
        tags.append((np.asarray(f["goal"], float),
                     f"TGT {_fmt_range(np.linalg.norm(f['goal'] - p))}", AMBER_COLOR))
    elif task == "quadx_track":
        tags.append((np.asarray(f["tgt"], float),
                     f"TGT {_fmt_range(np.linalg.norm(f['tgt'] - p))}", AMBER_COLOR))
    elif task in ("fixedwing", "quadplane"):
        tags.append((np.asarray(f["goal"], float),
                     f"WPT{int(f.get('wpt', 0)) + 1} "
                     f"{_fmt_range(np.linalg.norm(f['goal'] - p))}", AMBER_COLOR))
    elif task == "rocket":
        if p[2] > 30.0:  # near the pad the bracket would sit on the airframe
            tags.append((np.zeros(3), f"PAD {_fmt_range(np.linalg.norm(p))}", AMBER_COLOR))
    elif task == "missile":
        tags.append((np.asarray(f["tgt"], float),
                     f"TGT {_fmt_range(np.linalg.norm(f['tgt'] - p))}", AMBER_COLOR))
    elif task == "transfer":
        tags.append((p, f"SAT {np.linalg.norm(p) * 1000.0:.0f} km", accent))
    elif task in ("docking", "rendezvous"):
        rng = float(np.linalg.norm(p))
        tags.append((p, f"CHASER {_fmt_range(rng)}", accent))
        tags.append((np.zeros(3), "STATION", accent))
    drawn = []
    for world, label, color in tags:
        pr = _project(world)
        if pr is None:
            continue
        sx, sy, z = pr
        if not (40 <= sx <= W - 40 and 40 <= sy <= H - 40):
            continue  # never leave a half-drawn box on the frame edge
        if any((sx - px) ** 2 + (sy - py) ** 2 < 30.0 ** 2 for px, py in drawn):
            continue  # co-located brackets (e.g. docked chaser/station) de-dupe
        drawn.append((sx, sy))
        _tag(d, sx, sy, color, label, _occluded(sx, sy, z))


def _gcs_overlay(canvas, task, gcs):
    """Mission-planner layer: ADI + heading tape, stage strip, plan view and
    3D target tags, alpha-composited over the finished frame."""
    from PIL import Image, ImageDraw

    W, H = canvas.size
    accent = TASK_META.get(task, ("", "", ACCENT_COLOR, True))[2]
    frames, fi = gcs["frames"], gcs["fi"]
    roll, pitch, yaw = _attitude(task, frames, fi)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    hero = task == "satellite"
    _draw_adi(d, ov, accent, roll, pitch, hero)
    _draw_heading(d, accent, yaw, hero)
    _draw_stages(d, task, accent, gcs["telem"], frames, gcs["extras"], fi)
    if task == "satellite":
        _att_sphere(d, ov, accent, frames, fi)
    if gcs["plan"] is not None:
        _draw_plan(d, ov, task, accent, gcs["plan"], frames, fi, yaw)
    _draw_target_tags(d, task, accent, frames, fi)
    return Image.alpha_composite(canvas, ov)



def build_scene(task, extras, first, close_res=CLOSE_RES):
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
        vis_options=gs.options.VisOptions(
            background_color=SKY_COLOR if outdoor else (0.006, 0.011, 0.020),
            ambient_light=(0.36, 0.42, 0.50) if outdoor else (0.38, 0.42, 0.50),
            shadow=True,
        ),
    )
    ents = {}
    p0 = first["pos"]
    if task == "quadx":
        # PyFlyt-style visualization (mirrors PyFlyt 0.29 quadx_envs): grass
        # ground, the cf2x URDF (enlarged 6x so the ~9 cm airframe is
        # readable at dome scale), an opaque solid green sphere at the active
        # waypoint (PyFlyt's waypoint_handler scaling, enlarged to stay visible),
        # and a fixed global camera. The close-up inset keeps the true camera
        # geometry of PyFlyt's yaw-30 / pitch-30 debug view.
        _ground(scene, color=GRASS_GROUND)
        _grid(scene, 12.0, 1.0, color=GRASS_GRID)
        ents["body"] = _model(scene, "crazyflie", _build_quadrotor(), scale=9.0)
        _add_quadrotor_motion(scene, ents, 9.0)
        ents["goal"] = _marker(scene, 0.18, (0.0, 1.0, 0.0, 1.0), extras["target"])
        gx, gy, gz = (float(v) for v in extras["target"])
        _ring(scene, 0.55, (0.20, 1.0, 0.55, 0.95), (gx, gy, gz), n=48, tube=0.028)
        _beam(scene, (gx, gy, 0.0), gz, (0.20, 1.0, 0.55, 0.30), 0.018)
        cam = scene.add_camera(
            res=MAIN_RES,
            pos=tuple(0.5 * (np.asarray(p0, float) + np.asarray(extras["target"], float))
                      + np.array((4.6, -6.6, 4.2))),
            lookat=tuple(0.5 * (np.asarray(p0, float) + np.asarray(extras["target"], float))
                         + np.array((0.0, 0.0, 1.3))),
            fov=_hfov_match(46, MAIN_RES), far=2000
        )
        ents["close_cam"] = scene.add_camera(
            res=close_res, fov=_hfov_match(45, close_res), far=5000
        )
    elif task == "quadx_track":
        # Reuse the PyFlyt cf2x model (enlarged 8x for readability). Only the
        # target, true LOS and the environment's geometric occluders are added.
        _ground(scene, color=GRASS_GROUND)
        _grid(scene, 24.0, 2.0, color=GRASS_GRID)
        ents["body"] = _model(scene, "crazyflie", _build_quadrotor(), scale=14.0)
        _add_quadrotor_motion(scene, ents, 14.0)
        ents["target"] = _marker(scene, 0.8, (0.9, 0.15, 0.1, 1.0), first["tgt"])
        tx, ty, tz = (float(v) for v in first["tgt"])
        _ring(scene, 1.6, (1.0, 0.22, 0.20, 0.95), (tx, ty, tz), n=56, tube=0.075)
        _beam(scene, (tx, ty, 0.0), max(4.0, tz), (1.0, 0.22, 0.20, 0.30), 0.055)
        ents["occluders"] = []
        for x, y, radius, height in extras["occluders"]:
            ent = _cyl(
                scene, radius, height, (0.105, 0.135, 0.155, 1.0), (x, y, height / 2.0)
            )
            ents["occluders"].append(ent)
        ents["los"] = [_marker(scene, 0.05, (1.0, 1.0, 1.0, 0.65)) for _ in range(24)]
        ents["trail"] = _trail(scene, 0.06, (0.2, 0.4, 0.9, 0.6))
        ents["target_trail"] = _trail(scene, 0.06, (0.9, 0.15, 0.1, 0.7))
        mid = 0.5 * (np.asarray(p0, float) + np.asarray(first["tgt"], float))
        cam = scene.add_camera(
            res=MAIN_RES,
            pos=tuple(mid + np.array((14.5, -15.5, 9.5))),
            lookat=tuple(mid + np.array((0.0, 0.0, 2.5))),
            fov=_hfov_match(52, MAIN_RES), far=3000
        )
        ents["close_cam"] = scene.add_camera(
            res=close_res, fov=_hfov_match(45, close_res), far=5000
        )
    elif task == "fixedwing":
        _ground(scene, color=GRASS_GROUND)
        _grid(scene, 220.0, 20.0, color=GRASS_GRID)
        ents["body"] = _model(scene, "cessna", _build_fixedwing(), scale=10.0)
        _add_fixedwing_motion(scene, ents, 10.0)
        ents["wps"] = [_marker(scene, 1.2, (0.92, 0.18, 0.14, 1.0), t) for t in extras["targets"]]
        for t in extras["targets"]:
            wx, wy, wz = (float(v) for v in t)
            # slim gate ring: the previous 8 m / 0.55 m tube read as a fat
            # donut whenever the close-up flew past a waypoint
            _ring(scene, 3.2, (0.95, 0.24, 0.18, 1.0), (wx, wy, wz), n=64, tube=0.085)
            _beam(scene, (wx, wy, 0.0), wz, (0.95, 0.24, 0.18, 0.22), 0.05)
            _ring(scene, 3.2, (0.55, 0.10, 0.08, 1.0), (wx, wy, 0.03), n=64, tube=0.05)
        ents["trail"] = _trail(scene, 0.35, (0.2, 0.4, 0.9, 0.6))
        # fixed global camera centred on the waypoint cluster; the panorama is
        # its own zone in the composed frame, so nothing is panned away
        cam = scene.add_camera(
            res=MAIN_RES, pos=(85, -85, 70), lookat=(0, 0, 30),
            fov=_hfov_match(55, MAIN_RES), far=5000
        )
        # slow pan: the camera position stays fixed (the waypoint circuit must
        # never leave frame) but the look-at point eases toward the aircraft so
        # it cannot drift behind the HUD panel / close-up inset.
        ents["cam_base"] = (
            np.array((85.0, -85.0, 70.0)), np.array((0.0, 0.0, 30.0))
        )
        ents["close_cam"] = scene.add_camera(
            res=close_res, fov=_hfov_match(68, close_res), far=5000
        )
    elif task == "quadplane":
        # Same PyFlyt waypoint scene convention as Fixedwing, with the native
        # quadplane model (enlarged 10x); no synthetic tilt/mode animation is
        # introduced.
        _ground(scene, color=GRASS_GROUND)
        _grid(scene, 220.0, 20.0, color=GRASS_GRID)
        ents["body"] = _model(scene, "quadplane", _build_quadplane(), scale=10.0)
        _add_quadplane_motion(scene, ents, 10.0)
        ents["wps"] = [_marker(scene, 1.2, (0.92, 0.18, 0.14, 1.0), t) for t in extras["targets"]]
        for t in extras["targets"]:
            wx, wy, wz = (float(v) for v in t)
            # slim gate ring: the previous 8 m / 0.55 m tube read as a fat
            # donut whenever the close-up flew past a waypoint
            _ring(scene, 3.2, (0.95, 0.24, 0.18, 1.0), (wx, wy, wz), n=64, tube=0.085)
            _beam(scene, (wx, wy, 0.0), wz, (0.95, 0.24, 0.18, 0.22), 0.05)
            _ring(scene, 3.2, (0.55, 0.10, 0.08, 1.0), (wx, wy, 0.03), n=64, tube=0.05)
        ents["trail"] = _trail(scene, 0.35, (0.2, 0.4, 0.9, 0.6))
        # same fixed global framing as Fixedwing
        cam = scene.add_camera(
            res=MAIN_RES, pos=(100, -100, 74), lookat=(0, 0, 32),
            fov=_hfov_match(55, MAIN_RES), far=5000
        )
        ents["cam_base"] = (
            np.array((100.0, -100.0, 74.0)), np.array((0.0, 0.0, 32.0))
        )
        ents["close_cam"] = scene.add_camera(
            res=close_res, fov=_hfov_match(68, close_res), far=5000
        )
    elif task == "rocket":
        _ground(scene, color=GRASS_GROUND)
        # no ground grid: at this near-ground camera the 25 m lines crowd
        # together towards the horizon into moire scan lines
        _urdf(scene, MODELS / "landing_pad.urdf", pos=(0, 0, 0.1))
        _cyl(scene, 30.0, 0.10, (0.09, 0.18, 0.22, 1.0), (0, 0, 0.06))
        for rr, tt in ((6.0, 1.0), (10.0, 0.8), (15.0, 0.65), (22.0, 0.55), (30.0, 0.45)):
            _ring(scene, rr, (0.45, 0.90, 1.0, 1.0), (0, 0, 0.14), n=80, tube=tt)
        # pad cross + dashed altitude ladder: gives the empty descent corridor
        # a readable structure at 700-900 m camera range. The ladder is offset
        # to the side of the descent axis so the 16x airframe never intersects
        # it (on-axis bars showed up as cyan dashes across the close-up body).
        _box(scene, (56.0, 0.35, 0.04), (0.35, 0.75, 0.88, 1.0), (0.0, 0.0, 0.15))
        _box(scene, (0.35, 56.0, 0.04), (0.35, 0.75, 0.88, 1.0), (0.0, 0.0, 0.15))
        for z in range(15, 460, 15):
            ents.setdefault("ladder", []).append(
                _box(scene, (7.5, 2.2, 0.5), (0.45, 0.86, 1.0, 1.0), (-120.0, 0.0, float(z)))
            )
        # 5.4 m airframe at ~450 m altitude would be a few pixels: enlarged 16x.
        # The display scale is anchored at the airframe's lowest point (computed
        # after scene.build below), otherwise the 16x body would sink ~39 m into
        # the pad as it lands (the true base only clears the pad by centimetres).
        ents["body"] = _model(scene, "falcon9", _build_rocket(), scale=16.0)
        _add_rocket_motion(scene, ents, 16.0)
        # emissive plume: white-hot spine, irregular silhouette, ember tail
        flame_t = np.linspace(0.0, 1.0, 30)
        flame_r = tuple(
            (1.6 + 3.8 * (t / 0.32) ** 0.8) if t < 0.32
            else (5.4 * (1.0 - (t - 0.32) / 0.68) ** 0.95 + 0.3)
            for t in flame_t
        )
        ents["flame"] = _flame_chain(
            scene, flame_r, steps=[0.50 * r for r in flame_r]
        )
        ents["flame_len"] = sum(ents["flame"]["steps"])
        ents["flame_nozzle"] = _marker(
            scene, 1.6, (0.0, 0.0, 0.0, 1.0), HIDDEN,
            emissive=(1.0, 0.86, 0.56), opacity=1.0,
        )
        # impingement pool: warm discs on the pad, revealed progressively as
        # the plume tail reaches the concrete (a real landing burn splashes)
        ents["plume_pad"] = [
            _cyl(scene, 3.5, 0.08, (0.0, 0.0, 0.0, 1.0), HIDDEN,
                 emissive=(1.0, 0.52, 0.16), opacity=0.50),
            _cyl(scene, 7.5, 0.08, (0.0, 0.0, 0.0, 1.0), HIDDEN,
                 emissive=(1.0, 0.40, 0.10), opacity=0.30),
            _cyl(scene, 12.5, 0.08, (0.0, 0.0, 0.0, 1.0), HIDDEN,
                 emissive=(0.95, 0.30, 0.06), opacity=0.18),
        ]
        # embers shed by the exhaust: a few tiny glowing sparks under the tail
        ents["embers"] = []
        for j in range(10):
            ang = 2.0 * math.pi * j / 10.0
            ents["embers"].append(
                {
                    "m": _marker(
                        scene, 0.16 + 0.10 * (j % 3), (0.0, 0.0, 0.0, 1.0),
                        HIDDEN, emissive=(1.0, 0.42, 0.07), opacity=1.0,
                    ),
                    "dx": math.cos(ang) * 1.9,
                    "dy": math.sin(ang) * 1.9,
                    "ph": j / 10.0,
                }
            )
        ents["trail"] = _trail(scene, 1.2, (0.25, 0.55, 0.95, 0.75))
        # descent corridor: pad in the lower third, apex near the top
        cam = scene.add_camera(
            res=MAIN_RES, pos=(520, -520, 300), lookat=(0, 0, 200),
            fov=_hfov_match(55, MAIN_RES), far=12000
        )
        ents["close_cam"] = scene.add_camera(
            res=close_res, fov=_hfov_match(14, close_res), far=12000
        )

    elif task == "satellite":
        # Recognizable bus: grey body + equipment compartments on the +/-z
        # faces (up/down), sensor head on +x (front), nozzle on -x (rear),
        # antenna on top, solar panels on +/-y with a light grid. Panels and
        # details are rigidly attached to the bus: each stores (entity,
        # local_pos, local_quat) and is posed per frame from the recorded
        # quaternion. All geometry here is display-only (collision=False,
        # scene never stepped) and only built when visualize.py is run.
        # Realistic bus: gold MLI body, radiators, dish, antenna and two solar
        # wings, all as one multi-material model in the bus frame. Only the
        # engineering overlays (body triad, target triad, CMG axes, ghost) stay
        # separate entities, then follow the recorded quaternion.
        ents["body"] = _model(scene, "comsat", _build_satellite())
        ents["attach"] = []

        def _attach(ent, pos, quat=(1.0, 0.0, 0.0, 0.0)):
            ents["attach"].append((ent, np.asarray(pos, float), tuple(quat)))

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
        # static star decoration (fixed positions, never moves); varied
        # magnitudes read as depth instead of a uniform dot shell
        rng = np.random.default_rng(3)
        for i, p in enumerate(rng.normal(size=(120, 3))):
            p = p / np.linalg.norm(p) * 22.0
            mag = float(rng.random())
            col = (1.0, 0.94, 0.84, 0.85) if i % 5 == 0 else (0.78, 0.86, 1.0, 0.8)
            _marker(scene, 0.055 + 0.14 * mag * mag, col, p)
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
            res=close_res, fov=_hfov_match(45, close_res), far=100
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
        pts0 = pf @ R.T * scale
        ents["orbit0"] = _polyline(  # true initial ellipse, pixel-thin line
            scene, pts0, 0.085, (0.24, 0.92, 0.44, 1.0), closed=True
        )
        # cyan target orbit: circle at the initial apogee radius
        # (ideal perigee-raising end state, same plane at reference thrust)
        circ = np.stack((o["ra0"] * np.cos(E), o["ra0"] * np.sin(E), np.zeros_like(E)), -1)
        pts_t = circ @ R.T * scale
        ents["orbit_t"] = _polyline(
            scene, pts_t, 0.055, (0.28, 0.86, 0.96, 1.0), closed=True
        )
        # perigee (red) / apogee (yellow) of the initial orbit
        ents["peri"] = _marker(scene, 0.30, (0.9, 0.2, 0.2, 1.0), R[:, 0] * a * (1.0 - e) * scale)
        ents["apo"] = _marker(scene, 0.30, (0.95, 0.85, 0.2, 1.0), R[:, 0] * -a * (1.0 + e) * scale)
        ents["earth"] = _marker(scene, 6.371, (0.15, 0.35, 0.75, 1.0))
        rng = np.random.default_rng(5)
        for i, sp in enumerate(rng.normal(size=(190, 3))):
            sp = sp / np.linalg.norm(sp) * 140.0
            _marker(scene, 0.09 + 0.16 * float(rng.random()) ** 2,
                    (0.80, 0.88, 1.0, 0.85), sp)
        ents["body"] = _model(scene, "smallsat", _build_smallsat())
        ents["trail"] = _trail(scene, 0.22, (1.0, 0.65, 0.1, 1.0))
        ents["plume"] = [
            _marker(scene, 0.10, (0.25, 0.85, 1.0, 0.85)) for _ in range(6)
        ]
        # Live osculating prediction: dotted violet ellipse + moving apsides.
        # This is recomputed from the recorded r/v state, not propagated with a
        # second simulation and not fitted to the already-flown trajectory.
        pred0 = _orbit_elements(first, o["mu"])
        ents["prediction"] = [
            _marker(scene, 0.15, (0.98, 0.30, 0.90, 1.0), pt)
            for pt in pred0["points"]
        ]
        ents["pred_peri"] = _marker(
            scene, 0.25, (0.95, 0.35, 0.78, 1.0), pred0["peri"]
        )
        ents["pred_apo"] = _marker(
            scene, 0.25, (0.68, 0.48, 0.98, 1.0), pred0["apo"]
        )
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
            res=close_res, fov=_hfov_match(45, close_res), far=300
        )
        # 2000 km scale bar (2 scene units), below the frame center
        view = -campos / np.linalg.norm(campos)
        right = np.cross(view, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, view)
        base = -up * 15.0 - right
        ents["scalebar"] = [
            _marker(scene, 0.28, (1.0, 1.0, 1.0, 0.95), base + right * i) for i in range(5)
        ]
    elif task == "missile":
        # km-scale air-intercept scene at ~4 km altitude: missile URDF (enlarged
        # 200x, otherwise a 3 m body is sub-pixel at engagement range), red
        # marker for the kinematic target aircraft (posed per frame from the
        # recorded target state), a yellow locator dot riding on the missile
        # (the narrow white airframe alone is easy to lose against the ground),
        # engagement trail, fixed global camera.
        # A far grassland plane plus the cloud sky background give the
        # engagement an unambiguous horizon instead of an empty dark frame.
        _ground(scene, 100000.0, color=GRASS_GROUND)
        ents["body"] = _model(scene, "amraam", _build_missile(), scale=200.0)
        _add_missile_motion(scene, ents, 200.0)
        ents["me"] = _marker(scene, 24.0, (1.0, 0.9, 0.25, 1.0), tuple(p0))
        ents["target"] = _marker(scene, 60.0, (0.9, 0.2, 0.2, 0.9), tuple(first["tgt"]))
        tg = np.asarray(first["tgt"], dtype=float)
        _ring(scene, 150.0, (1.0, 0.25, 0.22, 0.75), tuple(tg), n=72, tube=4.0)
        _beam(scene, (tg[0], tg[1], 0.0), float(tg[2]), (1.0, 0.25, 0.22, 0.16), 6.0)
        # plume down the -x body axis: white-hot near the nozzle, ember tail
        flame_t = np.linspace(0.0, 1.0, 40)
        flame_r = tuple(26.0 * (1.0 - t) ** 0.9 + 0.8 for t in flame_t)
        ents["flame"] = _flame_chain(
            scene, flame_r, steps=[0.30 * r for r in flame_r]
        )
        ents["flame_nozzle"] = _marker(
            scene, 10.0, (0.0, 0.0, 0.0, 1.0),
            HIDDEN, emissive=(1.0, 0.92, 0.74), opacity=1.0,
        )
        # dashed line of sight: reads the closing geometry at km range
        ents["los"] = [_marker(scene, 11.0, (1.0, 0.42, 0.32, 0.85)) for _ in range(30)]
        ents["trail"] = _trail(scene, 8.0, (0.25, 0.52, 0.92, 0.65))
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
            res=close_res, fov=_hfov_match(45, close_res), far=20000
        )
    else:  # docking / rendezvous
        # Target station: realistic pressurised core + docking node + solar
        # wings, with the docking port on the +x face (green ring edge +
        # centre marker + translucent approach corridor). The chaser is a
        # Crew-Dragon-like capsule whose +x nose carries the docking ring;
        # all marker positions follow the recorded relative state each frame.
        big = task == "rendezvous"
        # Local bodies are magnified (display-only) because a 1.5 m station at
        # approach range is a handful of pixels; positions, ranges, goals and
        # keep-out radii stay in true metres and the overlay states the factor.
        k = 7.0 if big else 5.0
        s = (3.0 if big else 1.5) * k  # station body edge [m]
        c = (2.0 if big else 1.0) * k  # chaser body edge [m]
        px = s / 2.0  # docking-port face offset from origin
        ents["station"] = _model(scene, "station", _build_station(), scale=s)
        # target docking port on the +x face: ring edge, centre marker, axis
        rr = (0.7 if big else 0.35) * k  # port ring radius [m]
        ents["t_ring"] = _ring(
            scene, rr, (0.2, 0.95, 0.45, 0.95), (px, 0, 0),
            n=48, plane="yz", tube=(0.045 if big else 0.022) * k,
        )
        ents["t_center"] = _marker(
            scene, (0.16 if big else 0.07) * k, (0.1, 0.95, 0.3, 1.0), (px, 0, 0)
        )
        ents["t_axis"] = _cyl(
            scene,
            (0.12 if big else 0.05) * k,
            3.5 * s if big else 2.2 * k,
            (0.2, 0.9, 0.4, 0.28),
        )
        # Chaser: capsule + trunk built as one model (nose on +x). The old
        # per-frame attach list is no longer needed -- the model carries the
        # nose adapter, heat shield, trunk and RCS pods.
        ents["body"] = _model(scene, "dragon", _build_chaser(), scale=c)
        ents["plume"] = [
            _marker(scene, 0.035 * k, (0.25, 0.85, 1.0, 0.82)) for _ in range(6)
        ]
        ents["c_attach"] = []
        # chaser port ring (8 spheres) + centre marker, positioned on the
        # target-facing (+X local) face each frame
        ents["c_ring"] = [
            _marker(scene, (0.06 if big else 0.035) * k, (1.0, 0.8, 0.2, 0.95))
            for _ in range(16)
        ]
        ents["c_center"] = _marker(
            scene, (0.09 if big else 0.05) * k, (1.0, 0.85, 0.3, 1.0)
        )
        # dashed line-of-sight pool between the two centres
        ents["link"] = [
            _marker(scene, 1.2 if big else 0.2, (0.95, 0.95, 0.95, 0.55))
            for _ in range(20)
        ]
        # static starfield, scaled to each task's working volume
        rng = np.random.default_rng(7 if big else 9)
        for i, sp in enumerate(rng.normal(size=(140 if big else 120, 3))):
            sp = sp / np.linalg.norm(sp) * (3600.0 if big else 320.0)
            _marker(scene, (9.0 + 12.0 * float(rng.random()) ** 2) if big
                    else (0.60 + 0.80 * float(rng.random()) ** 2),
                    (0.82, 0.90, 1.0, 0.78), sp)
        ents["trail"] = _trail(
            scene, 2.0 if big else 0.35, (1.0, 0.6, 0.1, 0.9)
        )
        if extras.get("keep_out"):
            ents["keep_out"] = _marker(scene, extras["keep_out"], (0.9, 0.2, 0.2, 0.12))
        # relative-velocity arrow pool (quantized; true speed * per-task scale)
        if big:
            ents["arrows"] = {
                L: _box(scene, (1.1, 1.1, L), (1.0, 0.9, 0.25, 0.95), HIDDEN)
                for L in ARROW_SIZES
            }
        else:
            ents["arrows"] = {
                L: _box(scene, (1.5, 1.5, L), (1.0, 0.9, 0.25, 0.95), HIDDEN)
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
            ents["rings"] = []
            for ri, r in enumerate(RING_RADII):
                ents["rings"].extend(
                    _ring(
                        scene, r, (0.62, 0.60, 0.42, 0.24 - 0.04 * ri),
                        (0, 0, RING_PLANE), n=RING_SEGMENTS, tube=1.6,
                    )
                )
        if task == "rendezvous":
            # fixed LVLH view centered on the target: the approach zone (rings
            # 250/500 m, goal and keep-out) stays readable; the far spawn only
            # appears in the chaser inset until the chaser enters this zone
            u = np.asarray(p0) / np.linalg.norm(p0)
            side = np.cross(u, (0.0, 0.0, 1.0))
            side /= np.linalg.norm(side)
            view = 0.35 * u + 0.75 * side + 0.45 * np.array((0.0, 0.0, 1.0))
            view /= np.linalg.norm(view)
            cam_pos = 250.0 * view
            look = (0.0, 0.0, 0.0)
        else:
            # docking: fixed 3/4 view perpendicular to the recorded approach
            # axis, so the chaser and station read side-by-side instead of the
            # chaser hiding directly behind the station
            u = np.asarray(p0) / np.linalg.norm(p0)
            side = np.cross(u, (0.0, 0.0, 1.0))
            side /= np.linalg.norm(side)
            view = 0.82 * side + 0.45 * np.array((0.0, 0.0, 1.0)) + 0.20 * u
            view /= np.linalg.norm(view)
            cam_pos = 46.0 * view
            look = (0.0, 0.0, 0.0)
        cam = scene.add_camera(
            res=MAIN_RES, pos=tuple(cam_pos), lookat=look,
            fov=_hfov_match(60, MAIN_RES), far=5000
        )
        ents["close_cam"] = scene.add_camera(
            res=close_res, fov=_hfov_match(55, close_res), far=3000
        )
    scene.build(n_envs=1)
    if task == "rocket":
        # display-only: after build the visual geometry is final, so the anchor
        # offset for the 16x airframe can be measured exactly from the model
        ents["body_lift"] = _scale_ground_lift(ents["body"].lowest, 16.0)
    return scene, ents, cam


def _telemetry(task, frames, dt):
    """Per-step trace + phase label for the HUD; no new simulator state."""
    n = len(frames)
    pos = np.array([f["pos"] for f in frames], dtype=float)
    times = np.array([f.get("_t", i * dt) for i, f in enumerate(frames)], dtype=float)
    phases = []

    if task in ("quadx", "quadx_track", "fixedwing", "quadplane", "rocket"):
        series = pos[:, 2]
        label = "ALTITUDE / m"
    elif task == "missile":
        tgt = np.array([f["tgt"] for f in frames], dtype=float)
        series = np.linalg.norm(pos - tgt, axis=1)
        label = "RANGE TO TARGET / m"
    elif task == "satellite":
        vals = []
        for f in frames:
            q, qt = np.asarray(f["quat"], float), np.asarray(f["target_quat"], float)
            vals.append(math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(q, qt)))))))
        series = np.array(vals)
        label = "ATTITUDE ERROR / deg"
    elif task == "transfer":
        series = np.linalg.norm(pos, axis=1)
        label = "ORBIT RADIUS / 1000 km"
    else:
        series = np.linalg.norm(pos, axis=1)
        label = "RANGE TO TARGET / m"

    if task == "quadx":
        goals = []
        for f in frames:
            g = tuple(np.round(np.asarray(f.get("goal", (0, 0, 0)), float), 3))
            if not goals or g != goals[-1]:
                goals.append(g)
        index = []
        for f in frames:
            g = tuple(np.round(np.asarray(f.get("goal", (0, 0, 0)), float), 3))
            index.append(goals.index(g) + 1)
        phases = [f"WAYPOINT {i} / {len(goals)}" for i in index]
    elif task == "quadx_track":
        phases = ["TARGET LOCK" if f.get("visible") else "OCCLUDED" for f in frames]
    elif task in ("fixedwing", "quadplane"):
        phases = ["WAYPOINT NAVIGATION"] * n
    elif task == "rocket":
        phases = ["TERMINAL DESCENT" if z < 90.0 else "GUIDED DESCENT" for z in series]
    elif task == "missile":
        phases = ["TERMINAL INTERCEPT" if r < 650.0 else "MIDCOURSE GUIDANCE"
                  for r in series]
    elif task == "satellite":
        phases = ["CMG SLEW" if e > 2.5 else "ATTITUDE HOLD" for e in series]
    elif task == "transfer":
        phases = ["LOW-THRUST TRANSFER"] * n
    elif task == "docking":
        phases = ["FINAL APPROACH" if r < 20.0 else "CLOSING" for r in series]
    else:
        phases = ["FINAL APPROACH" if r < 300.0 else "MEDIUM APPROACH" for r in series]

    speed = None
    speed_unit = "m/s"
    if n > 1 and task != "satellite":
        dts = np.diff(times)
        step = float(np.median(dts)) if dts.size else float(dt)
        step = max(step, 1e-6)
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) / step
        speed = np.concatenate(([seg[0]], seg))
        if task == "transfer":
            speed = speed * 1000.0  # model unit is 1000 km
            speed_unit = "km/s"
    return {"series": series.tolist(), "label": label, "phase": phases,
            "times": times, "speed": None if speed is None else speed.tolist(),
            "speed_unit": speed_unit}


def render(task, frames, extras, out_path, fps, closeup=False):
    global _CLOSEUP_ONLY, _PLUME
    _CLOSEUP_ONLY = closeup
    scene, ents, cam = build_scene(
        task, extras, frames[0], close_res=CLOSE_HERO_RES if closeup else CLOSE_RES
    )
    if closeup:
        # wide-view furniture that reads as floating dashes in the tight frame
        for ent in ents.get("ladder", []):
            ent.set_pos(tuple(HIDDEN))
    trail = ents.get("trail")
    path = []
    target_path = []
    rgb_frames = []
    big = task == "rendezvous"
    space = task in ("docking", "rendezvous")
    k = 7.0 if big else 5.0
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
    telem = _telemetry(task, frames, extras.get("dt", 1.0 / 30.0))
    # phase-change ticks for the mission timeline (Open MCT time conductor)
    marks = []
    prev_phase = None
    for i, ph in enumerate(telem["phase"]):
        if ph != prev_phase:
            marks.append((i / max(1, n_frames - 1), ph))
            prev_phase = ph
    plan_ctx = _plan_context(task, frames, extras)  # fixed map frame, no jitter
    # fixedwing / quadplane: zero-phase slow pan. The camera anchors the
    # waypoint circuit; the look-at point eases toward the aircraft so the
    # vehicle cannot drift behind the HUD panel or the close-up inset.
    pan_look = None
    if task in ("fixedwing", "quadplane") and "cam_base" in ents:
        _, base_look = ents["cam_base"]
        want = np.array(
            [0.28 * base_look + 0.72 * np.asarray(f["pos"], float) for f in frames]
        )
        win = 15
        pad = np.pad(want, ((win // 2, win // 2), (0, 0)), mode="edge")
        pan_look = np.stack([pad[i : i + win].mean(0) for i in range(len(want))])
    # quadx: the dome is small but the camera is closer still, so a fixed
    # framing loses the drone behind the inset near the end of the capture.
    # Pan on the midpoint of drone and active waypoint (both stay in frame).
    pan_quad = None
    if task == "quadx":
        want = np.array(
            [
                0.5 * (np.asarray(f["pos"], float) + np.asarray(f["goal"], float))
                for f in frames
            ]
        )
        win = 15
        pad = np.pad(want, ((win // 2, win // 2), (0, 0)), mode="edge")
        pan_quad = np.stack([pad[i : i + win].mean(0) for i in range(len(want))])

    def _state_task(name, hero, rows=(), legend=(), note=None, fi=0, result=None):
        frac = (fi + 1) / n_frames
        elapsed = float(telem["times"][fi])
        total = float(telem["times"][-1])
        clock = f"T+ {_fmt_time(elapsed)} / {_fmt_time(total)}"
        row_list = list(rows)
        if telem.get("speed") is not None and len(row_list) < 3:
            row_list.append(
                ("Speed", f"{telem['speed'][fi]:.2f} {telem['speed_unit']}")
            )
        out = dict(task=name, hero=hero, rows=row_list, legend=list(legend),
                   note=note, progress=frac, clock=clock, task_key=task,
                   phase=telem["phase"][fi], marks=marks, spark_idx=fi,
                   spark=(telem["series"], telem["label"]),
                   gcs=dict(fi=fi, frames=frames, extras=extras, telem=telem,
                            plan=plan_ctx))
        if result:
            out["result"] = result
        return out

    spin_phases = np.zeros(len(ents.get("motion", [])))
    for fi, f in enumerate(frames):
        # display-only lift (rocket): keeps the enlarged airframe off the pad
        pos = tuple(map(float, f["pos"]))
        lift = ents.get("body_lift", 0.0)
        ents["body"].set_pos((pos[0], pos[1], pos[2] + lift))
        if f["quat"] is not None:
            ents["body"].set_quat(tuple(map(float, f["quat"])))
        _update_motion(ents, f, spin_phases)
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
            # keep the newest beads out of the displayed body: the rocket is
            # 83 m tall at 16x and the space chasers are 5 / 14 m long, so
            # without clearance the trail looks like a spike through the nose
            clearance = {
                "rocket": 100.0,
                "missile": 320.0,
                "quadx_track": 2.5,
                "docking": 6.0,
                "rendezvous": 16.0,
                "fixedwing": 9.0,
                "quadplane": 9.0,
                "transfer": 3.0,
            }.get(task, 0.0)
            if _CLOSEUP_ONLY and task in ("rocket", "missile"):
                # the narrow close-up frame turns the wide-view trail into a
                # column of floating beads above the vehicle: park it
                for m in trail:
                    m.set_pos(tuple(HIDDEN))
            else:
                _update_trail(trail, path, clearance=clearance)
        p = np.asarray(f["pos"], dtype=float)
        if task == "quadx":
            ents["goal"].set_pos(tuple(map(float, f["goal"])))
            # Slow pan on the drone + active waypoint midpoint. The inset
            # follows the micro-drone; the main frame keeps both subjects
            # clear of the HUD panel and the inset.
            if pan_quad is not None:
                mid = pan_quad[fi]
                cam.set_pose(
                    pos=tuple(mid + np.array((4.6, -6.6, 4.2))),
                    lookat=tuple(mid + np.array((0.0, 0.0, 1.3))),
                    up=WORLD_UP,
                )
            yaw, pitch, dist = math.radians(30.0), math.radians(-30.0), 1.95
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
                [("Waypoints", "4 inside the 5 m dome"), ("Body", "scaled 9x")],
                [("green", "active waypoint"), ("grey", "ground shadow")],
                "slow pan · throttle-driven props · PyFlyt close-up",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                _main_render(cam),
                _close_render(ents["close_cam"]),
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
            yaw, pitch, dist = math.radians(122.0), math.radians(-25.0), 1.8
            off = np.array(
                (
                    dist * math.cos(pitch) * math.cos(yaw),
                    -dist * math.cos(pitch) * math.sin(yaw),
                    -dist * math.sin(pitch),
                )
            )
            ents["close_cam"].set_pose(
                pos=tuple(p + off), lookat=tuple(p + np.array((0.0, 0.0, 0.12))),
                up=WORLD_UP
            )
            state = _state_task(
                "QuadX-Track",
                ("Altitude", f"{p[2]:.2f} m"),
                [("Target", "VISIBLE" if f["visible"] else "LOST"),
                 ("Body", "scaled 14x")],
                [("red", "ground target"), ("grey", "LOS occluder"),
                 ("white", "line of sight")],
                "fixed global · true target · throttle-driven props",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                _main_render(cam),
                _close_render(ents["close_cam"]),
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
            qc = _quat_x_to(d)
            ents["body"].set_quat(qc)
            _update_thruster(
                ents["plume"], pd - d * (0.45 * c_size), f["action"], spacing=0.14 * k
            )
            for ent, lp, lq in ents.get("c_attach", []):
                ent.set_pos(tuple(map(float, pd + _quat_rot(qc, lp))))
                ent.set_quat(tuple(map(float, _quat_mul_np(qc, lq))))
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
            # constant chaser framing far out, tightening to the pair frame as
            # the range closes; the cap keeps the chaser readable instead of
            # shrinking to a dot at spawn range
            near_off = 2.0 * (0.5 * span + e)
            off = float(min(4.5 * c_size, near_off))
            w = 0.5 + 0.5 * float(np.clip((span - min_sep) / (3.0 * e), 0.0, 1.0))
            look = pd * w
            direction = 0.86 * uu + 0.40 * ww + 0.10 * d
            direction /= np.linalg.norm(direction)
            ents["close_cam"].set_pose(
                pos=tuple(look + off * direction), lookat=tuple(look), up=WORLD_UP
            )
        if task == "rendezvous":
            r = float(np.linalg.norm(p))
            rgb = _main_render(cam)
            state = _state_task(
                "Rendezvous · CWH LVLH",
                ("Range to target", f"{r:.0f} m"),
                _dock_rows(f),
                [("green", "goal 10 m"), ("red", "keep-out 5 m"),
                 ("grey", "range rings 250 / 500 m")],
                "goal 10 m ball · keep-out 5 m · bodies 7x · arrow 100 m per m/s",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                rgb,
                _close_render(ents["close_cam"]),
                state,
                "CLOSE-UP: chaser",
            )
            rgb_frames.append(rgb)
            continue
        if task == "docking":
            r = float(np.linalg.norm(p))
            rgb = _main_render(cam)
            state = _state_task(
                "Docking · CWH LVLH",
                ("Range to target", f"{float(r):.2f} m"),
                _dock_rows(f),
                [("grey", "target station"), ("green", "docking port"),
                 ("orange", "chaser")],
                f"goal <{extras['pos_goal']} m & <{extras['vel_goal']} m/s · "
                f"fail >{extras['range_fail']} m · bodies 5x",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                rgb,
                _close_render(ents["close_cam"]),
                state,
                "CLOSE-UP: chaser + station",
            )
            rgb_frames.append(rgb)
        elif task in ("fixedwing", "quadplane", "rocket"):
            if pan_look is not None:
                cam.set_pose(
                    pos=tuple(ents["cam_base"][0]),
                    lookat=tuple(pan_look[fi]),
                    up=WORLD_UP,
                )
            png = _main_render(cam)
            scale = {"fixedwing": 10, "quadplane": 10, "rocket": 16}[task]
            # follow the DISPLAYED body (rocket's scaled airframe is lifted so
            # its base sits on the pad), never the raw link origin
            lift = np.array((0.0, 0.0, ents.get("body_lift", 0.0)))
            if task == "rocket":
                # Fixed ground-based long-lens tracking shot. The narrow 14 deg
                # field and ~570 m stand-off keep the 16x booster readable at
                # apogee and still contain the full airframe at touchdown.
                centre = p + np.array((0.0, 0.0, ents["body_lift"] - 2.1))
                ents["close_cam"].set_pose(
                    pos=(360.0, -440.0, 9.0),
                    lookat=tuple(centre),
                    up=WORLD_UP,
                )
                # Real ignition / fuel / throttle state drives the engine flame.
                aux = np.asarray(f["aux"], float)
                burning = aux[4] > 0.5 and aux[5] > 0.0 and f["throttle"][0] > 0.01
                _PLUME = None
                if burning:
                    # engine-bell exit in the 16x display frame: the model's
                    # lowest point is the landing legs (-2.645), not the bells
                    nozzle = p + np.array((0.0, 0.0, ents["body_lift"] - 2.43 * scale))
                    nx, ny = float(nozzle[0]), float(nozzle[1])
                    thr = float(np.clip(f["throttle"][0], 0.15, 1.0))
                    flick = (0.90 + 0.10 * thr) * (
                        0.94 + 0.05 * math.sin(fi * 2.7) + 0.03 * math.sin(fi * 7.3)
                    )
                    flame = ents["flame"]
                    n = flame["rings"]
                    parts = flame["parts"]
                    # compress the plume into the space above the pad instead of
                    # letting it pile up into a blob at touchdown
                    gap = max(float(nozzle[2]) - 1.0, 0.0)
                    squash = float(
                        np.clip(gap / max(ents["flame_len"], 1e-6), 0.45, 1.0)
                    )
                    zs = []
                    z = float(nozzle[2])
                    for k in range(n):
                        # no clamp: rings below the pad are simply occluded by
                        # the ground plane, which reads as the plume splashing
                        # flat on the concrete instead of piling up
                        z = z - flame["steps"][k] * flick * squash
                        zs.append(z)
                    for idx, m in enumerate(flame["marks"]):
                        k = idx // parts
                        t = k / max(n - 1, 1)
                        ph = flame["phase"][idx]
                        # turbulence: small at the bell, growing downstream
                        wob = 0.10 + 0.55 * t ** 1.6
                        ox, oy = flame["offs"][idx]
                        x = nx + ox + wob * math.sin(fi * 1.7 + ph * 3.0)
                        y = ny + oy + wob * math.cos(fi * 1.3 + ph * 2.0)
                        zz = zs[k] + flame["dz"][idx] * 0.6 * math.sin(fi * 2.3 + ph * 2.0)
                        m.set_pos((x, y, zz))
                    ents["flame_nozzle"].set_pos((nx, ny, max(nozzle[2] + 0.3, 1.0)))
                    _PLUME = (
                        tuple(float(v) for v in nozzle),
                        (0.0, 0.0, -1.0),
                        ents["flame_len"] * flick * squash,
                        4.3,
                        float(fi),
                    )
                    # sparks shed under the tail, twinkling with the burn
                    tailz = zs[-1]
                    for e in ents["embers"]:
                        ph = e["ph"]
                        e["m"].set_pos(
                            (
                                nx + e["dx"] * (0.7 + 0.5 * math.sin(fi * 0.9 + ph * 6.3)),
                                ny + e["dy"] * (0.7 + 0.5 * math.cos(fi * 0.7 + ph * 5.1)),
                                max(tailz - 1.0 - 2.6 * (0.5 + 0.5 * math.sin(fi * 1.3 + ph * 9.4)),
                                    0.7),
                            )
                        )
                    # impingement pool on the pad, revealed as the plume tail
                    # reaches the concrete
                    for j, disc in enumerate(ents["plume_pad"]):
                        if zs[-1] < 3.0 - 4.0 * j:
                            disc.set_pos((0.0, 0.0, 0.42 + 0.04 * j))
                        else:
                            disc.set_pos(tuple(HIDDEN))
                else:
                    for m in ents["flame"]["marks"]:
                        m.set_pos(tuple(HIDDEN))
                    ents["flame_nozzle"].set_pos(tuple(HIDDEN))
                    for disc in ents["plume_pad"]:
                        disc.set_pos(tuple(HIDDEN))
                    for e in ents["embers"]:
                        e["m"].set_pos(tuple(HIDDEN))
            else:
                # Body-frame 3/4 front-left tracking shot. The previous wingtip
                # mount could not see the nose propeller at all; this framing
                # keeps the propeller, the left aileron and the tail surfaces in
                # the same close-up, so their motion is actually observable.
                q = np.asarray(f["quat"], float)
                mount = np.array((0.78, -0.95, 0.55)) * scale
                target = np.array((-0.02, 0.0, 0.03)) * scale
                origin = p + lift
                ents["close_cam"].set_pose(
                    pos=tuple(origin + _quat_rot(q, mount)),
                    lookat=tuple(origin + _quat_rot(q, target)),
                    up=tuple(_quat_rot(q, np.array((0.0, 0.0, 1.0)))),
                )
            names = {
                "fixedwing": "Fixedwing-Waypoints",
                "quadplane": "Quadplane · VTOL",
                "rocket": "Rocket landing",
            }
            if task == "rocket":
                s = np.asarray(f["surfaces"], float) * 45.0
                gim = np.asarray(f["aux"], float)[-2:] * 5.0
                rows = [
                    ("Grid fins", f"{s[0]:+.0f}/{s[1]:+.0f}/{s[2]:+.0f}/{s[3]:+.0f} deg"),
                    ("TVC X/Y", f"{gim[0]:+.1f} / {gim[1]:+.1f} deg"),
                ]
                legend = [("grey", "ground plane"), ("blue", "descent path")]
                note = "ground telephoto · live grid fins/TVC · 16x pad-anchored model"
            else:
                rows = _surf_rows(f["surfaces"])
                legend = [("red", "waypoint target"), ("blue", "flown path")]
                note = (
                    "3/4 front-left · 5 throttle-driven props · baseline surfaces neutral"
                    if task == "quadplane"
                    else "3/4 front-left · live prop, aileron, elevator, rudder"
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
                _close_render(ents["close_cam"]),
                state,
                "GROUND TELEPHOTO · TRACKING" if task == "rocket" else "3/4 FRONT-LEFT · MOVING PARTS",
            )
            rgb_frames.append(rgb)
        elif task == "transfer":
            # fixed camera on the central body: following the spacecraft would
            # hide the orbit shape; the full trajectory must stay in frame
            orb = _orbit_elements(f, extras["orbit"]["mu"])
            for marker, pt in zip(ents["prediction"], orb["points"]):
                marker.set_pos(tuple(map(float, pt)))
            ents["pred_peri"].set_pos(tuple(map(float, orb["peri"])))
            ents["pred_apo"].set_pos(tuple(map(float, orb["apo"])))
            _update_thruster(ents["plume"], p, f["action"], spacing=0.42)
            rgb = _main_render(cam)
            state = _state_task(
                "Orbital transfer · GCRF",
                ("Perigee radius", f"{orb['rp'] * 1e-3:.0f} km"),
                [("Apogee radius", f"{orb['ra'] * 1e-3:.0f} km"),
                 ("Inclination", f"{math.degrees(orb['inc']):.2f} deg")],
                [("green", "initial orbit"), ("cyan", "target circle"),
                 ("violet", "osculating prediction")],
                "orange = flown · cyan plume = recorded thrust · oversized spacecraft",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            # close-up kept on the day-side/outboard of the spacecraft so the
            # Earth ball stays behind the camera: the inset reads as a proper
            # chase shot against stars and orbit lines
            u = p / max(float(np.linalg.norm(p)), 1e-9)
            side = np.cross(u, np.array((0.0, 0.0, 1.0)))
            side = side / max(float(np.linalg.norm(side)), 1e-9)
            ents["close_cam"].set_pose(
                pos=tuple(p + 8.8 * side + 0.18 * u + np.array((0.0, 0.0, 1.8))),
                lookat=tuple(p),
                up=WORLD_UP,
            )
            rgb = _frame(
                rgb,
                _close_render(ents["close_cam"]),
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
                pos=(1.9, 2.3, 1.35), lookat=(0.0, 0.0, 0.0), up=WORLD_UP
            )
            rgb_frames.append(
                _frame(
                    _main_render(cam),
                    _close_render(ents["close_cam"]),
                    state,
                    "CLOSE-UP: bus + CMG",
                )
            )
        elif task == "missile":
            # Fixed global engagement view; only the inset follows the missile.
            tgt = np.asarray(f["tgt"], dtype=float)
            ents["me"].set_pos(tuple(p))
            for i, m in enumerate(ents["los"]):
                m.set_pos(tuple(p + (tgt - p) * (i + 1) / (len(ents["los"]) + 1)))
            # display-only exhaust plume behind the airframe (it flies nose-on
            # to the target, so the tail is away from the line of sight)
            _PLUME = None
            if float(f["aux"][5]) > 0.5:  # actual boost-coast burning flag
                # nozzle is the body -x end; taper the layered flame down that
                # axis so it stays behind the airframe through the whole turn
                back = _quat_rot(np.asarray(f["quat"], float), np.array((-1.0, 0.0, 0.0)))
                side = np.cross(back, np.array((0.0, 0.0, 1.0)))
                nside = np.linalg.norm(side)
                side = side / nside if nside > 1e-6 else np.array((0.0, 1.0, 0.0))
                up2 = np.cross(back, side)
                flick = 0.94 + 0.05 * math.sin(fi * 2.9) + 0.03 * math.sin(fi * 6.7)
                flame = ents["flame"]
                n = flame["rings"]
                parts = flame["parts"]
                acc = 300.0 + np.cumsum(flame["steps"]) * flick
                for idx, m in enumerate(flame["marks"]):
                    k = idx // parts
                    t = k / max(n - 1, 1)
                    ph = flame["phase"][idx]
                    ox, oy = flame["offs"][idx]
                    # turbulent swirl grows downstream of the nozzle
                    wob = 1.0 + 7.0 * t ** 1.5
                    off = (
                        side * (ox + wob * math.sin(fi * 1.8 + ph * 3.0))
                        + up2 * (oy + wob * math.cos(fi * 1.4 + ph * 2.0))
                    )
                    dist = acc[k] + flame["dz"][idx] * 0.6 * math.sin(fi * 2.1 + ph * 2.0)
                    m.set_pos(tuple(p + back * dist + off))
                ents["flame_nozzle"].set_pos(tuple(p + back * 288.0))
                _PLUME = (
                    tuple(float(v) for v in (p + back * 292.0)),
                    tuple(float(v) for v in back),
                    float(np.sum(ents["flame"]["steps"])) * flick,
                    22.0,
                    float(fi),
                )
            else:
                for m in ents["flame"]["marks"]:
                    m.set_pos(tuple(HIDDEN))
                ents["flame_nozzle"].set_pos(tuple(HIDDEN))
            png = _main_render(cam)
            # inset: chaser-framed far out, sliding to the missile+target pair
            # as the engagement closes (same convention as docking)
            rng = float(np.linalg.norm(tgt - p))
            off = float(np.clip(1.35 * rng + 520.0, 760.0, 1400.0))
            w = 0.5 * float(np.clip((1600.0 - rng) / 1400.0, 0.0, 1.0))
            look = p + (tgt - p) * w
            direction = np.array((0.55, -1.0, 0.52))
            direction /= np.linalg.norm(direction)
            ents["close_cam"].set_pose(
                pos=tuple(look + off * direction),
                lookat=tuple(look),
                up=WORLD_UP,
            )
            state = _state_task(
                "Missile intercept",
                ("Altitude", f"{p[2] / 1000.0:.2f} km"),
                _missile_rows(f["surfaces"]),
                [("yellow", "missile + locator"), ("red", "target marker")],
                "live fin actuation + boost-coast plume · global engagement view",
                fi,
                terminal if fi == n_frames - 1 else None,
            )
            rgb = _frame(
                png,
                _close_render(ents["close_cam"]),
                state,
                "CLOSE-UP: missile",
            )
            rgb_frames.append(rgb)
        elif task != "quadx":  # quadx is rendered in its own branch above
            # catch-all (not reachable for the supported tasks): raw panorama
            _follow(cam, p, (-9, -9, 5))
            rgb_frames.append(_main_render(cam))
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
    _CLOSEUP_ONLY = False


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
        "--clip-frames",
        type=int,
        default=0,
        help="render only N evenly spaced frames (time-lapse highlight; 0 = all)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="mp4 path (default results/<task>/vis.mp4)"
    )
    parser.add_argument(
        "--ckpt",
        action="store_true",
        help="replay the trained checkpoint (results/<task>/checkpoint.pt) "
        "instead of the classical baseline",
    )
    parser.add_argument(
        "--closeup",
        action="store_true",
        help="render only the close-up camera at native 960x540 "
        "(for the standalone close-up showcase replays)",
    )
    args = parser.parse_args()

    out = args.out or Path(__file__).resolve().parent.parent / "results" / args.vehicle / (
        "vis_closeup.mp4" if args.closeup else ("vis_ckpt.mp4" if args.ckpt else "vis.mp4")
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    env, cfg = build_env(args.vehicle)
    policy = (
        _CkptPolicy(args.vehicle, env, cfg) if args.ckpt else _BaselinePolicy(args.vehicle, env)
    )
    frames, extras = record(args.vehicle, env, policy, args.max_steps)
    dt = getattr(env, "dt", None)
    if dt is None:
        dt = 1.0 / float(getattr(env, "agent_hz", 30))
    extras["dt"] = float(dt)
    for i, f in enumerate(frames):
        f["_t"] = i * float(dt)
    env.close()

    if args.clip_frames and args.clip_frames < len(frames):
        idx = np.linspace(0, len(frames) - 1, args.clip_frames).round().astype(int)
        frames = [frames[i] for i in idx]

    render(args.vehicle, frames, extras, out, args.fps, closeup=args.closeup)
    print(f"{args.vehicle}: recorded {len(frames)} agent steps -> {out}")


if __name__ == "__main__":
    main()
