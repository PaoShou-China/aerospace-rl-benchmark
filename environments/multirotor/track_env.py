"""QuadX-Track: persistent observation of a cooperative ground moving target.

Task
    A 27 g Crazyflie 2.x quadrotor must keep a cooperative ground vehicle
    inside a nadir-stabilized camera's field of view while holding a fixed
    standoff offset and a fixed altitude. The target drives on the ground
    plane with a pre-drawn schedule of straight, turning and stop-and-go
    segments inside a 15 m radius area.

Sensing (geometric, no rendering)
    The camera is a downward cone of half-angle FOV_HALF_DEG mounted on a
    stabilized gimbal, so the optical axis is world -z and the ground
    footprint radius at altitude dz is dz * tan(FOV_HALF_DEG). A detection
    requires ALL of:
      * the target inside the FOV cone (and below the vehicle),
      * the slant range inside [SENSOR_RANGE_MIN, SENSOR_RANGE_MAX],
      * an unobstructed line of sight past the static cylindrical occluders.
    A detection yields a 3-D position measurement in WORLD axes relative to
    the vehicle, y - p, corrupted by isotropic N(0, POS_NOISE_STD^2) per
    axis. When any condition fails the environment reports a ZERO relative
    measurement and a 0 visibility flag: the true target state is never
    exposed. Occluders are pure line-of-sight geometry -- they carry no
    collision dynamics, and there is no interception, contact or weapon
    objective anywhere in the task.

Action Space (4,)
    Same interface as QuadX-Waypoints: policy output in [-1, 1]^4 affinely
    remapped onto roll/pitch/yaw rate commands [-pi, pi] rad/s and
    collective throttle [0, 0.8], feeding the plant's 120 Hz rate PID.

Observation Space (25,)
    float32, 'policy' TensorDict group:
    [body angular velocity (3), quaternion xyzw (4), body-frame linear
    velocity (3), world position (3), last physical action (4), per-rotor
    throttle (4), visible target measurement relative to the vehicle in
    world axes (3, zero when invisible), visibility flag (1)].

Reward
    Per policy step, evaluated on the end-of-step state:
        r = +0.10 * vis - 0.10 * (1 - vis)
            - 0.05 * ref_error * vis - 0.01 * (z - ALT_REF)^2
            - 0.002 * |a_phys|^2 - 10 * crash,
    where ref_error is the horizontal distance from the vehicle to the
    reference point (target_xy + REF_OFFSET_XY). The tracking term is only
    paid while visible: when the target is lost the agent receives no
    target-derived signal beyond the -0.10 loss penalty, matching the
    observation (there is no hidden target state in the reward channel
    that the policy could exploit through observations).

Episode End
    Termination: ground contact (collision) or ||p|| > 30 m.
    Truncation: 1200 policy steps (30 s).
    Success: recorded at the time-limit truncation when the episode's
    visible-time fraction is at least SUCCESS_VISIBLE_FRACTION and the
    vehicle has not crashed. Success does not cut the episode short.

Metrics (per completed episode, accumulated in completed_metrics)
    visible_fraction, lost_seconds (total invisible time), max_dropout_seconds,
    track_error (mean horizontal standoff error over visible steps),
    flight_distance (path length), collisions, out_of_bounds, timeouts, plus
    the loss-cause counts fov_loss, range_loss and occlusion_steps.

References
    - PyFlyt QuadX-Waypoints vehicle/plant defaults (cf2x, Genesis backend).
    - Method documentation: docs/method/env-quadx-track.md.
"""
import math
import torch
from tensordict import TensorDict
from environments.common.plant import TensorPlant, TensorRandom
from environments.common.dynamics import euler, inverse_rotate
from algorithms.baselines.baselines import QuadXTrackKFBaseline

# ---- camera / geometry -----------------------------------------------------
FOV_HALF_DEG = 30.0
SENSOR_RANGE_MIN = 1.5       # m
SENSOR_RANGE_MAX = 12.0      # m
POS_NOISE_STD = 0.15         # m, per axis
ALT_REF = 6.0                # m, commanded camera altitude
REF_OFFSET_XY = (1.5, 0.0)   # m, fixed horizontal offset of the reference point
TARGET_Z = 0.15              # m, ground vehicle antenna height
TARGET_SPAWN_MIN = 0.5       # m, initial horizontal distance from the vehicle
TARGET_SPAWN_MAX = 2.0       # m
TARGET_AREA_RADIUS = 15.0    # m, the target reflects off this circle
BOUNDS_RADIUS = 30.0         # m, vehicle out-of-bounds termination

# ---- target motion schedule ------------------------------------------------
NUM_SEGMENTS = 12
SEG_DURATION_MIN, SEG_DURATION_MAX = 2.0, 5.0   # s
SEG_SPEED_MIN, SEG_SPEED_MAX = 0.4, 1.2         # m/s
SEG_TURN_MIN, SEG_TURN_MAX = 0.3, 1.0           # rad/s
MODE_STRAIGHT, MODE_TURN, MODE_STOP = 0, 1, 2

# ---- static line-of-sight occluders: (x, y, radius, height) -----------------
OCCLUDERS = (
    (4.5, 3.0, 1.0, 5.0),
    (-3.5, 5.5, 1.2, 6.0),
    (6.0, -4.0, 1.0, 4.5),
    (-6.0, -3.0, 1.1, 5.5),
    (1.0, -6.5, 1.0, 4.0),
    (-1.5, 8.5, 1.0, 6.5),
)

# ---- reward -----------------------------------------------------------------
VIS_BONUS = 0.10
LOST_PENALTY = 0.10
TRACK_GAIN = 0.05
ALT_GAIN = 0.01
ACTION_GAIN = 0.002
CRASH_PENALTY = 10.0
SUCCESS_VISIBLE_FRACTION = 0.85
MAX_STEPS = 1200            # 30 s at 40 Hz


def _occluded(p_uav, p_tgt):
    """(n,) bool: any static cylinder intersects the UAV-target line of sight."""
    ux, uy, uz = p_uav.unbind(-1)
    dx, dy, dz = (p_tgt - p_uav).unbind(-1)
    denom = (dx * dx + dy * dy).clamp_min(1e-9)
    blocked = torch.zeros_like(ux, dtype=torch.bool)
    for cx, cy, radius, height in OCCLUDERS:
        t = (((cx - ux) * dx + (cy - uy) * dy) / denom).clamp(0., 1.)
        qx, qy = ux + t * dx, uy + t * dy
        dist2 = (qx - cx) ** 2 + (qy - cy) ** 2
        z_at = uz + t * dz
        blocked |= (dist2 < radius * radius) & (z_at < height)
    return blocked


class QuadXTrackEnv:
    # public device-side constants (the classical baseline reads these at
    # construction, like every other baseline in the suite)
    ALT_REF = ALT_REF
    REF_OFFSET_XY = REF_OFFSET_XY
    FOV_HALF_DEG = FOV_HALF_DEG
    SENSOR_RANGE_MIN = SENSOR_RANGE_MIN
    SENSOR_RANGE_MAX = SENSOR_RANGE_MAX
    POS_NOISE_STD = POS_NOISE_STD
    TARGET_Z = TARGET_Z
    TARGET_AREA_RADIUS = TARGET_AREA_RADIUS
    SUCCESS_VISIBLE_FRACTION = SUCCESS_VISIBLE_FRACTION
    conventional_controller = QuadXTrackKFBaseline

    bounds_radius = BOUNDS_RADIUS

    def __init__(self, num_envs=64, backend='cpu', seed=7):
        if num_envs < 1:
            raise ValueError('n must be positive')
        self.num_envs = num_envs
        self.task_cfg = {'vehicle': 'quadx', 'task': 'quadx_track', 'num_envs': num_envs,
                         'backend': backend, 'implementation': 'fully_tensorized',
                         'task_defaults': 'AerospaceRL quadx-track v1'}
        self.clip_actions = 1.0

        # same plant, rate-PID and solver guard as QuadX-Waypoints
        self.plant = TensorPlant('quadx', num_envs, backend, seed, vel_clamp=(15., 30.))
        self.device = self.plant.device
        self.agent_hz = 40
        self.inner_steps = int(self.plant.control_hz // self.agent_hz)
        self.max_steps = MAX_STEPS
        self.max_episode_length = self.max_steps + 2
        self.num_actions = 4
        self.dt = 1. / self.agent_hz
        self.action_low = torch.tensor([-math.pi, -math.pi, -math.pi, 0.],
                                       device=self.device, dtype=torch.float64)
        self.action_high = torch.tensor([math.pi, math.pi, math.pi, .8],
                                        device=self.device, dtype=torch.float64)
        # a separate counter stream keeps the camera noise from perturbing the
        # plant's motor-noise sequence
        self.sensor_rng = TensorRandom(num_envs, self.device, seed + 7919)
        self.tan_fov = math.tan(math.radians(FOV_HALF_DEG))
        self.ref_offset = torch.tensor(REF_OFFSET_XY, device=self.device, dtype=torch.float64)

        d = self.device
        zeros = lambda *s: torch.zeros(*s, device=d, dtype=torch.float64)
        self.action = zeros(num_envs, self.num_actions)
        self.reward = zeros(num_envs)
        self.count = torch.zeros(num_envs, device=d, dtype=torch.long)
        self.terminated = torch.zeros(num_envs, device=d, dtype=torch.bool)
        self.truncated = self.terminated.clone()
        self.collision = self.terminated.clone()
        self.out_of_bounds = self.terminated.clone()
        self.complete = self.terminated.clone()
        self._observation = None
        self.obs_buf = None
        self.rew_buf = torch.zeros(num_envs, device=d, dtype=torch.float32)
        self.reset_buf = torch.zeros(num_envs, device=d, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, device=d, dtype=torch.long)
        self.returns = zeros(num_envs)
        self.completed_returns = zeros(num_envs)
        self.completed_episodes = torch.zeros(num_envs, device=d, dtype=torch.long)
        self.completed_successes = torch.zeros(num_envs, device=d, dtype=torch.long)

        # target kinematic state (reset-owned)
        self.target_pos = zeros(num_envs, 3)
        self.target_heading = zeros(num_envs)
        self.target_speed = zeros(num_envs)
        self.target_yaw_rate = zeros(num_envs)
        self.target_mode = torch.zeros(num_envs, device=d, dtype=torch.long)
        self.target_seg_index = torch.zeros(num_envs, device=d, dtype=torch.long)
        self.target_seg_timer = zeros(num_envs)
        self.seg_mode = torch.zeros((num_envs, NUM_SEGMENTS), device=d, dtype=torch.long)
        self.seg_dur = zeros(num_envs, NUM_SEGMENTS)
        self.seg_speed = zeros(num_envs, NUM_SEGMENTS)
        self.seg_yaw = zeros(num_envs, NUM_SEGMENTS)

        # perception / metric accumulators (per current episode)
        self.vis = torch.zeros(num_envs, device=d, dtype=torch.bool)
        self.ep_visible = zeros(num_envs)
        self.ep_invisible = zeros(num_envs)
        self.ep_dropout = zeros(num_envs)
        self.ep_max_dropout = zeros(num_envs)
        self.ep_ref_err = zeros(num_envs)
        self.ep_visible_steps = zeros(num_envs)
        self.ep_path = zeros(num_envs)
        self.ep_fov_loss = zeros(num_envs)
        self.ep_range_loss = zeros(num_envs)
        self.ep_occlusion = zeros(num_envs)

        keys = ('visible_fraction', 'lost_seconds', 'max_dropout_seconds', 'track_error',
                'flight_distance', 'collisions', 'out_of_bounds', 'timeouts',
                'fov_loss_steps', 'range_loss_steps', 'occlusion_steps')
        self.completed_metrics = {k: torch.zeros((), device=d, dtype=torch.float64) for k in keys}

        self.extras = dict()
        self.cfg = dict(self.task_cfg)
        self.reset()

    # ------------------------------------------------------------------ #
    # action mapping: identical to QuadX-Waypoints
    # ------------------------------------------------------------------ #
    def _scale_actions(self, actions):
        raw = actions.to(torch.float64).clamp(-self.clip_actions, self.clip_actions)
        return self.action_low + (raw + 1) * .5 * (self.action_high - self.action_low)

    # ------------------------------------------------------------------ #
    # cooperative ground target
    # ------------------------------------------------------------------ #
    def _advance_target(self, mask):
        """Advance the target by one policy period on active replicas."""
        t = self.target_pos
        moving = self.target_mode != MODE_STOP
        speed = torch.where(moving, self.target_speed, torch.zeros_like(self.target_speed))
        heading = self.target_heading
        vx = speed * torch.cos(heading)
        vy = speed * torch.sin(heading)
        nx = torch.where(mask, t[:, 0] + vx * self.dt, t[:, 0])
        ny = torch.where(mask, t[:, 1] + vy * self.dt, t[:, 1])
        nz = t[:, 2]
        heading = torch.where(mask, heading + self.target_yaw_rate * self.dt, heading)
        # reflect off the circular operating area so a 30 s episode cannot
        # drive the target (and the chase) outside the vehicle's flight bounds
        rad = torch.sqrt(nx * nx + ny * ny).clamp_min(1e-9)
        outside = mask & (rad > TARGET_AREA_RADIUS)
        normal = torch.stack((nx / rad, ny / rad), -1)
        scale = torch.where(outside, TARGET_AREA_RADIUS / rad, torch.ones_like(rad))
        nx, ny = nx * scale, ny * scale
        dvec = torch.stack((torch.cos(heading), torch.sin(heading)), -1)
        dot = (dvec * normal).sum(-1)
        dvec = torch.where(outside[:, None], dvec - 2. * dot[:, None] * normal, dvec)
        heading = torch.where(outside, torch.atan2(dvec[:, 1], dvec[:, 0]), heading)
        self.target_pos.copy_(torch.stack((nx, ny, nz), -1))
        self.target_heading.copy_(heading)

        # segment timer / schedule switch (at most one boundary per 25 ms step)
        timer = torch.where(mask, self.target_seg_timer - self.dt, self.target_seg_timer)
        switch = mask & (timer <= 0.)
        nxt = (self.target_seg_index + switch.to(torch.long)).clamp_max(NUM_SEGMENTS - 1)
        idx = nxt[:, None]
        mode = self.seg_mode.gather(1, idx).squeeze(1)
        dur = self.seg_dur.gather(1, idx).squeeze(1)
        spd = self.seg_speed.gather(1, idx).squeeze(1)
        yaw = self.seg_yaw.gather(1, idx).squeeze(1)
        self.target_seg_index.copy_(torch.where(switch, nxt, self.target_seg_index))
        self.target_seg_timer.copy_(torch.where(switch, dur, timer))
        self.target_mode.copy_(torch.where(switch, mode, self.target_mode))
        self.target_speed.copy_(torch.where(switch, spd, self.target_speed))
        self.target_yaw_rate.copy_(torch.where(switch, yaw, self.target_yaw_rate))

    # ------------------------------------------------------------------ #
    # geometric perception
    # ------------------------------------------------------------------ #
    def _perception(self, mask):
        """Visibility, loss causes and the (possibly noisy) relative measurement."""
        p, _, _, _ = self.plant.state()
        tp = self.target_pos
        rel = tp - p
        dh = rel[:, :2].norm(dim=-1)
        d3 = rel.norm(dim=-1)
        dz = p[:, 2] - tp[:, 2]
        in_fov = (dz > 0.) & (dh <= dz * self.tan_fov)
        in_range = (d3 >= SENSOR_RANGE_MIN) & (d3 <= SENSOR_RANGE_MAX)
        occ = _occluded(p, tp)
        vis = mask & in_fov & in_range & ~occ
        noise = torch.stack([self.sensor_rng.normal(vis, 0.) for _ in range(3)], -1) * POS_NOISE_STD
        rel_meas = torch.where(vis[:, None], tp + noise - p, torch.zeros_like(rel))
        return vis, in_fov, in_range, occ, rel_meas

    def _build_observation(self, mask, rel_meas, vis):
        p, q, v, w = self.plant.state()
        wb = inverse_rotate(q, w)
        vb = inverse_rotate(q, v)
        observation = torch.cat((wb, q[:, 1:], q[:, :1], vb, p, self.action,
                                 self.plant.auxiliary(), rel_meas,
                                 vis.to(torch.float64)[:, None]), -1)
        if self._observation is None:
            self._observation = observation.clone()
        else:
            self._observation.copy_(torch.where(mask[:, None], observation, self._observation))

    def _update_observation(self):
        if self.obs_buf is None:
            self.obs_buf = torch.zeros(self._observation.shape, device=self.device, dtype=torch.float32)
        self.obs_buf.copy_(self._observation)

    def get_observations(self):
        return TensorDict({'policy': self.obs_buf.clone()}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def step(self, actions):
        if actions.device != torch.device(self.device):
            raise ValueError('Actions must already be on env.device; implicit host/device copies are disabled')
        physical = self._scale_actions(actions)
        self.action.copy_(physical)
        active0 = ~(self.terminated | self.truncated)
        self._advance_target(active0)
        self.count += 1

        crash = torch.zeros_like(self.terminated)
        for _ in range(self.inner_steps):
            active = ~(self.terminated | self.truncated)
            self.plant.clear_contacts()
            self.plant.controller(physical, active)
            for _ in range(self.plant.physics_per_control):
                self.plant.physics_step(physical, active)
            p, _, v, _ = self.plant.state()
            collision = self.plant.ground_contact
            bounds = p.norm(dim=-1) > BOUNDS_RADIUS
            term = collision | bounds
            crash |= term & active
            self.terminated |= term & active
            self.collision |= collision & active
            self.out_of_bounds |= bounds & active

        # the time limit is applied after the full control step
        active = ~(self.terminated | self.truncated)
        timeout = (self.count >= self.max_steps) & active
        self.truncated |= timeout

        # perception and the step reward, evaluated on the end-of-step state
        vis, in_fov, in_range, occ, rel_meas = self._perception(active0)
        self.vis.copy_(torch.where(active0, vis, self.vis))
        self._build_observation(active0, rel_meas, vis)
        p, _, v, _ = self.plant.state()
        ref_err = ((self.target_pos[:, :2] + self.ref_offset) - p[:, :2]).norm(dim=-1)
        alt_err = p[:, 2] - ALT_REF
        self.reward.copy_(torch.where(active0,
                                      VIS_BONUS * vis.to(torch.float64)
                                      - LOST_PENALTY * (~vis).to(torch.float64)
                                      - TRACK_GAIN * ref_err * vis.to(torch.float64)
                                      - ALT_GAIN * alt_err * alt_err
                                      - ACTION_GAIN * physical.square().sum(-1)
                                      - CRASH_PENALTY * crash.to(torch.float64),
                                      torch.zeros_like(self.reward)))

        # per-episode metric accumulation
        self.ep_visible += vis.to(torch.float64)
        self.ep_invisible += (active0 & ~vis).to(torch.float64)
        self.ep_dropout.copy_(torch.where(active0, torch.where(vis, torch.zeros_like(self.ep_dropout),
                                                                self.ep_dropout + self.dt), self.ep_dropout))
        self.ep_max_dropout.copy_(torch.where(active0, torch.maximum(self.ep_max_dropout, self.ep_dropout),
                                              self.ep_max_dropout))
        self.ep_ref_err += torch.where(vis, ref_err, torch.zeros_like(ref_err))
        self.ep_visible_steps += vis.to(torch.float64)
        self.ep_path += torch.where(active0, v.norm(dim=-1) * self.dt, torch.zeros_like(self.ep_path))
        self.ep_fov_loss += (active0 & ~in_fov).to(torch.float64)
        self.ep_range_loss += (active0 & in_fov & ~in_range).to(torch.float64)
        self.ep_occlusion += (active0 & in_fov & in_range & occ).to(torch.float64)

        # success = full visible-time fraction at the natural time limit, no crash
        vis_frac = self.ep_visible / self.count.to(torch.float64).clamp_min(1)
        self.complete |= timeout & (vis_frac >= SUCCESS_VISIBLE_FRACTION)

        self.reset_buf = self.terminated | self.truncated
        self.rew_buf.copy_(self.reward)
        self.episode_length_buf += 1
        self.returns += self.reward
        done = self.reset_buf
        self.completed_returns += torch.where(done, self.returns, 0.)
        self.completed_episodes += done.to(torch.long)
        self.completed_successes += (done & self.complete).to(torch.long)
        term_out, trunc_out = self.terminated.clone(), self.truncated.clone()

        self.extras['time_outs'] = self.truncated & ~self.terminated
        self.extras['terminal_observations'] = TensorDict(
            {'policy': self._observation.to(torch.float32)}, batch_size=[self.num_envs])

        done_any = bool(done.any())
        if done_any:
            n = done.sum().clamp_min(1)
            steps = self.count[done].to(torch.float64).clamp_min(1)
            self.completed_metrics['visible_fraction'] += (self.ep_visible[done] / steps).sum()
            self.completed_metrics['lost_seconds'] += (self.ep_invisible[done] * self.dt).sum()
            self.completed_metrics['max_dropout_seconds'] += self.ep_max_dropout[done].sum()
            self.completed_metrics['track_error'] += (self.ep_ref_err[done]
                                                     / self.ep_visible_steps[done].clamp_min(1)).sum()
            self.completed_metrics['flight_distance'] += self.ep_path[done].sum()
            self.completed_metrics['collisions'] += (done & self.collision).to(torch.float64).sum()
            self.completed_metrics['out_of_bounds'] += (done & self.out_of_bounds).to(torch.float64).sum()
            self.completed_metrics['timeouts'] += (done & self.truncated & ~self.complete).to(torch.float64).sum()
            self.completed_metrics['fov_loss_steps'] += self.ep_fov_loss[done].sum()
            self.completed_metrics['range_loss_steps'] += self.ep_range_loss[done].sum()
            self.completed_metrics['occlusion_steps'] += self.ep_occlusion[done].sum()
            self.extras['episode'] = {
                'success_rate': (done & self.complete).to(torch.float64).sum() / n,
                'visible_fraction': (self.ep_visible[done] / steps).mean(),
                'lost_seconds': (self.ep_invisible[done] * self.dt).mean(),
                'max_dropout_seconds': self.ep_max_dropout[done].mean(),
                'track_error': (self.ep_ref_err[done] / self.ep_visible_steps[done].clamp_min(1)).mean(),
                'flight_distance': self.ep_path[done].mean(),
                'collision_rate': (done & self.collision).to(torch.float64).sum() / n,
            }
        else:
            self.extras.pop('episode', None)

        if done_any:
            self.reset_idx(self.reset_buf)
        self._update_observation()
        return self.get_observations(), self.rew_buf, term_out, trunc_out, self.extras

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset_idx(self, envs_idx):
        """Reset selected replicas; envs_idx is a device-resident boolean mask."""
        mask = envs_idx
        p, q, v, w = self.plant.state()
        spawn = torch.zeros_like(p)
        spawn[:, 2] = ALT_REF
        rotation = torch.zeros_like(q)
        rotation[:, 0] = 1.
        self.plant.set_state(torch.where(mask[:, None], spawn, p),
                             torch.where(mask[:, None], rotation, q),
                             torch.where(mask[:, None], torch.zeros_like(v), v),
                             torch.where(mask[:, None], torch.zeros_like(w), w))
        self.count.masked_fill_(mask, 0)
        self.terminated.masked_fill_(mask, False)
        self.truncated.masked_fill_(mask, False)
        self.collision.masked_fill_(mask, False)
        self.out_of_bounds.masked_fill_(mask, False)
        self.complete.masked_fill_(mask, False)
        self.action.masked_fill_(mask[:, None], 0.)
        self.plant.zero(mask)

        # target spawn inside the camera footprint + a full segment schedule;
        # one uniform draw for everything, reset-masked (inactive replicas do
        # not consume random numbers)
        width = 3 + 5 * NUM_SEGMENTS
        u = self.plant.rng.uniform(width, mask)
        radius = TARGET_SPAWN_MIN + u[:, 0] * (TARGET_SPAWN_MAX - TARGET_SPAWN_MIN)
        angle = u[:, 1] * 2. * math.pi
        heading = u[:, 2] * 2. * math.pi
        seg = u[:, 3:].unflatten(-1, (NUM_SEGMENTS, 5))
        mode = (seg[..., 0] * 3.).clamp(0., 2.).to(torch.long)
        dur = SEG_DURATION_MIN + seg[..., 1] * (SEG_DURATION_MAX - SEG_DURATION_MIN)
        speed = SEG_SPEED_MIN + seg[..., 2] * (SEG_SPEED_MAX - SEG_SPEED_MIN)
        turn = SEG_TURN_MIN + seg[..., 3] * (SEG_TURN_MAX - SEG_TURN_MIN)
        turn = torch.where(seg[..., 4] > .5, turn, -turn)
        speed = torch.where(mode == MODE_STOP, torch.zeros_like(speed), speed)
        turn = torch.where(mode == MODE_TURN, turn, torch.zeros_like(turn))
        self.seg_mode.copy_(torch.where(mask[:, None], mode, self.seg_mode))
        self.seg_dur.copy_(torch.where(mask[:, None], dur, self.seg_dur))
        self.seg_speed.copy_(torch.where(mask[:, None], speed, self.seg_speed))
        self.seg_yaw.copy_(torch.where(mask[:, None], turn, self.seg_yaw))

        target = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle),
                              torch.full_like(radius, TARGET_Z)), -1)
        self.target_pos.copy_(torch.where(mask[:, None], target, self.target_pos))
        self.target_heading.copy_(torch.where(mask, heading, self.target_heading))
        self.target_mode.copy_(torch.where(mask, mode[:, 0], self.target_mode))
        self.target_speed.copy_(torch.where(mask, speed[:, 0], self.target_speed))
        self.target_yaw_rate.copy_(torch.where(mask, turn[:, 0], self.target_yaw_rate))
        self.target_seg_index.masked_fill_(mask, 0)
        self.target_seg_timer.copy_(torch.where(mask, dur[:, 0], self.target_seg_timer))

        self.ep_visible.masked_fill_(mask, 0.)
        self.ep_invisible.masked_fill_(mask, 0.)
        self.ep_dropout.masked_fill_(mask, 0.)
        self.ep_max_dropout.masked_fill_(mask, 0.)
        self.ep_ref_err.masked_fill_(mask, 0.)
        self.ep_visible_steps.masked_fill_(mask, 0.)
        self.ep_path.masked_fill_(mask, 0.)
        self.ep_fov_loss.masked_fill_(mask, 0.)
        self.ep_range_loss.masked_fill_(mask, 0.)
        self.ep_occlusion.masked_fill_(mask, 0.)

        # settle the vehicle, then seed the observation with the first
        # (guaranteed in-footprint) measurement
        self.plant.warmup(mask)
        vis, _, _, _, rel_meas = self._perception(mask)
        self.vis.copy_(torch.where(mask, vis, self.vis))
        self._build_observation(mask, rel_meas, vis)
        self.episode_length_buf.masked_fill_(mask, 0)
        self.returns.masked_fill_(mask, 0)

    def reset(self):
        mask = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.reset_idx(mask)
        self.reset_buf = mask.clone()
        self._update_observation()
        return self.get_observations()

    def close(self):
        self.plant.close()
