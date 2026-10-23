"""Small observable-state feedback repairs for the original rocket actions.

Raw actions are [fin_x, fin_y, fin_yaw, ignition, throttle, gimbal_x,
gimbal_y].  Only ignition/throttle are mapped from [-1, 1] to [0, 1] by
the environment.  This module changes commands, never actuator states or
physics.  The caller owns intervention timing, duration and episode memory.
"""
from __future__ import annotations

import torch


PARAM_DIM = 12
FEATURE_DIM = 30
PARAMETER_NAMES = (
    "throttle_bias", "descent_gain", "pulse_speed", "position_gain",
    "lateral_velocity_gain", "tilt_gain", "angular_rate_gain", "fin_gain",
    "yaw_damping_gain", "gimbal_x_bias", "gimbal_y_bias", "throttle_rate_per_second",
)
FEATURE_NAMES = (
    "log_height", "height", "world_vx", "world_vy", "world_vz",
    "body_wx", "body_wy", "body_wz", "qx", "qy", "qz", "qw",
    "world_x", "world_y", "upright_error_x", "upright_error_y",
    "fuel", "actual_thrust_fraction", "actual_gimbal_x", "actual_gimbal_y",
    "ignited", "pad_contact", "parent_ignition_raw", "parent_throttle_physical",
    "parent_gimbal_x", "parent_gimbal_y", "braking_margin", "speed",
    "projected_lateral_distance", "time_to_ground_proxy",
)

_GRAVITY = 9.81
_MIN_THRUST = 2966.7
_MAX_THRUST = 7607.0
_DT = 1. / 40.


def default_parameters(n, device="cpu"):
    """Neutral feedback; mode 0 exactly preserves clipped float32 actions."""
    value = torch.zeros((n, PARAM_DIM), device=device, dtype=torch.float32)
    value[:, 2] = .35
    value[:, 11] = 80.  # >= 2 command units per step: no [0,1] rate restriction.
    return value


def parameter_bounds(device="cpu"):
    low = torch.tensor([-.15, 0., .25, 0., 0., 0., 0., 0., 0., -.15, -.15, .2],
                       device=device, dtype=torch.float32)
    high = torch.tensor([.15, .15, .60, 2., 3., .4, .6, .2, .3, .15, .15, 80.],
                        device=device, dtype=torch.float32)
    return low, high


def _rotate(q, vector):
    """Body-to-world rotation for a normalized xyzw quaternion."""
    twice_cross = 2. * torch.linalg.cross(q[:, :3], vector, dim=-1)
    return vector + q[:, 3:4] * twice_cross + torch.linalg.cross(q[:, :3], twice_cross, dim=-1)


def _state(raw_observation):
    if raw_observation.ndim != 2 or raw_observation.shape[1] != 30:
        raise ValueError("raw_observation must have shape [N,30]")
    raw = raw_observation.float()
    q = raw[:, 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = q * torch.where(q[:, 3:4] < 0, -1., 1.)
    velocity = _rotate(q, raw[:, 7:10])
    height = raw[:, 12] - 2.575
    mass = 138.2 + 410.9 * raw[:, 25].clamp(0., 1.)
    return raw, q, velocity, height, mass


def feedback_action(raw_observation, parent_action, parameters, ignition_mode):
    """Return bounded float32 commands without touching the real actuators.

    Parameters follow PARAMETER_NAMES and are clipped to parameter_bounds.
    Modes: 0 inherit; 1 force off; 2 force on; 3 pulse below 2 m, inherit
    above it; 4 cut on the *currently observed* pad contact, otherwise inherit.
    Mode 3 uses observed ignited state for speed hysteresis, cuts on contact
    and on ascent.  Mode 4 has no hidden ever-contact latch; the caller can
    implement that by switching to mode 1 in its explicit decision memory.
    """
    raw, q, velocity, height, mass = _state(raw_observation)
    n = len(raw)
    if parent_action.shape != (n, 7) or parameters.shape != (n, PARAM_DIM):
        raise ValueError("expected parent_action[N,7] and parameters[N,12]")
    if ignition_mode.shape != (n,):
        raise ValueError("ignition_mode must have shape [N]")
    base = parent_action.float().clamp(-1., 1.)
    low, high = parameter_bounds(raw.device)
    params = torch.maximum(torch.minimum(parameters.float(), high), low)
    bu, ks, vc, kp, kv, kr, kw, kf, ky, bgx, bgy, rate = params.unbind(-1)
    descent = -velocity[:, 2]  # Signed: ascent must not become a relight request.
    maximum_braking = (_MAX_THRUST / mass - _GRAVITY).clamp_min(1.)
    descent_reference = (vc.square() + 1.2 * maximum_braking * (height - 2.).clamp_min(0.)).sqrt()
    parent_throttle = (base[:, 4] + 1.) * .5
    target_throttle = (parent_throttle + bu + ks * (descent - descent_reference)).clamp(0., 1.)
    # Observation 17 is the previous physical throttle COMMAND in [0,1].
    # Observation 26 is actual thrust/max, already including the thrust floor.
    previous_throttle = raw[:, 17].clamp(0., 1.)
    limit = rate * _DT
    throttle = (previous_throttle + torch.maximum(torch.minimum(
        target_throttle - previous_throttle, limit), -limit)).clamp(0., 1.)

    position_weight = ((height - 2.) / 8.).clamp(0., 1.)
    lateral = -kp[:, None] * position_weight[:, None] * raw[:, 10:12] / 2. - kv[:, None] * velocity[:, :2]
    vertical_reference = _MIN_THRUST / mass
    absolute_cap = torch.where(height <= 2., 1.5, 4.5)
    lateral_cap = torch.minimum(absolute_cap, .1 * vertical_reference)
    lateral = torch.maximum(torch.minimum(lateral, lateral_cap[:, None]), -lateral_cap[:, None])
    desired_up = torch.cat((lateral, vertical_reference[:, None]), -1)
    desired_up = desired_up / desired_up.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    inverse_q = torch.cat((-q[:, :3], q[:, 3:4]), -1)
    desired_body_up = _rotate(inverse_q, desired_up)
    tilt_error = torch.stack((-desired_body_up[:, 1], desired_body_up[:, 0]), -1)
    torque_direction = kr[:, None] * tilt_error / .1 - kw[:, None] * raw[:, :2] / .1
    # Both positive gimbal commands produce NEGATIVE body-axis torque.
    gimbal_delta = (-torque_direction + torch.stack((bgx, bgy), -1)).clamp(-.35, .35)
    fin_delta = kf[:, None] * torch.stack((torque_direction[:, 1], -torque_direction[:, 0]), -1)
    fins_effective = velocity.norm(dim=-1) >= 2.
    fin_delta = fin_delta.clamp(-.1, .1) * fins_effective[:, None]
    yaw_delta = (-ky * raw[:, 2] / .1).clamp(-.1, .1) * fins_effective

    action = base.clone()
    action[:, :2] = base[:, :2] + fin_delta
    action[:, 2] = base[:, 2] + yaw_delta
    action[:, 5:7] = base[:, 5:7] + gimbal_delta
    neutral_throttle = (bu == 0) & (ks == 0) & (rate >= 80.)
    # Preserve bit identity rather than introducing a [-1,1] -> [0,1] roundtrip.
    action[:, 4] = torch.where(neutral_throttle, base[:, 4], 2. * throttle - 1.)
    pad_contact = raw[:, 29] > .5
    lit = raw[:, 24] > .5
    pulse_on = torch.where(lit, descent >= vc - .05, descent >= vc + .05) & ~pad_contact
    pulse_command = torch.where(pulse_on, 1., -1.)
    ignition = torch.where((ignition_mode == 3) & (height <= 2.), pulse_command, base[:, 3])
    ignition = torch.where((ignition_mode == 4) & pad_contact, -1., ignition)
    ignition = torch.where(ignition_mode == 1, -1., ignition)
    action[:, 3] = torch.where(ignition_mode == 2, 1., ignition)
    return action.clamp(-1., 1.).float()


def repair_features(raw_observation, parent_action):
    """30 observable features with fixed scales; no fitted stats or identifiers.

    Features follow FEATURE_NAMES.  Distances, speeds, rates and auxiliary
    channels use fixed physical scales; final clipping bounds unusual states
    for local distance calculations.  Braking margin and projected landing
    position are heuristic features, not reachability guarantees.
    """
    raw, q, velocity, height, mass = _state(raw_observation)
    if parent_action.shape != (len(raw), 7):
        raise ValueError("parent_action must have shape [N,7]")
    base = parent_action.float().clamp(-1., 1.)
    inverse_q = torch.cat((-q[:, :3], q[:, 3:4]), -1)
    upright = torch.zeros_like(velocity)
    upright[:, 2] = 1.
    upright_body = _rotate(inverse_q, upright)
    upright_error = torch.stack((-upright_body[:, 1], upright_body[:, 0]), -1)
    descent = (-velocity[:, 2]).clamp_min(0.)
    maximum_braking = (_MAX_THRUST / mass - _GRAVITY).clamp_min(1.)
    braking_margin = height - descent.square() / (2. * maximum_braking)
    time_to_ground = (height.clamp_min(0.) / descent.clamp_min(.3)).clamp(0., 5.)
    projected_xy = raw[:, 10:12] + velocity[:, :2] * time_to_ground[:, None]
    scaled_q = q * raw.new_tensor([5., 5., 5., 1.])
    values = torch.cat((
        (height.sign() * torch.log1p(height.abs()) / 4.)[:, None], height[:, None] / 10.,
        velocity / 2., raw[:, :3] / .1, scaled_q, raw[:, 10:12] / 2., upright_error / .1,
        raw[:, 25:26] / .05, raw[:, 26:27] / .4, raw[:, 27:29],
        raw[:, 24:25], raw[:, 29:30], base[:, 3:4], (base[:, 4:5] + 1.) * .5,
        base[:, 5:7], braking_margin[:, None] / 10., velocity.norm(dim=-1, keepdim=True),
        projected_xy.norm(dim=-1, keepdim=True) / 2., time_to_ground[:, None] / 5.,
    ), -1)
    return values.clamp(-25., 25.).float()
