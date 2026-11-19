"""Coupled short-horizon shooting through the frozen full physical operators.

Only the first 25 ms command is executable without replanning. The remaining
zero-order-hold plan is a non-anticipative prefix, not a touchdown guarantee.
The finite-difference Jacobian is detached: gradients through the target and
solve are a local decoder surrogate, not an exact derivative of an optimizer.
"""
from dataclasses import dataclass
import math
import torch
from frozen.core.phi_rocket_physics import PhiPhysics, rotation_increment, state_minus
from frozen.core.rocket_geometry import rotation_error
from .touchdown_intent import IntentBatch

CONTINUOUS = (0, 1, 2, 4, 5, 6)


def interval(physics, state, command, theta, seconds):
    # Differentiable interval lengths with a fixed conservative substep count.
    # PhiPhysics.forward converts dt to float; use its same flight/contact
    # composition here so the arrival-time coordinate has a gradient path.
    if not torch.isfinite(seconds).all() or bool((seconds<=0).any()) or bool((seconds>.3+1e-12).any()):
        raise ValueError('four-node prefix intervals must be in (0,0.3] seconds')
    # All legal arrival times are <=1.2 s. A universal grid prevents a batch
    # companion from changing this candidate's dynamics or uncertainty check.
    n = 24
    dt = seconds / n
    for _ in range(n):
        state = physics.contact(physics.flight(state, command, theta, dt), theta, dt)
    return state


def rollout(physics, state, controls, theta, arrival):
    states = []
    for k in range(4):
        state = interval(physics, state, controls[:, k], theta, arrival / 4)
        states.append(state)
    return torch.stack(states, 1)


def prefix_margins(states):
    """Conservative design targets in SI, separate from benchmark success."""
    s = states[:, -1]
    tilt_cos = 1 - 2 * (s[:, 6].square() + s[:, 7].square())
    margins = torch.stack((1.65 - s[:, :2].norm(dim=-1), 1. - s[:, 3:5].norm(dim=-1),
                        1. - s[:, 5].abs(), .3 - s[:, 10:13].norm(dim=-1),
                        tilt_cos - math.cos(.1), s[:, 20]), -1)
    return margins / s.new_tensor([1.65, 1., 1., .3, 1-math.cos(.1), .25])


@dataclass
class DecodedPlan:
    snapshot_id: str
    controls: torch.Tensor
    states: torch.Tensor
    cost: torch.Tensor
    margins: torch.Tensor
    arrival: torch.Tensor
    ignition: torch.Tensor
    raw_intent: torch.Tensor
    iterations: int


class TouchdownDecoder:
    def __init__(self, iterations=1, damping=.05):
        if iterations < 1 or damping <= 0:
            raise ValueError("positive decoder iterations/damping required")
        self.iterations, self.damping = iterations, damping
        self.physics = PhiPhysics().double().requires_grad_(False)

    def _residual(self, x, theta, controls, intent):
        states = rollout(self.physics, x, controls, theta, intent.arrival)
        previous = torch.cat((x[:, None], states[:, :-1]), 1)
        dt = intent.arrival[:, None, None] / 4
        acceleration = (states[:, :, 3:6] - previous[:, :, 3:6]) / dt
        qref = rotation_increment(intent.attitude)
        attitude_error = rotation_error(states[..., 6:10], qref)
        prev_attitude = torch.cat((torch.zeros_like(intent.attitude[:, :1]), intent.attitude[:, :-1]), 1)
        omega_reference = (intent.attitude - prev_attitude) / dt
        matching = torch.cat(((acceleration - intent.acceleration) / 5,
                              attitude_error * 4,
                              (states[..., 10:13] - omega_reference) * .3), -1).flatten(1)
        s = states[:, -1]
        terminal = torch.cat(((s[:, :3] - s.new_tensor([0., 0., 2.575])) * .5,
                               s[:, 3:6] * .5, s[:, 10:13],
                               rotation_error(s[:, 6:10], s.new_tensor([0., 0., 0., 1.]).expand(len(s), -1))), -1)
        smooth = (controls[:, 1:, list(CONTINUOUS)] - controls[:, :-1, list(CONTINUOUS)]).flatten(1) * .2
        effort = controls[:, :, list(CONTINUOUS)].flatten(1) * .02
        return torch.cat((matching, terminal, smooth, effort), -1), states

    def decode(self, snapshot, intent, ignition=None):
        if intent.snapshot_id != snapshot.fingerprint:
            raise ValueError("intent and decoder snapshot mismatch")
        raw = intent.raw
        self.physics = self.physics.to(device=raw.device, dtype=raw.dtype)
        n = len(raw)
        x = snapshot.state.to(raw).expand(n, -1)
        theta = snapshot.theta.to(raw).expand(n, -1)
        if ignition is None:
            ignition = torch.ones(n, 4, dtype=torch.bool, device=raw.device)
        if ignition.shape != (n, 4) or ignition.dtype != torch.bool:
            raise ValueError("each candidate needs an explicit four-node ignition schedule")
        controls = raw.new_zeros(n, 4, 7)
        controls[:, :, 3] = ignition.to(raw) * 2 - 1
        # A physical hover command is a warm start only; optimization consumes
        # every intent node and its result is the action returned to execution.
        mass = self.physics.mass_properties(x[:, 20:21], theta)[0][:, 0]
        thrust_gain = self.physics.values(theta)[:, 0]
        engine = (mass * 9.81 / (7607 * thrust_gain)).clamp(2966.7 / 7607, 1.)
        controls[:, :, 4] = ((engine - 2966.7 / 7607) / (1 - 2966.7 / 7607) * 2 - 1)[:, None]
        for _ in range(self.iterations):
            r, _ = self._residual(x, theta, controls, intent)
            # Vectorize all 24 actuator-coordinate perturbations. The same
            # theta/state snapshot is used throughout this coupled solve.
            with torch.no_grad():
                trial = controls[:, None].expand(-1, 24, -1, -1).clone()
                signed = []
                for j in range(24):
                    k, a = divmod(j, 6)
                    channel = CONTINUOUS[a]
                    delta = torch.where(controls[:, k, channel] > 0, -.005, .005)
                    trial[:, j, k, channel] += delta
                    signed.append(delta)
                repeated = IntentBatch(raw.detach()[:, None].expand(-1, 24, -1).reshape(-1, 25), intent.snapshot_id)
                rt, _ = self._residual(x[:, None].expand(-1, 24, -1).reshape(-1, 21),
                                       theta[:, None].expand(-1, 24, -1).reshape(-1, 67),
                                       trial.reshape(-1, 4, 7), repeated)
                J = ((rt.reshape(n, 24, -1) - r.detach()[:, None]) /
                     torch.stack(signed, 1)[..., None]).transpose(1, 2)
            normal = J.transpose(1, 2) @ J + self.damping * torch.eye(24, device=raw.device, dtype=raw.dtype)
            delta = -torch.linalg.solve(normal, J.transpose(1, 2) @ r[..., None])[..., 0]
            continuous = (controls[:, :, list(CONTINUOUS)] + delta.reshape(n, 4, 6)).clamp(-1, 1)
            controls = torch.cat((continuous[..., :3], (ignition.to(raw)*2-1)[..., None], continuous[..., 3:]), -1)
        residual, states = self._residual(x, theta, controls, intent)
        if not torch.isfinite(residual).all() or not torch.isfinite(controls).all():
            raise ValueError("nonfinite physical decode")
        return DecodedPlan(snapshot.fingerprint, controls, states, residual.square().mean(-1),
                           prefix_margins(states), intent.arrival, ignition, raw, self.iterations)
