"""Experimental train-through-physics flight core; not a qualified landing actor.

One differentiable free-flight operator is used for causal identification,
prediction, and action inference. One learned coordinate updater is shared
between identification and control. Contact and discrete ignition search are
deliberately not approximated by differentiable surrogates in this prototype.

Units are SI; the public observation quaternion is xyzw, angular velocity
and observed velocity are body-frame, and position is the authored base origin.
The internal state is [p_w(3), v_w(3), q_xyzw(4), omega_b(3), fins(4),
                      throttle(1), gimbals(2), fuel_fraction(1)].
"""
from __future__ import annotations

import math

import torch
from torch import nn


def rotate(q, v):
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def inverse_rotate(q, v):
    return rotate(torch.cat((-q[..., :3], q[..., 3:4]), -1), v)


def observation_state(obs):
    return torch.cat((obs[..., 10:13], rotate(obs[..., 3:7], obs[..., 7:10]),
                      obs[..., 3:7], obs[..., :3], obs[..., 20:24],
                      obs[..., 26:29], obs[..., 25:26]), -1)


class RocketFlightOperator(nn.Module):
    """Reduced rigid-body flight model with the repository's actuator/aero laws.

This is not Genesis: deterministic thrust uses the mean noise multiplier;
there is no collision solver, and Euler integration and variable-mass rigid
body transport can differ from the actual simulator. Learned coefficients
are effective corrections, not independently identified mass/thrust truth.
"""
    parameter_names = ("thrust_per_mass_gain", "drag_x_gain", "drag_y_gain",
                       "drag_z_gain", "transverse_inertia_gain", "axial_inertia_gain",
                       "fin_aerodynamic_gain", "engine_lag_gain", "gimbal_lag_gain")

    def __init__(self):
        super().__init__()
        self.substeps = 6
        self.register_buffer("fin_positions", torch.tensor([
            [.35, 0., 2.051], [-.35, 0., 2.051],
            [0., .35, 2.051], [0., -.35, 2.051]]))
        self.register_buffer("lift", torch.tensor([[0., 1., 0.], [0., 1., 0.],
                                                   [1., 0., 0.], [1., 0., 0.]]))
        self.register_buffer("forward_axis", torch.tensor([[0., 0., -1.]]).expand(4, -1).clone())
        self.register_buffer("drag", .5 * 1.225 * torch.tensor([1.16*1.7649, 1.16*1.7649, 2*.1075]))
        self.register_buffer("gravity", torch.tensor([0., 0., -9.81]))
        # Scales define loss coordinates, never change simulator units.
        self.register_buffer("error_scale", torch.tensor(
            [.04]*3 + [.15]*3 + [.01]*4 + [.03]*3 + [.02]*4 + [.02] + [.02]*2 + [.0001]))
        self.register_buffer("encoder_scale", torch.tensor(
            [50., 50., 450.] + [50.]*3 + [1.]*4 + [2.]*3 + [1.]*7 + [.05]))

    def coefficients(self, alpha, actuation):
        # The upstream PyFlyt lifting-surface formula (MIT), expressed without
        # mutable actuator state so gradients can traverse the actual rollout.
        slope = 6.283 / (1 + 2 * 5 / 3)
        delta = actuation * (math.pi / 4)
        dcl = slope * .65 * delta
        ap, an = math.radians(20), math.radians(-20)
        a0 = -dcl / slope
        clp, cln = slope * ap + dcl, slope * an + dcl
        cl = slope * (alpha - a0)
        effective = alpha - a0 - cl / math.pi
        ct = .01 * torch.cos(effective)
        cn = (cl + ct*torch.sin(effective)) / torch.cos(effective)
        cd = cn*torch.sin(effective) + ct*torch.cos(effective)
        cm = -cn * (.25 - .175*(1 - 2*effective/math.pi))
        aip = clp/math.pi * (1 - ((alpha-ap)/(math.pi/2-ap)).clamp(0, 1))
        ain = cln/math.pi * ((alpha+math.pi/2)/(an+math.pi/2)).clamp(0, 1)
        es = alpha - a0 - torch.where(alpha > 0, aip, ain)
        sn, cs = torch.sin(es), torch.cos(es)
        cd90 = -.0426*delta.square() + .21*delta + 1.98
        cns = cd90*sn*(1/(.56+.44*sn.abs()) - .41*(1-math.exp(-17)))
        cts = .005*cs
        cls, cds = cns*cs-cts*sn, cns*sn+cts*cs
        cms = -cns*(.25-.175*(1-2*es.abs()/math.pi))
        linear = (alpha > an) & (alpha < ap)
        return tuple(torch.where(linear, x, y) for x, y in ((cl, cls), (cd, cds), (cm, cms)))

    def forward(self, state, raw_action, theta):
        shape = state.shape
        s = state.reshape(-1, 21)
        a = raw_action.reshape(-1, 7).clamp(-1, 1)
        all_theta = theta.reshape(-1, theta.shape[-1])
        gain = all_theta[:, :9].clamp(-.7, .7).exp()
        p, v, q, w = s[:, :3], s[:, 3:6], s[:, 6:10], s[:, 10:13]
        fins, throttle, gimbal, fuel = s[:, 13:17], s[:, 17:18], s[:, 18:20], s[:, 20:21]
        fin_cmd = torch.stack((a[:, 1]+a[:, 2], a[:, 1]-a[:, 2],
                               a[:, 0]-a[:, 2], a[:, 0]+a[:, 2]), -1).clamp(-1, 1)
        ignition = (a[:, 3:4] > 0).to(s.dtype)
        minimum = 2966.7/7607
        target = ignition*(minimum + (a[:, 4:5]+1)*.5*(1-minimum))
        dt = 1 / (40*self.substeps)
        steps_per_interval = 6//self.substeps
        for _ in range(self.substeps):
            if self.substeps == 6:
                fins = fins + dt/.05 * (fin_cmd-fins)
                # TensorRandom.normal(mask, 1) has mean 1 in the frozen source,
                # so the multiplicative noise has mean 1.01, not 1.00.
                throttle = (throttle + dt/(.01*gain[:, 7:8])*(target-throttle)) * 1.01 * (fuel > 0)
                gimbal = gimbal + dt/(.01*gain[:, 8:9])*(a[:, 5:7]-gimbal)
                force_fins, force_throttle, force_gimbal = fins, throttle, gimbal
            else:
                # Exact endpoint and average of the original affine discrete
                # actuator recurrence over six 240Hz ticks. Rigid-body forces
                # use this average in a cheaper 40Hz prediction macro step.
                def affine_interval(value, recurrence, drive):
                    power = recurrence.pow(steps_per_interval)
                    steady = drive/(1-recurrence)
                    average_power = recurrence*(1-power)/(steps_per_interval*(1-recurrence))
                    return steady+power*(value-steady), steady+average_power*(value-steady)
                rf = fins.new_full((1,),1-1/240/.05)
                fins, force_fins = affine_interval(fins,rf,fin_cmd*(1-rf))
                rt = (1-1/240/(.01*gain[:, 7:8]))*1.01
                throttle, force_throttle = affine_interval(throttle,rt,target*(1/240/(.01*gain[:, 7:8]))*1.01)
                throttle, force_throttle = throttle*(fuel>0), force_throttle*(fuel>0)
                rg = 1-1/240/(.01*gain[:, 8:9])
                gimbal, force_gimbal = affine_interval(gimbal,rg,a[:,5:7]*(1-rg))
            fuel = (fuel-force_throttle*(1.451/410.9)*dt).clamp_min(0)
            mass = 138.2 + 410.9*fuel
            com_z = (-94. + .2*2.051) / mass
            rcom = torch.cat((torch.zeros_like(com_z).expand(-1, 2), com_z), -1)
            ixy = (753.8835702+1678*fuel-mass*com_z.square()) * gain[:, 4:5]
            iz = (2.3845+7.01*fuel) * gain[:, 5:6]
            inertia = torch.cat((ixy, ixy, iz), -1)
            vb = inverse_rotate(q, v)
            vf = vb[:, None] + torch.cross(w[:, None].expand(-1, 4, -1),
                                           self.fin_positions.expand(len(s), -1, -1), dim=-1)
            vl, va = (vf*self.lift).sum(-1), (vf*self.forward_axis).sum(-1)
            va = va + ((vl.square()+va.square()) < 1e-12).to(va.dtype)*1e-6
            alpha = torch.atan2(-vl, va)
            cl, cd, cm = self.coefficients(alpha, force_fins)
            qa = .5*1.225*.25*vf.square().sum(-1)
            normal = qa*(cl*torch.cos(alpha)+cd*torch.sin(alpha))
            parallel = qa*(cl*torch.sin(alpha)-cd*torch.cos(alpha))
            ff = (normal[..., None]*self.lift + parallel[..., None]*self.forward_axis) * gain[:, 6:7, None]
            ft = (qa*cm*.5)[..., None] * torch.cross(self.lift, self.forward_axis, dim=-1) * gain[:, 6:7, None]
            x, y = (force_gimbal*math.radians(5)).unbind(-1)
            direction = torch.stack((torch.sin(y), -torch.sin(x)*torch.cos(y),
                                     torch.cos(x)*torch.cos(y)), -1)
            thrust = direction*force_throttle*7607*gain[:, :1]
            drag = -self.drag*vb.abs()*vb*gain[:, 1:4]
            force = ff.sum(1)+thrust+drag
            booster_arm = torch.zeros_like(rcom)
            booster_arm[:, 2] = -2
            torque_origin = (torch.cross(self.fin_positions.expand_as(ff), ff, dim=-1)+ft).sum(1)
            torque_origin = torque_origin + torch.cross(booster_arm, thrust, dim=-1)
            extra_force, extra_torque = self.external_wrench(p, v, q, w, all_theta, mass)
            force = force+extra_force
            torque_origin = torque_origin+extra_torque
            torque = torque_origin-torch.cross(rcom, force, dim=-1)
            dw = (torque-torch.cross(w, inertia*w, dim=-1))/inertia
            transport = torch.cross(dw, rcom, dim=-1) + torch.cross(w, torch.cross(w, rcom, dim=-1), dim=-1)
            dv = rotate(q, force/mass-transport)+self.gravity
            quadrature = .5*(1+1/steps_per_interval)
            p = p+dt*(v+quadrature*dt*dv)
            v = v+dt*dv
            rotation_w = w+quadrature*dt*dw
            w = w+dt*dw
            dq = .5*torch.cat((q[:, 3:4]*rotation_w+torch.cross(q[:, :3], rotation_w, dim=-1),
                               -(q[:, :3]*rotation_w).sum(-1, keepdim=True)), -1)
            q = nn.functional.normalize(q+dt*dq, dim=-1)
        return torch.cat((p, v, q, w, fins, throttle, gimbal, fuel), -1).reshape(shape)

    def external_wrench(self, p, v, q, w, theta, mass):
        return torch.zeros_like(p), torch.zeros_like(p)

    def rollout(self, state, actions, theta):
        states = []
        for action in actions.unbind(1):
            state = self(state, action, theta)
            states.append(state)
        return torch.stack(states, 1)

    def residual(self, predicted, actual):
        # Quaternion antipodes describe identical rotations.
        actual = actual.expand_as(predicted)
        sign = torch.where((predicted[..., 6:10]*actual[..., 6:10]).sum(-1, keepdim=True) < 0, -1., 1.)
        aligned = torch.cat((actual[..., :6], actual[..., 6:10]*sign, actual[..., 10:]), -1)
        return (predicted-aligned)/self.error_scale


class SharedCoordinateUpdate(nn.Module):
    """The same learned optimizer weights for theta and action coordinates."""
    def __init__(self, hidden=48):
        super().__init__()
        self.types = nn.Embedding(16, 8)  # nine theta and seven action types
        self.net = nn.Sequential(nn.Linear(28, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value, gradient, context, *, mode):
        width = value.shape[-1]
        offset = 0 if mode == "id" else 9
        type_emb = self.types(torch.arange(width, device=value.device)+offset)
        type_emb = type_emb[None].expand(len(value), -1, -1)
        mode_feature = value.new_full((*value.shape, 1), 0. if mode == "id" else 1.)
        x = torch.cat((value[..., None], gradient.tanh()[..., None],
                       torch.log1p(gradient.abs())[..., None], mode_feature,
                       context[:, None].expand(-1, width, -1), type_emb), -1)
        step = .002 + .078*torch.sigmoid(self.net(x).squeeze(-1))
        return value-step*gradient.tanh()


class RocketPhysicalCore(nn.Module):
    """Causal initializer + shared learned inference + one physical operator.

theta is static over each window. Control cannot change factual theta, and
task gradients do not enter its initializer. The prototype does not provide
a calibrated posterior; expose causal fit errors instead of invented certainty.
"""
    def __init__(self, history=4, iterations=2):
        super().__init__()
        self.history, self.iterations = history, iterations
        self.physics = RocketFlightOperator()
        self.encoder = nn.Sequential(nn.Linear(history*(21+7+21), 96), nn.SiLU(),
                                     nn.Linear(96, 32), nn.SiLU())
        self.belief_seed = nn.Linear(32, 9)
        # A tiny nonzero seed also lets the causal encoder learn on update 1.
        nn.init.normal_(self.belief_seed.weight, std=.001)
        nn.init.zeros_(self.belief_seed.bias)
        self.context = nn.Linear(32, 16)
        self.update = SharedCoordinateUpdate()

    def history_energy(self, theta, before, actions, after):
        prediction = self.physics(before, actions, theta[:, None].expand(-1, before.shape[1], -1))
        return self.physics.residual(prediction, after).square().mean((1, 2)) + .002*theta.square().mean(-1)

    def identify(self, before, actions, after, *, create_graph=True):
        features = torch.cat((before/self.physics.encoder_scale, actions,
                              (after-before)/self.physics.error_scale), -1).clamp(-20, 20)
        hidden = self.encoder(features.flatten(1))
        context = self.context(hidden).tanh()
        theta = .5*torch.tanh(self.belief_seed(hidden))
        energies = []
        for _ in range(self.iterations):
            energy = self.history_energy(theta, before, actions, after)
            energies.append(energy)
            gradient, = torch.autograd.grad(energy.sum(), theta, create_graph=create_graph)
            theta = self.update(theta, gradient, context, mode="id").clamp(-.7, .7)
        final = self.history_energy(theta, before, actions, after)
        return {"theta": theta, "context": context, "history_energy": final,
                "initial_history_energy": energies[0], "uncertainty_calibrated": False}

    def predict(self, current_state, executed_actions, belief):
        return self.physics.rollout(current_state, executed_actions, belief["theta"])

    def control_energy(self, action, state, theta, seed, horizon=2):
        sequence = action[:, None].expand(-1, horizon, -1)
        future = self.physics.rollout(state, sequence, theta)[:, -1]
        h = (state[:, 2]-2.575).clamp_min(0)
        desired_v = torch.cat(((-.4*state[:, :2]).clamp(-5, 5),
                               -torch.sqrt(4*h+.0001).clamp_max(30)[:, None]), -1)
        accel = ((desired_v-state[:, 3:6])/.8).clamp(-20, 20)
        target_v = state[:, 3:6]+accel*(horizon/40)
        task = ((future[:, 3:6]-target_v)/.5).square().mean(-1)
        task = task + .1*future[:, 10:13].square().sum(-1)
        return task + 3*(action-seed).square().sum(-1)

    def control(self, state, seed_action, belief, *, create_graph=True):
        theta, context = belief["theta"].detach(), belief["context"].detach()
        seed = seed_action.detach().clamp(-1, 1)
        action = seed.clone().requires_grad_(True)
        before = self.control_energy(action, state, theta, seed)
        mask = action.new_tensor([1., 1., 1., 0., 1., 1., 1.])
        for _ in range(self.iterations):
            energy = self.control_energy(action, state, theta, seed)
            gradient, = torch.autograd.grad(energy.sum(), action, create_graph=create_graph)
            proposed = self.update(action, gradient, context, mode="control")
            action = (action+(proposed-action)*mask).clamp(-1, 1)
        after = self.control_energy(action, state, theta, seed)
        return {"action": action, "initial_energy": before, "energy": after,
                "theta_used": theta, "discrete_ignition_optimized": False}

    def forward(self, history_before, history_actions, history_after,
                current_state, seed_action, executed_prediction_actions, *, create_graph=True):
        belief = self.identify(history_before, history_actions, history_after, create_graph=create_graph)
        prediction = self.predict(current_state, executed_prediction_actions, belief)
        control = self.control(current_state, seed_action, belief, create_graph=create_graph)
        return {"belief": belief, "prediction": prediction, "control": control}
