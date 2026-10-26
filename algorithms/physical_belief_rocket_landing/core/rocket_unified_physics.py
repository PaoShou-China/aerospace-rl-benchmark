"""Unified recurrent physical-belief actor with train-through-physics PPO.

The model owns causal recurrent belief, a weighted physical hypothesis set,
flight/contact operators, a shared ID/control optimizer, discrete ignition
branches, and the final policy distribution. The real simulator remains the
source of rewards and transitions. Contact is a learned compliant surrogate,
not a claim to duplicate the Genesis rigid-contact solver.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Bernoulli, Normal

from .rocket_physical_core import RocketFlightOperator, observation_state, rotate, inverse_rotate
from .rocket_phase_policy import features


class RocketContactOperator(RocketFlightOperator):
    parameter_names = RocketFlightOperator.parameter_names + (
        "contact_stiffness_gain", "contact_damping_gain", "friction_gain")

    def __init__(self):
        super().__init__()
        self.substeps = 1
        points = []
        # Four bottom box corners for each of the three authored landing legs.
        for x, y, yaw in ((0., .35, 0.), (.3031, -.175, 4.188), (-.3031, -.175, -4.188)):
            for dx in (-.025, .025):
                for dy in (-.25, .25):
                    points.append([x+math.cos(yaw)*dx-math.sin(yaw)*dy,
                                   y+math.sin(yaw)*dx+math.cos(yaw)*dy, -2.425])
        self.register_buffer("feet", torch.tensor(points))

    def external_wrench(self, p, v, q, w, theta, mass):
        arm = self.feet[None].expand(p.shape[0], -1, -1)
        orientation = q[:, None].expand(-1, len(self.feet), -1)
        world_arm = rotate(orientation, arm)
        points = p[:, None]+world_arm
        point_v = v[:, None]+rotate(orientation, torch.cross(w[:, None].expand_as(arm), arm, dim=-1))
        # The unchanged pad has radius 2m and top z=.15m; ground top is z=0.
        on_pad = points[..., :2].square().sum(-1) <= 4.
        gap = points[..., 2]-torch.where(on_pad, .15, 0.)
        penetration = nn.functional.softplus(-gap/.0005)*.0005
        active = torch.sigmoid(-gap/.0005)
        gains = theta[:, 9:12].clamp(-1.2, 1.2).exp()
        normal = 80000*gains[:, 0:1]*penetration
        normal = normal+500*gains[:, 1:2]*active*nn.functional.relu(-point_v[..., 2])
        # Implicit spring/damper effective response regularizes the 40Hz
        # internal model. Actual contacts remain owned by Genesis at 240Hz.
        dt = 1/(40*self.substeps)
        active_count = active.sum(-1,keepdim=True).clamp_min(1)
        denominator = 1+active_count*(dt*500*gains[:,1:2]+dt*dt*80000*gains[:,:1])/mass
        normal = normal/denominator
        # Smooth Coulomb sliding; no fictitious tangential impulse at rest.
        friction = -.6*gains[:, 2:3, None]*normal[..., None]*torch.tanh(point_v[..., :2]/.02)
        stopping_force = mass[:,None]/active_count[:,None]*point_v[...,:2].abs()/dt
        friction = friction.sign()*torch.minimum(friction.abs(),stopping_force)
        force_w = torch.cat((friction, normal[..., None]), -1)
        force_b = inverse_rotate(orientation, force_w)
        torque_b = torch.cross(arm, force_b, dim=-1)
        return force_b.sum(1), torque_b.sum(1)


class SharedPhysicalUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.types = nn.Embedding(19, 8)
        self.net = nn.Sequential(nn.Linear(28, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value, gradient, context, mode):
        width = value.shape[-1]
        types = self.types(torch.arange(width, device=value.device)+(0 if mode == "id" else 12))
        x = torch.cat((value[..., None], gradient.tanh()[..., None],
                       torch.log1p(gradient.abs())[..., None],
                       value.new_full((*value.shape, 1), float(mode == "control")),
                       context[:, None].expand(-1, width, -1),
                       types[None].expand(len(value), -1, -1)), -1)
        step = .001+.059*self.net(x).sigmoid().squeeze(-1)
        return value-step*gradient.tanh()


class UnifiedRocketActor(nn.Module):
    memory_width = 88  # recurrent hidden64, mean12, log-scale12

    def __init__(self, seed_policy, *, horizon=6, iterations=2, hypotheses=3):
        super().__init__()
        if hypotheses != 3:
            raise ValueError("this weighted cubature implementation has three hypotheses")
        self.seed_policy = seed_policy
        self.horizon, self.iterations, self.hypotheses = horizon, iterations, hypotheses
        self.physics = RocketContactOperator()
        self.history_encoder = nn.GRUCell(50, 64)
        self.belief_head = nn.Linear(64, 24)
        nn.init.normal_(self.belief_head.weight, std=.001)
        nn.init.zeros_(self.belief_head.bias)
        self.context = nn.Linear(64, 16)
        self.shared_update = SharedPhysicalUpdate()
        self.action_head = nn.Sequential(nn.Linear(109, 128), nn.SiLU(), nn.Linear(128, 128),
                                         nn.SiLU(), nn.Linear(128, 13))
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)
        self.log_std = nn.Parameter(torch.tensor([.01, .01, .01, .03, .01, .01]).log())
        self.physical_mix_logit = nn.Parameter(torch.tensor(-1.1))
        self.risk_logit = nn.Parameter(torch.tensor(-1.1))
        self.register_buffer("hypothesis_directions", torch.tensor([
            [0.]*12, [1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1.]]))
        self.register_buffer("continuous", torch.tensor([0, 1, 2, 4, 5, 6]))

    def initial_memory(self, count, device):
        memory = torch.zeros((count, self.memory_width), device=device)
        memory[:, 76:] = -2.3
        return memory

    def infer_belief(self, obs, memory, previous_state, executed_action, history_valid, *, create_graph=True):
        state = observation_state(obs)
        evidence = torch.cat((previous_state/self.physics.encoder_scale, executed_action,
                              (state-previous_state)/self.physics.error_scale,
                              history_valid[:, None].to(state.dtype)), -1).clamp(-20, 20)
        evidence = torch.where(history_valid[:, None], evidence, torch.zeros_like(evidence))
        hidden = self.history_encoder(evidence, memory[:, :64])
        raw = self.belief_head(hidden)
        prior, old_log_scale = memory[:, 64:76], memory[:, 76:]
        mean = (prior+.03*raw[:, :12].tanh()).clamp(-.7, .7)
        log_scale = (old_log_scale+.02*raw[:, 12:].tanh()).clamp(-4.6, -.7)
        hypotheses = (mean[:, None]+log_scale.exp()[:, None]*self.hypothesis_directions).clamp(-.7, .7)
        context = self.context(hidden).tanh()
        count = len(obs)
        safe_previous = torch.where(history_valid[:, None], previous_state, state)

        def energy(theta):
            predicted = self.physics(safe_previous[:, None].expand(-1, 3, -1),
                                     executed_action[:, None].expand(-1, 3, -1), theta)
            residual = self.physics.residual(predicted, state[:, None])
            likelihood = residual.square().clamp_max(100).mean(-1)
            # Do not identify from an auto-reset discontinuity or a zero history.
            likelihood = torch.where(history_valid[:, None], likelihood, torch.zeros_like(likelihood))
            prior_cost = .01*((theta-prior[:, None])/log_scale.exp()[:, None]).square().mean(-1)
            return likelihood+prior_cost, likelihood

        for _ in range(self.iterations):
            objective, _ = energy(hypotheses)
            gradient, = torch.autograd.grad(objective.sum(), hypotheses, create_graph=create_graph)
            proposed = self.shared_update(hypotheses.reshape(-1, 12), gradient.reshape(-1, 12),
                                          context[:, None].expand(-1, 3, -1).reshape(-1, 16), "id")
            hypotheses = proposed.reshape(count, 3, 12).clamp(-.7, .7)
        _, likelihood = energy(hypotheses)
        weights = (-.5*likelihood).softmax(-1)
        posterior = (weights[..., None]*hypotheses).sum(1)
        variance = (weights[..., None]*(hypotheses-posterior[:, None]).square()).sum(1)+.0001
        # Gradual evidence accumulation, with a floor for unexcited dimensions.
        posterior_log_scale = (.95*log_scale+.05*.5*variance.log()).clamp(-4.6, -.7)
        new_memory = torch.cat((hidden, posterior, posterior_log_scale), -1)
        return {"theta": hypotheses, "weights": weights, "mean": posterior,
                "log_scale": posterior_log_scale, "context": context,
                "memory": new_memory, "state": state, "fit": (weights*likelihood).sum(-1)}

    def control_cost(self, actions, state, theta, weights, reference):
        # Two ignition hypotheses share the same physical belief. Each branch
        # shares one action across all three physical hypotheses.
        n = len(state)
        s = state[:, None, None].expand(-1, 2, 3, -1).reshape(-1, 21)
        u = actions[:, :, None].expand(-1, -1, 3, -1).reshape(-1, 7)
        th = theta[:, None].expand(-1, 2, -1, -1).reshape(-1, 12)
        accumulated = torch.zeros((n*6,), device=state.device)
        for _ in range(self.horizon):
            s = self.physics(s, u, th)
            # Physical state costs are evaluated on each predicted hypothesis.
            h = (s[:, 2]-2.575).clamp_min(0)
            blend = ((h-2)/8).clamp(0, 1)
            blend = blend.square()*(3-2*blend)
            desired_down = (1-blend)*torch.minimum((24*h+1e-6).sqrt(), .8*h)+blend*(24*h+1e-6).sqrt().clamp_max(80)
            desired_xy = (-s[:, :2]/(2+h)[:, None]).clamp(-2, 2)
            target_v = torch.cat((desired_xy, -desired_down[:, None]), -1)
            near = torch.exp(-h/2)
            speed_cost = ((s[:, 3:6]-target_v)/(.2+2*(1-near))[:, None]).square().mean(-1)
            tilt = s[:, 6:8].square().sum(-1)
            task = speed_cost+20*tilt+(.1+near)*s[:, 10:13].square().sum(-1)
            task = task+.02*s[:, :2].square().sum(-1)+10*nn.functional.relu(2.54-s[:, 2]).square()
            accumulated = accumulated+task.clamp_max(1000)/self.horizon
        costs = accumulated.reshape(n, 2, 3)
        expected = (weights[:, None]*costs).sum(-1)
        risk = self.risk_logit.sigmoid()
        robust = (1-risk)*expected+risk*costs.max(-1).values
        return robust+3*(actions-reference).square().sum(-1)

    def forward(self, obs, memory, previous_state, executed_action, history_valid, *, create_graph=True):
        with torch.enable_grad():
            belief = self.infer_belief(obs, memory, previous_state, executed_action, history_valid,
                                       create_graph=create_graph)
            seed = self.seed_policy({"policy": obs})
            # Factual belief is conditioned on interaction, never on task loss.
            descriptor = torch.cat((features(obs), belief["memory"][:, :64].detach(), belief["mean"].detach()), -1)
            raw = self.action_head(descriptor)
            continuous_seed = seed[:, self.continuous].clamp(-1, 1)
            controls = continuous_seed[:, None]+.1*raw[:, 1:].reshape(-1, 2, 6).tanh()
            bits = obs.new_tensor([-1., 1.])[None, :, None].expand(len(obs), -1, -1)
            proposed = torch.cat((controls[..., :3], bits, controls[..., 3:]), -1).clamp(-1, 1)
            reference = proposed.detach()
            theta, weights = belief["theta"].detach(), belief["weights"].detach()
            context = belief["context"].detach()[:, None].expand(-1, 2, -1).reshape(-1, 16)
            mask = obs.new_tensor([1., 1., 1., 0., 1., 1., 1.])
            for _ in range(self.iterations):
                cost = self.control_cost(proposed, belief["state"], theta, weights, reference)
                gradient, = torch.autograd.grad(cost.sum(), proposed, create_graph=create_graph)
                adjusted = self.shared_update(proposed.reshape(-1, 7), gradient.reshape(-1, 7), context, "control")
                proposed = (proposed+(adjusted.reshape_as(proposed)-proposed)*mask).clamp(-1, 1)
            cost = self.control_cost(proposed, belief["state"], theta, weights, reference)
            near = torch.sigmoid((20-(obs[:, 12]-2.575))/3)
            blend = self.physical_mix_logit.sigmoid()*near
            means = controls+blend[:, None, None]*(proposed[:, :, self.continuous]-controls)
            logits = (12*seed[:, 3]).clamp(-8, 8)+raw[:, 0]+blend*(cost[:, 0]-cost[:, 1]).clamp(-4, 4)
            scales = self.log_std.clamp(-7, -1).exp().expand_as(means)
            return {"logits": logits, "means": means, "scales": scales,
                    "belief": belief, "model_control_cost": cost.mean(-1)}

    def prediction_loss(self, belief, action, next_obs, valid):
        predicted = self.physics(belief["state"][:, None].expand(-1, 3, -1),
                                 action[:, None].expand(-1, 3, -1), belief["theta"])
        target = observation_state(next_obs)
        weights = belief["weights"]
        expectation = (weights[..., None]*predicted).sum(1)
        residual = self.physics.residual(expectation, target)
        normalized_spread = ((predicted-expectation[:, None])/self.physics.error_scale).square()
        variance = (weights[..., None]*normalized_spread).sum(1)+.05
        mse = residual.square().clamp_max(100).mean(-1)
        nll = (.5*(residual.square()/variance+variance.log())).clamp(-10, 100).mean(-1)
        prior = .001*belief["mean"].square().mean(-1)
        loss = mse+.02*nll+prior
        return (loss*valid.float()).sum()/valid.sum().clamp_min(1)

    @staticmethod
    def action(output, deterministic=False):
        gate = (output["logits"] > 0).long() if deterministic else Bernoulli(logits=output["logits"]).sample().long()
        control = output["means"] if deterministic else Normal(output["means"], output["scales"]).sample()
        control = control[torch.arange(len(gate), device=gate.device), gate]
        return torch.cat((control[:, :3], (2*gate-1).to(control.dtype)[:, None], control[:, 3:]), -1)

    @staticmethod
    def log_prob(output, action):
        gate = (action[:, 3] > 0).long()
        continuous = torch.cat((action[:, :3], action[:, 4:]), -1)
        lp = Normal(output["means"], output["scales"]).log_prob(continuous[:, None]).sum(-1)
        return Bernoulli(logits=output["logits"]).log_prob(gate.float())+lp.gather(1, gate[:, None]).squeeze(1)

    @staticmethod
    def kl_divergence(old, new):
        """Exact joint KL for the ignition bit and its conditional controls."""
        old_probability = old["logits"].sigmoid()
        gate_kl = old_probability*(nn.functional.logsigmoid(old["logits"])
                                  -nn.functional.logsigmoid(new["logits"]))
        gate_kl += (1-old_probability)*(nn.functional.logsigmoid(-old["logits"])
                                       -nn.functional.logsigmoid(-new["logits"]))
        normal_kl = (new["scales"].log()-old["scales"].log()
                     +(old["scales"].square()+(old["means"]-new["means"]).square())
                     /(2*new["scales"].square())-.5).sum(-1)
        return (gate_kl+(1-old_probability)*normal_kl[:, 0]
                +old_probability*normal_kl[:, 1]).clamp_min(0)

    @staticmethod
    def entropy(output):
        probability = output["logits"].sigmoid()
        normal = Normal(output["means"], output["scales"]).entropy().sum(-1)
        return Bernoulli(logits=output["logits"]).entropy()+(1-probability)*normal[:, 0]+probability*normal[:, 1]


class PhysicalMemory:
    def __init__(self, actor, num_envs, device):
        self.actor = actor
        self.memory = actor.initial_memory(num_envs, device)
        self.previous_state = torch.zeros((num_envs, 21), device=device)
        self.executed_action = torch.zeros((num_envs, 7), device=device)
        self.history_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def inputs(self):
        return {key: getattr(self, key).detach().clone() for key in
                ("memory", "previous_state", "executed_action", "history_valid")}

    def advance(self, obs, action, new_memory, done):
        self.memory = torch.where(done[:, None], self.actor.initial_memory(len(done), done.device), new_memory.detach())
        self.previous_state = observation_state(obs).detach()
        self.executed_action = action.detach()
        self.history_valid = ~done
