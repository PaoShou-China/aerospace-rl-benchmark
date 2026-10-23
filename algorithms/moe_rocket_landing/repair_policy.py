"""Distil simulator-discovered feedback programs into an observation-only gate.

Search and labels are training-only.  At execution the frozen parent supplies
the base action, and a learned classifier can commit to one finite intervention.
The gate never receives seeds, termination labels, or future observations.
"""
from __future__ import annotations

import torch
from torch import nn

from feedback_program import FEATURE_DIM, feedback_action, repair_features


class RepairGate(nn.Module):
    def __init__(self, programs: int, width: int = 96):
        super().__init__()
        self.programs, self.width = programs, width
        self.register_buffer("mean", torch.zeros(FEATURE_DIM))
        self.register_buffer("scale", torch.ones(FEATURE_DIM))
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(),
                                     nn.Linear(width, programs + 1))

    def forward(self, features):
        return self.network(((features - self.mean) / self.scale).clamp(-8, 8))

    def fit(self, features, labels, *, seed=719007, updates=800):
        """Positive labels are actual successful interventions; 0 means retain.

        Background label 0 on old successful paths is a preservation objective,
        not an assertion that every alternative would fail on those states.
        """
        if len(features) != len(labels) or not len(features):
            raise ValueError("nonempty aligned feature/label tensors required")
        features = features.to(self.mean.device).float()
        labels = labels.to(features.device).long()
        self.mean.copy_(features.mean(0))
        self.scale.copy_(features.std(0, unbiased=False).clamp_min(.1))
        positive = (labels > 0).nonzero(as_tuple=False).flatten()
        negative = (labels == 0).nonzero(as_tuple=False).flatten()
        if not len(positive) or not len(negative):
            raise ValueError("both repaired and retention examples are required")
        generator = torch.Generator(device=features.device).manual_seed(seed)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=.01)
        self.train().requires_grad_(True)
        losses = []
        for iteration in range(updates):
            p = positive[torch.randint(len(positive), (64,), generator=generator,
                                       device=features.device)]
            n = negative[torch.randint(len(negative), (192,), generator=generator,
                                       device=features.device)]
            indices = torch.cat((p, n))
            loss = nn.functional.cross_entropy(self(features[indices]), labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.)
            optimizer.step()
            if iteration % 100 == 0 or iteration == updates - 1:
                losses.append(float(loss.detach()))
        self.eval().requires_grad_(False)
        with torch.no_grad():
            probability = self(features).softmax(-1)
            predicted = probability.argmax(-1)
            # Calibrate only on training retention states.  This is not a
            # guarantee on entire episodes or on unobserved states.
            maximum = probability[:, 1:].max(-1).values
            retention_quantile = float(torch.quantile(maximum[negative], .999))
            threshold = max(.65, min(.98, retention_quantile))
            return {"updates": updates, "loss_samples": losses,
                    "positive_examples": len(positive), "retention_examples": len(negative),
                    "positive_class_accuracy": float((predicted[positive] == labels[positive]).float().mean()),
                    "retention_accuracy": float((predicted[negative] == 0).float().mean()),
                    "threshold": threshold,
                    "threshold_source": "training-only 99.9th percentile of retention scores, clipped [.65,.98]"}


class CounterfactualRepairPolicy(nn.Module):
    """One observation-triggered, finite closed-loop intervention per episode."""
    def __init__(self, parent, gate, parameters, modes, durations, centers,
                 *, threshold=.75, support_radius=1.5):
        super().__init__()
        self.parent, self.gate = parent, gate
        self.register_buffer("parameters_library", parameters)
        self.register_buffer("modes", modes.long())
        self.register_buffer("durations", durations.long())
        self.register_buffer("centers", centers)
        self.threshold, self.support_radius = float(threshold), float(support_radius)
        for key, dtype in (("selected", torch.long), ("remaining", torch.long),
                           ("used", torch.bool)):
            self.register_buffer(key, torch.empty(0, dtype=dtype), persistent=False)
        self.last_trigger = None
        self.eval().requires_grad_(False)

    def _allocate(self, count, device):
        if self.used.shape != (count,) or self.used.device != device:
            self.selected = torch.zeros(count, dtype=torch.long, device=device)
            self.remaining = torch.zeros_like(self.selected)
            self.used = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, done_mask=None):
        self.parent.reset(done_mask)
        if done_mask is None:
            self.selected.zero_(); self.remaining.zero_(); self.used.zero_()
        elif self.used.numel():
            self.selected.masked_fill_(done_mask, 0)
            self.remaining.masked_fill_(done_mask, 0)
            self.used.masked_fill_(done_mask, False)

    def apply_to_parent(self, observation, parent_action, enabled=None):
        self._allocate(len(observation), observation.device)
        if enabled is None:
            enabled = torch.ones_like(self.used)
        self.last_trigger = None
        available = ~self.used & enabled & (observation[:, 12] - 2.575 <= 150.)
        if self.gate.programs and bool(available.any()):
            ids = available.nonzero(as_tuple=False).flatten()
            features = repair_features(observation[ids], parent_action[ids])
            probabilities = self.gate(features).softmax(-1)
            confidence, choice = probabilities[:, 1:].max(-1)
            distance = ((features - self.centers[choice]).square().mean(-1)).sqrt()
            trigger = ((confidence >= self.threshold) & (confidence > probabilities[:, 0])
                       & (distance <= self.support_radius))
            chosen_ids = ids[trigger]
            if len(chosen_ids):
                programs = choice[trigger]
                self.selected[chosen_ids] = programs
                self.remaining[chosen_ids] = self.durations[programs]
                self.used[chosen_ids] = True
                self.last_trigger = {"indices": chosen_ids.clone(), "features": features[trigger].clone(),
                                     "program": programs.clone(), "confidence": confidence[trigger].clone(),
                                     "distance": distance[trigger].clone()}
        active = (self.remaining > 0) & enabled
        result = parent_action.clone()
        if bool(active.any()):
            programs = self.selected[active]
            result[active] = feedback_action(observation[active], parent_action[active],
                                            self.parameters_library[programs], self.modes[programs])
        self.remaining.sub_(1).clamp_min_(0)
        return result

    def forward(self, obs):
        parent_action = self.parent(obs)
        return self.apply_to_parent(obs["policy"], parent_action)

    act = forward


def load_repair_policy(payload, observations):
    """Restore a self-contained checkpoint into a ready, episode-aware policy.

    Load the saved experiment's immutable module versions for exact replay.
    Call policy.reset(done_mask) after each natural environment reset.
    """
    import copy
    from expert_router import FrozenExpertSelector, RidgeAdvantageGate
    from phase_policy import RocketPhaseModel

    if payload["format"] != "rocket_counterfactual_repair_v1":
        raise ValueError("unsupported counterfactual repair checkpoint")
    if payload.get("activation_height", 150.) != 150.:
        raise ValueError("this experiment preserves the measured 150 m activation ceiling")
    device = observations["policy"].device
    parent_payload = payload["parent_payload"]
    actors = []
    for config, weights in zip(parent_payload["expert_configs"], parent_payload["expert_actor_state_dicts"]):
        config = copy.deepcopy(config)
        kwargs = config["actor"]
        kwargs.pop("class_name")
        std = kwargs.get("distribution_cfg", {}).get("init_std")
        if isinstance(std, list):
            kwargs["distribution_cfg"]["init_std"] = torch.tensor(std, device=device)
        actor = RocketPhaseModel(observations, config["obs_groups"], "actor", 7, **kwargs).to(device)
        actor.load_state_dict(weights, strict=True)
        actors.append(actor.eval().requires_grad_(False))
    parent_gate = RidgeAdvantageGate().to(device)
    parent_gate.load_state_dict(parent_payload["gate_state_dict"])
    parent = FrozenExpertSelector(*actors, parent_gate)
    gate = RepairGate(payload["gate_programs"], payload["gate_width"]).to(device)
    gate.load_state_dict(payload["gate_state_dict"], strict=True)
    policy = CounterfactualRepairPolicy(parent, gate, payload["parameters"].to(device),
        payload["modes"].to(device), payload["durations"].to(device), payload["centers"].to(device),
        threshold=payload["threshold"], support_radius=payload["support_radius"],
        )
    return policy.eval().requires_grad_(False)
