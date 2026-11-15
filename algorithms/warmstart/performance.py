"""Optional execution adapter for the existing BC minibatch equations."""
import torch

from algorithms.performance import TensorGraph


def prepare_collection(env):
    from environments import ENV_REGISTRY
    from algorithms.performance import enable_environment

    if (torch.device(env.device).type == "cuda"
            and type(env) in {entry[0] for entry in ENV_REGISTRY.values()}):
        return enable_environment(env, compile_step=False)
    return []


def make_bc_step(model, optimizer, *, normalize, critic=False, weighted=False):
    from rsl_rl.models import MLPModel
    from rsl_rl.modules import EmpiricalNormalization

    norm = model.obs_normalizer
    if (type(model) is not MLPModel or model.is_recurrent or model.obs_groups != ["policy"]
            or type(norm) not in (EmpiricalNormalization, torch.nn.Identity)
            or getattr(norm, "until", None) is not None
            or next(model.parameters()).device.type != "cuda"):
        return None
    for group in optimizer.param_groups:
        group.update(fused=True, foreach=None, capturable=True)
    return _BCStep(model, optimizer, normalize=normalize, critic=critic, weighted=weighted)


class _BCStep:
    def __init__(self, model, optimizer, *, normalize, critic, weighted):
        self.model, self.optimizer = model, optimizer
        self.normalize, self.critic, self.weighted = normalize, critic, weighted
        self.step = TensorGraph(self._step, allow_grad=True, guard=self._guard)

    def _guard(self):
        m, norm = self.model, self.model.obs_normalizer
        return (tuple(id(x) for x in m.parameters()), tuple(id(x) for x in m.buffers()),
                norm.training, getattr(norm, "eps", None),
                tuple((g["lr"], g["betas"], g["eps"], g["weight_decay"]) for g in self.optimizer.param_groups))

    def _step(self, observations, targets, *weights):
        model, norm = self.model, self.model.obs_normalizer
        if self.normalize:
            old_std = getattr(norm, "_std", None)
            model.update_normalization({"policy": observations})
            if old_std is not None and norm._std is not old_std:
                old_std.copy_(norm._std)
                norm._std = old_std
        prediction = model({"policy": observations})
        if self.critic:
            prediction = prediction.squeeze(-1)
        if self.weighted:
            per_pair = (prediction - targets).pow(2).mean(dim=-1)
            loss = (weights[0] * per_pair).sum() / weights[0].sum().clamp_min(1e-12)
        else:
            loss = torch.nn.functional.mse_loss(prediction, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def __call__(self, *inputs):
        # Borrowed until the next minibatch; the caller immediately accumulates
        # it on the current stream before the next replay can overwrite it.
        return self.step(*inputs)
