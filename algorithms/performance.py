"""Opt-in execution optimizations; task equations and PPO stay in their owners.

TensorGraph is for fixed-shape, tensor-only functions (gradients opt in).
Outputs are borrowed until the next call. State buffers must keep their
addresses; the guard invalidates the capture when an owner replaces them.
"""
# PPO minibatch equations adapted from rsl-rl 5.3.0.
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (see third_party/rsl-rl-LICENSE.txt).
from __future__ import annotations

import torch


class TensorGraph:
    def __init__(self, fn, *, guard=lambda: (), allow_grad=False):
        self.fn = fn
        self.guard = guard
        self.signature = None
        self.graph = None
        self.warmups = 0
        self.allow_grad = allow_grad
        self.stream = None

    def __call__(self, *args):
        if (torch.is_grad_enabled() and not self.allow_grad) or not all(x.is_cuda for x in args):
            return self.fn(*args)
        signature = (tuple((tuple(x.shape), x.dtype, x.device) for x in args),
                     torch.is_inference_mode_enabled(), self.guard())
        if signature != self.signature:
            self.signature = signature
            self.graph = None
            self.warmups = 0
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=args[0].device)
        # Real warmup calls each advance the state exactly once.
        if self.warmups < 2:
            self.warmups += 1
            current = torch.cuda.current_stream()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                result = self.fn(*args)
            current.wait_stream(self.stream)
            return result
        if self.graph is None:
            self.inputs = tuple(x.clone() for x in args)
            torch.cuda.synchronize()
            inference = torch.is_inference_mode_enabled()
            # Graph/RNG bookkeeping must remain normal tensors when an env
            # graph (inference mode) and a backward graph share one generator.
            with torch.inference_mode(False):
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.stream):
                    with torch.inference_mode(inference):
                        self.outputs = self.fn(*self.inputs)
            # Capture records kernels; replay performs this call's state update.
        for dst, src in zip(self.inputs, args):
            dst.copy_(src)
        self.graph.replay()
        return self.outputs


def _guard(owner, names):
    def signature():
        return tuple((name, id(value), tuple(value.shape)) if torch.is_tensor(value) else (name, value)
                     for name in names if (value := getattr(owner, name, None)) is not None)
    return signature


class _PhysicsProfile:
    """Replay Torch segments; keep Genesis calls outside CUDA capture.

    Native kernels own temporary launch buffers and internal graphs. Capturing
    them as Torch kernels is unsafe. Public setters and scene.step therefore
    remain live calls between graphs, including all solver bookkeeping.
    """
    def __init__(self, plant, original, restore):
        self.plant = plant
        self.original = original
        self.restore = restore
        self.graphs = None
        self.warmups = 0
        self.stream = torch.cuda.Stream(device=plant.device)
        self.signature = None
        self.recording = None
        self.last_stream = None

    def _signature(self, args):
        p = self.plant
        buffers = tuple((k, id(v)) for owner in (p, p.rng, getattr(p, "surfaces", None),
                                                getattr(p, "rocket_variation", None))
                        if owner is not None for k, v in vars(owner).items() if torch.is_tensor(v))
        return (tuple((x.shape, x.dtype, x.device) for x in args),
                torch.is_inference_mode_enabled(), buffers)

    def _finish(self):
        self.recording.capture_end()
        self.recording.replay()
        self.graphs.append((self.recording, None, None))
        self.recording = None

    def _begin(self):
        # Normal tensors are needed for RNG bookkeeping shared with PPO graphs.
        with torch.inference_mode(False):
            self.recording = torch.cuda.CUDAGraph()
            self.recording.capture_begin()

    def _native(self, fn, args, kwargs):
        self._finish()
        result = self._run_native(fn, args, kwargs)
        self.graphs.append((fn, args, kwargs))
        self._begin()
        return result

    def _run_native(self, fn, args, kwargs):
        import quadrants as qd
        prog = qd.lang.impl.get_runtime().prog
        prog.set_current_cuda_stream(torch.cuda.current_stream(self.plant.device).cuda_stream)
        try:
            return fn(*args, **kwargs)
        finally:
            prog.set_current_cuda_stream(0)

    def _capture(self, args):
        p = self.plant
        self.graphs = []
        self.inputs = tuple(x.clone() for x in args)
        # Main caches getter results per physics frame. Captures must consume
        # owned pre-state buffers, not the particular frame cached at capture.
        self.state_inputs = tuple(x.clone() for x in p.state())
        p._state_cache = self.state_inputs
        self.index_caches = []  # graph kernels borrow these constant tensors
        replaced = []
        # Cache static Python index selections used by tensor-only getters.
        for entity in (p.drone, p.target):
            if entity is None:
                continue
            index_fn = entity._get_global_idx
            cache = {}
            self.index_caches.append(cache)
            def indices(idx, *a, _fn=index_fn, _cache=cache, **kw):
                if torch.is_tensor(idx):
                    return _fn(idx, *a, **kw)
                key = (str(idx), a, tuple(kw.items()))
                if key not in _cache:
                    recording = self.recording is not None
                    if recording:
                        self._finish()
                    value = _fn(idx, *a, **kw)
                    if isinstance(value, slice):
                        if recording:
                            self._begin()
                        return value
                    # Genesis converts one-element index arrays to a scalar.
                    # Keep that immutable scalar on the host; a CUDA tensor
                    # would force its .item() path to synchronize in capture.
                    index = torch.as_tensor(value, dtype=torch.int32)
                    _cache[key] = int(index.item()) if index.numel() == 1 else index.to(p.device)
                    if recording:
                        self._begin()
                return _cache[key]
            entity._get_global_idx = indices
            replaced.append((entity, "_get_global_idx", index_fn))
            for name in ("set_pos", "set_quat", "set_dofs_velocity", "apply_links_external_wrench",
                         "set_links_mass", "set_links_inertia", "set_links_COM"):
                if not hasattr(entity, name):
                    continue
                fn = getattr(entity, name)
                def boundary(*a, _fn=fn, **kw):
                    return self._native(_fn, a, kw)
                setattr(entity, name, boundary)
                replaced.append((entity, name, fn))
        fn = p.scene.step
        def scene_boundary(*a, **kw):
            return self._native(fn, a, kw)
        p.scene.step = scene_boundary
        replaced.append((p.scene, "step", fn))
        try:
            self._begin()
            self.original(*self.inputs[:2], restore=self.restore,
                          **({"target_state": self.inputs[2:]} if len(args) > 2 else {}))
            self._finish()
            self.state_outputs = p._state_cache
        finally:
            if self.recording is not None:
                try:
                    self.recording.capture_end()
                except RuntimeError:
                    pass  # Preserve the original capture error after cleanup.
                self.recording = None
            for owner, name, original in reversed(replaced):
                setattr(owner, name, original)

    def __call__(self, command, mask, target_state=None):
        current = torch.cuda.current_stream(command.device)
        if self.last_stream is not None and current != self.last_stream:
            current.wait_stream(self.last_stream)
        if torch.is_grad_enabled():
            self.last_stream = current
            return self._run_native(self.original, (command, mask),
                                    {"target_state": target_state, "restore": self.restore})
        args = (command, mask) + (() if target_state is None else tuple(target_state))
        signature = self._signature(args)
        if signature != self.signature:
            if self.last_stream is not None:
                self.last_stream.synchronize()  # Retire borrowed graph buffers before replacing them.
            self.signature = signature
            self.graphs = None
            self.warmups = 0
        self.last_stream = current
        if self.warmups < 2:
            self.warmups += 1
            self._run_native(self.original, (command, mask),
                             {"target_state": target_state, "restore": self.restore})
        elif self.graphs is None:
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                self._capture(args)
            current.wait_stream(self.stream)
        else:
            # Only capture needs a side stream. Replay on the caller stream so
            # native default-stream launches can avoid cross-stream barriers
            # and per-launch argument-buffer allocations.
            for dst, src in zip(self.state_inputs, self.plant.state()):
                dst.copy_(src)
            for dst, src in zip(self.inputs, args):
                dst.copy_(src)
            for operation, a, kw in self.graphs:
                if a is None:
                    operation.replay()
                else:
                    self._run_native(operation, a, kw)
            self.plant.physics_steps += 1
            self.plant._state_cache = self.state_outputs


class _PhysicsStep:
    """Keep main's pause restoration, contact cadence and frame-cache contract."""

    def __init__(self, plant):
        self.plant = plant
        self.original = plant.physics_step
        self.profiles = {}
        self.signature = None
        self.last_stream = None

    def __call__(self, command, mask, target_state=None, *, restore=True):
        p = self.plant
        current = torch.cuda.current_stream(command.device)
        if self.last_stream is not None and current != self.last_stream:
            current.wait_stream(self.last_stream)
        signature = (torch.is_inference_mode_enabled(),
                     p.contact_sample_every, p.physics_dt, p.vel_clamp, p.vehicle,
                     getattr(p, "f9_real", False))
        if signature != self.signature:
            if self.last_stream is not None:
                self.last_stream.synchronize()
            # The live cache may borrow outputs from a profile being retired.
            if self.profiles and p._state_cache is not None:
                p._state_cache = tuple(x.clone() for x in p._state_cache)
                current.synchronize()  # Complete copies before retiring graph pools.
            self.profiles.clear()
            self.signature = signature
        self.last_stream = current
        # physics_step increments its counter before deciding to read contacts.
        sample_contacts = (p.physics_steps + 1) % p.contact_sample_every == 0
        key = (bool(restore), sample_contacts)
        if key not in self.profiles:
            self.profiles[key] = _PhysicsProfile(p, self.original, bool(restore))
        return self.profiles[key](command, mask, target_state=target_state)


def _finish_environment(env, selected, compile_step, compile_aerodynamics):
    plant = getattr(env, "plant", None)
    # FP32 fusion changes aerodynamic rounding and drifts from main's native
    # trajectory. Segmented graphs still remove Torch launch overhead without
    # changing those arithmetic operations.
    if (compile_aerodynamics and plant is not None and hasattr(plant, "surfaces")
            and plant.dtype == torch.float64
            and "fused_aerodynamics" not in selected):
        if isinstance(plant.physics_step, _PhysicsStep):
            # A short evaluation may already have captured unfused segments.
            # Finish those calls and recapture after changing the aero owner.
            if plant.physics_step.last_stream is not None:
                plant.physics_step.last_stream.synchronize()
            plant.physics_step.signature = None
        plant.surfaces.step = torch.compile(plant.surfaces.step, fullgraph=True,
                                             options={"triton.cudagraphs": False})
        selected.append("fused_aerodynamics")
    # Whole-step fusion changes cancellation in FP32 CWH rewards. Preserve
    # their eager rounding; the integrator and PPO/GAE still use CUDA graphs.
    if (compile_step and getattr(env, "vehicle", None) in ("docking", "rendezvous")
            and env.p.dtype == torch.float64
            and not env.residual and "fused_cwh_step" not in selected):
        env._integrate = torch.compiler.disable(env._integrate)
        env.reset_idx = torch.compiler.disable(env.reset_idx)
        env.step = torch.compile(env.step, fullgraph=False, options={"triton.cudagraphs": False})
        selected.append("fused_cwh_step")
    env._cuda_performance_paths = tuple(selected)
    return selected


def enable_environment(env, *, compile_step=True, compile_aerodynamics=True):
    """Install once; short evaluations can defer costly compilation."""
    if torch.device(env.device).type != "cuda":
        raise ValueError("the performance backend requires CUDA")
    previous = vars(env).get("_cuda_performance_paths")
    if previous is not None:
        return _finish_environment(env, list(previous), compile_step, compile_aerodynamics)
    selected = []
    plant = getattr(env, "plant", None)
    if plant is None:
        names = ("q", "omega", "gimbal", "gimbal_rate", "dt", "substeps", "inertia", "rotor_h_dyn",
                 "rotor_axes_ref", "g_cross_r0", "gimbal_axes", "r", "v", "m", "f_max", "isp", "g0",
                 "mu", "p", "mass", "n", "fuel")
        env._integrate = TensorGraph(env._integrate, guard=_guard(env, names))
        # Keep parameter addresses stable when per-episode DR updates their values.
        apply = env._apply_perturbation
        def stable_perturbation(name, scale):
            old = getattr(env, name)
            apply(name, scale)
            new = getattr(env, name)
            if torch.is_tensor(old) and torch.is_tensor(new) and old.shape == new.shape:
                old.copy_(new)
                setattr(env, name, old)
        env._apply_perturbation = stable_perturbation
        env._skip_empty_resets = not env.residual
        selected.append("graphed_integrator")
    else:
        # For coincident authored/internal origins, the quaternion write does
        # not need the intermediate position FK. It performs the final FK.
        origins = getattr(plant.drone._solver, "_links_offset_pos_is_identity", None)
        if origins is not None and bool(origins[plant.drone.base_link_idx].all()):
            plant._defer_position_forward = True
            selected.append("deferred_position_forward")
        if hasattr(plant, "pid"):
            pid = plant.pid
            pid.step = TensorGraph(pid.step, guard=_guard(pid, ("integral", "previous", "kp", "ki", "kd", "limit", "control_hz")))
            selected.append("graphed_rate_pid")
        import genesis as gs
        import quadrants as qd
        if (gs.use_zerocopy and plant.scene.sim._rigid_only
                and not plant.scene.sim._requires_grad
                and hasattr(qd.lang.impl.get_runtime().prog, "set_current_cuda_stream")):
            plant.physics_step = _PhysicsStep(plant)
            selected.append("segmented_physics_step")
    return _finish_environment(env, selected, compile_step, compile_aerodynamics)


class _PPOUpdate:
    """Capture the upstream fixed-schedule MLP PPO minibatch, including Adam.

    The clipping, entropy, loss, sampling, separate actor/critic gradient
    clipping and minibatch order follow rsl-rl 5.3. Extensions use upstream PPO.
    """
    def __init__(self, alg):
        self.alg = alg
        self.step = TensorGraph(self._step, allow_grad=True, guard=self._guard)

    def _guard(self):
        a = self.alg
        return (tuple(g["lr"] for g in a.optimizer.param_groups), a.clip_param, a.value_loss_coef,
                a.entropy_coef, a.max_grad_norm, a.use_clipped_value_loss,
                a.normalize_advantage_per_mini_batch,
                tuple(id(getattr(model.obs_normalizer, "_std", None)) for model in (a.actor, a.critic)))

    def _step(self, obs_tensor, actions, old_log_prob, advantages, old_values, returns):
        from tensordict import TensorDict
        a = self.alg
        obs = TensorDict({"policy": obs_tensor}, batch_size=[obs_tensor.shape[0]])
        if a.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        a.actor(obs, stochastic_output=True)
        dist = a.actor.distribution
        self.distribution_state = (dist._distribution, dist._alpha, dist._beta)
        log_prob = a.actor.get_output_log_prob(actions)
        values = a.critic(obs)
        entropy = a.actor.output_entropy
        ratio = torch.exp(log_prob - old_log_prob.squeeze())
        surrogate = -advantages.squeeze() * ratio
        clipped = -advantages.squeeze() * ratio.clamp(1 - a.clip_param, 1 + a.clip_param)
        surrogate_loss = torch.maximum(surrogate, clipped).mean()
        if a.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(-a.clip_param, a.clip_param)
            value_loss = torch.maximum((values - returns).square(), (value_clipped - returns).square()).mean()
        else:
            value_loss = (returns - values).square().mean()
        loss = surrogate_loss + a.value_loss_coef * value_loss - a.entropy_coef * entropy.mean()
        a.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a.actor.parameters(), a.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(a.critic.parameters(), a.max_grad_norm)
        a.optimizer.step()
        return torch.stack((value_loss.detach(), surrogate_loss.detach(), entropy.detach().mean()))

    def __call__(self):
        a = self.alg
        total = torch.zeros(3, device=a.device)
        for batch in a.storage.mini_batch_generator(a.num_mini_batches, a.num_learning_epochs):
            total.add_(self.step(batch.observations["policy"], batch.actions, batch.old_actions_log_prob,
                                 batch.advantages, batch.values, batch.returns))
            # Graph replay updates captured tensors, but does not repeat Python
            # assignments. Restore the latest minibatch distribution for logging.
            dist = a.actor.distribution
            dist._distribution, dist._alpha, dist._beta = self.distribution_state
        means = (total / (a.num_mini_batches * a.num_learning_epochs)).tolist()
        a.storage.clear()
        return dict(zip(("value", "surrogate", "entropy"), means))


class _RolloutAct:
    def __init__(self, alg):
        self.alg = alg
        self.step = TensorGraph(self._step, guard=lambda: tuple(
            id(getattr(model.obs_normalizer, "_std", None)) for model in (alg.actor, alg.critic)))

    def _step(self, policy):
        from tensordict import TensorDict
        a = self.alg
        obs = TensorDict({"policy": policy}, batch_size=[policy.shape[0]])
        actions = a.actor(obs, stochastic_output=True).detach()
        values = a.critic(obs).detach()
        log_prob = a.actor.get_output_log_prob(actions).detach()
        params = tuple(x.detach() for x in a.actor.output_distribution_params)
        dist = a.actor.distribution
        self.distribution_state = (dist._distribution, dist._alpha, dist._beta)
        return actions, values, log_prob, *params

    def __call__(self, obs):
        outputs = self.step(obs["policy"])
        t = self.alg.transition
        t.hidden_states = (None, None)
        t.actions, t.values, t.actions_log_prob = outputs[:3]
        t.distribution_params = outputs[3:]
        t.observations = obs
        dist = self.alg.actor.distribution
        dist._distribution, dist._alpha, dist._beta = self.distribution_state
        return t.actions


def enable_ppo(alg):
    """Use CUDA graphs only for the validated, unextended fixed-schedule PPO."""
    from rsl_rl.algorithms import PPO
    if torch.device(alg.device).type != "cuda":
        return ["upstream_ppo_fallback"]
    if type(alg) is not PPO or alg.schedule != "fixed" or alg.rnd or alg.symmetry or alg.is_multi_gpu:
        return ["upstream_ppo_fallback"]
    if alg.actor.is_recurrent or alg.critic.is_recurrent:
        return ["upstream_ppo_fallback"]
    if alg.actor.obs_groups != ["policy"] or alg.critic.obs_groups != ["policy"]:
        return ["upstream_ppo_fallback"]
    if type(alg.optimizer) is not torch.optim.Adam:
        return ["upstream_ppo_fallback"]
    if type(alg.actor.distribution).__name__ != "BetaDistribution":
        return ["upstream_ppo_fallback"]
    for group in alg.optimizer.param_groups:
        group.update(fused=True, foreach=None, capturable=True)
    for param, state in alg.optimizer.state.items():
        if "step" in state:
            state["step"] = state["step"].to(param.device)
    for model in (alg.actor, alg.critic):
        norm = model.obs_normalizer
        if not hasattr(norm, "_std"):
            continue
        original = norm.update
        def stable_update(x, _norm=norm, _original=original):
            old = _norm._std
            _original(x)
            if _norm._std is not old:
                old.copy_(_norm._std)
                _norm._std = old
        if norm.until is None and norm.training:
            norm.update = TensorGraph(stable_update, guard=_guard(norm, ("_mean", "_var", "_std", "count", "training")))
        else:
            norm.update = stable_update
    alg.act = _RolloutAct(alg)
    alg.update = _PPOUpdate(alg)
    return ["graphed_ppo_minibatch", "graphed_rollout_policy", "graphed_normalizers", "fused_adam"]


class _Returns:
    def __init__(self, alg):
        self.alg = alg
        self.original = alg.compute_returns
        self.step = TensorGraph(self._step, guard=lambda: (
            alg.gamma, alg.lam, alg.normalize_advantage_per_mini_batch,
            tuple(id(getattr(alg.storage, k)) for k in ("rewards", "dones", "values", "returns")),
            id(getattr(alg.critic.obs_normalizer, "_std", None))))

    def _step(self, policy):
        from tensordict import TensorDict
        self.original(TensorDict({"policy": policy}, batch_size=[policy.shape[0]]))
        return self.alg.storage.advantages

    def __call__(self, obs):
        self.alg.storage.advantages = self.step(obs["policy"])


def enable_performance(env, runner):
    paths = enable_environment(env) + enable_ppo(runner.alg)
    if getattr(env, "vehicle", None) in ("docking", "rendezvous") and "graphed_ppo_minibatch" in paths:
        runner.alg.compute_returns = _Returns(runner.alg)
        paths.append("graphed_returns")
    # run_training keeps main's asynchronous episode logger and phase timings.
    return paths
