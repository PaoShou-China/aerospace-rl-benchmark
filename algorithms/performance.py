"""Opt-in execution optimizations; task equations and PPO stay in their owners.

TensorGraph is for fixed-shape, tensor-only functions (gradients opt in).
Outputs are borrowed until the next call. State buffers must keep their
addresses; the guard invalidates the capture when an owner replaces them.

SegmentGraph extends this to task code that also calls Genesis: Quadrants
kernels are recorded into the same CUDA graph (``_native_capture``), and the
host effects a replay skips are audited at capture (see its docstring).
``configure_backend`` selects Genesis' performance mode, which the whole-step
capture requires. Environment variables: AEROSPACE_GENESIS_PERFORMANCE,
AEROSPACE_NATIVE_CAPTURE, AEROSPACE_SEGMENT_GRAPHS (default on),
AEROSPACE_FUSE (Inductor fusion, default off), AEROSPACE_SEGMENT_DEBUG.
"""
# PPO minibatch equations adapted from rsl-rl 5.3.0.
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (see third_party/rsl-rl-LICENSE.txt).
from __future__ import annotations

import contextlib
import importlib

import torch

# Quadrants launch contexts created while a native capture is open. Each launch
# copies its host-side argument buffer into a graph memcpy node, which re-reads
# that host memory on every replay, so the context must outlive the graph.
_RETAINED_LAUNCH_CONTEXTS = None


def _install_launch_context_retention():
    from quadrants._lib import core

    if getattr(_install_launch_context_retention, "installed", False):
        return
    original = core.KernelCxx.make_launch_context

    def make_launch_context(self):
        context = original(self)
        if _RETAINED_LAUNCH_CONTEXTS is not None:
            _RETAINED_LAUNCH_CONTEXTS.append(context)
        return context

    core.KernelCxx.make_launch_context = make_launch_context
    _install_launch_context_retention.installed = True


@contextlib.contextmanager
def _native_capture(stream, retained, solver=None):
    """Route Quadrants launches into ``stream`` (being captured) as plain kernels.

    Quadrants' own graph kernels cannot be launched into a capturing stream, so
    they run as their individual tasks for the capture only; the captured
    kernels are the ones the eager graph would launch. The errno check is a
    host read and is replayed by the caller at Genesis' own cadence.
    """
    global _RETAINED_LAUNCH_CONTEXTS
    import quadrants as qd
    qd_kernel = importlib.import_module("quadrants.lang.kernel")

    _install_launch_context_retention()
    prog = qd.lang.impl.get_runtime().prog
    graph_enabled = qd_kernel._GRAPH_ENABLED
    check_errno = None if solver is None else solver.check_errno
    qd_kernel._GRAPH_ENABLED = False
    _RETAINED_LAUNCH_CONTEXTS = retained
    if solver is not None:
        solver.check_errno = lambda: None
    prog.set_current_cuda_stream(stream.cuda_stream)
    try:
        yield
    finally:
        prog.set_current_cuda_stream(0)
        if solver is not None:
            del solver.check_errno  # restore the class method
            assert solver.check_errno.__func__ is check_errno.__func__
        _RETAINED_LAUNCH_CONTEXTS = None
        qd_kernel._GRAPH_ENABLED = graph_enabled


class TensorGraph:
    def __init__(self, fn, *, guard=lambda: (), allow_grad=False):
        self.fn = fn
        self.guard = guard
        self.signature = None
        self.graph = None
        self.warmups = 0
        self.allow_grad = allow_grad
        self.stream = None

    @torch.compiler.disable
    def __call__(self, *args):
        if (torch.is_grad_enabled() and not self.allow_grad) or not all(x.is_cuda for x in args):
            return self.fn(*args)
        if torch.cuda.is_current_stream_capturing():
            return self.fn(*args)  # an enclosing segment graph records these kernels
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


_PINNED_CONSTANTS = []  # host sources of constants copied inside captures


def _constant_to_device(cpu_tensor, device):
    """Copy a host constant to the device, also while a stream is capturing.

    Outside capture this is a plain copy. Inside, the source is pinned and kept
    alive: the recorded memcpy node re-reads it on every replay.
    """
    if not torch.cuda.is_current_stream_capturing():
        return cpu_tensor.to(device)
    pinned = cpu_tensor.contiguous().pin_memory()
    _PINNED_CONSTANTS.append(pinned)
    out = torch.empty(pinned.shape, dtype=pinned.dtype, device=device)
    out.copy_(pinned, non_blocking=True)
    return out


def _sanitize_on_host(original, index, expected_size, max_size, dim, name):
    """Run Genesis' index sanitizer with its device pinned to the CPU."""
    import genesis as gs

    device = gs.device
    gs.device = torch.device("cpu")
    try:
        return original(index, expected_size, max_size, dim, name)
    finally:
        gs.device = device


def _segment_graphs_enabled():
    import os

    return os.environ.get("AEROSPACE_SEGMENT_GRAPHS", "1") != "0"


def _rate_check_errno():
    from genesis.engine.simulator import RATE_CHECK_ERRNO

    return RATE_CHECK_ERRNO


def _native_capture_supported(plant):
    """Whole-step capture needs a host-loop-free solve (the monolith arm).

    The decomposed arm iterates ``graph_do_while`` on the host below SM 9.0,
    reading a device flag per iteration, which no stream capture can hold.
    """
    import os

    if os.environ.get("AEROSPACE_NATIVE_CAPTURE", "1") == "0":
        return False
    try:
        qd_kernel = importlib.import_module("quadrants.lang.kernel")
        from quadrants._lib import core
    except ImportError:
        return False
    solver = plant.scene.sim.rigid_solver
    config = (getattr(solver, "_static_rigid_sim_config", None)
              or getattr(solver, "rigid_config", None))
    prefer = getattr(config, "prefer_decomposed_solver", None)
    if prefer is None and isinstance(config, dict):
        prefer = config.get("prefer_decomposed_solver")
    return (hasattr(qd_kernel, "_GRAPH_ENABLED") and hasattr(core.KernelCxx, "make_launch_context")
            and prefer == 0)


class _PhysicsProfile:
    """Replay Torch segments; keep Genesis calls outside CUDA capture.

    Fallback for generic (ndarray) Genesis kernels, whose decomposed solve
    loops on the host: public setters and scene.step remain live calls
    between graphs, including all solver bookkeeping. Whole-substep capture
    (``_native_capture_supported``) uses a :class:`SegmentGraph` instead.
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
                if dst is not src:
                    dst.copy_(src)
            for dst, src in zip(self.inputs, args):
                if dst is not src:
                    dst.copy_(src)
            for operation, a, kw in self.graphs:
                if a is None:
                    operation.replay()
                else:
                    self._run_native(operation, a, kw)
            self.plant.physics_steps += 1
            self.plant._state_cache = self.state_outputs


def _install_capture_caches(plant, replaced):
    """Plant-level caches for Genesis' host-to-device index conversions.

    Getters/setters called with Python index lists convert them to device
    tensors on every call, a host-to-device copy of a temporary that a capture
    cannot record. Eager calls fill these caches, so captures read the cached
    constants; a miss inside a capture copies from pinned memory kept alive.
    ``replaced`` collects (owner, name, original) to restore afterwards.
    """
    import genesis.utils.misc as gs_misc

    caches = vars(plant).setdefault("_capture_index_caches", {})
    for entity in (plant.drone, plant.target):
        if entity is None:
            continue
        index_fn = entity._get_global_idx
        cache = caches.setdefault(id(entity), {})
        def indices(idx, *a, _fn=index_fn, _cache=cache, **kw):
            if torch.is_tensor(idx):
                return _fn(idx, *a, **kw)
            key = (str(idx), a, tuple(kw.items()))
            if key not in _cache:
                value = _fn(idx, *a, **kw)
                if isinstance(value, slice):
                    return value
                # Genesis converts one-element index arrays to a scalar. Keep
                # that immutable scalar on the host; a CUDA tensor would force
                # its .item() path to synchronize in capture.
                index = torch.as_tensor(value, dtype=torch.int32)
                _cache[key] = (int(index.item()) if index.numel() == 1
                               else _constant_to_device(index, plant.device))
            return _cache[key]
        entity._get_global_idx = indices
        replaced.append((entity, "_get_global_idx", index_fn))

    sanitized = caches.setdefault("sanitize_index", {})
    original = gs_misc.sanitize_index

    def sanitize_index(index, expected_size, max_size, dim, name, _original=original):
        if not isinstance(index, (int, tuple, list, range, type(None))):
            return _original(index, expected_size, max_size, dim, name)
        value = tuple(index) if isinstance(index, (tuple, list, range)) else index
        key = (type(index).__name__, value, expected_size, max_size, dim, name)
        value = sanitized.get(key)
        if value is None:
            if torch.cuda.is_current_stream_capturing():
                cpu = _sanitize_on_host(_original, index, expected_size, max_size, dim, name)
                value = _constant_to_device(cpu, plant.device)
            else:
                value = _original(index, expected_size, max_size, dim, name)
            sanitized[key] = value
        return value

    gs_misc.sanitize_index = sanitize_index
    replaced.append((gs_misc, "sanitize_index", original))


class _PhysicsStep:
    """Keep main's pause restoration, contact cadence and frame-cache contract."""

    def __init__(self, plant):
        self.plant = plant
        self.original = plant.physics_step
        self.profiles = {}
        self.signature = None
        self.last_stream = None
        # Whole-substep capture (Genesis kernels included) when the solve has
        # no host loop; otherwise the segmented profiles below.
        self.segment = None
        if _native_capture_supported(plant):
            original = self.original

            def physics_step(command, mask, restore, *target_state):
                original(command, mask, target_state or None, restore=restore)

            self.segment = SegmentGraph(physics_step, None, plant=plant, name="physics_step")

    @torch.compiler.disable
    def __call__(self, command, mask, target_state=None, *, restore=True):
        p = self.plant
        if torch.cuda.is_current_stream_capturing():
            # An enclosing segment graph records the substep itself.
            return self.original(command, mask, target_state, restore=restore)
        if self.segment is not None:
            return self.segment(command, mask, bool(restore), *(target_state or ()))
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


_SCALARS = (bool, int, float, str, type(None))
_SEGMENT_DEBUG = __import__("os").environ.get("AEROSPACE_SEGMENT_DEBUG", "0") == "1"


def _watched_owners(roots):
    """Project objects reachable from ``roots`` whose attributes task code mutates."""
    seen = {}
    stack = [r for r in roots if r is not None]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen[id(obj)] = obj
        for value in vars(obj).values():
            module = type(value).__module__ or ""
            if (module.startswith(("environments.", "algorithms.")) and module != __name__
                    and hasattr(value, "__dict__") and not isinstance(value, type)):
                stack.append(value)
    return list(seen.values())


def _snapshot(owners):
    snap = {}
    for k, obj in enumerate(owners):
        for name, value in vars(obj).items():
            if torch.is_tensor(value) or isinstance(value, _SCALARS):
                snap[(k, name)] = value
            elif isinstance(value, tuple) and value and all(torch.is_tensor(x) for x in value):
                snap[(k, name)] = value
            elif isinstance(value, dict):
                for key, item in value.items():
                    if torch.is_tensor(item):
                        snap[(k, name, key)] = item
    return snap


def _same(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return a is b
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(x is y for x, y in zip(a, b))
    return type(a) is type(b) and a == b


class SegmentGraph:
    """Capture one segment of task code -- Torch and Genesis kernels -- as a CUDA graph.

    Contract for ``fn``: it updates persistent tensors in place and returns
    None, a tensor or a tuple of tensors (static outputs, borrowed until the
    next call). Python arguments that are not tensors select separate graphs.

    Host effects a replay does not repeat are audited at capture. Integer
    counters the capture advanced are advanced by the same amount after
    every replay; the plant frame cache, simulator substep counter and
    Genesis' periodic errno check are replayed explicitly. Any other host
    effect (a tensor attribute rebound instead of updated in place, a float
    or flag changed) keeps the segment eager from then on.
    """

    # Rebound by the eager step code between segments, never read inside one.
    _VOLATILE = frozenset({"_state_cache", "reset_buf"})

    def __init__(self, fn, env, *, plant=None, name=None):
        self.fn = fn
        self.env = env
        self.plant = plant
        self.name = name or getattr(fn, "__name__", "segment")
        self.native = plant is not None and _native_capture_supported(plant)
        self.graphs = {}
        self.disabled = None
        self.stream = None
        self.owners = None
        self.recaptures = 0
        self.unstable = 0

    # -- bookkeeping ---------------------------------------------------------
    def _guard(self):
        snap = _snapshot(self.owners)
        return tuple((key, id(v) if torch.is_tensor(v) else v) for key, v in snap.items()
                     if len(key) == 2 and key[1] not in self._VOLATILE
                     and key[1] not in self.counters
                     and not (isinstance(v, int) and not isinstance(v, bool))
                     and not isinstance(v, tuple))

    def _install_caches(self, replaced):
        if self.plant is not None and self.native:
            _install_capture_caches(self.plant, replaced)

    @torch.compiler.disable
    def __call__(self, *args):
        if self.disabled is not None or torch.is_grad_enabled():
            return self.fn(*args)
        if torch.cuda.is_current_stream_capturing():
            return self.fn(*args)
        tensors = [a for a in args if torch.is_tensor(a)]
        if any(not t.is_cuda for t in tensors):
            return self.fn(*args)
        if self.owners is None:
            self.owners = _watched_owners([self.env, self.plant])
            self.counters = set()
        key = (tuple((tuple(a.shape), a.dtype) if torch.is_tensor(a) else ("py", a) for a in args),
               torch.is_inference_mode_enabled(),
               # physics_step samples contacts on a host-counted cadence
               None if self.plant is None
               else self.plant.physics_steps % self.plant.contact_sample_every)
        entry = self.graphs.get(key)
        guard = self._guard()
        if entry is not None and entry["guard"] != guard:
            if entry.get("graph") is None:
                self.unstable += 1
                if self.unstable == 4:
                    old = dict(entry["guard"])
                    changed = [k for k, v in guard if old.get(k, object()) != v][:6]
                    print(f"[performance] segment {self.name}: state it reads keeps being replaced "
                          f"between calls {changed}; it runs eagerly until that stops", flush=True)
            if _SEGMENT_DEBUG:
                old = dict(entry["guard"])
                changed = [k for k, v in guard if old.get(k, object()) != v][:6]
                print(f"[performance] segment {self.name}: guard changed {changed}", flush=True)
            if entry.get("graph") is not None:
                torch.cuda.current_stream().synchronize()
                self.recaptures += 1
                if self.recaptures in (8, 64):
                    changed = [k for k, v in zip(entry["guard"], guard) if k != v][:4]
                    print(f"[performance] segment {self.name}: recaptured {self.recaptures}x; "
                          f"changed {changed}", flush=True)
            entry = None
        if entry is None:
            entry = self.graphs[key] = {"guard": guard, "warmups": 0, "graph": None}
        if entry["warmups"] < 2:
            entry["warmups"] += 1
            # Warm up on the static input buffers the capture will record, so
            # the capture sees exactly the tensors (and tensor kinds) warmed.
            inputs = self._stage_inputs(entry, args)
            replaced = []
            self._install_caches(replaced)
            # The second warmup is audited like the capture (first-call lazy
            # allocations are done by then): replaced state disables the segment.
            before = _snapshot(self.owners) if entry["warmups"] == 2 else None
            try:
                result = self.fn(*inputs)
            finally:
                for owner, attr, original in reversed(replaced):
                    setattr(owner, attr, original)
            if before is not None:
                unsafe = self._audit(before, _snapshot(self.owners))[1]
                if unsafe:
                    self._disable(unsafe, "host state changed in warmup")
                    return result
            entry["guard"] = self._guard()
            return result
        if entry["graph"] is None:
            return self._capture(entry, args)
        return self._replay(entry, args)

    def _stage_inputs(self, entry, args):
        if "inputs" not in entry:
            entry["inputs"] = tuple(a.clone() if torch.is_tensor(a) else a for a in args)
            if self.plant is not None:
                entry["state_in"] = tuple(x.clone() for x in self.plant.state())
        for dst, src in zip(entry["inputs"], args):
            if torch.is_tensor(dst) and dst is not src:
                dst.copy_(src)
        p = self.plant
        if p is not None:
            for dst, src in zip(entry["state_in"], p.state()):
                if dst is not src:
                    dst.copy_(src)
            p._state_cache = entry["state_in"]
        return entry["inputs"]

    def _audit(self, before, after):
        """Split host-state changes into replayable counters and the rest."""
        counters, unsafe = {}, []
        for k in set(before) | set(after):
            a, b = before.get(k), after.get(k)
            if k in before and k in after and _same(a, b):
                continue
            if len(k) == 2 and k[1] in self._VOLATILE:
                continue
            if (isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool)
                    and not isinstance(b, bool)):
                counters[k] = b - a
                self.counters.add(k[1])
                continue
            unsafe.append(k)
        return counters, unsafe

    def _disable(self, keys, reason):
        names = sorted({".".join(map(str, (type(self.owners[k[0]]).__name__,) + k[1:]))
                        for k in keys})
        print(f"[performance] segment {self.name} stays eager: {reason}: {names[:6]}", flush=True)
        self.disabled = names

    def _capture(self, entry, args):
        p = self.plant
        sim = p.scene.sim if p is not None else None
        solver = sim.rigid_solver if sim is not None else None
        if (solver is not None and solver.is_active
                and sim._cur_substep_global % _rate_check_errno() == 0):
            solver.check_errno()
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        inputs = self._stage_inputs(entry, args)
        state_in = entry.get("state_in")
        before = _snapshot(self.owners)
        substep = sim._cur_substep_global if sim is not None else 0
        retained = []
        replaced = []
        self._install_caches(replaced)
        inference = torch.is_inference_mode_enabled()
        self.stream.wait_stream(current)
        try:
            with torch.cuda.stream(self.stream):
                context = (_native_capture(self.stream, retained, solver) if self.native
                           else contextlib.nullcontext())
                with context:
                    with torch.inference_mode(False):
                        graph = torch.cuda.CUDAGraph()
                        graph.capture_begin(capture_error_mode="thread_local")
                    try:
                        with torch.inference_mode(inference):
                            outputs = self.fn(*inputs)
                    finally:
                        with torch.inference_mode(False):
                            graph.capture_end()
        except Exception as error:  # noqa: BLE001 - any capture failure falls back
            # Nothing ran on the device; undo the host effects of the partial
            # call and run it eagerly instead.
            for k, value in before.items():
                owner = self.owners[k[0]]
                if len(k) == 2:
                    setattr(owner, k[1], value)
                else:
                    getattr(owner, k[1])[k[2]] = value
            if sim is not None:
                sim._cur_substep_global = substep
            if p is not None:
                p._state_cache = state_in
            torch.cuda.synchronize()
            self.disabled = [f"capture failed: {type(error).__name__}: {str(error)[:200]}"]
            print(f"[performance] segment {self.name} stays eager: {self.disabled[0]}", flush=True)
            return self.fn(*args)
        finally:
            for owner, attr, original in reversed(replaced):
                setattr(owner, attr, original)
        current.wait_stream(self.stream)
        counters, unsafe = self._audit(before, _snapshot(self.owners))
        entry.update(graph=graph, outputs=outputs, retained=retained,
                     counters=[(self.owners[k[0]], k[1], d) for k, d in counters.items()],
                     substeps=(sim._cur_substep_global - substep) if sim is not None else 0)
        if p is not None:
            entry["state_out"] = p._state_cache
        # The capture call's Python effects already happened; this replay
        # performs its device work.
        graph.replay()
        if unsafe:
            self._disable(unsafe, "host state changed in capture")
            entry["graph"] = None
        entry["guard"] = self._guard()
        return outputs

    def _replay(self, entry, args):
        for dst, src in zip(entry["inputs"], args):
            if torch.is_tensor(dst) and dst is not src:
                dst.copy_(src)
        p = self.plant
        if p is not None:
            for dst, src in zip(entry["state_in"], p.state()):
                if dst is not src:
                    dst.copy_(src)
            sim = p.scene.sim
            solver = sim.rigid_solver
            if entry["substeps"] and solver.is_active:
                rate = _rate_check_errno()
                start = sim._cur_substep_global
                per_step = max(int(sim._substeps), 1)
                if any((start + k) % rate == 0 for k in range(0, entry["substeps"], per_step)):
                    solver.check_errno()
        entry["graph"].replay()
        for owner, attr, delta in entry["counters"]:
            setattr(owner, attr, getattr(owner, attr) + delta)
        if p is not None:
            p.scene.sim._cur_substep_global += entry["substeps"]
            p._state_cache = entry["state_out"]
        return entry["outputs"]


# Genesis entity calls a fused (Inductor) region must leave to the engine.
_GENESIS_ENTITY_CALLS = (
    "apply_links_external_wrench", "get_ang", "get_contacts", "get_links_mass", "get_links_quat",
    "get_links_vel", "get_pos", "get_quat", "get_vel", "set_dofs_velocity", "set_links_COM",
    "set_links_inertia", "set_links_mass", "set_pos", "set_quat",
)


def _fusion_enabled():
    import os

    return os.environ.get("AEROSPACE_FUSE", "0") == "1"


def _fuse_task_math(env, plant):
    """Fuse the task's elementwise Torch math into Triton kernels (Inductor).

    Genesis calls stay engine calls (graph breaks); the fused kernels are then
    recorded by the segment graphs like any other kernel. Fusion contracts
    multiply-adds and may reorder reductions, so results agree with the
    unfused path to float32 rounding rather than bitwise.
    """
    options = dict(dynamic=False, options={"triton.cudagraphs": False})
    # Variants (restore/contact flags, inference vs normal tensors) are all
    # compiled during warmups; allow them before Dynamo falls back to eager.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    torch._dynamo.config.accumulated_cache_size_limit = max(
        torch._dynamo.config.accumulated_cache_size_limit, 512)
    for entity in (plant.drone, plant.target):
        if entity is None:
            continue
        for name in _GENESIS_ENTITY_CALLS:
            fn = getattr(entity, name, None)
            if fn is not None:
                setattr(entity, name, torch.compiler.disable(fn))
    plant.scene.step = torch.compiler.disable(plant.scene.step)
    plant._physics_step = torch.compile(plant._physics_step, **options)
    plant.controller = torch.compile(plant.controller, **options)
    for name in ("_observe", "_potential"):
        fn = getattr(env, name, None)
        if callable(fn):
            setattr(env, name, torch.compile(fn, **options))


def _fuse_tensor_task_math(env):
    """Inductor fusion for the tensor-only spacecraft tasks (see ``_fuse_task_math``)."""
    options = dict(dynamic=False, options={"triton.cudagraphs": False})
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    for name in ("_integrate", "_observe", "_potential", "_manipulability", "_attitude_error"):
        fn = getattr(env, name, None)
        if callable(fn):
            setattr(env, name, torch.compile(fn, **options))


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


def configure_backend():
    """Select process-global backend options; call before the first env exists.

    Genesis performance mode compiles static-array kernels specialised to the
    scene. For these single-body vehicles that selects the monolith constraint
    solve, which runs without the per-iteration host loop the generic
    decomposed solve needs below SM 9.0, and makes whole-substep CUDA-graph
    capture possible. Solver results agree to float32 rounding, not bitwise.
    Set ``AEROSPACE_GENESIS_PERFORMANCE=0`` to keep the generic kernels.
    """
    import os

    from environments.common import original_backend

    if os.environ.get("AEROSPACE_GENESIS_PERFORMANCE", "1") == "0":
        original_backend.GENESIS_PERFORMANCE_MODE = False
        return {"genesis_performance_mode": False}
    try:
        import genesis as gs
    except ImportError:
        return {"genesis_performance_mode": False}
    if gs._initialized:
        return {"genesis_performance_mode": not gs.use_ndarray}
    original_backend.GENESIS_PERFORMANCE_MODE = True
    return {"genesis_performance_mode": True}


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
        segments = _segment_graphs_enabled() and not env.residual
        integrate = env._integrate
        if _fusion_enabled():
            _fuse_tensor_task_math(env)
            integrate = env._integrate
            selected.append("fused_task_math")
        # Segment graphs record the integrator with the rest of the step.
        env._integrate = integrate if segments else TensorGraph(integrate, guard=_guard(env, names))
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
        if segments:
            names = [n for n in ("_step_core", "_reset_replicas")
                     if callable(getattr(env, n, None))]
            for name in names:
                setattr(env, name, SegmentGraph(getattr(env, name), env, name=name))
            if names:
                selected.append("graphed_step_segments")
    else:
        # For coincident authored/internal origins, the quaternion write does
        # not need the intermediate position FK. It performs the final FK.
        origins = getattr(plant.drone._solver, "_links_offset_pos_is_identity", None)
        if origins is not None and bool(origins[plant.drone.base_link_idx].all()):
            plant._defer_position_forward = True
            selected.append("deferred_position_forward")
        segments = (_segment_graphs_enabled() and _native_capture_supported(plant)
                    and not getattr(env, "residual", False))
        if hasattr(plant, "pid") and not segments:
            # Segment graphs record the PID with the rest of the control step.
            pid = plant.pid
            pid.step = TensorGraph(pid.step, guard=_guard(pid, ("integral", "previous", "kp", "ki", "kd", "limit", "control_hz")))
            selected.append("graphed_rate_pid")
        import genesis as gs
        import quadrants as qd
        if not gs.use_ndarray:
            selected.append("genesis_performance_mode")
        if _fusion_enabled():
            _fuse_task_math(env, plant)
            selected.append("fused_task_math")
        if (gs.use_zerocopy and plant.scene.sim._rigid_only
                and not plant.scene.sim._requires_grad
                and hasattr(qd.lang.impl.get_runtime().prog, "set_current_cuda_stream")):
            plant.physics_step = _PhysicsStep(plant)
            selected.append("native_graphed_physics_step" if _native_capture_supported(plant)
                            else "segmented_physics_step")
            if segments:
                segments = [name for name in ("_step_begin", "_control_step", "_step_end")
                            if callable(getattr(env, name, None))]
                if len(segments) == 3:
                    for name in segments:
                        segment = SegmentGraph(getattr(env, name), env, plant=plant, name=name)
                        setattr(env, name, segment)
                    if callable(getattr(env, "_episode_stats", None)):
                        env._episode_stats = SegmentGraph(env._episode_stats, env,
                                                          name="_episode_stats")
                    if callable(getattr(env, "_reset_replicas", None)):
                        env._reset_replicas = SegmentGraph(env._reset_replicas, env, plant=plant,
                                                           name="_reset_replicas")
                    selected.append("graphed_step_segments")
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
