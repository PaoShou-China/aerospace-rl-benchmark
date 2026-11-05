"""Unified parameter mismatch and domain randomization for all vehicle plants.

Two capabilities, both driven by env_kwargs so the evaluation/training code
stays unchanged:

* **param_mismatch** (static):  ``env_kwargs={"param_mismatch": {"mass": 0.2}}``
  scales the *plant's* mass up by 20 %.  The classical controller keeps its
  nominal value (hardcoded for atmospheric vehicles; passed through ``design``
  for spacecraft vehicles), so the controller operates on a model that
  deliberately differs from the true plant.

* **domain_randomize** (per-episode):
  ``env_kwargs={"domain_randomize": {"mass": (0.8, 1.2)}}`` samples a fresh
  scale factor uniformly in [0.8, 1.2] for each replica on every reset and
  applies it to the plant parameter.  The controller's nominal model stays
  fixed, so each episode is a different plant realization under the same
  assumed model.

Each env declares its perturbable parameters in ``PERTURBABLE_PARAMS`` (a dict
mapping parameter name -> SCALE placeholder, i.e. 1.0) and implements
``_apply_perturbation(name, scale)`` to scale the live plant tensor.  The
design model handed to controllers is kept separately in
``env.controller_design`` and stores the PHYSICAL nominal (thrust in N, mass
in kg, ...), resolved through ``env._nominal_param(name)`` (atmospheric plants
delegate to ``TensorPlant.nominal_param``); the 1.0 placeholder is never
written there.  The shared helpers below keep the sampling/normalization
logic in one place.
"""

import torch

from .dynamics import COMPUTE_DTYPE


def resolve_domain_randomize(spec, default):
    """Resolve the ``domain_randomize`` constructor argument.

    Rules (single source of truth — no silent fallback to a different range):

    * ``None`` or ``True`` or ``"default"`` -> use the env's built-in
      ``DEFAULT_DOMAIN_RANDOMIZE`` table (domain randomization ON).
    * ``False`` or ``{}`` -> disabled (nominal plant, benchmark contract).
    * ``dict`` -> use the caller's explicit ranges (merged over defaults so a
      partial override only changes the named keys).

    Returns the resolved range dict or ``None`` when disabled.
    """
    if spec is False or spec == {}:
        return None
    base = dict(default) if default else {}
    if spec is None or spec is True or spec == "default":
        return base or None
    if isinstance(spec, dict):
        merged = dict(base)
        merged.update(spec)
        return merged
    raise TypeError(f"unsupported domain_randomize spec: {spec!r}")


def parse_perturbation_spec(spec):
    """Normalize a mismatch/randomization spec into ``{name: (low, high)}``.

    Accepts:
      * ``{"mass": 0.2}``                       -> {"mass": (0.8, 1.2)}
      * ``{"mass": (0.85, 1.15)}``               -> {"mass": (0.85, 1.15)}
      * ``{"mass": {"low": 0.9, "high": 1.1}}``  -> {"mass": (0.9, 1.1)}
    """
    out = {}
    for name, val in (spec or {}).items():
        if isinstance(val, (int, float)):
            frac = float(val)
            out[name] = (1.0 - frac, 1.0 + frac)  # fraction -> symmetric interval
        elif isinstance(val, (tuple, list)) and len(val) == 2:
            out[name] = (float(val[0]), float(val[1]))
        elif isinstance(val, dict):
            out[name] = (float(val["low"]), float(val["high"]))
        else:
            raise TypeError(f"unsupported perturbation spec for {name!r}: {val!r}")
    return out


def sample_scales(spec, num_envs, device, seed, base_counter=0):
    """Sample one uniform scale factor per (replica, param) for one episode.

    Returns a dict ``{name: tensor(num_envs,)}`` of scale factors in the
    declared [low, high] range.  Deterministic given (seed, counter).
    """
    scales = {}
    g = torch.Generator(device="cpu")
    for i, (name, (lo, hi)) in enumerate(spec.items()):
        g.manual_seed(int(seed) * 1000003 + base_counter * 97 + i)
        u = torch.rand(num_envs, generator=g, device="cpu")
        scales[name] = (lo + (hi - lo) * u).to(COMPUTE_DTYPE).to(device)
    return scales


def apply_static_perturbation(env, mismatch_spec):
    """Apply a one-time static perturbation to all declared params.

    Spec forms (per param):
      * ``0.2``        -> scale = 1.2  (plant gets +20 %)
      * ``-0.1``       -> scale = 0.9  (plant gets -10 %)
      * ``1.3``        -> scale = 1.3  (explicit multiplier, >1 treated as scale)
      * ``(0.8, 1.2)`` -> scale = 1.0  (midpoint of the range)

    After this call the plant runs with perturbed parameters;
    ``env.controller_design`` holds the nominal values for the controller.
    """
    env._static_scales = {}
    for name, val in (mismatch_spec or {}).items():
        if name not in env.PERTURBABLE_PARAMS:
            raise KeyError(
                f"{type(env).__name__}: unknown perturbable param {name!r}; "
                f"known: {sorted(env.PERTURBABLE_PARAMS)}"
            )
        if isinstance(val, (int, float)):
            v = float(val)
            # abs(v) > 1 is treated as explicit multiplier (e.g. 1.3 = 1.3x);
            # small values are fractional deviations (0.2 -> +20%, -0.1 -> -10%).
            scale = v if abs(v) > 1.0 else 1.0 + v
        elif isinstance(val, (tuple, list)) and len(val) == 2:
            scale = (float(val[0]) + float(val[1])) / 2.0  # midpoint of a range spec
        else:
            raise TypeError(f"unsupported static spec for {name!r}: {val!r}")
        # Capture the nominal value BEFORE perturbing the plant so controllers
        # keep their design-model reference on the unmodified parameters.
        # ``PERTURBABLE_PARAMS`` holds only SCALE placeholders (1.0 for every
        # atmospheric plant), so it is the last-resort fallback: envs backed by
        # a ``TensorPlant`` expose their SI nominal through ``_nominal_param``.
        nominal = getattr(env, "_nominal_param", None)
        if callable(nominal):
            nominal = nominal(name)
        else:
            nominal = getattr(env, name, env.PERTURBABLE_PARAMS[name])
        if not hasattr(env, "controller_design") or env.controller_design is None:
            env.controller_design = {}
        env.controller_design[name] = nominal
        env._apply_perturbation(name, scale)
        env._static_scales[name] = scale


def as_env_mask(env, mask):
    """Normalize a reset argument to a ``(num_envs,)`` boolean mask on-device."""
    if mask is None:
        return None
    m = torch.as_tensor(mask, device=env.device).reshape(-1)
    if m.dtype == torch.bool:
        return m
    bool_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    bool_mask[m.to(torch.long)] = True
    return bool_mask


def apply_domain_randomization(env, randomize_spec, seed, counter, mask=None):
    """Per-episode: sample fresh scale factors and apply to the plant.

    ``mask`` (boolean ``(num_envs,)`` mask or an index tensor) restricts the
    resample to the replicas that actually reset; every other replica keeps
    the plant realization of the episode it is still running, so one replica's
    reset can never re-randomize its neighbours mid-episode.  ``None``
    re-randomizes all replicas (full ``reset()``).

    ``env._static_scales`` (from param_mismatch) are multiplied in so a static
    bias plus per-episode spread compose correctly.

    Returns the per-env, per-parameter scales actually applied now.
    """
    spec = parse_perturbation_spec(randomize_spec)
    scales = sample_scales(spec, env.num_envs, env.device, seed, counter)
    sel = as_env_mask(env, mask)
    applied = getattr(env, "_dr_scales", None)
    if applied is None:
        applied = {}
        env._dr_scales = applied
    for name, s in scales.items():
        if name not in env.PERTURBABLE_PARAMS:
            raise KeyError(
                f"{type(env).__name__}: unknown perturbable param {name!r}"
            )
        base = env._static_scales.get(name, 1.0)
        new = s * base
        if sel is not None:
            keep = applied.get(name)
            if keep is None:
                # nothing randomized yet for this param: the live plant runs on
                # the static perturbation (or the nominal value) for every env
                keep = torch.full_like(new, float(base))
            new = torch.where(sel, new, keep)
        applied[name] = new
        env._apply_perturbation(name, new)
    return dict(applied)
