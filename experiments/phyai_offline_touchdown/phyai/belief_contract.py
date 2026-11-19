"""Typed, detached facts at one decision time; covariance uses core tangent units."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import torch
from frozen.core.phi_rocket_types import NJ, NP, ND, STATE_SCALE

MODES = ("flight", "contact", "slide", "bounce", "support")


def _copy(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")
    return value.detach().clone()


@dataclass(frozen=True, init=False)
class BeliefSnapshot:
    _state: torch.Tensor
    _theta: torch.Tensor
    _covariance: torch.Tensor
    _evidence_json: str
    version: int
    clock: float
    modes: tuple[str, ...]
    model_id: str
    history_id: str

    def __init__(self, state, theta, covariance, *, version, clock, modes, model_id, history_id, evidence=None):
        x = _copy(state, (21,), "state")
        c = _copy(theta, (NP + ND,), "theta (61 physical + 6 disturbance = 67)")
        p = _copy(covariance, (NJ, NJ), "covariance")
        if any(t.dtype != x.dtype or t.device != x.device for t in (c, p)):
            raise ValueError("snapshot tensors must share dtype and device")
        if not torch.allclose(p, p.T, atol=1e-8, rtol=1e-6):
            raise ValueError("covariance must be symmetric")
        if torch.linalg.eigvalsh(p).min() < -1e-8:
            raise ValueError("covariance must be positive semidefinite")
        if abs(float(x[6:10].norm()) - 1) > 1e-5:
            raise ValueError("state must use a unit xyzw quaternion")
        if c.abs().max() > 1:
            raise ValueError("normalized physical parameters must lie in [-1,1]")
        if not isinstance(version, int) or version < 0 or not math.isfinite(clock) or clock < 0:
            raise ValueError("invalid belief version/clock")
        if not modes or not set(modes) <= set(MODES) or not model_id or not history_id:
            raise ValueError("explicit modes and model/history identifiers required")
        data = dict(_state=x, _theta=c, _covariance=p, version=version, clock=float(clock),
                    modes=tuple(modes), model_id=model_id, history_id=history_id,
                    _evidence_json=json.dumps(evidence or {"qualified": False, "representation": "unavailable"},
                                              sort_keys=True, allow_nan=False))
        for k, v in data.items():
            object.__setattr__(self, k, v)

    @property
    def state(self): return self._state.clone()

    @property
    def theta(self): return self._theta.clone()

    @property
    def covariance(self): return self._covariance.clone()

    @property
    def evidence(self): return json.loads(self._evidence_json)

    @property
    def factual_valid(self):
        e = self.evidence
        if e.get('source') == 'frozen_filter':
            return (e.get('assumption_invalid') is False and e.get('numerical_valid') is True
                    and e.get('history_valid') is True)
        return not (e.get('assumption_invalid') is True or e.get('numerical_valid') is False)

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        for t in (self._state, self._theta, self._covariance):
            h.update(t.cpu().contiguous().numpy().tobytes())
        h.update(json.dumps([self.version, self.clock, self.modes, self.model_id, self.history_id,
                             self._evidence_json], sort_keys=True).encode())
        return h.hexdigest()

    def features(self):
        # Include every covariance entry, including state/parameter cross terms.
        scale = self._state.new_tensor([10]*3 + [5]*3 + [1]*4 + [1]*3 + [1]*8)
        mode = self._state.new_tensor([m in self.modes for m in MODES])
        return torch.cat((self._state / scale, self._theta, self._covariance.flatten(), mode)).detach()

    @classmethod
    def from_filter(cls, belief, index=0, **metadata):
        fields = ('lower','upper','evidence_matrix','evidence_radius','evidence_center',
                  'evidence_window','evidence_valid','assumption_invalid','assumption_invalid_latched',
                  'numerical_valid','history_valid','possible_modes','mode_probability','mode',
                  'contact_observed','contact_age','evidence_rejected','transition_inconsistent',
                  'filtered_state','planning_state','prequential_error','innovation_nis')
        evidence = {k: belief[k][index].detach().cpu().tolist() if k in belief else None for k in fields}
        evidence.update(qualified=False,source='frozen_filter',selected_state='state',
                        representation='legacy evidence window and local slab; missing fields are explicit nulls')
        for key in ('version','clock'):
            if key in belief:
                value = belief[key][index].item()
                if key in metadata and metadata[key] != value:
                    raise ValueError(f'{key} disagrees with factual filter output')
                metadata[key] = int(value) if key=='version' else float(value)
        if evidence['possible_modes'] is not None:
            possible = evidence['possible_modes']
            if len(possible) != len(MODES):
                raise ValueError('possible modes must use the five-mode registry')
            modes = tuple(m for m,valid in zip(MODES,possible) if valid)
            if 'modes' in metadata and tuple(metadata['modes']) != modes:
                raise ValueError('modes disagree with factual filter output')
            metadata['modes'] = modes
        return cls(belief["state"][index], torch.cat((belief["mean"][index], belief["disturbance"][index])),
                   belief["covariance"][index], evidence=evidence, **metadata)


@dataclass(frozen=True, init=False)
class TransitionRecord:
    transition_id: str
    belief_version: int
    start: float
    end: float
    _proposed: torch.Tensor
    _sent: torch.Tensor
    _applied: torch.Tensor
    _actuator_response: torch.Tensor
    _observation: torch.Tensor
    _next_observation: torch.Tensor

    def __init__(self, transition_id, belief_version, start, end, proposed, sent, applied,
                 actuator_response, observation, next_observation):
        if not transition_id or not math.isfinite(start + end) or not 0 <= start < end:
            raise ValueError("a unique transition and positive observed time interval are required")
        for k, v in dict(transition_id=transition_id, belief_version=belief_version, start=start, end=end).items():
            object.__setattr__(self, k, v)
        for k, v, n in (("proposed", proposed, 7), ("sent", sent, 7), ("applied", applied, 7),
                        ("actuator_response", actuator_response, 7), ("observation", observation, 30),
                        ("next_observation", next_observation, 30)):
            object.__setattr__(self, "_" + k, _copy(v, (n,), k))
        if any(t.abs().max() > 1 for t in (self._sent, self._applied)):
            raise ValueError("sent/applied normalized commands outside actuator limits")

    @property
    def proposed(self): return self._proposed.clone()
    @property
    def sent(self): return self._sent.clone()
    @property
    def applied(self): return self._applied.clone()
    @property
    def actuator_response(self): return self._actuator_response.clone()
    @property
    def observation(self): return self._observation.clone()
    @property
    def next_observation(self): return self._next_observation.clone()


class TransitionLedger:
    def __init__(self):
        self._records = []
        self._ids = set()

    def accept(self, record: TransitionRecord):
        if record.transition_id in self._ids:
            raise ValueError("duplicate transition evidence")
        self._ids.add(record.transition_id)
        self._records.append(record)

    @property
    def records(self): return tuple(self._records)
    @property
    def real_transition_count(self): return len(self._records)
