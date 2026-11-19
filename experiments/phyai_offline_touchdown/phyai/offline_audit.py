"""Replay retained real transitions. No simulator, training, or collection."""
import torch
from frozen.core.phi_rocket_types import PARAMETERS
from frozen.core.phi_rocket_physics import state_minus
from frozen.core.rocket_physical_core import observation_state


def event_windows(data, before=8, after=8):
    obs, nxt, first = data['obs'], data['next_obs'], data['first_episode']
    result = {k: [] for k in ("pre_contact", "contact_crossing", "post_contact")}
    events = []
    for e in range(obs.shape[1]):
        cross = first[:, e] & (obs[:, e, 29] <= .5) & (nxt[:, e, 29] > .5)
        ids = cross.nonzero().flatten()
        if not len(ids):
            continue
        t = int(ids[0])
        # The row t is the observed state BEFORE crossing; next_obs[t] may
        # already include a collision impulse or the captured terminal state.
        events.append(dict(episode=e, contact_transition=t, pre_contact_observation=t))
        result['contact_crossing'].append((t, e))
        result['pre_contact'] += [(k, e) for k in range(max(0, t-before), t) if bool(first[k,e])]
        result['post_contact'] += [(k, e) for k in range(t+1, min(len(obs), t+after+1)) if bool(first[k,e])]
    return result, events


def truth_coordinates(physics, reference, true_parameters):
    """Diagnostic mapping only; log range clipping rather than silently hide it."""
    tp = {k: torch.as_tensor(v, dtype=reference.dtype, device=reference.device) for k,v in true_parameters.items()}
    n = len(tp['thrust'])
    theta = reference[None].expand(n, -1).clone()
    values = physics.values(reference[None])[0]
    names = [p[0] for p in PARAMETERS]
    clipping = {}
    def put(name, physical):
        j = names.index(name)
        scale, transform = PARAMETERS[j][1:3]
        raw = physical.log() if transform == 'log' else physical
        raw = raw / scale
        clipping[name] = int((raw.abs() > 1).sum())
        theta[:, j] = raw.clamp(-1,1)
    for name,key in [('thrust_per_mass_gain','thrust'),('fin_aerodynamic_gain','fin')]:
        put(name, values[names.index(name)] * tp[key])
    put('engine_lag_gain', tp['booster_tau']/.01)
    put('gimbal_lag_gain', tp['gimbal_tau']/.01)
    for k,name in enumerate(['gimbal_x_bias','gimbal_y_bias']):
        put(name, values[names.index(name)] + tp['gimbal_bias'][:,k])
    for k,name in enumerate(['wind_world_x','wind_world_y','wind_world_z']):
        put(name, tp['wind'][:,k])
    return theta, clipping


def distribution(values):
    x = torch.as_tensor(values, dtype=torch.float64)
    finite = x[torch.isfinite(x)]
    return dict(n=x.numel(), nonfinite=int(x.numel()-finite.numel()),
                median=float(finite.median()) if len(finite) else None,
                p95=float(torch.quantile(finite,.95)) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None)


def replay_rows(physics, data, theta, rows):
    if not rows:
        return {"samples": 0}
    t, e = torch.tensor(rows).T
    obs, nxt, actions = data['obs'][t,e].double(), data['next_obs'][t,e].double(), data['action'][t,e].double()
    x, actual = observation_state(obs), observation_state(nxt)
    with torch.no_grad():
        # Initialization is the measured state, not a newly inferred belief.
        predicted = physics.sensor(physics(x, actions, theta[e], .025), theta[e])
        error = state_minus(predicted, actual)
    norms = dict(position_m=error[:,:3].norm(dim=-1), velocity_m_s=error[:,3:6].norm(dim=-1),
                 attitude_rad=error[:,6:9].norm(dim=-1), angular_rate_rad_s=error[:,9:12].norm(dim=-1),
                 actuator_normalized=error[:,12:19].abs().amax(-1))
    pred_w, real_w = predicted[:,10:13].norm(dim=-1), actual[:,10:13].norm(dim=-1)
    flags = {}
    for limit in (.02, .3, .35):
        flags[str(limit)] = dict(false_clear=int(((pred_w < limit)&(real_w >= limit)).sum()),
                                 false_alarm=int(((pred_w >= limit)&(real_w < limit)).sum()))
    return dict(samples=len(rows), errors={k:distribution(v) for k,v in norms.items()},
                angular_threshold_disagreements=flags,
                predicted_angular_rate=distribution(pred_w), measured_angular_rate=distribution(real_w))
