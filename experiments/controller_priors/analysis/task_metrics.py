"""Independent physical metrics per task (official reward/success untouched).

Everything here reads only the policy observation, the *applied* normalized
action and public task parameters.  No env-private state is read.  Metrics are
accumulated per replica until that replica's FIRST episode ends, matching the
``FirstEpisodes`` accounting; no fixed short window is used.

Channel semantics differ per task and are recorded in the eval report:
- docking / rendezvous: action = normalized per-axis thrust, F_i = a_i f_max;
- satellite: action = normalized CMG gimbal rate, rate_i = a_i action_scale.
"""
import torch


def _z(env, shape=()):
    return torch.zeros(shape, device=env.device, dtype=torch.float64)


def init_accumulators(task, env, n):
    acc = {
        'steps': _z(env, n), 'action_abs_sum': _z(env, n), 'action_sq_sum': _z(env, n),
        'sat_envsteps': _z(env, n), 'nominal_sat_envsteps': _z(env, n),
        'terminal_obs': torch.zeros(n, int(env.get_observations()['policy'].shape[-1]),
                                    device=env.device, dtype=torch.float64),
        'has_terminal': _z(env, n).bool(),
    }
    if task in ('docking', 'rendezvous'):
        acc['propellant_kg'] = _z(env, n)
        acc['min_range_m'] = torch.full((n,), float('inf'), device=env.device, dtype=torch.float64)
        acc['terminal_range_m'] = _z(env, n)
        acc['terminal_speed_m_s'] = _z(env, n)
        if task == 'rendezvous':
            acc['keepout_envsteps'] = _z(env, n)
    elif task == 'satellite':
        acc['gimbal_rate_sq_integral'] = _z(env, n)   # actuator-usage proxy, NOT electric energy
        acc['min_manipulability'] = torch.full((n,), float('inf'), device=env.device, dtype=torch.float64)
        acc['terminal_attitude_error_rad'] = _z(env, n)
        acc['terminal_rate_rad_s'] = _z(env, n)
    else:
        raise KeyError(f'no physical metric profile for task {task!r}')
    return acc


def accumulate(task, env, acc, *, obs, applied_action, nominal_action, active, done,
               terminal_obs, dt_control):
    """Vectorised update for the replicas whose first episode is still in flight."""
    act = applied_action.to(torch.float64)
    a = active.to(torch.float64)
    acc['steps'] += a
    acc['action_abs_sum'] += act.abs().sum(-1) * a
    acc['action_sq_sum'] += (act ** 2).sum(-1) * a
    sat = (act.abs() >= 1.0 - 1e-6).any(-1).to(torch.float64)
    acc['sat_envsteps'] += sat * a
    if nominal_action is not None:
        nsat = (nominal_action.abs() >= 1.0 - 1e-6).any(-1).to(torch.float64)
        acc['nominal_sat_envsteps'] += nsat * a
    if task in ('docking', 'rendezvous'):
        # Plant constants may be per-env tensors under domain randomization; a
        # scalar conversion here would raise, so keep the per-env shape (the
        # propellant sum is elementwise anyway) and only collapse when scalar.
        def _const(name):
            value = getattr(env, name)
            return value.to(torch.float64) if torch.is_tensor(value) else float(value)

        f_max = _const('f_max')
        isp, g0 = _const('isp'), _const('g0')
        acc['propellant_kg'] += (act.norm(dim=-1) * a) * f_max * dt_control / (isp * g0)
        rng = obs[:, :3].norm(dim=-1)
        acc['min_range_m'] = torch.minimum(acc['min_range_m'], torch.where(active, rng,
                                                                          torch.full_like(rng, float('inf'))))
        if task == 'rendezvous':
            inside = (rng < float(env.keep_out)).to(torch.float64)
            acc['keepout_envsteps'] += inside * a
        pred = done
        if bool(pred.any()):
            term = terminal_obs.to(torch.float64)
            acc['terminal_range_m'] += torch.where(done, term[:, :3].norm(dim=-1), _z(env, rng.shape[-1]))
            acc['terminal_speed_m_s'] += torch.where(done, term[:, 3:6].norm(dim=-1), _z(env, rng.shape[-1]))
            acc['terminal_obs'] = torch.where(done[:, None], term, acc['terminal_obs'])
            acc['has_terminal'] |= done
    elif task == 'satellite':
        scale = float(env.action_scale)
        rate = act * scale
        acc['gimbal_rate_sq_integral'] += (rate ** 2).sum(-1) * a * dt_control
        n_cmg = int(env.num_gimbals)
        m_norm = obs[:, 10 + 3 * n_cmg]
        acc['min_manipulability'] = torch.minimum(
            acc['min_manipulability'], torch.where(active, m_norm, torch.full_like(m_norm, float('inf'))))
        if bool(done.any()):
            term = terminal_obs.to(torch.float64)
            acc['terminal_attitude_error_rad'] += torch.where(done, term[:, 4:7].norm(dim=-1),
                                                              _z(env, term.shape[0]))
            acc['terminal_rate_rad_s'] += torch.where(done, term[:, 7:10].norm(dim=-1),
                                                      _z(env, term.shape[0]))
            acc['terminal_obs'] = torch.where(done[:, None], term, acc['terminal_obs'])
            acc['has_terminal'] |= done


def finalize(task, env, acc, *, dt_control):
    """Per-replica and aggregated physical metrics over the first episodes."""
    n = acc['steps'].shape[0]
    steps = acc['steps'].clamp_min(1.)
    per = {
        'steps': acc['steps'].cpu().tolist(),
        'maneuver_time_s': (acc['steps'] * dt_control).cpu().tolist(),
        'mean_abs_action': (acc['action_abs_sum'] / steps).cpu().tolist(),
        'rms_action': (acc['action_sq_sum'] / steps).sqrt().cpu().tolist(),
        'action_saturation_fraction': (acc['sat_envsteps'] / steps).cpu().tolist(),
    }
    if task in ('docking', 'rendezvous'):
        per['propellant_kg'] = acc['propellant_kg'].cpu().tolist()
        per['min_range_m'] = acc['min_range_m'].cpu().tolist()
        per['terminal_range_m'] = acc['terminal_range_m'].cpu().tolist()
        per['terminal_speed_m_s'] = acc['terminal_speed_m_s'].cpu().tolist()
        if task == 'rendezvous':
            per['keepout_fraction'] = (acc['keepout_envsteps'] / steps).cpu().tolist()
        per['terminal_error_m'] = acc['terminal_range_m'].cpu().tolist()
    else:
        per['gimbal_rate_squared_integral_proxy'] = acc['gimbal_rate_sq_integral'].cpu().tolist()
        per['min_manipulability'] = acc['min_manipulability'].cpu().tolist()
        per['terminal_attitude_error_rad'] = acc['terminal_attitude_error_rad'].cpu().tolist()
        per['terminal_rate_rad_s'] = acc['terminal_rate_rad_s'].cpu().tolist()
        per['terminal_error_rad'] = acc['terminal_attitude_error_rad'].cpu().tolist()
    agg = {}
    for k, v in per.items():
        finite = [x for x in v if isinstance(x, (int, float)) and x == x and abs(x) != float('inf')]
        agg[k] = sum(finite) / len(finite) if finite else None
    return {'per_replica': per, 'mean': agg}


PROFILE_NOTE = {
    'docking': 'action = normalized per-axis thrust; propellant from the same '
               'Isp=300 s / 10 N model as the env; time = control steps (dt=1 s)',
    'rendezvous': 'action = normalized per-axis thrust; propellant from Isp=300 s; '
                  'keepout_fraction is the fraction of first-episode steps inside r<5 m',
    'satellite': 'gimbal-rate-squared integral is an ACTUATOR-USAGE PROXY (rad^2/s), '
                 'not electrical energy (no motor power model in the env); time = '
                 'control steps x 0.025 s (40 Hz)',
}
