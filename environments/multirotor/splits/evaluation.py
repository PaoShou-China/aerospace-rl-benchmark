"""Independent full-track evaluation: exactly the first episode of each replica."""
import copy
import json
import math
from pathlib import Path

import torch
from algorithms.train_utils import RslVecEnv
from .env import QuadXSplitsEnv
from .ppo import SplitsRunner


def load_env_policy(out,num_envs,backend,seed,laps=1,max_episode_seconds=None):
    out=Path(out)
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if cfg.get('splits_schema')!='racing36_v1':raise ValueError('expected a new 36-D Split-S checkpoint')
    kwargs=dict(cfg.get('env_kwargs') or {})
    kwargs.update(full_track=True,laps=laps,initialization='fixed')
    kwargs['max_episode_seconds']=(max_episode_seconds if max_episode_seconds is not None
                                   else kwargs.get('max_episode_seconds',20.)*laps)
    torch.manual_seed(seed)
    env=QuadXSplitsEnv(num_envs=num_envs,backend=backend,seed=seed,**kwargs)
    try:
        runner=SplitsRunner(RslVecEnv(env),copy.deepcopy(cfg),device=str(env.device))
        runner.load(str(out/'checkpoint.pt'),map_location=str(env.device))
        return env,runner.get_inference_policy(),cfg
    except BaseException:
        env.close();raise


def _number(value):
    value=float(value)
    return value if math.isfinite(value) else None


@torch.no_grad()
def evaluate_policy(env,policy,max_steps=None):
    seen=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    records=[];obs=env.get_observations();steps=0
    for _ in range(max_steps if max_steps is not None else env.max_steps):
        actions=policy(obs)
        if not torch.isfinite(actions).all():raise ValueError('non-finite policy action')
        obs,_,term,trunc,extra=env.step(actions);steps+=1
        take=(term|trunc)&~seen
        if bool(take.any()):
            outcome=extra['outcome']
            for i in take.nonzero().flatten().tolist():
                row=dict(replica=i,success=bool(outcome['success'][i]),
                         gates_passed=int(outcome['gates_passed'][i]),laps=int(outcome['laps'][i]),
                         crash=bool(outcome['crash'][i]),gate_collision=bool(outcome['gate_collision'][i]),
                         out_of_bounds=bool(outcome['out_of_bounds'][i]),timeout=bool(outcome['timeout'][i]))
                for key in ('time_s','min_clearance_m','mean_speed','max_speed','mean_body_rate'):
                    row[key]=_number(outcome[key][i])
                row['lap_times_s']=[_number(t) for t in outcome['lap_times'][i]]
                records.append(row)
            seen|=take
        if bool(seen.all()):break
    successes=sum(r['success'] for r in records)
    def mean(key,success_only=False):
        values=[r[key] for r in records if r[key] is not None and (not success_only or r['success'])]
        return sum(values)/len(values) if values else None
    return dict(protocol='first_episode_per_replica',env_steps=steps,scheduled_episodes=env.num_envs,
                completed_episodes=len(records),all_replicas_finished=bool(seen.all()),
                incomplete_replicas=(~seen).nonzero().flatten().tolist(),
                success_rate=successes/env.num_envs,successes=successes,
                mean_success_time_s=mean('time_s',True),mean_gates_passed=mean('gates_passed'),
                crash_rate=sum(r['crash'] for r in records)/env.num_envs,
                gate_collision_rate=sum(r['gate_collision'] for r in records)/env.num_envs,
                mean_clearance_m=mean('min_clearance_m'),mean_speed=mean('mean_speed'),
                mean_body_rate=mean('mean_body_rate'),episodes=records)


@torch.no_grad()
def record_episode(env,policy,max_steps=None):
    """Single-replica poses including the genuine terminal pose before auto-reset."""
    if env.num_envs!=1:raise ValueError('recording requires one replica')
    frames=[];obs=env.get_observations();outcome=None
    def snap(state,time_s,passed,target_id):
        frame={key:value[0].detach().cpu().numpy().copy() for key,value in state.items()}
        frame.update(time_s=time_s,passed=passed,target_id=target_id)
        frames.append(frame)
    snap(dict(zip(('p','q','v','w'),env.plant.state())),0.,0,int(env.gate_index[0]))
    for step in range(max_steps if max_steps is not None else env.max_steps):
        obs,_,term,trunc,extra=env.step(policy(obs))
        if bool((term|trunc)[0]):
            outcome=extra['outcome']
            passed=int(outcome['gates_passed'][0])
            target=min(passed,env.num_targets-1)%7
            snap(extra['terminal_state'],float(outcome['time_s'][0]),passed,target)
            break
        snap(dict(zip(('p','q','v','w'),env.plant.state())),(step+1)/40,
             int(env.target_index[0]),int(env.gate_index[0]))
    result=('SUCCESS' if outcome is not None and bool(outcome['success'][0]) else
            'TIMEOUT' if outcome is not None and bool(outcome['timeout'][0]) else
            'FAILURE' if outcome is not None else 'INCOMPLETE')
    return frames,result
