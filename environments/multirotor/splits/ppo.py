"""Split-S-local PPO/model/checkpoint code; no shared algorithm defaults change."""
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from algorithms.train_utils import RslVecEnv, build_cfg


class SplitsMLPModel(MLPModel):
    """256x256 MLP; tanh actor mean, linear critic head.

    Gaussian samples remain raw PPO actions and are clipped by the existing
    QuadX action mapping. Log probabilities are computed for those raw samples,
    not incorrectly treated as a squashed-Gaussian density.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.distribution is not None:
            self.mlp=torch.nn.Sequential(self.mlp,torch.nn.Tanh())


class SplitsPPO(PPO):
    """Bootstrap time limits from pre-reset terminal observations, not V(s_t)."""
    def process_env_step(self,obs,rewards,dones,extras):
        timeouts=extras.get('time_outs')
        if timeouts is not None:
            with torch.no_grad():
                terminal_value=self.critic(extras['terminal_observations'].to(self.device)).squeeze(-1)
            rewards=rewards+self.gamma*terminal_value*timeouts.to(self.device)
            extras={k:v for k,v in extras.items() if k!='time_outs'}
        super().process_env_step(obs,rewards,dones,extras)


def build_config(*,learning_rate=3e-4,entropy_coef=.001,init_std=.5,
                 num_steps_per_env=128,num_learning_epochs=5,num_mini_batches=4):
    # Optimizer values and Critic width are local choices, not verified paper settings.
    cfg=build_cfg(arch='mlp',lr_schedule='fixed',learning_rate=learning_rate,
                  entropy_coef=entropy_coef,init_std=init_std,num_steps_per_env=num_steps_per_env,
                  num_learning_epochs=num_learning_epochs,num_mini_batches=num_mini_batches,
                  gamma=.99,lam=.95)
    for name in ('actor','critic'):
        cfg[name].update(class_name='environments.multirotor.splits.ppo:SplitsMLPModel',
                         hidden_dims=[256,256],activation='tanh')
    cfg['algorithm']['class_name']='environments.multirotor.splits.ppo:SplitsPPO'
    cfg['splits_schema']='racing36_v1'
    return cfg


class SplitsRunner(OnPolicyRunner):
    def save(self,path,infos=None):
        state=self.alg.save()
        # rsl-rl exposes the last executed iteration; resume must start at the next.
        state.update(iter=self.current_learning_iteration+1,infos=infos,
                     splits_training_state=self.env.training_state_dict(),torch_rng=torch.get_rng_state())
        if torch.cuda.is_available():state['cuda_rng']=torch.cuda.get_rng_state_all()
        torch.save(state,path)
        if hasattr(self.logger,'writer'):
            self.logger.save_model(path,self.current_learning_iteration)

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        infos=super().load(path,load_cfg,strict,map_location)
        if not self.env.full_track:
            state=torch.load(path,weights_only=False,map_location='cpu')
            if 'splits_training_state' not in state:raise ValueError('checkpoint lacks Split-S curriculum state')
            self.env.load_training_state_dict(state['splits_training_state'])
            torch.set_rng_state(state['torch_rng'])
            if 'cuda_rng' in state and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
        return infos


def run_training(env_cls,*,num_envs=1024,updates=1000,seed=7,backend='gpu',resume=False,
                 out_dir=None,env_kwargs=None,**ppo_kwargs):
    out=Path(out_dir) if out_dir is not None else Path(__file__).resolve().parents[3]/'results'/'quadx_splits'
    cfg=build_config(**ppo_kwargs);cfg['env_kwargs']=dict(env_kwargs or {})
    cfg.update(training_seed=seed,training_num_envs=num_envs)
    if not resume and (list(out.glob('*.pt')) or (out/'config.json').exists()):
        raise FileExistsError(f'{out} already contains a run; choose another OUT_DIR or deliberately resume')
    if resume:
        saved=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if saved!=cfg:raise ValueError('resume requires identical task/network/PPO configuration')
        if not (out/'checkpoint.pt').exists():raise FileNotFoundError(out/'checkpoint.pt')
    torch.manual_seed(seed);out.mkdir(parents=True,exist_ok=True)
    env=env_cls(num_envs=num_envs,backend=backend,seed=seed,**cfg['env_kwargs'])
    try:
        runner=SplitsRunner(RslVecEnv(env),copy.deepcopy(cfg),log_dir=str(out),device=str(env.device))
        if resume:runner.load(str(out/'checkpoint.pt'))
        (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        start=time.perf_counter();runner.learn(updates);elapsed=time.perf_counter()-start
        runner.save(str(out/'checkpoint.pt'))
        episodes=int(env.completed_episodes.sum())
        source=Path(__file__).resolve().parent
        report=dict(task=env.cfg,seed=seed,updates=updates,algo='ppo',
                    transitions=updates*cfg['num_steps_per_env']*num_envs,train_wall_seconds=elapsed,
                    completed_episodes=env.completed_episodes.cpu().tolist(),
                    completed_successes=env.completed_successes.cpu().tolist(),
                    completed_metrics={k:float(v)/max(episodes,1) for k,v in env.completed_metrics.items()},
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
                    dynamics_sha256=hashlib.sha256((source.parents[1]/'common'/'plant.py').read_bytes()).hexdigest(),
                    resume_note='curriculum and RNG restored; trajectories restart at reset, not bitwise in-flight replay')
        (out/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)
        return report
    finally:env.close()
