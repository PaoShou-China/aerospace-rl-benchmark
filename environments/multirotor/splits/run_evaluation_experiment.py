"""Fresh training with episode-weighted curriculum and layered validation.

All artifacts live under this directory. Run --smoke before a full experiment.
Evaluation workers use separate processes so they cannot reset training physics,
update observation normalization, or consume training RNG state.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
# Keep writable compiler/runtime caches inside the permitted experiment folder.
for key,name in (('NUMBA_CACHE_DIR','numba'),('GS_CACHE_FILE_PATH','genesis'),
                 ('QD_OFFLINE_CACHE_FILE_PATH','quadrants'),('CUDA_CACHE_PATH','cuda'),
                 ('MPLCONFIGDIR','matplotlib')):
    cache=HERE/'experiments'/'runtime_cache'/name
    cache.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(cache)


def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def evaluate_worker(args):
    from environments.multirotor.splits.evaluation import load_env_policy,evaluate_policy
    result={}
    for name,full in (('random_four_gates',False),('fixed_full_track',True)):
        runs=[]
        for seed in ([7] if args.smoke else [7,17,27]):
            env,policy,_=load_env_policy(args.checkpoint.parent,8 if args.smoke else 64,
                'gpu',seed,full_track=full,checkpoint=args.checkpoint,
                max_episode_seconds=1. if args.smoke else 20.)
            try:
                runs.append(evaluate_policy(env,policy))
                runs[-1]['seed']=seed
            finally:env.close()
        episodes=[e for r in runs for e in r['episodes']]
        planned=sum(r['scheduled_episodes'] for r in runs)
        assert len(episodes)==planned,'Evaluation did not finish every first episode'
        summary={key:sum(e[key] for e in episodes)/planned
                 for key in ('success','gates_passed','crash','gate_collision','out_of_bounds','timeout')}
        result[name]=dict(summary=summary,runs=runs)
    write_json(args.report,dict(checkpoint=str(args.checkpoint),smoke=args.smoke,protocols=result))


def main(args):
    from environments.multirotor.splits import train
    from environments.multirotor.splits.ppo import build_config,SplitsRunner
    from algorithms.train_utils import RslVecEnv
    import torch
    out=HERE/'experiments'/('window1024_smoke' if args.smoke else 'window1024_seed7_1000')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():
        raise FileExistsError(f'{out} already has an experiment; refusing to overwrite')
    def status(phase,**extra):
        write_json(out/'status.json',dict(phase=phase,pid=os.getpid(),updated=datetime.now().isoformat(),**extra))
    updates=2 if args.smoke else 1000
    nenv=8 if args.smoke else train.NUM_ENVS
    ppo_kwargs=dict(learning_rate=train.LEARNING_RATE,entropy_coef=train.ENTROPY_COEF,
        init_std=train.INIT_STD,num_steps_per_env=8 if args.smoke else train.NUM_STEPS_PER_ENV,
        num_learning_epochs=1 if args.smoke else train.NUM_LEARNING_EPOCHS,
        num_mini_batches=2 if args.smoke else train.NUM_MINI_BATCHES)
    kwargs=copy.deepcopy(train.ENV_KWARGS)
    kwargs['buffer_cfg']['success_window']=1024
    if args.smoke:kwargs['max_episode_seconds']=.2
    manifest=dict(seed=7,updates=updates,num_envs=nenv,env_kwargs=kwargs,ppo_kwargs=ppo_kwargs,
        description='Fresh policy; corrected yaw; last 1024 completed episode outcomes; 60% curriculum threshold.',
        limitations='Combined implementation experiment, not a single-factor causal ablation. Fixed-start replicas are not distinct initial states.',
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    write_json(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    evaluations=[]
    def evaluate(path,label):
        report=out/'evaluations'/f'{label}.json'
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
                 '--checkpoint',str(path),'--report',str(report)]
        if args.smoke:command.append('--smoke')
        with (out/f'evaluation_{label}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        data=json.loads(report.read_text(encoding='utf-8'))
        summary={name:v['summary'] for name,v in data['protocols'].items()}
        evaluations.append(dict(label=label,report=str(report),summary=summary))
        write_json(out/'comparison.json',evaluations)
        print(json.dumps(dict(evaluation=label,summary=summary)),flush=True)
        return summary
    best=None
    def callback(path,runner):
        nonlocal best
        iteration=runner.current_learning_iteration+1
        if path.name!='checkpoint.pt':
            if args.smoke or runner.current_learning_iteration==0 or runner.current_learning_iteration%200:
                return
        b=runner.env.initial_states
        training=dict(iteration=iteration,completed_episodes=int(runner.env.completed_episodes.sum()),
            completed_successes=int(runner.env.completed_successes.sum()),
            window_success_rate=b.success_rate,window_count=len(b.success_history),
            reset_count=b.reset_count,replay_count=b.replay_count,ready=b.ready)
        write_json(out/'training_metrics'/f'{iteration}.json',training)
        status('validation',iteration=iteration)
        summary=evaluate(path,f'new_{iteration}')
        full=summary['fixed_full_track'];segment=summary['random_four_gates']
        score=(full['success'],full['gates_passed'],segment['success'],segment['gates_passed'])
        if best is None or score>best:
            best=score
            shutil.copy2(path,out/'best.pt')
            write_json(out/'best.json',dict(iteration=iteration,score=score,source=str(path)))
        status('training',iteration=iteration)
    try:
        status('baseline_evaluation')
        if not args.smoke:
            for name,label in (('quadx_splits_extended','old_2000'),('quadx_splits_4000','old_4000')):
                evaluate(ROOT/'results'/name/'checkpoint.pt',label)
        # Save a genuine untrained baseline. run_training reseeds to the same
        # seed and constructor order, so its starting policy is identical.
        initial=out/'initial';initial.mkdir()
        cfg=build_config(**ppo_kwargs)
        cfg.update(env_kwargs=kwargs,training_seed=7,training_num_envs=nenv)
        write_json(initial/'config.json',cfg)
        torch.manual_seed(7)
        env=train.QuadXSplitsEnv(num_envs=nenv,backend='gpu',seed=7,**kwargs)
        try:
            runner=SplitsRunner(RslVecEnv(env),copy.deepcopy(cfg),device=str(env.device))
            torch.save(runner.alg.save(),initial/'checkpoint.pt')
        finally:env.close()
        del runner,env
        torch.cuda.empty_cache()
        evaluate(initial/'checkpoint.pt','new_untrained')
        status('training',iteration=0)
        train.run_training(train.QuadXSplitsEnv,num_envs=nenv,updates=updates,seed=7,
            backend='gpu',resume=False,out_dir=out,env_kwargs=kwargs,
            checkpoint_callback=callback,**ppo_kwargs)
        status('completed',updates=updates,comparison=str(out/'comparison.json'),best=str(out/'best.pt'))
    except BaseException:
        import traceback
        status('failed',error=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.worker:
        if args.checkpoint is None or args.report is None:parser.error('worker requires checkpoint and report')
        evaluate_worker(args)
    else:main(args)
