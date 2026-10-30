"""Continue 1000 updates with a fifth-gate preview and four-gate termination."""
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

# Import locally to configure writable caches before environments/__init__.
from run_evaluation_experiment import HERE,ROOT,write_json


def main(smoke=False):
    from environments.multirotor.splits import train
    source=HERE/'experiments/window1024_seed7_1000'
    out=HERE/'experiments'/('lookahead5_resume_smoke' if smoke else 'lookahead5_resume_2000')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists():raise FileExistsError(f'Experiment exists: {out}')
    def status(phase,**extra):
        write_json(out/'status.json',dict(phase=phase,pid=os.getpid(),updated=datetime.now().isoformat(),**extra))
    updates=1 if smoke else 1000
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    kwargs=copy.deepcopy(cfg['env_kwargs'])
    ppo_kwargs=dict(learning_rate=train.LEARNING_RATE,entropy_coef=train.ENTROPY_COEF,
        init_std=train.INIT_STD,num_steps_per_env=train.NUM_STEPS_PER_ENV,
        num_learning_epochs=train.NUM_LEARNING_EPOCHS,num_mini_batches=train.NUM_MINI_BATCHES)
    manifest=dict(source=str(source/'checkpoint.pt'),additional_updates=updates,
        total_updates=1000+updates,change='Preview gate 5 while approaching gate 4; finish after gate 4.',
        restored='Policy, critic, optimizer, observation normalization, curriculum window/buffer, RNG; trajectories restart.',
        source_checkpoint_sha256=hashlib.sha256((source/'checkpoint.pt').read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    write_json(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    for name in ('config.json','checkpoint.pt'):shutil.copy2(source/name,out/name)
    shutil.copy2(source/'evaluations/new_1000.json',out/'previous_protocol_evaluation.json')
    comparisons=[];best=None
    def evaluate(path,label):
        report=out/'evaluations'/f'{label}.json'
        command=[sys.executable,'-u',str(HERE/'run_evaluation_experiment.py'),'--worker',
                 '--checkpoint',str(path),'--report',str(report)]
        if smoke:command.append('--smoke')
        with (out/f'evaluation_{label}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        data=json.loads(report.read_text(encoding='utf-8'))
        summary={name:v['summary'] for name,v in data['protocols'].items()}
        comparisons.append(dict(label=label,summary=summary,report=str(report)))
        write_json(out/'comparison.json',comparisons)
        print(json.dumps(dict(evaluation=label,summary=summary)),flush=True)
        return summary
    def consider_best(path,summary,iteration):
        nonlocal best
        full=summary['fixed_full_track'];segment=summary['random_four_gates']
        score=(full['success'],full['gates_passed'],segment['success'],segment['gates_passed'])
        if best is None or score>best:
            best=score;shutil.copy2(path,out/'best.pt')
            write_json(out/'best.json',dict(iteration=iteration,score=score,source=str(path)))
    def callback(path,runner):
        iteration=runner.current_learning_iteration+1
        if path.name!='checkpoint.pt':
            if smoke or runner.current_learning_iteration<=1000 or runner.current_learning_iteration%200:return
        b=runner.env.initial_states
        write_json(out/'training_metrics'/f'{iteration}.json',dict(iteration=iteration,
            success_window_rate=b.success_rate,window_count=len(b.success_history),
            replay_count=b.replay_count,reset_count=b.reset_count,ready=b.ready))
        status('validation',iteration=iteration)
        summary=evaluate(path,f'continued_{iteration}')
        consider_best(path,summary,iteration)
        status('training',iteration=iteration)
    try:
        status('baseline_evaluation')
        summary=evaluate(out/'checkpoint.pt','before_1000_new_observation')
        consider_best(out/'checkpoint.pt',summary,1000)
        status('training',iteration=1000)
        train.run_training(train.QuadXSplitsEnv,num_envs=cfg['training_num_envs'],updates=updates,
            seed=cfg['training_seed'],backend='gpu',resume=True,out_dir=out,env_kwargs=kwargs,
            checkpoint_callback=callback,**ppo_kwargs)
        status('completed',total_updates=1000+updates)
    except BaseException:
        import traceback
        status('failed',error=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args().smoke)
