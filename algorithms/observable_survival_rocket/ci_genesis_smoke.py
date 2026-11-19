"""Bounded CPU Genesis integration smoke, not a policy-performance experiment.

Creates 16 new transitions (4 environments x 4 policy steps), performs one
PPO/auxiliary update and checks the saved checkpoint. No complete landing
trial, remote connection, benchmark score or generalization claim is made.
"""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def run(prepared: Path, evidence: Path):
    evidence.mkdir(parents=True, exist_ok=True)
    configuration=json.loads((prepared/'smoke.json').read_text())
    configuration.update(backend='cpu',num_envs=4,rollout=4,updates=1,
        additional_transitions=16,minibatch=16,microbatch=4,max_microbatch=4,
        inference_batch=4,horizon=2,knots=1,iterations=1,tail_steps=2,
        evaluation_workers=1,skip_evaluation=True)
    config=prepared/'ci_cpu.json'
    config.write_text(json.dumps(configuration,indent=2)+'\n')
    output=prepared/'ci_cpu_run'
    status={'scope':'CPU Genesis software integration only', 'new_transitions_requested':16,
            'remote_gpu_connected':False,'performance_evaluation_run':False,
            'landing_success_rate':None,'status':'starting'}
    started=time.perf_counter()
    try:
        command=[sys.executable,'-u',str(prepared/'bundle/scripts/run.py'),'train',
                 '--config',str(config),'--out',str(output),'--skip-evaluation']
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (evidence/'genesis_smoke.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                     env=env,start_new_session=True)
            try:
                code=process.wait(timeout=480)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                raise
        if code:
            raise RuntimeError(f'CPU smoke exited {code}; see genesis_smoke.log')
        training=json.loads((output/'status.json').read_text())
        manifest=json.loads((output/'manifest.json').read_text())
        if training.get('status')!='trained':
            raise RuntimeError('smoke training did not finish')
        actual=training['new_environment_transitions']-manifest['resume_environment_transitions']
        if actual!=16 or not (output/'checkpoint.pt').is_file():
            raise RuntimeError('incorrect interaction budget or missing checkpoint')
        status.update(status='passed',new_transitions_completed=actual,
                      completed_updates=training['completed_updates'],checkpoint_saved=True,
                      historical_transitions=manifest['resume_environment_transitions'])
    except BaseException as error:
        status.update(status='failed',error=repr(error))
        raise
    finally:
        status['wall_seconds']=time.perf_counter()-started
        (evidence/'genesis_smoke_status.json').write_text(json.dumps(status,indent=2)+'\n')
        for name in ('status.json','manifest.json','progress.jsonl'):
            if (output/name).exists():
                (evidence/f'genesis_{name}').write_bytes((output/name).read_bytes())
        print(json.dumps(status,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    args=p.parse_args()
    run(args.prepared.resolve(),args.evidence.resolve())

if __name__=='__main__': main()
