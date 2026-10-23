"""Run a bounded, resumable four-arm physical-control experiment campaign."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((snapshot.ROOT/'configs/confidence_sequence.json').read_text())
    config.update(gpu_memory_budget_gib=22. if args.workers == 2 else 44.,
                  microbatch=4096, gpu_memory_fraction=.94)
    definitions = [
        ('confidence_sequence', {}),
        ('compact_sequence', {'hypotheses':3, 'horizon':8, 'knots':2}),
        ('fixed_belief', {'ablation':'fixed_belief'}),
        ('no_physical_solver', {'ablation':'no_solver'}),
    ]
    for name, changes in definitions:
        path = out/(name+'.json')
        if not path.exists():
            snapshot.write_json(path, {**config, **changes})
    active = {}
    status_path = out/'campaign.json'
    statuses = json.loads(status_path.read_text()).get('arms', {}) if status_path.exists() else {}
    # A fresh coordinator never assumes a previously live child is finished.
    # It marks interrupted runs for explicit continuation instead of erasing them.
    for name, _ in definitions:
        if (out/name).exists() and name not in statuses:
            statuses[name] = {'status':'needs_resume'}
        if statuses.get(name, {}).get('status') == 'running':
            statuses[name]['status'] = 'needs_resume'
    pending = [name for name, _ in definitions if name not in statuses]
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                name = pending.pop(0)
                log = (out/(name+'.log')).open('a')
                command = [sys.executable, '-u', str(snapshot.ROOT/'scripts/run.py'), 'train',
                           '--config', str(out/(name+'.json')), '--out', str(out/name)]
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                log.close()
                active[name] = child
                statuses[name] = {'status':'running', 'pid':child.pid, 'started_at_unix':time.time()}
            for name, child in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                statuses[name].update(status='completed' if code == 0 else 'failed',
                                      exit_code=code, finished_at_unix=time.time())
                result = out/name/'development512.json'
                if code == 0 and result.exists():
                    statuses[name]['evaluation'] = json.loads(result.read_text())['summary']
                del active[name]
            snapshot.write_json(status_path, {'status':'running' if pending or active else 'finished',
                'arms':statuses, 'pending':pending, 'max_workers':args.workers,
                'goal_success_rate':.90, 'development_episodes':512,
                'new_transitions_per_arm':2097152, 'planned_new_transitions':8388608,
                'source_checkpoint':'artifacts/policy.pt', 'updated_at_unix':time.time()})
            if pending or active:
                time.sleep(15)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
