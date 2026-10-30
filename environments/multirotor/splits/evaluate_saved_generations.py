"""Evaluate every saved checkpoint of the 2000-to-5000 continuation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_evaluation_experiment import HERE, ROOT, write_json

OUT = HERE / 'experiments/lookahead5_resume_5000'
SWEEP = OUT / 'generation_sweep'


def summarize(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for name, protocol in data['protocols'].items():
        episodes = [e for run in protocol['runs'] for e in run['episodes']]
        assert len(episodes) == 192
        successful = [e['time_s'] for e in episodes if e['success']]
        failures = {}
        for e in episodes:
            if not e['success']:
                key = str(e['target_gate_at_end'])
                failures[key] = failures.get(key, 0) + 1
        result[name] = dict(protocol['summary'], episodes=len(episodes),
            successes=len(successful), mean_success_time_s=sum(successful)/len(successful) if successful else None,
            ground_contact=sum(e['crash'] and not e['gate_collision'] for e in episodes)/len(episodes),
            failures_by_target_gate=failures)
    return result


def main():
    SWEEP.mkdir(exist_ok=True)
    checkpoints = sorted(OUT.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
    manifest = dict(pid=os.getpid(), started=datetime.now().isoformat(),
        seeds=[7, 17, 27], replicas_per_seed=64, protocols=['random_four_gates','fixed_full_track'],
        max_episode_seconds=20, workers=2,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints})
    write_json(SWEEP/'manifest.json', manifest)
    results = []; pending = []; errors = []
    def save_status(phase):
        write_json(SWEEP/'status.json', dict(phase=phase, pid=os.getpid(),
            updated=datetime.now().isoformat(), completed=len(results), total=len(checkpoints), errors=errors))
        write_json(SWEEP/'summary.json', sorted(results, key=lambda r:r['completed_updates']))
    def run(checkpoint, report):
        with (SWEEP/f'{checkpoint.stem}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(HERE/'run_evaluation_experiment.py'),
                '--worker','--checkpoint',str(checkpoint),'--report',str(report)],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        return summarize(report)
    for checkpoint in checkpoints:
        iteration = int(checkpoint.stem.split('_')[-1]) + 1
        existing = OUT/'evaluations'/f'continued_{iteration}.json'
        report = existing if existing.exists() else SWEEP/f'{checkpoint.stem}.json'
        row = dict(checkpoint=checkpoint.name, completed_updates=iteration, report=str(report))
        if report.exists():
            row['protocols'] = summarize(report)
            results.append(row)
        else:
            pending.append((checkpoint, report, row))
    save_status('running')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run,p,r): row for p,r,row in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                row['protocols'] = future.result()
                results.append(row)
                print(json.dumps(row), flush=True)
            except Exception as exc:
                errors.append(dict(checkpoint=row['checkpoint'],error=str(exc)))
                print(f'FAILED: {row["checkpoint"]}: {exc}', flush=True)
            save_status('running')
    save_status('failed' if errors else 'completed')
    if errors:
        raise RuntimeError(f'{len(errors)} checkpoint evaluations failed')


if __name__ == '__main__':
    main()
