"""Resume the completed 2000-update Split-S run for 3000 further updates."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from datetime import datetime

# Configure workspace-local runtime caches before importing environments.
from run_evaluation_experiment import HERE, ROOT, write_json


def main():
    import torch
    from environments.multirotor.splits import train
    from environments.multirotor.splits.ppo import build_config

    source = HERE / 'experiments/lookahead5_resume_2000'
    out = HERE / 'experiments/lookahead5_resume_5000'
    if (out / 'manifest.json').exists():
        raise FileExistsError(f'Experiment already exists: {out}')
    cfg = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=False)
    start = int(checkpoint['iter'])
    assert start == 2000, f'Unexpected source iteration: {start}'
    del checkpoint
    alg = cfg['algorithm']
    ppo_kwargs = dict(learning_rate=alg['learning_rate'], entropy_coef=alg['entropy_coef'],
                      init_std=cfg['actor']['distribution_cfg']['init_std'],
                      num_steps_per_env=cfg['num_steps_per_env'],
                      num_learning_epochs=alg['num_learning_epochs'],
                      num_mini_batches=alg['num_mini_batches'])
    expected = build_config(**ppo_kwargs)
    expected.update(env_kwargs=cfg['env_kwargs'], training_seed=cfg['training_seed'],
                    training_num_envs=cfg['training_num_envs'])
    assert expected == cfg, 'Current configuration differs from the source run'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'manifest.json', dict(
        source=str(source / 'checkpoint.pt'), start_iteration=start,
        additional_updates=3000, total_updates=5000,
        source_checkpoint_sha256=hashlib.sha256((source / 'checkpoint.pt').read_bytes()).hexdigest(),
        restored='Actor, critic, optimizer, normalization, curriculum buffer/window and RNG; episodes reset.',
        sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for p in HERE.glob('*.py'):
        shutil.copy2(p, snapshot / p.name)
    for name in ('checkpoint.pt', 'config.json'):
        shutil.copy2(source / name, out / name)
    baseline = source / 'evaluations/continued_2000.json'
    if baseline.exists():
        shutil.copy2(baseline, out / 'baseline_2000.json')

    def status(phase, iteration, **extra):
        write_json(out / 'status.json', dict(phase=phase, pid=os.getpid(),
            updated=datetime.now().isoformat(), iteration=iteration, total_updates=5000, **extra))

    def callback(path, runner):
        iteration = runner.current_learning_iteration + 1
        if path.name != 'checkpoint.pt':
            temporary = out / 'checkpoint.pending.pt'
            shutil.copy2(path, temporary)
            temporary.replace(out / 'checkpoint.pt')
        b = runner.env.initial_states
        write_json(out / 'training_metrics' / f'{iteration}.json', dict(
            iteration=iteration, success_window_rate=b.success_rate,
            window_count=len(b.success_history), replay_count=b.replay_count,
            reset_count=b.reset_count, ready=b.ready))
        status('training', iteration)
        # Native checkpoint filenames use zero-based update indices.
        if path.name != 'checkpoint.pt' and (
                runner.current_learning_iteration <= start or runner.current_learning_iteration % 500):
            return
        status('validation', iteration)
        report = out / 'evaluations' / f'continued_{iteration}.json'
        command = [sys.executable, '-u', str(HERE / 'run_evaluation_experiment.py'),
                   '--worker', '--checkpoint', str(path), '--report', str(report)]
        with (out / f'evaluation_{iteration}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if result.returncode:
            write_json(out / 'evaluations' / f'failed_{iteration}.json',
                       dict(iteration=iteration, returncode=result.returncode))
            print(f'Evaluation failed at {iteration}; training continues. See evaluation log.', flush=True)
        status('training', iteration)

    try:
        status('initializing', start)
        train.run_training(train.QuadXSplitsEnv, num_envs=cfg['training_num_envs'], updates=3000,
            seed=cfg['training_seed'], backend='gpu', resume=True, out_dir=out,
            env_kwargs=cfg['env_kwargs'], checkpoint_callback=callback, **ppo_kwargs)
        status('completed', 5000)
    except BaseException:
        status('failed', -1, error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
