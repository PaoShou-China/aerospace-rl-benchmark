"""Prepare, evaluate, or train the packaged physical actor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import snapshot as baseline

ROOT = baseline.ROOT


def package_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f'{path}: extract the artifacts archive into the package directory')


def check_recorded_artifact(path):
    require_file(path)
    records = json.loads((ROOT / 'provenance.json').read_text())['artifact_sha256']
    for name, expected in records.items():
        if path.resolve() == (ROOT / name).resolve() and baseline.sha256(path) != expected:
            raise RuntimeError(f'checkpoint checksum mismatch: {name}')


def prepare_evaluation(out, checkpoint):
    check_recorded_artifact(checkpoint)
    manifest_path = ROOT / 'results/training_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['source_sha256'].items():
        if baseline.sha256(ROOT / 'core' / name) != expected:
            raise RuntimeError(f'packaged model source differs from the recorded experiment: {name}')
    source = baseline.prepare_snapshot(out)
    for name in manifest['source_sha256']:
        shutil.copy2(ROOT / 'core' / name, source / 'src/common' / name)
    shutil.copy2(checkpoint, out / 'checkpoint.pt')
    shutil.copy2(ROOT / 'artifacts/parent/config.json', out / 'parent_config.json')
    shutil.copy2(manifest_path, out / 'manifest.json')
    baseline.write_json(out / 'status.json', {'status': 'prepared', 'checkpoint': str(checkpoint),
                                             'checkpoint_sha256': baseline.sha256(checkpoint)})
    baseline.verify_snapshot(out)


def train(args):
    config = json.loads(args.config.read_text())
    if args.from_parent:
        config.pop('resume', None)
    if args.resume is not None:
        config['resume'] = str(args.resume)
    for key in ('total_transitions', 'num_envs', 'microbatch', 'backend', 'evaluation_workers'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.num_envs is not None:
        capacity = config['num_envs'] * config['rollout']
        config['inference_batch'] = min(config['inference_batch'], config['num_envs'])
        for key in ('minibatch', 'microbatch', 'max_microbatch'):
            config[key] = min(config[key], capacity)
    parent = package_path(config['parent'])
    check_recorded_artifact(parent / config['checkpoint'])
    require_file(parent / 'config.json')
    config['parent'] = str(parent)
    if config.get('resume'):
        resume = package_path(config['resume'])
        check_recorded_artifact(resume)
        config['resume'] = str(resume)
    command = [sys.executable, '-u', str(ROOT / 'scripts/train.py'),
               '--out', str(args.out.resolve())]
    for key, value in config.items():
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend((option, str(value)))
    if args.skip_evaluation and '--skip-evaluation' not in command:
        command.append('--skip-evaluation')
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-eval', 'evaluate'):
        p = commands.add_parser(name)
        p.add_argument('--out', type=Path, required=True, help='New output directory')
        p.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/policy.pt')
        if name == 'evaluate':
            p.add_argument('--backend', choices=('gpu', 'cpu'), default='gpu')
            p.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    p = commands.add_parser('train')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/rocket_landing.json')
    start = p.add_mutually_exclusive_group()
    start.add_argument('--resume', type=Path)
    start.add_argument('--from-parent', action='store_true', help='Initialize a new physical actor from DAgger')
    p.add_argument('--total-transitions', type=int, help='Total lineage budget, including resumed transitions')
    p.add_argument('--num-envs', type=int)
    p.add_argument('--microbatch', type=int)
    p.add_argument('--backend', choices=('gpu', 'cpu'))
    p.add_argument('--evaluation-workers', type=int, choices=range(1, 5))
    p.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
        return
    out = args.out.resolve()
    model = out if args.command == 'prepare-eval' else out / 'model'
    if out.exists():
        raise FileExistsError(f'choose a new output directory: {out}')
    prepare_evaluation(model, package_path(args.checkpoint))
    if args.command == 'prepare-eval':
        print(model)
        return
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/evaluate.py'),
                    '--run-dir', str(model), '--out', str(out / 'evaluation'),
                    '--backend', args.backend, '--workers', str(args.workers)], check=True)


if __name__ == '__main__':
    main()
