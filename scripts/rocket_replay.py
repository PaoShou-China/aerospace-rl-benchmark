"""Rocket logging/replay CLI; --help and inspect do not import Genesis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visualization.rocket_data import Episode


def collect(args):
    import torch
    from environments.rocket import RocketLandingEnv
    from algorithms.baselines.baselines import build_baseline
    from visualization.rocket_recorder import RocketRecorder
    config = json.loads(args.config.read_text()) if args.config else {}
    env_kwargs = dict(config.get('env_kwargs') or {})
    if any(k in env_kwargs for k in ('seed', 'backend', 'num_envs')):
        raise ValueError('env_kwargs must not override seed/backend/num_envs')
    if args.checkpoint and not args.config:
        raise ValueError('--checkpoint requires the corresponding --config')
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(args.seed)
    env = RocketLandingEnv(num_envs=1, backend=args.backend, seed=args.seed, **env_kwargs)
    try:
        source = 'benchmark_pdg_baseline'
        if args.checkpoint:
            # Match the checkpoint reconstruction used by scripts/visualize.py.
            from scripts.visualize import _JsonStd
            from algorithms.train_utils import RslVecEnv
            dist = config.get('actor', {}).get('distribution_cfg', {})
            if isinstance(dist.get('init_std'), list):
                dist['init_std'] = _JsonStd(dist['init_std'])
            if config.get('algo') == 'flash_sac':
                from algorithms.flashsac.flash_sac import FlashSACRunner
                runner = FlashSACRunner(env, config, log_dir=str(args.out.parent/'replay_tb'), device=str(env.device))
                runner.load(str(args.checkpoint)); policy = runner.deterministic_action
            else:
                from rsl_rl.runners import OnPolicyRunner
                runner = OnPolicyRunner(RslVecEnv(env), config, log_dir=str(args.out.parent/'replay_tb'), device=str(env.device))
                runner.load(str(args.checkpoint)); policy = runner.alg.actor.eval()
            source = 'benchmark_checkpoint'
        else:
            controller = build_baseline('rocket', env)
            policy = controller.actions
        recorder = RocketRecorder(env, seed=args.seed, source=source, config=config, terminal_height=args.terminal_height)
        if args.checkpoint:
            recorder.metadata['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        env.visual_recorder = recorder
        with torch.no_grad():
            for _ in range(args.max_steps or env.max_steps):
                actions = policy(env.get_observations())
                env.step(actions)
                if recorder.ended:
                    break
        episode = recorder.episode()
        episode.save(args.out)
        (args.out/'policy_steps.json').write_text(json.dumps(recorder.policy_steps, indent=2, allow_nan=False))
        print(json.dumps(episode.data['result'], indent=2))
        print(f'Saved {len(episode.times)} real samples to {args.out}')
    finally:
        env.visual_recorder = None
        env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect', help='record the first real episode of one replica')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--backend', choices=('cpu','gpu'), default='cpu')
    c.add_argument('--seed', type=int, default=7)
    c.add_argument('--config', type=Path, help='explicit config.json; omitted = default calm environment')
    c.add_argument('--checkpoint', type=Path, help='optional trained policy; omitted = PDG baseline')
    c.add_argument('--max-steps', type=int, help='collection cap; never changes the task time limit')
    c.add_argument('--terminal-height', type=float, default=20.)
    for name in ('render','inspect','html'):
        q = sub.add_parser(name)
        q.add_argument('--episode', type=Path, required=True)
        if name == 'render':
            q.add_argument('--out', type=Path, required=True)
            q.add_argument('--backend', choices=('cpu','gpu'), default='cpu')
            q.add_argument('--fps', type=int, default=30);q.add_argument('--speed', type=float, default=1.)
            q.add_argument('--start', type=float, default=0.);q.add_argument('--end', type=float)
            q.add_argument('--width', type=int, default=1280);q.add_argument('--height', type=int, default=720)
            q.add_argument('--font', type=Path)
        elif name == 'inspect':
            q.add_argument('--time', type=float, default=0.)
        else:
            q.add_argument('--video', type=Path, required=True);q.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == 'collect':
            if args.max_steps is not None and args.max_steps < 1:
                raise ValueError('--max-steps must be positive')
            collect(args)
        else:
            e = Episode.load(args.episode)
            if args.command == 'inspect':
                print(json.dumps(e.metrics(args.time), indent=2))
            elif args.command == 'render':
                from visualization.rocket_render import render_video
                render_video(e, args.out, fps=args.fps, speed=args.speed, start=args.start, end=args.end,
                             backend=args.backend, width=args.width, height=args.height, font_path=args.font)
                print(args.out)
            else:
                from visualization.rocket_render import export_html
                export_html(e, args.video, args.out);print(args.out)
    except (ValueError, OSError, ModuleNotFoundError, ImportError) as exc:
        p.exit(2, f'rocket_replay: {exc}\nSee docs/rocket_genesis_replay.md for runtime requirements.\n')


if __name__ == '__main__':
    main()
