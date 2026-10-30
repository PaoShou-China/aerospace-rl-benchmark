"""Versioned 4 x 128 first-episode evaluation of the repository PDG baseline.

Each cohort runs in a separate process; raw episode rows and environment
contracts are saved before aggregation. This does not load research policies.
"""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    p.add_argument('--variation', choices=('nominal', 'shifted', 'stress'), default='nominal')
    p.add_argument('--agent-hz', type=int, choices=(40, 60), default=40)
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--seeds', type=int, nargs='+', default=[1000007, 2000007, 3000007, 4000007])
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.num_envs < 1 or len(a.seeds) != len(set(a.seeds)):
        p.error('positive num-envs and unique seeds required')
    if (a.out/'evaluation.json').exists():p.error('Output already exists; use a new directory')
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.worker:
        reports = []
        for seed in a.seeds:
            dest = a.out / f'seed_{seed}'
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                '--out', str(dest), '--backend', a.backend,
                '--variation', a.variation, '--agent-hz', str(a.agent_hz),
                '--num-envs', str(a.num_envs), '--seeds', str(seed)], check=True)
            reports.append(json.loads((dest / 'evaluation.json').read_text()))
        n = sum(r['episodes_finished'] for r in reports)
        k = sum(r['successes'] for r in reports)
        z = 1.959963984540054
        den = 1 + z*z/n
        mid = (k/n + z*z/(2*n))/den
        half = z*((k/n)*(1-k/n)/n + z*z/(4*n*n))**.5/den
        report = dict(complete=all(r['complete'] for r in reports), n=n, successes=k,
            success_rate=k/n, wilson95=[mid-half, mid+half], seeds=a.seeds,
            num_envs_per_cohort=a.num_envs, benchmark=reports[0]['benchmark'],
            formal_512=(len(a.seeds)==4 and a.num_envs==128),
            cohorts=reports)
        (a.out / 'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='cohorts'}, indent=2))
        return

    import torch
    torch.set_num_threads(1)
    from environments.rocket import RocketLandingEnv
    from algorithms.baselines.baselines import build_baseline
    from scripts.first_episode import FirstEpisodes
    env = RocketLandingEnv(a.num_envs, a.backend, a.seeds[0],
        variation=a.variation, agent_hz=a.agent_hz)
    try:
        controller = build_baseline('rocket', env)
        cohort = FirstEpisodes(env)
        obs = env.get_observations()
        initial_sha = hashlib.sha256(obs['policy'].cpu().numpy().tobytes()).hexdigest()
        with torch.no_grad():
            for step in range(env.max_steps):
                obs, _, term, trunc, _ = env.step(controller.actions(obs))
                controller.reset_idx(term | trunc)
                if cohort.update(env):break
        report = cohort.report()
        report.update(benchmark=env.cfg['benchmark'], backend=a.backend, seed=a.seeds[0],
            controller=controller.name, env_steps=step+1, initial_observation_sha256=initial_sha,
            versions={k:importlib.metadata.version(k) for k in ('torch','genesis-world')},
            source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [ROOT/'environments/rocket/env.py', ROOT/'environments/common/plant.py',
                    ROOT/'environments/common/rocket_benchmark.py',
                    ROOT/'environments/common/original_backend.py',
                    ROOT/'algorithms/baselines/baselines.py', Path(__file__).resolve()]})
        (a.out/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
        if not report['complete']:raise RuntimeError('Incomplete first-episode cohort')
    finally:
        env.close()


if __name__ == '__main__':
    main()
