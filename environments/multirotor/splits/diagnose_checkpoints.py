"""Evaluate saved policies on the same full-track initial conditions."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from algorithms.train_utils import RslVecEnv
from environments.multirotor.splits.env import QuadXSplitsEnv
from environments.multirotor.splits.ppo import SplitsRunner
from environments.multirotor.splits.evaluation import evaluate_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--final-only', action='store_true')
    args = parser.parse_args()
    folder = ROOT/'results/quadx_splits_4000'
    baseline = ROOT/'results/quadx_splits_extended/checkpoint.pt'
    paths = [baseline] + ([] if args.final_only else sorted(
        folder.glob('model_*.pt'), key=lambda p:int(p.stem.split('_')[1])))
    paths += [folder/'checkpoint.pt']
    cfg = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    rows = []
    for path in paths:
        torch.manual_seed(7)
        kwargs = dict(cfg['env_kwargs'])
        kwargs.update(full_track=True, initialization='fixed', laps=1)
        env = QuadXSplitsEnv(num_envs=64, backend='gpu', seed=7, **kwargs)
        try:
            runner = SplitsRunner(RslVecEnv(env), copy.deepcopy(cfg), device=str(env.device))
            runner.load(str(path), map_location=str(env.device))
            start = {k:v[0].tolist() for k,v in zip(('p','q','v','w'),env.plant.state())}
            report = evaluate_policy(env, runner.get_inference_policy())
            episodes = report['episodes']
            row = dict(checkpoint=str(path), next_iteration=runner.current_learning_iteration,
                initial_state=start, report=report,
                out_of_bounds_rate=sum(e['out_of_bounds'] for e in episodes)/len(episodes),
                timeout_rate=sum(e['timeout'] for e in episodes)/len(episodes),
                mean_time_s=sum(e['time_s'] for e in episodes)/len(episodes))
            rows.append(row)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps(dict(model=path.name, iteration=row['next_iteration'],
                gates=report['mean_gates_passed'], success=report['success_rate'],
                crash=report['crash_rate'], bounds=row['out_of_bounds_rate'])),flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
