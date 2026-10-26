"""Evaluate a Split-S checkpoint over fixed-start full-track episodes."""
import argparse
import json
import sys
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_ROOT))
from environments.multirotor.splits.evaluation import load_env_policy,evaluate_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=_ROOT/'results'/'quadx_splits')
    parser.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    parser.add_argument('--laps',type=int,default=1)
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    reports=[]
    for seed in args.seeds:
        env,policy,_=load_env_policy(args.out,args.num_envs,args.backend,seed,args.laps)
        try:
            report=evaluate_policy(env,policy,args.max_steps)
            report.update(seed=seed,task=env.cfg)
            reports.append(report)
        finally:env.close()
    report=dict(checkpoint=str(args.out/'checkpoint.pt'),seeds=args.seeds,laps=args.laps,runs=reports,
                success_rate=sum(r['successes'] for r in reports)/sum(r['scheduled_episodes'] for r in reports))
    (args.out/'eval_splits.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
