"""Engine-off randomized placement diagnostic; not a landing-policy score."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', choices=('v1', 'v2'), default='v2')
    p.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--seed', type=int, default=15000007)
    p.add_argument('--agent-hz', type=int, choices=(40,60), default=60)
    p.add_argument('--seconds', type=int, default=10)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():p.error('Output already exists; use a new path')
    if a.num_envs < 1 or not 1 <= a.seconds <= 30:p.error('invalid diagnostic size/duration')
    import torch
    from environments.rocket import RocketLandingEnv
    from environments.common.dynamics import quat_from_euler
    from scripts.first_episode import FirstEpisodes
    torch.set_num_threads(1)
    env = RocketLandingEnv(a.num_envs, a.backend, a.seed,
                          benchmark_version=a.version, agent_hz=a.agent_hz)
    try:
        d = env.device
        n = a.num_envs
        u = torch.rand(n,5,generator=torch.Generator().manual_seed(a.seed+1),dtype=torch.float64).to(d)
        position = torch.zeros(n,3,device=d,dtype=torch.float64)
        position[:,2] = 2.575 + .001 + .019*u[:,0]
        angles = torch.zeros_like(position);angles[:,2] = math.pi*(2*u[:,4]-1)
        velocity = torch.zeros_like(position);velocity[:,2] = -.4*u[:,1]
        heading = 2*math.pi*u[:,2]
        velocity[:,:2] = torch.stack((heading.cos(),heading.sin()),-1)*(.03*u[:,3:4])
        env.plant.set_state(position,quat_from_euler(angles),velocity,torch.zeros_like(velocity))
        mask = torch.ones(n,device=d,dtype=torch.bool)
        env._observe(mask);env._update_observation()
        # All replicas finish the same diagnostic horizon unless naturally fatal.
        env.max_steps = a.seconds*a.agent_hz
        rest = torch.zeros(n,7,device=d);rest[:,3:5] = -1
        cohort = FirstEpisodes(env)
        speeds = []
        with torch.no_grad():
            for step in range(env.max_steps):
                env.step(rest)
                if step >= max(0,env.max_steps-a.agent_hz):
                    v = env.plant.state()[2].norm(dim=-1)
                    speeds.extend(v[~cohort.seen & ~env.reset_buf].cpu().tolist())
                if cohort.update(env):break
        report = cohort.report()
        report.update(kind='passive-contact diagnostic, not policy evaluation',
            benchmark=env.cfg['benchmark'], seed=a.seed, placement_rng_offset=1,
            seconds=a.seconds, backend=a.backend,
            late_surviving_speed_median=float(torch.tensor(speeds).median()) if speeds else None)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
        if not report['complete']:raise RuntimeError('Incomplete passive diagnostic')
    finally:
        env.close()


if __name__=='__main__':main()
