"""Quick experiment: conventional-controller SR under nominal plant vs default
domain randomization, for a few representative tasks.

Fixed seeds for reproducibility. Each task uses a first-episode cohort
(every replica finishes exactly one episode) so SR is unbiased.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
from environments import ENV_REGISTRY, make
from algorithms.baselines.baselines import build_baseline
from scripts.first_episode import FirstEpisodes

TASKS = ["quadx", "quadx_track", "fixedwing", "quadplane", "rocket", "missile"]
SEEDS = [7, 13, 21]
NUM_ENVS = 128

MAX_STEPS = {
    "quadx": 1000,
    "quadx_track": 1000,
    "fixedwing": 4000,
    "quadplane": 1400,
    "rocket": 2000,
    "missile": 1000,
}


def run_one(task, seed, domain_randomize, aero="realistic"):
    env_kwargs = {"domain_randomize": domain_randomize, "aero": aero}
    env = make(task, num_envs=NUM_ENVS, backend="cpu", seed=seed, **env_kwargs)
    try:
        controller = build_baseline(task, env, None)
        cap = MAX_STEPS.get(task, 1000)
        cohort = FirstEpisodes(env)
        obs = env.get_observations()
        steps = 0
        done = False
        with torch.no_grad():
            while steps < cap and not done:
                actions = controller.actions(obs)
                obs, _, _, _, _ = env.step(actions)
                cohort.update(env)
                steps += 1
                if steps % 32 == 0 and cohort.finished():
                    done = True
        r = cohort.report()
        return r["successes"] / r["episodes_finished"] if r["episodes_finished"] else 0.0
    finally:
        env.close()


def main():
    print(f"{'task':12s} {'seed':>5s}  {'SR_nominal':>11s}  {'SR_DR':>8s}  {'drop':>7s}")
    print("-" * 52)
    for task in TASKS:
        for seed in SEEDS:
            sr_nom = run_one(task, seed, False)
            sr_dr = run_one(task, seed, None)
            drop = sr_nom - sr_dr
            print(f"{task:12s} {seed:5d}  {sr_nom:11.3f}  {sr_dr:8.3f}  {drop:+7.3f}")
        print()


if __name__ == "__main__":
    main()
