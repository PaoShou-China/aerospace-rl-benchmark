"""Paired evaluation of final-gate zero padding versus a gate-1 preview.

Only the policy input is changed. The environment, seven-gate termination,
checkpoint, dynamics and initial conditions are identical between conditions.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Set writable runtime caches before importing environments.
from run_evaluation_experiment import HERE, ROOT, write_json

RUN = HERE / 'experiments/lookahead5_resume_5000'
OUT = RUN / 'last_gate_ablation_4800'
CHECKPOINT = RUN / 'model_4800.pt'
SEEDS = (7, 17, 27)


def worker(condition, seed):
    import torch
    from environments.multirotor.splits.evaluation import load_env_policy, evaluate_policy

    env, policy, _ = load_env_policy(RUN, 64, 'gpu', seed, full_track=True,
                                    checkpoint=CHECKPOINT, max_episode_seconds=20.)
    try:
        assert env.full_track and env.num_targets == 7 and env.max_steps == 800
        entries = {}
        modified_rows = 0
        with torch.no_grad():
            initial_state = env._capture_state().cpu().tolist()
            initial_observation = env.get_observations()['policy'].cpu().tolist()

        @torch.no_grad()
        def controlled_policy(observations):
            nonlocal modified_rows
            final_gate = env.target_index == 6
            raw = observations['policy']
            baseline_actions = policy(observations)
            if not bool(final_gate.any()):
                return baseline_actions
            assert bool((raw[final_gate, 24:36] == 0).all())
            changed = observations.clone()
            # Match env._observe's float64 body-frame corner transform exactly.
            rotation = env._observation[:, 3:12].reshape(-1, 3, 3)
            position = env.plant.state()[0]
            corners = env.gate_geometry.corners()[0]
            preview = torch.einsum('nkj,nck->ncj', rotation,
                                   corners[None] - position[:, None]).flatten(1)
            if condition == 'gate1_preview':
                changed['policy'][final_gate, 24:36] = preview[final_gate].to(raw.dtype)
                assert torch.equal(changed['policy'][:, :24], raw[:, :24])
                assert torch.equal(changed['policy'][~final_gate], raw[~final_gate])
                assert bool((changed['policy'][final_gate, 24:36] != 0).any(dim=1).all())
                modified_rows += int(final_gate.sum())
                actions = policy(changed)
            else:
                actions = baseline_actions
            first_episode = final_gate & (env.completed_episodes == 0)
            for index in first_episode.nonzero().flatten().tolist():
                if str(index) not in entries:
                    entries[str(index)] = dict(
                        env_step=int(env.count[index]),
                        physical_state=env._capture_state()[index].cpu().tolist(),
                        original_observation=raw[index].cpu().tolist(),
                        applied_observation=changed['policy'][index].cpu().tolist(),
                        baseline_action=baseline_actions[index].cpu().tolist(),
                        applied_action=actions[index].cpu().tolist())
            return actions

        result = evaluate_policy(env, controlled_policy)
        assert result['all_replicas_finished'] and len(result['episodes']) == 64
        assert all(e['gate_sequence'] == list(range(1, 8)) for e in result['episodes'])
        assert all(e['gates_passed'] == 7 for e in result['episodes'] if e['success'])
        if condition == 'gate1_preview':
            assert modified_rows > 0
        result.update(condition=condition, seed=seed, initial_state=initial_state,
                      initial_observation=initial_observation, gate7_entries=entries,
                      modified_policy_rows=modified_rows)
        write_json(OUT / f'{condition}_seed{seed}.json', result)
        print(json.dumps(dict(condition=condition, seed=seed, successes=result['successes'],
            mean_gates_passed=result['mean_gates_passed'],
            mean_success_time_s=result['mean_success_time_s'])), flush=True)
    finally:
        env.close()


def summary(runs):
    episodes = [e for r in runs for e in r['episodes']]
    times = [e['time_s'] for e in episodes if e['success']]
    failures = {}
    for e in episodes:
        if not e['success']:
            key = str(e['target_gate_at_end'])
            failures[key] = failures.get(key, 0) + 1
    return dict(episodes=len(episodes), successes=len(times), success_rate=len(times)/len(episodes),
        mean_gates_passed=sum(e['gates_passed'] for e in episodes)/len(episodes),
        mean_success_time_s=sum(times)/len(times) if times else None,
        gate_collisions=sum(e['gate_collision'] for e in episodes),
        ground_contacts=sum(e['crash'] and not e['gate_collision'] for e in episodes),
        out_of_bounds=sum(e['out_of_bounds'] for e in episodes), timeouts=sum(e['timeout'] for e in episodes),
        failures_by_target_gate=failures)


def main():
    if (OUT / 'manifest.json').exists():
        raise FileExistsError(f'Experiment already exists: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = [CHECKPOINT, RUN/'config.json', HERE/'env.py', HERE/'evaluation.py', HERE/'ppo.py',
               HERE/'gate_geometry.py', HERE/'split_s.py',
               ROOT/'environments/common/plant.py', ROOT/'environments/common/dynamics.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    write_json(OUT/'manifest.json', dict(checkpoint=str(CHECKPOINT), seeds=SEEDS,
        replicas=64, seconds=20, completed_gates=7, files_sha256=hashes,
        intervention='Only policy observation columns 24:36 when target_index == 6; preview gate 1.',
        script=Path(__file__).read_text(encoding='utf-8')))
    runs = {name: [] for name in ('zero_padding', 'gate1_preview')}
    def status(phase, **extra):
        write_json(OUT/'status.json', dict(phase=phase, pid=os.getpid(),
            updated=datetime.now().isoformat(), **extra))
    try:
        for seed in SEEDS:
            for condition in runs:
                status('running', condition=condition, seed=seed)
                with (OUT/f'{condition}_seed{seed}.log').open('w', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                        '--condition', condition, '--seed', str(seed)], cwd=str(ROOT),
                        stdout=log, stderr=subprocess.STDOUT, check=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                result = json.loads((OUT/f'{condition}_seed{seed}.json').read_text(encoding='utf-8'))
                runs[condition].append(result)
                print(json.dumps(dict(condition=condition, seed=seed,
                    successes=result['successes'], mean_gates=result['mean_gates_passed'])), flush=True)
        import numpy as np
        paired = []
        for baseline, preview in zip(runs['zero_padding'], runs['gate1_preview']):
            assert baseline['initial_state'] == preview['initial_state']
            assert baseline['initial_observation'] == preview['initial_observation']
            a = baseline['gate7_entries']; b = preview['gate7_entries']
            shared = sorted(set(a) & set(b))
            max_state_delta = max((float(np.max(np.abs(np.array(a[i]['physical_state']) -
                                                       np.array(b[i]['physical_state'])))) for i in shared), default=0.)
            max_obs_delta = max((float(np.max(np.abs(np.array(a[i]['original_observation']) -
                                                     np.array(b[i]['original_observation'])))) for i in shared), default=0.)
            action_deltas = [float(np.linalg.norm(np.array(b[i]['applied_action']) -
                                                  np.array(b[i]['baseline_action']))) for i in shared]
            paired.append(dict(seed=baseline['seed'], initial_states_identical=True,
                baseline_gate7_entries=len(a), preview_gate7_entries=len(b), shared_gate7_entries=len(shared),
                max_gate7_entry_state_delta=max_state_delta,
                max_gate7_entry_original_observation_delta=max_obs_delta,
                mean_action_change_at_gate7_entry=float(np.mean(action_deltas)) if action_deltas else None))
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in tracked)
        result = dict(checkpoint=str(CHECKPOINT), summaries={k: summary(v) for k,v in runs.items()},
                      per_seed_pairing=paired, source_and_checkpoint_unchanged=True)
        write_json(OUT/'comparison.json', result)
        status('completed')
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        import traceback
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('zero_padding', 'gate1_preview'))
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    worker(args.condition, args.seed) if args.condition else main()
