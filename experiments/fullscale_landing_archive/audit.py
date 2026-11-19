"""Read-only consistency audit of the bundled historical aggregates (stdlib only)."""
import hashlib
import json
from pathlib import Path


def audit(root):
    def read(name):
        return json.loads((root / 'evidence' / name).read_text())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    manifest = json.loads((root / 'manifest.json').read_text())
    for name, hashes in manifest['files'].items():
        actual = hashlib.sha256((root / 'evidence' / name).read_bytes()).hexdigest()
        require(actual == hashes['curated_sha256'], f'Changed evidence: {name}')
    p, r = read('plan.json'), read('results.json')
    names = ['bc'] + [f'ppo_{s}' for s in p['training_seeds']]
    expected = {(n, s) for n in names for s in p['evaluation_seeds']}
    rows = r['evaluations']
    require(len(rows) == len(expected), 'Wrong panel count')
    require({(x['name'], x['eval_seed']) for x in rows} == expected, 'Missing/duplicate panels')
    totals = {}
    for name in names:
        group = [x for x in rows if x['name'] == name]
        require(len({x['model_sha256'] for x in group}) == 1, 'Checkpoint changed within panel')
        require(len({x['ppo_updates'] for x in group}) == 1, 'Updates changed within panel')
        for x in group:
            require(x['n'] == p['eval_replicas_per_seed'], 'Wrong denominator')
            require(0 <= x['successes'] <= x['n'], 'Invalid count')
            require(x['successes_on_pad'] == x['successes'], 'Metric disagreement')
        totals[name] = {'successes': sum(x['successes'] for x in group),
                        'n': sum(x['n'] for x in group),
                        'updates': group[0]['ppo_updates']}
    p, choice = read('scaling_plan.json'), read('scaling_choice.json')
    require(choice['n_per_cell'] == len(p['selection_seeds']) * p['eval_replicas'], 'BC denominator')
    require(set(p['train_shards']['small']) < set(p['train_shards']['large']), 'Non-nested data')
    for name, cell in choice['cells'].items():
        require(cell['train_episodes'] == len(p['train_shards'][cell['data']]) * 256, 'Training episodes')
        require(0 <= choice['successes'][name] <= choice['n_per_cell'], 'BC count')
    winner = min(choice['cells'], key=lambda n: (-choice['successes'][n],
                 choice['cells'][n]['width'], choice['cells'][n]['train_episodes']))
    require(winner == choice['name'], 'BC selection mismatch')
    frozen, selected, result = (read(n) for n in
        ['scaling_freeze.json', 'scaling_final_choice.json', 'scaling_results.json'])
    names = {f'{prefix}_{s}' for prefix in ['ppo', 'scaled_ppo'] for s in p['train_seeds']}
    require(set(frozen['hashes']) == set(selected['hashes']) == set(result['successes']) == names,
            'Scaling model coverage')
    require(selected['n_per_update'] == len(names) * len(p['checkpoint_validation_seeds']) * p['eval_replicas'],
            'Selection denominator')
    counts = selected['validation_successes']
    require(set(map(int, counts)) == set(frozen['updates']), 'Missing milestone')
    chosen = min(frozen['updates'], key=lambda u: (-counts[str(u)], u))
    require(chosen == selected['update'] == result['matched_update'], 'Milestone mismatch')
    for n in names:
        require(selected['hashes'][n] == frozen['hashes'][n][str(chosen)], 'Frozen hash mismatch')
    require(result['n_per_model'] == len(p['test_seeds']) * p['eval_replicas'], 'Test denominator')
    require(all(v == 0 for v in result['successes'].values()), 'Archive expected all-zero test')
    require(result['paired_difference'] == 0 and result['crossed_seed_bootstrap_95'] == [0, 0],
            'All-zero aggregate mismatch')
    return {'record_consistency': 'passed', 'original_run_reproduced': False,
            'first_study': totals, 'scaling_n_per_model': result['n_per_model'],
            'scaling_models': len(names), 'selected_update': chosen}


if __name__ == '__main__':
    print(json.dumps(audit(Path(__file__).resolve().parent), indent=2))
