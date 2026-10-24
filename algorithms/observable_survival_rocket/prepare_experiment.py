"""Create an isolated experiment bundle; never edit the original benchmark.

Run from a checkout of the requested branch. No SSH credentials are read,
written or required by this utility. Remote execution is a separate action.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

BASE_COMMIT = '91a17f96e2ee289720f6bee521aab35f0ceb7647'
SOURCE_BLOBS = {
    'core/rocket_confident_physics.py':'7e0c94a2302c931f896a54684e7c3d7b8c0120f6',
    'scripts/train.py':'f5d049b4c253e4bdc5f0bf5165a4f52be6ff5d5e',
    'scripts/run.py':'b25e4bf3913b73b4623e7f63ad2f297dcfe78a89',
}


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{name}: expected one patch site, got {count}; source drift, stop')
    return text.replace(old, new, 1)


def patch_factory(text):
    marker = 'def build_physical_actor(seed_policy, config):\n'
    addition = '''    if config.get('architecture') == 'observable_survival':
        from .rocket_observable_survival import ObservableSurvivalActor
        return ObservableSurvivalActor(seed_policy,
            horizon=config['horizon'], iterations=config['iterations'],
            hypotheses=config.get('hypotheses', 25), knots=config.get('knots', 3),
            ablation=config.get('ablation', 'full'),
            confidence=not config.get('disable_confidence', False),
            tail_steps=config.get('tail_steps', 40), tail_weight=config.get('tail_weight', .15),
            observable_update=not config.get('disable_observable_update', False))
'''
    return replace_once(text, marker, marker+addition, 'actor factory')


def patch_train(text):
    text = replace_once(text, '"rocket_ppo_practical_support.py", "rocket_confident_physics.py")',
        '"rocket_ppo_practical_support.py", "rocket_confident_physics.py",\n'
        '            "observable_belief.py", "rocket_observable_survival.py", "survival_metrics.py")',
        'source overlay manifest')
    text = replace_once(text, 'choices=("published", "confidence_sequence")',
        'choices=("published", "confidence_sequence", "observable_survival")', 'architecture option')
    text = replace_once(text, '    args = parser.parse_args()\n',
        '    parser.add_argument("--tail-steps", type=int, default=40)\n'
        '    parser.add_argument("--tail-weight", type=float, default=.15)\n'
        '    parser.add_argument("--disable-observable-update", action="store_true")\n'
        '    args = parser.parse_args()\n', 'new options')
    text = replace_once(text, '    args = parser.parse_args()\n',
        '    parser.add_argument("--additional-transitions", type=int)\n'
        '    args = parser.parse_args()\n'
        '    if args.additional_transitions is not None:\n'
        '        if args.total_transitions is not None:\n'
        '            parser.error("additional-transitions and total-transitions are exclusive")\n'
        '        if args.additional_transitions <= 0 or args.additional_transitions % args.num_envs:\n'
        '            parser.error("additional transition budget must be positive and divisible by num-envs")\n',
        'additional interaction budget')
    marker = '        remaining_transitions = transition_target-status["new_environment_transitions"]\n'
    text = replace_once(text, marker,
        '        if args.additional_transitions is not None:\n'
        '            transition_target = status["new_environment_transitions"] + args.additional_transitions\n'
        '            status["planned_environment_transitions"] = transition_target\n' + marker,
        'lineage-aware interaction target')
    text = replace_once(text, "            if args.architecture == 'confidence_sequence':\n",
        "            if args.architecture in ('confidence_sequence', 'observable_survival'):\n",
        'initial physical authority')
    marker = '        baseline.write_json(out/"manifest.json", manifest)\n'
    metadata = '''        if args.architecture == 'observable_survival':
            manifest['components'] = [
                'causal recurrent context', 'Joseph-form full-covariance effective-parameter belief',
                'local physical sensitivities', '25-point moment-matching cubature',
                'six-DOF flight', 'three-leg contact surrogate',
                'learned sequence optimizer', 'two ignition branches',
                'fixed engine-off settling continuation', 'final hybrid likelihood PPO',
                'actual-executed-action prediction loss', 'original evaluation plus sampled terminal dwell']
            manifest['belief_is_calibrated_probability'] = False
            manifest['contact_parameters_identified'] = False
            manifest['state_estimation_covariance_propagated'] = False
            manifest['tail_is_safety_certificate'] = False
'''
    text = replace_once(text, marker, metadata+marker, 'accurate architecture metadata')
    return text


def patch_run(text):
    # Fix conflicting initialization choices in this isolated copy only.
    text = replace_once(text, "    if args.from_parent:\n        config.pop('resume', None)\n",
        "    if args.from_parent:\n        config.pop('resume', None)\n        config.pop('initialize_from', None)\n",
        'from-parent precedence')
    text = replace_once(text, "    if args.resume is not None:\n        config['resume'] = str(args.resume)\n",
        "    if args.resume is not None:\n        config.pop('initialize_from', None)\n        config['resume'] = str(args.resume)\n",
        'resume precedence')
    if "    parent = package_path(config['parent'])\n" in text:
        text = replace_once(text, "    parent = package_path(config['parent'])\n",
            '    if args.total_transitions is not None:\n'
            '        config.pop("additional_transitions", None)\n'
            "    parent = package_path(config['parent'])\n", 'explicit total-budget override')
    return text


def patch_evaluate(text):
    addition = '''        from common.survival_metrics import StableLandingMonitor
        survival_monitor = StableLandingMonitor(128, dt=1/40, hold_seconds=2.)

        def observe_survival(active):
            _, q_s, v_s, w_s = env.plant.state()
            tilt_s = euler(q_s)[:, :2].norm(dim=-1)
            settled_s = (env.plant.pad_contact & ~env.fatal_collision &
                         (v_s.norm(dim=-1)<.02) & (w_s.norm(dim=-1)<.02) & (tilt_s<.1))
            survival_monitor.update(*[x.detach().cpu().numpy() for x in
                (settled_s, env.complete, env.fatal_collision, env.out_of_bounds)],
                active=active.detach().cpu().numpy())

'''
    text = replace_once(text, '        original_reset = env.reset_idx\n',
        addition+'        original_reset = env.reset_idx\n', 'monitor creation')
    text = replace_once(text, '            if ids:\n                _, q, v, w = env.plant.state()\n',
        '            if ids:\n                observe_survival(first)\n                _, q, v, w = env.plant.state()\n',
        'terminal pre-reset sample')
    text = replace_once(text, '                    completed += int(book.add(i, row))\n',
        '                    row.update(survival_monitor.finish(i))\n'
        '                    completed += int(book.add(i, row))\n', 'terminal metric fields')
    text = replace_once(text, '                following, _, done, extra = env.step(action)\n',
        '                following, _, done, extra = env.step(action)\n'
        '                observe_survival(live_before & ~done)\n', 'nonterminal metric samples')
    for filename in ('report.json', 'development512.json'):
        marker = f'        baseline.write_json(out/"{filename}", report)\n'
        addition = '''        rows = [row for row in report['episodes'] if row is not None]
        report['survival_summary'] = {
            'denominator':len(rows),
            'durable_successes':sum(bool(row.get('durable_success', False)) for row in rows),
            'sampled_terminal_two_second_successes':sum(bool(row.get('sampled_two_second_terminal_settle', False)) for row in rows),
            'metric_sampling_hz':40, 'required_terminal_dwell_seconds':2.,
            'changes_original_success_criterion':False}
'''
        text = replace_once(text, marker, addition+marker, f'{filename} metric summary')
    return text


def prepare(repo: Path, out: Path) -> Path:
    source = repo.resolve()/'algorithms/physical_belief_rocket_landing'
    own = Path(__file__).resolve().parent
    if out.exists():
        raise FileExistsError(f'output already exists: {out}; choose a fresh directory')
    for name, expected in SOURCE_BLOBS.items():
        actual = git_blob(source/name)
        if actual != expected:
            raise RuntimeError(f'{name}: source hash differs from reviewed branch; reconcile before retry')
    if not (source/'vendor/benchmark_source.tar.gz').is_file():
        raise FileNotFoundError('frozen environment archive missing')
    bundle = out/'bundle'
    out.mkdir(parents=True)
    shutil.copytree(source,bundle,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','runs'))
    for name in ('observable_belief.py','rocket_observable_survival.py','survival_metrics.py'):
        shutil.copy2(own/name,bundle/'core'/name)
    changes = {'core/rocket_confident_physics.py':patch_factory,
               'scripts/train.py':patch_train,'scripts/run.py':patch_run,
               'scripts/evaluate.py':patch_evaluate}
    for name, patcher in changes.items():
        path = bundle/name
        result = patcher(path.read_text())
        ast.parse(result)
        path.write_text(result)
    # All methods use the same persistent training reward and the SAME 25
    # hypotheses. This is a new experiment, not a rerun of the published score.
    base = dict(parent='artifacts/parent',checkpoint='parent.pt',
                architecture='observable_survival',hypotheses=25,knots=3,
                initialize_from='artifacts/policy.pt',reward_variant='persistent',
                backend='gpu',num_envs=16,rollout=8,updates=2,epochs=1,
                minibatch=128,microbatch=8,max_microbatch=8,inference_batch=8,
                batch_tune_every=2,gpu_memory_fraction=.7,horizon=6,iterations=1,
                seed=730101,tail_steps=40,tail_weight=.15,evaluation_workers=1,
                skip_evaluation=True)
    configs = {'smoke':dict(base,tail_steps=2),
               'full':dict(base,updates=256,rollout=32),
               'no_tail':dict(base,updates=256,rollout=32,tail_weight=0.),
               'fixed_belief':dict(base,updates=256,rollout=32,disable_observable_update=True),
               'no_solver':dict(base,updates=256,rollout=32,ablation='no_solver'),
               'confidence_control':dict(base,architecture='confidence_sequence',updates=256,rollout=32)}
    for name, cfg in configs.items():
        cfg['additional_transitions'] = cfg['updates'] * cfg['rollout'] * cfg['num_envs']
        (out/f'{name}.json').write_text(json.dumps(cfg,indent=2)+'\n')
    provenance = {'base_commit':BASE_COMMIT,'remote_training_has_run':False,
                  'reference_environment_archive_changed':False,
                  'source_blob_checks':SOURCE_BLOBS,
                  'modified_source_sha256':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in changes},
                  'extension_sha256':{name:hashlib.sha256((own/name).read_bytes()).hexdigest() for name in
                      ('observable_belief.py','rocket_observable_survival.py','survival_metrics.py')},
                  'important':'Original benchmark success retained. New durable-success metric is supplemental.'}
    (out/'preparation.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Prepared only. Training has NOT started.\nBundle: {bundle}')
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    prepare(args.repo,args.out.resolve())

if __name__=='__main__':
    main()
