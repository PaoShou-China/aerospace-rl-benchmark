"""Shared machinery for the per-task training entries (train_<task>.py).

Importing this module bootstraps the `pyflyt_genesis` package (src/ as the
package root: env subpackages keep their relative imports and the frozen
original_pyflyt sources keep their absolute ones), so each per-task entry
only needs `src/` on sys.path plus one import of its environment class.

`run_training()` reproduces the exact rsl-rl PPO loop, checkpoint handling
and run reporting of the original train.py. All run parameters are passed in
as plain constants from the per-task entry files -- nothing is read from the
command line -- and the PPO/model configuration is identical across tasks so
results stay directly comparable.
"""
import copy
import json
import math
import shutil
import sys
import time
import types
from pathlib import Path

# --- pyflyt_genesis bootstrap (idempotent) ---
_src = Path(__file__).resolve().parents[1]
if 'pyflyt_genesis' not in sys.modules:
    _pkg = types.ModuleType('pyflyt_genesis')
    _pkg.__path__ = [str(_src)]
    sys.modules['pyflyt_genesis'] = _pkg

import torch
from rsl_rl.runners import OnPolicyRunner


def build_cfg(*, lr_schedule='fixed',
              learning_rate=.001, desired_kl=.01, entropy_coef=.001, init_std=.5,
              gamma=.99, lam=.95, num_steps_per_env=32, num_learning_epochs=4,
              num_mini_batches=4, clip_param=.2, value_loss_coef=1.0,
              arch='flash_residual', trunk_hidden=128, critic_trunk_hidden=256):
    """PPO/model config, identical for every task (fair-comparison baseline).

    Keyword overrides exist so a task entry can vary exactly the variables
    under test while everything else stays at the shared defaults.

    arch: 'flash_residual' (default, the FlashSAC network structure via
    src/common/ppo_flashnet.py: BN embedder + 2 residual blocks + RMSNorm
    trunk, actor 2x128 / critic 2x256, UnitLinear/RMSNorm weight
    renormalization on the update path; BN-free for PPO -- BatchNorm is
    incompatible with the on-policy ratio) or 'mlp' (opt-out 2x128 tanh MLP,
    kept for architecture A/B arms). Trunk is the project-wide default since
    the 2026-10-21 pendulum 3-way A/B: under the shared warmup-cosine
    schedule it beats the MLP by 66 return points (-153.1 vs -219.4) and
    nearly matches native FlashSAC (-144.4).

    trunk_hidden / critic_trunk_hidden: widths of the residual trunk for the
    actor / critic under arch='flash_residual'. Defaults (128/256) are the
    project-wide baseline; narrow widths can be pinned as an explicit A/B
    (e.g. rendezvous, where 64x64 beat 512x256x128 by 22 points).

    Shared lr defaults follow the 2026-10-20 tuning history: the KL-adaptive
    schedule is the project-wide pathology (rides its 1e-2 ceiling on
    rendezvous/satellite, slams to its 1e-5 floor on fixedwing, thrashes
    1e-5..8.65e-4 elsewhere) and fixed 3e-4 stalls from scratch (quadx stuck
    at -377 with 0% success, results/quadx_sweep). Fixed 1e-3 is the A/B
    winner on every task where the two were compared (quadx 300-update sweep,
    fixedwing 150-update A/B, docking, rendezvous), so it is the shared
    default; entries with a different proven optimum pin it explicitly
    (docking fixed 3e-4, transfer adaptive 3e-4)."""
    if arch not in ('flash_residual', 'mlp'):
        raise ValueError(f'unknown arch {arch!r} (expected flash_residual | mlp)')
    if arch == 'flash_residual':
        flash_cfg = {'class_name': 'common.ppo_flashnet:FlashResMLPModel'}
        # use_bn=False: BatchNorm is incompatible with PPO's on-policy ratio --
        # with batch stats the log-prob of a sample depends on the OTHER samples
        # in its batch, so the rollout policy (stats over 1024 replicas/step) and
        # the update policy (stats over 8k minibatches) are different functions
        # no matter how the stats are shared (verified 2026-10-21 bisection:
        # BN split / batch-everywhere / renorm-off / head-gain / no-value-clip /
        # no-BN-critic-only ALL flat-lined; the critic is innocent -- offline it
        # fits returns 3x better than the MLP). The trunk keeps FlashSAC's
        # UnitLinear residual geometry + RMSNorm + weight renormalization;
        # input whitening moves to EmpiricalNormalization (as in the MLP arm).
        actor = {**flash_cfg, 'trunk_hidden': trunk_hidden, 'trunk_blocks': 2, 'use_bn': False,
                 'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': init_std, 'std_type': 'log'}}
        critic = {**flash_cfg, 'trunk_hidden': critic_trunk_hidden, 'trunk_blocks': 2, 'use_bn': False}
    else:
        actor = {'class_name': 'MLPModel', 'hidden_dims': [128, 128], 'activation': 'tanh', 'obs_normalization': True,
                 'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': init_std, 'std_type': 'log'}}
        critic = {'class_name': 'MLPModel', 'hidden_dims': [128, 128], 'activation': 'tanh', 'obs_normalization': True}
    return {'num_steps_per_env': num_steps_per_env, 'save_interval': 100, 'obs_groups': {'actor': ['policy'], 'critic': ['policy']},
            'actor': actor,
            'critic': critic,
            'algorithm': {'class_name': 'PPO', 'learning_rate': learning_rate, 'num_learning_epochs': num_learning_epochs, 'num_mini_batches': num_mini_batches,
                          'schedule': lr_schedule, 'desired_kl': desired_kl, 'entropy_coef': entropy_coef, 'gamma': gamma, 'lam': lam,
                          'clip_param': clip_param, 'value_loss_coef': value_loss_coef},
            'logger': 'tensorboard'}


def run_training(task, env_cls, num_envs=64, updates=1000, seed=7, backend='gpu',
                 resume=False, clean=True, out_dir=None, lr_decay_updates=None,
                 lr_cosine=False, learning_rate_end=1.5e-4,
                 freeze_obs_norm=False, lr_sigma_coupling=None, sigma_pin=None,
                 algo='ppo', env_kwargs=None, **algo_overrides):
    """Train one task end-to-end and write checkpoint.pt / run.json / config.json.

    env_kwargs: extra keyword arguments forwarded to the env constructor
    (e.g. rendezvous's opt-in range curriculum); None leaves every other
    task entry's call signature unchanged.

    lr_decay_updates: optional exponential lr half-life in updates (opt-in;
    None keeps the constant lr the other task entries use).

    lr_cosine (PPO only): warmup-cosine lr schedule, the IDENTICAL schedule
    flash_sac's _set_lr applies -- same _warmup_cosine_decay formula, same
    lr_warmup_rate=1e-6 fraction of total updates, same peak->end span. Use it
    to put both algorithms under one shared schedule for fair A/Bs (with
    schedule='fixed' rsl-rl never reapplies alg.learning_rate, so the
    optimizer's param_groups are written directly, as in lr_decay_updates).
    learning_rate_end is the cosine floor (flash_sac_cfg's paper default
    1.5e-4; for flash_sac it simply forwards to flash_sac_cfg).

    freeze_obs_norm (PPO only): pin both empirical obs normalizers in eval
    mode after the checkpoint load -- EmpiricalNormalization.update() no-ops
    there, so a warm start's fitted input mapping cannot drift while PPO
    rollouts stream through. The pin is re-applied inside a wrapped
    alg.train_mode() because every learn() call recursively re-enables it.

    lr_sigma_coupling (PPO only): opt-in lr = k * sigma schedule, mutually
    exclusive with lr_decay_updates. The per-update mean policy drift scales
    as lr/sigma, so a fixed lr pushes the policy progressively harder as the
    exploration std anneals; after every 25-update chunk the lr is rewritten
    as k * mean action std to keep the drift rate constant. The coupling is
    applied once BEFORE the first chunk too -- at a pinned small sigma the
    uncoupled initial lr can sit in the collapse band (lr/sigma > ~3.3e-4).

    sigma_pin (PPO only): re-pin the exploration std to this value at every
    chunk (the entropy bonus and policy gradient otherwise drift it). With
    lr_sigma_coupling the lr follows the pinned sigma. Pinned sigma keeps the
    noisy-rollout objective aligned with greedy behavior: at sigma=0.5 the
    in-rollout success collapses and PPO's gradient chases noise (measured
    dose-response, 100-update gates: sigma 0.5 -> 12% greedy, 0.35 -> 46%,
    0.2 -> holds ~60-64%).

    algo: 'ppo' (default, rsl-rl OnPolicyRunner) or 'flash_sac'
    (src/common/flash_sac.py, Kim et al. 2026). Both consume the identical
    batched env protocol, checkpoint naming and reporting pipeline so results
    stay directly comparable across algorithms.

    Extra keyword arguments are forwarded to the selected algorithm's config
    builder (build_cfg for PPO, flash_sac.flash_sac_cfg for FlashSAC) so a
    task entry can vary exactly the variables under test while everything
    else stays at the shared fair-comparison defaults.
    """
    torch.manual_seed(seed)
    root = Path(__file__).resolve().parents[2]
    out = Path(out_dir) if out_dir is not None else root / 'results' / task
    out.mkdir(parents=True, exist_ok=True)
    if not resume and clean:
        # fresh run: drop stale models, tensorboard events and reports
        for p in list(out.glob('model_*.pt')) + [out / 'checkpoint.pt', out / 'run.json',
                                                 out / 'config.json', out / 'params']:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        for p in out.glob('events.out.tfevents.*'):
            p.unlink()

    env = env_cls(num_envs=num_envs, backend=backend, seed=seed, **(env_kwargs or {}))
    if algo == 'flash_sac':
        from .flash_sac import FlashSACRunner, flash_sac_cfg
        valid = set(flash_sac_cfg())
        unknown = set(algo_overrides) - valid
        if unknown:
            print(f'warning: {sorted(unknown)} are PPO-only overrides, ignored for flash_sac', flush=True)
        cfg = {'algo': 'flash_sac', **flash_sac_cfg(**{k: v for k, v in algo_overrides.items() if k in valid})}
        runner = FlashSACRunner(env, cfg, log_dir=str(out), device=str(env.device))
        steps_per_iter = 1  # one vectorized env step (num_envs transitions) per iteration
    elif algo == 'ppo':
        cfg = build_cfg(**algo_overrides)
        # OnPolicyRunner mutates cfg in place (resolving/popping class_name); pass a
        # deep copy so the dumped config.json stays a faithful training config.
        runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir=str(out), device=str(env.device))
        steps_per_iter = cfg['num_steps_per_env']
    else:
        raise ValueError(f'unknown algo {algo!r} (expected ppo | flash_sac)')
    if resume:
        cands = sorted(out.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
        ckpt = cands[-1] if cands else out / 'checkpoint.pt'
        if ckpt.exists():
            runner.load(str(ckpt))
            print(f'resumed from {ckpt}', flush=True)
        else:
            print('no checkpoint found, training from scratch', flush=True)
    if freeze_obs_norm:
        if algo != 'ppo':
            raise ValueError('freeze_obs_norm is PPO-only')

        def _pin_norms():
            for m in (runner.alg.actor, runner.alg.critic):
                if hasattr(m, 'obs_normalizer'):
                    m.obs_normalizer.eval()

        _orig_train_mode = runner.alg.train_mode

        def _train_mode():
            _orig_train_mode()
            _pin_norms()

        runner.alg.train_mode = _train_mode
        _pin_norms()
        print('obs normalizers frozen (eval mode)', flush=True)

    if lr_decay_updates and (lr_sigma_coupling or sigma_pin):
        raise ValueError('lr_decay_updates and lr_sigma_coupling/sigma_pin are mutually exclusive')
    if lr_cosine and (lr_decay_updates or lr_sigma_coupling or sigma_pin):
        raise ValueError('lr_cosine is mutually exclusive with lr_decay_updates/lr_sigma_coupling/sigma_pin')
    start = time.perf_counter()
    if algo == 'ppo' and lr_cosine:
        # warmup-cosine lr, byte-identical schedule to flash_sac's _set_lr
        from .flash_sac import _warmup_cosine_decay
        lr0 = cfg['algorithm']['learning_rate']
        warmup = max(1, int(1e-6 * updates))
        step = max(1, updates // 100)
        done = 0
        while done < updates:
            n = min(step, updates - done)
            runner.learn(n)
            done += n
            lr = _warmup_cosine_decay(done, lr0, lr0, learning_rate_end, warmup, updates)
            runner.alg.learning_rate = lr
            for group in runner.alg.optimizer.param_groups:
                group['lr'] = lr
    elif algo == 'ppo' and lr_decay_updates:
        # exponential lr half-life; with schedule='fixed' rsl-rl never reapplies
        # alg.learning_rate to the optimizer (that path is inside the adaptive
        # branch), so write the optimizer's param_groups directly
        lr0 = cfg['algorithm']['learning_rate']
        step = max(1, lr_decay_updates // 4)
        done = 0
        while done < updates:
            n = min(step, updates - done)
            runner.learn(n)
            done += n
            lr = lr0 * 0.5 ** (done / lr_decay_updates)
            runner.alg.learning_rate = lr
            for group in runner.alg.optimizer.param_groups:
                group['lr'] = lr
    elif algo == 'ppo' and (lr_sigma_coupling or sigma_pin):
        def _couple():
            if sigma_pin:
                with torch.no_grad():
                    runner.alg.actor.distribution.log_std_param.fill_(math.log(sigma_pin))
                sigma = sigma_pin
            else:
                with torch.no_grad():
                    sigma = float(runner.alg.actor.distribution.log_std_param.exp().mean())
            lr = lr_sigma_coupling * sigma if lr_sigma_coupling else None
            if lr is not None:
                runner.alg.learning_rate = lr
                for group in runner.alg.optimizer.param_groups:
                    group['lr'] = lr
            return sigma, lr

        sigma, lr = _couple()
        print(f'[lr-sigma] init: sigma {sigma:.3f} -> lr {lr:.2e}', flush=True)
        done = 0
        while done < updates:
            n = min(25, updates - done)
            runner.learn(n)
            done += n
            sigma, lr = _couple()
            print(f'[lr-sigma] update {done}/{updates}: sigma {sigma:.3f} -> lr {lr:.2e}', flush=True)
    else:
        runner.learn(updates)
    elapsed = time.perf_counter() - start
    runner.save(str(out / 'checkpoint.pt'))

    report = {'task': env.cfg, 'device': str(env.device), 'updates': updates, 'algo': algo,
              'transitions': updates * steps_per_iter * num_envs,
              'train_wall_seconds': elapsed,
              'completed_episodes': env.completed_episodes.detach().cpu().tolist(),
              'completed_successes': env.completed_successes.detach().cpu().tolist(),
              'physics_step_calls': getattr(getattr(env, 'plant', None), 'physics_steps', None),
              'note': 'Original default tasks on device-resident tensors; NumPy seeded RNG bitstreams and physics engine numerics are not identical to PyFlyt.'}
    if getattr(env, 'completed_metrics', None):
        eps = max(int(env.completed_episodes.detach().sum()), 1)
        report['completed_metrics'] = {k: float(v.detach().sum()) / eps for k, v in env.completed_metrics.items()}
    (out / 'run.json').write_text(json.dumps(report, indent=2))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    return report
