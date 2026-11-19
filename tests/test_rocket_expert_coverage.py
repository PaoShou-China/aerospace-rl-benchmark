"""Full-scale BC must include each terminal action and exclude post-reset samples."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from algorithms.warmstart.bc_warmstart import fit_behavior_cloning, bc_recipe_fingerprint
from algorithms.warmstart.expert_dataset import collect_landing_first_episodes
from scripts.landing_score import benchmark_method_dir


class Expert:
    def __init__(self, env, design):
        self.name = "test_expert"

    def actions(self, obs):
        return torch.zeros(len(obs['policy']), 1)

    def reset_idx(self, mask):
        pass


class Episodes:
    conventional_controller = Expert
    controller_design = {}
    cfg = {'vehicle': 'rocket'}
    device = 'cpu'
    num_envs = 2
    num_actions = 1
    agent_hz = 40
    max_steps = 4

    def __init__(self):
        self.age = torch.zeros(2, dtype=torch.long)
        self.completed_successes = torch.zeros_like(self.age)

    def get_observations(self):
        return {'policy': torch.stack((self.age.float(), (self.age >= 2).float()), dim=1)}

    def step(self, unused_actions):
        self.age += 1
        term = torch.tensor([self.age[0] >= 2, False])
        trunc = torch.tensor([False, self.age[1] >= 4])
        self.completed_successes += trunc
        before_reset = self.get_observations()
        self.age[term | trunc] = 0
        return self.get_observations(), torch.ones(2), term, trunc, {'terminal_observations': before_reset}


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.seen = []

    def update_normalization(self, obs):
        self.seen.extend(obs['policy'].flatten().tolist())

    def forward(self, obs):
        return self.linear(obs['policy'])


class ExpertCoverageTests(unittest.TestCase):
    def test_keep_first_terminal_actions_when_neighbor_resets(self):
        data, stats = collect_landing_first_episodes(Episodes(), seed=11)
        self.assertEqual(data['pair_mask'].T.tolist(), [[True, True, False, False], [True]*4])
        self.assertEqual(stats['complete_episodes'], 2)
        self.assertEqual(stats['expert_pairs'], 6)
        self.assertEqual(stats['observed_pad_contact_pairs'], 2)
        self.assertEqual(stats['pairs_in_2s_before_first_observed_contact'], 2)
        self.assertEqual(stats['terminal_stable_episodes'], 1)
        env = Episodes()
        env.max_steps = 3
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            collect_landing_first_episodes(env, seed=11)

    def test_actor_and_critic_exclude_resets_and_respect_timeouts(self):
        actor, critic = RecordingModel(), RecordingModel()
        runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic))
        obs = torch.arange(6.).reshape(3, 2, 1)
        done = torch.tensor([[False, False], [True, False], [False, True]])
        mask = torch.tensor([[True, True], [True, True], [False, True]])
        term = done.clone()
        term[-1, 1] = False
        report = fit_behavior_cloning(runner, obs, torch.zeros_like(obs), torch.ones(3, 2), done,
                                      pair_mask=mask, terminated=term, truncated=done & ~term,
                                      fit_critic=True, epochs=1, batch_size=8, gamma=0.99,
                                      lr=1e-3, log=lambda _: None)
        self.assertCountEqual(actor.seen, [0, 1, 2, 3, 5])
        self.assertCountEqual(critic.seen, [0, 2])
        self.assertEqual(report['critic_valid_pairs'], 2)
        self.assertEqual(report['critic_excluded_pairs'], 3)

    def test_precomputed_returns_remain_compatible_with_pair_mask(self):
        actor, critic = RecordingModel(), RecordingModel()
        runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic))
        obs = torch.arange(6.).reshape(3, 2, 1)
        mask = torch.tensor([[True, True], [True, True], [False, True]])
        valid = torch.tensor([[True, False], [True, False], [True, False]])
        report = fit_behavior_cloning(
            runner, obs, torch.zeros_like(obs), torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.bool),
            pair_mask=mask, returns=torch.ones(3, 2), returns_valid=valid,
            epochs=1, batch_size=8, gamma=.99, lr=1e-3, log=lambda _: None)
        self.assertCountEqual(actor.seen, [0, 1, 2, 3, 5])
        self.assertCountEqual(critic.seen, [0, 2])
        self.assertEqual(report['critic_valid_pairs'], 2)
        self.assertEqual(report['critic_excluded_pairs'], 3)
        self.assertIn('precomputed', report['critic_target'])

    def test_cache_fingerprint_distinguishes_collection_protocol(self):
        a = dict(steps=None, select=None, collection='first_complete_episode')
        b = dict(a, collection='fixed_steps')
        self.assertNotEqual(bc_recipe_fingerprint('rocket', a), bc_recipe_fingerprint('rocket', b))

    def test_new_bc_runs_never_reuse_old_seed_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            rocket = Path(temp) / 'rocket'
            legacy = rocket / 'bc_ppo'
            (legacy / 'seed_11').mkdir(parents=True)
            revised = benchmark_method_dir(rocket, 'bc_ppo')
            self.assertNotEqual(legacy, revised)
            self.assertEqual(benchmark_method_dir(rocket, 'bc_ppo', read_existing=True), legacy)
            revised.mkdir()
            self.assertEqual(benchmark_method_dir(rocket, 'bc_ppo', read_existing=True), revised)
            self.assertEqual(benchmark_method_dir(rocket, 'ppo_scratch'), rocket / 'ppo_scratch')
