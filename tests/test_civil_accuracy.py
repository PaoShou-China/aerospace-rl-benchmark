import unittest
from types import SimpleNamespace

import torch

from scripts.civil_accuracy import CivilAccuracy


class CivilAccuracyTest(unittest.TestCase):
    def test_first_episode_terminal_values_ignore_later_episodes(self):
        env = SimpleNamespace(num_envs=2, device="cpu")
        stats = CivilAccuracy("rendezvous", env)
        obs = torch.tensor([[3., 4., 0., 0., 0., 2.], [0., 0., 7., 0., 3., 0.]])
        stats.update(obs, torch.tensor([True, True]), torch.tensor([True, False]))
        stats.update(obs * 2, torch.tensor([False, True]), torch.tensor([False, True]))
        rows = [{}, {}]
        stats.attach(None, rows)
        self.assertEqual(rows[0]["terminal_relative_distance_m"], 5.)
        self.assertEqual(rows[1]["terminal_relative_distance_m"], 14.)
        self.assertEqual([row["policy_steps"] for row in rows], [1., 2.])

    def test_invisible_camera_error_is_null(self):
        env = SimpleNamespace(num_envs=1, device="cpu", ref_offset=torch.zeros(2))
        stats = CivilAccuracy("quadx_track", env)
        stats.update(torch.zeros(1, 4), torch.tensor([True]), torch.tensor([True]))
        rows = [{}]
        stats.attach(None, rows)
        self.assertEqual(rows[0]["camera_visible_fraction"], 0.)
        self.assertIsNone(rows[0]["visible_measured_offset_error_m"])
