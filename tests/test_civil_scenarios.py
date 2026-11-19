import unittest

import torch

from environments import make
from environments.civil_scenarios import CIVIL_TASKS, scenario_kwargs


class CivilScenariosTest(unittest.TestCase):
    def test_profiles_are_opt_in_and_repeatable(self):
        for task in CIVIL_TASKS:
            self.assertEqual(scenario_kwargs(task, "reference", 7), {})
            self.assertNotEqual(scenario_kwargs(task, "hard", 7), scenario_kwargs(task, "development", 7))
        original = scenario_kwargs("quadx", "hard", 7)
        self.assertNotEqual(original["steady_wind"], scenario_kwargs("quadx", "hard", 8)["steady_wind"])
        original["domain_randomize"].clear()
        self.assertIn("mass", scenario_kwargs("quadx", "hard", 7)["domain_randomize"])
        with self.assertRaises(ValueError):
            scenario_kwargs("unknown", "hard", 7)

    def test_eclipse_scales_thrust_and_fuel_only_in_shadow(self):
        env = make("transfer", 3, civil_profile="hard")
        try:
            # Two points on the Sun axis, and one behind Earth but outside its shadow.
            sun = env.sun_direction
            cross = torch.tensor([0., 0., 1.], device=env.device)
            r = torch.stack((sun, -sun, -sun + cross)) * 1e7
            m = torch.full((3, 1), 100., device=env.device)
            state = torch.cat((r, torch.zeros_like(r), m), dim=-1)
            action = torch.ones_like(r) * 0.5
            eclipse = env._deriv(state, action)
            env.eclipse_thrust_fraction = 1.
            full = env._deriv(state, action)
            grav = env._deriv(state, torch.zeros_like(action))
            scale = torch.tensor([1., .1, 1.], device=env.device)
            torch.testing.assert_close(eclipse[:, 6], scale * full[:, 6], atol=1e-13, rtol=1e-5)
            torch.testing.assert_close(eclipse[:, 5]-grav[:, 5], scale*(full[:, 5]-grav[:, 5]), atol=5e-7, rtol=.01)
        finally:
            env.close()

    def test_navigation_hold_uses_old_packet_and_resets_only_finished_lanes(self):
        env = make("rendezvous", 8, seed=3, civil_profile="navigation_outage")
        try:
            old = env.get_observations()["policy"].clone()
            obs, _, term, trunc, extras = env.step(torch.zeros(8, 3))
            stale = env.stale
            self.assertTrue(stale.any() and not stale.all())
            self.assertFalse((term | trunc).any())
            torch.testing.assert_close(obs["policy"][stale], old[stale])
            torch.testing.assert_close(obs["policy"][~stale], extras["civil_truth_terminal"][~stale])
            env.env.max_steps = 2
            obs, _, term, trunc, extras = env.step(torch.zeros(8, 3))
            self.assertTrue((term | trunc).all())
            self.assertFalse(env.stale.any())
            torch.testing.assert_close(obs["policy"], env.env.get_observations()["policy"])
            self.assertFalse(torch.equal(extras["civil_truth_terminal"], obs["policy"]))
        finally:
            env.close()

    def test_transfer_integrates_sub_ulp_propellant_and_resets_independent_lanes(self):
        env = make("transfer", 3, seed=3, domain_randomize=False, eclipse_thrust_fraction=.1)
        try:
            env.r.copy_(torch.stack((env.sun_direction, -env.sun_direction, env.sun_direction)) * 1.2e7)
            env.v.zero_()
            action = torch.tensor([[.2, 0., 0.]]).expand(3, -1)
            env._integrate(action, torch.tensor([True, True, False]))
            burn = env.f_max * .2 * env.dt / (env.isp * env.g0)
            expected = torch.tensor([burn, burn * .1, 0.])
            torch.testing.assert_close(env.propellant_used, expected, atol=1e-11, rtol=2e-6)
            torch.testing.assert_close(env.m, env.mass0 - expected, atol=0., rtol=0.)
            env.reset_idx(torch.tensor([True, False, False]))
            self.assertEqual(float(env.propellant_used[0]), 0.)
            torch.testing.assert_close(env.propellant_used[1], expected[1], atol=1e-11, rtol=2e-6)
        finally:
            env.close()

    def test_invalid_profiles_and_outage_tasks_fail_before_construction(self):
        with self.assertRaises(ValueError):
            scenario_kwargs("transfer", "typo", 1)
        with self.assertRaises(ValueError):
            make("satellite", 1, civil_profile="navigation_outage")
        for value in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                make("transfer", 1, eclipse_thrust_fraction=value)
        for value in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make("quadx_track", 1, camera_sigma_m=value)

    def test_reference_matches_unwrapped_initial_state(self):
        plain = make("rendezvous", 3, seed=11)
        reference = make("rendezvous", 3, seed=11, civil_profile="reference")
        try:
            torch.testing.assert_close(plain.get_observations()["policy"],
                                       reference.get_observations()["policy"])
        finally:
            plain.close()
            reference.close()

    def test_manual_partial_reset_keeps_other_navigation_packets(self):
        env = make("rendezvous", 3, seed=3, civil_profile="navigation_outage")
        try:
            env.step(torch.zeros(3, 3))
            old = env.get_observations()["policy"].clone()
            ages = env.age.clone()
            mask = torch.tensor([True, False, False])
            env.reset_idx(mask)
            torch.testing.assert_close(env.get_observations()["policy"][~mask], old[~mask])
            torch.testing.assert_close(env.age[~mask], ages[~mask])
            torch.testing.assert_close(env.get_observations()["policy"][mask],
                                       env.env.get_observations()["policy"][mask])
            self.assertEqual(env.age[0].item(), 0)
            self.assertFalse(env.stale[0].item())
        finally:
            env.close()

    def test_transfer_outage_retains_onboard_time_and_fuel(self):
        env = make("transfer", 3, seed=0, civil_profile="navigation_outage")
        try:
            old = env.get_observations()["policy"].clone()
            obs, _, term, trunc, extras = env.step(torch.ones(3, 3) * .2)
            self.assertTrue(env.stale.any())
            self.assertFalse((term | trunc).any())
            truth = extras["civil_truth_terminal"]
            torch.testing.assert_close(obs["policy"][:, [0, 7]], truth[:, [0, 7]])
            torch.testing.assert_close(obs["policy"][env.stale, 1:7], old[env.stale, 1:7])
        finally:
            env.close()
