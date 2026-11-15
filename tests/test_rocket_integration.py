"""Opt-in Genesis CPU integration: ROCKET_INTEGRATION=1 python -m unittest discover -s tests -v."""

import os
import unittest
import torch


@unittest.skipUnless(os.environ.get("ROCKET_INTEGRATION") == "1", "opt-in Genesis simulation")
class RocketIntegrationTests(unittest.TestCase):
    def test_free_flight_finite_and_variation_hidden(self):
        from environments.rocket import RocketLandingEnv

        env = RocketLandingEnv(2, "cpu", 23000007)
        try:
            action = torch.zeros(2, 7, device=env.device)
            action[:, 3:5] = 0.5
            for _ in range(4):
                obs, _, _, _, _ = env.step(action)
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            self.assertEqual(env.cfg["benchmark"]["version"], "v2")
        finally:
            env.close()
        env = RocketLandingEnv(2, "cpu", 23000007, variation="shifted", dryden=0.3)
        try:
            self.assertEqual(tuple(env.get_observations()["policy"].shape), (2, 30))
            before = env.plant.rocket_variation.wind.clone()
            env.reset_idx(torch.tensor([True, False], device=env.device))
            self.assertTrue(torch.equal(before[1], env.plant.rocket_variation.wind[1]))
            self.assertIsNotNone(env._wind)
            obs, _, _, _, _ = env.step(torch.zeros(2, 7, device=env.device))
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            self.assertEqual(env.cfg["benchmark"]["external_wind"]["dryden"], 0.3)
        finally:
            env.close()

    def test_f9_real_dual_tank_reset_and_masked_step(self):
        from environments.rocket import RocketLandingEnv

        env = RocketLandingEnv(2, "cpu", 23000007, benchmark_version="v1")
        try:
            self.assertEqual(tuple(env.get_observations()["policy"].shape), (2, 30))
            masses = env.plant.drone.get_links_mass().sum(1)
            propellant = env.plant.fuel * env.plant.params["propellant_params"]["total_fuel"]
            # Default domain randomization does not touch dry mass, so the
            # total must be exactly nominal dry mass + residual propellant.
            torch.testing.assert_close(
                masses.to(torch.float64), 25600.0 + propellant, rtol=1e-3, atol=0.2
            )
            fuel_before = env.plant.fuel.clone()
            env.reset_idx(torch.tensor([True, False], device=env.device))
            self.assertEqual(fuel_before[1], env.plant.fuel[1])
            fuel_after_reset = env.plant.fuel.clone()
            action = torch.zeros(2, 7, device=env.device)
            action[:, 3:5] = 1.0
            obs, _, _, _, _ = env.step(action)
            self.assertTrue(torch.isfinite(obs["policy"]).all())
            self.assertTrue((env.plant.fuel < fuel_after_reset).all())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
