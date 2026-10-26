"""Opt-in Genesis CPU integration: ROCKET_INTEGRATION=1 python -m unittest discover -s tests -v."""
import os
import unittest
import torch


@unittest.skipUnless(os.environ.get('ROCKET_INTEGRATION')=='1', 'opt-in Genesis simulation')
class RocketIntegrationTests(unittest.TestCase):
    def test_nominal_free_flight_unchanged_and_variation_hidden(self):
        from environments.rocket import RocketLandingEnv
        traces = []
        for version in ('v1','v2'):
            env = RocketLandingEnv(2,'cpu',23000007,benchmark_version=version)
            try:
                trace = [env.get_observations()['policy'].clone()]
                action = torch.zeros(2,7,device=env.device)
                action[:,3:5] = .5
                for _ in range(4):
                    obs,_,_,_,_ = env.step(action)
                    trace.append(obs['policy'].clone())
                traces.append(torch.stack(trace))
            finally:env.close()
        self.assertTrue(torch.equal(*traces), 'nominal initial state/free flight differs')
        env = RocketLandingEnv(2,'cpu',23000007,variation='shifted',dryden=.3)
        try:
            self.assertEqual(tuple(env.get_observations()['policy'].shape),(2,30))
            before = env.plant.rocket_variation.wind.clone()
            env.reset_idx(torch.tensor([True,False],device=env.device))
            self.assertTrue(torch.equal(before[1],env.plant.rocket_variation.wind[1]))
            self.assertIsNotNone(env._wind)
            obs,_,_,_,_ = env.step(torch.zeros(2,7,device=env.device))
            self.assertTrue(torch.isfinite(obs['policy']).all())
            self.assertEqual(env.cfg['benchmark']['external_wind']['dryden'],.3)
        finally:env.close()


if __name__=='__main__':unittest.main()
