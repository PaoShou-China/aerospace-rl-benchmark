"""Offline G0 diagnostic; paths supplied explicitly, never discovers remote hosts."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from frozen.core.phi_rocket_physics import PhiPhysics
from phyai.offline_audit import event_windows, truth_coordinates, replay_rows, distribution


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    windows, events = event_windows(data)
    physics = PhiPhysics().double().eval()
    reference = torch.tensor(json.loads((ROOT/'frozen/candidate/reference_theta.json').read_text())['reference_theta'],dtype=torch.float64)
    truth, clipped = truth_coordinates(physics, reference, data['summary']['true_first_episode_parameters'])
    fixed = reference[None].expand(len(truth), -1)
    modes = {}
    for name, parameters in [('reference',fixed),('truth_mapped_diagnostic',truth)]:
        modes[name] = {phase:replay_rows(physics,data,parameters,rows) for phase,rows in windows.items()}
    pre = [float(data['obs'][r['contact_transition'],r['episode'],:3].norm()) for r in events]
    report = dict(evidence_class='offline one-step replay of retained real transitions',
                  new_real_interactions=0, new_training_updates=0,
                  source=str(args.data), source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  source_policy=data['summary']['options'], cohort_seed=data['summary']['cohort_seed'],
                  episodes=data['obs'].shape[1], contact_episodes=len(events),
                  events=events, pre_contact_measured_angular_rate=distribution(pre),
                  truth_mapping_clipped_episodes=clipped, replay=modes,
                  interpretation='Reference and truth-mapped diagnostics initialize from measured states. No online-belief replay or new candidate performance is inferred. Contact-crossing next states include impulses; pre-contact observation is separately indexed.',
                  model_gate='requires comparison against angular-rate resolution and contact error; not passed by position accuracy alone',
                  wall_seconds=time.perf_counter()-start, torch_version=torch.__version__)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['episodes','contact_episodes','truth_mapping_clipped_episodes','wall_seconds']}))


if __name__=='__main__': main()
