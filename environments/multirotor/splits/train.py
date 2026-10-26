"""Train Split-S PPO on the unchanged QuadX plant. Edit constants; no CLI modes.

Paper-inspired architecture/reward/initialization; optimizer values are local.
Existing run directories are never cleaned automatically.
"""
import sys
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(_ROOT))

from environments.multirotor.splits.env import QuadXSplitsEnv
from environments.multirotor.splits.ppo import run_training

NUM_ENVS=1024
UPDATES=1000
SEED=7
BACKEND='gpu'
RESUME=False
OUT_DIR=_ROOT/'results'/'quadx_splits'
LEARNING_RATE=3e-4
ENTROPY_COEF=.001
INIT_STD=.5
NUM_STEPS_PER_ENV=128
NUM_LEARNING_EPOCHS=5
NUM_MINI_BATCHES=4
ENV_KWARGS=dict(initialization='curriculum',steady_wind=None,dryden=0.,severity='moderate',
                buffer_cfg=dict(capacity=4096,min_states=128,min_episodes=64,
                                success_threshold=.6,replay_fraction=.5))

if __name__=='__main__':
    run_training(QuadXSplitsEnv,num_envs=NUM_ENVS,updates=UPDATES,seed=SEED,
                 backend=BACKEND,resume=RESUME,out_dir=OUT_DIR,env_kwargs=ENV_KWARGS,
                 learning_rate=LEARNING_RATE,entropy_coef=ENTROPY_COEF,init_std=INIT_STD,
                 num_steps_per_env=NUM_STEPS_PER_ENV,num_learning_epochs=NUM_LEARNING_EPOCHS,
                 num_mini_batches=NUM_MINI_BATCHES)
