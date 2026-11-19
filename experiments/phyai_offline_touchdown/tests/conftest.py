from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)


@pytest.fixture
def state():
    x = torch.zeros(21, dtype=torch.float64)
    x[2], x[5], x[9], x[17], x[20] = 4.0, -0.4, 1.0, 0.45, 0.25
    return x


@pytest.fixture
def snapshot(state):
    from phyai.belief_contract import BeliefSnapshot
    return BeliefSnapshot(state, torch.zeros(67, dtype=state.dtype),
                          torch.eye(87, dtype=state.dtype) * 0.0025,
                          version=7, clock=3.0, modes=("flight",),
                          model_id="frozen-full-physics", history_id="episode-1:120")
