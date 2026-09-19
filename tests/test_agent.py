from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from flyhigh.agent import FlyAgent
from flyhigh.brain.lif import LIFBrain
from flyhigh.motor.readout import Readout
from flyhigh.senses.frame import PanoramicFrame
from tests.test_lif import make_connectome


class FakeSampler:
    def __init__(self, n_eyes=2):
        self.eyes = [None] * n_eyes

    def sample(self, frame):
        return np.full((len(self.eyes), 721), float(frame.lum.mean()), dtype=np.float32)


class FakeEye:
    """'flyvis' with 2 neurons per eye: neuron 0 activity = luminance, neuron 1 = 0.5 always."""
    n_neurons = 2
    types = np.array(["T4a", "Mi1"])
    rest = np.array([0.5, 0.5], dtype=np.float32)

    def reset(self, batch_size):
        self.batch_size = batch_size

    def step(self, lum):
        return torch.stack([lum[:, 0], torch.full_like(lum[:, 0], 0.5)], 1)


class FakeBridge:
    """Bright input (activity above rest, left eye) drives LIF neuron 0 (the 'GF')."""
    driven_indices = np.array([0])

    def __init__(self, n_lif):
        self.n_lif = n_lif

    def ext_i(self, activity):
        n_agents = activity.shape[0] // 2
        out = torch.zeros(n_agents, self.n_lif)
        out[:, 0] = torch.clamp(activity[0::2, 0] - 0.5, min=0) * 100.0
        return out


def agent_with_fakes(n_agents=2):
    c = make_connectome(3, [])
    c.neurons = c.neurons.with_columns(
        pl.Series("type", ["DNp01", "DNp04", "DNa02"]), pl.Series("side", ["L", "L", "R"]),
    )
    brain = LIFBrain(c, n_agents=n_agents, device="cpu")
    return FlyAgent.from_parts(FakeSampler(), FakeEye(), FakeBridge(3), brain, Readout(c))


def test_grey_world_gives_idle_command():
    agent = agent_with_fakes()
    assert agent.eye.batch_size == 4 and agent.steps_per_tick == 100
    cmds = [agent.tick([PanoramicFrame.grey(), PanoramicFrame.grey()]) for _ in range(3)][-1]
    assert all(not c.escape and c.yaw == 0.0 for c in cmds)
    assert agent.brain.step_count == 300 and agent.brain_counts_last_tick.sum() == 0


def test_bright_world_for_agent_0_only_triggers_only_its_escape():
    agent = agent_with_fakes()
    for _ in range(3):
        cmds = agent.tick([PanoramicFrame.grey(1.0), PanoramicFrame.grey(0.5)])
    assert cmds[0].escape is True and cmds[1].escape is False
    assert agent.dn_rates.shape == (2, 3) and agent.dn_rates[0, 0] > 50 and agent.dn_rates[1, 0] == 0
    assert agent.brain_counts_last_tick[0] > 0 and agent.last_activity.shape == (4, 2)


def test_rates_are_a_two_tick_window():
    agent = agent_with_fakes(n_agents=1)
    agent.tick([PanoramicFrame.grey(1.0)])
    busy = agent.dn_rates[0, 0]
    assert busy > 0
    agent.tick([PanoramicFrame.grey(0.5)])  # the drive stops: the window still holds last tick
    assert agent.dn_rates[0, 0] == busy
    agent.tick([PanoramicFrame.grey(0.5)])  # ... and forgets it one tick later
    assert agent.dn_rates[0, 0] == 0


def test_tick_checks_frame_count():
    agent = agent_with_fakes()
    with pytest.raises(ValueError):
        agent.tick([PanoramicFrame.grey()])


@pytest.mark.data
@pytest.mark.flyvis
def test_real_agent_grey_is_silent():
    from flyhigh.data.connectome import Connectome
    if not Path("data/raw").exists() or not Path("data/cache/alignment.parquet").exists():
        pytest.skip("needs real data + alignment table")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    agent = FlyAgent(Connectome.load("data/raw"), n_agents=1)
    cmds = [agent.tick([PanoramicFrame.grey()]) for _ in range(5)][-1]
    assert cmds[0].escape is False and abs(cmds[0].yaw) < 0.05
