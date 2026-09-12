import pytest
import torch

from flyhigh.brain.lif import LIFBrain
from flyhigh.brain.recorder import SpikeRecorder
from flyhigh.brain.stimulus import PoissonActivation
from tests.test_lif import make_connectome


def test_poisson_activation_hits_target_rate():
    k = 400
    brain = LIFBrain(make_connectome(k, []), device="cpu")
    stim = PoissonActivation(range(k), rate_hz=150.0)
    torch.manual_seed(1)
    kicks = 0
    for _ in range(10_000):  # 1 s
        kicks += (stim.ext_v(brain) > 0).sum().item()
    assert abs(kicks / k - 150.0) < 150.0 * 0.05


def test_poisson_activation_only_touches_its_targets():
    brain = LIFBrain(make_connectome(5, []), n_agents=2, device="cpu")
    stim = PoissonActivation([1, 3], rate_hz=5000.0)
    v = stim.ext_v(brain)
    assert v.shape == (2, 5)
    assert (v[:, [0, 2, 4]] == 0).all()


def test_run_with_activation_makes_targets_fire_every_kick():
    brain = LIFBrain(make_connectome(1, []), device="cpu")
    torch.manual_seed(2)
    rec = brain.run(10_000, stimuli=[PoissonActivation([0], rate_hz=150.0)], recorder=SpikeRecorder())
    n = rec.counts[0, 0].item()
    assert 120 < n < 180  # refractory is switched off for activated neurons, so each kick spikes


def test_recorder_round_trip_and_rates():
    brain = LIFBrain(make_connectome(2, [(0, 1, 60)]), n_agents=2, device="cpu")
    drive = torch.zeros(2, 2); drive[1, 0] = 30.0
    rec = SpikeRecorder(neuron_idx=[1])
    for _ in range(5_000):
        rec.record(brain, brain.step(ext_i=drive))
    df = rec.to_polars()
    assert df.columns == ["t_ms", "agent", "neuron"]
    assert df["agent"].unique().to_list() == [1] and df["neuron"].unique().to_list() == [1]
    assert df.height == rec.counts[1, 1].item() > 0
    assert rec.counts[0].sum() == 0
    rates = rec.rates(duration_ms=500.0)
    assert rates[1, 1] == pytest.approx(rec.counts[1, 1].item() / 0.5)


def test_silence_removes_a_neuron_from_the_circuit():
    c = make_connectome(3, [(0, 1, 200), (1, 2, 200)])  # 200 syn -> ~8.6 mV peak, above the 7 mV gap
    drive = torch.zeros(1, 3); drive[0, 0] = 30.0
    intact = LIFBrain(c, device="cpu").run(3_000, ext_i=drive, recorder=SpikeRecorder())
    assert intact.counts[0, 2] > 0
    cut = LIFBrain(c, device="cpu")
    cut.silence([1])
    rec = cut.run(3_000, ext_i=drive, recorder=SpikeRecorder())
    assert rec.counts[0, 0] > 0 and rec.counts[0, 2] == 0  # neuron 1 may still spike (outgoing-only)
