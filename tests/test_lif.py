import math

import numpy as np
import polars as pl
import pytest
import torch

from flyhigh.brain.lif import LIFBrain, ShiuParams
from flyhigh.data.connectome import Connectome


def make_connectome(n: int, edges: list[tuple[int, int, int]]) -> Connectome:
    """Tiny connectome: n excitatory neurons plus (pre, post, signed_weight) edges."""
    neurons = pl.DataFrame({
        "body_id": list(range(100, 100 + n)), "index": list(range(n)),
        "type": [f"N{i}" for i in range(n)], "instance": [None] * n, "class": [None] * n,
        "superclass": ["cb_intrinsic"] * n, "region": ["brain"] * n, "side": ["L"] * n,
        "dimorphism": [None] * n, "hex1": [None] * n, "hex2": [None] * n,
        "nt": ["acetylcholine"] * n, "nt_sign": [1] * n, "nt_confidence": [1.0] * n,
    }, schema_overrides={"index": pl.Int64, "nt_sign": pl.Int8, "hex1": pl.Float64, "hex2": pl.Float64})
    e = pl.DataFrame({
        "pre_idx": [p for p, _, _ in edges], "post_idx": [q for _, q, _ in edges],
        "syn_count": [abs(w) for _, _, w in edges], "weight": [w for _, _, w in edges],
    }, schema={"pre_idx": pl.Int64, "post_idx": pl.Int64, "syn_count": pl.Int32, "weight": pl.Int32})
    return Connectome(neurons, e)


def spike_count(brain, n_steps, ext_i=None, ext_v=None):
    total = torch.zeros(brain.n_agents, brain.n_neurons, dtype=torch.int64)
    for _ in range(n_steps):
        total += brain.step(ext_i=ext_i, ext_v=ext_v).cpu()
    return total


P = ShiuParams()


def test_constant_drive_gives_analytic_firing_rate():
    brain = LIFBrain(make_connectome(1, []), device="cpu")
    drive = 20.0  # mV above rest at steady state
    t_to_th = -P.t_mbr_ms * math.log(1 - (P.v_th_mv - P.v_rest_mv) / drive)
    expected_hz = 1000.0 / (t_to_th + P.t_rfc_ms)
    n = spike_count(brain, 10_000, ext_i=torch.full((1, 1), drive)).item()  # 1 s
    assert abs(n - expected_hz) / expected_hz < 0.02


def test_subthreshold_drive_never_spikes():
    brain = LIFBrain(make_connectome(1, []), device="cpu")
    n = spike_count(brain, 5_000, ext_i=torch.full((1, 1), 5.0)).item()  # 7 mV needed
    assert n == 0


def test_refractory_period_caps_rate():
    brain = LIFBrain(make_connectome(1, []), device="cpu")
    n = spike_count(brain, 1_000, ext_i=torch.full((1, 1), 1e4)).item()  # 100 ms, huge drive
    # Brian2 semantics: refractory while (t - t_spike) <= t_rfc, so ISI = t_rfc + dt exactly.
    assert n == 1 + (1_000 - 1) // (brain.rfc_steps + 1)


def test_one_spike_raises_postsynaptic_g_by_w_syn_times_count():
    brain = LIFBrain(make_connectome(2, [(0, 1, 3)]), device="cpu")
    kick = torch.zeros(1, 2); kick[0, 0] = 100.0  # force neuron 0 to spike now
    spiked = brain.step(ext_v=kick)
    assert spiked[0, 0] and not spiked[0, 1]
    for _ in range(brain.delay_steps - 1):
        brain.step()
    assert brain.g[0, 1].item() == 0.0  # not arrived yet
    brain.step()
    assert brain.g[0, 1].item() == pytest.approx(3 * P.w_syn_mv, rel=1e-5)


def test_inhibitory_edge_lowers_g():
    brain = LIFBrain(make_connectome(2, [(0, 1, -2)]), device="cpu")
    kick = torch.zeros(1, 2); kick[0, 0] = 100.0
    brain.step(ext_v=kick)
    for _ in range(brain.delay_steps):
        brain.step()
    assert brain.g[0, 1].item() == pytest.approx(-2 * P.w_syn_mv, rel=1e-5)


def test_agents_are_independent():
    brain = LIFBrain(make_connectome(2, [(0, 1, 50)]), n_agents=2, device="cpu")
    drive = torch.zeros(2, 2); drive[0, 0] = 30.0  # drive neuron 0 in agent 0 only
    n = spike_count(brain, 5_000, ext_i=drive)
    assert n[0, 0] > 0 and n[0, 1] > 0  # agent 0: neuron 0 fires and drives neuron 1
    assert n[1].sum() == 0  # agent 1 silent


def test_no_input_no_activity():
    brain = LIFBrain(make_connectome(3, [(0, 1, 5), (1, 2, 5), (2, 0, 5)]), device="cpu")
    assert spike_count(brain, 1_000).sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    edges = [(0, 1, 4), (1, 2, -3), (2, 0, 6), (0, 2, 2)]
    drive = torch.zeros(1, 3); drive[0, 0] = 15.0
    cpu = spike_count(LIFBrain(make_connectome(3, edges), device="cpu"), 3_000, ext_i=drive)
    gpu = spike_count(LIFBrain(make_connectome(3, edges), device="cuda"), 3_000, ext_i=drive.cuda())
    assert torch.equal(cpu, gpu)


def random_connectome(n=200, n_edges=3000, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, n_edges); post = rng.integers(0, n, n_edges)
    keep = pre != post
    pairs = {(int(p), int(q)) for p, q in zip(pre[keep], post[keep])}
    edges = [(p, q, int(rng.integers(1, 30)) * int(rng.choice([-1, 1]))) for p, q in sorted(pairs)]
    return make_connectome(n, edges)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_event_driven_propagation_matches_spmv(device):
    c = random_connectome()
    torch.manual_seed(0)
    drive = (torch.rand(3, c.n_neurons) < 0.1).float() * 25.0
    drive = drive.to(device)
    spmv = LIFBrain(c, n_agents=3, device=device, propagation="spmv")
    event = LIFBrain(c, n_agents=3, device=device, propagation="event")
    for _ in range(2_000):
        s1 = spmv.step(ext_i=drive); s2 = event.step(ext_i=drive)
        assert torch.equal(s1, s2)
        assert torch.allclose(spmv.g, event.g, atol=1e-4)
    assert spmv.v.abs().sum() > 0 and (s1.sum() > 0 or spmv.g.abs().sum() > 0)


def test_spike_resets_synaptic_input_g_to_zero():
    """Shiu et al. reset rule: 'v = v_rst; g = 0' — a spike discards accumulated input."""
    brain = LIFBrain(make_connectome(2, [(0, 1, 300)]), device="cpu")
    kick = torch.zeros(1, 2); kick[0, 0] = 100.0
    brain.step(ext_v=kick)
    for _ in range(brain.delay_steps):
        brain.step()
    assert brain.g[0, 1].item() > 0  # 300 synapses arrived: g = 82.5 mV (peak v ≈ 13 mV)
    for _ in range(100):  # neuron 1 charges up and spikes within a few ms
        spiked = brain.step()
        if spiked[0, 1]:
            break
    assert spiked[0, 1]
    assert brain.g[0, 1].item() == 0.0


def test_silence_only_removes_outgoing_synapses():
    """Shiu's silence(): the neuron still spikes, but nobody hears it."""
    from flyhigh.brain.recorder import SpikeRecorder
    c = make_connectome(3, [(0, 1, 200), (1, 2, 200)])
    drive = torch.zeros(1, 3); drive[0, 0] = 30.0
    cut = LIFBrain(c, device="cpu"); cut.silence([1])
    rec = cut.run(3_000, ext_i=drive, recorder=SpikeRecorder())
    assert rec.counts[0, 0] > 0 and rec.counts[0, 1] > 0 and rec.counts[0, 2] == 0
