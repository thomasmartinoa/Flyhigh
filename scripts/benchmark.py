"""Step-time benchmark for LIFBrain on the real male-CNS connectome.

    uv run python scripts/benchmark.py [--steps 1000]
"""

import argparse
import time

import torch

from flyhigh.brain.lif import LIFBrain
from flyhigh.data.connectome import Connectome


def bench(connectome, n_agents, propagation, steps):
    torch.cuda.reset_peak_memory_stats()
    brain = LIFBrain(connectome, n_agents=n_agents, device="cuda", propagation=propagation)
    # random graded drive so ~2 % of neurons are near threshold: realistic sparse activity
    drive = (torch.rand(n_agents, connectome.n_neurons, device="cuda") < 0.02).float() * 12.0
    for _ in range(50):
        brain.step(ext_i=drive)
    torch.cuda.synchronize()
    t = time.perf_counter()
    spikes = 0
    for _ in range(steps):
        spikes += brain.step(ext_i=drive).sum()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / steps * 1e3
    return ms, spikes.item() / steps / n_agents, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args()
    c = Connectome.load(args.raw)
    graphs = {
        "full CNS": c,
        "brain+optic": c.subgraph(mask=(c.neurons["region"] != "vnc").to_numpy()),
        "brain only": c.subgraph(region="brain"),
    }
    print(f"{'graph':12s} {'N':>7s} {'edges':>10s} {'A':>2s} {'kernel':6s} {'ms/step':>8s} {'x realtime':>10s} {'spk/step':>8s} {'GPU GB':>6s}")
    for name, g in graphs.items():
        for prop in ("event", "spmv"):
            for A in (1, 2):
                ms, spk, gb = bench(g, A, prop, args.steps)
                print(f"{name:12s} {g.n_neurons:7d} {g.n_edges:10d} {A:2d} {prop:6s} {ms:8.2f} {0.1/ms:10.3f} {spk:8.0f} {gb:6.2f}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
