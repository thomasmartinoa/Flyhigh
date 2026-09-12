"""Spike recording for `LIFBrain` runs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
import torch

from flyhigh.brain.lif import LIFBrain


class SpikeRecorder:
    """Accumulates per-neuron spike counts for every agent, and exact spike times for
    `neuron_idx` (all neurons if None — fine for analysis, slower for long runs)."""

    def __init__(self, neuron_idx: Iterable[int] | None = None):
        self.neuron_idx = None if neuron_idx is None else torch.as_tensor(list(neuron_idx), dtype=torch.int64)
        self.counts: torch.Tensor | None = None
        self._chunks: list[torch.Tensor] = []  # each (k, 3): t_step, agent, neuron
        self.n_steps = 0

    def record(self, brain: LIFBrain, spiked: torch.Tensor) -> None:
        if self.counts is None:
            self.counts = torch.zeros_like(spiked, dtype=torch.int64, device="cpu")
        self.counts += spiked.to("cpu", torch.int64)
        self.n_steps += 1
        if self.neuron_idx is None:
            agent, neuron = spiked.nonzero(as_tuple=True)
        else:
            idx = self.neuron_idx.to(spiked.device)
            agent, j = spiked[:, idx].nonzero(as_tuple=True)
            neuron = idx[j]
        if agent.numel():
            t = torch.full_like(agent, brain.step_count - 1)
            self._chunks.append(torch.stack([t, agent, neuron], 1).cpu())

    def to_polars(self, dt_ms: float = 0.1) -> pl.DataFrame:
        if not self._chunks:
            return pl.DataFrame({"t_ms": [], "agent": [], "neuron": []},
                                schema={"t_ms": pl.Float64, "agent": pl.Int64, "neuron": pl.Int64})
        a = torch.cat(self._chunks).numpy()
        return pl.DataFrame({"t_ms": a[:, 0] * dt_ms, "agent": a[:, 1], "neuron": a[:, 2]})

    def save(self, path) -> None:
        self.to_polars().write_parquet(path)

    def rates(self, duration_ms: float) -> np.ndarray:
        """Mean firing rate (Hz) per agent and neuron over `duration_ms`."""
        return self.counts.numpy() / (duration_ms / 1000.0)
