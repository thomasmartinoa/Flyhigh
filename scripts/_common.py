"""Shared helpers for the validation scripts."""

import polars as pl
import torch

from flyhigh.brain.lif import LIFBrain
from flyhigh.brain.recorder import SpikeRecorder
from flyhigh.brain.stimulus import PoissonActivation
from flyhigh.data.connectome import Connectome

MAX_RATE_HZ = 1000.0 / 2.3  # refractory-limited ceiling (t_rfc + dt)


def load(raw="data/raw") -> Connectome:
    return Connectome.load(raw)


def experiment(connectome, activations, silence=None, duration_ms=1000.0, seed=0, brain_kw=None, record=()):
    """Run one trial: `activations` = [(neuron_idx, rate_hz), ...]. Returns rates (N,) and recorder."""
    torch.manual_seed(seed)
    brain = LIFBrain.for_male_cns(connectome, device="cuda", **(brain_kw or {}))
    if silence is not None and len(silence):
        brain.silence(silence)
    stimuli = [PoissonActivation(idx, hz) for idx, hz in activations]
    rec = brain.run(int(duration_ms / brain.p.dt_ms), stimuli=stimuli, recorder=SpikeRecorder(neuron_idx=list(record)))
    return rec.rates(duration_ms)[0], rec


def summary(connectome, rates, exclude_types=()):
    df = connectome.neurons.with_columns(pl.Series("rate", rates))
    active = int((rates > 0).sum())
    resp = df.filter(pl.col("rate") > 5)
    top = (
        resp.filter(~pl.col("type").str.contains("|".join(f"^{t}" for t in exclude_types) or "^$").fill_null(False))
        .group_by("type").agg(pl.len().alias("n"), pl.col("rate").mean().round(0).alias("hz"))
        .sort("hz", descending=True).head(6)
    )
    return active, resp.height, [(t, n, int(h)) for t, n, h in top.iter_rows()]
