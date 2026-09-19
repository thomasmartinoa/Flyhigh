"""The closed sensorimotor loop: frames → eyes → flyvis → LIF brain → descending neurons → command.

One `tick` is one camera frame (10 ms): the sampler reads both eyes, flyvis advances one step,
the bridge turns its activity into currents that are held for 100 LIF steps, and the readout
converts descending-neuron rates over the last two ticks (20 ms: enough spikes to count, short
enough to react) into a MotorCommand per agent. Agents are the batch dimension everywhere.
"""

from __future__ import annotations

import numpy as np
import torch

from flyhigh.brain.lif import LIFBrain
from flyhigh.data.connectome import Connectome
from flyhigh.motor.command import MotorCommand
from flyhigh.motor.readout import Readout, ReadoutParams
from flyhigh.senses.alignment import ColumnAlignment
from flyhigh.senses.bridge import FlyvisBridge
from flyhigh.senses.eye import EyeGeometry, EyeSampler
from flyhigh.senses.flyvis_eye import FlyvisEye
from flyhigh.senses.frame import TICK_MS, PanoramicFrame

DEFAULT_GAINS: dict[str, float] = {}  # per flyvis type; filled by scripts/calibrate_bridge.py (Task 9)
DEFAULT_GAIN = 20.0


class FlyAgent:
    steps_per_tick = round(TICK_MS / 0.1)
    window_ticks = 2  # descending-neuron rates over the last 20 ms

    def __init__(self, connectome: Connectome, n_agents: int = 1, gains: dict[str, float] | None = None,
                 params: ReadoutParams | None = None, alignment_path="data/cache/alignment.parquet",
                 frame_shape=(180, 360), device="cuda"):
        sampler = EyeSampler(frame_shape, [EyeGeometry("L"), EyeGeometry("R")])
        eye = FlyvisEye()
        eye.reset(batch_size=2 * n_agents)
        alignment = ColumnAlignment.load(alignment_path)
        bridge = FlyvisBridge(alignment, eye.types, eye.rest, connectome.n_neurons,
                              gains if gains is not None else DEFAULT_GAINS, DEFAULT_GAIN)
        brain = LIFBrain.for_male_cns(connectome, n_agents=n_agents, device=device,
                                      driven_only=bridge.driven_indices)
        self._init(sampler, eye, bridge, brain, Readout(connectome, params))

    @classmethod
    def from_parts(cls, sampler, eye, bridge, brain, readout) -> FlyAgent:
        """Assemble from ready-made (or fake) parts; resets the eye for the brain's batch."""
        self = cls.__new__(cls)
        eye.reset(batch_size=2 * brain.n_agents)
        self._init(sampler, eye, bridge, brain, readout)
        return self

    def _init(self, sampler, eye, bridge, brain, readout):
        self.sampler, self.eye, self.bridge, self.brain, self.readout = sampler, eye, bridge, brain, readout
        self.n_agents = brain.n_agents
        self._watch = torch.as_tensor(readout.watch, device=brain.device)
        self._counts = [np.zeros((self.n_agents, len(readout.watch))) for _ in range(self.window_ticks)]
        self.dn_rates = np.zeros((self.n_agents, brain.n_neurons))  # Hz; only `readout.watch` is filled
        self.brain_counts_last_tick = np.zeros(brain.n_neurons)  # all agents summed; calibration/plots
        self.last_activity: torch.Tensor | None = None  # flyvis, (2 * n_agents, n_flyvis)

    def tick(self, frames: list[PanoramicFrame]) -> list[MotorCommand]:
        if len(frames) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} frames, got {len(frames)}")
        lum = np.concatenate([self.sampler.sample(f) for f in frames])  # (n_agents*2, 721): L, R per agent
        self.last_activity = self.eye.step(torch.as_tensor(lum))
        ext_i = self.bridge.ext_i(self.last_activity).to(self.brain.device)
        counts = torch.zeros(self.n_agents, len(self.readout.watch), device=self.brain.device)
        total = torch.zeros(self.brain.n_neurons, device=self.brain.device)
        for _ in range(self.steps_per_tick):
            spiked = self.brain.step(ext_i=ext_i)
            counts += spiked[:, self._watch]
            total += spiked.sum(0)
        self._counts = self._counts[1:] + [counts.cpu().numpy()]
        self.brain_counts_last_tick = total.cpu().numpy()
        window_s = self.window_ticks * TICK_MS / 1000.0
        self.dn_rates[:] = 0.0
        self.dn_rates[:, self.readout.watch] = sum(self._counts) / window_s
        return self.readout.commands(self.dn_rates)
