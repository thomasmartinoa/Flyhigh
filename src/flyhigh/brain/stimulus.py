"""External inputs to an `LIFBrain`."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from flyhigh.brain.lif import LIFBrain, ShiuParams

# Shiu et al. 2024 optogenetic-activation proxy: Poisson kicks of w_syn * 250 mV straight
# into v (each kick fires the neuron), with the target's refractory period switched off.
DEFAULT_KICK_MV = ShiuParams().w_syn_mv * 250


class PoissonActivation:
    def __init__(
        self,
        neuron_idx: Iterable[int],
        rate_hz: float = 150.0,
        weight_mv: float = DEFAULT_KICK_MV,
        agents: Iterable[int] | None = None,
    ):
        self.neuron_idx = torch.as_tensor(list(neuron_idx), dtype=torch.int64)
        self.rate_hz = rate_hz
        self.weight_mv = weight_mv
        self.agents = None if agents is None else torch.as_tensor(list(agents), dtype=torch.int64)

    def attach(self, brain: LIFBrain) -> None:
        """Called once by `LIFBrain.run`: mimic optogenetics by removing the refractory cap."""
        brain.set_refractory(self.neuron_idx, 0.0)

    def ext_v(self, brain: LIFBrain) -> torch.Tensor:
        idx = self.neuron_idx.to(brain.device)
        agents = torch.arange(brain.n_agents, device=brain.device) if self.agents is None else self.agents.to(brain.device)
        p = self.rate_hz * brain.p.dt_ms / 1000.0
        kicks = (torch.rand(agents.numel(), idx.numel(), device=brain.device) < p).float() * self.weight_mv
        v = torch.zeros(brain.n_agents, brain.n_neurons, device=brain.device)
        v[agents.unsqueeze(1), idx.unsqueeze(0)] = kicks
        return v
