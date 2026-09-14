"""The pretrained flyvis optic lobe, run one 10 ms frame at a time with persistent state."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from flyhigh.senses.flyvis_compat import import_flyvis, preserve_torch_default_device


class FlyvisEye:
    def __init__(self, model: str = "flow/0000/000", dt: float = 0.01, device=None):
        flyvis = import_flyvis()

        self.dt = dt
        self.device = torch.device(device or flyvis.device)
        # flyvis's internals (parameter init, and every forward/steady_state call below)
        # assume torch's default device is the network's device -- that was true for the
        # whole process until import_flyvis() undid the side effect of `import flyvis`.
        # Restore it just for the duration of each flyvis call, via self._flyvis_device().
        with self._flyvis_device():
            view = flyvis.NetworkView(flyvis.results_dir / model)
            self.net = view.init_network().to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        nodes = self.net.connectome.nodes
        self.types = np.array([t.decode() for t in nodes.type[:]])
        self.u = np.asarray(nodes.u[:], dtype=np.int64)
        self.v = np.asarray(nodes.v[:], dtype=np.int64)
        self.n_neurons = len(self.types)
        self._state = None
        self.rest = None

    @contextmanager
    def _flyvis_device(self):
        """Make torch's default device the network's device, only for the wrapped call."""
        with preserve_torch_default_device():
            torch.set_default_device(self.device)
            yield

    # A few flyvis cell types have slow time constants: after 1 s of grey the activity still
    # drifts by ~1e-2; after 2 s the residual drift is < 1e-4 (and it costs 0.1 s).
    T_GREY_S = 2.0

    def reset(self, batch_size: int) -> None:
        """2 s of grey (0.5) → steady state; remember it as the per-neuron resting activity,
        so that later a grey screen injects nothing into the LIF brain."""
        with torch.no_grad(), self._flyvis_device():
            self._state = self.net.steady_state(t_pre=self.T_GREY_S, dt=self.dt, batch_size=batch_size)
        self.rest = self._state.nodes.activity[0].detach().cpu().numpy()
        self.batch_size = batch_size

    def step(self, lum: torch.Tensor) -> torch.Tensor:
        """lum: (batch, 721) in [0, 1]. Returns activity (batch, n_neurons) after one dt."""
        assert self._state is not None, "call reset(batch_size) first"
        x = lum.to(self.device).float()[:, None, None, :]  # (batch, frames=1, 1, hexals)
        with torch.no_grad(), self._flyvis_device():
            self.net.stimulus.zero(x.shape[0], 1)
            self.net.stimulus.buffer = self.net.stimulus.buffer.to(self.device)
            self.net.stimulus.add_input(x)
            self._state = self.net.forward(
                self.net.stimulus(), self.dt, state=self._state, as_states=True
            )[-1]
        return self._state.nodes.activity
