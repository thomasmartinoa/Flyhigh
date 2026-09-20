"""flyvis activity → graded drive (mV) for the LIF neurons in the same eye column.

flyvis activities are in arbitrary units around a per-neuron resting level (its steady state on
grey). The bridge subtracts that rest so a grey screen injects nothing, clamps at zero (the LIF
neurons have no hyperpolarising input to model), and scales by a per-type gain calibrated so a
strong stimulus drives T4/T5 to tens of Hz (Task 9). The result goes into `LIFBrain.step(ext_i=)`
for the neurons that `driven_only` cut off from their own synapses.
"""

from __future__ import annotations

import numpy as np
import torch

from flyhigh.senses.alignment import ColumnAlignment


class FlyvisBridge:
    def __init__(self, alignment: ColumnAlignment, fv_types, rest, n_lif: int,
                 gains: dict[str, float], default_gain: float = 20.0):
        self.n_lif = n_lif
        fv_types = np.asarray(fv_types)
        gain_per_fv = np.array([gains.get(t, default_gain) for t in fv_types], dtype=np.float32)
        self.rest = torch.as_tensor(np.asarray(rest, dtype=np.float32))
        self._fv, self._lif, self._gain = {}, {}, {}
        for eye in ("L", "R"):
            fv = alignment.fv_indices(eye)
            self._fv[eye] = torch.as_tensor(fv)
            self._lif[eye] = torch.as_tensor(alignment.lif_indices(eye))
            # a LIF neuron that stands for k flyvis cells (R1-R6) gets their mean: weight = 1/k
            self._gain[eye] = torch.as_tensor(gain_per_fv[fv] * alignment.weights(eye).astype(np.float32))
        self.driven_indices = np.unique(np.concatenate([alignment.lif_indices("L"), alignment.lif_indices("R")]))

    def ext_i(self, activity: torch.Tensor) -> torch.Tensor:
        """activity: (2 * n_agents, n_flyvis), rows ordered [agent0-L, agent0-R, agent1-L, ...].
        Returns (n_agents, n_lif) float32 on the activity's device."""
        dev = activity.device
        n_agents = activity.shape[0] // 2
        delta = torch.clamp(activity.float() - self.rest.to(dev), min=0.0)
        out = torch.zeros(n_agents, self.n_lif, device=dev)
        for k, eye in enumerate(("L", "R")):
            rows = delta[k::2]  # (n_agents, n_fv)
            drive = rows[:, self._fv[eye].to(dev)] * self._gain[eye].to(dev)
            out.index_add_(1, self._lif[eye].to(dev), drive)
        return out
