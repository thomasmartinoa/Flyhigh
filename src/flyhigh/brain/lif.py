"""Batched leaky integrate-and-fire brain on the male-CNS connectome.

Neuron model (Shiu et al. 2024, Nature; Brian2 "linear" method), per neuron:

    dv/dt = (v_rest - v + g) / t_mbr     (frozen while refractory)
    dg/dt = -g / tau                      (frozen while refractory)
    spike when v > v_th   ->  v = v_reset, g = 0 (input discarded), refractory for t_rfc
    presynaptic spike     ->  g[post] += w_syn * signed_synapse_count   (after delay t_dly)

Optional short-term synaptic depression (Tsodyks & Markram): each presynaptic neuron has a
resource x in [0, 1] that scales its outgoing weights, drops by a fraction `std_u` on every
spike and recovers with time constant `std_tau_ms`. Off by default (std_u = 0), which is the
exact Shiu model. We use it to stabilise the denser male-CNS graph — see docs/02-lif-brain.md.

The batch dimension is the *agent*: every agent gets its own membrane state but the
same wiring, so one sparse matmul propagates spikes for all agents at once.
Voltages are in mV and time in ms throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from flyhigh.data.connectome import Connectome


@dataclass(frozen=True)
class ShiuParams:
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_th_mv: float = -45.0
    t_mbr_ms: float = 20.0  # membrane time constant
    tau_ms: float = 5.0  # synaptic time constant
    t_rfc_ms: float = 2.2  # refractory period
    t_dly_ms: float = 1.8  # synaptic delay
    w_syn_mv: float = 0.275  # jump in g per synapse
    dt_ms: float = 0.1


# Our calibration for the male CNS (docs/02-lif-brain.md): the graph carries ~1.9x FlyWire's
# synapse density, so Shiu's w_syn ignites it; a lower gain plus short-term depression on
# central (non-sensory) synapses keeps activity local while sensory drive stays high-fidelity.
MALE_CNS_W_SYN = 0.20
MALE_CNS_STD_U = 0.2
MALE_CNS_STD_TAU_MS = 300.0


class LIFBrain:
    @classmethod
    def for_male_cns(cls, connectome: Connectome, n_agents: int = 1, device="cuda", **kw) -> "LIFBrain":
        sensory = connectome.neurons.filter(
            connectome.neurons["superclass"].str.contains("sensory").fill_null(False)
        )["index"].to_numpy().copy()
        return cls(
            connectome, n_agents=n_agents, device=device,
            params=ShiuParams(w_syn_mv=MALE_CNS_W_SYN),
            std_u=MALE_CNS_STD_U, std_tau_ms=MALE_CNS_STD_TAU_MS, std_exempt=sensory, **kw,
        )

    def __init__(
        self,
        connectome: Connectome,
        n_agents: int = 1,
        params: ShiuParams = ShiuParams(),
        device: str | torch.device = "cuda",
        propagation: str = "event",
        std_u: float = 0.0,
        std_tau_ms: float = 300.0,
        std_exempt=None,
    ):
        """propagation: "event" gathers only the out-edges of neurons that spiked (fast when
        <1% of neurons fire per step, which is the norm); "spmv" is a full sparse matmul."""
        self.connectome = connectome
        self.p = params
        self.n_agents = n_agents
        self.n_neurons = connectome.n_neurons
        self.device = torch.device(device)
        self.delay_steps = max(1, round(params.t_dly_ms / params.dt_ms))
        self.rfc_steps = round(params.t_rfc_ms / params.dt_ms)

        # Edge weights already carry the NT sign; scale by w_syn here.
        self.propagation = propagation
        if propagation == "event":
            # CSR indexed by *pre* neuron so a spike's out-edges are one contiguous run.
            e = connectome.edges.sort("pre_idx", "post_idx")
            pre = torch.from_numpy(e["pre_idx"].to_numpy().copy())
            counts = torch.bincount(pre, minlength=self.n_neurons)
            self._rowptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(self.device)
            self._col = torch.from_numpy(e["post_idx"].to_numpy().copy()).to(self.device)
            self._val = (torch.from_numpy(e["weight"].to_numpy().copy()).float() * params.w_syn_mv).to(self.device)
            self._propagate = self._propagate_event
        elif propagation == "spmv":
            self.W = connectome.to_sparse(device=self.device) * params.w_syn_mv  # W[post, pre]
            self._propagate = self._propagate_spmv
        else:
            raise ValueError(f"unknown propagation {propagation!r}")

        # Exact integration constants for the linear ODE pair over one step.
        dt, tm, tau = params.dt_ms, params.t_mbr_ms, params.tau_ms
        self._a = math.exp(-dt / tm)  # membrane decay
        self._b = math.exp(-dt / tau)  # synaptic decay
        self._c = tau / (tau - tm) * (self._b - self._a)  # g -> v coupling

        A, N = n_agents, self.n_neurons
        self.std = std_u > 0
        self.x = torch.ones((A, N), device=self.device)  # synaptic resource per presynaptic neuron
        self._std_u = torch.full((N,), float(std_u), device=self.device)
        if std_exempt is not None:
            self._std_u[torch.as_tensor(std_exempt, dtype=torch.int64, device=self.device)] = 0.0
        self._std_k = params.dt_ms / std_tau_ms
        self.v = torch.full((A, N), params.v_rest_mv, device=self.device)
        self.g = torch.zeros((A, N), device=self.device)
        self.refractory_left = torch.zeros((A, N), dtype=torch.int32, device=self.device)
        self._rfc_steps_per_neuron = torch.full((N,), self.rfc_steps, dtype=torch.int32, device=self.device)
        self._spike_ring = torch.zeros((self.delay_steps, A, N), device=self.device)
        self._ring_pos = 0
        self.step_count = 0

    def _propagate_spmv(self, arriving: torch.Tensor) -> None:
        self.g += (self.W @ arriving.T).T

    def _propagate_event(self, arriving: torch.Tensor) -> None:
        """`arriving` is (A, N): 1 for each delayed spike, scaled by the synaptic resource x."""
        agent, pre = arriving.nonzero(as_tuple=True)
        if pre.numel() == 0:
            return
        starts = self._rowptr[pre]
        lengths = self._rowptr[pre + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return
        # flat edge indices: for each spiking pre, the run starts[i] .. starts[i]+lengths[i]
        seg_offsets = torch.cumsum(lengths, 0) - lengths
        pos = torch.arange(total, device=self.device)
        seg = torch.repeat_interleave(torch.arange(pre.numel(), device=self.device), lengths)
        edge = starts[seg] + (pos - seg_offsets[seg])
        flat_target = agent[seg] * self.n_neurons + self._col[edge]
        self.g.view(-1).index_add_(0, flat_target, self._val[edge] * arriving[agent, pre][seg])

    def set_refractory(self, neuron_idx, t_rfc_ms: float) -> None:
        """Per-neuron refractory period (Shiu: 0 ms for optogenetically activated neurons)."""
        idx = torch.as_tensor(neuron_idx, dtype=torch.int64, device=self.device)
        self._rfc_steps_per_neuron[idx] = round(t_rfc_ms / self.p.dt_ms)

    def silence(self, neuron_idx) -> None:
        """Zero all *outgoing* synapses of these neurons (Shiu's silence(): the neuron may
        still spike, but it no longer influences anyone)."""
        idx = torch.as_tensor(neuron_idx, dtype=torch.int64, device=self.device)
        mask = torch.zeros(self.n_neurons, dtype=torch.bool, device=self.device)
        mask[idx] = True
        if self.propagation == "event":
            pre_of_edge = torch.repeat_interleave(
                torch.arange(self.n_neurons, device=self.device), self._rowptr[1:] - self._rowptr[:-1]
            )
            self._val[mask[pre_of_edge]] = 0.0
        else:
            W = self.W.to_sparse_coo().coalesce()
            post, pre = W.indices()
            keep = ~mask[pre]
            self.W = torch.sparse_coo_tensor(
                W.indices()[:, keep], W.values()[keep], W.shape
            ).coalesce().to_sparse_csr()

    def run(self, n_steps: int, stimuli=(), recorder=None, ext_i: torch.Tensor | None = None):
        """Step the brain `n_steps` times with Poisson activations and/or a constant drive.
        Returns the recorder (a fresh `SpikeRecorder` if none was given)."""
        from flyhigh.brain.recorder import SpikeRecorder

        recorder = recorder or SpikeRecorder(neuron_idx=[])
        for s in stimuli:
            s.attach(self)
        for _ in range(n_steps):
            ext_v = None
            for s in stimuli:
                v = s.ext_v(self)
                ext_v = v if ext_v is None else ext_v + v
            recorder.record(self, self.step(ext_i=ext_i, ext_v=ext_v))
        return recorder

    @property
    def t_ms(self) -> float:
        return self.step_count * self.p.dt_ms

    def step(self, ext_i: torch.Tensor | None = None, ext_v: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one time step.

        ext_i: (A, N) graded drive in mV — a constant input current expressed as the
               steady-state depolarisation it would produce (used for sensory currents).
        ext_v: (A, N) instantaneous jump in v (used for optogenetic-style Poisson kicks).
        Returns the (A, N) bool tensor of neurons that spiked this step.
        """
        p = self.p
        # 1. integrate everything that is not refractory (Brian2 order: state update,
        #    threshold/reset, then synaptic delivery — so arriving spikes land after decay)
        active = self.refractory_left == 0
        v_new = p.v_rest_mv + (self.v - p.v_rest_mv) * self._a + self.g * self._c
        if ext_i is not None:
            v_new = v_new + ext_i * (1.0 - self._a)
        self.v = torch.where(active, v_new, self.v)
        self.g = torch.where(active, self.g * self._b, self.g)
        if ext_v is not None:
            self.v = self.v + ext_v
        self.refractory_left = torch.clamp(self.refractory_left - 1, min=0)

        # 2. threshold, reset (v AND g — Shiu's reset rule 'v = v_rst; g = 0'), refractory
        spiked = self.v > p.v_th_mv
        self.v = torch.where(spiked, torch.full_like(self.v, p.v_reset_mv), self.v)
        self.g = torch.where(spiked, torch.zeros_like(self.g), self.g)
        self.refractory_left = torch.where(
            spiked, self._rfc_steps_per_neuron.expand_as(spiked), self.refractory_left
        )

        # 3. deliver the spikes emitted `delay_steps` ago, then queue this step's spikes
        #    in the slot they just vacated
        arriving = self._spike_ring[self._ring_pos]
        if self.std:
            self.x += (1.0 - self.x) * self._std_k
            self._propagate(arriving * self.x)
            self.x = torch.where(arriving > 0, self.x * (1.0 - self._std_u), self.x)
        else:
            self._propagate(arriving)
        self._spike_ring[self._ring_pos] = spiked.to(self._spike_ring.dtype)
        self._ring_pos = (self._ring_pos + 1) % self.delay_steps
        self.step_count += 1
        return spiked
