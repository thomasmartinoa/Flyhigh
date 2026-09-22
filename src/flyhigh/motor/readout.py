"""Hand-mapped channels from descending-neuron rates (Hz over a 20 ms window) to MotorCommand.

The brain reaches the body through ~1,300 descending neurons; a handful with known jobs are
read here: the giant fiber DNp01 (one spike = jump), DNp04 (its loom-driven partner), DNa01/02
(turn towards their side), the HS/VS tangential cells (whole-field yaw / pitch optic flow) and
DNp09 (forward drive). Each channel is a pure function so it can be tested -- and replaced by a
learned readout later -- on its own. Sign convention: `yaw` > 0 turns right, and the fly turns
*with* the perceived scene rotation (optomotor); the HS side→sign mapping is checked in
`scripts/validate_reflexes.py`, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyhigh.data.connectome import Connectome
from flyhigh.motor.command import MotorCommand


@dataclass(frozen=True)
class ReadoutParams:
    gf_escape_hz: float = 50.0
    gf_assist_hz: float = 20.0
    dnp04_assist_hz: float = 50.0
    k_dn: float = 1.0
    k_hs: float = 1.0
    k_vs: float = 0.01
    k_fwd: float = 0.005
    forward_bias: float = 0.2
    eps_hz: float = 5.0  # keeps (R-L)/(R+L) quiet when both sides barely fire
    escape_refractory_ms: float = 0.0  # ignore the escape channel for this long after an escape


def escape_channel(gf_hz, dnp04_hz, p: ReadoutParams) -> bool:
    return bool(gf_hz > p.gf_escape_hz or (gf_hz > p.gf_assist_hz and dnp04_hz > p.dnp04_assist_hz))


def yaw_channel(dna_l, dna_r, hs_l, hs_r, p: ReadoutParams) -> float:
    dn = p.k_dn * (dna_r - dna_l) / (dna_r + dna_l + p.eps_hz)
    hs = p.k_hs * (hs_r - hs_l) / (hs_r + hs_l + p.eps_hz)
    return float(np.clip(dn + hs, -1.0, 1.0))


def lift_channel(vs_hz, p: ReadoutParams) -> float:
    return float(np.clip(p.k_vs * vs_hz, -1.0, 1.0))


def forward_channel(dnp09_hz, escape: bool, p: ReadoutParams) -> float:
    if escape:
        return 0.0
    return float(np.clip(p.forward_bias + p.k_fwd * dnp09_hz, -1.0, 1.0))


class Readout:
    def __init__(self, connectome: Connectome, params: ReadoutParams | None = None):
        self.p = params or ReadoutParams()
        self._escape_block_ms: list[float] = []  # per agent, time left on the escape refractory
        side = connectome.neurons["side"].to_numpy()

        def ids(pattern, s=None):
            idx = connectome.ids_by_type(pattern)
            return idx if s is None else idx[side[idx] == s]

        self.gf = ids(r"^DNp01$")
        self.dnp04 = ids(r"^DNp04$")
        self.dna_l, self.dna_r = ids(r"^DNa0[12]$", "L"), ids(r"^DNa0[12]$", "R")
        self.hs_l, self.hs_r = ids(r"^HS[NES]$", "L"), ids(r"^HS[NES]$", "R")
        self.vs = ids(r"^VS")
        self.dnp09 = ids(r"^DNp09$")
        self.watch = np.unique(np.concatenate([
            self.gf, self.dnp04, self.dna_l, self.dna_r, self.hs_l, self.hs_r, self.vs, self.dnp09,
        ])).astype(np.int64)

    @staticmethod
    def _mean(rates, idx):
        return float(rates[idx].mean()) if len(idx) else 0.0

    def command(self, rates: np.ndarray, blocked: bool = False) -> MotorCommand:
        """rates: (N,) Hz for every neuron (only `watch` is read).
        `blocked`: the escape channel is refractory, so report no escape whatever the GF does."""
        def m(idx):
            return self._mean(rates, idx)

        escape = False if blocked else escape_channel(m(self.gf), m(self.dnp04), self.p)
        return MotorCommand(
            forward=forward_channel(m(self.dnp09), escape, self.p),
            yaw=0.0 if escape else yaw_channel(m(self.dna_l), m(self.dna_r), m(self.hs_l), m(self.hs_r), self.p),
            lift=0.0 if escape else lift_channel(m(self.vs), self.p),
            escape=escape,
        )

    def commands(self, rates: np.ndarray, dt_ms: float = 10.0) -> list[MotorCommand]:
        """rates: (n_agents, N); `dt_ms`: time since the last call (one tick).

        With `escape_refractory_ms` > 0 an escape silences the channel for that long -- the
        escape is a 100 ms full-authority manoeuvre, and without a refractory one stray giant-fiber
        spike cascades: the body's own motion re-fires the detector (docs/04 §5, docs/05 §4).
        Default 0 keeps the reflex exactly as validated in M2-M4b."""
        while len(self._escape_block_ms) < len(rates):
            self._escape_block_ms.append(0.0)
        out = []
        for i, r in enumerate(rates):
            blocked = self._escape_block_ms[i] > 0
            self._escape_block_ms[i] = max(0.0, self._escape_block_ms[i] - dt_ms)
            cmd = self.command(r, blocked=blocked)
            if cmd.escape:
                self._escape_block_ms[i] = self.p.escape_refractory_ms
            out.append(cmd)
        return out
