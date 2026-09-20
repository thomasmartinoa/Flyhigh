# scripts/calibrate_bridge.py
"""Sweep the flyvis→LIF gain: a 60°/s rotating grating should make the LIF T4/T5 fire at
50–100 Hz (their measured range) and grey must leave them silent. Prints one row per gain;
the chosen value becomes `DEFAULT_GAIN` in flyhigh/agent.py (table in docs/03-see-and-move.md)."""
import sys

import numpy as np

from flyhigh.agent import FlyAgent
from flyhigh.data.connectome import Connectome
from flyhigh.senses.alignment import ColumnAlignment
from flyhigh.senses.frame import PanoramicFrame, rotating_grating

GAINS = [float(g) for g in sys.argv[1:]] or [5.0, 10.0, 20.0, 40.0, 80.0]

c = Connectome.load("data/raw")
fv_types = list(ColumnAlignment.load("data/cache/alignment.parquet").coverage)
groups = {
    "T4/T5": c.ids_by_type(r"^T[45][abcd]$"), "LC4/LPLC2": c.ids_by_type(r"^(LC4|LPLC2)$"),
    "HS": c.ids_by_type(r"^HS[NES]$"), "GF": c.ids_by_type(r"^DNp01$"), "DNa": c.ids_by_type(r"^DNa0[12]$"),
}
grating = rotating_grating(wavelength_deg=30, deg_per_s=60, duration_ms=500)
grey = [PanoramicFrame.grey()] * 30

print(f"{'gain':>5} {'stim':8s} " + " ".join(f"{k:>12s}" for k in groups) + "   T4/T5 p90   active")
for gain in GAINS:
    agent = FlyAgent(c, n_agents=1, gains={t: gain for t in fv_types})
    driven = np.intersect1d(groups["T4/T5"], agent.bridge.driven_indices)
    for frames, label in ((grey, "grey"), (grating, "grating")):
        counts = np.zeros(c.n_neurons)
        for fr in frames:
            agent.tick([fr])
            counts += agent.brain_counts_last_tick
        hz = counts / (len(frames) * 0.01)
        print(f"{gain:5.1f} {label:8s} " + " ".join(f"{hz[idx].mean():10.1f} Hz" for idx in groups.values())
              + f"   {np.percentile(hz[driven], 90):7.1f}   {int((counts > 0).sum()):6d}")
