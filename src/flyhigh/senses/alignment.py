"""Map every flyvis neuron (type, u, v) to the male-CNS neurons (type, side, hex1, hex2) in the
same eye column.

Both lattices are hexagonal with one column per ommatidium, but they are written in different
bases (`mcns_to_axial`), and the rotation/reflection between them is one of the 12 symmetries
of a hex lattice. Counting matched columns cannot pick the symmetry -- flyvis's eye is a
perfectly symmetric disc, so all 12 tie -- but the *direction* of known synaptic offsets can:
Mi9 sits one column on the preferred-direction side of the T4 it feeds, and only the true
symmetry maps flyvis's offset onto the connectome's (`choose_symmetry`). The translation is
the lattice centre, aligned per eye.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from flyhigh.senses.type_map import mcns_types

# The 6 rotations of axial coords (cube-coordinate cycling) and their mirror images.
# Each entry maps (u, v) -> (u', v') via the cube coordinates (x=u, z=v, y=-u-v).
HEX_SYMMETRIES = [
    ("x", "z"), ("-y", "-x"), ("z", "y"), ("-x", "-z"), ("y", "x"), ("-z", "-y"),  # rotations
    ("z", "x"), ("-x", "-y"), ("y", "z"), ("-z", "-x"), ("x", "y"), ("-y", "-z"),  # reflections
]


def apply_symmetry(u, v, sym):
    cube = {"x": u, "z": v, "y": -u - v}

    def pick(name):
        return -cube[name[1]] if name.startswith("-") else cube[name]

    return pick(sym[0]), pick(sym[1])


def mcns_to_axial(hex1, hex2):
    """Male-CNS (hex1, hex2) -> flyvis-style axial (u, v).

    In axial hex coordinates (flyvis, and the cube trick above) a column's six neighbours are
    (±1,0), (0,±1), ±(1,-1) and the eye is bounded in u, v and u+v. In the male CNS the
    synapse-carrying neighbours are (±1,0), (0,±1), ±(1,+1) and the eye is bounded in
    hex1-hex2: the second axis points the other way. Flipping it makes the two lattices
    differ by a symmetry only.
    """
    return hex1, -hex2


def hex_to_cartesian(u, v):
    """Axial (u, v) -> plane coordinates in units of one column, for distances between offsets."""
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    return np.stack([u + v / 2, v * np.sqrt(3) / 2], axis=-1)


def choose_symmetry(fv_offsets: pl.DataFrame, mcns_offsets: pl.DataFrame):
    """Pick the symmetry that best maps flyvis's per-connection column offsets onto the
    connectome's. Both tables have columns s, t, du, dv (mean pre−post offset, axial
    convention); only (s, t) pairs present in both count. Returns (symmetry, {symmetry: rms})
    where rms is the root-mean-square cartesian distance between offsets, in columns."""
    both = fv_offsets.join(mcns_offsets, on=["s", "t"], suffix="_m")
    if both.height == 0:
        raise ValueError("no (s, t) pair is present in both offset tables")
    m = hex_to_cartesian(both["du_m"].to_numpy(), both["dv_m"].to_numpy())
    scores = {}
    for sym in HEX_SYMMETRIES:
        su, sv = apply_symmetry(both["du"].to_numpy(), both["dv"].to_numpy(), sym)
        scores[sym] = float(np.sqrt(np.mean(np.sum((hex_to_cartesian(su, sv) - m) ** 2, axis=1))))
    return min(scores, key=scores.get), scores


@dataclass
class ColumnAlignment:
    table: pl.DataFrame  # fv_index, eye, lif_index, weight
    # One row per (flyvis neuron, LIF neuron) pair in the same column. A LIF neuron driven by
    # several flyvis neurons (the six R1..R6 of a column all stand for the one `R1-R6` type)
    # gets weight 1/k on each row, so the bridge can sum rows and inject their mean.
    coverage: dict[str, float]
    transform: tuple

    @classmethod
    def build(cls, fv_types, fv_u, fv_v, neurons: pl.DataFrame, symmetry=None) -> ColumnAlignment:
        """`symmetry`: one of HEX_SYMMETRIES, normally from `choose_symmetry`. If None, the
        symmetry matching the most columns is used -- fine for test lattices, degenerate on
        the real (symmetric) flyvis eye, where it silently picks the first."""
        fv_types = np.asarray(fv_types); fv_u = np.asarray(fv_u); fv_v = np.asarray(fv_v)
        target_types = [mcns_types(t) for t in fv_types]
        cols = neurons.filter(pl.col("hex1").is_not_null()).select("index", "type", "side", "hex1", "hex2")
        au, av = mcns_to_axial(cols["hex1"].to_numpy(), cols["hex2"].to_numpy())
        cols = cols.with_columns(pl.Series("u", au), pl.Series("v", av))
        best = None
        for sym in [symmetry] if symmetry is not None else HEX_SYMMETRIES:
            su, sv = apply_symmetry(fv_u, fv_v, sym)
            rows, n_matched = [], 0
            for eye in ("L", "R"):
                side = cols.filter(pl.col("side") == eye)
                # shift: align lattice centres (flyvis centre is (0,0); male-CNS centre = mean of L1 columns)
                l1 = side.filter(pl.col("type") == "L1")
                if l1.height == 0:
                    l1 = side
                du, dv = round(l1["u"].mean()), round(l1["v"].mean())
                lut: dict[tuple, list[int]] = {}
                for t, a, b, i in zip(side["type"], side["u"], side["v"], side["index"]):
                    lut.setdefault((t, int(a), int(b)), []).append(int(i))
                for j in range(len(fv_types)):
                    for t in target_types[j]:
                        hits = lut.get((t, int(su[j]) + du, int(sv[j]) + dv), ())
                        rows.extend((j, eye, hit) for hit in hits)
                        n_matched += bool(hits)
            if best is None or n_matched > best[0]:
                best = (n_matched, sym, rows)
        n_matched, sym, rows = best
        table = pl.DataFrame(rows, schema={"fv_index": pl.Int64, "eye": pl.Utf8, "lif_index": pl.Int64}, orient="row")
        table = table.with_columns((1.0 / pl.len().over("eye", "lif_index")).alias("weight"))
        matched = table.filter(pl.col("eye") == "R")["fv_index"].to_numpy()
        coverage = {}
        for t in np.unique(fv_types):
            m = fv_types == t
            coverage[str(t)] = float(np.isin(np.where(m)[0], matched).mean())
        return cls(table, coverage, tuple(sym))

    def lif_indices(self, eye: str) -> np.ndarray:
        return self.table.filter(pl.col("eye") == eye)["lif_index"].to_numpy().copy()

    def fv_indices(self, eye: str) -> np.ndarray:
        return self.table.filter(pl.col("eye") == eye)["fv_index"].to_numpy().copy()

    def weights(self, eye: str) -> np.ndarray:
        return self.table.filter(pl.col("eye") == eye)["weight"].to_numpy().copy()

    def save(self, path) -> None:
        path = Path(path)
        self.table.write_parquet(path)
        pl.DataFrame({"type": list(self.coverage), "coverage": list(self.coverage.values()),
                      "transform": [",".join(self.transform)] * len(self.coverage)}).write_parquet(path.with_suffix(".meta.parquet"))

    @classmethod
    def load(cls, path) -> ColumnAlignment:
        path = Path(path)
        meta = pl.read_parquet(path.with_suffix(".meta.parquet"))
        return cls(pl.read_parquet(path), dict(zip(meta["type"], meta["coverage"])), tuple(meta["transform"][0].split(",")))
