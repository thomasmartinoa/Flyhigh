"""Map every flyvis neuron (type, u, v) to a male-CNS neuron (type, side, hex1, hex2).

Both lattices are axial hex coordinates. The unknown rotation/reflection between them is one
of the 12 symmetries of a hex lattice; the unknown translation is the lattice centre. We pick
the symmetry+shift that matches the most columns, separately per eye, and keep one table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from flyhigh.senses.type_map import mcns_type

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


@dataclass
class ColumnAlignment:
    table: pl.DataFrame  # fv_index, eye, lif_index
    coverage: dict[str, float]
    transform: tuple

    @classmethod
    def build(cls, fv_types, fv_u, fv_v, neurons: pl.DataFrame) -> ColumnAlignment:
        fv_types = np.asarray(fv_types); fv_u = np.asarray(fv_u); fv_v = np.asarray(fv_v)
        target_type = np.array([mcns_type(t) for t in fv_types])
        cols = neurons.filter(pl.col("hex1").is_not_null()).select("index", "type", "side", "hex1", "hex2")
        best = None
        for sym in HEX_SYMMETRIES:
            su, sv = apply_symmetry(fv_u, fv_v, sym)
            rows, n_matched = [], 0
            for eye in ("L", "R"):
                side = cols.filter(pl.col("side") == eye)
                # shift: align lattice centres (flyvis centre is (0,0); male-CNS centre = mean of L1 columns)
                l1 = side.filter(pl.col("type") == "L1")
                if l1.height == 0:
                    l1 = side
                du, dv = round(l1["hex1"].mean()), round(l1["hex2"].mean())
                lut = {(t, int(a), int(b)): i for t, a, b, i in zip(side["type"], side["hex1"], side["hex2"], side["index"])}
                for j in range(len(fv_types)):
                    hit = lut.get((target_type[j], int(su[j]) + du, int(sv[j]) + dv))
                    if hit is not None:
                        rows.append((j, eye, hit)); n_matched += 1
            if best is None or n_matched > best[0]:
                best = (n_matched, sym, rows)
        n_matched, sym, rows = best
        table = pl.DataFrame(rows, schema={"fv_index": pl.Int64, "eye": pl.Utf8, "lif_index": pl.Int64}, orient="row")
        matched = table.filter(pl.col("eye") == "R")["fv_index"].to_numpy()
        coverage = {}
        for t in np.unique(fv_types):
            m = fv_types == t
            coverage[str(t)] = float(np.isin(np.where(m)[0], matched).mean())
        return cls(table, coverage, sym)

    def lif_indices(self, eye: str) -> np.ndarray:
        return self.table.filter(pl.col("eye") == eye)["lif_index"].to_numpy().copy()

    def fv_indices(self, eye: str) -> np.ndarray:
        return self.table.filter(pl.col("eye") == eye)["fv_index"].to_numpy().copy()

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
